#!/usr/bin/env python3
"""
Chess LLM supervised fine-tuning (SFT) training script.

Trains a chat model to output a move (wrapped in `<uci_move>...</uci_move>`)
given a position prompt. The data collator masks the prompt so the loss is
computed only on the assistant response.

Supports optional performance features when installed:
- Cut Cross Entropy (CCE) for reduced activation memory
- Liger kernels for select fused ops (not cross entropy)
"""

import os
import sys
import argparse
import yaml
import random
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import warnings

# Ensure repo root is on sys.path when running from subfolders
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

from src.utils.chess_eval_callback import FastChessEvalCallback as ChessEvalCallback, prepare_eval_positions
from src.utils.chess_tokenizer import ChessTokenizerMode, add_chess_tokens, generate_all_uci_moves

# ============================================================================
# Performance optimizations - set early
# ============================================================================

def setup_torch_optimizations():
    """Configure PyTorch for optimal performance on modern GPUs."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')

setup_torch_optimizations()


# ============================================================================
# Liger Kernel Integration (Non-CrossEntropy kernels only)
# ============================================================================

LIGER_AVAILABLE = False
try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3
    LIGER_AVAILABLE = True
    print("Liger Kernel available")
except ImportError:
    print("Liger Kernel not installed. Install with: pip install liger-kernel")


def apply_liger_kernels_no_ce(model_name: str):
    """
    Apply Liger kernels for RoPE, RMSNorm, SwiGLU but NOT cross entropy.
    CCE handles cross entropy separately to allow weighted loss.
    """
    if not LIGER_AVAILABLE:
        return
    
    model_name_lower = model_name.lower()
    
    if 'qwen3' in model_name_lower or 'qwen/qwen3' in model_name_lower:
        apply_liger_kernel_to_qwen3(
            rope=True,
            rms_norm=True,
            swiglu=True,
            cross_entropy=False,
            fused_linear_cross_entropy=False,
        )
        print("Applied Liger kernels to Qwen3 (RoPE, RMSNorm, SwiGLU)")
        print("  Cross Entropy disabled (using CCE for weighted loss)")
    
    elif 'qwen2' in model_name_lower:
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2
            apply_liger_kernel_to_qwen2(
                rope=True, rms_norm=True, swiglu=True,
                cross_entropy=False, fused_linear_cross_entropy=False,
            )
            print("Applied Liger kernels to Qwen2 (RoPE, RMSNorm, SwiGLU)")
        except ImportError:
            print("Qwen2 Liger kernels not available")
    
    elif 'llama' in model_name_lower:
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_llama
            apply_liger_kernel_to_llama(
                rope=True, rms_norm=True, swiglu=True,
                cross_entropy=False, fused_linear_cross_entropy=False,
            )
            print("Applied Liger kernels to Llama (RoPE, RMSNorm, SwiGLU)")
        except ImportError:
            print("Llama Liger kernels not available")
    
    elif 'mistral' in model_name_lower:
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_mistral
            apply_liger_kernel_to_mistral(
                rope=True, rms_norm=True, swiglu=True,
                cross_entropy=False, fused_linear_cross_entropy=False,
            )
            print("Applied Liger kernels to Mistral (RoPE, RMSNorm, SwiGLU)")
        except ImportError:
            print("Mistral Liger kernels not available")
    else:
        print(f"No Liger kernel support for model: {model_name}")


# ============================================================================
# Cut Cross Entropy Integration
# ============================================================================

CCE_AVAILABLE = False
try:
    from cut_cross_entropy import linear_cross_entropy
    CCE_AVAILABLE = True
    print("Cut Cross Entropy available")
except ImportError:
    print("Cut Cross Entropy not installed. Install with:")
    print('  pip install "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git"')


# ============================================================================
# Data Collator with Prompt Masking
# ============================================================================

@dataclass
class SFTDataCollatorWithPromptMasking:
    """
    Data collator that:
    1. Tokenizes conversations
    2. Masks prompt/user tokens (sets labels to -100)
    3. Only computes loss on assistant response tokens
    4. Tracks sample weights for weighted loss
    
    This ensures we only train on the model's output, not on repeating the user input.
    """
    tokenizer: Any
    max_length: int = 1024
    pad_to_multiple_of: int = 8
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_weights = []
        
        for example in examples:
            # Get messages
            if 'messages' in example:
                messages = example['messages']
            else:
                from src.utils.formatting import position_to_messages
                messages = position_to_messages(example)['messages']
            
            # Tokenize the full conversation
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            # Tokenize just the user part (prompt only, with generation prompt)
            # This gives us the length of tokens to mask
            user_only_messages = [messages[0]]  # Just the user message
            prompt_text = self.tokenizer.apply_chat_template(
                user_only_messages,
                tokenize=False,
                add_generation_prompt=True  # Include the assistant header
            )
            
            # Tokenize both
            full_tokens = self.tokenizer(
                full_text,
                truncation=False,
                return_tensors=None,
            )
            
            prompt_tokens = self.tokenizer(
                prompt_text,
                truncation=False,
                return_tensors=None,
            )
            
            input_ids = full_tokens['input_ids']
            attention_mask = full_tokens.get('attention_mask') or [1] * len(input_ids)
             
            # Create labels: -100 for prompt tokens, actual ids for response
            labels = input_ids.copy()
            prompt_length = len(prompt_tokens['input_ids'])

            if len(input_ids) > self.max_length:
                overflow = len(input_ids) - self.max_length
                removed_from_assistant = max(0, overflow - prompt_length)
                if removed_from_assistant:
                    warnings.warn(
                        "Example exceeded max_length; removed "
                        f"{removed_from_assistant} assistant tokens (max_length={self.max_length}). "
                        "Consider reducing reasoning_trace.max_trace_tokens or increasing model.max_seq_length."
                    )
                input_ids = input_ids[overflow:]
                attention_mask = attention_mask[overflow:]
                labels = input_ids.copy()
                prompt_length = max(0, prompt_length - overflow)
             
            # Mask all prompt tokens (set to -100 = ignore_index)
            for i in range(min(prompt_length, len(labels))):
                labels[i] = -100
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
            batch_weights.append(example.get('loss_weight', 1.0))
        
        # Pad to same length
        max_len = max(len(ids) for ids in batch_input_ids)
        # Round up to multiple of pad_to_multiple_of
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) 
                       // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        pad_token_id = self.tokenizer.pad_token_id
        
        for input_ids, attn_mask, labels in zip(batch_input_ids, batch_attention_mask, batch_labels):
            padding_length = max_len - len(input_ids)
            
            # Pad on the right
            padded_input_ids.append(input_ids + [pad_token_id] * padding_length)
            padded_attention_mask.append(attn_mask + [0] * padding_length)
            padded_labels.append(labels + [-100] * padding_length)  # -100 for padding
        
        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(padded_attention_mask, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long),
            'loss_weights': torch.tensor(batch_weights, dtype=torch.float32),
        }


# ============================================================================
# Weighted CCE Trainer with Sample-Level Weighting
# ============================================================================

class WeightedCCETrainer(Trainer):
    """
    Custom Trainer with:
    - Cut Cross Entropy for memory efficiency
    - Sample-level weighting (weight per sample, not per token)
    - Correct gradient accumulation handling
    
    Loss computation:
    1. Get per-token losses (CCE with reduction="none")
    2. Labels already have -100 for prompt tokens (ignored by CCE)
    3. Sum token losses per sample -> per-sample loss
    4. Multiply by sample weight
    5. Average weighted losses across samples
    """
    
    def __init__(self, *args, use_cce: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cce = use_cce and CCE_AVAILABLE
        self._warned = False
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute weighted cross-entropy loss with sample-level weights.
        
        Key points:
        - Labels have -100 for prompt tokens (only train on assistant response)
        - Weights are per-sample, not per-token
        - Uses sum reduction then normalizes correctly
        """
        loss_weights = inputs.pop('loss_weights', None)
        labels = inputs.pop('labels', None)
        
        # Forward pass
        outputs = model(**inputs, output_hidden_states=True)
        
        if labels is None:
            labels = inputs.get('input_ids')
        
        # Get hidden states and classifier for CCE
        hidden_states = outputs.hidden_states[-1]  # (batch, seq, hidden)
        
        if hasattr(model, 'lm_head'):
            classifier = model.lm_head.weight
        elif hasattr(model, 'model') and hasattr(model.model, 'lm_head'):
            classifier = model.model.lm_head.weight
        else:
            return self._compute_standard_ce_loss(
                outputs.logits, labels, loss_weights, return_outputs, outputs
            )
        
        if self.use_cce:
            try:
                loss = self._compute_cce_weighted_loss(
                    hidden_states, classifier, labels, loss_weights
                )
                return (loss, outputs) if return_outputs else loss
            except Exception as e:
                if not self._warned:
                    print(f"CCE computation failed: {e}, using standard CE")
                    self._warned = True
        
        return self._compute_standard_ce_loss(
            outputs.logits, labels, loss_weights, return_outputs, outputs
        )
    
    def _compute_cce_weighted_loss(
        self,
        hidden_states: torch.Tensor,
        classifier: torch.Tensor,
        labels: torch.Tensor,
        loss_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute weighted loss using Cut Cross Entropy.
        
        Uses CCE's shift=1 parameter for memory-efficient causal LM shifting.
        This avoids allocating the shifted embeddings matrix.
        
        Sample-level weighting approach:
        1. Get per-token losses (prompt tokens already masked with -100)
        2. Sum losses per sample to get per-sample loss
        3. Divide by valid token count per sample -> mean loss per sample
        4. Multiply by sample weight
        5. Average across samples
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Create mask for valid (non-ignored) tokens after shift
        # After shift=1: labels[..., 1:] is used, so seq becomes seq-1
        shift_labels = labels[..., 1:].contiguous()
        valid_mask = (shift_labels != -100)
        
        # Count valid tokens per sample (for normalization)
        valid_tokens_per_sample = valid_mask.sum(dim=1).float().clamp(min=1)
        
        # CCE with shift=1 handles causal LM shifting internally without allocating
        # a new shifted embeddings matrix. This is more memory efficient.
        # Result shape: (batch * (seq-1),) with reduction="none"
        per_token_loss = linear_cross_entropy(
            hidden_states,  # (batch, seq, hidden) - NOT pre-shifted
            classifier,
            labels,  # (batch, seq) - NOT pre-shifted  
            shift=1,  # CCE handles shifting internally
            ignore_index=-100,
            reduction="none",
        )
        
        # Reshape to (batch, seq-1)
        per_token_loss = per_token_loss.view(batch_size, seq_len - 1)
        
        # Sum per sample, then divide by valid token count -> mean loss per sample
        per_sample_loss = per_token_loss.sum(dim=1) / valid_tokens_per_sample
        
        # Apply sample weights
        if loss_weights is not None:
            loss_weights = loss_weights.to(per_sample_loss.device)
            # Weighted mean: sum(w_i * loss_i) / sum(w_i)
            weighted_loss = (per_sample_loss * loss_weights).sum() / loss_weights.sum()
        else:
            # Simple mean across samples
            weighted_loss = per_sample_loss.mean()
        
        return weighted_loss
    
    def _compute_standard_ce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_weights: Optional[torch.Tensor],
        return_outputs: bool,
        outputs: Any,
    ) -> torch.Tensor:
        """Fallback to standard cross entropy with sample-level weighting."""
        
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        batch_size, seq_len, vocab_size = shift_logits.shape
        
        # Valid token mask
        valid_mask = (shift_labels != -100)
        
        valid_tokens_per_sample = valid_mask.sum(dim=1).float().clamp(min=1)
        
        # Per-token loss
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)
        
        per_token_loss = F.cross_entropy(
            flat_logits.float(),
            flat_labels,
            ignore_index=-100,
            reduction='none'
        )
        per_token_loss = per_token_loss.view(batch_size, seq_len)
        
        # Per-sample mean loss
        per_sample_loss = per_token_loss.sum(dim=1) / valid_tokens_per_sample
        
        # Apply sample weights
        if loss_weights is not None:
            loss_weights = loss_weights.to(per_sample_loss.device)
            weighted_loss = (per_sample_loss * loss_weights).sum() / loss_weights.sum()
        else:
            weighted_loss = per_sample_loss.mean()
        
        return (weighted_loss, outputs) if return_outputs else weighted_loss


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_model_and_tokenizer(config: Dict[str, Any], use_liger: bool = True):
    """Setup model and tokenizer with optimizations."""
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    tokenizer_config = config.get("tokenizer", {})
    
    model_name = model_config.get('name', 'Qwen/Qwen3-0.6B')
    print(f"Loading model: {model_name}")
    
    # Apply Liger kernels BEFORE loading model
    if use_liger and LIGER_AVAILABLE:
        apply_liger_kernels_no_ce(model_name)
    
    dtype_str = model_config.get('dtype', 'bfloat16')
    dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16}.get(dtype_str, torch.float32)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    attn_impl = model_config.get('attn_implementation', 'flash_attention_2')
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )
    except Exception as e:
        if attn_impl == "flash_attention_2":
            print(f"Warning: flash_attention_2 unavailable ({e}); falling back to eager attention.")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager",
            )
        else:
            raise
    
    if training_config.get('gradient_checkpointing', True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("Gradient checkpointing enabled (non-reentrant)")

    mode_str = tokenizer_config.get("chess_mode", "tags_only")
    if mode_str not in ("tags_only", "tags_and_moves"):
        raise ValueError(f"tokenizer.chess_mode must be 'tags_only' or 'tags_and_moves' (got {mode_str!r})")
    mode: ChessTokenizerMode = mode_str
    add_think_tags = bool(tokenizer_config.get("add_think_tags", False))

    all_uci_moves = generate_all_uci_moves() if mode == "tags_and_moves" else None
    added = add_chess_tokens(
        tokenizer=tokenizer,
        model=model,
        mode=mode,
        add_think_tags=add_think_tags,
        all_uci_moves=all_uci_moves,
    )
    if added["special_tokens"] or added["move_tokens"]:
        print(
            f"Added tokens: {added['special_tokens']} special, "
            f"{added['move_tokens']} UCI move tokens (mode={mode})"
        )
     
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is not None:
        model.config.bos_token_id = tokenizer.bos_token_id
    
    if hasattr(model, 'generation_config'):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    
    if training_config.get('torch_compile', False):
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled")
    
    return model, tokenizer


def load_or_create_dataset(
    config: Dict[str, Any],
    streaming: bool = False,
    preprocessed_path: Optional[str] = None,
    tokenizer=None,
):
    """Load existing dataset or create from scratch."""
    from src.utils.data_processing import create_streaming_dataset, preprocess_and_save
    from src.utils.formatting import add_messages_column
    
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    eval_size = training_config.get('eval_size', 1000)
    elo_weights = config.get('elo_weights', None)
    
    if preprocessed_path and Path(preprocessed_path).exists():
        print(f"Loading preprocessed dataset from {preprocessed_path}")
        dataset = load_from_disk(preprocessed_path)
        
        if 'messages' not in dataset.column_names:
            reasoning_cfg = config.get("reasoning_trace", {})
            formatting_cfg = config.get("formatting", {})

            if reasoning_cfg.get("enabled") and "move_evaluations" in dataset.column_names:
                if tokenizer is None:
                    raise ValueError("tokenizer must be provided to build reasoning-trace messages.")

                from src.distill.reasoning_trace import ReasoningTraceGenerator
                from src.sft.formatting_sft import add_messages_with_reasoning_trace, maybe_override_target_to_best

                distill_cfg = config.get("distillation", {})
                min_probability = float(distill_cfg.get("min_probability", 0.001))
                stockfish_temperature = float(distill_cfg.get("stockfish_temperature", 100.0))
                include_board = bool(formatting_cfg.get("include_board", False))
                always_choose_best = bool(
                    reasoning_cfg.get("always_choose_best_move", False)
                    or data_config.get("target_move") == "best"
                )

                trace_generator = ReasoningTraceGenerator(reasoning_cfg, tokenizer=tokenizer)
                if data_config.get("target_move") == "best" and "best_move_uci" in dataset.column_names:
                    dataset = dataset.map(maybe_override_target_to_best, desc="Setting target_move_uci=best_move_uci")

                def _add_messages(ex):
                    return add_messages_with_reasoning_trace(
                        ex,
                        reasoning_trace_generator=trace_generator,
                        include_board=include_board,
                        always_choose_best_move=always_choose_best,
                        min_probability=min_probability,
                        stockfish_temperature=stockfish_temperature,
                    )

                dataset = dataset.map(_add_messages, desc="Building reasoning-trace messages")
            else:
                prompt_config = config.get('prompt', {})
                dataset = add_messages_column(
                    dataset,
                    prompt_template=prompt_config.get('template'),
                    response_template=prompt_config.get('response_template'),
                    num_proc=os.cpu_count() or 4
                )
        
        if eval_size > 0 and len(dataset) > eval_size:
            dataset = dataset.shuffle(seed=training_config.get('seed', 42))
            eval_dataset = dataset.select(range(eval_size))
            train_dataset = dataset.select(range(eval_size, len(dataset)))
            print(f"Split: {len(train_dataset)} train, {len(eval_dataset)} eval")
            if 'source' in eval_dataset.column_names:
                eval_datasets = {'total': eval_dataset}
                eval_games = eval_dataset.filter(lambda ex: ex['source'] == 'game')
                eval_puzzles = eval_dataset.filter(lambda ex: ex['source'] == 'puzzle')
                if len(eval_games) > 0:
                    eval_datasets['games'] = eval_games
                if len(eval_puzzles) > 0:
                    eval_datasets['puzzles'] = eval_puzzles
                return train_dataset, eval_datasets
            return train_dataset, eval_dataset
        
        return dataset, None
    
    if streaming:
        print("Creating streaming dataset...")
        train_dataset = create_streaming_dataset(
            games_ratio=data_config.get('games_ratio', 0.7),
            config=config,
            seed=config.get('training', {}).get('seed', 42)
        )
        
        if eval_size > 0:
            print(f"Creating eval dataset ({eval_size} examples)...")
            from src.utils.data_processing import stream_game_positions, stream_puzzle_positions
            eval_examples = []
            eval_games_examples = []
            eval_puzzles_examples = []

            games_ratio = data_config.get('games_ratio', 0.7)
            eval_games = int(eval_size * games_ratio)
            eval_puzzles = eval_size - eval_games

            if eval_games > 0:
                games_count = 0
                for pos in stream_game_positions(
                    dataset_name=data_config.get('games_dataset', 'Lichess/standard-chess-games'),
                    min_elo=data_config.get('min_elo', 1200),
                    elo_weights=elo_weights,
                    sample_rate=data_config.get('sample_rate', 0.3),
                    skip_first_moves=data_config.get('skip_first_moves', 4),
                    skip_last_moves=data_config.get('skip_last_moves', 2),
                    config=config,
                    seed=training_config.get('seed', 42) + 1000
                ):
                    eval_examples.append(pos)
                    eval_games_examples.append(pos)
                    games_count += 1
                    if games_count >= eval_games:
                        break

            if eval_puzzles > 0:
                puzzles_count = 0
                for pos in stream_puzzle_positions(
                    dataset_name=data_config.get('puzzles_dataset', 'Lichess/chess-puzzles'),
                    min_rating=data_config.get('min_puzzle_rating', 1000),
                    max_rating=data_config.get('max_puzzle_rating', 2500),
                    config=config,
                    seed=training_config.get('seed', 42) + 2000
                ):
                    eval_examples.append(pos)
                    eval_puzzles_examples.append(pos)
                    puzzles_count += 1
                    if puzzles_count >= eval_puzzles:
                        break

            rng = random.Random(training_config.get('seed', 42) + 3000)
            rng.shuffle(eval_examples)

            eval_datasets = {'total': Dataset.from_list(eval_examples)}
            if eval_games_examples:
                eval_datasets['games'] = Dataset.from_list(eval_games_examples)
            if eval_puzzles_examples:
                eval_datasets['puzzles'] = Dataset.from_list(eval_puzzles_examples)

            prompt_config = config.get('prompt', {})
            eval_datasets = {
                name: add_messages_column(
                    ds,
                    prompt_template=prompt_config.get('template'),
                    response_template=prompt_config.get('response_template'),
                    num_proc=os.cpu_count() or 4
                )
                for name, ds in eval_datasets.items()
            }
            print(f"Eval dataset: {len(eval_examples)} examples")
            print(f"  Games: {len(eval_games_examples)}, Puzzles: {len(eval_puzzles_examples)}")
            return train_dataset, eval_datasets
        
        return train_dataset, None
    else:
        save_path = data_config.get('preprocessed_path', './data/chess_sft_preprocessed')
        print(f"Pre-processing dataset to {save_path}...")
        dataset = preprocess_and_save(
            output_path=save_path,
            target_size=data_config.get('target_size', 5_000_000),
            games_ratio=data_config.get('games_ratio', 0.7),
            config=config,
            seed=config.get('training', {}).get('seed', 42)
        )
        
        prompt_config = config.get('prompt', {})
        dataset = add_messages_column(
            dataset,
            prompt_template=prompt_config.get('template'),
            response_template=prompt_config.get('response_template'),
            num_proc=os.cpu_count() or 4
        )
        dataset.save_to_disk(save_path)
        
        if eval_size > 0 and len(dataset) > eval_size:
            dataset = dataset.shuffle(seed=training_config.get('seed', 42))
            eval_dataset = dataset.select(range(eval_size))
            train_dataset = dataset.select(range(eval_size, len(dataset)))
            if 'source' in eval_dataset.column_names:
                eval_datasets = {'total': eval_dataset}
                eval_games = eval_dataset.filter(lambda ex: ex['source'] == 'game')
                eval_puzzles = eval_dataset.filter(lambda ex: ex['source'] == 'puzzle')
                if len(eval_games) > 0:
                    eval_datasets['games'] = eval_games
                if len(eval_puzzles) > 0:
                    eval_datasets['puzzles'] = eval_puzzles
                return train_dataset, eval_datasets
            return train_dataset, eval_dataset
        
        return dataset, None


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    config: Dict[str, Any],
    streaming: bool = False,
    use_cce: bool = True,
    chess_eval_callback: Optional[TrainerCallback] = None
):
    """Create optimized trainer."""
    training_config = config.get('training', {})
    model_config = config.get('model', {})
    
    max_steps = training_config.get('max_steps', -1)
    if streaming and max_steps <= 0:
        data_config = config.get('data', {})
        target_size = data_config.get('target_size', 5_000_000)
        batch_size = training_config.get('per_device_train_batch_size', 8)
        grad_accum = training_config.get('gradient_accumulation_steps', 8)
        epochs = training_config.get('num_train_epochs', 1)
        max_steps = (target_size // (batch_size * grad_accum)) * epochs
        print(f"Setting max_steps to {max_steps} for streaming mode")
    
    eval_strategy = training_config.get('eval_strategy', 'no')
    metric_for_best_model = None
    if eval_dataset is None:
        eval_strategy = 'no'
    elif eval_strategy != 'no':
        if isinstance(eval_dataset, dict):
            if 'total' in eval_dataset:
                metric_for_best_model = 'eval_total_loss'
            else:
                first_key = next(iter(eval_dataset.keys()))
                metric_for_best_model = f"eval_{first_key}_loss"
        else:
            metric_for_best_model = 'eval_loss'
    
    training_args = TrainingArguments(
        output_dir=training_config.get('output_dir', './outputs/chess-sft'),
        
        num_train_epochs=training_config.get('num_train_epochs', 1) if not streaming else 1,
        max_steps=max_steps if streaming else training_config.get('max_steps', -1),
        
        per_device_train_batch_size=training_config.get('per_device_train_batch_size', 8),
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 16),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 8),
        
        learning_rate=training_config.get('learning_rate', 5e-6),
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
        warmup_ratio=training_config.get('warmup_ratio', 0.03),
        weight_decay=training_config.get('weight_decay', 0.01),
        
        logging_steps=training_config.get('logging_steps', 50),
        logging_first_step=True,
        
        save_strategy=training_config.get('save_strategy', 'steps'),
        save_steps=training_config.get('save_steps', 500),
        save_total_limit=training_config.get('save_total_limit', 3),
        
        eval_strategy=eval_strategy,
        eval_steps=training_config.get('eval_steps', 500) if eval_strategy == 'steps' else None,
        
        bf16=training_config.get('bf16', True),
        fp16=False,
        bf16_full_eval=True,
        
        optim=training_config.get('optim', 'adamw_torch_fused'),
        
        seed=training_config.get('seed', 42),
        data_seed=training_config.get('seed', 42),
        
        dataloader_num_workers=training_config.get('dataloader_num_workers', 0),
        dataloader_pin_memory=training_config.get('dataloader_pin_memory', True),
        
        gradient_checkpointing=training_config.get('gradient_checkpointing', True),
        
        report_to=training_config.get('report_to', 'wandb'),
        run_name=training_config.get('run_name', 'chess-sft-cce-liger'),
        
        remove_unused_columns=False,
        
        load_best_model_at_end=eval_strategy != 'no',
        metric_for_best_model=metric_for_best_model,
        
        # Do NOT use Liger via TrainingArguments - we apply manually without CE
        use_liger_kernel=False,
        
        torch_compile=training_config.get('torch_compile', False),
        
        push_to_hub=False,
        include_tokens_per_second=True,
        include_num_input_tokens_seen=True,
    )
    
    # Data collator with prompt masking
    data_collator = SFTDataCollatorWithPromptMasking(
        tokenizer=tokenizer,
        max_length=model_config.get('max_seq_length', 1024),
        pad_to_multiple_of=8,
    )
    
    callbacks = []
    if chess_eval_callback is not None:
        callbacks.append(chess_eval_callback)
        print("Fast chess evaluation callback enabled")
    
    trainer = WeightedCCETrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        use_cce=use_cce,
        callbacks=callbacks if callbacks else None,
    )
    
    return trainer


def save_model(model, tokenizer, config: Dict[str, Any]):
    """Save the trained model."""
    output_config = config.get('output', {})
    training_config = config.get('training', {})
    
    output_dir = training_config.get('output_dir', './outputs/chess-sft')
    final_path = f"{output_dir}-final"
    
    print(f"Saving model to {final_path}...")
    
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    if output_config.get('push_to_hub', False):
        hub_model_id = output_config.get('hub_model_id')
        if hub_model_id:
            print(f"Pushing to HuggingFace Hub: {hub_model_id}")
            model.push_to_hub(hub_model_id)
            tokenizer.push_to_hub(hub_model_id)
    
    print("Model saved successfully!")
    return final_path


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def main():
    parser = argparse.ArgumentParser(description='Chess LLM SFT with CCE + Liger')
    parser.add_argument('--config', type=str, default='configs/sft/config_sft.yaml')
    parser.add_argument('--streaming', action='store_true')
    parser.add_argument('--preprocessed_path', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--no-cce', action='store_true')
    parser.add_argument('--no-liger', action='store_true')
    
    args = parser.parse_args()
    
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    if args.debug:
        print("DEBUG MODE: Using small dataset")
        config['data']['target_size'] = 1000
        config['training']['max_steps'] = 100
        config['training']['logging_steps'] = 10
        config['training']['save_steps'] = 50
        config['training']['report_to'] = 'none'
        config['training']['per_device_train_batch_size'] = 2
        config['training']['gradient_accumulation_steps'] = 2
    
    use_cce = CCE_AVAILABLE and not args.no_cce
    use_liger = LIGER_AVAILABLE and not args.no_liger
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION SETTINGS")
    print("=" * 60)
    print(f"TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"Cut Cross Entropy: {'enabled' if use_cce else 'disabled'}")
    print(f"Liger Kernels (RoPE/RMSNorm/SwiGLU): {'enabled' if use_liger else 'disabled'}")
    print(f"Flash Attention: {config.get('model', {}).get('attn_implementation', 'flash_attention_2')}")
    print("Prompt masking: enabled (train only on assistant response)")
    print("Sample-level weighting: enabled")
    print("=" * 60)
    
    print("\nSetting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(config, use_liger=use_liger)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print_gpu_memory()
    
    print("\nLoading dataset...")
    train_dataset, eval_dataset = load_or_create_dataset(
        config,
        streaming=args.streaming,
        preprocessed_path=args.preprocessed_path,
        tokenizer=tokenizer,
    )
    
    if hasattr(train_dataset, '__len__'):
        print(f"Train dataset size: {len(train_dataset):,}")
    else:
        print("Train dataset: streaming mode")
    
    if eval_dataset is not None:
        if isinstance(eval_dataset, dict):
            for name, ds in eval_dataset.items():
                print(f"Eval dataset ({name}) size: {len(ds):,}")
        else:
            print(f"Eval dataset size: {len(eval_dataset):,}")
    
    # Chess evaluation callback
    chess_eval_callback = None
    training_config = config.get('training', {})
    eval_config = config.get('evaluation', {})

    if eval_dataset is not None and not args.debug:
        # Find Stockfish binary
        stockfish_path = None
        possible_paths = [
            eval_config.get('stockfish_path'),
            '/usr/bin/stockfish',
            '/usr/games/stockfish',
            '/usr/local/bin/stockfish',
            '/opt/homebrew/bin/stockfish',  # macOS ARM
        ]

        # Also check if stockfish is in PATH
        import shutil
        stockfish_in_path = shutil.which('stockfish')
        if stockfish_in_path:
            possible_paths.insert(0, stockfish_in_path)

        for path in possible_paths:
            if path and Path(path).exists():
                stockfish_path = path
                break

        eval_positions = prepare_eval_positions(
            eval_dataset,
            max_positions=training_config.get("chess_eval_positions", 500),
        )

        if eval_positions:
            chess_eval_callback = ChessEvalCallback(
                eval_positions=eval_positions,
                tokenizer=tokenizer,
                eval_batch_size=training_config.get('chess_eval_batch_size', 32),
                max_new_tokens=training_config.get('chess_eval_max_new_tokens', 64),
                max_total_tokens=training_config.get(
                    "chess_eval_max_total_tokens",
                    config.get("model", {}).get("max_seq_length", 2048),
                ),
                eval_every_n_steps=training_config.get('chess_eval_steps', 500),
                stockfish_path=stockfish_path,
                stockfish_workers=eval_config.get('stockfish_workers', 8),
                stockfish_depth=eval_config.get('stockfish_depth', 10),
                print_samples=training_config.get("chess_eval_print_samples", 0),
                print_max_chars=training_config.get("chess_eval_print_max_chars", 600),
            )
            print(f"Chess eval callback: {len(eval_positions)} positions")
            if stockfish_path:
                print(f"Stockfish found: {stockfish_path} (ACPL enabled)")
            else:
                print("Stockfish not found - ACPL metrics disabled")
    
    print("\nCreating trainer...")
    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
        streaming=args.streaming,
        use_cce=use_cce,
        chess_eval_callback=chess_eval_callback
    )
    
    resume_checkpoint = args.resume
    if resume_checkpoint and not Path(resume_checkpoint).exists():
        print(f"Warning: Checkpoint {resume_checkpoint} not found")
        resume_checkpoint = None
    
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    loss_config = config.get('loss_weighting', {})
    print(f"Loss weighting: {loss_config.get('function', 'linear')}")
    print(f"Weight range: [{loss_config.get('min_weight', 0.5)}, {loss_config.get('max_weight', 2.0)}]")
    print_gpu_memory()
    
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print_gpu_memory()
    
    print("\nSaving model...")
    save_model(model, tokenizer, config)


if __name__ == "__main__":
    main()
