#!/usr/bin/env python3
"""
Chess LLM Policy Distillation Training Script.

Trains a language model to play chess using policy distillation from Stockfish.
Uses Forward KL divergence (or JSD) with soft targets from Stockfish analysis.

Key features:
- Liger Kernel KL/JSD for memory-efficient distillation loss
- Support for both preprocessed and streaming data modes
- Distillation-specific metrics (KL divergence, top1_agreement)
- Combined soft (distillation) + hard (cross-entropy) loss

Usage:
    # With preprocessed data (recommended)
    python distill/train.py --config configs/distill/config_distill.yaml --preprocessed_path ./data/chess_distill

    # With streaming (on-the-fly Stockfish analysis)
    python distill/train.py --config configs/distill/config_distill.yaml --streaming

    # Debug mode
    python distill/train.py --config configs/distill/config_distill.yaml --preprocessed_path ./data/chess_distill --debug
"""

import os
import sys
import argparse
import json
import yaml
import random
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import warnings
import shutil

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
from transformers.trainer_pt_utils import AcceleratorConfig
import chess


# ============================================================================
# Performance optimizations
# ============================================================================

def setup_torch_optimizations():
    """Configure PyTorch for optimal performance."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')

setup_torch_optimizations()


# ============================================================================
# Liger Kernel Integration
# ============================================================================

LIGER_AVAILABLE = False
try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3
    LIGER_AVAILABLE = True
    print("Liger Kernel available")
except ImportError:
    print("Liger Kernel not installed. Install with: pip install liger-kernel")


def apply_liger_kernels_no_ce(model_name: str):
    """Apply Liger kernels for RoPE, RMSNorm, SwiGLU (not cross entropy)."""
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
# Imports from local modules
# ============================================================================

from src.distill.distillation_loss import (
    ChessDistillationLoss,
    compute_distillation_metrics,
    create_soft_target_tensor,
    LIGER_AVAILABLE as LIGER_LOSS_AVAILABLE,
)
from src.distill.collator_distill import (
    DistillationCollator,
    PrecomputedDistillationCollator,
)
from src.distill.stockfish_teacher import StockfishTeacher
from src.distill.reasoning_trace import ReasoningTraceGenerator
from src.utils.chess_tokenizer import (
    ChessTokenizerMode,
    add_chess_tokens,
    build_move_to_token_id,
    generate_all_uci_moves,
)


# ============================================================================
# Distillation Trainer
# ============================================================================

class DistillationTrainer(Trainer):
    """
    Custom Trainer for policy distillation.

    Loss structure:
        L_total = L_ce + kl_weight * L_kl

        L_ce: Cross-entropy on full sequence (thinking + move)
              Teaches the model to generate reasoning text

        L_kl: KL divergence on <uci_move> token only
              Soft distribution from Stockfish CP scores
              Teaches move quality ranking across ALL legal moves

    The soft targets teach relative move quality while CE teaches reasoning.
    """

    def __init__(
        self,
        *args,
        distill_loss_fn: Optional[ChessDistillationLoss] = None,
        move_to_token_id: Optional[Dict[str, int]] = None,
        move_distill_mode: str = "move_token",
        ce_loss_enabled: bool = True,
        kl_weight: float = 1.0,
        use_cce: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.distill_loss_fn = distill_loss_fn
        self.move_to_token_id = move_to_token_id or {}
        if move_distill_mode not in ("move_token", "move_prefix"):
            raise ValueError(
                "distillation.move_distill_mode must be 'move_token' or 'move_prefix' "
                f"(got {move_distill_mode!r})"
            )
        if move_distill_mode == "move_token" and not self.move_to_token_id:
            raise ValueError(
                "move_distill_mode='move_token' requires UCI move tokens. "
                "Set tokenizer.chess_mode='tags_and_moves' or switch to move_prefix."
            )
        self.move_distill_mode = move_distill_mode
        self.ce_loss_enabled = ce_loss_enabled
        self.kl_weight = kl_weight
        self.use_cce = use_cce and CCE_AVAILABLE
        self._cce_warned = False
        self._step_count = 0
        self._accumulated_metrics = {
            'ce_loss': 0.0,
            'soft_loss': 0.0,
            'hard_loss': 0.0,
            'top1_agreement': 0.0,
            'kl_divergence': 0.0,
            'move_pos_found': 0.0,
            'mapped_prob': 0.0,
            'mapped_move_frac': 0.0,
        }
        self._metric_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute combined loss for distillation training:
        1. CE loss on full sequence (thinking + move) - teaches reasoning text
        2. KL loss on move position only - teaches probability distribution

        The soft targets (teacher_probs) come from Stockfish analysis.
        """
        # Extract distillation-specific inputs
        soft_targets = inputs.pop('soft_targets', None)
        sample_weights = inputs.pop('sample_weights', None)
        target_info = inputs.pop('target_info', None)
        labels = inputs.pop('labels', None)
        move_positions = inputs.pop('move_positions', None)

        # Forward pass - need hidden states for CCE
        outputs = model(**inputs, output_hidden_states=self.use_cce)
        logits = outputs.logits  # (batch, seq, vocab)

        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device

        # STEP 1: Compute CE loss on full sequence (thinking + move tokens)
        # This teaches the model to produce the reasoning/analysis text
        # Note: For proper gradient accumulation, we use sum reduction and divide
        # by num_items_in_batch (total valid tokens across accumulated batches)
        ce_loss = torch.tensor(0.0, device=device)
        if self.ce_loss_enabled and labels is not None:
            if self.use_cce:
                # Use Cut Cross Entropy for memory efficiency
                ce_loss = self._compute_cce_loss(model, outputs, labels, num_items_in_batch)
            else:
                # Standard PyTorch cross entropy with proper gradient accumulation handling
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                if num_items_in_batch is not None:
                    # Use sum reduction and divide by total valid tokens across accumulated batches
                    ce_loss = F.cross_entropy(
                        shift_logits.view(-1, vocab_size),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    ce_loss = ce_loss / num_items_in_batch
                else:
                    # Fallback to mean reduction (single batch or no accumulation info)
                    ce_loss = F.cross_entropy(
                        shift_logits.view(-1, vocab_size),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )

        # STEP 2: If no distillation, return CE loss only
        if self.distill_loss_fn is None or soft_targets is None:
            return (ce_loss, outputs) if return_outputs else ce_loss

        # STEP 3: Compute KL loss on move position
        if self.move_distill_mode == "move_prefix":
            tokenizer = getattr(self, "processing_class", None)
            if tokenizer is None:
                raise ValueError("Trainer is missing tokenizer/processing_class (required for move_prefix distillation).")

            temperature = float(getattr(self.distill_loss_fn, "temperature", 1.0) or 1.0)
            loss_terms: List[torch.Tensor] = []
            move_pos_found_ratio = None
            mapped_prob_ratios = []
            mapped_move_fracs = []
            student_logits = None

            if move_positions is not None:
                move_positions = move_positions.to(device)
                valid_mask = (move_positions >= 0) & (move_positions < seq_len)
                move_pos_found_ratio = valid_mask.float().mean().item()

                if batch_size == 1 and move_positions.numel() != batch_size:
                    # Packed batch: all positions live in batch item 0.
                    logits_view = logits[0]
                    for idx, is_valid in enumerate(valid_mask.tolist()):
                        if not is_valid:
                            continue
                        soft_target = soft_targets[idx]
                        if not soft_target:
                            continue
                        best_move = max(soft_target.items(), key=lambda x: x[1])[0]
                        best_tokens = tokenizer.encode(best_move, add_special_tokens=False)
                        if not best_tokens:
                            continue

                        token_cache = {
                            move: tokenizer.encode(move, add_special_tokens=False)
                            for move in soft_target.keys()
                        }

                        base_pos = int(move_positions[idx])
                        for k in range(len(best_tokens)):
                            pos = base_pos + k
                            if pos >= logits_view.shape[0]:
                                break

                            prefix = best_tokens[:k]
                            next_token_probs: Dict[int, float] = {}
                            total = 0.0
                            for move, prob in soft_target.items():
                                seq = token_cache.get(move) or []
                                if len(seq) <= k or seq[:k] != prefix:
                                    continue
                                next_token = int(seq[k])
                                next_token_probs[next_token] = next_token_probs.get(next_token, 0.0) + float(prob)
                                total += float(prob)

                            if total <= 0.0:
                                break

                            token_ids = torch.tensor(list(next_token_probs.keys()), device=device, dtype=torch.long)
                            teacher = torch.tensor(list(next_token_probs.values()), device=device, dtype=torch.float32)
                            teacher = teacher / teacher.sum()

                            log_probs = F.log_softmax(logits_view[pos].float() / temperature, dim=-1)
                            loss_terms.append(-(teacher * log_probs.gather(0, token_ids)).sum())
                else:
                    # Standard batch: one sample per batch row.
                    for idx, is_valid in enumerate(valid_mask.tolist()):
                        if not is_valid:
                            continue
                        soft_target = soft_targets[idx]
                        if not soft_target:
                            continue
                        best_move = max(soft_target.items(), key=lambda x: x[1])[0]
                        best_tokens = tokenizer.encode(best_move, add_special_tokens=False)
                        if not best_tokens:
                            continue

                        token_cache = {
                            move: tokenizer.encode(move, add_special_tokens=False)
                            for move in soft_target.keys()
                        }

                        base_pos = int(move_positions[idx])
                        for k in range(len(best_tokens)):
                            pos = base_pos + k
                            if pos >= seq_len:
                                break

                            prefix = best_tokens[:k]
                            next_token_probs: Dict[int, float] = {}
                            total = 0.0
                            for move, prob in soft_target.items():
                                seq = token_cache.get(move) or []
                                if len(seq) <= k or seq[:k] != prefix:
                                    continue
                                next_token = int(seq[k])
                                next_token_probs[next_token] = next_token_probs.get(next_token, 0.0) + float(prob)
                                total += float(prob)

                            if total <= 0.0:
                                break

                            token_ids = torch.tensor(list(next_token_probs.keys()), device=device, dtype=torch.long)
                            teacher = torch.tensor(list(next_token_probs.values()), device=device, dtype=torch.float32)
                            teacher = teacher / teacher.sum()

                            log_probs = F.log_softmax(logits[idx, pos].float() / temperature, dim=-1)
                            loss_terms.append(-(teacher * log_probs.gather(0, token_ids)).sum())

            if loss_terms:
                kl_loss = torch.stack(loss_terms).mean()
                if temperature != 1.0:
                    kl_loss = kl_loss * (temperature ** 2)
            else:
                kl_loss = torch.tensor(0.0, device=device)

            loss_dict = {"soft_loss": float(kl_loss.detach().cpu()), "hard_loss": 0.0}
        else:
            # move_token mode: KL over a single move token.
            # Create teacher probability tensors
            teacher_probs_batch = []
            hard_targets_batch = []
            mapped_prob_ratios = []
            mapped_move_fracs = []

            for i, soft_target in enumerate(soft_targets):
                if soft_target is None:
                    # Fallback: uniform distribution
                    teacher_probs = torch.ones(vocab_size, device=device) / vocab_size
                    mapped_prob_ratios.append(0.0)
                    mapped_move_fracs.append(0.0)
                else:
                    total_prob = sum(soft_target.values())
                    mapped_prob = sum(
                        prob for move, prob in soft_target.items()
                        if move in self.move_to_token_id
                    )
                    mapped_count = sum(
                        1 for move in soft_target
                        if move in self.move_to_token_id
                    )
                    mapped_prob_ratios.append(
                        (mapped_prob / total_prob) if total_prob > 0 else 0.0
                    )
                    mapped_move_fracs.append(
                        (mapped_count / len(soft_target)) if soft_target else 0.0
                    )
                    # Convert move_probs dict to tensor
                    teacher_probs = create_soft_target_tensor(
                        move_probs=soft_target,
                        vocab_size=vocab_size,
                        move_to_token_id=self.move_to_token_id,
                        device=device,
                    )
                teacher_probs_batch.append(teacher_probs)

                # Get hard target (most likely move in teacher distribution)
                if soft_target:
                    best_move = max(soft_target.items(), key=lambda x: x[1])[0]
                    if best_move in self.move_to_token_id:
                        hard_targets_batch.append(self.move_to_token_id[best_move])
                    else:
                        hard_targets_batch.append(0)
                else:
                    hard_targets_batch.append(0)

            teacher_probs_tensor = torch.stack(teacher_probs_batch)  # (batch, vocab)
            hard_targets_tensor = torch.tensor(hard_targets_batch, device=device)

            # Get student logits at move position
            # For causal LM, logits at position t predict token t+1.
            # We want logits at the <uci_move> tag position to predict the move token.
            move_pos_found_ratio = None
            if move_positions is not None:
                move_positions = move_positions.to(device)
                valid_mask = (move_positions >= 0) & (move_positions < seq_len)
                move_pos_found_ratio = valid_mask.float().mean().item()
                if valid_mask.any():
                    if batch_size == 1 and move_positions.numel() != batch_size:
                        # Packed batch: a single flattened sequence that contains multiple
                        # original samples. All move positions belong to batch item 0.
                        student_logits = logits[0, move_positions[valid_mask]]
                    else:
                        # Standard batch: one move position per batch item.
                        batch_indices = torch.arange(batch_size, device=device)[valid_mask]
                        student_logits = logits[batch_indices, move_positions[valid_mask]]
                    teacher_probs_tensor = teacher_probs_tensor[valid_mask]
                    hard_targets_tensor = hard_targets_tensor[valid_mask]
                else:
                    student_logits = None
            else:
                # Fallback: use last non-padding position (less reliable)
                attention_mask = inputs.get('attention_mask')
                if attention_mask is not None:
                    seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
                    batch_indices = torch.arange(batch_size, device=device)
                    student_logits = logits[batch_indices, seq_lengths]  # (batch, vocab)
                else:
                    student_logits = logits[:, -2, :]  # (batch, vocab)

            # Compute distillation loss (KL on move position)
            if student_logits is None:
                kl_loss = torch.tensor(0.0, device=device)
                loss_dict = {'soft_loss': 0.0, 'hard_loss': 0.0}
            else:
                kl_loss, loss_dict = self.distill_loss_fn(
                    student_logits=student_logits,
                    teacher_probs=teacher_probs_tensor,
                    hard_targets=hard_targets_tensor,
                )

        # STEP 4: Combine losses
        # ce_loss: Teaches thinking text via cross-entropy
        # kl_loss: Teaches move distribution via KL divergence
        # Formula: L_total = L_ce + kl_weight * L_kl
        total_loss = ce_loss + self.kl_weight * kl_loss

        # Apply sample weights if provided
        if sample_weights is not None:
            sample_weights = sample_weights.to(device)
            # Note: ChessDistillationLoss already returns mean loss,
            # so we'd need to modify it for proper weighting
            # For now, we log but don't apply weights to distillation loss

        # Compute and accumulate metrics
        self._step_count += 1
        self._accumulated_metrics['ce_loss'] += ce_loss.item()
        self._accumulated_metrics['soft_loss'] += loss_dict.get('soft_loss', 0)
        self._accumulated_metrics['hard_loss'] += loss_dict.get('hard_loss', 0)
        if move_pos_found_ratio is not None:
            self._accumulated_metrics['move_pos_found'] += move_pos_found_ratio
        if mapped_prob_ratios:
            self._accumulated_metrics['mapped_prob'] += sum(mapped_prob_ratios) / len(mapped_prob_ratios)
        if mapped_move_fracs:
            self._accumulated_metrics['mapped_move_frac'] += sum(mapped_move_fracs) / len(mapped_move_fracs)
        self._metric_count += 1

        # Compute additional metrics periodically
        if self._step_count % 100 == 0 and student_logits is not None:
            with torch.no_grad():
                metrics = compute_distillation_metrics(
                    student_logits, teacher_probs_tensor, hard_targets_tensor
                )
                self._accumulated_metrics['top1_agreement'] += metrics.get('top1_agreement', 0)
                self._accumulated_metrics['kl_divergence'] += metrics.get('kl_divergence', 0)

        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Add distillation metrics to logs."""
        if self._metric_count > 0:
            logs['distill/ce_loss'] = self._accumulated_metrics['ce_loss'] / self._metric_count
            logs['distill/soft_loss'] = self._accumulated_metrics['soft_loss'] / self._metric_count
            logs['distill/hard_loss'] = self._accumulated_metrics['hard_loss'] / self._metric_count
            logs['distill/move_pos_found'] = self._accumulated_metrics['move_pos_found'] / self._metric_count
            logs['distill/mapped_prob'] = self._accumulated_metrics['mapped_prob'] / self._metric_count
            logs['distill/mapped_move_frac'] = self._accumulated_metrics['mapped_move_frac'] / self._metric_count

            if self._accumulated_metrics['top1_agreement'] > 0:
                logs['distill/top1_agreement'] = self._accumulated_metrics['top1_agreement'] / (self._metric_count // 100 + 1)
                logs['distill/kl_divergence'] = self._accumulated_metrics['kl_divergence'] / (self._metric_count // 100 + 1)

            # Reset accumulators
            self._accumulated_metrics = {k: 0.0 for k in self._accumulated_metrics}
            self._metric_count = 0

        super().log(logs, start_time)

    def _compute_cce_loss(
        self,
        model,
        outputs,
        labels: torch.Tensor,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss using Cut Cross Entropy.

        CCE is memory-efficient as it computes loss without materializing
        the full logits tensor. Uses shift=1 for causal LM shifting.

        For proper gradient accumulation, we use sum reduction and divide by
        num_items_in_batch (total valid tokens across all accumulated batches).

        Args:
            model: The model (to get lm_head weights)
            outputs: Model outputs with hidden_states
            labels: Target labels (batch, seq)
            num_items_in_batch: Total valid tokens across accumulated batches
                               (for proper gradient accumulation normalization)

        Returns:
            Scalar CE loss
        """
        # Get hidden states from last layer
        hidden_states = outputs.hidden_states[-1]  # (batch, seq, hidden)
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Get classifier weights
        if hasattr(model, 'lm_head'):
            classifier = model.lm_head.weight
        elif hasattr(model, 'model') and hasattr(model.model, 'lm_head'):
            classifier = model.model.lm_head.weight
        else:
            # Fallback to standard CE if we can't find lm_head
            if not self._cce_warned:
                print("CCE: Could not find lm_head, falling back to standard CE")
                self._cce_warned = True
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            if num_items_in_batch is not None:
                loss = F.cross_entropy(
                    shift_logits.view(-1, logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                return loss / num_items_in_batch
            else:
                return F.cross_entropy(
                    shift_logits.view(-1, logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

        try:
            # CCE with shift=1 handles causal LM shifting internally
            # This avoids allocating the shifted embeddings matrix
            per_token_loss = linear_cross_entropy(
                hidden_states,  # (batch, seq, hidden) - NOT pre-shifted
                classifier,     # (vocab, hidden)
                labels,         # (batch, seq) - NOT pre-shifted
                shift=1,        # CCE handles shifting internally
                ignore_index=-100,
                reduction="none",
            )

            # Reshape to (batch, seq-1)
            per_token_loss = per_token_loss.view(batch_size, seq_len - 1)

            # Create mask for valid tokens (not -100 after shift)
            shift_labels = labels[..., 1:].contiguous()
            valid_mask = (shift_labels != -100)

            # Sum the losses over valid tokens
            loss_sum = (per_token_loss * valid_mask.float()).sum()

            # Normalize by num_items_in_batch if provided (for gradient accumulation)
            # Otherwise, normalize by valid tokens in this batch
            if num_items_in_batch is not None:
                ce_loss = loss_sum / num_items_in_batch
            else:
                valid_count = valid_mask.sum().clamp(min=1)
                ce_loss = loss_sum / valid_count

            return ce_loss

        except Exception as e:
            # Fallback to standard CE if CCE fails
            if not self._cce_warned:
                print(f"CCE computation failed: {e}, using standard CE")
                self._cce_warned = True

            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            if num_items_in_batch is not None:
                loss = F.cross_entropy(
                    shift_logits.view(-1, logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                return loss / num_items_in_batch
            else:
                return F.cross_entropy(
                    shift_logits.view(-1, logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )


# ============================================================================
# Helper Functions
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_global_step_from_checkpoint(checkpoint_path: Path) -> Optional[int]:
    """
    Read `global_step` from a Hugging Face Trainer checkpoint directory.

    Returns `None` when `trainer_state.json` is missing or malformed.
    """

    state_path = checkpoint_path / "trainer_state.json"
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    global_step = state.get("global_step")
    if isinstance(global_step, int):
        return global_step
    return None


def validate_trainer_checkpoint(checkpoint_path: Path) -> List[str]:
    """
    Return a list of missing files required for full Trainer resume.

    Hugging Face Trainer resumes *model + optimizer + scheduler + state* when the
    checkpoint directory contains these artifacts.

    This does not validate shard completeness; it is a quick sanity check to
    catch the most common "resume didn't work" causes (wrong path, partial copy,
    or `save_only_model=True`).
    """

    required = [
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
    ]
    missing = [name for name in required if not (checkpoint_path / name).exists()]
    return missing


def setup_model_and_tokenizer(
    config: Dict[str, Any],
    use_liger: bool = True,
    use_torch_compile: bool = False,
):
    """Setup model and tokenizer with optimizations."""
    model_config = config.get('model', {})
    training_config = config.get('training', {})

    model_name = model_config.get('name', 'Qwen/Qwen3-0.6B')
    print(f"Loading model: {model_name}")

    # Apply Liger kernels BEFORE loading model
    if use_liger and LIGER_AVAILABLE:
        apply_liger_kernels_no_ce(model_name)

    dtype_str = model_config.get('torch_dtype', model_config.get('dtype', 'bfloat16'))
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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    if hasattr(model, 'generation_config'):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    # Apply torch.compile for faster training
    if use_torch_compile:
        print("Applying torch.compile to model...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile applied successfully")
        except Exception as e:
            print(f"Warning: torch.compile failed ({e}), continuing without compilation")

    return model, tokenizer


def log_tokenizer_stats(tokenizer, all_uci_moves: List[str], sample_moves: Optional[List[str]] = None) -> None:
    """Log how the tokenizer splits UCI moves and tag-wrapped moves."""
    if sample_moves is None:
        sample_moves = ["e2e4", "g1f3", "e7e8q", "a2a1n"]

    single_token = 0
    for move in all_uci_moves:
        if len(tokenizer.encode(move, add_special_tokens=False)) == 1:
            single_token += 1

    total_moves = len(all_uci_moves)
    print(f"UCI move single-token coverage: {single_token}/{total_moves}")

    tag_sample = "<uci_move>e2e4</uci_move>"
    tag_ids = tokenizer.encode(tag_sample, add_special_tokens=False)
    tag_tokens = tokenizer.convert_ids_to_tokens(tag_ids)
    print(f"Tokenization sample: {tag_sample}")
    print(f"  ids: {tag_ids}")
    print(f"  tokens: {tag_tokens}")

    for move in sample_moves:
        move_ids = tokenizer.encode(move, add_special_tokens=False)
        move_tokens = tokenizer.convert_ids_to_tokens(move_ids)
        print(f"Move tokenization: {move}")
        print(f"  ids: {move_ids}")
        print(f"  tokens: {move_tokens}")


def load_distillation_dataset(
    config: Dict[str, Any],
    preprocessed_path: Optional[str] = None,
    streaming: bool = False,
):
    """Load distillation dataset."""
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    eval_size = training_config.get('eval_size', 1000)

    if preprocessed_path and Path(preprocessed_path).exists():
        print(f"Loading preprocessed distillation dataset from {preprocessed_path}")
        dataset = load_from_disk(preprocessed_path)

        # Split into train/eval
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
        print("Streaming mode: Loading raw positions for on-the-fly Stockfish analysis")
        return load_streaming_dataset(config)

    raise ValueError(
        f"No dataset found at {preprocessed_path}. "
            "Run preprocessing first: python distill/preprocess.py"
    )


def load_streaming_dataset(config: Dict[str, Any]):
    """
    Load TRUE streaming dataset using IterableDataset.

    Pattern from sft/train.py:
    - Training: IterableDataset (true streaming, infinite data)
    - Eval: Small fixed Dataset (consistent evaluation)

    Data is generated on-the-fly - never materializes in memory.
    Stockfish analysis happens per-batch in the collator.
    """
    from src.utils.data_processing import (
        create_streaming_dataset,
        stream_game_positions,
        stream_puzzle_positions,
    )

    data_config = config.get('data', {})
    training_config = config.get('training', {})
    eval_size = training_config.get('eval_size', 1000)
    elo_weights = config.get('elo_weights', None)

    print("\n" + "=" * 60)
    print("TRUE STREAMING MODE (IterableDataset)")
    print("=" * 60)
    print("  - Training data generated on-the-fly, NOT loaded into memory")
    print("  - Stockfish analysis happens per-batch in collator")
    print("  - Infinite data stream from Lichess games + puzzles")

    # Training: IterableDataset for true streaming
    train_dataset = create_streaming_dataset(
        games_ratio=data_config.get('games_ratio', 0.7),
        config=config,
        seed=training_config.get('seed', 42),
    )

    print("Streaming train dataset ready")
    print(f"  Games ratio: {data_config.get('games_ratio', 0.7)}")
    print(f"  Shuffle buffer: {data_config.get('shuffle_buffer_size', 10000)}")

    # Eval: Small fixed Dataset for consistent evaluation
    eval_dataset = None
    if eval_size > 0:
        print(f"\nCreating eval dataset ({eval_size} examples)...")
        eval_examples = []
        eval_games_examples = []
        eval_puzzles_examples = []

        games_ratio = data_config.get('games_ratio', 0.7)
        eval_games = int(eval_size * games_ratio)
        eval_puzzles = eval_size - eval_games

        if eval_games > 0:
            for pos in stream_game_positions(
                dataset_name=data_config.get('games_dataset', 'Lichess/standard-chess-games'),
                min_elo=data_config.get('min_elo', 1200),
                elo_weights=elo_weights,
                sample_rate=data_config.get('sample_rate', 0.3),
                skip_first_moves=data_config.get('skip_first_moves', 4),
                skip_last_moves=data_config.get('skip_last_moves', 2),
                config=config,
                seed=training_config.get('seed', 42) + 1000,
            ):
                eval_examples.append(pos)
                eval_games_examples.append(pos)
                if len(eval_examples) >= eval_games:
                    break

        if eval_puzzles > 0:
            puzzle_count = 0
            for pos in stream_puzzle_positions(
                dataset_name=data_config.get('puzzles_dataset', 'Lichess/chess-puzzles'),
                min_rating=data_config.get('min_puzzle_rating', 1000),
                max_rating=data_config.get('max_puzzle_rating', 2500),
                config=config,
                seed=training_config.get('seed', 42) + 2000,
            ):
                eval_examples.append(pos)
                eval_puzzles_examples.append(pos)
                puzzle_count += 1
                if puzzle_count >= eval_puzzles:
                    break

        rng = random.Random(training_config.get('seed', 42) + 3000)
        rng.shuffle(eval_examples)

        eval_datasets = {'total': Dataset.from_list(eval_examples)}
        if eval_games_examples:
            eval_datasets['games'] = Dataset.from_list(eval_games_examples)
        if eval_puzzles_examples:
            eval_datasets['puzzles'] = Dataset.from_list(eval_puzzles_examples)

        eval_dataset = eval_datasets
        print(f"Eval dataset: {len(eval_examples)} examples (fixed, in memory)")
        print(f"  Games: {eval_games}, Puzzles: {eval_puzzles}")

    print("=" * 60)

    return train_dataset, eval_dataset


def create_distillation_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    config: Dict[str, Any],
    distill_loss_fn: ChessDistillationLoss,
    move_to_token_id: Dict[str, int],
    streaming: bool = False,
    teacher: Optional[StockfishTeacher] = None,
    chess_eval_callback: Optional[TrainerCallback] = None,
    padding_free: bool = False,
):
    """Create the distillation trainer."""
    training_config = config.get('training', {})
    model_config = config.get('model', {})
    formatting_config = config.get('formatting', {})
    reasoning_trace_config = config.get('reasoning_trace', {})
    trace_enabled = reasoning_trace_config.get('enabled', False)
    force_best_move = reasoning_trace_config.get('always_choose_best_move', False)
    reasoning_trace_generator = None
    if trace_enabled:
        reasoning_trace_generator = ReasoningTraceGenerator(
            reasoning_trace_config,
            tokenizer=tokenizer,
        )
        trace_status = reasoning_trace_generator.status()
        opening_detail = ""
        if not trace_status.get("opening_available"):
            opening_detail = trace_status.get("opening_error") or "unavailable"
        tablebase_detail = ""
        if not trace_status.get("tablebase_available"):
            tablebase_detail = trace_status.get("tablebase_error") or "unavailable"
        print(
            "Reasoning trace enabled: "
            f"opening={trace_status.get('opening_available')}"
            + (f" ({opening_detail})" if opening_detail else "")
            + ", "
            f"tablebase={trace_status.get('tablebase_available')}"
            + (f" ({tablebase_detail})" if tablebase_detail else "")
            + ", "
            f"force_best={force_best_move}"
        )
    distill_config = config.get('distillation', {})
    stockfish_config = config.get('stockfish', {})

    # Training arguments - handle streaming mode differently
    # IterableDataset requires max_steps, not num_train_epochs
    if streaming:
        max_steps = training_config.get('max_steps', 10000)
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0 for streaming mode. Use --max-steps or set in config.")
        num_train_epochs = 1  # Ignored for IterableDataset
    else:
        max_steps = training_config.get('max_steps', -1)
        num_train_epochs = training_config.get('num_train_epochs', 3)

    eval_strategy = training_config.get('eval_strategy', 'steps') if eval_dataset is not None else 'no'
    metric_for_best_model = None
    if eval_strategy != 'no':
        if isinstance(eval_dataset, dict):
            if 'total' in eval_dataset:
                metric_for_best_model = 'eval_total_loss'
            else:
                first_key = next(iter(eval_dataset.keys()))
                metric_for_best_model = f"eval_{first_key}_loss"
        else:
            metric_for_best_model = 'eval_loss'

    training_args = TrainingArguments(
        output_dir=training_config.get('output_dir', './outputs/chess-distill'),

        num_train_epochs=num_train_epochs,
        max_steps=max_steps,

        per_device_train_batch_size=training_config.get('per_device_train_batch_size', 4),
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 8),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 8),

        learning_rate=training_config.get('learning_rate', 2e-5),
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
        warmup_ratio=training_config.get('warmup_ratio', 0.1),
        weight_decay=training_config.get('weight_decay', 0.01),
        max_grad_norm=training_config.get('max_grad_norm', 1.0),

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

        # Streaming: num_workers=0 (Stockfish in main process), but keep pin_memory
        dataloader_num_workers=0 if streaming else training_config.get('dataloader_num_workers', 0),
        dataloader_pin_memory=True,  # Keep enabled for performance

        gradient_checkpointing=training_config.get('gradient_checkpointing', True),

        report_to=training_config.get('report_to', 'wandb'),
        run_name=training_config.get('run_name', 'chess-policy-distill'),

        remove_unused_columns=False,

        load_best_model_at_end=eval_strategy != 'no',
        metric_for_best_model=metric_for_best_model,

        use_liger_kernel=False,  # We apply manually

        push_to_hub=False,
        include_tokens_per_second=True,

        # When resuming from checkpoint, Trainer can spend a long time "skipping"
        # batches to reach the previous step. With streaming + on-the-fly Stockfish,
        # that skip work is wasted compute. Prefer continuing at the correct step
        # but with new streamed samples.
        ignore_data_skip=True if streaming else training_config.get("ignore_data_skip", False),

        # Disable token counting for streaming - soft_targets are non-tensor
        # and cause Accelerate concatenation errors
        include_num_input_tokens_seen=False if streaming else True,

        # For streaming with non-tensor batch fields (soft_targets), disable
        # dispatch_batches to prevent Accelerate concatenation errors
        accelerator_config=AcceleratorConfig(
            dispatch_batches=False,
        ) if streaming else None,
    )

    # Data collator - use streaming collator with teacher, or precomputed
    pad_to_multiple_of = training_config.get("pad_to_multiple_of", 8)
    return_flash_attn_kwargs = training_config.get("return_flash_attn_kwargs", True)

    if padding_free:
        print("Padding-free packing enabled (FlashAttention2 + position_ids).")
        pad_to_multiple_of = None

    if streaming and teacher is not None:
        print("Using streaming collator with on-the-fly Stockfish analysis")
        print(f"  Note: Each batch requires ~{training_config.get('per_device_train_batch_size', 4) * 80}ms for Stockfish analysis")
        data_collator = DistillationCollator(
            tokenizer=tokenizer,
            teacher=teacher,
            max_length=model_config.get('max_seq_length', 2048),
            pad_to_multiple_of=pad_to_multiple_of,
            max_display_moves=formatting_config.get('max_display_moves', 5),
            randomize_order=formatting_config.get('randomize_order', True),
            include_soft_targets=True,
            pv_length=formatting_config.get('pv_length', 5),
            include_board=formatting_config.get('include_board', True),
            source_overrides=stockfish_config.get('source_overrides'),
            reasoning_trace_generator=reasoning_trace_generator,
            force_best_move=force_best_move,
            padding_free=padding_free,
            return_flash_attn_kwargs=return_flash_attn_kwargs,
            seed=training_config.get('seed', 42),
        )
    else:
        data_collator = PrecomputedDistillationCollator(
            tokenizer=tokenizer,
            max_length=model_config.get('max_seq_length', 2048),
            pad_to_multiple_of=pad_to_multiple_of,
            max_display_moves=formatting_config.get('max_display_moves', 5),
            randomize_order=formatting_config.get('randomize_order', True),
            pv_length=formatting_config.get('pv_length', 5),
            include_board=formatting_config.get('include_board', True),
            reasoning_trace_generator=reasoning_trace_generator,
            rerandomize_reasoning_trace=reasoning_trace_config.get("rerandomize_in_collator", False),
            force_best_move=force_best_move,
            padding_free=padding_free,
            return_flash_attn_kwargs=return_flash_attn_kwargs,
            seed=training_config.get('seed', 42),
        )

    callbacks = []
    if chess_eval_callback is not None:
        callbacks.append(chess_eval_callback)
        print("Chess evaluation callback enabled")

    trainer = DistillationTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        distill_loss_fn=distill_loss_fn,
        move_to_token_id=move_to_token_id,
        move_distill_mode=distill_config.get("move_distill_mode", "move_token"),
        ce_loss_enabled=distill_config.get('ce_loss_enabled', True),
        kl_weight=distill_config.get('kl_weight', 1.0),
        use_cce=distill_config.get('use_cce', True),
        callbacks=callbacks if callbacks else None,
    )

    return trainer


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def main():
    parser = argparse.ArgumentParser(description='Chess LLM Policy Distillation Training')
    parser.add_argument('--config', type=str, default='configs/distill/config_distill.yaml',
                        help='Path to configuration file')
    parser.add_argument('--preprocessed_path', type=str, default=None,
                        help='Path to preprocessed distillation dataset')
    parser.add_argument('--streaming', action='store_true',
                        help='Use streaming mode with on-the-fly Stockfish analysis')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode with small dataset')
    parser.add_argument('--no-liger', action='store_true',
                        help='Disable Liger kernel optimizations')
    parser.add_argument('--torch-compile', action='store_true',
                        help='Enable torch.compile for faster training')
    parser.add_argument('--unpad', action='store_true',
                        help='Enable padding-free packing for FlashAttention2 (like TRL padding_free)')
    parser.add_argument('--no-cce', action='store_true',
                        help='Disable Cut Cross Entropy (use standard PyTorch CE)')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Max training steps (required for streaming mode)')

    args = parser.parse_args()

    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Handle --max-steps for streaming mode
    if args.max_steps is not None:
        config['training']['max_steps'] = args.max_steps

    # Debug mode overrides
    if args.debug:
        print("DEBUG MODE: Using small dataset and fewer steps")
        config['training']['max_steps'] = 100
        config['training']['logging_steps'] = 10
        config['training']['save_steps'] = 50
        config['training']['eval_steps'] = 50
        config['training']['report_to'] = 'none'
        config['training']['per_device_train_batch_size'] = 2
        config['training']['gradient_accumulation_steps'] = 2

    resume_checkpoint = args.resume
    if resume_checkpoint and not Path(resume_checkpoint).exists():
        print(f"Warning: Checkpoint {resume_checkpoint} not found")
        resume_checkpoint = None

    stream_skip_examples = 0

    if resume_checkpoint:
        checkpoint_path = Path(resume_checkpoint)
        global_step = load_global_step_from_checkpoint(checkpoint_path)
        missing = validate_trainer_checkpoint(checkpoint_path)
        print(f"Resume checkpoint: {checkpoint_path}")
        print(f"  trainer_state global_step: {global_step if global_step is not None else 'unknown'}")
        if missing:
            print(
                "Warning: checkpoint is missing Trainer state files: "
                + ", ".join(missing)
                + ". Resume may restart optimizer/scheduler and step counters."
            )

    # Get config sections
    training_config = config.get('training', {})
    distill_config = config.get('distillation', {})
    cce_config = config.get('cce', {})
    stockfish_config = config.get('stockfish', {})
    use_liger = distill_config.get('use_liger', True) and not args.no_liger
    use_cce = cce_config.get('enabled', True) and not args.no_cce and CCE_AVAILABLE
    loss_type = distill_config.get('loss_type', 'kl')
    ce_loss_enabled = distill_config.get('ce_loss_enabled', True)
    kl_weight = distill_config.get('kl_weight', 1.0)
    prob_mode = stockfish_config.get('prob_mode', 'cp')

    print("\n" + "=" * 60)
    print("DISTILLATION TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Loss structure: L_total = L_ce + {kl_weight} * L_kl")
    print(f"CE loss (thinking): {'ENABLED' if ce_loss_enabled else 'DISABLED'}")
    print(f"Cut Cross Entropy: {'ENABLED' if use_cce else 'DISABLED'}")
    print(f"KL loss (move dist): {loss_type.upper()} divergence, weight={kl_weight}")
    print(f"Move distill mode: {distill_config.get('move_distill_mode', 'move_token')}")
    print(f"Liger Kernel: {'enabled' if use_liger and LIGER_LOSS_AVAILABLE else 'disabled'}")
    print(f"Temperature: {distill_config.get('temperature', 1.0)}")
    if prob_mode == 'wdl':
        print(f"Stockfish WDL temperature: {stockfish_config.get('wdl_temperature', 1.0)}")
    else:
        print(f"Stockfish temp (CP->prob): {distill_config.get('stockfish_temperature', 100.0)}")
    print(f"Floor probability: {distill_config.get('min_probability', 0.001)}")
    print(f"Flash Attention: {config.get('model', {}).get('attn_implementation', 'flash_attention_2')}")
    use_torch_compile = args.torch_compile or training_config.get('torch_compile', False)
    padding_free = (
        args.unpad
        or training_config.get("padding_free", False)
        or training_config.get('use_unpad', False)
    )
    print(f"torch.compile: {'ENABLED' if use_torch_compile else 'disabled'}")
    print(f"Padding-free packing: {'ENABLED' if padding_free else 'disabled'}")
    print("=" * 60)

    if args.streaming and resume_checkpoint:
        global_step = load_global_step_from_checkpoint(Path(resume_checkpoint))
        if global_step is None:
            print(
                "Warning: streaming resume could not read checkpoint global_step; "
                "stream will restart near the beginning."
            )
        else:
            per_device_batch = int(training_config.get("per_device_train_batch_size", 4))
            grad_accum = int(training_config.get("gradient_accumulation_steps", 1))
            stream_skip_examples = global_step * per_device_batch * grad_accum
            if stream_skip_examples > 0:
                print(
                    f"Streaming resume: will skip {stream_skip_examples:,} streamed examples "
                    f"(global_step={global_step}, batch={per_device_batch}, grad_accum={grad_accum})."
                )

    # Store in config for trainer to use
    distill_config['ce_loss_enabled'] = ce_loss_enabled
    distill_config['kl_weight'] = kl_weight
    distill_config['use_cce'] = use_cce

    # Setup model and tokenizer
    print("\nSetting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(
        config,
        use_liger=LIGER_AVAILABLE and not args.no_liger,
        use_torch_compile=use_torch_compile,
    )

    actual_attn = getattr(model.config, "_attn_implementation", None)
    if padding_free and actual_attn is not None and actual_attn != "flash_attention_2":
        print(
            f"Warning: padding-free packing requires flash_attention_2, but model uses {actual_attn}. "
            "Disabling padding-free packing."
        )
        padding_free = False

    # Add chess tokens and build move-to-token mapping (if using move-vocab distillation).
    tokenizer_config = config.get("tokenizer", {})
    mode_str = tokenizer_config.get("chess_mode")
    if not mode_str:
        mode_str = "tags_and_moves" if distill_config.get("add_uci_move_tokens", True) else "tags_only"
    if mode_str not in ("tags_only", "tags_and_moves"):
        raise ValueError(f"tokenizer.chess_mode must be 'tags_only' or 'tags_and_moves' (got {mode_str!r})")
    mode: ChessTokenizerMode = mode_str
    add_think_tags = bool(tokenizer_config.get("add_think_tags", True))

    all_uci_moves = generate_all_uci_moves() if mode == "tags_and_moves" else None
    added = add_chess_tokens(
        tokenizer=tokenizer,
        model=model,
        mode=mode,
        add_think_tags=add_think_tags,
        all_uci_moves=all_uci_moves,
    )
    if added['special_tokens'] or added['move_tokens']:
        print(
            f"Added tokens: {added['special_tokens']} special, "
            f"{added['move_tokens']} UCI move tokens (mode={mode})"
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print_gpu_memory()

    if mode == "tags_and_moves":
        total_moves = len(all_uci_moves or [])
        move_to_token_id = build_move_to_token_id(tokenizer=tokenizer, all_uci_moves=all_uci_moves)
        print(f"Move vocab coverage: {len(move_to_token_id)}/{total_moves}")
    else:
        move_to_token_id = {}
        print("Move vocab coverage: 0 (tags_only mode)")

    if distill_config.get('log_tokenizer_stats', False):
        log_tokenizer_stats(tokenizer, generate_all_uci_moves())

    # Create distillation loss function
    # alpha=1.0 means pure KL loss (no additional CE on move token)
    # The CE on thinking is handled separately in DistillationTrainer
    distill_loss_fn = ChessDistillationLoss(
        alpha=1.0,  # Pure KL - CE weighting is handled at trainer level via kl_weight
        temperature=distill_config.get('temperature', 1.0),
        label_smoothing=distill_config.get('label_smoothing', 0.0),
        use_liger=use_liger,
        loss_type=loss_type,
    )
    print(f"Distillation loss initialized (use_liger={distill_loss_fn.use_liger}, type={loss_type})")

    # Load dataset
    print("\nLoading dataset...")
    preprocessed_path = args.preprocessed_path or config.get('data', {}).get('preprocessed_path')
    train_dataset, eval_dataset = load_distillation_dataset(
        config,
        preprocessed_path=preprocessed_path,
        streaming=args.streaming,
    )

    if args.streaming and stream_skip_examples > 0:
        if hasattr(train_dataset, "skip"):
            train_dataset = train_dataset.skip(stream_skip_examples)
        else:
            print(
                "Warning: train_dataset does not support `.skip()`. "
                "Streaming resume cannot advance the stream; you may repeat early samples."
            )

    if hasattr(train_dataset, '__len__'):
        print(f"Train dataset size: {len(train_dataset):,}")
    if eval_dataset is not None:
        if isinstance(eval_dataset, dict):
            for name, ds in eval_dataset.items():
                print(f"Eval dataset {name} size: {len(ds):,}")
        else:
            print(f"Eval dataset size: {len(eval_dataset):,}")

    # Setup chess evaluation callback (optional)
    chess_eval_callback = None
    if eval_dataset is not None and not args.debug:
        from src.utils.chess_eval_callback import FastChessEvalCallback, prepare_eval_positions

        eval_config = config.get("evaluation", {})
        training_config = config.get("training", {})

        stockfish_path = None
        possible_paths = [
            eval_config.get("stockfish_path"),
            shutil.which("stockfish"),
            "/usr/bin/stockfish",
            "/usr/games/stockfish",
            "/usr/local/bin/stockfish",
            "/opt/homebrew/bin/stockfish",
        ]
        for path in possible_paths:
            if path and Path(path).exists():
                stockfish_path = path
                break

        eval_positions = prepare_eval_positions(
            eval_dataset,
            max_positions=training_config.get("chess_eval_positions", 500),
        )

        if eval_positions:
            chess_eval_callback = FastChessEvalCallback(
                eval_positions=eval_positions,
                tokenizer=tokenizer,
                eval_batch_size=training_config.get("chess_eval_batch_size", 32),
                max_new_tokens=training_config.get("chess_eval_max_new_tokens", 128),
                max_total_tokens=training_config.get(
                    "chess_eval_max_total_tokens",
                    config.get("model", {}).get("max_seq_length", 2048),
                ),
                do_sample=training_config.get("chess_eval_do_sample", False),
                temperature=training_config.get("chess_eval_temperature"),
                top_p=training_config.get("chess_eval_top_p"),
                top_k=training_config.get("chess_eval_top_k"),
                min_p=training_config.get("chess_eval_min_p"),
                seed=training_config.get("seed", 42),
                eval_every_n_steps=training_config.get("chess_eval_steps", 500),
                stockfish_path=stockfish_path,
                stockfish_workers=eval_config.get("stockfish_workers", 8),
                stockfish_depth=eval_config.get("stockfish_depth", 10),
                print_samples=training_config.get("chess_eval_print_samples", 0),
                print_max_chars=training_config.get("chess_eval_print_max_chars", 600),
            )
            print(f"Chess eval callback: {len(eval_positions)} positions")
            if stockfish_path:
                print(f"  Stockfish: {stockfish_path} (ACPL enabled)")
            else:
                print("  Stockfish: not found (ACPL disabled)")

    # Initialize StockfishTeacher for streaming mode
    teacher = None
    if args.streaming:
        stockfish_config = config.get('stockfish', {})

        # Find Stockfish path
        sf_path = stockfish_config.get('path')
        if not sf_path:
            possible_paths = [
                shutil.which('stockfish'),
                '/usr/bin/stockfish',
                '/usr/games/stockfish',
                '/usr/local/bin/stockfish',
                '/opt/homebrew/bin/stockfish',
            ]
            for path in possible_paths:
                if path and Path(path).exists():
                    sf_path = path
                    break

        if not sf_path:
            raise FileNotFoundError(
                "Stockfish not found. Install with: sudo apt install stockfish (Linux) or brew install stockfish (macOS)"
            )

        print(f"\nInitializing StockfishTeacher...")
        print(f"  Path: {sf_path}")
        print(f"  Workers: {stockfish_config.get('num_workers', 8)}")
        print(f"  Threads/worker: {stockfish_config.get('threads_per_worker', 1)}")
        print(f"  Depth: {stockfish_config.get('depth', 12)}")
        if stockfish_config.get('time_limit_ms') is not None:
            print(f"  Time limit: {stockfish_config.get('time_limit_ms')}ms")
        if stockfish_config.get('nodes') is not None:
            print(f"  Nodes: {stockfish_config.get('nodes')}")
        print(f"  Top-k: {stockfish_config.get('top_k', 5)}")
        if stockfish_config.get('shallow_depth', 0):
            print(f"  Shallow depth: {stockfish_config.get('shallow_depth')}")
            if stockfish_config.get('shallow_max_moves') is not None:
                print(f"  Shallow max moves: {stockfish_config.get('shallow_max_moves')}")
        if stockfish_config.get('confirm_depth', 0):
            print(f"  Confirm depth: {stockfish_config.get('confirm_depth')}")
            print(f"  Confirm top-k: {stockfish_config.get('confirm_top_k', 0)}")
        print(f"  Prob mode: {stockfish_config.get('prob_mode', 'cp')}")
        if stockfish_config.get('prob_mode', 'cp') == 'wdl':
            print(f"  WDL temperature: {stockfish_config.get('wdl_temperature', 1.0)}")
        if stockfish_config.get('cache_size', 0):
            print(f"  Cache size: {stockfish_config.get('cache_size')}")
        if stockfish_config.get('syzygy_path'):
            print(f"  Syzygy path: {stockfish_config.get('syzygy_path')}")

        teacher = StockfishTeacher(
            stockfish_path=sf_path,
            num_workers=stockfish_config.get('num_workers', 8),
            depth=stockfish_config.get('depth', 12),
            top_k=stockfish_config.get('top_k', 5),
            hash_mb_per_worker=stockfish_config.get('hash_mb_per_worker', 64),
            temperature=distill_config.get('stockfish_temperature', 100.0),
            min_probability=distill_config.get('min_probability', 0.001),
            threads_per_worker=stockfish_config.get('threads_per_worker', 1),
            time_limit_ms=stockfish_config.get('time_limit_ms'),
            nodes=stockfish_config.get('nodes'),
            shallow_depth=stockfish_config.get('shallow_depth', 0),
            shallow_max_moves=stockfish_config.get('shallow_max_moves'),
            confirm_depth=stockfish_config.get('confirm_depth', 0),
            confirm_top_k=stockfish_config.get('confirm_top_k', 0),
            prob_mode=stockfish_config.get('prob_mode', 'cp'),
            wdl_temperature=stockfish_config.get('wdl_temperature', 1.0),
            cache_size=stockfish_config.get('cache_size', 0),
            syzygy_path=stockfish_config.get('syzygy_path'),
        )
        print("StockfishTeacher created")

        # Pre-initialize workers and test with a simple position
        print("  Initializing Stockfish workers...")
        test_analysis = teacher.analyze_position(chess.STARTING_FEN)
        print(f"  Workers ready (test: best={test_analysis.best_move_san})")

    # Create trainer
    print("\nCreating distillation trainer...")
    trainer = create_distillation_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
        distill_loss_fn=distill_loss_fn,
        move_to_token_id=move_to_token_id,
        streaming=args.streaming,
        teacher=teacher,
        chess_eval_callback=chess_eval_callback,
        padding_free=padding_free,
    )

    # Start training
    print("\n" + "=" * 60)
    print("Starting distillation training...")
    print("=" * 60)
    print_gpu_memory()

    print("\nCalling trainer.train() - first batch may take time for Stockfish analysis...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print_gpu_memory()

    # Cleanup StockfishTeacher
    if teacher is not None:
        print("\nShutting down StockfishTeacher...")
        teacher.close()
        print("StockfishTeacher closed")

    # Save model
    output_dir = config.get('training', {}).get('output_dir', './outputs/chess-distill')
    final_path = f"{output_dir}-final"
    print(f"\nSaving model to {final_path}...")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print("Model saved successfully!")


if __name__ == "__main__":
    main()
