"""
Data Collator for Streaming Chess Distillation.

Performs on-the-fly Stockfish analysis during training, creating
soft targets for knowledge distillation.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import random
import threading
from queue import Queue
import time
import warnings

import chess

from .stockfish_teacher import StockfishTeacher, PositionAnalysis
from .formatting_distill import (
    position_to_messages_distill,
    create_distillation_example,
    DISTILLATION_PROMPT_TEMPLATE,
    DISTILLATION_PROMPT_TEMPLATE_NO_BOARD,
    DISTILLATION_RESPONSE_TEMPLATE,
)
from .chess_utils import render_board_utf, get_legal_moves_uci


@dataclass
class DistillationCollator:
    """
    Collator that performs on-the-fly Stockfish analysis.

    For each batch:
    1. Analyzes positions with Stockfish (parallelized)
    2. Generates training examples with randomized thinking
    3. Creates soft target distributions for distillation loss

    This enables streaming training without pre-computing all analyses.
    """

    tokenizer: Any
    teacher: StockfishTeacher
    max_length: int = 2048
    pad_to_multiple_of: int = 8
    max_display_moves: int = 5
    randomize_order: bool = True
    include_soft_targets: bool = True
    pv_length: int = 5
    include_board: bool = True
    source_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    seed: Optional[int] = None
    
    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self._call_count = 0
        self.uci_move_token_id = self._resolve_token_id("<uci_move>")
        self.uci_move_end_token_id = self._resolve_token_id("</uci_move>")
        if self.uci_move_token_id is None:
            warnings.warn(
                "Tokenizer does not contain <uci_move> token; move_positions will be -1."
            )

    def _resolve_token_id(self, token: str) -> Optional[int]:
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            return None
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        if unk_id is not None and token_id == unk_id:
            return None
        return token_id

    def _find_last_token(self, input_ids: List[int], token_id: Optional[int]) -> int:
        if token_id is None:
            return -1
        for idx in range(len(input_ids) - 1, -1, -1):
            if input_ids[idx] == token_id:
                return idx
        return -1
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples with on-the-fly analysis.
        
        Args:
            examples: List of position dicts with 'fen' and 'target_move_uci'
        
        Returns:
            Batch dict with input_ids, attention_mask, labels, and optionally soft_targets
        """
        self._call_count += 1

        # Vary the random seed per batch for different randomizations
        batch_rng = random.Random(self.seed + self._call_count if self.seed else None)

        # Extract FENs for batch analysis
        fens = [ex['fen'] for ex in examples]

        # Parallel Stockfish analysis
        analysis_overrides = None
        if self.source_overrides:
            analysis_overrides = []
            for ex in examples:
                source = ex.get('source')
                override = self.source_overrides.get(source) if source else None
                if override:
                    analysis_overrides.append(dict(override))
                else:
                    analysis_overrides.append(None)

        try:
            analyses = self.teacher.analyze_batch(
                fens,
                analysis_overrides=analysis_overrides,
            )
        except Exception as e:
            warnings.warn(f"Stockfish analysis failed: {e}. Using fallback.")
            analyses = [None] * len(examples)
        
        # Process each example
        batch_messages = []
        batch_soft_targets = []
        batch_target_info = []
        
        for ex, analysis in zip(examples, analyses):
            if analysis is None:
                # Fallback: create simple example without analysis
                messages = self._create_fallback_messages(ex)
                soft_targets = None
                target_info = {'cp_loss': 0, 'rank': 1}
            else:
                # Create full distillation example
                result = create_distillation_example(
                    fen=ex['fen'],
                    target_move_uci=ex['target_move_uci'],
                    analysis=analysis,
                    board_utf=ex.get('board_utf'),
                    max_display_moves=self.max_display_moves,
                    randomize_order=self.randomize_order,
                    pv_length=self.pv_length,
                    include_board=self.include_board,
                    rng=batch_rng,
                )
                
                messages = result['messages']
                soft_targets = result['move_probs']
                target_info = {
                    'cp_loss': result['target_move_cp_loss'],
                    'rank': result['target_move_rank'],
                    'prob': result['target_move_prob'],
                }
            
            batch_messages.append(messages)
            batch_soft_targets.append(soft_targets)
            batch_target_info.append(target_info)
        
        # Tokenize
        batch = self._tokenize_batch(batch_messages)
        
        # Add soft targets if requested
        if self.include_soft_targets:
            batch['soft_targets'] = batch_soft_targets
            batch['target_info'] = batch_target_info
        
        return batch
    
    def _create_fallback_messages(self, example: Dict[str, Any]) -> List[Dict[str, str]]:
        """Create messages without Stockfish analysis."""
        fen = example['fen']
        target_move = example['target_move_uci']
        
        board = chess.Board(fen)
        board_utf = example.get('board_utf') or render_board_utf(board)
        legal_moves = example.get('legal_moves_uci') or get_legal_moves_uci(board)
        
        user_content = DISTILLATION_PROMPT_TEMPLATE.format(
            fen=fen,
            legal_moves=legal_moves,
            board=board_utf,
        ) if self.include_board else DISTILLATION_PROMPT_TEMPLATE_NO_BOARD.format(
            fen=fen,
            legal_moves=legal_moves,
        )
        
        # Simple thinking without analysis
        thinking = f"Selecting move {target_move}."
        assistant_content = DISTILLATION_RESPONSE_TEMPLATE.format(
            thinking=thinking,
            move=target_move,
        )
        
        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    
    def _tokenize_batch(
        self,
        batch_messages: List[List[Dict[str, str]]],
    ) -> Dict[str, torch.Tensor]:
        """Tokenize messages with prompt masking."""
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_move_positions = []
        
        for messages in batch_messages:
            # Full conversation
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            # User part only (for masking)
            user_only = [messages[0]]
            prompt_text = self.tokenizer.apply_chat_template(
                user_only,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            full_tokens = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                return_tensors=None,
            )
            
            prompt_tokens = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_length,
                return_tensors=None,
            )
            
            input_ids = full_tokens['input_ids']
            attention_mask = full_tokens['attention_mask']
            
            # Create labels with prompt masking
            labels = input_ids.copy()
            prompt_length = len(prompt_tokens['input_ids'])
            
            for i in range(min(prompt_length, len(labels))):
                labels[i] = -100  # Ignore prompt tokens
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
            move_pos = self._find_last_token(
                input_ids[prompt_length:],
                self.uci_move_token_id
            )
            if move_pos >= 0:
                move_pos += prompt_length
            batch_move_positions.append(move_pos)
        
        # Pad to same length
        max_len = max(len(ids) for ids in batch_input_ids)
        
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) 
                       // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        pad_token_id = self.tokenizer.pad_token_id or 0
        
        for i in range(len(batch_input_ids)):
            pad_len = max_len - len(batch_input_ids[i])
            
            batch_input_ids[i] = batch_input_ids[i] + [pad_token_id] * pad_len
            batch_attention_mask[i] = batch_attention_mask[i] + [0] * pad_len
            batch_labels[i] = batch_labels[i] + [-100] * pad_len
        
        return {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long),
            'labels': torch.tensor(batch_labels, dtype=torch.long),
            'move_positions': torch.tensor(batch_move_positions, dtype=torch.long),
        }


@dataclass
class PrecomputedDistillationCollator:
    """
    Collator for pre-computed distillation data.

    Use this when you've already run Stockfish analysis and saved
    the results. Faster than on-the-fly but requires preprocessing.
    """

    tokenizer: Any
    max_length: int = 2048
    pad_to_multiple_of: int = 8
    max_display_moves: int = 5
    randomize_order: bool = True
    pv_length: int = 5
    include_board: bool = True
    seed: Optional[int] = None
    
    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self._call_count = 0
        self.uci_move_token_id = self._resolve_token_id("<uci_move>")
        self.uci_move_end_token_id = self._resolve_token_id("</uci_move>")
        if self.uci_move_token_id is None:
            warnings.warn(
                "Tokenizer does not contain <uci_move> token; move_positions will be -1."
            )

    def _resolve_token_id(self, token: str) -> Optional[int]:
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            return None
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        if unk_id is not None and token_id == unk_id:
            return None
        return token_id

    def _find_last_token(self, input_ids: List[int], token_id: Optional[int]) -> int:
        if token_id is None:
            return -1
        for idx in range(len(input_ids) - 1, -1, -1):
            if input_ids[idx] == token_id:
                return idx
        return -1
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate pre-computed examples.
        
        Expects examples to have 'messages', 'move_probs', and analysis metadata.
        """
        self._call_count += 1
        batch_rng = random.Random(self.seed + self._call_count if self.seed else None)
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_soft_targets = []
        batch_weights = []
        batch_move_positions = []
        
        for ex in examples:
            # Get or regenerate messages
            if 'messages' in ex and not self.randomize_order:
                messages = ex['messages']
            else:
                # Regenerate with new randomization
                messages = self._regenerate_messages(ex, batch_rng)
            
            # Tokenize with prompt masking
            tokens = self._tokenize_single(messages)

            batch_input_ids.append(tokens['input_ids'])
            batch_attention_mask.append(tokens['attention_mask'])
            batch_labels.append(tokens['labels'])
            batch_move_positions.append(tokens['move_position'])
            
            # Soft targets
            if 'move_probs' in ex:
                batch_soft_targets.append(ex['move_probs'])
            
            # Loss weight
            batch_weights.append(ex.get('loss_weight', 1.0))
        
        # Pad
        batch = self._pad_batch(
            batch_input_ids,
            batch_attention_mask,
            batch_labels,
        )

        batch['soft_targets'] = batch_soft_targets
        batch['sample_weights'] = torch.tensor(batch_weights, dtype=torch.float)
        batch['move_positions'] = torch.tensor(batch_move_positions, dtype=torch.long)

        return batch
    
    def _regenerate_messages(
        self,
        example: Dict[str, Any],
        rng: random.Random,
    ) -> List[Dict[str, str]]:
        """Regenerate messages with new randomization."""
        from .stockfish_teacher import MoveAnalysis, PositionAnalysis
        
        # Reconstruct analysis from stored data
        move_analyses = []
        if 'move_evaluations' in example:
            for mv in example['move_evaluations']:
                move_analyses.append(MoveAnalysis(
                    uci=mv['uci'],
                    san=mv['san'],
                    centipawn=mv['centipawn'],
                    cp_loss=mv.get('cp_loss', 0),
                    category=mv.get('category', 'unknown'),
                    mate_in=mv.get('mate_in'),
                    win_probability=mv.get('win_probability', 0.5),
                ))
        
        analysis = PositionAnalysis(
            fen=example['fen'],
            move_analyses=move_analyses,
            best_move_uci=example.get('best_move_uci', ''),
            best_move_san=example.get('best_move_san', ''),
            best_score_cp=example.get('best_score_cp', 0),
            best_pv=example.get('best_pv', []),
            move_probs=example.get('move_probs', {}),
            top_k_moves=example.get('top_k_moves', []),
        )

        position = {
            'fen': example['fen'],
            'target_move_uci': example['target_move_uci'],
            'legal_moves_uci': example.get('legal_moves_uci', ''),
            'board_utf': example.get('board_utf', ''),
        }

        result = position_to_messages_distill(
            position=position,
            analysis=analysis,
            max_display_moves=self.max_display_moves,
            randomize_order=self.randomize_order,
            pv_length=self.pv_length,
            include_board=self.include_board,
            rng=rng,
        )

        return result['messages']
    
    def _tokenize_single(
        self,
        messages: List[Dict[str, str]],
    ) -> Dict[str, List[int]]:
        """Tokenize a single example with prompt masking."""
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        user_only = [messages[0]]
        prompt_text = self.tokenizer.apply_chat_template(
            user_only,
            tokenize=False,
            add_generation_prompt=True
        )
        
        full_tokens = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        
        prompt_tokens = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        
        input_ids = full_tokens['input_ids']
        attention_mask = full_tokens['attention_mask']
        labels = input_ids.copy()

        prompt_length = len(prompt_tokens['input_ids'])
        for i in range(min(prompt_length, len(labels))):
            labels[i] = -100
        
        move_pos = self._find_last_token(
            input_ids[prompt_length:],
            self.uci_move_token_id
        )
        if move_pos >= 0:
            move_pos += prompt_length

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'move_position': move_pos,
        }
    
    def _pad_batch(
        self,
        batch_input_ids: List[List[int]],
        batch_attention_mask: List[List[int]],
        batch_labels: List[List[int]],
    ) -> Dict[str, torch.Tensor]:
        """Pad batch to same length."""
        max_len = max(len(ids) for ids in batch_input_ids)
        
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) 
                       // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        pad_token_id = self.tokenizer.pad_token_id or 0
        
        padded_input_ids = []
        padded_attention = []
        padded_labels = []
        
        for i in range(len(batch_input_ids)):
            pad_len = max_len - len(batch_input_ids[i])
            
            padded_input_ids.append(batch_input_ids[i] + [pad_token_id] * pad_len)
            padded_attention.append(batch_attention_mask[i] + [0] * pad_len)
            padded_labels.append(batch_labels[i] + [-100] * pad_len)
        
        return {
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(padded_attention, dtype=torch.long),
            'labels': torch.tensor(padded_labels, dtype=torch.long),
        }


class PrefetchingDistillationCollator:
    """
    Collator with prefetching for reduced latency.
    
    Prefetches Stockfish analyses for the next batch while
    the current batch is being processed by the GPU.
    """
    
    def __init__(
        self,
        tokenizer: Any,
        teacher: StockfishTeacher,
        prefetch_batches: int = 2,
        **kwargs
    ):
        self.base_collator = DistillationCollator(
            tokenizer=tokenizer,
            teacher=teacher,
            **kwargs
        )
        self.prefetch_batches = prefetch_batches
        self._prefetch_queue: Queue = Queue(maxsize=prefetch_batches)
        self._prefetch_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def start_prefetching(self, dataloader: DataLoader):
        """Start prefetch thread."""
        def prefetch_worker():
            for batch in dataloader:
                if self._stop_event.is_set():
                    break
                processed = self.base_collator(batch)
                self._prefetch_queue.put(processed)
            self._prefetch_queue.put(None)  # Sentinel
        
        self._prefetch_thread = threading.Thread(target=prefetch_worker)
        self._prefetch_thread.start()
    
    def get_batch(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get next prefetched batch."""
        return self._prefetch_queue.get()
    
    def stop(self):
        """Stop prefetching."""
        self._stop_event.set()
        if self._prefetch_thread:
            self._prefetch_thread.join()


if __name__ == "__main__":
    # Test the collators
    print("Testing collators...")
    
    from transformers import AutoTokenizer
    
    # Mock tokenizer for testing
    class MockTokenizer:
        pad_token_id = 0
        
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            parts = []
            for m in messages:
                parts.append(f"<{m['role']}>{m['content']}</{m['role']}>")
            if add_generation_prompt:
                parts.append("<assistant>")
            return "".join(parts)
        
        def __call__(self, text, truncation=True, max_length=2048, return_tensors=None):
            # Simple character-level tokenization for testing
            tokens = [ord(c) % 100 for c in text[:max_length]]
            return {
                'input_ids': tokens,
                'attention_mask': [1] * len(tokens),
            }
    
    tokenizer = MockTokenizer()
    
    # Test examples
    examples = [
        {
            'fen': 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1',
            'target_move_uci': 'e7e5',
        },
        {
            'fen': 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2',
            'target_move_uci': 'g1f3',
        },
    ]
    
    try:
        from .stockfish_teacher import StockfishTeacher
        
        with StockfishTeacher(num_workers=2, depth=8, top_k=5) as teacher:
            collator = DistillationCollator(
                tokenizer=tokenizer,
                teacher=teacher,
                max_length=512,
                randomize_order=True,
            )
            
            print("\nProcessing batch...")
            start = time.time()
            batch = collator(examples)
            elapsed = time.time() - start
            
            print(f"Batch processed in {elapsed:.2f}s")
            print(f"Input shape: {batch['input_ids'].shape}")
            print(f"Has soft_targets: {'soft_targets' in batch}")
            
            if 'soft_targets' in batch:
                for i, st in enumerate(batch['soft_targets']):
                    if st:
                        print(f"  Example {i}: {len(st)} moves in distribution")
            
    except FileNotFoundError as e:
        print(f"Stockfish not found: {e}")
        print("Testing precomputed collator instead...")
        
        # Test precomputed collator
        precomputed_examples = [
            {
                'fen': 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1',
                'target_move_uci': 'e7e5',
                'messages': [
                    {'role': 'user', 'content': 'Test prompt'},
                    {'role': 'assistant', 'content': '<think>Test</think>\n<uci_move>e7e5</uci_move>'},
                ],
                'move_probs': {'e7e5': 0.4, 'c7c5': 0.3, 'd7d5': 0.2, 'g8f6': 0.1},
                'loss_weight': 1.0,
            },
        ]
        
        collator = PrecomputedDistillationCollator(
            tokenizer=tokenizer,
            max_length=512,
        )
        
        batch = collator(precomputed_examples)
        print(f"Precomputed batch shape: {batch['input_ids'].shape}")
