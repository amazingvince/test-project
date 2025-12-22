"""
Data collators for policy distillation training.

`DistillationCollator` is used in streaming mode (it calls Stockfish at batch
time). `PrecomputedDistillationCollator` is used with a preprocessed dataset
saved by `distill/preprocess.py`.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chess
import torch

from .stockfish_teacher import PositionAnalysis, StockfishTeacher
from .formatting_distill import (
    position_to_messages_distill,
    create_distillation_example,
    DISTILLATION_PROMPT_TEMPLATE,
    DISTILLATION_PROMPT_TEMPLATE_NO_BOARD,
    DISTILLATION_RESPONSE_TEMPLATE,
)
from .reasoning_trace import ReasoningTraceGenerator
from ..utils.chess_utils import get_legal_moves_uci, render_board_utf


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
    reasoning_trace_generator: Optional[ReasoningTraceGenerator] = None
    force_best_move: bool = False
    seed: Optional[int] = None
    
    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self._call_count = 0
        self.uci_move_token_id = self._resolve_token_id("<uci_move>")
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
                    source=ex.get('source'),
                    reasoning_trace_generator=self.reasoning_trace_generator,
                    force_best_move=self.force_best_move,
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
        side_to_move = "White" if board.turn == chess.WHITE else "Black"
        example_move = get_first_legal_move(board) or ""
        
        user_content = DISTILLATION_PROMPT_TEMPLATE.format(
            fen=fen,
            legal_moves=legal_moves,
            board=board_utf,
            side_to_move=side_to_move,
            example_move=example_move,
        ) if self.include_board else DISTILLATION_PROMPT_TEMPLATE_NO_BOARD.format(
            fen=fen,
            legal_moves=legal_moves,
            side_to_move=side_to_move,
            example_move=example_move,
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
    reasoning_trace_generator: Optional[ReasoningTraceGenerator] = None
    rerandomize_reasoning_trace: bool = False
    force_best_move: bool = False
    seed: Optional[int] = None
    
    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self._call_count = 0
        self.uci_move_token_id = self._resolve_token_id("<uci_move>")
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
            messages = None
            best_move_uci = ex.get("best_move_uci")
            target_move_uci = ex.get("target_move_uci")
            needs_force_best = (
                self.force_best_move
                and bool(best_move_uci)
                and best_move_uci != target_move_uci
            )

            if "messages" in ex and not needs_force_best:
                if self.reasoning_trace_generator is not None and not self.rerandomize_reasoning_trace:
                    # Reasoning traces are already randomized during preprocessing; regenerating them
                    # in the collator is expensive and typically unnecessary.
                    messages = ex["messages"]
                elif not self.randomize_order and self.reasoning_trace_generator is None:
                    messages = ex["messages"]

            if messages is None:
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
                    pv_uci=mv.get('pv_uci', []),
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
            shallow_move_cps=example.get('shallow_move_cps', {}),
            shallow_move_win_probs=example.get('shallow_move_win_probs', {}),
            confirm_move_cps=example.get('confirm_move_cps', {}),
            confirm_move_win_probs=example.get('confirm_move_win_probs', {}),
        )

        position = {
            'fen': example['fen'],
            'target_move_uci': example['target_move_uci'],
            'legal_moves_uci': example.get('legal_moves_uci', ''),
            'board_utf': example.get('board_utf', ''),
            'first_legal_move': example.get('first_legal_move', ''),
            'side_to_move': example.get('side_to_move', ''),
            'source': example.get('source'),
        }

        result = position_to_messages_distill(
            position=position,
            analysis=analysis,
            max_display_moves=self.max_display_moves,
            randomize_order=self.randomize_order,
            pv_length=self.pv_length,
            include_board=self.include_board,
            reasoning_trace_generator=self.reasoning_trace_generator,
            force_best_move=self.force_best_move,
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
