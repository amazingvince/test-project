"""
Streaming SFT collator that generates distill-style messages on the fly.

This is intended for `sft/train.py --streaming` when `reasoning_trace.enabled: true`.
It runs Stockfish at batch time, generates a reasoning trace (if configured), and
tokenizes with prompt masking so the loss is computed only on assistant tokens.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Sequence

import chess
from src.distill.formatting_distill import (
    DISTILLATION_PROMPT_TEMPLATE,
    DISTILLATION_PROMPT_TEMPLATE_NO_BOARD,
    DISTILLATION_RESPONSE_TEMPLATE,
    create_distillation_example,
)
from src.distill.reasoning_trace import ReasoningTraceGenerator
from src.utils.chess_utils import get_first_legal_move, get_legal_moves_uci, render_board_utf

if TYPE_CHECKING:  # pragma: no cover
    import torch
    from src.distill.stockfish_teacher import PositionAnalysis


class StockfishBatchAnalyzer(Protocol):
    """Minimum interface needed from `StockfishTeacher` for streaming SFT traces."""

    def analyze_batch(
        self,
        fens: Sequence[str],
        analysis_overrides: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> List[Optional["PositionAnalysis"]]:
        ...


def _fallback_messages(
    example: Dict[str, Any],
    *,
    include_board: bool,
    target_move_uci: str,
) -> List[Dict[str, str]]:
    fen = example["fen"]
    board = chess.Board(fen)

    fen_parts = fen.split()
    side_to_move = (
        example.get("side_to_move")
        or ("White" if len(fen_parts) > 1 and fen_parts[1] == "w" else "Black")
    )
    legal_moves = example.get("legal_moves_uci") or get_legal_moves_uci(board)
    example_move = (
        example.get("first_legal_move")
        or get_first_legal_move(board)
        or (legal_moves.split()[0] if legal_moves else "")
    )
    board_utf = example.get("board_utf") or render_board_utf(board)

    if include_board:
        user_content = DISTILLATION_PROMPT_TEMPLATE.format(
            fen=fen,
            legal_moves=legal_moves,
            board=board_utf,
            side_to_move=side_to_move,
            example_move=example_move,
        )
    else:
        user_content = DISTILLATION_PROMPT_TEMPLATE_NO_BOARD.format(
            fen=fen,
            legal_moves=legal_moves,
            side_to_move=side_to_move,
            example_move=example_move,
        )

    assistant_content = DISTILLATION_RESPONSE_TEMPLATE.format(
        thinking="Choosing a legal move.",
        move=target_move_uci,
    )
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


@dataclass
class StreamingSFTStockfishTraceCollator:
    tokenizer: Any
    teacher: StockfishBatchAnalyzer
    reasoning_trace_generator: Optional[ReasoningTraceGenerator] = None

    max_length: int = 2048
    pad_to_multiple_of: int = 8

    include_board: bool = False
    max_display_moves: int = 5
    randomize_order: bool = True
    pv_length: int = 5
    force_best_move: bool = False

    seed: int = 42

    def __post_init__(self) -> None:
        self._call_count = 0

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        import torch

        self._call_count += 1
        batch_rng = random.Random(self.seed + self._call_count)

        fens = [ex["fen"] for ex in examples]
        try:
            analyses = self.teacher.analyze_batch(fens)
        except Exception as exc:
            warnings.warn(f"Stockfish analysis failed: {exc}. Falling back to minimal messages.")
            analyses = [None] * len(examples)

        batch_input_ids: List[List[int]] = []
        batch_attention_mask: List[List[int]] = []
        batch_labels: List[List[int]] = []
        batch_weights: List[float] = []

        for ex, analysis in zip(examples, analyses):
            if analysis is None:
                board = chess.Board(ex["fen"])
                target_move = ex.get("target_move_uci") or get_first_legal_move(board) or ""
                messages = _fallback_messages(
                    ex,
                    include_board=self.include_board,
                    target_move_uci=target_move,
                )
            else:
                result = create_distillation_example(
                    fen=ex["fen"],
                    target_move_uci=ex.get("target_move_uci", ""),
                    analysis=analysis,
                    board_utf=ex.get("board_utf"),
                    max_display_moves=self.max_display_moves,
                    randomize_order=self.randomize_order,
                    pv_length=self.pv_length,
                    include_board=self.include_board,
                    source=ex.get("source"),
                    reasoning_trace_generator=self.reasoning_trace_generator,
                    force_best_move=self.force_best_move,
                    rng=batch_rng,
                )
                messages = result["messages"]

            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_text = self.tokenizer.apply_chat_template(
                [messages[0]],
                tokenize=False,
                add_generation_prompt=True,
            )

            full_tokens = self.tokenizer(full_text, truncation=False, return_tensors=None)
            prompt_tokens = self.tokenizer(prompt_text, truncation=False, return_tensors=None)

            input_ids = full_tokens["input_ids"]
            attention_mask = full_tokens.get("attention_mask") or [1] * len(input_ids)
            labels = input_ids.copy()
            prompt_length = len(prompt_tokens["input_ids"])

            if len(input_ids) > self.max_length:
                overflow = len(input_ids) - self.max_length
                input_ids = input_ids[overflow:]
                attention_mask = attention_mask[overflow:]
                labels = input_ids.copy()
                prompt_length = max(0, prompt_length - overflow)

            for i in range(min(prompt_length, len(labels))):
                labels[i] = -100

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
            batch_weights.append(float(ex.get("loss_weight", 1.0)))

        max_len = max(len(ids) for ids in batch_input_ids)
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        pad_id = int(self.tokenizer.pad_token_id)
        padded_input_ids: List[List[int]] = []
        padded_attention: List[List[int]] = []
        padded_labels: List[List[int]] = []
        for ids, attn, lab in zip(batch_input_ids, batch_attention_mask, batch_labels):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [pad_id] * pad_len)
            padded_attention.append(attn + [0] * pad_len)
            padded_labels.append(lab + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "loss_weights": torch.tensor(batch_weights, dtype=torch.float32),
        }
