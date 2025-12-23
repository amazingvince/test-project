"""
Streaming SFT collator with on-the-fly Stockfish analysis and reasoning traces.

This mirrors distillation's streaming collator:
- sample raw positions from HF streaming datasets
- run Stockfish per-batch via `StockfishTeacher`
- generate distill-style reasoning traces + `<uci_move>...</uci_move>` targets
- tokenize with prompt masking (train only on assistant tokens)

The output keys match what `sft/train.py`'s trainer expects:
`input_ids`, `attention_mask`, `labels`, `loss_weights`.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chess
try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for lightweight test envs
    torch = None  # type: ignore[assignment]

from src.distill.formatting_distill import create_distillation_example
from src.distill.reasoning_trace import ReasoningTraceGenerator
from src.distill.stockfish_teacher import StockfishTeacher


def _compute_elo_weight(example: Dict[str, Any], config: Dict[str, Any]) -> float:
    loss_cfg = (config or {}).get("loss_weighting", {})
    if not loss_cfg.get("enabled", False):
        return 1.0
    if loss_cfg.get("method") != "elo":
        return 1.0

    white = int(example.get("white_elo") or 1500)
    black = int(example.get("black_elo") or 1500)
    avg = (white + black) / 2.0

    elo_min = float(loss_cfg.get("elo_min", 1200))
    elo_max = float(loss_cfg.get("elo_max", 2400))
    w_min = float(loss_cfg.get("min_weight", 0.5))
    w_max = float(loss_cfg.get("max_weight", 2.0))
    fn = str(loss_cfg.get("function", "linear"))

    if elo_max <= elo_min:
        return 1.0

    if fn == "gaussian":
        target = float(loss_cfg.get("target_elo", 1900))
        sigma = float(loss_cfg.get("gaussian_sigma", 400))
        sigma = max(1.0, sigma)
        # Weight in [0, 1]
        import math

        z = (avg - target) / sigma
        t = math.exp(-0.5 * z * z)
    else:
        # Linear fallback.
        t = (avg - elo_min) / (elo_max - elo_min)
        t = max(0.0, min(1.0, t))

    return w_min + t * (w_max - w_min)


@dataclass
class StreamingSFTStockfishTraceCollator:
    tokenizer: Any
    teacher: StockfishTeacher
    config: Dict[str, Any]
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

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if torch is None:
            raise ImportError("StreamingSFTStockfishTraceCollator requires torch.")
        self._call_count += 1
        batch_rng = random.Random(self.seed + self._call_count)

        fens = [ex["fen"] for ex in examples]
        try:
            analyses = self.teacher.analyze_batch(fens)
        except Exception as exc:
            warnings.warn(f"Stockfish analysis failed: {exc}. Falling back to empty analyses.")
            analyses = [None] * len(examples)

        batch_input_ids: List[List[int]] = []
        batch_attention_mask: List[List[int]] = []
        batch_labels: List[List[int]] = []
        batch_weights: List[float] = []

        for ex, analysis in zip(examples, analyses):
            if analysis is None:
                # If Stockfish fails, don't crash training: emit a minimal target.
                # Use the provided move if present; otherwise try the first legal move.
                board = chess.Board(ex["fen"])
                target_move = ex.get("target_move_uci") or next(iter(board.legal_moves)).uci()
                messages = [
                    {"role": "user", "content": f"FEN: {ex['fen']}\nLegal moves (UCI): {' '.join(m.uci() for m in board.legal_moves)}"},
                    {"role": "assistant", "content": f"<think>Selecting a legal move.</think>\n<uci_move>{target_move}</uci_move>"},
                ]
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
            batch_weights.append(float(ex.get("loss_weight", _compute_elo_weight(ex, self.config))))

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
