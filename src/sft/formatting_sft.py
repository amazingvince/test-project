"""
SFT formatting that reuses the distillation-style reasoning traces.

The SFT preprocessing pipeline (`src/utils/data_processing_with_eval.py`) stores
Stockfish analysis as a list of per-move centipawn evaluations. Distillation
uses a richer `PositionAnalysis` class and an optional `ReasoningTraceGenerator`
to create human-like `<think>...</think>` text.

This module bridges those two worlds so SFT can:
- train on Stockfish's best move (or the played move)
- use the same reasoning-trace generator and response format as distillation
  without requiring the 8k move-token vocabulary.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import chess

from src.distill.formatting_distill import position_to_messages_distill
from src.distill.reasoning_trace import ReasoningTraceGenerator
from src.distill.stockfish_teacher import (
    MoveAnalysis,
    PositionAnalysis,
    categorize_move,
    cp_to_probability_distribution,
    cp_to_win_probability,
)
from src.utils.chess_utils import get_first_legal_move, get_legal_moves_uci, render_board_utf


def position_with_eval_to_distill_analysis(
    position: Dict[str, Any],
    *,
    min_probability: float = 0.001,
    stockfish_temperature: float = 100.0,
) -> PositionAnalysis:
    """
    Convert an SFT preprocessed position dict into a distillation `PositionAnalysis`.

    Expected keys (as produced by `src/utils/data_processing_with_eval.py`):
      - `fen`
      - `legal_moves_uci` (optional; will be derived from FEN if missing)
      - `move_evaluations`: list of {uci, san, centipawn, mate_in?}
      - `best_move_uci`, `best_move_san`, `best_score_cp`
    """

    fen = position.get("fen")
    if not fen:
        raise ValueError("position must contain 'fen'")

    board = chess.Board(fen)
    all_legal_moves = position.get("legal_moves_uci") or get_legal_moves_uci(board)
    all_legal_moves_list = [m for m in all_legal_moves.split() if m]

    move_evaluations = position.get("move_evaluations") or []
    best_move_uci = position.get("best_move_uci") or ""
    best_move_san = position.get("best_move_san") or ""
    best_score_cp = int(position.get("best_score_cp") or 0)

    move_cps: Dict[str, int] = {}
    move_analyses: List[MoveAnalysis] = []
    for mv in move_evaluations:
        uci = mv.get("uci")
        if not uci:
            continue
        cp = int(mv.get("centipawn") or 0)
        move_cps[uci] = cp

        mate_in = mv.get("mate_in")
        mate_in = int(mate_in) if mate_in is not None else None
        cp_loss = best_score_cp - cp
        move_analyses.append(
            MoveAnalysis(
                uci=str(uci),
                san=str(mv.get("san") or uci),
                centipawn=cp,
                cp_loss=cp_loss,
                category=categorize_move(cp_loss, mate_in),
                mate_in=mate_in,
                win_probability=cp_to_win_probability(cp),
                pv_uci=[],
            )
        )

    move_probs = cp_to_probability_distribution(
        move_cps=move_cps,
        all_legal_moves=all_legal_moves_list,
        temperature=float(stockfish_temperature),
        min_probability=float(min_probability),
    )

    return PositionAnalysis(
        fen=fen,
        move_analyses=move_analyses,
        best_move_uci=best_move_uci,
        best_move_san=best_move_san,
        best_score_cp=best_score_cp,
        best_pv=[],
        move_probs=move_probs,
        top_k_moves=[ma.uci for ma in move_analyses[:5]],
        shallow_move_cps={},
        shallow_move_win_probs={},
        confirm_move_cps={},
        confirm_move_win_probs={},
    )


def add_messages_with_reasoning_trace(
    example: Dict[str, Any],
    *,
    reasoning_trace_generator: Optional[ReasoningTraceGenerator],
    include_board: bool,
    always_choose_best_move: bool,
    min_probability: float = 0.001,
    stockfish_temperature: float = 100.0,
) -> Dict[str, Any]:
    """
    Build a `messages` field for SFT using distillation-style reasoning traces.

    Returns a dict containing a single key: `{"messages": ...}` so it can be
    used directly in `datasets.Dataset.map`.
    """

    analysis = position_with_eval_to_distill_analysis(
        example,
        min_probability=min_probability,
        stockfish_temperature=stockfish_temperature,
    )

    position = dict(example)
    if "legal_moves_uci" not in position or not position["legal_moves_uci"]:
        board = chess.Board(position["fen"])
        position["legal_moves_uci"] = get_legal_moves_uci(board)
    if "first_legal_move" not in position or not position["first_legal_move"]:
        board = chess.Board(position["fen"])
        position["first_legal_move"] = get_first_legal_move(board) or ""
    if "board_utf" not in position or not position["board_utf"]:
        board = chess.Board(position["fen"])
        position["board_utf"] = render_board_utf(board)
    if "side_to_move" not in position or not position["side_to_move"]:
        fen_parts = position["fen"].split()
        position["side_to_move"] = "White" if len(fen_parts) > 1 and fen_parts[1] == "w" else "Black"

    messages_data = position_to_messages_distill(
        position=position,
        analysis=analysis,
        include_board=include_board,
        reasoning_trace_generator=reasoning_trace_generator,
        force_best_move=always_choose_best_move,
    )
    return {"messages": messages_data["messages"]}


def maybe_override_target_to_best(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience mapping function: set `target_move_uci = best_move_uci` when present.

    This is useful for SFT runs that want to imitate Stockfish directly.
    """

    best = example.get("best_move_uci")
    if best:
        return {"target_move_uci": best}
    return {}
