"""
Training-time chess evaluation callback.

This module is intentionally lightweight and side-effect free so it can be
imported from both `sft/train.py` and `distill/train.py` without triggering
training-script setup (kernel patches, torch flags, etc.).
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import chess
import torch
from transformers import TrainerCallback

from src.utils.chess_utils import (
    extract_uci_from_response,
    get_first_legal_move,
    get_legal_moves_uci,
    render_board_utf,
    validate_uci_move,
)
from src.utils.formatting import DEFAULT_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

# Maximum total tokens (input + generation) during evaluation
MAX_TOTAL_TOKENS = 1024


@dataclass(frozen=True)
class EvalPosition:
    """Minimal position bundle for fast eval during training."""

    fen: str
    target_move_uci: str
    source: str = "unknown"  # "game" | "puzzle" | "unknown"


@dataclass(frozen=True)
class EvalMetrics:
    """Aggregate evaluation metrics."""

    total_positions: int
    legal_move_rate: float
    accuracy: float
    acpl: Optional[float] = None
    acpl_n: int = 0


def _stockfish_eval_worker(args: Tuple[int, str, str, str, int]) -> Tuple[int, Optional[int], bool]:
    """
    Compute centipawn loss for a predicted move.

    Returns:
        (index, centipawn_loss or None, is_white_to_move)
    """

    idx, fen, predicted_move, stockfish_path, depth = args
    board = chess.Board(fen)
    is_white = board.turn == chess.WHITE

    try:
        from stockfish import Stockfish
    except ImportError:
        return idx, None, is_white

    try:
        sf = Stockfish(
            path=stockfish_path,
            depth=depth,
            parameters={"Threads": 1, "Hash": 32},
        )
        sf.set_fen_position(fen)

        top_moves = sf.get_top_moves(2)
        if not top_moves:
            return idx, None, is_white

        best_eval = top_moves[0].get("Centipawn")
        if top_moves[0].get("Move") == predicted_move:
            return idx, 0, is_white

        try:
            board.push_uci(predicted_move)
        except (ValueError, chess.IllegalMoveError):
            return idx, None, is_white

        sf.set_fen_position(board.fen())
        eval_after = sf.get_evaluation()
        if eval_after.get("type") != "cp" or best_eval is None:
            return idx, None, is_white

        # After our move, it's opponent's turn, so negate to compare from mover's perspective.
        our_eval = -int(eval_after["value"])
        loss = int(best_eval) - our_eval
        return idx, max(0, loss), is_white
    except Exception:
        return idx, None, is_white


def prepare_eval_positions(
    eval_dataset: Any,
    *,
    max_positions: int = 500,
) -> List[EvalPosition]:
    """
    Build a small, in-memory list of evaluation positions.

    Supports:
    - a HuggingFace Dataset
    - a dict of datasets (e.g. {"total": ds, "games": ds, "puzzles": ds})

    Only `fen`, `target_move_uci`, and optional `source` are used.
    """

    def _iter_rows(ds: Iterable[Any], limit: int) -> List[EvalPosition]:
        picked: List[EvalPosition] = []
        for i, row in enumerate(ds):
            if i >= limit:
                break
            data = dict(row) if hasattr(row, "keys") else row
            fen = data.get("fen")
            target = data.get("target_move_uci")
            if not fen or not target:
                continue
            picked.append(EvalPosition(fen=str(fen), target_move_uci=str(target), source=str(data.get("source", "unknown"))))
        return picked

    if isinstance(eval_dataset, dict):
        datasets = [ds for ds in eval_dataset.values() if ds is not None]
        if not datasets:
            return []
        per = max(1, max_positions // len(datasets))
        combined: List[EvalPosition] = []
        for ds in datasets:
            combined.extend(_iter_rows(ds, per))
        return combined[:max_positions]

    return _iter_rows(eval_dataset, max_positions)


def _build_prompts(tokenizer, positions: List[EvalPosition]) -> Tuple[List[str], List[chess.Board]]:
    prompts: List[str] = []
    boards: List[chess.Board] = []

    for pos in positions:
        board = chess.Board(pos.fen)
        boards.append(board)

        user_content = DEFAULT_PROMPT_TEMPLATE.format(
            fen=board.fen(),
            legal_moves=get_legal_moves_uci(board),
            board=render_board_utf(board),
            example_move=get_first_legal_move(board) or "",
            side_to_move="White" if board.turn == chess.WHITE else "Black",
        )
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    return prompts, boards


def _compute_basic_metrics(results: List[Dict[str, Any]]) -> EvalMetrics:
    total = len(results)
    legal = sum(1 for r in results if r["is_legal"])
    correct = sum(1 for r in results if r["is_correct"])
    legal_rate = legal / total if total else 0.0
    accuracy = correct / total if total else 0.0

    cpl_values = [r["cpl"] for r in results if r.get("cpl") is not None]
    acpl = (sum(cpl_values) / len(cpl_values)) if cpl_values else None
    return EvalMetrics(
        total_positions=total,
        legal_move_rate=legal_rate,
        accuracy=accuracy,
        acpl=acpl,
        acpl_n=len(cpl_values),
    )


class FastChessEvalCallback(TrainerCallback):
    """
    Periodically evaluates the model on a fixed set of chess positions.

    Metrics:
    - legal move rate
    - exact-match accuracy vs. target move
    - optional ACPL (Average Centipawn Loss) if Stockfish is configured
    """

    def __init__(
        self,
        *,
        eval_positions: List[EvalPosition],
        tokenizer: Any,
        eval_batch_size: int = 32,
        max_new_tokens: int = 128,
        max_total_tokens: int = MAX_TOTAL_TOKENS,
        eval_every_n_steps: int = 500,
        stockfish_path: Optional[str] = None,
        stockfish_workers: int = 8,
        stockfish_depth: int = 10,
    ) -> None:
        self.eval_positions = list(eval_positions)
        self.tokenizer = tokenizer
        self.eval_batch_size = int(eval_batch_size)
        self.max_new_tokens = int(max_new_tokens)
        self.max_total_tokens = int(max_total_tokens)
        self.eval_every_n_steps = int(eval_every_n_steps)
        self.stockfish_path = stockfish_path
        self.stockfish_workers = int(stockfish_workers)
        self.stockfish_depth = int(stockfish_depth)
        self._original_padding_side = getattr(tokenizer, "padding_side", "right")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step <= 0:
            return
        if self.eval_every_n_steps <= 0:
            return
        if state.global_step % self.eval_every_n_steps != 0:
            return
        model = kwargs.get("model")
        if model is None:
            return
        self._run(model, step=state.global_step, final=False)

    def on_train_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        self._run(model, step=state.global_step, final=True)

    @torch.inference_mode()
    def _run(self, model, *, step: int, final: bool) -> None:
        if not self.eval_positions:
            return

        model.eval()
        self.tokenizer.padding_side = "left"

        try:
            results: List[Dict[str, Any]] = []
            positions = self.eval_positions

            max_input_length = self.max_total_tokens - self.max_new_tokens
            logger.debug(
                "Chess eval: max_total_tokens=%s, max_input=%s, max_new_tokens=%s",
                self.max_total_tokens, max_input_length, self.max_new_tokens
            )

            for start in range(0, len(positions), self.eval_batch_size):
                batch_positions = positions[start : start + self.eval_batch_size]
                prompts, boards = _build_prompts(self.tokenizer, batch_positions)

                inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_input_length,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                prompt_len = inputs["input_ids"].shape[1]

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                )

                for i, seq in enumerate(outputs):
                    response = self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                    predicted = extract_uci_from_response(response)
                    board = boards[i]
                    target = batch_positions[i].target_move_uci
                    is_legal = bool(predicted) and validate_uci_move(board, predicted)
                    is_correct = bool(predicted) and (predicted == target)
                    results.append(
                        {
                            "fen": board.fen(),
                            "source": batch_positions[i].source,
                            "predicted_move": predicted,
                            "target_move": target,
                            "is_legal": is_legal,
                            "is_correct": is_correct,
                            "response_text": response,
                            "cpl": None,
                            "is_white": board.turn == chess.WHITE,
                        }
                    )

            if self.stockfish_path:
                indexed = [
                    (i, r["fen"], r["predicted_move"])
                    for i, r in enumerate(results)
                    if r.get("predicted_move") and r.get("is_legal")
                ]
                if indexed:
                    args_list = [
                        (i, fen, move, self.stockfish_path, self.stockfish_depth) for i, fen, move in indexed
                    ]
                    with ProcessPoolExecutor(max_workers=self.stockfish_workers) as executor:
                        futures = [executor.submit(_stockfish_eval_worker, a) for a in args_list]
                        for fut in as_completed(futures):
                            idx, cpl, is_white = fut.result()
                            results[idx]["cpl"] = cpl
                            results[idx]["is_white"] = is_white

            metrics = _compute_basic_metrics(results)

            prefix = "final/" if final else ""
            logger.info(
                "Chess eval @ step %s: legal=%.2f%% acc=%.2f%% acpl=%s (n=%s)",
                step,
                metrics.legal_move_rate * 100,
                metrics.accuracy * 100,
                f"{metrics.acpl:.1f}" if metrics.acpl is not None else "n/a",
                metrics.acpl_n,
            )

            per_source: Dict[str, List[Dict[str, Any]]] = {}
            for r in results:
                per_source.setdefault(r["source"] or "unknown", []).append(r)
            for source, rows in sorted(per_source.items()):
                sm = _compute_basic_metrics(rows)
                logger.info(
                    "  source=%s legal=%.2f%% acc=%.2f%% acpl=%s (n=%s)",
                    source,
                    sm.legal_move_rate * 100,
                    sm.accuracy * 100,
                    f"{sm.acpl:.1f}" if sm.acpl is not None else "n/a",
                    sm.acpl_n,
                )

            try:
                import wandb

                if wandb.run is not None:
                    payload: Dict[str, float] = {
                        f"{prefix}chess/legal_move_rate": metrics.legal_move_rate,
                        f"{prefix}chess/accuracy": metrics.accuracy,
                    }
                    if metrics.acpl is not None:
                        payload[f"{prefix}chess/acpl"] = metrics.acpl
                    for source, rows in per_source.items():
                        sm = _compute_basic_metrics(rows)
                        payload[f"{prefix}chess/{source}_legal_move_rate"] = sm.legal_move_rate
                        payload[f"{prefix}chess/{source}_accuracy"] = sm.accuracy
                        if sm.acpl is not None:
                            payload[f"{prefix}chess/{source}_acpl"] = sm.acpl
                    wandb.log(payload, step=step)
            except Exception:
                return
        finally:
            self.tokenizer.padding_side = self._original_padding_side
            model.train()
