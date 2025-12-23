"""
Training-time chess evaluation callback.

This module is intentionally lightweight and side-effect free so it can be
imported from both `sft/train.py` and `distill/train.py` without triggering
training-script setup (kernel patches, torch flags, etc.).
"""

from __future__ import annotations

import logging
import random
import time
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
    trim_generated_token_ids,
    validate_uci_move,
)
from src.utils.formatting import DEFAULT_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

# Maximum total tokens (input + generation) during evaluation
MAX_TOTAL_TOKENS = 2048


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


def _compute_acpl_by_side(results: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """
    Compute Average Centipawn Loss split by side to move.

    The evaluation worker stores `is_white` based on the original position's
    side to move. This helper summarizes ACPL for:
    - positions where White is to play
    - positions where Black is to play
    """

    def _mean(values: List[int]) -> Optional[float]:
        return (sum(values) / len(values)) if values else None

    white = [r["cpl"] for r in results if r.get("cpl") is not None and r.get("is_white") is True]
    black = [r["cpl"] for r in results if r.get("cpl") is not None and r.get("is_white") is False]

    return {
        "acpl_white": _mean(white),
        "acpl_black": _mean(black),
        "acpl_white_n": float(len(white)),
        "acpl_black_n": float(len(black)),
    }


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
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        seed: int = 42,
        eval_every_n_steps: int = 500,
        stockfish_path: Optional[str] = None,
        stockfish_workers: int = 8,
        stockfish_depth: int = 10,
        print_samples: int = 0,
        print_max_chars: int = 600,
    ) -> None:
        self.eval_positions = list(eval_positions)
        self.tokenizer = tokenizer
        self.eval_batch_size = int(eval_batch_size)
        self.max_new_tokens = int(max_new_tokens)
        self.max_total_tokens = int(max_total_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.seed = int(seed)
        self.eval_every_n_steps = int(eval_every_n_steps)
        self.stockfish_path = stockfish_path
        self.stockfish_workers = int(stockfish_workers)
        self.stockfish_depth = int(stockfish_depth)
        self.print_samples = max(0, int(print_samples))
        self.print_max_chars = max(0, int(print_max_chars))
        self._original_padding_side = getattr(tokenizer, "padding_side", "right")
        self._last_eval_step = -1

        # Validate that there's room for input tokens
        min_input_tokens = 100  # Reasonable minimum for a chess prompt
        if self.max_new_tokens >= self.max_total_tokens - min_input_tokens:
            old_value = self.max_new_tokens
            self.max_new_tokens = self.max_total_tokens - min_input_tokens
            logger.warning(
                "max_new_tokens (%d) too large for max_total_tokens (%d), "
                "reduced to %d to leave room for input",
                old_value, self.max_total_tokens, self.max_new_tokens
            )

    def on_save(self, args, state, control, **kwargs):
        step = int(getattr(state, "global_step", 0) or 0)
        if step <= 0:
            return
        if self.eval_every_n_steps <= 0:
            return
        if step % self.eval_every_n_steps != 0:
            return
        if step == self._last_eval_step:
            return
        model = kwargs.get("model")
        if model is None:
            return
        self._last_eval_step = step
        try:
            self._run(model, step=step, final=False)
        except Exception:
            logger.exception("Chess eval failed at step %s (continuing training).", step)

    def on_train_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        try:
            self._run(model, step=state.global_step, final=True)
        except Exception:
            logger.exception("Final chess eval failed at step %s.", state.global_step)

    @torch.inference_mode()
    def _run(self, model, *, step: int, final: bool) -> None:
        if not self.eval_positions:
            return

        model.eval()
        self.tokenizer.padding_side = "left"

        eval_started = time.perf_counter()
        gen_seconds = 0.0
        stockfish_seconds = 0.0
        used_max_new_tokens: Optional[int] = None
        prompt_truncations = 0

        try:
            results: List[Dict[str, Any]] = []
            positions = self.eval_positions

            logger.debug(
                "Chess eval: max_total_tokens=%s, max_new_tokens=%s",
                self.max_total_tokens, self.max_new_tokens
            )

            for start in range(0, len(positions), self.eval_batch_size):
                batch_positions = positions[start : start + self.eval_batch_size]
                prompts, boards = _build_prompts(self.tokenizer, batch_positions)

                length_meta = self.tokenizer(
                    prompts,
                    padding=False,
                    truncation=False,
                    return_length=True,
                )
                lengths = length_meta.get("length")
                if lengths is None:
                    lengths = [len(ids) for ids in length_meta["input_ids"]]
                max_prompt_len = max(int(v) for v in lengths) if lengths else 0
                max_new_tokens = min(self.max_new_tokens, max(1, self.max_total_tokens - max_prompt_len))
                if max_prompt_len + max_new_tokens > self.max_total_tokens:
                    max_new_tokens = max(1, self.max_total_tokens - max_prompt_len)
                max_input_length = self.max_total_tokens - max_new_tokens
                if max_input_length < max_prompt_len:
                    prompt_truncations += 1

                inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_input_length,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                prompt_len = inputs["input_ids"].shape[1]

                eos_token_ids: List[int] = []
                if self.tokenizer.eos_token_id is not None:
                    eos_token_ids.append(int(self.tokenizer.eos_token_id))
                close_tag_id = self.tokenizer.convert_tokens_to_ids("</uci_move>")
                close_tag_token_id: Optional[int] = None
                if (
                    isinstance(close_tag_id, int)
                    and close_tag_id >= 0
                    and (self.tokenizer.unk_token_id is None or close_tag_id != self.tokenizer.unk_token_id)
                ):
                    close_tag_token_id = int(close_tag_id)
                    eos_token_ids.append(close_tag_token_id)
                eos_token_ids = list(dict.fromkeys(eos_token_ids))

                generate_kwargs = dict(inputs)
                generate_kwargs.update(
                    {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": self.do_sample,
                        "pad_token_id": self.tokenizer.pad_token_id,
                        "use_cache": True,
                    }
                )
                if eos_token_ids:
                    generate_kwargs["eos_token_id"] = eos_token_ids

                if self.do_sample:
                    if self.temperature is not None:
                        generate_kwargs["temperature"] = float(self.temperature)
                    if self.top_p is not None:
                        generate_kwargs["top_p"] = float(self.top_p)
                    if self.top_k is not None and hasattr(model.generation_config, "top_k"):
                        generate_kwargs["top_k"] = int(self.top_k)
                    if self.min_p is not None and hasattr(model.generation_config, "min_p"):
                        generate_kwargs["min_p"] = float(self.min_p)

                used_max_new_tokens = max_new_tokens
                gen_started = time.perf_counter()
                device = inputs["input_ids"].device
                fork_devices: List[int] = []
                if device.type == "cuda" and device.index is not None:
                    fork_devices = [int(device.index)]
                with torch.random.fork_rng(devices=fork_devices, enabled=True):
                    if self.do_sample:
                        seed_value = self.seed + int(step)
                        torch.manual_seed(seed_value)
                        if device.type == "cuda":
                            torch.cuda.manual_seed_all(seed_value)
                    outputs = model.generate(**generate_kwargs)
                gen_seconds += time.perf_counter() - gen_started

                for i, seq in enumerate(outputs):
                    generated_ids = seq[prompt_len:].tolist()
                    trimmed_ids = trim_generated_token_ids(
                        generated_ids,
                        close_tag_id=close_tag_token_id,
                        eos_token_ids=(
                            [int(self.tokenizer.eos_token_id)]
                            if self.tokenizer.eos_token_id is not None
                            else ()
                        ),
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    response = self.tokenizer.decode(trimmed_ids, skip_special_tokens=False)
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
                stockfish_started = time.perf_counter()
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
                stockfish_seconds += time.perf_counter() - stockfish_started

            metrics = _compute_basic_metrics(results)
            side_metrics = _compute_acpl_by_side(results)

            prefix = "final/" if final else ""
            eval_seconds = time.perf_counter() - eval_started
            logger.info(
                "Chess eval @ step %s: legal=%.2f%% acc=%.2f%% acpl=%s (n=%s) time=%.1fs (gen=%.1fs sf=%.1fs)",
                step,
                metrics.legal_move_rate * 100,
                metrics.accuracy * 100,
                f"{metrics.acpl:.1f}" if metrics.acpl is not None else "n/a",
                metrics.acpl_n,
                eval_seconds,
                gen_seconds,
                stockfish_seconds,
            )
            if prompt_truncations:
                logger.warning(
                    "Chess eval prompt truncations: %s batches truncated (max_total_tokens=%s).",
                    prompt_truncations,
                    self.max_total_tokens,
                )
            if used_max_new_tokens is not None and used_max_new_tokens != self.max_new_tokens:
                logger.warning(
                    "Chess eval reduced max_new_tokens from %s to %s to fit prompt budget (max_total_tokens=%s).",
                    self.max_new_tokens,
                    used_max_new_tokens,
                    self.max_total_tokens,
                )
            if side_metrics["acpl_white"] is not None or side_metrics["acpl_black"] is not None:
                logger.info(
                    "  ACPL by side: white=%s (n=%s) black=%s (n=%s)",
                    f"{side_metrics['acpl_white']:.1f}" if side_metrics["acpl_white"] is not None else "n/a",
                    int(side_metrics["acpl_white_n"]),
                    f"{side_metrics['acpl_black']:.1f}" if side_metrics["acpl_black"] is not None else "n/a",
                    int(side_metrics["acpl_black_n"]),
                )

            per_source: Dict[str, List[Dict[str, Any]]] = {}
            for r in results:
                per_source.setdefault(r["source"] or "unknown", []).append(r)
            for source, rows in sorted(per_source.items()):
                sm = _compute_basic_metrics(rows)
                sm_side = _compute_acpl_by_side(rows)
                logger.info(
                    "  source=%s legal=%.2f%% acc=%.2f%% acpl=%s (n=%s)",
                    source,
                    sm.legal_move_rate * 100,
                    sm.accuracy * 100,
                    f"{sm.acpl:.1f}" if sm.acpl is not None else "n/a",
                    sm.acpl_n,
                )
                if sm_side["acpl_white"] is not None or sm_side["acpl_black"] is not None:
                    logger.info(
                        "    ACPL by side: white=%s (n=%s) black=%s (n=%s)",
                        f"{sm_side['acpl_white']:.1f}" if sm_side["acpl_white"] is not None else "n/a",
                        int(sm_side["acpl_white_n"]),
                        f"{sm_side['acpl_black']:.1f}" if sm_side["acpl_black"] is not None else "n/a",
                        int(sm_side["acpl_black_n"]),
                    )

            self._print_sample_outputs(results, step=step, final=final)

            import importlib

            try:
                wandb = importlib.import_module("wandb")
            except ImportError:
                wandb = None

            if wandb is not None and getattr(wandb, "run", None) is not None:
                payload: Dict[str, float] = {
                    f"{prefix}chess/legal_move_rate": metrics.legal_move_rate,
                    f"{prefix}chess/accuracy": metrics.accuracy,
                    f"{prefix}chess/eval_seconds": eval_seconds,
                    f"{prefix}chess/eval_gen_seconds": gen_seconds,
                    f"{prefix}chess/eval_stockfish_seconds": stockfish_seconds,
                }
                if metrics.acpl is not None:
                    payload[f"{prefix}chess/acpl"] = metrics.acpl
                if side_metrics["acpl_white"] is not None:
                    payload[f"{prefix}chess/acpl_white"] = side_metrics["acpl_white"]
                if side_metrics["acpl_black"] is not None:
                    payload[f"{prefix}chess/acpl_black"] = side_metrics["acpl_black"]
                for source, rows in per_source.items():
                    sm = _compute_basic_metrics(rows)
                    sm_side = _compute_acpl_by_side(rows)
                    payload[f"{prefix}chess/{source}_legal_move_rate"] = sm.legal_move_rate
                    payload[f"{prefix}chess/{source}_accuracy"] = sm.accuracy
                    if sm.acpl is not None:
                        payload[f"{prefix}chess/{source}_acpl"] = sm.acpl
                    if sm_side["acpl_white"] is not None:
                        payload[f"{prefix}chess/{source}_acpl_white"] = sm_side["acpl_white"]
                    if sm_side["acpl_black"] is not None:
                        payload[f"{prefix}chess/{source}_acpl_black"] = sm_side["acpl_black"]
                try:
                    wandb.log(payload, step=step)
                except Exception:
                    logger.exception("wandb.log failed during chess eval @ step %s.", step)
        finally:
            self.tokenizer.padding_side = self._original_padding_side
            model.train()

    def _print_sample_outputs(self, results: List[Dict[str, Any]], *, step: int, final: bool) -> None:
        if self.print_samples <= 0:
            return
        if not results:
            return

        n = min(self.print_samples, len(results))
        rng = random.Random(step + (1 if final else 0))
        indices = sorted(rng.sample(range(len(results)), k=n))

        header = f"Chess eval samples @ step {step}"
        if final:
            header += " (final)"
        print("\n" + "=" * 80)
        print(header)
        print("=" * 80)

        for i, idx in enumerate(indices, start=1):
            r = results[idx]
            fen = str(r.get("fen", ""))
            board = chess.Board(fen) if fen else None

            predicted = r.get("predicted_move") or "<none>"
            target = r.get("target_move") or "<none>"
            source = r.get("source") or "unknown"
            is_legal = bool(r.get("is_legal"))
            is_correct = bool(r.get("is_correct"))
            cpl = r.get("cpl")

            print(f"\n[{i}/{n}] source={source} legal={is_legal} correct={is_correct} cpl={cpl if cpl is not None else 'n/a'}")
            if fen:
                print(f"FEN: {fen}")
            if board is not None:
                print(render_board_utf(board))
            print(f"Target: {target}  Pred: {predicted}")

            response = (r.get("response_text") or "").strip()
            if self.print_max_chars and len(response) > self.print_max_chars:
                head_chars = max(1, self.print_max_chars // 2)
                tail_chars = max(1, self.print_max_chars - head_chars)
                response = (
                    response[:head_chars].rstrip()
                    + "\n...\n"
                    + response[-tail_chars:].lstrip()
                )
            if response:
                print("Response:")
                print(response)
