"""
Helpers for constructing `StockfishTeacher` from a config dict.

Both `distill/train.py` and `sft/train.py` support "streaming" training where
positions are sampled on-the-fly. When reasoning traces (or distillation) need
Stockfish analysis, we create a single `StockfishTeacher` up-front and reuse it
inside the data collator.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from src.distill.stockfish_teacher import StockfishTeacher


@dataclass(frozen=True)
class StockfishTeacherConfig:
    """Normalized configuration for `StockfishTeacher`."""

    stockfish_path: str
    num_workers: int
    depth: int
    top_k: int
    hash_mb_per_worker: int
    threads_per_worker: int
    time_limit_ms: Optional[int]
    nodes: Optional[int]
    shallow_depth: int
    shallow_max_moves: Optional[int]
    confirm_depth: int
    confirm_top_k: int
    prob_mode: str
    wdl_temperature: float
    cache_size: int
    syzygy_path: Optional[str]
    temperature: float
    min_probability: float


def _find_stockfish_path(explicit_path: Optional[str]) -> Optional[str]:
    if explicit_path:
        if Path(explicit_path).exists():
            return explicit_path
        return None

    candidates = [
        shutil.which("stockfish"),
        "/usr/bin/stockfish",
        "/usr/games/stockfish",
        "/usr/local/bin/stockfish",
        "/opt/homebrew/bin/stockfish",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def normalize_stockfish_teacher_config(
    *,
    stockfish_config: Dict[str, Any],
    distillation_config: Optional[Dict[str, Any]] = None,
    require_path: bool = True,
) -> StockfishTeacherConfig:
    """
    Normalize disparate config schemas into a `StockfishTeacherConfig`.

    We accept both `workers` and `num_workers` to keep SFT and distill configs
    compatible.
    """

    distillation_config = distillation_config or {}

    sf_path = _find_stockfish_path(stockfish_config.get("path"))
    if not sf_path and require_path:
        raise FileNotFoundError(
            "Stockfish not found. Install it (e.g. `apt install stockfish`) or set `stockfish.path`."
        )

    num_workers = int(stockfish_config.get("num_workers", stockfish_config.get("workers", 8)))
    depth = int(stockfish_config.get("depth", 12))
    top_k = int(stockfish_config.get("top_k", 5))
    hash_mb = int(stockfish_config.get("hash_mb_per_worker", stockfish_config.get("hash_mb", 64)))
    threads_per_worker = int(stockfish_config.get("threads_per_worker", stockfish_config.get("threads", 1)))

    time_limit_ms = stockfish_config.get("time_limit_ms")
    time_limit_ms = int(time_limit_ms) if time_limit_ms is not None else None

    nodes = stockfish_config.get("nodes")
    nodes = int(nodes) if nodes is not None else None

    shallow_depth = int(stockfish_config.get("shallow_depth", 0))
    shallow_max_moves = stockfish_config.get("shallow_max_moves")
    shallow_max_moves = int(shallow_max_moves) if shallow_max_moves is not None else None

    confirm_depth = int(stockfish_config.get("confirm_depth", 0))
    confirm_top_k = int(stockfish_config.get("confirm_top_k", 0))

    prob_mode = str(stockfish_config.get("prob_mode", "cp"))
    wdl_temperature = float(stockfish_config.get("wdl_temperature", 1.0))

    cache_size = int(stockfish_config.get("cache_size", 0))
    syzygy_path = stockfish_config.get("syzygy_path")
    syzygy_path = str(syzygy_path) if syzygy_path else None

    temperature = float(distillation_config.get("stockfish_temperature", 100.0))
    min_probability = float(distillation_config.get("min_probability", 0.001))

    return StockfishTeacherConfig(
        stockfish_path=sf_path or "",
        num_workers=num_workers,
        depth=depth,
        top_k=top_k,
        hash_mb_per_worker=hash_mb,
        threads_per_worker=threads_per_worker,
        time_limit_ms=time_limit_ms,
        nodes=nodes,
        shallow_depth=shallow_depth,
        shallow_max_moves=shallow_max_moves,
        confirm_depth=confirm_depth,
        confirm_top_k=confirm_top_k,
        prob_mode=prob_mode,
        wdl_temperature=wdl_temperature,
        cache_size=cache_size,
        syzygy_path=syzygy_path,
        temperature=temperature,
        min_probability=min_probability,
    )


def create_stockfish_teacher(
    *,
    stockfish_config: Dict[str, Any],
    distillation_config: Optional[Dict[str, Any]] = None,
) -> StockfishTeacher:
    """
    Create a `StockfishTeacher` from config.

    This is used by both streaming distillation and streaming SFT with reasoning traces.
    """

    cfg = normalize_stockfish_teacher_config(
        stockfish_config=stockfish_config,
        distillation_config=distillation_config,
        require_path=True,
    )
    return StockfishTeacher(
        stockfish_path=cfg.stockfish_path,
        num_workers=cfg.num_workers,
        depth=cfg.depth,
        top_k=cfg.top_k,
        hash_mb_per_worker=cfg.hash_mb_per_worker,
        temperature=cfg.temperature,
        min_probability=cfg.min_probability,
        threads_per_worker=cfg.threads_per_worker,
        time_limit_ms=cfg.time_limit_ms,
        nodes=cfg.nodes,
        shallow_depth=cfg.shallow_depth,
        shallow_max_moves=cfg.shallow_max_moves,
        confirm_depth=cfg.confirm_depth,
        confirm_top_k=cfg.confirm_top_k,
        prob_mode=cfg.prob_mode,
        wdl_temperature=cfg.wdl_temperature,
        cache_size=cfg.cache_size,
        syzygy_path=cfg.syzygy_path,
    )

