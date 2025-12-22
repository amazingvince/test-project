#!/usr/bin/env python3
"""
Preprocess a dataset for SFT training using Stockfish annotations.

This is a thin CLI wrapper around `src.utils.data_processing_with_eval`, which
streams positions from HuggingFace datasets, evaluates them with Stockfish, and
writes a dataset to disk.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.data_processing_with_eval import preprocess_and_save_with_eval
from src.utils.stockfish_eval import StockfishEvaluator


logger = logging.getLogger(__name__)


def find_stockfish() -> Optional[str]:
    """Return the first Stockfish executable found on PATH or common locations."""

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


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess chess data with Stockfish evaluations'
    )
    parser.add_argument(
        '--config', type=str, default='configs/sft/config_with_eval.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output path for preprocessed dataset'
    )
    parser.add_argument(
        '--size', type=int, default=None,
        help='Number of positions to generate'
    )
    parser.add_argument(
        '--depth', type=int, default=None,
        help='Stockfish search depth'
    )
    parser.add_argument(
        '--workers', type=int, default=None,
        help='Number of Stockfish workers'
    )
    parser.add_argument(
        '--stockfish-path', type=str, default=None,
        help='Path to Stockfish binary'
    )
    parser.add_argument(
        '--games-ratio', type=float, default=None,
        help='Ratio of game positions (vs puzzles)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        logger.warning("Config not found: %s (using defaults)", config_path)
        config = {}
    
    # Override with command line arguments
    stockfish_config = config.get('stockfish', {})
    data_config = config.get('data', {})
    
    output_path = args.output or data_config.get('preprocessed_path', './data/chess_with_eval')
    target_size = args.size or data_config.get('target_size', 100000)
    depth = args.depth or stockfish_config.get('depth', 12)
    workers = args.workers or stockfish_config.get('workers', 8)
    stockfish_path = args.stockfish_path or stockfish_config.get('path')
    games_ratio = args.games_ratio or data_config.get('games_ratio', 0.7)
    multipv = stockfish_config.get('multipv')
    
    # Find Stockfish if not specified
    if stockfish_path is None:
        stockfish_path = find_stockfish()
    
    if stockfish_path is None:
        raise SystemExit(
            "Stockfish not found. Install it (e.g. `apt install stockfish`) or pass --stockfish-path."
        )
    
    logger.info("Output path: %s", output_path)
    logger.info("Target size: %s", f"{target_size:,}")
    logger.info("Games ratio: %.1f%%", games_ratio * 100)
    logger.info("Stockfish: %s (depth=%s, workers=%s)", stockfish_path, depth, workers)
    logger.info("MultiPV: %s", multipv or "all moves")
    
    # Verify Stockfish works
    logger.info("Verifying Stockfish...")
    try:
        with StockfishEvaluator(stockfish_path=stockfish_path, depth=depth) as evaluator:
            import chess
            board = chess.Board()
            analysis = evaluator.analyze_position(board, multipv=5)
            logger.info("Stockfish OK (best move in start position: %s)", analysis.best_move_san)
    except Exception as e:
        raise SystemExit(f"Stockfish verification failed: {e}") from e
    
    # Process data
    logger.info("Starting preprocessing...")
    dataset = preprocess_and_save_with_eval(
        output_path=output_path,
        target_size=target_size,
        games_ratio=games_ratio,
        stockfish_path=stockfish_path,
        stockfish_depth=depth,
        stockfish_workers=workers,
        multipv=multipv,
        config=config,
        seed=args.seed
    )

    logger.info("Dataset saved: %s", output_path)
    logger.info("Total examples: %s", f"{len(dataset):,}")
    
    # Show a sample row for a quick sanity check
    logger.info("Sample entry:")
    sample = dataset[0]
    logger.info("FEN: %s", sample["fen"])
    logger.info("Target move: %s", sample["target_move_uci"])
    logger.info(
        "Best move: %s (%+dcp)",
        sample["best_move_san"],
        sample["best_score_cp"],
    )
    logger.info("Target rank: %s", sample["target_move_rank"])
    logger.info("CP loss: %s", sample["target_move_cp_loss"])
    logger.info("Loss weight: %.3f", sample["loss_weight"])
    
    if sample['move_evaluations']:
        logger.info("Top 5 moves:")
        for i, mv in enumerate(sample['move_evaluations'][:5], 1):
            cp = mv['centipawn']
            logger.info("%d. %s: %+dcp", i, mv["san"], cp)


if __name__ == "__main__":
    raise SystemExit(main())
