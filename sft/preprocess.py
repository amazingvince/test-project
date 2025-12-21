#!/usr/bin/env python3
"""
Preprocess chess data with Stockfish evaluations.

This script:
1. Loads positions from Lichess games and puzzles
2. Analyzes each position with Stockfish to get all move evaluations
3. Saves the dataset with move evaluations for training

Usage:
    python sft/preprocess.py --output ./data/chess_with_eval --size 100000
    python sft/preprocess.py --config configs/sft/config_with_eval.yaml
"""

import argparse
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.data_processing_with_eval import preprocess_and_save_with_eval
from src.utils.stockfish_eval import StockfishEvaluator


def find_stockfish():
    """Find Stockfish binary."""
    import shutil
    
    paths = [
        shutil.which('stockfish'),
        '/usr/bin/stockfish',
        '/usr/games/stockfish',
        '/usr/local/bin/stockfish',
        '/opt/homebrew/bin/stockfish',
    ]
    
    for path in paths:
        if path and Path(path).exists():
            return path
    
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
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config not found: {config_path}, using defaults")
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
        print("ERROR: Stockfish not found!")
        print("Install with: sudo apt install stockfish")
        print("Or specify path with --stockfish-path")
        sys.exit(1)
    
    print("=" * 60)
    print("Chess Data Preprocessing with Stockfish")
    print("=" * 60)
    print(f"Output path: {output_path}")
    print(f"Target size: {target_size:,}")
    print(f"Games ratio: {games_ratio:.1%}")
    print(f"Stockfish path: {stockfish_path}")
    print(f"Stockfish depth: {depth}")
    print(f"Stockfish workers: {workers}")
    print(f"MultiPV: {multipv or 'all moves'}")
    print("=" * 60)
    
    # Verify Stockfish works
    print("\nVerifying Stockfish...")
    try:
        with StockfishEvaluator(stockfish_path=stockfish_path, depth=depth) as evaluator:
            import chess
            board = chess.Board()
            analysis = evaluator.analyze_position(board, multipv=5)
            print(f"✓ Stockfish working! Best move in starting position: {analysis.best_move_san}")
    except Exception as e:
        print(f"ERROR: Stockfish verification failed: {e}")
        sys.exit(1)
    
    # Process data
    print("\nStarting preprocessing...")
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
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)
    print(f"Dataset saved to: {output_path}")
    print(f"Total examples: {len(dataset):,}")
    
    # Show sample
    print("\nSample entry:")
    print("-" * 60)
    sample = dataset[0]
    print(f"FEN: {sample['fen']}")
    print(f"Target move: {sample['target_move_uci']}")
    print(f"Best move: {sample['best_move_san']} ({sample['best_score_cp']:+d}cp)")
    print(f"Target rank: {sample['target_move_rank']}")
    print(f"CP loss: {sample['target_move_cp_loss']}")
    print(f"Loss weight: {sample['loss_weight']:.3f}")
    
    if sample['move_evaluations']:
        print(f"\nTop 5 moves:")
        for i, mv in enumerate(sample['move_evaluations'][:5], 1):
            cp = mv['centipawn']
            print(f"  {i}. {mv['san']}: {cp:+d}cp")


if __name__ == "__main__":
    main()
