#!/usr/bin/env python3
"""
Preprocess chess data with Stockfish analysis for policy distillation.

This script:
1. Loads positions from Lichess games and puzzles
2. Analyzes each position with Stockfish (parallelized)
3. Generates probability distributions over all legal moves
4. Saves the dataset ready for distillation training

Usage:
    python preprocess_distill.py --output ./data/chess_distill --size 100000
    python preprocess_distill.py --config configs/config_distill.yaml
"""

import argparse
import yaml
from pathlib import Path
import sys
import random
import time
from typing import Dict, Any, Optional, Iterator
from dataclasses import asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import chess
from datasets import load_dataset, Dataset
from tqdm import tqdm

from src.stockfish_teacher import StockfishTeacher, PositionAnalysis
from src.chess_utils import (
    parse_movetext,
    render_board_utf,
    get_legal_moves_uci,
    get_first_legal_move,
)
from src.formatting_distill import create_distillation_example
from src.reasoning_trace import ReasoningTraceGenerator


def find_stockfish() -> Optional[str]:
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


def get_elo_weight(white_elo: int, black_elo: int, elo_weights: Dict[int, float]) -> float:
    """Get sampling weight based on average ELO."""
    avg_elo = (white_elo + black_elo) / 2
    sorted_thresholds = sorted(elo_weights.keys(), reverse=True)
    
    for threshold in sorted_thresholds:
        if avg_elo >= threshold:
            return elo_weights[threshold]
    
    return 0.01


def extract_positions_from_games(
    config: Dict[str, Any],
    max_positions: int,
    seed: int = 42,
) -> Iterator[Dict[str, Any]]:
    """
    Extract positions from Lichess games.
    
    Yields position dicts with fen, target_move_uci, and metadata.
    """
    data_config = config.get('data', {})
    elo_weights = config.get('elo_weights', {
        1200: 0.05, 1400: 0.10, 1600: 0.15,
        1800: 0.25, 2000: 0.30, 2200: 0.20, 2400: 0.10
    })
    
    rng = random.Random(seed)
    
    dataset = load_dataset(
        data_config.get('games_dataset', 'Lichess/standard-chess-games'),
        split="train",
        streaming=True,
    )
    
    position_count = 0
    skip_first = data_config.get('skip_first_moves', 4)
    skip_last = data_config.get('skip_last_moves', 2)
    sample_rate = data_config.get('sample_rate', 0.3)
    min_elo = data_config.get('min_elo', 1200)
    
    for game in dataset:
        if position_count >= max_positions:
            break
        
        # Filter games
        if game.get('Termination') == 'Time forfeit':
            continue
        
        result = game.get('Result', '')
        if result not in ['1-0', '0-1', '1/2-1/2']:
            continue
        
        white_elo = game.get('WhiteElo', 1500) or 1500
        black_elo = game.get('BlackElo', 1500) or 1500
        
        if isinstance(white_elo, str):
            white_elo = int(white_elo) if white_elo.isdigit() else 1500
        if isinstance(black_elo, str):
            black_elo = int(black_elo) if black_elo.isdigit() else 1500
        
        avg_elo = (white_elo + black_elo) / 2
        if avg_elo < min_elo:
            continue
        
        # ELO-based sampling
        weight = get_elo_weight(white_elo, black_elo, elo_weights)
        if rng.random() > weight:
            continue
        
        movetext = game.get('movetext', '') or game.get('moves', '')
        if not movetext:
            continue
        
        moves = parse_movetext(movetext)
        if len(moves) < skip_first + skip_last + 1:
            continue
        
        # Extract positions
        board = chess.Board()
        
        for move_idx, move_san in enumerate(moves):
            if position_count >= max_positions:
                break
            
            if move_idx < skip_first:
                try:
                    board.push_san(move_san)
                except:
                    break
                continue
            
            if move_idx >= len(moves) - skip_last:
                break
            
            if rng.random() > sample_rate:
                try:
                    board.push_san(move_san)
                except:
                    break
                continue
            
            try:
                target_move = board.parse_san(move_san)
                target_uci = target_move.uci()
            except:
                break
            
            yield {
                'fen': board.fen(),
                'target_move_uci': target_uci,
                'board_utf': render_board_utf(board),
                'legal_moves_uci': get_legal_moves_uci(board),
                'white_elo': white_elo,
                'black_elo': black_elo,
                'move_number': move_idx,
                'source': 'game',
            }
            
            position_count += 1
            
            try:
                board.push_san(move_san)
            except:
                break


def extract_positions_from_puzzles(
    config: Dict[str, Any],
    max_positions: int,
    seed: int = 42,
) -> Iterator[Dict[str, Any]]:
    """
    Extract positions from Lichess puzzles.

    Extracts ALL solver's moves (odd-indexed: moves[1], moves[3], moves[5]...).

    Puzzle format:
        moves[0] = opponent's setup move
        moves[1] = solver's first move (extracted)
        moves[2] = opponent's response
        moves[3] = solver's second move (extracted)
        ...and so on
    """
    data_config = config.get('data', {})

    dataset = load_dataset(
        data_config.get('puzzles_dataset', 'Lichess/chess-puzzles'),
        split="train",
        streaming=True,
    )

    min_rating = data_config.get('min_puzzle_rating', 1000)
    max_rating = data_config.get('max_puzzle_rating', 2500)

    position_count = 0

    for puzzle in dataset:
        if position_count >= max_positions:
            break

        rating = puzzle.get('Rating', 1500) or 1500
        if isinstance(rating, str):
            rating = int(rating) if rating.isdigit() else 1500

        if rating < min_rating or rating > max_rating:
            continue

        fen = puzzle.get('FEN')
        moves_str = puzzle.get('Moves', '')

        if not fen or not moves_str:
            continue

        moves = moves_str.split()
        if len(moves) < 2:
            continue

        try:
            board = chess.Board(fen)

            # Iterate through all moves in the puzzle
            for i, move_uci in enumerate(moves):
                if position_count >= max_positions:
                    break

                try:
                    move = chess.Move.from_uci(move_uci)
                except ValueError:
                    break  # Invalid move format, stop

                if move not in board.legal_moves:
                    break  # Illegal move, stop

                # Odd indices (1, 3, 5...) are solver's moves - extract these
                if i % 2 == 1:
                    yield {
                        'fen': board.fen(),
                        'target_move_uci': move_uci,
                        'board_utf': render_board_utf(board),
                        'legal_moves_uci': get_legal_moves_uci(board),
                        'white_elo': rating,
                        'black_elo': rating,
                        'move_number': i,
                        'source': 'puzzle',
                    }

                    position_count += 1

                # Apply move to advance board state
                board.push(move)

        except:
            continue


def analyze_and_format_batch(
    positions: list,
    teacher: StockfishTeacher,
    config: Dict[str, Any],
    rng: random.Random,
    source_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    reasoning_trace_generator: Optional[ReasoningTraceGenerator] = None,
    force_best_move: bool = False,
) -> list:
    """
    Analyze positions and create distillation examples.
    """
    format_config = config.get('formatting', {})
    
    fens = [p['fen'] for p in positions]
    analysis_overrides = None
    if source_overrides:
        analysis_overrides = []
        for pos in positions:
            source = pos.get('source')
            override = source_overrides.get(source) if source else None
            if override:
                analysis_overrides.append(dict(override))
            else:
                analysis_overrides.append(None)

    analyses = teacher.analyze_batch(fens, analysis_overrides=analysis_overrides)
    
    results = []
    for pos, analysis in zip(positions, analyses):
        try:
            example = create_distillation_example(
                fen=pos['fen'],
                target_move_uci=pos['target_move_uci'],
                analysis=analysis,
                board_utf=pos['board_utf'],
                max_display_moves=format_config.get('max_display_moves', 5),
                randomize_order=format_config.get('randomize_order', True),
                pv_length=format_config.get('pv_length', 5),
                include_board=format_config.get('include_board', True),
                source=pos.get('source'),
                reasoning_trace_generator=reasoning_trace_generator,
                force_best_move=force_best_move,
                rng=rng,
            )

            # Add metadata
            example['white_elo'] = pos['white_elo']
            example['black_elo'] = pos['black_elo']
            example['move_number'] = pos['move_number']
            example['source'] = pos['source']

            # Store best_pv for regeneration during training
            example['best_pv'] = analysis.best_pv

            # Convert move_evaluations for storage
            example['move_evaluations'] = [
                {
                    'uci': ma.uci,
                    'san': ma.san,
                    'centipawn': ma.centipawn,
                    'cp_loss': ma.cp_loss,
                    'category': ma.category,
                    'mate_in': ma.mate_in,
                    'win_probability': ma.win_probability,
                    'pv_uci': ma.pv_uci,
                }
                for ma in analysis.move_analyses
            ]

            results.append(example)
            
        except Exception as e:
            print(f"Warning: Failed to process position: {e}")
            continue
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess chess data for policy distillation'
    )
    parser.add_argument(
        '--config', type=str, default='configs/config_distill.yaml',
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
        '--time-limit-ms', type=int, default=None,
        help='Stockfish time limit per position (ms)'
    )
    parser.add_argument(
        '--nodes', type=int, default=None,
        help='Stockfish node limit per position'
    )
    parser.add_argument(
        '--top-k', type=int, default=None,
        help='Number of moves for deep analysis'
    )
    parser.add_argument(
        '--workers', type=int, default=None,
        help='Number of Stockfish workers'
    )
    parser.add_argument(
        '--threads-per-worker', type=int, default=None,
        help='Threads per Stockfish worker'
    )
    parser.add_argument(
        '--shallow-depth', type=int, default=None,
        help='Optional shallow depth for all-moves pass'
    )
    parser.add_argument(
        '--shallow-max-moves', type=int, default=None,
        help='Cap shallow multipv moves'
    )
    parser.add_argument(
        '--cache-size', type=int, default=None,
        help='LRU cache size for analyses (0 disables)'
    )
    parser.add_argument(
        '--prob-mode', type=str, default=None,
        help='Probability mode: "cp" or "wdl"'
    )
    parser.add_argument(
        '--wdl-temperature', type=float, default=None,
        help='Temperature for win-prob softmax'
    )
    parser.add_argument(
        '--stockfish-path', type=str, default=None,
        help='Path to Stockfish binary'
    )
    parser.add_argument(
        '--batch-size', type=int, default=100,
        help='Batch size for Stockfish analysis'
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
    distill_config = config.get('distillation', {})
    data_config = config.get('data', {})
    source_overrides = stockfish_config.get('source_overrides', {})
    reasoning_trace_config = config.get('reasoning_trace', {})
    trace_enabled = reasoning_trace_config.get('enabled', False)
    force_best_move = reasoning_trace_config.get('always_choose_best_move', False)
    reasoning_trace_generator = None
    if trace_enabled:
        reasoning_trace_generator = ReasoningTraceGenerator(reasoning_trace_config)
        trace_status = reasoning_trace_generator.status()
        print(
            "Reasoning trace enabled: "
            f"opening={trace_status.get('opening_available')}, "
            f"tablebase={trace_status.get('tablebase_available')}, "
            f"force_best={force_best_move}"
        )
    
    output_path = args.output or './data/chess_distill'
    target_size = args.size or data_config.get('target_size', 100000)
    depth = args.depth if args.depth is not None else stockfish_config.get('depth', 12)
    time_limit_ms = (
        args.time_limit_ms if args.time_limit_ms is not None
        else stockfish_config.get('time_limit_ms')
    )
    nodes = args.nodes if args.nodes is not None else stockfish_config.get('nodes')
    top_k = args.top_k if args.top_k is not None else stockfish_config.get('top_k', 5)
    workers = args.workers if args.workers is not None else stockfish_config.get('num_workers', 8)
    threads_per_worker = (
        args.threads_per_worker if args.threads_per_worker is not None
        else stockfish_config.get('threads_per_worker', 1)
    )
    shallow_depth = (
        args.shallow_depth if args.shallow_depth is not None
        else stockfish_config.get('shallow_depth', 0)
    )
    shallow_max_moves = (
        args.shallow_max_moves if args.shallow_max_moves is not None
        else stockfish_config.get('shallow_max_moves')
    )
    confirm_depth = stockfish_config.get('confirm_depth', 0)
    confirm_top_k = stockfish_config.get('confirm_top_k', 0)
    cache_size = (
        args.cache_size if args.cache_size is not None
        else stockfish_config.get('cache_size', 0)
    )
    prob_mode = (
        args.prob_mode if args.prob_mode is not None
        else stockfish_config.get('prob_mode', 'cp')
    )
    wdl_temperature = (
        args.wdl_temperature if args.wdl_temperature is not None
        else stockfish_config.get('wdl_temperature', 1.0)
    )
    stockfish_path = args.stockfish_path or stockfish_config.get('path')
    games_ratio = args.games_ratio if args.games_ratio is not None else data_config.get('games_ratio', 0.7)
    batch_size = args.batch_size
    
    # Find Stockfish
    if stockfish_path is None:
        stockfish_path = find_stockfish()
    
    if stockfish_path is None:
        print("ERROR: Stockfish not found!")
        print("Install with: sudo apt install stockfish")
        print("Or specify path with --stockfish-path")
        sys.exit(1)
    
    # Calculate targets
    games_target = int(target_size * games_ratio)
    puzzles_target = target_size - games_target
    
    print("=" * 70)
    print("Chess Policy Distillation - Data Preprocessing")
    print("=" * 70)
    print(f"Output path: {output_path}")
    print(f"Target size: {target_size:,}")
    print(f"  Games: {games_target:,} ({games_ratio:.0%})")
    print(f"  Puzzles: {puzzles_target:,} ({1-games_ratio:.0%})")
    print(f"Stockfish: {stockfish_path}")
    print(f"  Depth: {depth}")
    if time_limit_ms is not None:
        print(f"  Time limit: {time_limit_ms}ms")
    if nodes is not None:
        print(f"  Nodes: {nodes}")
    print(f"  Top-k: {top_k}")
    print(f"  Workers: {workers}")
    print(f"  Threads/worker: {threads_per_worker}")
    if shallow_depth:
        print(f"  Shallow depth: {shallow_depth}")
        if shallow_max_moves is not None:
            print(f"  Shallow max moves: {shallow_max_moves}")
    if confirm_depth:
        print(f"  Confirm depth: {confirm_depth}")
        print(f"  Confirm top-k: {confirm_top_k}")
    print(f"  Prob mode: {prob_mode}")
    if prob_mode == "wdl":
        print(f"  WDL temperature: {wdl_temperature}")
    if cache_size:
        print(f"  Cache size: {cache_size}")
    print(f"Batch size: {batch_size}")
    print("=" * 70)
    
    # Initialize
    rng = random.Random(args.seed)
    
    # Verify Stockfish
    print("\nVerifying Stockfish...")
    try:
        with StockfishTeacher(
            stockfish_path=stockfish_path,
            num_workers=1,
            depth=depth,
            top_k=top_k,
            temperature=distill_config.get('stockfish_temperature', 100.0),
            min_probability=distill_config.get('min_probability', 0.001),
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
        ) as test_teacher:
            analysis = test_teacher.analyze_position(chess.STARTING_FEN)
            print(f"✓ Stockfish working! Best opening move: {analysis.best_move_san}")
    except Exception as e:
        print(f"ERROR: Stockfish verification failed: {e}")
        sys.exit(1)
    
    # Create teacher with full worker pool
    teacher = StockfishTeacher(
        stockfish_path=stockfish_path,
        num_workers=workers,
        depth=depth,
        top_k=top_k,
        temperature=distill_config.get('stockfish_temperature', 100.0),
        min_probability=distill_config.get('min_probability', 0.001),
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
    )
    
    all_examples = []
    
    try:
        # Process games
        print(f"\n{'='*70}")
        print("Step 1: Processing game positions")
        print(f"{'='*70}")
        
        game_positions = list(tqdm(
            extract_positions_from_games(config, games_target, args.seed),
            total=games_target,
            desc="Extracting games"
        ))
        
        print(f"Extracted {len(game_positions):,} game positions")
        print("Analyzing with Stockfish...")
        
        for i in tqdm(range(0, len(game_positions), batch_size), desc="Analyzing"):
            batch = game_positions[i:i+batch_size]
            examples = analyze_and_format_batch(
                batch,
                teacher,
                config,
                rng,
                source_overrides=source_overrides,
                reasoning_trace_generator=reasoning_trace_generator,
                force_best_move=force_best_move,
            )
            all_examples.extend(examples)
        
        print(f"Processed {len(all_examples):,} game examples")
        
        # Process puzzles
        print(f"\n{'='*70}")
        print("Step 2: Processing puzzle positions")
        print(f"{'='*70}")
        
        puzzle_positions = list(tqdm(
            extract_positions_from_puzzles(config, puzzles_target, args.seed),
            total=puzzles_target,
            desc="Extracting puzzles"
        ))
        
        print(f"Extracted {len(puzzle_positions):,} puzzle positions")
        print("Analyzing with Stockfish...")
        
        game_count = len(all_examples)
        for i in tqdm(range(0, len(puzzle_positions), batch_size), desc="Analyzing"):
            batch = puzzle_positions[i:i+batch_size]
            examples = analyze_and_format_batch(
                batch,
                teacher,
                config,
                rng,
                source_overrides=source_overrides,
                reasoning_trace_generator=reasoning_trace_generator,
                force_best_move=force_best_move,
            )
            all_examples.extend(examples)
        
        puzzle_count = len(all_examples) - game_count
        print(f"Processed {puzzle_count:,} puzzle examples")
        
    finally:
        teacher.close()
    
    # Shuffle and save
    print(f"\n{'='*70}")
    print("Step 3: Shuffling and saving")
    print(f"{'='*70}")
    
    rng.shuffle(all_examples)
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(all_examples)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(output_path)
    
    # Print statistics
    print(f"\n{'='*70}")
    print("Dataset Statistics")
    print(f"{'='*70}")
    print(f"Total examples: {len(all_examples):,}")
    
    # Source distribution
    sources = [ex['source'] for ex in all_examples]
    game_count = sum(1 for s in sources if s == 'game')
    puzzle_count = sum(1 for s in sources if s == 'puzzle')
    print(f"Games: {game_count:,} ({100*game_count/len(all_examples):.1f}%)")
    print(f"Puzzles: {puzzle_count:,} ({100*puzzle_count/len(all_examples):.1f}%)")
    
    # Move quality distribution
    ranks = [ex['target_move_rank'] for ex in all_examples if ex.get('target_move_rank', 0) > 0]
    if ranks:
        print(f"\nMove quality:")
        print(f"  Best move (rank 1): {sum(1 for r in ranks if r == 1):,} ({100*sum(1 for r in ranks if r == 1)/len(ranks):.1f}%)")
        print(f"  Top 3: {sum(1 for r in ranks if r <= 3):,} ({100*sum(1 for r in ranks if r <= 3)/len(ranks):.1f}%)")
        print(f"  Top 5: {sum(1 for r in ranks if r <= 5):,} ({100*sum(1 for r in ranks if r <= 5)/len(ranks):.1f}%)")
    
    # CP loss distribution
    cp_losses = [ex['target_move_cp_loss'] for ex in all_examples if ex.get('target_move_cp_loss') is not None]
    if cp_losses:
        print(f"\nCentipawn loss:")
        print(f"  Average: {sum(cp_losses)/len(cp_losses):.1f}")
        print(f"  ≤10cp (excellent): {sum(1 for l in cp_losses if l <= 10):,} ({100*sum(1 for l in cp_losses if l <= 10)/len(cp_losses):.1f}%)")
        print(f"  ≤30cp (good): {sum(1 for l in cp_losses if l <= 30):,} ({100*sum(1 for l in cp_losses if l <= 30)/len(cp_losses):.1f}%)")
        print(f"  ≤100cp (inaccuracy): {sum(1 for l in cp_losses if l <= 100):,} ({100*sum(1 for l in cp_losses if l <= 100)/len(cp_losses):.1f}%)")
    
    # Probability stats
    probs = [ex['target_move_prob'] for ex in all_examples if ex.get('target_move_prob', 0) > 0]
    if probs:
        print(f"\nTarget move probability (from Stockfish):")
        print(f"  Average: {sum(probs)/len(probs):.4f}")
        print(f"  Min: {min(probs):.4f}")
        print(f"  Max: {max(probs):.4f}")
    
    print(f"\n{'='*70}")
    print(f"✓ Dataset saved to: {output_path}")
    print(f"{'='*70}")
    
    # Show sample
    print("\nSample entry:")
    print("-" * 70)
    sample = all_examples[0]
    print(f"FEN: {sample['fen']}")
    print(f"Target: {sample['target_move_uci']}")
    print(f"Best: {sample['best_move_uci']} ({sample['best_score_cp']:+d}cp)")
    print(f"Target rank: {sample['target_move_rank']}")
    print(f"Target CP loss: {sample['target_move_cp_loss']}")
    print(f"Target probability: {sample['target_move_prob']:.4f}")
    print(f"Moves in distribution: {len(sample['move_probs'])}")
    print("\nThinking section:")
    print(sample['messages'][1]['content'][:500] + "...")


if __name__ == "__main__":
    main()
