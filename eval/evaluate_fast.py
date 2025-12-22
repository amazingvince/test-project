#!/usr/bin/env python3
"""
Fast evaluation for chess models.

This script generates moves for a batch of positions, checks legality/accuracy,
and optionally computes ACPL via Stockfish.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Ensure repo root is on sys.path when running from subfolders
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chess
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

from src.utils.chess_utils import (
    render_board_utf, 
    get_legal_moves_uci, 
    get_first_legal_move,
    extract_uci_from_response,
    validate_uci_move
)
from src.utils.formatting import DEFAULT_PROMPT_TEMPLATE
from src.utils.data_processing import stream_game_positions, stream_puzzle_positions


@dataclass
class EvalResult:
    """Result of evaluating a single position."""
    fen: str
    source: str
    predicted_move: Optional[str]
    target_move: str
    is_legal: bool
    is_correct: bool
    model_centipawn_loss: Optional[int]
    response_text: str


def setup_torch_optimizations():
    """Configure PyTorch for fast inference."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(model_path: str, device: str = "auto", compile_model: bool = True):
    """Load model optimized for fast inference."""
    logger.info("Loading model from %s", model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Use bfloat16 for better performance on modern GPUs
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",  # Fast attention
        )
    except Exception as e:
        logger.warning("flash_attention_2 unavailable (%s); falling back to eager attention.", e)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="eager",
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Ensure padding is on the left for batch generation
    tokenizer.padding_side = "left"
    
    model.eval()
    
    # Compile model for faster inference
    if compile_model:
        logger.info("Compiling model with torch.compile (this may take a minute)...")
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            logger.info("Model compiled")
        except Exception as e:
            logger.warning("Could not compile model: %s", e)
    
    return model, tokenizer


def build_prompts_batch(
    boards: List[chess.Board],
    tokenizer,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
) -> Tuple[List[str], torch.Tensor]:
    """Build prompts for a batch of positions."""
    prompts = []
    
    for board in boards:
        fen = board.fen()
        legal_moves = get_legal_moves_uci(board)
        board_utf = render_board_utf(board)
        first_legal = get_first_legal_move(board) or ""
        side_to_move = "White" if board.turn == chess.WHITE else "Black"
        
        user_content = prompt_template.format(
            fen=fen,
            legal_moves=legal_moves,
            board=board_utf,
            example_move=first_legal,
            side_to_move=side_to_move,
        )
        
        messages = [{"role": "user", "content": user_content}]
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)
    
    # Tokenize batch with left padding
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    )
    
    return prompts, inputs


@torch.inference_mode()
def generate_moves_batch(
    model,
    tokenizer,
    boards: List[chess.Board],
    max_new_tokens: int = 64,  # Reduced - we only need the move
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
) -> List[Tuple[Optional[str], str]]:
    """
    Generate moves for a batch of positions.
    
    Returns:
        List of (uci_move or None, full_response_text) tuples
    """
    _, inputs = build_prompts_batch(boards, tokenizer, prompt_template)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    input_length = inputs['input_ids'].shape[1]
    
    # Generate with greedy decoding (faster than sampling)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Greedy is faster
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,  # Use KV cache
    )
    
    # Decode responses
    results = []
    for i, output in enumerate(outputs):
        response = tokenizer.decode(
            output[input_length:], 
            skip_special_tokens=True
        )
        uci_move = extract_uci_from_response(response)
        results.append((uci_move, response))
    
    return results


# ============================================================================
# Stockfish Evaluation - Optimized with persistent instance and parallelism
# ============================================================================

class StockfishEvaluator:
    """Persistent Stockfish instance for efficient evaluation."""
    
    def __init__(self, stockfish_path: str, depth: int = 15, time_limit_ms: int = 100):
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.time_limit_ms = time_limit_ms
        self._sf = None
    
    def _get_stockfish(self):
        """Lazy initialization of Stockfish."""
        if self._sf is None:
            from stockfish import Stockfish
            self._sf = Stockfish(
                path=self.stockfish_path,
                depth=self.depth,
                parameters={
                    "Threads": 1,  # Single thread per process
                    "Hash": 64,    # Small hash table
                }
            )
        return self._sf
    
    def evaluate_move(self, fen: str, predicted_move: str) -> Optional[int]:
        """
        Evaluate centipawn loss of a move.
        
        Optimized: Uses single analysis call instead of separate best_move + eval.
        """
        try:
            sf = self._get_stockfish()
            
            # Get evaluation of current position
            sf.set_fen_position(fen)
            
            # Get top moves with their evaluations in one call
            top_moves = sf.get_top_moves(2)
            if not top_moves:
                return None
            
            best_eval = top_moves[0].get('Centipawn')
            
            # Check if our move is the best
            if top_moves[0].get('Move') == predicted_move:
                return 0  # Best move, no loss
            
            # Evaluate our move
            board = chess.Board(fen)
            try:
                board.push_uci(predicted_move)
            except (ValueError, chess.IllegalMoveError):
                return None
            
            sf.set_fen_position(board.fen())
            eval_after = sf.get_evaluation()
            
            if eval_after['type'] != 'cp' or best_eval is None:
                return None
            
            # Calculate loss (from perspective of side that moved)
            # After our move, it's opponent's turn, so negate
            our_eval = -eval_after['value']
            loss = best_eval - our_eval
            
            return max(0, loss)
        
        except Exception as e:
            return None
    
    def close(self):
        """Clean up Stockfish process."""
        if self._sf is not None:
            try:
                self._sf.__del__()
            except Exception:
                logger.debug("Stockfish cleanup failed", exc_info=True)
            self._sf = None


def evaluate_position_stockfish(
    args: Tuple[str, str, str, int, int]
) -> Tuple[int, Optional[int], bool]:
    """
    Worker function for parallel Stockfish evaluation.
    
    Args:
        args: (index, fen, predicted_move, stockfish_path, depth, time_ms)
    
    Returns:
        (index, centipawn_loss or None, is_white_to_move)
    """
    idx, fen, predicted_move, stockfish_path, depth, time_ms = args
    
    try:
        evaluator = StockfishEvaluator(stockfish_path, depth, time_ms)
        board = chess.Board(fen)
        is_white = board.turn == chess.WHITE
        cpl = evaluator.evaluate_move(fen, predicted_move)
        evaluator.close()
        return (idx, cpl, is_white)
    except Exception as e:
        return (idx, None, True)


def parallel_stockfish_evaluation(
    positions_with_moves: List[Tuple[int, str, str]],  # (idx, fen, move)
    stockfish_path: str,
    depth: int = 12,  # Reduced depth for speed
    time_ms: int = 100,
    num_workers: int = 8
) -> Tuple[Dict[int, Optional[int]], Dict[int, bool]]:
    """
    Evaluate multiple positions in parallel using multiprocessing.
    
    Returns:
        Tuple of (cpl_dict, is_white_dict) mapping position index to values
    """
    if not positions_with_moves:
        return {}, {}
    
    # Prepare arguments for workers
    args_list = [
        (idx, fen, move, stockfish_path, depth, time_ms)
        for idx, fen, move in positions_with_moves
    ]
    
    cpl_results = {}
    white_results = {}
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(evaluate_position_stockfish, args): args[0]
            for args in args_list
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Stockfish eval"):
            try:
                idx, cpl, is_white = future.result(timeout=10)
                cpl_results[idx] = cpl
                white_results[idx] = is_white
            except Exception as e:
                idx = futures[future]
                cpl_results[idx] = None
                white_results[idx] = True
    
    return cpl_results, white_results


# ============================================================================
# Main Evaluation Logic
# ============================================================================

def compute_metrics_for_predictions(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(predictions)
    legal_moves = sum(1 for p in predictions if p['is_legal'])
    correct_moves = sum(1 for p in predictions if p['is_correct'])

    cpl_all = [p['cpl'] for p in predictions if p['cpl'] is not None]
    cpl_white = [p['cpl'] for p in predictions if p['cpl'] is not None and p['is_white']]
    cpl_black = [p['cpl'] for p in predictions if p['cpl'] is not None and not p['is_white']]

    metrics = {
        'total_positions': total,
        'legal_move_rate': legal_moves / total if total > 0 else 0,
        'accuracy': correct_moves / total if total > 0 else 0,
        'legal_moves': legal_moves,
        'correct_moves': correct_moves,
    }

    if cpl_all:
        metrics['acpl'] = sum(cpl_all) / len(cpl_all)
        metrics['acpl_median'] = sorted(cpl_all)[len(cpl_all) // 2]
        metrics['acpl_n'] = len(cpl_all)

    if cpl_white:
        metrics['acpl_white'] = sum(cpl_white) / len(cpl_white)
        metrics['acpl_white_n'] = len(cpl_white)

    if cpl_black:
        metrics['acpl_black'] = sum(cpl_black) / len(cpl_black)
        metrics['acpl_black_n'] = len(cpl_black)

    return metrics


def load_test_positions(
    num_positions: int = 1000,
    seed: int = 42,
    config: Optional[Dict[str, Any]] = None,
    source: str = "mixed",
    games_ratio: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Load test positions from games and/or puzzles."""
    import random

    if config is None:
        config = {}

    data_config = config.get("data", {})
    elo_weights = config.get("elo_weights", None)

    source = source.lower()
    if source not in {"mixed", "games", "puzzles"}:
        raise ValueError("source must be one of: mixed, games, puzzles")

    if games_ratio is None:
        games_ratio = data_config.get("games_ratio", 0.7)

    if source == "games":
        target_games = num_positions
        target_puzzles = 0
    elif source == "puzzles":
        target_games = 0
        target_puzzles = num_positions
    else:
        target_games = int(num_positions * games_ratio)
        target_puzzles = num_positions - target_games

    positions = []
    games_added = 0
    puzzles_added = 0

    if target_games > 0:
        for pos in stream_game_positions(
            dataset_name=data_config.get("games_dataset", "Lichess/standard-chess-games"),
            min_elo=data_config.get("min_elo", 1200),
            elo_weights=elo_weights,
            sample_rate=data_config.get("sample_rate", 0.3),
            skip_first_moves=data_config.get("skip_first_moves", 4),
            skip_last_moves=data_config.get("skip_last_moves", 2),
            config=config,
            seed=seed,
        ):
            positions.append({
                "fen": pos["fen"],
                "target_move": pos["target_move_uci"],
                "source": "game",
            })
            games_added += 1
            if games_added >= target_games:
                break

    if target_puzzles > 0:
        for pos in stream_puzzle_positions(
            dataset_name=data_config.get("puzzles_dataset", "Lichess/chess-puzzles"),
            min_rating=data_config.get("min_puzzle_rating", 1000),
            max_rating=data_config.get("max_puzzle_rating", 2500),
            config=config,
            seed=seed + 1,
        ):
            positions.append({
                "fen": pos["fen"],
                "target_move": pos["target_move_uci"],
                "source": "puzzle",
            })
            puzzles_added += 1
            if puzzles_added >= target_puzzles:
                break

    rng = random.Random(seed)
    rng.shuffle(positions)

    logger.info("Loaded %s test positions", len(positions))
    logger.info("  Games: %s, Puzzles: %s", games_added, puzzles_added)
    return positions


def evaluate_model(
    model,
    tokenizer,
    positions: List[Dict[str, Any]],
    batch_size: int = 16,
    max_new_tokens: int = 64,
    stockfish_path: Optional[str] = None,
    stockfish_depth: int = 12,
    stockfish_time_ms: int = 100,
    stockfish_workers: int = 8,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluate model on positions with batched generation.
    """
    results = []
    all_predictions = []
    
    # Phase 1: Batched model inference
    logger.info("Phase 1: generating moves (batch_size=%s)", batch_size)
    
    for i in tqdm(range(0, len(positions), batch_size), desc="Generating"):
        batch_positions = positions[i:i + batch_size]
        boards = [chess.Board(pos['fen']) for pos in batch_positions]
        
        # Generate moves for batch
        batch_results = generate_moves_batch(
            model,
            tokenizer,
            boards,
            max_new_tokens=max_new_tokens,
        )
        
        for j, (predicted, response) in enumerate(batch_results):
            pos = batch_positions[j]
            board = boards[j]
            source = pos.get('source', 'unknown')
            
            is_legal = False
            if predicted:
                is_legal = validate_uci_move(board, predicted)
            
            is_correct = predicted == pos['target_move']
            is_white = board.turn == chess.WHITE
            
            all_predictions.append({
                'idx': i + j,
                'fen': pos['fen'],
                'target_move': pos['target_move'],
                'predicted_move': predicted,
                'response': response,
                'is_legal': is_legal,
                'is_correct': is_correct,
                'is_white': is_white,
                'cpl': None,
                'source': source,
            })
    
    # Phase 2: Parallel Stockfish evaluation (if enabled)
    if stockfish_path:
        logger.info("Phase 2: Stockfish evaluation (workers=%s)", stockfish_workers)
        
        # Collect legal moves for Stockfish evaluation
        legal_positions = [
            (p['idx'], p['fen'], p['predicted_move'])
            for p in all_predictions
            if p['is_legal'] and p['predicted_move']
        ]
        
        if legal_positions:
            cpl_results, white_results = parallel_stockfish_evaluation(
                legal_positions,
                stockfish_path,
                depth=stockfish_depth,
                time_ms=stockfish_time_ms,
                num_workers=stockfish_workers
            )
            
            # Update predictions with CPL
            for p in all_predictions:
                if p['idx'] in cpl_results:
                    p['cpl'] = cpl_results[p['idx']]
                    p['is_white'] = white_results.get(p['idx'], p['is_white'])
    
    # Calculate metrics
    metrics = compute_metrics_for_predictions(all_predictions)
    metrics_by_source = {}
    sources = sorted(set(p['source'] for p in all_predictions))
    for source in sources:
        subset = [p for p in all_predictions if p['source'] == source]
        metrics_by_source[source] = compute_metrics_for_predictions(subset)
    metrics['by_source'] = metrics_by_source
    
    # Build results
    for p in all_predictions:
        results.append(EvalResult(
            fen=p['fen'],
            source=p['source'],
            predicted_move=p['predicted_move'],
            target_move=p['target_move'],
            is_legal=p['is_legal'],
            is_correct=p['is_correct'],
            model_centipawn_loss=p['cpl'],
            response_text=p['response']
        ))
    
    return {
        'metrics': metrics,
        'results': [asdict(r) for r in results]
    }


def main():
    parser = argparse.ArgumentParser(description='Fast Chess LLM Evaluation')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--config', type=str, default=None,
                        help='Optional config file for data settings')
    parser.add_argument('--num_positions', type=int, default=1000,
                        help='Number of positions to evaluate')
    parser.add_argument('--source', type=str, default='mixed',
                        help='Data source: mixed, games, puzzles')
    parser.add_argument('--games_ratio', type=float, default=None,
                        help='Games ratio when source=mixed (overrides config)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for generation')
    parser.add_argument('--max_new_tokens', type=int, default=64,
                        help='Max tokens to generate per position')
    parser.add_argument('--stockfish', type=str, default=None,
                        help='Path to Stockfish binary')
    parser.add_argument('--stockfish_depth', type=int, default=12,
                        help='Stockfish search depth (lower = faster)')
    parser.add_argument('--stockfish_time', type=int, default=100,
                        help='Stockfish time per move (milliseconds)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel Stockfish workers')
    parser.add_argument('--output', type=str, default='eval_results.json',
                        help='Output file for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-compile', action='store_true',
                        help='Disable torch.compile')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode (verbose output)')
    
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    
    # Setup optimizations
    setup_torch_optimizations()

    config = load_config(args.config)
    
    # Load model
    model, tokenizer = load_model(
        args.model, 
        compile_model=not args.no_compile
    )
    
    # Load test positions
    positions = load_test_positions(
        num_positions=args.num_positions,
        seed=args.seed,
        config=config,
        source=args.source,
        games_ratio=args.games_ratio,
    )
    
    # Warmup (important for torch.compile)
    logger.info("Warming up model...")
    warmup_board = chess.Board()
    _ = generate_moves_batch(model, tokenizer, [warmup_board, warmup_board])
    logger.info("Model warmed up")
    
    # Evaluate
    start_time = time.time()
    logger.info("Starting evaluation...")
    
    results = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        positions=positions,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        stockfish_path=args.stockfish,
        stockfish_depth=args.stockfish_depth,
        stockfish_time_ms=args.stockfish_time,
        stockfish_workers=args.workers,
        verbose=args.debug
    )
    
    elapsed = time.time() - start_time
    
    # Report results
    metrics = results['metrics']
    logger.info("Total positions: %s", metrics["total_positions"])
    logger.info("Legal move rate: %.2f%%", metrics["legal_move_rate"] * 100)
    logger.info("Accuracy: %.2f%%", metrics["accuracy"] * 100)
    
    if 'acpl' in metrics:
        logger.info("ACPL: %.1f (n=%s)", metrics["acpl"], metrics["acpl_n"])
        if 'acpl_white' in metrics:
            logger.info("ACPL White: %.1f (n=%s)", metrics["acpl_white"], metrics["acpl_white_n"])
        if 'acpl_black' in metrics:
            logger.info("ACPL Black: %.1f (n=%s)", metrics["acpl_black"], metrics["acpl_black_n"])
        logger.info("ACPL Median: %.1f", metrics["acpl_median"])
    
    logger.info("Total time: %.1fs (%.1fms/position)", elapsed, elapsed / len(positions) * 1000)
    logger.info("Throughput: %.1f positions/second", len(positions) / elapsed)

    if metrics.get('by_source'):
        for source, sm in metrics['by_source'].items():
            logger.info("Source: %s", source)
            logger.info("  Total positions: %s", sm["total_positions"])
            logger.info("  Legal move rate: %.2f%%", sm["legal_move_rate"] * 100)
            logger.info("  Accuracy: %.2f%%", sm["accuracy"] * 100)
            if 'acpl' in sm:
                logger.info("  ACPL: %.1f (n=%s)", sm["acpl"], sm.get("acpl_n", 0))
    
    # Save results
    logger.info("Saving results to %s", args.output)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Evaluation complete")


if __name__ == "__main__":
    raise SystemExit(main())
