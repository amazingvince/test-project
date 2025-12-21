"""
Data processing module for Chess SFT training with Stockfish evaluations.

Handles loading and processing of:
- Lichess game positions with Stockfish analysis
- Lichess puzzles with Stockfish analysis

Includes centipawn-based loss weighting for reward shaping.
"""

import math
import random
import chess
from typing import Iterator, Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from datasets import load_dataset, IterableDataset, Dataset, interleave_datasets
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings

from .chess_utils import (
    ChessPosition,
    parse_movetext,
    render_board_utf,
    get_legal_moves_uci,
    get_first_legal_move,
)


# =============================================================================
# Extended Position Dataclass with Stockfish Evaluations
# =============================================================================

@dataclass
class ChessPositionWithEval:
    """Chess position with Stockfish move evaluations."""
    fen: str
    legal_moves_uci: str
    target_move_uci: str
    first_legal_move: str
    board_utf: str
    side_to_move: str
    white_elo: int
    black_elo: int
    move_number: int
    source: str
    
    # Move evaluations from Stockfish
    move_evaluations: List[Dict[str, Any]] = field(default_factory=list)
    # Best move according to Stockfish
    best_move_uci: str = ""
    best_move_san: str = ""
    best_score_cp: int = 0
    # Target move quality
    target_move_rank: int = 0  # 1 = best, higher = worse
    target_move_cp_loss: int = 0  # Centipawn loss vs best move
    
    # Loss weighting (now based on move quality, not just ELO)
    loss_weight: float = 1.0


# =============================================================================
# Loss Weight Functions (Centipawn-Based)
# =============================================================================

def compute_loss_weight_from_quality(
    target_move_rank: int,
    target_move_cp_loss: int,
    num_legal_moves: int,
    weight_type: str = "cp_loss",
    min_weight: float = 0.1,
    max_weight: float = 2.0,
    **kwargs
) -> float:
    """
    Compute loss weight based on move quality (Stockfish evaluation).
    
    Better moves get higher weight to encourage learning good moves.
    
    Args:
        target_move_rank: Rank of target move (1=best)
        target_move_cp_loss: Centipawn loss vs best move
        num_legal_moves: Total number of legal moves
        weight_type: How to compute weight:
            - "cp_loss": Based on centipawn loss
            - "rank": Based on move rank
            - "combined": Combination of both
            - "quality_gate": High if good move, low otherwise
            - "uniform": Always 1.0 (no quality weighting)
        min_weight: Minimum weight
        max_weight: Maximum weight
    
    Returns:
        Loss weight for training
    """
    if weight_type == "uniform":
        return 1.0
    
    elif weight_type == "cp_loss":
        # Higher weight for lower CP loss (better moves)
        max_cp_loss = kwargs.get("max_cp_loss", 200)
        # Clamp CP loss
        clamped_loss = min(target_move_cp_loss, max_cp_loss)
        ratio = clamped_loss / max_cp_loss
        return max_weight - ratio * (max_weight - min_weight)
    
    elif weight_type == "rank":
        # Higher weight for better ranked moves
        if num_legal_moves <= 1:
            return max_weight
        ratio = (target_move_rank - 1) / (num_legal_moves - 1)
        return max_weight - ratio * (max_weight - min_weight)
    
    elif weight_type == "combined":
        # Combine CP loss and rank
        max_cp_loss = kwargs.get("max_cp_loss", 200)
        alpha = kwargs.get("alpha", 0.7)  # Weight on CP loss
        
        cp_ratio = min(target_move_cp_loss / max_cp_loss, 1.0)
        rank_ratio = (target_move_rank - 1) / max(num_legal_moves - 1, 1)
        
        combined_ratio = alpha * cp_ratio + (1 - alpha) * rank_ratio
        return max_weight - combined_ratio * (max_weight - min_weight)
    
    elif weight_type == "quality_gate":
        # High weight if good move, low otherwise
        top_n = kwargs.get("top_n", 3)
        cp_threshold = kwargs.get("cp_threshold", 50)
        
        is_good = target_move_rank <= top_n or target_move_cp_loss <= cp_threshold
        return max_weight if is_good else min_weight
    
    elif weight_type == "soft_rank":
        # Exponential decay based on rank
        decay = kwargs.get("decay", 0.7)
        raw_weight = decay ** (target_move_rank - 1)
        # Scale to weight range
        return min_weight + raw_weight * (max_weight - min_weight)
    
    else:
        return 1.0


def compute_loss_weight_elo(
    white_elo: int,
    black_elo: int,
    config: Optional[Dict[str, Any]] = None
) -> float:
    """
    Legacy ELO-based loss weight (for comparison/fallback).
    """
    if config is None:
        config = {}
    
    loss_config = config.get('loss_weighting', {})
    
    if not loss_config.get('elo_weighting_enabled', False):
        return 1.0
    
    avg_elo = (white_elo + black_elo) / 2
    
    elo_min = loss_config.get('elo_min', 1200)
    elo_max = loss_config.get('elo_max', 2400)
    weight_min = loss_config.get('min_weight', 0.5)
    weight_max = loss_config.get('max_weight', 2.0)
    
    normalized = (avg_elo - elo_min) / (elo_max - elo_min)
    normalized = max(0.0, min(1.0, normalized))
    
    return weight_min + normalized * (weight_max - weight_min)


# =============================================================================
# Stockfish Analysis Integration
# =============================================================================

def analyze_position_with_stockfish(
    board: chess.Board,
    evaluator: Any,  # StockfishEvaluator
    multipv: Optional[int] = None
) -> Dict[str, Any]:
    """
    Analyze a position and return evaluation data.
    
    Args:
        board: Chess board position
        evaluator: StockfishEvaluator instance
        multipv: Number of moves to analyze (None = all)
    
    Returns:
        Dictionary with evaluation data
    """
    try:
        analysis = evaluator.analyze_position(board, multipv=multipv)
        
        return {
            "move_evaluations": [
                {
                    "uci": mv.uci,
                    "san": mv.san,
                    "centipawn": mv.centipawn,
                    "mate_in": mv.mate_in
                }
                for mv in analysis.move_evaluations
            ],
            "best_move_uci": analysis.best_move_uci,
            "best_move_san": analysis.best_move_san,
            "best_score_cp": analysis.best_score_cp,
        }
    except Exception as e:
        warnings.warn(f"Stockfish analysis failed: {e}")
        return {
            "move_evaluations": [],
            "best_move_uci": "",
            "best_move_san": "",
            "best_score_cp": 0,
        }


def add_eval_to_position(
    position: ChessPositionWithEval,
    eval_data: Dict[str, Any]
) -> ChessPositionWithEval:
    """
    Add Stockfish evaluation data to a position.
    """
    position.move_evaluations = eval_data.get("move_evaluations", [])
    position.best_move_uci = eval_data.get("best_move_uci", "")
    position.best_move_san = eval_data.get("best_move_san", "")
    position.best_score_cp = eval_data.get("best_score_cp", 0)
    
    # Calculate rank and CP loss for target move
    if position.move_evaluations:
        for i, mv in enumerate(position.move_evaluations):
            if mv["uci"] == position.target_move_uci:
                position.target_move_rank = i + 1
                position.target_move_cp_loss = position.best_score_cp - mv["centipawn"]
                break
        else:
            # Target move not found (shouldn't happen)
            position.target_move_rank = len(position.move_evaluations) + 1
            position.target_move_cp_loss = 9999
    
    return position


# =============================================================================
# Position Creation
# =============================================================================

def position_from_board_with_eval(
    board: chess.Board,
    target_move_uci: str,
    evaluator: Optional[Any] = None,
    white_elo: int = 1500,
    black_elo: int = 1500,
    move_number: int = 0,
    source: str = "game",
    multipv: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None
) -> ChessPositionWithEval:
    """
    Create a ChessPositionWithEval from a chess.Board.
    
    Args:
        board: Chess board position
        target_move_uci: The move to train on
        evaluator: StockfishEvaluator instance (optional)
        white_elo: White player ELO
        black_elo: Black player ELO
        move_number: Move number in game
        source: "game" or "puzzle"
        multipv: Number of moves to analyze
        config: Configuration dict
    
    Returns:
        ChessPositionWithEval instance
    """
    legal_moves = get_legal_moves_uci(board)
    first_legal = get_first_legal_move(board)
    board_utf = render_board_utf(board)
    side = "White" if board.turn else "Black"
    
    position = ChessPositionWithEval(
        fen=board.fen(),
        legal_moves_uci=legal_moves,
        target_move_uci=target_move_uci,
        first_legal_move=first_legal or "",
        board_utf=board_utf,
        side_to_move=side,
        white_elo=white_elo,
        black_elo=black_elo,
        move_number=move_number,
        source=source
    )
    
    # Add Stockfish evaluation if available
    if evaluator is not None:
        eval_data = analyze_position_with_stockfish(board, evaluator, multipv)
        position = add_eval_to_position(position, eval_data)
        
        # Compute loss weight from move quality
        if config is None:
            config = {}
        loss_config = config.get('loss_weighting', {})
        
        position.loss_weight = compute_loss_weight_from_quality(
            target_move_rank=position.target_move_rank,
            target_move_cp_loss=position.target_move_cp_loss,
            num_legal_moves=len(position.move_evaluations) or len(legal_moves.split()),
            weight_type=loss_config.get('weight_type', 'cp_loss'),
            min_weight=loss_config.get('min_weight', 0.1),
            max_weight=loss_config.get('max_weight', 2.0),
            **loss_config
        )
    else:
        # Fallback to ELO-based weighting
        position.loss_weight = compute_loss_weight_elo(white_elo, black_elo, config)
    
    return position


# =============================================================================
# Game Processing with Evaluation
# =============================================================================

def game_to_positions_with_eval(
    movetext: str,
    white_elo: int,
    black_elo: int,
    evaluator: Optional[Any] = None,
    skip_first_n: int = 4,
    skip_last_n: int = 2,
    sample_rate: float = 0.3,
    multipv: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    rng: Optional[random.Random] = None
) -> Iterator[ChessPositionWithEval]:
    """
    Convert a game's movetext to positions with Stockfish evaluations.
    """
    if rng is None:
        rng = random.Random()
    
    moves = parse_movetext(movetext)
    if len(moves) < skip_first_n + skip_last_n + 1:
        return
    
    board = chess.Board()
    
    for move_idx, move_san in enumerate(moves):
        if move_idx < skip_first_n:
            try:
                board.push_san(move_san)
            except (ValueError, chess.InvalidMoveError):
                return
            continue
        
        if move_idx >= len(moves) - skip_last_n:
            break
        
        if rng.random() > sample_rate:
            try:
                board.push_san(move_san)
            except (ValueError, chess.InvalidMoveError):
                return
            continue
        
        try:
            target_move = board.parse_san(move_san)
            target_uci = target_move.uci()
        except (ValueError, chess.InvalidMoveError):
            return
        
        position = position_from_board_with_eval(
            board=board,
            target_move_uci=target_uci,
            evaluator=evaluator,
            white_elo=white_elo,
            black_elo=black_elo,
            move_number=move_idx,
            source="game",
            multipv=multipv,
            config=config
        )
        
        yield position
        
        try:
            board.push_san(move_san)
        except (ValueError, chess.InvalidMoveError):
            return


def puzzle_to_positions_with_eval(
    puzzle: Dict[str, Any],
    evaluator: Optional[Any] = None,
    multipv: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None
) -> List[ChessPositionWithEval]:
    """
    Convert Lichess puzzle to multiple training positions with Stockfish evaluation.

    Extracts ALL solver's moves (odd-indexed: moves[1], moves[3], moves[5]...).

    Puzzle format:
        moves[0] = opponent's setup move
        moves[1] = solver's first move (extracted)
        moves[2] = opponent's response
        moves[3] = solver's second move (extracted)
        ...and so on

    Returns:
        List of ChessPositionWithEval objects, one for each solver's move.
    """
    positions = []

    try:
        fen = puzzle.get('FEN')
        moves_str = puzzle.get('Moves', '')

        if not fen or not moves_str:
            return positions

        moves = moves_str.split()
        if len(moves) < 2:
            return positions

        # Get puzzle rating
        puzzle_rating = puzzle.get('Rating', 1500)
        if puzzle_rating is None:
            puzzle_rating = 1500
        if isinstance(puzzle_rating, str):
            try:
                puzzle_rating = int(puzzle_rating)
            except ValueError:
                puzzle_rating = 1500

        board = chess.Board(fen)

        # Iterate through all moves in the puzzle
        for i, move_uci in enumerate(moves):
            try:
                move = chess.Move.from_uci(move_uci)
            except ValueError:
                break  # Invalid move format, stop

            if move not in board.legal_moves:
                break  # Illegal move, stop

            # Odd indices (1, 3, 5...) are solver's moves - extract these
            if i % 2 == 1:
                position = position_from_board_with_eval(
                    board=board,
                    target_move_uci=move_uci,
                    evaluator=evaluator,
                    white_elo=puzzle_rating,
                    black_elo=puzzle_rating,
                    move_number=i,
                    source="puzzle",
                    multipv=multipv,
                    config=config
                )
                positions.append(position)

            # Apply move to advance board state
            board.push(move)

        return positions

    except Exception:
        return positions


# Backward compatibility alias
def puzzle_to_position_with_eval(
    puzzle: Dict[str, Any],
    evaluator: Optional[Any] = None,
    multipv: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None
) -> Optional[ChessPositionWithEval]:
    """
    Convert Lichess puzzle to training position with Stockfish evaluation.

    DEPRECATED: Use puzzle_to_positions_with_eval() instead to get all solver's moves.
    This function only returns the first solver's move for backward compatibility.
    """
    positions = puzzle_to_positions_with_eval(puzzle, evaluator, multipv, config)
    return positions[0] if positions else None


# =============================================================================
# Batch Processing with Stockfish
# =============================================================================

def process_positions_batch(
    positions: List[Dict[str, Any]],
    stockfish_path: Optional[str] = None,
    depth: int = 12,
    num_workers: int = 4,
    multipv: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Add Stockfish evaluations to a batch of positions.
    
    Args:
        positions: List of position dicts (must have 'fen' and 'target_move_uci')
        stockfish_path: Path to Stockfish binary
        depth: Stockfish search depth
        num_workers: Number of parallel workers
        multipv: Number of moves to evaluate per position
        config: Configuration dict
    
    Returns:
        List of position dicts with added evaluation data
    """
    from .stockfish_eval import StockfishEvaluator
    
    def process_one(pos_dict):
        evaluator = StockfishEvaluator(
            stockfish_path=stockfish_path,
            depth=depth,
            threads=1
        )
        try:
            board = chess.Board(pos_dict['fen'])
            eval_data = analyze_position_with_stockfish(board, evaluator, multipv)
            
            # Create position object
            position = ChessPositionWithEval(
                fen=pos_dict['fen'],
                legal_moves_uci=pos_dict.get('legal_moves_uci', get_legal_moves_uci(board)),
                target_move_uci=pos_dict['target_move_uci'],
                first_legal_move=pos_dict.get('first_legal_move', get_first_legal_move(board) or ''),
                board_utf=pos_dict.get('board_utf', render_board_utf(board)),
                side_to_move=pos_dict.get('side_to_move', "White" if board.turn else "Black"),
                white_elo=pos_dict.get('white_elo', 1500),
                black_elo=pos_dict.get('black_elo', 1500),
                move_number=pos_dict.get('move_number', 0),
                source=pos_dict.get('source', 'game')
            )
            
            position = add_eval_to_position(position, eval_data)
            
            # Compute loss weight
            if config is None:
                cfg = {}
            else:
                cfg = config
            loss_config = cfg.get('loss_weighting', {})
            
            position.loss_weight = compute_loss_weight_from_quality(
                target_move_rank=position.target_move_rank,
                target_move_cp_loss=position.target_move_cp_loss,
                num_legal_moves=len(position.move_evaluations) or 1,
                weight_type=loss_config.get('weight_type', 'cp_loss'),
                min_weight=loss_config.get('min_weight', 0.1),
                max_weight=loss_config.get('max_weight', 2.0),
                **loss_config
            )
            
            return asdict(position)
            
        finally:
            evaluator.close()
    
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_one, pos): pos for pos in positions}
        for future in tqdm(as_completed(futures), total=len(positions), desc="Analyzing"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                warnings.warn(f"Failed to process position: {e}")
    
    return results


# =============================================================================
# Streaming with Evaluation (for large datasets)
# =============================================================================

def stream_positions_with_eval(
    dataset_name: str = "Lichess/standard-chess-games",
    stockfish_path: Optional[str] = None,
    depth: int = 12,
    min_elo: int = 1200,
    sample_rate: float = 0.3,
    skip_first_moves: int = 4,
    skip_last_moves: int = 2,
    multipv: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    max_positions: Optional[int] = None
) -> Iterator[Dict[str, Any]]:
    """
    Stream positions from games with real-time Stockfish analysis.
    
    Note: This is slower than batch processing but uses less memory.
    """
    from .stockfish_eval import StockfishEvaluator
    
    rng = random.Random(seed)
    
    evaluator = StockfishEvaluator(
        stockfish_path=stockfish_path,
        depth=depth,
        threads=1
    )
    
    try:
        dataset = load_dataset(
            dataset_name,
            split="train",
            streaming=True,
        )
        
        elo_weights = config.get('elo_weights', {
            1200: 0.05, 1400: 0.10, 1600: 0.15,
            1800: 0.25, 2000: 0.30, 2200: 0.20, 2400: 0.10
        }) if config else {}
        
        position_count = 0
        
        for game in dataset:
            if max_positions and position_count >= max_positions:
                break
            
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
            weight = 0.01
            for threshold in sorted(elo_weights.keys(), reverse=True):
                if avg_elo >= threshold:
                    weight = elo_weights[threshold]
                    break
            
            if rng.random() > weight:
                continue
            
            movetext = game.get('movetext', '') or game.get('moves', '')
            if not movetext:
                continue
            
            for pos in game_to_positions_with_eval(
                movetext=movetext,
                white_elo=white_elo,
                black_elo=black_elo,
                evaluator=evaluator,
                skip_first_n=skip_first_moves,
                skip_last_n=skip_last_moves,
                sample_rate=sample_rate,
                multipv=multipv,
                config=config,
                rng=rng
            ):
                yield asdict(pos)
                position_count += 1
                
                if max_positions and position_count >= max_positions:
                    break
    
    finally:
        evaluator.close()


# =============================================================================
# Dataset Creation with Evaluation
# =============================================================================

def preprocess_and_save_with_eval(
    output_path: str,
    target_size: int = 100_000,
    games_ratio: float = 0.7,
    stockfish_path: Optional[str] = None,
    stockfish_depth: int = 12,
    stockfish_workers: int = 8,
    multipv: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42
) -> Dataset:
    """
    Pre-process data with Stockfish evaluations and save to disk.
    
    This is the recommended approach for training:
    1. Collect positions from games/puzzles
    2. Batch analyze with Stockfish (parallelized)
    3. Save with evaluations for training
    
    Args:
        output_path: Where to save the dataset
        target_size: Number of positions to collect
        games_ratio: Ratio of game positions vs puzzles
        stockfish_path: Path to Stockfish binary
        stockfish_depth: Stockfish analysis depth
        stockfish_workers: Number of parallel Stockfish instances
        multipv: Moves to evaluate per position (None = all)
        config: Configuration dict
        seed: Random seed
    
    Returns:
        HuggingFace Dataset
    """
    if config is None:
        config = {}
    
    data_config = config.get('data', {})
    elo_weights = config.get('elo_weights', None)
    
    games_target = int(target_size * games_ratio)
    puzzles_target = target_size - games_target
    
    print(f"Target: {games_target:,} game positions, {puzzles_target:,} puzzles")
    print(f"Stockfish depth: {stockfish_depth}, workers: {stockfish_workers}")
    
    # Step 1: Collect positions (without Stockfish yet)
    print("\nStep 1: Collecting positions...")
    raw_positions = []
    
    # Collect game positions
    print("Processing games...")
    game_count = 0
    rng = random.Random(seed)
    
    dataset = load_dataset(
        data_config.get('games_dataset', 'Lichess/standard-chess-games'),
        split="train",
        streaming=True,
    )
    
    default_elo_weights = {
        1200: 0.05, 1400: 0.10, 1600: 0.15,
        1800: 0.25, 2000: 0.30, 2200: 0.20, 2400: 0.10
    }
    elo_weights = elo_weights or default_elo_weights
    
    for game in tqdm(dataset, total=games_target * 10, desc="Games"):
        if game_count >= games_target:
            break
        
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
        if avg_elo < data_config.get('min_elo', 1200):
            continue
        
        weight = 0.01
        for threshold in sorted(elo_weights.keys(), reverse=True):
            if avg_elo >= threshold:
                weight = elo_weights[threshold]
                break
        
        if rng.random() > weight:
            continue
        
        movetext = game.get('movetext', '') or game.get('moves', '')
        if not movetext:
            continue
        
        # Collect positions without evaluation
        for pos in game_to_positions_with_eval(
            movetext=movetext,
            white_elo=white_elo,
            black_elo=black_elo,
            evaluator=None,  # No Stockfish yet
            skip_first_n=data_config.get('skip_first_moves', 4),
            skip_last_n=data_config.get('skip_last_moves', 2),
            sample_rate=data_config.get('sample_rate', 0.3),
            config=config,
            rng=rng
        ):
            raw_positions.append(asdict(pos))
            game_count += 1
            if game_count >= games_target:
                break
    
    print(f"Collected {game_count:,} game positions")
    
    # Collect puzzle positions
    print("Processing puzzles...")
    puzzle_count = 0
    
    puzzle_dataset = load_dataset(
        data_config.get('puzzles_dataset', 'Lichess/chess-puzzles'),
        split="train",
        streaming=True,
    )
    
    for puzzle in tqdm(puzzle_dataset, total=puzzles_target * 2, desc="Puzzles"):
        if puzzle_count >= puzzles_target:
            break
        
        rating = puzzle.get('Rating', 1500) or 1500
        if isinstance(rating, str):
            rating = int(rating) if rating.isdigit() else 1500
        
        min_rating = data_config.get('min_puzzle_rating', 1000)
        max_rating = data_config.get('max_puzzle_rating', 2500)
        
        if rating < min_rating or rating > max_rating:
            continue

        # Get all solver's moves from this puzzle
        positions = puzzle_to_positions_with_eval(puzzle, evaluator=None, config=config)
        for pos in positions:
            raw_positions.append(asdict(pos))
            puzzle_count += 1
            if puzzle_count >= puzzles_target:
                break

        if puzzle_count >= puzzles_target:
            break

    print(f"Collected {puzzle_count:,} puzzle positions")
    print(f"Total raw positions: {len(raw_positions):,}")
    
    # Step 2: Batch analyze with Stockfish
    print("\nStep 2: Analyzing with Stockfish...")
    analyzed_positions = process_positions_batch(
        positions=raw_positions,
        stockfish_path=stockfish_path,
        depth=stockfish_depth,
        num_workers=stockfish_workers,
        multipv=multipv,
        config=config
    )
    
    print(f"Analyzed {len(analyzed_positions):,} positions")
    
    # Step 3: Shuffle and save
    print("\nStep 3: Shuffling and saving...")
    rng.shuffle(analyzed_positions)
    
    dataset = Dataset.from_list(analyzed_positions)
    dataset.save_to_disk(output_path)
    
    # Print statistics
    print(f"\nSaved to {output_path}")
    print(f"Total examples: {len(analyzed_positions):,}")
    
    # Quality statistics
    ranks = [p['target_move_rank'] for p in analyzed_positions if p.get('target_move_rank', 0) > 0]
    cp_losses = [p['target_move_cp_loss'] for p in analyzed_positions if p.get('target_move_cp_loss', 0) >= 0]
    weights = [p['loss_weight'] for p in analyzed_positions if 'loss_weight' in p]
    
    if ranks:
        print(f"\nMove quality stats:")
        print(f"  Avg rank: {sum(ranks)/len(ranks):.2f}")
        print(f"  Top-1 moves: {sum(1 for r in ranks if r == 1):,} ({100*sum(1 for r in ranks if r == 1)/len(ranks):.1f}%)")
        print(f"  Top-3 moves: {sum(1 for r in ranks if r <= 3):,} ({100*sum(1 for r in ranks if r <= 3)/len(ranks):.1f}%)")
    
    if cp_losses:
        print(f"\nCentipawn loss stats:")
        print(f"  Avg CP loss: {sum(cp_losses)/len(cp_losses):.1f}")
        print(f"  ≤20cp: {sum(1 for l in cp_losses if l <= 20):,} ({100*sum(1 for l in cp_losses if l <= 20)/len(cp_losses):.1f}%)")
        print(f"  ≤50cp: {sum(1 for l in cp_losses if l <= 50):,} ({100*sum(1 for l in cp_losses if l <= 50)/len(cp_losses):.1f}%)")
    
    if weights:
        print(f"\nLoss weight stats:")
        print(f"  Min: {min(weights):.3f}")
        print(f"  Max: {max(weights):.3f}")
        print(f"  Mean: {sum(weights)/len(weights):.3f}")
    
    return dataset


if __name__ == "__main__":
    # Test with a small batch
    print("Testing data processing with Stockfish...")
    
    test_config = {
        'loss_weighting': {
            'weight_type': 'cp_loss',
            'min_weight': 0.1,
            'max_weight': 2.0,
            'max_cp_loss': 200
        }
    }
    
    # Test loss weight computation
    print("\nTesting loss weight computation:")
    test_cases = [
        (1, 0, 20),    # Best move
        (2, 15, 20),   # Second best, 15cp loss
        (3, 50, 20),   # Third, 50cp loss
        (5, 100, 20),  # Fifth, 100cp loss
        (10, 200, 20), # Tenth, 200cp loss
    ]
    
    for rank, cp_loss, num_moves in test_cases:
        weight = compute_loss_weight_from_quality(
            rank, cp_loss, num_moves,
            weight_type='cp_loss',
            min_weight=0.1,
            max_weight=2.0,
            max_cp_loss=200
        )
        print(f"  Rank {rank}, CP loss {cp_loss}: weight = {weight:.3f}")
