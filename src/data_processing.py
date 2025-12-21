"""
Data processing module for Chess SFT training.

Handles loading and processing of:
- Lichess game positions
- Lichess puzzles

Supports both streaming and pre-processing modes.
Includes ELO-based loss weighting.
"""

import math
import random
import chess
from typing import Iterator, Dict, Any, Optional, List
from dataclasses import asdict
from datasets import load_dataset, IterableDataset, Dataset, interleave_datasets

from .chess_utils import (
    ChessPosition,
    parse_movetext,
    render_board_utf,
    get_legal_moves_uci,
    get_first_legal_move,
    position_from_board
)


# =============================================================================
# Loss Weight Functions
# =============================================================================

def compute_loss_weight_linear(
    elo: float,
    elo_min: float = 1200,
    elo_max: float = 2400,
    weight_min: float = 0.5,
    weight_max: float = 2.0
) -> float:
    """
    Linear loss weight based on ELO.
    Higher ELO = higher weight.
    """
    # Normalize ELO to [0, 1]
    normalized = (elo - elo_min) / (elo_max - elo_min)
    normalized = max(0.0, min(1.0, normalized))
    
    # Map to weight range
    weight = weight_min + normalized * (weight_max - weight_min)
    return weight


def compute_loss_weight_gaussian(
    elo: float,
    target_elo: float = 1900,
    sigma: float = 400,
    weight_min: float = 0.5,
    weight_max: float = 2.0
) -> float:
    """
    Gaussian loss weight centered on target ELO.
    Positions near target_elo get highest weight.
    """
    # Gaussian centered on target
    gaussian = math.exp(-((elo - target_elo) ** 2) / (2 * sigma ** 2))
    
    # Map to weight range
    weight = weight_min + gaussian * (weight_max - weight_min)
    return weight


def compute_loss_weight_step(
    elo: float,
    elo_weights: Dict[int, float]
) -> float:
    """
    Step function loss weight based on ELO brackets.
    Uses the same weights as sampling.
    """
    sorted_thresholds = sorted(elo_weights.keys(), reverse=True)
    
    for threshold in sorted_thresholds:
        if elo >= threshold:
            # Normalize to reasonable range (weights are sampling probs)
            # Scale so max sampling weight -> 2.0, min -> 0.5
            raw_weight = elo_weights[threshold]
            return 0.5 + (raw_weight / 0.30) * 1.5  # 0.30 is max weight
    
    return 0.5


def compute_loss_weight(
    white_elo: int,
    black_elo: int,
    config: Optional[Dict[str, Any]] = None
) -> float:
    """
    Compute loss weight for a position based on ELO.
    
    Args:
        white_elo: White player's rating
        black_elo: Black player's rating
        config: Loss weighting configuration
    
    Returns:
        Loss weight (typically 0.5 to 2.0)
    """
    if config is None:
        config = {}
    
    loss_config = config.get('loss_weighting', {})
    
    if not loss_config.get('enabled', True):
        return 1.0
    
    avg_elo = (white_elo + black_elo) / 2
    
    method = loss_config.get('function', 'gaussian')
    weight_min = loss_config.get('min_weight', 0.5)
    weight_max = loss_config.get('max_weight', 2.0)
    
    if method == 'linear':
        return compute_loss_weight_linear(
            elo=avg_elo,
            elo_min=loss_config.get('elo_min', 1200),
            elo_max=loss_config.get('elo_max', 2400),
            weight_min=weight_min,
            weight_max=weight_max
        )
    elif method == 'gaussian':
        return compute_loss_weight_gaussian(
            elo=avg_elo,
            target_elo=loss_config.get('target_elo', 1900),
            sigma=loss_config.get('gaussian_sigma', 400),
            weight_min=weight_min,
            weight_max=weight_max
        )
    elif method == 'step':
        elo_weights = config.get('elo_weights', {})
        return compute_loss_weight_step(avg_elo, elo_weights)
    else:
        return 1.0


# =============================================================================
# Sampling Weight Functions
# =============================================================================

def get_elo_weight(
    white_elo: int, 
    black_elo: int, 
    elo_weights: Dict[int, float]
) -> float:
    """
    Get sampling weight based on average ELO.
    """
    avg_elo = (white_elo + black_elo) / 2
    sorted_thresholds = sorted(elo_weights.keys(), reverse=True)
    
    for threshold in sorted_thresholds:
        if avg_elo >= threshold:
            return elo_weights[threshold]
    
    return 0.01


# =============================================================================
# Game Processing
# =============================================================================

def game_to_positions(
    movetext: str,
    white_elo: int,
    black_elo: int,
    skip_first_n: int = 4,
    skip_last_n: int = 2,
    sample_rate: float = 0.3,
    config: Optional[Dict[str, Any]] = None,
    rng: Optional[random.Random] = None
) -> Iterator[ChessPosition]:
    """
    Convert a game's movetext to individual positions.
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
        
        pos = position_from_board(
            board=board,
            target_move_uci=target_uci,
            white_elo=white_elo,
            black_elo=black_elo,
            move_number=move_idx,
            source="game"
        )
        
        # Add loss weight
        pos.loss_weight = compute_loss_weight(white_elo, black_elo, config)
        
        yield pos
        
        try:
            board.push_san(move_san)
        except (ValueError, chess.InvalidMoveError):
            return


def puzzle_to_positions(
    puzzle: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> List[ChessPosition]:
    """
    Convert Lichess puzzle to multiple training positions.

    Extracts ALL solver's moves (odd-indexed: moves[1], moves[3], moves[5]...).

    Puzzle format:
        moves[0] = opponent's setup move
        moves[1] = solver's first move (extracted)
        moves[2] = opponent's response
        moves[3] = solver's second move (extracted)
        ...and so on

    Returns:
        List of ChessPosition objects, one for each solver's move.
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

        # Get puzzle rating for loss weighting
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
                pos = position_from_board(
                    board=board,
                    target_move_uci=move_uci,
                    white_elo=puzzle_rating,
                    black_elo=puzzle_rating,
                    move_number=i,
                    source="puzzle"
                )

                # Add loss weight (puzzles use their rating as ELO proxy)
                pos.loss_weight = compute_loss_weight(puzzle_rating, puzzle_rating, config)
                positions.append(pos)

            # Apply move to advance board state
            board.push(move)

        return positions

    except Exception:
        return positions


# Backward compatibility alias
def puzzle_to_position(
    puzzle: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> Optional[ChessPosition]:
    """
    Convert Lichess puzzle to training position.

    DEPRECATED: Use puzzle_to_positions() instead to get all solver's moves.
    This function only returns the first solver's move for backward compatibility.
    """
    positions = puzzle_to_positions(puzzle, config)
    return positions[0] if positions else None


# =============================================================================
# Streaming Functions
# =============================================================================

def stream_game_positions(
    dataset_name: str = "Lichess/standard-chess-games",
    min_elo: int = 1200,
    elo_weights: Optional[Dict[int, float]] = None,
    sample_rate: float = 0.3,
    skip_first_moves: int = 4,
    skip_last_moves: int = 2,
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42
) -> Iterator[Dict[str, Any]]:
    """
    Stream positions from Lichess games with ELO weighting.
    """
    if elo_weights is None:
        elo_weights = {
            1200: 0.05, 1400: 0.10, 1600: 0.15,
            1800: 0.25, 2000: 0.30, 2200: 0.20, 2400: 0.10
        }
    
    rng = random.Random(seed)
    
    dataset = load_dataset(
        dataset_name,
        split="train",
        streaming=True,
    )
    
    for game in dataset:
        if game.get('Termination') == 'Time forfeit':
            continue
        
        result = game.get('Result', '')
        if result not in ['1-0', '0-1', '1/2-1/2']:
            continue
        
        white_elo = game.get('WhiteElo', 1500)
        black_elo = game.get('BlackElo', 1500)
        
        # Handle None or missing ELO
        if white_elo is None:
            white_elo = 1500
        if black_elo is None:
            black_elo = 1500
        
        # Handle string ELO values
        if isinstance(white_elo, str):
            try:
                white_elo = int(white_elo)
            except ValueError:
                white_elo = 1500
        if isinstance(black_elo, str):
            try:
                black_elo = int(black_elo)
            except ValueError:
                black_elo = 1500
        
        avg_elo = (white_elo + black_elo) / 2
        
        if avg_elo < min_elo:
            continue
        
        weight = get_elo_weight(white_elo, black_elo, elo_weights)
        if rng.random() > weight:
            continue
        
        movetext = game.get('movetext', '') or game.get('moves', '')
        if not movetext:
            continue
        
        for pos in game_to_positions(
            movetext=movetext,
            white_elo=white_elo,
            black_elo=black_elo,
            skip_first_n=skip_first_moves,
            skip_last_n=skip_last_moves,
            sample_rate=sample_rate,
            config=config,
            rng=rng
        ):
            yield asdict(pos)


def stream_puzzle_positions(
    dataset_name: str = "Lichess/chess-puzzles",
    min_rating: int = 1000,
    max_rating: int = 2500,
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42
) -> Iterator[Dict[str, Any]]:
    """
    Stream positions from Lichess puzzles.

    Yields all solver's moves from each puzzle (moves[1], moves[3], moves[5]...).
    """
    dataset = load_dataset(
        dataset_name,
        split="train",
        streaming=True,
    )

    for puzzle in dataset:
        rating = puzzle.get('Rating', 1500)

        # Handle None rating
        if rating is None:
            rating = 1500

        if isinstance(rating, str):
            try:
                rating = int(rating)
            except ValueError:
                rating = 1500

        if rating < min_rating or rating > max_rating:
            continue

        # Get all solver's moves from this puzzle
        positions = puzzle_to_positions(puzzle, config)
        for pos in positions:
            yield asdict(pos)


def create_streaming_dataset(
    games_ratio: float = 0.7,
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42
) -> IterableDataset:
    """
    Create a streaming dataset that interleaves games and puzzles.
    """
    if config is None:
        config = {}
    
    data_config = config.get('data', {})
    elo_weights = config.get('elo_weights', None)
    
    def game_gen():
        yield from stream_game_positions(
            dataset_name=data_config.get('games_dataset', 'Lichess/standard-chess-games'),
            min_elo=data_config.get('min_elo', 1200),
            elo_weights=elo_weights,
            sample_rate=data_config.get('sample_rate', 0.3),
            skip_first_moves=data_config.get('skip_first_moves', 4),
            skip_last_moves=data_config.get('skip_last_moves', 2),
            config=config,
            seed=seed
        )
    
    def puzzle_gen():
        yield from stream_puzzle_positions(
            dataset_name=data_config.get('puzzles_dataset', 'Lichess/chess-puzzles'),
            min_rating=data_config.get('min_puzzle_rating', 1000),
            max_rating=data_config.get('max_puzzle_rating', 2500),
            config=config,
            seed=seed
        )
    
    games_ds = IterableDataset.from_generator(game_gen)
    puzzles_ds = IterableDataset.from_generator(puzzle_gen)
    
    puzzles_ratio = 1.0 - games_ratio
    combined = interleave_datasets(
        [games_ds, puzzles_ds],
        probabilities=[games_ratio, puzzles_ratio],
        seed=seed,
        stopping_strategy="all_exhausted"
    )
    
    buffer_size = data_config.get('shuffle_buffer_size', 10000)
    combined = combined.shuffle(seed=seed, buffer_size=buffer_size)
    
    return combined


def preprocess_and_save(
    output_path: str,
    target_size: int = 5_000_000,
    games_ratio: float = 0.7,
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42
) -> Dataset:
    """
    Pre-process data and save to disk.
    """
    from tqdm import tqdm
    
    if config is None:
        config = {}
    
    games_target = int(target_size * games_ratio)
    puzzles_target = target_size - games_target
    
    print(f"Target: {games_target:,} game positions, {puzzles_target:,} puzzles")
    
    examples = []
    
    print("Processing games...")
    data_config = config.get('data', {})
    elo_weights = config.get('elo_weights', None)
    
    game_count = 0
    for pos in tqdm(
        stream_game_positions(
            dataset_name=data_config.get('games_dataset', 'Lichess/standard-chess-games'),
            min_elo=data_config.get('min_elo', 1200),
            elo_weights=elo_weights,
            sample_rate=data_config.get('sample_rate', 0.3),
            skip_first_moves=data_config.get('skip_first_moves', 4),
            skip_last_moves=data_config.get('skip_last_moves', 2),
            config=config,
            seed=seed
        ),
        total=games_target,
        desc="Games"
    ):
        examples.append(pos)
        game_count += 1
        if game_count >= games_target:
            break
    
    print(f"Collected {game_count:,} game positions")
    
    print("Processing puzzles...")
    puzzle_count = 0
    for pos in tqdm(
        stream_puzzle_positions(
            dataset_name=data_config.get('puzzles_dataset', 'Lichess/chess-puzzles'),
            min_rating=data_config.get('min_puzzle_rating', 1000),
            max_rating=data_config.get('max_puzzle_rating', 2500),
            config=config,
            seed=seed
        ),
        total=puzzles_target,
        desc="Puzzles"
    ):
        examples.append(pos)
        puzzle_count += 1
        if puzzle_count >= puzzles_target:
            break
    
    print(f"Collected {puzzle_count:,} puzzle positions")
    
    print("Shuffling...")
    rng = random.Random(seed)
    rng.shuffle(examples)
    
    print("Creating dataset...")
    dataset = Dataset.from_list(examples)
    
    print(f"Saving to {output_path}...")
    dataset.save_to_disk(output_path)
    print(f"Saved {len(examples):,} examples")
    
    # Print loss weight statistics
    weights = [e['loss_weight'] for e in examples if 'loss_weight' in e]
    if weights:
        print(f"\nLoss weight stats:")
        print(f"  Min: {min(weights):.3f}")
        print(f"  Max: {max(weights):.3f}")
        print(f"  Mean: {sum(weights)/len(weights):.3f}")
    
    return dataset


if __name__ == "__main__":
    # Test loss weight functions
    print("Testing loss weight functions...")
    
    test_elos = [1200, 1400, 1600, 1800, 1900, 2000, 2200, 2400]
    
    print("\nLinear weighting:")
    for elo in test_elos:
        w = compute_loss_weight_linear(elo)
        print(f"  ELO {elo}: weight = {w:.3f}")
    
    print("\nGaussian weighting (target=1900, sigma=400):")
    for elo in test_elos:
        w = compute_loss_weight_gaussian(elo, target_elo=1900, sigma=400)
        print(f"  ELO {elo}: weight = {w:.3f}")