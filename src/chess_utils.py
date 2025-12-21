"""
Chess utilities for board rendering and move parsing.
"""

import re
import chess
from dataclasses import dataclass
from typing import Optional, List


# Unicode piece symbols for board rendering
UNICODE_PIECES = {
    'P': '♙', 'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔',
    'p': '♟', 'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚',
}


@dataclass
class ChessPosition:
    """Represents a single chess position for training."""
    fen: str
    legal_moves_uci: str  # Space-separated list
    target_move_uci: str
    first_legal_move: str  # For format example in prompt
    board_utf: str
    side_to_move: str
    # Metadata
    white_elo: int
    black_elo: int
    move_number: int
    source: str  # "game" or "puzzle"
    # Loss weighting
    loss_weight: float = 1.0  # Weight for cross-entropy loss


def render_board_utf(board: chess.Board) -> str:
    """
    Render board in UTF-8 format matching competition format.
    
    Example output:
       a  b  c  d  e  f  g  h  
       +------------------------+
    8 | ♜  ♞  ♝  ♛  ♚  ♝  ♞  ♜ | 8
    7 | ♟  ♟  ♟  ♟  ♟  ♟  ♟  ♟ | 7
    ...
    """
    lines = []
    files = 'abcdefgh'
    
    # Top coordinates
    coord_line = "   " + "".join(f" {f} " for f in files) + "  "
    lines.append(coord_line)
    lines.append("   +" + "-" * 24 + "+")
    
    for rank in range(7, -1, -1):  # 8 down to 1
        rank_num = str(rank + 1)
        line_parts = [f"{rank_num} |"]
        
        for file in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece:
                char = UNICODE_PIECES.get(piece.symbol(), piece.symbol())
            else:
                char = "·"
            line_parts.append(f" {char} ")
        
        line_parts.append(f"| {rank_num}")
        lines.append("".join(line_parts))
    
    lines.append("   +" + "-" * 24 + "+")
    lines.append(coord_line)
    
    return "\n".join(lines)


def parse_movetext(movetext: str) -> List[str]:
    """
    Parse Lichess movetext format to list of SAN moves.
    
    Movetext format: "1. e4 e5 2. Nf3 Nc6 3. Bb5 ..."
    Also handles: "1. e4 { [%clk 0:05:00] } e5 { [%clk 0:05:00] } ..."
    
    Returns:
        List of SAN moves: ["e4", "e5", "Nf3", "Nc6", ...]
    """
    if not movetext:
        return []
    
    # Remove clock annotations { [%clk ...] }
    cleaned = re.sub(r'\{[^}]*\}', '', movetext)
    
    # Remove move numbers and dots (1. or 1...)
    cleaned = re.sub(r'\d+\.+\s*', '', cleaned)
    
    # Remove game results
    cleaned = re.sub(r'(1-0|0-1|1/2-1/2|\*)\s*$', '', cleaned)
    
    # Remove evaluation annotations
    cleaned = re.sub(r'\?+|\!+', '', cleaned)
    
    # Split and filter
    moves = cleaned.split()
    return [m.strip() for m in moves if m.strip()]


def get_legal_moves_uci(board: chess.Board) -> str:
    """Get space-separated string of all legal moves in UCI format."""
    return ' '.join(m.uci() for m in board.legal_moves)


def get_first_legal_move(board: chess.Board) -> Optional[str]:
    """Get the first legal move in UCI format (for prompt example)."""
    try:
        return next(iter(board.legal_moves)).uci()
    except StopIteration:
        return None


def validate_uci_move(board: chess.Board, uci_move: str) -> bool:
    """Check if a UCI move string is legal in the current position."""
    try:
        move = chess.Move.from_uci(uci_move)
        return move in board.legal_moves
    except (ValueError, chess.InvalidMoveError):
        return False


def extract_uci_from_response(response: str) -> Optional[str]:
    """
    Extract UCI move from model response.
    
    Expected format: <uci_move>e2e4</uci_move>
    """
    match = re.search(r'<uci_move>([a-h][1-8][a-h][1-8][qrbn]?)</uci_move>', response)
    if match:
        return match.group(1)
    return None


def position_from_board(
    board: chess.Board,
    target_move_uci: str,
    white_elo: int = 1500,
    black_elo: int = 1500,
    move_number: int = 0,
    source: str = "game"
) -> ChessPosition:
    """Create a ChessPosition from a chess.Board and target move."""
    legal_moves = get_legal_moves_uci(board)
    first_legal = get_first_legal_move(board)
    board_utf = render_board_utf(board)
    side = "White" if board.turn else "Black"
    
    return ChessPosition(
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


def setup_position_from_fen(fen: str) -> Optional[chess.Board]:
    """Safely create a board from FEN string."""
    try:
        board = chess.Board(fen)
        return board
    except ValueError:
        return None


if __name__ == "__main__":
    # Test the utilities
    board = chess.Board()
    print("Starting position:")
    print(render_board_utf(board))
    print(f"\nLegal moves: {get_legal_moves_uci(board)}")
    print(f"First legal: {get_first_legal_move(board)}")
    
    # Test movetext parsing
    movetext = "1. e4 { [%clk 0:05:00] } e5 2. Nf3 Nc6 3. Bb5 a6 1-0"
    moves = parse_movetext(movetext)
    print(f"\nParsed moves: {moves}")
