"""
Chess helpers used across SFT, distillation, and evaluation.

This module is intentionally small and stable:
- Parse movetext (SAN) from common Lichess exports
- Render a readable ASCII board (for prompts and reports)
- Validate/extract UCI moves in model outputs
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

import chess


@dataclass
class ChessPosition:
    """
    A single chess training position with minimal metadata.

    Fields are kept close to what the training scripts need so datasets can be
    stored as plain dicts (via `dataclasses.asdict`) without custom serialization.
    """

    fen: str
    legal_moves_uci: str
    target_move_uci: str
    first_legal_move: str
    board_utf: str
    side_to_move: str
    white_elo: int
    black_elo: int
    move_number: int
    source: str  # "game" or "puzzle"
    loss_weight: float = 1.0


def render_board_utf(board: chess.Board) -> str:
    """
    Render an ASCII board with coordinates.

    The output is stable across terminals and avoids Unicode piece glyphs, which
    can render inconsistently depending on font/encoding.

    Example:
       a  b  c  d  e  f  g  h
       +------------------------+
    8 | r  n  b  q  k  b  n  r | 8
    7 | p  p  p  p  p  p  p  p | 7
    ...
    """

    files = "abcdefgh"
    coord_line = "   " + "".join(f" {file_letter} " for file_letter in files) + "  "

    lines: List[str] = [coord_line, "   +" + "-" * 24 + "+"]
    for rank in range(7, -1, -1):
        rank_label = str(rank + 1)
        row: List[str] = [f"{rank_label} |"]
        for file_idx in range(8):
            square = chess.square(file_idx, rank)
            piece = board.piece_at(square)
            row.append(f" {(piece.symbol() if piece else '.')} ")
        row.append(f"| {rank_label}")
        lines.append("".join(row))
    lines.extend(["   +" + "-" * 24 + "+", coord_line])
    return "\n".join(lines)


def parse_movetext(movetext: str) -> List[str]:
    """
    Parse a Lichess movetext string into SAN moves.

    Supports common Lichess PGN-like movetext formats, including clock blocks:
        "1. e4 { [%clk 0:05:00] } e5 2. Nf3 Nc6 3. Bb5 a6 1-0"

    Returns:
        A list of SAN moves in order (e.g. ["e4", "e5", "Nf3", ...]).
    """

    if not movetext:
        return []

    cleaned = re.sub(r"\{[^}]*\}", "", movetext)  # {...} annotations
    cleaned = re.sub(r"\d+\.+\s*", "", cleaned)  # "1." / "1..." prefixes
    cleaned = re.sub(r"(1-0|0-1|1/2-1/2|\*)\s*$", "", cleaned)  # results
    cleaned = re.sub(r"\?+|\!+", "", cleaned)  # simple punctuation nags

    return [token.strip() for token in cleaned.split() if token.strip()]


def get_legal_moves_uci(board: chess.Board) -> str:
    """Return all legal moves as a space-separated UCI string."""

    return " ".join(move.uci() for move in board.legal_moves)


def get_first_legal_move(board: chess.Board) -> Optional[str]:
    """Return the first legal move in UCI form, or `None` if none exist."""

    move = next(iter(board.legal_moves), None)
    return move.uci() if move else None


def validate_uci_move(board: chess.Board, uci_move: str) -> bool:
    """Return True if `uci_move` is well-formed and legal in `board`."""

    try:
        move = chess.Move.from_uci(uci_move)
    except ValueError:
        return False
    return move in board.legal_moves


_UCI_TAG_RE = re.compile(r"<uci_move>\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*</uci_move>")
_UCI_BARE_RE = re.compile(r"(?i)\b([a-h][1-8][a-h][1-8][qrbn]?)\b")


def extract_uci_from_response(response: str) -> Optional[str]:
    """
    Extract a UCI move from an assistant response.

    Expected format:
        <uci_move>e2e4</uci_move>
    """

    text = response or ""
    match = _UCI_TAG_RE.search(text)
    if match:
        return match.group(1).lower()

    # Fallback: some models will emit a bare UCI move (often inside <think> ... </think>)
    # before they learn the strict `<uci_move>...</uci_move>` format. Returning the last
    # match keeps evaluation usable while we still track format compliance separately.
    matches = list(_UCI_BARE_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).lower()


def trim_generated_token_ids(
    generated_ids: List[int],
    *,
    close_tag_id: Optional[int] = None,
    eos_token_ids: Sequence[int] = (),
    pad_token_id: Optional[int] = None,
) -> List[int]:
    """
    Trim a generated token sequence at the first stop token.

    Hugging Face `generate()` pads shorter generations to match the longest item
    in a batch. When `pad_token_id == eos_token_id` this can appear as repeated
    `<|endoftext|>` tokens in decoded output.

    This helper:
    - cuts at the first `</uci_move>` token if present (inclusive)
    - otherwise cuts at the first EOS token (exclusive)
    - strips trailing PAD/EOS tokens
    """

    if not generated_ids:
        return []

    close_tag_id = int(close_tag_id) if close_tag_id is not None else None
    if close_tag_id is not None and close_tag_id < 0:
        close_tag_id = None
    eos_set = set(int(v) for v in eos_token_ids if v is not None)
    if close_tag_id is not None:
        eos_set.discard(close_tag_id)

    stop_idx: Optional[int] = None
    include_stop = False

    if close_tag_id is not None:
        try:
            stop_idx = generated_ids.index(close_tag_id)
            include_stop = True
        except ValueError:
            stop_idx = None

    if eos_set:
        for idx, token_id in enumerate(generated_ids):
            if token_id in eos_set:
                if stop_idx is None or idx < stop_idx:
                    stop_idx = idx
                    include_stop = False
                break

    if stop_idx is not None:
        generated_ids = generated_ids[: stop_idx + (1 if include_stop else 0)]

    strip_ids = set()
    if pad_token_id is not None:
        strip_ids.add(int(pad_token_id))
    strip_ids.update(eos_set)

    while generated_ids and generated_ids[-1] in strip_ids:
        generated_ids.pop()

    return generated_ids


def position_from_board(
    board: chess.Board,
    target_move_uci: str,
    *,
    white_elo: int = 1500,
    black_elo: int = 1500,
    move_number: int = 0,
    source: str = "game",
) -> ChessPosition:
    """
    Build a `ChessPosition` from a `chess.Board` and a target move.

    This is a convenience helper for dataset generation. The returned object is
    fully self-contained (FEN, legal move list, and a rendered board).
    """

    legal_moves = get_legal_moves_uci(board)
    first_legal = get_first_legal_move(board) or ""
    board_utf = render_board_utf(board)
    side_to_move = "White" if board.turn == chess.WHITE else "Black"

    return ChessPosition(
        fen=board.fen(),
        legal_moves_uci=legal_moves,
        target_move_uci=target_move_uci,
        first_legal_move=first_legal,
        board_utf=board_utf,
        side_to_move=side_to_move,
        white_elo=white_elo,
        black_elo=black_elo,
        move_number=move_number,
        source=source,
    )


def setup_position_from_fen(fen: str) -> Optional[chess.Board]:
    """Create a `chess.Board` from FEN, returning `None` if it is invalid."""

    try:
        return chess.Board(fen)
    except ValueError:
        return None
