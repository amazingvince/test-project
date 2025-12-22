"""
Tokenizer helpers for chess move modeling.

This repo supports two common approaches for emitting UCI moves:

1) **Tags-only** (recommended for SFT):
   - Add only the move delimiters: `<uci_move>` and `</uci_move>`
   - Keep the move itself (e.g. `e2e4`) as regular text, which will be
     tokenized into 1+ base-model tokens.

2) **Tags + move vocabulary** (recommended for distillation-on-one-token):
   - Add `<uci_move>` and `</uci_move>` plus an explicit vocabulary containing
     all 8,064 UCI strings (including promotions).
   - This makes every legal move a single token, enabling a direct KL loss over
     the move distribution at the `<uci_move>` position.

The functions here are intentionally small so both `sft/train.py` and
`distill/train.py` can share the same tokenizer behavior.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Literal, Optional

import chess

ChessTokenizerMode = Literal["tags_only", "tags_and_moves"]


def generate_all_uci_moves() -> List[str]:
    """
    Generate the complete set of UCI move strings.

    This enumerates:
    - all from/to square pairs (excluding from==to)
    - plus promotion suffixes (`q`, `r`, `b`, `n`) for moves that land on the 1st
      or 8th rank (rank index 0 or 7 in python-chess).

    The resulting list is stable and contains 8,064 unique strings.
    """

    moves: List[str] = []
    for from_sq in range(64):
        for to_sq in range(64):
            if from_sq == to_sq:
                continue
            from_str = chess.SQUARE_NAMES[from_sq]
            to_str = chess.SQUARE_NAMES[to_sq]
            uci_move = f"{from_str}{to_str}"
            moves.append(uci_move)

            to_rank = chess.square_rank(to_sq)
            if to_rank in (0, 7):
                for promo in ("q", "r", "b", "n"):
                    moves.append(f"{uci_move}{promo}")

    # Deterministic uniqueness check (should always be true).
    if len(moves) != len(set(moves)):
        raise RuntimeError("generate_all_uci_moves() produced duplicates.")

    return moves


def add_chess_tokens(
    *,
    tokenizer,
    model=None,
    mode: ChessTokenizerMode,
    add_think_tags: bool = False,
    all_uci_moves: Optional[Iterable[str]] = None,
    pad_to_multiple_of: Optional[int] = 64,
) -> Dict[str, int]:
    """
    Add chess-related tokens to a tokenizer + model.

    Args:
        tokenizer: HF tokenizer instance.
        model: Optional HF causal LM. When provided, embeddings are resized to
            match the tokenizer after adding tokens.
        mode:
            - "tags_only": add `<uci_move>` and `</uci_move>` only.
            - "tags_and_moves": add tags + all UCI move strings as tokens.
        add_think_tags: When True, also add `<think>` and `</think>` as special
            tokens. This is optional: the model can learn them as plain text.
        all_uci_moves: Optional iterable of move strings (defaults to the full set).
        pad_to_multiple_of: Passed to `resize_token_embeddings` for tensor-core friendly
            vocab padding. Use `None` to disable.

    Returns:
        Dict with counts of added tokens, e.g. `{"special_tokens": 2, "move_tokens": 8064}`.
    """

    added = {"special_tokens": 0, "move_tokens": 0}

    special: List[str] = ["<uci_move>", "</uci_move>"]
    if add_think_tags:
        special = ["<think>", "</think>", *special]

    added["special_tokens"] = int(
        tokenizer.add_special_tokens({"additional_special_tokens": special})
    )

    if mode == "tags_and_moves":
        if all_uci_moves is None:
            all_uci_moves = generate_all_uci_moves()
        added["move_tokens"] = int(
            tokenizer.add_tokens(list(all_uci_moves), special_tokens=False)
        )
    elif mode != "tags_only":
        raise ValueError(f"Unknown chess tokenizer mode: {mode}")

    if model is not None and (added["special_tokens"] or added["move_tokens"]):
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=pad_to_multiple_of)
        if hasattr(model, "tie_weights"):
            model.tie_weights()

    return added


def build_move_to_token_id(
    *,
    tokenizer,
    all_uci_moves: Optional[Iterable[str]] = None,
) -> Dict[str, int]:
    """
    Build a mapping from UCI move string -> token id for single-token moves.

    This is only meaningful when the tokenizer vocabulary explicitly contains
    UCI move strings as individual tokens (i.e. mode `tags_and_moves`).
    """

    if all_uci_moves is None:
        all_uci_moves = generate_all_uci_moves()

    move_to_token: Dict[str, int] = {}
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for uci_move in all_uci_moves:
        ids = tokenizer.encode(uci_move, add_special_tokens=False)
        if len(ids) == 1 and (unk_id is None or ids[0] != unk_id):
            move_to_token[uci_move] = int(ids[0])
    return move_to_token
