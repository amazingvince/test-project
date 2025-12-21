#!/usr/bin/env python3
"""
Quick tokenizer + streaming sanity check for chess distillation.

- Loads config and tokenizer
- Optionally adds <uci_move> tags and UCI move tokens
- Prints tokenizer coverage and sample tokenization
- Streams a few positions and prints prompt + token alignment info
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml

try:
    from transformers import AutoTokenizer
except ImportError:
    print("ERROR: transformers is not installed.")
    sys.exit(1)

try:
    import chess
except ImportError:
    print("ERROR: python-chess is not installed.")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.formatting_distill import (
    DISTILLATION_PROMPT_TEMPLATE,
    DISTILLATION_RESPONSE_TEMPLATE,
)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_all_uci_moves() -> List[str]:
    moves = []
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
                for promo in ["q", "r", "b", "n"]:
                    moves.append(f"{uci_move}{promo}")
    return moves


def add_distillation_tokens(tokenizer, add_move_tokens: bool, all_uci_moves: List[str]) -> Dict[str, int]:
    added = {"special_tokens": 0, "move_tokens": 0}
    added["special_tokens"] = tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<think>", "</think>", "<uci_move>", "</uci_move>"]}
    )
    if add_move_tokens:
        added["move_tokens"] = tokenizer.add_tokens(all_uci_moves, special_tokens=False)
    return added


def count_single_token_moves(tokenizer, all_uci_moves: List[str]) -> int:
    unk_id = getattr(tokenizer, "unk_token_id", None)
    count = 0
    for move in all_uci_moves:
        ids = tokenizer.encode(move, add_special_tokens=False)
        if len(ids) == 1 and (unk_id is None or ids[0] != unk_id):
            count += 1
    return count


def print_tokenizer_info(base_tok, tok, all_uci_moves: List[str]) -> None:
    base_single = count_single_token_moves(base_tok, all_uci_moves)
    new_single = count_single_token_moves(tok, all_uci_moves)
    print(f"UCI move single-token coverage (base): {base_single}/{len(all_uci_moves)}")
    print(f"UCI move single-token coverage (aug):  {new_single}/{len(all_uci_moves)}")

    sample = "<uci_move>e2e4</uci_move>"
    ids = tok.encode(sample, add_special_tokens=False)
    tokens = tok.convert_ids_to_tokens(ids)
    print(f"Tokenization sample: {sample}")
    print(f"  ids: {ids}")
    print(f"  tokens: {tokens}")

    for move in ["e2e4", "g1f3", "e7e8q", "a2a1n"]:
        move_ids = tok.encode(move, add_special_tokens=False)
        move_tokens = tok.convert_ids_to_tokens(move_ids)
        print(f"Move tokenization: {move}")
        print(f"  ids: {move_ids}")
        print(f"  tokens: {move_tokens}")


def summarize_moves(legal_moves_uci: str, max_show: int = 10) -> str:
    moves = legal_moves_uci.split()
    if len(moves) <= max_show:
        return legal_moves_uci
    return " ".join(moves[:max_show]) + f" ... ({len(moves)} total)"


def analyze_sample(tokenizer, pos: Dict[str, Any]) -> None:
    user_content = DISTILLATION_PROMPT_TEMPLATE.format(
        fen=pos["fen"],
        legal_moves=pos["legal_moves_uci"],
        board=pos.get("board_utf", ""),
    )
    assistant_content = DISTILLATION_RESPONSE_TEMPLATE.format(
        thinking=f"Selecting move {pos['target_move_uci']}.",
        move=pos["target_move_uci"],
    )
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        [messages[0]],
        tokenize=False,
        add_generation_prompt=True,
    )

    full_ids = tokenizer(full_text, truncation=True, return_tensors=None)["input_ids"]
    prompt_ids = tokenizer(prompt_text, truncation=True, return_tensors=None)["input_ids"]

    uci_tag_id = tokenizer.convert_tokens_to_ids("<uci_move>")
    move_pos = -1
    if uci_tag_id is not None and uci_tag_id != tokenizer.unk_token_id:
        for idx in range(len(full_ids) - 1, len(prompt_ids) - 1, -1):
            if full_ids[idx] == uci_tag_id:
                move_pos = idx
                break

    move_ids = tokenizer.encode(pos["target_move_uci"], add_special_tokens=False)
    print("Sample:")
    print(f"  source: {pos.get('source', 'unknown')}")
    print(f"  fen: {pos['fen']}")
    print(f"  target_move_uci: {pos['target_move_uci']}")
    print(f"  legal_moves_uci: {summarize_moves(pos['legal_moves_uci'])}")
    print(f"  prompt_len: {len(prompt_ids)}")
    print(f"  total_len: {len(full_ids)}")
    print(f"  uci_tag_id: {uci_tag_id}")
    print(f"  uci_tag_pos: {move_pos}")
    print(f"  move_token_ids: {move_ids}")
    print("")


def stream_positions(
    config: Dict[str, Any],
    samples_per_source: int = 2,
    seed: int = 42,
) -> None:
    try:
        from src.data_processing import stream_game_positions, stream_puzzle_positions
    except ImportError as exc:
        raise ImportError("datasets is not installed (required for streaming).") from exc

    data_config = config.get("data", {})
    elo_weights = config.get("elo_weights", None)

    game_iter = stream_game_positions(
        dataset_name=data_config.get("games_dataset", "Lichess/standard-chess-games"),
        min_elo=data_config.get("min_elo", 1200),
        elo_weights=elo_weights,
        sample_rate=data_config.get("sample_rate", 0.3),
        skip_first_moves=data_config.get("skip_first_moves", 4),
        skip_last_moves=data_config.get("skip_last_moves", 2),
        config=config,
        seed=seed,
    )
    puzzle_iter = stream_puzzle_positions(
        dataset_name=data_config.get("puzzles_dataset", "Lichess/chess-puzzles"),
        min_rating=data_config.get("min_puzzle_rating", 1000),
        max_rating=data_config.get("max_puzzle_rating", 2500),
        config=config,
        seed=seed,
    )

    samples = []
    for _ in range(samples_per_source):
        samples.append(next(game_iter))
    for _ in range(samples_per_source):
        samples.append(next(puzzle_iter))

    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Tokenizer + stream sanity check")
    parser.add_argument("--config", type=str, default="configs/config_distill.yaml")
    parser.add_argument("--samples-per-source", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = config.get("model", {}).get("name", "Qwen/Qwen3-0.6B")
    distill_config = config.get("distillation", {})

    print(f"Loading tokenizer: {model_name}")
    base_tok = AutoTokenizer.from_pretrained(model_name)
    tok = AutoTokenizer.from_pretrained(model_name)

    all_uci_moves = generate_all_uci_moves()
    added = add_distillation_tokens(
        tok,
        add_move_tokens=distill_config.get("add_uci_move_tokens", True),
        all_uci_moves=all_uci_moves,
    )
    print(f"Added tokens: {added['special_tokens']} special, {added['move_tokens']} moves")

    print_tokenizer_info(base_tok, tok, all_uci_moves)

    try:
        samples = stream_positions(
            config,
            samples_per_source=args.samples_per_source,
            seed=args.seed,
        )
    except Exception as exc:
        print(f"ERROR: failed to stream samples: {exc}")
        return 1

    for pos in samples:
        analyze_sample(tok, pos)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
