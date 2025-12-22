"""
Merge two Qwen-style checkpoints with task arithmetic.

This script implements a simple, parameter-wise merge:

    merged = chess + scale * (reasoning - base)

where:
- `base` is the pretrained base model,
- `reasoning` is an instruction / reasoning-tuned variant of the same base,
- `chess` is your chess-tuned checkpoint.

The merge is heuristic; always evaluate the resulting checkpoint on your chess
and generation-style benchmarks.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for task-arithmetic merging."""
    parser = argparse.ArgumentParser(
        description="Merge checkpoints with task arithmetic: chess + scale * (reasoning - base)"
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-0.6B-Base",
        help="Base pretrained model (same architecture as the others).",
    )
    parser.add_argument(
        "--reasoning-model",
        default="Qwen/Qwen3-0.6B",
        help="Instruction/reasoning-tuned model built from --base-model.",
    )
    parser.add_argument(
        "--chess-model",
        required=True,
        help="Your chess-tuned checkpoint (local path or HF repo id).",
    )
    parser.add_argument(
        "--output",
        default="./merged-model",
        help="Directory to write merged weights + tokenizer.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="How much of (reasoning - base) to add to chess.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help="dtype used while loading and merging on CPU.",
    )
    parser.add_argument(
        "--tokenizer-from",
        choices=("chess", "reasoning", "base"),
        default="chess",
        help="Which checkpoint to load the tokenizer/chat template from.",
    )
    return parser.parse_args()


def _torch_dtype(dtype: str) -> torch.dtype:
    """Convert a string dtype flag to a torch dtype."""
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]


def load_state_dict(model_path: str, *, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    """
    Load a checkpoint's state dict on CPU.

    The returned tensors keep the underlying storage alive after the model is
    deleted, so no explicit `.clone()` is required.
    """
    print(f"Loading weights: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    )
    state_dict = {k: v.detach() for k, v in model.state_dict().items()}
    del model
    gc.collect()
    return state_dict


def task_arithmetic_merge(
    base: Dict[str, torch.Tensor],
    chess: Dict[str, torch.Tensor],
    reasoning: Dict[str, torch.Tensor],
    scale: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """
    Merge using task arithmetic.

    Formula:
        merged = chess + scale * (reasoning - base)

    Keys that are missing in `base`/`reasoning` or have mismatched shapes are
    copied from `chess` unchanged.
    """
    merged: Dict[str, torch.Tensor] = {}
    stats = {"merged": 0, "copied": 0, "missing": 0, "shape_mismatch": 0}

    with torch.no_grad():
        for key, chess_tensor in chess.items():
            base_tensor = base.get(key)
            reasoning_tensor = reasoning.get(key)
            if base_tensor is None or reasoning_tensor is None:
                merged[key] = chess_tensor
                stats["missing"] += 1
                continue

            if chess_tensor.shape != base_tensor.shape or chess_tensor.shape != reasoning_tensor.shape:
                merged[key] = chess_tensor
                stats["shape_mismatch"] += 1
                continue

            merged[key] = chess_tensor + scale * (reasoning_tensor - base_tensor)
            stats["merged"] += 1

    stats["copied"] = len(chess) - stats["merged"]
    return merged, stats


def main():
    args = parse_args()
    dtype = _torch_dtype(args.dtype)

    print("Task arithmetic merge")
    print(f"  base:      {args.base_model}")
    print(f"  reasoning: {args.reasoning_model}")
    print(f"  chess:     {args.chess_model}")
    print(f"  scale:     {args.scale}")
    print(f"  output:    {args.output}")
    print(f"  dtype:     {args.dtype}")

    base_weights = load_state_dict(args.base_model, dtype=dtype)
    reasoning_weights = load_state_dict(args.reasoning_model, dtype=dtype)
    chess_weights = load_state_dict(args.chess_model, dtype=dtype)

    merged_weights, stats = task_arithmetic_merge(
        base=base_weights,
        chess=chess_weights,
        reasoning=reasoning_weights,
        scale=args.scale,
    )
    print(f"Merged parameters: {stats['merged']}/{len(chess_weights)}")
    if stats["missing"]:
        print(f"Copied (missing key): {stats['missing']}")
    if stats["shape_mismatch"]:
        print(f"Copied (shape mismatch): {stats['shape_mismatch']}")

    del base_weights, reasoning_weights, chess_weights
    gc.collect()

    print("Loading output model skeleton...")
    model = AutoModelForCausalLM.from_pretrained(
        args.chess_model,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    )
    missing, unexpected = model.load_state_dict(merged_weights, strict=False)
    if missing or unexpected:
        print(f"load_state_dict(strict=False): missing={len(missing)} unexpected={len(unexpected)}")

    tokenizer_source = {
        "chess": args.chess_model,
        "reasoning": args.reasoning_model,
        "base": args.base_model,
    }[args.tokenizer_from]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Saved merged model to: {output_path}")


if __name__ == "__main__":
    main()
