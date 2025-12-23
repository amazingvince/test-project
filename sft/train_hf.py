#!/usr/bin/env python3
"""
Minimal SFT training entrypoint using the standard Hugging Face Trainer loss.

This script is intentionally "boring":
  - No weighted loss.
  - No custom Trainer.compute_loss.
  - Uses the model's native causal-LM cross entropy by passing `labels`.

It is designed for the "Stockfish reasoning trace" SFT workflow:
  1) Run `sft/preprocess.py` with `reasoning_trace.enabled: true` and
     `data.target_move: best` (or `reasoning_trace.always_choose_best_move: true`)
     so each example contains a `messages` field whose assistant message ends with
     `<uci_move>...</uci_move>`.
  2) Train on those `messages` with prompt masking (loss only on assistant tokens).

If you want SFT sample-weighting or Cut Cross Entropy, use `sft/train.py` instead.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from src.utils.chess_eval_callback import FastChessEvalCallback as ChessEvalCallback, prepare_eval_positions
from src.utils.chess_tokenizer import ChessTokenizerMode, add_chess_tokens, generate_all_uci_moves


def setup_torch_optimizations() -> None:
    """Enable common performance knobs for Ampere+ GPUs."""

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


setup_torch_optimizations()


LIGER_AVAILABLE = False
try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3

    LIGER_AVAILABLE = True
except ImportError:
    pass


def apply_liger_kernels_full(model_name: str) -> None:
    """
    Apply Liger kernels including cross entropy, when available.

    Liger patches must be applied *before* loading the model.
    """

    if not LIGER_AVAILABLE:
        return

    name = model_name.lower()
    if "qwen3" in name:
        apply_liger_kernel_to_qwen3(
            rope=True,
            rms_norm=True,
            swiglu=True,
            cross_entropy=True,
            fused_linear_cross_entropy=True,
        )
        return

    # Other model families are optional. We keep this script minimal and only
    # guarantee Qwen3 support (the repo's default).


@dataclass
class PromptMaskingCollator:
    """
    Collator that tokenizes chat messages and masks the prompt.

    The dataset is expected to provide a `messages` field compatible with
    `tokenizer.apply_chat_template`. Labels are `-100` for prompt tokens so the
    model loss is computed only on assistant tokens.

    Truncation behavior:
      - If the example exceeds `max_length`, the collator will drop tokens from
        the *prompt* side only. If the assistant portion would be truncated,
        it raises a ValueError (the fix is to reduce trace length or increase
        `model.max_seq_length`).
    """

    tokenizer: Any
    max_length: int
    pad_to_multiple_of: int = 8

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_input_ids: List[List[int]] = []
        batch_attention_mask: List[List[int]] = []
        batch_labels: List[List[int]] = []

        for example in examples:
            messages = example.get("messages")
            if not messages:
                raise ValueError(
                    "Dataset example is missing `messages`. "
                    "Run `sft/preprocess.py` with `reasoning_trace.enabled: true` "
                    "to generate distill-style messages."
                )

            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_text = self.tokenizer.apply_chat_template(
                [messages[0]],
                tokenize=False,
                add_generation_prompt=True,
            )

            full_tokens = self.tokenizer(full_text, truncation=False, return_tensors=None)
            prompt_tokens = self.tokenizer(prompt_text, truncation=False, return_tensors=None)

            input_ids = full_tokens["input_ids"]
            attention_mask = full_tokens.get("attention_mask") or [1] * len(input_ids)
            labels = input_ids.copy()
            prompt_length = len(prompt_tokens["input_ids"])

            if len(input_ids) > self.max_length:
                overflow = len(input_ids) - self.max_length
                if overflow > prompt_length:
                    raise ValueError(
                        "Example exceeds max_length and would truncate assistant tokens. "
                        f"max_length={self.max_length} full_length={len(input_ids)} "
                        f"prompt_length={prompt_length}. "
                        "Reduce reasoning_trace.max_trace_tokens or increase model.max_seq_length."
                    )

                input_ids = input_ids[overflow:]
                attention_mask = attention_mask[overflow:]
                labels = input_ids.copy()
                prompt_length -= overflow

            for i in range(min(prompt_length, len(labels))):
                labels[i] = -100

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        max_len = max(len(ids) for ids in batch_input_ids)
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        pad_id = int(self.tokenizer.pad_token_id)
        padded_input_ids: List[List[int]] = []
        padded_attention: List[List[int]] = []
        padded_labels: List[List[int]] = []
        for ids, attn, lab in zip(batch_input_ids, batch_attention_mask, batch_labels):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [pad_id] * pad_len)
            padded_attention.append(attn + [0] * pad_len)
            padded_labels.append(lab + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_model_and_tokenizer(config: Dict[str, Any]) -> Tuple[Any, Any]:
    model_cfg = config.get("model", {})
    tok_cfg = config.get("tokenizer", {})
    train_cfg = config.get("training", {})

    model_name = model_cfg.get("name")
    if not model_name:
        raise ValueError("config.model.name is required")

    if LIGER_AVAILABLE and train_cfg.get("use_liger", True):
        apply_liger_kernels_full(model_name)
        print("Liger kernels: enabled (including cross entropy)")
    else:
        print("Liger kernels: disabled or unavailable")

    dtype_str = str(model_cfg.get("dtype", "bfloat16")).lower()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype_str, torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    attn_impl = model_cfg.get("attn_implementation", "flash_attention_2")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation=attn_impl,
    )

    if bool(train_cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    mode_str = tok_cfg.get("chess_mode", "tags_only")
    if mode_str not in ("tags_only", "tags_and_moves"):
        raise ValueError(f"tokenizer.chess_mode must be 'tags_only' or 'tags_and_moves' (got {mode_str!r})")
    mode: ChessTokenizerMode = mode_str

    add_think_tags = bool(tok_cfg.get("add_think_tags", False))
    all_uci_moves = generate_all_uci_moves() if mode == "tags_and_moves" else None
    added = add_chess_tokens(
        tokenizer=tokenizer,
        model=model,
        mode=mode,
        add_think_tags=add_think_tags,
        all_uci_moves=all_uci_moves,
    )
    if added["special_tokens"] or added["move_tokens"]:
        print(
            f"Added tokens: {added['special_tokens']} special, "
            f"{added['move_tokens']} UCI move tokens (mode={mode})"
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is not None:
        model.config.bos_token_id = tokenizer.bos_token_id
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    if bool(train_cfg.get("torch_compile", False)):
        model = torch.compile(model, mode="reduce-overhead")

    return model, tokenizer


def split_train_eval(dataset: Dataset, *, eval_size: int, seed: int) -> Tuple[Dataset, Optional[Dataset]]:
    if eval_size <= 0:
        return dataset, None
    if len(dataset) <= eval_size:
        return dataset, dataset
    shuffled = dataset.shuffle(seed=seed)
    eval_ds = shuffled.select(range(eval_size))
    train_ds = shuffled.select(range(eval_size, len(shuffled)))
    return train_ds, eval_ds


def main() -> int:
    parser = argparse.ArgumentParser(description="SFT training (vanilla HF loss)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--preprocessed_path", type=str, required=True, help="Dataset from `sft/preprocess.py`.")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint directory to resume from.")
    args = parser.parse_args()

    config = load_config(args.config)
    model, tokenizer = setup_model_and_tokenizer(config)

    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    dataset_path = Path(args.preprocessed_path)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}. Run `sft/preprocess.py` first.")

    dataset = load_from_disk(str(dataset_path))
    if "messages" not in dataset.column_names:
        raise SystemExit(
            "Preprocessed dataset is missing `messages`. "
            "Re-run `sft/preprocess.py` with `reasoning_trace.enabled: true`."
        )

    seed = int(train_cfg.get("seed", 42))
    eval_size = int(train_cfg.get("eval_size", 200))
    train_dataset, eval_dataset = split_train_eval(dataset, eval_size=eval_size, seed=seed)

    callbacks = []
    chess_eval_positions = int(train_cfg.get("chess_eval_positions", 0) or 0)
    if chess_eval_positions > 0:
        positions = prepare_eval_positions(
            dataset=train_dataset,
            num_positions=chess_eval_positions,
            seed=seed,
        )
        callbacks.append(
            ChessEvalCallback(
                tokenizer=tokenizer,
                positions=positions,
                batch_size=int(train_cfg.get("chess_eval_batch_size", 32)),
                steps=int(train_cfg.get("chess_eval_steps", train_cfg.get("eval_steps", 500))),
                max_new_tokens=int(train_cfg.get("chess_eval_max_new_tokens", 1024)),
                max_total_tokens=int(train_cfg.get("chess_eval_max_total_tokens", 2048)),
                do_sample=bool(train_cfg.get("chess_eval_do_sample", True)),
                temperature=float(train_cfg.get("chess_eval_temperature", 0.6)),
                top_p=float(train_cfg.get("chess_eval_top_p", 0.95)),
                top_k=int(train_cfg.get("chess_eval_top_k", 20)),
                min_p=float(train_cfg.get("chess_eval_min_p", 0.0)),
                stockfish_path=config.get("evaluation", {}).get("stockfish_path"),
                stockfish_depth=int(config.get("evaluation", {}).get("stockfish_depth", 20)),
                stockfish_workers=int(config.get("evaluation", {}).get("stockfish_workers", 24)),
                print_samples=int(train_cfg.get("chess_eval_print_samples", 0)),
                print_max_chars=int(train_cfg.get("chess_eval_print_max_chars", 5000)),
            )
        )

    if config.get("loss_weighting", {}).get("enabled"):
        print("Note: loss_weighting is configured but ignored in `sft/train_hf.py` (vanilla CE only).")

    eval_strategy = train_cfg.get("eval_strategy", "steps") if eval_dataset is not None else "no"

    training_args = TrainingArguments(
        output_dir=train_cfg.get("output_dir", "./outputs/chess-sft-hf"),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 1)),
        max_steps=int(train_cfg.get("max_steps", -1)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 8)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 8)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "cosine")),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.1)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        logging_steps=int(train_cfg.get("logging_steps", 50)),
        logging_first_step=True,
        save_strategy=str(train_cfg.get("save_strategy", "steps")),
        save_steps=int(train_cfg.get("save_steps", 500)),
        save_total_limit=int(train_cfg.get("save_total_limit", 3)),
        eval_strategy=eval_strategy,
        eval_steps=int(train_cfg.get("eval_steps", 500)) if eval_strategy == "steps" else None,
        bf16=bool(train_cfg.get("bf16", True)),
        fp16=False,
        bf16_full_eval=True,
        optim=str(train_cfg.get("optim", "adamw_torch_fused")),
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 0)),
        dataloader_pin_memory=bool(train_cfg.get("dataloader_pin_memory", True)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        report_to=str(train_cfg.get("report_to", "wandb")),
        run_name=str(train_cfg.get("run_name", "chess-sft-hf")),
        remove_unused_columns=False,
        load_best_model_at_end=eval_strategy != "no",
        metric_for_best_model="eval_loss" if eval_strategy != "no" else None,
        use_liger_kernel=False,
        torch_compile=bool(train_cfg.get("torch_compile", False)),
        push_to_hub=False,
        include_tokens_per_second=True,
        include_num_input_tokens_seen=True,
    )

    collator = PromptMaskingCollator(
        tokenizer=tokenizer,
        max_length=int(model_cfg.get("max_seq_length", 2048)),
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks or None,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

