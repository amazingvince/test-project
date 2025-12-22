# RTX 5090 Training Optimization Guide

This repo is set up to train on a single 32GB GPU (for example an RTX 5090).
Most optimizations are optional; the training entrypoints fall back to standard
PyTorch/Transformers implementations if an optional dependency is missing.

**Entry points**
- SFT: `sft/train.py`
- Distillation: `distill/train.py`

## Quick Start (Linux)

```bash
pip install -r requirements.txt

# Optional speedups (install what you want)
pip install flash-attn --no-build-isolation
pip install liger-kernel
pip install "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git"
```

Run distillation (preprocessed, recommended):

```bash
python distill/preprocess.py --config configs/distill/config_distill.yaml --output ./data/chess_distill
python distill/train.py --config configs/distill/config_distill.yaml --preprocessed_path ./data/chess_distill
```

Or run distillation in streaming mode (Stockfish called inside the collator):

```bash
python distill/train.py --config configs/distill/config_distill.yaml --streaming
```

## High-Impact Optimizations

### Flash Attention 2

- Config: `model.attn_implementation: flash_attention_2`
- If `flash-attn` is not available, `sft/train.py`, `distill/train.py`, and `eval/evaluate_fast.py` fall back to eager attention automatically.

### Cut Cross Entropy (CCE)

CCE avoids materializing the full `(batch, seq, vocab)` logits tensor for CE.

- SFT: controlled by `cce.enabled` in the SFT config and `--no-cce` in the CLI.
- Distillation: controlled by `distillation.use_cce` and `--no-cce`.

### Liger Kernels

Liger provides fused kernels for several transformer components (e.g., RoPE / RMSNorm / SwiGLU).

- The training scripts apply Liger kernels before model load.
- Disable with `--no-liger` if you hit kernel/build issues.

### Padding-Free Packing (`--unpad`)

Distillation supports a padding-free (“packed”) batching mode for FlashAttention2:

- `training.use_unpad: true` (or `--unpad`) flattens a batch into a single sequence (like HF `DataCollatorWithFlattening` / TRL `padding_free`).
- Boundaries are enforced by resetting `position_ids` per sample (prevents cross-sample attention) and inserting a `-100` label separator (prevents cross-sample CE loss).
- Requires `model.attn_implementation: flash_attention_2`; if FlashAttention2 is unavailable, `distill/train.py` disables packing automatically.

## DataLoader and Stockfish Notes

- Distillation streaming mode runs Stockfish in the collator, so `dataloader_num_workers=0` is intentional.
- Preprocessed training can use multiple workers (bounded by CPU cores and dataset storage throughput).

## Troubleshooting

- **OOM**: reduce `per_device_train_batch_size`, increase `gradient_accumulation_steps`, or reduce `model.max_seq_length`.
- **flash-attn install issues**: set `model.attn_implementation: eager` and continue; training will still work.
