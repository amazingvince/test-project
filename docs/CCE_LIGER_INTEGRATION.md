# CCE + Liger Kernel Integration

This repo optionally uses:
- **Liger kernels** for fused transformer components (RoPE / RMSNorm / SwiGLU, depending on model support).
- **Cut Cross Entropy (CCE)** for cross-entropy loss without materializing full logits.

The training entrypoints are written so that these are additive optimizations:
if either dependency is missing, training still runs with standard PyTorch/Transformers codepaths.

## Where It’s Used

- `sft/train.py`
  - Applies Liger kernels before model load (disabled via `--no-liger`).
  - Uses CCE in `WeightedCCETrainer` when installed.
- `distill/train.py`
  - Uses CCE for the CE component (reasoning + move text).
  - Uses `src/distill/distillation_loss.py` for KL/JSD distillation on the move token.

## Installation

```bash
pip install liger-kernel
pip install "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git"
```

## Prompt Masking and Shifting

- Collators set `labels=-100` for prompt and padding tokens, so CE trains only on assistant tokens.
- For causal LMs, the logits at position `t` predict the token at `t+1`; CCE is called with `shift=1` to match that convention.

