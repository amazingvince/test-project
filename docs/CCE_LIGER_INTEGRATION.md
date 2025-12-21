# CCE + Liger Kernel Integration

## Overview

This project combines **Cut Cross Entropy (CCE)** and **Liger Kernels** for optimal training:

| Component | Tool | Why |
|-----------|------|-----|
| RoPE, RMSNorm, SwiGLU | Liger Kernels | 3X speedup, 3X memory reduction |
| Cross Entropy | CCE | 99.9% memory reduction, allows weighted loss |

## Why Not Use Liger's Cross Entropy?

From the [CCE paper](https://arxiv.org/abs/2411.09009):

> "Liger Kernels compute the **loss+gradient simultaneously**. Any transform applied to the loss **must be implemented in the kernel itself**. CCE has **separate forward and backward stages**, enabling user-defined transformations on the loss."

**Liger CE** = Fused computation → Cannot apply custom weights  
**CCE** = Separate forward/backward → Can apply sample weights

## Implementation

### 1. Apply Liger Kernels (Without CE)

```python
from liger_kernel.transformers import apply_liger_kernel_to_qwen3

# BEFORE loading model
apply_liger_kernel_to_qwen3(
    rope=True,
    rms_norm=True,
    swiglu=True,
    cross_entropy=False,           # DISABLED - use CCE instead
    fused_linear_cross_entropy=False,
)

model = AutoModelForCausalLM.from_pretrained(...)
```

### 2. Use CCE with shift=1

CCE's `shift=1` parameter handles causal LM shifting without allocating a new tensor:

```python
from cut_cross_entropy import linear_cross_entropy

# Memory-efficient: no shifted tensor allocation
per_token_loss = linear_cross_entropy(
    hidden_states,  # (batch, seq, hidden) - NOT pre-shifted
    classifier,
    labels,         # (batch, seq) - NOT pre-shifted
    shift=1,        # CCE handles shifting internally
    ignore_index=-100,
    reduction="none",
)
```

### 3. Prompt Masking

The data collator sets `labels=-100` for all user/prompt tokens:

```python
# Full conversation tokenized
labels = input_ids.copy()

# Mask prompt tokens
prompt_length = len(tokenize(user_message + assistant_header))
for i in range(prompt_length):
    labels[i] = -100  # Ignored by CCE
```

### 4. Sample-Level Weighting

```python
# Per-token losses (prompt tokens return 0)
per_token_loss = linear_cross_entropy(..., reduction="none")

# Reshape: (batch * (seq-1),) -> (batch, seq-1)
per_token_loss = per_token_loss.view(batch_size, seq_len - 1)

# Per-sample mean
per_sample_loss = per_token_loss.sum(dim=1) / valid_tokens_per_sample

# Weighted average
loss = (per_sample_loss * weights).sum() / weights.sum()
```

## TrainingArguments

```python
TrainingArguments(
    ...
    use_liger_kernel=False,  # We apply manually without CE
)
```

## Memory Savings

| Component | Standard | With CCE+Liger | Savings |
|-----------|----------|----------------|---------|
| Cross Entropy (128K vocab) | 24 GB | ~1 MB | 99.9% |
| RoPE | 3X | 1X | 66% |
| RMSNorm | 3X | 1X | 66% |

## References

1. [Cut Cross Entropy Paper](https://arxiv.org/abs/2411.09009)
2. [Liger Kernel Paper](https://arxiv.org/abs/2410.10989)
3. [Apple ml-cross-entropy](https://github.com/apple/ml-cross-entropy)
4. [LinkedIn Liger-Kernel](https://github.com/linkedin/Liger-Kernel)
