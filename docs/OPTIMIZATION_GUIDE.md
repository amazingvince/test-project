# RTX 5090 Training Optimization Guide

This guide explains all the optimizations implemented for training on a single RTX 5090 (32GB VRAM).

## Quick Start

```bash
# Install optimized dependencies
pip install -r requirements_optimized.txt

# Install Flash Attention 2 (highly recommended)
pip install flash-attn --no-build-isolation

# Run optimized training
python distill/train.py --config configs/distill/config_distill.yaml --streaming
```

## Optimization Summary

| Optimization | Memory Savings | Speed Improvement | Notes |
|-------------|---------------|-------------------|-------|
| Cut Cross Entropy | 10-28GB → <1GB | ~1.5x | Huge for large vocab |
| Liger Kernel | ~20% | ~1.2x | Fused operations |
| Flash Attention 2 | ~50% | ~2x | Must install separately |
| Gradient Checkpointing | ~60% | 0.7x | Trade speed for memory |
| BF16 Training | ~50% | ~1.1x | Best for Blackwell |
| TF32 Matmul | None | ~1.2x | Free speedup |
| Fused AdamW | ~10% | ~1.1x | Built into PyTorch |

## Detailed Optimizations

### 1. Cut Cross Entropy (CCE)

**What it does:** Computes cross-entropy loss without materializing the full logits tensor.

**Memory impact:** For Gemma 2 (2B), reduces loss memory from 24GB to 1MB.

**How it works:**
- Instead of computing `logits = hidden @ vocab_weights` (huge tensor)
- CCE computes loss directly using custom Triton kernels
- Only materializes logits for the correct token on-the-fly

**Installation:**
```bash
pip install cut-cross-entropy
```

**In your code:**
```python
from cut_cross_entropy import linear_cross_entropy
from cut_cross_entropy.transformers import cce_patch

# Patch a model
model = cce_patch(model, impl="cce", reduction="none")
```

### 2. Liger Kernel

**What it does:** Provides fused GPU kernels that combine multiple operations.

**Fused operations:**
- RMSNorm
- RoPE (Rotary Position Embedding)
- SwiGLU activation
- Cross Entropy Loss
- FusedLinearCrossEntropy

**How to enable:**
```python
# In TrainingArguments
TrainingArguments(
    use_liger_kernel=True,
    ...
)
```

### 3. Flash Attention 2

**What it does:** Memory-efficient attention that doesn't materialize the full attention matrix.

**Requirements:**
- GPU with compute capability >= 8.0 (RTX 5090 = 12.0 ✓)
- PyTorch 2.0+
- CUDA 11.6+

**Installation:**
```bash
pip install flash-attn --no-build-isolation
```

**In your code:**
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",
    ...
)
```

### 4. TF32 Precision

**What it does:** Uses TensorFloat-32 format for matrix multiplications - same range as FP32 but with FP16 precision.

**Speed improvement:** ~1.2x faster matmuls with minimal accuracy loss.

**How to enable:**
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
```

### 5. Gradient Checkpointing

**What it does:** Trades compute for memory by recomputing activations during backward pass.

**Memory savings:** ~60% reduction in activation memory.

**How to enable:**
```python
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
```

**Note:** Use `use_reentrant=False` for better performance with torch.compile.

### 6. Fused AdamW Optimizer

**What it does:** Combines multiple optimizer operations into single CUDA kernels.

**How to enable:**
```python
TrainingArguments(
    optim="adamw_torch_fused",
    ...
)
```

### 7. DataLoader Optimizations

**Key settings:**
```python
TrainingArguments(
    dataloader_num_workers=8,           # Parallel data loading
    dataloader_pin_memory=True,         # Faster CPU→GPU transfer
    dataloader_persistent_workers=True, # Keep workers alive
    dataloader_prefetch_factor=4,       # Prefetch batches
    ...
)
```

### 8. Memory Allocation Optimization

**Environment variable:**
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This reduces memory fragmentation by using expandable memory segments.

## Batch Size Guide for RTX 5090

For Qwen3-0.6B with sequence length 1024:

| Config | Batch Size | Grad Accum | Effective Batch | VRAM Usage |
|--------|-----------|------------|-----------------|------------|
| Conservative | 8 | 8 | 64 | ~18GB |
| Balanced | 16 | 4 | 64 | ~24GB |
| Aggressive | 24 | 4 | 96 | ~28GB |
| Maximum | 32 | 2 | 64 | ~30GB |

**Recommendations:**
- Start with balanced config (16 batch, 4 grad accum)
- Monitor VRAM with `nvidia-smi` or the built-in memory logging
- If OOM, reduce batch size or increase gradient accumulation

## HuggingFace Datasets Best Practices

### 1. Use Streaming for Large Datasets

```python
dataset = load_dataset("Lichess/standard-chess-games", streaming=True)
```

**Benefits:**
- No memory limit on dataset size
- Start training immediately
- Efficient for single-pass training

### 2. Preprocess and Cache

```python
# Process once, save to disk
dataset = dataset.map(preprocess_fn, num_proc=os.cpu_count())
dataset.save_to_disk("./preprocessed_data")

# Load fast in subsequent runs
dataset = load_from_disk("./preprocessed_data")
```

### 3. Use `.with_format("torch")`

```python
# Avoids repeated conversions
dataset = dataset.with_format("torch")
```

### 4. Efficient Shuffling

```python
# For streaming datasets
dataset = dataset.shuffle(seed=42, buffer_size=10000)

# For in-memory datasets
dataset = dataset.shuffle(seed=42)
```

### 5. Interleave Multiple Datasets

```python
from datasets import interleave_datasets

combined = interleave_datasets(
    [games_ds, puzzles_ds],
    probabilities=[0.7, 0.3],
    seed=42,
    stopping_strategy="all_exhausted"
)
```

## Accelerate Configuration

For single GPU, you typically don't need `accelerate config`, but you can use it for consistency:

```yaml
# ~/.cache/huggingface/accelerate/default_config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
mixed_precision: bf16
dynamo_backend: 'NO'
machine_rank: 0
num_machines: 1
num_processes: 1
```

## Monitoring Training

### GPU Memory
```python
def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
```

### Training Speed
The optimized trainer includes `include_tokens_per_second=True` in TrainingArguments.

## Troubleshooting

### Out of Memory (OOM)

1. **Reduce batch size** - Most common fix
2. **Enable gradient checkpointing** - Already enabled in optimized config
3. **Reduce sequence length** - `max_seq_length: 512` in config
4. **Use CCE** - Install cut-cross-entropy

### Slow Training

1. **Check GPU utilization** - Should be >90%
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Profile data loading**
   ```python
   # Check if dataloader is the bottleneck
   TrainingArguments(
       dataloader_num_workers=0,  # Try this
       ...
   )
   ```

3. **Enable torch.compile** - For long training runs
   ```yaml
   training:
     torch_compile: true
   ```

### CUDA Compatibility

RTX 5090 requires CUDA 12.8+. Check your installation:
```bash
python -c "import torch; print(torch.version.cuda)"
```

If you see errors about sm_120, upgrade PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Advanced: Custom Loss with CCE

For weighted loss with CCE:

```python
from cut_cross_entropy import linear_cross_entropy

def compute_weighted_cce_loss(hidden_states, classifier_weights, labels, weights):
    # hidden_states: (B, T, hidden_dim)
    # classifier_weights: (vocab_size, hidden_dim)
    # labels: (B, T)
    # weights: (B,)
    
    loss = linear_cross_entropy(
        hidden_states,
        classifier_weights,
        labels,
        reduction="none"  # Get per-token loss
    )
    
    # Aggregate per-sample, then weight
    mask = labels != -100
    per_sample_loss = (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    weighted_loss = (per_sample_loss * weights).mean()
    
    return weighted_loss
```

## Performance Expectations

On RTX 5090 with Qwen3-0.6B:

| Metric | Without Optimization | With Optimization |
|--------|---------------------|-------------------|
| VRAM Usage | ~28GB | ~16GB |
| Throughput | ~2000 tokens/s | ~4000 tokens/s |
| Time per epoch (5M samples) | ~14 hours | ~7 hours |

*Numbers are approximate and depend on specific configuration.*

## References

- [Cut Cross Entropy Paper](https://arxiv.org/abs/2411.09009)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [HuggingFace Efficient Training Guide](https://huggingface.co/docs/transformers/perf_train_gpu_one)
