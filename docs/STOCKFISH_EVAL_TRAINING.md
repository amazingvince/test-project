# Chess LLM Training with Stockfish-Based Reward Shaping

This document explains the enhanced training approach that uses Stockfish evaluations to:
1. Include move analysis in `<think>` tags (distillation)
2. Shape rewards based on centipawn values
3. Weight training examples by move quality

## Overview

The key insight is that instead of just training on "this is the move that was played," we can provide the model with:
- **All legal moves and their evaluations** (in the thinking section)
- **Quality-based loss weighting** (better moves get more training signal)

This is a form of **knowledge distillation** from Stockfish into the language model.

## How It Works

### 1. Data Generation with Stockfish Analysis

For each position, we run Stockfish to evaluate ALL legal moves:

```
Position: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1

Stockfish analysis:
  1. e5:  -20cp (best)
  2. c5:  -25cp
  3. d5:  -30cp
  4. Nf6: -35cp
  5. e6:  -40cp
  ...
```

### 2. Training Format with Evaluations

The response includes move analysis in `<think>` tags:

```
<think>
Analyzing position with 20 legal moves...

Top candidates:
→ 1. e5: -20cp
  2. c5: -25cp (loses 5cp)
  3. d5: -30cp (loses 10cp)
  4. Nf6: -35cp (loses 15cp)
  5. e6: -40cp (loses 20cp)

Best continuation: e5 (-20cp)
</think>
<uci_move>e7e5</uci_move>
```

### 3. Reward Shaping via Loss Weighting

The training loss is weighted based on move quality:

| Move Quality | CP Loss | Loss Weight |
|-------------|---------|-------------|
| Best move   | 0       | 2.0 (max)   |
| Top-3       | ≤20     | ~1.6        |
| Good        | ≤50     | ~1.3        |
| Acceptable  | ≤100    | ~1.0        |
| Inaccuracy  | ≤200    | ~0.5        |
| Blunder     | >200    | 0.1 (min)   |

This means:
- **Best moves contribute 20x more** to the gradient than blunders
- The model learns more from good examples
- Bad moves are still learned but with less emphasis

## Files Structure

```
src/
├── src/utils/stockfish_eval.py # Stockfish evaluation utilities
├── data_processing_with_eval.py # Data generation with evals
├── formatting_with_eval.py     # Format with evals in <think>
└── ...

configs/
├── configs/sft/config_with_eval.yaml       # Configuration with eval settings

sft/preprocess.py               # Script to generate training data
```

## Quick Start

### 1. Install Dependencies

```bash
# Install Stockfish
sudo apt install stockfish

# Or on macOS
brew install stockfish

# Install Python packages
pip install chess python-chess datasets tqdm
```

### 2. Generate Training Data

```bash
# Small test run (1000 positions)
python sft/preprocess.py \
    --output ./data/chess_eval_test \
    --size 1000 \
    --depth 10 \
    --workers 4

# Full dataset (500K positions)
python sft/preprocess.py \
    --config configs/sft/config_with_eval.yaml \
    --size 500000
```

### 3. Train the Model

```bash
python train.py \
    --config configs/sft/config_with_eval.yaml \
    --preprocessed_path ./data/chess_eval_test
```

## Configuration Options

### Stockfish Settings

```yaml
stockfish:
  depth: 12        # Higher = more accurate, slower
  workers: 8       # Parallel analysis workers
  multipv: null    # null = all moves, or N for top-N
```

### Loss Weighting Options

```yaml
loss_weighting:
  enabled: true
  weight_type: "cp_loss"  # Options below
  min_weight: 0.1
  max_weight: 2.0
```

**Weight Types:**

1. **`cp_loss`** (recommended): Weight inversely proportional to centipawn loss
   ```
   weight = max_weight - (cp_loss / max_cp_loss) * (max_weight - min_weight)
   ```

2. **`rank`**: Weight based on move rank
   ```
   weight = max_weight - (rank - 1) / (num_moves - 1) * range
   ```

3. **`combined`**: Mix of CP loss and rank
   ```
   weight = alpha * cp_weight + (1 - alpha) * rank_weight
   ```

4. **`quality_gate`**: Binary - high if good, low otherwise
   ```
   weight = max_weight if (rank <= top_n or cp_loss <= threshold) else min_weight
   ```

5. **`soft_rank`**: Exponential decay by rank
   ```
   weight = min_weight + (decay ^ (rank - 1)) * range
   ```

6. **`uniform`**: No quality weighting (baseline)

### Formatting Options

```yaml
formatting:
  include_eval: true
  eval_style: "detailed"  # or "compact", "minimal"
  top_n_moves: 5
```

**Eval Styles:**

- **detailed**: Full analysis with commentary
- **compact**: Single line `e5(-20) c5(-25) d5(-30)`
- **minimal**: Just `Best: e5 -20cp`

## Example Output

### Training Example with Detailed Style

```
[User Message]
You are an expert chess player. Here is the position in FEN format:
r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4

Legal moves: a2a3 a2a4 b2b3 b2b4 ...

Board looks like this right now
   a  b  c  d  e  f  g  h  
   +------------------------+
8 | ♜  ·  ♝  ♛  ♚  ♝  ·  ♜ | 8
7 | ♟  ♟  ♟  ♟  ·  ♟  ♟  ♟ | 7
...

Select the best move...

[Assistant Response]
<think>
Analyzing position with 33 legal moves...

Top candidates:
→ 1. Qxf7#: M1 (checkmate!)
  2. Qf3: +150cp
  3. d3: +45cp
  4. Nc3: +40cp
  5. Nf3: +35cp

Best continuation: Qxf7# (M1)
</think>
<uci_move>h5f7</uci_move>
```

## Benefits of This Approach

1. **Distillation**: The model learns HOW to evaluate moves, not just WHICH move to play
2. **Reward Shaping**: Better moves get amplified in training
3. **Reasoning**: The `<think>` section teaches move comparison
4. **Robustness**: Model sees that multiple moves can be good (not just one "correct" answer)

## Performance Considerations

- **Stockfish Analysis**: ~50-200ms per position at depth 12
- **Batch Processing**: Use multiple workers (8-16) for parallel analysis
- **Dataset Size**: 500K positions takes ~2-4 hours to preprocess
- **Storage**: ~1GB per 500K positions with evaluations

## Advanced: Custom Reward Functions

You can define custom reward functions in `src/utils/stockfish_eval.py`:

```python
def compute_move_reward(
    analysis: PositionAnalysis,
    played_move_uci: str,
    reward_type: str = "centipawn_loss",
    **kwargs
) -> float:
    """Custom reward computation."""
    # Implement your own reward logic here
    ...
```

## Comparison: With vs Without Evaluations

| Metric | Without Eval | With Eval |
|--------|--------------|-----------|
| Learns reasoning | ❌ | ✓ |
| Quality-weighted | ❌ | ✓ |
| Training signal | Equal for all | Better moves emphasized |
| Model output | Just move | Analysis + move |

## Troubleshooting

### Stockfish Not Found
```bash
# Check if installed
which stockfish

# Install on Ubuntu/Debian
sudo apt install stockfish

# Specify path manually
python sft/preprocess.py --stockfish-path /path/to/stockfish
```

### Slow Processing
- Reduce `depth` (10 instead of 12)
- Increase `workers` (up to CPU cores)
- Use `multipv: 10` instead of all moves

### Out of Memory
- Process in smaller batches
- Use streaming mode for large datasets
