# Chess LLM Training: SFT + Policy Distillation

Train language models to play chess using a two-stage approach:
1. **SFT (Supervised Fine-Tuning)**: Learn basic chess from human games
2. **Policy Distillation**: Refine with Stockfish's probability distribution

## Key Features

- **Two-Stage Workflow**: SFT for foundation, distillation for refinement
- **Policy Distillation**: Learn from Stockfish's probability distribution over moves
- **Forward KL Divergence**: Proper knowledge distillation that preserves move quality gradients
- **Win Probability Categories**: Human-readable move quality (winning, better, slight edge, equal, slight disadvantage, worse, losing)
- **Randomized Thinking**: Prevents positional shortcuts by shuffling move order
- **Floor Probability**: All legal moves get non-zero probability (teaches move legality!)
- **Liger Kernel Support**: Memory-efficient KL/JSD loss computation
- **Tokenizer-aware Move Tags**: Optional `<think>`/`<uci_move>` special tokens + single-token UCI moves for distillation alignment
- **Mixed-Source Eval**: Games + puzzles evaluation with per-source metrics
- **Optional Board Rendering**: Drop the board from distillation prompts to save tokens
- **Shallow All-Moves Pass**: Cheap priors for legal moves in streaming distillation
- **Stockfish LRU Cache**: Reuse analyses across repeated positions
- **Reasoning Trace Generator**: Longer, human-like traces with optional opening and tablebase context

## Training Workflow

### Stage 1: SFT (Supervised Fine-Tuning)
```
Input: Position
Target: Single move that was played
Loss: Cross-entropy on one move
```
Learn basic chess patterns from human games.

### Stage 2: Policy Distillation
```
Input: Position
Target: Probability distribution from Stockfish over ALL moves
Loss: Cross-entropy on thinking + Forward KL on move prediction

Thinking shows category + win probability for all moves:
- Nf3: slight edge (56% win)
- e4: equal (54% win)
- d4: equal (52% win)
- h4: slight disadvantage (42% win)
- a4: worse (28% win)

Best line: Nf3 d5 Nc3 Nc6 Bb5
```

Optional: replace the move list with a configurable reasoning trace (see `reasoning_trace` settings).

The model learns:
1. **Position evaluation** (win probability categories)
2. **Move quality** (higher win % = better move)
3. **That multiple moves can be good** (soft distribution)
4. **What moves are legal** (floor probability on all legal moves)
5. **Principal variation** (best line of play)

## Quick Start

### 1. Install Dependencies

```bash
# Clone and setup
git clone <repo>
cd chess-llm-sft-cleanup-1
pip install -r requirements.txt

# Install Stockfish
sudo apt install stockfish  # Linux
brew install stockfish       # macOS
```

### Optional: Tokenizer Probe

```bash
# Inspect how <uci_move> and UCI moves are tokenized, plus stream a few samples
python scripts/tokenizer_probe.py --config configs/distill/config_distill.yaml --samples-per-source 2
```

### Optional: Openings + Tablebases

```bash
# Download lichess opening TSVs to ./data/openings
python scripts/download_openings.py --output-dir ./data/openings

# Download Syzygy tablebases (3-4-5 pieces) to ./data/syzygy
# Use --dry-run first to see the file list
python scripts/download_tablebases.py --output-dir ./data/syzygy --pieces 3,4,5 --dry-run
python scripts/download_tablebases.py --output-dir ./data/syzygy --pieces 3,4,5
```

### Optional: Distillation Gut-Check Report

```bash
# Samples random preprocessed positions and writes a Markdown report with:
# - ASCII board
# - input prompt
# - reasoning trace
#
# Also checks for openings/tablebases and downloads them if missing.
python scripts/gut_check_report.py \
  --dataset ./data/chess_distill \
  --num-samples 20 \
  --output ./reports/gut_check_report.md
```

If `--dataset` does not exist, the script auto-runs `distill/preprocess.py` to create a small dataset for the report (override with `--preprocess-size` or disable with `--no-auto-preprocess`).

### 2. Stage 1: SFT Training

```bash
# Train on human games (streaming from Lichess)
python sft/train.py --config configs/sft/config_sft.yaml --streaming

# Or with preprocessed data
python sft/train.py --config configs/sft/config_sft.yaml --preprocessed_path ./data/chess_sft
```

### 3. Stage 2: Policy Distillation

```bash
# First, preprocess with Stockfish analysis
python distill/preprocess.py \
    --output ./data/chess_distill \
    --size 500000 \
    --depth 12 \
    --workers 8

# Then train with distillation (on SFT checkpoint or base model)
python distill/train.py \
    --config configs/distill/config_distill.yaml \
    --preprocessed_path ./data/chess_distill
```

Hardware presets:
- `configs/distill/config_distill.yaml` is tuned for a single RTX 5090 + 32 CPU cores.
- `configs/distill/config_distill_h100.yaml` is tuned for a single H100 + 32 CPU cores.

## Evaluation

```bash
# Mixed games + puzzles (uses data settings from config)
python eval/evaluate_fast.py \
  --model ./outputs/chess-sft-final \
  --config configs/sft/config_sft.yaml \
  --source mixed \
  --num_positions 1000 \
  --max_new_tokens 128

# Puzzles only
python eval/evaluate_fast.py \
  --model ./outputs/chess-sft-final \
  --source puzzles \
  --num_positions 500
```

Shortcut:
```bash
./scripts/evaluate.sh ./outputs/chess-sft-final --config configs/sft/config_sft.yaml --source mixed
```

Notes:
- Metrics include a per-source breakdown when using mixed sources.
- Training eval now supports `eval_total_loss`, `eval_games_loss`, and `eval_puzzles_loss` when sources are present.

## Architecture

```
Training Scripts
- sft/train.py             Stage 1: Supervised fine-tuning
- distill/train.py         Stage 2: Policy distillation

Configurations
- configs/sft/config_sft.yaml
- configs/distill/config_distill.yaml

Core Modules
- src/distill/stockfish_teacher.py
- src/distill/formatting_distill.py
- src/distill/distillation_loss.py
- src/distill/collator_distill.py

Preprocessing
- distill/preprocess.py

Scripts
- scripts/train_sft.sh
- scripts/train_distill.sh
- scripts/evaluate.sh
- scripts/tokenizer_probe.py
- scripts/gut_check_report.py
```

## Training Example Format

### User Prompt
```
You are an expert chess player. Analyze this position and select the best move.

Position (FEN): rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2

Legal moves: a2a3 a2a4 b2b3 ...

Board:
   a  b  c  d  e  f  g  h  
   +------------------------+
8 | ♜  ♞  ♝  ♛  ♚  ♝  ♞  ♜ | 8
...
```

### Assistant Response (with distillation)
```
<think>
Analyzing position...
- e6: equal (46% win)
- e5: equal (45% win)
- c5: equal (45% win)
- d6: slight disadvantage (44% win)
- c6: slight disadvantage (38% win)

Best line: e6 d4 d5 Nc3 Nf6
Playing e6 gives 46% win chance.
</think>
<uci_move>e7e5</uci_move>
```

**Key points:**
- Move order is **randomized** to prevent the model from learning "first = best"
- All moves show category + win probability (intuitive AlphaZero-style evaluation)
- **"Playing X"** always shows Stockfish's best move and its win probability
- **`<uci_move>`** can be forced to Stockfish's best move via `reasoning_trace.always_choose_best_move`
- The model learns the thinking text (CE loss) AND the probability distribution over all moves (KL loss)

## Key Design Decisions

### Why Forward KL (not Reverse)?

| Property | Forward KL | Reverse KL |
|----------|-----------|------------|
| Behavior | Mean-seeking | Mode-seeking |
| Learns | All reasonable moves | Only best move |
| Chess fit | ✓ Multiple moves often good | ✗ Ignores alternatives |

Forward KL teaches that e4, d4, Nf3, c4 can all be good opening moves.
Reverse KL would collapse to just one.

### Why Win Probability Categories?

Win probability (WDL) is more intuitive than centipawns:
- Based on AlphaZero's evaluation approach
- "56% win" is clearer than "+15cp"
- Directly represents expected game outcome

Categories based on win probability:
| Category | Win % | Description |
|----------|-------|-------------|
| winning | 80%+ | Clear advantage |
| better | 65-80% | Significant edge |
| slight edge | 55-65% | Small advantage |
| equal | 45-55% | Balanced position |
| slight disadvantage | 35-45% | Small deficit |
| worse | 20-35% | Significant deficit |
| losing | <20% | Critical disadvantage |

Every move displays its category and win probability.

### Why Randomize Move Order?

Without randomization:
```
Training data always shows: [best, 2nd, 3rd, 4th, 5th]
Model learns: "pick the first one"
```

With randomization:
```
Training data shows: [3rd, best, 5th, 2nd, 4th]
Model must: actually parse and compare
```

### Why Floor Probability for All Moves?

```python
# Without floor (bad)
probs = {"e4": 0.4, "d4": 0.35, "Nf3": 0.25}  # Only 3 moves!
# Model never sees that a3, h3, etc. are legal

# With floor (good)  
probs = {"e4": 0.35, "d4": 0.30, "Nf3": 0.22, "a3": 0.001, "h3": 0.001, ...}
# Model learns ALL moves are legal, just varying quality
```

This is critical for the model to learn move legality, not just "what's good".

## Configuration Reference

### Stockfish Settings

```yaml
stockfish:
  depth: 12             # Higher = more accurate (10-14 recommended)
  time_limit_ms: null   # Optional time cap (ms), overrides depth
  nodes: null           # Optional node cap, overrides depth
  top_k: 5              # Moves for deep analysis (5-10 recommended)
  shallow_depth: 4      # Cheap all-moves pass (0 disables)
  shallow_max_moves: 64 # Cap shallow multipv
  confirm_depth: 14     # Optional extra-deep confirm pass (0 disables)
  confirm_top_k: 2      # Moves to confirm at confirm_depth
  num_workers: 8        # Parallel Stockfish instances
  threads_per_worker: 1 # Engine threads per worker
  prob_mode: "cp"       # "cp" or "wdl"
  wdl_temperature: 1.0  # Win-prob softmax temperature
  cache_size: 4096      # LRU cache (0 disables)
```

Optional per-source overrides:
```yaml
stockfish:
  source_overrides:
    game:
      depth: 10
      top_k: 5
    puzzle:
      depth: 12
      top_k: 8
```

### Distillation Settings

```yaml
distillation:
  # Loss structure: L_total = L_ce + kl_weight * L_kl
  ce_loss_enabled: true         # CE on thinking text (teaches reasoning)
  kl_weight: 1.0                # Weight for KL loss on move distribution

  temperature: 1.0              # Student softmax temperature
  min_probability: 0.001        # Floor for unanalyzed moves
  stockfish_temperature: 100.0  # CP -> probability conversion
  add_uci_move_tokens: true     # Add single-token UCI moves to vocab
  log_tokenizer_stats: false    # Print tokenizer coverage diagnostics
  use_liger: true               # Liger Kernel for memory efficiency
  loss_type: "kl"               # "kl" or "jsd"
```

**Loss Structure**:
- **CE loss** on full sequence (thinking + move) teaches the model to generate reasoning
- **KL loss** on `<uci_move>` token teaches the soft probability distribution over all moves

### Formatting Settings

```yaml
formatting:
  max_display_moves: 5     # Moves shown in thinking
  include_board: true      # Include board rendering in prompt
  randomize_order: true    # CRITICAL: prevent shortcuts
  pv_length: 5             # Moves in "Best line:" (principal variation)
```

### Reasoning Trace Settings

```yaml
reasoning_trace:
  enabled: true
  always_choose_best_move: true
  max_trace_tokens: 1024
  move_notation: "uci"
  include_opening: true
  include_tablebase: true
  min_candidates: 3
  max_candidates: 5
  include_threat_scan: true
  threat_max_checks: 2
  threat_max_hanging: 2
  include_plan: true
  plan_prob: 0.5
  include_opponent_perspective: true
  opponent_perspective_prob: 0.4
  unclear_win_margin: 0.04
  include_motifs: true
  include_quiet_move_motif: true
  perpetual_max_plies: 6
  max_motifs_per_candidate: 1
  include_positional_cues: true
  max_positional_cues: 4
  pv_prune_quiet: true
  pv_prune_quiet_plies: 2
  pv_prune_min_moves: 2
  pv_quiet_summary: true
  include_trap_detection: true
  trap_min_cp_swing: 80
  trap_min_win_prob_swing: 0.12
  trap_confirm_cp_tolerance: 30
  trap_confirm_win_prob_tolerance: 0.05
  trap_refutation_max_len: 4
  style_weights:
    thorough: 0.6
    concise: 0.25
    tactical: 0.15
  source_overrides:
    puzzle:
      style: "tactical"
      max_candidates: 4
```

Note: trap detection uses shallow‑vs‑deep comparison, so set `stockfish.shallow_depth > 0`.

### Training Eval Settings

```yaml
training:
  chess_eval_max_new_tokens: 128  # Longer generations for eval sanity
```

## Performance Notes

### Stockfish Analysis Speed

| Depth | Top-k | Time/position | With 8 workers |
|-------|-------|---------------|----------------|
| 10 | 5 | ~40ms | ~200 pos/sec |
| 12 | 5 | ~80ms | ~100 pos/sec |
| 12 | 10 | ~150ms | ~50 pos/sec |

### Preprocessing Time

| Dataset Size | Depth 10 | Depth 12 |
|-------------|----------|----------|
| 10K | ~5 min | ~10 min |
| 100K | ~50 min | ~2 hours |
| 500K | ~4 hours | ~8 hours |

### Streaming vs Preprocessing

**Preprocessing** (recommended):
- Faster training (no Stockfish during training)
- Reproducible (same analysis each epoch)
- Requires disk space

**Streaming** (on-the-fly):
- No preprocessing needed
- Different randomization each epoch
- Slower training (~50-100ms overhead per batch)

## Loss Function Details

The distillation training uses a combined loss:

```python
L_total = L_ce + L_kl

# Cross-entropy loss on full sequence (thinking + move)
L_ce = CrossEntropy(model_output, target_tokens)

# KL divergence on move token only
L_kl = KL(P_stockfish || P_model)
     = Σ P_stockfish(m) * log(P_stockfish(m) / P_model(m))
```

This combined approach:
- **CE loss**: Teaches the model to generate the thinking text (best move analysis, win probabilities, PV)
- **KL loss**: Teaches the model the soft probability distribution over ALL legal moves

**Key insight**: The reasoning trace is anchored to Stockfish's best move. If `reasoning_trace.always_choose_best_move` is enabled, the `<uci_move>` output is also the best move; otherwise it can stay as the source move while KL still teaches the soft distribution. This means:
- The model learns to reason about the best move (via CE loss on thinking)
- The model learns a quality-weighted distribution (via KL loss)

The KL (soft) loss teaches:
- The full distribution over all moves
- That multiple moves can be good (e.g., e6 at 16.3% and e5 at 16.0%)
- Relative move quality based on win probability

Both losses work together - CE for reasoning, KL for move quality.

## Probability Distribution

Stockfish centipawn scores are converted to probabilities:

```python
def cp_to_probability(centipawns, temperature=100):
    """
    Convert centipawn scores to probability distribution.
    
    temperature=100 (default): balanced distribution
    temperature=50:  sharper (focus on best moves)
    temperature=200: softer (more uniform)
    """
    exp_scores = exp(centipawns / temperature)
    return exp_scores / sum(exp_scores)
```

Example with moves evaluated at `[+50, +30, +10, -20]` cp:
- **T=50** (sharp): `[0.73, 0.20, 0.05, 0.02]` 
- **T=100** (default): `[0.52, 0.30, 0.13, 0.05]`
- **T=200** (soft): `[0.35, 0.28, 0.22, 0.15]`

## Monitoring Training

Key metrics to watch:

| Metric | Good | Bad |
|--------|------|-----|
| `top1_agreement` | >0.6 | <0.3 |
| `kl_divergence` | Decreasing | Increasing |
| `prob_on_correct` | >0.3 | <0.1 |

## Example Output

After preprocessing, each example contains:

```python
{
    'fen': 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1',
    'target_move_uci': 'e7e5',
    'best_move_uci': 'e7e5',
    'best_move_san': 'e5',
    'best_score_cp': -20,
    'best_pv': ['e5', 'Nf3', 'Nc6', 'Bb5', 'a6'],  # Principal variation
    'target_move_rank': 1,
    'target_move_cp_loss': 0,
    'target_move_prob': 0.42,
    'move_probs': {
        'e7e5': 0.42,
        'c7c5': 0.28,
        'd7d5': 0.15,
        'g8f6': 0.08,
        # ... all other legal moves with floor probability
        'a7a6': 0.001,
        'h7h6': 0.001,
    },
    'move_evaluations': [
        {'uci': 'e7e5', 'san': 'e5', 'centipawn': -20, 'cp_loss': 0, 'win_probability': 0.52},
        {'uci': 'c7c5', 'san': 'c5', 'centipawn': -28, 'cp_loss': 8, 'win_probability': 0.50},
        # ... categories derived from win_probability: winning, better, slight edge, equal, etc.
    ],
    'messages': [
        {'role': 'user', 'content': '...'},
        {'role': 'assistant', 'content': '<think>...</think>\n<uci_move>e7e5</uci_move>'}
    ]
}
```

## Troubleshooting

### "Stockfish not found"
```bash
# Check installation
which stockfish

# Install
sudo apt install stockfish  # Ubuntu/Debian
brew install stockfish       # macOS

# Or specify path
python distill/preprocess.py --stockfish-path /path/to/stockfish
```

### Slow preprocessing
- Reduce `depth` (10 instead of 12)
- Increase `workers` (up to CPU cores)
- Use `top_k=5` instead of higher values

### Out of memory during training
- Reduce `per_device_train_batch_size`
- Enable `gradient_checkpointing`
- Use smaller model

## Files Overview

| File | Purpose |
|------|---------|
| `sft/train.py` | Stage 1: Supervised fine-tuning |
| `distill/train.py` | Stage 2: Policy distillation |
| `configs/sft/config_sft.yaml` | SFT configuration |
| `configs/distill/config_distill.yaml` | Distillation configuration |
| `distill/preprocess.py` | Batch preprocessing with Stockfish |
| `src/distill/stockfish_teacher.py` | Parallel Stockfish analysis |
| `src/distill/formatting_distill.py` | Generate thinking with categories |
| `src/distill/distillation_loss.py` | Forward KL / JSD loss (Liger support) |
| `src/distill/collator_distill.py` | Data collation for training |
| `eval/evaluate_fast.py` | Fast batched evaluation with optional Stockfish |
| `scripts/evaluate.sh` | CLI wrapper for evaluation |
| `scripts/tokenizer_probe.py` | Tokenizer and data stream sanity check |
| `scripts/gut_check_report.py` | Generate a Markdown gut-check report from preprocessed distill data |
| `scripts/upload_to_hub.py` | Upload model to Hugging Face Hub |

## Uploading to Hugging Face

After training, upload your model to Hugging Face Hub:

```bash
# Login first (one-time)
huggingface-cli login

# Upload model
python scripts/upload_to_hub.py \
    --model_path ./outputs/chess-distill-final \
    --repo_id your-username/chess-llm \
    --base_model Qwen/Qwen3-0.6B

# Private repo
python scripts/upload_to_hub.py \
    --model_path ./outputs/chess-distill-final \
    --repo_id your-username/chess-llm \
    --private
```

## License

MIT License - see LICENSE file.
