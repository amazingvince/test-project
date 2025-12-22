# Chess LLM Training (SFT + Stockfish Distillation)

This repo trains a pretrained chat LLM to play chess with a two-stage workflow:
1. **SFT** (supervised fine-tuning) on human games/puzzles.
2. **Policy distillation** from a Stockfish teacher distribution over legal moves.

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

Stockfish is optional but strongly recommended for distillation and richer eval:
- Linux: `sudo apt install stockfish`
- macOS: `brew install stockfish`

### Stage 1: SFT

```bash
# Streaming from Hugging Face (no local dataset)
python sft/train.py --config configs/sft/config_sft.yaml --streaming

# Or train from a preprocessed dataset
python sft/train.py --config configs/sft/config_sft.yaml --preprocessed_path ./data/chess_sft
```

More details: `sft/README.md`

### Stage 2: Distillation

```bash
# Preprocess with Stockfish analysis
python distill/preprocess.py --config configs/distill/config_distill.yaml --output ./data/chess_distill

# Train with distillation
python distill/train.py --config configs/distill/config_distill.yaml --preprocessed_path ./data/chess_distill
```

Hardware presets:
- `configs/distill/config_distill.yaml`: single RTX 5090 + 32 CPU cores
- `configs/distill/config_distill_h100.yaml`: single H100 + 32 CPU cores

More details: `distill/README.md`

## Evaluation

```bash
# Mixed games + puzzles
python eval/evaluate_fast.py --model ./outputs/chess-sft-final --config configs/sft/config_sft.yaml --source mixed --num_positions 1000

# Puzzles only
python eval/evaluate_fast.py --model ./outputs/chess-sft-final --source puzzles --num_positions 500
```

More details: `eval/README.md`

## Useful Scripts

- Tokenization sanity checks: `python scripts/tokenizer_probe.py --config configs/distill/config_distill.yaml`
- End-to-end report (prompt + trace samples): `python scripts/gut_check_report.py --dataset ./data/chess_distill --output ./reports/gut_check_report.md`
- Download openings: `python scripts/download_openings.py --output-dir ./data/openings`
- Download tablebases: `python scripts/download_tablebases.py --output-dir ./data/syzygy --pieces 3,4,5`

More details: `scripts/README.md`

## Competition Submission

- Global Chess Challenge 2025 (AIcrowd): `competition/global_chess_challenge_2025/README.md`

## Design Docs

- `docs/chess_reasoning_trace_generator.md`: reasoning trace generator design and knobs
- `docs/STOCKFISH_EVAL_TRAINING.md`: Stockfish-eval dataset generation and weighting ideas
- `docs/OPTIMIZATION_GUIDE.md`: performance notes (CCE/Liger/etc.)
- End-to-end Linux GPU guide: `docs/END_TO_END_LINUX_GPU.md`
