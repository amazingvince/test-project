# End-to-End Linux GPU Setup (Train → Upload → AIcrowd Submit)

This guide is the “happy path” for taking this repo from a fresh Linux GPU VM to:
- a trained model on Hugging Face, and
- a local AIcrowd starter-kit evaluation run using the same prompt template you trained with.

It assumes Ubuntu 22.04+ and a single NVIDIA GPU.

## 0) Create a GPU Instance

Recommended:
- Ubuntu 22.04+ (or equivalent)
- 1× GPU (RTX 5090 24GB+ or H100 80GB)
- 16–32+ CPU cores (Stockfish preprocessing is CPU-heavy)
- 200GB+ disk (more if downloading 6-piece tablebases)

Sanity check:

```bash
nvidia-smi
```

## 1) Clone the Repo

```bash
git clone <YOUR_REPO_URL>
cd chess-llm-sft-cleanup-1
```

## 2) Install Dependencies (GPU + Training Stack)

This repo includes a one-shot setup helper:

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

What it does:
- Creates `./venv`
- Installs PyTorch (CUDA build), repo dependencies, FlashAttention2 (best-effort), Cut Cross Entropy (optional)
- Installs an optimized Stockfish binary for your CPU (AVX2/BMI2/etc)

If the PyTorch CUDA index in `scripts/setup.sh` does not match your environment, replace it with the correct one from https://pytorch.org/get-started/locally/.

Activate the venv in new shells:

```bash
source venv/bin/activate
```

## 3) Configure Weights & Biases (wandb)

```bash
wandb login
export WANDB_PROJECT="chess-distill"
export WANDB_ENTITY="<your_wandb_user_or_org>"   # optional
```

Training configs already set `report_to: wandb`.

## 4) Download Openings + Endgame Tablebases (Optional, Recommended)

Openings:

```bash
python scripts/download_openings.py --output-dir ./data/openings
```

Syzygy tablebases (3/4/5 pieces; reasonable size):

```bash
python scripts/download_tablebases.py --output-dir ./data/syzygy --pieces 3,4,5
```

Notes:
- 6-piece tables are much larger; only download if you want them.
- Distillation configs expect `./data/syzygy` by default (`configs/distill/config_distill*.yaml`).

## 5) Stage 1: SFT (Supervised Fine-Tuning)

### Option A: Fast SFT (streaming, no Stockfish)

Streaming from HF (no local dataset). This does not run Stockfish by default:

```bash
python sft/train.py --config configs/sft/config_sft.yaml --streaming
```

If you want *distill-style reasoning traces* in streaming SFT, enable `reasoning_trace.enabled: true`
and provide `stockfish` settings (this is much slower).

### Option B: SFT with distill-style reasoning traces (recommended)

If you want the *same* reasoning-trace format used in distillation (and a “best move” target),
preprocess with Stockfish first:

```bash
python sft/preprocess.py --config configs/sft/config_with_eval.yaml --output ./data/chess_sft_best
python sft/train.py --config configs/sft/config_with_eval.yaml --preprocessed_path ./data/chess_sft_best
```

H100 preset (no 8k move tokens; distill-style traces + best move):

```bash
python sft/preprocess.py --config configs/sft/config_sft_h100_no_moves.yaml --output ./data/chess_sft_h100_best
python sft/train.py --config configs/sft/config_sft_h100_no_moves.yaml --preprocessed_path ./data/chess_sft_h100_best
```

### Option C: SFT from a local dataset (no Stockfish)

Train from a local preprocessed dataset (no Stockfish evals / no reasoning traces):

```bash
python sft/train.py --config configs/sft/config_sft.yaml --preprocessed_path ./data/chess_sft
```

Outputs land under `training.output_dir` in the config (default: `./outputs/...`).

## 5.1) Tokenization Sanity Check (Optional)

If you want to confirm the special tags are present and see how UCI moves tokenize:

```bash
python scripts/tokenizer_probe.py --config configs/distill/config_distill.yaml
```

## 6) Stage 2: Distillation Preprocess (Stockfish Teacher)

This generates a distillation dataset with Stockfish move probabilities and reasoning traces.

```bash
python distill/preprocess.py --config configs/distill/config_distill.yaml --output ./data/chess_distill
```

Useful knobs:
- `configs/distill/config_distill.yaml`: tuned for single RTX 5090 + 32 CPU cores
- `configs/distill/config_distill_h100.yaml`: tuned for single H100 + 32 CPU cores
- Adjust `stockfish.num_workers` for your CPU
- Use `--size` to iterate faster:
  - `--size 50000` (smoke test)
  - `--size 500000` (meaningful run)

## 7) Stage 2: Distillation Train

```bash
python distill/train.py --config configs/distill/config_distill.yaml --preprocessed_path ./data/chess_distill
```

If you are training on H100:

```bash
python distill/train.py --config configs/distill/config_distill_h100.yaml --preprocessed_path ./data/chess_distill
```

## 8) Quick Sanity Check Report (Recommended)

```bash
python scripts/gut_check_report.py --dataset ./data/chess_distill --num-samples 20 --output ./reports/gut_check_report.md
```

This produces a Markdown report with prompt + reasoning trace examples.

## 9) Evaluate

```bash
python eval/evaluate_fast.py --model ./outputs/<your-output-dir> --source mixed --num_positions 1000
```

For ACPL, ensure Stockfish is installed and provide `--stockfish /usr/bin/stockfish` if needed.

## 10) Upload to Hugging Face Hub

```bash
huggingface-cli login
python scripts/upload_to_hub.py --model_path ./outputs/<your-output-dir> --repo_id <you>/<model-name>
```

## 11) Local AIcrowd Starter-Kit Compatibility Test

This clones the starter kit into `./tmp/`, clones `chess-env` via HTTPS, and copies the prompt templates:

```bash
python scripts/global_chess_challenge_2025_setup.py
```

Then follow `competition/global_chess_challenge_2025/README.md`.

## 12) Submit to AIcrowd

Use:
- your HF model repo (from Step 10), and
- the prompt template `competition/global_chess_challenge_2025/prompt_minimal.jinja`

See: `competition/global_chess_challenge_2025/README.md`.
