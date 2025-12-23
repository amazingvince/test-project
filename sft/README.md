# SFT (Supervised Fine-Tuning)

Stage 1 trains a chat model to output a single chess move for a position prompt.
The training target is wrapped in `<uci_move>...</uci_move>`.

This repo supports two common SFT targets:
- **Played move** (from games / puzzle solutions): closer to the raw data distribution.
- **Best move** (Stockfish): directly optimizes for strength and pairs well with the
  distillation-style reasoning traces.

**Entry points**
- `sft/train.py`: SFT training (streaming or preprocessed dataset).
- `sft/train_hf.py`: SFT training using vanilla HF loss (no weighting / CCE).
- `sft/preprocess.py`: Optional offline dataset generation with Stockfish move evaluations (useful for analysis/ablations; distillation lives in `distill/`).

**Typical commands**
- Streaming SFT (no local dataset): `python sft/train.py --config configs/sft/config_sft.yaml --streaming`
- Streaming SFT with Stockfish reasoning traces (slow): `python sft/train.py --config configs/sft/config_with_eval.yaml --streaming`
- Preprocessed SFT: `python sft/train.py --config configs/sft/config_sft.yaml --preprocessed_path ./data/chess_sft`
- Preprocess with Stockfish evals: `python sft/preprocess.py --config configs/sft/config_with_eval.yaml --output ./data/chess_with_eval`
- Vanilla-loss SFT on preprocessed reasoning traces: `python sft/train_hf.py --config configs/sft/config_sft_h100_no_moves_hf.yaml --preprocessed_path ./data/chess_sft_h100_best`

**Configuration**
- `configs/sft/config_sft.yaml`: baseline SFT training.
- `configs/sft/config_with_eval.yaml`: Stockfish-annotated datasets + reasoning traces (trains on best move by default).
- `configs/sft/config_sft_h100_no_moves_hf.yaml`: distill-style traces + best move target, trained with standard HF CE loss.

**Tokenizer**
- `tokenizer.chess_mode: tags_only` adds only `<uci_move>` and `</uci_move>` (recommended for SFT).
- `tokenizer.chess_mode: tags_and_moves` additionally adds all 8,064 UCI move strings as tokens (mostly useful for distillation).

**Openings + tablebases (optional)**
If `reasoning_trace.enabled: true`:
- `reasoning_trace.include_opening: true` enriches traces with opening names when `./data/openings/*.tsv` is present.
  - Download: `python scripts/download_openings.py --output-dir ./data/openings`
- `reasoning_trace.include_tablebase: true` enriches traces with Syzygy endgame info when `./data/syzygy` is present.
  - Download: `python scripts/download_tablebases.py --output-dir ./data/syzygy --pieces 3,4,5`

These resources only affect the *trace text*; the SFT target move still comes from the dataset (played/best).
