# SFT (Supervised Fine-Tuning)

Stage 1 trains a chat model to output a single chess move for a position prompt.
The training target is the played move wrapped in `<uci_move>...</uci_move>`.

**Entry points**
- `sft/train.py`: SFT training (streaming or preprocessed dataset).
- `sft/preprocess.py`: Optional offline dataset generation with Stockfish move evaluations (useful for analysis/ablations; distillation lives in `distill/`).

**Typical commands**
- Streaming SFT (no local dataset): `python sft/train.py --config configs/sft/config_sft.yaml --streaming`
- Preprocessed SFT: `python sft/train.py --config configs/sft/config_sft.yaml --preprocessed_path ./data/chess_sft`
- Preprocess with Stockfish evals: `python sft/preprocess.py --config configs/sft/config_with_eval.yaml --output ./data/chess_with_eval`

**Configuration**
- `configs/sft/config_sft.yaml`: baseline SFT training.
- `configs/sft/config_with_eval.yaml`: preprocessing/training settings for Stockfish-annotated datasets.

