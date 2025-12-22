# Policy Distillation

Stage 2 refines the move policy by distilling a Stockfish teacher distribution over legal moves.

**Entry points**
- `distill/preprocess.py`: Streams positions, runs Stockfish, and writes a distillation dataset to disk.
- `distill/train.py`: Trains the student with cross-entropy over the full response plus a distillation loss on the move token(s).

**Typical commands**
- Preprocess: `python distill/preprocess.py --config configs/distill/config_distill.yaml --output ./data/chess_distill`
- Train: `python distill/train.py --config configs/distill/config_distill.yaml --preprocessed_path ./data/chess_distill`

**Configuration**
- `configs/distill/config_distill.yaml`: tuned for a single RTX 5090 + 32 CPU cores.
- `configs/distill/config_distill_h100.yaml`: tuned for a single H100 + 32 CPU cores.

**Related docs**
- `docs/chess_reasoning_trace_generator.md`: reasoning trace design and configuration knobs.

