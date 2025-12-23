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

**Tokenizer + KL modes**
- `tokenizer.chess_mode: tags_and_moves` adds `<uci_move>` tags plus all 8,064 UCI moves as single tokens.
  - Use with `distillation.move_distill_mode: move_token` (fastest; KL is applied at the `<uci_move>` position).
- `tokenizer.chess_mode: tags_only` adds only `<uci_move>` and `</uci_move>`.
  - Use with `distillation.move_distill_mode: move_prefix` (no 8k move vocab; KL is applied token-by-token over the move string).

**Related docs**
- `docs/chess_reasoning_trace_generator.md`: reasoning trace design and configuration knobs.

**Openings + tablebases (optional)**
If `reasoning_trace.enabled: true`:
- `reasoning_trace.include_opening: true` enriches traces with opening names when `./data/openings/*.tsv` is present.
  - Download: `python scripts/download_openings.py --output-dir ./data/openings`
- `reasoning_trace.include_tablebase: true` enriches traces with Syzygy endgame info when `./data/syzygy` is present.
  - Download: `python scripts/download_tablebases.py --output-dir ./data/syzygy --pieces 3,4,5`
