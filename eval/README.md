# Evaluation

`eval/evaluate_fast.py` runs a lightweight move-level evaluation on streamed positions from games and/or puzzles.

**Typical commands**
- Mixed (games + puzzles): `python eval/evaluate_fast.py --model ./outputs/chess-sft-final --config configs/sft/config_sft.yaml --source mixed --num_positions 1000`
- Puzzles only: `python eval/evaluate_fast.py --model ./outputs/chess-sft-final --source puzzles --num_positions 500`

**Notes**
- Supports per-source breakdown metrics when evaluating mixed sources.
- Can optionally compute ACPL if Stockfish is available.

