"""
`src` is the internal Python package for this repo.

This project is script-first: the user-facing entry points live in:
- `sft/`     (supervised fine-tuning)
- `distill/` (policy distillation with Stockfish)
- `eval/`    (evaluation utilities)

Reusable code lives in subpackages:
- `src.utils`   shared helpers (chess parsing/rendering, dataset plumbing, etc.)
- `src.distill` distillation-specific logic (teacher, loss, collators, traces)

To keep imports predictable (and avoid importing heavyweight deps at package
import time), this `__init__` intentionally does not re-export submodules.
Import what you need directly, e.g.:

    from src.utils.chess_utils import render_board_utf
    from src.distill.stockfish_teacher import StockfishTeacher
"""

__all__: list[str] = []

