#!/usr/bin/env python3
"""
Generate an end-to-end gut-check report for distillation training data.

What this script does:
1) Ensures opening TSVs and Syzygy tablebases exist (downloads if missing).
2) Loads a preprocessed distillation dataset from disk.
3) Samples random positions and exports a Markdown report with:
   - ASCII board
   - Input prompt
   - Regenerated reasoning trace (from stored Stockfish analysis)

Typical usage:
  python scripts/gut_check_report.py --dataset ./data/chess_distill --num-samples 20
"""

from __future__ import annotations

import argparse
import datetime as _dt
import random
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

# Ensure repo root is on sys.path when running from subfolders
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Suppress optional-import warnings in src/__init__.py for environments without torch.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"Distillation modules not available:.*",
)

import chess
from datasets import DatasetDict, load_from_disk

from src.distill.reasoning_trace import ReasoningTraceGenerator
from src.distill.stockfish_teacher import MoveAnalysis, PositionAnalysis
from src.utils.chess_utils import render_board_utf


THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", flags=re.DOTALL | re.IGNORECASE)


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _count_files(folder: Path, patterns: Sequence[str]) -> int:
    if not folder.exists():
        return 0
    count = 0
    for pattern in patterns:
        count += len(list(folder.glob(pattern)))
    return count


def _run_python(script: Path, args: List[str]) -> None:
    cmd = [sys.executable, str(script), *args]
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def ensure_openings(openings_dir: Path, download: bool) -> Tuple[bool, int]:
    openings_dir = openings_dir.resolve()
    existing = _count_files(openings_dir, ("*.tsv",))
    if existing > 0:
        return True, existing
    if not download:
        return False, 0
    script = ROOT / "scripts" / "download_openings.py"
    _run_python(script, ["--output-dir", str(openings_dir)])
    return True, _count_files(openings_dir, ("*.tsv",))


def ensure_tablebases(
    tablebase_dir: Path,
    download: bool,
    pieces: str,
    max_files: int,
) -> Tuple[bool, int]:
    tablebase_dir = tablebase_dir.resolve()
    existing = _count_files(tablebase_dir, ("*.rtbw", "*.rtbz"))
    if existing > 0:
        return True, existing
    if not download:
        return False, 0
    if not max_files or max_files <= 0:
        print(
            "Note: Syzygy tablebases can be multiple GB; "
            "use --tablebase-max-files to limit downloads for a quick gut-check."
        )
    script = ROOT / "scripts" / "download_tablebases.py"
    args = ["--output-dir", str(tablebase_dir), "--pieces", pieces]
    if max_files and max_files > 0:
        args.extend(["--max-files", str(max_files)])
    _run_python(script, args)
    return True, _count_files(tablebase_dir, ("*.rtbw", "*.rtbz"))


def _extract_thinking_from_messages(messages: Any) -> str:
    if not messages or not isinstance(messages, list):
        return ""
    assistant = None
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            assistant = msg.get("content", "")
            break
    if not assistant:
        return ""
    match = THINK_RE.search(assistant)
    if not match:
        return assistant.strip()
    return match.group(1).strip()


def analysis_from_row(row: Dict[str, Any]) -> PositionAnalysis:
    fen = row.get("fen")
    if not fen:
        raise ValueError("Row missing 'fen'")

    evals = row.get("move_evaluations") or []
    if not evals:
        raise ValueError("Row missing 'move_evaluations' (need preprocessed distill data)")

    move_analyses: List[MoveAnalysis] = []
    for item in evals:
        move_analyses.append(
            MoveAnalysis(
                uci=str(item.get("uci", "")),
                san=str(item.get("san", item.get("uci", ""))),
                centipawn=int(item.get("centipawn", 0) or 0),
                cp_loss=int(item.get("cp_loss", 0) or 0),
                category=str(item.get("category", "")),
                mate_in=item.get("mate_in"),
                win_probability=float(item.get("win_probability", 0.5) or 0.5),
                pv_uci=list(item.get("pv_uci") or []),
            )
        )

    # Best move fields are stored by preprocess; fall back to the first deep entry.
    best_move_uci = row.get("best_move_uci") or move_analyses[0].uci
    best_move_san = row.get("best_move_san") or move_analyses[0].san
    best_score_cp = int(row.get("best_score_cp", move_analyses[0].centipawn) or 0)

    return PositionAnalysis(
        fen=fen,
        move_analyses=move_analyses,
        best_move_uci=best_move_uci,
        best_move_san=best_move_san,
        best_score_cp=best_score_cp,
        best_pv=list(row.get("best_pv") or []),
        move_probs=dict(row.get("move_probs") or {}),
        top_k_moves=list(row.get("top_k_moves") or [m.uci for m in move_analyses]),
        shallow_move_cps=dict(row.get("shallow_move_cps") or {}),
        shallow_move_win_probs=dict(row.get("shallow_move_win_probs") or {}),
        confirm_move_cps=dict(row.get("confirm_move_cps") or {}),
        confirm_move_win_probs=dict(row.get("confirm_move_win_probs") or {}),
    )


def _pick_rows(
    ds: Any,
    rng: random.Random,
    num_samples: int,
    source: str,
    max_tries: int,
) -> List[Dict[str, Any]]:
    if num_samples <= 0:
        return []
    total = len(ds)
    if total <= 0:
        return []

    wanted_source = None if source == "mixed" else source
    picked: List[Dict[str, Any]] = []
    used: set[int] = set()

    tries = 0
    while len(picked) < num_samples and tries < max_tries:
        tries += 1
        idx = rng.randrange(total)
        if idx in used:
            continue
        row = ds[int(idx)]
        if not isinstance(row, dict):
            continue
        if wanted_source is not None and row.get("source") != wanted_source:
            continue
        used.add(idx)
        picked.append(row)

    return picked


def _code_block(text: str) -> str:
    text = (text or "").rstrip()
    return f"```text\n{text}\n```"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a distillation gut-check report.")
    parser.add_argument("--config", type=Path, default=Path("configs/distill/config_distill.yaml"))
    parser.add_argument("--dataset", type=Path, default=Path("data/chess_distill"))
    parser.add_argument("--output", type=Path, default=Path("reports/gut_check_report.md"))
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", type=str, default="mixed", choices=("mixed", "game", "puzzle"))
    parser.add_argument("--max-tries", type=int, default=5000)

    parser.add_argument("--no-download", action="store_true", help="Do not download openings/tablebases")
    parser.add_argument("--openings-dir", type=Path, default=Path("data/openings"))
    parser.add_argument("--tablebase-dir", type=Path, default=Path("data/syzygy"))
    parser.add_argument("--tablebase-pieces", type=str, default="3,4,5")
    parser.add_argument("--tablebase-max-files", type=int, default=0)

    parser.add_argument(
        "--no-auto-preprocess",
        action="store_true",
        help="If the dataset folder is missing, do not run distill/preprocess.py automatically.",
    )
    parser.add_argument(
        "--preprocess-size",
        type=int,
        default=0,
        help="If auto-preprocessing, number of positions to generate (0 = choose a small default).",
    )
    parser.add_argument(
        "--preprocess-output",
        type=Path,
        default=None,
        help="If auto-preprocessing, where to write the dataset (default: --dataset).",
    )
    parser.add_argument("--preprocess-depth", type=int, default=None)
    parser.add_argument("--preprocess-workers", type=int, default=None)
    parser.add_argument("--preprocess-batch-size", type=int, default=None)
    parser.add_argument("--preprocess-games-ratio", type=float, default=None)
    parser.add_argument("--preprocess-stockfish-path", type=str, default=None)

    parser.add_argument(
        "--use-existing-trace",
        action="store_true",
        help="Use <think> from stored messages instead of regenerating with current trace generator.",
    )
    args = parser.parse_args()

    config = _read_yaml(args.config)
    reasoning_cfg = dict(config.get("reasoning_trace", {}) or {})
    reasoning_cfg.setdefault("include_opening", True)
    reasoning_cfg.setdefault("include_tablebase", True)

    download = not args.no_download
    openings_ok, openings_count = ensure_openings(args.openings_dir, download)
    tablebase_ok, tablebase_count = ensure_tablebases(
        args.tablebase_dir,
        download,
        pieces=args.tablebase_pieces,
        max_files=args.tablebase_max_files,
    )

    # If the user passed custom directories and the config does not override, wire them in.
    if "opening_paths" not in reasoning_cfg and args.openings_dir.exists():
        reasoning_cfg["opening_paths"] = [
            str(p) for p in sorted(args.openings_dir.glob("*.tsv"))
        ]
    if "tablebase_paths" not in reasoning_cfg and args.tablebase_dir.exists():
        reasoning_cfg["tablebase_paths"] = [str(args.tablebase_dir)]

    dataset_path = args.dataset
    if not dataset_path.exists():
        if args.no_auto_preprocess:
            print(f"Dataset not found: {dataset_path}")
            print("Run: python distill/preprocess.py --config configs/distill/config_distill.yaml")
            return 2

        preprocess_out = (args.preprocess_output or dataset_path).resolve()
        preprocess_size = int(args.preprocess_size or max(args.num_samples * 50, 500))
        print(f"Dataset not found: {dataset_path}")
        print(f"Auto-preprocessing a small dataset ({preprocess_size} positions) to: {preprocess_out}")
        print("Tip: use --preprocess-size to change this, or run distill/preprocess.py for a full dataset.")

        cmd = [
            sys.executable,
            str(ROOT / "distill" / "preprocess.py"),
            "--config",
            str(args.config),
            "--output",
            str(preprocess_out),
            "--size",
            str(preprocess_size),
        ]
        if args.preprocess_depth is not None:
            cmd.extend(["--depth", str(args.preprocess_depth)])
        if args.preprocess_workers is not None:
            cmd.extend(["--workers", str(args.preprocess_workers)])
        if args.preprocess_batch_size is not None:
            cmd.extend(["--batch-size", str(args.preprocess_batch_size)])
        if args.preprocess_games_ratio is not None:
            cmd.extend(["--games-ratio", str(args.preprocess_games_ratio)])
        if args.preprocess_stockfish_path:
            cmd.extend(["--stockfish-path", args.preprocess_stockfish_path])

        subprocess.run(cmd, check=True, cwd=str(ROOT))
        dataset_path = preprocess_out

    ds = load_from_disk(str(dataset_path))
    if isinstance(ds, DatasetDict):
        if "train" in ds:
            ds = ds["train"]
        else:
            ds = next(iter(ds.values()))

    rng = random.Random(args.seed)
    rows = _pick_rows(ds, rng, args.num_samples, args.source, args.max_tries)
    if not rows:
        print("No rows selected. Check dataset/source filters.")
        return 3

    trace_generator = ReasoningTraceGenerator(reasoning_cfg)
    trace_status = trace_generator.status()

    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("# Distillation Gut Check Report")
    lines.append("")
    lines.append(f"- Generated: `{now}`")
    lines.append(f"- Config: `{args.config.as_posix()}`")
    lines.append(f"- Dataset: `{dataset_path.as_posix()}`")
    lines.append(f"- Samples: `{len(rows)}` (source filter: `{args.source}`)")
    lines.append("")
    lines.append("## Assets")
    lines.append(f"- Openings: `{openings_ok}` (tsv files: `{openings_count}`)")
    lines.append(f"- Tablebases: `{tablebase_ok}` (rtb files: `{tablebase_count}`)")
    lines.append(f"- Trace status: opening=`{trace_status.get('opening_available')}` tablebase=`{trace_status.get('tablebase_available')}`")
    lines.append("")

    for idx, row in enumerate(rows, start=1):
        fen = row.get("fen", "")
        source = row.get("source", "unknown")
        white_elo = row.get("white_elo", row.get("rating", ""))
        black_elo = row.get("black_elo", row.get("rating", ""))
        move_number = row.get("move_number", "")

        board_utf = row.get("board_utf") or render_board_utf(chess.Board(fen))

        prompt = ""
        messages = row.get("messages")
        if isinstance(messages, list) and messages:
            first = messages[0]
            if isinstance(first, dict):
                prompt = str(first.get("content", ""))

        thinking = ""
        if args.use_existing_trace:
            thinking = _extract_thinking_from_messages(messages)
        else:
            try:
                analysis = analysis_from_row(row)
                thinking = trace_generator.generate(
                    position={"fen": fen, "source": source},
                    analysis=analysis,
                    rng=random.Random(args.seed + idx),
                )
            except Exception as exc:
                thinking = f"[trace generation failed: {exc}]"

        lines.append(f"## Sample {idx}")
        lines.append(f"- source: `{source}`  move_number: `{move_number}`  white_elo: `{white_elo}`  black_elo: `{black_elo}`")
        lines.append(f"- fen: `{fen}`")
        lines.append("")
        lines.append("### Board")
        lines.append(_code_block(board_utf))
        lines.append("")
        lines.append("### Prompt")
        lines.append(_code_block(prompt))
        lines.append("")
        lines.append("### Reasoning Trace")
        lines.append(_code_block(thinking))
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
