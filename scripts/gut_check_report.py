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
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

# Ensure repo root is on sys.path when running from subfolders
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chess
from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer

from src.distill.reasoning_trace import ReasoningTraceGenerator
from src.distill.stockfish_teacher import MoveAnalysis, PositionAnalysis
from src.utils.chess_tokenizer import ChessTokenizerMode, add_chess_tokens, generate_all_uci_moves
from src.utils.chess_utils import render_board_utf


THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", flags=re.DOTALL | re.IGNORECASE)


def setup_tokenizer(config: Dict[str, Any]) -> Any:
    """Setup tokenizer with distillation tokens (matching distill/train.py)."""
    model_config = config.get('model', {})
    model_name = model_config.get('name', 'Qwen/Qwen3-0.6B')

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer_config = config.get("tokenizer", {})
    distill_config = config.get("distillation", {})

    mode_str = tokenizer_config.get("chess_mode")
    if mode_str is None:
        mode_str = "tags_and_moves" if distill_config.get("add_uci_move_tokens", True) else "tags_only"
    if mode_str not in ("tags_only", "tags_and_moves"):
        raise ValueError(f"tokenizer.chess_mode must be 'tags_only' or 'tags_and_moves' (got {mode_str!r})")
    mode: ChessTokenizerMode = "tags_only" if mode_str == "tags_only" else "tags_and_moves"

    all_uci_moves = generate_all_uci_moves() if mode == "tags_and_moves" else None
    added = add_chess_tokens(
        tokenizer=tokenizer,
        mode=mode,
        add_think_tags=bool(tokenizer_config.get("add_think_tags", False)),
        all_uci_moves=all_uci_moves,
    )
    print(f"Added tokens: {added['special_tokens']} special, {added['move_tokens']} UCI move tokens (mode={mode})")
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    return tokenizer


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


def _extract_full_response(messages: Any) -> str:
    """Extract full assistant response including <think> and <uci_move>."""
    if not messages or not isinstance(messages, list):
        return ""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return str(msg.get("content", "")).strip()
    return ""


def _classify_position(fen: str) -> str:
    """Classify position as 'opening', 'middlegame', or 'endgame'."""
    try:
        board = chess.Board(fen)
        piece_count = len(board.piece_map())

        if piece_count <= 10:
            return "endgame"
        if board.fullmove_number <= 10:
            return "opening"
        return "middlegame"
    except Exception:
        return "unknown"


def _format_kl_distribution(move_probs: Dict[str, float], top_n: int = 10) -> str:
    """Format move probability distribution for display."""
    if not move_probs:
        return "(no distribution available)"

    # Filter out None values and sort by probability descending
    valid_probs = [(k, v) for k, v in move_probs.items() if v is not None]
    if not valid_probs:
        return "(no valid probabilities)"
    sorted_probs = sorted(valid_probs, key=lambda x: -x[1])

    lines = []
    lines.append(f"Top {min(top_n, len(sorted_probs))} moves (of {len(sorted_probs)} legal):")

    for i, (move, prob) in enumerate(sorted_probs[:top_n]):
        bar_len = int(prob * 40)  # 40-char max bar
        bar = "=" * bar_len + "." * (40 - bar_len)
        lines.append(f"  {move:6s} {prob:6.2%} [{bar}]")

    # Show sum of top-k vs rest
    rest_sum = sum(p for _, p in sorted_probs[top_n:])
    if rest_sum > 0:
        lines.append(f"  (remaining {len(sorted_probs) - top_n} moves: {rest_sum:.2%})")

    return "\n".join(lines)


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


def _pick_rows_stratified(
    ds: Any,
    rng: random.Random,
    num_samples: int,
    source: str,
    max_tries: int,
    opening_min: int = 0,
    endgame_min: int = 0,
) -> List[Dict[str, Any]]:
    """Pick rows with stratified sampling across game phases.

    Ensures a mix of opening, middlegame, and endgame positions.
    """
    if num_samples <= 0:
        return []
    total = len(ds)
    if total <= 0:
        return []

    # Target distribution: ~25% opening, ~25% endgame, ~50% middlegame
    opening_target = opening_min if opening_min > 0 else max(2, num_samples // 4)
    endgame_target = endgame_min if endgame_min > 0 else max(2, num_samples // 4)
    middlegame_target = num_samples - opening_target - endgame_target

    targets = {
        "opening": opening_target,
        "endgame": endgame_target,
        "middlegame": middlegame_target,
        "unknown": 0,  # Don't specifically target unknown
    }

    picked: Dict[str, List[Dict[str, Any]]] = {
        "opening": [],
        "middlegame": [],
        "endgame": [],
        "unknown": [],
    }
    used: set[int] = set()
    wanted_source = None if source == "mixed" else source

    tries = 0
    while sum(len(v) for v in picked.values()) < num_samples and tries < max_tries:
        tries += 1
        idx = rng.randrange(total)
        if idx in used:
            continue

        row = ds[int(idx)]
        if not isinstance(row, dict):
            continue

        # Source filter
        if wanted_source and row.get("source") != wanted_source:
            continue

        # Classify and check quota
        phase = _classify_position(row.get("fen", ""))
        if len(picked[phase]) < targets.get(phase, num_samples):
            used.add(idx)
            picked[phase].append(row)
        elif sum(len(v) for v in picked.values()) < num_samples:
            # Allow overflow to other categories if not full
            for fallback_phase in ["middlegame", "opening", "endgame"]:
                if len(picked[fallback_phase]) < targets.get(fallback_phase, 0) + 2:
                    used.add(idx)
                    picked[fallback_phase].append(row)
                    break

    # Combine all phases in order: opening, middlegame, endgame
    result = picked["opening"] + picked["middlegame"] + picked["endgame"] + picked["unknown"]
    return result


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
    parser.add_argument(
        "--stratified",
        action="store_true",
        default=True,
        help="Use stratified sampling to ensure mix of opening/middlegame/endgame (default: True).",
    )
    parser.add_argument(
        "--no-stratified",
        action="store_true",
        help="Disable stratified sampling, use random sampling instead.",
    )
    parser.add_argument(
        "--opening-samples",
        type=int,
        default=0,
        help="Minimum opening samples (0 = auto ~25%%).",
    )
    parser.add_argument(
        "--endgame-samples",
        type=int,
        default=0,
        help="Minimum endgame samples (0 = auto ~25%%).",
    )
    parser.add_argument(
        "--top-k-moves",
        type=int,
        default=10,
        help="Number of moves to show in KL distribution (default: 10).",
    )
    args = parser.parse_args()

    config = _read_yaml(args.config)
    reasoning_cfg = dict(config.get("reasoning_trace", {}) or {})
    reasoning_cfg.setdefault("include_opening", True)
    reasoning_cfg.setdefault("include_tablebase", True)

    # Setup tokenizer with distillation tokens
    tokenizer = setup_tokenizer(config)

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
    use_stratified = args.stratified and not args.no_stratified
    if use_stratified:
        rows = _pick_rows_stratified(
            ds, rng, args.num_samples, args.source, args.max_tries,
            opening_min=args.opening_samples,
            endgame_min=args.endgame_samples,
        )
    else:
        rows = _pick_rows(ds, rng, args.num_samples, args.source, args.max_tries)
    if not rows:
        print("No rows selected. Check dataset/source filters.")
        return 3

    trace_generator = ReasoningTraceGenerator(reasoning_cfg)
    trace_status = trace_generator.status()

    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Count samples by phase for header
    phase_counts = {"opening": 0, "middlegame": 0, "endgame": 0, "unknown": 0}
    for row in rows:
        phase = _classify_position(row.get("fen", ""))
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

    lines: List[str] = []
    lines.append("# Distillation Gut Check Report")
    lines.append("")
    lines.append(f"- Generated: `{now}`")
    lines.append(f"- Config: `{args.config.as_posix()}`")
    lines.append(f"- Dataset: `{dataset_path.as_posix()}`")
    lines.append(f"- Samples: `{len(rows)}` (source filter: `{args.source}`, stratified: `{use_stratified}`)")
    lines.append(f"- Phase distribution: opening=`{phase_counts['opening']}`, middlegame=`{phase_counts['middlegame']}`, endgame=`{phase_counts['endgame']}`")
    lines.append("")
    lines.append("## Assets")
    lines.append(f"- Openings: `{openings_ok}` (tsv files: `{openings_count}`)")
    lines.append(f"- Tablebases: `{tablebase_ok}` (rtb files: `{tablebase_count}`)")
    lines.append(f"- Trace status: opening=`{trace_status.get('opening_available')}` tablebase=`{trace_status.get('tablebase_available')}`")
    lines.append(f"- Tokenizer: `{config.get('model', {}).get('name', 'Qwen/Qwen3-0.6B')}` (vocab size: `{len(tokenizer)}`)")
    lines.append("")

    # Token count tracking
    token_stats = {
        'input_tokens': [],
        'output_tokens': [],
        'total_tokens': [],
    }

    for idx, row in enumerate(rows, start=1):
        fen = row.get("fen", "")
        source = row.get("source", "unknown")
        phase = _classify_position(fen)
        white_elo = row.get("white_elo", row.get("rating", ""))
        black_elo = row.get("black_elo", row.get("rating", ""))
        move_number = row.get("move_number", "")
        best_move_uci = row.get("best_move_uci", "")

        board_utf = row.get("board_utf") or render_board_utf(chess.Board(fen))

        prompt = ""
        messages = row.get("messages")
        if isinstance(messages, list) and messages:
            first = messages[0]
            if isinstance(first, dict):
                prompt = str(first.get("content", ""))

        # Get full response or generate it
        full_response = ""
        if args.use_existing_trace:
            full_response = _extract_full_response(messages)
        else:
            try:
                analysis = analysis_from_row(row)
                thinking = trace_generator.generate(
                    position={"fen": fen, "source": source},
                    analysis=analysis,
                    rng=random.Random(args.seed + idx),
                )
                # Build full training format
                move_to_use = best_move_uci or (analysis.best_move_uci if analysis else "")
                full_response = f"<think>\n{thinking}\n</think>\n<uci_move>{move_to_use}</uci_move>"
            except Exception as exc:
                full_response = f"[trace generation failed: {exc}]"

        # Format KL distribution
        move_probs = row.get("move_probs", {})
        kl_dist = _format_kl_distribution(move_probs, top_n=args.top_k_moves)

        # Count tokens for input and output
        input_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        output_token_ids = tokenizer.encode(full_response, add_special_tokens=False)
        input_token_count = len(input_token_ids)
        output_token_count = len(output_token_ids)
        total_token_count = input_token_count + output_token_count

        token_stats['input_tokens'].append(input_token_count)
        token_stats['output_tokens'].append(output_token_count)
        token_stats['total_tokens'].append(total_token_count)

        lines.append(f"## Sample {idx}")
        lines.append(f"- **Phase**: `{phase}` | source: `{source}` | move: `{move_number}` | ELO: `{white_elo}`/`{black_elo}`")
        lines.append(f"- fen: `{fen}`")
        lines.append(f"- **Tokens**: input=`{input_token_count}`, output=`{output_token_count}`, total=`{total_token_count}`")
        lines.append("")
        lines.append("### Board")
        lines.append(_code_block(board_utf))
        lines.append("")
        lines.append("### Prompt")
        lines.append(_code_block(prompt))
        lines.append("")
        lines.append("### Full Training Response")
        lines.append(_code_block(full_response))
        lines.append("")
        lines.append("### KL Distribution (Stockfish)")
        lines.append(_code_block(kl_dist))
        lines.append("")

    # Summary statistics at end
    lines.append("---")
    lines.append("")
    lines.append("## Summary Statistics")
    lines.append("")
    lines.append(f"- Total samples: `{len(rows)}`")
    lines.append(f"- Opening positions: `{phase_counts['opening']}` ({100*phase_counts['opening']/len(rows):.1f}%)")
    lines.append(f"- Middlegame positions: `{phase_counts['middlegame']}` ({100*phase_counts['middlegame']/len(rows):.1f}%)")
    lines.append(f"- Endgame positions: `{phase_counts['endgame']}` ({100*phase_counts['endgame']/len(rows):.1f}%)")
    lines.append(f"- Opening book available: `{trace_status.get('opening_available')}`")
    lines.append(f"- Tablebase available: `{trace_status.get('tablebase_available')}`")
    lines.append("")

    # Token statistics
    def _compute_stats(values: List[int]) -> Dict[str, float]:
        if not values:
            return {'mean': 0, 'min': 0, 'max': 0, 'std': 0, 'sum': 0}
        n = len(values)
        total = sum(values)
        mean = total / n
        min_val = min(values)
        max_val = max(values)
        variance = sum((x - mean) ** 2 for x in values) / n
        std = variance ** 0.5
        return {'mean': mean, 'min': min_val, 'max': max_val, 'std': std, 'sum': total}

    lines.append("### Token Statistics")
    lines.append("")
    lines.append("| Metric | Input Tokens | Output Tokens | Total Tokens |")
    lines.append("|--------|-------------|---------------|--------------|")

    input_stats = _compute_stats(token_stats['input_tokens'])
    output_stats = _compute_stats(token_stats['output_tokens'])
    total_stats = _compute_stats(token_stats['total_tokens'])

    lines.append(f"| Mean | {input_stats['mean']:.1f} | {output_stats['mean']:.1f} | {total_stats['mean']:.1f} |")
    lines.append(f"| Std Dev | {input_stats['std']:.1f} | {output_stats['std']:.1f} | {total_stats['std']:.1f} |")
    lines.append(f"| Min | {input_stats['min']} | {output_stats['min']} | {total_stats['min']} |")
    lines.append(f"| Max | {input_stats['max']} | {output_stats['max']} | {total_stats['max']} |")
    lines.append(f"| Sum | {input_stats['sum']} | {output_stats['sum']} | {total_stats['sum']} |")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
