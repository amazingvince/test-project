#!/usr/bin/env python3
"""
Prepare a local copy of the Global Chess Challenge 2025 starter kit.

This script is a convenience helper for running the AIcrowd starter kit
evaluation loop against a model trained in this repo.

What it does:
  1) Clones the starter kit repo into `./tmp/` (if missing).
  2) Clones the `aicrowd/chess-env` dependency via HTTPS (the upstream starter kit
     uses an SSH submodule URL which often fails without GitHub SSH keys).
  3) Copies this repo's recommended prompt templates into the starter kit's
     `player_agents/` directory.

It does NOT install dependencies or start vLLM automatically. It prints the
exact commands to run next.

Usage:
  python scripts/global_chess_challenge_2025_setup.py

  # Custom location
  python scripts/global_chess_challenge_2025_setup.py --dir ./tmp/gcc2025
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


STARTER_KIT_URL = "https://github.com/AIcrowd/global-chess-challenge-2025-starter-kit.git"
CHESS_ENV_URL = "https://github.com/aicrowd/chess-env.git"


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.check_call(cmd, cwd=str(cwd) if cwd is not None else None)


def _ensure_clone(url: str, dest: Path, *, sentinel: Path, force: bool) -> None:
    """
    Clone `url` into `dest` if the repo is missing or incomplete.

    Args:
        url: Git URL.
        dest: Destination directory.
        sentinel: A file that must exist for the clone to be considered usable.
        force: If True, delete `dest` when it exists but is incomplete.
    """
    if dest.exists() and sentinel.exists() and not force:
        return

    if dest.exists():
        if not force:
            # Common case: the starter kit has an empty `chess-env/` directory because the
            # submodule URL uses SSH. If it's empty, it's safe to delete and re-clone.
            entries = list(dest.iterdir()) if dest.is_dir() else []
            if entries:
                raise RuntimeError(
                    f"Refusing to overwrite non-empty directory: {dest}\n"
                    "Re-run with --force to delete and re-clone."
                )
        shutil.rmtree(dest, ignore_errors=True)

    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", url, str(dest)])


def _copy_prompt_templates(repo_root: Path, starter_kit_dir: Path) -> list[Path]:
    source_dir = repo_root / "competition" / "global_chess_challenge_2025"
    dest_dir = starter_kit_dir / "player_agents"
    dest_dir.mkdir(parents=True, exist_ok=True)

    templates = [
        source_dir / "prompt_minimal.jinja",
        source_dir / "prompt_with_board_ascii.jinja",
    ]

    copied: list[Path] = []
    for src in templates:
        if not src.exists():
            raise FileNotFoundError(f"Missing prompt template: {src}")
        out_name = f"chess_distill_{src.name}"
        dst = dest_dir / out_name
        shutil.copyfile(src, dst)
        copied.append(dst)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set up the Global Chess Challenge 2025 starter kit in ./tmp and copy prompt templates."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="tmp/global-chess-challenge-2025-starter-kit",
        help="Destination path for the starter kit clone (default: tmp/global-chess-challenge-2025-starter-kit).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing starter kit folders and re-clone.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    starter_kit_dir = (repo_root / args.dir).resolve()
    chess_env_dir = starter_kit_dir / "chess-env"

    _ensure_clone(
        STARTER_KIT_URL,
        starter_kit_dir,
        sentinel=starter_kit_dir / "README.md",
        force=args.force,
    )
    _ensure_clone(
        CHESS_ENV_URL,
        chess_env_dir,
        sentinel=chess_env_dir / "requirements.txt",
        force=args.force,
    )
    copied = _copy_prompt_templates(repo_root, starter_kit_dir)

    print("Starter kit ready:")
    print(f"  {starter_kit_dir}")
    print("")
    print("Copied prompt templates:")
    for p in copied:
        print(f"  {p}")
    print("")
    print("Next steps (recommended):")
    print(f"  pip install -r {starter_kit_dir / 'requirements.txt'}")
    print("")
    print("  # Start vLLM server in a separate terminal")
    print(f"  cd {starter_kit_dir / 'player_agents'}")
    print("  pip install vllm")
    print('  vllm serve "<your_hf_repo_or_local_path>" --served-model-name aicrowd-chess-model --dtype bfloat16 --gpu-memory-utilization 0.9 --enforce-eager --disable-log-stats --host 0.0.0.0 --port 5000')
    print("")
    print("  # Run local evaluation (in the starter kit directory)")
    print(f"  cd {starter_kit_dir}")
    print("  # Pick a prompt template:")
    for p in copied:
        print(f"  #   player_agents/{p.name}")
    print(
        "  python local_evaluation.py"
        f" --template-file player_agents/{copied[0].name}"
        " --endpoint http://localhost:5000/v1"
        " --games-per-opponent 10"
    )


if __name__ == "__main__":
    main()
