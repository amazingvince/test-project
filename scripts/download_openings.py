#!/usr/bin/env python3
"""
Download lichess-org chess openings TSV files into ./data/openings.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path


DEFAULT_BASE_URL = "https://raw.githubusercontent.com/lichess-org/chess-openings/master/"
DEFAULT_FILES = ("a.tsv", "b.tsv", "c.tsv", "d.tsv", "e.tsv")


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download chess opening TSV files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/openings"),
        help="Directory to store TSV files",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="Base URL for TSV files",
    )
    parser.add_argument(
        "--files",
        type=str,
        default=",".join(DEFAULT_FILES),
        help="Comma-separated list of TSV files to download",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files if they already exist",
    )
    args = parser.parse_args()

    files = [name.strip() for name in args.files.split(",") if name.strip()]
    if not files:
        print("No files requested.")
        return 1

    for name in files:
        url = f"{args.base_url.rstrip('/')}/{name}"
        dest = args.output_dir / name
        if dest.exists() and not args.overwrite:
            print(f"Skipping existing {dest}")
            continue
        print(f"Downloading {url} -> {dest}")
        download_file(url, dest)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
