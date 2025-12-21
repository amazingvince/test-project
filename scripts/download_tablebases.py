#!/usr/bin/env python3
"""
Download Syzygy tablebases (WDL/DTZ) from a public index.
"""

from __future__ import annotations

import argparse
import re
import shutil
import urllib.request
from pathlib import Path
from typing import Iterable, List


DEFAULT_BASE_URL = "https://tablebase.lichess.ovh/tables/standard/"


def fetch_index(base_url: str) -> List[str]:
    with urllib.request.urlopen(base_url) as response:
        html = response.read().decode("utf-8", errors="ignore")
    return sorted(set(re.findall(r'href="([^"]+\\.rtb[zw])"', html)))


def piece_count(name: str) -> int:
    return sum(1 for ch in name if ch.isalpha() and ch.isupper())


def filter_files(
    files: Iterable[str],
    pieces: Iterable[int],
    include_wdl: bool,
    include_dtz: bool,
) -> List[str]:
    wanted = set(int(p) for p in pieces)
    results = []
    for name in files:
        if name.endswith(".rtbw") and not include_wdl:
            continue
        if name.endswith(".rtbz") and not include_dtz:
            continue
        if piece_count(Path(name).stem) in wanted:
            results.append(name)
    return results


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Syzygy tablebases.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/syzygy"),
        help="Directory to store tablebase files",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="Base URL for the tablebase index",
    )
    parser.add_argument(
        "--pieces",
        type=str,
        default="3,4,5",
        help="Comma-separated piece counts to download (e.g. 3,4,5)",
    )
    parser.add_argument(
        "--wdl",
        action="store_true",
        help="Download WDL files (.rtbw)",
    )
    parser.add_argument(
        "--dtz",
        action="store_true",
        help="Download DTZ files (.rtbz)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit number of files to download (0 = no limit)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the files that would be downloaded without downloading",
    )
    args = parser.parse_args()

    include_wdl = args.wdl or not args.dtz
    include_dtz = args.dtz or not args.wdl

    pieces = [int(p.strip()) for p in args.pieces.split(",") if p.strip()]
    if not pieces:
        print("No piece counts provided.")
        return 1

    print(f"Fetching index from {args.base_url}")
    index_files = fetch_index(args.base_url)
    filtered = filter_files(index_files, pieces, include_wdl, include_dtz)

    if args.max_files and args.max_files > 0:
        filtered = filtered[: args.max_files]

    if args.dry_run:
        print(f"Found {len(filtered)} files:")
        for name in filtered:
            print(f"  {name}")
        return 0

    for name in filtered:
        url = f"{args.base_url.rstrip('/')}/{name}"
        dest = args.output_dir / name
        if dest.exists():
            print(f"Skipping existing {dest}")
            continue
        print(f"Downloading {url} -> {dest}")
        download_file(url, dest)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
