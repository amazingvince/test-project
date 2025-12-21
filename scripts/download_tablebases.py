#!/usr/bin/env python3
"""
Download Syzygy tablebases (WDL/DTZ) from a public index.
"""

from __future__ import annotations

import argparse
import re
import shutil
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Iterable, List


DEFAULT_BASE_URL = "https://tablebase.lichess.ovh/tables/standard/"


def _ensure_trailing_slash(url: str) -> str:
    return url if url.endswith("/") else url + "/"


def _fetch_html(url: str) -> str:
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8", errors="ignore")


def _list_hrefs(url: str) -> List[str]:
    html = _fetch_html(url)
    return sorted(set(re.findall(r'href="([^"]+)"', html)))


def _list_dirs(url: str) -> List[str]:
    return [
        href for href in _list_hrefs(url)
        if href.endswith("/") and href not in ("../", "/")
    ]


def _list_rtb_files(url: str) -> List[str]:
    return [
        href for href in _list_hrefs(url)
        if href.endswith(".rtbw") or href.endswith(".rtbz")
    ]


def _resolve_source_dirs(
    base_url: str,
    pieces: Iterable[int],
    include_wdl: bool,
    include_dtz: bool,
) -> List[str]:
    base_url = _ensure_trailing_slash(base_url)

    # If the URL already lists rtb files, treat it as a leaf directory.
    if _list_rtb_files(base_url):
        return [base_url]

    pieces_set = set(int(p) for p in pieces)
    source_dirs: List[str] = []

    if pieces_set & {3, 4, 5}:
        if include_wdl:
            source_dirs.append(urllib.parse.urljoin(base_url, "3-4-5-wdl/"))
        if include_dtz:
            source_dirs.append(urllib.parse.urljoin(base_url, "3-4-5-dtz/"))

    if 6 in pieces_set:
        if include_wdl:
            source_dirs.append(urllib.parse.urljoin(base_url, "6-wdl/"))
        if include_dtz:
            source_dirs.append(urllib.parse.urljoin(base_url, "6-dtz/"))

    if 7 in pieces_set:
        # 7-piece tables are grouped into subfolders (pawnful/pawnless buckets).
        seven_url = urllib.parse.urljoin(base_url, "7/")
        for subdir in _list_dirs(seven_url):
            source_dirs.append(urllib.parse.urljoin(seven_url, subdir))

    # If nothing matched, fall back to any leaf dirs we can find (best effort).
    if not source_dirs:
        for subdir in _list_dirs(base_url):
            candidate = urllib.parse.urljoin(base_url, subdir)
            if _list_rtb_files(candidate):
                source_dirs.append(candidate)

    return source_dirs


def piece_count(name: str) -> int:
    return sum(1 for ch in name if ch.isalpha() and ch.isupper())


def filter_files(
    files: Iterable[tuple[str, str]],
    pieces: Iterable[int],
) -> List[str]:
    wanted = set(int(p) for p in pieces)
    results = []
    for url, name in files:
        if piece_count(Path(name).stem) in wanted:
            results.append((url, name))
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

    source_dirs = _resolve_source_dirs(
        args.base_url,
        pieces,
        include_wdl=include_wdl,
        include_dtz=include_dtz,
    )
    if not source_dirs:
        print("No suitable tablebase directories found at that URL.")
        return 1

    index_files: List[tuple[str, str]] = []
    for folder_url in source_dirs:
        folder_url = _ensure_trailing_slash(folder_url)
        for href in _list_rtb_files(folder_url):
            if href.endswith(".rtbw") and not include_wdl:
                continue
            if href.endswith(".rtbz") and not include_dtz:
                continue
            file_url = urllib.parse.urljoin(folder_url, href)
            index_files.append((file_url, href))

    filtered = filter_files(index_files, pieces)

    if args.max_files and args.max_files > 0:
        filtered = filtered[: args.max_files]

    if args.dry_run:
        print(f"Found {len(filtered)} files:")
        for _, name in filtered:
            print(f"  {name}")
        return 0

    for url, name in filtered:
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
