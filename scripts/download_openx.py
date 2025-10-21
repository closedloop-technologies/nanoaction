#!/usr/bin/env python3
"""
Download subsets of the Open-X Embodiment dataset that have been mirrored on Hugging Face
in the LeRobot-compatible format.

The script relies on `huggingface_hub.snapshot_download` so downloads are cached and
resumable. By default it only materialises lightweight symlinks that point into your
Hugging Face cache (use `--copy` to force physical copies).

Example usage:

  # Inspect the repository structure without downloading anything
  python scripts/download_openx.py --list

  # Download a single source dataset (e.g. ALOHA) into data/openx/
  python scripts/download_openx.py --datasets aloha

  # Download multiple datasets into a custom directory
  python scripts/download_openx.py --datasets aloha bridge --local-dir ~/datasets/openx
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Set

try:
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.hf_api import RepoFile, RepoFolder
except ImportError as exc:  # pragma: no cover - dependency message
    raise SystemExit(
        "huggingface_hub is required. Install it with `pip install huggingface_hub`."
    ) from exc


def humanize_bytes(num_bytes: int) -> str:
    """Return a human readable byte count."""
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} {units[-1]}"


def collect_repo_tree(
    api: HfApi, repo_id: str, *, repo_type: str, revision: Optional[str]
) -> Sequence[Any]:
    """Fetch a recursive tree listing for the target repository."""
    return api.list_repo_tree(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        recursive=True,
    )


def summarise_top_level(files: Sequence[Any]) -> tuple[Counter, Counter, Set[str]]:
    """
    Produce counts, size totals, and the set of top-level directories present.

    Returns:
        (file_counter, size_counter, top_level_dirs)
    """
    file_counter: Counter[str] = Counter()
    size_counter: Counter[str] = Counter()
    top_level_dirs: Set[str] = set()

    for entry in files:
        if not isinstance(entry, RepoFile):
            continue
        parts = entry.path.split("/")
        if len(parts) == 1:
            top_level_dirs.add(parts[0])
            file_counter[parts[0]] += 1
            size_counter[parts[0]] += entry.size or 0
        else:
            top = parts[0]
            top_level_dirs.add(top)
            file_counter[top] += 1
            size_counter[top] += entry.size or 0

    return file_counter, size_counter, top_level_dirs


def build_allow_patterns(
    *,
    datasets: Optional[Iterable[str]],
    metadata_files: Iterable[str],
    include_patterns: Optional[Iterable[str]],
    include_metadata: bool,
) -> Optional[Sequence[str]]:
    """
    Construct the allow_patterns argument for snapshot_download.

    When no datasets or patterns are provided we return None so that the Hub
    library mirrors the entire repository.
    """
    patterns: Set[str] = set()

    if datasets:
        for dataset in datasets:
            dataset = dataset.strip("/")
            if not dataset:
                continue
            patterns.add(f"{dataset}/*")
            # Some datasets place auxiliary assets alongside the folder (e.g. tarballs)
            patterns.add(f"{dataset}*")

    if include_metadata:
        patterns.update(metadata_files)

    if include_patterns:
        for pattern in include_patterns:
            if pattern:
                patterns.add(pattern)

    if not patterns:
        return None
    return sorted(patterns)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Open-X Embodiment data that has been curated in LeRobot format on Hugging Face."
    )
    parser.add_argument(
        "--repo-id",
        default="jxu124/OpenX-Embodiment",
        help="Hugging Face dataset repository ID to mirror (default: jxu124/OpenX-Embodiment).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional revision (branch, tag, or commit SHA). Defaults to the repo's main branch.",
    )
    parser.add_argument(
        "--local-dir",
        default="data/openx",
        help="Local directory where the dataset snapshot will be materialised (default: data/openx).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional path to a Hugging Face cache directory. Defaults to the global cache.",
    )
    parser.add_argument(
        "--datasets",
        "-d",
        nargs="+",
        help="One or more top-level dataset folders to download (e.g. aloha bridge rt1). "
        "Use --list to discover available options.",
    )
    parser.add_argument(
        "--include-patterns",
        "-i",
        nargs="+",
        help="Additional glob-style patterns passed to snapshot_download's allow_patterns parameter.",
    )
    parser.add_argument(
        "--ignore-patterns",
        "-x",
        nargs="+",
        help="Glob-style patterns passed to snapshot_download's ignore_patterns parameter.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face access token (falls back to the HF_TOKEN environment variable if unset).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Disable symlinks so files are fully copied into --local-dir instead of referencing the cache.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of matching files even if they are already present in the cache.",
    )
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Do not materialise top-level metadata files (README, dataset_infos.json, etc.).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only list the top-level assets available in the repository without downloading.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    api = HfApi()
    repo_type = "dataset"

    print(f"Inspecting repository: {args.repo_id} (type={repo_type}, revision={args.revision or 'default'})")
    repo_tree = collect_repo_tree(api, args.repo_id, repo_type=repo_type, revision=args.revision)

    file_counter, size_counter, top_level_dirs = summarise_top_level(repo_tree)
    metadata_files = sorted(
        entry.path
        for entry in repo_tree
        if isinstance(entry, RepoFile) and "/" not in entry.path
    )

    if args.list:
        print("\nTop-level assets:")
        for name in sorted(top_level_dirs):
            count = file_counter.get(name, 0)
            size = size_counter.get(name, 0)
            size_str = humanize_bytes(size) if size else "n/a"
            print(f"  {name:<30} files={count:6d} size≈{size_str}")

        if metadata_files:
            print("\nTop-level metadata files:")
            for meta in metadata_files:
                print(f"  {meta}")
        return 0

    if args.datasets:
        missing = sorted(set(args.datasets) - top_level_dirs)
        if missing:
            print(
                "The following requested datasets were not found in the repository:\n"
                + "\n".join(f"  - {name}" for name in missing),
                file=sys.stderr,
            )
            return 1

    allow_patterns = build_allow_patterns(
        datasets=args.datasets,
        metadata_files=metadata_files,
        include_patterns=args.include_patterns,
        include_metadata=not args.skip_metadata,
    )

    if allow_patterns:
        print("\nAllow patterns:")
        for pattern in allow_patterns:
            print(f"  - {pattern}")
    if args.ignore_patterns:
        print("\nIgnore patterns:")
        for pattern in args.ignore_patterns:
            print(f"  - {pattern}")

    local_dir = Path(args.local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else None
    if cache_dir:
        cache_dir = cache_dir.resolve()

    print(f"\nDownloading to: {local_dir}")
    if cache_dir:
        print(f"Using cache directory: {cache_dir}")
    print("Materialisation mode:", "copy" if args.copy else "symlink")

    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type=repo_type,
        revision=args.revision,
        token=args.token,
        local_dir=str(local_dir),
        local_dir_use_symlinks=not args.copy,
        allow_patterns=allow_patterns,
        ignore_patterns=args.ignore_patterns,
        force_download=args.force_download,
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    print(f"\nDownload complete. Snapshot materialised at: {snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
