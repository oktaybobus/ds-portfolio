#!/usr/bin/env python
"""Fetch the datasets declared in assets.yaml into data/raw/<project>/.

Resolution order per asset: already on disk, committed in the repo, copied from
the original course tree if this is the author's machine, otherwise downloaded
from the Hugging Face Hub. That ordering means the script is a no-op for anyone
who already has the data and a single command for anyone who does not.

Usage:
    python scripts/fetch_assets.py --all
    python scripts/fetch_assets.py --project laptop_price
    python scripts/fetch_assets.py --list
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
COURSE_ROOT = REPO_ROOT.parent  # the original "MY AI JOURNEY" tree
ASSETS_FILE = REPO_ROOT / "assets.yaml"
RAW_DIR = REPO_ROOT / "data" / "raw"


@dataclass(frozen=True)
class Asset:
    """One declared dataset."""

    key: str
    project: str
    file: str
    size_mb: float
    committed_in_repo: bool
    local_source: str | None
    hf_repo: str | None
    hf_path: str | None
    description: str

    @property
    def destination(self) -> Path:
        return RAW_DIR / self.project / self.file


def load_assets() -> list[Asset]:
    """Parse assets.yaml into Asset records."""
    document = yaml.safe_load(ASSETS_FILE.read_text(encoding="utf-8")) or {}
    return [
        Asset(
            key=key,
            project=str(entry["project"]),
            file=str(entry["file"]),
            size_mb=float(entry.get("size_mb", 0)),
            committed_in_repo=bool(entry.get("committed_in_repo", False)),
            local_source=entry.get("local_source"),
            hf_repo=entry.get("hf_repo"),
            hf_path=entry.get("hf_path"),
            description=str(entry.get("description", "")),
        )
        for key, entry in (document.get("assets") or {}).items()
    ]


def fetch(asset: Asset, *, force: bool = False) -> str:
    """Make one asset available on disk and report how it got there."""
    destination = asset.destination
    if destination.is_file() and not force:
        return "present"

    destination.parent.mkdir(parents=True, exist_ok=True)

    if asset.local_source:
        source = COURSE_ROOT / asset.local_source
        if source.is_file():
            shutil.copy2(source, destination)
            return f"copied from {source.parent.name}/"

    if asset.hf_repo and asset.hf_path:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            return "MISSING (install the 'hub' extra to download from Hugging Face)"
        try:
            downloaded = hf_hub_download(
                repo_id=asset.hf_repo, filename=asset.hf_path, repo_type="dataset"
            )
        except Exception as error:
            return f"FAILED ({type(error).__name__}: {error})"
        shutil.copy2(downloaded, destination)
        return f"downloaded from {asset.hf_repo}"

    return "MISSING (no local source and no Hub location declared)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--all", action="store_true", help="fetch every declared asset")
    parser.add_argument("--project", help="fetch only the assets of one project")
    parser.add_argument(
        "--list", action="store_true", help="show what is declared and what is present"
    )
    parser.add_argument("--force", action="store_true", help="re-fetch even if the file exists")
    args = parser.parse_args(argv)

    assets = load_assets()
    if args.project:
        assets = [a for a in assets if a.project == args.project]
        if not assets:
            print(f"no assets declared for project {args.project!r}", file=sys.stderr)
            return 2

    if args.list or not (args.all or args.project):
        print(f"{'asset':24} {'project':20} {'size':>8}  status")
        for asset in load_assets():
            status = "present" if asset.destination.is_file() else "missing"
            print(f"{asset.key:24} {asset.project:20} {asset.size_mb:>6.1f}MB  {status}")
        return 0

    failures = 0
    for asset in assets:
        result = fetch(asset, force=args.force)
        if result.startswith(("MISSING", "FAILED")):
            failures += 1
        print(f"{asset.key:24} {result}")

    if failures:
        print(f"\n{failures} asset(s) could not be fetched", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
