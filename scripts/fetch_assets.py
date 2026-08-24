#!/usr/bin/env python
"""Fetch the datasets declared in assets.yaml into data/raw/<project>/.

Resolution order per asset: already on disk, copied from the original course
tree if this is the author's machine, downloaded from the Hugging Face mirror,
or fetched from the publisher's own URL. That ordering means the script is a
no-op for anyone who already has the data and a single command for anyone who
does not.

The mirror (``OKTAYBBS/ds-portfolio-data``) is private, so downloading from it
needs `hf auth login` with an account that has access. MovieLens is not on it
at all - its licence says "the user may not redistribute the data without
separate permission" - so it carries a ``source_url`` pointing at GroupLens
instead, and arrives as the pristine 100,000-rating file rather than the
edited copy in the course tree.

Usage:
    python scripts/fetch_assets.py --all
    python scripts/fetch_assets.py --project laptop_price
    python scripts/fetch_assets.py --list
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
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
    directory: bool = False
    copy_patterns: tuple[str, ...] = ()
    source_url: str | None = None
    archive_member: str | None = None

    @property
    def destination(self) -> Path:
        """Where the asset lands.

        Directory assets are unpacked *into* data/raw/<project>/ rather than
        into a subfolder, so a project reads its files from one place whether
        they arrived as a tree or as a single download.
        """
        return RAW_DIR / self.project if self.directory else RAW_DIR / self.project / self.file


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
            directory=bool(entry.get("directory", False)),
            copy_patterns=tuple(entry.get("copy_patterns", ()) or ()),
            source_url=entry.get("source_url"),
            archive_member=entry.get("archive_member"),
        )
        for key, entry in (document.get("assets") or {}).items()
    ]


def fetch(asset: Asset, *, force: bool = False) -> str:
    """Make one asset available on disk and report how it got there."""
    destination = asset.destination
    if _present(asset) and not force:
        return "present"

    destination.parent.mkdir(parents=True, exist_ok=True)

    if asset.local_source:
        source = COURSE_ROOT / asset.local_source
        if asset.directory and source.is_dir():
            return f"copied {_copy_tree(asset, source)} file(s) from {source.name}/"
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
        if asset.directory:
            count = _unpack(Path(downloaded), destination)
            return f"downloaded and unpacked {count} file(s) from {asset.hf_repo}"
        shutil.copy2(downloaded, destination)
        return f"downloaded from {asset.hf_repo}"

    if asset.source_url:
        try:
            return _fetch_url(asset, destination)
        except Exception as error:
            return f"FAILED ({type(error).__name__}: {error})"

    return "MISSING (no local source, mirror entry or source URL declared)"


def _fetch_url(asset: Asset, destination: Path) -> str:
    """Download from the publisher, extracting one member when it is an archive."""
    assert asset.source_url is not None
    with tempfile.TemporaryDirectory() as directory:
        archive = Path(directory) / Path(asset.source_url).name
        with urllib.request.urlopen(asset.source_url, timeout=120) as response:
            archive.write_bytes(response.read())

        if not asset.archive_member:
            shutil.copy2(archive, destination)
            return f"downloaded from {_host(asset.source_url)}"

        with (
            zipfile.ZipFile(archive) as bundle,
            bundle.open(asset.archive_member) as member,
            destination.open("wb") as target,
        ):
            shutil.copyfileobj(member, target)
        return f"downloaded {asset.archive_member} from {_host(asset.source_url)}"


def _unpack(archive: Path, directory: Path) -> int:
    """Extract a zip into a directory asset's destination."""
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members = [m for m in bundle.namelist() if not m.endswith("/")]
        bundle.extractall(directory)
    return len(members)


def _host(url: str) -> str:
    """Return the hostname of a URL, for the status line."""
    from urllib.parse import urlparse

    return urlparse(url).netloc or url


def _present(asset: Asset) -> bool:
    """True when the asset is already on disk."""
    if not asset.directory:
        return asset.destination.is_file()
    return asset.destination.is_dir() and any(asset.destination.iterdir())


def _copy_tree(asset: Asset, source: Path) -> int:
    """Copy a directory asset's files, honouring copy_patterns when given."""
    asset.destination.mkdir(parents=True, exist_ok=True)
    patterns = asset.copy_patterns or ("*",)

    copied = 0
    for pattern in patterns:
        for path in sorted(source.glob(pattern)):
            if path.is_file():
                shutil.copy2(path, asset.destination / path.name)
                copied += 1
    return copied


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
            status = "present" if _present(asset) else "missing"
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
