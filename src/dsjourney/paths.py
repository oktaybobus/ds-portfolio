"""Filesystem layout of the monorepo.

Every path is derived from the installed package location so the helpers work
identically from a source checkout, an editable install, and inside the Docker
image, without any caller having to pass a base directory around.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_ROOT = "DSJOURNEY_ROOT"


def _discover_repo_root() -> Path:
    """Return the monorepo root.

    Resolution order: the ``DSJOURNEY_ROOT`` environment variable (used by the
    Docker image, where the package is installed outside the source tree), then
    the first ancestor directory that contains ``pyproject.toml``, then the
    current working directory as a last resort.
    """
    override = os.environ.get(_ENV_ROOT)
    if override:
        return Path(override).expanduser().resolve()

    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd().resolve()


REPO_ROOT: Path = _discover_repo_root()
SRC_DIR: Path = REPO_ROOT / "src"
PROJECTS_DIR: Path = REPO_ROOT / "projects"
DATA_DIR: Path = REPO_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"
ARTIFACTS_DIR: Path = REPO_ROOT / "artifacts"
DOCS_DIR: Path = REPO_ROOT / "docs"


def project_dir(name: str) -> Path:
    """Return the source directory of a portfolio project."""
    return PROJECTS_DIR / name


def project_artifacts_dir(name: str, *, create: bool = False) -> Path:
    """Return the artifact directory of a project, optionally creating it."""
    path = ARTIFACTS_DIR / name
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def project_data_dir(name: str, *, create: bool = False) -> Path:
    """Return the raw-data directory reserved for a project."""
    path = RAW_DATA_DIR / name
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def available_projects() -> list[str]:
    """Return the names of every project that ships a ``config.yaml``."""
    if not PROJECTS_DIR.is_dir():
        return []
    return sorted(p.name for p in PROJECTS_DIR.iterdir() if (p / "config.yaml").is_file())
