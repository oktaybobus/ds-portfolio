"""Dataset loading.

The course notebooks each opened their data a slightly different way - bare
``pd.read_csv`` here, a ``sqlite3`` connection there, a Colab download somewhere
else. This module funnels all of that through one function so every project
reports the same clear error when its data has not been fetched yet.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from dsjourney.config import DatasetConfig, ProjectConfig
from dsjourney.paths import project_data_dir


class DatasetNotFoundError(FileNotFoundError):
    """Raised when a project's data file is missing from data/raw/."""


def dataset_path(config: ProjectConfig) -> Path:
    """Return the expected on-disk location of a project's dataset."""
    if config.dataset is None:
        raise ValueError(f"project {config.name!r} has no dataset; it drives an environment")
    return project_data_dir(config.name) / config.dataset.file


def load_dataset(config: ProjectConfig) -> pd.DataFrame:
    """Read a project's dataset into a DataFrame.

    Raises:
        DatasetNotFoundError: when the file is absent, with the fetch command to run.
    """
    path = dataset_path(config)
    assert config.dataset is not None  # dataset_path raises otherwise
    if not path.is_file():
        raise DatasetNotFoundError(
            f"dataset '{config.dataset.id}' not found at {path}. "
            f"Run: uv run python scripts/fetch_assets.py --project {config.name}"
        )
    return read_tabular(path, config.dataset)


def read_tabular(path: Path, dataset: DatasetConfig) -> pd.DataFrame:
    """Read a tabular file according to its declared format."""
    options = dict(dataset.read_options)

    if dataset.format == "csv":
        return pd.read_csv(path, **options)
    if dataset.format == "excel":
        return pd.read_excel(path, **options)
    if dataset.format == "text":
        # One row per line, so a line-oriented source (an adjacency list, a
        # book) still answers `dsj eda-report`. The encoding is explicit and
        # strict: these files are not all UTF-8, and a silent replacement
        # character is how the source notebook lost punctuation.
        encoding = str(options.pop("encoding", "utf-8"))
        text = path.read_text(encoding=encoding, errors="strict")
        return pd.DataFrame({"line": text.splitlines()})
    if dataset.format == "sqlite":
        if not dataset.table:
            raise ValueError(f"dataset '{dataset.id}' is sqlite but declares no table")
        with sqlite3.connect(path) as conn:
            return pd.read_sql_query(f"SELECT * FROM {dataset.table}", conn, **options)
    raise ValueError(f"cannot read format {dataset.format!r} as a table")


def describe_dataset(frame: pd.DataFrame) -> dict[str, object]:
    """Return a small, JSON-serialisable fingerprint of a DataFrame.

    Stored alongside every trained model so a metrics file can later be traced
    back to the exact shape of data that produced it.
    """
    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "column_names": list(frame.columns.astype(str)),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "memory_mb": round(frame.memory_usage(deep=True).sum() / 1_048_576, 3),
    }
