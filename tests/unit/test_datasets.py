"""Unit tests for dataset loading."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from dsjourney.config import DatasetConfig, ProjectConfig
from dsjourney.datasets import DatasetNotFoundError, describe_dataset, load_dataset, read_tabular

BASE = {
    "name": "demo",
    "title": "Demo",
    "task": "regression",
    "target": "y",
    "model": {"estimator": "linear"},
}


def test_load_dataset_names_the_fetch_command_when_absent() -> None:
    config = ProjectConfig.model_validate(
        {**BASE, "dataset": {"id": "nothing", "file": "nothing.csv"}}
    )
    with pytest.raises(DatasetNotFoundError, match=r"fetch_assets\.py"):
        load_dataset(config)


def test_read_tabular_reads_csv(tmp_path: Path) -> None:
    path = tmp_path / "d.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(path, index=False)
    frame = read_tabular(path, DatasetConfig(id="d", file="d.csv", format="csv"))
    assert frame["a"].tolist() == [1, 2]


def test_read_tabular_reads_sqlite(tmp_path: Path) -> None:
    path = tmp_path / "d.db"
    with sqlite3.connect(path) as conn:
        pd.DataFrame({"a": [3, 4]}).to_sql("things", conn, index=False)
    frame = read_tabular(path, DatasetConfig(id="d", file="d.db", format="sqlite", table="things"))
    assert frame["a"].tolist() == [3, 4]


def test_read_tabular_requires_a_table_for_sqlite(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="declares no table"):
        read_tabular(tmp_path / "d.db", DatasetConfig(id="d", file="d.db", format="sqlite"))


def test_read_tabular_rejects_a_non_tabular_format(tmp_path: Path) -> None:
    config = DatasetConfig(id="d", file="d", format="kagglehub")
    with pytest.raises(ValueError, match="cannot read format"):
        read_tabular(tmp_path / "d", config)


def test_describe_dataset_fingerprints_a_frame(messy_frame: pd.DataFrame) -> None:
    summary = describe_dataset(messy_frame)
    assert summary["rows"] == 6
    assert summary["columns"] == 8
    assert summary["missing_cells"] == 2
    assert summary["duplicate_rows"] == 0
