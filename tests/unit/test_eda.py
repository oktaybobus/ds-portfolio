"""Unit tests for the exploratory-analysis helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from dsjourney import eda


def test_overview_reports_one_row_per_column(messy_frame: pd.DataFrame) -> None:
    result = eda.overview(messy_frame)
    assert len(result) == messy_frame.shape[1]
    assert set(result.columns) == {"dtype", "missing", "missing_pct", "unique", "sample"}


def test_overview_counts_missing_values(messy_frame: pd.DataFrame) -> None:
    result = eda.overview(messy_frame)
    assert result.loc["score", "missing"] == 2
    assert result.loc["score", "missing_pct"] == pytest.approx(33.33, abs=0.01)


def test_overview_handles_an_empty_frame() -> None:
    result = eda.overview(pd.DataFrame({"a": pd.Series(dtype=float)}))
    assert result.loc["a", "missing_pct"] == 0.0
    assert result.loc["a", "sample"] is None


def test_missing_report_lists_only_incomplete_columns(messy_frame: pd.DataFrame) -> None:
    result = eda.missing_report(messy_frame)
    assert list(result.index) == ["score"]


def test_missing_report_can_include_complete_columns(messy_frame: pd.DataFrame) -> None:
    result = eda.missing_report(messy_frame, only_missing=False)
    assert len(result) == messy_frame.shape[1]


def test_correlation_with_target_excludes_the_target(regression_frame: pd.DataFrame) -> None:
    result = eda.correlation_with_target(regression_frame, "target")
    assert "target" not in result.index
    assert result.index[0] == "feature_a"


def test_correlation_with_target_rejects_a_missing_column(regression_frame: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="not in the frame"):
        eda.correlation_with_target(regression_frame, "absent")


def test_suggest_feature_filter_applies_the_course_rule(regression_frame: pd.DataFrame) -> None:
    """Signal features are kept, pure noise is flagged as too weak."""
    buckets = eda.suggest_feature_filter(regression_frame, "target")
    assert "feature_a" in buckets["keep"]
    assert "feature_b" in buckets["keep"]
    assert "noise_only" in buckets["too_weak"]


def test_suggest_feature_filter_flags_a_leaking_copy(regression_frame: pd.DataFrame) -> None:
    leaked = regression_frame.assign(target_in_disguise=regression_frame["target"] * 1.001)
    buckets = eda.suggest_feature_filter(leaked, "target")
    assert "target_in_disguise" in buckets["too_collinear"]


def test_categorical_summary_skips_high_cardinality_columns(messy_frame: pd.DataFrame) -> None:
    result = eda.categorical_summary(messy_frame, max_unique=3)
    assert set(result["column"]) == {"brand"}


def test_categorical_summary_returns_empty_frame_when_nothing_qualifies() -> None:
    result = eda.categorical_summary(pd.DataFrame({"a": [1, 2, 3]}))
    assert result.empty
    assert list(result.columns) == ["column", "value", "count", "share_pct"]


def test_rare_categories_finds_the_long_tail(messy_frame: pd.DataFrame) -> None:
    assert eda.rare_categories(messy_frame["brand"], min_count=2) == ["Xiaomi"]
