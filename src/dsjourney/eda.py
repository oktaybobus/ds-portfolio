"""Exploratory data analysis.

Roughly the same twenty lines - ``head``, ``info``, ``isnull().sum()``,
``value_counts``, ``corr`` - opened almost every notebook in the course. They are
collected here as functions that return DataFrames instead of printing, so the
same code serves a notebook, a training script and a unit test.
"""

from __future__ import annotations

import pandas as pd

_HIGH_CARDINALITY_THRESHOLD = 30


def overview(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one row per column: dtype, missing counts, cardinality, sample value.

    Replaces the ``df.info()`` / ``df.isnull().sum()`` / ``df.nunique()`` trio
    with a single table that can be sorted, filtered and asserted on.
    """
    total = len(frame)
    rows = []
    for column in frame.columns:
        series = frame[column]
        missing = int(series.isna().sum())
        non_null = series.dropna()
        rows.append(
            {
                "column": str(column),
                "dtype": str(series.dtype),
                "missing": missing,
                "missing_pct": round(100 * missing / total, 2) if total else 0.0,
                "unique": int(series.nunique(dropna=True)),
                "sample": non_null.iloc[0] if len(non_null) else None,
            }
        )
    return pd.DataFrame(rows).set_index("column")


def missing_report(frame: pd.DataFrame, *, only_missing: bool = True) -> pd.DataFrame:
    """Return columns ranked by how much data they are missing."""
    total = len(frame)
    counts = frame.isna().sum().sort_values(ascending=False)
    report = pd.DataFrame(
        {
            "missing": counts.astype(int),
            "missing_pct": (100 * counts / total).round(2) if total else 0.0,
        }
    )
    return report[report["missing"] > 0] if only_missing else report


def correlation_with_target(
    frame: pd.DataFrame, target: str, *, absolute: bool = True
) -> pd.Series:
    """Return every numeric column's correlation with the target, strongest first.

    The course rule of thumb was to keep features with ``0.20 < |r| < 0.90`` -
    see :func:`suggest_feature_filter`, which applies exactly that band.
    """
    if target not in frame.columns:
        raise KeyError(f"target column {target!r} is not in the frame")

    correlations = frame.corr(numeric_only=True)[target].drop(labels=[target], errors="ignore")
    ordered = correlations.abs() if absolute else correlations
    return ordered.sort_values(ascending=False)


def suggest_feature_filter(
    frame: pd.DataFrame,
    target: str,
    *,
    lower: float = 0.20,
    upper: float = 0.90,
) -> dict[str, list[str]]:
    """Split numeric features into keep / too-weak / too-collinear buckets.

    This encodes the "golden rule" taught in week 3: below ``lower`` a feature
    contributes too little, above ``upper`` it is the target restated in another
    unit and will leak.
    """
    correlations = correlation_with_target(frame, target, absolute=True)
    return {
        "keep": [
            str(c) for c in correlations[(correlations >= lower) & (correlations <= upper)].index
        ],
        "too_weak": [str(c) for c in correlations[correlations < lower].index],
        "too_collinear": [str(c) for c in correlations[correlations > upper].index],
    }


def categorical_summary(
    frame: pd.DataFrame, *, max_unique: int = _HIGH_CARDINALITY_THRESHOLD
) -> pd.DataFrame:
    """Return value counts for every low-cardinality object/category column."""
    frames = []
    for column in frame.select_dtypes(include=["object", "category", "bool"]).columns:
        counts = frame[column].value_counts(dropna=False)
        if len(counts) > max_unique:
            continue
        part = counts.rename("count").to_frame()
        part["share_pct"] = (100 * part["count"] / len(frame)).round(2)
        part.insert(0, "column", str(column))
        part.index.name = "value"
        frames.append(part.reset_index())
    if not frames:
        return pd.DataFrame(columns=["column", "value", "count", "share_pct"])
    return pd.concat(frames, ignore_index=True)


def rare_categories(series: pd.Series, *, min_count: int = 10) -> list[str]:
    """Return the category labels that occur fewer than ``min_count`` times.

    Feeds :func:`dsjourney.preprocess.group_rare_categories`, which is how the
    notebooks collapsed one-off laptop brands into an ``Others`` bucket.
    """
    counts = series.value_counts(dropna=True)
    return [str(label) for label in counts[counts < min_count].index]
