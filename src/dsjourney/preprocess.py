"""Feature preparation.

Every function here returns a new DataFrame; nothing is modified in place. The
notebooks leaned on ``inplace=True``, which made cells order-dependent and
impossible to re-run - the same transformation written as a pure function can be
composed, tested and applied to unseen rows at prediction time.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dsjourney.config import SplitConfig

_NON_NUMERIC = r"[^\d.\-]"


NumericDtype = Literal["int", "float"]


def _cast(series: pd.Series[Any], dtype: NumericDtype) -> pd.Series[Any]:
    """Cast a parsed numeric Series, keeping float whenever a value is missing.

    Branching on the literal rather than forwarding the string lets the pandas
    ``astype`` overloads resolve statically, so the module type-checks strictly
    with no suppressions - and an integer column can never be asked to hold NaN.
    """
    if series.isna().any():
        return series.astype("float64")
    return series.astype("int64") if dtype == "int" else series.astype("float64")


def _as_text(frame: pd.DataFrame, column: str) -> pd.Series[str]:
    """Return a column as a string Series.

    pandas-stubs cannot always narrow ``frame[column]`` to a string dtype, so
    every ``.str`` accessor call downstream would need its own type ignore.
    Funnelling them through one annotated helper keeps that noise in a single
    place instead of scattered across the module.
    """
    return frame[column].astype(str)  # type: ignore[return-value]


def drop_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return a copy without the named columns, ignoring ones already absent."""
    return frame.drop(columns=[c for c in columns if c in frame.columns])


def strip_unit(
    frame: pd.DataFrame, column: str, unit: str, *, dtype: NumericDtype = "float"
) -> pd.DataFrame:
    """Turn a column such as ``"16GB"`` or ``"1.37kg"`` into a numeric column.

    Args:
        unit: The literal suffix to remove, e.g. ``"GB"`` or ``"kg"``.
        dtype: ``"int"`` or ``"float"`` for the resulting column.
    """
    cleaned = _as_text(frame, column).str.replace(unit, "", regex=False).str.strip()
    return frame.assign(**{column: _cast(pd.to_numeric(cleaned, errors="coerce"), dtype)})


def to_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Strip currency symbols, separators and stray characters, then coerce to numbers.

    Values that still cannot be parsed become ``NaN`` rather than raising, so the
    imputation step downstream stays in charge of what to do with them.
    """
    updates = {}
    for column in columns:
        if column not in frame.columns:
            continue
        cleaned = _as_text(frame, column).str.replace(_NON_NUMERIC, "", regex=True)
        updates[column] = pd.to_numeric(cleaned.replace("", np.nan), errors="coerce")
    return frame.assign(**updates)


def group_rare_categories(
    frame: pd.DataFrame,
    column: str,
    *,
    min_count: int = 10,
    other_label: str = "Others",
) -> pd.DataFrame:
    """Collapse infrequent labels of a categorical column into a single bucket.

    Keeps one-hot encoding from exploding into dozens of columns that each carry
    a handful of rows.
    """
    counts = frame[column].value_counts(dropna=True)
    rare = counts[counts < min_count].index
    return frame.assign(**{column: frame[column].where(~frame[column].isin(rare), other_label)})


def binary_flag(frame: pd.DataFrame, source: str, keyword: str, new_column: str) -> pd.DataFrame:
    """Add a 0/1 column marking rows whose ``source`` text contains ``keyword``."""
    flag = _as_text(frame, source).str.contains(keyword, case=False, na=False)
    return frame.assign(**{new_column: flag.astype(int)})


def extract_pattern(
    frame: pd.DataFrame,
    source: str,
    pattern: str,
    new_columns: Sequence[str],
    *,
    dtype: NumericDtype = "float",
) -> pd.DataFrame:
    """Pull regex capture groups out of a text column into new typed columns.

    Example:
        ``extract_pattern(df, "ScreenResolution", r"(\\d+)x(\\d+)", ["x_res", "y_res"])``
    """
    extracted = _as_text(frame, source).str.extract(pattern)
    if extracted.shape[1] != len(new_columns):
        raise ValueError(
            f"pattern captures {extracted.shape[1]} group(s) but {len(new_columns)} name(s) were given"
        )
    updates = {
        name: _cast(pd.to_numeric(extracted[position], errors="coerce"), dtype)
        for position, name in enumerate(new_columns)
    }
    return frame.assign(**updates)


def map_values(frame: pd.DataFrame, mapping: Mapping[str, Mapping[object, object]]) -> pd.DataFrame:
    """Apply per-column value replacements, e.g. ``{"Term": {"Short Term": 0}}``.

    ``Series.replace`` on a string column that maps to numbers leaves the result
    as ``object`` dtype. scikit-learn then reports the label type as "unknown"
    and refuses to fit, so the replaced columns are re-inferred back to a real
    numeric dtype here rather than in every caller.
    """
    updates = {
        column: frame[column].replace(dict(replacements)).infer_objects()  # type: ignore[arg-type]
        for column, replacements in mapping.items()
        if column in frame.columns
    }
    return frame.assign(**updates)


def flag_and_fill_missing(
    frame: pd.DataFrame, column: str, *, fill_value: float = -1.0
) -> pd.DataFrame:
    """Record missingness as its own feature before filling the gap.

    For fields like "months since last delinquent", the absence of a value is
    itself informative - dropping that signal loses real predictive power.
    """
    indicator = frame[column].isna().astype(int)
    return frame.assign(
        **{f"{column}_missing": indicator, column: frame[column].fillna(fill_value)}
    )


def impute_numeric(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    strategy: str = "median",
    add_indicator: bool = True,
) -> pd.DataFrame:
    """Fill missing numeric values, optionally keeping a flag for what was filled.

    The indicator matters more than the fill value on financial data: a missing
    credit score is usually a thin-file applicant, which is itself predictive.
    Imputing without the flag throws that signal away.

    Args:
        strategy: ``"median"``, ``"mean"`` or ``"zero"``.
    """
    updates: dict[str, pd.Series] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        series = frame[column]
        if not series.isna().any():
            continue
        if add_indicator:
            updates[f"{column}_missing"] = series.isna().astype(int)
        if strategy == "median":
            fill: float = float(series.median())
        elif strategy == "mean":
            fill = float(series.mean())
        elif strategy == "zero":
            fill = 0.0
        else:
            raise ValueError(f"unknown imputation strategy {strategy!r}")
        updates[column] = series.fillna(fill)
    return frame.assign(**updates)


def drop_duplicate_rows(
    frame: pd.DataFrame, *, subset: Sequence[str] | None = None
) -> pd.DataFrame:
    """Return a copy with duplicate rows removed and the index reset."""
    return frame.drop_duplicates(subset=list(subset) if subset else None).reset_index(drop=True)


def safe_ratio(
    frame: pd.DataFrame, numerator: str, denominator: str, new_column: str
) -> pd.DataFrame:
    """Add a ratio column, treating a zero denominator as missing rather than infinite."""
    denominators = frame[denominator].replace(0, np.nan)
    return frame.assign(**{new_column: frame[numerator] / denominators})


EARTH_RADIUS_KM = 6371.0


def add_calendar_features(
    frame: pd.DataFrame, column: str, *, prefix: str = "", drop: bool = False
) -> pd.DataFrame:
    """Derive hour, day of week, month and a weekend flag from a datetime column."""
    when = pd.to_datetime(frame[column], errors="coerce")
    derived: dict[str, pd.Series[Any]] = {
        f"{prefix}hour": when.dt.hour,
        f"{prefix}day_of_week": when.dt.dayofweek,
        f"{prefix}month": when.dt.month,
        f"{prefix}is_weekend": (when.dt.dayofweek >= 5).astype(int),
    }
    updated = frame.assign(**derived)
    return drop_columns(updated, [column]) if drop else updated


def add_cyclical(
    frame: pd.DataFrame, column: str, *, period: int, drop: bool = False
) -> pd.DataFrame:
    """Encode a cyclical integer column as a sine/cosine pair.

    Hour 23 and hour 0 are one hour apart, but as raw numbers they are 23 apart
    and a linear model reads midnight as the opposite extreme of 11pm. Projecting
    onto a circle makes the distance between them what it actually is. Tree
    models can learn the wrap from splits alone, but only by spending depth on
    it, so the encoding helps them too.

    Args:
        period: The length of the cycle - 24 for hours, 7 for weekdays, 12 for
            months.
    """
    angle = 2 * np.pi * frame[column] / period
    updated = frame.assign(**{f"{column}_sin": np.sin(angle), f"{column}_cos": np.cos(angle)})
    return drop_columns(updated, [column]) if drop else updated


def haversine_km(
    lat1: pd.Series | float,
    lon1: pd.Series | float,
    lat2: pd.Series | float,
    lon2: pd.Series | float,
) -> pd.Series | float:
    """Great-circle distance in kilometres between two coordinate pairs.

    Straight-line distance in degrees is not a distance: a degree of longitude
    is 111 km at the equator and 0 at the poles, so a Euclidean gap between
    lat/lon pairs distorts with latitude.
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = phi2 - phi1
    delta_lambda = np.radians(np.asarray(lon2) - np.asarray(lon1))

    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def one_hot(
    frame: pd.DataFrame, columns: Iterable[str], *, drop_first: bool = True
) -> pd.DataFrame:
    """One-hot encode the given categorical columns."""
    present = [c for c in columns if c in frame.columns]
    if not present:
        return frame.copy()
    return pd.get_dummies(frame, columns=present, drop_first=drop_first, dtype=int)


def log_transform_target(frame: pd.DataFrame, target: str, new_name: str) -> pd.DataFrame:
    """Replace a skewed target with its ``log1p``, as done for laptop prices.

    Use :func:`inverse_log_transform` to bring predictions back to the original
    scale before showing them to a user.
    """
    return frame.assign(**{new_name: np.log1p(frame[target])}).drop(columns=[target])


def inverse_log_transform(values: np.ndarray | pd.Series) -> np.ndarray:
    """Undo :func:`log_transform_target` on model output."""
    return np.expm1(np.asarray(values, dtype=float))


@dataclass(frozen=True)
class SplitResult:
    """The four frames produced by a train/test split, plus the fitted scaler."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    scaler: StandardScaler | None = None

    @property
    def feature_names(self) -> list[str]:
        """Column order the model was trained on - must be replayed at predict time."""
        return [str(c) for c in self.x_train.columns]


def split_and_scale(
    frame: pd.DataFrame,
    target: str,
    config: SplitConfig,
    *,
    scale_columns: Sequence[str] = (),
) -> SplitResult:
    """Split into train/test and standardise the chosen numeric columns.

    The scaler is fitted on the training half only and merely applied to the test
    half. Several notebooks fitted it on the full frame first, which quietly
    leaks test statistics into training and inflates the reported score.
    """
    if target not in frame.columns:
        raise KeyError(f"target column {target!r} is not in the frame")

    features = frame.drop(columns=[target])
    labels = frame[target]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=labels if config.stratify else None,
    )

    columns = [c for c in scale_columns if c in features.columns]
    if not columns:
        return SplitResult(x_train, x_test, y_train, y_test, None)

    scaler = StandardScaler()
    x_train_scaled = x_train.assign(
        **dict(zip(columns, scaler.fit_transform(x_train[columns]).T, strict=True))
    )
    x_test_scaled = x_test.assign(
        **dict(zip(columns, scaler.transform(x_test[columns]).T, strict=True))
    )
    return SplitResult(x_train_scaled, x_test_scaled, y_train, y_test, scaler)


def align_to_training_columns(frame: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    """Reshape an inference frame to the exact columns and order used in training.

    Missing one-hot columns are filled with zeros and unexpected extras dropped,
    which is what makes a saved model usable against a single user-entered row.
    """
    aligned = frame.reindex(columns=list(feature_names), fill_value=0)
    return aligned.fillna(0)
