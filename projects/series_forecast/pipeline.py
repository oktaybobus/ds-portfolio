"""Series catalogue and loading for the forecasting project.

Two series, one code path. Each :class:`SeriesSpec` says where the data is, how
to turn its date column into a real index, how long a season is, and how far
ahead to forecast; everything else comes from :mod:`dsjourney.forecasting`.

The source notebooks fitted SARIMAX to the whole series and then predicted past
its end. That is not evaluation - there is nothing to compare the forecast to.
Here the last ``horizon`` observations are always withheld, and every model is
reported next to a naive "tomorrow equals today" baseline it has to beat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from dsjourney import forecasting
from dsjourney.config import load_project_config
from dsjourney.datasets import DatasetNotFoundError
from dsjourney.paths import project_data_dir

CONFIG = load_project_config("series_forecast")


@dataclass(frozen=True)
class SeriesSpec:
    """How to read and forecast one univariate series."""

    key: str
    title: str
    file: str
    date_column: str
    value_column: str
    period: int
    horizon: int
    frequency: str | None = None
    quarterly_labels: bool = False
    units: str = ""


SERIES: dict[str, SeriesSpec] = {
    "delhi_temperature": SeriesSpec(
        key="delhi_temperature",
        title="New Delhi daily mean temperature",
        file="DailyDelhiClimateTrain.csv",
        date_column="date",
        value_column="meantemp",
        period=365,
        horizon=60,
        frequency="D",
        units="degrees C",
    ),
    "adidas_revenue": SeriesSpec(
        key="adidas_revenue",
        title="Adidas quarterly revenue",
        file="adidas-quarterly-sales.csv",
        date_column="Time Period",
        value_column="Revenue",
        period=4,
        horizon=8,
        quarterly_labels=True,
        units="EUR millions",
    ),
}

DEFAULT_SERIES = "delhi_temperature"


def spec_for(series: str) -> SeriesSpec:
    """Look up a series spec by key."""
    try:
        return SERIES[series]
    except KeyError as error:
        raise KeyError(f"unknown series {series!r}; available: {sorted(SERIES)}") from error


def load_raw(series: str = DEFAULT_SERIES) -> pd.DataFrame:
    """Read one series' source file."""
    spec = spec_for(series)
    path = project_data_dir(CONFIG.name) / spec.file
    if not path.is_file():
        raise DatasetNotFoundError(
            f"series '{series}' not found at {path}. "
            f"Run: uv run python scripts/fetch_assets.py --project {CONFIG.name}"
        )
    return pd.read_csv(path)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the frame unchanged.

    Forecasting works on an indexed series rather than a feature matrix; use
    :func:`build_series` instead. This exists so the generic CLI commands still
    have something to call.
    """
    return frame


def build_series(frame: pd.DataFrame, spec: SeriesSpec) -> pd.Series:
    """Turn a raw frame into a sorted, date-indexed series.

    Quarter labels such as ``"2000Q1"`` are not parseable as dates by
    ``pd.to_datetime``; they are converted through a PeriodIndex first.
    """
    if spec.quarterly_labels:
        indexed = frame.assign(_when=forecasting.parse_quarter_index(frame, spec.date_column))
        return forecasting.load_series(indexed, date_column="_when", value_column=spec.value_column)
    return forecasting.load_series(
        frame, date_column=spec.date_column, value_column=spec.value_column, freq=spec.frequency
    )


def prepare_input(payload: dict[str, Any]) -> pd.DataFrame:
    """Not applicable: a forecast extends a series rather than scoring a record."""
    raise NotImplementedError(
        "series_forecast extends a series, it does not score records. "
        "Use: python projects/series_forecast/train.py --series <key>"
    )
