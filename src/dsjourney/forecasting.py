"""Time-series forecasting.

The course notebooks fitted SARIMAX to an entire series and then predicted past
its end. That produces a plot but not a number: with no held-out period there is
nothing to score the forecast against, so "it looks about right" was the only
available verdict.

This module splits chronologically instead. The last ``horizon`` observations
are withheld, the model sees only what came before, and the forecast is scored
against them - which is the one arrangement that answers "how wrong is this
likely to be next quarter?"

Never reuse :func:`dsjourney.preprocess.split_and_scale` here: it shuffles, and
a shuffled split lets a model learn from next month to predict last month.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_HORIZON = 30


@dataclass(frozen=True)
class ForecastSplit:
    """A chronological train/test split of one series."""

    train: pd.Series
    test: pd.Series

    @property
    def horizon(self) -> int:
        """How many periods were withheld."""
        return len(self.test)


def load_series(
    frame: pd.DataFrame, *, date_column: str, value_column: str, freq: str | None = None
) -> pd.Series:
    """Return a sorted, date-indexed series ready to forecast.

    Args:
        freq: Optional pandas offset alias to resample onto, e.g. ``"MS"`` for
            month start. Gaps in a daily series break a seasonal model, so
            passing a frequency here is usually worth it.
    """
    dated = frame.assign(_when=pd.to_datetime(frame[date_column], errors="coerce"))
    dated = dated.dropna(subset=["_when"]).sort_values("_when")
    series = pd.Series(
        pd.to_numeric(dated[value_column], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(dated["_when"]),
        name=value_column,
    ).dropna()

    if freq:
        series = series.resample(freq).mean().interpolate()
    return series


def parse_quarter_index(frame: pd.DataFrame, column: str) -> pd.DatetimeIndex:
    """Turn labels such as ``"2000Q1"`` into quarter-start timestamps."""
    return pd.PeriodIndex(frame[column].astype(str), freq="Q").to_timestamp()


def chronological_split(series: pd.Series, *, horizon: int = DEFAULT_HORIZON) -> ForecastSplit:
    """Withhold the final ``horizon`` observations as the test period.

    Raises:
        ValueError: when the series is too short to leave a usable training set.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if len(series) <= horizon + 2:
        raise ValueError(f"series of length {len(series)} cannot spare a {horizon}-period holdout")
    return ForecastSplit(series.iloc[:-horizon], series.iloc[-horizon:])


def forecast_scores(actual: pd.Series, predicted: pd.Series | np.ndarray) -> dict[str, float]:
    """Return MAE, RMSE, MAPE and MASE for one forecast.

    ``mase`` scales MAE by the average one-step change in the holdout, so it is
    comparable across series with different units. Read it as "how many typical
    one-day moves is this forecast off by" - not as a pass/fail against a naive
    forecast. For a multi-step horizon, a value above 1 is normal and expected,
    because a 60-day-ahead forecast is a much harder problem than a one-day-ahead
    one. To ask whether a model beats doing nothing, use the ``skill_vs_naive``
    column that :func:`compare_forecasters` adds, which compares like with like.
    """
    truth = np.asarray(actual, dtype=float).ravel()
    prediction = np.asarray(predicted, dtype=float).ravel()[: len(truth)]

    errors = truth - prediction
    naive_errors = np.abs(np.diff(truth)) if len(truth) > 1 else np.array([np.nan])
    naive_scale = float(np.nanmean(naive_errors)) if len(naive_errors) else float("nan")
    mae = float(np.mean(np.abs(errors)))

    with np.errstate(divide="ignore", invalid="ignore"):
        percentage = np.abs(errors / np.where(truth == 0, np.nan, truth))

    return {
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mape": float(np.nanmean(percentage)),
        "mase": float(mae / naive_scale) if naive_scale else float("nan"),
    }


def seasonal_strength(series: pd.Series, *, period: int) -> dict[str, float]:
    """Decompose the series and report how much of it is trend and seasonality.

    Returns NaNs rather than raising when the series is too short for the
    requested period, so an exploratory call never breaks a pipeline.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    if len(series) < 2 * period:
        return {"trend_strength": float("nan"), "seasonal_strength": float("nan")}

    decomposition = seasonal_decompose(series, model="additive", period=period)
    residual = np.asarray(decomposition.resid, dtype=float)
    seasonal = np.asarray(decomposition.seasonal, dtype=float)
    trend = np.asarray(decomposition.trend, dtype=float)
    mask = ~np.isnan(residual)

    residual_variance = float(np.var(residual[mask]))
    return {
        "trend_strength": _strength(residual_variance, trend[mask] + residual[mask]),
        "seasonal_strength": _strength(residual_variance, seasonal[mask] + residual[mask]),
    }


def fit_naive(train: pd.Series, horizon: int) -> pd.Series:
    """Repeat the last observed value - the baseline every model must beat."""
    return pd.Series(np.repeat(float(train.iloc[-1]), horizon), index=_future_index(train, horizon))


def fit_seasonal_naive(train: pd.Series, horizon: int, *, period: int = 12) -> pd.Series:
    """Repeat the value from one season ago."""
    if len(train) < period:
        return fit_naive(train, horizon)
    season = np.asarray(train.iloc[-period:], dtype=float)
    values = np.resize(season, horizon)
    return pd.Series(values, index=_future_index(train, horizon))


def fit_sarima(
    train: pd.Series,
    horizon: int,
    *,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> pd.Series:
    """Fit SARIMAX on the training period and forecast ``horizon`` steps ahead."""
    import statsmodels.api as sm

    model = sm.tsa.statespace.SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    forecast = fitted.forecast(steps=horizon)
    return pd.Series(np.asarray(forecast, dtype=float), index=_future_index(train, horizon))


def fit_holt_winters(
    train: pd.Series, horizon: int, *, period: int = 12, trend: str | None = "add"
) -> pd.Series:
    """Fit exponential smoothing with an additive trend and seasonal component."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    seasonal = "add" if len(train) >= 2 * period else None
    model = ExponentialSmoothing(
        train, trend=trend, seasonal=seasonal, seasonal_periods=period if seasonal else None
    )
    fitted = model.fit()
    forecast = fitted.forecast(horizon)
    return pd.Series(np.asarray(forecast, dtype=float), index=_future_index(train, horizon))


Forecaster = Callable[..., pd.Series]

FORECASTERS: dict[str, Forecaster] = {
    "naive": fit_naive,
    "seasonal_naive": fit_seasonal_naive,
    "sarima": fit_sarima,
    "holt_winters": fit_holt_winters,
}


def available_models() -> list[str]:
    """Return the registered forecaster names."""
    return sorted(FORECASTERS)


def build_forecast(name: str, train: pd.Series, horizon: int, **params: Any) -> pd.Series:
    """Produce a forecast with a registered method.

    Raises:
        KeyError: naming the valid methods, so a config typo is obvious.
    """
    forecaster = FORECASTERS.get(name)
    if forecaster is None:
        raise KeyError(f"unknown forecaster {name!r}; available: {available_models()}")
    return forecaster(train, horizon, **params)


def compare_forecasters(
    split: ForecastSplit,
    *,
    methods: list[str] | None = None,
    params: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Score every registered method on the same holdout, best MAE first.

    A method that fails to fit is kept in the table with NaN scores and its
    error message, rather than aborting the comparison.
    """
    names = methods or available_models()
    overrides = params or {}
    rows: list[dict[str, Any]] = []

    for name in names:
        try:
            prediction = build_forecast(name, split.train, split.horizon, **overrides.get(name, {}))
            scores = forecast_scores(split.test, prediction)
        except Exception as error:
            rows.append({"method": name, "error": str(error)[:120]})
            continue
        rows.append({"method": name, **scores, "error": None})

    table = pd.DataFrame(rows)
    if "mae" not in table.columns:
        raise RuntimeError("every forecaster failed; see the 'error' column")

    ordered = table.sort_values("mae", na_position="last").reset_index(drop=True)
    return ordered.assign(skill_vs_naive=_skill_against_naive(ordered))


def _skill_against_naive(table: pd.DataFrame) -> pd.Series:
    """Return 1 - MAE / naive MAE for each row, over the same horizon.

    Positive means the method beats repeating the last observed value across the
    whole forecast period; zero means it matches it; negative means it is worse.
    This is the like-for-like comparison that ``mase`` is not.
    """
    naive_rows = table[table["method"] == "naive"]["mae"].dropna()
    if naive_rows.empty:
        return pd.Series([float("nan")] * len(table), index=table.index)
    baseline = float(naive_rows.iloc[0])
    if baseline <= 0:
        return pd.Series([float("nan")] * len(table), index=table.index)
    return (1 - table["mae"] / baseline).round(4)


def _future_index(train: pd.Series, horizon: int) -> pd.DatetimeIndex:
    """Build the timestamp index the forecast lands on."""
    index = train.index
    if isinstance(index, pd.DatetimeIndex) and len(index) > 1:
        step = index[-1] - index[-2]
        return pd.DatetimeIndex([index[-1] + step * (i + 1) for i in range(horizon)])
    return pd.DatetimeIndex(pd.RangeIndex(len(train), len(train) + horizon))


def _strength(residual_variance: float, combined: np.ndarray) -> float:
    """Return 1 - Var(residual) / Var(component + residual), floored at zero."""
    total = float(np.var(combined))
    if total <= 0:
        return float("nan")
    return float(max(0.0, 1 - residual_variance / total))
