"""Unit tests for time-series forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dsjourney import forecasting


def test_chronological_split_keeps_the_order(seasonal_series: pd.Series) -> None:
    """The holdout must be the tail, never a random sample."""
    split = forecasting.chronological_split(seasonal_series, horizon=30)
    assert split.horizon == 30
    assert len(split.train) == len(seasonal_series) - 30
    assert split.train.index.max() < split.test.index.min()
    pd.testing.assert_series_equal(split.test, seasonal_series.iloc[-30:])


def test_chronological_split_rejects_a_bad_horizon(seasonal_series: pd.Series) -> None:
    with pytest.raises(ValueError, match="horizon must be positive"):
        forecasting.chronological_split(seasonal_series, horizon=0)


def test_chronological_split_rejects_too_short_a_series() -> None:
    short = pd.Series(range(5), index=pd.date_range("2020-01-01", periods=5))
    with pytest.raises(ValueError, match="cannot spare"):
        forecasting.chronological_split(short, horizon=10)


def test_load_series_sorts_and_indexes() -> None:
    frame = pd.DataFrame(
        {"when": ["2020-01-03", "2020-01-01", "2020-01-02"], "value": [3.0, 1.0, 2.0]}
    )
    series = forecasting.load_series(frame, date_column="when", value_column="value")
    assert series.tolist() == [1.0, 2.0, 3.0]
    assert isinstance(series.index, pd.DatetimeIndex)


def test_load_series_drops_unparseable_dates() -> None:
    frame = pd.DataFrame({"when": ["2020-01-01", "not a date"], "value": [1.0, 2.0]})
    assert len(forecasting.load_series(frame, date_column="when", value_column="value")) == 1


def test_parse_quarter_index_handles_quarter_labels() -> None:
    """pd.to_datetime cannot read "2000Q1"; a PeriodIndex can."""
    frame = pd.DataFrame({"period": ["2000Q1", "2000Q2", "2001Q1"]})
    index = forecasting.parse_quarter_index(frame, "period")
    assert index[0] == pd.Timestamp("2000-01-01")
    assert index[2] == pd.Timestamp("2001-01-01")


def test_forecast_scores_are_zero_on_a_perfect_forecast() -> None:
    actual = pd.Series([1.0, 2.0, 3.0, 4.0])
    scores = forecasting.forecast_scores(actual, actual)
    assert scores["mae"] == pytest.approx(0.0)
    assert scores["rmse"] == pytest.approx(0.0)
    assert scores["mase"] == pytest.approx(0.0)


def test_forecast_scores_handle_a_zero_actual() -> None:
    """MAPE is undefined at zero; it must be skipped, not returned as infinity."""
    scores = forecasting.forecast_scores(pd.Series([0.0, 2.0]), np.array([1.0, 2.0]))
    assert np.isfinite(scores["mape"])


def test_naive_repeats_the_last_value(seasonal_series: pd.Series) -> None:
    split = forecasting.chronological_split(seasonal_series, horizon=10)
    forecast = forecasting.fit_naive(split.train, 10)
    assert len(forecast) == 10
    assert forecast.nunique() == 1
    assert forecast.iloc[0] == pytest.approx(float(split.train.iloc[-1]))


def test_seasonal_naive_repeats_a_season(seasonal_series: pd.Series) -> None:
    split = forecasting.chronological_split(seasonal_series, horizon=12)
    forecast = forecasting.fit_seasonal_naive(split.train, 12, period=12)
    np.testing.assert_allclose(forecast.to_numpy(), split.train.iloc[-12:].to_numpy())


def test_seasonal_naive_falls_back_when_history_is_short() -> None:
    short = pd.Series(range(10), index=pd.date_range("2020-01-01", periods=10), dtype=float)
    forecast = forecasting.fit_seasonal_naive(short, 3, period=50)
    assert forecast.nunique() == 1


def test_forecast_index_continues_the_series(seasonal_series: pd.Series) -> None:
    split = forecasting.chronological_split(seasonal_series, horizon=5)
    forecast = forecasting.fit_naive(split.train, 5)
    assert forecast.index[0] > split.train.index[-1]
    assert (forecast.index == split.test.index).all()


def test_seasonal_strength_detects_a_strong_cycle(seasonal_series: pd.Series) -> None:
    strength = forecasting.seasonal_strength(seasonal_series, period=365)
    assert strength["seasonal_strength"] > 0.5


def test_seasonal_strength_returns_nan_when_too_short() -> None:
    short = pd.Series(range(10), index=pd.date_range("2020-01-01", periods=10), dtype=float)
    assert np.isnan(forecasting.seasonal_strength(short, period=365)["seasonal_strength"])


def test_available_models_lists_the_registry() -> None:
    assert set(forecasting.available_models()) >= {"naive", "seasonal_naive", "holt_winters"}


def test_build_forecast_names_the_alternatives_on_a_typo(seasonal_series: pd.Series) -> None:
    with pytest.raises(KeyError, match="available:"):
        forecasting.build_forecast("crystal_ball", seasonal_series, 5)


@pytest.mark.slow
def test_compare_forecasters_ranks_and_scores_skill(seasonal_series: pd.Series) -> None:
    """A seasonal method should beat naive on a strongly seasonal series."""
    split = forecasting.chronological_split(seasonal_series, horizon=30)
    table = forecasting.compare_forecasters(
        split,
        methods=["naive", "seasonal_naive"],
        params={"seasonal_naive": {"period": 365}},
    )
    assert table["mae"].is_monotonic_increasing
    assert (table.loc[table["method"] == "naive", "skill_vs_naive"] == 0).all()
    assert table.iloc[0]["skill_vs_naive"] >= 0


def test_compare_forecasters_records_a_failure(seasonal_series: pd.Series) -> None:
    split = forecasting.chronological_split(seasonal_series, horizon=10)
    table = forecasting.compare_forecasters(
        split, methods=["naive", "holt_winters"], params={"holt_winters": {"period": -5}}
    )
    assert table["error"].notna().any()
    assert "naive" in table["method"].tolist()
