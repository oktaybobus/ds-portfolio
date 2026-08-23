"""Unit tests for the pure feature transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dsjourney import preprocess
from dsjourney.config import SplitConfig


def test_transforms_never_mutate_their_input(messy_frame: pd.DataFrame) -> None:
    """Every helper must leave the caller's frame untouched."""
    before = messy_frame.copy(deep=True)

    preprocess.strip_unit(messy_frame, "ram", "GB", dtype="int")
    preprocess.group_rare_categories(messy_frame, "brand", min_count=2)
    preprocess.binary_flag(messy_frame, "screen", "IPS", "ips")
    preprocess.impute_numeric(messy_frame, ["score"])
    preprocess.drop_columns(messy_frame, ["price"])

    pd.testing.assert_frame_equal(messy_frame, before)


def test_strip_unit_converts_to_numbers(messy_frame: pd.DataFrame) -> None:
    result = preprocess.strip_unit(messy_frame, "ram", "GB", dtype="int")
    assert result["ram"].tolist() == [8, 16, 4, 8, 32, 8]


def test_strip_unit_handles_floats(messy_frame: pd.DataFrame) -> None:
    result = preprocess.strip_unit(messy_frame, "weight", "kg", dtype="float")
    assert result["weight"].iloc[0] == pytest.approx(1.37)


def test_group_rare_categories_collapses_the_long_tail(messy_frame: pd.DataFrame) -> None:
    result = preprocess.group_rare_categories(messy_frame, "brand", min_count=2)
    assert result["brand"].tolist() == ["Dell", "Dell", "Dell", "HP", "Others", "HP"]


def test_binary_flag_is_case_insensitive(messy_frame: pd.DataFrame) -> None:
    result = preprocess.binary_flag(messy_frame, "screen", "touchscreen", "touch")
    assert result["touch"].tolist() == [1, 0, 0, 1, 0, 0]


def test_extract_pattern_pulls_out_capture_groups(messy_frame: pd.DataFrame) -> None:
    result = preprocess.extract_pattern(messy_frame, "screen", r"(\d+)x(\d+)", ["width", "height"])
    assert result["width"].tolist() == [1920, 1366, 2560, 1920, 3840, 1920]
    assert result["height"].iloc[4] == 2160


def test_extract_pattern_rejects_a_name_count_mismatch(messy_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="captures 2 group"):
        preprocess.extract_pattern(messy_frame, "screen", r"(\d+)x(\d+)", ["only_one"])


def test_impute_numeric_keeps_a_missingness_flag(messy_frame: pd.DataFrame) -> None:
    result = preprocess.impute_numeric(messy_frame, ["score"], strategy="median")
    assert result["score"].isna().sum() == 0
    assert result["score_missing"].tolist() == [0, 1, 0, 0, 1, 0]


def test_impute_numeric_skips_complete_columns(messy_frame: pd.DataFrame) -> None:
    result = preprocess.impute_numeric(messy_frame, ["price"])
    assert "price_missing" not in result.columns


def test_impute_numeric_rejects_an_unknown_strategy(messy_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unknown imputation strategy"):
        preprocess.impute_numeric(messy_frame, ["score"], strategy="wishful")


def test_safe_ratio_treats_a_zero_denominator_as_missing(messy_frame: pd.DataFrame) -> None:
    result = preprocess.safe_ratio(messy_frame, "balance", "limit", "utilisation")
    assert np.isnan(result["utilisation"].iloc[1])
    assert result["utilisation"].iloc[0] == pytest.approx(0.2)


def test_log_transform_round_trips(messy_frame: pd.DataFrame) -> None:
    transformed = preprocess.log_transform_target(messy_frame, "price", "price_log")
    assert "price" not in transformed.columns
    restored = preprocess.inverse_log_transform(transformed["price_log"])
    np.testing.assert_allclose(restored, messy_frame["price"].to_numpy())


def test_one_hot_drops_the_first_level(messy_frame: pd.DataFrame) -> None:
    result = preprocess.one_hot(messy_frame, ["brand"], drop_first=True)
    assert "brand" not in result.columns
    assert "brand_Dell" not in result.columns
    assert "brand_HP" in result.columns


def test_one_hot_ignores_absent_columns(messy_frame: pd.DataFrame) -> None:
    result = preprocess.one_hot(messy_frame, ["not_here"])
    pd.testing.assert_frame_equal(result, messy_frame)


def test_split_and_scale_fits_the_scaler_on_training_data_only(
    regression_frame: pd.DataFrame,
) -> None:
    """The regression test for the leak the source notebooks had.

    A scaler fitted on the full frame would centre the training half exactly on
    zero. Fitted on the training half alone, the test half is merely transformed
    and its mean drifts away from zero - which is what we assert.
    """
    result = preprocess.split_and_scale(
        regression_frame, "target", SplitConfig(test_size=0.25), scale_columns=["feature_a"]
    )
    assert result.x_train["feature_a"].mean() == pytest.approx(0.0, abs=1e-9)
    assert abs(result.x_test["feature_a"].mean()) > 1e-9


def test_split_and_scale_without_columns_returns_no_scaler(regression_frame: pd.DataFrame) -> None:
    result = preprocess.split_and_scale(regression_frame, "target", SplitConfig())
    assert result.scaler is None


def test_split_and_scale_rejects_a_missing_target(regression_frame: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="not in the frame"):
        preprocess.split_and_scale(regression_frame, "nope", SplitConfig())


def test_align_to_training_columns_fills_and_orders() -> None:
    row = pd.DataFrame([{"b": 1, "unexpected": 9}])
    aligned = preprocess.align_to_training_columns(row, ["a", "b", "c"])
    assert list(aligned.columns) == ["a", "b", "c"]
    assert aligned.iloc[0].tolist() == [0, 1, 0]


def test_drop_duplicate_rows_resets_the_index() -> None:
    frame = pd.DataFrame({"a": [1, 1, 2]})
    result = preprocess.drop_duplicate_rows(frame)
    assert len(result) == 2
    assert result.index.tolist() == [0, 1]


def test_add_cyclical_makes_the_wrap_adjacent() -> None:
    """Hour 23 and hour 0 are an hour apart; as raw integers they are 23 apart."""
    frame = pd.DataFrame({"hour": [0, 12, 23]})
    encoded = preprocess.add_cyclical(frame, "hour", period=24)

    def distance(a: int, b: int) -> float:
        return float(
            np.hypot(
                encoded["hour_sin"].iloc[a] - encoded["hour_sin"].iloc[b],
                encoded["hour_cos"].iloc[a] - encoded["hour_cos"].iloc[b],
            )
        )

    assert distance(0, 2) < distance(0, 1)  # 0 is nearer 23 than it is to 12


def test_add_cyclical_can_drop_the_source() -> None:
    frame = pd.DataFrame({"month": [1, 6, 12]})
    assert "month" not in preprocess.add_cyclical(frame, "month", period=12, drop=True).columns


def test_add_calendar_features_derives_the_parts() -> None:
    frame = pd.DataFrame({"when": ["2016-01-02 08:00:00", "2016-01-04 23:30:00"]})
    result = preprocess.add_calendar_features(frame, "when")
    assert result["hour"].tolist() == [8, 23]
    assert result["is_weekend"].tolist() == [1, 0]  # 2 Jan 2016 was a Saturday
    assert result["month"].tolist() == [1, 1]


def test_haversine_is_zero_for_the_same_point() -> None:
    assert preprocess.haversine_km(37.8, -122.3, 37.8, -122.3) == pytest.approx(0.0, abs=1e-9)


def test_haversine_matches_a_known_distance() -> None:
    """Oakland 12th St to San Francisco Embarcadero is about 13 km apart."""
    distance = preprocess.haversine_km(37.803768, -122.271450, 37.792874, -122.396742)
    assert 10 < float(distance) < 16


def test_haversine_works_on_series() -> None:
    frame = pd.DataFrame(
        {
            "lat1": [37.8, 37.8],
            "lon1": [-122.3, -122.3],
            "lat2": [37.8, 38.0],
            "lon2": [-122.3, -122.0],
        }
    )
    result = preprocess.haversine_km(frame["lat1"], frame["lon1"], frame["lat2"], frame["lon2"])
    assert isinstance(result, pd.Series)
    assert result.iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert result.iloc[1] > 20
