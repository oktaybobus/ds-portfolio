"""Tests for the second group of ported projects.

``series_forecast`` runs in CI: both its series are small enough to commit, so
the whole path from CSV to scored forecast is covered on every push. The
recommender and the housing model need fetched data and are marked accordingly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dsjourney import forecasting, recommend
from dsjourney.datasets import DatasetNotFoundError
from dsjourney.training import train_supervised
from projects.istanbul_housing import pipeline as housing
from projects.movie_recommender import pipeline as movies
from projects.series_forecast import pipeline as series


class TestSeriesForecast:
    """Committed data, so these run everywhere."""

    @pytest.mark.parametrize("key", sorted(series.SERIES))
    def test_every_series_loads_and_indexes(self, key: str) -> None:
        spec = series.spec_for(key)
        values = series.build_series(series.load_raw(key), spec)
        assert isinstance(values.index, pd.DatetimeIndex)
        assert values.index.is_monotonic_increasing
        assert values.notna().all()

    def test_quarter_labels_span_real_years(self) -> None:
        """ "2000Q1" is unparseable by pd.to_datetime and needs a PeriodIndex."""
        spec = series.spec_for("adidas_revenue")
        values = series.build_series(series.load_raw("adidas_revenue"), spec)
        assert values.index.min().year == 2000
        assert values.index.max().year > 2015

    def test_delhi_series_covers_four_years(self) -> None:
        spec = series.spec_for("delhi_temperature")
        values = series.build_series(series.load_raw("delhi_temperature"), spec)
        assert len(values) > 1400
        assert values.between(0, 45).all()  # degrees C in New Delhi

    def test_spec_for_names_the_alternatives_on_a_typo(self) -> None:
        with pytest.raises(KeyError, match="available:"):
            series.spec_for("dehli_temperature")

    def test_prepare_input_explains_the_right_entry_point(self) -> None:
        with pytest.raises(NotImplementedError, match=r"train\.py"):
            series.prepare_input({})

    @pytest.mark.slow
    def test_seasonality_beats_naive_on_temperature(self) -> None:
        """The finding the notebook's no-holdout approach could not produce."""
        spec = series.spec_for("delhi_temperature")
        values = series.build_series(series.load_raw("delhi_temperature"), spec)
        split = forecasting.chronological_split(values, horizon=spec.horizon)

        table = forecasting.compare_forecasters(
            split,
            methods=["naive", "seasonal_naive", "holt_winters"],
            params={
                "seasonal_naive": {"period": spec.period},
                "holt_winters": {"period": spec.period},
            },
        )
        assert table.iloc[0]["skill_vs_naive"] > 0.2

    @pytest.mark.slow
    def test_nothing_beats_naive_on_adidas_revenue(self) -> None:
        """Locks in the more useful of the two results: do not deploy a model here."""
        spec = series.spec_for("adidas_revenue")
        values = series.build_series(series.load_raw("adidas_revenue"), spec)
        split = forecasting.chronological_split(values, horizon=spec.horizon)

        table = forecasting.compare_forecasters(
            split,
            methods=["naive", "seasonal_naive", "holt_winters"],
            params={"seasonal_naive": {"period": 4}, "holt_winters": {"period": 4}},
        )
        assert table.iloc[0]["method"] == "naive"


@pytest.mark.needs_data
class TestMovieRecommender:
    @pytest.fixture(scope="class")
    def ratings(self) -> pd.DataFrame:
        try:
            return movies.load_raw()
        except DatasetNotFoundError as error:
            pytest.skip(str(error))

    def test_ratings_keep_the_timestamp(self, ratings: pd.DataFrame) -> None:
        """The notebooks read only three columns, which made a time split impossible."""
        assert "timestamp" in ratings.columns
        assert ratings["timestamp"].nunique() > 1000

    def test_catalogue_carries_genre_flags(self) -> None:
        items = movies.load_items()
        assert "action" in items.columns
        assert items[["action", "drama", "comedy"]].isin([0, 1]).all().all()

    def test_holdout_is_chronological(self, ratings: pd.DataFrame) -> None:
        """No training rating may come after the holdout begins.

        The boundary is ``<=`` rather than ``<``: MovieLens timestamps have
        second resolution and a user often rates several films in the same
        second, so an exact tie at the split point is normal.
        """
        split = recommend.split_ratings(ratings, holdout_per_user=5)
        sample = split.test["user_id"].drop_duplicates().head(20)
        for user_id in sample:
            latest_train = split.train.loc[split.train["user_id"] == user_id, "timestamp"].max()
            earliest_test = split.test.loc[split.test["user_id"] == user_id, "timestamp"].min()
            assert latest_train <= earliest_test

    @pytest.mark.slow
    def test_star_wars_neighbours_are_the_trilogy(self, ratings: pd.DataFrame) -> None:
        """A readable check that the support floor fixed the similarity list."""
        items = movies.load_items()
        matrix = recommend.user_item_matrix(ratings)
        similar = recommend.similar_by_ratings(matrix, movies.DEMO_ITEM_ID, items, limit=3)
        titles = " ".join(similar["title"].tolist())
        assert "Empire Strikes Back" in titles
        assert "Return of the Jedi" in titles

    @pytest.mark.slow
    def test_ranking_beats_random(self, ratings: pd.DataFrame) -> None:
        """Without the support floor this scored below random - see MIN_SUPPORT_FOR_RANKING."""
        split = recommend.split_ratings(ratings, holdout_per_user=5)
        model = recommend.fit_svd(split.train, components=50)
        metrics = recommend.evaluate_recommender(model, split, k=10)

        random_baseline = 10 / int(ratings["item_id"].nunique())
        assert metrics["precision_at_10"] > 3 * random_baseline
        assert metrics["rmse"] < 1.2


@pytest.mark.needs_data
class TestIstanbulHousing:
    @pytest.fixture(scope="class")
    def raw(self) -> pd.DataFrame:
        try:
            return housing.load_raw()
        except DatasetNotFoundError as error:
            pytest.skip(str(error))

    def test_columns_are_ascii_snake_case(self, raw: pd.DataFrame) -> None:
        for column in raw.columns:
            assert column == column.lower()
            assert " " not in column
            assert column.isascii(), f"{column} still has Turkish characters"

    def test_every_building_age_label_is_mapped(self, raw: pd.DataFrame) -> None:
        """The gap that silently discarded 30% of the file.

        Four labels - including 2,838 new builds - were missing from the source
        notebook's map, turned into NaN, and were dropped before training.
        """
        labels = set(raw["bina_yasi"].dropna().astype(str).str.strip())
        unmapped = labels - set(housing.BUILDING_AGE_MAP)
        assert not unmapped, f"unmapped building-age labels: {sorted(unmapped)}"

    def test_feature_build_retains_almost_every_listing(self, raw: pd.DataFrame) -> None:
        features = housing.build_features(raw)
        assert len(features) / len(raw) > 0.95

    def test_features_are_numeric_and_complete(self, raw: pd.DataFrame) -> None:
        features = housing.build_features(raw)
        assert features.select_dtypes(exclude="number").empty
        assert features.isna().sum().sum() == 0
        assert "fiyat" not in features.columns

    def test_room_counts_sum_both_sides(self) -> None:
        assert housing._room_count("4.5+1") == pytest.approx(5.5)
        assert housing._room_count("3+1") == pytest.approx(4.0)
        assert np.isnan(housing._room_count("Stüdyo"))

    def test_areas_lose_their_unit(self) -> None:
        parsed = housing._area_to_number(pd.Series(["292 m²", "110m2", "not an area"]))
        assert parsed.iloc[0] == pytest.approx(292.0)
        assert parsed.iloc[1] == pytest.approx(110.0)
        assert np.isnan(parsed.iloc[2])

    def test_prices_are_clipped_to_the_residential_range(self, raw: pd.DataFrame) -> None:
        features = housing.build_features(raw)
        prices = housing.postprocess(features["fiyat_log"].to_numpy())
        assert prices.min() >= 0.1
        assert prices.max() <= housing.MAX_PRICE_MILLIONS

    @pytest.mark.slow
    def test_training_reports_both_scales(self, raw: pd.DataFrame) -> None:
        """A log-scale R2 and a price-scale R2 are different numbers."""
        features = housing.build_features(raw)
        report = train_supervised(
            housing.CONFIG,
            features,
            save=False,
            make_plots=False,
            inverse_transform=housing.postprocess,
        )
        metrics = report.bundle.metrics
        assert metrics["r2"] > 0.75
        assert metrics["r2_original"] > 0.70
        assert metrics["r2"] != metrics["r2_original"]


@pytest.mark.needs_data
class TestBartRidership:
    """Kaggle-sourced, so marked needs_data even though kagglehub caches it."""

    @pytest.fixture(scope="class")
    def raw(self) -> pd.DataFrame:
        from projects.bart_ridership import pipeline as bart

        try:
            return bart.load_raw(sample=20_000)
        except Exception as error:
            pytest.skip(f"BART dataset unavailable: {error}")

    def test_station_coordinates_are_parsed_out_of_prose(self) -> None:
        """Coordinates live inside a free-text Location field, in either order."""
        from projects.bart_ridership import pipeline as bart

        try:
            stations = bart.load_stations()
        except Exception as error:
            pytest.skip(f"BART dataset unavailable: {error}")

        assert len(stations) > 40
        assert stations["latitude"].between(37.0, 38.5).all()
        assert stations["longitude"].between(-123.0, -121.0).all()

    def test_coordinate_parser_is_order_independent(self) -> None:
        from projects.bart_ridership import pipeline as bart

        assert bart._parse_coordinates("-122.271450,37.803768,0") == (37.803768, -122.27145)
        assert bart._parse_coordinates("37.803768,-122.271450") == (37.803768, -122.27145)
        assert bart._parse_coordinates(None) == (None, None)

    def test_features_exclude_the_raw_target(self, raw: pd.DataFrame) -> None:
        """Leaving Throughput in the frame would hand the model the answer."""
        from projects.bart_ridership import pipeline as bart

        features = bart.build_features(raw)
        assert "Throughput" not in features.columns
        assert "year" not in features.columns
        assert "throughput_log" in features.columns

    def test_cyclical_encoding_wraps(self, raw: pd.DataFrame) -> None:
        """Hour 23 and hour 0 must be neighbours, not opposite extremes."""
        from projects.bart_ridership import pipeline as bart

        features = bart.build_features(raw)
        for column in ("hour_sin", "hour_cos", "day_of_week_sin", "month_cos"):
            assert column in features.columns
            assert features[column].between(-1.0, 1.0).all()

    def test_distance_is_geographic(self, raw: pd.DataFrame) -> None:
        """BART's longest trip is under 100 km; same-station trips are zero."""
        from projects.bart_ridership import pipeline as bart

        features = bart.build_features(raw)
        assert features["distance_km"].max() < 100
        same = features[features["same_station"] == 1]
        if len(same):
            assert same["distance_km"].max() == pytest.approx(0.0, abs=1e-6)

    def test_chronological_frames_split_by_year(self) -> None:
        from projects.bart_ridership import pipeline as bart

        try:
            train_frame, test_frame = bart.chronological_frames(sample=5_000)
        except Exception as error:
            pytest.skip(f"BART dataset unavailable: {error}")

        assert len(train_frame) > 0
        assert len(test_frame) > 0
        assert list(train_frame.columns) == list(test_frame.columns)
