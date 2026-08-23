"""Unit tests for the recommender."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dsjourney import recommend


def test_split_ratings_withholds_the_most_recent(ratings_log: pd.DataFrame) -> None:
    """A random split would let a model see a user's future taste."""
    split = recommend.split_ratings(ratings_log, holdout_per_user=2)

    for user_id in split.test["user_id"].unique():
        latest_train = split.train.loc[split.train["user_id"] == user_id, "timestamp"].max()
        earliest_test = split.test.loc[split.test["user_id"] == user_id, "timestamp"].min()
        assert latest_train < earliest_test


def test_split_ratings_keeps_every_rating(ratings_log: pd.DataFrame) -> None:
    split = recommend.split_ratings(ratings_log, holdout_per_user=2)
    assert len(split.train) + len(split.test) == len(ratings_log)


def test_split_ratings_spares_users_with_too_little_history() -> None:
    """A user with three ratings cannot give up two of them."""
    sparse = pd.DataFrame(
        {"user_id": [1, 1, 1], "item_id": [1, 2, 3], "rating": [5, 4, 3], "timestamp": [1, 2, 3]}
    )
    split = recommend.split_ratings(sparse, holdout_per_user=2)
    assert split.test.empty
    assert len(split.train) == 3


def test_split_ratings_rejects_a_bad_holdout(ratings_log: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        recommend.split_ratings(ratings_log, holdout_per_user=0)


def test_user_item_matrix_is_users_by_items(ratings_log: pd.DataFrame) -> None:
    matrix = recommend.user_item_matrix(ratings_log)
    assert matrix.shape[0] == ratings_log["user_id"].nunique()
    assert matrix.shape[1] == ratings_log["item_id"].nunique()


def test_popularity_ranking_orders_by_rating_mass(
    ratings_log: pd.DataFrame, item_catalogue: pd.DataFrame
) -> None:
    ranking = recommend.popularity_ranking(ratings_log, item_catalogue)
    assert ranking["share_pct"].sum() == pytest.approx(100.0, abs=0.01)
    assert ranking.iloc[0]["rank"] == 1.0  # competition ranking: ties share rank 1
    assert ranking["rank"].is_monotonic_increasing
    assert 99 not in ranking.head(4).index  # the single-rating item is not popular


def test_similar_by_ratings_needs_minimum_support(
    ratings_log: pd.DataFrame, item_catalogue: pd.DataFrame
) -> None:
    """Item 99 has one rating; without a floor it would top the list."""
    matrix = recommend.user_item_matrix(ratings_log)
    similar = recommend.similar_by_ratings(matrix, 1, item_catalogue, min_ratings=5)
    assert 99 not in similar.index
    assert 1 not in similar.index  # the seed item is excluded


def test_similar_by_ratings_finds_the_taste_group(
    ratings_log: pd.DataFrame, item_catalogue: pd.DataFrame
) -> None:
    matrix = recommend.user_item_matrix(ratings_log)
    similar = recommend.similar_by_ratings(matrix, 1, item_catalogue, min_ratings=5, limit=2)
    assert set(similar.index) <= {2, 3}


def test_ranking_surfaces_the_taste_group(ratings_log: pd.DataFrame) -> None:
    """A user who liked items 1-3 should be offered the rest of that group."""
    model = recommend.fit_svd(ratings_log, components=3)
    ranked = model.rank_for_user(0, seen={1}, limit=2, min_support=5)
    assert set(ranked.index) & {2, 3}


def test_similar_by_ratings_rejects_an_unrated_item(
    ratings_log: pd.DataFrame, item_catalogue: pd.DataFrame
) -> None:
    matrix = recommend.user_item_matrix(ratings_log)
    with pytest.raises(KeyError, match="no ratings"):
        recommend.similar_by_ratings(matrix, 12345, item_catalogue)


def test_similar_by_genre_matches_the_genre(item_catalogue: pd.DataFrame) -> None:
    similar = recommend.similar_by_genre(item_catalogue, 1, limit=2)
    assert set(similar.index) <= {2, 3}
    assert 1 not in similar.index


def test_similar_by_genre_rejects_an_unknown_item(item_catalogue: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="not in the catalogue"):
        recommend.similar_by_genre(item_catalogue, 12345)


def test_svd_predicts_within_the_rating_range(ratings_log: pd.DataFrame) -> None:
    model = recommend.fit_svd(ratings_log, components=3)
    prediction = model.predict(0, 1)
    assert 1.0 <= prediction <= 5.5


def test_svd_falls_back_for_unknown_users_and_items(ratings_log: pd.DataFrame) -> None:
    model = recommend.fit_svd(ratings_log, components=3)
    assert model.predict(9999, 1) == pytest.approx(model.predict(9998, 1))
    assert model.predict(0, 12345) == pytest.approx(model.global_mean)


def test_svd_requires_fitting_first() -> None:
    with pytest.raises(RuntimeError, match="fit\\(\\) must be called"):
        recommend.SVDRecommender().predict(1, 1)


def test_ranking_excludes_low_support_items(ratings_log: pd.DataFrame) -> None:
    """The regression test for precision@10 scoring below random.

    Item 99 was rated 5 by one user, so its item mean is 5.0 and it outranks
    everything for every user until a support floor removes it.
    """
    model = recommend.fit_svd(ratings_log, components=3)

    unfiltered = model.rank_for_user(5, seen=set(), limit=10, min_support=1)
    filtered = model.rank_for_user(5, seen=set(), limit=10, min_support=5)

    assert 99 in unfiltered.index
    assert 99 not in filtered.index


def test_ranking_excludes_already_seen_items(ratings_log: pd.DataFrame) -> None:
    model = recommend.fit_svd(ratings_log, components=3)
    ranked = model.rank_for_user(0, seen={1, 2, 3}, limit=10, min_support=5)
    assert not {1, 2, 3} & set(ranked.index)


def test_ranking_is_empty_for_an_unknown_user(ratings_log: pd.DataFrame) -> None:
    model = recommend.fit_svd(ratings_log, components=3)
    assert model.rank_for_user(9999, seen=set()).empty


def test_recommend_returns_titles(ratings_log: pd.DataFrame, item_catalogue: pd.DataFrame) -> None:
    model = recommend.fit_svd(ratings_log, components=3)
    table = model.recommend(0, seen={1, 2, 3}, items=item_catalogue, limit=3)
    assert list(table.columns) == ["item_id", "predicted_rating", "title"]
    assert table["title"].notna().all()


def test_evaluate_recommender_reports_ranking_and_error(ratings_log: pd.DataFrame) -> None:
    split = recommend.split_ratings(ratings_log, holdout_per_user=2)
    model = recommend.fit_svd(split.train, components=3)
    # This fixture is tiny, so the production support floor of 20 would leave
    # nothing eligible to rank; the floor itself is covered by its own test.
    metrics = recommend.evaluate_recommender(model, split, k=3, min_support=2)

    assert np.isfinite(metrics["rmse"])
    assert 0.0 <= metrics["precision_at_3"] <= 1.0
    assert 0.0 <= metrics["recall_at_3"] <= 1.0


def test_available_models_lists_the_registry() -> None:
    assert set(recommend.available_models()) == {
        "svd",
        "popularity",
        "item_correlation",
        "content_genre",
    }
