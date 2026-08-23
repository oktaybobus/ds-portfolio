"""Unit tests for the text helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression

from dsjourney import text


def test_clean_text_lowercases_and_strips_noise() -> None:
    assert text.clean_text("Great FOOD!! 10/10, would return.") == "great food would return"


def test_clean_text_tolerates_non_strings() -> None:
    assert text.clean_text(None) == ""
    assert text.clean_text(3.14) == ""


def test_remove_stopwords_keeps_content_words() -> None:
    assert text.remove_stopwords("the food was very good") == "food good"


def test_normalise_combines_both_steps() -> None:
    assert text.normalise("The Pizza Was NOT Good!!!") == "pizza good"


def test_normalise_series_maps_over_a_column() -> None:
    series = pd.Series(["Amazing Service!", "Awful, cold food."])
    assert text.normalise_series(series).tolist() == ["amazing service", "awful cold food"]


def test_build_text_pipeline_fits_end_to_end(review_frame: pd.DataFrame) -> None:
    documents = text.normalise_series(review_frame["text"])
    labels = (review_frame["stars"] >= 4).astype(int)

    pipeline = text.build_text_pipeline(LogisticRegression(max_iter=200), min_df=1)
    pipeline.fit(documents, labels)

    assert pipeline.predict(pd.Series(["wonderful amazing food"]))[0] == 1
    assert pipeline.predict(pd.Series(["terrible rude dirty"]))[0] == 0


def test_top_features_reports_both_directions(review_frame: pd.DataFrame) -> None:
    documents = text.normalise_series(review_frame["text"])
    labels = (review_frame["stars"] >= 4).astype(int)
    pipeline = text.build_text_pipeline(LogisticRegression(max_iter=200), min_df=1)
    pipeline.fit(documents, labels)

    result = text.top_features(pipeline, limit=3)
    assert set(result["direction"]) == {"positive", "negative"}
    assert len(result) == 6


def test_top_features_is_empty_for_a_non_linear_model() -> None:
    from sklearn.ensemble import RandomForestClassifier

    pipeline = text.build_text_pipeline(RandomForestClassifier(n_estimators=2), min_df=1)
    pipeline.fit(pd.Series(["good food", "bad food"]), [1, 0])
    assert text.top_features(pipeline).empty
