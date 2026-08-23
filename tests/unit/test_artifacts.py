"""Unit tests for model persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from dsjourney.artifacts import ModelBundle, load_bundle, save_bundle


@pytest.fixture
def fitted_bundle() -> ModelBundle:
    """A tiny fitted bundle whose scaler covers one of two features."""
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [0.0, 1.0, 0.0, 1.0]})
    target = frame["a"] * 2 + frame["b"]

    scaler = StandardScaler().fit(frame[["a"]])
    scaled = frame.assign(a=scaler.transform(frame[["a"]]).ravel())
    model = LinearRegression().fit(scaled, target)

    return ModelBundle(
        project="unit_demo",
        task="regression",
        model=model,
        feature_names=["a", "b"],
        metrics={"r2": 1.0},
        scaler=scaler,
        scaled_columns=["a"],
    )


def test_metadata_records_provenance(fitted_bundle: ModelBundle) -> None:
    metadata = fitted_bundle.metadata()
    assert metadata["model_class"] == "LinearRegression"
    assert metadata["feature_count"] == 2
    assert metadata["has_scaler"] is True
    assert metadata["scaled_columns"] == ["a"]
    assert metadata["created_at"]


def test_save_and_load_round_trip(fitted_bundle: ModelBundle, tmp_path: Path) -> None:
    save_bundle(fitted_bundle, directory=tmp_path)
    restored = load_bundle("unit_demo", directory=tmp_path)

    assert restored.feature_names == ["a", "b"]
    assert restored.scaled_columns == ["a"]
    assert restored.metrics == {"r2": 1.0}
    assert restored.scaler is not None
    assert restored.extra["created_at"]
    assert type(restored.model).__name__ == "LinearRegression"


def test_load_bundle_names_the_fix_when_untrained(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="dsj train"):
        load_bundle("never_trained", directory=tmp_path)


def test_prepare_applies_the_saved_scaler(fitted_bundle: ModelBundle) -> None:
    """The regression test for silently unscaled inference input.

    Without the scaler the raw value 3.0 reaches a model trained on values with
    mean 0 and unit variance, and the prediction is wrong by a wide margin.
    """
    row = pd.DataFrame([{"a": 3.0, "b": 1.0}])
    prepared = fitted_bundle.prepare(row)

    assert prepared["a"].iloc[0] != pytest.approx(3.0)
    assert prepared["b"].iloc[0] == pytest.approx(1.0)
    prediction = float(fitted_bundle.model.predict(prepared)[0])
    assert prediction == pytest.approx(7.0, abs=1e-6)  # 3*2 + 1


def test_prepare_reorders_and_fills_columns(fitted_bundle: ModelBundle) -> None:
    row = pd.DataFrame([{"b": 1.0, "a": 3.0, "unexpected": 99.0}])
    prepared = fitted_bundle.prepare(row)
    assert list(prepared.columns) == ["a", "b"]


def test_prepare_rejects_a_frame_missing_a_scaled_column(fitted_bundle: ModelBundle) -> None:
    bundle = ModelBundle(
        project="unit_demo",
        task="regression",
        model=fitted_bundle.model,
        feature_names=["b"],
        scaler=fitted_bundle.scaler,
        scaled_columns=["a"],
    )
    with pytest.raises(ValueError, match="missing scaled column"):
        bundle.prepare(pd.DataFrame([{"b": 1.0}]))


def test_prepare_is_a_plain_alignment_without_a_scaler() -> None:
    bundle = ModelBundle(
        project="unit_demo", task="regression", model=object(), feature_names=["a", "b"]
    )
    prepared = bundle.prepare(pd.DataFrame([{"b": 2.0}]))
    np.testing.assert_array_equal(prepared.to_numpy(), np.array([[0.0, 2.0]]))


def test_prepare_hands_text_models_a_series(review_frame: pd.DataFrame) -> None:
    """The regression test for a text model returning one answer for everything.

    A TF-IDF pipeline treats its input as a sequence of documents. Iterating a
    DataFrame yields column *names*, so passing one vectorises the literal
    string "text" - every review then scores identically, and nothing raises.
    """
    from sklearn.linear_model import LogisticRegression

    from dsjourney.text import build_text_pipeline, normalise_series

    documents = normalise_series(review_frame["text"])
    labels = (review_frame["stars"] >= 4).astype(int)
    model = build_text_pipeline(LogisticRegression(max_iter=200), min_df=1).fit(documents, labels)

    bundle = ModelBundle(
        project="unit_text",
        task="text-classification",
        model=model,
        feature_names=["text"],
    )

    positive = bundle.prepare(pd.DataFrame({"text": ["wonderful delicious great service"]}))
    negative = bundle.prepare(pd.DataFrame({"text": ["awful terrible rude cold"]}))

    assert isinstance(positive, pd.Series)
    assert model.predict(positive)[0] == 1
    assert model.predict(negative)[0] == 0


def test_prepare_rejects_a_text_frame_without_its_column() -> None:
    bundle = ModelBundle(
        project="unit_text", task="text-classification", model=object(), feature_names=["text"]
    )
    with pytest.raises(ValueError, match="no 'text' column"):
        bundle.prepare(pd.DataFrame({"review": ["hello"]}))
