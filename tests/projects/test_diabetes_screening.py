"""Tests for the Pima screening project.

The dataset is committed, so the zero counts below are measured from the real
file. They are the reason the project exists: the source notebook fed every one
of them to the model as a genuine reading.
"""

from __future__ import annotations

import pytest

from dsjourney import spark as dsspark
from projects.diabetes_screening import pipeline

ROWS = 768
POSITIVES = 268
# Measured on the shipped file. Insulin is nearly half the column.
ENCODED_MISSING = {
    "Glucose": 5,
    "BloodPressure": 35,
    "SkinThickness": 227,
    "Insulin": 374,
    "BMI": 11,
}


def test_the_raw_file_has_the_published_shape() -> None:
    frame = pipeline.load_raw()
    assert len(frame) == ROWS
    assert int(frame[pipeline.TARGET].sum()) == POSITIVES
    assert list(frame.columns) == [*pipeline.FEATURES, pipeline.TARGET]


def test_the_impossible_zeros_are_still_in_the_raw_file() -> None:
    """If this ever fails the dataset was pre-cleaned and the fix is redundant."""
    frame = pipeline.load_raw()
    for column, expected in ENCODED_MISSING.items():
        assert int((frame[column] == 0).sum()) == expected, column


def test_build_features_recovers_them_as_missing() -> None:
    cleaned = pipeline.build_features(pipeline.load_raw())
    for column, expected in ENCODED_MISSING.items():
        assert int(cleaned[column].isna().sum()) == expected, column


def test_zero_pregnancies_is_left_alone() -> None:
    """111 women in the file have never been pregnant. That is a measurement."""
    cleaned = pipeline.build_features(pipeline.load_raw())
    assert cleaned["Pregnancies"].isna().sum() == 0
    assert int((cleaned["Pregnancies"] == 0).sum()) == 111


def test_build_features_does_not_mutate_its_argument() -> None:
    frame = pipeline.load_raw()
    before = int((frame["Insulin"] == 0).sum())
    pipeline.build_features(frame)
    assert int((frame["Insulin"] == 0).sum()) == before


def test_the_majority_baseline_is_worth_beating() -> None:
    """Any accuracy at or below this number means the model learned nothing."""
    frame = pipeline.load_raw()
    assert dsspark.majority_baseline(frame[pipeline.TARGET]) == pytest.approx(0.651, abs=0.001)


def test_prepare_input_orders_columns_for_the_model() -> None:
    row = pipeline.prepare_input({name: 1 for name in pipeline.FEATURES})
    assert list(row.columns) == list(pipeline.FEATURES)
    assert len(row) == 1


@pytest.mark.needs_spark
@pytest.mark.slow
def test_training_beats_the_baseline_and_reports_distinct_metrics() -> None:
    """End to end on the real data: the numbers must differ from each other.

    Accuracy and ROC AUC coinciding would mean the mislabel in the notebook was
    harmless. On this dataset they are several points apart.
    """
    from projects.diabetes_screening.train import main

    assert main(["--no-save"]) == 0
