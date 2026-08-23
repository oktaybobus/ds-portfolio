"""Shared fixtures.

The synthetic frames here mirror the shapes the real datasets have - packed
strings, unit suffixes, rare categories, structural missingness - so the unit
tests exercise the same code paths as a real run without needing any data on
disk.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

TENSORFLOW_INSTALLED = importlib.util.find_spec("tensorflow") is not None


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip ``needs_dl`` tests when the optional deep-learning extra is missing.

    The marker alone only lets a caller deselect them; without this hook a plain
    `pytest` run on a machine with no TensorFlow reports failures for an
    environment choice rather than a defect.
    """
    if TENSORFLOW_INSTALLED:
        return
    skip = pytest.mark.skip(reason="TensorFlow not installed (uv sync --extra dl)")
    for item in items:
        if "needs_dl" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def messy_frame() -> pd.DataFrame:
    """A frame with unit suffixes, packed strings, rare labels and missing values."""
    return pd.DataFrame(
        {
            "ram": ["8GB", "16GB", "4GB", "8GB", "32GB", "8GB"],
            "weight": ["1.37kg", "2.1kg", "1.8kg", "1.37kg", "2.9kg", "1.5kg"],
            "brand": ["Dell", "Dell", "Dell", "HP", "Xiaomi", "HP"],
            "screen": [
                "IPS Panel Touchscreen 1920x1080",
                "1366x768",
                "IPS Panel 2560x1600",
                "Touchscreen 1920x1080",
                "3840x2160",
                "1920x1080",
            ],
            "price": [1000.0, 2200.0, 1500.0, 900.0, 4000.0, 1100.0],
            "score": [700.0, np.nan, 640.0, 7400.0, np.nan, 710.0],
            "balance": [1000.0, 2000.0, 0.0, 500.0, 3000.0, 250.0],
            "limit": [5000.0, 0.0, 4000.0, 2000.0, 10000.0, 1000.0],
        }
    )


@pytest.fixture
def regression_frame() -> pd.DataFrame:
    """A small, learnable regression problem with a known signal."""
    rng = np.random.default_rng(0)
    size = 200
    feature_a = rng.normal(size=size)
    feature_b = rng.normal(size=size)
    noise = rng.normal(scale=0.1, size=size)
    return pd.DataFrame(
        {
            "feature_a": feature_a,
            "feature_b": feature_b,
            "noise_only": rng.normal(size=size),
            "target": 3 * feature_a - 2 * feature_b + noise,
        }
    )


@pytest.fixture
def classification_frame() -> pd.DataFrame:
    """A small, separable binary classification problem."""
    rng = np.random.default_rng(1)
    size = 240
    feature_a = rng.normal(size=size)
    feature_b = rng.normal(size=size)
    label = ((2 * feature_a + feature_b) > 0).astype(int)
    return pd.DataFrame({"feature_a": feature_a, "feature_b": feature_b, "target": label})


@pytest.fixture
def review_frame() -> pd.DataFrame:
    """Star-rated reviews with enough repetition for a real TF-IDF vocabulary.

    Small enough to stay fast, large enough that ``min_df=2`` keeps terms and a
    stratified split leaves both classes on each side - the conditions the real
    project runs under.
    """
    positive = [
        "Absolutely wonderful food and the service was great",
        "Really good pasta and wonderful service, would return",
        "Best brunch in town, wonderful coffee and great staff",
        "Great food, great service, wonderful evening",
        "The pasta was delicious and the staff were wonderful",
        "Delicious food and great coffee, highly recommend",
        "Wonderful evening, delicious pasta, great value",
        "Great staff and delicious brunch, will return",
        "Delicious coffee and wonderful service every time",
        "Good food, great atmosphere, wonderful staff",
    ]
    negative = [
        "Terrible food and the service was awful",
        "Awful pasta and rude staff, never again",
        "Cold food, slow service, terrible experience",
        "Rude staff and dirty tables, awful evening",
        "The pasta was cold and the service terrible",
        "Awful coffee and rude staff, would not return",
        "Terrible evening, cold food, slow service",
        "Dirty tables and awful food, never again",
        "Slow service and terrible coffee every time",
        "Bad food, rude staff, awful atmosphere",
    ]
    neutral = ["It was okay, nothing special about it", "Average food, average service"]

    return pd.DataFrame(
        {
            "stars": [5] * 5 + [4] * 5 + [1] * 5 + [2] * 5 + [3] * 2,
            "text": positive + negative + neutral,
        }
    )
