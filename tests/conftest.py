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


def _spark_runnable() -> bool:
    """True when PySpark is installed and a JVM is present to run it."""
    from dsjourney.spark import spark_available

    return spark_available()


def _rl_runnable() -> bool:
    """True when the gymnasium environments are installed."""
    from dsjourney.rl import gymnasium_installed

    return gymnasium_installed()


def _deeprl_runnable() -> bool:
    """True when stable-baselines3 and torch are installed."""
    from dsjourney.rl import sb3_installed

    return _rl_runnable() and sb3_installed()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip tests whose optional runtime is not installed.

    The marker alone only lets a caller deselect them; without this hook a plain
    `pytest` run on a machine with no TensorFlow reports failures for an
    environment choice rather than a defect. ``needs_spark`` works the same way.
    """
    if not TENSORFLOW_INSTALLED:
        skip = pytest.mark.skip(reason="TensorFlow not installed (uv sync --extra dl)")
        for item in items:
            if "needs_dl" in item.keywords:
                item.add_marker(skip)

    # Same reasoning for Spark, which additionally needs a Java runtime the
    # Python environment cannot install for itself.
    if not _spark_runnable():
        skip = pytest.mark.skip(
            reason="PySpark or a Java runtime is missing (uv sync --extra spark; brew install openjdk@17)"
        )
        for item in items:
            if "needs_spark" in item.keywords:
                item.add_marker(skip)

    for marker, runnable, hint in (
        ("needs_rl", _rl_runnable, "gymnasium not installed (uv sync --extra rl)"),
        (
            "needs_deeprl",
            _deeprl_runnable,
            "stable-baselines3 not installed (uv sync --extra deeprl)",
        ),
    ):
        if runnable():
            continue
        skip = pytest.mark.skip(reason=hint)
        for item in items:
            if marker in item.keywords:
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


@pytest.fixture
def seasonal_series() -> pd.Series:
    """Two years of daily data with a clear yearly cycle, a trend and light noise."""
    index = pd.date_range("2020-01-01", periods=730, freq="D")
    rng = np.random.default_rng(7)
    day_of_year = np.arange(730) % 365
    seasonal = 10 * np.sin(2 * np.pi * day_of_year / 365)
    trend = np.linspace(0, 5, 730)
    return pd.Series(20 + seasonal + trend + rng.normal(scale=0.5, size=730), index=index)


@pytest.fixture
def ratings_log() -> pd.DataFrame:
    """A small interaction log with two clear taste groups and a popularity skew.

    Users 0-9 like items 1-3, users 10-19 like items 4-6, and item 99 is rated
    5 by exactly one person - the pattern that breaks unfiltered top-N ranking.
    """
    rows = []
    timestamp = 1_000_000
    for user in range(20):
        favourites = (1, 2, 3) if user < 10 else (4, 5, 6)
        others = (4, 5, 6) if user < 10 else (1, 2, 3)
        for item in others:
            rows.append((user, item, 2, timestamp))
            timestamp += 10
        for item in (7, 8):
            rows.append((user, item, 3, timestamp))
            timestamp += 10
        # Favourites are rated last so that a chronological holdout contains
        # liked items; otherwise precision@k has no relevant set to score.
        for item in favourites:
            rows.append((user, item, 5, timestamp))
            timestamp += 10
    rows.append((0, 99, 5, timestamp))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


@pytest.fixture
def item_catalogue() -> pd.DataFrame:
    """A catalogue matching ``ratings_log`` with two-genre structure."""
    from dsjourney.recommend import GENRE_NAMES

    rows = []
    for item_id in [1, 2, 3, 4, 5, 6, 7, 8, 99]:
        flags = dict.fromkeys(GENRE_NAMES, 0)
        flags["action" if item_id <= 3 else "drama"] = 1
        rows.append({"item_id": item_id, "title": f"Film {item_id}", **flags})
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_corpus() -> dict[str, str]:
    """Three short documents on clearly separate topics."""
    return {
        "solar": (
            "Photovoltaic panels convert sunlight directly into electricity. "
            "A solar farm needs land, sunshine and a grid connection. "
            "Panel efficiency has risen steadily while installation cost has fallen. "
            "Storage remains the limiting factor for overnight supply. "
        )
        * 4,
        "whales": (
            "Humpback whales migrate thousands of kilometres between feeding and breeding grounds. "
            "Their songs travel far through deep water. "
            "Commercial whaling reduced several populations to near extinction. "
            "Protection measures have allowed a slow recovery. "
        )
        * 4,
        "bridges": (
            "A suspension bridge carries its deck from cables slung between towers. "
            "The cables transfer load to anchorages at either end. "
            "Wind-induced oscillation destroyed the Tacoma Narrows bridge. "
            "Modern decks are shaped to shed vortices. "
        )
        * 4,
    }
