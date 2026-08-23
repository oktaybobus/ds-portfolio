"""Loading and feature construction for the MovieLens recommender.

The source notebooks produced similarity lists and stopped. Nothing was held
out, so "films correlated with Star Wars" could not be told apart from noise -
and in fact the top of that list was dominated by obscure titles rated by three
people, because no minimum-support floor was applied.

This module keeps the timestamp column the notebooks discarded, which is what
lets :func:`dsjourney.recommend.split_ratings` withhold each user's most recent
ratings and turn the whole thing into something measurable.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from dsjourney import recommend
from dsjourney.config import load_project_config
from dsjourney.datasets import DatasetNotFoundError
from dsjourney.paths import project_data_dir

CONFIG = load_project_config("movie_recommender")

RATINGS_FILE = "u.data"
ITEMS_FILE = "u.item"

# Star Wars (1977) - the seed title the source notebook used for its similarity
# demo, kept so the two can be compared directly.
DEMO_ITEM_ID = 50


def _path(file: str):  # type: ignore[no-untyped-def]
    """Resolve a data file, with the fetch command in the error when absent."""
    path = project_data_dir(CONFIG.name) / file
    if not path.is_file():
        raise DatasetNotFoundError(
            f"{file} not found at {path}. "
            f"Run: uv run python scripts/fetch_assets.py --project {CONFIG.name}"
        )
    return path


def load_raw() -> pd.DataFrame:
    """Read the ratings log."""
    return recommend.load_ratings(_path(RATINGS_FILE))


def load_items() -> pd.DataFrame:
    """Read the film catalogue with its genre flags."""
    return recommend.load_items(_path(ITEMS_FILE))


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the ratings log unchanged.

    A recommender trains on interactions rather than a feature matrix; the
    modelling entry point is :func:`dsjourney.recommend.fit_svd`.
    """
    return frame


def prepare_input(payload: dict[str, Any]) -> pd.DataFrame:
    """Not applicable: recommendations are produced per user, not per record."""
    raise NotImplementedError(
        "movie_recommender ranks items for a user id. "
        "Use: python projects/movie_recommender/recommend_for.py --user 1"
    )


def catalogue_summary(ratings: pd.DataFrame, items: pd.DataFrame) -> dict[str, int]:
    """Return the headline shape of the interaction log."""
    return {
        "ratings": len(ratings),
        "users": int(ratings["user_id"].nunique()),
        "films": int(ratings["item_id"].nunique()),
        "catalogue_size": len(items),
    }
