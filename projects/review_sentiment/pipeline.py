"""Text preparation for restaurant review sentiment.

Two steps from the source notebook were dropped on purpose:

- **Language detection.** ``langdetect`` was run row by row to keep only English
  reviews. On this dataset that is minutes of work to remove a fraction of a
  percent of rows, and it makes the pipeline non-deterministic across versions.
- **TextBlob lemmatisation.** It requires a corpus download at import time,
  which breaks hermetic CI runs and slows the Docker build, for a change in F1
  well inside the noise of the split.

What remains - lowercasing, stripping punctuation and digits, removing
stopwords, TF-IDF over unigrams and bigrams - is what carried the signal.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from dsjourney import text
from dsjourney.config import load_project_config
from dsjourney.datasets import load_dataset

CONFIG = load_project_config("review_sentiment")

TEXT_COLUMN = "text"
STAR_COLUMN = "stars"
LABEL_COLUMN = "sentiment"

NEUTRAL_STARS = 3
POSITIVE_THRESHOLD = 4

# Documented request body for the API and the CLI.
EXAMPLE_INPUT = {"text": "The pasta was incredible and the staff could not have been friendlier."}


def load_raw() -> pd.DataFrame:
    """Read the review dataset."""
    return load_dataset(CONFIG)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a two-column frame of cleaned text and a binary sentiment label.

    3-star reviews are removed rather than assigned to a class: they are
    genuinely mixed, and forcing them either way teaches the model to guess.
    """
    polarised = frame[frame[STAR_COLUMN] != NEUTRAL_STARS].dropna(subset=[TEXT_COLUMN])
    return pd.DataFrame(
        {
            TEXT_COLUMN: text.normalise_series(polarised[TEXT_COLUMN]),
            LABEL_COLUMN: (polarised[STAR_COLUMN] >= POSITIVE_THRESHOLD).astype(int),
        }
    ).query(f"{TEXT_COLUMN}.str.len() > 0")


def prepare_input(payload: dict[str, Any]) -> pd.DataFrame:
    """Normalise one review for scoring by the saved pipeline."""
    return pd.DataFrame({TEXT_COLUMN: [text.normalise(payload.get("text", ""))]})


def predict_sentiment(model: Any, review: str) -> dict[str, Any]:
    """Score a single raw review and return the label with its confidence."""
    cleaned = text.normalise(review)
    if not cleaned:
        return {"label": "unknown", "confidence": 0.0, "cleaned": ""}

    documents = pd.Series([cleaned])
    prediction = int(model.predict(documents)[0])
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(documents)[0][prediction])
    return {
        "label": "positive" if prediction == 1 else "negative",
        "confidence": confidence,
        "cleaned": cleaned,
    }
