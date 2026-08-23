"""Text preparation for the NLP projects.

Deliberately dependency-light. The source notebook used NLTK stopwords,
TextBlob lemmatisation and langdetect language filtering; all three need a
corpus download at import time, which makes a CI run non-hermetic and a Docker
build slow. The stopword list is inlined here instead, and the cleaning steps
that actually moved the metric are kept.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# The standard English stopword list, inlined so no corpus download is required.
STOPWORDS: frozenset[str] = frozenset(
    [
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "a",
        "an",
        "the",
        "and",
        "but",
        "if",
        "or",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "can",
        "will",
        "just",
        "don",
        "should",
        "now",
    ]
)

_PUNCTUATION = re.compile(r"[^\w\s]")
_DIGITS = re.compile(r"\d+")
_WHITESPACE = re.compile(r"\s+")


def clean_text(value: Any) -> str:
    """Lowercase a review and strip punctuation, digits and repeated whitespace."""
    if not isinstance(value, str):
        return ""
    lowered = value.lower()
    without_punctuation = _PUNCTUATION.sub(" ", lowered)
    without_digits = _DIGITS.sub(" ", without_punctuation)
    return _WHITESPACE.sub(" ", without_digits).strip()


def remove_stopwords(value: str, *, stopwords: frozenset[str] = STOPWORDS) -> str:
    """Drop stopwords from an already-cleaned string."""
    return " ".join(word for word in value.split() if word not in stopwords)


def normalise(value: Any) -> str:
    """Clean and de-stopword a raw document in one step."""
    return remove_stopwords(clean_text(value))


def normalise_series(series: pd.Series) -> pd.Series:
    """Apply :func:`normalise` across a column of documents."""
    return series.map(normalise)


def build_text_pipeline(
    estimator: BaseEstimator,
    *,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
) -> Pipeline:
    """Return a TF-IDF -> estimator pipeline.

    Bundling the vectoriser with the classifier is what makes the saved model
    usable on raw text later; saving them separately is how vocabularies and
    models drift out of sync.
    """
    vectoriser = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
    )
    return Pipeline([("tfidf", vectoriser), ("estimator", estimator)])


def top_features(pipeline: Pipeline, *, limit: int = 20) -> pd.DataFrame:
    """Return the most positive and most negative terms of a linear text model.

    Only meaningful for linear estimators; returns an empty frame otherwise.
    """
    estimator = pipeline.named_steps.get("estimator")
    vectoriser = pipeline.named_steps.get("tfidf")
    coefficients = getattr(estimator, "coef_", None)
    if coefficients is None or vectoriser is None:
        return pd.DataFrame(columns=["term", "weight", "direction"])

    weights = pd.Series(coefficients[0], index=vectoriser.get_feature_names_out())
    positive = weights.nlargest(limit).rename("weight").to_frame().assign(direction="positive")
    negative = weights.nsmallest(limit).rename("weight").to_frame().assign(direction="negative")
    combined = pd.concat([positive, negative])
    combined.index.name = "term"
    return combined.reset_index()
