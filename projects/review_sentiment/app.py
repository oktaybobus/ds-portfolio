"""Streamlit demo for restaurant review sentiment.

Run with ``dsj serve review_sentiment``. Paste a review and see the predicted
label, the confidence, and which terms the linear model weights most heavily.
"""

from __future__ import annotations

import streamlit as st

from dsjourney.artifacts import load_bundle
from dsjourney.text import top_features
from projects.review_sentiment import pipeline

EXAMPLES = [
    "The pasta was incredible and the staff could not have been friendlier.",
    "Waited 40 minutes for cold food. The waiter never came back.",
    "Decent enough, but nothing I would go out of my way for.",
]

st.set_page_config(page_title="Review Sentiment", page_icon="=", layout="centered")


@st.cache_resource
def _load():
    """Load the trained TF-IDF pipeline once per session."""
    return load_bundle("review_sentiment")


def main() -> None:
    st.title("Restaurant Review Sentiment")
    st.caption("TF-IDF over unigrams and bigrams into a logistic regression.")

    try:
        bundle = _load()
    except FileNotFoundError:
        st.error("No trained model found. Run `uv run python projects/review_sentiment/train.py`.")
        return

    with st.sidebar:
        st.subheader("Model")
        st.metric("F1", f"{bundle.metrics.get('f1', 0):.3f}")
        st.metric("ROC AUC", f"{bundle.metrics.get('roc_auc', 0):.3f}")
        st.caption(
            f"{bundle.extra.get('vocabulary_size', '?')} terms, "
            f"{bundle.extra.get('train_rows', '?')} training reviews"
        )

    example = st.selectbox("Try an example", ["(write your own)", *EXAMPLES])
    default = "" if example.startswith("(") else example
    review = st.text_area("Review text", value=default, height=140)

    if st.button("Analyse", type="primary", use_container_width=True) and review.strip():
        result = pipeline.predict_sentiment(bundle.model, review)
        if result["label"] == "positive":
            st.success(f"Positive ({result['confidence']:.1%} confidence)")
        elif result["label"] == "negative":
            st.error(f"Negative ({result['confidence']:.1%} confidence)")
        else:
            st.warning("Nothing left after cleaning - try a longer review.")
        with st.expander("Cleaned text the model actually saw"):
            st.code(result["cleaned"] or "(empty)")

    with st.expander("Most influential terms"):
        st.dataframe(top_features(bundle.model, limit=15), use_container_width=True)


main()
