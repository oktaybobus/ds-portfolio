"""Streamlit demo for article search.

Run with ``dsj serve article_search``. The index is built once per session -
390 articles chunk and embed in a couple of seconds, so there is no bundle to
load and nothing can go stale.
"""

from __future__ import annotations

import streamlit as st

from dsjourney import retrieval
from projects.article_search import pipeline
from projects.article_search.search import WEAK_MATCH

st.set_page_config(page_title="Article Search", page_icon="=", layout="wide")


@st.cache_resource
def _index():
    """Load the corpus and build the index once."""
    documents = pipeline.load_documents()
    return documents, pipeline.build_index(documents)


@st.cache_data
def _quality(_index_key: int, probes: int = 150):
    """Score the index so the page can show what it is worth."""
    documents, index = _index()
    return retrieval.evaluate_retrieval(index, retrieval.build_probes(documents, count=probes), k=5)


def main() -> None:
    st.title("Wikipedia Article Search")
    st.caption("Latent semantic analysis over 390 human-rights and conflict articles.")

    try:
        _documents, index = _index()
    except FileNotFoundError as error:
        st.error(str(error))
        return

    with st.sidebar:
        st.subheader("Index")
        st.metric("Documents", index.document_count)
        st.metric("Chunks", f"{len(index.chunks):,}")
        metrics = _quality(len(index.chunks))
        st.metric("MRR", f"{metrics['mrr']:.3f}")
        st.metric("Recall@5", f"{metrics['recall_at_5']:.3f}")
        st.caption(
            "Scored on sentences taken out of known articles: can the index find "
            "its way back to the source?"
        )

    query = st.selectbox("Try a query", ["(write your own)", *pipeline.EXAMPLE_QUERIES])
    default = "" if query.startswith("(") else query
    text = st.text_input("Search", value=default)
    top_k = st.slider("Results", 1, 10, 5)

    if not text.strip():
        return

    hits = index.search_documents(text, k=top_k)
    if hits.empty:
        st.warning("Nothing matched.")
        return

    best = float(hits.iloc[0]["score"])
    if best < WEAK_MATCH:
        st.warning(
            f"The best score is {best:.3f}, below {WEAK_MATCH} - 94% of answerable "
            "queries clear that line, so the corpus probably has no answer here."
        )
    else:
        st.caption(
            f"Top score {best:.3f}. Scores are not calibrated confidence: an "
            "out-of-domain query can still score high, so read the passage."
        )

    for _, row in hits.iterrows():
        with st.expander(f"{row['score']:.3f}  ·  {row['document_id'].replace('_', ' ')}"):
            st.write(row["text"])
            st.caption(f"chunk {row['position']} of {row['document_id']}")


main()
