"""Corpus loading for the article search project.

The source notebook pushed whole articles into ChromaDB - one vector per
document, some of them 150 KB. A vector averaged over that much text matches
everything a little and nothing precisely, which is why the retrieved passages
in the notebook were topical but rarely the specific paragraph asked about.

Here the corpus is chunked with overlap before indexing, and the chunk size is
chosen by measurement rather than by default - see ``train.py --scan``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from dsjourney import retrieval
from dsjourney.config import load_project_config
from dsjourney.datasets import DatasetNotFoundError
from dsjourney.paths import project_data_dir

CONFIG = load_project_config("article_search")

# The queries the demo offers, chosen to span the corpus rather than to flatter
# it - the last one has no good answer in these articles.
EXAMPLE_QUERIES = [
    "mass rape by security forces during a search operation",
    "regional charter protecting the rights of peoples in Africa",
    "obligations of an occupying power under the Geneva Conventions",
    "truth and reconciliation commission after apartheid",
    "how do I renew a driving licence",
]


def corpus_directory():  # type: ignore[no-untyped-def]
    """Return the directory holding the article files."""
    return project_data_dir(CONFIG.name)


def load_raw() -> pd.DataFrame:
    """Return one row per article, for the generic CLI commands."""
    documents = load_documents()
    return pd.DataFrame(
        [
            {"document_id": key, "characters": len(text), "words": len(text.split())}
            for key, text in documents.items()
        ]
    )


def load_documents(*, limit: int | None = None) -> dict[str, str]:
    """Read the article corpus as ``{document_id: text}``."""
    try:
        return retrieval.load_corpus(corpus_directory(), limit=limit)
    except FileNotFoundError as error:
        raise DatasetNotFoundError(
            f"{error}. Run: uv run python scripts/fetch_assets.py --project {CONFIG.name}"
        ) from error


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the article summary unchanged; retrieval indexes text, not rows."""
    return frame


def build_index(
    documents: dict[str, str] | None = None, **overrides: Any
) -> retrieval.RetrievalIndex:
    """Chunk the corpus and build a searchable index using the config settings."""
    params = {**CONFIG.model.params, **overrides}
    corpus = documents if documents is not None else load_documents()
    chunks = retrieval.chunk_corpus(
        corpus, size=int(params["chunk_words"]), overlap=int(params["overlap"])
    )
    return retrieval.RetrievalIndex().build(chunks, components=int(params["components"]))


def prepare_input(payload: dict[str, Any]) -> pd.DataFrame:
    """Not applicable: retrieval answers a query, it does not score a record."""
    raise NotImplementedError(
        "article_search answers text queries. "
        "Use: python projects/article_search/search.py 'your question'"
    )
