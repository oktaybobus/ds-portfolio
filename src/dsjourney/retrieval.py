"""Document retrieval: chunking, indexing, search, and measuring the result.

The source notebook loaded 390 articles into ChromaDB and ran one query -
"Turkey human rights violations?" - then looked at the five results and moved
on. That tells you the pipeline runs. It does not tell you whether the right
document came back, which is the only question a retrieval system exists to
answer.

Two things are added here. Documents are **chunked** with overlap, and
retrieval is **scored**: :func:`build_probes` takes sentences out of known
articles and asks the index to find its way home, over hundreds of queries
instead of one impression.

The scoring is deliberately at two levels. Measuring only whether the right
*article* comes back rewards ever-larger chunks - in the limit, one vector per
document, which is exactly what the notebook did. Measuring whether the returned
*passage* contains the sentence moves the opposite way. Both numbers are
reported, because a chunk size chosen on one of them alone is chosen on a metric
that was not measuring the thing it justified.

The default index is TF-IDF projected through SVD - latent semantic analysis -
so it captures word co-occurrence without downloading a transformer. Pass a
different ``embedder`` for dense embeddings when the extra weight is worth it.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

DEFAULT_CHUNK_WORDS = 180
DEFAULT_OVERLAP_WORDS = 40
MIN_CHUNK_WORDS = 20

# Below this many chunks, a document-frequency floor of 2 prunes the vocabulary
# to nothing. See RetrievalIndex.build.
SMALL_CORPUS_CHUNKS = 20
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """One passage of one document."""

    document_id: str
    position: int
    text: str

    @property
    def chunk_id(self) -> str:
        """Stable identifier of the passage within the corpus."""
        return f"{self.document_id}#{self.position}"


@dataclass(frozen=True)
class Probe:
    """A query with the document it was taken from - the ground truth."""

    query: str
    document_id: str


def load_corpus(
    directory: Path, *, pattern: str = "*.txt", limit: int | None = None
) -> dict[str, str]:
    """Read a directory of text files into ``{document_id: text}``.

    The file stem is the document id, which keeps the mapping readable in
    search results instead of turning every hit into an opaque index.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"no corpus directory at {directory}")

    documents: dict[str, str] = {}
    for path in sorted(directory.glob(pattern))[:limit]:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            documents[path.stem] = text
    if not documents:
        raise FileNotFoundError(f"no readable {pattern} files under {directory}")
    return documents


def chunk_text(
    text: str,
    *,
    size: int = DEFAULT_CHUNK_WORDS,
    overlap: int = DEFAULT_OVERLAP_WORDS,
) -> list[str]:
    """Split text into overlapping word windows.

    Overlap exists because a passage that answers a question often straddles a
    boundary; without it the sentence is split across two vectors and neither
    one matches it well. ``passage_at_k`` in :func:`evaluate_retrieval` is the
    metric that shows this happening.

    Raises:
        ValueError: when overlap is not smaller than the window, which would
            never advance.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")

    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []

    step = size - overlap
    windows = [" ".join(words[start : start + size]) for start in range(0, len(words), step)]
    return [w for w in windows if len(w.split()) >= min(MIN_CHUNK_WORDS, size)]


def chunk_corpus(
    documents: dict[str, str],
    *,
    size: int = DEFAULT_CHUNK_WORDS,
    overlap: int = DEFAULT_OVERLAP_WORDS,
) -> list[Chunk]:
    """Chunk every document in a corpus."""
    chunks: list[Chunk] = []
    for document_id, text in documents.items():
        for position, passage in enumerate(chunk_text(text, size=size, overlap=overlap)):
            chunks.append(Chunk(document_id=document_id, position=position, text=passage))
    return chunks


def build_embedder(
    *, components: int = 256, max_features: int = 50_000, min_df: int = 2
) -> Pipeline:
    """Return a TF-IDF -> SVD -> L2 normalise pipeline (latent semantic analysis).

    Normalising afterwards is what makes a dot product a cosine similarity, so
    search can be one matrix multiply rather than a pairwise loop.

    Args:
        min_df: Minimum document frequency for a term. Two is right for a real
            corpus and fatal for a small one - on a handful of chunks it prunes
            every term and the vectoriser raises. :meth:`RetrievalIndex.build`
            picks it from the corpus size rather than leaving the caller to
            discover this.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=max_features,
                    ngram_range=(1, 2),
                    min_df=min_df,
                    sublinear_tf=True,
                ),
            ),
            ("svd", TruncatedSVD(n_components=components, random_state=42)),
            ("normalise", Normalizer(copy=False)),
        ]
    )


@dataclass
class RetrievalIndex:
    """A searchable index over chunked documents."""

    chunks: list[Chunk] = field(default_factory=list)
    embedder: Pipeline | None = None
    _vectors: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    @property
    def is_built(self) -> bool:
        """True once :meth:`build` has run."""
        return self._vectors.size > 0

    @property
    def document_count(self) -> int:
        """Number of distinct documents behind the index."""
        return len({chunk.document_id for chunk in self.chunks})

    def build(self, chunks: Sequence[Chunk], *, components: int = 256) -> RetrievalIndex:
        """Embed every chunk and store the matrix.

        ``components`` is capped to what the corpus can support - asking for
        more latent dimensions than there are chunks raises inside SVD.
        """
        if not chunks:
            raise ValueError("cannot build an index over zero chunks")

        self.chunks = list(chunks)
        usable = max(min(components, len(self.chunks) - 1), 1)
        # A term has to appear in two chunks to survive min_df=2. On a corpus of
        # three chunks that removes almost everything, so small corpora keep
        # every term instead.
        min_df = 2 if len(self.chunks) >= SMALL_CORPUS_CHUNKS else 1
        self.embedder = build_embedder(components=usable, min_df=min_df)
        self._vectors = np.asarray(
            self.embedder.fit_transform([chunk.text for chunk in self.chunks]), dtype=float
        )
        return self

    def search(self, query: str, *, k: int = 5) -> pd.DataFrame:
        """Return the k most similar chunks, most similar first."""
        if not self.is_built or self.embedder is None:
            raise RuntimeError("build() must be called before search()")
        if not query.strip():
            return pd.DataFrame(columns=["document_id", "position", "score", "text"])

        vector = np.asarray(self.embedder.transform([query]), dtype=float)
        scores = (self._vectors @ vector.T).ravel()
        top = np.argsort(scores)[::-1][:k]

        return pd.DataFrame(
            [
                {
                    "document_id": self.chunks[i].document_id,
                    "position": self.chunks[i].position,
                    "score": float(scores[i]),
                    "text": self.chunks[i].text,
                }
                for i in top
            ]
        )

    def search_documents(self, query: str, *, k: int = 5, pool: int = 30) -> pd.DataFrame:
        """Return the k best *documents*, scored by their strongest chunk.

        A long article can occupy every slot in a chunk-level result list, which
        looks like five hits and is really one. Collapsing to documents is what
        a reader actually wants back.
        """
        hits = self.search(query, k=pool)
        if hits.empty:
            return hits
        best = hits.sort_values("score", ascending=False).drop_duplicates("document_id")
        return best.head(k).reset_index(drop=True)


def build_probes(
    documents: dict[str, str],
    *,
    count: int = 200,
    words: int = 12,
    seed: int = 42,
) -> list[Probe]:
    """Take sentences out of documents to use as queries with known answers.

    A retrieval system that cannot find the article a sentence came from cannot
    find anything. It is a proxy for real user queries, not a substitute - but
    it is a measurement, and one eyeballed query is not.
    """
    generator = random.Random(seed)
    candidates = list(documents.items())
    if not candidates:
        return []

    probes: list[Probe] = []
    attempts = 0
    while len(probes) < count and attempts < count * 20:
        attempts += 1
        document_id, text = generator.choice(candidates)
        sentences = [s for s in _SENTENCE_END.split(text) if len(s.split()) >= words]
        if not sentences:
            continue
        sentence = generator.choice(sentences)
        query = " ".join(sentence.split()[:words])
        probes.append(Probe(query=query, document_id=document_id))
    return probes


def evaluate_retrieval(
    index: RetrievalIndex, probes: Sequence[Probe], *, k: int = 5
) -> dict[str, float]:
    """Score an index at two levels, because they disagree.

    **Document level** - recall@1, recall@k, MRR - asks "did the right article
    come back?". This metric rewards large chunks: the bigger the window, the
    more of an article each vector covers, and the easier the article is to
    find. Taken alone it would recommend one vector per document, which is what
    the source notebook did.

    **Passage level** - ``passage_at_k`` - asks "did the returned text actually
    contain the sentence?". On its own this also favours large chunks, for a
    trivial reason: a wider window is likelier to contain any given sentence.

    **Cost** - ``context_words`` and ``hits_per_1k_words`` - is what makes the
    trade real. A RAG pipeline pastes the retrieved chunks into a prompt and
    pays for every token, so the question is not "can this find the sentence?"
    but "how much irrelevant text comes with it?". Reporting hit rate without
    context size is how a chunking decision gets justified by a number that was
    never measuring the thing it was used for.
    """
    if not probes:
        return {"probes": 0.0}

    hits_at_1 = 0
    hits_at_k = 0
    passage_hits = 0
    reciprocal_ranks = 0.0
    context_words = 0

    for probe in probes:
        documents = index.search_documents(probe.query, k=k)
        ranked = documents["document_id"].tolist() if not documents.empty else []
        if probe.document_id in ranked:
            rank = ranked.index(probe.document_id) + 1
            hits_at_k += 1
            reciprocal_ranks += 1 / rank
            if rank == 1:
                hits_at_1 += 1

        chunks = index.search(probe.query, k=k)
        if not chunks.empty:
            context_words += int(sum(len(text.split()) for text in chunks["text"]))
            if any(probe.query in text for text in chunks["text"]):
                passage_hits += 1

    total = len(probes)
    passage_rate = passage_hits / total
    words_per_query = context_words / total
    return {
        "probes": float(total),
        "recall_at_1": hits_at_1 / total,
        f"recall_at_{k}": hits_at_k / total,
        "mrr": reciprocal_ranks / total,
        f"passage_at_{k}": passage_rate,
        "context_words": words_per_query,
        "hits_per_1k_words": 1000 * passage_rate / words_per_query if words_per_query else 0.0,
    }


INDEXERS = {"lsa": build_embedder}


def available_models() -> list[str]:
    """Return the registered index types."""
    return sorted(INDEXERS)


def compare_chunk_sizes(
    documents: dict[str, str],
    probes: Sequence[Probe],
    *,
    candidates: Iterable[tuple[int, int]] = (
        (120, 20),
        (300, 60),
        (600, 120),
        (1000, 200),
        (2000, 0),
    ),
    k: int = 5,
) -> pd.DataFrame:
    """Score several chunk-size/overlap settings so the choice is evidence-based."""
    rows: list[dict[str, Any]] = []
    for size, overlap in candidates:
        chunks = chunk_corpus(documents, size=size, overlap=overlap)
        index = RetrievalIndex().build(chunks)
        rows.append(
            {
                "chunk_words": size,
                "overlap": overlap,
                "chunks": len(chunks),
                **evaluate_retrieval(index, probes, k=k),
            }
        )
    return pd.DataFrame(rows).sort_values("mrr", ascending=False).reset_index(drop=True)
