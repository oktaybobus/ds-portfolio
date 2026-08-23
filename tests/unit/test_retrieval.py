"""Unit tests for document retrieval."""

from __future__ import annotations

from pathlib import Path

import pytest

from dsjourney import retrieval


def test_chunk_text_returns_the_whole_text_when_short() -> None:
    assert retrieval.chunk_text("one two three", size=100) == ["one two three"]


def test_chunk_text_windows_with_overlap() -> None:
    words = " ".join(str(n) for n in range(100))
    chunks = retrieval.chunk_text(words, size=40, overlap=10)
    assert len(chunks) > 1
    first, second = chunks[0].split(), chunks[1].split()
    assert first[-10:] == second[:10]  # the overlap is real


def test_chunk_text_rejects_overlap_that_never_advances() -> None:
    with pytest.raises(ValueError, match="must be smaller than size"):
        retrieval.chunk_text("a b c", size=10, overlap=10)


def test_chunk_text_rejects_a_bad_size() -> None:
    with pytest.raises(ValueError, match="size must be positive"):
        retrieval.chunk_text("a b c", size=0)


def test_chunk_text_handles_empty_input() -> None:
    assert retrieval.chunk_text("") == []


def test_chunk_corpus_labels_every_chunk(tiny_corpus: dict[str, str]) -> None:
    chunks = retrieval.chunk_corpus(tiny_corpus, size=30, overlap=5)
    assert {chunk.document_id for chunk in chunks} == set(tiny_corpus)
    assert all(chunk.chunk_id.startswith(chunk.document_id) for chunk in chunks)


def test_load_corpus_reads_a_directory(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha text", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta text", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")

    corpus = retrieval.load_corpus(tmp_path)
    assert set(corpus) == {"a", "b"}  # the blank file is skipped


def test_load_corpus_reports_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no corpus directory"):
        retrieval.load_corpus(tmp_path / "nowhere")


def test_load_corpus_reports_an_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no readable"):
        retrieval.load_corpus(tmp_path)


def test_index_finds_the_right_document(tiny_corpus: dict[str, str]) -> None:
    index = retrieval.RetrievalIndex().build(
        retrieval.chunk_corpus(tiny_corpus, size=30, overlap=5), components=8
    )
    hits = index.search_documents("cables slung between towers", k=1)
    assert hits.iloc[0]["document_id"] == "bridges"


def test_index_reports_its_shape(tiny_corpus: dict[str, str]) -> None:
    index = retrieval.RetrievalIndex().build(retrieval.chunk_corpus(tiny_corpus), components=4)
    assert index.is_built
    assert index.document_count == 3


def test_search_before_build_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="build\\(\\) must be called"):
        retrieval.RetrievalIndex().search("anything")


def test_build_rejects_an_empty_corpus() -> None:
    with pytest.raises(ValueError, match="zero chunks"):
        retrieval.RetrievalIndex().build([])


def test_search_returns_nothing_for_a_blank_query(tiny_corpus: dict[str, str]) -> None:
    index = retrieval.RetrievalIndex().build(retrieval.chunk_corpus(tiny_corpus), components=4)
    assert index.search("   ").empty


def test_search_documents_deduplicates(tiny_corpus: dict[str, str]) -> None:
    """Chunk-level results let one long article fill every slot."""
    index = retrieval.RetrievalIndex().build(
        retrieval.chunk_corpus(tiny_corpus, size=20, overlap=5), components=8
    )
    documents = index.search_documents("whale songs travel through deep water", k=3)
    assert documents["document_id"].is_unique


def test_build_probes_are_traceable(tiny_corpus: dict[str, str]) -> None:
    probes = retrieval.build_probes(tiny_corpus, count=20, words=6)
    assert len(probes) == 20
    for probe in probes:
        assert probe.document_id in tiny_corpus
        assert len(probe.query.split()) <= 6


def test_build_probes_is_deterministic(tiny_corpus: dict[str, str]) -> None:
    first = retrieval.build_probes(tiny_corpus, count=10, seed=1)
    second = retrieval.build_probes(tiny_corpus, count=10, seed=1)
    assert [p.query for p in first] == [p.query for p in second]


def test_build_probes_on_an_empty_corpus() -> None:
    assert retrieval.build_probes({}, count=5) == []


def test_evaluate_reports_both_levels_and_cost(tiny_corpus: dict[str, str]) -> None:
    """Document recall, passage hit rate and context size are all needed.

    Judging chunk size on document recall alone always recommends bigger
    chunks; the context cost is what makes the trade visible.
    """
    index = retrieval.RetrievalIndex().build(
        retrieval.chunk_corpus(tiny_corpus, size=30, overlap=6), components=8
    )
    probes = retrieval.build_probes(tiny_corpus, count=20, words=8)
    metrics = retrieval.evaluate_retrieval(index, probes, k=3)

    assert 0.0 <= metrics["recall_at_1"] <= 1.0
    assert 0.0 <= metrics["passage_at_3"] <= 1.0
    assert metrics["context_words"] > 0
    assert metrics["hits_per_1k_words"] >= 0


def test_evaluate_handles_no_probes(tiny_corpus: dict[str, str]) -> None:
    index = retrieval.RetrievalIndex().build(retrieval.chunk_corpus(tiny_corpus), components=4)
    assert retrieval.evaluate_retrieval(index, [])["probes"] == 0.0


def test_larger_chunks_cost_more_context(tiny_corpus: dict[str, str]) -> None:
    """The trade the two-level metric exists to expose."""
    probes = retrieval.build_probes(tiny_corpus, count=15, words=8)
    table = retrieval.compare_chunk_sizes(tiny_corpus, probes, candidates=((20, 5), (80, 10)), k=3)
    small = table[table["chunk_words"] == 20].iloc[0]
    large = table[table["chunk_words"] == 80].iloc[0]
    assert large["context_words"] > small["context_words"]


def test_available_models_lists_the_registry() -> None:
    assert "lsa" in retrieval.available_models()
