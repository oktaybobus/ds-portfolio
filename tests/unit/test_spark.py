"""Tests for the Spark helpers.

The ones that need a JVM are marked ``needs_spark`` and skipped when there is
none; the rest run everywhere. Each defect the module was written to prevent has
a named test here, so the fix cannot be undone quietly.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from dsjourney import evaluate
from dsjourney import spark as dsspark


def test_majority_baseline_is_the_largest_class_share() -> None:
    labels = np.array([0] * 65 + [1] * 35)
    assert dsspark.majority_baseline(labels) == pytest.approx(0.65)


def test_majority_baseline_on_a_balanced_target() -> None:
    assert dsspark.majority_baseline([0, 1, 0, 1]) == pytest.approx(0.5)


def test_timed_reports_elapsed_seconds_and_throughput() -> None:
    result, timing = dsspark.timed("noop", lambda: 21 * 2, rows=100)
    assert result == 42
    assert timing.seconds >= 0
    assert timing.label == "noop"
    assert timing.rows_per_second > 0


def test_timing_throughput_is_zero_without_a_row_count() -> None:
    assert dsspark.Timing(label="x", seconds=1.0).rows_per_second == 0.0


def test_available_models_lists_the_configured_algorithms() -> None:
    assert set(dsspark.available_models()) == {"spark_bfs", "spark_logistic"}


def test_java_home_matches_spark_availability() -> None:
    """Whatever the machine has, the two answers must be consistent."""
    if dsspark.java_home() is None or not dsspark.pyspark_installed():
        assert not dsspark.spark_available()
    else:
        assert dsspark.spark_available()


def test_install_hint_names_both_halves_of_the_requirement() -> None:
    assert "--extra spark" in dsspark.INSTALL_HINT
    assert "openjdk" in dsspark.INSTALL_HINT


@pytest.fixture(scope="module")
def spark() -> Iterator[object]:
    """One session for the whole module; starting a JVM per test costs seconds."""
    if not dsspark.spark_available():
        pytest.skip("no Spark")
    with dsspark.session("dsjourney-tests", cores="2", shuffle_partitions=2) as session:
        yield session


@pytest.mark.needs_spark
def test_session_stops_itself_on_exit() -> None:
    from pyspark.sql import SparkSession

    with dsspark.session("stop-check", cores="1", shuffle_partitions=1):
        assert SparkSession.getActiveSession() is not None
    assert SparkSession.getActiveSession() is None


@pytest.mark.needs_spark
def test_read_text_lines_honours_a_non_utf8_encoding(spark: object, tmp_path: Path) -> None:
    """The defect: Spark's text reader assumes UTF-8 and replaces what fails.

    Marvel-names.txt and book.txt are both cp1252/Latin-1 in this course tree,
    so reading them as UTF-8 corrupts words rather than raising.
    """
    path = tmp_path / "latin1.txt"
    path.write_bytes("café\nrésumé\n".encode("latin-1"))

    correct = dsspark.read_text_lines(spark, path, encoding="latin-1").collect()
    assert [row.line for row in correct] == ["café", "résumé"]

    corrupted = dsspark.read_text_lines(spark, path, encoding="utf-8").collect()
    assert [row.line for row in corrupted] != ["café", "résumé"]


@pytest.mark.needs_spark
def test_word_frequencies_ranks_by_count(spark: object, tmp_path: Path) -> None:
    path = tmp_path / "book.txt"
    path.write_text("the cat sat\nthe cat ran\nthe end\n", encoding="utf-8")

    counts = {
        row.word: row["count"]
        for row in dsspark.word_frequencies(
            dsspark.read_text_lines(spark, path), min_length=1
        ).collect()
    }
    assert counts["the"] == 3
    assert counts["cat"] == 2
    assert counts["end"] == 1


@pytest.mark.needs_spark
def test_adjacency_aggregates_a_node_split_over_several_lines(
    spark: object, tmp_path: Path
) -> None:
    """The defect: mapping each line to a degree undercounts multi-line nodes.

    74 of the 6,486 Marvel characters are split this way.
    """
    path = tmp_path / "graph.txt"
    path.write_text("1 2 3\n1 4\n2 1\n", encoding="utf-8")

    degrees = {
        row.id: row.degree
        for row in dsspark.degree_table(
            dsspark.adjacency_from_lines(dsspark.read_text_lines(spark, path))
        ).collect()
    }
    assert degrees == {1: 3, 2: 1}


@pytest.mark.needs_spark
def test_adjacency_keeps_a_node_with_no_neighbours(spark: object, tmp_path: Path) -> None:
    path = tmp_path / "graph.txt"
    path.write_text("1 2\n3\n", encoding="utf-8")

    degrees = {
        row.id: row.degree
        for row in dsspark.degree_table(
            dsspark.adjacency_from_lines(dsspark.read_text_lines(spark, path))
        ).collect()
    }
    assert degrees == {1: 1, 3: 0}


@pytest.mark.needs_spark
def test_bfs_finds_distances_and_omits_unreachable_nodes(spark: object, tmp_path: Path) -> None:
    path = tmp_path / "graph.txt"
    path.write_text("1 2\n2 3\n3 1\n7 8\n8 7\n", encoding="utf-8")

    adjacency = dsspark.adjacency_from_lines(dsspark.read_text_lines(spark, path))
    distances = {
        row.id: row.distance for row in dsspark.bfs_distances(adjacency, 1, max_depth=5).collect()
    }
    assert distances == {1: 0, 2: 1, 3: 2}
    assert 7 not in distances
    assert 8 not in distances


@pytest.mark.needs_spark
def test_bfs_stops_at_max_depth(spark: object, tmp_path: Path) -> None:
    path = tmp_path / "chain.txt"
    path.write_text("1 2\n2 3\n3 4\n4 5\n", encoding="utf-8")

    adjacency = dsspark.adjacency_from_lines(dsspark.read_text_lines(spark, path))
    distances = dsspark.bfs_distances(adjacency, 1, max_depth=2).collect()
    assert max(row.distance for row in distances) == 2


@pytest.mark.needs_spark
def test_spark_and_sklearn_score_identical_predictions_identically(spark: object) -> None:
    """The defect: the notebook printed ROC AUC under the label 'Accuracy'.

    Scoring the same predictions with both libraries proves the Spark numbers
    mean what their names say - and that accuracy and AUC are not each other.
    """
    from pyspark.ml.functions import array_to_vector, vector_to_array

    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=200)
    # Deliberately imperfect scores, so accuracy and AUC land apart.
    scores = np.clip(labels * 0.35 + rng.random(200) * 0.6, 0, 1)

    rows = [
        (float(label), float(score >= 0.5), [float(1 - score), float(score)])
        for label, score in zip(labels, scores, strict=True)
    ]
    frame = spark.createDataFrame(  # type: ignore[attr-defined]
        rows, "label DOUBLE, prediction DOUBLE, probability ARRAY<DOUBLE>"
    ).withColumn("probability", array_to_vector("probability"))

    spark_scores = dsspark.binary_classification_scores(frame)

    collected = frame.withColumn("p", vector_to_array("probability")).toPandas()
    sklearn_scores = evaluate.classification_scores(
        collected["label"],
        collected["prediction"],
        y_proba=np.stack(collected["p"].to_numpy()),
    )

    for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert spark_scores[name] == pytest.approx(sklearn_scores[name], abs=1e-6), name

    assert spark_scores["accuracy"] != pytest.approx(spark_scores["roc_auc"], abs=0.01)
