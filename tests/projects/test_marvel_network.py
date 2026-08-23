"""Tests for the Marvel co-appearance graph.

The data files are committed, so these assert facts about the real graph rather
than a fixture. Every number here was measured from the shipped files; if a
replacement dataset changes one, the test says so instead of the pipeline
quietly producing a different answer.
"""

from __future__ import annotations

import pytest

from dsjourney import spark as dsspark
from projects.marvel_network import pipeline

HEROES = 6486
CO_APPEARANCE_PAIRS = 336534
ISOLATED_HEROES = 19
MOST_CONNECTED_ID = 859
MOST_CONNECTED_DEGREE = 1933
MULTI_LINE_HEROES = 74


def test_the_names_file_is_not_utf8() -> None:
    """The declared Latin-1 encoding is load-bearing, not decoration.

    If this ever passes, ``Marvel-names.txt`` has been re-encoded and
    ``NAMES_ENCODING`` should be revisited - not the other way round.
    """
    raw = pipeline.names_path().read_bytes()
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")


def test_names_load_completely_under_the_declared_encoding() -> None:
    names = pipeline.load_names()
    assert len(names) == 19428
    assert names["id"].is_unique
    assert names.set_index("id").loc[MOST_CONNECTED_ID, "name"] == "CAPTAIN AMERICA"


def test_names_keep_the_whole_quoted_string() -> None:
    """Splitting on whitespace instead of quotes truncates most of the file."""
    names = pipeline.load_names().set_index("id")["name"]
    assert names.loc[1] == "24-HOUR MAN/EMMANUEL"
    assert names.str.contains(" ").sum() > 1000


def test_some_heroes_are_split_over_several_lines() -> None:
    """The reason degrees must be aggregated by key rather than read per line."""
    lines = pipeline.load_raw()
    first_field = lines["line"].str.split().str[0]
    assert (first_field.value_counts() > 1).sum() == MULTI_LINE_HEROES


def test_pandas_degrees_match_the_published_shape() -> None:
    degrees = pipeline.degrees_with_pandas()
    assert len(degrees) == HEROES
    assert int(degrees["degree"].sum()) == CO_APPEARANCE_PAIRS
    assert int(degrees.iloc[0]["id"]) == MOST_CONNECTED_ID
    assert int(degrees.iloc[0]["degree"]) == MOST_CONNECTED_DEGREE
    assert int((degrees["degree"] == 0).sum()) == ISOLATED_HEROES


def test_the_most_connected_hero_wins_outright() -> None:
    """``flipped.max()`` in the notebook is only correct because of this.

    On a tie it would return whichever id sorted higher, silently.
    """
    degrees = pipeline.degrees_with_pandas()
    top = int(degrees.iloc[0]["degree"])
    assert (degrees["degree"] == top).sum() == 1
    # 1,933 against 1,741: the README quotes this margin as the reason the
    # notebook's tie-blind `max()` happened to be safe.
    assert top - int(degrees.iloc[1]["degree"]) == 192


def test_every_hero_in_the_graph_has_a_name() -> None:
    """Guards the defect class where a lookup table silently drops rows."""
    degrees = pipeline.degrees_with_pandas()
    known = set(pipeline.load_names()["id"])
    assert set(degrees["id"]) <= known


def test_prepare_input_explains_the_right_entry_point() -> None:
    with pytest.raises(NotImplementedError, match=r"train\.py"):
        pipeline.prepare_input({})


@pytest.mark.needs_spark
@pytest.mark.slow
def test_spark_degrees_equal_the_pandas_degrees() -> None:
    """The distributed answer and the single-machine answer must be the same."""
    reference = pipeline.degrees_with_pandas().set_index("id")["degree"]

    with dsspark.session("marvel-test", cores="2", shuffle_partitions=2) as spark:
        lines = dsspark.read_text_lines(spark, pipeline.graph_path())
        degrees = dsspark.degree_table(dsspark.adjacency_from_lines(lines)).toPandas()

    # Spark returns int32 where pandas returns int64, so compare the values
    # rather than the dtypes - the claim being tested is about the counts.
    computed = degrees.set_index("id")["degree"].astype("int64")
    assert len(computed) == HEROES
    assert computed.sort_index().equals(reference.sort_index().astype("int64"))


@pytest.mark.needs_spark
@pytest.mark.slow
def test_bfs_from_captain_america_reaches_almost_everyone() -> None:
    with dsspark.session("marvel-bfs-test", cores="2", shuffle_partitions=2) as spark:
        lines = dsspark.read_text_lines(spark, pipeline.graph_path())
        adjacency = dsspark.adjacency_from_lines(lines)
        distances = dsspark.bfs_distances(adjacency, MOST_CONNECTED_ID, max_depth=12).toPandas()

    assert int(distances.loc[distances["id"] == MOST_CONNECTED_ID, "distance"].iloc[0]) == 0
    assert len(distances) / HEROES > 0.9
    assert int(distances["distance"].max()) <= 12
