"""Loading the Marvel co-appearance graph.

Two things here differ from the source notebook and both change the answer.

``Marvel-names.txt`` is Latin-1. The notebook read it through
``sc.textFile``, which decodes as UTF-8 and replaces whatever fails - two lines
of this file - without raising. The encoding is declared explicitly below, and
``tests/projects/test_marvel_network.py`` asserts that a strict UTF-8 read of
the shipped file still fails, so the declaration cannot rot into decoration.

74 of the 6,486 characters have their neighbours spread over more than one
line. The notebook's ``reduceByKey`` handled that correctly; the ``map``-only
version people usually write from the same tutorial does not, and undercounts
exactly those 74. :func:`dsjourney.spark.adjacency_from_lines` aggregates by
key for the same reason, and a test pins the count.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from dsjourney.config import load_project_config
from dsjourney.datasets import DatasetNotFoundError
from dsjourney.paths import project_data_dir

CONFIG = load_project_config("marvel_network")

GRAPH_FILE = "Marvel-graph.txt"
NAMES_FILE = "Marvel-names.txt"

# Marvel-names.txt was published in 1990s-era Latin-1 and never converted.
NAMES_ENCODING = "latin-1"

# Captain America - the character the notebook's single question landed on, and
# the default BFS root so the two results sit side by side.
DEMO_HERO_ID = 859


def data_path(file: str) -> Path:
    """Resolve a data file, naming the fetch command when it is absent."""
    path = project_data_dir(CONFIG.name) / file
    if not path.is_file():
        raise DatasetNotFoundError(
            f"{file} not found at {path}. "
            f"Run: uv run python scripts/fetch_assets.py --project {CONFIG.name}"
        )
    return path


def graph_path() -> Path:
    """Path to the adjacency-list file."""
    return data_path(GRAPH_FILE)


def names_path() -> Path:
    """Path to the character id-to-name file."""
    return data_path(NAMES_FILE)


def load_raw() -> pd.DataFrame:
    """Read the adjacency list as one row per line, for the generic CLI commands."""
    lines = graph_path().read_text(encoding="utf-8").splitlines()
    return pd.DataFrame({"line": [line for line in lines if line.strip()]})


def load_names() -> pd.DataFrame:
    """Read the character names as ``(id, name)``.

    Parsed on the quote character rather than on whitespace: every name in the
    file is quoted and most contain spaces, so splitting on whitespace truncates
    them at the first word.
    """
    rows: list[tuple[int, str]] = []
    text = names_path().read_text(encoding=NAMES_ENCODING)
    for line in text.splitlines():
        parts = line.split('"')
        if len(parts) < 2 or not parts[0].strip().isdigit():
            continue
        rows.append((int(parts[0].strip()), parts[1]))
    return pd.DataFrame(rows, columns=["id", "name"])


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the line frame unchanged; a graph is traversed, not tabulated."""
    return frame


def degrees_with_pandas(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute ``(id, degree)`` in pandas, for cross-checking and timing.

    The Spark result must equal this exactly. Having both is what turns "Spark
    is faster" from a claim in the notebook's opening paragraph into a number in
    the README.
    """
    lines = frame if frame is not None else load_raw()
    degrees: dict[int, int] = {}
    for line in lines["line"]:
        fields = line.split()
        if not fields:
            continue
        degrees[int(fields[0])] = degrees.get(int(fields[0]), 0) + len(fields) - 1
    return (
        pd.DataFrame({"id": list(degrees), "degree": list(degrees.values())})
        .sort_values(["degree", "id"], ascending=[False, True])
        .reset_index(drop=True)
    )


def prepare_input(payload: dict[str, Any]) -> pd.DataFrame:
    """Not applicable: this project answers questions about a graph, not a row."""
    raise NotImplementedError(
        "marvel_network measures a graph. Use: python projects/marvel_network/train.py --root 859"
    )
