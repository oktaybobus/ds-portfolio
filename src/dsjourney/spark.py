"""Distributed computation with Spark, and an honest account of when it helps.

The source notebook opened with the claim that Spark is "up to 100x faster"
and then ran every job with ``setMaster("local")`` - one thread - over files of
a few megabytes. Nothing in it was ever timed, so the claim went untested.
``projects/marvel_network/train.py --benchmark`` tests it with :func:`timed`:
at that scale pandas wins by roughly ten to one, and the JVM takes longer to
start than pandas takes to finish. That is not an argument against Spark; it is
the reason to know where the line sits before reaching for it.

Three defects from the notebook are fixed by construction:

* **Text was read as UTF-8 regardless of what the file actually was.**
  ``book.txt`` has 269 lines of cp1252 and ``Marvel-names.txt`` has two; Spark's
  ``text`` reader replaces every undecodable byte and raises nothing. See
  :func:`read_text_lines`, which routes through the one reader that accepts an
  encoding.
* **A binary classifier's ROC AUC was printed as "Accuracy".** They are
  different numbers on the same predictions. :func:`binary_classification_scores`
  returns all five, so there is nothing to mislabel.
* **``SparkContext.getOrCreate(conf=conf)`` silently discards ``conf``** when a
  context is already alive, which is why the notebook is littered with bare
  ``sc.stop()`` cells. :func:`session` is a context manager instead.

Two more are environment traps rather than notebook defects, and both fail in
ways that point away from their cause:

* Spark launches its Python workers from whatever ``python3`` resolves to on
  PATH. On a machine with conda or a system interpreter that is not the one
  running the driver, and the job dies mid-stage with ``PYTHON_VERSION_MISMATCH``
  rather than at start-up. :func:`session` pins both to ``sys.executable``.
* The JVM validates charset names against its own list, which has no ``latin-1``
  and no ``cp1252``. :func:`jvm_charset` translates the Python spelling.
"""

from __future__ import annotations

import functools
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql import SparkSession

T = TypeVar("T")

# Homebrew keeps versioned JDKs keg-only, so they are installed but absent from
# PATH. Spark supports 17 and 21; 8 and 11 are gone from Spark 4 and 25 is not
# supported by any release yet, so those two are what we look for.
_JDK_CANDIDATES = (
    "/opt/homebrew/opt/openjdk@17",
    "/opt/homebrew/opt/openjdk@21",
    "/usr/local/opt/openjdk@17",
    "/usr/local/opt/openjdk@21",
    "/usr/lib/jvm/temurin-17-jdk-amd64",
    "/usr/lib/jvm/java-17-openjdk-amd64",
)

INSTALL_HINT = (
    "PySpark needs the optional extra and a Java runtime.\n"
    "  uv sync --extra spark\n"
    "  brew install openjdk@17        # macOS; apt install openjdk-17-jdk on Debian"
)


@functools.lru_cache(maxsize=8)
def _runs_java(home: Path) -> bool:
    """True when ``home`` is a real JDK and its ``java`` actually starts.

    macOS ships a ``/usr/bin/java`` stub that exists whether or not a JDK does;
    it prints an install prompt and exits non-zero. Testing for the file alone
    therefore reports a runtime that is not there, so the binary is executed.
    """
    binary = home / "bin" / "java"
    if not binary.is_file():
        return False
    try:
        completed = subprocess.run(
            [str(binary), "-version"], capture_output=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment dependent
        return False
    return completed.returncode == 0


def java_home() -> Path | None:
    """Return the Java installation Spark should use, or ``None`` if there is none.

    Resolution order: an existing ``JAVA_HOME``, the known Homebrew and Debian
    locations, macOS's ``java_home`` helper, then whatever ``java`` is on PATH.
    Discovering it here means ``dsj train marvel_network`` works on a fresh
    clone without the reader first editing their shell profile.
    """
    configured = os.environ.get("JAVA_HOME")
    if configured and _runs_java(Path(configured)):
        return Path(configured)

    for candidate in _JDK_CANDIDATES:
        path = Path(candidate)
        if _runs_java(path):
            return path

    if platform.system() == "Darwin":
        try:
            completed = subprocess.run(
                ["/usr/libexec/java_home"], capture_output=True, text=True, timeout=30, check=False
            )
        except (OSError, subprocess.SubprocessError):  # pragma: no cover
            completed = None
        if completed is not None and completed.returncode == 0:
            path = Path(completed.stdout.strip())
            if _runs_java(path):
                return path

    executable = shutil.which("java")
    if executable is not None:
        path = Path(executable).resolve().parent.parent
        if _runs_java(path):
            return path
    return None


def pyspark_installed() -> bool:
    """True when the ``spark`` extra is installed."""
    import importlib.util

    return importlib.util.find_spec("pyspark") is not None


def spark_available() -> bool:
    """True when a Spark job could actually run: PySpark installed and a JVM present."""
    return pyspark_installed() and java_home() is not None


def require_spark() -> None:
    """Raise with installation instructions when Spark cannot run."""
    if not pyspark_installed():
        raise ImportError(f"PySpark is not installed.\n{INSTALL_HINT}")
    if java_home() is None:
        raise RuntimeError(f"No Java runtime found; Spark cannot start.\n{INSTALL_HINT}")


@contextmanager
def session(
    app_name: str = "dsjourney",
    *,
    cores: str = "*",
    shuffle_partitions: int = 8,
    log_level: str = "ERROR",
) -> Iterator[SparkSession]:
    """Yield a local Spark session and stop it on the way out.

    ``cores`` defaults to ``"*"`` - every core on the machine - where the
    notebook hard-coded ``local``, which is one thread. ``shuffle_partitions``
    drops Spark's default of 200, which on a local run means 200 near-empty
    tasks per shuffle and more scheduling than work.

    Stopping is what the context manager is for. A leaked ``SparkContext`` makes
    the next ``getOrCreate`` silently return the old one, configuration and all,
    which is the bug behind every stray ``sc.stop()`` in the source notebook.
    """
    require_spark()
    from pyspark.sql import SparkSession as _SparkSession

    home = java_home()
    if home is not None:
        os.environ["JAVA_HOME"] = str(home)

    # Spark starts its Python workers from whatever `python3` PATH resolves to,
    # which on a machine with a system or conda interpreter is not the one
    # running this code. The job then dies mid-stage with PYTHON_VERSION_MISMATCH
    # rather than at start-up, so the failure surfaces far from its cause.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    builder = (
        _SparkSession.builder.appName(app_name)
        .master(f"local[{cores}]")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.ui.showConsoleProgress", "false")
        # A local run has no cluster to lose, and the event log and metrics
        # server only add start-up cost.
        .config("spark.ui.enabled", "false")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(log_level)
    try:
        yield spark
    finally:
        spark.stop()


# Spark validates the charset name against the JVM's list, which does not
# include Python's aliases: "latin-1" is rejected outright, "cp1252" silently is
# not one of the accepted eight either. Both are spelled differently on the JVM.
_CHARSET_ALIASES = {
    "latin-1": "iso-8859-1",
    "latin1": "iso-8859-1",
    "iso8859-1": "iso-8859-1",
    "cp1252": "windows-1252",
    "windows_1252": "windows-1252",
    "utf8": "utf-8",
    "utf_8": "utf-8",
    "ascii": "us-ascii",
}


def jvm_charset(encoding: str) -> str:
    """Return the JVM name for a Python encoding.

    ``spark.read.option("encoding", "latin-1")`` raises INVALID_PARAMETER_VALUE
    - the JVM calls it ``iso-8859-1``. Passing the Python name through is a
    hard failure rather than a silent one, but it fails at read time, well after
    the encoding was chosen.
    """
    key = encoding.strip().lower()
    return _CHARSET_ALIASES.get(key, key)


def read_text_lines(
    spark: SparkSession,
    path: Path | str,
    *,
    encoding: str = "utf-8",
    column: str = "line",
) -> SparkDataFrame:
    """Read a text file as one row per line, decoded with the encoding given.

    ``spark.read.text`` has no encoding option and assumes UTF-8; undecodable
    bytes become U+FFFD with no warning. The CSV reader does accept one, so a
    separator that cannot occur in text turns it into a line reader that decodes
    correctly. ``book.txt`` loses 269 lines' worth of punctuation without this,
    which silently splits contractions into separate words.
    """
    frame = (
        spark.read.option("encoding", jvm_charset(encoding))
        .option("header", "false")
        .option("quote", "")
        .option("escape", "")
        .option("sep", "\x00")
        .csv(str(path))
    )
    return frame.toDF(column)


def word_frequencies(
    lines: SparkDataFrame,
    *,
    column: str = "line",
    min_length: int = 1,
) -> SparkDataFrame:
    """Count words across a line frame, most frequent first.

    Returns a DataFrame rather than the notebook's ``countByValue``: that is an
    action which collects the entire vocabulary into the driver as a Python
    dict, so the one operation the whole exercise exists to demonstrate is
    performed on a single machine.
    """
    from pyspark.sql import functions as sf

    words = sf.explode(sf.split(sf.lower(sf.col(column)), r"\W+")).alias("word")
    return (
        lines.select(words)
        .where(sf.length("word") >= min_length)
        .groupBy("word")
        .count()
        .orderBy(sf.desc("count"), "word")
    )


def adjacency_from_lines(
    lines: SparkDataFrame,
    *,
    column: str = "line",
) -> SparkDataFrame:
    """Parse whitespace-separated adjacency lists into ``(id, neighbours)``.

    A node may be spread over several lines - 74 of the 6,486 Marvel heroes are
    - so the neighbour lists are concatenated by key. Mapping each line straight
    to a degree, without the aggregation, undercounts exactly those nodes and
    raises nothing.
    """
    from pyspark.sql import functions as sf

    parts = sf.split(sf.trim(sf.col(column)), r"\s+")
    return (
        lines.where(sf.length(sf.trim(sf.col(column))) > 0)
        .select(
            parts.getItem(0).cast("int").alias("id"),
            sf.slice(parts, 2, sf.size(parts) - 1).cast("array<int>").alias("neighbours"),
        )
        .groupBy("id")
        .agg(sf.flatten(sf.collect_list("neighbours")).alias("neighbours"))
    )


def degree_table(adjacency: SparkDataFrame) -> SparkDataFrame:
    """Return ``(id, degree)`` ordered by degree, descending."""
    from pyspark.sql import functions as sf

    return adjacency.select("id", sf.size("neighbours").alias("degree")).orderBy(
        sf.desc("degree"), "id"
    )


def bfs_distances(
    adjacency: SparkDataFrame,
    start: int,
    *,
    max_depth: int = 20,
) -> SparkDataFrame:
    """Breadth-first distances from ``start``, as ``(id, distance)``.

    One distributed join per level, which is how a graph too large for one
    machine is traversed. Nodes never reached are simply absent from the result;
    the caller decides what an unreachable node means.
    """
    from pyspark.sql import functions as sf

    spark = adjacency.sparkSession
    distances = spark.createDataFrame([(int(start), 0)], "id INT, distance INT")
    frontier = distances

    for depth in range(1, max_depth + 1):
        discovered = (
            frontier.join(adjacency, on="id", how="inner")
            .select(sf.explode("neighbours").alias("id"))
            .distinct()
            .join(distances, on="id", how="left_anti")
            .withColumn("distance", sf.lit(depth))
        )
        # Spark is lazy, so without a checkpoint the query plan grows by one
        # join per level and eventually costs more to analyse than to run.
        discovered = discovered.localCheckpoint(eager=True)
        if discovered.isEmpty():
            break
        distances = distances.unionByName(discovered).localCheckpoint(eager=True)
        frontier = discovered

    return distances.orderBy("distance", "id")


def binary_classification_scores(
    predictions: SparkDataFrame,
    *,
    label_col: str = "label",
    prediction_col: str = "prediction",
    probability_col: str = "probability",
    positive_label: float = 1.0,
) -> dict[str, float]:
    """Score binary predictions on accuracy, precision, recall, F1 and ROC AUC.

    The notebook built a ``BinaryClassificationEvaluator``, left ``metricName``
    at its default of ``areaUnderROC``, and printed the result as ``Accuracy``.
    On the Pima data the two differ by roughly eight points, in the flattering
    direction. Returning the whole set removes the choice of what to call it.

    Keys match :func:`dsjourney.evaluate.classification_scores` so a Spark model
    and a scikit-learn one can be put in the same table.
    """
    from pyspark.ml.evaluation import (
        BinaryClassificationEvaluator,
        MulticlassClassificationEvaluator,
    )

    multiclass = functools.partial(
        MulticlassClassificationEvaluator,
        labelCol=label_col,
        predictionCol=prediction_col,
        metricLabel=positive_label,
    )
    scores = {
        "accuracy": multiclass(metricName="accuracy").evaluate(predictions),
        "precision": multiclass(metricName="precisionByLabel").evaluate(predictions),
        "recall": multiclass(metricName="recallByLabel").evaluate(predictions),
        # `f1` on this evaluator is the support-weighted mean over both classes,
        # not the positive class - a different number again.
        "f1": multiclass(metricName="fMeasureByLabel").evaluate(predictions),
        "roc_auc": BinaryClassificationEvaluator(
            labelCol=label_col, rawPredictionCol=probability_col, metricName="areaUnderROC"
        ).evaluate(predictions),
    }
    return {name: float(value) for name, value in scores.items()}


def majority_baseline(labels: Any) -> float:
    """Accuracy of always predicting the most common label.

    Quoted next to every accuracy in this module. Pima is 65% negative, so a
    model that has learned nothing still scores 0.651.
    """
    import numpy as np

    values, counts = np.unique(np.asarray(labels).ravel(), return_counts=True)
    del values
    return float(counts.max() / counts.sum())


# Algorithms this module can run, by the name a project config uses. The
# contract test in tests/projects/test_pipelines.py resolves any estimator
# named `spark_*` against this list.
SPARK_MODELS = ("spark_bfs", "spark_logistic")


def available_models() -> list[str]:
    """Return the Spark algorithms a project config may name."""
    return list(SPARK_MODELS)


@dataclass(frozen=True)
class Timing:
    """How long one labelled run took, in seconds."""

    label: str
    seconds: float
    rows: int = 0

    @property
    def rows_per_second(self) -> float:
        """Throughput, or 0 when the run processed no rows."""
        return self.rows / self.seconds if self.seconds > 0 and self.rows else 0.0


def timed(label: str, work: Callable[[], T], *, rows: int = 0) -> tuple[T, Timing]:
    """Run ``work``, returning its result alongside a :class:`Timing`."""
    started = time.perf_counter()
    result = work()
    return result, Timing(label=label, seconds=time.perf_counter() - started, rows=rows)
