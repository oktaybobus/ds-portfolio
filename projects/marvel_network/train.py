#!/usr/bin/env python
"""Measure the Marvel co-appearance graph with Spark.

Usage:
    python projects/marvel_network/train.py
    python projects/marvel_network/train.py --root 5306
    python projects/marvel_network/train.py --benchmark
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from dsjourney import spark as dsspark
from dsjourney.paths import project_artifacts_dir
from projects.marvel_network import pipeline


def _distance_summary(distances: pd.DataFrame, heroes: int) -> dict[str, float]:
    """Reduce a BFS result to reach, mean hop count and eccentricity."""
    reached = len(distances)
    return {
        "reachable_fraction": reached / heroes if heroes else 0.0,
        "mean_distance": float(distances["distance"].mean()) if reached else 0.0,
        "eccentricity": float(distances["distance"].max()) if reached else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    params = pipeline.CONFIG.model.params
    parser.add_argument("--root", type=int, default=int(params["root_id"]))
    parser.add_argument("--max-depth", type=int, default=int(params["max_depth"]))
    parser.add_argument(
        "--benchmark", action="store_true", help="time the same job in Spark and in pandas"
    )
    parser.add_argument("--no-save", action="store_true", help="do not write artifacts")
    args = parser.parse_args(argv)

    if not dsspark.spark_available():
        print(dsspark.INSTALL_HINT)
        return 2

    names = pipeline.load_names().set_index("id")["name"]
    print(f"{len(names):,} character names ({pipeline.NAMES_ENCODING})")

    with dsspark.session(
        "marvel_network",
        cores=str(params["cores"]),
        shuffle_partitions=int(params["shuffle_partitions"]),
    ) as spark:
        lines = dsspark.read_text_lines(spark, pipeline.graph_path())
        adjacency = dsspark.adjacency_from_lines(lines).cache()
        degrees = dsspark.degree_table(adjacency).toPandas()

        heroes = len(degrees)
        pairs = int(degrees["degree"].sum())
        print(f"\n{heroes:,} characters | {pairs:,} co-appearance pairs")

        # The Spark answer has to equal the single-machine answer. Without this
        # the distributed version could be quietly wrong in either direction and
        # the only symptom would be a plausible number.
        reference = pipeline.degrees_with_pandas()
        merged = degrees.merge(reference, on="id", suffixes=("_spark", "_pandas"))
        assert len(merged) == heroes == len(reference), "Spark and pandas disagree on the hero set"
        assert (merged["degree_spark"] == merged["degree_pandas"]).all(), (
            "Spark and pandas disagree on degrees"
        )
        print("cross-check: Spark degrees match pandas exactly")

        top = degrees.head(10).assign(name=lambda f: f["id"].map(names))
        print("\nMost connected characters:")
        for row in top.itertuples():
            print(f"  {row.degree:5,}  {row.name}")

        # `flipped.max()` in the notebook returns one character and hides how
        # close the race was; printing the runner-up is the whole cost of
        # knowing whether the answer is robust.
        lead = int(top.iloc[0]["degree"]) - int(top.iloc[1]["degree"])
        print(f"  (the leader is {lead} co-appearances clear of second place)")

        print(f"\nBFS from {names.get(args.root, args.root)}...")
        distances = dsspark.bfs_distances(adjacency, args.root, max_depth=args.max_depth).toPandas()
        summary = _distance_summary(distances, heroes)
        spread = distances.groupby("distance").size()
        for hops, count in spread.items():
            print(f"  {hops} hop(s): {count:,} characters")
        unreachable = heroes - len(distances)
        print(f"  unreachable: {unreachable:,}")

        timings: list[dsspark.Timing] = []
        if args.benchmark:
            _, spark_timing = dsspark.timed(
                "spark",
                lambda: dsspark.degree_table(
                    dsspark.adjacency_from_lines(
                        dsspark.read_text_lines(spark, pipeline.graph_path())
                    )
                ).toPandas(),
                rows=heroes,
            )
            _, pandas_timing = dsspark.timed("pandas", pipeline.degrees_with_pandas, rows=heroes)
            timings = [spark_timing, pandas_timing]
            print("\nSame job, both engines (session already warm):")
            for timing in timings:
                print(f"  {timing.label:8} {timing.seconds:7.3f}s")
            ratio = spark_timing.seconds / pandas_timing.seconds
            print(f"  pandas is {ratio:.1f}x faster at this size")

    metrics = {
        "heroes": float(heroes),
        "co_appearance_pairs": float(pairs),
        "max_degree": float(degrees["degree"].max()),
        "mean_degree": float(degrees["degree"].mean()),
        "median_degree": float(degrees["degree"].median()),
        "isolated_heroes": float((degrees["degree"] == 0).sum()),
        **summary,
    }
    print("\n" + " | ".join(f"{name} {value:.4g}" for name, value in metrics.items()))

    if not args.no_save:
        directory = project_artifacts_dir("marvel_network", create=True)
        (directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        metadata = {
            "model_class": "SparkBFS",
            "root_id": args.root,
            "root_name": str(names.get(args.root, "")),
            "max_depth": args.max_depth,
            "names": len(names),
            "engine": "pyspark",
        }
        if timings:
            metadata["seconds"] = {t.label: round(t.seconds, 3) for t in timings}
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        degrees.assign(name=lambda f: f["id"].map(names)).to_csv(
            directory / "degrees.csv", index=False
        )
        print(f"\nartifacts: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
