#!/usr/bin/env python
"""Analyse the earthquake catalogue and score it against the literature.

Usage:
    python projects/earthquake_atlas/train.py
    python projects/earthquake_atlas/train.py --cell-degrees 2 --no-save
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dsjourney import geo
from dsjourney.paths import project_artifacts_dir
from projects.earthquake_atlas import pipeline

# The global b-value reported across the seismology literature.
LITERATURE_B = 1.0


def _density_figure(quakes, cells):  # type: ignore[no-untyped-def]
    """A world scatter of the catalogue with the busiest cells marked."""
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.scatter(
        quakes["Longitude"], quakes["Latitude"], s=2, alpha=0.15, linewidths=0, color="#1f77b4"
    )
    top = cells.head(20)
    axis.scatter(
        top["cell_lon"],
        top["cell_lat"],
        s=top["count"] * 0.6,
        facecolors="none",
        edgecolors="#d62728",
        linewidths=1.5,
        label="20 busiest 5-degree cells",
    )
    axis.set_xlim(-180, 180)
    axis.set_ylim(-90, 90)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_title("Significant earthquakes 1965-2016 (M >= 5.5)")
    axis.legend(loc="lower left")
    figure.tight_layout()
    return figure


def _magnitude_figure(frequency, fit):  # type: ignore[no-untyped-def]
    """Observed cumulative counts against the fitted Gutenberg-Richter line."""
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.semilogy(
        frequency["magnitude"],
        frequency["cumulative_count"],
        "o",
        markersize=3,
        label="observed N(>=M)",
    )
    grid = np.linspace(fit.completeness, frequency["magnitude"].max(), 50)
    axis.semilogy(
        grid, [fit.expected_count(m) for m in grid], "-", label=f"fit: b = {fit.b_value:.3f}"
    )
    axis.set_xlabel("Magnitude")
    axis.set_ylabel("Cumulative count")
    axis.set_title("Gutenberg-Richter fit")
    axis.legend()
    figure.tight_layout()
    return figure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    params = pipeline.CONFIG.model.params
    parser.add_argument("--cell-degrees", type=float, default=float(params["cell_degrees"]))
    parser.add_argument("--no-save", action="store_true", help="do not write artifacts")
    args = parser.parse_args(argv)

    quakes = pipeline.load_raw()
    span = f"{quakes['Date'].min():%Y}-{quakes['Date'].max():%Y}"
    print(
        f"{len(quakes):,} earthquakes, {span}, M {quakes['Magnitude'].min()}-"
        f"{quakes['Magnitude'].max()}"
    )

    # Density: the map as a table, busiest first.
    cells = geo.grid_density(
        quakes, lat="Latitude", lon="Longitude", cell_degrees=args.cell_degrees
    )
    print(f"\n{len(cells):,} occupied {args.cell_degrees:g}-degree cells; busiest:")
    for row in cells.head(5).itertuples():
        print(f"  {row.count:5,} quakes near ({row.cell_lat:+.1f}, {row.cell_lon:+.1f})")

    # Proximity: the spatial join the notebook never made.
    joined = pipeline.quakes_near_cities(quakes)
    near = joined[joined["distance_km"] <= 100]
    print(f"\n{len(near):,} quakes within 100 km of a top-1000 US city")
    if len(near):
        closest = near.nsmallest(3, "distance_km")
        for row in closest.itertuples():
            print(f"  M{row.Magnitude} {row.distance_km:5.1f} km from {row.nearest_city}")

    # The law: a fitted constant with a literature value to answer to.
    fit = geo.gutenberg_richter(
        quakes["Magnitude"],
        completeness=float(params["completeness"]),
        bin_width=float(params["bin_width"]),
    )
    gap = abs(fit.b_value - LITERATURE_B)
    print(
        f"\nGutenberg-Richter: b = {fit.b_value:.3f} +/- {fit.b_stderr:.3f} "
        f"over {fit.events:,} events\n"
        f"  literature value ~{LITERATURE_B}; this catalogue lands {gap:.3f} away"
    )

    metrics = {
        "b_value": fit.b_value,
        "b_stderr": fit.b_stderr,
        "b_gap_from_literature": gap,
        "events": float(fit.events),
        "busiest_cell_count": float(cells["count"].iloc[0]),
        "quakes_within_100km_of_a_city": float(len(near)),
    }
    print("\n" + " | ".join(f"{k} {v:.4g}" for k, v in metrics.items()))

    if not args.no_save:
        directory = project_artifacts_dir("earthquake_atlas", create=True)
        (directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        metadata = {
            "model_class": "GutenbergRichter",
            "events": fit.events,
            "completeness": fit.completeness,
            "a_value": round(fit.a_value, 4),
            "cell_degrees": args.cell_degrees,
            "span": span,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        cells.head(100).to_csv(directory / "densest_cells.csv", index=False)
        frequency = geo.magnitude_frequency(quakes["Magnitude"])
        _density_figure(quakes, cells).savefig(directory / "density_map.png", dpi=110)
        _magnitude_figure(frequency, fit).savefig(directory / "gutenberg_richter.png", dpi=110)
        plt.close("all")
        print(f"\nartifacts: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
