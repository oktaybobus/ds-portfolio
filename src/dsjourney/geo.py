"""Geospatial analysis: joins, density, and one law of nature to check against.

The source notebook is fourteen cells of plotly gallery examples, and five of
its eight maps plot plotly's own bundled demo data - gapminder, elections,
car-sharing - rather than anything from the course. Nothing is computed from
any of it: every cell renders a picture and moves on. It also cannot run as
saved: ``from plotly.oflfine import init_notebook_mode`` raises ImportError on
the typo, and the cell after calls ``iplot``, which is never imported.

This module is what the pictures were missing. A map of earthquake density is
a claim about where earthquakes are; :func:`grid_density` computes it.
"Which quakes were near cities" is a spatial join; :func:`nearest_neighbour`
answers it in one BallTree query. And earthquake magnitudes follow the
Gutenberg-Richter law with b close to 1.0 - a published constant the data
either reproduces or does not, which makes it the rare chance to score an
analysis against nature rather than against a held-out split.

Everything here uses scikit-learn and matplotlib, both already core
dependencies; there is no plotly and no basemap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0


def _as_radians(frame: pd.DataFrame, lat: str, lon: str) -> np.ndarray:
    """Return an (n, 2) array of radians, validating the coordinate ranges."""
    coords = frame[[lat, lon]].to_numpy(dtype=float)
    if np.isnan(coords).any():
        raise ValueError(f"columns {lat!r}/{lon!r} contain NaN coordinates")
    if (np.abs(coords[:, 0]) > 90).any() or (np.abs(coords[:, 1]) > 180).any():
        raise ValueError(f"coordinates outside [-90, 90] x [-180, 180] in {lat!r}/{lon!r}")
    return np.radians(coords)


def nearest_neighbour(
    points: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    lat: str = "lat",
    lon: str = "lon",
    ref_lat: str = "lat",
    ref_lon: str = "lon",
) -> pd.DataFrame:
    """Join each point to its nearest reference point on the sphere.

    Returns ``points`` with two added columns: ``nearest_index`` (positional
    index into ``reference``) and ``distance_km`` along the great circle. One
    BallTree query replaces the quadratic pandas crossjoin people usually
    write; 23,000 quakes against 1,000 cities takes ~50 ms.
    """
    from sklearn.neighbors import BallTree

    if reference.empty:
        raise ValueError("reference frame is empty; nothing to join against")

    tree = BallTree(_as_radians(reference, ref_lat, ref_lon), metric="haversine")
    distance, index = tree.query(_as_radians(points, lat, lon), k=1)
    joined = points.copy()
    joined["nearest_index"] = index[:, 0]
    joined["distance_km"] = distance[:, 0] * EARTH_RADIUS_KM
    return joined


def grid_density(
    frame: pd.DataFrame,
    *,
    lat: str = "lat",
    lon: str = "lon",
    cell_degrees: float = 5.0,
) -> pd.DataFrame:
    """Count points per lat/lon cell, returning only occupied cells.

    The centre of each cell is reported rather than its corner, so plotting the
    result puts markers where the mass is. A 5-degree cell at the equator is
    roughly 550 km across; it shrinks towards the poles, which is fine for
    density *ranking* and wrong for density *per km²* - callers comparing
    areas should weight by cos(latitude).
    """
    if cell_degrees <= 0:
        raise ValueError(f"cell_degrees must be positive, got {cell_degrees}")
    half = cell_degrees / 2
    return (
        frame.assign(
            cell_lat=(frame[lat] // cell_degrees) * cell_degrees + half,
            cell_lon=(frame[lon] // cell_degrees) * cell_degrees + half,
        )
        .groupby(["cell_lat", "cell_lon"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


@dataclass(frozen=True)
class GutenbergRichterFit:
    """A maximum-likelihood fit of the magnitude-frequency law."""

    b_value: float
    b_stderr: float
    a_value: float
    completeness: float
    events: int

    def expected_count(self, magnitude: float) -> float:
        """Expected number of events at or above ``magnitude``."""
        return float(10 ** (self.a_value - self.b_value * magnitude))


def gutenberg_richter(
    magnitudes: pd.Series | np.ndarray,
    *,
    completeness: float,
    bin_width: float = 0.1,
) -> GutenbergRichterFit:
    """Fit log10 N(>=M) = a - bM by maximum likelihood (Aki 1965, Utsu 1966).

    ``completeness`` is the magnitude above which the catalogue records every
    event; below it the data is missing quakes, not the Earth. Fitting through
    an incomplete tail is the classic way to get a wrong b-value with a good
    R². Events below the threshold are excluded, which is the honest cut.

    The standard error is Aki's b/sqrt(n). For a global catalogue the
    literature value is b close to 1.0.
    """
    values = np.asarray(magnitudes, dtype=float)
    values = values[~np.isnan(values)]
    kept = values[values >= completeness]
    if len(kept) < 50:
        raise ValueError(
            f"only {len(kept)} events at or above M{completeness}; too few for a stable fit"
        )
    # The half-bin correction accounts for magnitudes being reported rounded.
    b_value = np.log10(np.e) / (kept.mean() - (completeness - bin_width / 2))
    a_value = np.log10(len(kept)) + b_value * completeness
    return GutenbergRichterFit(
        b_value=float(b_value),
        b_stderr=float(b_value / np.sqrt(len(kept))),
        a_value=float(a_value),
        completeness=completeness,
        events=len(kept),
    )


def magnitude_frequency(
    magnitudes: pd.Series | np.ndarray, *, bin_width: float = 0.1
) -> pd.DataFrame:
    """Cumulative counts N(>=M) per magnitude bin, for plotting against the fit."""
    values = np.sort(np.asarray(magnitudes, dtype=float))
    values = values[~np.isnan(values)]
    bins = np.arange(values.min(), values.max() + bin_width, bin_width)
    counts = [(values >= m).sum() for m in bins]
    return pd.DataFrame({"magnitude": np.round(bins, 3), "cumulative_count": counts})


def available_models() -> list[str]:
    """Return the analyses a project config may name."""
    return ["gutenberg_richter"]
