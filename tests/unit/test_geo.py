"""Tests for the geospatial helpers.

The Gutenberg-Richter fit is checked against synthetic catalogues drawn from
the law itself, so the estimator must recover the b that generated the data -
a stronger check than any fixture of magic numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dsjourney import geo


@pytest.fixture
def city_pair() -> pd.DataFrame:
    """Two reference points a known distance apart: Istanbul and Ankara."""
    return pd.DataFrame(
        {"name": ["Istanbul", "Ankara"], "lat": [41.0082, 39.9334], "lon": [28.9784, 32.8597]}
    )


def test_nearest_neighbour_picks_the_closer_reference(city_pair: pd.DataFrame) -> None:
    points = pd.DataFrame({"lat": [41.0, 39.9], "lon": [29.0, 32.8]})
    joined = geo.nearest_neighbour(points, city_pair)
    assert joined["nearest_index"].tolist() == [0, 1]


def test_nearest_neighbour_distance_matches_the_known_geodesic(city_pair: pd.DataFrame) -> None:
    """Istanbul to Ankara is ~351 km along the great circle."""
    just_istanbul = city_pair.iloc[[0]]
    ankara = city_pair.iloc[[1]].rename(columns={"lat": "lat", "lon": "lon"})
    joined = geo.nearest_neighbour(ankara, just_istanbul)
    assert joined["distance_km"].iloc[0] == pytest.approx(351, abs=5)


def test_nearest_neighbour_adds_columns_without_mutating_the_input(
    city_pair: pd.DataFrame,
) -> None:
    points = pd.DataFrame({"lat": [10.0], "lon": [10.0]})
    before = list(points.columns)
    geo.nearest_neighbour(points, city_pair)
    assert list(points.columns) == before


def test_nearest_neighbour_rejects_an_empty_reference() -> None:
    points = pd.DataFrame({"lat": [0.0], "lon": [0.0]})
    with pytest.raises(ValueError, match="empty"):
        geo.nearest_neighbour(points, points.iloc[:0])


def test_coordinates_outside_the_globe_are_rejected() -> None:
    bad = pd.DataFrame({"lat": [95.0], "lon": [0.0]})
    reference = pd.DataFrame({"lat": [0.0], "lon": [0.0]})
    with pytest.raises(ValueError, match="outside"):
        geo.nearest_neighbour(bad, reference)


def test_nan_coordinates_are_rejected_not_propagated() -> None:
    bad = pd.DataFrame({"lat": [np.nan], "lon": [0.0]})
    reference = pd.DataFrame({"lat": [0.0], "lon": [0.0]})
    with pytest.raises(ValueError, match="NaN"):
        geo.nearest_neighbour(bad, reference)


def test_grid_density_counts_and_centres() -> None:
    frame = pd.DataFrame({"lat": [1.0, 2.0, 3.0, 7.0], "lon": [1.0, 2.0, 3.0, 7.0]})
    cells = geo.grid_density(frame, cell_degrees=5.0)
    assert cells["count"].tolist() == [3, 1]
    assert cells.iloc[0]["cell_lat"] == 2.5  # centre, not corner
    assert cells.iloc[1][["cell_lat", "cell_lon"]].tolist() == [7.5, 7.5]


def test_grid_density_handles_negative_coordinates() -> None:
    """Flooring must bin -1 into the [-5, 0) cell, not the [0, 5) one."""
    frame = pd.DataFrame({"lat": [-1.0], "lon": [-1.0]})
    cells = geo.grid_density(frame, cell_degrees=5.0)
    assert cells.iloc[0]["cell_lat"] == -2.5


def test_grid_density_rejects_a_nonpositive_cell() -> None:
    frame = pd.DataFrame({"lat": [0.0], "lon": [0.0]})
    with pytest.raises(ValueError, match="positive"):
        geo.grid_density(frame, cell_degrees=0)


def test_gutenberg_richter_recovers_the_generating_b() -> None:
    """Draw a catalogue from the law with b=1 and the fit must find it."""
    rng = np.random.default_rng(0)
    b_true, mc = 1.0, 5.5
    beta = b_true * np.log(10)
    magnitudes = mc + rng.exponential(1 / beta, size=20_000)
    fit = geo.gutenberg_richter(magnitudes, completeness=mc, bin_width=0.0)
    assert fit.b_value == pytest.approx(b_true, abs=3 * fit.b_stderr)
    assert fit.events == 20_000


def test_gutenberg_richter_is_sensitive_to_the_generating_b() -> None:
    """A steeper catalogue must fit a larger b - the estimator is not constant."""
    rng = np.random.default_rng(1)
    fits = []
    for b_true in (0.8, 1.4):
        beta = b_true * np.log(10)
        magnitudes = 5.5 + rng.exponential(1 / beta, size=20_000)
        fits.append(geo.gutenberg_richter(magnitudes, completeness=5.5, bin_width=0.0))
    assert fits[0].b_value < fits[1].b_value
    assert fits[0].b_value == pytest.approx(0.8, abs=0.05)
    assert fits[1].b_value == pytest.approx(1.4, abs=0.08)


def test_gutenberg_richter_ignores_the_incomplete_tail() -> None:
    """Events below the completeness threshold must not drag the fit."""
    rng = np.random.default_rng(2)
    beta = np.log(10)
    complete = 5.5 + rng.exponential(1 / beta, size=10_000)
    junk = np.full(5_000, 4.0)  # a spike of under-threshold events
    fit = geo.gutenberg_richter(np.concatenate([complete, junk]), completeness=5.5, bin_width=0.0)
    assert fit.events == 10_000
    assert fit.b_value == pytest.approx(1.0, abs=0.05)


def test_gutenberg_richter_refuses_a_tiny_catalogue() -> None:
    with pytest.raises(ValueError, match="too few"):
        geo.gutenberg_richter(np.array([5.6, 5.7]), completeness=5.5)


def test_expected_count_matches_the_catalogue_at_the_threshold() -> None:
    rng = np.random.default_rng(3)
    magnitudes = 5.5 + rng.exponential(1 / np.log(10), size=10_000)
    fit = geo.gutenberg_richter(magnitudes, completeness=5.5, bin_width=0.0)
    assert fit.expected_count(5.5) == pytest.approx(10_000, rel=0.01)


def test_magnitude_frequency_is_cumulative_and_monotonic() -> None:
    frame = geo.magnitude_frequency(np.array([5.5, 5.5, 6.0, 7.0]), bin_width=0.5)
    assert frame["cumulative_count"].iloc[0] == 4
    assert frame["cumulative_count"].is_monotonic_decreasing


def test_available_models_names_the_analysis() -> None:
    assert geo.available_models() == ["gutenberg_richter"]
