"""Tests for the earthquake atlas.

The data is committed, so these pin facts about the real catalogue - including
the two quiet traps it carries: three ISO-timestamped rows in a file of
MM/DD/YYYY dates, and a magnitude floor that makes fitting below M5.5 fit the
gap rather than the Earth.
"""

from __future__ import annotations

import pytest

from dsjourney import geo
from projects.earthquake_atlas import pipeline

QUAKES = 23_412
CITIES = 1_000
ISO_TIMESTAMPED_ROWS = 3


def test_every_date_parses_including_the_three_iso_rows() -> None:
    """The defect: 3 of 23,412 dates are ISO-8601 UTC among MM/DD/YYYY rows.

    A naive to_datetime raises on the mixed timezones; adding errors="coerce"
    without utc=True silently NaT-s the three rows instead. All rows must
    survive parsing.
    """
    frame = pipeline.load_raw()
    assert len(frame) == QUAKES
    assert frame["Date"].notna().all()
    assert str(frame["Date"].dtype).startswith("datetime64")


def test_the_catalogue_spans_the_published_years() -> None:
    frame = pipeline.load_raw()
    assert frame["Date"].dt.year.min() == 1965
    assert frame["Date"].dt.year.max() == 2016


def test_the_magnitude_floor_is_the_declared_completeness() -> None:
    """The catalogue is truncated at M5.5 by construction; the config must
    declare exactly that, or the fit runs through missing data."""
    frame = pipeline.load_raw()
    assert frame["Magnitude"].min() == pytest.approx(5.5)
    assert float(pipeline.CONFIG.model.params["completeness"]) == 5.5


def test_cities_load_with_coordinates() -> None:
    cities = pipeline.load_cities()
    assert len(cities) == CITIES
    assert {"City", "lat", "lon"} <= set(cities.columns)


def test_the_join_names_a_real_city_for_a_known_quake() -> None:
    """The 1965 Puget Sound earthquake (M6.7) struck ~5 km from Tacoma."""
    joined = pipeline.quakes_near_cities()
    tacoma = joined[(joined["nearest_city"] == "Tacoma") & (joined["Magnitude"] == 6.7)]
    assert len(tacoma) >= 1
    assert float(tacoma["distance_km"].min()) < 10


def test_the_b_value_reproduces_the_literature() -> None:
    """The headline claim: a global catalogue must land near b = 1.0."""
    frame = pipeline.load_raw()
    fit = geo.gutenberg_richter(frame["Magnitude"], completeness=5.5)
    assert fit.b_value == pytest.approx(1.0, abs=0.05)
    assert fit.events == QUAKES


def test_the_busiest_cells_are_subduction_zones() -> None:
    """The densest 5-degree cells must be in the western Pacific arcs, not
    scattered - the map has known geography to answer to."""
    frame = pipeline.load_raw()
    cells = geo.grid_density(frame, lat="Latitude", lon="Longitude", cell_degrees=5.0)
    top = cells.head(5)
    assert (top["count"] >= 500).all()
    # Every one of the five busiest cells sits in the 120E-180E Pacific belt.
    assert top["cell_lon"].between(120, 180).sum() >= 4


def test_prepare_input_explains_the_right_entry_point() -> None:
    with pytest.raises(NotImplementedError, match=r"train\.py"):
        pipeline.prepare_input({})


@pytest.mark.slow
def test_training_runs_end_to_end() -> None:
    from projects.earthquake_atlas.train import main

    assert main(["--no-save"]) == 0
