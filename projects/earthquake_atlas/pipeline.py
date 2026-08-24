"""Loading and joining the earthquake catalogue.

Two files, both committed: 23,412 significant earthquakes (M >= 5.5,
1965-2016, USGS via plotly/datasets) and the 1,000 largest US cities. The
notebook read the first, plotted it once, and never put the two together -
the join is where the analysis starts.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from dsjourney import geo
from dsjourney.config import load_project_config
from dsjourney.datasets import load_dataset
from dsjourney.paths import project_data_dir

CONFIG = load_project_config("earthquake_atlas")

CITIES_FILE = "us-cities.csv"

# The catalogue is USGS "significant earthquakes": magnitude 5.5 and above by
# construction. Everything below that is absent by design, not by seismology.
CATALOGUE_FLOOR = 5.5


def load_raw() -> pd.DataFrame:
    """Read the earthquake catalogue with parsed dates.

    23,409 rows carry plain ``MM/DD/YYYY`` dates; three carry full ISO-8601
    UTC timestamps, which makes a naive ``to_datetime`` either raise on the
    mixed timezones or - with ``errors="coerce"`` alone - quietly turn rows
    into NaT. ``utc=True`` parses both forms onto one axis, and the date-only
    rows simply land at midnight UTC. A test pins that all 23,412 rows parse.
    """
    frame = load_dataset(CONFIG)
    frame["Date"] = pd.to_datetime(frame["Date"], format="mixed", utc=True)
    return frame


def load_cities() -> pd.DataFrame:
    """Read the US cities reference table."""
    return pd.read_csv(project_data_dir(CONFIG.name) / CITIES_FILE)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the catalogue unchanged; the analysis works on coordinates."""
    return frame


def quakes_near_cities(
    quakes: pd.DataFrame | None = None, cities: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Join every quake to its nearest US city.

    A note on reading the result: the cities file covers the United States
    only, so for most of the planet "nearest city" means "nearest US city,
    an ocean away". The distances are meaningful near the US and an upper
    bound everywhere else.
    """
    quake_frame = quakes if quakes is not None else load_raw()
    city_frame = cities if cities is not None else load_cities()
    joined = geo.nearest_neighbour(quake_frame, city_frame, lat="Latitude", lon="Longitude")
    joined["nearest_city"] = city_frame["City"].to_numpy()[joined["nearest_index"]]
    return joined


def prepare_input(payload: dict[str, Any]) -> pd.DataFrame:
    """Not applicable: the atlas answers questions about the catalogue."""
    raise NotImplementedError(
        "earthquake_atlas analyses a catalogue. Use: python projects/earthquake_atlas/train.py"
    )
