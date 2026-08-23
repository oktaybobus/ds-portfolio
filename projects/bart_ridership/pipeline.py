"""Feature engineering for BART ridership demand.

Every trip in the raw file is an (origin, destination, hour) row with a
passenger count. Three things carry the signal, and the raw columns express
none of them directly:

- **Time is cyclical.** Hour 23 and hour 0 are one hour apart; as integers they
  are 23 apart. Encoded as a sine/cosine pair they sit next to each other.
- **Stations are places.** `station_info.csv` holds coordinates inside a free-text
  ``Location`` field, so they have to be parsed out before a distance exists.
- **Distance is not a coordinate difference.** Great-circle distance between the
  two stations is one number that captures what a subtraction of latitudes and
  longitudes does not.

The split is by year, not at random: 2016 trains, 2017 tests. A shuffled split
on time-stamped data lets a model see the same station-pair-hour on either side
of the boundary and reports a score no deployment would reproduce.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dsjourney import preprocess
from dsjourney.config import load_project_config

CONFIG = load_project_config("bart_ridership")

KAGGLE_HANDLE = "saulfuh/bart-ridership"
TRAIN_FILE = "date-hour-soo-dest-2016.csv"
TEST_FILE = "date-hour-soo-dest-2017.csv"
STATION_FILE = "station_info.csv"

# 13.3M rows fit in memory but make every experiment a coffee break. A
# deterministic sample keeps the whole pipeline honest and runnable; pass
# sample=None to train on everything.
DEFAULT_SAMPLE = 400_000
RANDOM_STATE = 42

# The Bay Area sits around 37-38 N, -121 to -123 E. The Location field stores
# the pair in an inconsistent order, so the values are identified by range
# rather than by position.
LATITUDE_RANGE = (30.0, 45.0)
LONGITUDE_RANGE = (-130.0, -110.0)

EXAMPLE_INPUT = {
    "origin": "EMBR",
    "destination": "12TH",
    "hour": 8,
    "day_of_week": 1,
    "month": 3,
}


def dataset_root() -> Path:
    """Download the BART dataset with kagglehub and return its directory."""
    try:
        import kagglehub
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "kagglehub is required for this project. Install it with: uv sync --extra data"
        ) from error
    return Path(kagglehub.dataset_download(KAGGLE_HANDLE))


def load_stations() -> pd.DataFrame:
    """Read the station table with coordinates parsed out of the Location text."""
    stations = pd.read_csv(dataset_root() / STATION_FILE)
    coordinates = stations["Location"].map(_parse_coordinates)
    return pd.DataFrame(
        {
            "station": stations["Abbreviation"].str.strip(),
            "station_name": stations["Name"],
            "latitude": [pair[0] for pair in coordinates],
            "longitude": [pair[1] for pair in coordinates],
        }
    ).dropna(subset=["latitude", "longitude"])


def load_raw(*, sample: int | None = DEFAULT_SAMPLE) -> pd.DataFrame:
    """Read both years of trips, tagged with the year they came from.

    Args:
        sample: Rows to keep per year, sampled deterministically. ``None`` reads
            all 13.3 million.
    """
    root = dataset_root()
    frames = []
    for year, file in ((2016, TRAIN_FILE), (2017, TEST_FILE)):
        frame = pd.read_csv(root / file)
        if sample is not None and len(frame) > sample:
            frame = frame.sample(n=sample, random_state=RANDOM_STATE)
        frames.append(frame.assign(year=year))
    return pd.concat(frames, ignore_index=True)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn raw trips into a numeric frame with a ``throughput_log`` target."""
    stations = load_stations()

    joined = (
        frame.rename(columns={"Origin": "origin", "Destination": "destination"})
        .pipe(preprocess.add_calendar_features, "DateTime", drop=True)
        .pipe(_join_station_coordinates, stations)
        .pipe(_add_geography)
        .pipe(_encode_time)
    )
    # Throughput is the target: leaving the raw column in the feature set would
    # let the model read the answer. `year` is split bookkeeping, not a feature -
    # it is constant within each half and would only teach the model which side
    # of the split it is on.
    numeric = preprocess.drop_columns(joined, ["origin", "destination", "Throughput", "year"])
    with_target = numeric.assign(throughput=pd.to_numeric(frame["Throughput"], errors="coerce"))
    return preprocess.log_transform_target(with_target.dropna(), "throughput", "throughput_log")


def chronological_frames(sample: int | None = DEFAULT_SAMPLE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (2016 features, 2017 features) - the split that respects time.

    Rows dropped during feature building are dropped from the year labels too,
    by aligning on the index rather than assuming the two stay the same length.
    """
    raw = load_raw(sample=sample).reset_index(drop=True)
    features = build_features(raw)
    year = raw.loc[features.index, "year"]
    return features[year == 2016], features[year == 2017]


def prepare_input(payload: dict[str, Any]) -> pd.DataFrame:
    """Build a single model-ready row from a trip description."""
    stations = load_stations().set_index("station")
    origin = str(payload.get("origin", "EMBR"))
    destination = str(payload.get("destination", "12TH"))

    row = pd.DataFrame(
        [
            {
                "hour": int(payload.get("hour", 8)),
                "day_of_week": int(payload.get("day_of_week", 1)),
                "month": int(payload.get("month", 3)),
                "is_weekend": int(int(payload.get("day_of_week", 1)) >= 5),
                "origin_latitude": _coordinate(stations, origin, "latitude"),
                "origin_longitude": _coordinate(stations, origin, "longitude"),
                "destination_latitude": _coordinate(stations, destination, "latitude"),
                "destination_longitude": _coordinate(stations, destination, "longitude"),
                "same_station": int(origin == destination),
            }
        ]
    )
    with_distance = _add_geography(row)
    return _encode_time(with_distance)


def postprocess(prediction: np.ndarray) -> np.ndarray:
    """Convert a log-scale prediction back to a passenger count."""
    return preprocess.inverse_log_transform(prediction)


def _join_station_coordinates(frame: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Attach origin and destination coordinates."""
    lookup = stations.set_index("station")[["latitude", "longitude"]]
    joined = frame.join(lookup.add_prefix("origin_"), on="origin")
    joined = joined.join(lookup.add_prefix("destination_"), on="destination")
    return joined.assign(same_station=(frame["origin"] == frame["destination"]).astype(int))


def _add_geography(frame: pd.DataFrame) -> pd.DataFrame:
    """Add great-circle distance between the two stations."""
    return frame.assign(
        distance_km=preprocess.haversine_km(
            frame["origin_latitude"],
            frame["origin_longitude"],
            frame["destination_latitude"],
            frame["destination_longitude"],
        )
    )


def _encode_time(frame: pd.DataFrame) -> pd.DataFrame:
    """Project hour, weekday and month onto circles."""
    return (
        frame.pipe(preprocess.add_cyclical, "hour", period=24)
        .pipe(preprocess.add_cyclical, "day_of_week", period=7)
        .pipe(preprocess.add_cyclical, "month", period=12)
    )


def _parse_coordinates(value: Any) -> tuple[float | None, float | None]:
    """Pull a (latitude, longitude) pair out of the free-text Location field.

    The field stores ``"-122.271450,37.803768,0"`` - longitude first here, but
    not consistently - so each number is assigned by which range it falls in
    rather than by its position.
    """
    numbers = [float(match) for match in re.findall(r"[-+]?\d*\.\d+|\d+", str(value))]
    latitude = next((n for n in numbers if LATITUDE_RANGE[0] < n < LATITUDE_RANGE[1]), None)
    longitude = next((n for n in numbers if LONGITUDE_RANGE[0] < n < LONGITUDE_RANGE[1]), None)
    return latitude, longitude


def _coordinate(stations: pd.DataFrame, station: str, column: str) -> float:
    """Look up one station coordinate, falling back to the network centre."""
    if station in stations.index:
        return float(stations.loc[station, column])
    return float(stations[column].mean())
