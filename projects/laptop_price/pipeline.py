"""Feature engineering for laptop price prediction.

The dataset ships three columns that are really packed records: screen
resolution ("IPS Panel Retina Display 2560x1600"), storage ("128GB SSD +
1TB HDD") and CPU ("Intel Core i5 2.3GHz"). Most of the signal is inside those
strings, so the bulk of this module is turning them into numbers - pixel
density, SSD and HDD capacity, clock speed and CPU generation.

Everything generic - splitting, scaling, fitting, scoring - lives in dsjourney.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dsjourney import preprocess
from dsjourney.config import load_project_config
from dsjourney.datasets import load_dataset

CONFIG = load_project_config("laptop_price")

RAW_COLUMN_RENAMES = {
    "Company": "company",
    "TypeName": "type_name",
    "Inches": "inches",
    "ScreenResolution": "screen_resolution",
    "Cpu": "cpu",
    "Ram": "ram_gb",
    "Memory": "memory",
    "Gpu": "gpu",
    "OpSys": "operating_system",
    "Weight": "weight_kg",
    "Price": "price",
}

CATEGORICAL_COLUMNS = ["company", "type_name", "cpu_brand", "gpu_brand"]

# Operating system is dropped rather than encoded: in this dataset it is almost
# perfectly collinear with brand (every macOS row is an Apple row), so keeping it
# adds columns without adding information.
DROPPED_COLUMNS = ["index", "operating_system"]

MIN_BRAND_COUNT = 15


def load_raw() -> pd.DataFrame:
    """Read the laptop dataset and normalise its column names."""
    frame = load_dataset(CONFIG)
    renames = {**RAW_COLUMN_RENAMES, "Unnamed: 0": "index"}
    return frame.rename(columns=renames)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw listing table into a numeric, model-ready frame.

    Returns a frame whose only non-feature column is the ``price_log`` target.
    """
    prepared = (
        frame.pipe(preprocess.drop_columns, DROPPED_COLUMNS)
        .pipe(preprocess.strip_unit, "ram_gb", "GB", dtype="int")
        .pipe(preprocess.strip_unit, "weight_kg", "kg", dtype="float")
        .pipe(preprocess.group_rare_categories, "company", min_count=MIN_BRAND_COUNT)
        .pipe(_add_screen_features)
        .pipe(_add_storage_features)
        .pipe(_add_gpu_brand)
        .pipe(_add_cpu_features)
    )
    encoded = preprocess.one_hot(prepared, CATEGORICAL_COLUMNS, drop_first=True)
    return preprocess.log_transform_target(encoded, "price", "price_log")


def prepare_input(payload: dict[str, Any]) -> pd.DataFrame:
    """Build a single model-ready row from a user-supplied specification.

    Accepts the human-facing fields a demo form collects (brand, RAM, screen
    size, resolution, storage, CPU) and derives the same features the model was
    trained on. The caller aligns the result to the saved column order with
    :func:`dsjourney.preprocess.align_to_training_columns`.
    """
    width = float(payload.get("screen_width", 1920))
    height = float(payload.get("screen_height", 1080))
    inches = float(payload.get("inches", 15.6))
    ghz = float(payload.get("cpu_ghz", 2.5))
    generation = float(payload.get("cpu_generation", 8))

    row = pd.DataFrame(
        [
            {
                "company": payload.get("company", "Others"),
                "type_name": payload.get("type_name", "Notebook"),
                "ram_gb": int(payload.get("ram_gb", 8)),
                "weight_kg": float(payload.get("weight_kg", 2.0)),
                "touchscreen": int(bool(payload.get("touchscreen", False))),
                "ips": int(bool(payload.get("ips", False))),
                "ppi": _pixel_density(width, height, inches),
                "ssd_gb": int(payload.get("ssd_gb", 256)),
                "hdd_gb": int(payload.get("hdd_gb", 0)),
                "cpu_brand": payload.get("cpu_brand", "Intel Core i5"),
                "cpu_ghz": ghz,
                "cpu_generation": generation,
                "cpu_performance": ghz * generation,
                "gpu_brand": payload.get("gpu_brand", "Intel Graphics"),
            }
        ]
    )
    return preprocess.one_hot(row, CATEGORICAL_COLUMNS, drop_first=False)


def postprocess(prediction: np.ndarray) -> np.ndarray:
    """Convert a log-scale prediction back to a currency amount."""
    return preprocess.inverse_log_transform(prediction)


def _add_screen_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive touchscreen, IPS and pixel density, then drop the raw text column.

    Pixel density (PPI) captures resolution and physical size in one number and
    correlates with price far better than either does alone, which is why the
    three inputs are collapsed rather than kept.
    """
    with_flags = (
        frame.pipe(preprocess.binary_flag, "screen_resolution", "Touchscreen", "touchscreen")
        .pipe(preprocess.binary_flag, "screen_resolution", "IPS", "ips")
        .pipe(
            preprocess.extract_pattern,
            "screen_resolution",
            r"(\d+)x(\d+)",
            ["x_resolution", "y_resolution"],
        )
    )
    with_ppi = with_flags.assign(
        ppi=_pixel_density(
            with_flags["x_resolution"], with_flags["y_resolution"], with_flags["inches"]
        )
    )
    return preprocess.drop_columns(
        with_ppi, ["screen_resolution", "inches", "x_resolution", "y_resolution"]
    )


def _add_storage_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Split the free-text memory column into SSD and HDD capacities in GB."""
    normalised = (
        frame["memory"]
        .astype(str)
        .str.replace("GB", "", regex=False)
        .str.replace("TB", "000", regex=False)
    )
    with_storage = frame.assign(
        ssd_gb=normalised.str.extract(r"(\d+)\s*SSD")[0].fillna(0).astype(int),
        hdd_gb=normalised.str.extract(r"(\d+)\s*HDD")[0].fillna(0).astype(int),
    )
    return preprocess.drop_columns(with_storage, ["memory"])


def _add_gpu_brand(frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce ~110 distinct GPU model strings to a handful of performance tiers."""
    return preprocess.drop_columns(frame.assign(gpu_brand=frame["gpu"].map(_classify_gpu)), ["gpu"])


def _add_cpu_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive CPU tier, clock speed, generation and their product."""
    ghz = frame["cpu"].map(_parse_clock_speed)
    generation = frame["cpu"].map(_parse_generation)
    with_cpu = frame.assign(
        cpu_brand=frame["cpu"].map(_classify_cpu),
        cpu_ghz=ghz,
        cpu_generation=generation,
        # A 2.3GHz 8th-generation chip outruns a 2.3GHz 3rd-generation one, so
        # the interaction carries information neither factor holds alone.
        cpu_performance=ghz * generation,
    )
    return preprocess.drop_columns(with_cpu, ["cpu"])


def _pixel_density(width: Any, height: Any, inches: Any) -> Any:
    """Return pixels per inch from a resolution and a diagonal screen size."""
    return np.sqrt(np.square(width) + np.square(height)) / inches


def _classify_gpu(text: str) -> str:
    """Map a GPU model string to a coarse performance tier."""
    value = str(text)
    if "Nvidia" in value:
        for tier in ("GTX", "RTX", "Quadro"):
            if tier in value:
                return f"Nvidia {tier}"
        return "Nvidia Other"
    if "Intel" in value:
        return "Intel Graphics"
    if "AMD" in value:
        return "AMD Radeon"
    return "Other"


def _classify_cpu(text: str) -> str:
    """Map a CPU model string to a market tier such as 'Intel Core i7'."""
    value = str(text)
    if "Intel Core i" in value:
        return " ".join(value.split()[:3])
    for keyword, label in (
        ("Celeron", "Intel Celeron"),
        ("Pentium", "Intel Pentium"),
        ("Core M", "Intel Core M"),
        ("Atom", "Intel Atom"),
    ):
        if keyword in value:
            return label
    if "AMD" in value:
        if "Ryzen" in value:
            return "AMD Ryzen"
        if any(series in value for series in ("A-Series", "E-Series", "A6", "A9", "A10", "A12")):
            return "AMD A/E Series"
        return "AMD Other"
    return "Other"


def _parse_clock_speed(text: str) -> float:
    """Read the trailing clock speed, e.g. 'Intel Core i5 2.3GHz' -> 2.3."""
    try:
        return float(str(text).split()[-1].replace("GHz", ""))
    except (ValueError, IndexError):
        return float("nan")


def _parse_generation(text: str) -> float:
    """Read the Intel Core generation digit; 0 for chips that have no generation.

    Celeron, Pentium and Atom parts are not generational, so they get 0 rather
    than a missing value the imputer would otherwise have to invent.
    """
    parts = str(text).split()
    if len(parts) > 3 and parts[3][:1].isdigit():
        return float(parts[3][0])
    return 0.0
