"""Feature engineering for Istanbul apartment prices.

The listings come from a scrape of Emlakjet, so almost every field is free text
written for a human: room counts as ``"4.5+1"``, areas as ``"292 m2"``, building
age as ``"21 Ve Uzeri"``. Turning those into numbers is the bulk of this module.

Column names arrive with Turkish characters and spaces (``"Oda Sayisi"``,
``"Bina Yasi"``), which makes every downstream reference fragile;
:func:`normalise_columns` folds them to ASCII snake_case once, at the door.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dsjourney import preprocess
from dsjourney.config import load_project_config
from dsjourney.datasets import load_dataset

CONFIG = load_project_config("istanbul_housing")

TURKISH_TO_ASCII = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")

# Dropped rather than encoded: identifiers and free-text titles carry no
# transferable signal, and the "selected province/district" columns are scraper
# bookkeeping that duplicates ilce. The remaining four are >90% missing.
DROPPED_COLUMNS = [
    "url",
    "baslik",
    "il",
    "secilen_il",
    "secilen_ilce",
    "kaynak_dosya",
    "isitma",
    "yapi_durumu",
    "kullanim_durumu",
    "krediye_uygunluk",
    "tapu_durumu",
]

CATEGORICAL_COLUMNS = ["ilce", "mahalle"]

# Ranges map to their midpoint; the open-ended bands take a representative value
# rather than being dropped.
#
# The two "0 (...)" labels matter more than they look: together they are 2,838
# listings, a quarter of the file. The source notebook's map omitted them along
# with "21-25" and "31 Ve Uzeri". Those rows became NaN, and the notebook's
# dropna() then discarded 3,264 listings - 30% of the data, and the entire
# new-build segment. A test asserts every label in the file is covered here.
BUILDING_AGE_MAP = {
    "0": 0.0,
    "0 (Oturuma Hazır)": 0.0,
    "0 (Yapım Aşamasında)": 0.0,
    "1": 1.0,
    "2": 2.0,
    "3": 3.0,
    "4": 4.0,
    "5": 5.0,
    "6-10": 8.0,
    "11-15": 13.0,
    "16-20": 18.0,
    "21-25": 23.0,
    "21 Ve Üzeri": 25.0,
    "26-30": 28.0,
    "31 Ve Üzeri": 35.0,
}

# Listings above this are penthouses and land-plot mislistings whose prices are
# an order of magnitude above the rest; keeping them makes the model fit them
# instead of the 99% of listings a user would actually search for.
MAX_PRICE_MILLIONS = 50.0
# Neighbourhood is the strongest price driver in this city, so the collapse
# threshold is a real modelling decision, not tidying. Measured on the holdout:
# min_count 50 -> R2 0.741, 25 -> 0.761, 10 -> 0.780, 5 -> 0.781, 1 -> 0.778.
# Five keeps almost all the location signal while still folding away singletons
# that would only add noise columns.
MIN_NEIGHBOURHOOD_LISTINGS = 5


def load_raw() -> pd.DataFrame:
    """Read the scraped listings and normalise their column names."""
    return normalise_columns(load_dataset(CONFIG)).rename(columns={"fiyat_tl": "fiyat"})


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Fold Turkish column names to ASCII snake_case."""
    renamed = {
        column: str(column).strip().translate(TURKISH_TO_ASCII).lower().replace(" ", "_")
        for column in frame.columns
    }
    return frame.rename(columns=renamed)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn raw listings into a numeric frame with a ``fiyat_log`` target."""
    prepared = (
        preprocess.drop_columns(frame, DROPPED_COLUMNS)
        .pipe(_price_in_millions)
        .pipe(_parse_areas)
        .pipe(_parse_rooms)
        .pipe(_parse_building_age)
        .pipe(preprocess.group_rare_categories, "mahalle", min_count=MIN_NEIGHBOURHOOD_LISTINGS)
        # Impute before deriving: a ratio built from a missing input is missing
        # too, and the final dropna would then discard the row anyway.
        .pipe(preprocess.impute_numeric, ["bina_yasi", "toplam_oda"])
        .pipe(_add_ratios)
        .pipe(preprocess.impute_numeric, ["net_brut_orani", "oda_basi_net_m2"])
    )
    encoded = preprocess.one_hot(prepared, CATEGORICAL_COLUMNS, drop_first=True)
    return preprocess.log_transform_target(encoded.dropna(), "fiyat", "fiyat_log")


def prepare_input(payload: dict[str, Any]) -> pd.DataFrame:
    """Build a single model-ready row from a listing description."""
    brut = float(payload.get("brut_m2", 120))
    net = float(payload.get("net_m2", 100))
    rooms = float(payload.get("toplam_oda", 3.5))
    age = float(payload.get("bina_yasi", 8))

    row = pd.DataFrame(
        [
            {
                "brut_m2": brut,
                "net_m2": net,
                "toplam_oda": rooms,
                "bina_yasi": age,
                "net_brut_orani": net / brut if brut else np.nan,
                "oda_basi_net_m2": net / rooms if rooms else np.nan,
                "yas_alan_etkisi": age * net,
                "ilce": payload.get("ilce", "Kadıköy"),
                "mahalle": payload.get("mahalle", "Others"),
            }
        ]
    )
    return preprocess.one_hot(row, CATEGORICAL_COLUMNS, drop_first=False)


def postprocess(prediction: np.ndarray) -> np.ndarray:
    """Convert a log-scale prediction back to millions of Turkish lira."""
    return preprocess.inverse_log_transform(prediction)


def _price_in_millions(frame: pd.DataFrame) -> pd.DataFrame:
    """Rescale prices to millions and clip the luxury tail.

    Working in millions keeps the target in a range where a 0.01 difference is
    meaningful; raw lira values run to nine figures and make every plot
    unreadable.
    """
    scaled = frame.assign(fiyat=pd.to_numeric(frame["fiyat"], errors="coerce") / 1_000_000)
    return scaled[scaled["fiyat"].between(0.1, MAX_PRICE_MILLIONS)]


def _parse_areas(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn ``"292 m2"`` into a number, dropping rows whose area is unparseable."""
    parsed = frame.assign(
        brut_m2=_area_to_number(frame["brut"]),
        net_m2=_area_to_number(frame["net"]),
    )
    cleaned = parsed.dropna(subset=["brut_m2", "net_m2"])
    return preprocess.drop_columns(cleaned, ["brut", "net"])


def _parse_rooms(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn ``"4.5+1"`` into 5.5 total rooms.

    Turkish listings write rooms as bedrooms plus living rooms, and a half
    counts. Summing both sides is what the number means to a buyer.
    """
    totals = frame["oda_sayisi"].map(_room_count)
    return preprocess.drop_columns(frame.assign(toplam_oda=totals), ["oda_sayisi"])


def _parse_building_age(frame: pd.DataFrame) -> pd.DataFrame:
    """Map age bands such as ``"6-10"`` to their midpoint."""
    ages = frame["bina_yasi"].astype(str).str.strip().map(BUILDING_AGE_MAP)
    return frame.assign(bina_yasi=pd.to_numeric(ages, errors="coerce"))


def _add_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive layout-efficiency features from area and room count.

    ``net_brut_orani`` is how much of the paid-for area is usable, which varies
    a lot between older buildings and new complexes; ``oda_basi_net_m2`` says
    whether those rooms are spacious or merely numerous.
    """
    return frame.assign(
        net_brut_orani=frame["net_m2"] / frame["brut_m2"].replace(0, np.nan),
        oda_basi_net_m2=frame["net_m2"] / frame["toplam_oda"].replace(0, np.nan),
        yas_alan_etkisi=frame["bina_yasi"] * frame["net_m2"],
    )


def _area_to_number(series: pd.Series) -> pd.Series:
    """Strip the ``m2`` suffix and thousands separators from an area column."""
    text = series.astype(str).str.replace("m²", "", regex=False).str.replace("m2", "", regex=False)
    return pd.to_numeric(text.str.replace(",", ".").str.strip(), errors="coerce")


def _room_count(value: Any) -> float:
    """Sum both sides of a ``"3+1"`` style room count."""
    text = str(value)
    if "+" not in text:
        return float("nan")
    try:
        left, right = text.split("+", 1)
        return float(left) + float(right)
    except ValueError:
        return float("nan")
