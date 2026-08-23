"""Loading and cleaning the Pima clinical records.

Five columns encode a missing measurement as ``0``: a living patient does not
have a glucose level, blood pressure, skinfold thickness, insulin level or BMI
of zero. Insulin is 48.7% zeros and skinfold thickness 29.6%, so the source
notebook handed nearly half its insulin column to the model as a real reading
at the bottom of the scale. Nothing raises; the model simply learns that a
large cluster of patients has no insulin.

:data:`ZERO_IS_MISSING` names those columns and :func:`build_features` turns the
zeros into nulls, which is what lets MLlib's ``Imputer`` see them. The count is
asserted in ``tests/projects/test_diabetes_screening.py`` against the shipped
file, so a replacement dataset with different conventions fails loudly rather
than silently reverting the fix.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dsjourney.config import load_project_config
from dsjourney.datasets import load_dataset

CONFIG = load_project_config("diabetes_screening")

TARGET = "Outcome"

# Columns where 0 is physiologically impossible and therefore means "not
# recorded". Pregnancies is deliberately absent: zero pregnancies is a fact.
ZERO_IS_MISSING = (
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
)

FEATURES = (
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
)


def load_raw() -> pd.DataFrame:
    """Read the 768 clinical records exactly as published."""
    return load_dataset(CONFIG)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace the impossible zeros with nulls, leaving imputation to the model.

    Imputing here would fit the fill value on every row, test rows included -
    the leak that :mod:`dsjourney.preprocess` exists to avoid. The nulls are
    carried into the pipeline instead and filled from training-fold statistics.
    """
    cleaned = frame.copy()
    for column in ZERO_IS_MISSING:
        cleaned[column] = cleaned[column].replace(0, np.nan)
    return cleaned


def missing_after_cleaning(frame: pd.DataFrame | None = None) -> pd.Series:
    """Count the encoded-missing values per column, for the README and the tests."""
    cleaned = build_features(frame if frame is not None else load_raw())
    return cleaned[list(ZERO_IS_MISSING)].isna().sum()


def prepare_input(payload: dict[str, Any]) -> pd.DataFrame:
    """Build a one-row frame from a request payload, in training column order."""
    row = {name: payload.get(name) for name in FEATURES}
    return build_features(pd.DataFrame([row]))
