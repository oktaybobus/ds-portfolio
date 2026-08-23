"""End-to-end test for the laptop price project.

Runs against the real dataset, which is small enough to be committed - so CI
verifies the full path from CSV to scored model on every push, not just the
unit-level behaviour of the toolkit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dsjourney.artifacts import ModelBundle
from dsjourney.training import train_supervised
from projects.laptop_price import pipeline


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return pipeline.load_raw()


@pytest.fixture(scope="module")
def features(raw: pd.DataFrame) -> pd.DataFrame:
    return pipeline.build_features(raw)


def test_raw_data_has_the_expected_shape(raw: pd.DataFrame) -> None:
    assert len(raw) == 1303
    assert {"company", "cpu", "gpu", "memory", "price"} <= set(raw.columns)


def test_build_features_produces_a_fully_numeric_frame(features: pd.DataFrame) -> None:
    assert features.select_dtypes(exclude="number").empty
    assert "price_log" in features.columns
    assert "price" not in features.columns


def test_build_features_extracts_the_packed_columns(features: pd.DataFrame) -> None:
    for column in ("ppi", "ssd_gb", "hdd_gb", "cpu_ghz", "cpu_generation", "touchscreen", "ips"):
        assert column in features.columns
    for column in ("screen_resolution", "memory", "cpu", "gpu"):
        assert column not in features.columns


def test_build_features_leaves_no_missing_values(features: pd.DataFrame) -> None:
    assert features.isna().sum().sum() == 0


def test_pixel_density_is_physically_plausible(features: pd.DataFrame) -> None:
    """Panels in this catalogue run from 1366x768 on 17" up to 4K on 12.5"."""
    assert features["ppi"].between(90, 360).all()
    assert features["ppi"].median() == pytest.approx(141, abs=5)


def test_rare_brands_are_collapsed(features: pd.DataFrame) -> None:
    assert "company_Others" in features.columns


@pytest.mark.parametrize(
    ("cpu", "expected"),
    [
        ("Intel Core i7 2.8GHz", "Intel Core i7"),
        ("Intel Celeron Dual Core N3350 1.1GHz", "Intel Celeron"),
        ("AMD Ryzen 1700 3GHz", "AMD Ryzen"),
        ("AMD A9-Series 9420 3GHz", "AMD A/E Series"),
        ("Samsung Cortex A72 2.0GHz", "Other"),
    ],
)
def test_cpu_tiers_are_classified(cpu: str, expected: str) -> None:
    assert pipeline._classify_cpu(cpu) == expected


@pytest.mark.parametrize(
    ("gpu", "expected"),
    [
        ("Nvidia GeForce GTX 1050", "Nvidia GTX"),
        ("Nvidia Quadro M1200", "Nvidia Quadro"),
        ("Intel HD Graphics 620", "Intel Graphics"),
        ("AMD Radeon R5", "AMD Radeon"),
        ("ARM Mali T860 MP4", "Other"),
    ],
)
def test_gpu_tiers_are_classified(gpu: str, expected: str) -> None:
    assert pipeline._classify_gpu(gpu) == expected


def test_non_generational_cpus_get_generation_zero() -> None:
    assert pipeline._parse_generation("Intel Celeron Dual Core N3350 1.1GHz") == 0.0
    assert pipeline._parse_generation("Intel Core i7 8550U 1.8GHz") == 8.0


@pytest.mark.slow
def test_training_beats_the_notebook_baseline(features: pd.DataFrame) -> None:
    """The source notebook reported R2 = 0.845 with LinearRegression."""
    report = train_supervised(pipeline.CONFIG, features, save=False, make_plots=False)
    assert report.bundle.metrics["r2"] > 0.85


@pytest.mark.slow
def test_a_single_prediction_lands_in_a_sane_range(features: pd.DataFrame) -> None:
    """Guards the inference path: unscaled input used to double the estimate."""
    report = train_supervised(pipeline.CONFIG, features, save=False, make_plots=False)
    bundle = ModelBundle(
        project="laptop_price",
        task="regression",
        model=report.bundle.model,
        feature_names=report.bundle.feature_names,
        scaler=report.bundle.scaler,
        scaled_columns=report.bundle.scaled_columns,
    )

    payload = {
        "company": "Dell",
        "type_name": "Gaming",
        "ram_gb": 16,
        "weight_kg": 2.5,
        "inches": 15.6,
        "screen_width": 1920,
        "screen_height": 1080,
        "ips": True,
        "ssd_gb": 512,
        "hdd_gb": 1000,
        "cpu_brand": "Intel Core i7",
        "cpu_ghz": 2.8,
        "cpu_generation": 8,
        "gpu_brand": "Nvidia GTX",
    }
    row = bundle.prepare(pipeline.prepare_input(payload))
    price = float(pipeline.postprocess(bundle.model.predict(row))[0])

    observed = np.expm1(features["price_log"])
    assert observed.min() <= price <= observed.max()
    assert 60_000 <= price <= 180_000  # a mid-to-high-end gaming laptop in this dataset
