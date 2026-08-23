"""Unit tests for the shared training orchestration."""

from __future__ import annotations

import pandas as pd
import pytest

from dsjourney.config import ProjectConfig
from dsjourney.training import train_clustering, train_supervised, train_text_classifier

REGRESSION = {
    "name": "unit_regression",
    "title": "Unit regression",
    "task": "regression",
    "target": "target",
    "dataset": {"id": "x", "file": "x.csv"},
    "model": {"estimator": "linear", "scale_features": ["feature_a"]},
    "split": {"test_size": 0.3},
}

CLASSIFICATION = {
    "name": "unit_classification",
    "title": "Unit classification",
    "task": "classification",
    "target": "target",
    "dataset": {"id": "x", "file": "x.csv"},
    "model": {"estimator": "logistic"},
    "split": {"test_size": 0.3, "stratify": True},
}

CLUSTERING = {
    "name": "unit_clustering",
    "title": "Unit clustering",
    "task": "clustering",
    "target": None,
    "dataset": {"id": "x", "file": "x.csv"},
    "model": {"estimator": "kmeans", "params": {"n_clusters": 2, "random_state": 0}},
}


def test_train_supervised_learns_the_signal(regression_frame: pd.DataFrame) -> None:
    config = ProjectConfig.model_validate(REGRESSION)
    report = train_supervised(config, regression_frame, save=False, make_plots=False)
    assert report.bundle.metrics["r2"] > 0.95
    assert report.bundle.scaled_columns == ["feature_a"]
    assert report.bundle.extra["selected_by"] == "config"


def test_train_supervised_benchmark_picks_a_winner(regression_frame: pd.DataFrame) -> None:
    config = ProjectConfig.model_validate(REGRESSION)
    report = train_supervised(
        config, regression_frame, benchmark=True, save=False, make_plots=False
    )
    assert report.benchmark is not None
    assert report.bundle.extra["selected_by"] == "benchmark"


def test_train_supervised_requires_a_target(regression_frame: pd.DataFrame) -> None:
    config = ProjectConfig.model_validate({**REGRESSION, "target": None})
    with pytest.raises(ValueError, match="declares no target"):
        train_supervised(config, regression_frame, save=False)


def test_train_supervised_scores_a_classifier(classification_frame: pd.DataFrame) -> None:
    config = ProjectConfig.model_validate(CLASSIFICATION)
    report = train_supervised(config, classification_frame, save=False, make_plots=False)
    assert report.bundle.metrics["f1"] > 0.85
    assert "roc_auc" in report.bundle.metrics


def test_train_clustering_scans_k_and_scores(regression_frame: pd.DataFrame) -> None:
    config = ProjectConfig.model_validate(CLUSTERING)
    features = regression_frame[["feature_a", "feature_b"]]
    report = train_clustering(config, features, k_range=range(2, 5), save=False, make_plots=False)
    assert report.bundle.extra["n_clusters"] == 2
    assert set(report.bundle.extra["k_scan"]) == {"2", "3", "4"}
    assert "silhouette" in report.bundle.metrics


def test_train_text_classifier_fits_a_pipeline(review_frame: pd.DataFrame) -> None:
    config = ProjectConfig.model_validate(
        {
            "name": "unit_text",
            "title": "Unit text",
            "task": "text-classification",
            "target": "label",
            "dataset": {"id": "x", "file": "x.csv"},
            "model": {"estimator": "logistic"},
            "split": {"test_size": 0.3},
        }
    )
    labels = (review_frame["stars"] >= 4).astype(int)
    report = train_text_classifier(
        config, review_frame["text"], labels, save=False, make_plots=False
    )
    assert report.bundle.extra["vocabulary_size"] > 0
    assert "f1" in report.bundle.metrics


def test_summary_is_a_single_line(regression_frame: pd.DataFrame) -> None:
    config = ProjectConfig.model_validate(REGRESSION)
    report = train_supervised(config, regression_frame, save=False, make_plots=False)
    assert "\n" not in report.summary()
    assert "r2=" in report.summary()
