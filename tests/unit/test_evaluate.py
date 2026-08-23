"""Unit tests for the metric functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dsjourney import evaluate


def test_regression_scores_are_perfect_on_exact_predictions() -> None:
    truth = [1.0, 2.0, 3.0, 4.0]
    scores = evaluate.regression_scores(truth, truth)
    assert scores["r2"] == pytest.approx(1.0)
    assert scores["rmse"] == pytest.approx(0.0)
    assert scores["mae"] == pytest.approx(0.0)


def test_regression_rmse_is_the_root_of_mse() -> None:
    scores = evaluate.regression_scores([0.0, 0.0], [3.0, 4.0])
    assert scores["rmse"] == pytest.approx(3.5355, abs=1e-3)


def test_classification_scores_cover_the_imbalanced_case() -> None:
    """A model that predicts the majority class scores well on accuracy and badly on recall."""
    truth = [0] * 90 + [1] * 10
    predicted = [0] * 100
    scores = evaluate.classification_scores(truth, predicted)
    assert scores["accuracy"] == pytest.approx(0.9)
    assert scores["recall"] == pytest.approx(0.0)
    assert scores["f1"] == pytest.approx(0.0)


def test_classification_scores_add_roc_auc_when_probabilities_are_given() -> None:
    truth = [0, 0, 1, 1]
    predicted = [0, 0, 1, 1]
    probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.1, 0.9]])
    scores = evaluate.classification_scores(truth, predicted, y_proba=probabilities)
    assert scores["roc_auc"] == pytest.approx(1.0)


def test_classification_scores_survive_a_single_class_roc() -> None:
    scores = evaluate.classification_scores([1, 1], [1, 1], y_proba=[0.9, 0.8])
    assert np.isnan(scores["roc_auc"])


def test_clustering_scores_need_at_least_two_clusters() -> None:
    features = np.random.default_rng(0).normal(size=(20, 3))
    scores = evaluate.clustering_scores(features, np.zeros(20))
    assert np.isnan(scores["silhouette"])


def test_clustering_scores_reward_well_separated_groups() -> None:
    features = np.vstack([np.zeros((10, 2)), np.full((10, 2), 100.0)])
    labels = np.array([0] * 10 + [1] * 10)
    assert evaluate.clustering_scores(features, labels)["silhouette"] > 0.9


def test_confusion_frame_is_labelled() -> None:
    result = evaluate.confusion_frame([0, 1, 1], [0, 1, 0])
    assert list(result.index) == ["true_0", "true_1"]
    assert list(result.columns) == ["pred_0", "pred_1"]


def test_classification_report_frame_returns_a_dataframe() -> None:
    result = evaluate.classification_report_frame([0, 1, 1, 0], [0, 1, 0, 0])
    assert isinstance(result, pd.DataFrame)
    assert "precision" in result.columns


def test_score_dispatches_on_task() -> None:
    assert "r2" in evaluate.score("regression", [1.0, 2.0], [1.0, 2.0])
    assert "f1" in evaluate.score("classification", [0, 1], [0, 1])


def test_score_rejects_clustering() -> None:
    with pytest.raises(ValueError, match="clustering uses clustering_scores"):
        evaluate.score("clustering", [0], [0])
