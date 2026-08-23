"""Metrics.

One scoring function per task family, each returning a plain dict so results can
be printed, asserted on in a test, and written straight to ``metrics.json`` next
to the model without any further formatting.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics as skm

ArrayLike = np.ndarray | pd.Series | list[float]


def regression_scores(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Return R2, RMSE, MAE and MAPE for a regression model."""
    truth = np.asarray(y_true, dtype=float).ravel()
    prediction = np.asarray(y_pred, dtype=float).ravel()
    return {
        "r2": float(skm.r2_score(truth, prediction)),
        "rmse": float(np.sqrt(skm.mean_squared_error(truth, prediction))),
        "mae": float(skm.mean_absolute_error(truth, prediction)),
        "mape": float(skm.mean_absolute_percentage_error(truth, prediction)),
    }


def classification_scores(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    y_proba: ArrayLike | None = None,
    average: str = "binary",
) -> dict[str, float]:
    """Return accuracy, precision, recall, F1 and - when probabilities are given - ROC AUC.

    Accuracy alone is misleading on the imbalanced datasets in this portfolio
    (loan defaults, fraud), so recall and F1 are always reported alongside it.
    """
    truth = np.asarray(y_true).ravel()
    prediction = np.asarray(y_pred).ravel()
    kwargs: dict[str, Any] = {"average": average, "zero_division": 0}
    scores = {
        "accuracy": float(skm.accuracy_score(truth, prediction)),
        "precision": float(skm.precision_score(truth, prediction, **kwargs)),
        "recall": float(skm.recall_score(truth, prediction, **kwargs)),
        "f1": float(skm.f1_score(truth, prediction, **kwargs)),
    }
    if y_proba is not None:
        probabilities = np.asarray(y_proba, dtype=float)
        if probabilities.ndim == 2 and probabilities.shape[1] == 2:
            probabilities = probabilities[:, 1]
        try:
            scores["roc_auc"] = float(skm.roc_auc_score(truth, probabilities))
        except ValueError:
            scores["roc_auc"] = float("nan")
    return scores


def clustering_scores(features: pd.DataFrame | np.ndarray, labels: ArrayLike) -> dict[str, float]:
    """Return silhouette, Calinski-Harabasz and Davies-Bouldin for a clustering.

    Silhouette is undefined for a single cluster, so it is reported as NaN rather
    than crashing the run.
    """
    values = np.asarray(features, dtype=float)
    assignments = np.asarray(labels).ravel()
    if len(set(assignments.tolist())) < 2:
        return {
            "silhouette": float("nan"),
            "calinski_harabasz": float("nan"),
            "davies_bouldin": float("nan"),
        }
    return {
        "silhouette": float(skm.silhouette_score(values, assignments)),
        "calinski_harabasz": float(skm.calinski_harabasz_score(values, assignments)),
        "davies_bouldin": float(skm.davies_bouldin_score(values, assignments)),
    }


def confusion_frame(
    y_true: ArrayLike, y_pred: ArrayLike, *, labels: list[Any] | None = None
) -> pd.DataFrame:
    """Return the confusion matrix as a labelled DataFrame."""
    matrix = skm.confusion_matrix(
        np.asarray(y_true).ravel(), np.asarray(y_pred).ravel(), labels=labels
    )
    names = labels if labels is not None else sorted(set(np.asarray(y_true).ravel().tolist()))
    index = pd.Index([f"true_{n}" for n in names], name="actual")
    columns = pd.Index([f"pred_{n}" for n in names], name="predicted")
    return pd.DataFrame(matrix, index=index, columns=columns)


def classification_report_frame(y_true: ArrayLike, y_pred: ArrayLike) -> pd.DataFrame:
    """Return sklearn's per-class report as a DataFrame instead of a text blob."""
    report = skm.classification_report(
        np.asarray(y_true).ravel(), np.asarray(y_pred).ravel(), output_dict=True, zero_division=0
    )
    return pd.DataFrame(report).transpose()


def score(task: str, y_true: ArrayLike, y_pred: ArrayLike, **kwargs: Any) -> dict[str, float]:
    """Dispatch to the scorer that matches a project's declared task."""
    if task in {"regression"}:
        return regression_scores(y_true, y_pred)
    if task in {"classification", "text-classification", "image-classification"}:
        return classification_scores(y_true, y_pred, **kwargs)
    raise ValueError(f"no scorer for task {task!r}; clustering uses clustering_scores()")
