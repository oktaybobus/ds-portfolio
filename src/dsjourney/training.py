"""Shared training orchestration.

A project supplies two things: a config and a function that turns the raw frame
into model-ready features. Everything after that - splitting, optional model
sweep, fitting, scoring, saving - is identical across projects and lives here.
That is what lets each ``projects/<name>/train.py`` stay under a hundred lines.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dsjourney import evaluate, viz
from dsjourney.artifacts import ModelBundle, save_bundle, save_table
from dsjourney.benchmark import BenchmarkResult, build_model, compare_models
from dsjourney.config import ProjectConfig
from dsjourney.preprocess import SplitResult, split_and_scale

FeatureBuilder = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class TrainingReport:
    """Everything one training run produced."""

    bundle: ModelBundle
    split: SplitResult
    predictions: np.ndarray
    benchmark: BenchmarkResult | None = None
    artifacts_dir: Path | None = None

    def summary(self) -> str:
        """A one-line, log-friendly summary of the run."""
        scores = ", ".join(f"{k}={v:.4f}" for k, v in self.bundle.metrics.items() if _is_finite(v))
        return f"{self.bundle.project}: {type(self.bundle.model).__name__} -> {scores}"


def train_supervised(
    config: ProjectConfig,
    frame: pd.DataFrame,
    *,
    benchmark: bool = False,
    save: bool = True,
    make_plots: bool = True,
    inverse_transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> TrainingReport:
    """Split, fit and score a supervised model, then persist the bundle.

    Args:
        frame: Model-ready features including the target column.
        benchmark: Sweep every registered estimator and keep the winner instead
            of using the one named in the config.
        save: Write the model, metrics and plots to ``artifacts/<project>/``.
        make_plots: Render the diagnostic figures (skipped by fast unit tests).
        inverse_transform: For a project trained on a transformed target, the
            function that undoes it. A second set of metrics suffixed
            ``_original`` is then reported on the scale a reader cares about -
            R2 on ``log1p(price)`` and R2 on price are different numbers, and
            quoting one as the other is how portfolio metrics stop being
            comparable to anything.
    """
    if config.target is None:
        raise ValueError(f"project '{config.name}' is supervised but declares no target")

    split = split_and_scale(
        frame, config.target, config.split, scale_columns=config.model.scale_features
    )

    sweep: BenchmarkResult | None = None
    if benchmark:
        sweep = compare_models(
            config.task, split.x_train, split.y_train, split.x_test, split.y_test
        )
        model = sweep.best_model
        chosen = sweep.best_name
    else:
        model = build_model(config.task, config.model.estimator, **config.model.params)
        model.fit(split.x_train, np.ravel(split.y_train))
        chosen = config.model.estimator

    predictions = model.predict(split.x_test)
    metrics = _score_supervised(config, model, split, predictions)
    if inverse_transform is not None and config.task == "regression":
        metrics |= _original_scale_scores(split.y_test, predictions, inverse_transform)

    bundle = ModelBundle(
        project=config.name,
        task=config.task,
        model=model,
        feature_names=split.feature_names,
        metrics=metrics,
        scaler=split.scaler,
        scaled_columns=list(config.model.scale_features),
        extra={
            "estimator_key": chosen,
            "selected_by": "benchmark" if benchmark else "config",
            "target": config.target,
            "train_rows": len(split.x_train),
            "test_rows": len(split.x_test),
        },
    )

    directory = None
    if save:
        directory = save_bundle(bundle)
        if sweep is not None:
            save_table(sweep.table, directory / "benchmark.csv")
        if make_plots:
            _write_supervised_plots(config, split, predictions, sweep, directory)

    return TrainingReport(bundle, split, predictions, sweep, directory)


def train_text_classifier(
    config: ProjectConfig,
    documents: pd.Series,
    labels: pd.Series,
    *,
    save: bool = True,
    make_plots: bool = True,
    max_features: int = 5000,
    min_df: int = 2,
) -> TrainingReport:
    """Fit a TF-IDF + linear classifier pipeline on raw text and persist it.

    The vectoriser is fitted inside the pipeline, on the training split only, so
    the test documents contribute nothing to the vocabulary or the IDF weights.
    """
    from sklearn.model_selection import train_test_split

    from dsjourney.text import build_text_pipeline

    x_train, x_test, y_train, y_test = train_test_split(
        documents,
        labels,
        test_size=config.split.test_size,
        random_state=config.split.random_state,
        stratify=labels if config.split.stratify else None,
    )

    estimator = build_model(config.task, config.model.estimator, **config.model.params)
    model = build_text_pipeline(estimator, max_features=max_features, min_df=min_df)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    probabilities = model.predict_proba(x_test) if hasattr(model, "predict_proba") else None
    average = "binary" if pd.Series(y_train).nunique() <= 2 else "macro"
    metrics = evaluate.classification_scores(
        y_test, predictions, y_proba=probabilities, average=average
    )

    bundle = ModelBundle(
        project=config.name,
        task=config.task,
        model=model,
        feature_names=["text"],
        metrics=metrics,
        scaler=None,
        extra={
            "estimator_key": config.model.estimator,
            "selected_by": "config",
            "target": config.target,
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "vocabulary_size": len(model.named_steps["tfidf"].vocabulary_),
        },
    )

    directory = None
    if save:
        directory = save_bundle(bundle)
        if make_plots:
            matrix = evaluate.confusion_frame(y_test, predictions)
            viz.save_figure(viz.confusion_matrix_plot(matrix), directory / "confusion_matrix.png")

    split = SplitResult(
        x_train.to_frame(name="text"), x_test.to_frame(name="text"), y_train, y_test, None
    )
    return TrainingReport(bundle, split, np.asarray(predictions), None, directory)


def train_clustering(
    config: ProjectConfig,
    frame: pd.DataFrame,
    *,
    k_range: range = range(2, 11),
    save: bool = True,
    make_plots: bool = True,
) -> TrainingReport:
    """Scan a range of cluster counts, fit the configured k and score the result.

    The elbow and silhouette curves over ``k_range`` are always computed so the
    chosen k is justified by evidence rather than asserted.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(frame), columns=frame.columns, index=frame.index)

    inertias: list[float] = []
    silhouettes: list[float] = []
    for k in k_range:
        probe = KMeans(n_clusters=k, n_init=10, random_state=config.split.random_state).fit(scaled)
        inertias.append(float(probe.inertia_))
        silhouettes.append(evaluate.clustering_scores(scaled, probe.labels_)["silhouette"])

    model = build_model(config.task, config.model.estimator, **config.model.params)
    labels = model.fit_predict(scaled)
    metrics = evaluate.clustering_scores(scaled, labels)

    bundle = ModelBundle(
        project=config.name,
        task=config.task,
        model=model,
        feature_names=[str(c) for c in frame.columns],
        metrics=metrics,
        scaler=scaler,
        scaled_columns=[str(c) for c in frame.columns],
        extra={
            "estimator_key": config.model.estimator,
            "n_clusters": int(getattr(model, "n_clusters", len(set(labels)))),
            "rows": len(frame),
            "k_scan": {str(k): round(s, 4) for k, s in zip(k_range, silhouettes, strict=True)},
        },
    )

    directory = None
    if save:
        directory = save_bundle(bundle)
        if make_plots:
            viz.save_figure(
                viz.elbow_plot(list(k_range), inertias, silhouettes),
                directory / "cluster_selection.png",
            )

    empty = pd.DataFrame()
    split = SplitResult(
        scaled, empty, pd.Series(labels, index=frame.index), pd.Series(dtype=float), scaler
    )
    return TrainingReport(bundle, split, np.asarray(labels), None, directory)


def _original_scale_scores(
    y_test: pd.Series,
    predictions: np.ndarray,
    inverse_transform: Callable[[np.ndarray], np.ndarray],
) -> dict[str, float]:
    """Score a regression on the untransformed target scale.

    Both sides are inverted, so this answers "how far off is the predicted price
    in currency?" rather than "how far off is the predicted logarithm?".
    """
    restored_truth = inverse_transform(np.asarray(y_test, dtype=float))
    restored_prediction = inverse_transform(np.asarray(predictions, dtype=float))
    scores = evaluate.regression_scores(restored_truth, restored_prediction)
    return {f"{name}_original": value for name, value in scores.items()}


def _score_supervised(
    config: ProjectConfig, model: Any, split: SplitResult, predictions: np.ndarray
) -> dict[str, float]:
    """Score a fitted model with the metric family that matches its task."""
    if config.task == "regression":
        return evaluate.regression_scores(split.y_test, predictions)

    probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(split.x_test)
        except Exception:
            probabilities = None
    average = "binary" if pd.Series(split.y_train).nunique() <= 2 else "macro"
    return evaluate.classification_scores(
        split.y_test, predictions, y_proba=probabilities, average=average
    )


def _write_supervised_plots(
    config: ProjectConfig,
    split: SplitResult,
    predictions: np.ndarray,
    sweep: BenchmarkResult | None,
    directory: Path,
) -> None:
    """Render the diagnostic figures appropriate to the task."""
    if config.task == "regression":
        viz.save_figure(viz.residual_plot(split.y_test, predictions), directory / "residuals.png")
    else:
        matrix = evaluate.confusion_frame(split.y_test, predictions)
        viz.save_figure(viz.confusion_matrix_plot(matrix), directory / "confusion_matrix.png")

    if sweep is not None:
        metric = "r2" if config.task == "regression" else "f1"
        viz.save_figure(
            viz.model_comparison_plot(sweep.table, metric), directory / "model_comparison.png"
        )


def _is_finite(value: float) -> bool:
    return isinstance(value, (int, float)) and np.isfinite(value)
