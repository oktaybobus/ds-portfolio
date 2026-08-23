"""Plotting helpers.

Every function builds and returns a Matplotlib ``Figure`` and never calls
``plt.show()``. That keeps them usable from a headless training run (save to
disk), from Streamlit (``st.pyplot``), from a notebook, and from a test that
only asserts the figure was produced.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DEFAULT_PALETTE = "viridis"


def count_plot(
    frame: pd.DataFrame, column: str, *, top: int | None = None, figsize: tuple[int, int] = (9, 5)
) -> plt.Figure:
    """Horizontal category counts with the value written on each bar."""
    order = frame[column].value_counts().index
    if top is not None:
        order = order[:top]
    figure, axes = plt.subplots(figsize=figsize)
    sns.countplot(
        data=frame,
        y=column,
        order=order,
        hue=column,
        legend=False,
        palette=DEFAULT_PALETTE,
        ax=axes,
    )
    for container in axes.containers:
        # Axes.containers is typed as Container; only BarContainer reaches here.
        axes.bar_label(container, padding=2)  # type: ignore[arg-type]
    axes.set_title(f"Distribution of {column}")
    figure.tight_layout()
    return figure


def distribution_plot(
    frame: pd.DataFrame, column: str, *, figsize: tuple[int, int] = (8, 6)
) -> plt.Figure:
    """Box plot stacked over a histogram, with mean and median marked.

    Shows the spread, the shape and the outliers of a numeric column in one
    glance - the combination used to inspect credit scores in the loan project.
    """
    figure, (box_axes, hist_axes) = plt.subplots(
        2, sharex=True, gridspec_kw={"height_ratios": (0.2, 0.8)}, figsize=figsize
    )
    sns.boxplot(data=frame, x=column, ax=box_axes, color="skyblue")
    box_axes.set(xlabel="")
    sns.histplot(data=frame, x=column, ax=hist_axes, kde=True, color="navy")

    mean_value = float(frame[column].mean())
    median_value = float(frame[column].median())
    hist_axes.axvline(
        mean_value, color="red", linestyle="--", linewidth=2, label=f"mean {mean_value:.2f}"
    )
    hist_axes.axvline(median_value, color="green", linewidth=2, label=f"median {median_value:.2f}")
    hist_axes.legend()

    figure.suptitle(f"{column}: distribution and summary statistics")
    figure.tight_layout()
    return figure


def correlation_heatmap(
    frame: pd.DataFrame, *, figsize: tuple[int, int] = (12, 9), annot: bool = True
) -> plt.Figure:
    """Absolute correlation matrix of every numeric column."""
    correlations = frame.corr(numeric_only=True).abs()
    figure, axes = plt.subplots(figsize=figsize)
    sns.heatmap(correlations, annot=annot, fmt=".2f", cmap="rocket_r", vmin=0, vmax=1, ax=axes)
    axes.set_title("Absolute correlation matrix")
    figure.tight_layout()
    return figure


def model_comparison_plot(
    table: pd.DataFrame, metric: str, *, figsize: tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Bar chart of a :func:`dsjourney.benchmark.compare_models` result."""
    data = table.dropna(subset=[metric]).sort_values(metric, ascending=False)
    figure, axes = plt.subplots(figsize=figsize)
    sns.barplot(
        data=data, y="model", x=metric, hue="model", legend=False, palette=DEFAULT_PALETTE, ax=axes
    )
    for container in axes.containers:
        axes.bar_label(container, fmt="%.3f", padding=2)  # type: ignore[arg-type]
    axes.set_title(f"Model comparison by {metric}")
    figure.tight_layout()
    return figure


def confusion_matrix_plot(matrix: pd.DataFrame, *, figsize: tuple[int, int] = (6, 5)) -> plt.Figure:
    """Annotated heatmap of a :func:`dsjourney.evaluate.confusion_frame` result."""
    figure, axes = plt.subplots(figsize=figsize)
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes)
    axes.set_title("Confusion matrix")
    figure.tight_layout()
    return figure


def residual_plot(
    y_true: pd.Series | np.ndarray, y_pred: np.ndarray, *, figsize: tuple[int, int] = (10, 4)
) -> plt.Figure:
    """Predicted-vs-actual scatter next to a residual distribution."""
    truth = np.asarray(y_true, dtype=float).ravel()
    prediction = np.asarray(y_pred, dtype=float).ravel()
    residuals = truth - prediction

    figure, (scatter_axes, hist_axes) = plt.subplots(1, 2, figsize=figsize)
    scatter_axes.scatter(truth, prediction, alpha=0.4, edgecolor="none")
    limits = [min(truth.min(), prediction.min()), max(truth.max(), prediction.max())]
    scatter_axes.plot(limits, limits, "r--", linewidth=1)
    scatter_axes.set(xlabel="actual", ylabel="predicted", title="Predicted vs actual")

    sns.histplot(residuals, kde=True, ax=hist_axes, color="slateblue")
    hist_axes.axvline(0, color="red", linestyle="--")
    hist_axes.set(xlabel="residual", title="Residual distribution")

    figure.tight_layout()
    return figure


def elbow_plot(
    k_values: list[int],
    inertias: list[float],
    silhouettes: list[float] | None = None,
    *,
    figsize: tuple[int, int] = (10, 4),
) -> plt.Figure:
    """Inertia (elbow) and, optionally, silhouette against the number of clusters."""
    panels = 2 if silhouettes else 1
    figure, axes = plt.subplots(1, panels, figsize=figsize, squeeze=False)

    axes[0][0].plot(k_values, inertias, marker="o")
    axes[0][0].set(xlabel="k", ylabel="inertia", title="Elbow method")

    if silhouettes:
        axes[0][1].plot(k_values, silhouettes, marker="o", color="darkorange")
        axes[0][1].set(xlabel="k", ylabel="silhouette", title="Silhouette by k")

    figure.tight_layout()
    return figure


def forecast_plot(
    train: pd.Series,
    test: pd.Series,
    forecast: pd.Series | np.ndarray,
    *,
    title: str = "Forecast",
    context: int = 200,
    figsize: tuple[int, int] = (11, 5),
) -> plt.Figure:
    """Plot the tail of the training period, the holdout, and the forecast.

    Showing the holdout and the forecast on the same axes is the point: a
    forecast drawn past the end of a fully-fitted series looks convincing no
    matter how wrong it is, because there is nothing beside it to disagree.
    """
    figure, axes = plt.subplots(figsize=figsize)

    recent = train.iloc[-context:] if context and len(train) > context else train
    axes.plot(recent.index, recent.to_numpy(), label="train", color="#4c72b0", linewidth=1.2)
    axes.plot(test.index, test.to_numpy(), label="actual (holdout)", color="#111111", linewidth=1.6)

    values = forecast.to_numpy() if isinstance(forecast, pd.Series) else np.asarray(forecast)
    axes.plot(
        test.index[: len(values)],
        values[: len(test)],
        label="forecast",
        color="#c44e52",
        linestyle="--",
        linewidth=1.8,
    )

    axes.axvline(test.index[0], color="grey", linestyle=":", linewidth=1)
    axes.set_title(title)
    axes.legend(loc="best")
    axes.grid(alpha=0.25)
    figure.tight_layout()
    return figure


def save_figure(figure: plt.Figure, path: Path, *, dpi: int = 120) -> Path:
    """Write a figure to disk, creating parent directories, and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path
