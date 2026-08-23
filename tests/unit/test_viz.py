"""Unit tests for the plotting helpers.

The figures are not inspected pixel by pixel; what matters is that every helper
returns a Figure (never calls ``show``), survives a headless backend, and can be
written to disk - which is what breaks in CI when a plot smuggles in ``plt.show``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

from dsjourney import viz


def test_backend_is_headless() -> None:
    assert matplotlib.get_backend().lower() == "agg"


def test_count_plot_returns_a_figure(messy_frame: pd.DataFrame) -> None:
    figure = viz.count_plot(messy_frame, "brand")
    assert figure.axes


def test_count_plot_can_limit_categories(messy_frame: pd.DataFrame) -> None:
    figure = viz.count_plot(messy_frame, "brand", top=1)
    assert len(figure.axes[0].get_yticklabels()) == 1


def test_distribution_plot_marks_mean_and_median(messy_frame: pd.DataFrame) -> None:
    figure = viz.distribution_plot(messy_frame, "price")
    labels = [line.get_label() for line in figure.axes[1].lines]
    assert any("mean" in str(label) for label in labels)
    assert any("median" in str(label) for label in labels)


def test_correlation_heatmap_renders(regression_frame: pd.DataFrame) -> None:
    assert viz.correlation_heatmap(regression_frame).axes


def test_model_comparison_plot_skips_failed_models() -> None:
    table = pd.DataFrame({"model": ["a", "b", "c"], "r2": [0.9, None, 0.5]})
    figure = viz.model_comparison_plot(table, "r2")
    assert len(figure.axes[0].get_yticklabels()) == 2


def test_confusion_matrix_plot_renders() -> None:
    matrix = pd.DataFrame(
        [[5, 1], [2, 7]], index=["true_0", "true_1"], columns=["pred_0", "pred_1"]
    )
    assert viz.confusion_matrix_plot(matrix).axes


def test_residual_plot_has_two_panels(regression_frame: pd.DataFrame) -> None:
    figure = viz.residual_plot(regression_frame["target"], regression_frame["feature_a"].to_numpy())
    assert len(figure.axes) == 2


def test_elbow_plot_adds_a_silhouette_panel_when_given_one() -> None:
    assert len(viz.elbow_plot([2, 3, 4], [10.0, 6.0, 5.0]).axes) == 1
    assert len(viz.elbow_plot([2, 3, 4], [10.0, 6.0, 5.0], [0.4, 0.5, 0.3]).axes) == 2


def test_save_figure_creates_parent_directories(tmp_path: Path, messy_frame: pd.DataFrame) -> None:
    target = tmp_path / "nested" / "plot.png"
    result = viz.save_figure(viz.count_plot(messy_frame, "brand"), target)
    assert result.is_file()
    assert result.stat().st_size > 0
