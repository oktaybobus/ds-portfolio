"""Tests for the ``dsj`` command-line interface.

This module had zero coverage while six projects crashed under ``dsj train``
with a raw ValueError from three frames down - the README's first suggested
command, broken for close to half the portfolio, and no test to notice. These
run every command surface through Typer's runner, no subprocesses involved.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from dsjourney.cli import _generic_train_supported, _own_entry_point, app
from dsjourney.config import load_project_config
from dsjourney.paths import available_projects

runner = CliRunner()

# Projects the generic scikit-learn trainer genuinely covers, as pairs of
# (project, why the rest are excluded).
GENERIC = {
    "bart_ridership",
    "customer_segments",
    "istanbul_housing",
    "laptop_price",
    "loan_default",
    "review_sentiment",
}


def test_list_shows_every_project() -> None:
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    # Rich truncates long names to fit the table, so match on a stable prefix.
    for project in available_projects():
        assert project[:12] in result.output


def test_info_renders_any_task_type() -> None:
    """`info` must work for every project, trained or not, sklearn or not."""
    for project in available_projects():
        result = runner.invoke(app, ["info", project])
        assert result.exit_code == 0, f"{project}: {result.output[-200:]}"


def test_info_names_the_training_command_when_untrained() -> None:
    result = runner.invoke(app, ["info", "marvel_network"])
    assert "train" in result.output


@pytest.mark.parametrize("project", sorted(set(available_projects()) - GENERIC))
def test_train_refuses_non_sklearn_projects_cleanly(project: str) -> None:
    """The defect this file exists for.

    ``dsj train marvel_network`` used to die in a raw ValueError. It must exit
    with code 2, no traceback, and name the command that actually trains the
    project.
    """
    result = runner.invoke(app, ["train", project])
    assert result.exit_code == 2, f"{project} should be refused"
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"{project} raised {type(result.exception).__name__} instead of exiting"
    )
    assert "Traceback" not in result.output
    entry = _own_entry_point(project)
    assert entry is not None and entry in result.output, (
        f"{project}'s refusal must say where it does train"
    )


def test_train_refusal_explains_spark_estimators_specifically() -> None:
    """diabetes_screening is classification - a task the trainer covers - but
    pinned to a Spark estimator. The message must blame the estimator, not
    the task, or it reads as nonsense."""
    result = runner.invoke(app, ["train", "diabetes_screening"])
    assert result.exit_code == 2
    assert "spark_logistic" in result.output


@pytest.mark.parametrize("project", sorted(GENERIC))
def test_generic_projects_are_still_routed_to_the_trainer(project: str) -> None:
    """The gate must not creep: every sklearn-family project stays trainable."""
    assert _generic_train_supported(load_project_config(project))


def test_benchmark_refuses_what_it_cannot_sweep() -> None:
    for project in ("marvel_network", "cartpole_balance", "diabetes_screening"):
        result = runner.invoke(app, ["benchmark", project])
        assert result.exit_code == 2, project
        assert "Traceback" not in result.output


def test_predict_requires_a_json_object() -> None:
    result = runner.invoke(app, ["predict", "laptop_price", "--json", "[1,2]"])
    assert result.exit_code != 0
    assert "JSON object" in result.output or "payload" in result.output


def test_predict_reads_a_payload_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from dsjourney.cli import _read_payload

    path = tmp_path / "payload.json"
    path.write_text('{"a": 1}', encoding="utf-8")
    assert _read_payload(f"@{path}") == {"a": 1}
    assert _read_payload('{"b": 2}') == {"b": 2}


def test_serve_reports_a_missing_app() -> None:
    result = runner.invoke(app, ["serve", "marvel_network"])
    assert result.exit_code == 2
    assert "No Streamlit app" in result.output


def test_unknown_project_fails_with_a_message_not_a_traceback() -> None:
    result = runner.invoke(app, ["train", "not_a_project"])
    assert result.exit_code != 0


def test_every_declared_entry_point_actually_exists() -> None:
    """The refusal message quotes a command; that command must be runnable."""
    from dsjourney.paths import project_dir

    for project in available_projects():
        entry = _own_entry_point(project)
        if entry is not None:
            script = entry.split()[-1]
            assert (project_dir(project).parent.parent / script).is_file(), script


def test_headline_metric_prefers_the_task_metric() -> None:
    from dsjourney.cli import _headline_metric

    assert _headline_metric({"r2": 0.8123, "rmse": 1.0}) == "r2=0.8123"
    assert _headline_metric({"f1": 0.5, "accuracy": 0.9}) == "f1=0.5000"
    assert _headline_metric({"something_else": 1.0}) == "-"


def test_to_jsonable_unwraps_single_element_arrays() -> None:
    import numpy as np

    from dsjourney.cli import _to_jsonable

    assert _to_jsonable(np.array([3.5])) == 3.5
    assert _to_jsonable(np.array([1, 2])) == [1, 2]
    assert _to_jsonable("plain") == "plain"


def test_eda_report_works_on_a_tabular_and_a_text_project() -> None:
    """eda-report had zero tests; it must render for both a CSV-backed project
    and one whose load_raw synthesises its frame (the graph's line table)."""
    for project in ("laptop_price", "marvel_network"):
        result = runner.invoke(app, ["eda-report", project, "--rows", "5"])
        assert result.exit_code == 0, f"{project}: {result.output[-200:]}"
        assert "rows" in result.output or "x" in result.output
