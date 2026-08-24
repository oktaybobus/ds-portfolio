"""Unit tests for the serving layer and its HTTP surface.

These run against the models that happen to be trained in this checkout, so
they assert behaviour that holds either way: that a project reports its own
servability honestly, and that the API turns each failure into the right status
code.
"""

from __future__ import annotations

import pytest

# FastAPI is an optional extra. Importing it at module scope makes *collection*
# fail on any install that omits `--extra api` - which takes the whole suite
# down, including tests that have nothing to do with serving. Skipping is the
# same rule conftest applies to TensorFlow and Spark, applied at the one place
# a missing extra is reached through an import rather than a marker.
pytest.importorskip("fastapi", reason="FastAPI not installed (uv sync --extra api)")

from fastapi.testclient import TestClient

from dsjourney import serving
from dsjourney.paths import available_projects
from service.app import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.mark.parametrize("project", available_projects())
def test_every_project_can_be_inspected(project: str) -> None:
    described = serving.inspect_project(project)
    assert described.name == project
    assert described.title


@pytest.mark.parametrize("project", available_projects())
def test_unservable_projects_explain_themselves(project: str) -> None:
    """A bare False is useless to a caller; the reason is the useful part."""
    described = serving.inspect_project(project)
    if not described.servable:
        assert described.reason


def test_non_record_tasks_are_reported_by_task_not_by_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forecaster is unservable whatever its training state.

    Reporting "not trained yet" would send the caller to run a command that
    still would not make the record endpoint work.
    """
    described = serving.inspect_project("series_forecast")
    assert not described.servable
    assert "forecasting" in described.reason
    assert "not trained" not in described.reason


def test_predict_refuses_a_non_record_project() -> None:
    with pytest.raises(serving.ProjectNotServableError, match="do not score individual records"):
        serving.predict_record("movie_recommender", {})


def test_record_projects_document_an_example() -> None:
    """An API is unusable if the caller has to guess the field names."""
    for described in serving.servable_projects():
        if described.servable:
            assert described.example_input, f"{described.name} has no EXAMPLE_INPUT"


def test_servable_projects_round_trip_their_own_example() -> None:
    for described in serving.servable_projects():
        if not described.servable:
            continue
        result = serving.predict_record(described.name, described.example_input)
        assert result["project"] == described.name
        assert result["prediction"] is not None


def test_health_counts_projects(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["projects"] == len(available_projects())
    assert body["servable"] <= body["projects"]


def test_list_projects_returns_every_project(client: TestClient) -> None:
    body = client.get("/projects").json()
    assert {entry["name"] for entry in body} == set(available_projects())


def test_describe_unknown_project_is_404(client: TestClient) -> None:
    assert client.get("/projects/not_a_project").status_code == 404


def test_predict_on_a_non_record_project_is_409(client: TestClient) -> None:
    response = client.post("/projects/series_forecast/predict", json={"record": {}})
    assert response.status_code == 409
    assert "forecasting" in response.json()["detail"]


def test_predict_with_a_broken_record_is_422(client: TestClient) -> None:
    """A bad record is the caller's mistake, not a server fault."""
    servable = [p for p in serving.servable_projects() if p.servable and p.task != "clustering"]
    if not servable:
        pytest.skip("no trained record-scoring project in this checkout")

    response = client.post(
        f"/projects/{servable[0].name}/predict", json={"record": {"ram_gb": object.__name__}}
    )
    assert response.status_code in {200, 422}


def test_predict_returns_probabilities_for_classifiers(client: TestClient) -> None:
    described = serving.inspect_project("loan_default")
    if not described.servable:
        pytest.skip("loan_default is not trained in this checkout")

    body = client.post(
        "/projects/loan_default/predict", json={"record": described.example_input}
    ).json()
    assert body["probabilities"] is not None
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)


def test_reload_clears_the_bundle_cache(client: TestClient) -> None:
    assert client.post("/admin/reload").json() == {"status": "reloaded"}


def test_no_unservable_reason_recommends_a_command_that_refuses() -> None:
    """The loop this closes: serving said "run dsj train marvel_network" while
    the CLI (correctly) exits 2 on exactly that command. Any project whose
    reason still points at `dsj train` must be one the generic trainer accepts."""
    from dsjourney.cli import _generic_train_supported
    from dsjourney.config import load_project_config

    for project in serving.servable_projects():
        if not project.servable and "dsj train" in project.reason:
            assert _generic_train_supported(load_project_config(project.name)), (
                f"{project.name}: reason sends the caller to a command that refuses it"
            )


def test_every_task_family_has_a_servability_verdict() -> None:
    """A new task type must be classified, not fall through to a wrong hint."""
    from typing import get_args

    from dsjourney.config import TaskType
    from dsjourney.serving import NON_RECORD_TASKS

    record_tasks = {"regression", "classification", "text-classification", "clustering"}
    for task in get_args(TaskType):
        assert task in NON_RECORD_TASKS or task in record_tasks, (
            f"task {task!r} is classified by neither serving rule"
        )
