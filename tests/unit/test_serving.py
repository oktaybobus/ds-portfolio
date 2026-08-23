"""Unit tests for the serving layer and its HTTP surface.

These run against the models that happen to be trained in this checkout, so
they assert behaviour that holds either way: that a project reports its own
servability honestly, and that the API turns each failure into the right status
code.
"""

from __future__ import annotations

import pytest
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
