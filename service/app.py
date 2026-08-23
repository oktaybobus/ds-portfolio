"""REST API over the portfolio's trained models.

The MLOps notebook's service hard-coded one model, three field names and one
response shape; adding a second model meant copying the file. Here the routes
are generic and the projects are discovered at start-up, so a project becomes
servable simply by having a saved bundle and a ``prepare_input``.

Run it with:

    uv run uvicorn service.app:app --reload

Interactive documentation is then at http://127.0.0.1:8000/docs.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dsjourney import __version__, serving

app = FastAPI(
    title="ds-portfolio model service",
    description=__doc__,
    version=__version__,
    contact={"name": "ds-portfolio", "url": "https://github.com/oktaybobus/ds-portfolio"},
)


class PredictionRequest(BaseModel):
    """One record to score.

    The fields differ per project - ``GET /projects/{name}`` returns a working
    example for whichever one you are calling.
    """

    record: dict[str, Any] = Field(
        description="Field/value pairs matching the project's example_input",
        examples=[{"company": "Dell", "type_name": "Gaming", "ram_gb": 16}],
    )


class PredictionResponse(BaseModel):
    """The model's answer, on the scale a reader cares about."""

    project: str
    task: str
    prediction: Any = Field(description="Regression: the value. Classification: the class label.")
    probabilities: dict[str, float] | None = Field(
        default=None, description="Per-class probabilities, when the estimator provides them"
    )


@app.get("/health", tags=["service"])
def health() -> dict[str, Any]:
    """Liveness probe: reports how many projects are ready to score."""
    projects = serving.servable_projects()
    return {
        "status": "ok",
        "version": __version__,
        "projects": len(projects),
        "servable": sum(1 for project in projects if project.servable),
    }


@app.get("/projects", tags=["projects"])
def list_projects() -> list[dict[str, Any]]:
    """Every project, whether it can score records, and why not when it cannot."""
    return [project.as_dict() for project in serving.servable_projects()]


@app.get("/projects/{name}", tags=["projects"])
def describe_project(name: str) -> dict[str, Any]:
    """One project's configuration, saved metrics and a working example record."""
    try:
        return serving.inspect_project(name).as_dict()
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.post("/projects/{name}/predict", response_model=PredictionResponse, tags=["predict"])
def predict(name: str, request: PredictionRequest) -> PredictionResponse:
    """Score one record against a project's saved model.

    Returns 404 for an unknown project, 409 when the project exists but cannot
    score records (a forecaster or a recommender), and 422 when the record
    itself cannot be turned into features.
    """
    try:
        result = serving.predict_record(name, request.record)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except serving.ProjectNotServableError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except (KeyError, ValueError, TypeError) as error:
        raise HTTPException(
            status_code=422, detail=f"could not build features from that record: {error}"
        ) from error
    return PredictionResponse(**result)


@app.post("/admin/reload", tags=["service"])
def reload_models() -> dict[str, str]:
    """Drop cached bundles so a retrained model is served without a restart."""
    serving.clear_cache()
    return {"status": "reloaded"}
