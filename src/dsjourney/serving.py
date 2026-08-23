"""Serving trained models over an API.

The MLOps notebook's FastAPI app hard-coded one model, three feature names and
one response shape. Adding a second model meant copying the file. This module
inverts that: a project becomes servable by having a saved bundle and a
``prepare_input``, and the service discovers it at start-up.

Nothing here imports FastAPI - the web layer lives in ``service/app.py`` and
this stays a plain, testable Python API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from dsjourney.artifacts import ModelBundle, bundle_exists, load_bundle
from dsjourney.config import ProjectConfig, load_project_config
from dsjourney.paths import available_projects
from dsjourney.pipeline import load_pipeline


class ProjectNotServableError(LookupError):
    """Raised when a project cannot answer a record-scoring request."""


@dataclass(frozen=True)
class ServableProject:
    """What the service can tell a caller about one project."""

    name: str
    title: str
    task: str
    summary: str
    trained: bool
    servable: bool
    reason: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    feature_count: int = 0
    example_input: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view for an API response."""
        return {
            "name": self.name,
            "title": self.title,
            "task": self.task,
            "summary": self.summary.strip(),
            "trained": self.trained,
            "servable": self.servable,
            "reason": self.reason,
            "metrics": self.metrics,
            "feature_count": self.feature_count,
            "example_input": self.example_input,
        }


def inspect_project(name: str) -> ServableProject:
    """Describe one project and say whether it can score a record.

    Never raises for an unservable project - it reports *why* instead, because
    "this model forecasts a series, it does not score records" is information a
    caller wants, not an error.
    """
    config = load_project_config(name)
    trained = bundle_exists(name)

    metrics: dict[str, float] = {}
    feature_count = 0
    if trained:
        bundle = load_bundle(name)
        metrics = bundle.metrics
        feature_count = len(bundle.feature_names)

    servable, reason = _servability(name, config, trained=trained)
    return ServableProject(
        name=name,
        title=config.title,
        task=config.task,
        summary=config.summary,
        trained=trained,
        servable=servable,
        reason=reason,
        metrics=metrics,
        feature_count=feature_count,
        example_input=example_input(name),
    )


def servable_projects() -> list[ServableProject]:
    """Describe every project in the repository, servable or not."""
    return [inspect_project(name) for name in available_projects()]


def example_input(name: str) -> dict[str, Any]:
    """Return a project's documented example record, or an empty dict."""
    try:
        module = load_pipeline(name)
    except ModuleNotFoundError:
        return {}
    example = getattr(module, "EXAMPLE_INPUT", None)
    return dict(example) if isinstance(example, dict) else {}


def predict_record(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Score one record against a project's saved model.

    Raises:
        ProjectNotServableError: when the project has no model, or its task does not
            take a single record.
    """
    described = inspect_project(name)
    if not described.servable:
        raise ProjectNotServableError(f"{name}: {described.reason}")

    module = load_pipeline(name)
    bundle = _cached_bundle(name)

    row = bundle.prepare(module.prepare_input(payload))
    raw = bundle.model.predict(row)

    postprocess = getattr(module, "postprocess", None)
    value = postprocess(raw) if callable(postprocess) else raw

    response: dict[str, Any] = {
        "project": name,
        "task": described.task,
        "prediction": _scalar(value),
    }

    probabilities = _class_probabilities(bundle, row)
    if probabilities is not None:
        response["probabilities"] = probabilities
    return response


def clear_cache() -> None:
    """Drop cached bundles, so a retrained model is picked up without a restart."""
    _cached_bundle.cache_clear()


@lru_cache(maxsize=16)
def _cached_bundle(name: str) -> ModelBundle:
    """Load a bundle once per process.

    Deserialising a forest takes long enough that doing it per request would
    dominate the response time.
    """
    return load_bundle(name)


# These task families produce a series, a ranked list or a label for a file -
# never a score for one row of features - so the record endpoint does not apply
# to them however well trained they are.
NON_RECORD_TASKS = frozenset(
    {"forecasting", "recommendation", "image-classification", "retrieval", "detection"}
)


def _servability(name: str, config: ProjectConfig, *, trained: bool) -> tuple[bool, str]:
    """Decide whether a project can score a single record, and say why not.

    The task check comes first on purpose: telling a caller that a forecasting
    project "is not trained yet" would send them to run a command that still
    would not make the record endpoint work.
    """
    if config.task in NON_RECORD_TASKS:
        return False, f"{config.task} projects do not score individual records"

    if not trained:
        return False, f"not trained yet - run: dsj train {name}"

    try:
        module = load_pipeline(name)
    except ModuleNotFoundError as error:
        return False, str(error)

    if not hasattr(module, "prepare_input"):
        return False, "pipeline does not implement prepare_input"

    return True, ""


def _scalar(value: Any) -> Any:
    """Reduce a single-element array to a plain Python number."""
    if hasattr(value, "tolist"):
        listed = value.tolist()
        return listed[0] if isinstance(listed, list) and len(listed) == 1 else listed
    return value


def _class_probabilities(bundle: ModelBundle, row: Any) -> dict[str, float] | None:
    """Return per-class probabilities when the estimator can produce them."""
    model = bundle.model
    if not hasattr(model, "predict_proba"):
        return None
    try:
        probabilities = model.predict_proba(row)[0]
        classes = getattr(model, "classes_", range(len(probabilities)))
    except Exception:
        return None
    return {str(label): float(value) for label, value in zip(classes, probabilities, strict=False)}
