"""Trained-model persistence.

A saved model on its own is not reproducible: the scaler that shaped its inputs,
the exact column order it expects and the scores it earned all have to travel
with it. :class:`ModelBundle` keeps those together in one directory so a
Streamlit app or a prediction script can restore the full state with one call.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn

from dsjourney import __version__
from dsjourney.paths import project_artifacts_dir

MODEL_FILE = "model.joblib"
SCALER_FILE = "scaler.joblib"
METADATA_FILE = "metadata.json"
METRICS_FILE = "metrics.json"


@dataclass
class ModelBundle:
    """A fitted model together with everything needed to reuse it."""

    project: str
    task: str
    model: Any
    feature_names: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    scaler: Any | None = None
    scaled_columns: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Reshape an inference frame exactly the way training reshaped it.

        Aligning the columns is not enough on its own: if the model was trained
        on standardised features, an unscaled row is fed values several standard
        deviations from anything the model saw, and the prediction is silently
        wrong rather than an error. This applies the saved scaler to the same
        columns it was fitted on.
        """
        from dsjourney.preprocess import align_to_training_columns

        aligned = align_to_training_columns(frame, self.feature_names)
        if self.scaler is None or not self.scaled_columns:
            return aligned

        columns = [c for c in self.scaled_columns if c in aligned.columns]
        if len(columns) != len(self.scaled_columns):
            missing = sorted(set(self.scaled_columns) - set(columns))
            raise ValueError(f"inference frame is missing scaled column(s): {missing}")

        scaled = self.scaler.transform(aligned[self.scaled_columns])
        return aligned.assign(**dict(zip(self.scaled_columns, scaled.T, strict=True)))

    def metadata(self) -> dict[str, Any]:
        """Return the provenance record written next to the model."""
        return {
            "project": self.project,
            "task": self.task,
            "model_class": type(self.model).__name__,
            "feature_names": self.feature_names,
            "feature_count": len(self.feature_names),
            "has_scaler": self.scaler is not None,
            "scaled_columns": self.scaled_columns,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "dsjourney_version": __version__,
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
            **self.extra,
        }


def save_bundle(bundle: ModelBundle, *, directory: Path | None = None) -> Path:
    """Persist a bundle to ``artifacts/<project>/`` and return that directory."""
    target = directory or project_artifacts_dir(bundle.project, create=True)
    target.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle.model, target / MODEL_FILE)
    if bundle.scaler is not None:
        joblib.dump(bundle.scaler, target / SCALER_FILE)

    _write_json(target / METADATA_FILE, bundle.metadata())
    _write_json(target / METRICS_FILE, bundle.metrics)
    return target


def load_bundle(project: str, *, directory: Path | None = None) -> ModelBundle:
    """Restore a bundle previously written by :func:`save_bundle`.

    Raises:
        FileNotFoundError: when the project has not been trained yet, naming the
            command that would produce the artifact.
    """
    target = directory or project_artifacts_dir(project)
    model_path = target / MODEL_FILE
    if not model_path.is_file():
        raise FileNotFoundError(
            f"no trained model at {model_path}. Run: uv run dsj train {project}"
        )

    metadata = _read_json(target / METADATA_FILE)
    scaler_path = target / SCALER_FILE
    known = {
        "project",
        "task",
        "model_class",
        "feature_names",
        "feature_count",
        "has_scaler",
        "scaled_columns",
    }
    return ModelBundle(
        project=str(metadata.get("project", project)),
        task=str(metadata.get("task", "unknown")),
        model=joblib.load(model_path),
        feature_names=[str(name) for name in metadata.get("feature_names", [])],
        metrics=_read_json(target / METRICS_FILE),
        scaler=joblib.load(scaler_path) if scaler_path.is_file() else None,
        scaled_columns=[str(name) for name in metadata.get("scaled_columns", [])],
        extra={k: v for k, v in metadata.items() if k not in known},
    )


def bundle_exists(project: str) -> bool:
    """True when the project already has a saved model."""
    return (project_artifacts_dir(project) / MODEL_FILE).is_file()


def save_table(table: Any, path: Path) -> Path:
    """Write a DataFrame to CSV, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    content = json.loads(path.read_text(encoding="utf-8"))
    return content if isinstance(content, dict) else {}
