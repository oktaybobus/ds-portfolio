"""The contract every portfolio project implements.

Keeping this as a Protocol rather than a base class means a project module is
just a module - no inheritance, no registration decorator - while the CLI can
still type-check what it is about to call.
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from dsjourney.config import ProjectConfig


@runtime_checkable
class ProjectPipeline(Protocol):
    """What ``projects/<name>/pipeline.py`` must expose."""

    CONFIG: ProjectConfig

    def load_raw(self) -> pd.DataFrame:
        """Read the project's dataset from ``data/raw/<project>/``."""
        ...

    def build_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Turn the raw frame into model-ready features including the target."""
        ...

    def prepare_input(self, payload: dict[str, Any]) -> pd.DataFrame:
        """Turn one user-supplied record into a single model-ready row."""
        ...


def load_pipeline(project: str) -> Any:
    """Import a project's pipeline module by project name.

    Raises:
        ModuleNotFoundError: with the expected path, when the project has no
            pipeline module yet.
    """
    module_name = f"projects.{project}.pipeline"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name and not error.name.startswith("projects"):
            raise
        raise ModuleNotFoundError(
            f"project '{project}' has no pipeline module (expected projects/{project}/pipeline.py)"
        ) from error
