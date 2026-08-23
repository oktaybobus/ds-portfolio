"""Declarative project configuration.

Each portfolio project owns a ``config.yaml`` that states what the project is,
where its data comes from and how it should be trained and scored. Validating
that file with Pydantic means a typo fails at load time with a precise message
instead of surfacing as a confusing error halfway through a training run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from dsjourney.paths import PROJECTS_DIR

TaskType = Literal[
    "regression",
    "classification",
    "clustering",
    "text-classification",
    "image-classification",
    "forecasting",
    "recommendation",
]

# Tasks that do not fit a features-plus-target frame: clustering has no target,
# and a recommender is trained on an interaction log rather than rows of
# features. Both are exempt from the target requirement.
UNSUPERVISED_TASKS = frozenset({"clustering", "recommendation"})


class DatasetConfig(BaseModel):
    """Where a project's input data lives and how to read it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Key into assets.yaml used by scripts/fetch_assets.py")
    file: str = Field(description="File name as stored under data/raw/<project>/")
    format: Literal["csv", "excel", "sqlite", "kagglehub"] = "csv"
    table: str | None = Field(default=None, description="Table name for sqlite sources")
    kaggle_handle: str | None = Field(default=None, description="Handle for kagglehub sources")
    read_options: dict[str, Any] = Field(default_factory=dict)


class SplitConfig(BaseModel):
    """Train/test split parameters, shared by every supervised project."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    random_state: int = 42
    stratify: bool = False


class ModelConfig(BaseModel):
    """Which estimator to fit and with what hyper-parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    estimator: str = Field(description="Key into dsjourney.benchmark model registries")
    params: dict[str, Any] = Field(default_factory=dict)
    scale_features: list[str] = Field(
        default_factory=list,
        description="Numeric columns to standardise; empty means no scaling",
    )


class ProjectConfig(BaseModel):
    """The full, validated description of one portfolio project."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    title: str
    task: TaskType
    summary: str = ""
    source_notebook: str = Field(
        default="",
        description="Relative path of the course notebook this project was distilled from",
    )
    target: str | None = None
    dataset: DatasetConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    model: ModelConfig
    metrics: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_is_a_slug(cls, value: str) -> str:
        if not value.replace("_", "").isalnum() or value != value.lower():
            raise ValueError(f"project name must be a lowercase snake_case slug, got {value!r}")
        return value

    @property
    def is_supervised(self) -> bool:
        """True when the task needs a target column.

        Forecasting counts as supervised: it has a target series, even though it
        is split chronologically rather than at random.
        """
        return self.task not in UNSUPERVISED_TASKS


def load_project_config(name_or_path: str | Path) -> ProjectConfig:
    """Load and validate a project config.

    Accepts a project name (``"laptop_price"``), a project directory, or a direct
    path to a ``config.yaml``.
    """
    path = Path(name_or_path)
    if not path.suffix:
        path = PROJECTS_DIR / path / "config.yaml"
    elif path.is_dir():
        path = path / "config.yaml"

    if not path.is_file():
        raise FileNotFoundError(f"no project config at {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(raw).__name__}")
    return ProjectConfig.model_validate(raw)
