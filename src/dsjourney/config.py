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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dsjourney.paths import PROJECTS_DIR

TaskType = Literal[
    "regression",
    "classification",
    "clustering",
    "text-classification",
    "image-classification",
    "forecasting",
    "recommendation",
    "retrieval",
    "detection",
    "graph",
    "control",
    "geospatial",
]

# Tasks that do not fit a features-plus-target frame: clustering has no target,
# a recommender is trained on an interaction log rather than rows of features,
# a graph is an edge list, and a control agent generates its own data by acting.
# All are exempt from the target requirement.
UNSUPERVISED_TASKS = frozenset(
    {"clustering", "recommendation", "retrieval", "detection", "graph", "control", "geospatial"}
)


class DatasetConfig(BaseModel):
    """Where a project's input data lives and how to read it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Key into assets.yaml used by scripts/fetch_assets.py")
    file: str = Field(description="File name as stored under data/raw/<project>/")
    format: Literal["csv", "excel", "sqlite", "kagglehub", "text"] = "csv"
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
    # A control project has no file to read: its data is whatever the
    # environment produces while the agent acts in it. One of the two must be
    # declared, and the validator below enforces that.
    dataset: DatasetConfig | None = None
    environment: str | None = Field(
        default=None, description="Gymnasium environment id, for control tasks"
    )
    split: SplitConfig = Field(default_factory=SplitConfig)
    model: ModelConfig
    metrics: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_is_a_slug(cls, value: str) -> str:
        if not value.replace("_", "").isalnum() or value != value.lower():
            raise ValueError(f"project name must be a lowercase snake_case slug, got {value!r}")
        return value

    @model_validator(mode="after")
    def _has_a_data_source(self) -> ProjectConfig:
        """Every project reads a dataset or drives an environment."""
        if self.dataset is None and self.environment is None:
            raise ValueError(f"project {self.name!r} declares neither a dataset nor an environment")
        return self

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
