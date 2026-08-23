"""dsjourney - a shared toolkit distilled from a 15-week AI/ML curriculum.

The package collects the data-preparation, exploration, modelling and evaluation
steps that were repeated across ~100 course notebooks into a single, tested API.
Each portfolio project under ``projects/`` is a thin, declarative layer on top of
it: load a config, build features, call the shared trainer, save the artifact.

All public helpers are pure with respect to their inputs - DataFrames are copied
rather than mutated, so a pipeline step never surprises the caller.
"""

from dsjourney.config import ProjectConfig, load_project_config
from dsjourney.paths import (
    ARTIFACTS_DIR,
    DATA_DIR,
    PROJECTS_DIR,
    REPO_ROOT,
    project_artifacts_dir,
    project_dir,
)

__version__ = "0.1.0"

__all__ = [
    "ARTIFACTS_DIR",
    "DATA_DIR",
    "PROJECTS_DIR",
    "REPO_ROOT",
    "ProjectConfig",
    "__version__",
    "load_project_config",
    "project_artifacts_dir",
    "project_dir",
]
