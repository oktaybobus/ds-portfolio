"""Unit tests for project configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from dsjourney.config import ProjectConfig, SplitConfig, load_project_config
from dsjourney.paths import available_projects

MINIMAL = {
    "name": "demo",
    "title": "Demo",
    "task": "regression",
    "target": "y",
    "dataset": {"id": "demo", "file": "demo.csv"},
    "model": {"estimator": "linear"},
}


def test_minimal_config_validates() -> None:
    config = ProjectConfig.model_validate(MINIMAL)
    assert config.name == "demo"
    assert config.split.test_size == 0.2
    assert config.is_supervised


def test_clustering_is_not_supervised() -> None:
    config = ProjectConfig.model_validate({**MINIMAL, "task": "clustering", "target": None})
    assert not config.is_supervised


def test_name_must_be_a_snake_case_slug() -> None:
    with pytest.raises(ValidationError, match="snake_case slug"):
        ProjectConfig.model_validate({**MINIMAL, "name": "Demo Project"})


def test_unknown_keys_are_rejected() -> None:
    """A typo in a config should fail loudly, not be silently ignored."""
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate({**MINIMAL, "targt": "y"})


def test_test_size_must_be_a_proportion() -> None:
    with pytest.raises(ValidationError):
        SplitConfig(test_size=1.5)


def test_config_is_immutable() -> None:
    config = ProjectConfig.model_validate(MINIMAL)
    with pytest.raises(ValidationError):
        config.name = "other"  # type: ignore[misc]


def test_load_project_config_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no project config"):
        load_project_config(tmp_path / "nowhere.yaml")


def test_load_project_config_accepts_a_direct_path(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(MINIMAL), encoding="utf-8")
    assert load_project_config(path).name == "demo"


def test_load_project_config_accepts_a_directory(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(MINIMAL), encoding="utf-8")
    assert load_project_config(tmp_path).title == "Demo"


@pytest.mark.parametrize("project", available_projects())
def test_every_shipped_project_config_is_valid(project: str) -> None:
    """Guards against a config drifting out of sync with the schema."""
    config = load_project_config(project)
    assert config.name == project
    assert config.title
    assert config.source_notebook, f"{project} should record the notebook it came from"
    if config.is_supervised:
        assert config.target, f"{project} is supervised and needs a target"
