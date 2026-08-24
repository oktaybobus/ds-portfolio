"""Contract tests every project pipeline must satisfy.

These run without any dataset on disk: they check the module surface, not the
data. A new project that forgets ``build_features`` or names its config
inconsistently fails here rather than halfway through a training run.
"""

from __future__ import annotations

import pytest

from dsjourney import detection, forecasting, recommend, retrieval, rl, spark
from dsjourney.benchmark import available_models
from dsjourney.config import load_project_config
from dsjourney.paths import available_projects, project_dir
from dsjourney.pipeline import load_pipeline

PROJECTS = available_projects()


def test_the_expected_projects_are_present() -> None:
    assert set(PROJECTS) == {
        "article_search",
        "bart_ridership",
        "cartpole_balance",
        "customer_segments",
        "diabetes_screening",
        "frozenlake_control",
        "marvel_network",
        "object_detection",
        "image_classifiers",
        "istanbul_housing",
        "laptop_price",
        "loan_default",
        "movie_recommender",
        "review_sentiment",
        "series_forecast",
    }


@pytest.mark.parametrize("project", PROJECTS)
def test_pipeline_module_is_importable(project: str) -> None:
    module = load_pipeline(project)
    assert module.CONFIG.name == project


@pytest.mark.parametrize("project", PROJECTS)
def test_pipeline_exposes_the_contract(project: str) -> None:
    module = load_pipeline(project)
    for attribute in ("CONFIG", "load_raw", "build_features", "prepare_input"):
        assert hasattr(module, attribute), f"{project} is missing {attribute}"


@pytest.mark.parametrize("project", PROJECTS)
def test_configured_estimator_exists_in_the_registry(project: str) -> None:
    """Each task family has its own registry; a config must name one of its members."""
    config = load_project_config(project)

    if config.task == "image-classification":
        pytest.skip("image projects build Keras models, not registry estimators")
    # Spark algorithms live in their own registry: they are not scikit-learn
    # estimators and cannot be swept by `dsj benchmark`.
    if config.model.estimator.startswith("spark_"):
        assert config.model.estimator in spark.available_models()
        return
    registries = {
        "forecasting": forecasting.available_models,
        "recommendation": recommend.available_models,
        "retrieval": retrieval.available_models,
        "detection": detection.available_models,
        "control": rl.available_models,
    }
    if config.task in registries:
        assert config.model.estimator in registries[config.task]()
        return
    assert config.model.estimator in available_models(config.task)


# Not every project trains a model: retrieval builds an index, detection runs a
# classifier over images. What every project must ship is one runnable entry
# point, whatever it is called.
ENTRY_POINTS = ("train.py", "detect.py", "search.py", "predict.py")


@pytest.mark.parametrize("project", PROJECTS)
def test_project_ships_an_entry_point(project: str) -> None:
    directory = project_dir(project)
    present = [name for name in ENTRY_POINTS if (directory / name).is_file()]
    assert present, f"{project} ships none of {ENTRY_POINTS}"


@pytest.mark.parametrize("project", PROJECTS)
def test_project_documents_itself_in_both_languages(project: str) -> None:
    directory = project_dir(project)
    assert (directory / "README.md").is_file(), f"{project} needs an English README"
    assert (directory / "README.tr.md").is_file(), f"{project} needs a Turkish README"


def test_load_pipeline_reports_an_unknown_project() -> None:
    with pytest.raises(ModuleNotFoundError, match="has no pipeline module"):
        load_pipeline("not_a_project")
