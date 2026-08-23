"""Model registry and benchmarking.

The course notebooks each pasted a ~150-line ``algo_test`` block that fit twenty
odd estimators and printed a table. That block is generalised here into a
registry plus one :func:`compare_models` function shared by every project, and
extended so the winner can be handed straight to a training script.

Gradient-boosting libraries (XGBoost, CatBoost, LightGBM) are optional: if the
``boost`` extra is not installed the registry simply offers fewer models rather
than failing to import.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDClassifier,
)
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from dsjourney.evaluate import classification_scores, regression_scores

Factory = Callable[..., BaseEstimator]

# Factories merge their defaults with the caller's kwargs rather than passing
# both positionally, so a project config can override `max_iter` or `verbose`
# without colliding with the default the registry supplies.

# Tree ensembles are scale-invariant; everything else is min-max scaled before fitting.
_SCALE_EXEMPT = {
    "decision_tree",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "adaboost",
    "xgboost",
    "catboost",
    "lightgbm",
}

REGRESSORS: dict[str, Factory] = {
    "linear": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elastic_net": ElasticNet,
    "huber": HuberRegressor,
    "bayesian_ridge": BayesianRidge,
    "knn": KNeighborsRegressor,
    "svr": SVR,
    "mlp": lambda **kw: MLPRegressor(**{"max_iter": 500, "hidden_layer_sizes": (100, 50), **kw}),
    "decision_tree": DecisionTreeRegressor,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "hist_gradient_boosting": HistGradientBoostingRegressor,
    "adaboost": AdaBoostRegressor,
}

CLASSIFIERS: dict[str, Factory] = {
    "logistic": lambda **kw: LogisticRegression(**{"max_iter": 1000, **kw}),
    "sgd": SGDClassifier,
    "linear_svc": LinearSVC,
    "svc": SVC,
    "knn": KNeighborsClassifier,
    "gaussian_nb": GaussianNB,
    "multinomial_nb": MultinomialNB,
    "bernoulli_nb": BernoulliNB,
    "mlp": lambda **kw: MLPClassifier(**{"max_iter": 500, **kw}),
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "extra_trees": ExtraTreesClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "adaboost": AdaBoostClassifier,
}

CLUSTERERS: dict[str, Factory] = {
    "kmeans": lambda **kw: KMeans(**{"n_init": 10, **kw}),
    "agglomerative": AgglomerativeClustering,
    "dbscan": DBSCAN,
}


def _register_optional_boosters() -> None:
    """Add XGBoost/CatBoost/LightGBM to the registries when they actually work.

    A plain ``ImportError`` guard is not enough on macOS: XGBoost and LightGBM
    install cleanly but raise at import time when the OpenMP runtime is absent
    (``brew install libomp`` fixes it). Catching every exception means a machine
    without libomp simply gets a smaller registry instead of an unusable
    package.
    """
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except Exception:
        pass
    else:
        REGRESSORS["xgboost"] = lambda **kw: XGBRegressor(**{"verbosity": 0, "n_jobs": -1, **kw})
        CLASSIFIERS["xgboost"] = lambda **kw: XGBClassifier(**{"verbosity": 0, "n_jobs": -1, **kw})

    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except Exception:
        pass
    else:
        REGRESSORS["catboost"] = lambda **kw: CatBoostRegressor(**{"verbose": 0, **kw})
        CLASSIFIERS["catboost"] = lambda **kw: CatBoostClassifier(**{"verbose": 0, **kw})

    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except Exception:
        pass
    else:
        REGRESSORS["lightgbm"] = lambda **kw: LGBMRegressor(**{"verbose": -1, **kw})
        CLASSIFIERS["lightgbm"] = lambda **kw: LGBMClassifier(**{"verbose": -1, **kw})


_register_optional_boosters()

_REGISTRIES: Mapping[str, dict[str, Factory]] = {
    "regression": REGRESSORS,
    "classification": CLASSIFIERS,
    "text-classification": CLASSIFIERS,
    "image-classification": CLASSIFIERS,
    "clustering": CLUSTERERS,
}


def available_models(task: str) -> list[str]:
    """Return the estimator keys registered for a task."""
    if task not in _REGISTRIES:
        raise ValueError(f"unknown task {task!r}; expected one of {sorted(_REGISTRIES)}")
    return sorted(_REGISTRIES[task])


def build_model(task: str, estimator: str, **params: Any) -> BaseEstimator:
    """Instantiate a registered estimator by name.

    Raises:
        KeyError: with the list of valid names, so a config typo is obvious.
    """
    registry = _REGISTRIES.get(task)
    if registry is None:
        raise ValueError(f"unknown task {task!r}; expected one of {sorted(_REGISTRIES)}")
    factory = registry.get(estimator)
    if factory is None:
        raise KeyError(f"unknown {task} estimator {estimator!r}; available: {sorted(registry)}")
    return factory(**params)


@dataclass(frozen=True)
class BenchmarkResult:
    """The comparison table plus the name and fitted instance of the winner."""

    table: pd.DataFrame
    best_name: str
    best_model: BaseEstimator

    @property
    def best_score(self) -> float:
        """The winning model's value on the ranking metric."""
        return float(self.table.iloc[0][self.table.columns[1]])


def compare_models(
    task: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    models: Sequence[str] | None = None,
    rank_by: str | None = None,
) -> BenchmarkResult:
    """Fit several estimators on the same split and rank them on a shared metric.

    Non-tree models are min-max scaled first, matching the convention used
    throughout the course. When the caller already standardised some columns the
    two scalings compose harmlessly - both are monotonic and both are fitted on
    the training split only - so no leakage is introduced either way. A model
    that fails to fit is recorded with NaN scores instead of aborting the whole
    comparison.

    Args:
        models: Subset of registry keys to try; defaults to the full registry.
        rank_by: Metric to sort on; defaults to ``r2`` for regression and ``f1``
            for classification. Lower-is-better metrics are sorted ascending.
    """
    registry = _REGISTRIES.get(task)
    if registry is None or task == "clustering":
        raise ValueError(f"compare_models does not support task {task!r}")

    names = list(models) if models else sorted(registry)
    is_regression = task == "regression"
    metric = rank_by or ("r2" if is_regression else "f1")
    lower_is_better = metric in {"rmse", "mae", "mape"}

    rows: list[dict[str, Any]] = []
    fitted: dict[str, BaseEstimator] = {}

    for name in names:
        started = time.perf_counter()
        try:
            model = build_model(task, name)
            train_features, test_features = _maybe_scale(name, x_train, x_test)
            model.fit(train_features, np.ravel(y_train))
            predictions = model.predict(test_features)
            scores = (
                regression_scores(y_test, predictions)
                if is_regression
                else classification_scores(y_test, predictions, average=_average_for(y_train))
            )
            fitted[name] = model
        except Exception as error:
            scores = {}
            rows.append({"model": name, **scores, "seconds": None, "error": str(error)[:120]})
            continue
        rows.append(
            {
                "model": name,
                **scores,
                "seconds": round(time.perf_counter() - started, 3),
                "error": None,
            }
        )

    table = pd.DataFrame(rows)
    if metric not in table.columns:
        reasons: list[str] = (
            [str(reason) for reason in table["error"].dropna().unique()[:3]]
            if "error" in table
            else []
        )
        detail = "; ".join(reasons) or "no errors were recorded"
        raise RuntimeError(
            f"every candidate model failed to fit, so no {metric!r} column exists. "
            f"First failures: {detail}"
        )

    ordered = table.sort_values(metric, ascending=lower_is_better, na_position="last").reset_index(
        drop=True
    )
    ordered = ordered[
        ["model", metric, *[c for c in ordered.columns if c not in {"model", metric}]]
    ]

    best_name = str(ordered.iloc[0]["model"])
    if best_name not in fitted:
        raise RuntimeError("every candidate model failed to fit; see the 'error' column")
    return BenchmarkResult(ordered, best_name, fitted[best_name])


def compare_text_models(
    x_train: pd.Series[str],
    y_train: pd.Series[int],
    x_test: pd.Series[str],
    y_test: pd.Series[int],
    *,
    models: Sequence[str] | None = None,
    rank_by: str = "f1",
    max_features: int = 5000,
    min_df: int = 2,
) -> BenchmarkResult:
    """Rank classifiers on raw text by wrapping each in a TF-IDF pipeline.

    :func:`compare_models` cannot be used directly here: a text project's
    features are documents, and handing raw strings to an estimator makes every
    candidate fail. Each model gets its own vectoriser, fitted on the training
    documents only, so the comparison stays leak-free.
    """
    from dsjourney.evaluate import classification_scores
    from dsjourney.text import build_text_pipeline

    names = list(models) if models else sorted(CLASSIFIERS)
    average = _average_for(y_train)
    lower_is_better = rank_by in {"log_loss"}

    rows: list[dict[str, Any]] = []
    fitted: dict[str, BaseEstimator] = {}

    for name in names:
        started = time.perf_counter()
        try:
            pipeline = build_text_pipeline(
                build_model("classification", name), max_features=max_features, min_df=min_df
            )
            pipeline.fit(x_train, y_train)
            scores = classification_scores(y_test, pipeline.predict(x_test), average=average)
            fitted[name] = pipeline
        except Exception as error:
            rows.append({"model": name, "seconds": None, "error": str(error)[:120]})
            continue
        rows.append(
            {
                "model": name,
                **scores,
                "seconds": round(time.perf_counter() - started, 3),
                "error": None,
            }
        )

    table = pd.DataFrame(rows)
    if rank_by not in table.columns:
        raise RuntimeError(f"every candidate model failed to fit; no {rank_by!r} column exists")

    ordered = table.sort_values(rank_by, ascending=lower_is_better, na_position="last").reset_index(
        drop=True
    )
    ordered = ordered[
        ["model", rank_by, *[c for c in ordered.columns if c not in {"model", rank_by}]]
    ]

    best_name = str(ordered.iloc[0]["model"])
    return BenchmarkResult(ordered, best_name, fitted[best_name])


def _maybe_scale(
    name: str, x_train: pd.DataFrame, x_test: pd.DataFrame
) -> tuple[np.ndarray | pd.DataFrame, np.ndarray | pd.DataFrame]:
    """Min-max scale the features unless the estimator is a tree ensemble."""
    if name in _SCALE_EXEMPT:
        return x_train, x_test
    scaler = MinMaxScaler()
    return scaler.fit_transform(x_train), scaler.transform(x_test)


def _average_for(y: pd.Series) -> str:
    """Pick the right averaging mode from the number of classes present."""
    return "binary" if pd.Series(y).nunique() <= 2 else "macro"
