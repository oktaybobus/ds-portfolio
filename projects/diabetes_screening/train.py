#!/usr/bin/env python
"""Train the Pima classifier on Spark MLlib and check it against scikit-learn.

Usage:
    python projects/diabetes_screening/train.py
    python projects/diabetes_screening/train.py --keep-zeros
    python projects/diabetes_screening/train.py --no-save
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dsjourney import evaluate
from dsjourney import spark as dsspark
from dsjourney.paths import project_artifacts_dir
from projects.diabetes_screening import pipeline


def _split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split once, stratified, and hand the same rows to both engines.

    Spark's ``randomSplit`` is not stratified and draws from its own RNG, so
    letting each engine split independently would leave the two sets of metrics
    incomparable - the difference could be the model or could be the split.
    """
    config = pipeline.CONFIG.split
    train, test = train_test_split(
        frame,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=frame[pipeline.TARGET] if config.stratify else None,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _fit_spark(
    spark: object, train: pd.DataFrame, test: pd.DataFrame
) -> tuple[dict[str, float], dict[str, float]]:
    """Fit MLlib logistic regression, returning its scores and its coefficients."""
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler

    params = pipeline.CONFIG.model.params
    features = list(pipeline.FEATURES)
    imputed = [f"{name}_filled" for name in features]

    stages = [
        # Fitted inside the pipeline, so the medians come from the training
        # fold only. This is the same leak the scaler caused in week 1.
        Imputer(inputCols=features, outputCols=imputed, strategy="median"),
        VectorAssembler(inputCols=imputed, outputCol="raw_features"),
        # Spark centres nothing by default (`withMean=False`); scikit-learn
        # centres by default. Left alone, the two engines fit different models
        # and the cross-check below would be comparing preprocessing, not
        # implementations.
        StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True),
        LogisticRegression(
            labelCol=pipeline.TARGET,
            featuresCol="features",
            maxIter=int(params["max_iter"]),
            regParam=float(params["reg_param"]),
            elasticNetParam=float(params["elastic_net_param"]),
        ),
    ]

    train_sdf = spark.createDataFrame(train)  # type: ignore[attr-defined]
    test_sdf = spark.createDataFrame(test)  # type: ignore[attr-defined]
    model = Pipeline(stages=stages).fit(train_sdf)
    predictions = model.transform(test_sdf)

    scores = dsspark.binary_classification_scores(predictions, label_col=pipeline.TARGET)
    weights = model.stages[-1].coefficients.toArray()
    coefficients = {name: float(value) for name, value in zip(features, weights, strict=True)}
    return scores, coefficients


def _fit_sklearn(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, float]:
    """Fit the same model in scikit-learn on the same rows, as a cross-check."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features = list(pipeline.FEATURES)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        # C=inf is "no regularisation", matching Spark's regParam of 0.
        # scikit-learn applies L2 at C=1.0 unless told otherwise, which is a
        # different model - and `penalty=None` is deprecated as of 1.8.
        LogisticRegression(C=np.inf, max_iter=int(pipeline.CONFIG.model.params["max_iter"])),
    )
    model.fit(train[features], train[pipeline.TARGET])
    predicted = model.predict(test[features])
    probabilities = model.predict_proba(test[features])
    return evaluate.classification_scores(test[pipeline.TARGET], predicted, y_proba=probabilities)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-zeros",
        action="store_true",
        help="train on the raw file, zeros and all, the way the notebook did",
    )
    parser.add_argument("--no-save", action="store_true", help="do not write artifacts")
    args = parser.parse_args(argv)

    if not dsspark.spark_available():
        print(dsspark.INSTALL_HINT)
        return 2

    raw = pipeline.load_raw()
    frame = raw if args.keep_zeros else pipeline.build_features(raw)
    print(f"{len(frame)} records | {frame[pipeline.TARGET].mean():.1%} diabetic")

    if args.keep_zeros:
        print("--keep-zeros: impossible zeros left in place, as in the source notebook")
    else:
        missing = pipeline.missing_after_cleaning(raw)
        print("Zeros recovered as missing:")
        for column, count in missing.items():
            print(f"  {column:26} {count:4,} ({count / len(frame):5.1%})")

    train, test = _split(frame)
    baseline = dsspark.majority_baseline(test[pipeline.TARGET])
    print(
        f"\ntrain {len(train)} | test {len(test)} | "
        f"predicting the majority class alone scores {baseline:.3f}"
    )

    params = pipeline.CONFIG.model.params
    with dsspark.session(
        "diabetes_screening",
        cores=str(params["cores"]),
        shuffle_partitions=int(params["shuffle_partitions"]),
    ) as spark:
        scores, coefficients = _fit_spark(spark, train, test)

    reference = _fit_sklearn(train, test)

    print("\n                 Spark MLlib   scikit-learn")
    for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {name:12} {scores[name]:11.4f} {reference[name]:14.4f}")

    # The notebook printed its AUC under the label "Accuracy". Showing the gap
    # is the point of this project.
    gap = scores["roc_auc"] - scores["accuracy"]
    print(
        f"\n  The notebook's 'Accuracy: 0.854' was this ROC AUC ({scores['roc_auc']:.3f}).\n"
        f"  Actual accuracy is {scores['accuracy']:.3f} - {gap:.3f} lower, and only "
        f"{scores['accuracy'] - baseline:.3f} above always guessing 'not diabetic'."
    )

    print("\nStandardised coefficients (positive raises predicted risk):")
    for name, value in sorted(coefficients.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {name:26} {value:+.4f}")

    metrics = {**scores, "majority_baseline": baseline}
    if not args.no_save:
        directory = project_artifacts_dir("diabetes_screening", create=True)
        (directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        metadata = {
            "model_class": "SparkLogisticRegression",
            "engine": "pyspark",
            "rows": len(frame),
            "train_rows": len(train),
            "test_rows": len(test),
            "zeros_treated_as_missing": not args.keep_zeros,
            "sklearn_cross_check": {k: round(v, 4) for k, v in reference.items()},
            "coefficients": {k: round(v, 4) for k, v in coefficients.items()},
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"\nartifacts: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
