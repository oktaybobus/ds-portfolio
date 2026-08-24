#!/usr/bin/env python
"""Train one or more image classifiers.

Requires the optional deep-learning extra and a Kaggle-capable network:

    uv sync --extra dl --extra data
    python projects/image_classifiers/train.py --dataset grape
    python projects/image_classifiers/train.py --all --epochs 5

Each run writes ``artifacts/image_classifiers/<dataset>/`` containing the Keras
model, the class labels, the metrics, the training history and a confusion
matrix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dsjourney import evaluate, vision, viz
from dsjourney.paths import project_artifacts_dir
from projects.image_classifiers import pipeline


def train_one(dataset: str, *, epochs: int | None = None, save: bool = True) -> dict[str, float]:
    """Download, train and evaluate a single image dataset; return its metrics."""
    spec = pipeline.spec_for(dataset)
    print(f"[{dataset}] downloading {spec.kaggle_handle}")
    root = vision.download_dataset(spec)
    image_root = vision.find_image_root(root)
    print(f"[{dataset}] class folders under {image_root}")

    train_data, validation_data, class_names = vision.load_image_datasets(
        image_root, image_size=spec.image_size, batch_size=spec.batch_size
    )
    print(f"[{dataset}] {len(class_names)} classes: {', '.join(class_names)}")

    model = vision.build_model(spec, len(class_names))
    history = vision.train_image_model(
        model, train_data, validation_data, epochs=epochs or spec.epochs
    )

    truths, predictions = vision.collect_predictions(model, validation_data)
    metrics = evaluate.classification_scores(truths, predictions, average="macro")
    print(f"[{dataset}] " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    if save:
        _save_outputs(dataset, model, class_names, metrics, history, truths, predictions)
    return metrics


def _save_outputs(
    dataset: str,
    model: object,
    class_names: list[str],
    metrics: dict[str, float],
    history: object,
    truths: object,
    predictions: object,
) -> Path:
    """Write the model, labels, metrics, history, metadata and confusion matrix."""
    spec = pipeline.spec_for(dataset)
    directory = project_artifacts_dir("image_classifiers", create=True) / dataset
    directory.mkdir(parents=True, exist_ok=True)

    frame = vision.history_frame(history)
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                # Named the way every other project names its estimator, so
                # RESULTS.md has something to put in the model column.
                "model_class": "MobileNetV2" if spec.architecture == "transfer" else "CNN",
                "dataset": dataset,
                "title": spec.title,
                "kaggle_handle": spec.kaggle_handle,
                "classes": len(class_names),
                "class_names": list(class_names),
                "image_size": spec.image_size,
                "epochs_run": len(frame),
                "best_val_accuracy": round(float(frame["val_accuracy"].max()), 4),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    model.save(directory / "model.keras")  # type: ignore[attr-defined]
    (directory / "labels.json").write_text(
        json.dumps({str(index): name for index, name in enumerate(class_names)}, indent=2),
        encoding="utf-8",
    )
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    frame.to_csv(directory / "history.csv", index=False)

    matrix = evaluate.confusion_frame(truths, predictions, labels=list(range(len(class_names))))
    viz.save_figure(viz.confusion_matrix_plot(matrix), directory / "confusion_matrix.png")

    print(f"[{dataset}] artifacts: {directory}")
    return directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset", choices=sorted(pipeline.DATASETS), help="which dataset to train"
    )
    parser.add_argument("--all", action="store_true", help="train every dataset in sequence")
    parser.add_argument("--epochs", type=int, help="override the per-dataset epoch count")
    parser.add_argument("--no-save", action="store_true", help="do not write artifacts")
    args = parser.parse_args(argv)

    if not args.dataset and not args.all:
        parser.error("pass --dataset <key> or --all")

    targets = sorted(pipeline.DATASETS) if args.all else [args.dataset]
    failures = 0
    for dataset in targets:
        try:
            train_one(dataset, epochs=args.epochs, save=not args.no_save)
        except Exception as error:
            failures += 1
            print(f"[{dataset}] FAILED: {type(error).__name__}: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
