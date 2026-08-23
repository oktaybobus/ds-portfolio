#!/usr/bin/env python
"""Classify a single image with one of the trained models.

Usage:
    python projects/image_classifiers/predict.py --dataset grape --image leaf.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dsjourney import vision
from dsjourney.paths import project_artifacts_dir
from projects.image_classifiers import pipeline


def predict(dataset: str, image_path: Path, *, top: int = 3) -> list[dict[str, float]]:
    """Return the top-k predicted classes with their probabilities."""
    tf = vision.require_tensorflow()
    spec = pipeline.spec_for(dataset)
    directory = project_artifacts_dir("image_classifiers") / dataset

    model_file = directory / "model.keras"
    if not model_file.is_file():
        raise FileNotFoundError(
            f"no trained model at {model_file}. "
            f"Run: python projects/image_classifiers/train.py --dataset {dataset}"
        )

    labels = json.loads((directory / "labels.json").read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(model_file)

    image = tf.keras.utils.load_img(image_path, target_size=(spec.image_size, spec.image_size))
    batch = np.expand_dims(tf.keras.utils.img_to_array(image), axis=0)
    probabilities = model.predict(batch, verbose=0)[0]

    ranked = np.argsort(probabilities)[::-1][:top]
    return [
        {"label": labels[str(index)], "probability": float(probabilities[index])}
        for index in ranked
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(pipeline.DATASETS))
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args(argv)

    for entry in predict(args.dataset, args.image, top=args.top):
        print(f"{entry['label']:30} {entry['probability']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
