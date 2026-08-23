#!/usr/bin/env python
"""Run detection over the sample images and write annotated figures.

Usage:
    python projects/object_detection/detect.py
    python projects/object_detection/detect.py --image g8.jpg --sweep
    python projects/object_detection/detect.py --yolo --image cars.jpg
"""

from __future__ import annotations

import argparse
import json

from dsjourney import detection, viz
from dsjourney.paths import project_artifacts_dir
from projects.object_detection import pipeline


def run_faces(*, save: bool = True) -> dict[str, float]:
    """Detect faces in every image with a known count and score the result."""
    directory = project_artifacts_dir("object_detection", create=True)
    detected_total = expected_total = 0

    for name, expected in pipeline.FACE_COUNTS.items():
        image = detection.load_image(pipeline.image_path(name))
        boxes = pipeline.detect_in(name)
        detected_total += len(boxes)
        expected_total += expected
        print(f"{name:16} expected {expected} | detected {len(boxes)}")

        if save:
            viz.save_figure(
                detection.draw_boxes(image, boxes, title=f"{name}: {len(boxes)} face(s)"),
                directory / f"{name.rsplit('.', 1)[0]}_faces.png",
            )

    metrics = {
        "detected": float(detected_total),
        "expected": float(expected_total),
        "error": float(abs(detected_total - expected_total)),
    }
    print(" | ".join(f"{k} {v:.0f}" for k, v in metrics.items()))

    if save:
        (directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        metadata = {
            "model_class": "HaarCascade",
            "images_scored": len(pipeline.FACE_COUNTS),
            **{k: float(v) for k, v in pipeline.CONFIG.model.params.items()},
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"artifacts: {directory}")
    return metrics


def run_sweep(name: str) -> None:
    """Print the cascade parameter sweep for one image."""
    image = detection.load_image(pipeline.image_path(name))
    table = detection.sweep_cascade_parameters(
        image, pipeline.cascade_path(), expected=pipeline.FACE_COUNTS.get(name)
    )
    print(table.to_string(index=False))


def run_yolo(name: str, *, save: bool = True) -> None:
    """Detect objects in one image with YOLO."""
    image = detection.load_image(pipeline.image_path(name))
    boxes = detection.detect_objects(image)
    counts: dict[str, int] = {}
    for box in boxes:
        counts[box.label] = counts.get(box.label, 0) + 1
    print(f"{name}: " + (", ".join(f"{n} x {label}" for label, n in counts.items()) or "nothing"))

    if save:
        directory = project_artifacts_dir("object_detection", create=True)
        viz.save_figure(
            detection.draw_boxes(image, boxes, title=f"{name}: YOLO"),
            directory / f"{name.rsplit('.', 1)[0]}_yolo.png",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="run on one image instead of the scored set")
    parser.add_argument("--sweep", action="store_true", help="sweep cascade parameters")
    parser.add_argument("--yolo", action="store_true", help="use YOLO instead of Haar cascades")
    parser.add_argument("--no-save", action="store_true", help="do not write figures")
    args = parser.parse_args(argv)

    if args.sweep:
        run_sweep(args.image or pipeline.REFERENCE_IMAGE)
        return 0
    if args.yolo:
        run_yolo(args.image or "cars.jpg", save=not args.no_save)
        return 0

    run_faces(save=not args.no_save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
