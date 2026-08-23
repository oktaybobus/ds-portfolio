"""Sample images and known face counts for the detection project.

The source notebook displayed everything through ``cv2.imshow`` followed by
``cv2.waitKey()``. That opens a desktop window and blocks until a key is
pressed - fine on the laptop it was written on, and a hang on any server, in
CI, or in a notebook someone else runs. Nothing here opens a window.

``FACE_COUNTS`` is the ground truth the parameter sweep scores against, counted
by looking at the photographs rather than by trusting a detector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from dsjourney import detection
from dsjourney.config import load_project_config
from dsjourney.datasets import DatasetNotFoundError
from dsjourney.paths import project_data_dir

CONFIG = load_project_config("object_detection")

FRONTAL_CASCADE = "haarcascade_frontalface_default.xml"
EYE_CASCADE = "haarcascade_eye.xml"

# Faces visible and roughly frontal, counted by eye. g8.jpg is the reference
# image for tuning: seven leaders, all facing the camera, no ambiguity.
FACE_COUNTS = {
    # Seven leaders; six faces are fully frontal and the seventh is occluded by
    # the person in front. At the configured settings the detector finds those
    # six with no false positives - verified by drawing the boxes and looking.
    "g8.jpg": 7,
    # Students photographed from behind, so a *frontal* cascade should find
    # nothing, and nothing is the right answer rather than a failure. At
    # min_neighbours=2 it returns two boxes: one covering half the room and one
    # on the whiteboard. Both are false positives, also verified by looking.
    "classroom.jpg": 0,
}

REFERENCE_IMAGE = "g8.jpg"

# Images with no people in them, used to show what the detector does when there
# is nothing to detect.
OBJECT_IMAGES = ["cars.jpg", "traffic.jpg", "dogs.jpg", "kitchen.jpeg", "Sunflowers.jpg"]


def images_directory() -> Path:
    """Return the directory holding the sample images and cascade files."""
    return project_data_dir(CONFIG.name)


def cascade_path(name: str = FRONTAL_CASCADE) -> Path:
    """Return the path to a Haar cascade file."""
    path = images_directory() / name
    if not path.is_file():
        raise DatasetNotFoundError(
            f"{name} not found at {path}. "
            f"Run: uv run python scripts/fetch_assets.py --project {CONFIG.name}"
        )
    return path


def image_path(name: str) -> Path:
    """Return the path to a sample image."""
    path = images_directory() / name
    if not path.is_file():
        raise DatasetNotFoundError(
            f"{name} not found at {path}. "
            f"Run: uv run python scripts/fetch_assets.py --project {CONFIG.name}"
        )
    return path


def load_raw() -> pd.DataFrame:
    """Return one row per sample image, for the generic CLI commands."""
    try:
        paths = detection.list_images(images_directory())
    except FileNotFoundError as error:
        raise DatasetNotFoundError(str(error)) from error

    return pd.DataFrame(
        [
            {
                "image": path.name,
                "kilobytes": round(path.stat().st_size / 1024, 1),
                "known_faces": FACE_COUNTS.get(path.name),
            }
            for path in paths
        ]
    )


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the image listing unchanged; detection reads pixels, not rows."""
    return frame


def detect_in(name: str, **overrides: Any) -> list[detection.BoundingBox]:
    """Run the configured face detector over one sample image."""
    params = {**CONFIG.model.params, **overrides}
    image = detection.load_image(image_path(name))
    boxes = detection.detect_faces(
        image,
        cascade_path(),
        scale_factor=float(params["scale_factor"]),
        min_neighbours=int(params["min_neighbours"]),
        min_size=int(params["min_size"]),
    )
    return detection.non_max_suppression(boxes)


def prepare_input(payload: dict[str, Any]) -> pd.DataFrame:
    """Not applicable: detection reads an image file, not a record."""
    raise NotImplementedError(
        "object_detection reads images. "
        "Use: python projects/object_detection/detect.py --image cars.jpg"
    )
