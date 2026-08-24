"""Object detection: boxes, overlap, and figures instead of windows.

The source notebook drove OpenCV through `cv2.imshow` and `cv2.waitKey`, which
opens a desktop window and blocks. That cannot run on a server, in CI, or in a
notebook someone else opens - the cell just hangs. Everything here returns
arrays and Matplotlib figures instead, the same rule :mod:`dsjourney.viz`
follows.

Haar cascades need no weights download and run in milliseconds, so they carry
the project. YOLO is available through :func:`detect_objects` when the optional
``yolo`` extra is installed, behind a lazy import.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Detections whose boxes overlap by more than this are treated as the same
# object by non_max_suppression.
DEFAULT_IOU_THRESHOLD = 0.3
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class BoundingBox:
    """One detection, in pixel coordinates with the origin at the top left."""

    x: int
    y: int
    width: int
    height: int
    label: str = "object"
    score: float = 1.0

    @property
    def area(self) -> int:
        """Box area in pixels."""
        return max(self.width, 0) * max(self.height, 0)

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        """Corners as (left, top, right, bottom)."""
        return self.x, self.y, self.x + self.width, self.y + self.height

    @property
    def centre(self) -> tuple[float, float]:
        """Box centre as (x, y)."""
        return self.x + self.width / 2, self.y + self.height / 2


def require_opencv() -> Any:
    """Import OpenCV, or explain how to install it."""
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "OpenCV is required for detection. Install it with: uv sync --extra detect"
        ) from error
    return cv2


def load_image(path: Path) -> np.ndarray:
    """Read an image as an RGB array.

    OpenCV reads BGR. Every plotting and model library expects RGB, and the
    channel swap is the single most common source of images that look correct
    in one window and blue in another - so it happens once, here.
    """
    cv2 = require_opencv()
    if not Path(path).is_file():
        raise FileNotFoundError(f"no image at {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"{path} is not a readable image")
    return np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB array to single-channel grey."""
    cv2 = require_opencv()
    if image.ndim == 2:
        return image
    return np.asarray(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))


def detect_faces(
    image: np.ndarray,
    cascade_path: Path,
    *,
    scale_factor: float = 1.1,
    min_neighbours: int = 5,
    min_size: int = 24,
) -> list[BoundingBox]:
    """Detect faces with a Haar cascade classifier.

    Args:
        scale_factor: How much the search window shrinks per pyramid level.
            Closer to 1.0 finds more faces and costs more time; above ~1.3 it
            steps over faces between scales.
        min_neighbours: How many overlapping detections a region needs to count.
            Low values produce false positives, high values miss real faces.
            :func:`sweep_cascade_parameters` measures the trade rather than
            leaving it at whatever the tutorial used.
    """
    cv2 = require_opencv()
    if not Path(cascade_path).is_file():
        raise FileNotFoundError(f"no cascade file at {cascade_path}")

    # A malformed file makes OpenCV raise SystemError from its C++ parser rather
    # than returning an empty classifier, so both outcomes are funnelled into
    # one clear error naming the file.
    try:
        classifier = cv2.CascadeClassifier(str(cascade_path))
        unusable = classifier.empty()
    except Exception as error:
        raise ValueError(f"{cascade_path} did not load as a cascade classifier") from error
    if unusable:
        raise ValueError(f"{cascade_path} did not load as a cascade classifier")

    found = classifier.detectMultiScale(
        to_grayscale(image),
        scaleFactor=scale_factor,
        minNeighbors=min_neighbours,
        minSize=(min_size, min_size),
    )
    return [BoundingBox(int(x), int(y), int(w), int(h), label="face") for x, y, w, h in found]


def detect_objects(
    image: np.ndarray, *, weights: str = "yolov8n.pt", confidence: float = 0.25
) -> list[BoundingBox]:
    """Detect objects with YOLO, downloading the weights on first use.

    Raises:
        ImportError: naming the extra to install, when ultralytics is absent.
    """
    try:
        from ultralytics import YOLO
        from ultralytics.engine.results import Results
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "ultralytics is required for YOLO detection. Install it with: uv sync --extra yolo"
        ) from error

    model = YOLO(weights)
    results = model.predict(image, conf=confidence, verbose=False)

    boxes: list[BoundingBox] = []
    for result in results:
        # predict() is typed to also cover streaming/embedding calls, whose
        # results carry a bare Tensor instead of a Results object - neither
        # applies to the plain detection call above, but the type checker
        # still has to be shown that. Boxes.__getitem__ exists without
        # __iter__, so indexing (rather than a for-in loop) is what keeps
        # mypy convinced it is safe to read.
        if not isinstance(result, Results) or result.boxes is None:
            continue
        names = result.names
        detected = result.boxes
        for index in range(len(detected)):
            box = detected[index]
            left, top, right, bottom = (float(v) for v in box.xyxy[0].tolist())
            boxes.append(
                BoundingBox(
                    x=int(left),
                    y=int(top),
                    width=int(right - left),
                    height=int(bottom - top),
                    label=str(names[int(box.cls[0])]),
                    score=float(box.conf[0]),
                )
            )
    return boxes


def iou(first: BoundingBox, second: BoundingBox) -> float:
    """Intersection over union of two boxes, from 0 (disjoint) to 1 (identical)."""
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.x + first.width, second.x + second.width)
    bottom = min(first.y + first.height, second.y + second.height)

    if right <= left or bottom <= top:
        return 0.0

    overlap = (right - left) * (bottom - top)
    union = first.area + second.area - overlap
    return overlap / union if union else 0.0


def non_max_suppression(
    boxes: list[BoundingBox], *, threshold: float = DEFAULT_IOU_THRESHOLD
) -> list[BoundingBox]:
    """Keep the highest-scoring box in each cluster of overlapping detections.

    A cascade fires several times around one face at neighbouring scales, so a
    raw detection count is an overcount. Suppression is what turns "seventeen
    detections" into "five faces".
    """
    ordered = sorted(boxes, key=lambda box: (box.score, box.area), reverse=True)
    kept: list[BoundingBox] = []
    for candidate in ordered:
        if all(iou(candidate, keeper) <= threshold for keeper in kept):
            kept.append(candidate)
    return kept


def draw_boxes(
    image: np.ndarray,
    boxes: list[BoundingBox],
    *,
    title: str = "",
    show_labels: bool = True,
    figsize: tuple[int, int] = (10, 7),
) -> Any:
    """Return a figure of the image with its detections outlined."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    figure, axes = plt.subplots(figsize=figsize)
    axes.imshow(image)
    axes.axis("off")

    for box in boxes:
        axes.add_patch(
            Rectangle(
                (box.x, box.y),
                box.width,
                box.height,
                linewidth=2,
                edgecolor="#00d17a",
                facecolor="none",
            )
        )
        if show_labels:
            caption = box.label if box.score >= 1.0 else f"{box.label} {box.score:.2f}"
            axes.text(
                box.x,
                max(box.y - 6, 8),
                caption,
                color="#00d17a",
                fontsize=9,
                weight="bold",
                bbox={"facecolor": "black", "alpha": 0.5, "pad": 1, "edgecolor": "none"},
            )

    axes.set_title(title or f"{len(boxes)} detection(s)")
    figure.tight_layout()
    return figure


def sweep_cascade_parameters(
    image: np.ndarray,
    cascade_path: Path,
    *,
    expected: int | None = None,
    scale_factors: tuple[float, ...] = (1.05, 1.1, 1.2, 1.3),
    neighbour_counts: tuple[int, ...] = (3, 5, 8),
) -> Any:
    """Count detections across cascade settings, so the defaults are a choice.

    When ``expected`` is given, the error column makes the best setting obvious
    instead of leaving it to whichever numbers the tutorial happened to use.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for scale_factor in scale_factors:
        for neighbours in neighbour_counts:
            boxes = detect_faces(
                image, cascade_path, scale_factor=scale_factor, min_neighbours=neighbours
            )
            deduplicated = non_max_suppression(boxes)
            row: dict[str, Any] = {
                "scale_factor": scale_factor,
                "min_neighbours": neighbours,
                "raw": len(boxes),
                "after_nms": len(deduplicated),
            }
            if expected is not None:
                row["error"] = abs(len(deduplicated) - expected)
            rows.append(row)

    table = pd.DataFrame(rows)
    return table.sort_values("error").reset_index(drop=True) if expected is not None else table


DETECTORS = {"haar_cascade": detect_faces, "yolo": detect_objects}


def available_models() -> list[str]:
    """Return the registered detector names."""
    return sorted(DETECTORS)


def list_images(directory: Path) -> list[Path]:
    """Return the image files in a directory, sorted by name."""
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
