"""Unit tests for detection geometry and the cascade wrapper.

The box arithmetic is tested with plain numbers; the image paths use the
committed sample photographs, so these run in CI without any download.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dsjourney import detection
from dsjourney.paths import project_data_dir

SAMPLES = project_data_dir("object_detection")
pytestmark = pytest.mark.skipif(
    not (SAMPLES / "g8.jpg").is_file(), reason="detection samples not fetched"
)


def _box(x: int, y: int, w: int, h: int, score: float = 1.0) -> detection.BoundingBox:
    return detection.BoundingBox(x, y, w, h, score=score)


def test_box_geometry() -> None:
    box = _box(10, 20, 30, 40)
    assert box.area == 1200
    assert box.xyxy == (10, 20, 40, 60)
    assert box.centre == (25.0, 40.0)


def test_iou_is_one_for_identical_boxes() -> None:
    assert detection.iou(_box(0, 0, 10, 10), _box(0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_is_zero_for_disjoint_boxes() -> None:
    assert detection.iou(_box(0, 0, 10, 10), _box(50, 50, 10, 10)) == 0.0


def test_iou_is_zero_when_boxes_only_touch() -> None:
    assert detection.iou(_box(0, 0, 10, 10), _box(10, 0, 10, 10)) == 0.0


def test_iou_matches_a_hand_calculation() -> None:
    """Two 10x10 boxes offset by 5: overlap 25, union 175."""
    assert detection.iou(_box(0, 0, 10, 10), _box(5, 5, 10, 10)) == pytest.approx(25 / 175)


def test_non_max_suppression_collapses_a_cluster() -> None:
    """A cascade fires several times around one face; the count must not."""
    boxes = [_box(0, 0, 10, 10, 0.9), _box(1, 1, 10, 10, 0.8), _box(2, 2, 10, 10, 0.7)]
    kept = detection.non_max_suppression(boxes, threshold=0.3)
    assert len(kept) == 1
    assert kept[0].score == pytest.approx(0.9)  # the strongest survives


def test_non_max_suppression_keeps_separate_objects() -> None:
    boxes = [_box(0, 0, 10, 10), _box(100, 100, 10, 10)]
    assert len(detection.non_max_suppression(boxes)) == 2


def test_non_max_suppression_on_an_empty_list() -> None:
    assert detection.non_max_suppression([]) == []


def test_load_image_returns_rgb() -> None:
    image = detection.load_image(SAMPLES / "g8.jpg")
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert image.dtype == np.uint8


def test_load_image_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no image at"):
        detection.load_image(tmp_path / "nothing.jpg")


def test_load_image_rejects_a_non_image(tmp_path: Path) -> None:
    path = tmp_path / "not_an_image.jpg"
    path.write_text("this is text", encoding="utf-8")
    with pytest.raises(ValueError, match="not a readable image"):
        detection.load_image(path)


def test_to_grayscale_drops_the_channels() -> None:
    image = detection.load_image(SAMPLES / "g8.jpg")
    grey = detection.to_grayscale(image)
    assert grey.ndim == 2
    assert detection.to_grayscale(grey).ndim == 2  # idempotent


def test_detect_faces_finds_the_known_faces() -> None:
    """Six of the seven leaders in g8.jpg, verified by drawing the boxes."""
    image = detection.load_image(SAMPLES / "g8.jpg")
    boxes = detection.detect_faces(
        image,
        SAMPLES / "haarcascade_frontalface_default.xml",
        scale_factor=1.05,
        min_neighbours=5,
    )
    assert len(boxes) == 6
    assert all(box.label == "face" for box in boxes)


def test_the_tutorial_scale_factor_finds_nothing() -> None:
    """1.30 is a common default and detects zero of seven faces here.

    Locking this in keeps the README's claim honest if OpenCV changes.
    """
    image = detection.load_image(SAMPLES / "g8.jpg")
    boxes = detection.detect_faces(
        image,
        SAMPLES / "haarcascade_frontalface_default.xml",
        scale_factor=1.30,
        min_neighbours=5,
    )
    assert len(boxes) == 0


def test_detect_faces_reports_a_missing_cascade(tmp_path: Path) -> None:
    image = detection.load_image(SAMPLES / "g8.jpg")
    with pytest.raises(FileNotFoundError, match="no cascade file"):
        detection.detect_faces(image, tmp_path / "nope.xml")


def test_detect_faces_rejects_a_bad_cascade(tmp_path: Path) -> None:
    path = tmp_path / "broken.xml"
    path.write_text("<not a cascade/>", encoding="utf-8")
    image = detection.load_image(SAMPLES / "g8.jpg")
    with pytest.raises(ValueError, match="did not load as a cascade"):
        detection.detect_faces(image, path)


def test_sweep_ranks_by_error() -> None:
    image = detection.load_image(SAMPLES / "g8.jpg")
    table = detection.sweep_cascade_parameters(
        image,
        SAMPLES / "haarcascade_frontalface_default.xml",
        expected=7,
        scale_factors=(1.05, 1.30),
        neighbour_counts=(5,),
    )
    assert table["error"].is_monotonic_increasing
    assert table.iloc[0]["scale_factor"] == pytest.approx(1.05)


def test_draw_boxes_returns_a_figure() -> None:
    image = detection.load_image(SAMPLES / "g8.jpg")
    figure = detection.draw_boxes(image, [_box(10, 10, 20, 20)], title="test")
    assert figure.axes


def test_list_images_skips_non_images() -> None:
    names = {path.name for path in detection.list_images(SAMPLES)}
    assert "g8.jpg" in names
    assert not any(name.endswith(".xml") for name in names)


def test_available_models_lists_the_registry() -> None:
    assert set(detection.available_models()) == {"haar_cascade", "yolo"}
