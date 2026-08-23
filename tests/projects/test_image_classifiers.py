"""Tests for the image-classification suite.

The catalogue and root-discovery logic are checked without TensorFlow or a
network; anything that needs Keras is marked ``needs_dl``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dsjourney import vision
from projects.image_classifiers import pipeline


def test_catalogue_lists_every_dataset() -> None:
    frame = pipeline.load_raw()
    assert len(frame) == 7
    assert set(frame["architecture"]) == {"cnn", "transfer"}


def test_spec_for_returns_the_right_dataset() -> None:
    spec = pipeline.spec_for("fish")
    assert spec.image_size == 170
    assert spec.architecture == "transfer"


def test_spec_for_names_the_alternatives_on_a_typo() -> None:
    with pytest.raises(KeyError, match="available:"):
        pipeline.spec_for("fishh")


def test_prepare_input_explains_the_right_entry_point() -> None:
    with pytest.raises(NotImplementedError, match=r"predict\.py"):
        pipeline.prepare_input({})


def _make_image(path: Path) -> None:
    """Write a one-byte-header file with an image suffix; content is irrelevant."""
    path.write_bytes(b"\x89PNG\r\n\x1a\n")


def test_find_image_root_locates_nested_class_folders(tmp_path: Path) -> None:
    """The archive layout the notebook hard-coded per dataset."""
    nested = tmp_path / "archive" / "Final Training Data"
    for label in ("healthy", "black_rot", "esca"):
        (nested / label).mkdir(parents=True)
        _make_image(nested / label / "a.jpg")

    assert vision.find_image_root(tmp_path) == nested


def test_find_image_root_prefers_the_directory_with_most_classes(tmp_path: Path) -> None:
    shallow = tmp_path / "shallow"
    for label in ("a", "b"):
        (shallow / label).mkdir(parents=True)
        _make_image(shallow / label / "x.png")

    deep = tmp_path / "deep" / "images"
    for label in ("a", "b", "c", "d"):
        (deep / label).mkdir(parents=True)
        _make_image(deep / label / "x.png")

    assert vision.find_image_root(tmp_path) == deep


def test_find_image_root_ignores_empty_class_folders(tmp_path: Path) -> None:
    root = tmp_path / "data"
    (root / "with_images").mkdir(parents=True)
    _make_image(root / "with_images" / "x.jpg")
    (root / "empty").mkdir()

    with pytest.raises(FileNotFoundError, match="at least 2 class folders"):
        vision.find_image_root(root)


@pytest.mark.needs_dl
def test_build_cnn_has_the_expected_output_shape() -> None:
    spec = pipeline.spec_for("grape")
    model = vision.build_model(spec, num_classes=4)
    assert model.output_shape == (None, 4)
    assert model.input_shape == (None, 128, 128, 3)


@pytest.mark.needs_dl
def test_transfer_model_freezes_the_backbone() -> None:
    spec = pipeline.spec_for("fish")
    model = vision.build_model(spec, num_classes=9)
    backbone = next(layer for layer in model.layers if layer.name.startswith("mobilenet"))
    assert not backbone.trainable


@pytest.mark.needs_dl
def test_unknown_architecture_is_rejected() -> None:
    spec = vision.ImageDatasetSpec(key="x", kaggle_handle="a/b", title="X", architecture="magic")
    with pytest.raises(ValueError, match="unknown architecture"):
        vision.build_model(spec, num_classes=2)
