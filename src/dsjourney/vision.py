"""Image-classification helpers.

TensorFlow is imported lazily inside each function so that importing
``dsjourney`` - and therefore running the CLI, the tests and the tabular
projects - never pays the multi-second Keras import cost or requires the
optional ``dl`` extra to be installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
MIN_CLASSES = 2


@dataclass(frozen=True)
class ImageDatasetSpec:
    """How to obtain and train on one image dataset."""

    key: str
    kaggle_handle: str
    title: str
    image_size: int = 128
    architecture: str = "cnn"
    dense_units: int = 256
    epochs: int = 15
    batch_size: int = 16


def require_tensorflow() -> Any:
    """Import TensorFlow, or explain how to install it.

    Raises:
        ImportError: with the exact command to run, when the extra is missing.
    """
    try:
        import tensorflow as tf
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "TensorFlow is required for image projects. Install it with: uv sync --extra dl"
        ) from error
    return tf


def download_dataset(spec: ImageDatasetSpec) -> Path:
    """Download a Kaggle dataset with kagglehub and return its local root."""
    try:
        import kagglehub
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "kagglehub is required to fetch image datasets. Install it with: uv sync --extra data"
        ) from error
    return Path(kagglehub.dataset_download(spec.kaggle_handle))


def find_image_root(root: Path, *, min_classes: int = MIN_CLASSES) -> Path:
    """Locate the directory whose immediate subdirectories are the class folders.

    Kaggle archives nest their data differently every time - sometimes one level
    down, sometimes behind a "Final Training Data" folder, sometimes split into
    train/test. Searching for the deepest directory that has the most
    image-bearing children removes that per-dataset guesswork.
    """
    best: tuple[int, Path] | None = None
    for candidate in [root, *(p for p in root.rglob("*") if p.is_dir())]:
        class_dirs = [child for child in sorted(candidate.iterdir()) if child.is_dir()]
        populated = [child for child in class_dirs if _contains_images(child)]
        if len(populated) < min_classes:
            continue
        if best is None or len(populated) > best[0]:
            best = (len(populated), candidate)

    if best is None:
        raise FileNotFoundError(
            f"no directory under {root} has at least {min_classes} class folders containing images"
        )
    return best[1]


def load_image_datasets(
    data_dir: Path,
    *,
    image_size: int = 128,
    batch_size: int = 16,
    validation_split: float = 0.2,
    seed: int = 123,
) -> tuple[Any, Any, list[str]]:
    """Build training and validation ``tf.data`` pipelines from a class-folder tree."""
    tf = require_tensorflow()
    shared = {
        "validation_split": validation_split,
        "seed": seed,
        "label_mode": "categorical",
        "image_size": (image_size, image_size),
        "batch_size": batch_size,
    }
    train = tf.keras.utils.image_dataset_from_directory(data_dir, subset="training", **shared)
    validation = tf.keras.utils.image_dataset_from_directory(
        data_dir, subset="validation", **shared
    )
    class_names = list(train.class_names)

    autotune = tf.data.AUTOTUNE
    return train.prefetch(autotune), validation.prefetch(autotune), class_names


def build_cnn(
    input_shape: tuple[int, int, int], num_classes: int, *, dense_units: int = 256
) -> Any:
    """A small three-block convolutional network with rescaling baked in.

    Keeping ``Rescaling`` inside the model means the saved ``.keras`` file takes
    raw 0-255 pixels, so inference code cannot forget to normalise the way it did
    during training.
    """
    tf = require_tensorflow()
    layers = tf.keras.layers
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(dense_units, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )


def build_transfer_model(
    input_shape: tuple[int, int, int], num_classes: int, *, dense_units: int = 128
) -> Any:
    """MobileNetV2 with frozen ImageNet weights and a fresh classification head."""
    tf = require_tensorflow()
    layers = tf.keras.layers
    base = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=input_shape
    )
    base.trainable = False
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            layers.Rescaling(1.0 / 127.5, offset=-1.0),
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(dense_units, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )


def build_model(spec: ImageDatasetSpec, num_classes: int) -> Any:
    """Build the architecture a dataset spec asks for."""
    shape = (spec.image_size, spec.image_size, 3)
    if spec.architecture == "transfer":
        return build_transfer_model(shape, num_classes, dense_units=spec.dense_units)
    if spec.architecture == "cnn":
        return build_cnn(shape, num_classes, dense_units=spec.dense_units)
    raise ValueError(f"unknown architecture {spec.architecture!r}; expected 'cnn' or 'transfer'")


def train_image_model(
    model: Any,
    train_data: Any,
    validation_data: Any,
    *,
    epochs: int = 15,
    learning_rate: float = 1e-3,
    patience: int = 3,
) -> Any:
    """Compile and fit an image classifier with early stopping on validation loss."""
    tf = require_tensorflow()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )
    ]
    return model.fit(
        train_data, validation_data=validation_data, epochs=epochs, callbacks=callbacks, verbose=2
    )


def collect_predictions(model: Any, dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return (true labels, predicted labels) as integer class indices.

    The notebook version iterated a Keras generator and guessed when it had
    wrapped around; a ``tf.data`` dataset is finite, so a plain loop terminates
    on its own and cannot double-count or truncate the validation set.
    """
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for images, labels in dataset:
        batch = model.predict(images, verbose=0)
        label_array = np.asarray(labels)
        truths.append(np.argmax(label_array, axis=1) if label_array.ndim > 1 else label_array)
        predictions.append(np.argmax(batch, axis=1))
    return np.concatenate(truths), np.concatenate(predictions)


def history_frame(history: Any) -> pd.DataFrame:
    """Return a Keras history as a tidy DataFrame with an ``epoch`` column."""
    frame = pd.DataFrame(history.history)
    return frame.assign(epoch=range(1, len(frame) + 1))


def _contains_images(directory: Path) -> bool:
    """True when a directory holds at least one image file directly."""
    return any(
        child.suffix.lower() in IMAGE_SUFFIXES for child in directory.iterdir() if child.is_file()
    )
