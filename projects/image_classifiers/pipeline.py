"""The image-classification suite.

Unlike the tabular projects this one trains several models, one per dataset, so
it exposes a catalogue rather than a single ``build_features``. The
:class:`~dsjourney.vision.ImageDatasetSpec` for each dataset records the Kaggle
handle, input resolution and architecture; everything else is shared.

Dataset roots are discovered rather than hard-coded: the source notebook had a
different ``os.path.join`` for every archive, plus a ``next(os.walk(...))``
search for the rice one. ``vision.find_image_root`` replaces all of that.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from dsjourney import vision
from dsjourney.config import load_project_config

CONFIG = load_project_config("image_classifiers")

DATASETS: dict[str, vision.ImageDatasetSpec] = {
    "grape": vision.ImageDatasetSpec(
        key="grape",
        kaggle_handle="rm1000/augmented-grape-disease-detection-dataset",
        title="Grape leaf disease",
        image_size=128,
        architecture="cnn",
        dense_units=256,
    ),
    "rice": vision.ImageDatasetSpec(
        key="rice",
        kaggle_handle="muratkokludataset/rice-image-dataset",
        title="Rice variety",
        image_size=128,
        architecture="cnn",
        dense_units=256,
    ),
    "fruits_veg": vision.ImageDatasetSpec(
        key="fruits_veg",
        kaggle_handle="kritikseth/fruit-and-vegetable-image-recognition",
        title="Fruit and vegetable",
        image_size=128,
        architecture="cnn",
        dense_units=256,
    ),
    "fish": vision.ImageDatasetSpec(
        key="fish",
        kaggle_handle="crowww/a-large-scale-fish-dataset",
        title="Fish species",
        image_size=170,
        architecture="transfer",
        dense_units=128,
    ),
    "tomato": vision.ImageDatasetSpec(
        key="tomato",
        kaggle_handle="kaustubhb999/tomatoleaf",
        title="Tomato leaf disease",
        image_size=128,
        architecture="cnn",
        dense_units=256,
    ),
    "animal": vision.ImageDatasetSpec(
        key="animal",
        kaggle_handle="emirhanai/animal-computer-vision-clean-dataset-code-cnnai",
        title="Animal species",
        image_size=128,
        architecture="transfer",
        dense_units=128,
    ),
    "brain": vision.ImageDatasetSpec(
        key="brain",
        kaggle_handle="sartajbhuvaji/brain-tumor-classification-mri",
        title="Brain MRI tumour type",
        image_size=128,
        architecture="cnn",
        dense_units=128,
    ),
}


def load_raw() -> pd.DataFrame:
    """Return the dataset catalogue.

    Image data is never loaded into a DataFrame; this exists so ``dsj info`` and
    the CLI's generic commands have something meaningful to show.
    """
    return pd.DataFrame(
        [
            {
                "dataset": spec.key,
                "title": spec.title,
                "kaggle_handle": spec.kaggle_handle,
                "image_size": spec.image_size,
                "architecture": spec.architecture,
            }
            for spec in DATASETS.values()
        ]
    )


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Not applicable: image pipelines stream from disk rather than a frame."""
    return frame


def prepare_input(payload: dict[str, Any]) -> pd.DataFrame:
    """Not applicable: use ``predict.py --image <path>`` to score an image."""
    raise NotImplementedError(
        "image_classifiers scores image files, not records. "
        "Use: python projects/image_classifiers/predict.py --dataset <key> --image <path>"
    )


def spec_for(dataset: str) -> vision.ImageDatasetSpec:
    """Look up a dataset spec by key."""
    try:
        return DATASETS[dataset]
    except KeyError as error:
        raise KeyError(f"unknown dataset {dataset!r}; available: {sorted(DATASETS)}") from error
