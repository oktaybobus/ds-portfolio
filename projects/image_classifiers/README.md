# Image Classification Suite

Seven image classifiers over public Kaggle datasets, all built from the same
two architectures and one training loop in `dsjourney.vision`.

| | |
|---|---|
| Task | Multi-class image classification |
| Datasets | 7 declared, **3 trained here**, downloaded on demand via `kagglehub` |
| Architectures | Scratch CNN (3 conv blocks) and MobileNetV2 transfer learning |
| **Best** | **Animal species, accuracy 0.971** (MobileNetV2) |
| Source | `HW19_AOB_CNN_ModelsTraining.ipynb` |

```bash
uv sync --extra dl --extra data
uv run python projects/image_classifiers/train.py --dataset grape
uv run python projects/image_classifiers/train.py --all --epochs 5
uv run python projects/image_classifiers/predict.py --dataset grape --image leaf.jpg
```

Requires the optional deep-learning extra and network access to Kaggle. The
datasets are tens of gigabytes in total and are never mirrored into this repo.

## The catalogue

| Key | Dataset | Input | Architecture | Classes | Accuracy | F1 |
|---|---|---|---|---|---|---|
| `grape` | Augmented grape leaf disease | 128² | CNN | - | not trained | - |
| `rice` | Rice variety | 128² | CNN | - | not trained | - |
| `fruits_veg` | Fruit and vegetable recognition | 128² | CNN | - | not trained | - |
| `fish` | Large-scale fish species | 170² | MobileNetV2 | - | not trained | - |
| **`tomato`** | Tomato leaf disease | 128² | CNN | 10 | **0.902** | 0.902 |
| **`animal`** | Animal species | 128² | MobileNetV2 | 4 | **0.971** | 0.971 |
| **`brain`** | Brain MRI tumour type | 128² | CNN | 4 | **0.895** | 0.893 |

Small, visually distinctive datasets get a scratch CNN; the harder ones get
frozen ImageNet features and a fresh head.

The four untrained rows each need a multi-gigabyte Kaggle archive that is not
on this machine; the three that are trained were already in the `kagglehub`
cache. Nothing about them is special - `train.py --dataset <key>` trains any of
the seven, and the row fills itself in from `metrics.json`.

### What the three runs show

`animal` is the easiest of the three and gets the best score: four visually
unmistakable species (buffalo, elephant, rhino, zebra) and transfer learning
from ImageNet, which has seen all four. It converged in 7 epochs.

`tomato` is the hardest brief - ten classes, most of them a green leaf with
slightly different blotches - and a scratch CNN still reaches 0.902 over 11,000
images.

`brain` is the one to read carefully. 0.895 accuracy across four MRI classes
sounds strong, and for a coursework CNN on 3,264 images it is. It is also not a
medical claim: the split is random rather than by patient, so slices from one
scan can land on both sides of it, which inflates the number by an amount this
project does not measure. The confusion matrix in
`artifacts/image_classifiers/brain/confusion_matrix.png` is the honest view of
where it fails.

Each run writes `artifacts/image_classifiers/<key>/` containing `model.keras`,
`labels.json`, `metrics.json`, `metadata.json`, `history.csv` and
`confusion_matrix.png`. The `.keras` files are gitignored - a trained model is
40 MB and reproducible from the command above.

## Three fixes carried over from the notebook

**Dataset roots are discovered, not hard-coded.** Every Kaggle archive nests its
class folders differently - one level down, behind `Final Training Data`, or
split into train/test. The notebook had a bespoke `os.path.join` per dataset
plus a `next(os.walk(...))` search that hard-coded the five rice class names.
`vision.find_image_root` walks the tree and picks the directory with the most
image-bearing subdirectories, so a new dataset needs no new code path.

**Validation predictions are collected correctly.** The notebook iterated a
Keras generator and broke out when `batch_index` wrapped back to 0. That is
fragile: it silently truncates or double-counts depending on where the generator
happened to be. A finite `tf.data` dataset ends on its own, so the loop in
`vision.collect_predictions` is a plain `for`.

**Rescaling lives inside the model.** Both builders start with a `Rescaling`
layer, so the saved `.keras` file accepts raw 0-255 pixels. Inference code
cannot forget to normalise the way training did - a mismatch that produces
confident nonsense rather than an error.

## Not ported

The notebook's eighth model (dental radiography) is driven by a CSV of
bounding-box annotations rather than a class-folder tree, so it needs a
different loader. It is out of scope for this suite.

Türkçe açıklamalar: [README.tr.md](README.tr.md)
