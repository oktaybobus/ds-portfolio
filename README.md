# ds-portfolio

Production-shaped machine-learning projects distilled from a 15-week AI/ML curriculum.

The course produced 102 distinct Jupyter notebooks and about 17,700 lines of
code: a good learning record, and unusable as software. The same twenty lines of
exploratory analysis, the same train/test/score block and the same 150-line
model sweep were re-pasted notebook after notebook, none of it importable,
testable, or runnable outside a Colab session.

This repository is the other half of that work. The recurring logic became one
tested package, `dsjourney`; each project became a thin declarative layer on top
of it - a config, a feature builder, a training entry point and a demo app.

```bash
uv sync --all-extras
uv run python scripts/fetch_assets.py --all
uv run dsj list
uv run dsj train laptop_price
```

## Projects

| Project | Task | Model | Headline | Source notebook |
|---|---|---|---|---|
| [`laptop_price`](projects/laptop_price) | Regression | CatBoost | **R² 0.895** | Laptop Price Prediction with ML |
| [`loan_default`](projects/loan_default) | Classification | Random forest | **Recall 0.737** | Loan Prediction - Classification |
| [`customer_segments`](projects/customer_segments) | Clustering | KMeans, k=4 | **Silhouette 0.337** | Customer Segmentation |
| [`review_sentiment`](projects/review_sentiment) | Text classification | TF-IDF → logistic | **F1 0.957** | NLP Class & Sentiment Analysis |
| [`image_classifiers`](projects/image_classifiers) | Image classification | CNN / MobileNetV2 | 7 datasets | CNN Model Training |

Full metric tables in [RESULTS.md](RESULTS.md), regenerated from
`artifacts/*/metrics.json` rather than typed by hand.

## How it fits together

```
src/dsjourney/          the reusable half - tested toolkit
  config.py             Pydantic-validated project configuration
  datasets.py           one loader for csv / excel / sqlite sources
  eda.py                column overview, missing report, correlation filter
  preprocess.py         pure feature transforms - nothing mutates in place
  benchmark.py          estimator registry + one-call model sweep
  evaluate.py           regression / classification / clustering metrics
  text.py               dependency-light NLP cleaning and TF-IDF pipelines
  vision.py             CNN and transfer-learning builders (TensorFlow, lazy)
  viz.py                figures returned, never shown - headless by default
  artifacts.py          model + scaler + columns + metrics saved as one bundle
  training.py           split, fit, score, persist - shared by every project
  cli.py                the `dsj` command

projects/<name>/        config.yaml, pipeline.py, train.py, app.py, READMEs
tests/                  174 tests: unit for the toolkit, smoke per project
scripts/                asset fetching and RESULTS.md generation
```

A project never re-implements a split, a metric, or a plot.
`projects/laptop_price/pipeline.py` is ~200 lines of genuinely project-specific
feature engineering; everything else it needs is a function call.

## The `dsj` command

| Command | What it does |
|---|---|
| `dsj list` | Every project, its task, and its headline metric |
| `dsj info <project>` | Validated config plus saved metrics |
| `dsj eda-report <project>` | Column overview and missing-value report |
| `dsj train <project>` | Train, score and save the model bundle |
| `dsj train <project> --benchmark` | Sweep every registered estimator, keep the winner |
| `dsj benchmark <project>` | Comparison table without saving anything |
| `dsj predict <project> --json '{...}'` | Score one record against the saved model |
| `dsj serve <project>` | Launch the project's Streamlit demo |

## What changed on the way out of the notebooks

Eight defects were found and fixed while porting. None of them raised an error;
all of them silently changed results. The full list with reproductions is in
[docs/tr/tekrar-eden-hatalar.md](docs/tr/tekrar-eden-hatalar.md); the four that
mattered most:

**Scaler leakage.** `StandardScaler` was fitted on the full frame before
splitting, leaking test statistics into training.
`preprocess.split_and_scale` fits on the training half only, and a test asserts
the test half's mean is *not* centred on zero.

**Unscaled inference input.** Models trained on standardised features were
called with raw values at prediction time. On `laptop_price` this over-estimated
a 16 GB gaming laptop at 208,153 against a real median of 102,777; with the
scaler applied the estimate is 96,239. `ModelBundle.prepare` now replays both
the column order and the scaler.

**Unix timestamps parsed without `unit="s"`.** Every order date in the
segmentation project collapsed onto 1970-01-01, so Recency was ~0 for every
customer and the "RFM" clustering was really an FM clustering.

**Accuracy on imbalanced data.** Loan defaults are 27% of the rows; predicting
"never defaults" scores 73%. The model is now selected and reported on recall.

One data-quality finding is worth its own line: all 16,611 duplicate rows in the
loan dataset are charged-off loans, so the raw file overstates the default rate
by 4.7 points.

## Docker

The image ships the CLI, the tabular projects and the Streamlit apps, with
`dsj` as its entrypoint - so the arguments you pass are the subcommand:

```bash
docker build -t ds-portfolio .
docker run --rm ds-portfolio list
docker run --rm -v "$PWD/artifacts:/app/artifacts" ds-portfolio train laptop_price
```

TensorFlow and XGBoost are left out on purpose: the first would add ~2 GB for
projects whose data is downloaded at training time anyway, the second pulls
~326 MB of CUDA runtime a CPU-only image never touches.

## Development

```bash
make install     # uv sync --all-extras
make check       # lint + typecheck + fast tests + format check
make test-fast   # unit tests only - no data or TensorFlow needed
make test        # everything, including real training runs
make train-all   # retrain every tabular project and refresh RESULTS.md
make cov         # coverage report
make docker      # build the runtime image
```

CI runs lint, `ruff format --check`, strict `mypy`, the test suite on Python 3.11
and 3.12, an end-to-end training run from the committed dataset, and a Docker
build. `mypy` is strict over the whole package with no suppressions.

### Data

Only `laptop_data.csv` (178 KB) is committed - it lets CI verify the full path
from CSV to scored model with no network access. Everything else is declared in
[`assets.yaml`](assets.yaml) and fetched by `scripts/fetch_assets.py`, which
resolves each asset in order: already on disk, committed, copied from the
original course tree, then downloaded from the Hugging Face Hub.

> **Status:** the Hub mirror (`OKTAYBBS/ds-portfolio-data`) is declared but not
> populated yet, so the download step is the only one that will not currently
> succeed on a fresh machine. On the author's machine the local-copy step
> handles all four assets.

Image datasets are pulled straight from Kaggle at training time by `kagglehub`.

### Optional extras

| Extra | Adds |
|---|---|
| `boost` | XGBoost, CatBoost, LightGBM |
| `dl` | TensorFlow, for `image_classifiers` |
| `nlp` | NLTK and wordcloud (not required - `dsjourney.text` is self-contained) |
| `app` | Streamlit |
| `hub` | `huggingface-hub`, for fetching datasets |
| `data` | `kagglehub` and `openpyxl` |
| `dev` | pytest, ruff, mypy |

On macOS, XGBoost and LightGBM need `brew install libomp`. Without it the
registry simply offers fewer models - it does not fail.

## Turkish notes

Code, documentation and commits are English. The learning notes stay in Turkish
under [docs/tr/](docs/tr/), and every project carries a `README.tr.md` alongside
its English one.

## License

MIT
