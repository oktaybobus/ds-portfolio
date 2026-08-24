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
| [`laptop_price`](projects/laptop_price) | Regression | CatBoost | **R² 0.805** | Laptop Price Prediction with ML |
| [`istanbul_housing`](projects/istanbul_housing) | Regression | CatBoost | **R² 0.814** | Regression Final Project |
| [`loan_default`](projects/loan_default) | Classification | Random forest | **Recall 0.737** | Loan Prediction - Classification |
| [`customer_segments`](projects/customer_segments) | Clustering | KMeans, k=4 | **Silhouette 0.337** | Customer Segmentation |
| [`review_sentiment`](projects/review_sentiment) | Text classification | TF-IDF → logistic | **F1 0.957** | NLP Class & Sentiment Analysis |
| [`series_forecast`](projects/series_forecast) | Forecasting | Holt-Winters / naive | **+44.7% vs naive** | Prophet & Time-Series Analysis |
| [`movie_recommender`](projects/movie_recommender) | Recommendation | Truncated SVD | **Precision@10 0.018** | RS KNN / SK22 / MatrixFactorization |
| [`bart_ridership`](projects/bart_ridership) | Regression (spatio-temporal) | HistGradientBoosting | **R² 0.818** | BART Analysis |
| [`article_search`](projects/article_search) | Retrieval | TF-IDF + SVD | **MRR 0.569** | AI Agents |
| [`object_detection`](projects/object_detection) | Detection | Haar cascade / YOLO | **6 of 7 faces** | Computer Vision |
| [`image_classifiers`](projects/image_classifiers) | Image classification | CNN / MobileNetV2 | 7 datasets | CNN Model Training |
| [`marvel_network`](projects/marvel_network) | Graph (PySpark) | Distributed BFS | **99.4% within 3 hops** | Big Data Hadoop Spark |
| [`diabetes_screening`](projects/diabetes_screening) | Classification (PySpark) | MLlib logistic | **Recall 0.531** | Big Data Hadoop Spark |

Every trained project is also reachable over HTTP - see [service/](service/).

Regression headlines are on the original price scale, not the log-transformed
target the models are fitted on - both are reported in [RESULTS.md](RESULTS.md).

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
  detection.py          bounding boxes, IoU, NMS, cascades - figures not windows
  evaluate.py           regression / classification / clustering metrics
  forecasting.py        chronological splits, forecasters, skill against naive
  recommend.py          rating splits, similarity, SVD, precision@k
  retrieval.py          chunking, LSA index, recall@k and context cost
  text.py               dependency-light NLP cleaning and TF-IDF pipelines
  vision.py             CNN and transfer-learning builders (TensorFlow, lazy)
  viz.py                figures returned, never shown - headless by default
  artifacts.py          model + scaler + columns + metrics saved as one bundle
  serving.py            which projects can score records, and how
  training.py           split, fit, score, persist - shared by every project
  cli.py                the `dsj` command

service/                FastAPI app serving every trained bundle
projects/<name>/        config.yaml, pipeline.py, train.py, app.py, READMEs
tests/                  362 tests: unit for the toolkit, smoke per project
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
| `dsj api` | Serve every trained project over HTTP, docs at `/docs` |

## What changed on the way out of the notebooks

Twenty-three defects were found and fixed while porting. None of them raised an
error; all of them silently changed results. The full list with reproductions is in
[docs/tr/tekrar-eden-hatalar.md](docs/tr/tekrar-eden-hatalar.md); the ones that
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

**No holdout at all.** Both forecasting notebooks fitted a model to the entire
series and predicted past its end, which produces a plot with nothing to
disagree with it. Scored on a chronological holdout, the SARIMAX in one of them
turns out to be 84% *worse* than repeating last quarter's number.

**Recommendations nobody measured.** The recommender notebooks produced
similarity lists and stopped. Adding a held-out set exposed a second problem:
without a minimum-support floor, precision@10 measured 0.0005 - below what
random ranking achieves - because films rated 5 by one person outranked
everything for everyone.

**Evaluation that measured the wrong thing.** Scoring the article index on
document recall alone recommends ever-larger chunks - in the limit, one vector
per document, which is what the notebook did. Adding a context-cost metric
inverts the answer: 120-word chunks retrieve eight times more efficiently per
word of context. Neither number is wrong; reporting only one is.

**A tutorial default that detects nothing.** The face cascade's `scaleFactor`
was left at the value the notebook used. Swept against a photograph with seven
faces, 1.05 finds six and 1.30 finds zero.

**A text model that answered the same thing every time.** Building the API
surfaced a defect that had been shipping quietly in the CLI: a TF-IDF pipeline
handed a one-column DataFrame iterates its *column names*, so every review was
scored as the literal string `"text"`. A glowing review and a scathing one both
came back positive at 0.696, and nothing raised.

**A metric reported under the wrong name.** The Spark notebook scored its
classifier with a `BinaryClassificationEvaluator`, left `metricName` at its
default of `areaUnderROC`, and printed the result as `Accuracy: 0.854`. Nothing
was broken - the label was. Real accuracy on the same predictions is 0.745,
against 0.649 for always answering "not diabetic", so the mislabel triples the
apparent margin over doing nothing.

**Text read as UTF-8 whatever it actually was.** `Marvel-names.txt` is Latin-1
and `book.txt` is cp1252. Spark's text reader assumes UTF-8 and substitutes
U+FFFD for anything that fails - 269 lines in `book.txt` - so every curly
apostrophe vanishes and `don't` is counted as two words. Its `read.text` has no
encoding option at all; `dsjourney.spark.read_text_lines` routes through the one
reader that does.

Two data-quality findings are worth their own lines. All 16,611 duplicate rows
in the loan dataset are charged-off loans, so the raw file overstates the
default rate by 4.7 points. And the Istanbul housing notebook's building-age
map omitted four labels, so 3,264 listings - 30% of the file, and the whole
new-build segment - became NaN and were dropped before training.

## Docker

The image ships the CLI, the tabular projects and the Streamlit apps, with
`dsj` as its entrypoint - so the arguments you pass are the subcommand:

```bash
docker build -t ds-portfolio .
docker run --rm ds-portfolio list
docker run --rm -p 8000:8000 -v "$PWD/artifacts:/app/artifacts" \
  ds-portfolio api --host 0.0.0.0
docker run --rm -v "$PWD/artifacts:/app/artifacts" ds-portfolio train laptop_price
docker run --rm -v "$PWD/artifacts:/app/artifacts" ds-portfolio predict laptop_price \
  --json '{"company":"Dell","type_name":"Gaming","ram_gb":16,"ssd_gb":512}'
```

Mount the volume: with `--rm` and no mount, the model a training run writes is
discarded when the container exits, and the next `docker run` has nothing to
score against.

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

Only `laptop_data.csv` (178 KB), the two forecasting series (77 KB), the 390-article
corpus (9.4 MB) and the detection samples (2.7 MB) are committed - it lets CI verify the full path
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
| `boost` | XGBoost, CatBoost and LightGBM (also available singly) |
| `dl` | TensorFlow, for `image_classifiers` |
| `nlp` | NLTK and wordcloud (not required - `dsjourney.text` is self-contained) |
| `app` | Streamlit |
| `api` | FastAPI and uvicorn, for `dsj api` |
| `detect` | Headless OpenCV, for Haar cascade detection |
| `yolo` | Ultralytics, for YOLO object detection |
| `hub` | `huggingface-hub`, for fetching datasets |
| `data` | `kagglehub` and `openpyxl` |
| `spark` | PySpark, for `marvel_network` and `diabetes_screening` |
| `dev` | pytest, ruff, mypy |

On macOS, XGBoost and LightGBM need `brew install libomp`. Without it the
registry simply offers fewer models - it does not fail.

PySpark needs a JVM as well as the extra: `brew install openjdk@17`, or
`apt install openjdk-17-jdk`. Homebrew keeps versioned JDKs off `PATH`, so
`dsjourney.spark.java_home()` looks in the usual places and sets `JAVA_HOME`
for the child process - no shell configuration required. Without a JVM the
Spark tests skip themselves and the two Spark projects print the install
command instead of failing.

## Turkish notes

Code, documentation and commits are English. The learning notes stay in Turkish
under [docs/tr/](docs/tr/), and every project carries a `README.tr.md` alongside
its English one.

## License

MIT
