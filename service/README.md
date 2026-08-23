# Model service

A REST API over every trained project in this repository.

```bash
uv sync --extra api
uv run dsj api                    # http://127.0.0.1:8000/docs
```

| Route | What it does |
|---|---|
| `GET /health` | Liveness, and how many projects are ready to score |
| `GET /projects` | Every project, whether it is servable, and why not when it is not |
| `GET /projects/{name}` | Config, saved metrics, and a **working example record** |
| `POST /projects/{name}/predict` | Score one record |
| `POST /admin/reload` | Drop cached bundles so a retrained model is picked up |

```bash
curl -s localhost:8000/projects/laptop_price | jq .example_input
curl -s -X POST localhost:8000/projects/laptop_price/predict \
  -H 'content-type: application/json' \
  -d '{"record": {"company": "Dell", "type_name": "Gaming", "ram_gb": 16, "ssd_gb": 512}}'
```

## Why it is generic

The MLOps notebook this comes from hard-coded one model, three field names and
one response shape:

```python
model = pickle.load(open("maas.pkl", "rb"))


class PredictionRequest(BaseModel):
    tecrube: float
    yazili: float
    mulakat: float
```

Adding a second model meant copying the file. Here the routes take a project
name and a free-form record, and `dsjourney.serving` resolves the rest from the
saved bundle: the column order, the scaler, the inverse transform for a
log-trained target, and the class labels for probabilities. A project becomes
servable by having a trained bundle and a `prepare_input` - no route to write.

## Honest refusals

Not every project scores a single record, and the API says so with a status
code rather than a stack trace:

| Situation | Status | Body |
|---|---|---|
| Unknown project | 404 | the missing path |
| Forecaster or recommender | 409 | `forecasting projects do not score individual records` |
| Untrained project | 409 | `not trained yet - run: dsj train <name>` |
| Record that cannot be featurised | 422 | what failed to parse |

The task check runs **before** the training check on purpose: telling a caller
that a forecasting project "is not trained yet" would send them to run a command
that still would not make the endpoint work.

## The bug this layer exposed

Building it surfaced a defect in the CLI that had been shipping quietly: a text
model was being handed a one-column DataFrame, and a TF-IDF pipeline iterating a
DataFrame sees its *column names*. Every review was scored as the literal string
`"text"`, so `dsj predict review_sentiment` returned the same answer - positive,
0.696 - for a glowing review and a scathing one alike. Nothing raised.

`ModelBundle.prepare` now returns a Series for text tasks, and
`test_prepare_hands_text_models_a_series` asserts that two opposite reviews get
opposite predictions.

Türkçe açıklamalar: [README.tr.md](README.tr.md)
