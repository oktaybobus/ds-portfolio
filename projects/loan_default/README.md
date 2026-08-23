# Loan Default Prediction

Predict whether a consumer loan will be charged off.

| | |
|---|---|
| Task | Binary classification (imbalanced) |
| Data | 240,373 applications after de-duplication, 26.7% default rate |
| Model | RandomForestClassifier, `class_weight="balanced"` |
| **Recall** | **0.737** |
| ROC AUC | 0.775 |
| Precision / F1 | 0.441 / 0.552 |
| Accuracy | 0.680 |
| Source | `HW11-AOB- 3-Loan prediction-Classification.ipynb` |

```bash
uv run python scripts/fetch_assets.py --project loan_default
uv run dsj train loan_default
uv run dsj train loan_default --benchmark
```

## Read the metrics in the right order

Accuracy is the least useful number on this page. Predicting "never defaults"
for every application scores 73% accuracy and is worth nothing to a lender.

The model here is deliberately tuned the other way: it catches **74% of the
loans that actually default**, at the cost of flagging some good applicants
(precision 0.44). Whether that trade is right depends on the cost of a default
versus the cost of a lost customer - which is a business decision, so
`class_weight` is a config value, not a constant.

## Two departures from the source notebook

**The target is inverted.** The notebook predicted `Fully Paid = 1`, making the
majority class positive. Every metric then flatters the model and recall answers
the wrong question. Here `charged_off = 1`, so recall means "of the bad loans,
how many did we catch?"

**Missingness is kept as a feature.** The gaps in this data are structural, not
random:

| Column | Missing | Why |
|---|---|---|
| `months_since_delinquent` | 55% | The applicant has never been delinquent |
| `credit_score` | 24% | Thin credit file |
| `annual_income` | 24% | Not supplied |

Each imputed column keeps a `*_missing` flag, so the model can learn that a
thin file is itself a risk signal. The source notebook used `miceforest`, which
fills the gap and discards the fact that it was there.

## Cleaning and features

- **Credit-score scale.** 16,187 rows record the score ten times too large
  (7400 instead of 740). Anything above 850 is divided by ten rather than
  dropped.
- **Currency columns.** `monthly_debt` and `max_open_credit` arrive as strings
  with symbols and separators; they are stripped and coerced, with unparseable
  values becoming NaN for the imputer to handle.
- **16,611 duplicate rows removed - every one of them a default.** This is not
  a rounding detail: the raw file reports a 31.4% default rate, and the
  de-duplicated data 26.7%. The duplication is entirely one-sided, so anyone
  quoting a default rate from the raw CSV is 4.7 points high. The source
  notebook also called `drop_duplicates()`, but reported the raw rate. A test
  (`test_every_duplicate_row_is_a_default`) locks the finding down.
- **Derived ratios**: `credit_utilisation` (balance / limit) and
  `debt_to_income` (monthly debt / annual income), both treating a zero
  denominator as missing rather than infinite.

Türkçe açıklamalar: [README.tr.md](README.tr.md)
