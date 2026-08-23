# Istanbul Apartment Price Prediction

Predict the asking price of an Istanbul apartment from 10,735 listings scraped
from Emlakjet.

| | |
|---|---|
| Task | Regression |
| Data | 10,599 usable listings (98.7% of the scrape), 169 features |
| Model | CatBoost, 1,200 iterations, depth 10 |
| **R² (price scale)** | **0.814** |
| MAE / RMSE | 1.89 / 3.22 million TL |
| R² (log scale) | 0.849 |
| Source | `AOB_Regression_Final_Project.ipynb` |

```bash
uv run python scripts/fetch_assets.py --project istanbul_housing
uv run dsj train istanbul_housing
uv run dsj train istanbul_housing --benchmark
uv run dsj serve istanbul_housing
```

## Everything is free text

The scrape returns what a human would read, not what a model can use:

| Raw | Example | Parsed into |
|---|---|---|
| `Oda Sayısı` | `4.5+1` | `toplam_oda` = 5.5 |
| `Brüt` / `Net` | `292 m²` | `brut_m2` = 292.0 |
| `Bina Yaşı` | `21 Ve Üzeri` | `bina_yasi` = 25.0 |

Column names arrive with Turkish characters and spaces, so `normalise_columns`
folds them to ASCII snake_case once at the door rather than leaving every
downstream reference to spell `Oda Sayısı` correctly.

## The building-age map that dropped 30% of the data

The source notebook mapped age bands to numbers with this dictionary:

```python
{'0': 0, '1': 1, ..., '6-10': 8, '11-15': 13, '16-20': 18,
 '21 Ve Üzeri': 25, '26-30': 28}
```

The file actually contains four more labels:

| Label | Listings |
|---|---|
| `0 (Oturuma Hazır)` | 2,680 |
| `31 Ve Üzeri` | 327 |
| `0 (Yapım Aşamasında)` | 158 |
| `21-25` | 141 |

Unmapped values became `NaN`, the derived age feature became `NaN` with them,
and the final `dropna()` silently discarded **3,264 rows - 30% of the file, and
the entire new-build segment**. New builds are not a random sample: their median
price is 8.8 M TL against 7.3 M overall, so the model was trained on a
systematically cheaper slice of the market than it would be asked to price.

Completing the map lifts retention from 68% to 98.7%.

## Neighbourhood granularity is a modelling decision

Location dominates price in this city, so how aggressively rare neighbourhoods
are collapsed into `Others` matters more than it looks:

| `min_count` | Feature columns | R² (price scale) | MAE (M TL) |
|---|---|---|---|
| 25 | 168 | 0.794 | 1.96 |
| 10 | 296 | 0.807 | 1.91 |
| **5** | **379** | **0.814** | **1.89** |
| 1 | 494 | 0.811 | 1.90 |

Five is the peak: it keeps almost all the location signal while still folding
away singleton neighbourhoods that only add noise columns.

## Against the notebook

The notebook reported R² 0.8292 with a hyperparameter-searched CatBoost. That
number is on the original price scale, so it is not comparable to a log-scale R²
- which is why `train_supervised` now reports both, and why the headline here is
the price-scale figure.

Run on the notebook's own 7,335-row subset with the settings in this repo, the
result is R² 0.7997 (MAE 2.02 M TL). On the full 10,599 rows it is **0.8138**
(MAE 1.89 M TL) - so recovering the new-build segment does not just add rows, it
makes the model better: +0.014 R² and 6.5% less error per listing. Letting the
model see that age-0 buildings are priced on their own terms helps it price
everything else too.

The remaining gap to the notebook's 0.8292 is most likely split variance plus
its `RandomizedSearchCV` having been tuned on this same data. It is not claimed
here as a win; the honest summary is comparable accuracy over 44% more listings.

### A note on how this section was written

The building-age fix was applied twice. The first attempt silently failed - the
map was never actually extended - and retention still read 98.7%, because
median imputation was quietly filling the unmapped rows with an age of 8. The
model was training on 2,838 new-build listings labelled as eight years old, and
every metric looked plausible. A test that asserts *every label in the file is
covered by the map*, rather than a metric that looks reasonable, is what caught
it.

## Cleaning

- Prices are converted to millions of lira and clipped to the 0.1-50 M range;
  above that the listings are penthouses and mislisted land plots whose prices
  are an order of magnitude out, and a model that fits them fits nothing else.
- Identifiers, listing titles and scraper bookkeeping columns are dropped;
  `Isıtma`, `Yapı Durumu`, `Kullanım Durumu`, `Krediye Uygunluk` and
  `Tapu Durumu` are over 90% missing.
- Imputation happens **before** the derived ratios, not after: a ratio built
  from a missing input is missing too, and the final `dropna()` would then
  discard the row anyway. This ordering is what recovered the last of the 30%.

Türkçe açıklamalar: [README.tr.md](README.tr.md)
