# Laptop Price Prediction

Predict the retail price of a laptop from its hardware configuration.

| | |
|---|---|
| Task | Regression |
| Data | 1,303 retail listings, 12 raw columns |
| Model | CatBoostRegressor over 38 engineered features |
| **R²** | **0.895** |
| RMSE / MAE | 0.195 / 0.138 (log scale) |
| MAPE | 1.27% |
| Source | `14- AOB-Laptop Price Prediction with ML.ipynb` |

```bash
uv run dsj train laptop_price
uv run dsj train laptop_price --benchmark   # sweep 15 estimators
uv run dsj serve laptop_price               # Streamlit demo
```

## Where the signal actually is

Three of the twelve raw columns are packed records rather than values, and most
of the predictive power is locked inside them:

| Raw column | Example | Extracted features |
|---|---|---|
| `ScreenResolution` | `IPS Panel Retina Display 2560x1600` | `touchscreen`, `ips`, `ppi` |
| `Memory` | `128GB SSD + 1TB HDD` | `ssd_gb`, `hdd_gb` |
| `Cpu` | `Intel Core i5 2.3GHz` | `cpu_brand`, `cpu_ghz`, `cpu_generation` |

Two derived features carry more than their inputs do separately:

- **`ppi`** — pixel density, `sqrt(width² + height²) / inches`. It folds
  resolution and physical size into the one number a buyer actually pays for;
  `inches` and the raw resolution are dropped once it exists.
- **`cpu_performance`** — clock speed × generation. A 2.3 GHz 8th-generation
  chip is not a 2.3 GHz 3rd-generation chip, and the interaction says so.

The target is `log1p(price)`: prices span 9k to 325k and the tail would
otherwise dominate the loss. `postprocess` converts predictions back.

## Model selection

`--benchmark` fits every registered regressor on the same split:

| Model | R² | RMSE |
|---|---|---|
| **CatBoost** | **0.901** | 0.189 |
| SVR | 0.884 | 0.205 |
| HistGradientBoosting | 0.880 | 0.208 |
| GradientBoosting | 0.878 | 0.210 |
| RandomForest | 0.872 | 0.215 |
| LinearRegression *(notebook's choice)* | 0.845 | 0.237 |

CatBoost is what the config now names. The gain over the notebook's linear model
is 0.056 R², most of it from the non-linear interaction between CPU tier, GPU
tier and RAM that a linear model cannot express.

## Notes

- `OpSys` is dropped rather than encoded: in this catalogue it is nearly
  collinear with brand (every macOS row is an Apple row).
- Brands with fewer than 15 listings are collapsed into `Others`, which keeps
  one-hot encoding from adding a dozen near-empty columns.
- Prices are in Indian Rupees, as in the source dataset.

Türkçe açıklamalar: [README.tr.md](README.tr.md)
