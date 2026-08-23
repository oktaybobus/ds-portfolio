# Univariate Time-Series Forecasting

Forecast two very different series through one code path, and score both against
a baseline that has to be beaten.

| Series | Observations | Holdout | Winner | MAE | vs naive |
|---|---|---|---|---|---|
| New Delhi daily mean temperature | 1,462 | 60 days | Holt-Winters | 2.17 °C | **+44.7%** |
| Adidas quarterly revenue | 88 | 8 quarters | *naive* | 879 M EUR | **0.0%** |

```bash
uv run python projects/series_forecast/train.py --all
uv run python projects/series_forecast/train.py --series adidas_revenue --horizon 12
uv run dsj serve series_forecast
```

## The missing step

Both source notebooks fitted a model to the **entire** series and then predicted
past its end:

```python
model = sm.tsa.statespace.SARIMAX(df["Revenue"])
result = model.fit()
predictions = result.predict(len(df), len(df) + 7)
```

There is nothing to compare those predictions against. The plot looks
convincing because a forecast drawn beyond the last observation has nothing
beside it to disagree with. `chronological_split` withholds the final periods
instead, so the forecast lands on top of data the model never saw.

Never reach for `train_test_split` here: it shuffles, and a shuffled split lets
a model learn from next month to predict last month.

## What the holdout revealed

**On Delhi temperature, seasonality is nearly everything.** Seasonal strength
measures 0.945 against a trend strength of 0.174. Holt-Winters, which models a
yearly cycle explicitly, cuts the naive error by 45%. SARIMA with the order the
notebook used is *worse than naive* (-25%).

**On Adidas revenue, nothing beats doing nothing.**

| Method | MAE | vs naive |
|---|---|---|
| **naive** (repeat last quarter) | **879** | **0.0%** |
| seasonal naive | 951 | −8.2% |
| Holt-Winters | 1,408 | −60.1% |
| SARIMA | 1,614 | −83.5% |

The notebook's SARIMAX is 84% worse than repeating the last quarter's number.
With 79 training quarters and a seasonal order to estimate, the model has too
many parameters and too little history; it fits the past and extrapolates
confidently in the wrong direction. That result is invisible without a holdout -
and it is the more useful of the two findings, because it says "do not deploy
this."

## Reading the metrics

- **`skill_vs_naive`** is the number to read: `1 - MAE / naive MAE` over the same
  horizon. Positive beats doing nothing, negative loses to it.
- **`mase`** scales MAE by the average one-step change, which makes it
  comparable across series with different units. For a multi-step forecast a
  value above 1 is normal - it is not a pass/fail line.
- **Trend and seasonal strength** come from an additive decomposition and say
  which methods are even worth trying.

## Adding a series

Add a `SeriesSpec` to `SERIES` in `pipeline.py`: file name, date column, value
column, season length and horizon. Quarter labels like `"2000Q1"` are handled by
`quarterly_labels=True` - `pd.to_datetime` cannot parse them, so they go through
a `PeriodIndex` first.

Türkçe açıklamalar: [README.tr.md](README.tr.md)
