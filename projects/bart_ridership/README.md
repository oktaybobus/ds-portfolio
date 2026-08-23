# BART Ridership Demand

Predict hourly passenger throughput between Bay Area Rapid Transit station pairs.

| | |
|---|---|
| Task | Regression (spatio-temporal) |
| Data | 13.3M origin-destination-hour records, 2016-2017, 46 stations |
| Split | **Chronological**: train 2016, test 2017 |
| Model | HistGradientBoosting, 400 iterations |
| **R² (2017)** | **0.818** |
| MAE | 4.50 passengers · 0.354 on the log scale |
| Source | `HW20_AOB_BART_analysis.ipynb` |

```bash
uv sync --extra data
uv run python projects/bart_ridership/train.py
uv run python projects/bart_ridership/train.py --sample 0            # all 13.3M rows
uv run python projects/bart_ridership/train.py --compare-random
```

Data comes from Kaggle (`saulfuh/bart-ridership`, 410 MB) via `kagglehub` at
training time; it is never mirrored into this repo. The default run samples
400,000 rows per year so an experiment takes seconds - `--sample 0` uses
everything.

## Three things the raw columns do not say

**Time is cyclical.** Hour 23 and hour 0 are one hour apart; as integers they
are 23 apart, and a model reads midnight as the far extreme of 11pm. `hour`,
`day_of_week` and `month` are each projected onto a circle as a sine/cosine
pair, so the wrap costs nothing. Tree models can learn the discontinuity from
splits alone, but only by spending depth on it.

**Stations are places, and their coordinates are buried in prose.**
`station_info.csv` stores them inside a free-text `Location` field as
`"-122.271450,37.803768,0"` - longitude first here, but not consistently. Each
number is assigned by which range it falls in rather than by its position.

**Distance is not a coordinate difference.** A degree of longitude is 111 km at
the equator and 0 at the poles, so subtracting latitudes and longitudes gives a
number that distorts with latitude. `distance_km` is the great-circle distance,
which is the single feature that says how far the trip actually is.

## The split, and what measuring it showed

Ridership is time-stamped, so a shuffled split puts the same station pair at the
same hour in the same season on both sides of the boundary. This project trains
on 2016 and tests on 2017 instead.

`--compare-random` scores both on the same rows:

| Split | R² | MAE (log scale) |
|---|---|---|
| Chronological (2016 → 2017) | 0.8182 | 0.354 |
| Random shuffle | 0.8219 | 0.350 |

**The difference is 0.004 - effectively nothing.** That is worth stating plainly
rather than dressing up: on this dataset the shuffled split does not inflate the
score meaningfully, because every feature is structural (which stations, what
hour, how far) and none of them identify an individual row. Ridership patterns
are stable enough year to year that 2017 is not a harder problem than a held-out
slice of 2016.

The chronological split is still the right method, and running it is the only
reason anyone can say the above with a number attached. A leak-free split is
cheap insurance; the point of measuring is that you stop having to guess whether
you needed it.

## Feature set

| Feature | Why |
|---|---|
| `hour_sin`, `hour_cos` | Commute peaks, with midnight adjacent to 11pm |
| `day_of_week_sin/cos`, `is_weekend` | Weekday commuting versus weekend travel |
| `month_sin`, `month_cos` | Seasonal variation |
| `origin_latitude/longitude`, `destination_latitude/longitude` | Where in the network |
| `distance_km` | Great-circle trip length |
| `same_station` | Entry and exit at one station behave differently |

The target is `log1p(throughput)`: counts run from 1 to several hundred and the
busy pairs would otherwise dominate the loss. Metrics are reported on both
scales.

Türkçe açıklamalar: [README.tr.md](README.tr.md)
