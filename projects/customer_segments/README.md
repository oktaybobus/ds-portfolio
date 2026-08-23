# RFM Customer Segmentation

Group e-commerce customers by Recency, Frequency and Monetary value.

| | |
|---|---|
| Task | Clustering |
| Data | 4,194 order lines → 3,054 customers |
| Model | KMeans, k = 4 |
| Silhouette | 0.337 |
| Calinski-Harabasz | 1701.2 |
| Davies-Bouldin | 0.888 |
| Source | `HW12-AOB-customersegment.ipynb` |

```bash
uv run python scripts/fetch_assets.py --project customer_segments
uv run python projects/customer_segments/train.py --max-k 12
```

## A bug worth reading about

`Orders.placed_date` holds Unix timestamps in **seconds**:

```
1426019099  ->  2015-03-10
```

The source notebook parsed it with `pd.to_datetime(series, errors="coerce")` and
no `unit` argument. pandas defaults to **nanoseconds**, so every one of those
integers landed within two seconds of 1970-01-01:

```python
pd.to_datetime(1426019099)  # 1970-01-01 00:00:01.426019099
pd.to_datetime(1426019099, unit="s")  # 2015-03-10 20:24:59
```

`errors="coerce"` meant no exception was raised. Recency was then
`snapshot_date - max(order_date)` over a two-second span, so it was ~0 for every
customer and contributed nothing to the clustering. The notebook's "RFM
segmentation" was in practice an FM segmentation.

Parsing with `unit="s"` restores dates spanning 2013-2016, and Recency becomes
the strongest separator between segments - see the 25-day versus 265-day medians
in the table below.

## The segments

| Segment | Customers | Median recency | Median frequency | Median spend | Total spend |
|---|---|---|---|---|---|
| Loyal | 82 | 97 days | 4 orders | 223.59 | 31,812 |
| Dormant high-value | 1,173 | 251 days | 1 order | 116.10 | 180,133 |
| Recent low-value | 727 | 25 days | 1 order | 37.99 | 39,287 |
| Lapsed low-value | 1,072 | 266 days | 1 order | 27.39 | 31,230 |

The 82 loyal customers order four times as often as anyone else. The dormant
group is the commercially interesting one: 1,173 people who each spent well but
have not come back in eight months - a re-engagement campaign, not a discount.

## Why k = 4 when k = 2 scores better

The silhouette scan is saved to `artifacts/customer_segments/cluster_selection.png`:

| k | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| silhouette | **0.539** | 0.352 | 0.337 | 0.361 | 0.341 | 0.354 | 0.362 | 0.363 | 0.351 |

k = 2 is the statistically cleanest split, and it separates "bought recently"
from "did not" - true, and useless for planning a campaign. k = 4 gives four
groups a marketing team can act on differently, at a cost of 0.2 silhouette.
That is a judgement call, so it is written in `config.yaml` where it can be
changed, and the scan is kept so the trade-off stays visible.

## Data shape

One wide denormalised export, 181 columns prefixed `Customers.`, `Orders.`,
`Order_Items.` and `Products.`, one row per order line. The pipeline
de-duplicates by order-line id, aggregates per customer, and `log1p`-transforms
all three metrics - each is heavily right-skewed, and KMeans on raw values would
simply isolate the largest spender.

Türkçe açıklamalar: [README.tr.md](README.tr.md)
