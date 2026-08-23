# Results

Every number here is read from `artifacts/<project>/metrics.json` by
`scripts/update_results.py`. Regenerate after training rather than editing
by hand; `--check` fails when the file has drifted.

Last generated: 2026-08-23

| Project | Task | Model | Headline | All metrics |
|---|---|---|---|---|
| `customer_segments` | clustering | KMeans | **Silhouette 0.337** | silhouette 0.337, calinski_harabasz 1701.229, davies_bouldin 0.888 |
| `laptop_price` | regression | CatBoostRegressor | **R² 0.895** | r2 0.895, rmse 0.195, mae 0.138, mape 0.013 |
| `loan_default` | classification | RandomForestClassifier | **Recall 0.737** | accuracy 0.680, precision 0.441, recall 0.737, f1 0.552, roc_auc 0.775 |
| `review_sentiment` | text-classification | Pipeline | **F1 0.957** | accuracy 0.934, precision 0.975, recall 0.939, f1 0.957, roc_auc 0.978 |

## Not trained in this checkout

- `image_classifiers` - run `uv sync --extra dl --extra data && python projects/image_classifiers/train.py --dataset grape`

## Reading these numbers

- **`loan_default` is ranked on recall, not accuracy.** The data is 27%
  defaults, so predicting 'never defaults' for everyone scores 73% accuracy
  and catches nothing.
- **`laptop_price` metrics are on the log-transformed target.** MAPE is on
  the same scale; the demo app converts predictions back to currency.
- **`customer_segments` uses k = 4 although k = 2 scores higher.** Four
  segments are actionable; two are not. The scan is in
  `artifacts/customer_segments/cluster_selection.png`.
