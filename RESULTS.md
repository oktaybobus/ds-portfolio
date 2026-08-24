# Results

Every number here is read from `artifacts/<project>/metrics.json` by
`scripts/update_results.py`. Regenerate after training rather than editing
by hand; `--check` fails when the file has drifted.

Last generated: 2026-08-24

| Project | Task | Model | Headline | All metrics |
|---|---|---|---|---|
| `article_search` | retrieval | TfidfSVD | **MRR 0.569** | probes 300.000, recall_at_1 0.470, recall_at_5 0.737, mrr 0.569, passage_at_5 0.627, context_words 1398.173, hits_per_1k_words 0.448 |
| `bart_ridership` | regression | HistGradientBoostingRegressor | **R² 0.818** | r2 0.818, rmse 0.457, mae 0.354, mape 0.282, r2_original 0.818, rmse_original 14.234, mae_original 4.497, mape_original 0.602 |
| `customer_segments` | clustering | KMeans | **Silhouette 0.337** | silhouette 0.337, calinski_harabasz 1701.229, davies_bouldin 0.888 |
| `diabetes_screening` | classification | SparkLogisticRegression | **Recall 0.531** | accuracy 0.745, precision 0.672, recall 0.531, f1 0.593, roc_auc 0.836, majority_baseline 0.649 |
| `istanbul_housing` | regression | CatBoostRegressor | **R² 0.814** | r2 0.849, rmse 0.240, mae 0.179, mape 0.093, r2_original 0.814, rmse_original 3.219, mae_original 1.889, mape_original 0.218 |
| `laptop_price` | regression | CatBoostRegressor | **R² 0.805** | r2 0.895, rmse 0.195, mae 0.138, mape 0.013, r2_original 0.805, rmse_original 17443.991, mae_original 8849.380, mape_original 0.140 |
| `loan_default` | classification | RandomForestClassifier | **Recall 0.737** | accuracy 0.680, precision 0.441, recall 0.737, f1 0.552, roc_auc 0.775 |
| `marvel_network` | graph | SparkBFS | **Reach 0.994** | heroes 6486.000, co_appearance_pairs 336534.000, max_degree 1933.000, mean_degree 51.886, median_degree 20.000, isolated_heroes 19.000, reachable_fraction 0.994, mean_distance 1.706, eccentricity 3.000 |
| `movie_recommender` | recommendation | TruncatedSVD | **Precision@10 0.018** | rmse 1.059, mae 0.845, precision_at_10 0.018, recall_at_10 0.058, evaluated_users 820.000 |
| `object_detection` | detection | HaarCascade | **Miscount 1.000** | detected 6.000, expected 7.000, error 1.000 |
| `review_sentiment` | text-classification | Pipeline | **F1 0.957** | accuracy 0.934, precision 0.975, recall 0.939, f1 0.957, roc_auc 0.978 |
| `series_forecast / adidas_revenue` | forecasting | naive | **Skill vs naive 0.000** | mae 879.250, rmse 1132.862, mape 0.209, mase 1.169, skill_vs_naive 0.000 |
| `series_forecast / delhi_temperature` | forecasting | holt_winters | **Skill vs naive 0.447** | mae 2.169, rmse 2.616, mape 0.111, mase 1.749, skill_vs_naive 0.447 |

## Not trained in this checkout

- `image_classifiers` - run `uv sync --extra dl --extra data && python projects/image_classifiers/train.py --dataset grape`

## Reading these numbers

- **`loan_default` is ranked on recall, not accuracy.** The data is 27%
  defaults, so predicting 'never defaults' for everyone scores 73% accuracy
  and catches nothing.
- **Regression headlines are on the original target scale.** Both projects
  train on `log1p(price)`; a log-scale R² and a price-scale R² are different
  numbers, so `r2` and `r2_original` are both reported.
- **`customer_segments` uses k = 4 although k = 2 scores higher.** Four
  segments are actionable; two are not. The scan is in
  `artifacts/customer_segments/cluster_selection.png`.
- **`series_forecast` is ranked on skill against a naive baseline.** On
  Adidas revenue that skill is 0.0 - nothing beats repeating the last
  quarter, which is a result, not a missing model.
- **`movie_recommender` precision@10 looks small by construction.** Each
  user has a handful of held-out films among 1,682 candidates; random
  ranking scores about 0.002.
- **`article_search` reports two levels plus a cost.** Document recall
  always favours bigger chunks; `hits_per_1k_words` is what makes the
  trade against context size visible.
- **`object_detection` is scored as a miscount, so lower is better.** Six
  of seven faces found on the reference image, zero false positives.
- **`diabetes_screening` is ranked on recall for the same reason as
  `loan_default`.** It is a screening test: a missed diabetic patient
  costs more than a false alarm. Its accuracy of 0.745 sits only 0.095
  above always answering 'not diabetic'.
- **`marvel_network` has no single score.** Reach is the fraction of the
  graph within `eccentricity` hops of Captain America; the degree
  distribution is the rest of the answer.
