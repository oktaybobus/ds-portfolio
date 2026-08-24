# MovieLens Recommender

Recommend films from 100,000 ratings, and measure whether the recommendations
are any good.

| | |
|---|---|
| Task | Recommendation |
| Data | MovieLens 100k - 100,000 ratings, 943 users, 1,682 films |
| Model | Truncated SVD, rank 50, over mean-centred ratings |
| RMSE / MAE | 1.059 / 0.845 |
| **Precision@10** | **0.0178** |
| Recall@10 | 0.0580 |
| Source | `Day10 AOB RSSKNN.ipynb`, `RSSK22.ipynb`, `RSMatrixFactorization.ipynb` |

```bash
uv run python scripts/fetch_assets.py --project movie_recommender
uv run python projects/movie_recommender/train.py --scan
uv run dsj serve movie_recommender
```

## About user 0

`u.info` in the original course tree declares 943 users and 100,000 ratings.
The shipped `u.data` disagrees: it has 944 users and 100,003 ratings, because
three rows at the top belong to `user_id == 0` - an id absent from `u.user`
and impossible under real MovieLens, where ids start at 1. Nobody documented
adding it; it just sat there.

**Decision: excluded from training.** `pipeline.load_raw()` drops the three
`user_id == 0` rows before anything else touches the data, so `train.py` fits
on 100,000 ratings from 943 users - the shape `u.info` actually declares.
Three rows out of 100,003 barely move any metric (RMSE by 0.0003, precision@10
by 0.0009 - see "Choosing the rank" below), so this is not a fix for a broken
result. It closes the gap between what the manifest
says and what the model was actually trained on, which had gone unnoticed
long enough that `metadata.json` shipped `"users": 944` without anyone
comparing it to `u.info`.

The unfiltered file - all 100,003 rows, all 944 ids, `user_id == 0` included -
is still what `pipeline.ratings_path()` points at, and its exact shape is
pinned by `test_the_raw_file_has_the_published_shape` in
`tests/projects/test_group2_projects.py`. If a future re-download changes
either number, or drops the synthetic row, that test fails instead of the
discrepancy sitting unnoticed again.

## Nothing was measured before

The three source notebooks produced similarity lists and stopped. With no
holdout, a recommender returning the same ten films to everyone would have
looked identical to a good one.

Here each user's five most recent ratings are withheld - chronologically, not at
random, because a deployed recommender never gets to see a user's future taste
while predicting their past. RMSE scores the predicted ratings; precision and
recall at 10 score what actually reaches the top of the list. A model can do
well on the first and badly on the second.

## Two floors that turned out to be load-bearing

**Similarity needs minimum support.** The notebook's `corrwith` list was topped
by obscure titles rated by three people who happened to also like the seed film.
Requiring 50 ratings gives a list that reads correctly:

| Film | Correlation | Ratings |
|---|---|---|
| Empire Strikes Back (1980) | 0.752 | 354 |
| Return of the Jedi (1983) | 0.679 | 489 |
| Raiders of the Lost Ark (1981) | 0.526 | 407 |
| Indiana Jones and the Last Crusade (1989) | 0.353 | 320 |

**Ranking needs it even more.** Without a support floor, precision@10 measured
**0.0005** - *below* the ~0.002 that random ranking would achieve. The cause:
SVD fills unseen cells with item means, so a film one person rated 5 gets a
predicted score of 5.0 for everybody and outranks anything a real user might
watch. Requiring 20 ratings to enter a top-N list lifted precision@10 to 0.0178,
a 36x difference from a single constant.

## Choosing the rank

`train.py --scan` scores several factor counts on the same holdout:

| Components | RMSE | Precision@10 | Recall@10 |
|---|---|---|---|
| 10 | 1.058 | 0.0152 | 0.0497 |
| 20 | **1.053** | 0.0162 | 0.0529 |
| 30 | 1.055 | **0.0178** | **0.0580** |
| **50** | 1.059 | **0.0178** | **0.0580** |
| 80 | 1.061 | 0.0171 | 0.0556 |
| 120 | 1.066 | 0.0154 | 0.0501 |

Rank 20 keeps the best RMSE, about 0.6% ahead of the config's rank 50. On the
ranking metrics rank 30 and rank 50 land exactly tied - both score precision@10
0.0178 and recall@10 0.0580, roughly 10% ahead of rank 20. A recommender is
judged on what it puts at the top of the list, so a ranking-metric win still
outweighs the RMSE win; between the tied 30 and 50, 50 is what the config
already named, and picking a different tied rank was out of scope for a fix
about a synthetic user.

## About that notebook title

`Day10 AOB RSMatrixFactorization.ipynb` contains no matrix factorisation. It
groups by title, sums ratings, computes each film's share of total rating mass
and ranks them - popularity, not factorisation. That is kept here as
`popularity_ranking`, which is an honest baseline and a poor recommender: it
ignores the user completely. The actual factorisation is `fit_svd`.

Türkçe açıklamalar: [README.tr.md](README.tr.md)
