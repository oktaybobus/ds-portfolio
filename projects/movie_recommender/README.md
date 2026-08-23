# MovieLens Recommender

Recommend films from 100,000 ratings, and measure whether the recommendations
are any good.

| | |
|---|---|
| Task | Recommendation |
| Data | MovieLens 100k - 100,003 ratings, 944 users, 1,682 films |
| Model | Truncated SVD, rank 50, over mean-centred ratings |
| RMSE / MAE | 1.059 / 0.846 |
| **Precision@10** | **0.0187** |
| Recall@10 | 0.0608 |
| Source | `Day10 AOB RSSKNN.ipynb`, `RSSK22.ipynb`, `RSMatrixFactorization.ipynb` |

```bash
uv run python scripts/fetch_assets.py --project movie_recommender
uv run python projects/movie_recommender/train.py --scan
uv run dsj serve movie_recommender
```

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
| Empire Strikes Back (1980) | 0.752 | 355 |
| Return of the Jedi (1983) | 0.679 | 489 |
| Raiders of the Lost Ark (1981) | 0.526 | 407 |
| Indiana Jones and the Last Crusade (1989) | 0.353 | 320 |

**Ranking needs it even more.** Without a support floor, precision@10 measured
**0.0005** - *below* the ~0.002 that random ranking would achieve. The cause:
SVD fills unseen cells with item means, so a film one person rated 5 gets a
predicted score of 5.0 for everybody and outranks anything a real user might
watch. Requiring 20 ratings to enter a top-N list lifted precision@10 to 0.0187,
a 32x difference from a single constant.

## Choosing the rank

`train.py --scan` scores several factor counts on the same holdout:

| Components | RMSE | Precision@10 | Recall@10 |
|---|---|---|---|
| 10 | 1.058 | 0.0159 | 0.0517 |
| 20 | **1.055** | 0.0156 | 0.0509 |
| 30 | 1.055 | 0.0166 | 0.0541 |
| **50** | 1.059 | **0.0187** | **0.0608** |
| 80 | 1.064 | 0.0166 | 0.0541 |
| 120 | 1.067 | 0.0165 | 0.0537 |

Rank 20 wins on RMSE by 0.4%; rank 50 wins on both ranking metrics by 20%. A
recommender is judged on what it puts at the top of the list, so 50 is what the
config names.

## About that notebook title

`Day10 AOB RSMatrixFactorization.ipynb` contains no matrix factorisation. It
groups by title, sums ratings, computes each film's share of total rating mass
and ranks them - popularity, not factorisation. That is kept here as
`popularity_ranking`, which is an honest baseline and a poor recommender: it
ignores the user completely. The actual factorisation is `fit_svd`.

Türkçe açıklamalar: [README.tr.md](README.tr.md)
