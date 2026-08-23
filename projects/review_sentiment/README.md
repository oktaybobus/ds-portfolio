# Restaurant Review Sentiment

Classify a restaurant review as positive or negative from its text alone.

| | |
|---|---|
| Task | Binary text classification |
| Data | 8,856 reviews after removing neutral ones |
| Model | TF-IDF (1-2 grams, 5,000 features) → LogisticRegression |
| **F1** | **0.957** |
| ROC AUC | 0.978 |
| Precision / Recall | 0.975 / 0.939 |
| Accuracy | 0.934 |
| Source | `HW16_AOB_NLP_ClassSentimentAnalysis.ipynb` |

```bash
uv run python scripts/fetch_assets.py --project review_sentiment
uv run python projects/review_sentiment/train.py
uv run python projects/review_sentiment/train.py --max-features 20000
```

## Labels

Sentiment is derived from the star rating: 4-5 positive, 1-2 negative. **3-star
reviews are dropped, not assigned.** They are genuinely mixed - forcing them
either way teaches the model to guess on exactly the cases where the text is
least decisive.

## What the model learned

`train.py` prints the strongest coefficients, which are a readable sanity check
that the model latched onto sentiment and not an artefact:

| Positive | weight | | Negative | weight |
|---|---|---|---|---|
| amazing | +5.33 | | mediocre | −4.36 |
| delicious | +4.10 | | disappointed | −4.22 |
| favorite | +3.48 | | worst | −3.98 |
| great | +3.41 | | dry | −3.65 |
| best buffet | +2.91 | | salty | −3.63 |

`best buffet` is a bigram, which is why `ngram_range=(1, 2)` earns its place.

## Three steps from the notebook that are not here

| Dropped | Why |
|---|---|
| `langdetect` per-row filtering | Minutes of runtime to remove a fraction of a percent of rows, and non-deterministic across library versions |
| TextBlob lemmatisation | Needs a corpus download at import time, which breaks hermetic CI and slows the Docker build; the F1 change was inside split noise |
| NLTK stopword corpus | Replaced by an inlined stopword list in `dsjourney.text` - same words, no download |

What remains - lowercasing, stripping punctuation and digits, removing
stopwords, TF-IDF over unigrams and bigrams - is what carried the signal.

## Leakage note

The vectoriser is fitted **inside** the scikit-learn `Pipeline`, so it sees the
training split only. Fitting TF-IDF on the full corpus before splitting - a
common shortcut - leaks the test set's vocabulary and IDF weights into training
and inflates the score.

Türkçe açıklamalar: [README.tr.md](README.tr.md)
