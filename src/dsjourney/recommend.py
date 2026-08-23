"""Recommender systems.

The course notebooks stopped at "here are ten films correlated with Star Wars"
and never scored anything - there was no holdout, so a recommender that returned
the same ten titles for every user would have looked identical to a good one.

Everything here is built around a held-out set of ratings instead:
:func:`split_ratings` withholds each user's most recent interactions, and
:func:`evaluate_recommender` reports RMSE on those ratings plus precision and
recall at k. Two of the three methods are the notebooks' own, rewritten;
:func:`fit_svd` supplies the matrix factorisation the notebook named but did not
actually implement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# The 19 genre flags occupy columns 5-23 of the MovieLens u.item file.
GENRE_COLUMNS = slice(5, 24)
GENRE_NAMES = [
    "unknown",
    "action",
    "adventure",
    "animation",
    "children",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "fantasy",
    "film_noir",
    "horror",
    "musical",
    "mystery",
    "romance",
    "sci_fi",
    "thriller",
    "war",
    "western",
]

MIN_RATINGS_FOR_SIMILARITY = 50

# Items rated by fewer than this many people are excluded from top-N lists. See
# SVDRecommender.rank_for_user for why the floor is load-bearing.
MIN_SUPPORT_FOR_RANKING = 20


@dataclass(frozen=True)
class RatingSplit:
    """Ratings divided into a fitting period and a held-out period."""

    train: pd.DataFrame
    test: pd.DataFrame

    @property
    def users(self) -> int:
        """Number of distinct users in the training half."""
        return int(self.train["user_id"].nunique())

    @property
    def items(self) -> int:
        """Number of distinct items in the training half."""
        return int(self.train["item_id"].nunique())


def load_ratings(path: Path) -> pd.DataFrame:
    """Read MovieLens ``u.data`` into user/item/rating/timestamp columns.

    The notebooks read only the first three columns. Keeping the timestamp is
    what makes an honest chronological split possible - see :func:`split_ratings`.
    """
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )


def load_items(path: Path) -> pd.DataFrame:
    """Read MovieLens ``u.item`` into item id, title and one column per genre."""
    raw = pd.read_csv(path, sep="|", header=None, encoding="iso-8859-1", engine="python")
    genres = raw.iloc[:, GENRE_COLUMNS].astype(int)
    genres.columns = GENRE_NAMES[: genres.shape[1]]
    return pd.concat([raw.iloc[:, [0, 1]].set_axis(["item_id", "title"], axis=1), genres], axis=1)


def split_ratings(ratings: pd.DataFrame, *, holdout_per_user: int = 5) -> RatingSplit:
    """Withhold each user's most recent ratings as the test set.

    A random split would let the model learn from a user's future taste to
    predict their past, which is exactly the situation a deployed recommender
    never faces. Users with too few ratings to spare any contribute only to
    training.
    """
    if holdout_per_user <= 0:
        raise ValueError(f"holdout_per_user must be positive, got {holdout_per_user}")

    ordered = ratings.sort_values(["user_id", "timestamp"])
    rank_from_end = ordered.groupby("user_id").cumcount(ascending=False)
    counts = ordered.groupby("user_id")["rating"].transform("size")

    is_holdout = (rank_from_end < holdout_per_user) & (counts > holdout_per_user * 2)
    return RatingSplit(
        ordered[~is_holdout].reset_index(drop=True), ordered[is_holdout].reset_index(drop=True)
    )


def user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """Pivot ratings into a users x items matrix with NaN for unseen pairs."""
    return ratings.pivot_table(index="user_id", columns="item_id", values="rating")


def popularity_ranking(ratings: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Rank items by rating count, mean rating and share of all rating mass.

    This is what the notebook labelled "matrix factorisation" actually computed.
    It is a useful baseline and a poor recommender: it ignores the user entirely.
    """
    stats = ratings.groupby("item_id")["rating"].agg(["size", "mean", "sum"])
    stats.columns = ["rating_count", "mean_rating", "rating_sum"]
    total = float(stats["rating_sum"].sum())
    stats["share_pct"] = (100 * stats["rating_sum"] / total).round(4) if total else 0.0
    # method="min" gives competition ranking (1, 2, 2, 4). The default "average"
    # would leave no item at rank 1 whenever the top two are tied.
    stats["rank"] = stats["share_pct"].rank(ascending=False, method="min")

    titled = stats.join(items.set_index("item_id")["title"])
    return titled.sort_values("rank")


def similar_by_ratings(
    matrix: pd.DataFrame,
    item_id: int,
    items: pd.DataFrame,
    *,
    limit: int = 10,
    min_ratings: int = MIN_RATINGS_FOR_SIMILARITY,
) -> pd.DataFrame:
    """Return items whose rating pattern correlates with the given item's.

    The ``min_ratings`` floor is what the notebook was missing: without it the
    top of the list is filled by obscure titles rated by three people who also
    happened to like the seed film, and the correlation means nothing.
    """
    if item_id not in matrix.columns:
        raise KeyError(f"item {item_id} has no ratings in this matrix")

    counts = matrix.count()
    # Correlating against columns with almost no ratings produces NaN and a
    # divide-by-zero warning from numpy, and those columns are filtered out
    # afterwards anyway - so drop them before the correlation, not after.
    popular_columns = counts[counts >= min_ratings].index
    correlations = matrix[popular_columns].corrwith(matrix[item_id]).dropna()

    table = pd.DataFrame({"correlation": correlations, "rating_count": counts[popular_columns]})
    popular = table[table.index != item_id]

    named = popular.join(items.set_index("item_id")["title"])
    return named.sort_values("correlation", ascending=False).head(limit)


def similar_by_genre(items: pd.DataFrame, item_id: int, *, limit: int = 10) -> pd.DataFrame:
    """Return items with the most similar genre vector, by cosine similarity."""
    indexed = items.set_index("item_id")
    genre_columns = [c for c in GENRE_NAMES if c in indexed.columns]
    vectors = indexed[genre_columns].to_numpy(dtype=float)

    if item_id not in indexed.index:
        raise KeyError(f"item {item_id} is not in the catalogue")

    positions = np.flatnonzero(np.asarray(indexed.index) == item_id)
    scores = cosine_similarity(vectors[positions[0] : positions[0] + 1], vectors).ravel()

    table = pd.DataFrame(
        {"similarity": scores, "title": indexed["title"].to_numpy()}, index=indexed.index
    )
    return table.drop(index=item_id).sort_values("similarity", ascending=False).head(limit)


@dataclass
class SVDRecommender:
    """Matrix factorisation over mean-centred ratings.

    Missing entries are filled with the item's mean rating before decomposition,
    so the factors describe how a user deviates from the crowd rather than
    treating "not rated" as "rated zero".

    Scores are held as a plain NumPy array with id-to-position lookups rather
    than a DataFrame: ranking every item for a user is an array slice instead of
    a label-based indexing round trip.
    """

    components: int = 20
    random_state: int = 42
    global_mean: float = 0.0
    _scores: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    _item_means: np.ndarray = field(default_factory=lambda: np.empty(0))
    _user_positions: dict[int, int] = field(default_factory=dict)
    _item_positions: dict[int, int] = field(default_factory=dict)
    _item_ids: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    _item_support: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))

    @property
    def is_fitted(self) -> bool:
        """True once :meth:`fit` has run."""
        return self._scores.size > 0

    def fit(self, ratings: pd.DataFrame) -> SVDRecommender:
        """Factorise the training ratings into latent user and item factors."""
        matrix = user_item_matrix(ratings)
        self.global_mean = float(ratings["rating"].mean())

        item_means = matrix.mean(axis=0).fillna(self.global_mean)
        filled = matrix.fillna(item_means)
        centred = filled.sub(item_means, axis=1).to_numpy(dtype=float)

        components = max(min(self.components, min(centred.shape) - 1), 1)
        model = TruncatedSVD(n_components=components, random_state=self.random_state)
        factors = model.fit_transform(centred)

        self._item_means = item_means.to_numpy(dtype=float)
        self._item_support = matrix.count().to_numpy(dtype=int)
        self._scores = factors @ model.components_ + self._item_means
        self._item_ids = np.asarray(matrix.columns, dtype=int)
        self._user_positions = {int(user): position for position, user in enumerate(matrix.index)}
        self._item_positions = {int(item): position for position, item in enumerate(matrix.columns)}
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        """Predict one user's rating for one item, falling back to means.

        A user or item unseen during training cannot have latent factors, so the
        item mean - or the global mean - stands in. Returning a number rather
        than raising is what lets a cold-start row be scored at all.
        """
        if not self.is_fitted:
            raise RuntimeError("fit() must be called before predict()")

        user_position = self._user_positions.get(int(user_id))
        item_position = self._item_positions.get(int(item_id))
        if user_position is not None and item_position is not None:
            return float(self._scores[user_position, item_position])
        if item_position is not None:
            return float(self._item_means[item_position])
        return self.global_mean

    def rank_for_user(
        self,
        user_id: int,
        seen: set[int],
        *,
        limit: int = 10,
        min_support: int = MIN_SUPPORT_FOR_RANKING,
    ) -> pd.Series:
        """Return the top scoring unseen items for a user, highest first.

        ``min_support`` is not a nicety. Without it the top of every list is
        filled by films a single person rated 5: their item mean is 5.0, no
        user deviates from it in the factors, and they outrank everything a real
        person might actually watch. Measured on MovieLens 100k, dropping the
        floor takes precision@10 below what random ranking would achieve.
        """
        user_position = self._user_positions.get(int(user_id))
        if user_position is None:
            return pd.Series(dtype=float)

        eligible = self._item_support >= min_support
        scores = pd.Series(self._scores[user_position][eligible], index=self._item_ids[eligible])
        unseen = scores.drop(labels=[i for i in seen if i in scores.index], errors="ignore")
        return unseen.sort_values(ascending=False).head(limit)

    def recommend(
        self, user_id: int, seen: set[int], items: pd.DataFrame, *, limit: int = 10
    ) -> pd.DataFrame:
        """Return the highest-scoring items this user has not already rated."""
        top = self.rank_for_user(user_id, seen, limit=limit)
        titles = items.set_index("item_id")["title"].reindex(top.index)
        return pd.DataFrame(
            {
                "item_id": np.asarray(top.index),
                "predicted_rating": top.to_numpy(),
                "title": titles.to_numpy(),
            }
        )


def fit_svd(
    ratings: pd.DataFrame, *, components: int = 20, random_state: int = 42
) -> SVDRecommender:
    """Fit an :class:`SVDRecommender` on a ratings frame."""
    return SVDRecommender(components=components, random_state=random_state).fit(ratings)


def evaluate_recommender(
    model: SVDRecommender,
    split: RatingSplit,
    *,
    k: int = 10,
    liked_threshold: float = 4.0,
    min_support: int = MIN_SUPPORT_FOR_RANKING,
) -> dict[str, float]:
    """Score predicted ratings and top-k relevance on the held-out period.

    RMSE answers "how close are the predicted scores?"; precision and recall at
    k answer "are the right things at the top of the list?" - a model can do
    well on the first and badly on the second, which is why the notebooks'
    eyeball check of a single similarity list proved nothing.

    ``min_support`` must match what serving uses, or the reported precision
    describes a different list from the one users would actually see.
    """
    users = split.test["user_id"].to_numpy(dtype=int)
    items = split.test["item_id"].to_numpy(dtype=int)
    predictions = np.array(
        [model.predict(int(user), int(item)) for user, item in zip(users, items, strict=True)]
    )
    truth = split.test["rating"].to_numpy(dtype=float)
    errors = truth - predictions

    seen_by_user = _items_by_user(split.train)
    liked = split.test[split.test["rating"] >= liked_threshold]
    relevant_by_user = _items_by_user(liked)

    hits, recommended, relevant_total = 0, 0, 0
    for user_id, relevant in relevant_by_user.items():
        suggestions = set(
            model.rank_for_user(
                user_id, seen_by_user.get(user_id, set()), limit=k, min_support=min_support
            ).index
        )
        hits += len(suggestions & relevant)
        recommended += len(suggestions)
        relevant_total += len(relevant)

    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        f"precision_at_{k}": float(hits / recommended) if recommended else float("nan"),
        f"recall_at_{k}": float(hits / relevant_total) if relevant_total else float("nan"),
        "evaluated_users": float(len(relevant_by_user)),
    }


RECOMMENDERS = {
    "svd": fit_svd,
    "popularity": popularity_ranking,
    "item_correlation": similar_by_ratings,
    "content_genre": similar_by_genre,
}


def available_models() -> list[str]:
    """Return the registered recommender names."""
    return sorted(RECOMMENDERS)


def _items_by_user(ratings: pd.DataFrame) -> dict[int, set[int]]:
    """Group item ids by user with concrete int keys.

    ``groupby(...).apply(set).to_dict()`` types its keys as ``Hashable``, which
    then has to be cast at every use; building the mapping explicitly keeps the
    signature honest.
    """
    grouped: dict[int, set[int]] = {}
    for user, item in zip(
        ratings["user_id"].to_numpy(dtype=int),
        ratings["item_id"].to_numpy(dtype=int),
        strict=True,
    ):
        grouped.setdefault(int(user), set()).add(int(item))
    return grouped


def compare_svd_components(
    split: RatingSplit,
    *,
    candidates: tuple[int, ...] = (5, 10, 20, 50),
    k: int = 10,
    min_support: int = MIN_SUPPORT_FOR_RANKING,
) -> pd.DataFrame:
    """Score SVD at several ranks so the chosen factor count is justified."""
    rows: list[dict[str, Any]] = []
    for components in candidates:
        model = fit_svd(split.train, components=components)
        scores = evaluate_recommender(model, split, k=k, min_support=min_support)
        rows.append({"components": components, **scores})
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
