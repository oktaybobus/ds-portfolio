"""Streamlit demo for the MovieLens recommender.

Run with ``dsj serve movie_recommender``. Fits on load rather than reading a
saved bundle: the factorisation takes a second or two on 100k ratings, and
keeping it live means the holdout logic stays visible in one place.
"""

from __future__ import annotations

import streamlit as st

from dsjourney import recommend
from projects.movie_recommender import pipeline

st.set_page_config(page_title="MovieLens Recommender", page_icon="=", layout="centered")


@st.cache_resource
def _fit():
    """Load the ratings, split them chronologically and fit the model once."""
    ratings = pipeline.load_raw()
    items = pipeline.load_items()
    split = recommend.split_ratings(ratings, holdout_per_user=5)
    components = int(pipeline.CONFIG.model.params.get("components", 50))
    model = recommend.fit_svd(split.train, components=components)
    metrics = recommend.evaluate_recommender(model, split, k=10)
    return ratings, items, split, model, metrics


def main() -> None:
    st.title("MovieLens Recommender")
    st.caption("SVD matrix factorisation over 100,000 ratings, scored on held-out recent ratings.")

    try:
        _ratings, items, split, model, metrics = _fit()
    except FileNotFoundError as error:
        st.error(str(error))
        return

    with st.sidebar:
        st.subheader("Held-out performance")
        st.metric("RMSE", f"{metrics['rmse']:.3f}")
        st.metric("Precision@10", f"{metrics['precision_at_10']:.4f}")
        st.metric("Recall@10", f"{metrics['recall_at_10']:.4f}")
        st.caption(
            "Precision@10 looks small because a user has only a handful of "
            "held-out films among 1,682 candidates. Random ranking scores "
            "about 0.002."
        )

    tab_user, tab_similar = st.tabs(["Recommend for a user", "Find similar films"])

    with tab_user:
        users = sorted(split.train["user_id"].unique())
        user_id = st.selectbox("User id", users, index=0)
        seen = set(split.train.loc[split.train["user_id"] == user_id, "item_id"])
        st.caption(f"This user rated {len(seen)} films in the training period.")
        if st.button("Recommend", type="primary", use_container_width=True):
            st.dataframe(
                model.recommend(int(user_id), seen, items, limit=10), use_container_width=True
            )

    with tab_similar:
        titles = items.set_index("item_id")["title"]
        popular = recommend.popularity_ranking(split.train, items).head(200)
        choice = st.selectbox("Film", popular["title"].tolist())
        item_id = int(titles[titles == choice].index[0])
        method = st.radio("Similarity", ["Rating correlation", "Genre overlap"], horizontal=True)
        if st.button("Find similar", type="primary", use_container_width=True):
            matrix = recommend.user_item_matrix(split.train)
            table = (
                recommend.similar_by_ratings(matrix, item_id, items)
                if method == "Rating correlation"
                else recommend.similar_by_genre(items, item_id)
            )
            st.dataframe(table, use_container_width=True)


main()
