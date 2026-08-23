#!/usr/bin/env python
"""Fit and score the MovieLens recommender.

Usage:
    python projects/movie_recommender/train.py
    python projects/movie_recommender/train.py --scan
"""

from __future__ import annotations

import argparse
import json

from dsjourney import recommend
from dsjourney.paths import project_artifacts_dir
from projects.movie_recommender import pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--components",
        type=int,
        default=int(pipeline.CONFIG.model.params.get("components", 50)),
        help="SVD rank",
    )
    parser.add_argument("--holdout", type=int, default=5, help="ratings withheld per user")
    parser.add_argument("--scan", action="store_true", help="score several SVD ranks")
    parser.add_argument("--no-save", action="store_true", help="do not write artifacts")
    args = parser.parse_args(argv)

    ratings = pipeline.load_raw()
    items = pipeline.load_items()
    summary = pipeline.catalogue_summary(ratings, items)
    print(
        f"{summary['ratings']:,} ratings | {summary['users']} users | "
        f"{summary['films']} films rated of {summary['catalogue_size']} in the catalogue"
    )

    split = recommend.split_ratings(ratings, holdout_per_user=args.holdout)
    print(f"train {len(split.train):,} | holdout {len(split.test):,} (most recent per user)")

    if args.scan:
        scan = recommend.compare_svd_components(split)
        print(scan.to_string(index=False))

    model = recommend.fit_svd(split.train, components=args.components)
    metrics = recommend.evaluate_recommender(model, split, k=10)
    print(" | ".join(f"{key} {value:.4f}" for key, value in metrics.items()))

    popular = recommend.popularity_ranking(split.train, items)
    print("\nTop 5 by share of all rating mass (the notebook's 'matrix factorisation'):")
    print(popular.head(5)[["title", "rating_count", "mean_rating", "share_pct"]].to_string())

    matrix = recommend.user_item_matrix(split.train)
    similar = recommend.similar_by_ratings(matrix, pipeline.DEMO_ITEM_ID, items)
    print("\nMost correlated with Star Wars (1977), min 50 ratings:")
    print(similar[["title", "correlation", "rating_count"]].to_string())

    if not args.no_save:
        directory = project_artifacts_dir("movie_recommender", create=True)
        (directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        metadata = {
            "model_class": "TruncatedSVD",
            "components": args.components,
            "holdout_per_user": args.holdout,
            **summary,
            "train_ratings": len(split.train),
            "holdout_ratings": len(split.test),
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        popular.head(50).to_csv(directory / "popular_films.csv")
        similar.to_csv(directory / "similar_to_star_wars.csv")
        if args.scan:
            scan.to_csv(directory / "svd_rank_scan.csv", index=False)
        print(f"\nartifacts: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
