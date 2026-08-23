#!/usr/bin/env python
"""Search the article corpus from the command line.

Usage:
    python projects/article_search/search.py "obligations of an occupying power"
    python projects/article_search/search.py "apartheid" --passages -k 3
"""

from __future__ import annotations

import argparse
import textwrap

from projects.article_search import pipeline

# Measured, not guessed: over 150 answerable probe queries and 12 deliberately
# out-of-domain ones, 94% of answerable queries score at least 0.40, so a top
# score below it is a strong signal that the corpus has no answer.
#
# The reverse does not hold. Half the out-of-domain queries also clear 0.40, and
# the worst of them reaches 0.760 - higher than most genuine matches. The two
# distributions overlap, and no threshold separates them: at 0.55 you keep 63%
# of answerable queries while rejecting only 67% of unanswerable ones.
#
# So this catches the obvious failures and not the plausible ones. Telling a
# user their query looks unanswerable when it scored 0.38 is honest; claiming
# anything above the line is answerable would not be. Separating those properly
# needs a reranker or a background-distribution comparison, neither of which is
# in this project.
WEAK_MATCH = 0.40


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="what to search for")
    parser.add_argument("-k", type=int, default=5, help="how many results")
    parser.add_argument("--passages", action="store_true", help="show the matching text")
    args = parser.parse_args(argv)

    index = pipeline.build_index()
    hits = index.search_documents(args.query, k=args.k)

    if hits.empty or float(hits.iloc[0]["score"]) < WEAK_MATCH:
        best = 0.0 if hits.empty else float(hits.iloc[0]["score"])
        print(f"No confident match (best score {best:.3f}, below {WEAK_MATCH}).")
        return 1

    for _, row in hits.iterrows():
        marker = " " if row["score"] >= WEAK_MATCH else "?"
        print(f"{marker} {row['score']:.3f}  {row['document_id']}")
        if args.passages:
            snippet = textwrap.shorten(row["text"], width=340, placeholder=" ...")
            print(textwrap.indent(textwrap.fill(snippet, width=88), "         "))
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
