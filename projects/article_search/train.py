#!/usr/bin/env python
"""Build the article index and measure how well it retrieves.

Usage:
    python projects/article_search/train.py
    python projects/article_search/train.py --scan
    python projects/article_search/train.py --probes 500
"""

from __future__ import annotations

import argparse
import json

from dsjourney import retrieval
from dsjourney.paths import project_artifacts_dir
from projects.article_search import pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=int, default=300, help="how many probe queries to score")
    parser.add_argument("--k", type=int, default=5, help="cut-off for recall@k")
    parser.add_argument("--scan", action="store_true", help="compare chunk sizes")
    parser.add_argument("--no-save", action="store_true", help="do not write artifacts")
    args = parser.parse_args(argv)

    documents = pipeline.load_documents()
    words = sum(len(text.split()) for text in documents.values())
    print(f"{len(documents)} documents | {words:,} words")

    probes = retrieval.build_probes(documents, count=args.probes)
    print(f"{len(probes)} probes (a sentence taken from a known article)")

    scan = None
    if args.scan:
        scan = retrieval.compare_chunk_sizes(documents, probes, k=args.k)
        print(scan.to_string(index=False))

    index = pipeline.build_index(documents)
    print(f"index: {len(index.chunks):,} chunks over {index.document_count} documents")

    metrics = retrieval.evaluate_retrieval(index, probes, k=args.k)
    print(" | ".join(f"{name} {value:.4f}" for name, value in metrics.items()))

    print("\nExample queries:")
    for query in pipeline.EXAMPLE_QUERIES:
        hits = index.search_documents(query, k=1)
        if hits.empty:
            print(f"  {query!r} -> nothing")
            continue
        top = hits.iloc[0]
        print(f"  {query[:52]!r:56} -> {top['document_id'][:44]} ({top['score']:.3f})")

    if not args.no_save:
        directory = project_artifacts_dir("article_search", create=True)
        (directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        metadata = {
            "model_class": "TfidfSVD",
            "documents": len(documents),
            "chunks": len(index.chunks),
            "words": words,
            **{k: int(v) for k, v in pipeline.CONFIG.model.params.items()},
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if scan is not None:
            scan.to_csv(directory / "chunk_size_scan.csv", index=False)
        print(f"\nartifacts: {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
