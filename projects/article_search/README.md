# Wikipedia Article Search

Semantic search over 390 articles on human rights, conflict and international law.

| | |
|---|---|
| Task | Retrieval |
| Corpus | 390 documents, 1.49M words |
| Index | TF-IDF → SVD (256 dims) → L2 normalise, 300-word chunks |
| **MRR** | **0.569** |
| Recall@1 / Recall@5 | 0.470 / 0.737 |
| Passage@5 | 0.627 at ~1,400 words of context |
| Source | `AI agents.ipynb` |

```bash
uv run python projects/article_search/train.py --scan
uv run python projects/article_search/search.py "obligations of an occupying power" --passages
uv run dsj serve article_search
```

## The notebook ran one query

The source loaded all 390 articles into ChromaDB - one vector per document,
some of them 150 KB - and ran a single query:

```python
res = col.query(query_texts=["Turkey human rights violations?"], n_results=5)
```

Five results came back, they looked topical, and that was the evaluation. It
tells you the pipeline runs. It does not tell you whether the right document
came back, which is the only thing a search index is for.

`build_probes` takes a sentence out of a known article and asks the index to
find its way home. Three hundred of those give recall@k and MRR - a number that
can be compared between settings.

## Chunk size: the two metrics disagree

`train.py --scan` scores several windows on the same probes:

| Chunk words | Overlap | Chunks | MRR | Passage@5 | Context words | Hits per 1k words |
|---|---|---|---|---|---|---|
| 2000 | 0 | 948 | **0.775** | **0.885** | 7,444 | 0.119 |
| 1000 | 200 | 2,034 | 0.707 | 0.765 | 4,188 | 0.183 |
| 600 | 120 | 3,295 | 0.650 | 0.705 | 2,653 | 0.266 |
| **300** | **60** | **6,390** | 0.586 | 0.675 | 1,396 | 0.483 |
| 180 | 40 | 10,794 | 0.508 | 0.530 | 856 | 0.619 |
| 120 | 20 | 15,048 | 0.473 | 0.555 | 583 | **0.953** |

Read down the MRR column and bigger is always better - which, taken to its
limit, recommends one vector per document. That is what the notebook did, and
by this metric alone the notebook was right.

Read the last column and it inverts. `hits_per_1k_words` is the hit rate
divided by how much text you had to carry to get it, and 120-word chunks are
**eight times more efficient** than 2000-word ones. For a RAG pipeline, where
retrieved chunks are pasted into a prompt and paid for by the token, that is
the column that matters.

There is no single right answer, only a question that has to be asked first:

- **A search box that returns article titles** - use large chunks. Nothing else
  matters, and 2000 words wins outright.
- **Context for a language model** - use small chunks. You are buying tokens.

This project uses 300/60 because its demo does both: it ranks articles *and*
shows the matching passage. That is a decision, and it is in `config.yaml`
where it can be changed, next to the numbers that justify it.

## Scores are not confidence

The corpus has nothing about renewing a driving licence. Ask anyway and the
index returns five results, the top one scoring 0.511 - higher than plenty of
genuine matches.

Measured over 150 answerable probes and 12 deliberately out-of-domain queries:

| | Mean top score | Range |
|---|---|---|
| Answerable | 0.606 | 0.350 - 0.9+ |
| Out of domain | 0.397 | up to 0.760 |

The distributions overlap, and no threshold separates them. At 0.55 you keep
63% of answerable queries while rejecting only 67% of unanswerable ones -
barely better than a coin flip.

So `WEAK_MATCH = 0.40` is set where it is useful and no further: 94% of
answerable queries clear it, so a score *below* it is a real signal that the
corpus has no answer. Above it means nothing in particular. Doing this properly
needs a reranker or a background-distribution comparison, and neither is in
this project - which is worth saying rather than shipping a threshold that
implies a confidence it does not have.

Türkçe açıklamalar: [README.tr.md](README.tr.md)
