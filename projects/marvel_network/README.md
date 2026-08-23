# Marvel Co-Appearance Network

The social graph of 6,486 Marvel characters, traversed with Spark.

| | |
|---|---|
| Task | Graph |
| Graph | 6,486 characters, 336,534 co-appearance pairs |
| Engine | PySpark, local mode |
| **Most connected** | **Captain America, 1,933 co-appearances** |
| Degree | mean 51.9, median 20, 19 characters with none |
| Reach from Captain America | 99.43% of the graph within 3 hops |
| Source | `Day10 AOB BigDataSpark.ipynb` |

```bash
uv run python projects/marvel_network/train.py
uv run python projects/marvel_network/train.py --root 5306 --benchmark
```

Needs a JVM. `brew install openjdk@17` on macOS, `apt install openjdk-17-jdk`
on Debian; `dsjourney.spark.java_home()` finds it without any shell setup.

## The notebook asked one question and got lucky

```python
mostPopular = flipped.max()
```

`flipped` is `(count, id)` pairs, so `max()` compares counts first and falls
back to comparing ids. On a tie it returns whichever character has the higher
id - silently, with no indication that a tie occurred.

Here it happens to be safe: Captain America has 1,933 co-appearances and the
runner-up has 1,741, so the answer is 192 clear and there is nothing to break.
That is luck, not correctness, and it is worth knowing which one you are
relying on. `test_the_most_connected_hero_wins_outright` asserts the margin
exists.

The rest of the distribution was never looked at, and it is more interesting
than the winner:

| | Characters |
|---|---|
| Degree 0 (never co-appear with anyone) | 19 |
| Degree 1-9 | 1,397 |
| Median degree | 20 |
| Degree 1,000+ | 26 |

A mean of 51.9 against a median of 20 is the usual shape of a social network:
a handful of protagonists carry most of the connections.

## 74 characters are split across lines

`Marvel-graph.txt` has 6,589 lines but only 6,486 characters, because a
character with many appearances is continued on a second line:

```
5988 748 1722 3752 ...
5988 1364 4126 ...
```

The notebook's `reduceByKey` handled this correctly. The `map`-only version
people usually write from the same tutorial does not, and undercounts exactly
those 74 - a 1% error concentrated entirely in the most connected characters,
which are the ones anybody would look at. `adjacency_from_lines` aggregates by
key, and `test_adjacency_aggregates_a_node_split_over_several_lines` pins it.

## The names file is not UTF-8

`Marvel-names.txt` is Latin-1. Two of its 19,428 lines contain bytes that are
not valid UTF-8, and `sc.textFile` - which the notebook used - decodes as UTF-8
and substitutes U+FFFD for anything that fails. No exception, no warning, just
two names silently wrong.

Spark's `read.text` has no encoding option at all; its CSV reader does. So
`dsjourney.spark.read_text_lines` routes through the CSV reader with a
separator that cannot occur in prose, and takes the encoding as an argument.
The same defect costs far more in the word-count exercise from the same
notebook: `book.txt` is cp1252 and 269 of its lines fail a strict UTF-8 read,
so every curly apostrophe becomes a replacement character and `don't` is
counted as two words.

`test_the_names_file_is_not_utf8` asserts the file really is undecodable as
UTF-8, so the declared encoding cannot decay into a comment that nobody
verifies.

## Degrees of separation

The notebook stopped at "who is most popular". The natural next question is how
far apart the characters are, which needs an iterative traversal - one
distributed join per level, in `bfs_distances`.

From Captain America:

| Hops | Characters |
|---|---|
| 0 | 1 |
| 1 | 1,933 |
| 2 | 4,477 |
| 3 | 38 |
| unreachable | 37 |

Mean distance 1.71, eccentricity 3. Essentially the whole Marvel universe is
within three hops of Captain America, and two thirds of it within two. The 37
unreachable characters include the 19 with no co-appearances at all.

The root matters: this is the most connected character in the graph, so it is
the best case. `--root` takes any id.

Türkçe açıklamalar: [README.tr.md](README.tr.md)

## Was Spark worth starting?

`--benchmark` runs the identical degree computation through both engines. The
graph is 1.6 MB, which is the point: this is the size the notebook used to
demonstrate a tool built for terabytes.

| | Seconds |
|---|---|
| Spark session start | 1.97 |
| First trivial Spark job (`spark.range(1).count()`) | 1.49 |
| Degree computation, Spark, session already warm | 0.126 |
| Degree computation, pandas | 0.013 |
| **Whole task in pandas, from a cold Python process** | **0.39** |

pandas is **9.6x faster** on the computation itself, and about **10x faster**
on the end-to-end task once the JVM's three and a half seconds of start-up are
counted. Spark spends more time getting ready than pandas spends finishing.

None of this is an argument against Spark. It is the argument for knowing where
the crossover is: Spark pays for its overhead when the data no longer fits in
memory on one machine, and 336,534 pairs is four orders of magnitude short of
that. The notebook opened by asserting Spark is "up to 100x faster" and then
ran every example at this scale without timing any of them, which is how a
reader ends up reaching for a cluster to process a spreadsheet.

The code is worth writing this way regardless - `bfs_distances` would run
unchanged on a graph a thousand times larger, and that portability is what the
API is for.
