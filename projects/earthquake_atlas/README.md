# Earthquake Atlas

Fifty-two years of significant earthquakes, analysed instead of only mapped.

| | |
|---|---|
| Task | Geospatial analysis |
| Data | 23,412 quakes, M >= 5.5, 1965-2016 (USGS) + 1,000 US cities |
| **Gutenberg-Richter b** | **1.004 +/- 0.007** - literature says ~1.0 |
| Busiest 5° cell | 654 quakes near (-7.5°, 152.5°), the New Britain trench |
| Quakes within 100 km of a US city | 99; the closest, M5.9 at 2.2 km from Rosemead |
| Source | `day13-AOB-CoğrafikSistemler.ipynb` |

```bash
uv run python projects/earthquake_atlas/train.py
uv run python projects/earthquake_atlas/train.py --cell-degrees 2
```

Both data files are committed (0.9 MB); no download, no API key, no plotly.

## The notebook cannot run

Its third cell reads:

```python
from plotly.oflfine import init_notebook_mode
```

`oflfine` is a typo for `offline`, so the import raises. The cell after calls
`iplot(...)`, which is never imported from anywhere. The first map - the only
one using course data - can never have rendered from a clean run of this file;
the saved outputs are from some other session whose code was edited afterwards.

Of the eight maps that follow, five plot plotly's own bundled demo datasets -
gapminder, Montreal elections, car sharing - so most of the notebook is the
library demonstrating itself. Nothing is computed from any dataset in the
file: no count, no distance, no fit. That is the gap this project fills.

## A map is a claim; here it is as a table

`grid_density` bins the catalogue into 5-degree cells. The five busiest:

| Quakes | Cell centre | What is there |
|---|---|---|
| 654 | 7.5°S, 152.5°E | New Britain trench, Papua New Guinea |
| 594 | 12.5°S, 167.5°E | Vanuatu subduction zone |
| 588 | 37.5°N, 142.5°E | Japan trench (the 2011 Tōhoku region) |
| 583 | 22.5°S, 177.5°W | Tonga trench |
| 558 | 2.5°N, 127.5°E | Molucca Sea |

All five are western-Pacific subduction zones, which is what half a century of
seismology says they should be - and a test asserts it, so the analysis has
geography to answer to rather than a picture to admire.
`artifacts/earthquake_atlas/density_map.png` is the same information drawn.

One honesty note: a 5-degree cell shrinks towards the poles, so these counts
rank density but are not per-km² - the docstring says so, and comparing areas
would need a cos(latitude) weight.

## The join the notebook never made

It loaded 23k earthquakes and 1k US cities into the same session and never put
them together. `nearest_neighbour` does it with one BallTree query in ~50 ms:
99 quakes within 100 km of a top-1000 US city, the closest an M5.9 at 2.2 km
from Rosemead, California - and the M6.7 at 4.9 km from Tacoma is the 1965
Puget Sound earthquake, which a test pins by name.

Caveat, stated rather than hidden: the reference file is US cities only, so
for most of the planet "nearest city" means "nearest *US* city, an ocean away".
The distances are meaningful near the US and an upper bound everywhere else.

## Scored against a law of nature

Earthquake magnitudes follow the Gutenberg-Richter law, log10 N(>=M) = a - bM,
and globally b sits near 1.0 across the literature. That makes this the rare
project scored against a published constant rather than a held-out split:

```
b = 1.004 +/- 0.007  over 23,412 events   (gap from literature: 0.004)
```

The fit is Aki-Utsu maximum likelihood, not a regression through the
log-counts - least squares on cumulative counts double-counts the tail and is
the textbook way to a biased b. Two details matter and both are tested:

- **Completeness.** The catalogue only contains M >= 5.5 by construction.
  Fitting below that fits the *missing data*, not the Earth; the fit refuses
  events under the threshold.
- **The estimator is checked against synthetic catalogues** drawn from the law
  itself with known b - it must recover 0.8 as 0.8 and 1.4 as 1.4, so a
  constant-output bug cannot hide behind the plausible-looking 1.0.

`artifacts/earthquake_atlas/gutenberg_richter.png` shows the observed
cumulative counts against the fitted line.

## One more data trap

Three of the 23,412 dates are full ISO-8601 UTC timestamps in a column of
`MM/DD/YYYY` strings. A naive `to_datetime` raises on the mixed timezones; the
reflex fix, `errors="coerce"`, silently turns exactly those three rows into
NaT instead. `utc=True` parses both forms, and a test asserts all rows survive.

Türkçe açıklamalar: [README.tr.md](README.tr.md)
