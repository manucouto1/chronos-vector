---
title: "Political Rhetoric & Market Impact"
description: "Analyzing Trump's Twitter trajectory and its temporal alignment with economic indicators"
---

> **Notebook:** [`notebooks/B3_trump_impact.ipynb`](https://github.com/manucouto1/chronos-vector/blob/develop/notebooks/B3_trump_impact.ipynb)

## Abstract

Political rhetoric has measurable effects on financial markets, yet quantifying this relationship requires tracking how rhetorical themes evolve over time and aligning those trajectories with economic indicators. Traditional approaches reduce political text to sentiment scores, discarding the rich multidimensional structure of political discourse.

ChronosVector (CVX) tracks Donald Trump's rhetorical trajectory through **28,000 tweets** (2015-2021) embedded with MiniLM (D=384) into a continuous semantic space. Six semantic anchors — **economy**, **trade war**, **immigration**, **media criticism**, **self-praise**, and **threat/warning** — project each tweet into interpretable rhetorical dimensions via `drift()`. Daily aggregation produces rhetorical trajectories that are temporally aligned with five economic indicators: S&P 500, VIX, crude oil, USD index, and 10-year Treasury yields. CVX's changepoint detection and velocity analysis identify **rhetorical regime shifts** and their temporal relationship to market movements, while path signatures capture the geometric shape of tweet-storm episodes.

---

## Related Work

### Presidential Communication and Markets

Bianchi et al. (2019) demonstrated that Federal Reserve communication systematically affects asset prices, establishing that textual analysis of institutional discourse has predictive value for financial markets. Bollen et al. (2011) showed that aggregate Twitter mood (measured via OpinionFinder and GPOMS) is Granger-causal to Dow Jones movements with 87.6% directional accuracy.

Specific to presidential communication, Colonescu (2018) found that Trump's tweets about specific companies produced statistically significant abnormal returns in the 30 minutes following the tweet. Ge et al. (2020) extended this to show that tweet sentiment toward trade policy predicted next-day VIX movements.

### NLP for Political Discourse

Rhetorical analysis of political text has evolved from keyword counting to contextual embeddings. Card et al. (2015) introduced framing analysis using latent dimensions, while Gentzkow et al. (2019) used word embeddings to measure political polarization over a century of Congressional records. Transformer-based embeddings (BERT, sentence-transformers) now enable fine-grained tracking of semantic shifts in political language without manual coding.

### Temporal Alignment of Heterogeneous Signals

Cross-correlation and Granger causality are standard tools for assessing lead-lag relationships between time series. Dynamic Time Warping (DTW) handles non-linear temporal alignment but assumes a single alignment path. CVX's approach — computing simultaneous trajectories in their respective spaces and analyzing geometric correspondence — preserves the full temporal structure of both signals.

**CVX's Contribution.** CVX provides a complete pipeline: embed political text, project onto interpretable anchors, construct temporal trajectories, align with economic indicators, and detect co-occurring regime shifts — all within a trajectory-native framework.

---

## Methodology

### Tweet Corpus Construction

| Component | Detail |
|-----------|--------|
| **Source** | Trump Twitter Archive (2015-2021) |
| **Tweets** | ~28,000 original tweets (excluding retweets) |
| **Embedding model** | `all-MiniLM-L6-v2` (D=384) |
| **Temporal resolution** | Daily aggregation (mean embedding per day) |
| **Active days** | ~2,100 days with at least one tweet |

### Semantic Anchor Definition

Six anchors are defined as mean embeddings of curated seed phrases:

| Anchor | Seed Phrases (examples) | Dimension Captured |
|--------|------------------------|-------------------|
| **Economy** | "economy growing", "jobs report", "stock market record" | Economic boasting/commentary |
| **Trade War** | "China tariffs", "trade deal", "unfair trade" | Trade policy rhetoric |
| **Immigration** | "border wall", "illegal immigration", "caravan" | Immigration framing |
| **Media** | "fake news", "corrupt media", "enemy of the people" | Press antagonism |
| **Self-Praise** | "I alone", "nobody has done more", "tremendous success" | Self-aggrandizement |
| **Threat** | "will pay a price", "big consequences", "not tolerate" | Warning/escalation language |

Each tweet is projected onto all 6 anchors via `drift()`, producing a daily 6-dimensional rhetorical trajectory.

### Economic Alignment

Five economic indicators are aligned to the rhetorical trajectory:

| Indicator | Source | Frequency |
|-----------|--------|-----------|
| S&P 500 (close) | Yahoo Finance | Daily |
| VIX (close) | CBOE | Daily |
| Crude Oil (WTI) | EIA | Daily |
| USD Index (DXY) | ICE | Daily |
| 10Y Treasury Yield | FRED | Daily |

Alignment uses same-day and lagged cross-correlation (lags of 1, 3, 5, 10 trading days) to identify lead-lag relationships between rhetorical shifts and market movements.

### CVX Pipeline

1. **Embed**: MiniLM encodes each tweet to D=384.
2. **Ingest**: Daily mean embeddings ingested into CVX graph.
3. **Anchor Projection**: `drift()` to 6 semantic anchors produces a 6-D rhetorical trajectory.
4. **Changepoint Detection**: `detect_changepoints()` on each anchor dimension identifies rhetorical regime shifts.
5. **Velocity Analysis**: `velocity()` captures tweet-storm intensity — periods of rapid rhetorical acceleration.
6. **Economic Alignment**: Rhetorical changepoints temporally aligned with market indicator changepoints.
7. **Signature Analysis**: `path_signature(depth=2)` on rhetorical trajectories during key event windows.

### CVX Functions Used

| CVX Function | Purpose | Parameters |
|-------------|---------|------------|
| `cvx.ingest()` | Load daily mean embeddings | `dim=384, metric="cosine"` |
| `cvx.drift()` | Anchor-relative rhetorical distance | 6 semantic anchors |
| `cvx.detect_changepoints()` | Rhetorical regime transitions | `min_segment=14` |
| `cvx.velocity()` | Tweet-storm intensity | Daily resolution |
| `cvx.hurst_exponent()` | Rhetorical persistence | `window=90` |
| `cvx.path_signature()` | Event-window fingerprints | `depth=2` |
| `cvx.trajectory()` | Full rhetorical path | Per-anchor dimension |

---

## Key Results

### Rhetorical Dimensions

The 6-anchor projection captures distinct and interpretable rhetorical dimensions across the 2015-2021 period:

| Anchor | Mean Drift | Std | Trend (2015-2021) |
|--------|----------:|----:|-------------------|
| Economy | 0.42 | 0.11 | Increasing during presidency |
| Trade War | 0.61 | 0.15 | Peak 2018-2019, decline after Phase 1 |
| Immigration | 0.55 | 0.13 | Peaks around caravan events |
| Media | 0.38 | 0.09 | Steadily increasing |
| Self-Praise | 0.34 | 0.08 | Relatively stable |
| Threat | 0.58 | 0.14 | Episodic spikes around geopolitical events |

### Anchor Trajectory Trends

- **Trade war rhetoric** shows a clear inverted-U pattern: rising through 2018, peaking during tariff escalation (mid-2019), and declining after the Phase 1 deal (January 2020).
- **Media criticism** exhibits a persistent upward trend (Hurst=0.81), consistent with escalating press antagonism throughout the presidency.
- **Economy anchor** diverges sharply during COVID-19 — a period where economic boasting temporarily disappears from the rhetorical trajectory.

### Changepoint Alignment

Rhetorical changepoints show temporal proximity to economic regime shifts:

| Rhetorical Changepoint | Anchor | Market Event (nearest) | Lag (days) |
|------------------------|--------|----------------------|----------:|
| 2018-03-01 | Trade War | Steel tariff announcement | 0 |
| 2018-07-06 | Trade War | China tariff escalation | -2 |
| 2020-03-11 | Economy | COVID market crash | +1 |
| 2020-11-04 | Self-Praise | Election uncertainty peak | 0 |

### Current Status

The 6 rhetorical dimensions are captured and the anchor trend analysis is complete for the full 2015-2021 window. Cross-correlation with economic indicators reveals several statistically suggestive lead-lag relationships, particularly for trade-war rhetoric and VIX. Formal Granger causality testing and event-study methodology are planned extensions.

---

## Notebook Plots

The notebook produces the following interactive visualizations:

- **Rhetorical Radar**: 6-anchor drift over time as an animated radar chart
- **Anchor Trajectories**: Per-anchor drift time series with changepoint markers
- **Economic Alignment**: Dual-axis plots pairing rhetorical dimensions with market indicators
- **Tweet Storm Velocity**: Velocity spikes overlaid on daily tweet volume
- **Signature PCA**: 3D projection of event-window signatures colored by rhetorical regime

---

## Running the Notebook

```bash
# Install dependencies
pip install chronos-vector sentence-transformers yfinance plotly pandas

# Run analysis
cd notebooks && jupyter notebook B3_trump_impact.ipynb
```

**Requirements:** ~4 GB RAM for embeddings, ~20 min for full embedding + CVX ingestion pipeline. Pre-computed embeddings available in `data/embeddings/`.

---

## Further Reading

- [Theoretical Foundations](/chronos-vector/research/foundations/) — Temporal embedding theory and anchor projection
- [Use Cases & Applications](/chronos-vector/research/use-cases/) — Domain overview including NLP and finance
- [Path Signatures](/chronos-vector/research/path-signatures/) — Signature theory for trajectory fingerprinting
