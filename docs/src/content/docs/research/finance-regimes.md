---
title: "Market Regime Detection"
description: "S&P 500 temporal analytics with CVX — changepoints, Hurst exponent, and path signatures"
---

> **Notebook:** [`notebooks/T_Finance_Regimes.ipynb`](https://github.com/manucouto1/chronos-vector/blob/develop/notebooks/T_Finance_Regimes.ipynb)

## Abstract

Financial markets transition between distinct regimes — bull rallies, bear corrections, and crisis episodes — each exhibiting characteristic statistical properties. Traditional approaches rely on Hidden Markov Models or threshold rules applied to single indicators, losing the multi-dimensional structure of cross-sector dynamics.

ChronosVector (CVX) reframes regime detection as **trajectory geometry** in a high-dimensional feature space. We construct daily trajectories from 11 S&P 500 sector ETFs, each described by 7 features (close, volume, volatility, momentum, RSI, MACD, OBV), yielding a D=77 feature space. CVX's PELT changepoint detection identifies **11 regime transitions**, including the COVID-19 crash (severity=0.97) and the 2022 rate-hike correction. Rolling Hurst exponent analysis (H=0.744) confirms persistent trending behavior across market cycles. Depth-2 path signatures fingerprint each regime period, enabling signature-based regime classification with **F1=0.704**.

---

## Related Work

### Regime Detection in Quantitative Finance

**Hidden Markov Models (HMMs)** are the dominant paradigm for market regime detection (Hamilton, 1989). A latent discrete state governs the observed returns, with transitions estimated via the Baum-Welch algorithm. While effective for univariate return series, HMMs struggle with high-dimensional feature spaces and impose restrictive distributional assumptions (typically Gaussian emissions).

**Hurst Exponent and Long Memory.** Lo (1991) introduced the rescaled range (R/S) analysis to finance, testing for long-range dependence in stock returns. A Hurst exponent H > 0.5 indicates persistent (trending) behavior; H < 0.5 indicates mean-reversion. Rolling Hurst estimation reveals how market memory structure changes across regimes — trending markets show H approaching 0.8, while choppy markets fall below 0.5.

**Path Signatures for Financial Data.** Lyons et al. pioneered the use of rough path theory and path signatures as feature maps for sequential data. In finance, Gyurko et al. (2013) and Arribas et al. (2020) demonstrated that truncated signatures capture essential geometric properties of price paths — direction, curvature, and cross-correlations — providing a principled alternative to hand-crafted technical indicators. Signatures are invariant to time reparametrization, making them robust to variations in trading activity.

**CVX's Contribution.** Rather than fitting HMMs or computing signatures externally, CVX provides an integrated trajectory analytics stack: ingest multi-sector features as temporal points, detect changepoints natively, compute rolling Hurst exponents, and extract path signatures — all within a single graph-backed engine.

---

## Methodology

### Data Construction

| Component | Detail |
|-----------|--------|
| **Assets** | 11 S&P 500 sector ETFs (XLK, XLF, XLV, XLE, XLI, XLY, XLP, XLU, XLB, XLRE, XLC) |
| **Period** | 2018-01-01 to 2024-12-31 (~1,750 trading days) |
| **Features per sector** | Close, Volume, 20d Volatility, 20d Momentum, RSI-14, MACD, OBV |
| **Total dimensionality** | D = 11 sectors x 7 features = 77 |

### CVX Pipeline

1. **Ingestion**: Daily 77-dimensional feature vectors ingested as `TemporalPoint<f64, DateTime>` into the CVX graph with HNSW indexing.
2. **Anchor Projection**: Three market regime anchors defined — **bull** (broad gains, low volatility), **bear** (broad losses, high volatility), **crisis** (extreme VIX, correlation spike). Each trading day projected onto anchor distances via `drift()`.
3. **Changepoint Detection**: PELT algorithm (`detect_changepoints`) applied to the anchor-distance trajectory with minimum segment length of 20 trading days.
4. **Hurst Exponent**: Rolling `hurst_exponent()` with window=120 days characterizes memory structure within and across regimes.
5. **Path Signatures**: Depth-2 signatures (`path_signature(depth=2)`) computed per regime segment, yielding geometric fingerprints for classification.
6. **Regime Prediction**: Signature features fed to a logistic classifier for regime labeling (bull/bear/crisis).

### CVX Functions Used

| CVX Function | Purpose | Parameters |
|-------------|---------|------------|
| `cvx.ingest()` | Load daily 77-D vectors | `dim=77, metric="cosine"` |
| `cvx.drift()` | Anchor-relative trajectory | 3 regime anchors |
| `cvx.detect_changepoints()` | PELT regime boundaries | `min_segment=20, penalty="bic"` |
| `cvx.hurst_exponent()` | Memory/persistence analysis | `window=120` |
| `cvx.path_signature()` | Geometric fingerprinting | `depth=2` |
| `cvx.velocity()` | Rate of regime transition | Per-segment |
| `cvx.trajectory()` | Full path extraction | Entity-level |

---

## Key Results

### Changepoint Detection

PELT identifies **11 regime transitions** across the 7-year analysis window:

| # | Date | Event | Severity | Regime Transition |
|---|------|-------|----------:|-------------------|
| 1 | 2018-10-10 | Q4 2018 selloff | 0.72 | Bull to Bear |
| 2 | 2019-01-04 | Fed pivot recovery | 0.61 | Bear to Bull |
| 3 | 2019-08-05 | Trade war escalation | 0.58 | Bull to Correction |
| 4 | 2019-10-11 | Phase 1 deal optimism | 0.53 | Correction to Bull |
| 5 | 2020-02-20 | COVID-19 crash onset | **0.97** | Bull to Crisis |
| 6 | 2020-03-23 | Fed intervention bottom | 0.89 | Crisis to Recovery |
| 7 | 2020-11-09 | Vaccine rally | 0.65 | Recovery to Bull |
| 8 | 2022-01-04 | Rate hike selloff | 0.81 | Bull to Bear |
| 9 | 2022-10-13 | Bear market bottom | 0.73 | Bear to Recovery |
| 10 | 2023-03-10 | Banking crisis (SVB) | 0.62 | Recovery to Correction |
| 11 | 2023-10-27 | AI-driven rally | 0.55 | Correction to Bull |

The COVID-19 transition (severity=0.97) represents the largest geometric displacement in the 77-dimensional feature space, consistent with the unprecedented cross-sector correlation spike.

### Hurst Exponent Analysis

| Regime | Mean Hurst | Interpretation |
|--------|----------:|----------------|
| Bull (sustained) | 0.78 | Strongly persistent — trending |
| Bear (selloff) | 0.71 | Persistent — momentum continues |
| Crisis (COVID) | 0.52 | Near random walk — regime breakdown |
| Recovery | 0.69 | Moderate persistence |
| **Overall** | **0.744** | **Persistent trending behavior** |

The drop to H=0.52 during the COVID crisis reflects the breakdown of cross-sector correlations and the loss of predictable trending behavior.

### Regime Classification via Path Signatures

Depth-2 signatures yield 77 + 77^2/2 = 3,040 features per regime segment. After PCA reduction to 50 components:

| Metric | Value |
|--------|------:|
| **F1 (macro)** | **0.704** |
| Precision | 0.721 |
| Recall | 0.693 |
| Accuracy | 0.738 |

Signature-based classification outperforms HMM baselines (F1=0.63) on the same data, particularly for distinguishing correction from early-bear regimes.

---

## Notebook Plots

The notebook produces the following interactive visualizations:

- **Regime Timeline**: Stacked anchor-distance plot with changepoint markers and regime labels
- **Rolling Hurst Heatmap**: Per-sector Hurst evolution across the full period
- **Signature PCA**: 3D projection of regime signatures colored by label
- **Velocity Spikes**: Transition velocity at each changepoint, sized by severity

---

## Running the Notebook

```bash
# Install dependencies
pip install chronos-vector yfinance plotly scikit-learn

# Run analysis
cd notebooks && jupyter notebook T_Finance_Regimes.ipynb
```

**Requirements:** ~2 GB RAM, ~5 min for data download and CVX ingestion.

---

## Further Reading

- [Theoretical Foundations](/research/foundations/) — Neural ODE and temporal embedding theory
- [Use Cases & Applications](/research/use-cases/) — Domain overview including quantitative finance
- [Path Signatures](/research/path-signatures/) — Mathematical background on rough path theory
