---
title: "Anomaly Detection (NAB)"
description: "Trajectory-geometric anomaly detection on the Numenta Anomaly Benchmark"
---

> **Notebook:** [`notebooks/T_NAB_Anomaly.ipynb`](https://github.com/manucouto1/chronos-vector/blob/develop/notebooks/T_NAB_Anomaly.ipynb)

## Abstract

Anomaly detection in time series is traditionally framed as a statistical outlier problem — deviations from a learned normal distribution. ChronosVector reframes this as **trajectory geometry**: anomalies are points where the trajectory's geometric properties — velocity, curvature, anchor deviation, or topological structure — deviate from expected behavior.

We evaluate CVX on the Numenta Anomaly Benchmark (NAB), a standard benchmark comprising 58 time series across 7 domains (cloud metrics, traffic, tweets, temperature, etc.) with 116 labeled anomaly windows. CVX applies four complementary detection strategies: (1) velocity spikes indicating sudden trajectory acceleration, (2) anchor deviation from a learned normal-behavior reference, (3) PELT changepoints marking regime shifts, and (4) topological disruption via persistence-based features. The pipeline is functional on **31 series** with multi-threshold evaluation. Scoring refinement against NAB's weighted scoring protocol is ongoing.

---

## Related Work

### The NAB Benchmark

Lavin and Ahmad (2015) introduced the Numenta Anomaly Benchmark as a standardized evaluation framework for real-time anomaly detection. NAB provides a **scoring protocol** that rewards early detection (detecting an anomaly before its labeled window) and penalizes false positives with configurable profiles (standard, reward low FP, reward low FN). Leading methods on NAB include:

- **HTM (Hierarchical Temporal Memory)**: Numenta's own cortically-inspired model, which learns temporal sequences and flags prediction errors.
- **Random Cut Forest (RCF)**: Amazon's ensemble of random trees that isolates anomalies via path length (Guha et al., 2016).
- **Twitter ADVec**: Seasonal-hybrid ESD for detecting anomalies in periodic signals.

### Delay Embeddings and Takens' Theorem

Takens' theorem (1981) establishes that a time series can be embedded into a higher-dimensional phase space via delay coordinates, reconstructing the topology of the underlying dynamical system. This provides theoretical grounding for CVX's approach: by constructing delay-embedded trajectories from univariate series, we gain access to geometric properties (velocity, curvature, attractor structure) that are invisible in the original 1D signal.

### Topological Data Analysis for Anomalies

Persistent homology has been applied to anomaly detection by tracking the birth/death of topological features (connected components, loops) in sliding windows over time series (Perea & Harer, 2015). Anomalies correspond to sudden changes in the persistence diagram — new topological features appearing or disappearing.

**CVX's Contribution.** CVX unifies delay embedding, velocity/acceleration analysis, changepoint detection, and topological features in a single trajectory-native engine. Rather than treating these as separate pipelines, CVX computes them as complementary views of the same underlying trajectory geometry.

---

## Methodology

### Delay Embedding Construction

Each univariate NAB time series is converted to a trajectory via delay embedding:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Window (W)** | 20 | Captures sufficient temporal context for most NAB series |
| **Stride** | 1 | Maximum resolution |
| **Dimensionality** | D = 20 | Each point is a length-20 window of the original series |

This transforms a 1D time series of length N into a trajectory of N-W+1 points in R^20.

### Detection Strategies

CVX applies four independent anomaly scoring strategies:

#### Strategy 1: Velocity Spikes

Compute `velocity()` between consecutive trajectory points. Anomalies produce sudden accelerations — the trajectory moves faster than expected given the learned baseline dynamics.

#### Strategy 2: Anchor Deviation

Define a **normal-behavior anchor** from the first 15% of each series (assumed anomaly-free). Compute `drift()` from this anchor at each time step. Anomalies appear as sudden increases in anchor distance.

#### Strategy 3: Changepoint Detection

Apply `detect_changepoints()` (PELT) to the trajectory. Changepoints mark moments where the trajectory's statistical properties shift — mean, variance, or direction. Each changepoint receives a severity score proportional to the geometric displacement.

#### Strategy 4: Topological Disruption

Compute local persistence features using sliding windows over the trajectory. Anomalies correspond to sudden changes in the topological structure — new cycles appearing or the attractor geometry deforming.

### Combined Scoring

The four strategy scores are combined via weighted averaging:

```
score(t) = w1 * velocity_score(t) + w2 * anchor_score(t)
         + w3 * changepoint_score(t) + w4 * topology_score(t)
```

Multiple threshold levels are evaluated against NAB's scoring protocol to generate detection windows.

### CVX Functions Used

| CVX Function | Purpose | Strategy |
|-------------|---------|----------|
| `cvx.ingest()` | Load delay-embedded vectors | All |
| `cvx.velocity()` | Trajectory speed between steps | Velocity Spikes |
| `cvx.drift()` | Distance from normal anchor | Anchor Deviation |
| `cvx.detect_changepoints()` | PELT regime boundaries | Changepoints |
| `cvx.hurst_exponent()` | Memory structure per window | Topology |
| `cvx.trajectory()` | Full path extraction | All |
| `cvx.path_signature()` | Local geometric features | Topology |

---

## Key Results

### Pipeline Status

The CVX anomaly detection pipeline is **functional on 31 of 58 NAB series** (the remaining 27 require domain-specific preprocessing for periodic signals). Current results represent the combined scoring approach with default weights.

### Per-Domain Performance

| NAB Domain | Series | Detected | Coverage |
|-----------|-------:|---------:|---------:|
| Cloud (AWS) | 17 | 14 | 82% |
| Traffic | 7 | 6 | 86% |
| Tweets | 10 | 5 | 50% |
| Temperature | 3 | 2 | 67% |
| CPU/Machine | 6 | 4 | 67% |
| Exchange Rate | 5 | 0 | 0% |
| Art. (no anomaly) | 10 | — | — |

### Strategy Contribution Analysis

| Strategy | Avg Precision | Avg Recall | Best Domain |
|----------|-------------:|----------:|-------------|
| Velocity Spikes | 0.68 | 0.42 | Cloud metrics |
| Anchor Deviation | 0.55 | 0.61 | Traffic |
| Changepoints | 0.72 | 0.38 | Temperature |
| Topological | 0.48 | 0.35 | Tweets |
| **Combined** | **0.63** | **0.51** | **Cloud** |

### Ongoing Refinements

- **Scoring calibration**: NAB's weighted scoring protocol penalizes late detection; current thresholds need tuning per domain.
- **Periodic series**: Tweet counts and some traffic series have strong seasonality that the delay embedding does not yet detrend.
- **Exchange rate series**: Very low signal-to-noise ratio; anomalies are subtle distributional shifts rather than geometric discontinuities.

---

## Notebook Plots

The notebook produces the following interactive visualizations:

- **Detection Timeline**: Original series with overlaid velocity, anchor deviation, and changepoint scores
- **Delay Embedding 3D**: PCA projection of the 20-D trajectory colored by anomaly score
- **Strategy Comparison**: Side-by-side score profiles for each detection strategy
- **Threshold Sensitivity**: Precision-recall curves at multiple detection thresholds

---

## Running the Notebook

```bash
# Install dependencies
pip install chronos-vector NAB plotly scikit-learn

# Run analysis
cd notebooks && jupyter notebook T_NAB_Anomaly.ipynb
```

**Requirements:** ~4 GB RAM for loading all NAB series, ~10 min CVX ingestion.

---

## Further Reading

- [Theoretical Foundations](/research/foundations/) — Delay embeddings and dynamical systems theory
- [Use Cases & Applications](/research/use-cases/) — Domain overview including anomaly detection
- [Stochastic Processes](/research/stochastic-processes/) — Hurst exponent and long-memory processes
