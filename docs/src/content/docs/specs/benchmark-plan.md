---
title: "Benchmark Strategy"
description: "How we measure CVX's advantages: functional benchmarks, competitive comparison, and datasets"
---

## Guiding Principle

> **"Benchmark what matters, not what's easy."**

We do not measure operations that every database handles identically (e.g., health check latency). We measure the operations where CVX should excel (temporal queries, trajectory analytics) and the operations where CVX must *not lose* (vanilla kNN, ingest throughput).

---

## Benchmark Categories

### Category A: Unique Capabilities (CVX-Only)

These benchmarks measure operations that **no existing vector database supports natively**. The comparison baseline is an ad-hoc solution a user would have to build on top of a generic VDB.

| Benchmark | What It Measures | Key Metric | Target |
|-----------|-----------------|------------|--------|
| **A1: Temporal kNN** | Snapshot kNN with native temporal awareness vs. Qdrant with timestamp as payload filter | Recall@10 vs. ground truth temporal kNN | CVX recall $\geq 0.95$ |
| **A2: Trajectory Reconstruction** | Retrieve full entity trajectory (single call) vs. N individual point lookups | Latency and storage comparison | $\geq 5\times$ faster, $3\text{-}5\times$ less storage |
| **A3: Change Point Detection** | Detect semantic shifts in entity trajectories | F1 score vs. planted ground truth | PELT F1 $\geq 0.85$ |
| **A4: Drift Attribution** | Measure and explain concept drift | Correlation with known semantic shifts | Top-10 dimensions capture $\geq 80\%$ of shifts |
| **A5: Prediction (Neural ODE)** | Predict future vector states | MSE vs. linear extrapolation baseline | $\geq 15\%$ lower MSE |
| **A6: Temporal Analogy** | "What in 2018 played the role that X does in 2024?" | MRR on curated analogy dataset | MRR $\geq 0.4$ |

### Category B: Competitive Parity (CVX vs. Qdrant)

These benchmarks demonstrate that CVX does **not sacrifice base performance** by adding temporal capabilities. The competitor is Qdrant (latest stable release) on identical hardware.

| Benchmark | What It Measures | Key Metric | Target |
|-----------|-----------------|------------|--------|
| **B1: Vanilla kNN** | Pure semantic kNN ($\alpha = 1.0$, no temporal component) | QPS at recall@10 $\geq 0.95$ | Within 80% of Qdrant's QPS |
| **B2: Ingest Throughput** | Sustained vector insertion rate | Vectors/second over 10M inserts | $\geq 50$K vectors/sec |
| **B3: Memory Efficiency** | RAM usage per million vectors | RSS per million vectors | Within $\pm 20\%$ of Qdrant |
| **B4: Concurrent Queries** | Query latency under concurrent load | Latency p50/p99 at 1-100 concurrent queries | Similar degradation curve |

### Category C: Storage Efficiency

| Benchmark | What It Measures | Key Metric | Target |
|-----------|-----------------|------------|--------|
| **C1: Delta Compression** | Storage savings from delta encoding vs. full vector storage | Compression ratio by drift rate | $\geq 3\times$ for slow drift |
| **C2: Tiered Storage** | Total storage cost across hot/warm/cold tiers | Cold tier size vs. hot tier | Cold $< 5\%$ of hot with recall $\geq 0.90$ |

### Category D: Stochastic Analytics

| Benchmark | What It Measures | Key Metric | Target |
|-----------|-----------------|------------|--------|
| **D1: Characterization Accuracy** | Classification accuracy of drift significance, mean reversion, Hurst exponent | Correct classification on synthetic processes | $\geq 95\%$ accuracy |
| **D2: Signature Quality** | Can signature-based kNN find trajectories with similar dynamics? | Recall@10 for same-pattern trajectories | $\geq 85\%$ recall |
| **D3: Neural SDE Calibration** | Does Neural SDE provide better calibrated uncertainty than Neural ODE? | % of true values within 95% confidence interval | Neural SDE calibration $\geq 90\%$ |

---

## Datasets

### Wikipedia Temporal Embeddings

Monthly snapshots of Wikipedia articles from 2018-2025, embedded with a sentence transformer. Known events (COVID-19, the Ukraine conflict, the AI boom) create natural ground-truth change points. Available in three sizes:

| Subset | Articles | Months | Total Points |
|--------|----------|--------|-------------|
| Small | 10K | 84 | ~840K |
| Medium | 100K | 84 | ~8.4M |
| Large | 500K | 84 | ~42M |

### Synthetic Planted Drift

Random walks in $D = 768$ with planted abrupt shifts at known timestamps. Parameters vary across drift rate ($\sigma \in [0.001, 0.05]$), change point magnitude ($0.1\text{-}2.0 \times \sigma$), number of change points (0-5), and trajectory length (100-100K). Total: 40K trajectories with ground truth.

### ArXiv Temporal Embeddings

ArXiv paper abstracts embedded with SPECTER2 ($D = 768$), with annual snapshots from 2010-2025. Ground truth includes known field evolution patterns: RNN to LSTM to Transformer, SVM to DNN to Foundation Models. Used primarily for temporal analogy benchmarks.

### Synthetic Uniform

Random uniform vectors in $D = 768$ with no temporal component. Used for apples-to-apples comparison with Qdrant on vanilla kNN, following the ann-benchmarks methodology. Sizes: 1M, 5M, 10M vectors.

---

## Fair Comparison Methodology

Every competitive benchmark follows these rules:

1. **Identical hardware.** Both CVX and Qdrant run on the same machine with the same resource limits.
2. **Default configurations.** Both systems use default settings unless tuning is explicitly part of the benchmark.
3. **Equivalent index parameters.** Qdrant uses HNSW with comparable $M$ and `ef_construction` values.
4. **Warm-up period.** The first 10% of queries are discarded before measurement begins.
5. **Statistical rigor.** Minimum 5 repetitions per measurement. Report median, p95, and p99 — not mean. Include 95% confidence intervals on all comparative claims.
6. **Version documentation.** Every run records the exact CVX commit hash and Qdrant version.
7. **Reproducibility.** All scripts, dataset generators (with fixed random seeds), and Docker Compose configurations are in the repository.

---

## Success Criteria

| Criterion | Target | Status |
|-----------|--------|--------|
| CVX temporal kNN recall $\geq$ Qdrant post-filter | recall@10 $\geq 0.95$ | Pending |
| CVX trajectory retrieval faster than N Qdrant lookups | $\geq 5\times$ speedup | Pending |
| PELT F1 on planted changes | $\geq 0.85$ | Pending |
| Neural ODE $<$ linear extrapolation error | $\geq 15\%$ lower MSE | Pending |
| CVX vanilla kNN within 80% of Qdrant QPS | At equivalent recall | Pending |
| Delta encoding $\geq 3\times$ compression | On slow-drift data | Pending |
| Cold tier $< 5\%$ of hot tier storage | With recall $\geq 0.90$ | Pending |
| Stochastic classification accuracy | $\geq 95\%$ on synthetic data | Pending |
| All benchmarks reproducible in CI | Green on weekly run | Pending |

---

## Infrastructure

### Benchmark Runner Structure

```
benches/
├── datasets/                          # Dataset generation scripts
│   ├── wikipedia_temporal.py
│   ├── synthetic_drift.py
│   ├── arxiv_temporal.py
│   └── synthetic_uniform.py
├── criterion/                         # Rust micro-benchmarks
│   ├── distance_kernels.rs
│   ├── hnsw_search.rs
│   ├── delta_encoding.rs
│   └── pelt.rs
├── integration/                       # Full system benchmarks
│   ├── temporal_knn_vs_qdrant.py      # A1
│   ├── trajectory_efficiency.py       # A2
│   ├── cpd_accuracy.py                # A3
│   ├── vanilla_knn.py                 # B1
│   └── ...
└── reports/
    └── generate_report.py             # Comparison charts
```

### CI Integration

| Mode | Trigger | Duration | Scope |
|------|---------|----------|-------|
| **Quick** | PRs touching `benches/` | ~5 min | Criterion micro-benchmarks only |
| **Full** | Weekly schedule + releases | ~60 min | All categories A/B/C/D with Qdrant comparison |

Criterion benchmarks track results across git commits and alert when performance regresses by more than 5%.

### Reporting

Each benchmark run produces three outputs:

1. **JSON results** — machine-readable for historical tracking
2. **Markdown summary** — human-readable for PR comments and the project README
3. **PNG charts** — visual comparisons (recall-QPS curves, compression ratios, latency distributions, F1 scores)
