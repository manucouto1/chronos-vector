# ChronosVector — Benchmark & Competitive Evaluation Plan

**Version:** 1.0
**Author:** Manuel Couto Pintos
**Date:** March 2026
**Status:** Draft
**Dependencies:** PRD §3.1 (Performance Targets), Architecture Doc §7-10, Roadmap Layer 12

---

## 1. Objectives

Este documento define la estrategia de benchmarking de ChronosVector en dos ejes:

1. **Benchmarks funcionales:** Demostrar que CVX resuelve problemas que los VDB existentes *no pueden* resolver, o resuelve mucho mejor.
2. **Benchmarks de rendimiento:** Demostrar que CVX no sacrifica rendimiento base (kNN puro) al añadir capacidades temporales.

### 1.1 Principio Rector

> **"Benchmark what matters, not what's easy."** No medimos operaciones que todos hacen igual (e.g., health check latency). Medimos las operaciones donde CVX debería brillar (temporal queries) y las operaciones donde CVX debe *no perder* (vanilla kNN).

### 1.2 Audiencia

| Audiencia | Lo que quiere ver |
|-----------|-------------------|
| **Hiring Manager / Portfolio Reviewer** | Gráficos comparativos claros, CVX vs alternativas |
| **Potential User (ML Engineer)** | "¿Es lo suficientemente rápido para producción?" |
| **Researcher** | Métricas rigurosas con intervalos de confianza, datasets reproducibles |
| **Contributor** | Benchmark suite que puede ejecutar localmente para validar PRs |

---

## 2. Benchmark Categories

### 2.1 Category A: Unique Capabilities (CVX-Only)

Estas son operaciones que *ningún* VDB existente soporta nativamente. La comparación es contra **soluciones ad-hoc** que un usuario tendría que construir sobre un VDB genérico.

#### A1: Temporal kNN vs Payload Filtering

| Aspect | Specification |
|--------|---------------|
| **What** | Snapshot kNN con conciencia temporal nativa vs Qdrant con timestamp como payload filter |
| **Hypothesis** | CVX con distancia compuesta (α=0.7) produce resultados más relevantes temporalmente que Qdrant con pre-filter |
| **Dataset** | Wikipedia temporal embeddings (ver §3.1) |
| **Metric** | - Recall@10 vs ground truth temporal kNN<br>- Latency p50/p99<br>- Relevance score: weighted combination of semantic similarity + temporal proximity |
| **Protocol** | 1. Load same 1M vectors in both CVX and Qdrant.<br>2. 1000 random queries with timestamp.<br>3. Ground truth: brute-force temporal kNN.<br>4. Compare recall and relevance. |
| **Expected Result** | CVX recall ≥ 0.95, Qdrant post-filter recall ≥ 0.90 but lower temporal relevance |

#### A2: Trajectory Reconstruction Efficiency

| Aspect | Specification |
|--------|---------------|
| **What** | Retrieve full trajectory of an entity over time |
| **Hypothesis** | CVX (native trajectory + delta encoding) is faster and uses less storage than N individual fetches on Qdrant |
| **Dataset** | 10K entities, 100 snapshots each, D=768 |
| **Metric** | - Latency: single CVX trajectory call vs N Qdrant point lookups + app-level sort<br>- Storage: CVX (with deltas) vs Qdrant (full vectors) |
| **Protocol** | 1. Load 1M vectors (10K entities × 100 timestamps) in both systems.<br>2. Retrieve 1000 random entity trajectories.<br>3. Measure latency distribution and total storage size. |
| **Expected Result** | CVX 5-10x faster (single call vs 100 calls), 3-5x less storage (delta encoding) |

#### A3: Change Point Detection Accuracy

| Aspect | Specification |
|--------|---------------|
| **What** | Detect semantic shifts in entity trajectories |
| **Hypothesis** | CVX's native PELT/BOCPD detects change points with high accuracy |
| **Dataset** | Synthetic planted changes (ver §3.2) + real Wikipedia temporal data |
| **Metric** | - F1 score vs ground truth (synthetic)<br>- Detection delay: timestamps between true change and detection<br>- False positive rate on stationary data |
| **Protocol** | 1. Generate 1000 synthetic trajectories with 0-5 planted change points each.<br>2. Run PELT offline + BOCPD online on each trajectory.<br>3. Compare detected vs true change points (within ±3 timestamps tolerance). |
| **Expected Result** | PELT F1 ≥ 0.85, BOCPD detection delay ≤ 10 observations, FPR < 5% |

#### A4: Drift Quantification & Attribution

| Aspect | Specification |
|--------|---------------|
| **What** | Measure and explain concept drift |
| **Hypothesis** | CVX provides richer drift analysis than manual distance computation |
| **Dataset** | Wikipedia articles with known semantic shifts (e.g., "corona", "transformer") |
| **Metric** | - Qualitative: does attribution correctly identify changed dimensions?<br>- Quantitative: correlation between attributed dimensions and known semantic shifts |
| **Protocol** | 1. Load temporal embeddings for 100 entities with known semantic changes.<br>2. Run drift attribution.<br>3. Expert evaluation: do top-K dimensions correspond to known semantic shifts? |
| **Expected Result** | Top-10 dimensions capture ≥ 80% of known semantic shifts |

#### A5: Prediction Accuracy (Neural ODE)

| Aspect | Specification |
|--------|---------------|
| **What** | Predict future vector states |
| **Hypothesis** | Neural ODE produces better predictions than linear extrapolation |
| **Dataset** | Held-out future snapshots from Wikipedia temporal embeddings |
| **Metric** | - MSE vs ground truth future vector<br>- Cosine similarity between predicted and actual<br>- Comparison vs baselines: linear extrapolation, historical mean, last-value |
| **Protocol** | 1. Train on first 80% of temporal data.<br>2. Predict vectors at timestamps in last 20%.<br>3. Compare against actual vectors at those timestamps. |
| **Expected Result** | Neural ODE MSE < linear extrapolation MSE by ≥ 15% |

#### A6: Temporal Analogy Quality

| Aspect | Specification |
|--------|---------------|
| **What** | "What in 2018 played the role that X does in 2024?" |
| **Hypothesis** | CVX temporal analogy produces semantically meaningful results |
| **Dataset** | ArXiv embeddings by year |
| **Metric** | - Qualitative evaluation by domain experts<br>- MRR (Mean Reciprocal Rank) against known analogies |
| **Protocol** | 1. Curate 50 known temporal analogies (e.g., "LSTM in 2016 → Transformer in 2020").<br>2. Run temporal analogy queries.<br>3. Measure rank of correct answer in results. |
| **Expected Result** | MRR ≥ 0.4 on curated analogy dataset |

### 2.2 Category B: Competitive Parity (CVX vs Existing VDBs)

Demostrar que CVX no sacrifica rendimiento base al añadir capacidades temporales.

#### B1: Vanilla kNN Throughput & Latency

| Aspect | Specification |
|--------|---------------|
| **What** | Pure semantic kNN (α=1.0, no temporal component) |
| **Competitor** | Qdrant (latest stable release) |
| **Dataset** | 1M random vectors D=768 (uniform distribution) |
| **Metric** | - QPS at recall@10 ≥ 0.95<br>- Latency p50, p95, p99<br>- Throughput vs recall tradeoff curve (varying ef_search) |
| **Protocol** | ann-benchmarks methodology: load, build index, query with varying ef_search, plot recall-QPS curve |
| **Target** | CVX within 80% of Qdrant's QPS at equivalent recall |

#### B2: Ingest Throughput

| Aspect | Specification |
|--------|---------------|
| **What** | Sustained vector insertion rate |
| **Competitor** | Qdrant |
| **Dataset** | 10M vectors D=768, batch size 1000 |
| **Metric** | - Vectors/second sustained over 10M inserts<br>- Latency per batch p50/p99<br>- Memory usage during ingestion |
| **Target** | ≥ 50K vectors/sec (within 70% of Qdrant) |

#### B3: Memory Efficiency

| Aspect | Specification |
|--------|---------------|
| **What** | RAM usage per million vectors |
| **Competitor** | Qdrant |
| **Dataset** | 1M, 5M, 10M vectors D=768 |
| **Metric** | - RSS per million vectors<br>- Index overhead (bytes per vector beyond raw vector storage) |
| **Target** | Comparable to Qdrant (±20%) for pure storage; *better* for temporal data due to delta encoding |

#### B4: Concurrent Query Performance

| Aspect | Specification |
|--------|---------------|
| **What** | Query latency under concurrent load |
| **Competitor** | Qdrant |
| **Dataset** | 1M vectors D=768 |
| **Metric** | - Latency p50/p99 at 1, 10, 50, 100 concurrent queries<br>- Throughput saturation point |
| **Target** | Similar degradation curve to Qdrant |

### 2.3 Category C: Storage Efficiency

#### C1: Delta Encoding Compression

| Aspect | Specification |
|--------|---------------|
| **What** | Storage savings from delta encoding vs full vector storage |
| **Dataset** | Wikipedia temporal embeddings with varying drift rates |
| **Metric** | - Compression ratio: full_size / delta_size<br>- Reconstruction latency per vector<br>- Compression ratio vs drift rate curve |
| **Expected Result** | ≥ 3x compression for slow drift, ≥ 1.5x for fast drift |

#### C2: Tiered Storage Efficiency

| Aspect | Specification |
|--------|---------------|
| **What** | Total storage cost across hot/warm/cold tiers |
| **Dataset** | 100M vectors with 90-day history |
| **Metric** | - Total bytes: hot + warm + cold<br>- vs baseline: all vectors in hot storage<br>- PQ recall at various compression levels |
| **Expected Result** | Cold tier uses < 5% of hot tier storage with recall ≥ 0.90 |

---

## 3. Datasets

### 3.1 Wikipedia Temporal Embeddings

| Property | Value |
|----------|-------|
| **Source** | Wikipedia article dumps, monthly snapshots 2018-2025 |
| **Embedding Model** | all-MiniLM-L6-v2 (D=384) or text-embedding-3-small (D=1536, truncated to 768) |
| **Size** | ~500K articles × 84 months = ~42M temporal points |
| **Ground Truth** | Known events (COVID, Ukraine, AI boom) create natural change points |
| **Generation** | Script in `benches/datasets/wikipedia_temporal.py` |
| **Subsets** | - **small**: 10K articles × 84 months (~840K points)<br>- **medium**: 100K articles × 84 months (~8.4M points)<br>- **large**: 500K articles × 84 months (~42M points) |

### 3.2 Synthetic Planted Drift

| Property | Value |
|----------|-------|
| **Purpose** | Ground truth for change point detection benchmarks |
| **Generation** | Random walks in D=768 with planted abrupt shifts at known timestamps |
| **Parameters** | - Drift rate (σ per step): [0.001, 0.01, 0.05]<br>- Change point magnitude: [0.1, 0.5, 1.0, 2.0] × σ<br>- Number of change points per trajectory: [0, 1, 3, 5]<br>- Trajectory length: [100, 1K, 10K, 100K] |
| **Size** | 10K trajectories × 4 length configs = 40K trajectories |
| **Script** | `benches/datasets/synthetic_drift.py` |

### 3.3 ArXiv Temporal Embeddings

| Property | Value |
|----------|-------|
| **Source** | ArXiv paper abstracts, annual snapshots 2010-2025 |
| **Purpose** | Temporal analogy benchmarks, research field evolution |
| **Embedding Model** | SPECTER2 (D=768) |
| **Size** | ~2M papers, 16 annual snapshots |
| **Ground Truth** | Known field evolution: RNN→LSTM→Transformer, SVM→DNN→Foundation Models |
| **Script** | `benches/datasets/arxiv_temporal.py` |

### 3.4 Synthetic Uniform (ANN-Benchmarks Compatible)

| Property | Value |
|----------|-------|
| **Purpose** | Apples-to-apples comparison with Qdrant on vanilla kNN |
| **Generation** | Random uniform vectors D=768, no temporal component |
| **Size** | 1M, 5M, 10M vectors |
| **Queries** | 10K random query vectors, k=10, ground truth via brute force |
| **Script** | `benches/datasets/synthetic_uniform.py` |

---

## 4. Infrastructure

### 4.1 Benchmark Runner

```
benches/
├── datasets/
│   ├── wikipedia_temporal.py      # Download + embed Wikipedia snapshots
│   ├── synthetic_drift.py         # Generate planted drift data
│   ├── arxiv_temporal.py          # Download + embed ArXiv snapshots
│   ├── synthetic_uniform.py       # Generate ANN-benchmark compatible data
│   └── README.md                  # Dataset instructions
├── criterion/
│   ├── distance_kernels.rs        # SIMD distance micro-benchmarks
│   ├── hnsw_search.rs             # HNSW search throughput
│   ├── delta_encoding.rs          # Delta encode/decode throughput
│   ├── pelt.rs                    # PELT throughput
│   └── trajectory_reconstruction.rs
├── integration/
│   ├── temporal_knn_vs_qdrant.py  # A1: temporal kNN comparison
│   ├── trajectory_efficiency.py   # A2: trajectory reconstruction
│   ├── cpd_accuracy.py            # A3: change point detection
│   ├── drift_attribution.py       # A4: drift quality evaluation
│   ├── prediction_accuracy.py     # A5: Neural ODE vs baselines
│   ├── temporal_analogy.py        # A6: analogy quality
│   ├── vanilla_knn.py             # B1: pure kNN parity
│   ├── ingest_throughput.py       # B2: ingestion comparison
│   ├── memory_efficiency.py       # B3: memory comparison
│   └── concurrent_queries.py      # B4: concurrency comparison
├── reports/
│   ├── generate_report.py         # Generate comparison charts
│   └── templates/
│       └── benchmark_report.html  # Report template
└── README.md                      # How to run benchmarks
```

### 4.2 CI Integration

**GitHub Actions workflow:** `.github/workflows/bench.yml`

```yaml
# Runs on: manual trigger + weekly schedule + tag releases
# Two modes:
#   - Quick (PR): criterion micro-benchmarks only, compare against baseline
#   - Full (release): all categories A/B/C with Qdrant comparison
```

| Mode | Trigger | Duration | Categories |
|------|---------|----------|------------|
| **Quick** | PR (on `benches/` changes) | ~5 min | Criterion micro-benchmarks |
| **Full** | Weekly + releases | ~60 min | A1-A6, B1-B4, C1-C2 |

### 4.3 Hardware Requirements

| Component | Quick | Full |
|-----------|-------|------|
| **CPU** | GitHub runner (2 vCPU) | 8+ cores (dedicated or self-hosted) |
| **RAM** | 8 GB | 32 GB (for 10M vector datasets) |
| **Disk** | 10 GB SSD | 100 GB SSD |
| **Qdrant** | Not needed | Docker container alongside CVX |

---

## 5. Reporting

### 5.1 Output Format

Cada benchmark run produce:

1. **JSON results:** Machine-readable para tracking histórico.
2. **Markdown summary:** Human-readable para README / PR comments.
3. **Charts:** PNG charts para visual comparison.

### 5.2 Key Charts

| Chart | Data | Format |
|-------|------|--------|
| **Recall-QPS Curve** | B1 results at varying ef_search | Line chart, CVX vs Qdrant |
| **Temporal Relevance Score** | A1 results | Bar chart, CVX vs Qdrant |
| **Storage Compression** | C1 results | Bar chart, compression ratio by drift rate |
| **CPD F1 Score** | A3 results | Bar chart by change point magnitude |
| **Prediction Error** | A5 results | Bar chart, Neural ODE vs baselines |
| **Ingest Throughput** | B2 results | Line chart over time (sustained rate) |
| **Latency Distribution** | B1, B4 results | Histogram / CDF |
| **Memory Scaling** | B3 results | Line chart, RSS vs vector count |

### 5.3 Historical Tracking

Criterion benchmarks track results across git commits, producing alerts when performance regresses by > 5%. Full benchmark results stored in `benches/reports/history/` as timestamped JSON.

---

## 6. Benchmark Methodology

### 6.1 Statistical Rigor

- Each measurement repeated **minimum 5 times** (Criterion default: 100+ samples with warm-up).
- Report **median, p95, p99** — not mean (skewed by outliers).
- Include **95% confidence intervals** on all comparative claims.
- For recall measurements: use **10-recall@10** (standard ANN benchmarks metric).

### 6.2 Fair Comparison Protocol

- Both CVX and Qdrant run on **identical hardware** (same machine, same resources).
- Both systems use **default configurations** unless tuning is part of the benchmark.
- Qdrant uses **HNSW index** (not brute force) with comparable M and ef_construction.
- Warm-up period before measurement (discard first 10% of queries).
- Document versions: exact CVX commit hash + Qdrant version.

### 6.3 Reproducibility

- All benchmark scripts in repo.
- Dataset generation scripts with fixed random seeds.
- Docker Compose for Qdrant setup.
- `benches/README.md` with step-by-step instructions.
- CI runs produce identical results (within statistical noise).

---

## 7. Success Criteria

| Criterion | Target | Status |
|-----------|--------|--------|
| CVX temporal kNN recall ≥ Qdrant post-filter | recall@10 ≥ 0.95 | Pending |
| CVX trajectory retrieval faster than N Qdrant lookups | ≥ 5x speedup | Pending |
| PELT F1 on planted changes | ≥ 0.85 | Pending |
| Neural ODE < linear extrapolation error | ≥ 15% lower MSE | Pending |
| CVX vanilla kNN within 80% of Qdrant QPS | at equivalent recall | Pending |
| Delta encoding ≥ 3x compression | on slow-drift data | Pending |
| Cold tier < 5% of hot tier storage | with recall ≥ 0.90 | Pending |
| All benchmarks reproducible in CI | green on weekly run | Pending |

---

### 2.4 Category D: Stochastic Analytics

#### D1: Stochastic Characterization Accuracy

| Aspect | Specification |
|--------|---------------|
| **What** | Accuracy of drift significance, mean reversion, and Hurst exponent tests |
| **Dataset** | Synthetic processes: Brownian motion (H=0.5), OU process (mean-reverting), trending (H>0.5) |
| **Metric** | Classification accuracy of ProcessClassification |
| **Expected Result** | ≥ 95% correct classification on synthetic data |

#### D2: Path Signature Quality

| Aspect | Specification |
|--------|---------------|
| **What** | Can signature-based kNN find trajectories with similar dynamics? |
| **Dataset** | Synthetic trajectories with planted patterns (linear drift, mean-reversion, regime switch) |
| **Metric** | Recall@10 for trajectory similarity (same pattern class) |
| **Expected Result** | ≥ 85% recall |

#### D3: Neural SDE vs Neural ODE Prediction

| Aspect | Specification |
|--------|---------------|
| **What** | Does Neural SDE provide better calibrated uncertainty than Neural ODE? |
| **Dataset** | Held-out trajectories from Wikipedia temporal + financial data |
| **Metric** | Calibration: % of true values within 95% confidence interval |
| **Expected Result** | Neural SDE calibration ≥ 90% vs Neural ODE overconfident intervals |

---

## 8. Timeline

| Phase | When (relative to roadmap) | Scope |
|-------|---------------------------|-------|
| **Phase 1** | Layer 1-2 complete | Distance kernel micro-benchmarks, HNSW recall/latency |
| **Phase 2** | Layer 4 complete | Temporal kNN benchmarks (A1), trajectory (A2) |
| **Phase 3** | Layer 8 complete | CPD accuracy (A3), drift attribution (A4) |
| **Phase 4** | Layer 10 complete | Prediction accuracy (A5), analogy (A6) |
| **Phase 5** | Layer 12 | Full competitive comparison (B1-B4), storage efficiency (C1-C2), report generation |
