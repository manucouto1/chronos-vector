---
title: "ChronosVector: Temporal Vector Analytics for Cross-Domain Intelligence"
description: "A white paper on temporal trajectory analysis across mental health, finance, anomaly detection, and political discourse"
---

## Abstract

Vector databases have become foundational infrastructure for AI applications, yet they treat entities as static points --- snapshots frozen in time. **ChronosVector (CVX)** introduces a fundamentally different paradigm: the temporal vector database, where every entity is a *trajectory* through embedding space and the database itself is the analytical engine.

CVX provides **19 native analytical functions** spanning differential calculus (velocity, drift), stochastic characterization (Hurst exponent, changepoint detection), path signatures (rough path theory), distributional distances (Wasserstein, Fisher-Rao, Hellinger), topological data analysis (persistent homology), and anchor projection. These functions compose into a 7-level analytical framework that decomposes temporal behavior from raw trajectories to topological structure.

The key innovation is **anchor projection** (RFC-006): a coordinate system transformation from opaque $\mathbb{R}^D$ embeddings into interpretable $\mathbb{R}^K$ coordinates defined by domain-specific reference points. On the eRisk 2022 depression detection task, anchor projection with DSM-5 clinical anchors achieves **F1 = 0.744** and **AUC = 0.886** on a temporal evaluation split --- a +0.144 F1 improvement over absolute-space features alone.

CVX has been validated across **6 domains** --- clinical NLP, quantitative finance, political discourse, anomaly detection, fraud detection, and insider threat --- using exclusively public datasets. All experiments are reproducible via open-source notebooks.

---

## 1. Introduction

### The Snapshot Problem

Modern vector databases (Pinecone, Milvus, Weaviate, Qdrant) excel at storing and retrieving high-dimensional embeddings. They answer the question *"what is similar to X right now?"* with remarkable efficiency. But they cannot answer questions about **change**:

- How fast is this patient's language shifting?
- When did this market regime begin?
- Is this user's behavioral trajectory consistent with insider threat patterns?
- How does political rhetoric shape financial sentiment over time?

These questions require treating entities not as points but as **trajectories** --- ordered sequences of embeddings that encode evolution, transformation, and regime dynamics.

### CVX's Approach: The Database Is the Analytical Engine

ChronosVector unifies storage and temporal analytics into a single system. Rather than exporting embeddings to external tools for analysis, CVX embeds the analytical functions directly into the database layer. The core index structure, **ST-HNSW** (Spatiotemporal Hierarchical Navigable Small World), supports both nearest-neighbor search and temporal trajectory retrieval natively.

### The 7-Level Analytical Framework

CVX's 19 functions organize into a layered framework, where each level answers progressively deeper questions about temporal behavior:

| Level | Question | CVX Functions | What It Reveals |
|-------|----------|---------------|-----------------|
| 1 | Where has the entity been? | `trajectory`, `search` | The raw path through embedding space |
| 2 | How fast is it changing? | `velocity`, `drift` | Rate and direction of transformation |
| 3 | Is change persistent or erratic? | `hurst_exponent` | Long-range dependence: trending vs. oscillating |
| 4 | When did regime transitions happen? | `detect_changepoints` | Structural breaks in behavior |
| 5 | How does the distribution transform? | `region_trajectory`, `wasserstein_drift`, `fisher_rao_distance` | Semantic migration between clusters |
| 6 | What is the *shape* of the transformation? | `path_signature`, `signature_distance` | Universal nonlinear trajectory fingerprint |
| 7 | How does the topology evolve? | `topological_features` | Fragmentation, convergence, structural change |

This layered decomposition applies identically across domains. The same `detect_changepoints` call identifies a psychiatric crisis inflection, a market regime shift, or an insider threat escalation.

---

## 2. Architecture Overview

### ST-HNSW: Spatiotemporal Index

CVX's core data structure extends the HNSW graph with temporal metadata. Each vector is annotated with timestamps and entity identifiers, enabling efficient retrieval patterns:

- **Point query:** nearest neighbors at a specific time or within a time window
- **Trajectory query:** ordered sequence of embeddings for a given entity
- **Region query:** all entities within a spatial neighborhood, partitioned by time

Temporal filtering uses **roaring bitmaps** for fast set operations on time-window predicates, avoiding full scans of the HNSW graph.

### Analytics Engine

The 19 analytical functions are implemented in Rust within the `cvx-analytics` crate, organized by mathematical domain:

| Module | Functions | Mathematical Basis |
|--------|----------|-------------------|
| `differential` | `velocity`, `drift`, `temporal_features` | Finite differences, feature engineering |
| `stochastic` | `hurst_exponent`, `detect_changepoints` | R/S analysis, PELT algorithm |
| `signatures` | `path_signature`, `log_signature`, `signature_distance` | Rough path theory (Lyons, 1998) |
| `comparison` | `frechet_distance` | Computational geometry |
| `distributional` | `wasserstein_drift`, `fisher_rao_distance`, `hellinger_distance` | Optimal transport, information geometry |
| `point_process` | `event_features` | Temporal point processes |
| `topology` | `topological_features` | Persistent homology (TDA) |
| `anchor` | `project_to_anchors`, `anchor_summary` | Coordinate system change |
| `prediction` | `predict` | Linear extrapolation / Neural ODE |

Python bindings are provided via **PyO3** through the `cvx-python` crate, exposing the full API as native Python functions with NumPy array interoperability.

### Anchor Projection: $\mathbb{R}^D \to \mathbb{R}^K$

Anchor projection (RFC-006) is a coordinate system transformation that re-expresses trajectories relative to user-defined reference points. Given $K$ anchor embeddings $\{\mathbf{a}_1, \ldots, \mathbf{a}_K\}$, each trajectory point $\mathbf{x}_t \in \mathbb{R}^D$ maps to:

$$\text{projected}_t[k] = d(\mathbf{x}_t, \mathbf{a}_k), \quad k = 1, \ldots, K$$

The result is a trajectory in $\mathbb{R}^K$ where each dimension has explicit semantic meaning (e.g., distance to "depression language", "anxiety language", "neutral language"). Crucially, the projected trajectory **composes with all existing CVX functions** --- velocity, changepoints, signatures, and topology all operate on the anchor-projected space without modification.

### Index Persistence

CVX supports save/load operations via **postcard** binary serialization, enabling persistent storage of HNSW indices with full temporal metadata. This allows pre-built indices to be distributed alongside datasets for reproducible analysis.

---

## 3. Related Work

### Vector Databases

| System | Vector Search | Temporal Trajectories | Analytical Functions |
|--------|:---:|:---:|:---:|
| Pinecone | Yes | No | No |
| Milvus | Yes | No | No |
| Weaviate | Yes | No | No |
| Qdrant | Yes | No | No |
| **CVX** | **Yes** | **Yes** | **19 native functions** |

Existing vector databases are optimized for retrieval. They support metadata filtering (including timestamps), but treat time as a filter predicate, not as a first-class analytical dimension. None provide trajectory-native operations like velocity, changepoint detection, or path signatures.

### Time Series Databases

Systems like InfluxDB and TimescaleDB handle temporal data natively but operate on scalar or low-dimensional metrics. They lack vector similarity search and cannot perform operations in high-dimensional embedding spaces. CVX bridges this gap: it applies time-series-style analytics (changepoints, Hurst exponent, stochastic characterization) to vector trajectories.

### Temporal Machine Learning

Neural ODEs (Chen et al., 2018), temporal transformers, and continuous-time models learn dynamics from temporal data. CVX is complementary: it provides the **data layer and feature extraction** that feeds these models. CVX's `temporal_features` function produces fixed-size summary vectors ($2D + 5$ dimensions) designed for downstream ML classification, while `path_signature` provides universal nonlinear trajectory descriptors.

### Path Signatures

The theory of rough paths and path signatures (Lyons, 1998; Kidger & Lyons, 2020) provides a mathematically rigorous framework for describing sequential data. CVX makes path signatures accessible via a simple API (`cvx.path_signature(trajectory, depth=3)`), computed in Rust for performance. Signature distance provides a metric for trajectory comparison that captures higher-order interactions between dimensions.

### Anchor-Based Interpretability

Anchor projection relates to concept-based explanations in interpretable ML, particularly TCAV (Kim et al., 2018), which tests model sensitivity to user-defined concepts. CVX's anchor projection applies a similar philosophy to trajectory analysis: rather than explaining a model, it explains *drift* by measuring movement relative to semantically meaningful reference points.

---

## 4. Cross-Domain Validation

CVX has been validated across 7 investigations spanning 6 domains. Each investigation uses exclusively public datasets and is fully reproducible from the repository's notebooks.

| Investigation | Domain | Dataset | Key Result | Page |
|---|---|---|---|---|
| B1: Mental Health Explorer | Clinical NLP | eRisk 2017--2022 | F1=0.600 (13 temporal features) | [Details](/research/mental-health-explorer) |
| B2: Clinical Anchoring | Clinical NLP | eRisk 2017--2022 | F1=0.744, AUC=0.886 (DSM-5 anchors) | [Details](/research/clinical-anchoring) |
| B3: Political Rhetoric & Markets | Political NLP / Finance | Trump Twitter + S&P 500 | Rhetorical anchor projection + market alignment | [Details](/research/trump-impact) |
| T1: Market Regime Detection | Quantitative Finance | S&P 500 Sector ETFs | 11 changepoints, Hurst=0.74, path signatures | [Details](/research/finance-regimes) |
| T2: Anomaly Detection | Time Series | Numenta NAB | Trajectory-geometric anomaly detection | [Details](/research/nab-anomaly) |
| T3: Fraud Detection | Cybersecurity | IEEE-CIS | Transaction trajectory fingerprinting | [Details](/research/fraud-detection) |
| T4: Insider Threat | Cybersecurity | CERT CMU | Behavioral regime shift detection | [Details](/research/insider-threat) |

### Reading Guide

- **B-series** (B1, B2, B3) investigations are **benchmark** studies with full experimental protocols, train/test splits, and quantitative evaluation against baselines.
- **T-series** (T1--T4) investigations are **technical** demonstrations showing how CVX's analytical toolkit applies to each domain, with qualitative and quantitative results.

---

## 5. Key Findings

Across the 7 investigations, several patterns emerge consistently:

**Anchor projection improves over absolute-space features.** In the clinical NLP domain, adding DSM-5 anchor projection to the B1 baseline improved F1 from 0.600 to 0.744 and AUC from 0.639 to 0.886. The improvement stems from transforming opaque high-dimensional drift into interpretable, domain-relevant coordinates. This pattern generalizes: political discourse analysis benefits from rhetorical anchors, and financial analysis from sector/regime anchors.

**Path signatures capture regime-level dynamics across domains.** Signature features encode the *shape* of trajectories, not just their endpoints. In market regime detection, signature distance distinguishes between accumulation, distribution, and crisis periods. In clinical NLP, signature features capture the nonlinear evolution of language that linear velocity cannot represent.

**Temporal analytics decompose complex behaviors into interpretable signals.** The 7-level framework provides a structured decomposition. Practitioners can identify *which level* of analysis reveals the most signal for their domain: mental health detection relies heavily on levels 2--4 (velocity, persistence, changepoints), while fraud detection emphasizes levels 5--6 (distributional shifts, signature fingerprints).

**CVX's API unifies analysis patterns across fundamentally different data.** The same function calls --- `velocity()`, `detect_changepoints()`, `path_signature()`, `project_to_anchors()` --- apply without modification to clinical text embeddings, financial time series, network traffic features, and behavioral logs. This universality is a consequence of operating in embedding space: once data is encoded as vectors, temporal dynamics follow the same mathematical structure regardless of the source domain.

---

## 6. Reproducibility

All investigations are designed for full reproducibility.

### Datasets

All datasets used are publicly available:

| Dataset | Source | Access |
|---------|--------|--------|
| eRisk 2017--2022 | CLEF eRisk shared task | Available upon request from organizers |
| Trump Twitter Archive | [thetrumparchive.com](https://www.thetrumparchive.com/) | Public download |
| S&P 500 Sector ETFs | Yahoo Finance (via `yfinance`) | Public API |
| Numenta NAB | [github.com/numenta/NAB](https://github.com/numenta/NAB) | Public repository |
| IEEE-CIS Fraud | [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection) | Public competition |
| CERT Insider Threat | [CMU SEI](https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247) | Public download |

### Code

All experiments are implemented as Jupyter notebooks in the `notebooks/` directory:

```bash
# Environment setup
conda activate cvx
cd crates/cvx-python && maturin develop --release && cd ../..

# Run any investigation notebook
jupyter notebook notebooks/B1_interactive_explorer.ipynb
```

### Dependencies

| Component | Purpose |
|-----------|---------|
| `cvx-python` | Rust-native CVX bindings (via PyO3 + maturin) |
| `sentence-transformers` | Text embedding (all-MiniLM-L6-v2 or all-mpnet-base-v2) |
| `yfinance` | Financial data retrieval |
| `scikit-learn` | Classification baselines and evaluation |
| `plotly` | Interactive 3D visualizations |

---

## 7. Future Work

### Additional Domains

- **Molecular dynamics:** Conformational trajectory analysis using graph-region clustering and signature-based state identification
- **Drug discovery:** Campaign navigation through chemical embedding spaces with anchor projection to pharmacophore references
- **Climate science:** Long-range climate model trajectory comparison using distributional distances and topological persistence

### Neural ODE Integration

CVX's `predict` function currently supports linear extrapolation. Future work integrates **Neural ODE** models (trained in Python via PyTorch, deployed in Rust via TorchScript) for nonlinear trajectory forecasting. The trained models will predict future embedding positions conditioned on observed trajectories, enabling proactive anomaly detection and early warning systems.

### Online Changepoint Detection

The current `detect_changepoints` implementation uses the offline PELT algorithm, requiring the full trajectory. Future work adds **Bayesian Online Changepoint Detection (BOCPD)** for streaming applications where trajectories grow incrementally and changepoints must be detected in real time.

### Expanded Anchor Libraries

Anchor projection's utility scales with the quality and coverage of the anchor set. Planned anchor libraries include:

- **ICD-11 diagnostic categories** for broader clinical NLP applications
- **Complete DSM-5 symptom dimensions** beyond the current depression-focused subset
- **Financial event taxonomy** (earnings, regulatory, geopolitical) for market regime anchoring
- **MITRE ATT&CK framework** anchors for cybersecurity trajectory analysis
