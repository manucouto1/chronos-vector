---
title: "B1: Mental Health Detection with MentalRoBERTa + Temporal Features"
description: "Detecting psychological distress using domain-adapted embeddings, behavioral posting patterns, and graph-based semantic regions — evaluated on eRisk and CLPsych"
---

> **Notebooks:** Per-dataset analysis following [Research Protocol 001 v2](https://github.com/manucouto1/chronos-vector/blob/develop/design/CVX_Research_Protocol_001_Mental_Health.md)
> - [`notebooks/B1_eRisk.ipynb`](https://github.com/manucouto1/chronos-vector/blob/develop/notebooks/B1_eRisk.ipynb) — Primary (Reddit, 2017–2022)
> - [`notebooks/B1_CLPsych.ipynb`](https://github.com/manucouto1/chronos-vector/blob/develop/notebooks/B1_CLPsych.ipynb) — Replication (Twitter, CLPsych 2015)

## Abstract

We present a multi-signal approach to early depression detection from social media combining: (1) **MentalRoBERTa** embeddings (D=768, domain-adapted to mental health language), (2) **behavioral posting patterns** (inter-post gaps, circadian activity, posting bursts), and (3) **graph-based semantic regions** from ChronosVector's HNSW hierarchy.

On eRisk (1.36M Reddit posts, 2,285 users), the full model achieves **ROC-AUC = 0.911 [0.875, 0.941]** on held-out 2022 test data, with early detection capability: **AUC = 0.849 using only 10% of a user's posts**. On CLPsych (Twitter), temporal embeddings reach **AUC = 0.804** with early detection at **AUC = 0.813 at 20%** of posts.

---

## Cross-Dataset Results

### Feature Ablation

| Model | eRisk AUC | eRisk F1 | CLPsych AUC | CLPsych F1 | Dims |
|-------|----------:|---------:|------------:|-----------:|-----:|
| Behavioral only | 0.643 | 0.020 | 0.585 | 0.309 | 11 |
| Static MentalRoBERTa | 0.901 | 0.457 | 0.787 | **0.579** | 768 |
| Temporal MentalRoBERTa | 0.910 | 0.443 | **0.801** | 0.559 | 768 |
| Region L3 only | 0.890 | 0.418 | 0.770 | 0.491 | ~80 |
| Temporal + Behavioral | 0.907 | 0.432 | **0.804** | 0.571 | 779 |
| **Full (Temp+Region+Behav)** | **0.911** | **0.458** | 0.790 | 0.522 | ~880 |

### Best Model — Test Set (95% Bootstrap CI)

**eRisk** (Full model, $n_{test}$ = 1,398):

| Metric | Point | 95% CI |
|--------|------:|-------:|
| **ROC-AUC** | **0.911** | [0.875, 0.941] |
| F1 | 0.456 | [0.346, 0.558] |
| Precision | 0.717 | [0.579, 0.848] |
| Recall | 0.336 | [0.239, 0.433] |

**CLPsych** (Static MentalRoBERTa, $n_{test}$ = 203):

| Metric | Point | 95% CI |
|--------|------:|-------:|
| **ROC-AUC** | **0.786** | [0.722, 0.850] |
| F1 | 0.576 | [0.464, 0.683] |
| Precision | 0.712 | [0.583, 0.838] |
| Recall | 0.487 | [0.370, 0.603] |

### Early Detection (AUC@k%)

| % Posts | eRisk AUC | CLPsych AUC |
|--------:|----------:|------------:|
| 10% | **0.849** | 0.780 |
| 20% | 0.858 | **0.813** |
| 50% | 0.888 | 0.795 |
| 100% | 0.908 | 0.792 |

**Key result:** With only 10% of a user's post history, the model already achieves AUC > 0.84 on eRisk. This validates early detection feasibility.

---

## Key Findings

### 1. Domain-Adapted Embeddings Matter

MentalRoBERTa (pre-trained on Reddit mental health forums) captures emotional and clinical language that general-purpose models miss. The static baseline with MentalRoBERTa (AUC = 0.901) is already strong — the embedding model is the single most impactful choice.

### 2. Behavioral Signals: Night Posting as Biomarker

On eRisk, the **night posting ratio** (posts between 00:00–06:00) is the strongest behavioral discriminator:
- Depression: 31% night posts vs Control: 22% (d = 0.534, p < 0.001)
- **Circadian disruption** is a known clinical marker of depression

On CLPsych, **posting gap variability** (gap_cv) is discriminative (d = 0.315, p < 0.001) — depression users have more irregular posting patterns.

### 3. Graph Regions Capture Topical Signal

At HNSW Level 3 (~60–97 coarse regions), several regions show large effect sizes:
- eRisk: regions r3_80 (d = 1.05) and r3_72 (d = 1.06) — depression users heavily over-represented
- With only 99 dimensions, Region L3 achieves AUC = 0.890 (eRisk) — comparable to 768-dim embeddings

### 4. Early Detection is Viable

The AUC@k% curves show signal is present early:
- **eRisk**: AUC > 0.84 with 10% of posts
- **CLPsych**: AUC > 0.80 with 20% of posts

This means screening can begin very early in a user's posting history.

---

## Methodology

### Pipeline

```
Posts → MentalRoBERTa (D=768) → CVX Index → Region Discovery (L3)
  │                                              │
  ├─ Relative time (t_rel, gap)                  ├─ Region distribution
  ├─ Behavioral features (gaps, circadian)       ├─ Region entropy
  └─ Recency-weighted aggregation                └─ Top-3 concentration
                    │                                      │
                    └──────── Feature Concatenation ────────┘
                                      │
                              Random Forest (balanced)
                                      │
                              Depression / Control
```

### CVX Features Used

```python
import chronos_vector as cvx

index = cvx.TemporalIndex(m=16, ef_construction=200, ef_search=100)
# ... ingest posts with relative timestamps ...

# Top-down region discovery
regions_l3 = index.regions(level=3)   # ~60-97 coarse regions
regions_l2 = index.regions(level=2)   # ~1,000 fine regions

# Region trajectory with EMA smoothing
traj = index.region_trajectory(entity_id=uid, level=3, window_days=14, alpha=0.3)
```

### Feature Groups

| Group | Features | Source |
|-------|----------|--------|
| **Behavioral** (11) | mean_gap, std_gap, gap_cv, gap_trend, night_ratio, burst_count, ... | Posting patterns |
| **Embedding** (768) | Recency-weighted mean of MentalRoBERTa embeddings | `mental/mental-roberta-base` |
| **Region L3** (~80) | Post distribution across coarse HNSW regions + entropy | CVX graph hierarchy |

---

## Running the Notebooks

```bash
# Setup
conda activate cvx
pip install transformers torch
cd crates/cvx-python && maturin develop --release && cd ../..

# Generate embeddings (on GPU)
python scripts/generate_embeddings_v2.py --dataset erisk

# Apply splits
python scripts/add_splits.py

# Run analysis
cd notebooks && jupyter notebook B1_eRisk.ipynb
```

**Requirements:** ~10 GB RAM for loading parquets, ~15 min CVX ingestion per dataset.

---

## References

1. Couto, M. et al. (2025). Temporal word embeddings for psychological disorder early detection. *JHIR*.
2. Ji, S. et al. (2022). MentalBERT: Publicly available pretrained language models for mental healthcare. *LREC*.
3. Coppersmith, G. et al. (2018). NLP of social media as screening for suicide risk. *BMI Insights*.
4. Losada, D. et al. (2019). eRisk: early risk prediction on the internet. *CLEF Working Notes*.
5. Killick, R. et al. (2012). Optimal detection of changepoints. *JASA*.
6. Malkov, Y.A. & Yashunin, D.A. (2018). Efficient and robust ANN using HNSW graphs. *IEEE TPAMI*.
7. DS@GT (2025). Temporal attention models for eRisk 2025. *CLEF Working Notes*.
