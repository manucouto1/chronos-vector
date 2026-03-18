---
title: "Clinical Anchoring — DSM-5 Anchor Projection for Depression Detection"
description: "Symptom-aware temporal analysis using CVX anchor projection on eRisk data"
---

## Abstract

Standard sentence embeddings compress clinical signals into opaque high-dimensional spaces where syntactic structure masks symptom evolution. We introduce **anchor projection** — a coordinate system transformation implemented natively in ChronosVector (CVX) — that re-expresses user trajectories relative to DSM-5 symptom reference vectors. On the eRisk depression detection task with a proper temporal split (train: 2017+2018, test: 2022), anchor-projected features achieve **F1=0.744 and AUC=0.886**, compared to F1=0.600 with absolute temporal features alone. Early detection at 10% of post history yields F1=0.673.

## Related Work

- **eRisk shared task** (Losada et al., 2017-2022): Early risk detection of depression from social media posts. Best systems use transformer-based classifiers on user-level features.
- **Concept-based explanations** (Kim et al., TCAV, 2018): Testing with Concept Activation Vectors measures model sensitivity to human-defined concepts. Our anchor projection applies a similar idea at the data level.
- **Clinical NLP for mental health** (Coppersmith et al., 2018; Harrigian et al., 2020): Feature engineering from social media text for mental health detection. Most approaches treat users as static feature vectors.
- **Temporal dynamics in depression** (De Choudhury et al., 2013): Pioneering work on temporal patterns in social media for depression, but without vector trajectory analysis.

## Methodology

### Data
- **eRisk dataset**: 1.36M Reddit posts from 2,285 users
- **Embeddings**: MentalRoBERTa (D=768), pre-trained on mental health corpora
- **Subset**: 466 balanced users (233 depression, 233 control), 225,962 posts

### CVX Pipeline

1. **Index Construction**: `bulk_insert` 225K vectors with `save`/`load` caching (avoids 500s rebuild)
2. **DSM-5 Anchor Vectors**: 9 symptom anchors (depressed mood, anhedonia, sleep disturbance, fatigue, worthlessness, concentration, suicidal ideation, appetite, psychomotor) + 1 healthy baseline, each encoded as MentalRoBERTa centroid of 3-5 representative phrases
3. **Anchor Projection**: `cvx.project_to_anchors(trajectory, anchors, metric='cosine')` → trajectory in ℝ¹⁰
4. **Feature Extraction**: `cvx.anchor_summary()` (mean, min, trend per anchor), `cvx.hurst_exponent()` on projected trajectory, `cvx.velocity()` in anchor space, topic polarization (dispersion), velocity differential (`cvx.drift()` for consecutive posts)
5. **Classification**: Logistic Regression with `class_weight='balanced'`, both 5-fold CV and temporal split

### Temporal Split (No Contamination)
- **Train**: eRisk 2017+2018 editions (226 users)
- **Test**: eRisk 2022 edition (236 users) — completely unseen

## Key Results

### Feature Ablation (5-Fold CV)

| Model | F1 | AUC | Precision | Recall |
|-------|-----|-----|-----------|--------|
| B1 Baseline (absolute features) | 0.600 | 0.639 | 0.590 | 0.614 |
| Anchor Only | 0.746 | 0.849 | 0.739 | 0.759 |
| Polarization Only | 0.599 | 0.665 | 0.653 | 0.556 |
| Velocity Only | 0.547 | 0.554 | 0.520 | 0.582 |
| **Combined (B2)** | **0.781** | **0.863** | **0.775** | **0.789** |

### Temporal Split (Train 2017+2018 → Test 2022)

| Metric | Value |
|--------|-------|
| **F1** | **0.744** |
| Precision | 0.659 |
| Recall | 0.856 |
| **AUC** | **0.886** |

### Early Detection

| % of Posts | F1 | AUC |
|-----------|-----|-----|
| 10% | 0.673 | 0.788 |
| 20% | 0.694 | 0.811 |
| 30% | 0.719 | 0.829 |
| 50% | 0.729 | 0.858 |
| 100% | 0.753 | 0.895 |

## CVX Functions Used

| Function | Role |
|----------|------|
| `project_to_anchors(metric='cosine')` | Transform ℝ⁷⁶⁸ → ℝ¹⁰ symptom coordinates |
| `anchor_summary()` | Mean, min, trend, last distance per anchor |
| `hurst_exponent(projected)` | Persistence of approach to depression |
| `velocity(projected)` | Rate of change in symptom space |
| `drift(post_t, post_t+1)` | Consecutive cosine displacement |
| `save()` / `load()` | Index persistence (avoid 500s rebuild) |

## Running the Notebook

```bash
conda activate cvx
cd crates/cvx-python && maturin develop --release && cd ../..
jupyter notebook notebooks/B2_clinical_anchoring.ipynb
```

## Interactive Visualizations

See the [full tutorial with Plotly plots](/chronos-vector/tutorials/b2-clinical-anchoring).

---

[← Back to White Paper](/chronos-vector/research/white-paper)
