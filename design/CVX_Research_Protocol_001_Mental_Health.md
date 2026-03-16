# Research Protocol 001: Temporal Trajectory Analysis for Mental Health Detection

**Status**: Draft
**Created**: 2026-03-16
**Authors**: Manuel Couto Pintos
**Related**: RFC-004 (Semantic Regions), Tutorial B1

---

## 1. Objective

Evaluate ChronosVector's graph-based semantic region trajectory analysis as a method for early detection of psychological distress from social media, across multiple benchmark datasets, with rigorous experimental design and interpretable post-hoc analysis.

### Primary Hypotheses

**H1**: Users with depression exhibit statistically different region trajectory dynamics (velocity, Hurst exponent, change point frequency) compared to control users.

**H2**: Temporal features extracted from smoothed region trajectories discriminate depression from control with F1 > 0.75 on held-out test data with proper temporal splits.

**H3**: The graph-based region approach (RFC-004) outperforms raw embedding trajectory analysis in all metrics.

### Secondary Hypotheses

**H4**: Change point detection on region trajectories identifies clinically meaningful regime transitions in depression users.

**H5**: Discriminative features generalize across datasets (eRisk, CLPsych, RSDD) — the same types of region dynamics are predictive in all three.

**H6**: Early detection is feasible — classification using only the first N% of a user's trajectory still achieves significant AUC.

---

## 2. Datasets

### 2.1 eRisk (Reddit)

| Split | Year | Users | Purpose |
|-------|------|-------|---------|
| **Train** | 2017 train | 486 (83 dep, 403 ctrl) | Model training + hyperparameter search |
| **Validation** | 2017 test / 2018 | ~400-820 | Hyperparameter selection |
| **Test** | 2022 | 1,400 (98 dep, 1,302 ctrl) | Final evaluation (touch once) |

Source: CLEF eRisk Lab. Labels from `risk_golden_truth.txt`.

### 2.2 CLPsych 2015 (Twitter)

| Split | Chunks | Users | Notes |
|-------|--------|-------|-------|
| **Train** | 0-50 | ~1,554 (477 dep, 872 ctrl) | Official training set |
| **Test** | 60-89 | Same users, later tweets | Temporal split |

Three conditions available: depression, PTSD, control. Protocol uses depression vs control (binary).

### 2.3 RSDD (Reddit)

| Split | File | Users | Notes |
|-------|------|-------|-------|
| **Train** | training-002 | ~39K | Large, imbalanced (~8% depression) |
| **Validation** | validation-001 | ~39K | Hyperparameter tuning |
| **Test** | testing-003 | ~39K | Final evaluation |

### 2.4 Pre-processing

All datasets are converted to unified JSONL format and embedded with `sentence-transformers/all-MiniLM-L6-v2` (D=384). Embeddings are stored as Parquet files. Original texts are retained for c-TF-IDF analysis.

**Temporal integrity**: Train/val/test splits respect the original dataset partitions. No user appears in multiple splits. No future data leaks into training.

---

## 3. Independent Variables (Hyperparameters)

### 3.1 Graph Regions

| Parameter | Search Space | Default |
|-----------|-------------|---------|
| HNSW level | {1, 2, 3} | 2 |
| K (resulting regions) | ~N/16^L | Determined by level |

### 3.2 Trajectory Smoothing

| Parameter | Search Space | Default |
|-----------|-------------|---------|
| Window size (days) | {7, 14, 21, 30} | 14 |
| EMA alpha | {0.1, 0.2, 0.3, 0.5} | 0.3 |

### 3.3 Change Point Detection (PELT)

| Parameter | Search Space | Default |
|-----------|-------------|---------|
| Penalty | {1.0, 2.0, 3.0, 5.0} × ln(n) | 3.0 × ln(n) |
| Min segment length | {2, 3, 5} | 2 |

### 3.4 Classifier

| Parameter | Search Space | Default |
|-----------|-------------|---------|
| n_estimators | {100, 200, 500} | 200 |
| max_depth | {5, 10, 20, None} | 10 |
| class_weight | {balanced, None} | balanced |

### 3.5 Search Strategy

Grid search on validation set. Best configuration evaluated once on test set. Reported with 95% confidence intervals via bootstrap (1000 iterations).

---

## 4. Dependent Variables (Metrics)

### 4.1 Classification (Primary)

| Metric | Computation | Threshold |
|--------|-------------|-----------|
| **ROC-AUC** | Area under ROC curve | Primary metric |
| **F1** | Harmonic mean of precision/recall | ≥ 0.75 for H2 |
| **Precision** | TP / (TP + FP) | Reported |
| **Recall** | TP / (TP + FN) | Reported |

All metrics on **test set only**, computed per fold in 5-fold stratified CV on train, and once on held-out test.

### 4.2 Trajectory Analytics (Secondary)

| Metric | Test | Significance |
|--------|------|-------------|
| Velocity (depression vs control) | Mann-Whitney U | p < 0.05 |
| Hurst exponent (depression vs control) | Mann-Whitney U | p < 0.05 |
| Change point frequency | Mann-Whitney U | p < 0.05 |
| Change point severity | Mann-Whitney U | p < 0.05 |
| Region distribution | Chi-squared | p < 0.05 |

Effect sizes reported as Cohen's d for continuous metrics, Cramér's V for categorical.

### 4.3 Comparison Metrics

| Comparison | Method |
|-----------|--------|
| Region trajectories vs raw embeddings | Paired test on same users |
| Across datasets | Report per-dataset + pooled |
| H3 (region > raw) | McNemar test on classification |

---

## 5. Experimental Plan

### Phase 1: Baseline Establishment

1. **Raw embedding baseline**: Reproduce current B1 results with proper train/test splits per dataset
2. **Static baseline**: Classify using mean embedding per user (no temporal features) as sanity check
3. Metrics: AUC, F1, precision, recall on test sets

### Phase 2: Region Trajectory Analysis

1. **Region extraction**: For each dataset, extract regions at levels 1, 2, 3
2. **c-TF-IDF labeling**: Label all regions, identify depression-related regions
3. **Trajectory computation**: Compute smoothed trajectories with default hyperparameters
4. **Analytics**: Velocity, Hurst, CPD on region trajectories for all users
5. **Statistical tests**: All secondary metrics (§4.2)

### Phase 3: Hyperparameter Optimization

1. **Grid search on validation set** (eRisk 2017 test / 2018):
   - HNSW level × window × alpha × penalty × classifier params
   - Optimize for F1 (primary) and AUC (secondary)
2. **Best config selection**: Choose config that maximizes F1 on validation
3. **Sensitivity analysis**: How much do results vary across the grid?

### Phase 4: Test Set Evaluation

1. **Apply best config** to all three test sets (eRisk 2022, CLPsych test chunks, RSDD test)
2. **Report all metrics** with 95% CI via bootstrap
3. **Compare region vs raw embedding** (H3) via McNemar test
4. **Cross-dataset comparison** (H5): Are the same features discriminative?

### Phase 5: Technique Exploration

Apply additional CVX analytics on region trajectories:

| Technique | CVX Function | What it Tests |
|-----------|-------------|---------------|
| Multi-scale analysis | Resample trajectories at 7d, 14d, 30d, 90d windows | Scale-dependent dynamics |
| Volatility (per-region) | Variance of region proportions over time | Stability of topical focus |
| Drift attribution | `cvx.drift()` on region distributions | Which regions changed most |
| Cohort divergence | Pairwise distance between dep/ctrl groups over time | When groups separate |
| Path signatures | Compute on region trajectories | Higher-order trajectory patterns |
| Temporal features fusion | Combine region features + raw embedding features | Complementarity |

### Phase 6: Early Detection Analysis

Test H6: classification using trajectory prefixes.

1. For each user, take first {25%, 50%, 75%, 100%} of their posts
2. Compute region trajectories on the prefix
3. Classify and report AUC at each prefix length
4. Plot AUC vs % trajectory observed (ERDE-style curve)
5. Determine minimum observation window for AUC > 0.80

---

## 6. Post-hoc Analysis

### 6.1 Error Analysis

- **False negatives**: Depression users classified as control
  - Do they have fewer posts? Shorter observation windows?
  - Are their trajectories more similar to control group?
  - Do they lack change points or show stable region distributions?
- **False positives**: Control users classified as depression
  - Do they discuss negative topics more than average controls?
  - Are they in specific life circumstances (grief, stress) that mimic depression?
- **Confusion matrix** per dataset, annotated with user characteristics

### 6.2 Ablation Study

Systematically remove feature groups and measure impact:

| Ablation | Features Removed | Expected Impact |
|----------|-----------------|----------------|
| No velocity features | Mean velocity, std velocity, max velocity | Moderate |
| No Hurst | Hurst exponent | Low-moderate |
| No CPD features | n_changepoints, max_severity, cp_rate | Low (currently 0) |
| No region distribution | Mean proportion per region | High |
| No drift features | L2 drift, cosine drift | Moderate |
| No volatility | Per-dimension volatility | Low-moderate |
| Only region distribution (no temporal) | Remove all temporal dynamics | Benchmark: is temporal info needed? |

### 6.3 Discriminative Region Analysis

- **Top discriminative regions**: Rank regions by chi-squared between dep/ctrl
- **c-TF-IDF of discriminative regions**: What topics separate the groups?
- **Region transition patterns**: Which region→region transitions are predictive?
  - Construct transition matrix per user, compare dep vs ctrl
- **Temporal region heatmap**: For top discriminative regions, plot proportion over time for dep vs ctrl (aggregate)

### 6.4 Temporal Signal Analysis

- **When does the signal emerge?** Plot per-feature discriminability (AUC) as function of time
- **Is there a "critical window"?** Identify the time period where features become most discriminative
- **Pre-transition features**: For users with detected change points, analyze features in the N days *before* the change point

### 6.5 Case Studies

Select 3-5 users for detailed qualitative analysis:

1. **True positive with clear trajectory**: Depression user with visible region shift + change point
2. **True positive subtle case**: Depression user with gradual drift, no obvious change point
3. **False negative**: Depression user that the model misses — why?
4. **False positive**: Control user flagged as depression — explain
5. **Early detection success**: User where prefix analysis detects signal early

For each case: region trajectory plot, c-TF-IDF of dominant regions, sample posts from key periods, timeline of velocity/CPD events.

### 6.6 Cross-Dataset Robustness

- **Feature correlation across datasets**: Do the same features rank high in importance for eRisk, CLPsych, RSDD?
  - Spearman correlation of feature importance rankings
- **Transfer learning**: Train on eRisk, test on CLPsych and RSDD (zero-shot cross-dataset)
- **Region topology comparison**: Do the same types of regions emerge across datasets?
  - Compare c-TF-IDF labels across datasets for similar-sized regions

### 6.7 Hyperparameter Robustness

- **Sensitivity heatmap**: AUC as function of (window_days × alpha) on validation set
- **Level sensitivity**: How much does performance change between level 1, 2, 3?
- **Penalty sensitivity for CPD**: At what penalty does PELT start detecting changes? Quality of detections at different penalty levels

---

## 7. Deliverables

### Notebooks

| Notebook | Content |
|----------|---------|
| `B1_main_analysis.ipynb` | Full pipeline: ingestion → regions → trajectories → classification → all three datasets |
| `B1_hyperparameter_search.ipynb` | Grid search on validation sets, sensitivity analysis |
| `B1_posthoc_analysis.ipynb` | Error analysis, ablation, case studies, cross-dataset |
| `B1_early_detection.ipynb` | Prefix analysis, ERDE curves, minimum observation window |

### Documentation

- Results integrated into docs site with key figures and tables
- Methodology section explaining graph semantic regions
- Comparison table across datasets and methods

### Figures

- Stacked area trajectories (case studies)
- Region distribution barplots (dep vs ctrl)
- AUC vs observation window (early detection curve)
- Sensitivity heatmaps (hyperparameters)
- Feature importance with ablation
- Discriminative region word clouds
- Cross-dataset feature correlation

---

## 8. Statistical Rigor

- All p-values reported with Bonferroni correction for multiple comparisons
- Effect sizes (Cohen's d, Cramér's V) alongside p-values
- 95% confidence intervals via bootstrap (n=1000) for all test-set metrics
- McNemar test for method comparisons (region vs raw)
- No results reported on test set until hyperparameters are frozen on validation set

---

## 9. Ethical Considerations

- All datasets obtained through official data use agreements
- No attempt to de-anonymize users
- Raw text data never committed to repository
- Results reported in aggregate; case studies use anonymized IDs
- Clinical deployment would require IRB approval and clinician oversight
- Tool aids clinicians — does not replace clinical judgment

---

## 10. Timeline

| Phase | Scope | Depends On |
|-------|-------|-----------|
| 1 | Baselines (3 datasets, proper splits) | Embeddings ready |
| 2 | Region analysis (3 datasets) | Phase 1 |
| 3 | Hyperparameter search (validation sets) | Phase 2 |
| 4 | Test set evaluation | Phase 3 (frozen config) |
| 5 | Technique exploration | Phase 4 |
| 6 | Early detection analysis | Phase 4 |
| Post-hoc | Error analysis, ablation, case studies | Phase 4 |

---

## References

1. Couto, M. et al. (2025). Temporal word embeddings for psychological disorder early detection. *JHIR*.
2. Coppersmith, G. et al. (2018). NLP of social media as screening for suicide risk. *BMI Insights*.
3. De Choudhury, M. et al. (2013). Predicting depression via social media. *ICWSM*.
4. Losada, D. et al. (2019). eRisk: early risk prediction on the internet. *CLEF Working Notes*.
5. Yates, A. et al. (2017). Depression and self-harm risk assessment in online forums. *EMNLP*.
6. Killick, R. et al. (2012). Optimal detection of changepoints. *JASA*.
7. Malkov, Y.A. & Yashunin, D.A. (2018). Efficient and robust ANN using HNSW graphs. *IEEE TPAMI*.
8. Bamler, R. & Mandt, S. (2017). Dynamic word embeddings. *ICML*.
9. Grootendorst, M. (2022). BERTopic: neural topic modeling with c-TF-IDF. *arXiv:2203.05794*.
