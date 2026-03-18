# Research Protocol 001: Temporal Trajectory Analysis for Mental Health Detection

**Status**: v2 — Major revision after Phase 1–4 results
**Created**: 2026-03-16
**Revised**: 2026-03-17
**Authors**: Manuel Couto Pintos
**Related**: RFC-004 (Semantic Regions), Tutorial B1

---

## 0. Lessons from v1 (Phase 1–4 Results)

The first iteration (v1) revealed fundamental issues:

| Issue | Evidence | Root Cause |
|-------|----------|------------|
| Static baseline surprisingly strong (AUC 0.90) | Mean embedding already captures "what topics" a user discusses | The *distributional* signal is strong; temporal dynamics add little with generic embeddings |
| Region trajectory F1 low (0.45 vs 0.62 baseline) | High-dimensional feature space (2K+5 ≈ 2,600) with few depression samples | Overfitting + wrong granularity (1,300 regions for 93 train dep users) |
| Velocity/Hurst not discriminative (p > 0.1) | Velocity measures topic change speed; depression ≠ faster topic switching | Wrong temporal encoding + wrong signal (topic ≠ emotion) |
| RSDD completely fails (F1 = 0) | 8% depression rate + noisy self-reported labels | Extreme imbalance + label noise |
| Change points always 0 | PELT penalty too high for smooth EMA distributions | EMA smoothing removes the very discontinuities CPD is designed to detect |

### Key Insights for v2

1. **Temporal encoding is wrong.** Using absolute calendar dates. In early detection, what matters is:
   - *Time since first observation* — track evolution from onset
   - *Time since previous post* (inter-post gap) — behavioral signal
   These are the two variants validated in DCWE (Couto et al., 2025).

2. **Embeddings are wrong.** MiniLM-L6-v2 is a general-purpose semantic model. It captures *topic* (sports, food, politics) but NOT *emotional valence*. **MentalRoBERTa** (Ji et al., 2022) is pre-trained on Reddit mental health forums and captures domain-specific emotional language.

3. **Hierarchy should be top-down.** Start from the coarsest level (level 3: ~60–80 regions) to find *which broad topic areas* differ. Then drill into finer levels only within discriminative coarse regions.

4. **Posts are events, not a continuous signal.** The "trajectory" metaphor breaks down for irregularly-spaced discrete posts. Instead:
   - Model posts as *events in a sequence* with inter-event intervals
   - Use *attention-weighted aggregation* (not EMA smoothing)
   - Weight by recency + content relevance (as in eRisk 2025 best systems)

5. **Need pattern-based approach.** Instead of computing global features, learn *characteristic temporal patterns* of the positive class during training, then detect those patterns in test users.

---

## 1. Objective (Revised)

Evaluate ChronosVector as a platform for **early sequential detection** of psychological distress from social media, using:
- Domain-adapted embeddings (MentalRoBERTa)
- Relative temporal encoding (time since first post, inter-post gaps)
- Hierarchical top-down region analysis
- Pattern-based classification with temporal attention

### Primary Hypotheses

**H1**: MentalRoBERTa embeddings + graph regions produce significantly more discriminative features than generic MiniLM embeddings.

**H2**: Relative temporal encoding (time since first post + inter-post gap features) improves classification over absolute timestamps.

**H3**: Top-down hierarchical analysis (coarse → fine) identifies interpretable depression-associated topic shifts.

**H4**: Sequential classification with ERDE metrics demonstrates early detection capability — achieving AUC > 0.80 with only 50% of a user's post history.

### Secondary Hypotheses

**H5**: Inter-post gap patterns (posting frequency, circadian shifts, gap variance) are independently discriminative behavioral biomarkers.

**H6**: Depression users concentrate their activity in fewer semantic regions at the coarsest level, with higher volatility at finer levels.

**H7**: The approach generalizes across datasets (eRisk, CLPsych) — same types of patterns are predictive.

---

## 2. Datasets

### 2.1 eRisk (Reddit) — Primary

| Split | Source | Users | Dep | Ctrl |
|-------|--------|------:|----:|-----:|
| **Train** | 2017+2018 pooled (80%) | 672 | 93 | 579 |
| **Val** | 2017+2018 pooled (20%) | 163 | 28 | 135 |
| **Test** | 2022 | 1,298 | 92 | 1,206 |

### 2.2 CLPsych 2015 (Twitter)

| Split | Source | Users | Dep | Ctrl |
|-------|--------|------:|----:|-----:|
| **Train** | User-level stratified 70% | 939 | 329 | 610 |
| **Val** | User-level stratified 15% | 201 | 70 | 131 |
| **Test** | User-level stratified 15% | 202 | 71 | 131 |

### 2.3 RSDD (Reddit) — Exploratory only

RSDD has extreme imbalance (~8% depression rate) and noisy self-reported labels. Include for completeness but do NOT use for primary hypothesis testing.

### 2.4 Pre-processing (Revised)

**Embedding models** (ablation):

| Model | Dimension | Domain | Purpose |
|-------|-----------|--------|---------|
| `mental/mental-roberta-base` | 768 | Mental health (Reddit) | **Primary** — emotion-aware |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | General purpose | Comparison baseline |

**Temporal encoding** (two features per post, added as metadata):

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `t_rel` | `(timestamp - user_first_post) / total_span` | Normalized position in user's history [0, 1] |
| `gap` | `timestamp - previous_timestamp` (seconds) | Inter-post interval; behavioral signal |

**Post-level features** (computed during preprocessing):

| Feature | Type | Source |
|---------|------|--------|
| `embedding` | float[768] | MentalRoBERTa |
| `t_rel` | float | Relative time [0, 1] |
| `gap_seconds` | float | Gap since previous post |
| `gap_log` | float | log(1 + gap_seconds) — normalized |
| `post_index` | int | Sequential index within user |
| `hour_of_day` | int | Circadian signal |
| `text_length` | int | Post length |

---

## 3. Feature Engineering (Revised)

### 3.1 Multi-Level Region Features (Top-Down)

For each HNSW hierarchy level L ∈ {3, 2, 1} (coarsest to finest):

| Feature Group | Description | Dim |
|--------------|-------------|-----|
| **Region distribution** | Proportion of posts per region | K_L |
| **Region entropy** | -Σ p·log(p) over regions | 1 |
| **Top-region concentration** | Fraction of posts in top-3 regions | 1 |
| **Region transition entropy** | Entropy of bigram transition matrix | 1 |

At the coarsest level (L=3, K≈60–80), these features are low-dimensional and interpretable.

### 3.2 Behavioral Temporal Features

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `mean_gap` | Mean inter-post interval | Posting frequency |
| `std_gap` | Std of inter-post intervals | Regularity |
| `max_gap` | Longest silence | Withdrawal periods |
| `gap_trend` | Slope of gap over time | Increasing → withdrawal |
| `n_posts` | Total number of posts | Activity level |
| `night_ratio` | Fraction of posts 00:00–06:00 | Circadian disruption |
| `burst_count` | Number of posting bursts (>5 posts in 1h) | Manic/crisis episodes |
| `gap_cv` | Coefficient of variation of gaps | Posting pattern stability |

### 3.3 Sequential Embedding Features (with Temporal Attention)

Instead of mean-pooling or EMA, aggregate post embeddings with **recency-weighted attention**:

$$\mathbf{u} = \sum_i \alpha_i \cdot \mathbf{e}_i, \quad \alpha_i = \text{softmax}(w \cdot t_{\text{rel},i})$$

Where $w$ is a learnable weight (or fixed linear ramp). This gives more weight to recent posts while preserving the full trajectory information. Implemented in Python using the embedding vectors from the CVX index.

### 3.4 Feature Hierarchy

**Level 0 (Simplest — Behavioral only):**
- Gap features (§3.2) — no embeddings needed

**Level 1 (Static Embedding):**
- Mean embedding per user (baseline)

**Level 2 (Temporal Embedding):**
- Recency-weighted embedding aggregation (§3.3)

**Level 3 (Region + Behavioral):**
- Coarse region distribution (L=3) + behavioral features

**Level 4 (Full Model):**
- All of the above + finer region levels

---

## 4. Metrics (Revised)

### 4.1 Classification

| Metric | Purpose |
|--------|---------|
| **ROC-AUC** | Ranking quality (primary) |
| **F1** | Classification quality (primary) |
| **Precision / Recall** | Operating point analysis |
| **Precision@k** | Top-k screening scenario |

### 4.2 Early Detection (NEW)

| Metric | Definition | Purpose |
|--------|-----------|---------|
| **ERDE₅** | Error penalized after 5 posts | Very early detection |
| **ERDE₅₀** | Error penalized after 50 posts | Moderate early detection |
| **AUC@k%** | AUC using only first k% of posts | Detection speed curve |
| **latency** | Number of posts needed for AUC > 0.80 | Minimum observation window |

### 4.3 Statistical Tests

| Test | Variables | Significance |
|------|-----------|-------------|
| Mann-Whitney U | Each feature (dep vs ctrl) | p < 0.05, Bonferroni-corrected |
| Effect size | Cohen's d | Report alongside p-values |
| McNemar | Method A vs Method B | Classification comparison |
| Ablation ΔAUC | Feature group removal | Contribution of each group |

---

## 5. Experimental Plan (Revised)

### Phase 0: Re-Embedding with MentalRoBERTa

1. Generate MentalRoBERTa embeddings (D=768) for all 3 datasets
2. Compute relative temporal features (t_rel, gap) per post
3. Store as updated Parquet files alongside existing MiniLM embeddings

**Location**: Run on HPC (hpc.glaciar.lab) with GPU.

### Phase 1: Feature-Level Analysis

Before any classification, analyze discriminative power of each feature group independently:

1. **Behavioral features only** — Are gap/frequency patterns discriminative?
2. **Region distribution at each level** — Which level is most discriminative?
3. **Effect of embedding model** — MentalRoBERTa vs MiniLM on same features
4. **Visualize**: Feature distributions (dep vs ctrl), effect sizes, p-values

This phase answers: *what types of signal exist in the data?*

### Phase 2: Hierarchical Region Analysis (Top-Down)

1. **Level 3** (60–80 regions): Identify coarse topic areas. c-TF-IDF label each. Which regions are over/under-represented in depression?
2. **Level 2** (~1,000 regions): Within discriminative L3 regions, drill down. Which sub-topics differ?
3. **Level 1** (~6,000+ regions): Fine-grained analysis only for the most discriminative L2 sub-regions.
4. **Transition analysis**: Build bigram transition matrices at L3. Are depression users "trapped" in certain regions or oscillating differently?

### Phase 3: Classification with Feature Ablation

Build classifiers with progressive feature sets:

| Experiment | Features | Expected AUC |
|-----------|---------|-------------|
| Behavioral only | §3.2 | 0.55–0.65 |
| Static embedding | Mean MentalRoBERTa per user | 0.85–0.92 |
| Temporal embedding | Recency-weighted aggregation | 0.88–0.94 |
| Region (L3) + behavioral | §3.1 + §3.2 | 0.80–0.90 |
| Full model | All §3.1–3.4 | 0.90–0.95 |

Each experiment: train on train split, tune threshold on val, evaluate on test with bootstrap CI.

### Phase 4: Early Detection (ERDE)

1. For each user, process posts sequentially in order
2. At each chunk boundary (10%, 20%, ..., 100%), extract features from posts seen so far
3. Classify and record decision + confidence
4. Compute ERDE₅, ERDE₅₀, and AUC@k% curves
5. Compare: how early can we detect with behavioral-only vs full model?

### Phase 5: Pattern Analysis & Interpretability

1. **Depression prototypes**: For train depression users, compute centroid trajectories at L3. What does a "typical depression pattern" look like?
2. **Discriminative regions**: Rank regions by mutual information with label. Map to topics via c-TF-IDF.
3. **Temporal pattern mining**: At what relative time (t_rel) does the depression signal become strongest? Is there a "critical window"?
4. **Case studies**: 5 users with detailed timeline analysis (TP, FP, FN)

---

## 6. Post-hoc Analysis

### 6.1 Ablation Study

| Ablation | Remove | Measures |
|----------|--------|---------|
| No behavioral features | Gap, frequency, circadian | Importance of posting patterns |
| No region features | All region distributions | Importance of topical structure |
| No temporal weighting | Use mean instead of recency attention | Importance of temporal attention |
| MiniLM instead of MentalRoBERTa | Swap embeddings | Importance of domain adaptation |
| Absolute timestamps | Replace relative time | Importance of relative encoding |
| Only L3 regions | Remove L2, L1 | Is coarse enough? |

### 6.2 Cross-Dataset Analysis

- Feature importance correlation (Spearman) across eRisk and CLPsych
- Transfer: train on eRisk, test on CLPsych (zero-shot)
- Do the same coarse regions emerge with similar c-TF-IDF labels?

### 6.3 Error Analysis

- False negatives: Do they have fewer posts? More "normal" posting patterns?
- False positives: Are they in grief/stress that mimics depression language?
- Confusion matrix per dataset with user characteristics

---

## 7. Deliverables

### Notebooks (Per Dataset)

| Notebook | Content |
|----------|---------|
| `B1_eRisk.ipynb` | Full pipeline for eRisk (primary) |
| `B1_CLPsych.ipynb` | CLPsych replication |
| `B1_RSDD.ipynb` | RSDD exploratory (expected to be weak) |
| `B1_ablation.ipynb` | Feature ablation study |
| `B1_early_detection.ipynb` | ERDE analysis + AUC@k% curves |

### Key Figures

- Feature discriminability heatmap (all features × datasets)
- Top-down region analysis visualization
- ERDE curves (AUC vs % posts observed)
- c-TF-IDF word clouds for discriminative regions
- Temporal signal emergence plot (discriminability vs relative time)
- Case study timelines

---

## 8. Statistical Rigor

- All p-values Bonferroni-corrected
- Effect sizes (Cohen's d) alongside p-values
- 95% bootstrap CI (n=1000) for all test-set metrics
- McNemar test for method comparisons
- No test-set results reported until hyperparameters frozen on validation
- ERDE metrics computed with official eRisk evaluation code

---

## 9. Ethical Considerations

- All datasets obtained through official data use agreements
- No attempt to de-anonymize users
- Raw text data never committed to repository
- Results reported in aggregate; case studies use anonymized IDs
- Clinical deployment would require IRB approval and clinician oversight
- Tool aids clinicians — does not replace clinical judgment

---

## 10. Implementation Priority

| Priority | Action | Impact | Effort |
|----------|--------|--------|--------|
| **P0** | Re-embed with MentalRoBERTa | High — domain-specific emotional signal | Medium (GPU, ~2h) |
| **P0** | Add relative time + gap features | High — correct temporal encoding | Low (preprocessing) |
| **P1** | Behavioral features (gap analysis) | Medium — new signal type | Low |
| **P1** | Top-down hierarchy (L3 first) | Medium — reduces dimensionality | Low |
| **P2** | Recency-weighted attention | Medium — better aggregation | Medium |
| **P2** | ERDE early detection evaluation | Medium — validates core use case | Medium |
| **P3** | Cross-dataset transfer | Low — validation | Low |

---

## References

1. Couto, M. et al. (2025). Temporal word embeddings for psychological disorder early detection. *JHIR*.
2. Ji, S. et al. (2022). MentalBERT: Publicly available pretrained language models for mental healthcare. *LREC*.
3. Coppersmith, G. et al. (2018). NLP of social media as screening for suicide risk. *BMI Insights*.
4. De Choudhury, M. et al. (2013). Predicting depression via social media. *ICWSM*.
5. Losada, D. et al. (2019). eRisk: early risk prediction on the internet. *CLEF Working Notes*.
6. Yates, A. et al. (2017). Depression and self-harm risk assessment in online forums. *EMNLP*.
7. Killick, R. et al. (2012). Optimal detection of changepoints. *JASA*.
8. Malkov, Y.A. & Yashunin, D.A. (2018). Efficient and robust ANN using HNSW graphs. *IEEE TPAMI*.
9. Bamler, R. & Mandt, S. (2017). Dynamic word embeddings. *ICML*.
10. Grootendorst, M. (2022). BERTopic: neural topic modeling with c-TF-IDF. *arXiv:2203.05794*.
11. Hofmann, V. et al. (2021). Dynamic Contextualized Word Embeddings. *ACL*.
12. DS@GT (2025). From prompts to predictions: temporal attention models for eRisk 2025. *CLEF Working Notes*.
13. MDHAN (2022). Hierarchical Attention Network for Explainable Depression Detection. *COLING*.
14. Bucur, A.M. (2024). Computational Approaches to Mental Health. *PhD Thesis, UPV*.
