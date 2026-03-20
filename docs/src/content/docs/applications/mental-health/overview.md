---
title: "Mental Health & Clinical NLP"
description: "Detecting psychological distress from social media using temporal trajectory analysis"
---

## Overview

CVX enables a fundamentally different approach to mental health detection from social media: instead of treating each post as an independent feature vector, it tracks **how a user's language evolves over time** — velocity, drift direction, change points, and proximity to clinical symptom anchors.

This is the most mature application domain for CVX, validated on the eRisk shared task dataset (1.36M Reddit posts, 2,285 users) with MentalRoBERTa embeddings (D=768).

## Key Findings

### Embedding Anisotropy Correction (30x Signal Improvement)

MentalRoBERTa embeddings occupy a narrow cone in high-dimensional space — all pairwise cosine similarities are ~0.96 regardless of content. This makes raw anchor projections useless.

**Centering** (subtracting the global mean vector) amplifies the discriminative signal 30x:

| Metric | Before centering | After centering |
|--------|-----------------|-----------------|
| Depression user → depressed_mood anchor | cosine sim 0.975 | cosine sim 0.42 |
| Control user → depressed_mood anchor | cosine sim 0.964 | cosine sim 0.09 |
| **Discriminative gap** | **0.011** | **0.33** |

This correction benefits ALL downstream CVX operations. See [RFC-012 Part B](/chronos-vector/rfc/rfc-012) for the academic references (Ethayarajh 2019, Su et al. 2021).

### DSM-5 Symptom Proximity Profiling (B3)

9 DSM-5 symptom anchors + 1 healthy baseline, encoded as MentalRoBERTa centroids of representative clinical phrases. After centering, population-level profiles clearly separate depression from control:

- **Depression users**: highest proximity to `depressed_mood` (0.35), `worthlessness` (0.31), `anhedonia` (0.28)
- **Control users**: uniformly low proximity across all symptoms (0.05-0.09)
- **Drift direction**: depression users show *increasing* proximity to symptoms over time; controls remain stable

### Hierarchical Semantic Clustering (B1)

HNSW's natural hierarchy produces semantic regions with strong clinical meaning:

- Level 2 regions show clusters with depression ratios from 0.15 to 0.85
- High-depression clusters contain posts about hopelessness, isolation, sleep disruption
- Low-depression clusters contain posts about social activities, hobbies, future plans
- This demonstrates **unsupervised specialization** — the graph structure separates clinical from non-clinical content without labels

### Classification Results (B2)

Anchor-projected temporal features on a proper temporal split (train 2017+2018 → test 2022):

| Model | F1 | AUC | Precision | Recall |
|-------|-----|-----|-----------|--------|
| B1 Baseline (absolute features) | 0.600 | 0.639 | 0.590 | 0.614 |
| **B2 Combined (anchor + polarization + velocity)** | **0.744** | **0.886** | **0.739** | **0.750** |
| Early detection (10% of posts) | 0.673 | — | — | — |

## CVX Features Used

| Feature | Purpose |
|---------|---------|
| `project_to_anchors()` | DSM-5 symptom proximity trajectories |
| `anchor_summary()` | Per-user mean, min, trend per symptom |
| `drift()` / `velocity()` | Rate and direction of linguistic change |
| `detect_changepoints()` | Onset/escalation event detection (PELT) |
| `regions()` / `region_assignments()` | Unsupervised semantic clustering via HNSW hierarchy |
| `hurst_exponent()` | Long-memory estimation (persistent vs antipersistent drift) |
| `path_signature()` | Trajectory shape classification |
| Centering (manual, RFC-012 pending) | Anisotropy correction for 30x signal improvement |

## Notebooks

| Notebook | Focus | Status |
|----------|-------|--------|
| B1_interactive_explorer | HNSW hierarchy visualization, depression ratio per cluster | Best: cluster visualization |
| B2_clinical_anchoring | Anchor projection pipeline, classification benchmarks | Complete |
| B3_clinical_dashboard | DSM-5 radar, symptom drift direction, clinical timeline | Best: population profiles, drift tracking |
| B1_erisk_rigorous | Rigorous eRisk evaluation (SVC+PCA, no data balancing) | Complete |

## Related

- [RFC-006: Anchor Projection](/chronos-vector/rfc/rfc-006)
- [RFC-012: Embedding Space Centering](/chronos-vector/rfc/rfc-012)
- [B1 Improvement Plan](/chronos-vector/research/B1-improvement-plan)
