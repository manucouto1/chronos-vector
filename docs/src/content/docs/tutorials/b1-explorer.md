---
title: "Mental Health Explorer"
description: "Interactive exploration of depression detection with 17 CVX analytical functions on eRisk data"
---

:::note
This tutorial requires the **eRisk dataset** (1.36M Reddit posts, D=768 MentalRoBERTa embeddings).
Run the notebook locally for full interactive Plotly visualizations with real outputs.
:::

## Overview

This notebook applies all 17 CVX analytical functions to real depression detection data from the eRisk shared task. Unlike the synthetic tutorials ([MAP-Elites](/tutorials/map-elites), [MLOps Drift](/tutorials/mlops-drift)), this uses **real clinical NLP data** — 233 depression users + 233 matched controls, embedded with MentalRoBERTa (D=768).

### What the notebook demonstrates

| Section | CVX Functions | What It Reveals |
|---------|--------------|-----------------|
| Ingestion | `bulk_insert`, `enable_quantization` | SQ8 accelerates HNSW construction ~2.4x |
| 3D Trajectories | `trajectory` | Users trace paths through embedding space over time |
| HNSW Hierarchy | `regions`, `region_members` | Semantic clusters emerge from the graph, labeled via c-TF-IDF |
| Temporal Calculus | `velocity`, `drift`, `hurst_exponent`, `detect_changepoints` | Depression shows anti-persistent dynamics (H<0.5) |
| Point Process | `event_features` | Burstiness and circadian disruption differ between groups |
| Distributional | `region_trajectory`, `wasserstein_drift`, `fisher_rao_distance`, `hellinger_distance` | Track semantic migration at L3 |
| Topology | `topological_features` | Betti curves reveal cluster structure differences |
| Signatures | `path_signature`, `signature_distance`, `frechet_distance` | Universal trajectory comparison in O(K²) |
| Deep-Dive | All of the above | 5-panel aligned dashboard for a single user |

## Running the notebook

```bash
# 1. Activate environment and build bindings
conda activate cvx
cd crates/cvx-python && maturin develop --release && cd ../..

# 2. Ensure eRisk embeddings are available
ls data/embeddings/erisk_mental_embeddings.parquet

# 3. Launch the notebook
jupyter notebook notebooks/B1_interactive_explorer.ipynb

# 4. (Optional) Generate docs page with outputs
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=1200 \
  notebooks/B1_interactive_explorer.ipynb
python scripts/nb2docs.py \
  --input notebooks/B1_interactive_explorer.ipynb \
  --output docs/src/content/docs/tutorials/b1-explorer.md \
  --plots-dir docs/public/plots/ \
  --title "Mental Health Explorer" \
  --description "Interactive exploration of depression detection"
```

Execution takes ~10 minutes (226K point ingestion at D=768 with SQ8).

## Key findings

From previous executions on the eRisk dataset:

- **Hurst exponent**: Depression users show more persistent dynamics (H=0.65 mean) — their language trends directionally rather than oscillating
- **Event features**: Burstiness B=0.64 (bursty posting patterns), circadian disruption visible in night_ratio
- **Region trajectory**: Fisher-Rao distances at L3 (~37 regions) are informative (d≈1.16), not saturated
- **Path signatures**: Depth-2 signatures at L3 produce ~1,482-dim fingerprints, tractable for PCA visualization
- **Topology**: L3 region centroids show meaningful Betti curves

## Related

- [B1: Classification Results](/tutorials/b1-mental-health) — Detailed AUC/F1 metrics with bootstrap CIs
- [Research Protocol 001](https://github.com/manucouto1/chronos-vector/blob/develop/design/CVX_Research_Protocol_001_Mental_Health.md) — Full research protocol
- [Temporal Analytics Toolkit](/specs/temporal-analytics) — API reference for all 17 functions
