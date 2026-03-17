---
title: "Mental Health Detection"
description: "Detecting psychological distress from social media using temporal vector analytics"
---

## Overview

Social media posts contain temporal linguistic signals that static models discard. By embedding each post with **MentalRoBERTa** (D=768, domain-adapted to mental health language) and storing them as timestamped trajectories in ChronosVector, we recover behavioral and linguistic patterns invisible to snapshot-based analysis:

- **Velocity** of linguistic change accelerates before crisis episodes
- **Posting rhythms** encode circadian disruption (a clinical depression marker)
- **Topical migration** reveals progressive withdrawal from social topics
- **Path signatures** fingerprint the *shape* of a user's linguistic evolution over time

This page walks through each CVX analytical function applied to eRisk (Reddit, depression detection) and CLPsych (Twitter, mental health triage).

---

## Data Model

Each social media user maps to a CVX entity. Each post becomes a timestamped vector. The timestamp is the post's publication time (Unix seconds).

```python
import chronos_vector as cvx
import numpy as np

# Create index with HNSW parameters tuned for 768-dim embeddings
index = cvx.TemporalIndex(m=16, ef_construction=200, ef_search=50)
index.enable_quantization(-1.0, 1.0)  # MentalRoBERTa outputs are normalized

# entity_ids: np.ndarray[uint64] — user IDs
# timestamps: np.ndarray[int64]  — post times (Unix seconds)
# vectors:    np.ndarray[float32, (N, 768)] — MentalRoBERTa embeddings
n = index.bulk_insert(entity_ids, timestamps, vectors, ef_construction=50)
print(f"Ingested {n} posts from {len(np.unique(entity_ids))} users")
```

Lowering `ef_construction` to 50 during bulk load speeds up ingestion substantially. The HNSW graph self-organizes: higher levels form coarse semantic regions used later for topical analysis.

---

## Temporal Calculus --- Velocity and Change Points

### Velocity: rate of linguistic change

`velocity()` computes the instantaneous rate of change in embedding space via central finite differences. A sudden velocity spike means the user's language is shifting rapidly, which can signal the onset of a depressive episode or crisis.

```python
traj = index.trajectory(entity_id=user_id)

# Velocity at a specific timestamp
mid_ts = traj[len(traj) // 2][0]
vel = cvx.velocity(traj, timestamp=mid_ts)
magnitude = np.linalg.norm(vel)
print(f"Velocity magnitude at t={mid_ts}: {magnitude:.4f}")
```

Clinically, sustained high velocity corresponds to rapid topic switching or emotional volatility. A velocity that spikes and does not return to baseline may indicate a regime change.

### Change point detection: behavioral regime shifts

`detect_changepoints()` uses the PELT algorithm to find timestamps where the user's linguistic behavior shifts significantly. Each change point includes a severity score proportional to the statistical cost reduction.

```python
changepoints = cvx.detect_changepoints(
    entity_id=user_id,
    trajectory=traj,
    penalty=None,         # Auto-calibrate via BIC
    min_segment_len=5     # At least 5 posts per segment
)

for ts, severity in changepoints:
    print(f"  Regime shift at t={ts}, severity={severity:.2f}")
```

On eRisk data, depression users show significantly more change points than controls, reflecting the episodic nature of depressive states. The severity scores help distinguish minor topic shifts from major behavioral transitions.

---

## Stochastic Characterization --- Hurst Exponent

The Hurst exponent H characterizes the long-range dependence of a trajectory via R/S (rescaled range) analysis:

| H value | Behavior | Clinical interpretation |
|---------|----------|----------------------|
| H < 0.5 | Anti-persistent (mean-reverting) | Erratic topic switching, inability to sustain focus |
| H = 0.5 | Random walk | No temporal memory |
| H > 0.5 | Persistent (trending) | Sustained topical engagement |

```python
h = cvx.hurst_exponent(traj)
print(f"Hurst exponent: {h:.3f}")
```

On CLPsych, depression users show significantly lower Hurst exponents than controls (Cohen's d = -0.41, p < 0.001). This anti-persistence reflects a clinical pattern: depressed individuals cycle erratically between topics rather than maintaining coherent thematic threads. The effect is consistent across both datasets, making H a robust temporal biomarker.

---

## Point Process Analysis --- Posting Patterns

`event_features()` analyzes the *timing* of posts as an independent behavioral signal, separate from content. It extracts 11 features from the inter-post interval distribution.

```python
timestamps_user = [ts for ts, _ in traj]
features = cvx.event_features(timestamps_user)
```

Key features and their clinical relevance:

| Feature | Depression signal | Effect size (eRisk) |
|---------|------------------|-------------------|
| `burstiness` | Higher — posting in intense bursts followed by withdrawal | d = 0.28 |
| `memory` | Lower — less temporal autocorrelation in gaps | d = -0.19 |
| `circadian_strength` | Lower — disrupted daily rhythm | d = -0.31 |
| `gap_cv` | Higher — irregular posting intervals | d = 0.32 |
| `max_gap` | Higher — longer withdrawal periods | d = 0.25 |

The strongest individual behavioral signal is **night posting ratio** (posts between 00:00-06:00): depression users post 31% at night vs. 22% for controls (d = 0.534, p < 0.001 on eRisk). Circadian disruption is a well-established clinical marker of major depressive disorder.

---

## Semantic Region Migration

CVX discovers semantic regions from the HNSW graph hierarchy. At Level 3, the graph produces 60-97 coarse regions that act as topical clusters. Tracking how a user's posts distribute across these regions over time reveals topical migration patterns.

### Region trajectory

```python
# Discover coarse semantic regions
regions = index.regions(level=3)
print(f"Found {len(regions)} regions at Level 3")

# Track user's region distribution over time (EMA-smoothed)
region_traj = index.region_trajectory(
    entity_id=user_id,
    level=3,
    window_days=14,   # 14-day sliding window
    alpha=0.3         # EMA smoothing factor
)
# region_traj: list of (timestamp, distribution) where distribution sums to ~1.0
```

### Wasserstein drift: geometry-aware topical shift

`wasserstein_drift()` measures how a user's topical distribution changes between two time points, respecting the spatial structure of regions. Unlike L2 distance between histograms, Wasserstein accounts for *which* regions are semantically close vs. distant.

```python
centroids = [centroid for _, centroid, _ in regions]

# Compare early vs. late distributions
early_dist = region_traj[0][1]
late_dist = region_traj[-1][1]

w_drift = cvx.wasserstein_drift(early_dist, late_dist, centroids, n_projections=50)
print(f"Wasserstein drift: {w_drift:.4f}")
```

### Fisher-Rao distance: invariant distributional metric

For pure distributional comparison (independent of region geometry), `fisher_rao_distance()` provides the unique Riemannian metric on the statistical manifold, invariant under sufficient statistics.

```python
fr_dist = cvx.fisher_rao_distance(
    [float(x) for x in early_dist],
    [float(x) for x in late_dist]
)
print(f"Fisher-Rao distance: {fr_dist:.4f}")  # Range [0, pi]
```

Depression users show higher Wasserstein drift over time, indicating progressive migration away from social and neutral topics toward isolation-related semantic regions.

---

## Topological Analysis

`topological_features()` applies persistent homology to region centroids, revealing the *structure* of a user's topic space rather than just its content.

```python
# Extract centroids for regions the user actually occupies
user_region_ids = [
    i for i, weight in enumerate(late_dist) if weight > 0.01
]
user_centroids = [centroids[i] for i in user_region_ids]

topo = cvx.topological_features(user_centroids, n_radii=20, persistence_threshold=0.1)
print(f"Significant clusters: {topo['n_components']}")
print(f"Max persistence:      {topo['max_persistence']:.3f}")
print(f"Persistence entropy:  {topo['persistence_entropy']:.3f}")
```

Depression users' topic spaces tend to show:
- **Higher fragmentation** (more connected components) — topics are scattered without coherent clustering
- **Lower max persistence** — no dominant topical cluster, suggesting lack of sustained interest
- **Different Betti curves** — the rate at which components merge as radius grows differs, reflecting a fundamentally different topological structure of their semantic engagement

---

## Path Signatures --- Trajectory Fingerprinting

Path signatures from rough path theory provide a *universal nonlinear feature* of sequential data. Applied to region trajectories, they capture the ordered, multi-scale shape of how a user's topical engagement evolves.

```python
# Compute Level-3 path signature on region trajectory
# Applied on region distributions (~80 dims), NOT raw embeddings (768 dims)
sig = cvx.path_signature(region_traj, depth=3, time_augmentation=True)
print(f"Signature dimension: {len(sig)}")
```

At depth 3, the signature captures:
- **Depth 1**: Net displacement — *where* did the user's topic focus move
- **Depth 2**: Signed areas — *how* it moved (rotation, correlation between topic shifts)
- **Depth 3**: Higher-order interactions — complex temporal patterns

### Cohort detection via signature distance

`signature_distance()` finds users whose linguistic evolution follows similar temporal patterns. This enables at-risk cohort detection: find users whose trajectory shape matches known depression cases.

```python
# Compare two users' trajectory signatures
sig_a = cvx.path_signature(region_traj_a, depth=3, time_augmentation=True)
sig_b = cvx.path_signature(region_traj_b, depth=3, time_augmentation=True)

dist = cvx.signature_distance(sig_a, sig_b)
print(f"Signature distance: {dist:.4f}")
# Lower distance = more similar temporal evolution pattern
```

Signature distance operates in O(K^2) per comparison, making it practical for screening large user populations against a library of known at-risk trajectory templates.

---

## Results Summary

| Dataset | Model | AUC | F1 | Early Detection |
|---------|-------|----:|---:|-----------------|
| eRisk (Reddit, N=2,285) | Full (Temporal + Region + Behavioral) | **0.911** | 0.458 | AUC@10% = 0.849 |
| CLPsych (Twitter, N=674) | Temporal + Behavioral | **0.804** | 0.571 | AUC@20% = 0.813 |

Key takeaways:
- **MentalRoBERTa embeddings** are the single most impactful choice (static baseline AUC = 0.901 on eRisk)
- **Temporal features** add signal beyond static aggregation, especially for early detection
- **Behavioral posting patterns** (night posting, burstiness) are lightweight and independently discriminative
- **Region-based features** compress 768-dim embeddings to ~80 dims with minimal AUC loss (0.890 vs 0.911)
- **Early detection is viable**: AUC > 0.84 with only 10% of a user's post history on eRisk

---

## References

1. Couto, M. et al. (2025). Temporal word embeddings for psychological disorder early detection. *Journal of Healthcare Informatics Research*.
2. Goh, K.-I. & Barabasi, A.-L. (2008). Burstiness and memory in complex systems. *EPL (Europhysics Letters)*, 81(4).
3. Lyons, T. (1998). Differential equations driven by rough signals. *Revista Matematica Iberoamericana*, 14(2).
4. Rao, C.R. (1945). Information and accuracy attainable in the estimation of statistical parameters. *Bulletin of the Calcutta Mathematical Society*, 37.
