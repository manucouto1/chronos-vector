---
title: "Examples"
description: "Code examples demonstrating ChronosVector's 17 analytical functions across 6 domains"
---

ChronosVector (CVX) ships **17 analytical functions** that turn a temporal vector index into a full analytical engine. This page demonstrates the core API with concise, practical examples drawn from the [Interactive Explorer notebook](/chronos-vector/notebooks/B1_interactive_explorer).

For domain-specific walkthroughs, see the [Cross-Domain Examples](#cross-domain-examples) table at the bottom.

---

## Quick Start

Create an index, ingest temporal vectors, enable quantization, and run your first search and trajectory retrieval.

```python
import chronos_vector as cvx
import numpy as np

# 1. Create the HNSW temporal index
index = cvx.TemporalIndex(m=16, ef_construction=200, ef_search=50)

# 2. Enable scalar quantization (SQ8) before insertion for ~2.4x speedup
index.enable_quantization(vmin=-1.5, vmax=1.5)

# 3. Bulk insert — entities, unix-second timestamps, float32 vectors
n = index.bulk_insert(entity_ids, timestamps, vectors, ef_construction=50)
# => "Ingested 48,312 points in 2.1s (23,006 pts/sec)"

# 4. Search: k-NN with optional temporal window
results = index.search(query_vec, k=10)
# => list of (entity_id, timestamp, distance)

# 5. Retrieve a single entity's full trajectory
traj = index.trajectory(entity_id=42)
# => [(ts_0, vec_0), (ts_1, vec_1), ...] sorted by time
```

---

## 3D Trajectory Visualization

Every entity traces a path through embedding space over time. PCA projects the vectors to 2D; the temporal axis provides the third dimension.

```python
traj = index.trajectory(entity_id=eid)
# traj: list of (unix_timestamp, np.ndarray) pairs, sorted chronologically

# Use traj directly for per-entity analysis, or collect all
# trajectories and project with PCA for a global view.
```

`trajectory` is the fundamental building block: every downstream function (velocity, drift, signatures) consumes this list.

---

## HNSW Hierarchy Exploration

CVX exposes the HNSW graph hierarchy as semantic regions. Higher levels yield coarser clusters.

```python
# Discover regions at each HNSW level
for level in [1, 2, 3]:
    regions = index.regions(level=level)
    print(f"Level {level}: {len(regions)} regions")
# Level 1: 4,218 regions
# Level 2:   897 regions
# Level 3:    71 regions

# Inspect a region: centroid vector + member posts
regions_l2 = index.regions(level=2)
# => list of (region_id, centroid_vec, n_members)

members = index.region_members(region_id=42, level=2)
# => list of (entity_id, timestamp) tuples belonging to this region
```

Region centroids live in the same vector space as your data, so you can project them with PCA, compute TF-IDF labels from member texts, or measure distributional distances between them.

---

## Temporal Calculus

CVX computes vector-space derivatives natively: instantaneous velocity, displacement drift, long-range dependence (Hurst exponent), and regime change detection.

### Velocity and Drift

```python
traj = index.trajectory(entity_id=eid)

# Instantaneous velocity at a given timestamp (central differences)
vel = cvx.velocity(traj, timestamp=ts)
vel_magnitude = np.linalg.norm(vel)
# => rate of semantic change at that moment

# Drift: displacement decomposition between any two vectors
l2_magnitude, cosine_drift, top_dims = cvx.drift(vec_a, vec_b, top_n=3)
# l2_magnitude : Euclidean distance
# cosine_drift : 1 - cosine_similarity
# top_dims     : indices of the 3 dimensions with largest absolute change
```

### Hurst Exponent

```python
h = cvx.hurst_exponent(traj)
# H < 0.5 : anti-persistent (erratic, mean-reverting)
# H = 0.5 : random walk
# H > 0.5 : persistent (trending)
```

In mental health data, depression users tend toward anti-persistent dynamics (H < 0.5), reflecting erratic semantic shifts.

### Change Point Detection

```python
changepoints = cvx.detect_changepoints(
    entity_id, traj, penalty=None, min_segment_len=5
)
# => list of (timestamp, severity) pairs marking regime transitions
```

Each change point marks a moment where the entity's trajectory undergoes a statistically significant shift.

---

## Point Process Analysis

CVX analyzes the **timing** of events independently from content: burstiness, memory, temporal entropy, and circadian patterns.

```python
ts_list = [t for t, _ in traj]

feats = cvx.event_features(ts_list)
# Returns a dict with:
#   burstiness        : (sigma - mu) / (sigma + mu), in [-1, 1]
#   memory            : lag-1 autocorrelation of inter-event times
#   n_events          : total count
#   circadian_strength: 24h Fourier amplitude (0 = no rhythm, 1 = strong)
#   temporal_entropy  : Shannon entropy of the inter-event distribution
```

Burstiness and memory form a 2D behavioral fingerprint: bursty-clustered posting (B > 0, M > 0) reveals different temporal patterns than regular-alternating posting (B < 0, M < 0).

---

## Distributional Distances

Track how an entity's semantic distribution migrates across regions over time, then measure that drift with three complementary metrics.

### Region Trajectory

```python
reg_traj = index.region_trajectory(
    entity_id=eid, level=2, window_days=30, alpha=0.3
)
# => list of (timestamp, distribution_vec) pairs
# Each distribution_vec is a probability vector over all regions at the given level
```

### Wasserstein, Fisher-Rao, and Hellinger

```python
p, q = reg_traj[0][1], reg_traj[-1][1]  # first and last distributions
centroids = [c for _, c, _ in index.regions(level=2)]

# Wasserstein (optimal transport, respects graph geometry)
wd = cvx.wasserstein_drift(list(p), list(q), centroids, n_projections=50)

# Fisher-Rao (unique invariant metric on the statistical manifold, d in [0, pi])
fr = cvx.fisher_rao_distance(list(p), list(q))

# Hellinger (bounded, symmetric, d in [0, 1])
hd = cvx.hellinger_distance(list(p), list(q))
```

Each metric captures a different aspect of distributional change. Wasserstein accounts for the geometry of the embedding space (regions that are close semantically contribute less to drift). Fisher-Rao is reparameterization-invariant. Hellinger is bounded and interpretable as overlap.

---

## Topological Analysis

Persistent homology (TDA) reveals the **shape** of the semantic space: how many clusters exist, how well-separated they are, and how topology changes at different scales.

```python
centroids = [c for _, c, _ in index.regions(level=2)]

topo = cvx.topological_features(
    centroids, n_radii=30, persistence_threshold=0.05
)
# Returns a dict with:
#   n_components        : number of significant connected components
#   max_persistence     : lifetime of the most persistent feature
#   persistence_entropy : Shannon entropy of the persistence diagram
#   betti_curve         : beta_0(r) values (connected components vs. radius)
#   radii               : filtration radii corresponding to betti_curve
```

The Betti curve shows how many disconnected clusters survive as you grow the neighborhood radius. A slow monotonic decay means well-separated clusters; a sharp drop indicates a few dominant hubs.

---

## Path Signatures

From rough path theory (Lyons, 1998): the **universal nonlinear feature** of sequential data. Any continuous function of a path can be approximated by a linear function of its signature.

```python
# Compute on region trajectories (K~71 at Level 3) for tractable dimensions
rt = index.region_trajectory(entity_id=eid, level=3, window_days=30, alpha=0.3)
sig_traj = [(t, [float(x) for x in d]) for t, d in rt]

sig = cvx.path_signature(sig_traj, depth=2, time_augmentation=True)
# => np.ndarray of dimension K + 1 + (K + 1)^2 (with time augmentation)
```

### Signature Distance and Frechet Distance

```python
# Compare two entities by their path signatures
d_sig = cvx.signature_distance(sig_a, sig_b)
# => L2 distance in signature space (captures full trajectory shape)

# Or compare raw trajectories with the Frechet distance
traj_a = index.trajectory(entity_id=eid_a)
traj_b = index.trajectory(entity_id=eid_b)
d_frechet = cvx.frechet_distance(traj_a[:100], traj_b[:100])
# => maximum deviation between optimally aligned paths
```

Signature distance operates in a lifted feature space where trajectory order and interactions are encoded. Frechet distance works directly in the embedding space and measures worst-case alignment.

---

## Cross-Domain Examples

CVX is domain-agnostic. The same 17 functions apply wherever you have entities evolving through vector space over time.

| Domain | Key CVX Functions | Page |
|--------|------------------|------|
| **Mental Health Detection** | `hurst_exponent`, `event_features`, `detect_changepoints` | [Mental Health](/chronos-vector/examples/mental-health) |
| **Quality-Diversity (MAP-Elites)** | `regions`, `topological_features`, `path_signature` | [MAP-Elites](/chronos-vector/examples/map-elites) |
| **Molecular Dynamics** | `velocity`, `frechet_distance`, `wasserstein_drift` | [Molecular Dynamics](/chronos-vector/examples/molecular-dynamics) |
| **Drug Discovery** | `region_trajectory`, `fisher_rao_distance`, `signature_distance` | [Drug Discovery](/chronos-vector/examples/drug-discovery) |
| **MLOps Drift Detection** | `wasserstein_drift`, `detect_changepoints`, `hellinger_distance` | [MLOps Drift](/chronos-vector/examples/mlops-drift) |
| **Quantitative Finance** | `hurst_exponent`, `path_signature`, `event_features` | [Finance](/chronos-vector/examples/finance) |
