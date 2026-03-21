---
title: "Python API Reference"
description: "Complete reference for the chronos-vector Python package"
---

## Installation

```bash
pip install chronos-vector
```

For development builds:
```bash
cd crates/cvx-python
maturin develop --release
```

---

## TemporalIndex

The main class. Wraps a spatiotemporal HNSW index with temporal edges for causal search.

### Constructor

```python
index = cvx.TemporalIndex(m=16, ef_construction=200, ef_search=50, model_path=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `m` | int | 16 | HNSW connections per node |
| `ef_construction` | int | 200 | Search width during construction |
| `ef_search` | int | 50 | Search width during queries |
| `model_path` | str? | None | Path to TorchScript Neural ODE model (requires torch-backend feature) |

### Insert & Ingest

```python
# Single point (with optional reward)
node_id = index.insert(entity_id, timestamp, vector, reward=None)

# Bulk insert from numpy arrays
n_inserted = index.bulk_insert(entity_ids, timestamps, vectors, ef_construction=None)
```

| Method | Parameters | Returns |
|--------|-----------|---------|
| `insert(entity_id, timestamp, vector, reward=None)` | entity_id: int, timestamp: int, vector: list[float], reward: float? | node_id: int |
| `bulk_insert(entity_ids, timestamps, vectors, ef_construction=None)` | numpy arrays (uint64, int64, float32[N,D]) | count: int |

**Reward**: Optional float (0.0-1.0) annotating outcome quality. Use `set_reward()` to annotate retroactively.

### Search

```python
# Standard kNN search
results = index.search(vector, k=10, alpha=1.0, query_timestamp=0,
                       filter_start=None, filter_end=None)
# Returns: list of (entity_id, timestamp, score)

# Reward-filtered search
results = index.search_with_reward(vector, k=10, min_reward=0.0, alpha=1.0,
                                   query_timestamp=0, filter_start=None, filter_end=None)
# Returns: list of (entity_id, timestamp, score) — only nodes with reward >= min_reward

# Causal search — find similar states + what happened next
results = index.causal_search(vector, k=5, temporal_context=5, alpha=1.0,
                              query_timestamp=0, filter_start=None, filter_end=None)
# Returns: list of dicts:
#   { "node_id": int, "score": float, "entity_id": int,
#     "successors": [(node_id, timestamp, vector), ...],
#     "predecessors": [(node_id, timestamp, vector), ...] }

# Hybrid search — beam search exploring semantic + temporal neighbors
results = index.hybrid_search(vector, k=10, beta=0.3, alpha=1.0,
                              query_timestamp=0, filter_start=None, filter_end=None)
# Returns: list of (entity_id, timestamp, score)
```

| Parameter | Description |
|-----------|-------------|
| `alpha` | Semantic vs temporal weight. 1.0 = pure semantic, 0.0 = pure temporal |
| `beta` | Temporal edge exploration weight (hybrid_search only). 0.0 = pure semantic, 1.0 = aggressive temporal |
| `temporal_context` | Steps to walk forward/backward in causal_search |
| `min_reward` | Minimum reward threshold (search_with_reward only) |
| `filter_start/end` | Optional temporal range filter |

### Trajectory

```python
traj = index.trajectory(entity_id, start=None, end=None)
# Returns: list of (timestamp, vector)
```

### Outcome / Reward

```python
index.set_reward(node_id, reward)       # Annotate retroactively
r = index.reward(node_id)               # Returns float or None
```

### Centering (Anisotropy Correction)

```python
centroid = index.compute_centroid()      # O(N×D) — returns list[float] or None
index.set_centroid(centroid)             # Enable centering
index.clear_centroid()                   # Disable centering
c = index.centroid()                     # Get current centroid or None
centered = index.centered_vector(vec)   # Returns vec - centroid
```

### Semantic Regions

```python
regions = index.regions(level=2)
# Returns: list of (hub_id, hub_vector, member_count)

assignments = index.region_assignments(level=3, start=None, end=None)
# Returns: dict { hub_id: [(entity_id, timestamp), ...] }

members = index.region_members(region_id, level=3, start=None, end=None)
# Returns: list of (entity_id, timestamp)

traj = index.region_trajectory(entity_id, level=2, window_days=7, alpha=0.3)
# Returns: list of (timestamp, distribution_vector)
```

### Persistence

```python
index.save(path)                        # Save to directory (index.bin + temporal_edges.bin)
index = cvx.TemporalIndex.load(path)    # Load (supports both directory and legacy .cvx format)
```

### Configuration

```python
index.set_ef_construction(ef)           # Tune build quality
index.set_ef_search(ef)                 # Tune search quality vs speed
index.enable_quantization(-1.0, 1.0)    # ~4× faster distance computation
index.disable_quantization()
len(index)                              # Number of points
```

---

## Analytics Functions

All functions are module-level: `cvx.function_name(...)`.

### Vector Calculus

```python
vel = cvx.velocity(trajectory, timestamp)
# Returns: list[float] — dv/dt at given timestamp

d = cvx.drift(v1, v2, top_n=5)
# Returns: (l2_magnitude, cosine_drift, top_dimensions)

features = cvx.temporal_features(trajectory)
# Returns: list[float] — fixed-size feature vector

h = cvx.hurst_exponent(trajectory)
# Returns: float — H > 0.5 trending, H < 0.5 mean-reverting
```

### Change Point Detection

```python
cps = cvx.detect_changepoints(entity_id, trajectory, penalty=None, min_segment_len=2)
# Returns: list of (timestamp, severity)
```

### Path Signatures

```python
sig = cvx.path_signature(trajectory, depth=2, time_augmentation=False)
# Returns: list[float] — depth 1: K features, depth 2: K + K²

log_sig = cvx.log_signature(trajectory, depth=2, time_augmentation=False)
# Returns: list[float] — compact (removes symmetric terms)

d = cvx.signature_distance(sig_a, sig_b)
# Returns: float

d = cvx.frechet_distance(traj_a, traj_b)
# Returns: float — discrete Fréchet (dog-walking) distance
```

### Distributional Distances

```python
d = cvx.fisher_rao_distance(p, q)
# Returns: float — range [0, π]

d = cvx.hellinger_distance(p, q)
# Returns: float — range [0, 1]

d = cvx.wasserstein_drift(dist_a, dist_b, centroids, n_projections=50)
# Returns: float — sliced Wasserstein with region geometry
```

### Anchor Projection

```python
projected = cvx.project_to_anchors(trajectory, anchors, metric="cosine")
# trajectory: list of (timestamp, vector)
# anchors: list of list[float] — K anchor vectors
# Returns: list of (timestamp, distances) where distances has K elements

summary = cvx.anchor_summary(projected_trajectory)
# Returns: dict { "mean": [...], "min": [...], "trend": [...], "last": [...] }
```

### Topology

```python
features = cvx.topological_features(points, n_radii=20, persistence_threshold=0.1)
# Returns: dict { "n_components", "total_persistence", "max_persistence",
#                  "mean_persistence", "persistence_entropy", "betti_curve", "radii" }
```

### Point Process

```python
features = cvx.event_features(timestamps)
# Returns: dict { "n_events", "span", "mean_gap", "std_gap", "burstiness",
#                  "memory", "temporal_entropy", "intensity_trend", "gap_cv",
#                  "max_gap", "circadian_strength" }
```

### Cohort & Causality

```python
report = cvx.cohort_drift(trajectories, t1, t2, top_n=5)
# Returns: dict with n_entities, mean/median/std drift, centroid metrics, outliers

windows = cvx.temporal_join(traj_a, traj_b, epsilon, window_us)
# Returns: list of (start, end, mean_distance, min_distance, points_a, points_b)

result = cvx.granger_causality(traj_a, traj_b, max_lag=5, significance=0.05)
# Returns: dict { direction, optimal_lag, f_statistic, p_value, effect_size }

motifs = cvx.discover_motifs(trajectory, window, max_motifs=5, exclusion_zone=0.5)
# Returns: list of dicts { canonical_index, occurrences, period, mean_match_distance }

discords = cvx.discover_discords(trajectory, window, max_discords=5)
# Returns: list of (index, timestamp, distance)

result = cvx.counterfactual_trajectory(pre_change, post_change, change_point)
# Returns: dict { total_divergence, max_divergence_value, max_divergence_time, divergence_curve }
```

### Prediction

```python
predicted, method = index.predict(entity_id, target_timestamp)
# Returns: (vector, "neural_ode" | "linear")

predicted = cvx.predict(trajectory, target_timestamp)
# Returns: list[float] — linear extrapolation (module-level, no Neural ODE)
```
