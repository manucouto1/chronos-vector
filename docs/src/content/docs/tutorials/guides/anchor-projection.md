---
title: "Anchor Projection & Centering"
description: "Project trajectories onto interpretable dimensions and correct embedding anisotropy"
---

Anchor projection maps high-dimensional trajectories onto clinically or domain-meaningful reference vectors. Centering corrects the anisotropy that makes raw cosine distances useless.

## Setup: Simulated Anisotropic Embeddings

```python
import chronos_vector as cvx
import numpy as np

np.random.seed(42)
D = 64

# Simulate anisotropic embeddings: shared dominant direction + small signal
dominant = np.random.randn(D).astype(np.float32)
dominant = dominant / np.linalg.norm(dominant) * 5.0  # strong shared component

index = cvx.TemporalIndex(m=16, ef_construction=100)

# Group A: dominant + signal toward anchor 0
for t in range(50):
    signal = np.zeros(D, dtype=np.float32)
    signal[0:5] = 0.3 + t * 0.005  # increasing proximity to anchor 0
    vec = dominant + signal + np.random.randn(D).astype(np.float32) * 0.05
    index.insert(1, t * 86400, vec.tolist())

# Group B: dominant + signal toward anchor 1
for t in range(50):
    signal = np.zeros(D, dtype=np.float32)
    signal[10:15] = 0.3
    vec = dominant + signal + np.random.randn(D).astype(np.float32) * 0.05
    index.insert(2, t * 86400, vec.tolist())
```

## The Anisotropy Problem

Without centering, all vectors are dominated by the shared component:

```python
traj_a = index.trajectory(1)
traj_b = index.trajectory(2)

# Raw cosine similarity — compressed in narrow range
from numpy.linalg import norm
v1, v2 = np.array(traj_a[0][1]), np.array(traj_b[0][1])
raw_sim = np.dot(v1, v2) / (norm(v1) * norm(v2))
print(f"Raw cosine similarity: {raw_sim:.4f}")
# Likely > 0.95 — the dominant direction masks the difference
```

## Centering: 30× Signal Improvement

```python
# Compute and set centroid
centroid = index.compute_centroid()
index.set_centroid(centroid)
centroid_np = np.array(centroid)

# Center vectors manually for comparison
v1c = v1 - centroid_np
v2c = v2 - centroid_np
centered_sim = np.dot(v1c, v2c) / (norm(v1c) * norm(v2c))
print(f"Centered cosine similarity: {centered_sim:.4f}")
# Much lower — the discriminative signal is revealed
print(f"Gap amplification: {(1-centered_sim)/(1-raw_sim):.1f}×")
```

## Define Anchors

Anchors are reference vectors representing interpretable dimensions:

```python
# 3 synthetic anchors (in real use: clinical phrases embedded by your model)
anchors = []
for i in range(3):
    a = np.zeros(D, dtype=np.float32)
    a[i*5:(i+1)*5] = 1.0  # each anchor activates different dimensions
    anchors.append(a.tolist())
```

## Project Trajectories

```python
proj_a = cvx.project_to_anchors(traj_a, anchors, metric="cosine")
proj_b = cvx.project_to_anchors(traj_b, anchors, metric="cosine")

print("Entity 1 anchor distances (first 3 time steps):")
for ts, dists in proj_a[:3]:
    print(f"  t={ts}: {[f'{d:.3f}' for d in dists]}")

print("\nEntity 2 anchor distances (first 3 time steps):")
for ts, dists in proj_b[:3]:
    print(f"  t={ts}: {[f'{d:.3f}' for d in dists]}")
```

## Centered Projection (Recommended)

For centered projection, subtract the centroid from both vectors and anchors:

```python
def project_centered(traj, anchors, centroid):
    """Project with centering for anisotropy correction."""
    c = np.array(centroid, dtype=np.float32)
    anchor_matrix = np.array(anchors) - c
    anchor_norms = np.linalg.norm(anchor_matrix, axis=1, keepdims=True)
    anchor_matrix = anchor_matrix / (anchor_norms + 1e-8)

    results = []
    for ts, vec in traj:
        v = np.array(vec, dtype=np.float32) - c
        v = v / (np.linalg.norm(v) + 1e-8)
        sims = v @ anchor_matrix.T
        dists = (1.0 - sims).tolist()
        results.append((ts, dists))
    return results

proj_a_centered = project_centered(traj_a, anchors, centroid)
proj_b_centered = project_centered(traj_b, anchors, centroid)

print("Centered Entity 1 (first 3):")
for ts, dists in proj_a_centered[:3]:
    print(f"  t={ts}: {[f'{d:.3f}' for d in dists]}")
```

## Anchor Summary

Aggregate statistics per anchor:

```python
summary_a = cvx.anchor_summary(proj_a)
print("Entity 1 anchor summary:")
for key in ["mean", "min", "trend", "last"]:
    print(f"  {key}: {[f'{v:.3f}' for v in summary_a[key]]}")

# trend < 0 means approaching the anchor over time
```

## Persistence

The centroid is saved with the index:

```python
index.save("centered_index")
loaded = cvx.TemporalIndex.load("centered_index")
assert loaded.centroid() is not None
print(f"Centroid preserved: {len(loaded.centroid())} dimensions")
```
