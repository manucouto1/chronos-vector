---
title: "Semantic Regions"
description: "Explore HNSW hierarchy as unsupervised clustering with distributional distances"
---

import { Aside } from '@astrojs/starlight/components';

## HNSW as Hierarchical Clustering

HNSW builds a multi-level graph where each level has fewer, more "central" nodes. These **hubs** emerge naturally and act as unsupervised cluster centroids. CVX assigns every node to its nearest hub via greedy descent — $\sim N/M^L$ regions at level $L$.

<Aside type="tip" title="No training required">
Unlike k-means or DBSCAN, HNSW regions emerge from index construction. No hyperparameter tuning, no iterative optimization. O(N) single-pass assignment.
</Aside>

## Setup

```python
import chronos_vector as cvx
import numpy as np
np.random.seed(42)

index = cvx.TemporalIndex(m=4, ef_construction=50, ef_search=50)
for cluster in range(3):
    center = np.random.randn(16).astype(np.float32) * 2
    for entity in range(10):
        for t in range(20):
            drift = np.random.randn(16).astype(np.float32) * 0.1 * t
            vec = center + drift + np.random.randn(16).astype(np.float32) * 0.3
            index.insert(cluster * 100 + entity, t * 86400, vec.tolist())
print(f"Index: {len(index)} points")
```

```text
Index: 600 points
```

## Region Discovery and Purity

```python
assignments = index.region_assignments(level=1)
for hub_id, members in sorted(assignments.items(), key=lambda x: -len(x[1]))[:3]:
    clusters = {}
    for eid, ts in members:
        clusters[eid // 100] = clusters.get(eid // 100, 0) + 1
    dominant = max(clusters.values())
    print(f"  Hub {hub_id}: {len(members)} members, purity={dominant/len(members):.0%}")
```

```text
  Hub 38: 32 members, purity=100%
  Hub 161: 28 members, purity=100%
  Hub 301: 26 members, purity=100%
```

<iframe src="/chronos-vector/plots/regions_purity.html"></iframe>

<iframe src="/chronos-vector/plots/regions_sizes.html"></iframe>

## Region Trajectory (EMA-Smoothed)

Track membership evolution with exponential smoothing: $\mathbf{s}(t) = \alpha \cdot \mathbf{c}(t) + (1-\alpha) \cdot \mathbf{s}(t-1)$

<iframe src="/chronos-vector/plots/regions_trajectory.html"></iframe>

## Distributional Distances

Compare region distributions with geometry-aware metrics:

| Metric | Formula | Range |
|--------|---------|-------|
| Fisher-Rao | $2\arccos(\sum_i \sqrt{p_i q_i})$ | $[0, \pi]$ |
| Hellinger | $\frac{1}{\sqrt{2}}\sqrt{\sum(\sqrt{p_i}-\sqrt{q_i})^2}$ | $[0, 1]$ |
| Wasserstein | Optimal transport with region centroids | $[0, \infty)$ |
