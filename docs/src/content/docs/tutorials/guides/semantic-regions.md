---
title: "Semantic Regions"
description: "Explore HNSW hierarchy as unsupervised clustering with distributional distances"
---

HNSW's multi-level hierarchy forms natural semantic clusters. CVX exposes these as **regions** — hub nodes at each level with their assigned members.

## Setup

```python
import chronos_vector as cvx
import numpy as np

np.random.seed(42)
D = 16

index = cvx.TemporalIndex(m=4, ef_construction=50, ef_search=50)

# 3 clusters with temporal evolution
for cluster in range(3):
    center = np.random.randn(D).astype(np.float32) * 2
    for entity in range(10):
        for t in range(20):
            drift = np.random.randn(D).astype(np.float32) * 0.1 * t
            vec = center + drift + np.random.randn(D).astype(np.float32) * 0.3
            eid = cluster * 100 + entity
            index.insert(eid, t * 86400, vec.tolist())

print(f"Index: {len(index)} points, 3 clusters × 10 entities × 20 timesteps")
```

## Discover Regions

```python
for level in range(3):
    regions = index.regions(level)
    if regions:
        print(f"Level {level}: {len(regions)} regions")
        for hub_id, hub_vec, count in regions[:5]:
            print(f"  Hub {hub_id}: {count} members")
```

## Region Assignments (O(N) Single Pass)

```python
assignments = index.region_assignments(level=1)
print(f"\nLevel 1: {len(assignments)} regions")

total = sum(len(m) for m in assignments.values())
print(f"Total assigned: {total} (index has {len(index)})")

# Analyze cluster composition per region
for hub_id, members in sorted(assignments.items(), key=lambda x: -len(x[1]))[:5]:
    clusters = {}
    for eid, ts in members:
        c = eid // 100
        clusters[c] = clusters.get(c, 0) + 1
    dominant = max(clusters, key=clusters.get)
    purity = clusters[dominant] / len(members)
    print(f"  Hub {hub_id}: {len(members)} members, "
          f"clusters={dict(clusters)}, purity={purity:.2f}")
```

## Temporal Region Filtering

```python
# Assignments in first half of time only
early = index.region_assignments(level=1, start=0, end=10 * 86400)
late = index.region_assignments(level=1, start=10 * 86400, end=20 * 86400)

early_total = sum(len(m) for m in early.values())
late_total = sum(len(m) for m in late.values())
print(f"\nEarly period: {early_total} points in {len(early)} regions")
print(f"Late period:  {late_total} points in {len(late)} regions")
```

## Region Trajectory (EMA-Smoothed)

Track how an entity's region membership evolves:

```python
entity_id = 5  # entity from cluster 0
region_traj = index.region_trajectory(entity_id, level=1, window_days=5, alpha=0.3)

print(f"\nEntity {entity_id} region trajectory ({len(region_traj)} windows):")
for ts, dist in region_traj[:5]:
    top_region = max(range(len(dist)), key=lambda i: dist[i])
    print(f"  t={ts}: top region idx={top_region}, weight={dist[top_region]:.3f}")
```

## Distributional Distances Between Regions

Compare how region distributions change:

```python
if len(region_traj) >= 2:
    d1 = region_traj[0][1]
    d2 = region_traj[-1][1]

    fr = cvx.fisher_rao_distance(d1, d2)
    h = cvx.hellinger_distance(d1, d2)
    print(f"\nRegion distribution drift:")
    print(f"  Fisher-Rao: {fr:.4f}")
    print(f"  Hellinger:  {h:.4f}")
```

## Wasserstein Drift (Geometry-Aware)

```python
# Get region centroids for geometry-aware comparison
regions = index.regions(level=1)
centroids = [vec for _, vec, _ in regions]

if len(centroids) >= 2 and len(region_traj) >= 2:
    w = cvx.wasserstein_drift(
        region_traj[0][1],
        region_traj[-1][1],
        centroids,
        n_projections=50,
    )
    print(f"  Wasserstein: {w:.4f}")
```
