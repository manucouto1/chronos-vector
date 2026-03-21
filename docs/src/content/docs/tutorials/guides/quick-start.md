---
title: "Quick Start"
description: "Install CVX, create an index, insert vectors, search, and save — in 5 minutes"
---

## Install

```bash
pip install chronos-vector
```

## Create an Index

```python
import chronos_vector as cvx
import numpy as np

index = cvx.TemporalIndex(m=16, ef_construction=200, ef_search=50)
```

## Insert Vectors

Each point has an **entity_id** (who), a **timestamp** (when), and a **vector** (what):

```python
# Single insert
index.insert(entity_id=1, timestamp=1000, vector=[0.1, 0.2, 0.3, 0.4])

# Bulk insert from numpy (much faster for large datasets)
n = 10_000
entity_ids = np.random.randint(0, 100, size=n, dtype=np.uint64)
timestamps = np.arange(n, dtype=np.int64) * 1000
vectors = np.random.randn(n, 128).astype(np.float32)

index.bulk_insert(entity_ids, timestamps, vectors)
print(f"Index has {len(index)} points")
```

## Search

```python
query = np.random.randn(128).astype(np.float32).tolist()

# Pure semantic search (alpha=1.0)
results = index.search(query, k=5)
for entity_id, timestamp, score in results:
    print(f"  entity={entity_id}, ts={timestamp}, score={score:.4f}")

# Mixed semantic + temporal (alpha=0.5, prefer recent)
results = index.search(query, k=5, alpha=0.5, query_timestamp=9_000_000)

# With temporal range filter
results = index.search(query, k=5, filter_start=5_000_000, filter_end=8_000_000)
```

## Trajectory

Extract all points for an entity, ordered by time:

```python
trajectory = index.trajectory(entity_id=1)
print(f"Entity 1 has {len(trajectory)} points")
for ts, vec in trajectory[:3]:
    print(f"  t={ts}, dim={len(vec)}")
```

## Save & Load

```python
# Save (directory format: index.bin + temporal_edges.bin)
index.save("my_index")

# Load (100× faster than rebuilding)
index = cvx.TemporalIndex.load("my_index")
```

## Basic Analytics

```python
traj = index.trajectory(entity_id=1)

if len(traj) >= 2:
    # Velocity at a specific timestamp
    vel = cvx.velocity(traj, timestamp=traj[len(traj)//2][0])
    print(f"Velocity magnitude: {np.linalg.norm(vel):.4f}")

    # Drift between first and last vector
    l2, cosine, top_dims = cvx.drift(traj[0][1], traj[-1][1], top_n=3)
    print(f"Drift: L2={l2:.4f}, cosine={cosine:.4f}")

    # Change point detection
    cps = cvx.detect_changepoints(entity_id=1, trajectory=traj)
    print(f"Change points: {len(cps)}")

    # Hurst exponent (> 0.5 = trending, < 0.5 = mean-reverting)
    h = cvx.hurst_exponent(traj)
    print(f"Hurst exponent: {h:.3f}")
```

## Semantic Regions

HNSW hierarchy forms natural clusters:

```python
regions = index.regions(level=1)
print(f"Level 1 has {len(regions)} regions")
for hub_id, hub_vec, count in regions[:5]:
    print(f"  Region {hub_id}: {count} members")

# O(N) single-pass assignment
assignments = index.region_assignments(level=1)
for hub_id, members in list(assignments.items())[:3]:
    print(f"  Region {hub_id}: {len(members)} members")
```

## What's Next?

- [Temporal Analytics Tutorial](/chronos-vector/tutorials/guides/temporal-analytics) — velocity, drift, changepoints, signatures
- [Anchor Projection Tutorial](/chronos-vector/tutorials/guides/anchor-projection) — project onto interpretable dimensions
- [Episodic Memory Tutorial](/chronos-vector/tutorials/guides/episodic-memory) — causal search for AI agents
- [Python API Reference](/chronos-vector/specs/python-api) — complete function reference
