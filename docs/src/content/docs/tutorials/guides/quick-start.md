---
title: "Quick Start"
description: "Install CVX, create an index, insert vectors, search, and explore temporal analytics — in 10 minutes"
---

import { Aside } from '@astrojs/starlight/components';

## Install

```bash
pip install chronos-vector
```

## Core Concept: Vectors in Time

Standard vector databases store embeddings as static points. CVX stores them as **trajectories** — ordered sequences of embeddings for the same entity over time.

Each point in CVX has three components:

| Component | Type | Meaning |
|-----------|------|---------|
| `entity_id` | `u64` | **Who** — the entity this vector belongs to (user, document, episode) |
| `timestamp` | `i64` | **When** — unix timestamp (seconds) |
| `vector` | `[f32]` | **What** — the embedding at this moment in time |

This enables questions that static stores cannot answer: *"How is this entity changing?"*, *"When did its behavior shift?"*, *"What trajectory shape does it follow?"*

## Create an Index and Insert Data

```python
import chronos_vector as cvx
import numpy as np

np.random.seed(42)

index = cvx.TemporalIndex(m=16, ef_construction=200, ef_search=50)

# Bulk insert: 1000 points, 10 entities, D=32
n = 1000
entity_ids = np.random.randint(0, 10, size=n, dtype=np.uint64)
timestamps = np.arange(n, dtype=np.int64) * 86400  # daily
vectors = np.random.randn(n, 32).astype(np.float32)

count = index.bulk_insert(entity_ids, timestamps, vectors)
print(f"{count} points inserted, index size: {len(index)}")
```

```text
1000 points inserted, index size: 1000
```

The HNSW index parameters control the speed/accuracy trade-off:

| Parameter | Effect | Typical value |
|-----------|--------|--------------|
| `m` | Connections per node — higher = better recall, more memory | 16 |
| `ef_construction` | Search width during build — higher = better graph, slower build | 100-200 |
| `ef_search` | Search width during query — higher = better recall, slower query | 50-200 |

## Search

CVX search combines **semantic distance** (cosine/L2 between vectors) with **temporal distance** (how far apart in time), controlled by the $\alpha$ parameter:

$$d_{ST} = \alpha \cdot d_{\text{semantic}} + (1 - \alpha) \cdot d_{\text{temporal}}$$

- $\alpha = 1.0$: pure semantic (ignore time)
- $\alpha = 0.5$: balanced — prefer recent AND similar
- $\alpha = 0.0$: pure temporal (ignore content)

```python
query = np.random.randn(32).astype(np.float32).tolist()

results = index.search(query, k=5, alpha=1.0)
for entity_id, timestamp, score in results:
    print(f"  entity={entity_id}, day={timestamp//86400}, score={score:.4f}")
```

```text
  entity=1, day=981, score=27.2262
  entity=1, day=458, score=28.8013
  entity=7, day=104, score=30.5205
  entity=2, day=122, score=30.8634
  entity=5, day=199, score=31.3905
```

## Trajectory Extraction

A trajectory is the ordered sequence of all embeddings for a single entity:

```python
traj = index.trajectory(entity_id=0)
print(f"Entity 0 has {len(traj)} points")
```

```text
Entity 0 has 118 points
```

## Temporal Analytics

### Velocity: $\frac{\partial \mathbf{v}}{\partial t}$

The rate of change of the embedding vector at a specific timestamp, computed via finite differences:

$$\mathbf{v}'(t) \approx \frac{\mathbf{v}(t + \Delta t) - \mathbf{v}(t - \Delta t)}{2\Delta t}$$

A high velocity indicates rapid semantic change — the entity's representation is shifting fast.

<iframe src="/chronos-vector/plots/quickstart_velocity.html"></iframe>

The **smooth drift** entity (green) shows low, consistent velocity. The **regime change** entity (red) shows a sharp velocity spike at day 50 when its embedding space flips. The **noise** entity (blue) shows moderate, erratic velocity.

### Hurst Exponent: Long-Memory Estimation

The Hurst exponent $H$ characterizes the long-term memory of a trajectory:

$$H \in [0, 1]$$

| Value | Meaning | Implication |
|-------|---------|-------------|
| $H > 0.5$ | **Persistent** (trending) | Past direction predicts future direction |
| $H = 0.5$ | **Random walk** | No long-term memory |
| $H < 0.5$ | **Anti-persistent** (mean-reverting) | Past direction predicts *opposite* future |

Computed via rescaled range analysis: $\mathbb{E}[R(n)/S(n)] \propto n^H$.

```python
h = cvx.hurst_exponent(traj)
print(f"Hurst exponent: {h:.4f}")
```

```text
Hurst exponent: 0.7074
```

<iframe src="/chronos-vector/plots/quickstart_hurst.html"></iframe>

### Change Point Detection (PELT)

PELT (Pruned Exact Linear Time, Killick et al. 2012) detects structural breaks — moments where the statistical properties of the trajectory change abruptly.

The algorithm minimizes a penalized cost:

$$\sum_{i=1}^{m+1} \left[ \mathcal{C}(\mathbf{y}_{\tau_{i-1}+1:\tau_i}) + \beta \right]$$

where $\mathcal{C}$ is the segment cost and $\beta$ is the penalty per change point. Higher $\beta$ = fewer, more significant change points.

```python
cps = cvx.detect_changepoints(1, traj1, min_segment_len=5)
print(f"{len(cps)} change point(s) detected")
```

```text
1 change point(s) detected
  day=50, severity=0.963
```

<iframe src="/chronos-vector/plots/quickstart_changepoints.html"></iframe>

PELT correctly identifies the regime change at day 50, where the entity's embedding flips from positive to negative values.

### Drift Measurement

Drift quantifies the displacement between two vectors:

```python
l2, cosine, top_dims = cvx.drift(traj[0][1], traj[-1][1], top_n=3)
print(f"L2 magnitude: {l2:.4f}")
print(f"Cosine drift:  {cosine:.4f}")
```

```text
L2 magnitude: 7.6693
Cosine drift:  0.9442
```

## Semantic Regions

HNSW's multi-level hierarchy forms natural clusters. Nodes at higher levels are **hub nodes** — they act as cluster centroids. Every node at level 0 is assigned to its nearest hub via greedy descent.

```python
regions = index.regions(level=1)
print(f"{len(regions)} regions at level 1")

assignments = index.region_assignments(level=1)
total = sum(len(m) for m in assignments.values())
print(f"{total} points assigned across {len(assignments)} regions")
```

```text
17 regions at level 1
1000 points assigned across 17 regions
```

<iframe src="/chronos-vector/plots/quickstart_regions.html"></iframe>

## Trajectory Visualization

Three synthetic entities show distinct behaviors in embedding space:

<iframe src="/chronos-vector/plots/quickstart_trajectories.html"></iframe>

<Aside type="tip" title="Color = time">
Points are colored by time (dark = early, light = late). The smooth drift entity traces a continuous path. The regime change entity shows two distinct clusters. The noise entity fills a cloud around the origin.
</Aside>

## Save & Load

```python
index.save("my_index")  # Directory: my_index/index.bin + temporal_edges.bin
loaded = cvx.TemporalIndex.load("my_index")
print(f"Loaded {len(loaded)} points")
```

```text
Loaded 1000 points
```

## What's Next?

- [Temporal Analytics](/chronos-vector/tutorials/guides/temporal-analytics) — velocity, drift, changepoints, signatures, topology
- [Anchor Projection & Centering](/chronos-vector/tutorials/guides/anchor-projection) — project onto interpretable dimensions
- [Semantic Regions](/chronos-vector/tutorials/guides/semantic-regions) — explore HNSW hierarchy as clustering
- [Episodic Memory for Agents](/chronos-vector/tutorials/guides/episodic-memory) — causal search, rewards, continuations
- [Python API Reference](/chronos-vector/specs/python-api) — complete function reference
