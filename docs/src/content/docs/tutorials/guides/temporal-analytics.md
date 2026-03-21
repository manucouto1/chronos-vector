---
title: "Temporal Analytics"
description: "Velocity, drift, changepoints, signatures, and distributional distances on embedding trajectories"
---

This tutorial covers CVX's 27+ analytical functions using synthetic data. No external datasets needed.

## Setup: Synthetic Trajectories

```python
import chronos_vector as cvx
import numpy as np

np.random.seed(42)
D = 32  # embedding dimension

index = cvx.TemporalIndex(m=16, ef_construction=100)

# Entity 1: smooth drift (trending)
for t in range(100):
    vec = np.sin(np.arange(D) * 0.1 + t * 0.05) + np.random.randn(D) * 0.01
    index.insert(1, t * 86400, vec.astype(np.float32).tolist())  # daily timestamps

# Entity 2: regime change at t=50
for t in range(100):
    if t < 50:
        vec = np.ones(D) * 0.5 + np.random.randn(D) * 0.02
    else:
        vec = -np.ones(D) * 0.5 + np.random.randn(D) * 0.02
    index.insert(2, t * 86400, vec.astype(np.float32).tolist())

# Entity 3: stationary noise
for t in range(100):
    vec = np.random.randn(D).astype(np.float32) * 0.1
    index.insert(3, t * 86400, vec.tolist())

traj1 = index.trajectory(1)
traj2 = index.trajectory(2)
traj3 = index.trajectory(3)
```

## Vector Calculus

### Velocity

Rate of change at a specific timestamp:

```python
mid_ts = traj1[50][0]
vel = cvx.velocity(traj1, mid_ts)
print(f"Entity 1 velocity magnitude: {np.linalg.norm(vel):.4f}")
# Should be nonzero — entity 1 is drifting

vel3 = cvx.velocity(traj3, traj3[50][0])
print(f"Entity 3 velocity magnitude: {np.linalg.norm(vel3):.4f}")
# Should be near zero — entity 3 is stationary
```

### Drift

Component-wise displacement between two vectors:

```python
l2, cosine, top_dims = cvx.drift(traj1[0][1], traj1[-1][1], top_n=5)
print(f"Entity 1 total drift: L2={l2:.4f}, cosine={cosine:.4f}")
print(f"Top changing dimensions: {top_dims}")

l2_2, cosine_2, _ = cvx.drift(traj2[0][1], traj2[-1][1], top_n=5)
print(f"Entity 2 drift (regime change): L2={l2_2:.4f}, cosine={cosine_2:.4f}")
# Entity 2 should show much larger drift
```

### Hurst Exponent

Long-memory estimation:

```python
h1 = cvx.hurst_exponent(traj1)
h2 = cvx.hurst_exponent(traj2)
h3 = cvx.hurst_exponent(traj3)
print(f"Entity 1 (trending): H={h1:.3f}")   # > 0.5
print(f"Entity 2 (regime):   H={h2:.3f}")   # depends on regime
print(f"Entity 3 (noise):    H={h3:.3f}")   # ≈ 0.5
```

## Change Point Detection

PELT finds structural breaks in trajectories:

```python
cps2 = cvx.detect_changepoints(2, traj2, penalty=None, min_segment_len=5)
print(f"Entity 2 changepoints: {len(cps2)}")
for ts, severity in cps2:
    day = ts // 86400
    print(f"  Day {day}: severity={severity:.3f}")
# Should detect the regime change around day 50

cps1 = cvx.detect_changepoints(1, traj1)
print(f"Entity 1 changepoints: {len(cps1)}")
# Smooth drift — fewer/no changepoints expected
```

## Path Signatures

Order-aware trajectory descriptors (invariant to reparametrization):

```python
sig1 = cvx.path_signature(traj1, depth=2)
sig2 = cvx.path_signature(traj2, depth=2)
sig3 = cvx.path_signature(traj3, depth=2)

print(f"Signature dimension: {len(sig1)} (depth=2, D={D}: {D} + {D*D} = {D + D*D})")

# Compare trajectory shapes
d12 = cvx.signature_distance(sig1, sig2)
d13 = cvx.signature_distance(sig1, sig3)
d23 = cvx.signature_distance(sig2, sig3)
print(f"sig_dist(trending, regime): {d12:.4f}")
print(f"sig_dist(trending, noise):  {d13:.4f}")
print(f"sig_dist(regime, noise):    {d23:.4f}")
```

### Fréchet Distance

Geometric trajectory comparison (order-preserving):

```python
d = cvx.frechet_distance(traj1, traj2)
print(f"Fréchet distance (trending vs regime): {d:.4f}")
```

## Distributional Distances

Compare probability distributions (e.g., region membership):

```python
p = np.array([0.5, 0.3, 0.2])  # region distribution at t1
q = np.array([0.2, 0.5, 0.3])  # region distribution at t2

fr = cvx.fisher_rao_distance(p.tolist(), q.tolist())
h = cvx.hellinger_distance(p.tolist(), q.tolist())
print(f"Fisher-Rao: {fr:.4f} (range [0, π])")
print(f"Hellinger:  {h:.4f} (range [0, 1])")
```

## Point Process Analysis

Extract features from event timing alone:

```python
timestamps = [traj1[i][0] for i in range(len(traj1))]
features = cvx.event_features(timestamps)
print(f"Burstiness: {features['burstiness']:.3f}")
print(f"Memory:     {features['memory']:.3f}")
print(f"Entropy:    {features['temporal_entropy']:.3f}")
```

## Temporal Features (Fixed-Size Vector)

Compact feature vector combining multiple statistics:

```python
features = cvx.temporal_features(traj1)
print(f"Feature vector size: {len(features)}")
# Can be used directly as input to sklearn classifiers
```

## Topology

Persistent homology tracks structural properties:

```python
vecs = [v for _, v in traj1]
topo = cvx.topological_features(vecs, n_radii=20, persistence_threshold=0.1)
print(f"Components: {topo['n_components']}")
print(f"Total persistence: {topo['total_persistence']:.4f}")
```

## Cohort Analysis

Compare multiple entities:

```python
all_trajs = {1: traj1, 2: traj2, 3: traj3}
t1_idx, t2_idx = 20, 80
vecs_t1 = [[v for ts, v in t if ts == t1_idx * 86400][0] for t in all_trajs.values() if any(ts == t1_idx * 86400 for ts, v in t)]
vecs_t2 = [[v for ts, v in t if ts == t2_idx * 86400][0] for t in all_trajs.values() if any(ts == t2_idx * 86400 for ts, v in t)]

if vecs_t1 and vecs_t2:
    report = cvx.cohort_drift(vecs_t1, vecs_t2, top_n=3)
    print(f"Cohort mean drift: {report['mean_drift_l2']:.4f}")
    print(f"Convergence score: {report['convergence_score']:.4f}")
```
