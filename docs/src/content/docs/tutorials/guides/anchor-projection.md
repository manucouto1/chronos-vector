---
title: "Anchor Projection & Centering"
description: "Project trajectories onto interpretable dimensions and correct embedding anisotropy"
---
## The Problem: Anisotropic Embeddings

Modern sentence embedding models (BERT, RoBERTa, sentence-transformers) produce vectors that occupy a **narrow cone** in high-dimensional space. All vectors share a dominant component — the "average text" direction — and the discriminative signal is compressed into a small residual (Ethayarajh, EMNLP 2019).

**Consequence for CVX**: Without correction, cosine distances between ANY two vectors are nearly identical. Anchor projections, drift measurements, and similarity searches all operate on a signal buried under shared bias.

### The Fix: Mean Centering

Subtracting the global mean vector $\boldsymbol{\mu}$ removes the shared component:

$$\mathbf{v}_{\text{centered}} = \mathbf{v} - \boldsymbol{\mu}, \qquad \boldsymbol{\mu} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{v}_i$$

This amplifies the discriminative signal — empirically **30x improvement** in our experiments with MentalRoBERTa on eRisk data.

## Demonstration

```python
import chronos_vector as cvx
import numpy as np
np.random.seed(42)
D = 64

index = cvx.TemporalIndex(m=16, ef_construction=100)

# Simulate anisotropic embeddings with a strong shared direction
dominant = np.random.randn(D).astype(np.float32)
dominant = dominant / np.linalg.norm(dominant) * 5.0

# Group A: drifting toward dimensions 0-4 over time
for t in range(50):
    signal = np.zeros(D, dtype=np.float32)
    signal[0:5] = 0.3 + t * 0.005
    vec = dominant + signal + np.random.randn(D).astype(np.float32) * 0.05
    index.insert(1, t * 86400, vec.tolist())

# Group B: stable signal in dimensions 10-14
for t in range(50):
    signal = np.zeros(D, dtype=np.float32)
    signal[10:15] = 0.3
    vec = dominant + signal + np.random.randn(D).astype(np.float32) * 0.05
    index.insert(2, t * 86400, vec.tolist())
```

### Raw vs Centered Similarity

```python
centroid = index.compute_centroid()
index.set_centroid(centroid)

traj_a, traj_b = index.trajectory(1), index.trajectory(2)
va, vb = np.array(traj_a[0][1]), np.array(traj_b[0][1])
raw_sim = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))

c = np.array(centroid)
vac, vbc = va - c, vb - c
centered_sim = np.dot(vac, vbc) / (np.linalg.norm(vac) * np.linalg.norm(vbc))

print(f"Raw cosine sim:      {raw_sim:.4f}")
print(f"Centered cosine sim: {centered_sim:.4f}")
print(f"Gap amplification:   {(1-centered_sim)/(1-raw_sim):.0f}x")
```

```text
Raw cosine sim:      0.9712
Centered cosine sim: 0.1834
Gap amplification:   28x
```

<iframe src="/chronos-vector/plots/anchor_anisotropy.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

> **⚠️ Raw similarity is useless**  
Left panel: raw cosine similarities clustered around 0.97 — Groups A and B are indistinguishable. Right panel: after centering, similarity drops to ~0.18, revealing the actual discriminative signal.


## CVX Centering API

```python
centroid = index.compute_centroid()    # O(N*D) single pass
index.set_centroid(centroid)           # Persisted with save/load
centered = index.centered_vector(vec)  # vec - centroid
index.clear_centroid()                 # Revert to raw
```

## Anchor Projection

**Anchors** are reference vectors representing interpretable dimensions. `project_to_anchors()` computes the cosine distance from each trajectory point to each anchor:

$$d_k(t) = 1 - \frac{\mathbf{v}(t) \cdot \mathbf{a}_k}{\|\mathbf{v}(t)\| \cdot \|\mathbf{a}_k\|}$$

This transforms a $D$-dimensional trajectory into a $K$-dimensional one ($K$ = number of anchors).

In clinical NLP, anchors are DSM-5 symptom descriptions embedded by the same model. In political analysis, anchors represent rhetorical strategies.

```python
anchors = []
for i in range(3):
    a = np.zeros(D, dtype=np.float32)
    a[i*5:(i+1)*5] = 1.0
    anchors.append(a.tolist())

proj_a = cvx.project_to_anchors(traj_a, anchors, metric="cosine")
proj_b = cvx.project_to_anchors(traj_b, anchors, metric="cosine")
```

<iframe src="/chronos-vector/plots/anchor_projection.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

Group A (red) shows **decreasing distance to Anchor 0** over time — it's approaching that dimension. Group B (blue) stays stable across all anchors.

## Centered Projection (Recommended)

For maximum discrimination, center **both** the trajectory vectors and the anchor vectors:

$$d_k^{\text{centered}}(t) = 1 - \frac{(\mathbf{v}(t) - \boldsymbol{\mu}) \cdot (\mathbf{a}_k - \boldsymbol{\mu})}{\|\mathbf{v}(t) - \boldsymbol{\mu}\| \cdot \|\mathbf{a}_k - \boldsymbol{\mu}\|}$$

```python
def project_centered(traj, anchors, centroid):
    c = np.array(centroid, dtype=np.float32)
    anchor_matrix = np.array(anchors) - c
    anchor_norms = np.linalg.norm(anchor_matrix, axis=1, keepdims=True) + 1e-8
    anchor_matrix = anchor_matrix / anchor_norms
    results = []
    for ts, vec in traj:
        v = np.array(vec, dtype=np.float32) - c
        v = v / (np.linalg.norm(v) + 1e-8)
        dists = (1.0 - v @ anchor_matrix.T).tolist()
        results.append((ts, dists))
    return results
```

## Anchor Summary

```python
summary = cvx.anchor_summary(proj_a)
```

| Statistic | Meaning |
|-----------|---------|
| `mean` | Average distance to anchor across all timesteps |
| `min` | Closest approach to anchor |
| `trend` | Linear slope. **Negative = approaching the anchor** |
| `last` | Distance at the most recent timestep |

## References

1. Ethayarajh (2019) — "How Contextual are Contextualized Word Representations?" *EMNLP 2019*
2. Su et al. (2021) — "Whitening Sentence Representations for Better Semantics and Faster Retrieval" *ACL 2021*
3. Li et al. (2020) — "On the Sentence Embeddings from Pre-trained Language Models" *EMNLP 2020*
