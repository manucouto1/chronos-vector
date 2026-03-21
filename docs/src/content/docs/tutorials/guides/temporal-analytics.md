---
title: "Temporal Analytics"
description: "Velocity, drift, changepoints, path signatures, distributional distances, and topology on embedding trajectories"
---

import { Aside } from '@astrojs/starlight/components';

CVX provides 27+ analytical functions that treat embedding trajectories as mathematical objects — differentiable curves, stochastic processes, and topological spaces. This tutorial covers each category with theory, code, and visualizations.

## Setup: Three Behavioral Archetypes

```python
import chronos_vector as cvx
import numpy as np

np.random.seed(42)
D = 32

index = cvx.TemporalIndex(m=16, ef_construction=100)

# Entity 0: Smooth directional drift (trending behavior)
for t in range(100):
    vec = np.sin(np.arange(D) * 0.1 + t * 0.05) + np.random.randn(D) * 0.01
    index.insert(0, t * 86400, vec.astype(np.float32).tolist())

# Entity 1: Regime change at t=50 (structural break)
for t in range(100):
    base = np.ones(D) * (0.5 if t < 50 else -0.5)
    index.insert(1, t * 86400, (base + np.random.randn(D) * 0.02).astype(np.float32).tolist())

# Entity 2: Stationary noise (no structure)
for t in range(100):
    index.insert(2, t * 86400, (np.random.randn(D) * 0.1).astype(np.float32).tolist())

traj0, traj1, traj2 = index.trajectory(0), index.trajectory(1), index.trajectory(2)
```

---

## 1. Vector Differential Calculus

### Drift: Displacement Between States

Drift measures the total displacement between two embedding vectors, decomposed into L2 magnitude, cosine distance, and per-dimension contributions:

$$\text{drift}_{L2} = \|\mathbf{v}(t_2) - \mathbf{v}(t_1)\|_2 \qquad \text{drift}_{\cos} = 1 - \frac{\mathbf{v}(t_1) \cdot \mathbf{v}(t_2)}{\|\mathbf{v}(t_1)\| \cdot \|\mathbf{v}(t_2)\|}$$

```python
l2, cosine, top_dims = cvx.drift(traj1[0][1], traj1[-1][1], top_n=3)
print(f"Regime entity drift: L2={l2:.3f}, cosine={cosine:.4f}")
```

```text
Regime entity drift: L2=5.627, cosine=1.9959
```

<iframe src="/chronos-vector/plots/analytics_drift.html" width="100%" height="450" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

The **regime change** entity (red) shows a sudden jump in cumulative drift at step 50. The **trending** entity (green) drifts steadily. The **noise** entity (blue) drifts randomly with no clear direction.

### Velocity: Instantaneous Rate of Change

$$\mathbf{v}'(t) \approx \frac{\mathbf{v}(t + \Delta t) - \mathbf{v}(t - \Delta t)}{2\Delta t}$$

The magnitude $\|\mathbf{v}'(t)\|$ indicates how fast the entity's semantic representation is shifting. See the [Quick Start velocity plot](/chronos-vector/tutorials/guides/quick-start) for a comparison across entity types.

### Hurst Exponent: Persistence vs Mean-Reversion

The rescaled range statistic:

$$\mathbb{E}\left[\frac{R(n)}{S(n)}\right] \sim c \cdot n^H \quad \text{as } n \to \infty$$

where $R(n)$ is the range of cumulative deviations and $S(n)$ is the standard deviation over window $n$.

| $H$ value | Behavior | Example |
|-----------|----------|---------|
| $H \approx 0.5$ | Random walk | Gaussian noise trajectory |
| $H > 0.5$ | **Persistent** — trends continue | Gradual semantic drift, disease progression |
| $H < 0.5$ | **Anti-persistent** — trends reverse | Mean-reverting oscillation |

```python
for traj, name in [(traj0, "Trending"), (traj1, "Regime"), (traj2, "Noise")]:
    print(f"  {name}: H={cvx.hurst_exponent(traj):.4f}")
```

```text
  Trending: H=0.5592
  Regime: H=0.7224
  Noise: H=0.7227
```

---

## 2. Change Point Detection (PELT)

Minimizes a penalized cost function over all possible segmentations:

$$\min_{\tau_1, \ldots, \tau_m} \sum_{i=1}^{m+1} \left[ \mathcal{C}(\mathbf{y}_{\tau_{i-1}+1:\tau_i}) + \beta \right]$$

- $\mathcal{C}$: segment cost (Gaussian log-likelihood on L2 distances between consecutive vectors)
- $\beta$: penalty per change point (default: BIC = $\frac{D \cdot \ln(n)}{2}$)

```python
cps = cvx.detect_changepoints(1, traj1, min_segment_len=5)
for ts, severity in cps:
    print(f"  Day {ts // 86400}: severity={severity:.3f}")
```

```text
  Day 50: severity=0.963
```

<Aside type="tip" title="BIC penalty for high dimensions">
For $D > 100$, the default BIC penalty may be too sensitive. Use `penalty = 3 * np.log(n)` to reduce false positives.
</Aside>

---

## 3. Path Signatures

From rough path theory (Lyons 1998), the **truncated path signature** is a universal, reparametrization-invariant descriptor of a trajectory's shape.

For a path $\mathbf{X}: [0,T] \to \mathbb{R}^D$, the depth-$k$ signature is:

$$S(\mathbf{X})^{i_1, \ldots, i_k} = \int_0^T \int_0^{t_k} \cdots \int_0^{t_2} dX^{i_1}_{t_1} \cdots dX^{i_k}_{t_k}$$

| Depth | Features | Captures |
|-------|----------|----------|
| 1 | $D$ | Net displacement per dimension |
| 2 | $D + D^2$ | Displacement + signed area (rotation, order) |
| 3 | $D + D^2 + D^3$ | Higher-order interactions |

```python
sig0 = cvx.path_signature(traj0, depth=2)
sig1 = cvx.path_signature(traj1, depth=2)
print(f"Signature dimension: {len(sig0)} (D={D}, depth=2)")

print(f"  trending vs regime: {cvx.signature_distance(sig0, sig1):.3f}")
```

```text
Signature dimension: 1056 (D=32, depth=2)
  trending vs regime: 18.148
```

<iframe src="/chronos-vector/plots/analytics_signatures.html" width="100%" height="500" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

The distance matrix shows that the trending and regime trajectories have the most different shapes — the abrupt regime change creates a very different signature from smooth drift.

### Fréchet Distance

The **dog-walking distance** — the minimum leash length needed to traverse both trajectories simultaneously, preserving order:

$$d_F(\mathbf{X}, \mathbf{Y}) = \inf_{\alpha, \beta} \max_{t \in [0,1]} \|\mathbf{X}(\alpha(t)) - \mathbf{Y}(\beta(t))\|$$

---

## 4. Point Process Analysis

When only **event timing** is available (no embeddings), point process features characterize temporal patterns:

| Feature | Formula | Interpretation |
|---------|---------|---------------|
| **Burstiness** | $B = \frac{\sigma_\tau - \mu_\tau}{\sigma_\tau + \mu_\tau}$ | $B > 0$: bursty, $B = 0$: Poisson, $B < 0$: periodic |
| **Memory** | $M = \text{corr}(\tau_i, \tau_{i+1})$ | Correlation between consecutive inter-event gaps |
| **Entropy** | $H = -\sum p_i \log p_i$ | Uniformity of event distribution over time bins |
| **Gap CV** | $\text{CV} = \sigma_\tau / \mu_\tau$ | Coefficient of variation of inter-event gaps |

<iframe src="/chronos-vector/plots/analytics_events.html" width="100%" height="450" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

Bursty events show high burstiness and gap CV; uniform events show near-zero burstiness and high entropy.

---

## 5. Distributional Distances

For comparing **probability distributions** (e.g., region membership proportions):

### Fisher-Rao Distance

The geodesic distance on the statistical manifold of categorical distributions:

$$d_{FR}(\mathbf{p}, \mathbf{q}) = 2 \arccos\left(\sum_i \sqrt{p_i \cdot q_i}\right)$$

Bounded in $[0, \pi]$. A true metric — symmetric, satisfies triangle inequality (unlike KL divergence).

### Hellinger Distance

$$d_H(\mathbf{p}, \mathbf{q}) = \frac{1}{\sqrt{2}} \sqrt{\sum_i \left(\sqrt{p_i} - \sqrt{q_i}\right)^2}$$

Bounded in $[0, 1]$. Related to Fisher-Rao: $d_H = \sqrt{2} \sin(d_{FR}/2)$.

### Wasserstein (Earth Mover's Distance)

The optimal transport cost between distributions, respecting the **geometry of the underlying space** (region centroids). Unlike Hellinger/Fisher-Rao, Wasserstein accounts for *how far apart* categories are, not just their probabilities.

```python
p, q = [0.5, 0.3, 0.2], [0.2, 0.5, 0.3]
print(f"Fisher-Rao: {cvx.fisher_rao_distance(p, q):.4f}")
print(f"Hellinger:  {cvx.hellinger_distance(p, q):.4f}")
```

```text
Fisher-Rao: 0.4510
Hellinger:  0.2228
```

---

## 6. Persistent Homology

Topological data analysis tracks how the **connectivity structure** of a point cloud changes across spatial scales.

The **Betti number** $\beta_0(r)$ counts the number of connected components at radius $r$. As $r$ increases, components merge. The radii at which they appear and disappear form a **persistence diagram**.

| Metric | Meaning |
|--------|---------|
| **Total persistence** | Sum of all lifetimes — overall topological complexity |
| **Max persistence** | Longest-lived feature — most robust structure |
| **Persistence entropy** | Shannon entropy of lifetimes — uniformity of scales |

```python
vecs = [v for _, v in traj0]
topo = cvx.topological_features(vecs, n_radii=20, persistence_threshold=0.1)
print(f"Total persistence: {topo['total_persistence']:.4f}")
print(f"Max persistence:   {topo['max_persistence']:.4f}")
```

<iframe src="/chronos-vector/plots/analytics_topology.html" width="100%" height="450" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

The Betti curve shows how connected components merge as the radius grows. A sharp drop indicates well-separated clusters.

---

## Summary: Choosing the Right Tool

| Question | Function | Output |
|----------|----------|--------|
| How fast is it changing? | `velocity()` | Vector — direction + magnitude |
| How much has it changed? | `drift()` | Scalar + top dimensions |
| Is it trending or reverting? | `hurst_exponent()` | $H \in [0,1]$ |
| When did behavior change? | `detect_changepoints()` | Timestamps + severity |
| What shape is the trajectory? | `path_signature()` | Fixed-size descriptor |
| Are two trajectories similar? | `signature_distance()`, `frechet_distance()` | Scalar distance |
| How do distributions differ? | `fisher_rao_distance()`, `wasserstein_drift()` | Scalar distance |
| Is the data clustered? | `topological_features()` | Betti curves + persistence |
| What's the temporal pattern? | `event_features()` | Burstiness, memory, entropy |
