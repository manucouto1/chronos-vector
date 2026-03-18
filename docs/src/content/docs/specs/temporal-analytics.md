---
title: "Temporal Analytics Toolkit"
description: "Complete reference of CVX analytical capabilities: mathematical foundations, Python API, and cross-domain applications"
---

ChronosVector provides **19 analytical functions** for extracting insights from temporal vector data. Each function is grounded in a specific mathematical framework and applicable across multiple domains.

## Overview

| Category | Functions | Mathematical Basis |
|----------|----------|-------------------|
| [Differential Calculus](#differential-calculus) | `velocity`, `drift`, `temporal_features` | Finite differences, feature engineering |
| [Stochastic Characterization](#stochastic-characterization) | `hurst_exponent`, `detect_changepoints` | R/S analysis, PELT algorithm |
| [Path Signatures](#path-signatures) | `path_signature`, `log_signature`, `signature_distance` | Rough path theory (Lyons, 1998) |
| [Trajectory Comparison](#trajectory-comparison) | `frechet_distance` | Computational geometry |
| [Distributional Distances](#distributional-distances) | `wasserstein_drift`, `fisher_rao_distance`, `hellinger_distance` | Optimal transport, information geometry |
| [Point Process Analysis](#point-process-analysis) | `event_features` | Temporal point processes |
| [Topological Analysis](#topological-analysis) | `topological_features` | Persistent homology (TDA) |
| [Anchor Projection](#anchor-projection) | `project_to_anchors`, `anchor_summary` | Coordinate system change to reference frame |
| [Prediction](#prediction) | `predict` | Linear extrapolation / Neural ODE |

---

## Differential Calculus

### `cvx.velocity(trajectory, timestamp)`

**Theory:** Instantaneous rate of change $\frac{dv}{dt}$ via central finite differences. Measures *how fast* an entity is moving through the embedding space.

**Output:** Velocity vector (same dimensionality as input).

**Applications:**
- **Mental health:** Rate of linguistic change — accelerating drift may signal crisis
- **Finance:** Speed of portfolio evolution — regime transitions show velocity spikes
- **Molecular dynamics:** Conformational transition speed between states
- **MAP-Elites:** Rate of archive exploration — stagnation detection

### `cvx.drift(v1, v2, top_n=5)`

**Theory:** Total displacement between two vectors, decomposed by dimension. Identifies *what* changed and *how much*.

**Output:** `(l2_magnitude, cosine_drift, top_dimensions)` — the overall magnitude, angular change, and the dimensions contributing most.

### `cvx.temporal_features(trajectory)`

**Theory:** Fixed-size summary vector combining mean, std, velocity statistics, Hurst exponent, and change point count. Designed for downstream ML classification.

**Output:** Vector of size $2D + 5$ where $D$ = input dimensionality.

---

## Stochastic Characterization

### `cvx.hurst_exponent(trajectory)`

**Theory:** The Hurst exponent $H \in (0, 1)$ characterizes the long-range dependence of a time series via R/S (rescaled range) analysis:

- $H = 0.5$: random walk (no memory)
- $H > 0.5$: persistent (trending) — past increases predict future increases
- $H < 0.5$: anti-persistent (mean-reverting) — past increases predict future decreases

**Applications:**
- **Mental health:** Depression users show anti-persistent topical dynamics ($H < 0.5$, $d = -0.41$ on CLPsych) — erratic topic switching
- **Finance:** $H > 0.5$ indicates trending markets; $H < 0.5$ indicates mean-reversion opportunities
- **Evolutionary optimization:** $H \to 0.5$ indicates convergence (search becoming random)

### `cvx.detect_changepoints(entity_id, trajectory, penalty, min_segment_len)`

**Theory:** PELT (Pruned Exact Linear Time) algorithm for offline change point detection. Minimizes:

$$\sum_{i=1}^{K+1} C(y_{t_{i-1}:t_i}) + K \cdot \beta$$

where $C$ is the segment cost and $\beta$ is the penalty per change point.

**Output:** List of `(timestamp, severity)` pairs marking regime transitions.

**Applications:**
- **Molecular dynamics:** Conformational state transitions
- **MLOps:** Model drift onset detection
- **Social media:** Behavioral regime changes in users

---

## Path Signatures

*Based on rough path theory (Lyons, 1998). See [Path Signatures](/research/path-signatures) for full mathematical treatment.*

### `cvx.path_signature(trajectory, depth=2, time_augmentation=False)`

**Theory:** The truncated path signature is the *universal nonlinear feature* of sequential data. Any continuous function of a path can be approximated by a linear function of its signature. It captures ordered, multi-scale structure:

- **Depth 1** ($K$ features): Net displacement — *where* did the entity move
- **Depth 2** ($K + K^2$ features): Signed areas — *how* it moved (rotation, correlation, volatility). Distinguishes "right-then-up" from "up-then-right"
- **Depth 3** ($K + K^2 + K^3$ features): Higher-order interactions

**Key property — Chen's Identity:** $S(\alpha \ast \beta) = S(\alpha) \otimes S(\beta)$. When a new point is appended, the signature updates in $O(K^2)$ instead of full recomputation in $O(N \cdot K^2)$. This makes path signatures **native to incremental-insert databases**.

**Practical note:** Applied on **region trajectories** ($K \sim 80$ at HNSW level 3), not raw embeddings ($D = 768$), keeping the computation tractable.

### `cvx.log_signature(trajectory, depth=2, time_augmentation=False)`

**Theory:** The log-signature removes redundant symmetric components via the Baker-Campbell-Hausdorff formula. Contains the same information in fewer dimensions: $K + K(K-1)/2$ instead of $K + K^2$ at depth 2.

### `cvx.signature_distance(sig_a, sig_b)`

**Theory:** $L_2$ distance between path signatures. A fast trajectory similarity measure: $O(K^2)$ per comparison capturing all order-dependent temporal dynamics.

**Applications (all domains):**
- **Molecular dynamics:** Find simulations with similar conformational dynamics
- **Drug discovery:** Identify optimization campaigns that followed similar paths through chemical space
- **MAP-Elites:** Find solutions with similar evolutionary trajectories
- **Mental health:** Detect users with similar temporal patterns to known at-risk cases

---

## Trajectory Comparison

### `cvx.frechet_distance(traj_a, traj_b)`

**Theory:** The discrete Fréchet distance measures the maximum minimum distance between corresponding points when both paths are traversed monotonically. Informally: *the shortest leash needed if you walk along path A while your dog walks along path B, both moving only forward.*

$$d_F(A, B) = \inf_{\alpha, \beta} \max_t \| A(\alpha(t)) - B(\beta(t)) \|$$

**Complexity:** $O(n \times m)$ via dynamic programming.

**When to use vs. signature distance:**
- **Signature distance** ($O(K^2)$): universal features, very fast — **recommended default**
- **Fréchet distance** ($O(nm)$): exact geometric comparison — when the precise geometric shape of the path matters

---

## Distributional Distances

These operate on **region distributions** — histograms over HNSW graph regions that describe an entity's topical composition at a given time.

### `cvx.wasserstein_drift(dist_a, dist_b, centroids, n_projections=50)`

**Theory:** The Wasserstein (optimal transport) distance measures the minimum cost of transforming one distribution into another, *respecting the geometry of the space*. Unlike $L_2$ between histograms, Wasserstein accounts for which regions are *close* vs *far*:

$$W_1(p, q) = \inf_{\gamma \in \Pi(p,q)} \int \|x - y\| \, d\gamma(x, y)$$

Implemented as Sliced Wasserstein (random 1D projections) for computational efficiency.

**Key insight:** Shifting mass between *neighboring* regions costs less than between *distant* ones. Verified: distant shift (1.2) > adjacent shift (0.7) in tests.

**Applications:**
- **MLOps:** Detect concept drift that respects feature space geometry
- **Drug discovery:** Measure how a campaign's chemical space focus shifted
- **Mental health:** Track topical migration between related vs. unrelated topics

### `cvx.fisher_rao_distance(p, q)`

**Theory:** The Fisher-Rao metric is the *unique* Riemannian metric on the statistical manifold that is invariant under sufficient statistics (Chentsov's theorem, 1982). For categorical distributions, it has a closed form via the Bhattacharyya angle:

$$d_{FR}(p, q) = 2 \arccos\left(\sum_i \sqrt{p_i \cdot q_i}\right)$$

**Properties:**
- Symmetric, metric (satisfies triangle inequality)
- Bounded: $[0, \pi]$ ($0$ = identical, $\pi$ = disjoint support)
- Invariant under reparametrization of the probability space

**When to use vs. Wasserstein:** Fisher-Rao is the *mathematically natural* distance between distributions. Wasserstein additionally incorporates the *geometry of the support space* (region centroids). Use Fisher-Rao for pure distributional comparison; Wasserstein when the spatial structure of regions matters.

### `cvx.hellinger_distance(p, q)`

**Theory:** Related to Fisher-Rao, bounded in $[0, 1]$: $H(p, q) = \sqrt{\frac{1 - BC(p,q)}{2}}$ where $BC$ is the Bhattacharyya coefficient.

---

## Point Process Analysis

### `cvx.event_features(timestamps)`

**Theory:** Extracts features from the *timing* of events, independent of vector content. The inter-event intervals themselves encode behavioral patterns modeled by temporal point process theory.

**Output:** Dictionary with 11 features:

| Feature | Formula | Interpretation |
|---------|---------|---------------|
| `burstiness` | $B = \frac{\sigma - \mu}{\sigma + \mu}$ | $-1$ = perfectly regular, $0$ = Poisson (random), $+1$ = maximally bursty |
| `memory` | Autocorrelation at lag 1 | $> 0$: short gaps follow short gaps (clustering). $< 0$: alternating |
| `temporal_entropy` | $H = -\sum p_i \log p_i$ | Higher = more unpredictable timing |
| `intensity_trend` | Slope of event rate | Positive = accelerating, negative = decelerating |
| `gap_cv` | $\sigma_{gap} / \mu_{gap}$ | Coefficient of variation of intervals |
| `circadian_strength` | 24h Fourier amplitude | $0$ = no daily rhythm, $1$ = strong rhythm |
| `max_gap` | Longest silence | Withdrawal period duration |

**Applications:**
- **Mental health:** Night posting ratio ($d = 0.534$, $p < 0.001$ on eRisk) and burstiness are behavioral biomarkers of depression. Circadian disruption is a known clinical marker.
- **Cybersecurity:** Bursty network activity patterns indicate automated attacks or data exfiltration
- **Evolutionary computation:** Accelerating event rate indicates convergence; stagnation shows flattening
- **Finance:** Trading bursts precede volatility spikes; memory coefficient detects algorithmic trading patterns

**References:** Goh & Barabási (2008) introduced burstiness/memory. Hawkes (1971) formalized self-exciting processes.

---

## Topological Analysis

### `cvx.topological_features(points, n_radii=20, persistence_threshold=0.1)`

**Theory:** Persistent homology from Topological Data Analysis (TDA) tracks how the *topology* of a point cloud changes as a filtration radius grows. For dimension 0 (connected components):

- Build a Vietoris-Rips complex: connect points within radius $r$
- Track when components appear (birth) and merge (death)
- **Betti number** $\beta_0(r)$ = number of connected components at radius $r$

Implemented via Union-Find on the pairwise distance graph (single-linkage equivalent). Applied on **region centroids** ($K \sim 80$) for tractability.

**Output:** Dictionary with:
- `n_components`: significant clusters (persistence $>$ threshold)
- `max_persistence`: most prominent topological gap
- `persistence_entropy`: $-\sum \frac{p_i}{P} \log \frac{p_i}{P}$ — uniformity of feature lifetimes
- `betti_curve`: $\beta_0(r)$ sampled at `n_radii` points

**What topology reveals that geometry doesn't:**
- Increasing $\beta_0$ over time → **fragmentation** (topic space splitting)
- Decreasing $\beta_0$ → **convergence** (topics merging)
- High persistence entropy → **uniform cluster structure** (no dominant gap)
- Low persistence entropy → **clear cluster separation** (one dominant gap)

**Applications:**
- **Drug discovery:** Track how the active chemical space fragments or converges during optimization campaigns
- **Molecular dynamics:** Detect when the conformational landscape develops new basins (new $\beta_0$)
- **MAP-Elites:** Monitor archive topology — is the solution space connected or fragmented?
- **MLOps:** Detect structural changes in embedding space topology (not just distributional shifts)

**References:** Edelsbrunner & Harer (2010). Zigzag persistence for temporal networks (EPJ Data Science, 2023).

---

## Anchor Projection

*See [RFC-006: Anchor Projection](/rfc/rfc-006) for design rationale and clinical validation.*

### `cvx.project_to_anchors(trajectory, anchors, metric='cosine')`

**Theory:** Projects a trajectory from absolute embedding space $\mathbb{R}^D$ into an anchor-relative coordinate system $\mathbb{R}^K$, where $K$ is the number of anchors. Each output dimension $k$ is the distance (cosine or L2) from the trajectory point to anchor $k$:

$$\text{projected}_t[k] = d(\mathbf{x}_t, \mathbf{a}_k), \quad k = 1, \ldots, K$$

This is a **coordinate system change**, not a new analytics paradigm. The output is itself a trajectory, so all existing CVX functions (`velocity`, `hurst_exponent`, `detect_changepoints`, `path_signature`, etc.) compose with it natively.

**Output:** $(T, K)$ array — a trajectory in $\mathbb{R}^K$ where each dimension represents distance to the corresponding anchor.

**Applications:**
- **Mental health:** Measure drift toward/away from clinical poles (depression, anxiety, neutral). Anchor-relative features improved F1 from 0.600 to 0.781 on eRisk
- **Finance:** Track portfolio proximity to sector archetypes (tech-heavy, defensive, balanced)
- **Drug discovery:** Monitor compound evolution relative to known active/toxic/selective reference molecules
- **MLOps:** Measure model embedding drift relative to known-good and known-bad reference distributions

### `cvx.anchor_summary(projected)`

**Theory:** Aggregates per-anchor distance dynamics into a fixed-size summary. For each anchor $k$, computes statistics over the projected trajectory's $k$-th dimension.

**Output:** Dictionary per anchor with:

| Statistic | Formula | Interpretation |
|-----------|---------|---------------|
| `mean` | $\bar{d}_k = \frac{1}{T}\sum_t d_k(t)$ | Average proximity to anchor $k$ |
| `min` | $\min_t d_k(t)$ | Closest approach to anchor $k$ |
| `trend` | Linear slope of $d_k(t)$ | Positive = drifting away, negative = approaching |
| `last` | $d_k(T)$ | Current distance to anchor $k$ |

**Applications:**
- **Mental health:** Trend toward depression anchor with decreasing `min` signals progressive deterioration
- **Finance:** `last` vs `mean` reveals whether current positioning is typical or extreme
- **MLOps:** Rising `trend` across all anchors indicates the model is entering an out-of-distribution region

---

## Prediction

### `cvx.predict(trajectory, target_timestamp)`

**Theory:** Linear extrapolation from trajectory history. For Neural ODE prediction, use `TemporalIndex(model_path="model.pt").predict()`.

---

## Cross-Domain Application Matrix

| Function | Mental Health | Finance | Drug Discovery | MAP-Elites | Molecular Dynamics | MLOps |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| `velocity` | Crisis speed | Regime transition | Optimization speed | Exploration rate | Transition speed | Drift rate |
| `hurst_exponent` | Anti-persistence (d=-0.41) | Trending vs reverting | Convergence | Stagnation | Basin stability | Drift persistence |
| `detect_changepoints` | Behavioral regime change | Market regime | Campaign phase | Archive reorganization | Conformational transition | Drift onset |
| `path_signature` | Trajectory pattern matching | Path-dependent options | Campaign similarity | Solution lineage | Folding pathway | Training dynamics |
| `frechet_distance` | Similar user trajectories | Portfolio path comparison | Campaign comparison | Archive comparison | Trajectory similarity | Model comparison |
| `wasserstein_drift` | Topical migration | Sector rotation | Chemical space shift | Niche redistribution | State redistribution | Feature drift |
| `fisher_rao_distance` | Distribution shift | Risk profile change | Activity profile change | Diversity metric | State population change | Class balance drift |
| `event_features` | Night posting, burstiness | Trading patterns | Experiment cadence | Generation dynamics | Simulation events | Request patterns |
| `topological_features` | Topic fragmentation | Market structure | Landscape topology | Archive connectivity | Conformational basins | Embedding structure |
| `project_to_anchors` | Drift toward clinical poles | Sector proximity | Reference compound distance | Archetype tracking | State proximity | Distribution drift |
| `anchor_summary` | Deterioration trend | Position vs. norm | Campaign summary | Exploration summary | Basin residence | Drift summary |
