---
title: "Unified Theory: Temporal Vector Analytics for Intelligent Memory"
description: "How time transforms vectors into trajectories, and why trajectories need structure, probability, and causality"
---

## 1. The Core Thesis

**Time converts vectors into trajectories. Trajectories have mathematical structure that static vectors cannot capture. Exploiting this structure produces better decisions than similarity-based retrieval alone.**

ChronosVector (CVX) implements this thesis across six layers, from raw storage to probabilistic reasoning. Each layer builds on the previous, and no layer alone is sufficient.

---

## 2. Layer 0: Vectors in Time

### The Fundamental Object

A **temporal point** $(e, t, \mathbf{v})$ captures an entity $e$ observed at time $t$ with embedding $\mathbf{v} \in \mathbb{R}^D$. A **trajectory** is a time-ordered sequence:

$$\mathcal{T}_e = \{(t_1, \mathbf{v}_1), (t_2, \mathbf{v}_2), \ldots, (t_n, \mathbf{v}_n)\} \quad t_1 < t_2 < \cdots < t_n$$

Standard vector databases store only $\mathbf{v}$ — they discard $t$ and $e$. CVX preserves all three, enabling every subsequent layer.

### Storage

- **HNSW index**: $O(\log N)$ approximate nearest neighbor search with SIMD distance kernels
- **Temporal filtering**: RoaringBitmap pre-filter by time range (< 1 byte per vector)
- **Episode encoding**: `entity_id = episode_id × 10000 + step_index` groups steps into episodes

### The Anisotropy Problem

Modern embedding models produce vectors in a narrow cone — all pairwise cosine similarities $\sim 0.96$. Centering (subtracting the global mean $\boldsymbol{\mu}$) amplifies the discriminative signal **30×**:

$$\mathbf{v}_{\text{centered}} = \mathbf{v} - \boldsymbol{\mu}, \quad \boldsymbol{\mu} = \frac{1}{N}\sum_{i=1}^N \mathbf{v}_i$$

This is not optional — without centering, all downstream analytics operate on noise (Ethayarajh, EMNLP 2019; Su et al., ACL 2021).

---

## 3. Layer 1: Differential Calculus on Trajectories

Treating trajectories as differentiable curves in $\mathbb{R}^D$ enables **kinematic analysis**:

### Velocity

$$\mathbf{v}'(t) \approx \frac{\mathbf{v}(t + \Delta t) - \mathbf{v}(t - \Delta t)}{2\Delta t}$$

The magnitude $\|\mathbf{v}'(t)\|$ measures **rate of semantic change**. High velocity = rapid behavioral shift.

### Drift

$$\text{drift}_{L2}(t_1, t_2) = \|\mathbf{v}(t_2) - \mathbf{v}(t_1)\|_2$$

Cumulative displacement from an initial state. When projected onto anchor vectors, drift measures **proximity to domain-specific concepts** (e.g., DSM-5 symptoms).

### Hurst Exponent

$$\mathbb{E}\left[\frac{R(n)}{S(n)}\right] \sim c \cdot n^H$$

$H > 0.5$: trajectory is **persistent** (trends continue). $H < 0.5$: **anti-persistent** (trends reverse). $H = 0.5$: random walk.

### Change Point Detection (PELT)

$$\min_{\tau_1, \ldots, \tau_m} \sum_{i=1}^{m+1} \left[\mathcal{C}(\mathbf{y}_{\tau_{i-1}+1:\tau_i}) + \beta\right]$$

Detects moments when the statistical properties of the trajectory change abruptly — regime transitions, onset events, behavioral shifts (Killick et al., 2012).

---

## 4. Layer 2: Algebraic Invariants

### Path Signatures (Lyons, 1998)

The depth-$k$ truncated signature of a path $\mathbf{X}: [0,T] \to \mathbb{R}^D$ is:

$$S(\mathbf{X})^{i_1, \ldots, i_k} = \int_0^T \int_0^{t_k} \cdots \int_0^{t_2} dX^{i_1}_{t_1} \cdots dX^{i_k}_{t_k}$$

This is a **universal**, **reparametrization-invariant** descriptor: two trajectories with the same shape (regardless of speed) produce similar signatures. At depth 2, the $D + D^2$ features capture both displacement and signed area (rotation/order).

### Topological Features

Persistent homology tracks the **connectivity structure** of trajectory point clouds:

$$\beta_0(r) = \text{number of connected components at radius } r$$

The persistence diagram encodes the birth/death of topological features across scales — more robust than single-scale clustering.

---

## 5. Layer 3: Distributional Geometry

When trajectories are projected onto regions or anchors, they become **probability distributions** over discrete states. The geometry of distributions requires specialized metrics:

### Fisher-Rao Distance

The geodesic on the statistical manifold of categorical distributions:

$$d_{FR}(\mathbf{p}, \mathbf{q}) = 2 \arccos\left(\sum_i \sqrt{p_i \cdot q_i}\right)$$

Unlike KL divergence, this is a true metric (symmetric, triangle inequality). Range: $[0, \pi]$.

### Wasserstein (Earth Mover's) Distance

The optimal transport cost, respecting the **geometry of the underlying space**:

$$W(\mathbf{p}, \mathbf{q}) = \inf_{\gamma \in \Pi(\mathbf{p}, \mathbf{q})} \int \|\mathbf{x} - \mathbf{y}\| \, d\gamma(\mathbf{x}, \mathbf{y})$$

Unlike Fisher-Rao, Wasserstein accounts for *how far apart* the categories are — moving mass between nearby regions costs less than between distant ones.

---

## 6. Layer 4: Temporal Causality

### Temporal Edges

Each entity's trajectory has an intrinsic order — step $n$ **precedes** step $n+1$. Temporal edges (`TemporalEdgeLayer`) encode this:

- `successor(node)` → what happened next for this entity
- `predecessor(node)` → what happened before
- `causal_search(query, k, temporal_context)` → find similar states + walk forward/backward

This enables the **continuation pattern**: "find where someone was in my situation, show me what they did next."

### Typed Edges (RFC-013)

Beyond temporal succession, typed edges encode **relational structure**:

| Edge Type | Meaning | Example |
|-----------|---------|---------|
| `CausedSuccess` | This action contributed to a win | retrieve + follow → win → edge |
| `CausedFailure` | This action was present during failure | retrieve + follow → fail → edge |
| `SameActionType` | Same abstract action in different contexts | "navigate" in episode A ↔ "navigate" in episode B |
| `RegionTransition` | Observed movement between semantic clusters | Region 5 → Region 12 with probability 0.7 |

### Granger Causality

For cross-entity causal discovery:

$$X \xrightarrow{\text{Granger}} Y \iff P(Y_t \mid Y_{<t}, X_{<t}) \neq P(Y_t \mid Y_{<t})$$

"Does entity A's trajectory history improve prediction of entity B's future?" — implemented as F-test on lagged regression residuals.

---

## 7. Layer 5: Probabilistic Reasoning

### Region MDP (RFC-013 Part A)

HNSW regions define a **discrete state space**. Observed trajectories define **transitions**. The result is a Markov Decision Process:

$$P(s' \mid s, a) = \frac{\text{count}(s \xrightarrow{a} s')}{\sum_{s''} \text{count}(s \xrightarrow{a} s'')}$$

$$P(\text{success} \mid s, a) = \frac{1 + \sum[\text{reward} > 0.5]}{2 + n} \quad \text{(Beta prior)}$$

This answers: "in states like mine, which action type has the highest success probability?"

### Bayesian Network (cvx-bayes)

When variables have **conditional dependencies** that a linear scorer cannot capture:

$$P(\text{success} \mid \text{task}, \text{region}, \text{action}) = \frac{P(\text{task}, \text{region}, \text{action}, \text{success})}{P(\text{task}, \text{region}, \text{action})}$$

The network factorizes the joint distribution via the DAG structure:

$$P(\mathbf{X}) = \prod_{i} P(X_i \mid \text{parents}(X_i))$$

Each CPT is learned from observations with Laplace smoothing. Inference computes posteriors via variable elimination.

### Context-Aware Decay

The discovery from E7c/E7d/E7e: **blind reward decay destroys useful experts**. Context-aware decay only penalizes experts when:

1. Task type matches the failed game
2. The agent actually followed the expert's action
3. The expert is in a low-quality region (informed by MDP)

---

## 8. Layer 6: Structural Knowledge

### Knowledge Graph (cvx-graph)

Encodes **compositional structure** that neither vectors nor probabilities capture:

- **Task plans**: `heat_then_place` requires find → take → heat → take → put
- **Shared sub-plans**: `heat` and `clean` both start with find → take
- **Constraints**: after `take`, valid next actions are `go`/`use`/`put`, not `take` again
- **Transfer**: if I know how to `find → take` for cleaning, I can reuse it for heating

The graph enables **structural guidance** during retrieval: the agent knows what step comes next from the graph, and uses CVX to find the best concrete realization.

---

## 9. The Integrated System

The six layers compose into a **closed-loop active memory**:

```
OBSERVATION → embed → HNSW search → candidates
                           ↓
                    Bayesian scoring (Layer 5)
                    ├── Similarity (Layer 0)
                    ├── Recency (Layer 1)
                    ├── Reward (Layer 4 — typed edges)
                    ├── P(success | context) (Layer 5 — BN/MDP)
                    └── Task plan step (Layer 6 — KG)
                           ↓
                    LLM chooses action
                           ↓
                    OUTCOME → update
                    ├── Win → add to index (Layer 0)
                    ├── Win → add CausedSuccess edges (Layer 4)
                    ├── Fail → context-aware decay (Layer 5)
                    ├── Update MDP transitions (Layer 5)
                    ├── Update BN posteriors (Layer 5)
                    └── Update KG if new structure learned (Layer 6)
```

### Why No Layer Alone Is Sufficient

| Layer | What it provides | What it cannot do |
|-------|-----------------|-------------------|
| 0 (HNSW) | Find similar states | Distinguish success from failure |
| 1 (Calculus) | Measure change speed | Predict next action |
| 2 (Signatures) | Compare trajectory shapes | Reason about task structure |
| 3 (Distributions) | Compare population-level patterns | Make individual decisions |
| 4 (Causality) | Attribute outcomes to actions | Estimate probabilities |
| 5 (Bayesian) | Compute conditional probabilities | Represent compositional knowledge |
| 6 (Knowledge) | Encode task structure | Score candidates numerically |

Each layer addresses a specific limitation of the layers below it. The full system is more than the sum of its parts.

---

## 10. Empirical Validation

| Experiment | What it tested | Result |
|-----------|---------------|--------|
| B2 (clinical anchoring) | Layer 0+1: centered drift detection | F1=0.744 on eRisk depression |
| B8 (ParlaMint) | Layer 0+3: rhetorical profiling | F1=0.94 predicting speaker gender |
| E5 (ALFWorld GPT-4o) | Layer 0+4: causal retrieval | 20% → 43.3% task completion |
| E7b (online learning) | Layer 0+4+5: reward decay | 6.7% → 26.7% across 3 rounds |
| E7e (context-aware) | Layer 5: conditional decay | Peak 30%, plateau 19.5% (vs 14.8%) |

---

## 11. References

### Temporal Embeddings & Anisotropy
1. Ethayarajh (2019). "How Contextual are Contextualized Word Representations?" EMNLP
2. Su et al. (2021). "Whitening Sentence Representations for Better Semantics and Faster Retrieval." ACL

### Path Signatures & Topology
3. Lyons (1998). *Differential Equations Driven by Rough Signals*
4. Carlsson (2009). "Topology and Data." Bulletin of the AMS

### Change Point Detection
5. Killick et al. (2012). "Optimal detection of changepoints with a linear computational cost." JASA

### Information Geometry
6. Amari & Nagaoka (2000). *Methods of Information Geometry*

### Bayesian Networks
7. Pearl (1988). *Probabilistic Reasoning in Intelligent Systems*
8. Koller & Friedman (2009). *Probabilistic Graphical Models*

### Knowledge Graphs
9. Hogan et al. (2021). "Knowledge Graphs." ACM Computing Surveys

### Agent Memory
10. Park et al. (2023). "Generative Agents." UIST
11. Shinn et al. (2023). "Reflexion." NeurIPS
12. Chen et al. (2021). "Decision Transformer." NeurIPS
13. Hafner et al. (2023). "DreamerV3." arXiv

### Causal Inference
14. Bareinboim et al. (2022-2024). Causal Reinforcement Learning

### Optimal Transport
15. Villani (2008). *Optimal Transport: Old and New*
