# RFC-004: Graph-Based Semantic Regions for Temporal Trajectory Analysis

**Status**: Proposed
**Created**: 2026-03-16
**Authors**: Manuel Couto Pintos
**Related**: RFC-003 (Neural ODE), Tutorial B1 (Mental Health)

---

## Summary

This RFC proposes leveraging the hierarchical structure of the HNSW graph to define **semantic regions** — natural clusters derived from the graph topology itself — and using these regions to transform noisy per-post embedding trajectories into smooth, interpretable temporal signals. This addresses a fundamental limitation discovered during real-data experiments: individual post embeddings are too noisy for direct temporal analysis because users discuss many topics regardless of their mental state.

---

## Problem Statement

### The Signal-to-Noise Problem in Temporal Embedding Analysis

When analyzing social media user trajectories for mental health monitoring (Tutorial B1), we discovered that operating directly on per-post embedding trajectories produces poor results:

| Metric | Result (eRisk, N=200) | Expected |
|--------|----------------------|----------|
| Velocity discrimination (Cohen's d) | -0.028 | Significant effect size |
| PELT change points detected | 0 of 200 users | At least some in depression group |
| Velocity p-value (Mann-Whitney) | 0.96 | < 0.05 |
| Hurst p-value | 0.25 | < 0.05 |
| ROC-AUC (temporal features) | 0.79 | > 0.85 |

**Root cause**: A user with depression still writes about sports, weather, food, work — topics unrelated to their mental state. Each post embedding captures the *topic* of that post, not the user's underlying *psychological state*. The resulting trajectory in $\mathbb{R}^{384}$ jumps chaotically between topical regions regardless of the user's actual condition.

```
User timeline (depressed):
  Day 1: "Great game yesterday"        → sports region
  Day 2: "Can't sleep again"           → distress region  ← signal
  Day 3: "New recipe for dinner"       → food region
  Day 4: "Everything feels pointless"  → distress region  ← signal
  Day 5: "Meeting at 3pm"              → work region
```

Direct trajectory analysis sees: sports → distress → food → distress → work — noise dominates signal.

### Why External Clustering Is Not the Answer

Running k-means or HDBSCAN on embeddings is possible, but:
1. **Doesn't use CVX's graph structure** — redundant computation on data already organized in a navigable graph
2. **Arbitrary K** — requires hyperparameter tuning with no principled choice
3. **Doesn't showcase CVX** — the tutorial should demonstrate the database's capabilities, not scikit-learn

### The HNSW Graph Already Has the Answer

The HNSW graph is a **hierarchical navigable structure** where:
- **Level 0**: All nodes, densely connected — fine-grained neighborhoods
- **Level 1**: Fewer nodes ($\sim N/M$) — natural community representatives
- **Level 2**: Even fewer ($\sim N/M^2$) — macro-region centroids
- **Level L**: The "skip list" nodes that reached level L during random insertion

Nodes at higher levels are the graph's **natural hubs** — they were randomly promoted during construction and became highly connected waypoints for navigation. Their neighborhoods at level 0 define organic semantic regions. This is not a coincidence: the HNSW construction algorithm produces a structure topologically equivalent to a hierarchical clustering (Malkov & Yashunin, 2018, §3.1).

---

## Proposed Solution

### Concept: Graph-Based Semantic Regions

Define a **semantic region** as the Voronoi cell of a high-level HNSW node:

$$\text{region}(v) = \arg\min_{h \in \text{hubs}(L)} d(v, h)$$

where $\text{hubs}(L)$ is the set of nodes present at HNSW level $L$. Each hub is the natural centroid of its region.

The number of regions is determined by the graph structure itself:
- Level 1: $\sim N / M$ regions (e.g., 7,600 for N=121K, M=16) — too many
- Level 2: $\sim N / M^2$ regions (e.g., 475) — reasonable
- Level 3: $\sim N / M^3$ regions (e.g., 30) — coarse, highly interpretable
- Adjustable by choosing the level

### User Trajectory as Region Distribution

Instead of tracking a user's raw embedding trajectory, track their **temporal distribution over regions**:

1. For each post, assign it to its nearest region (hub at level $L$)
2. In a sliding window of $W$ days, count the proportion of posts in each region
3. Apply Exponential Moving Average (EMA) for smoothing:

$$\mathbf{s}_t = \alpha \cdot \mathbf{x}_t + (1 - \alpha) \cdot \mathbf{s}_{t-1}$$

where $\mathbf{x}_t$ is the region distribution at time $t$ and $\alpha = 2/(W+1)$.

The resulting trajectory is a smooth curve in $\mathbb{R}^K$ (K = number of regions) that captures **how the user's topical focus evolves over time**.

### Interpretability via Region Labeling

Each region can be interpreted by examining its representative posts:

```python
region = index.region_info(region_id=7)
# → {
#     "centroid_node": 42381,
#     "size": 3200,
#     "sample_texts": [
#         "I feel so alone these days",
#         "Nobody understands what I'm going through",
#         "Can't stop crying"
#     ],
#     "label": "emotional_distress"  # manually or auto-assigned
# }
```

This transforms CVX from "your embedding drifted 0.3 in dimension 127" to **"your posts shifted from 40% social to 60% emotional distress over 3 weeks"**.

---

## Architecture

### New Components

```
┌──────────────────────────────────────────────────────────┐
│                  CVX Semantic Regions                      │
│                                                           │
│  ┌─────────────┐     ┌──────────────────┐                │
│  │ HNSW Graph  │────→│ RegionExtractor  │                │
│  │ (existing)  │     │   regions(level) │                │
│  └─────────────┘     │   assign(vector) │                │
│                      └────────┬─────────┘                │
│                               │                           │
│  ┌────────────────────────────▼──────────────────────┐   │
│  │           TemporalRegionTracker                    │   │
│  │                                                    │   │
│  │  user_region_trajectory(entity_id, window, alpha)  │   │
│  │  → Vec<(timestamp, region_distribution)>           │   │
│  │                                                    │   │
│  │  region_transitions(entity_id)                     │   │
│  │  → Vec<(timestamp, from_region, to_region)>        │   │
│  └────────────────────────────────────────────────────┘   │
│                               │                           │
│  ┌────────────────────────────▼──────────────────────┐   │
│  │           Existing CVX Analytics                   │   │
│  │  velocity() / detect_changepoints() / hurst()      │   │
│  │  → Applied to smooth region-distribution vectors   │   │
│  └────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

### API Design

#### Rust (cvx-index)

```rust
impl<D: DistanceMetric> HnswGraph<D> {
    /// Return node IDs present at the given HNSW level.
    /// These are the natural "hub" nodes of the graph hierarchy.
    /// Level 0 = all nodes, level max_level = entry point only.
    pub fn nodes_at_level(&self, level: usize) -> Vec<u32>;

    /// Assign a vector to its nearest hub at the given level.
    /// Uses greedy descent from the entry point — O(log N).
    pub fn assign_region(&self, vector: &[f32], level: usize) -> u32;
}

impl<D: DistanceMetric> TemporalHnsw<D> {
    /// Get the semantic regions (hubs) at a given HNSW level.
    /// Returns (node_id, vector, n_assigned) for each hub.
    pub fn regions(&self, level: usize) -> Vec<RegionInfo>;

    /// Compute the smoothed region-distribution trajectory for an entity.
    ///
    /// - `level`: HNSW level to use for regions
    /// - `window_days`: sliding window width in days
    /// - `alpha`: EMA smoothing factor (0 = no smoothing, 1 = no memory)
    ///
    /// Returns a trajectory where each point is a K-dimensional vector
    /// representing the entity's distribution over K regions at that time.
    pub fn region_trajectory(
        &self,
        entity_id: u64,
        level: usize,
        window_days: i64,
        alpha: f32,
    ) -> Vec<(i64, Vec<f32>)>;

    /// Get posts assigned to a specific region, optionally filtered by time.
    pub fn region_members(
        &self,
        region_id: u32,
        level: usize,
        filter: TemporalFilter,
    ) -> Vec<(u64, i64, u32)>;  // (entity_id, timestamp, node_id)
}
```

#### Python (cvx-python)

```python
import chronos_vector as cvx

index = cvx.TemporalIndex()
# ... ingest posts ...

# Discover semantic regions from graph structure
regions = index.regions(level=2)
# → [(region_id, centroid_vector, n_members), ...]

# Get smoothed trajectory for a user
traj = index.region_trajectory(
    entity_id=42,
    level=2,
    window_days=7,
    alpha=0.3,
)
# → [(timestamp, [0.3, 0.0, 0.5, 0.1, ...]), ...]
# Each vector sums to 1.0 (probability distribution over regions)

# Inspect what a region contains (for labeling)
members = index.region_members(region_id=7, level=2)
# → [(entity_id, timestamp, node_id), ...]
# Retrieve original texts via node_id → map back to dataset

# Apply CVX analytics on the smooth trajectory
vel = cvx.velocity(traj, timestamp=mid_point)
cps = cvx.detect_changepoints(entity_id=42, trajectory=traj, penalty=5.0)
h = cvx.hurst_exponent(traj)
```

#### TemporalIndexAccess Trait Extension

```rust
pub trait TemporalIndexAccess: Send + Sync {
    // ... existing methods ...

    /// Get hub nodes at a given HNSW level (semantic region centroids).
    fn regions(&self, level: usize) -> Vec<(u32, Vec<f32>, usize)>;

    /// Compute smoothed region-distribution trajectory for an entity.
    fn region_trajectory(
        &self,
        entity_id: u64,
        level: usize,
        window_days: i64,
        alpha: f32,
    ) -> Vec<(i64, Vec<f32>)>;
}
```

---

## Implementation Plan

### Phase 1: Graph Region Extraction (cvx-index)

**Effort**: Low

Add `nodes_at_level()` and `assign_region()` to `HnswGraph`. These are read-only operations on the existing graph structure — no new data structures needed.

`nodes_at_level(L)` iterates all nodes and returns those whose `neighbors.len() > L` (meaning they have edges at level L or above).

`assign_region(vector, level)` performs greedy descent from the entry point to level L+1, then returns the closest node at level L. This is exactly the first phase of HNSW search, already implemented.

### Phase 2: Region Trajectory Computation (cvx-index)

**Effort**: Medium

Add `region_trajectory()` to `TemporalHnsw`. Algorithm:

1. Get all regions at level L: `hubs = nodes_at_level(L)`
2. Get entity's posts: `trajectory(entity_id, All)`
3. For each post, assign to nearest hub: `assign_region(vector, L)`
4. Group posts by time window (W days)
5. For each window, compute proportion in each region
6. Apply EMA smoothing with factor α
7. Return smoothed trajectory as `Vec<(timestamp, Vec<f32>)>`

### Phase 3: Region Inspection (cvx-index)

**Effort**: Low

Add `region_members()` — filter entity_index by region assignment. Enables interpretability: retrieve posts belonging to a region, read original text, understand what the region represents.

### Phase 4: Python Bindings (cvx-python)

**Effort**: Low

Expose `regions()`, `region_trajectory()`, `region_members()` via PyO3.

### Phase 5: Tutorial B1 Rewrite (notebooks)

**Effort**: Medium

Rewrite B1 with the new pipeline:
1. Ingest all posts into CVX
2. Extract regions from graph (level 2-3)
3. Label regions by inspecting sample posts
4. Compute smoothed region trajectories per user
5. Visualize: stacked area charts showing topical evolution
6. Apply CVX analytics (velocity, CPD, Hurst) on smooth trajectories
7. Train classifier on region-trajectory features
8. Expected: much higher ROC-AUC, meaningful change points, interpretable results

---

## Expected Impact

### On B1 Tutorial Results

| Metric | Current (raw embeddings) | Expected (region trajectories) |
|--------|------------------------|---------------------------------|
| Velocity discrimination | d = -0.03 (none) | Meaningful effect size |
| PELT change points | 0 detected | Detectable regime transitions |
| Hurst p-value | 0.25 (not significant) | < 0.05 |
| ROC-AUC | 0.79 | > 0.85 |
| Interpretability | "dim_127 changed" | "shift from social to isolation" |

### On CVX as a Product

This feature transforms CVX from a vector database that happens to track time into a **temporal semantic analysis engine**:

- **No external clustering needed** — regions emerge from the graph
- **Adaptive granularity** — change level for coarser/finer regions
- **Built-in interpretability** — inspect region contents to understand what changed
- **Smooth signals** — EMA over region distributions produces clean trajectories
- **Works with any embedding** — not specific to NLP; applies to finance, IoT, etc.

---

## Mathematical Foundation

### HNSW Level Distribution

The probability that a node reaches level $L$ follows a geometric distribution:

$$P(\text{level} \geq L) = \left(\frac{1}{M}\right)^L$$

For $N$ nodes with $M=16$:
- Level 0: $N$ nodes (all)
- Level 1: $\sim N/16$ nodes
- Level 2: $\sim N/256$ nodes
- Level 3: $\sim N/4096$ nodes

This gives natural region counts of $N/M^L$, controllable by choosing $L$.

### Region Assignment Complexity

Assignment uses the existing HNSW greedy descent, which is $O(\log N)$ per query. For $P$ posts per user, the full region trajectory computation is $O(P \log N)$ — the same as inserting the posts.

### EMA Smoothing

The Exponential Moving Average with factor $\alpha$ has effective window width $W_{\text{eff}} = 2/\alpha - 1$. For $\alpha = 0.3$, $W_{\text{eff}} \approx 5.7$ time steps. This suppresses high-frequency noise while preserving genuine shifts.

The smoothed region distribution is guaranteed to remain a valid probability distribution (sums to 1, non-negative) since EMA is a convex combination of distributions.

---

## References

1. Malkov, Y.A. & Yashunin, D.A. (2018). Efficient and Robust ANN Using HNSW Graphs. IEEE TPAMI. §3.1: layer structure as probabilistic skip list.
2. Bamler, R. & Mandt, S. (2017). Dynamic Word Embeddings. ICML. Temporal smoothing of embedding trajectories.
3. Blondel, V. et al. (2008). Fast Unfolding of Communities in Large Networks. JSTAT. Community detection on graphs (Louvain method) — alternative to level-based regions.
4. Couto, M. et al. (2025). Temporal Word Embeddings for Psychological Disorder Early Detection. JHIR.
5. Coppersmith, G. et al. (2018). NLP of Social Media as Screening for Suicide Risk. BMI Insights.
