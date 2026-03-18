# RFC-007: Advanced Temporal Primitives

**Status**: Proposed
**Created**: 2026-03-18
**Authors**: Manuel Couto Pintos
**Related**: RFC-004 (Semantic Regions), RFC-005 (Query Capabilities), RFC-006 (Anchor Projection)

---

## Summary

This RFC proposes five new temporal primitives that exploit CVX's unique structural advantage: co-located temporal, entity, and vector data within a single index. These operations are **impossible or prohibitively expensive** in generic vector databases (Qdrant, Milvus, Pinecone) because they require cross-entity temporal reasoning — correlating how multiple entities move through embedding space over time.

| Primitive | What it answers |
|-----------|----------------|
| **Temporal Join** | "When were entities A and B semantically close simultaneously?" |
| **Granger Causality** | "Do A's movements in embedding space predict B's?" |
| **Temporal Motifs** | "Does entity A exhibit recurring semantic patterns?" |
| **Cohort Drift** | "How did a *group* of entities evolve collectively?" |
| **Counterfactual Trajectories** | "What would have happened if the trajectory diverged at change point C?" |

---

## Motivation

### Current State

CVX's analytics operate on **single entities**: velocity, drift, change points, Hurst exponent — all take one `entity_id` and analyze its trajectory in isolation. The only cross-entity operation is `search_trajectories()` (RFC-005) which finds similar trajectories but doesn't analyze their temporal relationship.

Real-world questions are inherently **multi-entity and temporal**:

- **Clinical NLP**: "Did the patient's language shift *before* or *after* their social network's language shifted?" (contagion vs. isolation)
- **Finance**: "Do sector leaders' embeddings Granger-cause followers?" (information flow)
- **Cybersecurity**: "Do compromised hosts exhibit the same recurring drift pattern?" (lateral movement signature)
- **Social media**: "How did the collective discourse of a political community change after event X?" (cohort drift)

None of these can be answered with single-entity analytics.

### Why Qdrant Cannot Do This

In Qdrant, entity trajectories require: (1) scroll with entity_id filter, (2) sort by timestamp client-side, (3) repeat for entity B, (4) align timestamps client-side, (5) compute cross-entity statistics client-side. This is:
- **N+1 queries** for N entities (cohort drift on 1000 entities = 1000 scrolls)
- **No temporal alignment** — client must bin/interpolate timestamps manually
- **No index-level optimization** — each scroll is a full scan of the payload index

CVX's `entity_index: BTreeMap<u64, Vec<(i64, u32)>>` provides O(log N) trajectory access. Cross-entity operations can be executed server-side with a single read lock, accessing all trajectories from shared memory.

---

## Proposed Changes

### 1. Temporal Join (Priority: P0)

#### Problem

Two entities may have periods of semantic convergence and divergence that are invisible in point-level kNN. A **temporal join** finds time windows where two (or more) entities were within semantic distance ε of each other.

#### Definition

Given entities A and B, distance threshold ε, and window size w:

```
TemporalJoin(A, B, ε, w) → Vec<(t_start, t_end, mean_distance)>
```

Returns time intervals [t_start, t_end] where:
- Both A and B have at least one point within the interval
- The minimum pairwise distance between A's and B's points within any sliding window of size w is ≤ ε

#### Algorithm

```
1. Retrieve trajectory_A = entity_index[A]  // O(log N + |A|)
2. Retrieve trajectory_B = entity_index[B]  // O(log N + |B|)
3. Merge-scan both trajectories by timestamp (both sorted)
4. For each window of size w:
   a. Compute min distance between A-points and B-points in window
   b. If min_dist ≤ ε: extend current convergence interval
   c. If min_dist > ε: close interval if open, record (t_start, t_end, mean_dist)
```

**Complexity**: O(|A| + |B|) for the merge-scan, plus O(k_A × k_B × D) per window for distance computation, where k_A, k_B are points-per-window (typically small, 1-5).

#### API

**Rust** (in `cvx-query`):
```rust
pub struct TemporalJoinResult {
    pub start: i64,
    pub end: i64,
    pub mean_distance: f32,
    pub min_distance: f32,
    pub points_a: usize,
    pub points_b: usize,
}

pub fn temporal_join(
    index: &impl TemporalIndexAccess,
    entity_a: u64,
    entity_b: u64,
    epsilon: f32,
    window_us: i64,
    filter: TemporalFilter,
) -> Vec<TemporalJoinResult>;
```

**REST** (`POST /v1/temporal-join`):
```json
{
  "entity_a": 42,
  "entity_b": 99,
  "epsilon": 0.3,
  "window_days": 7,
  "time_range": {"start": "2025-01-01T00:00:00Z", "end": "2025-12-31T00:00:00Z"}
}
```

**Python**:
```python
joins = index.temporal_join(
    entity_a=42, entity_b=99,
    epsilon=0.3, window_days=7,
    start=t1, end=t2,
)
# Returns: [(t_start, t_end, mean_dist, min_dist, n_a, n_b), ...]
```

#### Multi-entity variant

For cohort analysis, a **group temporal join** finds windows where a set of entities converge:

```rust
pub fn group_temporal_join(
    index: &impl TemporalIndexAccess,
    entities: &[u64],
    epsilon: f32,
    min_entities: usize,  // at least this many must be within ε
    window_us: i64,
    filter: TemporalFilter,
) -> Vec<GroupJoinResult>;
```

This enables questions like: "When did at least 5 of these 20 users converge semantically?"

#### Use Cases

| Domain | Query |
|--------|-------|
| Clinical NLP | Find periods where a patient's language converged with known distress patterns |
| Finance | Detect correlation regimes between sector ETFs |
| Cybersecurity | Identify hosts that simultaneously converged (coordinated attack) |
| Social media | Detect echo chamber formation (users converging semantically) |

---

### 2. Granger Causality in Embedding Space (Priority: P1)

#### Problem

Standard Granger causality tests whether past values of time series X improve prediction of time series Y beyond Y's own past. We extend this to **embedding trajectories**: does entity A's movement through embedding space *precede* entity B's movement?

#### Definition

Given entities A and B, max lag L, and significance threshold p:

```
GrangerTest(A, B, L, p) → GrangerResult {
    direction: A→B | B→A | Bidirectional | None,
    optimal_lag: i64,
    f_statistic: f64,
    p_value: f64,
    effect_size: f64,
}
```

#### Algorithm

The key challenge is that embedding trajectories are multivariate (D dimensions). We use **dimensionality-reduced Granger**:

```
1. Retrieve trajectories A, B
2. Align to common timestamps (linear interpolation at union of timestamps)
3. Reduce dimensionality:
   Option a) Project to anchor space (RFC-006) → K dimensions
   Option b) Project to region distribution space (RFC-004) → K regions
   Option c) PCA to top-k principal components
4. For each dimension d in reduced space:
   a. Fit VAR(L) model: B_d(t) = Σ_l α_l·B_d(t-l) + Σ_l β_l·A_d(t-l) + ε
   b. F-test: H₀: all β_l = 0 (A does not Granger-cause B in dimension d)
5. Combine per-dimension p-values via Fisher's method
6. Repeat with A↔B swapped for bidirectionality test
```

**Dimensionality reduction is critical**: Granger on raw 768-dim embeddings requires fitting 768 × L parameters per direction — massively overparameterized for typical trajectory lengths (50-500 points). Region distributions (K ≈ 60-80) or anchor projections (K ≈ 5-20) are tractable.

#### Complexity

- Trajectory retrieval: O(log N + |A| + |B|)
- Interpolation alignment: O(|A| + |B|)
- VAR fitting per dimension: O(K × L² × T) where T = aligned length
- Total: O(K² × L² × T) — feasible for K ≤ 80, L ≤ 20, T ≤ 1000

#### API

**Rust** (in `cvx-analytics`):
```rust
pub struct GrangerResult {
    pub direction: GrangerDirection,
    pub optimal_lag_us: i64,
    pub f_statistic: f64,
    pub p_value: f64,
    pub effect_size: f64,        // partial R² improvement
    pub per_dimension: Vec<DimensionGranger>,
}

pub enum GrangerDirection {
    AToB,
    BToA,
    Bidirectional,
    None,
}

pub fn granger_causality(
    traj_a: &[(i64, &[f32])],
    traj_b: &[(i64, &[f32])],
    max_lag: usize,
    significance: f64,
) -> Result<GrangerResult, AnalyticsError>;
```

**REST** (`POST /v1/granger`):
```json
{
  "entity_a": 42,
  "entity_b": 99,
  "max_lag_days": 14,
  "significance": 0.05,
  "projection": "regions",
  "level": 3
}
```

#### Use Cases

| Domain | Query |
|--------|-------|
| Clinical NLP | Does social network language change precede patient deterioration? |
| Finance | Do large-cap embeddings Granger-cause small-cap? (information cascade) |
| Social media | Does influencer discourse predict follower discourse shift? |
| Cybersecurity | Does C2 server traffic pattern precede infected host behavior change? |

#### References

- Granger, C.W.J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3).
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
- Tank, A. et al. (2021). Neural Granger causality. *IEEE TPAMI*, 44(8).

---

### 3. Temporal Motifs (Priority: P1)

#### Problem

Entities may exhibit **recurring patterns** in their semantic trajectory — seasonal cycles, behavioral loops, or pathological repetitions. Current analytics (velocity, drift, Hurst) capture statistical properties but not **structural recurrence**.

#### Definition

A temporal motif is a subsequence of the trajectory that recurs with similarity above threshold θ. We adapt the **Matrix Profile** algorithm to embedding trajectories:

```
TemporalMotifs(entity_id, window_size, θ) → Vec<Motif {
    pattern: Vec<(i64, Vec<f32>)>,   // the canonical motif
    occurrences: Vec<(i64, f32)>,    // (start_timestamp, match_distance)
    period: Option<i64>,             // detected periodicity if regular
}>
```

#### Algorithm: Matrix Profile on Region Trajectories

The Matrix Profile (Yeh et al., 2016) computes the nearest-neighbor distance for every subsequence of a time series. Adapted for embedding trajectories:

```
1. Retrieve region_trajectory(entity_id, level=3) → Vec<(i64, Vec<f32>)>
   // K-dimensional distribution per time window (K ≈ 60-80)
2. Extract all subsequences of length w (sliding window)
3. For each subsequence s_i:
   a. Compute DTW or Euclidean distance to all other subsequences s_j (j ≠ i, non-overlapping)
   b. Record nearest-neighbor distance → Matrix Profile[i]
   c. Record nearest-neighbor index → Matrix Profile Index[i]
4. Motif discovery:
   a. Find minima in Matrix Profile → these are the most-repeated patterns
   b. Follow Matrix Profile Index chain to find all occurrences
   c. Test for periodicity: if occurrences are regularly spaced, compute period
5. Discord discovery (bonus):
   a. Find maxima in Matrix Profile → these are the most *unusual* subsequences
   b. Return as anomalous temporal patterns
```

**Key insight**: Operating on region trajectories (K ≈ 80) rather than raw embeddings (D = 768) makes the Matrix Profile tractable. The STOMP algorithm (Zhu et al., 2016) computes the full Matrix Profile in O(N² × K) — for a 500-window trajectory with K=80, this is ~20M operations (milliseconds).

#### API

**Rust** (in `cvx-analytics`):
```rust
pub struct Motif {
    pub pattern: Vec<Vec<f32>>,     // canonical subsequence (region distributions)
    pub occurrences: Vec<MotifOccurrence>,
    pub period_us: Option<i64>,
    pub discord_score: f32,         // how unusual the pattern is globally
}

pub struct MotifOccurrence {
    pub start_timestamp: i64,
    pub distance: f32,              // match quality
}

pub fn discover_motifs(
    trajectory: &[(i64, &[f32])],
    window_size: usize,
    max_motifs: usize,
    exclusion_zone: f32,            // fraction of window_size for non-trivial matches
) -> Result<Vec<Motif>, AnalyticsError>;

pub fn discover_discords(
    trajectory: &[(i64, &[f32])],
    window_size: usize,
    max_discords: usize,
) -> Result<Vec<Discord>, AnalyticsError>;
```

**REST** (`GET /v1/entities/{id}/motifs`):
```json
{
  "window_days": 30,
  "max_motifs": 5,
  "level": 3,
  "start": "2025-01-01T00:00:00Z",
  "end": "2025-12-31T00:00:00Z"
}
```

#### Use Cases

| Domain | Query |
|--------|-------|
| Clinical NLP | Detect cyclic depressive episodes (recurrence is a key diagnostic criterion) |
| Finance | Identify recurring market regimes (risk-on/risk-off cycles) |
| Social media | Detect seasonal behavioral patterns (election cycles, holidays) |
| IoT/Industrial | Identify recurring failure modes in sensor embedding trajectories |

#### References

- Yeh, C.-C. M. et al. (2016). Matrix Profile I: All Pairs Similarity Joins for Time Series Subsequences. *IEEE ICDM*.
- Zhu, Y. et al. (2016). Matrix Profile II: Exploiting a Novel Algorithm and GPUs (STOMP). *IEEE ICDM*.
- Gharghabi, S. et al. (2018). Domain agnostic online semantic segmentation for multi-dimensional time series. *PVLDB*, 12(4).

---

### 4. Cohort Drift (Priority: P0)

#### Problem

All existing drift analytics operate on individual entities. But many research questions are about **groups**: "How did the depressed cohort's language change after the intervention?" requires aggregating drift across multiple entities.

#### Definition

Given a set of entity IDs (the cohort) and two time points/ranges:

```
CohortDrift(entities, t1, t2) → CohortDriftReport {
    mean_drift: f32,                    // average individual drift magnitude
    median_drift: f32,
    std_drift: f32,
    centroid_drift: DriftReport,        // drift of the cohort centroid
    dispersion_change: f32,             // did the cohort spread or compress?
    convergence_score: f32,             // are entities moving toward each other?
    top_dimensions: Vec<(usize, f32)>,  // most changed dims across cohort
    outliers: Vec<(u64, f32)>,          // entities drifting most/least vs cohort
}
```

#### Algorithm

```
1. For each entity in cohort:
   a. Retrieve vector at t1 (nearest neighbor interpolation)
   b. Retrieve vector at t2
   c. Compute individual drift vector: Δ_i = v_i(t2) - v_i(t1)
2. Cohort-level statistics:
   a. mean_drift = mean(||Δ_i||)
   b. centroid_t1 = mean(v_i(t1)), centroid_t2 = mean(v_i(t2))
   c. centroid_drift = drift_report(centroid_t1, centroid_t2)
   d. dispersion_t1 = mean(||v_i(t1) - centroid_t1||)
   e. dispersion_t2 = mean(||v_i(t2) - centroid_t2||)
   f. dispersion_change = dispersion_t2 - dispersion_t1
      // > 0 = cohort diverging, < 0 = cohort converging
   g. convergence_score = cosine_similarity of individual drift vectors
      // high = all moving in same direction, low = moving in random directions
3. Outlier detection:
   a. Compute z-score of each ||Δ_i|| against cohort distribution
   b. Flag entities with |z| > 2 as outliers
4. Dimensional attribution:
   a. mean_delta = mean(Δ_i) across cohort
   b. top_dimensions = argsort(|mean_delta|)[:top_n]
```

#### API

**Rust** (in `cvx-analytics`):
```rust
pub struct CohortDriftReport {
    pub n_entities: usize,
    pub mean_drift_l2: f32,
    pub median_drift_l2: f32,
    pub std_drift_l2: f32,
    pub centroid_drift: DriftReport,
    pub dispersion_t1: f32,
    pub dispersion_t2: f32,
    pub dispersion_change: f32,
    pub convergence_score: f32,
    pub top_dimensions: Vec<(usize, f32)>,
    pub outliers: Vec<CohortOutlier>,
}

pub struct CohortOutlier {
    pub entity_id: u64,
    pub drift_magnitude: f32,
    pub z_score: f32,
    pub drift_direction_alignment: f32, // cosine with cohort mean direction
}

pub fn cohort_drift(
    trajectories: &[(u64, Vec<(i64, &[f32])>)],
    t1: i64,
    t2: i64,
    top_n: usize,
) -> Result<CohortDriftReport, AnalyticsError>;
```

**REST** (`POST /v1/cohort/drift`):
```json
{
  "entity_ids": [1, 2, 3, 42, 99],
  "t1": "2025-06-01T00:00:00Z",
  "t2": "2025-12-01T00:00:00Z",
  "top_n": 10
}
```

#### Region-level Cohort Drift

For interpretability, cohort drift can also be computed on region distributions:

```rust
pub fn cohort_region_drift(
    index: &impl TemporalIndexAccess,
    entities: &[u64],
    level: usize,
    t1: i64,
    t2: i64,
    window_us: i64,
    alpha: f32,
) -> Result<CohortRegionDriftReport, AnalyticsError>;
```

This tracks how the cohort's distribution over semantic regions changes — e.g., "the depressed cohort shifted 15% of their mass from the 'social' region to the 'isolation' region between June and December."

#### Use Cases

| Domain | Query |
|--------|-------|
| Clinical NLP | Measure treatment effect: did the intervention cohort converge toward healthy patterns? |
| Finance | Sector rotation: is the tech sector converging or diverging? |
| Social media | Polarization: are political communities drifting apart? |
| Education | Learning outcomes: did the cohort converge toward target knowledge embeddings? |

#### References

- Wager, S. & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects. *JASA*, 113(523).
- Rubin, D. B. (2005). Causal inference using potential outcomes. *JASA*, 100(469).

---

### 5. Counterfactual Trajectories (Priority: P2)

#### Problem

When a change point is detected, a natural question is: "What *would* have happened if the change hadn't occurred?" This requires extrapolating the pre-change trajectory beyond the change point and comparing with the actual post-change trajectory.

#### Definition

Given entity A, change point at time t_cp, and horizon h:

```
Counterfactual(A, t_cp, h) → CounterfactualResult {
    actual_trajectory: Vec<(i64, Vec<f32>)>,       // what actually happened
    counterfactual_trajectory: Vec<(i64, Vec<f32>)>, // what would have happened
    divergence_curve: Vec<(i64, f32)>,              // distance between the two over time
    total_divergence: f32,                          // area under divergence curve
    max_divergence: (i64, f32),                     // peak divergence
}
```

#### Algorithm

Two methods, depending on available infrastructure:

**Method A: Linear extrapolation** (no Neural ODE required)
```
1. Retrieve trajectory before t_cp
2. Fit linear trend to last N pre-change points (OLS on each dimension)
3. Extrapolate linearly from t_cp to t_cp + h
4. Compare with actual post-change trajectory
```

**Method B: Neural ODE extrapolation** (RFC-003, when available)
```
1. Retrieve trajectory before t_cp
2. Use trained Neural ODE to predict trajectory from t_cp to t_cp + h
   conditioned only on pre-change dynamics
3. Compare with actual post-change trajectory
```

**Divergence computation** (shared):
```
4. For each timestamp t in [t_cp, t_cp + h]:
   a. Interpolate actual vector at t
   b. Interpolate counterfactual vector at t
   c. divergence(t) = cosine_distance(actual(t), counterfactual(t))
5. total_divergence = ∫ divergence(t) dt  (trapezoidal rule)
6. max_divergence = argmax_t divergence(t)
```

#### API

**Rust** (in `cvx-analytics`):
```rust
pub struct CounterfactualResult {
    pub change_point: i64,
    pub actual: Vec<(i64, Vec<f32>)>,
    pub counterfactual: Vec<(i64, Vec<f32>)>,
    pub divergence_curve: Vec<(i64, f32)>,
    pub total_divergence: f32,
    pub max_divergence_time: i64,
    pub max_divergence_value: f32,
    pub method: CounterfactualMethod,
}

pub enum CounterfactualMethod {
    LinearExtrapolation,
    NeuralODE,
}

pub fn counterfactual_trajectory(
    pre_change: &[(i64, &[f32])],
    post_change: &[(i64, &[f32])],
    change_point: i64,
    method: CounterfactualMethod,
) -> Result<CounterfactualResult, AnalyticsError>;
```

**REST** (`POST /v1/entities/{id}/counterfactual`):
```json
{
  "change_point": "2025-06-15T00:00:00Z",
  "horizon_days": 90,
  "method": "linear"
}
```

#### Use Cases

| Domain | Query |
|--------|-------|
| Clinical NLP | Quantify the impact of a life event on language trajectory |
| Finance | "What would the stock's trajectory be without the earnings shock?" |
| Policy evaluation | Measure treatment effect as divergence between actual and counterfactual |
| Cybersecurity | "How much did the compromise alter the host's behavior?" |

#### References

- Brodersen, K. H. et al. (2015). Inferring causal impact using Bayesian structural time-series models. *Annals of Applied Statistics*, 9(1).
- Abadie, A. (2021). Using synthetic controls: Feasibility, data requirements, and methodological aspects. *JEL*, 59(2).

---

## Implementation Plan

### Phase 1: `feat/cohort-drift` (P0)

| Task | Crate | Details |
|------|-------|---------|
| `cohort_drift()` | cvx-analytics | Vector-level cohort drift with all statistics |
| `cohort_region_drift()` | cvx-analytics | Region-distribution-level cohort drift |
| `nearest_vector_at()` helper | cvx-index | Nearest-neighbor temporal interpolation for arbitrary timestamp |
| REST endpoint `POST /v1/cohort/drift` | cvx-api | — |
| Python `index.cohort_drift()` | cvx-python | — |
| Tests: known cohort with controlled drift | cvx-analytics | Verify convergence_score, outlier detection |

### Phase 2: `feat/temporal-join` (P0)

| Task | Crate | Details |
|------|-------|---------|
| `temporal_join()` | cvx-analytics | Pairwise entity convergence windows |
| `group_temporal_join()` | cvx-analytics | Multi-entity convergence |
| REST endpoint `POST /v1/temporal-join` | cvx-api | — |
| Python `index.temporal_join()` | cvx-python | — |
| Tests: two entities that converge/diverge at known times | cvx-analytics | — |

### Phase 3: `feat/temporal-motifs` (P1)

| Task | Crate | Details |
|------|-------|---------|
| Matrix Profile computation (STOMP) | cvx-analytics | O(N²×K) on region trajectories |
| `discover_motifs()` | cvx-analytics | Top-k motifs with occurrences + periodicity |
| `discover_discords()` | cvx-analytics | Top-k unusual subsequences |
| REST endpoint `GET /v1/entities/{id}/motifs` | cvx-api | — |
| Python `index.motifs()` | cvx-python | — |
| Tests: synthetic trajectory with planted motif | cvx-analytics | — |

### Phase 4: `feat/granger-embeddings` (P1)

| Task | Crate | Details |
|------|-------|---------|
| Temporal alignment (interpolation) | cvx-analytics | Linear interp to common grid |
| VAR(L) fitting | cvx-analytics | Per-dimension OLS with lag terms |
| F-test + Fisher's method | cvx-analytics | Combined p-value across dimensions |
| `granger_causality()` | cvx-analytics | Full pipeline with direction detection |
| REST endpoint `POST /v1/granger` | cvx-api | — |
| Python `index.granger_test()` | cvx-python | — |
| Tests: synthetic A→B causality with known lag | cvx-analytics | — |

### Phase 5: `feat/counterfactual` (P2)

| Task | Crate | Details |
|------|-------|---------|
| Linear extrapolation (OLS per dim) | cvx-analytics | Baseline method |
| Divergence curve computation | cvx-analytics | Cosine distance over time + integration |
| `counterfactual_trajectory()` | cvx-analytics | Linear method initially, Neural ODE later |
| REST endpoint `POST /v1/entities/{id}/counterfactual` | cvx-api | — |
| Python `index.counterfactual()` | cvx-python | — |
| Tests: trajectory with known break + linear pre-trend | cvx-analytics | — |

---

## Impact on Existing Code

| Component | Change |
|-----------|--------|
| `cvx-analytics` | New modules: `cohort.rs`, `temporal_join.rs`, `motifs.rs`, `granger.rs`, `counterfactual.rs` |
| `cvx-index` | Add `nearest_vector_at(entity_id, timestamp)` helper to TemporalHnsw |
| `cvx-query/types.rs` | Extend `TemporalQuery` enum with new variants |
| `cvx-api/handlers.rs` | 5 new REST endpoints |
| `cvx-python` | 5 new public functions |

No breaking changes to existing API. All additions are new methods/endpoints.

---

## Verification

### Test Plan

| Primitive | Test Strategy |
|-----------|--------------|
| Temporal Join | Two entities with controlled convergence at t=1000: verify window detected. Non-overlapping entities: verify empty result. |
| Granger | Synthetic: A(t) = sin(t), B(t) = sin(t - lag). Verify A→B detected, not B→A. |
| Motifs | Trajectory with repeated subsequence at positions [0, 100, 200]. Verify 3 occurrences, correct period. |
| Cohort Drift | 10 entities all shifting +0.1 in dim 0. Verify convergence_score ≈ 1.0, top_dim = 0. |
| Counterfactual | Linear trajectory with break at t=50. Verify counterfactual follows original slope. |

### Benchmark Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Temporal Join (2 entities, 1K pts each) | < 10ms | Dominated by distance computation |
| Cohort Drift (100 entities) | < 50ms | 100 trajectory lookups + statistics |
| Motifs (500-window trajectory) | < 100ms | STOMP on K=80 dim region trajectory |
| Granger (2 entities, 500 pts, L=10) | < 200ms | VAR fitting is the bottleneck |

---

## References

1. Granger, C.W.J. (1969). Investigating causal relations. *Econometrica*, 37(3).
2. Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
3. Tank, A. et al. (2021). Neural Granger causality. *IEEE TPAMI*, 44(8).
4. Yeh, C.-C. M. et al. (2016). Matrix Profile I. *IEEE ICDM*.
5. Zhu, Y. et al. (2016). Matrix Profile II: STOMP. *IEEE ICDM*.
6. Brodersen, K. H. et al. (2015). Causal impact. *Annals of Applied Statistics*, 9(1).
7. Abadie, A. (2021). Synthetic controls. *JEL*, 59(2).
8. Wager, S. & Athey, S. (2018). Heterogeneous treatment effects. *JASA*, 113(523).
9. Gharghabi, S. et al. (2018). Semantic segmentation for time series. *PVLDB*, 12(4).
