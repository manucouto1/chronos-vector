# RFC-005: Extending Query Capabilities and Ingestion Performance

**Status**: Proposed
**Created**: 2026-03-17
**Authors**: Manuel Couto Pintos
**Related**: RFC-002 (Performance), RFC-004 (Semantic Regions)

---

## Summary

ChronosVector is a temporal vector database — a system for storing, retrieving, and analyzing entities that evolve in a high-dimensional vector space over time. This RFC proposes foundational improvements to make CVX a practical tool across domains: high-throughput batch ingestion, region-based queries, temporal neighborhood queries, and trajectory similarity search. These are domain-agnostic capabilities that serve any application where entities have temporal vector representations — from financial instruments and cybersecurity events to molecular simulations and population dynamics.

---

## Motivation

### Current State

After building scientific tutorials (B1: mental health detection), we identified that the bottlenecks are not in analytics but in **core database operations**:

1. **Ingestion**: ~450 pts/sec from Python. A 1M-post dataset takes ~37 minutes. Competing systems (Qdrant, Milvus) ingest at 10K-100K pts/sec.
2. **Region queries**: We can discover regions (RFC-004), but cannot query "give me all points in region R in time range T1-T2".
3. **Trajectory comparison**: We can retrieve a single entity's trajectory, but cannot search for "entities whose trajectory is similar to this one".
4. **Temporal neighbors**: We can search kNN at a point in time, but cannot ask "who were entity X's neighbors at each timestep?"

These are not nice-to-haves — they are fundamental database operations that every domain needs.

### Competitive Landscape

| Feature | Qdrant | Milvus | Weaviate | **CVX** |
|---------|--------|--------|----------|---------|
| Vector kNN | Yes | Yes | Yes | Yes |
| Metadata filtering | Rich | Rich | Rich | Basic |
| **Temporal-native queries** | No | No | No | **Yes (unique)** |
| **Hierarchical graph regions** | No | No | Partial (graph model) | **Yes (unique)** |
| **Entity trajectories** | No | No | No | **Yes (unique)** |
| **Trajectory analytics** | No | No | No | **Yes (unique)** |
| Batch ingestion | Yes (fast) | Yes (fast) | Yes (fast) | **No (450 pts/sec)** |

CVX's differentiator is temporal-native operations. But the ingestion bottleneck undermines credibility. No one will adopt a database that takes 37 minutes to load 1M records.

---

## Proposed Changes

### 1. Batch Ingestion API (Priority: P0)

#### Problem

Each Python `insert()` call:
1. Crosses the PyO3 FFI boundary
2. Acquires a write lock on the HNSW graph
3. Performs full graph traversal: O(ef_construction × log N) with ef_construction=200
4. Releases the lock

For N inserts, this is N lock acquisitions and N independent traversals.

#### Solution

Expose `ConcurrentTemporalHnsw::queue_insert()` and `flush_inserts()` (already implemented in Rust, RFC-002-04) to Python, plus add a high-level `bulk_insert()` that accepts NumPy arrays directly.

**Python API:**

```python
# Low-level: queue + flush
index.queue_insert(entity_id=1, timestamp=1000, vector=[0.1, 0.2, ...])
index.flush()  # Applies all queued inserts under single write lock

# High-level: numpy batch (zero-copy where possible)
index.bulk_insert(
    entity_ids=np.array([1, 1, 2, 2, ...], dtype=np.uint64),
    timestamps=np.array([1000, 2000, 1000, 2000, ...], dtype=np.int64),
    vectors=embeddings,  # np.ndarray shape (N, D), float32
)
```

**Rust implementation:**

```rust
/// Accept numpy arrays via PyO3 + numpy crate
#[pyo3(signature = (entity_ids, timestamps, vectors, ef_construction=None))]
fn bulk_insert(
    &mut self,
    entity_ids: PyReadonlyArray1<u64>,
    timestamps: PyReadonlyArray1<i64>,
    vectors: PyReadonlyArray2<f32>,
    ef_construction: Option<usize>,
) -> usize {
    // Optionally lower ef_construction during bulk load
    let original_ef = self.inner.config().ef_construction;
    if let Some(ef) = ef_construction {
        self.inner.set_ef_construction(ef);
    }

    let ids = entity_ids.as_slice().unwrap();
    let ts = timestamps.as_slice().unwrap();
    let vecs = vectors.as_array();

    for i in 0..ids.len() {
        self.inner.insert(ids[i], ts[i], &vecs.row(i).to_vec());
    }

    if ef_construction.is_some() {
        self.inner.set_ef_construction(original_ef);
    }
    ids.len()
}
```

**Expected improvement:**
- Eliminates N-1 lock acquisitions → single lock for entire batch
- Eliminates PyO3 FFI overhead per row (~50μs × N)
- Optional reduced ef_construction during bulk load (200 → 50-100)
- **Target: 3,000-10,000 pts/sec** (7-22× improvement)

**References:**
- pgvector parallel HNSW build: 85% reduction in build time (GSI Technology, 2024)
- Qdrant bulk optimization: disable indexing during load, rebuild after (Qdrant docs)
- hnswlib: supports batch add with pre-allocated graph (Malkov & Yashunin, 2018)

---

### 2. Region Members Query (Priority: P1)

#### Problem

RFC-004 introduced `regions(level)` which returns region centroids, and `region_trajectory()` which tracks distribution over regions. But there is no way to ask: **"which points belong to region R?"**

This is a fundamental database query. Without it, regions are opaque statistical constructs rather than navigable data structures.

#### Solution

Add `region_members()` to the core API:

```python
# Get all points in region R at level L, optionally time-filtered
members = index.region_members(
    region_id=42,
    level=3,
    filter_start=1000,  # optional
    filter_end=5000,     # optional
)
# Returns: [(node_id, entity_id, timestamp, vector), ...]
```

**Rust trait extension (TemporalIndexAccess):**

```rust
fn region_members(
    &self,
    region_hub: u32,
    level: usize,
    filter: TemporalFilter,
) -> Vec<(u32, u64, i64, Vec<f32>)>;
```

**Implementation:** For each node at level 0, compute `assign_region(node_vector, level)` and check if it maps to the requested hub. This is O(N) for a full scan but can be cached after first computation.

**Use cases (domain-agnostic):**
- "Which entities are in the high-volatility cluster right now?" (finance)
- "Which hosts exhibit anomalous behavior in this time window?" (security)
- "Which molecules converged to this region of the fitness landscape?" (drug discovery)
- "Which users are in the distress-related topic cluster this month?" (mental health)

**References:**
- IVFFlat indexes (pgvector) provide analogous cluster-membership retrieval via Voronoi partitions
- Community detection in graphs provides multi-scale cluster membership (Fortunato, 2010; Blondel et al., 2008)

---

### 3. Distance Acceleration via Quantization (Priority: P0)

#### Problem

Each distance computation during HNSW construction is O(D) — for D=768 (MentalRoBERTa embeddings), this means 768 float32 multiplications per candidate. With ef_construction=200, each insert evaluates ~200-400 candidates → **150K-300K FLOPs per insert**. This is the dominant cost (>80% of insert time).

Competing systems (Qdrant, Milvus, Faiss) accelerate distance computation using vector quantization — compact codes that allow fast approximate distances.

#### Solution

A `Quantizer` trait abstracts over acceleration strategies. The HNSW graph stores full float32 vectors (needed for trajectories, signatures, exact queries) alongside compact codes for fast approximate distances during construction.

**Trait definition** (in `cvx-core/traits/quantizer.rs`):

```rust
pub trait Quantizer: Send + Sync {
    type Code: Clone + Send + Sync;
    fn encode(&self, vector: &[f32]) -> Self::Code;
    fn distance_approx(&self, a: &Self::Code, b: &Self::Code) -> f32;
    fn distance_exact(&self, a: &[f32], b: &[f32]) -> f32;
    fn is_accelerated(&self) -> bool;
    fn name(&self) -> &str;
}
```

**Implementations:**

| Strategy | Code size (D=768) | Speedup | Recall impact | Training |
|----------|-------------------|---------|---------------|----------|
| `NoQuantizer` (default) | 0 bytes | 1× | None | No |
| `ScalarQuantizer` (SQ8) | 768 bytes | ~4× | < 1% | No |
| `ProductQuantizer` (PQ) | 96 bytes | ~8× | 1-3% | Yes (codebook) |
| `BinaryQuantizer` (BQ) | 96 bytes | ~32× | 5-10% | No |

**Two-phase distance:** During candidate selection, use `distance_approx` on codes. During final neighbor selection (heuristic pruning), use `distance_exact` on full vectors. This mirrors Qdrant's approach.

**References:**
- Qdrant uses Scalar Quantization for HNSW construction by default
- Jégou, H. et al. (2011). Product quantization for nearest neighbor search. *IEEE TPAMI*.
- Guo, R. et al. (2020). Accelerating large-scale inference with anisotropic vector quantization. *ICML*.

---

### 4. Trajectory Similarity Search (Priority: P2)

#### Problem

ChronosVector stores trajectories (sequences of temporal vectors per entity) but cannot compare them. The query "find entities whose trajectory is similar to entity X" requires computing trajectory-level similarity — a fundamentally different operation from point-level kNN.

#### Solution

```python
# Find entities with similar temporal evolution
similar = index.search_trajectories(
    entity_id=42,          # reference entity
    k=10,                   # number of results
    method="frechet",       # similarity measure
    filter=TemporalFilter.Range(start, end),
)
# Returns: [(entity_id, similarity_score), ...]
```

**Supported similarity measures:**

| Measure | Complexity | Properties | Best for |
|---------|-----------|-----------|----------|
| **Discrete Fréchet** | O(nm) | Respects ordering, handles unequal lengths | General trajectories |
| **DTW** (Dynamic Time Warping) | O(nm) | Handles temporal warping | Misaligned sequences |
| **Region-level Fréchet** | O(nm) on K dims | Operates on region distributions, not raw embeddings | Robust, fast |
| **Cosine on mean** | O(D) | Simple, fast | Baseline |

The key insight: **Region-level Fréchet** operates on K-dimensional region distributions (~60-100 dims at L3) rather than D-dimensional raw embeddings (768 dims). This is both faster (lower dimensionality) and more robust (region distributions are smoother than raw embeddings).

**Implementation strategy:**
1. Phase 1: Brute-force — compute trajectory similarity between query and all entities. Feasible for <10K entities.
2. Phase 2: Index — build a secondary index on trajectory summaries (mean vector + temporal features) for approximate trajectory retrieval, then refine with exact measures.

**Use cases:**
- "Which users had a similar posting evolution to this depressed user?" (mental health)
- "Which stocks followed AAPL's trajectory in the last quarter?" (finance)
- "Which training runs had similar loss curves?" (ML research)
- "Which populations evolved similarly?" (evolutionary computation)

**References:**
- Trajectory similarity survey: Toohey & Duckham (2015), "Trajectory Similarity Measures", ACM SIGSPATIAL
- Spatio-temporal trajectory similarity: comprehensive survey (arXiv:2303.05012, 2023)
- FastDTW: linear-time approximation of DTW (Salvador & Chan, 2007)
- Distributed trajectory similarity: DITA system (Xie et al., 2017), VLDB

---

### 5. Path Signatures for Trajectory Representation (Priority: P1)

#### Problem

Current trajectory features (velocity, Hurst, CPD) are ad-hoc summaries that discard the sequential structure of trajectories. We need a principled, mathematically grounded feature of trajectories that:
- Captures the **order** of events (not just statistics)
- Is a **universal nonlinearity** (any continuous function of the path can be approximated by a linear function of its signature)
- Supports **incremental updates** via Chen's Identity: S(α * β) = S(α) ⊗ S(β)

The last property is crucial for a database: when a new point is inserted, the signature updates in O(K²) instead of recomputing from scratch in O(N·K²).

#### Solution

Implement truncated path signatures on region trajectories (K~80 dims at L3):

```python
# Compute path signature of an entity's region trajectory
sig = index.path_signature(entity_id=42, level=3, depth=2)
# Returns: float vector of dimension K + K² = 80 + 6400 = 6,480

# Log-signature (compact, removes redundancies via BCH formula)
log_sig = index.log_signature(entity_id=42, level=3, depth=2)
# Returns: float vector of dimension K + K(K-1)/2 = 80 + 3,160 = 3,240

# Trajectory similarity via signature distance
similar = index.search_by_signature(entity_id=42, k=10, level=3, depth=2)
```

**Key insight:** Path signatures on **region trajectories** (not raw embeddings) are tractable:
- Region trajectory at L3: ~80 dims → depth 2 signature = 6,480 features
- Raw embedding trajectory: 768 dims → depth 2 = 590K features (unusable)
- The HNSW graph hierarchy provides the dimensionality reduction that makes signatures practical

**Chen's Identity enables incremental updates:**
- On `insert(entity, timestamp, vector)`: assign region → extend region trajectory → update signature via tensor product with existing partial signature
- Cost: O(K²) per insert (K=80 → 6,400 operations)
- Without Chen's: O(N·K²) full recomputation on every new point

**References:**
- Lyons, T.J. (1998). Differential equations driven by rough signals. *Revista Matemática Iberoamericana*.
- Chevyrev, I. & Kormilitzin, A. (2016). A primer on the signature method in ML. *arXiv:1603.03788*.
- Kidger, P. & Lyons, T. (2021). Signatory: differentiable computations of the signature. *ICLR*.
- Existing CVX spec: `design/CVX_Stochastic_Analytics_Spec.md` §3.

---

## Non-Goals

The following are explicitly **out of scope** for this RFC:

| Non-goal | Reason |
|----------|--------|
| Integrated visualization | Client responsibility. The DB returns data; Plotly/D3 renders it. |
| Text processing / NLP | Preprocessing concern. CVX is vector-agnostic. |
| Classification / ML models | Downstream application logic. |
| Payload storage | Vectors + metadata is sufficient. Full document storage is a different system. |
| Sequential prediction / streaming ML | Application-level concern built on top of insert + search. |

---

## Implementation Plan

Each phase is a separate feature branch, merged to develop independently.

### Completed Phases

#### Phase 1: `feat/bulk-insert-quantizer` (P0) ✅

- `set_ef_construction()` / `set_ef_search()` in cvx-index + Python
- `bulk_insert(numpy)` in cvx-python
- `Quantizer` trait with `NoQuantizer` and `ScalarQuantizer` in cvx-core
- Benchmark: ef=25 → 668 pts/sec (1.7× vs baseline 395)

#### Phase 2: `feat/quantizer-integration` (P0) ✅

- SQ8 integrated into `HnswGraph`: stores uint8 codes alongside float32 vectors
- `search_layer` uses `distance_fast` for candidate exploration, exact for re-ranking
- `enable/disable_scalar_quantization()` in cvx-index + Python
- Benchmark: SQ8 + ef=25 → 805 pts/sec (2.4× vs baseline)
- 80 cvx-index tests pass, 7/10 search quality overlap

#### Phase 3: `feat/region-members` (P1) ✅

- `region_members(hub, level, filter)` in TemporalHnsw
- Added to `TemporalIndexAccess` trait + `ConcurrentTemporalHnsw`
- Python: `index.region_members(region_id, level, start, end)`
- Verified: counts match `regions()`, time filter works correctly

#### Phase 4: `feat/path-signatures` (P1) ✅

- `compute_signature()`: truncated at depth 1-3 with iterated integrals
- `compute_log_signature()`: compact antisymmetric alternative
- `update_signature_incremental()`: Chen's Identity for O(K²) updates
- `signature_distance()`: L2 between signatures
- Python: `cvx.path_signature()`, `cvx.log_signature()`, `cvx.signature_distance()`
- 6 tests pass including Chen's Identity validation

#### Phase 5: `feat/trajectory-similarity` (P2) ✅

- `discrete_frechet()`: O(n×m) DP algorithm
- Python: `cvx.frechet_distance(traj_a, traj_b)`
- 91 total tests pass across cvx-analytics

---

### Pending Phases

#### Phase 6: `feat/wasserstein-distance` (P1)

**Goal:** Measure drift between region distributions using optimal transport instead of L2.

Wasserstein respects the topology of the region graph: two distributions that shift mass between *neighboring* regions have lower distance than two that shift between *distant* regions. L2 treats all region pairs as equally far.

| Task | Crate | Details |
|------|-------|---------|
| Implement Sliced Wasserstein distance | cvx-analytics | 1D projections + sort + L1, O(K × n_proj × K log K) |
| Region graph distance matrix | cvx-index | Pairwise distances between region hubs for exact Wasserstein |
| `wasserstein_drift(dist_t1, dist_t2)` | cvx-analytics | Compare two region distributions |
| Expose in Python | cvx-python | `cvx.wasserstein_drift(dist_a, dist_b)` |
| Tests: metric axioms, triangle inequality | cvx-analytics | — |

**Spec reference:** `CVX_Stochastic_Analytics_Spec.md` §6.4.

**References:**
- Villani, C. (2008). *Optimal Transport: Old and New*. Springer.
- Bonneel, N. et al. (2015). Sliced and Radon Wasserstein barycenters. *JMIV*.

#### Phase 7: `feat/temporal-point-process` (P1)

**Goal:** Extract features from the *timestamps themselves* — not the vectors. The pattern of *when* events occur is an independent signal from *what* the vectors contain.

| Task | Crate | Details |
|------|-------|---------|
| Intensity estimation λ(t) via kernel smoothing | cvx-analytics | Gaussian kernel, bandwidth selection |
| Inter-event interval statistics | cvx-analytics | Beyond mean/std: entropy, burstiness, memory coefficient |
| Self-exciting detection (Hawkes test) | cvx-analytics | Fit exponential kernel, test α > 0 |
| Temporal entropy H(gaps) | cvx-analytics | Regularity of posting/event pattern |
| Expose as `cvx.event_features(timestamps)` | cvx-python | Returns dict of features |
| Tests: known Poisson vs Hawkes sequences | cvx-analytics | — |

**Key features to extract:**

| Feature | Formula | What it captures |
|---------|---------|-----------------|
| `intensity_mean` | mean(λ(t)) | Average event rate |
| `intensity_trend` | slope of λ(t) over time | Accelerating or decelerating |
| `burstiness` | (σ_gap - μ_gap) / (σ_gap + μ_gap) | −1 = regular, 0 = Poisson, +1 = bursty |
| `memory_coefficient` | autocorrelation of gaps at lag 1 | Do short gaps follow short gaps? |
| `temporal_entropy` | −Σ p(bin) log p(bin) | Regularity of event timing |
| `hawkes_alpha` | Fitted self-exciting parameter | Strength of event clustering |
| `circadian_strength` | Amplitude of 24h Fourier component | Presence of daily rhythm |

**References:**
- Hawkes, A.G. (1971). Spectra of some self-exciting and mutually exciting point processes. *Biometrika*.
- Goh, K.-I. & Barabási, A.-L. (2008). Burstiness and memory in complex systems. *EPL*.
- Detecting Anomalous Event Sequences with Temporal Point Processes (NeurIPS 2021).

#### Phase 8: `feat/topological-features` (P2)

**Goal:** Track topological changes in the embedding space over time using persistent homology. Detects structural shifts like "the user's topic space fragmented" or "two clusters merged".

| Task | Crate | Details |
|------|-------|---------|
| Vietoris-Rips persistent homology (depth 0-1) | cvx-analytics | Connected components + loops |
| Betti curve computation over time windows | cvx-analytics | β₀(t), β₁(t) as temporal signal |
| Change detection on Betti curves | cvx-analytics | Apply PELT to topological features |
| Expose as `cvx.topological_features(entity_id, ...)` | cvx-python | Returns Betti curves |
| Tests: known topologies (sphere, torus, clusters) | cvx-analytics | — |

**What Betti numbers reveal:**
- β₀ = number of connected components (clusters)
- β₁ = number of loops (cyclic patterns)
- A sudden increase in β₀ = topic space fragmentation
- A decrease in β₀ = convergence to fewer topics

**Note:** This is computationally expensive for large point clouds. Apply on region centroids (K~80 at L3) rather than raw points, making it tractable.

**References:**
- Edelsbrunner, H. & Harer, J. (2010). *Computational Topology*. AMS.
- Zigzag persistence for temporal networks. *EPJ Data Science*, 2023.
- Persistence-based statistics for detecting structural changes. *arXiv:2511.00938*, 2025.

#### Phase 9: `feat/fisher-rao-distance` (P3)

**Goal:** Use the natural Riemannian metric on the space of probability distributions (region distributions) for principled distributional comparison.

| Task | Crate | Details |
|------|-------|---------|
| Fisher-Rao distance for categorical distributions | cvx-analytics | Closed-form for multinomials |
| Fisher information matrix computation | cvx-analytics | Curvature of the statistical manifold |
| Natural gradient on region distribution space | cvx-analytics | Direction of steepest distributional change |
| Expose in Python | cvx-python | `cvx.fisher_rao_distance(dist_a, dist_b)` |
| Tests: known distribution pairs | cvx-analytics | — |

**Why it matters:** The Fisher-Rao metric is the *unique* Riemannian metric (up to scale) that is invariant under sufficient statistics (Chentsov's theorem). It's the mathematically correct way to measure distance between distributions. For categorical distributions (region proportions), it has a closed-form solution via the Bhattacharyya angle.

**References:**
- Rao, C.R. (1945). Information and accuracy attainable in estimation. *Bull. Calcutta Math. Soc.*
- Chentsov, N.N. (1982). *Statistical Decision Rules and Optimal Inference*. AMS.

---

## Impact on Existing Code

| Component | Change |
|-----------|--------|
| `cvx-core/traits` | `Quantizer` trait ✅; `region_members()` in TemporalIndexAccess ✅ |
| `cvx-index/hnsw` | SQ8 integrated ✅; `region_members()` ✅ |
| `cvx-analytics` | `signatures.rs` ✅; `trajectory.rs` ✅; pending: `wasserstein.rs`, `point_process.rs`, `topology.rs`, `fisher_rao.rs` |
| `cvx-python` | 12 public functions ✅; pending: `wasserstein_drift`, `event_features`, `topological_features`, `fisher_rao_distance` |

No breaking changes to existing API. All additions are new methods.

---

## Verification

### Completed Benchmarks

```
Ingestion (D=768, 10K points):
  Individual insert (ef=200): 395 pts/sec
  bulk_insert (ef=25):        668 pts/sec (1.7×)
  SQ8 + ef=25:                805 pts/sec (2.4×)
  Search quality at SQ8+ef25: 7/10 overlap with exact baseline

Region query:
  region_members(): verified counts match regions(), time filter correct

Path signatures:
  Chen's Identity: incremental == full computation (within 1e-8)
  Log-signature dimension: K + K(K-1)/2 verified

Fréchet:
  Identical paths → 0, parallel offset → exact offset value
  Different-order paths → positive distance
```

### Completed Test Counts

| Crate | Tests | Status |
|-------|------:|--------|
| cvx-core | 9 | ✅ All pass (incl. quantizer) |
| cvx-index | 80 | ✅ All pass |
| cvx-analytics | 91 | ✅ All pass (incl. signatures + trajectory) |

### Pending Test Plans

| Phase | Tests to write |
|-------|---------------|
| Wasserstein | Metric axioms (non-negative, symmetric, triangle inequality), known transport plans |
| Point process | Poisson sequence → burstiness ≈ 0, regular → burstiness ≈ −1, Hawkes → α > 0 |
| Topology | Unit circle → β₁ = 1, two clusters → β₀ = 2, single cluster → β₀ = 1 |
| Fisher-Rao | Distance between identical distributions = 0, orthogonal distributions = π/2 |

---

## References

1. Malkov, Y.A. & Yashunin, D.A. (2018). HNSW graphs. *IEEE TPAMI*, 42(4).
2. Jégou, H. et al. (2011). Product quantization. *IEEE TPAMI*, 33(1).
3. Guo, R. et al. (2020). Anisotropic vector quantization. *ICML*.
4. Toohey, K. & Duckham, M. (2015). Trajectory similarity measures. *ACM SIGSPATIAL*.
5. Xie, D. et al. (2017). Distributed trajectory similarity search. *VLDB*, 10(11).
6. Salvador, S. & Chan, P. (2007). FastDTW. *Intelligent Data Analysis*, 11(5).
7. Spatio-temporal trajectory similarity survey. *arXiv:2303.05012*, 2023.
8. Fortunato, S. (2010). Community detection in graphs. *Physics Reports*, 486(3-5).
9. Lyons, T.J. (1998). Differential equations driven by rough signals. *Rev. Mat. Iberoam.*
10. Chevyrev, I. & Kormilitzin, A. (2016). Signature primer. *arXiv:1603.03788*.
11. Kidger, P. & Lyons, T. (2021). Signatory. *ICLR*.
12. Vassiliades, V. et al. (2018). CVT-MAP-Elites. *IEEE Trans. Evol. Comp.*, 22(4).
13. MOSCITO (2024). Temporal subspace clustering for MD. *arXiv:2408.00056*.
14. Villani, C. (2008). *Optimal Transport: Old and New*. Springer.
15. Bonneel, N. et al. (2015). Sliced Wasserstein barycenters. *JMIV*.
16. Hawkes, A.G. (1971). Self-exciting point processes. *Biometrika*, 58(1).
17. Goh, K.-I. & Barabási, A.-L. (2008). Burstiness and memory. *EPL*, 81(4).
18. NeurIPS (2021). Detecting anomalous event sequences with TPPs.
19. Edelsbrunner, H. & Harer, J. (2010). *Computational Topology*. AMS.
20. Zigzag persistence for temporal networks. *EPJ Data Science*, 2023.
21. Rao, C.R. (1945). Information and accuracy. *Bull. Calcutta Math. Soc.*
22. Chentsov, N.N. (1982). *Statistical Decision Rules and Optimal Inference*. AMS.
