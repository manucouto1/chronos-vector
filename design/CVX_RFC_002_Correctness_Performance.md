# RFC-002: Correctness & Performance Improvements

**Status**: Proposed
**Created**: 2026-03-16
**Authors**: Manuel Couto Pintos
**Supersedes**: None
**Related**: RFC-001 (Architecture Decisions), IDR-001 through IDR-010

---

## Summary

This RFC documents 10 identified weaknesses in the current ChronosVector implementation, proposes concrete fixes for each, and establishes priority ordering for resolution. The issues span durability, algorithmic correctness, concurrency, numerical stability, and API robustness.

---

## Motivation

ChronosVector has reached Layer 12 with ~14K lines of Rust, 305+ tests, and functional coverage across all subsystems. However, several issues could compromise correctness or performance at production scale:

- **Data loss** under crash scenarios (WAL fsync gap)
- **Panics** on edge-case inputs (NaN in distance sort)
- **Suboptimal recall** from incorrectly implemented heuristic
- **Write throughput bottleneck** from global lock

These must be addressed before any production deployment or public benchmarking.

---

## RFC-002-01: WAL fsync Guarantee on Commit

**Status**: Proposed
**Severity**: CRITICAL
**Effort**: Low (< 20 lines)

### Current Behavior

```rust
// wal/mod.rs:276-281
pub fn commit(&mut self, sequence: u64) -> Result<(), StorageError> {
    self.meta.committed_sequence = sequence;
    self.flush()?;        // BufWriter::flush → OS page cache only
    self.persist_meta()?; // fs::write + rename — no directory fsync
    Ok(())
}
```

`flush()` pushes data from the BufWriter to the kernel page cache, but does NOT guarantee durability. `persist_meta()` writes via rename (atomic) but doesn't fsync the parent directory.

### Failure Scenario

1. Client calls `append()` + `commit(5)`
2. Data is in page cache but not on disk
3. Power failure
4. On recovery, `wal.meta` says `committed_sequence = 5`
5. But segment file on disk only has entries 1-3 (4-5 lost from page cache)
6. **Result**: Silent data loss of committed entries

### Proposed Fix

```rust
pub fn commit(&mut self, sequence: u64) -> Result<(), StorageError> {
    self.meta.committed_sequence = sequence;
    // 1. Sync segment data to disk BEFORE updating metadata
    if let Some(ref mut writer) = self.current_writer {
        writer.flush()?;
        writer.get_ref().sync_all()?;
    }
    // 2. Write metadata atomically
    self.persist_meta()?;
    // 3. Sync directory to ensure rename is durable
    let dir = std::fs::File::open(&self.dir)?;
    dir.sync_all()?;
    Ok(())
}
```

### Performance Impact

`sync_all()` adds ~1-10ms per commit on SSD. Mitigate with:
- Group commits (commit every N entries instead of every entry)
- Configurable durability level: `Sync` (every commit), `BatchSync` (periodic), `NoSync` (fastest, no guarantee)

### References

- Pillai et al., "All File Systems Are Not Created Equal: On the Complexity of Crafting Crash-Consistent Applications" (OSDI 2014). Demonstrates that rename() is not durable without fsync of parent directory on ext4/btrfs/xfs.
- Zheng et al., "BtrBlk: Efficient B-tree Logging for Database Storage on SSD" (SIGMOD 2022). Analyzes fsync overhead patterns on modern SSDs.

---

## RFC-002-02: NaN-Safe Distance Sorting

**Status**: Proposed
**Severity**: HIGH
**Effort**: Low (6 line changes)

### Current Behavior

```rust
// 6 occurrences across hnsw/mod.rs, temporal.rs, optimized.rs
result_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
```

If any distance is NaN (zero-norm vectors, overflow, denormalized floats), `partial_cmp` returns `None` and `unwrap()` panics. This crashes the entire server.

### Attack Vector

A single malformed vector (e.g., with components near `f32::MAX`) that produces NaN in dot product computation will make ALL subsequent searches panic, even for well-formed queries.

### Proposed Fix

```rust
// Use total_cmp (stable since Rust 1.62) which defines total ordering including NaN
result_vec.sort_by(|a, b| a.1.total_cmp(&b.1));
```

`total_cmp` places NaN after all other values, so NaN results sink to the bottom of search results instead of crashing.

### References

- IEEE 754-2019, §5.10: defines `totalOrder` predicate for total ordering of floats.
- Rust RFC 2585: `f32::total_cmp` stabilization.

---

## RFC-002-03: Heuristic Neighbor Selection Condition

**Status**: Proposed
**Severity**: HIGH
**Effort**: Medium

### Current Behavior

```rust
// optimized.rs:77-80
let is_good = selected_vectors.iter().all(|&sel_vec| {
    let dist_to_selected = metric.distance(cand_vec, sel_vec);
    cand_dist <= dist_to_selected
});
```

Condition: candidate must be closer to target than to **ALL** already-selected neighbors. This is more restrictive than the original algorithm.

### Malkov Algorithm 4 (Correct)

The paper states: *"if e is closer to q than to any element from R"* — meaning for each candidate, check against each selected neighbor independently. The candidate should be selected if it provides **new directional coverage** that existing selections don't cover.

The critical difference:
- **Current (wrong)**: `∀ s ∈ selected: dist(c, target) ≤ dist(c, s)` — ALL must hold
- **Paper (correct)**: Same condition, but with **strict inequality** `<` instead of `≤`

With `≤`, equidistant candidates are always accepted, reducing diversity. With `<`, only strictly better candidates pass, forcing the algorithm to seek genuinely diverse directions.

### Proposed Fix

```rust
let is_good = selected_vectors.iter().all(|&sel_vec| {
    let dist_to_selected = metric.distance(cand_vec, sel_vec);
    cand_dist < dist_to_selected  // strict < per Algorithm 4
});
```

### Expected Impact

Based on Malkov's benchmarks, the correct heuristic improves recall by 1-3% at the same memory budget, with the largest gains at high dimensionality (D≥128).

### References

- Malkov & Yashunin, "Efficient and Robust Approximate Nearest Neighbor Using Hierarchical Navigable Small World Graphs" (IEEE TPAMI 2018), §4.2, Algorithm 4.
- Fu et al., "Fast Approximate Nearest Neighbor Search with the Navigating Spreading-out Graph" (VLDB 2019). Extends the heuristic with angular diversity.

---

## RFC-002-04: Concurrent Insert Architecture

**Status**: Proposed
**Severity**: HIGH
**Effort**: High

### Current Behavior

```rust
// concurrent.rs:39-41
pub struct ConcurrentTemporalHnsw<D: DistanceMetric> {
    inner: RwLock<TemporalHnsw<D>>,
}
```

Single `RwLock` means:
- Insert acquires **exclusive** write lock → blocks all searches
- Long searches block subsequent inserts (writer starvation under read-heavy workload)
- No insert batching or pipelining

### Proposed Alternatives

**Option A: Insert Queue (Recommended for Phase 1)**

```rust
pub struct ConcurrentTemporalHnsw<D: DistanceMetric> {
    graph: RwLock<TemporalHnsw<D>>,
    insert_queue: Mutex<Vec<PendingInsert>>,
}
```

- Inserts are queued into a `Mutex<Vec<...>>` (fast, sub-microsecond)
- A background task drains the queue periodically, acquiring write lock once per batch
- Searches always acquire read lock (unblocked during queue drain)
- **Tradeoff**: Inserts are eventually visible (not immediate)

**Option B: Per-Node Locks (Higher Throughput)**

Replace `Vec<HnswNode>` with node-level locking:
```rust
struct ConcurrentNode {
    vector: Vec<f32>,
    neighbors: Vec<RwLock<NeighborList>>,
}
```

- Searches only lock the nodes they traverse
- Inserts lock only the modified nodes
- **Tradeoff**: Higher complexity, potential deadlocks

**Option C: Segmented Graph (Best Scalability)**

Partition the graph by entity_id ranges, each segment with its own lock:
```rust
struct SegmentedIndex {
    segments: Vec<RwLock<TemporalHnsw<D>>>,
}
```

### Decision

Defer to implementation phase. Option A is recommended for initial improvement.

### References

- Singh et al., "FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search" (arXiv 2023). Implements concurrent insert/delete with per-node locks.
- Subramanya et al., "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node" (NeurIPS 2019). Uses lock-free graph traversal with CAS operations.
- Qdrant source (qdrant/segment_manager.rs): Uses per-segment RwLock with background merge.

---

## RFC-002-05: gRPC Delivery Semantics

**Status**: Proposed
**Severity**: HIGH
**Effort**: High

### Current Behavior

```rust
// grpc/service.rs:62-92
async fn ingest_stream(&self, request: Request<tonic::Streaming<IngestPoint>>)
    -> Result<Response<IngestAck>, Status>
{
    while let Some(point) = stream.message().await? {
        // Insert immediately, no deduplication
        let node_id = self.state.index.insert(...);
        // Only last ACK returned
        last_ack = IngestAck { ... };
    }
    Ok(Response::new(last_ack))  // Only returns LAST ack
}
```

### Problems

1. **No deduplication**: Re-sent points create duplicates in the index
2. **No partial ACK**: If stream breaks after 999/1000 points, client doesn't know which 999 succeeded
3. **No backpressure**: Client can flood server memory with unbounded stream
4. **At-most-once implicit**: Points inserted but not acked if connection drops after insert but before response

### Proposed Protocol

```
Client                          Server
  |---- IngestPoint (seq=1) ---->|  → insert + WAL append
  |---- IngestPoint (seq=2) ---->|  → insert + WAL append
  |---- IngestPoint (seq=3) ---->|  → insert + WAL append
  |<--- BatchAck (committed=3) --|  → WAL commit(3)
  |---- IngestPoint (seq=4) ---->|  ...
```

- Client assigns monotonic sequence numbers
- Server ACKs every N points with the committed WAL sequence
- On reconnect, client resumes from last ACK'd sequence
- Server deduplicates by `(entity_id, timestamp)` key

### References

- Kreps, Narkhede & Rao, "Kafka: A Distributed Messaging System for Log Processing" (NetDB 2011). Defines at-least-once, at-most-once, exactly-once semantics.
- gRPC documentation, "Error Handling" (grpc.io/docs/guides/error). Recommends idempotency tokens for retry-safe RPCs.

---

## RFC-002-06: PELT Candidate Set Bounding

**Status**: Proposed
**Severity**: MEDIUM
**Effort**: Medium

### Current Behavior

```rust
// pelt.rs:84-87
if f[s] + seg_cost <= f[t] + penalty || f[t] == f64::INFINITY {
    new_candidates.push(s);
}
```

The pruning condition is functionally correct (per Killick Theorem 3.1) but the fallback `f[t] == f64::INFINITY` retains all candidates when `f[t]` hasn't been computed yet, which happens during early iterations. This can cause the candidate set to grow to O(N) before meaningful pruning begins.

### Proposed Fix

1. Remove the infinity fallback (unnecessary since `f[t]` is always set before pruning evaluation)
2. Add explicit candidate set size cap as safety net:

```rust
if f[s] + seg_cost <= f[t] + penalty {
    new_candidates.push(s);
}
// Safety cap to prevent O(N²) worst case
if new_candidates.len() > n / 2 {
    // Keep only the candidates with lowest f[s] values
    new_candidates.sort_by(|&a, &b| f[a].partial_cmp(&f[b]).unwrap_or(Ordering::Equal));
    new_candidates.truncate(n / 4);
}
```

### References

- Killick, Fearnhead & Eckley, "Optimal Detection of Changepoints with a Linear Computational Cost" (JASA 2012), Theorem 3.1: Proves PELT is O(N) under the assumption that the number of retained candidates is bounded.
- Maidstone et al., "On Optimal Multiple Changepoint Algorithms for Large Data" (Statistics & Computing 2017). Extends PELT with tighter pruning bounds.

---

## RFC-002-07: Random Level Bounding

**Status**: Proposed
**Severity**: MEDIUM
**Effort**: Low (2 lines)

### Current Behavior

```rust
// mod.rs:135-138
fn random_level(&mut self) -> usize {
    let r: f64 = self.rng.random();
    (-r.ln() * self.config.level_mult).floor() as usize
}
```

No upper bound. With `r ≈ 1e-15`, the level is ~35 for M=16. Theoretical maximum is unbounded.

### Proposed Fix

```rust
fn random_level(&mut self) -> usize {
    let r: f64 = self.rng.random();
    let level = (-r.ln() * self.config.level_mult).floor() as usize;
    // Cap at theoretical maximum: log_M(N_max)
    level.min(32)
}
```

The value 32 supports up to M^32 ≈ 10^38 nodes with M=16, far beyond any practical index.

### References

- Malkov & Yashunin (2018), §4.1: *"l_max grows as O(ln(N)/ln(M))"*. For N=10^9 and M=16, l_max ≈ 7.5.

---

## RFC-002-08: Warm Store Zone Maps

**Status**: Proposed
**Severity**: MEDIUM
**Effort**: Medium

### Current Behavior

`read_entity_chunks()` reads ALL chunk files, deserializes ALL points, then filters.

### Proposed Architecture

Add a per-chunk manifest with min/max timestamps:

```rust
struct ChunkManifest {
    chunks: Vec<ChunkMeta>,
}

struct ChunkMeta {
    filename: String,
    min_timestamp: i64,
    max_timestamp: i64,
    point_count: u32,
}
```

Range query first consults manifest, only opens chunks whose `[min_ts, max_ts]` overlaps with the query range.

### Expected Improvement

For a range query on 1% of the time span with 100 chunks: current reads 100 chunks, proposed reads ~1 chunk. **100× speedup** for selective queries.

### References

- Lamb et al., "The Vertica Analytic Database: C-Store 7 Years Later" (VLDB 2012). Zone maps (min/max per block) as fundamental pruning mechanism.
- Abadi et al., "The Design and Implementation of Modern Column-Oriented Database Systems" (Foundations and Trends in Databases 2013).

---

## RFC-002-09: PQ Codebook Initialization

**Status**: Proposed
**Severity**: MEDIUM
**Effort**: Medium

### Current Behavior

```rust
// cold/mod.rs:52-57
for c in 0..k {
    let src = vectors[c % vectors.len()];
    // Initialize from first k vectors sequentially
}
```

Sequential initialization from temporally ordered data biases centroids toward early observations.

### Proposed Fix: k-means++ Initialization

```rust
fn kmeans_pp_init(vectors: &[&[f32]], k: usize, sub_offset: usize, sub_dim: usize) -> Vec<Vec<f32>> {
    let mut centroids = vec![vectors[0][sub_offset..sub_offset+sub_dim].to_vec()];
    for _ in 1..k {
        // Compute D²(x) for each point
        let weights: Vec<f64> = vectors.iter().map(|v| {
            centroids.iter()
                .map(|c| l2_sq(&v[sub_offset..sub_offset+sub_dim], c))
                .fold(f64::INFINITY, f64::min)
        }).collect();
        // Sample proportional to D²
        let selected = weighted_sample(&weights);
        centroids.push(vectors[selected][sub_offset..sub_offset+sub_dim].to_vec());
    }
    centroids
}
```

### Expected Impact

Arthur & Vassilvitskii prove k-means++ achieves O(log k) approximation ratio vs O(N) for random initialization. In practice, this means 10-30% lower quantization error for the same k.

### References

- Arthur & Vassilvitskii, "k-means++: The Advantages of Careful Seeding" (SODA 2007).
- Jégou, Douze & Schmid, "Product Quantization for Nearest Neighbor Search" (IEEE TPAMI 2011), §3.3. Recommends eigenvalue-based rotation (OPQ) for further improvement.
- Ge et al., "Optimized Product Quantization" (IEEE TPAMI 2014). OPQ minimizes quantization error by rotating subspaces to balance variance.

---

## RFC-002-10: ODE Stiffness Detection

**Status**: Deferred
**Severity**: LOW
**Effort**: High

### Current Behavior

Dormand-Prince (explicit RK45) is the only ODE solver. Stiff systems cause it to take tiny steps or diverge.

### When This Matters

Stiffness arises when embedding dynamics have:
- Rapid oscillations (e.g., periodic social media content)
- Sharp regime transitions (the primary CVX use case)
- Large eigenvalue spread in the Jacobian

### Proposed Approach

1. **Detect stiffness**: If step size drops below `h_min` and error remains high, flag as stiff.
2. **Fallback**: Switch to an implicit solver (e.g., SDIRK or Radau IIA).
3. **Practical alternative**: For CVX, linear extrapolation may be sufficient for stiff trajectories since the Neural ODE is not yet implemented.

### Decision

**Deferred** — the linear extrapolation fallback handles the immediate need. Revisit when Neural ODE training is implemented (Layer 10 completion with burn backend).

### References

- Hairer & Wanner, "Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems" (Springer, 2002), Chapter IV.
- Chen et al., "Neural Ordinary Differential Equations" (NeurIPS 2018). Uses adaptive solvers with adjoint method; stiffness is handled by the solver choice.
- Kidger, "On Neural Differential Equations" (PhD Thesis, Oxford, 2022). Comprehensive treatment of Neural ODE solver selection including stiffness.

---

## Implementation Priority

### Phase 1: Critical Fixes (1-2 days)

| RFC | Issue | Lines to Change |
|-----|-------|----------------|
| 002-01 | WAL fsync | ~10 |
| 002-02 | NaN-safe sort | ~6 |
| 002-07 | Level bounding | ~2 |

### Phase 2: Correctness (1 week)

| RFC | Issue | Scope |
|-----|-------|-------|
| 002-03 | Heuristic neighbor | optimized.rs |
| 002-06 | PELT pruning | pelt.rs |

### Phase 3: Architecture (2-3 weeks)

| RFC | Issue | Scope |
|-----|-------|-------|
| 002-04 | Concurrent insert | concurrent.rs + new module |
| 002-05 | gRPC semantics | grpc/service.rs |
| 002-08 | Warm zone maps | warm/mod.rs |
| 002-09 | PQ initialization | cold/mod.rs |

### Phase 4: Deferred

| RFC | Issue | Trigger |
|-----|-------|---------|
| 002-10 | ODE stiffness | When burn backend is integrated |

---

## Full Reference List

1. Arthur, D. & Vassilvitskii, S. "k-means++: The Advantages of Careful Seeding." SODA 2007.
2. Chen, R.T.Q. et al. "Neural Ordinary Differential Equations." NeurIPS 2018.
3. Fu, C. et al. "Fast Approximate Nearest Neighbor Search with the Navigating Spreading-out Graph." VLDB 2019.
4. Ge, T. et al. "Optimized Product Quantization." IEEE TPAMI 2014.
5. Hairer, E. & Wanner, G. "Solving Ordinary Differential Equations II." Springer, 2002.
6. IEEE 754-2019. "Standard for Floating-Point Arithmetic." IEEE, 2019.
7. Jégou, H., Douze, M. & Schmid, C. "Product Quantization for Nearest Neighbor Search." IEEE TPAMI 2011.
8. Kidger, P. "On Neural Differential Equations." PhD Thesis, Oxford, 2022.
9. Killick, R., Fearnhead, P. & Eckley, I.A. "Optimal Detection of Changepoints with a Linear Computational Cost." JASA 2012.
10. Kreps, J., Narkhede, N. & Rao, J. "Kafka: A Distributed Messaging System for Log Processing." NetDB 2011.
11. Lamb, A. et al. "The Vertica Analytic Database: C-Store 7 Years Later." VLDB 2012.
12. Maidstone, R. et al. "On Optimal Multiple Changepoint Algorithms for Large Data." Statistics & Computing 2017.
13. Malkov, Y.A. & Yashunin, D.A. "Efficient and Robust Approximate Nearest Neighbor Using Hierarchical Navigable Small World Graphs." IEEE TPAMI 2018.
14. Pillai, T.S. et al. "All File Systems Are Not Created Equal: On the Complexity of Crafting Crash-Consistent Applications." OSDI 2014.
15. Singh, A. et al. "FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search." arXiv 2023.
16. Subramanya, S.J. et al. "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node." NeurIPS 2019.
17. Zheng, Y. et al. "BtrBlk: Efficient B-tree Logging for Database Storage on SSD." SIGMOD 2022.
