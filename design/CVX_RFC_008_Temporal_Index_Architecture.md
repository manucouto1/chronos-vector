# RFC-008: Advanced Temporal Index Architecture

**Status**: Proposed
**Created**: 2026-03-18
**Authors**: Manuel Couto Pintos
**Related**: RFC-001 (Architecture Decisions, ADR-002/003), RFC-002 (Performance), RFC-005 (Query Capabilities)

---

## Summary

This RFC proposes three architectural improvements to CVX's indexing layer that address scalability, query latency, and streaming workloads. These improvements transform CVX from a "temporal vector index" into a **temporally-aware storage engine** — a distinction that no existing vector database makes.

| Improvement | What it solves |
|-------------|---------------|
| **Temporal Locality-Sensitive Hashing (T-LSH)** | Eliminates the over-fetch×4 penalty for composite distance queries (α < 1.0) |
| **Time-Partitioned Sharding** | Enables sub-linear query latency on historical data by pruning entire time partitions |
| **Streaming Window Index** | Maintains a hot index for real-time queries while compacting historical data |

---

## Motivation

### Current Bottlenecks

**1. Over-fetch penalty for composite distance (α < 1.0)**

When `alpha < 1.0` (temporal component active), `TemporalHnsw::search()` must over-fetch 4× candidates semantically, then re-rank with the composite distance (`temporal.rs:174-210`). This is because the HNSW graph is organized by **semantic distance only** — temporally relevant but semantically distant nodes are never explored.

```
Current flow (alpha=0.5, k=10):
  1. Semantic kNN with k=40 (4× over-fetch)    → 40 candidates
  2. Compute temporal distance for each          → 40 × O(1)
  3. Re-rank by composite distance               → sort 40
  4. Return top 10                               → truncate

Problem: If the best temporal match is semantically at rank 100,
         it will never be found (over-fetch only gets top 40).
```

**2. Linear scan for temporal filtering**

`build_filter_bitmap()` (`temporal.rs:137-146`) iterates ALL timestamps to build a Roaring Bitmap:

```rust
for (i, &ts) in self.timestamps.iter().enumerate() {
    if filter.matches(ts) {
        bitmap.insert(i as u32);
    }
}
```

For 10M points, this is a 10M-iteration loop on every temporally-filtered query. The bitmap itself is efficient (< 1 byte/point), but **building** it is O(N).

**3. No partition pruning**

A query for "similar vectors in the last 24 hours" still traverses the full HNSW graph (built over ALL timestamps). In time-series databases (TimescaleDB, InfluxDB), queries on recent data only touch recent partitions. CVX has no equivalent.

### Why This Matters

| Scenario | Current | Target |
|----------|---------|--------|
| 10M points, temporal query (last day) | Filter bitmap: 10M iterations + full HNSW traversal | Partition prune → touch only ~10K points |
| Composite distance (α=0.5) | 4× over-fetch, misses temporally-close/semantically-far | T-LSH: native spatiotemporal candidate generation |
| Streaming ingestion (1K pts/sec) | Rebuild index or online insert with degradation | Hot index absorbs writes, periodic compaction to cold |

---

## Proposed Changes

### 1. Temporal Locality-Sensitive Hashing (T-LSH) (Priority: P1)

#### Problem

The fundamental issue with composite distance in HNSW is that the graph is built with semantic distance. The greedy search on this graph has **no mechanism** to prefer temporally-close nodes during traversal. The over-fetch is a heuristic workaround.

#### Solution: Dual-Hash LSH

Build an auxiliary LSH index where hash functions incorporate both semantic and temporal components. This provides approximate nearest neighbors under the composite distance `d_ST` without relying on the semantic HNSW graph.

**Hash function design:**

For a point (vector v, timestamp t):

```
h_ST(v, t) = h_sem(v) ⊕ h_time(t)

Where:
  h_sem(v) = sign(r · v)               // random hyperplane LSH for cosine
  h_time(t) = floor(t / bucket_size)    // temporal bucketing

Combined hash: concatenate b_sem semantic bits + b_time temporal bits
```

The number of semantic vs. temporal bits controls the balance:
- More semantic bits → behaves like standard LSH (ignores time)
- More temporal bits → biases toward temporal locality
- **The ratio should match α**: for α=0.5, use equal bits; for α=0.8, use 4:1 semantic:temporal

**Multi-probe strategy:**

```
1. Compute primary hash h_ST(query_vector, query_time)
2. Probe primary bucket → collect candidates
3. Probe neighboring temporal buckets (t ± 1, t ± 2) → more candidates
4. Probe neighboring semantic buckets (flip 1 bit) → more candidates
5. Score all candidates with exact composite distance
6. Return top-k
```

#### Architecture

```
┌─────────────────────────────────────────────┐
│                TemporalHnsw                 │
│  ┌─────────────┐    ┌────────────────────┐  │
│  │  HnswGraph  │    │   T-LSH Index      │  │
│  │  (semantic)  │    │  (spatiotemporal)  │  │
│  └──────┬──────┘    └────────┬───────────┘  │
│         │                     │              │
│         ▼                     ▼              │
│   α ≥ 1.0: use HNSW    α < 1.0: use T-LSH  │
│   (pure semantic)       (composite distance) │
└─────────────────────────────────────────────┘
```

When α = 1.0 (pure semantic), the HNSW graph is optimal. When α < 1.0, T-LSH provides candidates that are naturally distributed in both semantic and temporal space.

#### Data Structures

```rust
pub struct TemporalLSH {
    /// Hash tables: each table uses a different random projection family
    tables: Vec<HashMap<u64, Vec<u32>>>,  // hash → node_ids
    /// Random hyperplanes for semantic hashing
    semantic_planes: Vec<Vec<Vec<f32>>>,  // [table][bit][dim]
    /// Temporal bucket size in microseconds
    temporal_bucket_us: i64,
    /// Bits allocated to semantic vs temporal
    semantic_bits: usize,
    temporal_bits: usize,
    /// Number of hash tables (more tables = higher recall, more memory)
    n_tables: usize,
}

impl TemporalLSH {
    /// Build from existing index data
    pub fn build(
        vectors: &[Vec<f32>],
        timestamps: &[i64],
        config: TLSHConfig,
    ) -> Self;

    /// Query: find candidates under composite distance
    pub fn query(
        &self,
        vector: &[f32],
        timestamp: i64,
        n_probes: usize,
    ) -> Vec<u32>;  // candidate node_ids

    /// Incremental insert
    pub fn insert(&mut self, node_id: u32, vector: &[f32], timestamp: i64);
}
```

#### Configuration

```rust
pub struct TLSHConfig {
    pub n_tables: usize,          // default: 16
    pub semantic_bits: usize,     // default: 12
    pub temporal_bits: usize,     // default: 4 (adjustable via alpha)
    pub temporal_bucket_us: i64,  // default: 86_400_000_000 (1 day)
    pub n_probes: usize,          // default: 3 (multi-probe depth)
}
```

#### Expected Improvement

| Metric | Current (over-fetch) | T-LSH |
|--------|---------------------|-------|
| Candidate quality | Semantic top-40, re-ranked | Spatiotemporal candidates |
| Recall@10 (α=0.5) | ~70% (limited by 4× over-fetch) | ~90% (multi-probe) |
| Latency (10M points) | ~5ms (HNSW) + re-rank | ~3ms (hash + probe + score) |
| Memory overhead | None | ~2 bytes/point × n_tables |

#### References

- Indyk, P. & Motwani, R. (1998). Approximate nearest neighbors: towards removing the curse of dimensionality. *STOC*.
- Andoni, A. & Indyk, P. (2006). Near-optimal hashing algorithms for approximate nearest neighbor. *FOCS*.
- Lv, Q. et al. (2007). Multi-probe LSH: efficient indexing for high-dimensional similarity search. *VLDB*.
- Zheng, B. et al. (2020). PM-LSH: a fast and accurate LSH framework for high-dimensional approximate NN search. *VLDB*.

---

### 2. Time-Partitioned Sharding (Priority: P0)

#### Problem

CVX stores all points (regardless of timestamp) in a single HNSW graph. A query for "last 24 hours" traverses the same graph as "last 10 years." Time-series databases solved this decades ago with time-partitioned storage.

#### Solution: Temporal Partitions

Divide the timeline into fixed-duration partitions. Each partition contains its own HNSW graph, covering only points within that time range.

```
Timeline: ─────────────────────────────────────────────────────►
           │  Partition 0  │  Partition 1  │  Partition 2  │ ...
           │  [t0, t0+Δ)  │  [t0+Δ, t0+2Δ) │  [t0+2Δ, t0+3Δ) │
           │  HNSW_0       │  HNSW_1        │  HNSW_2          │
           │  8K points    │  12K points    │  3K points       │
```

**Query routing:**

```
1. Parse temporal filter → determine which partitions overlap
2. Query only overlapping partitions (in parallel)
3. Merge results across partitions by composite distance
4. Return top-k global
```

#### Architecture

```rust
pub struct PartitionedTemporalHnsw<D: DistanceMetric> {
    /// Sorted partitions by start time
    partitions: Vec<Partition<D>>,
    /// Partition duration in microseconds
    partition_duration_us: i64,
    /// Global entity index: entity_id → vec of (partition_idx, local_node_id)
    global_entity_index: BTreeMap<u64, Vec<(usize, u32)>>,
    /// Active partition (receives new inserts)
    active_partition: usize,
}

pub struct Partition<D: DistanceMetric> {
    pub start_us: i64,
    pub end_us: i64,
    pub hnsw: TemporalHnsw<D>,
    pub point_count: usize,
    /// Whether this partition is in memory or on disk
    pub state: PartitionState,
}

pub enum PartitionState {
    Hot,      // fully in memory, writable
    Warm,     // in memory, read-only (compacted)
    Cold,     // on disk, loaded on demand
}
```

**Partition lifecycle:**

```
Insert → Active (Hot) partition
         ↓ (when partition time range ends)
Compact → Warm partition (read-only, optimized HNSW)
          ↓ (when memory pressure or age threshold)
Evict → Cold partition (serialized to disk, mmapped on query)
```

#### Insert Flow

```rust
pub fn insert(&mut self, entity_id: u64, timestamp: i64, vector: &[f32]) -> u32 {
    let partition_idx = self.partition_for_timestamp(timestamp);

    // If timestamp is in the future beyond active partition, create new partition(s)
    while partition_idx >= self.partitions.len() {
        self.create_partition();
    }

    let local_id = self.partitions[partition_idx].hnsw.insert(entity_id, timestamp, vector);

    // Update global entity index
    self.global_entity_index
        .entry(entity_id)
        .or_default()
        .push((partition_idx, local_id));

    self.encode_global_id(partition_idx, local_id)
}
```

#### Query Flow

```rust
pub fn search(
    &self,
    query: &[f32],
    k: usize,
    alpha: f32,
    filter: TemporalFilter,
) -> Vec<ScoredResult> {
    // 1. Determine which partitions overlap with the filter
    let relevant: Vec<&Partition<D>> = self.partitions.iter()
        .filter(|p| p.overlaps(&filter))
        .collect();

    // 2. Query each partition (parallelizable via rayon)
    let mut candidates: Vec<ScoredResult> = relevant
        .par_iter()
        .flat_map(|p| p.hnsw.search(query, k, alpha, filter.clone()))
        .collect();

    // 3. Merge and return global top-k
    candidates.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
    candidates.truncate(k);
    candidates
}
```

#### Trajectory Queries

The `global_entity_index` enables efficient cross-partition trajectory retrieval:

```rust
pub fn trajectory(&self, entity_id: u64, filter: TemporalFilter) -> Vec<(i64, u32)> {
    let entries = self.global_entity_index.get(&entity_id)?;
    entries.iter()
        .filter(|(part_idx, _)| self.partitions[*part_idx].overlaps(&filter))
        .flat_map(|(part_idx, local_id)| {
            // Retrieve timestamp from partition
            let ts = self.partitions[*part_idx].hnsw.timestamp_of(*local_id);
            if filter.matches(ts) { Some((ts, self.encode_global_id(*part_idx, *local_id))) } else { None }
        })
        .sorted_by_key(|(ts, _)| *ts)
        .collect()
}
```

#### Configuration

```rust
pub struct PartitionConfig {
    pub partition_duration_us: i64,  // default: 7 days (604_800_000_000 µs)
    pub max_hot_partitions: usize,   // default: 4 (last 4 weeks in memory)
    pub max_warm_partitions: usize,  // default: 12 (3 months read-only)
    pub compaction_threshold: usize, // compact when this many inserts since last compact
    pub cold_storage_path: PathBuf,
}
```

**Default partition duration guidelines:**

| Use case | Partition duration | Rationale |
|----------|-------------------|-----------|
| Social media monitoring | 7 days | Weekly posting patterns |
| Financial markets | 1 day | Trading day boundaries |
| IoT sensors | 1 hour | High-frequency data |
| Clinical longitudinal | 30 days | Monthly assessment cycles |

#### Expected Improvement

| Metric | Current (single index) | Partitioned |
|--------|----------------------|-------------|
| "Last 24h" query on 1-year data | Scan 10M timestamps, full HNSW | Touch 1 partition (~30K points) |
| Filter bitmap build | O(N) always | O(N/P) per partition, skip P-1 partitions |
| Memory for 10M points | All in HNSW | Hot: 2M, Warm: 2M mmapped, Cold: 6M on disk |
| Insert latency (active partition) | O(log N) with N=10M | O(log N/P) with N/P ≈ 30K |

#### References

- TimescaleDB hypertable architecture: automatic time-based partitioning (Timescale, 2017)
- InfluxDB TSI (Time-Structured Merge Tree): time-aware indexing (InfluxData, 2017)
- DiskANN: Fresh index serving with tiered storage (Subramanya et al., 2019)
- Milvus time-travel: partition-based temporal queries (Zilliz, 2022)

---

### 3. Streaming Window Index (Priority: P2)

#### Problem

Many CVX use cases are **streaming**: social media posts, market ticks, sensor readings arrive continuously. The index must simultaneously:
1. Absorb high-throughput writes without blocking readers
2. Serve low-latency queries over recent data
3. Maintain long-term historical data for trajectory analytics

The current `ConcurrentTemporalHnsw` (`concurrent.rs`) uses a single RwLock, which works for moderate write rates but degrades under sustained high-throughput ingestion.

#### Solution: Write-Ahead Buffer + Compaction Pipeline

```
Write path:
  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
  │  Write-Ahead │ ──→ │   Hot Buffer  │ ──→ │  Compacted HNSW  │
  │  Log (WAL)   │     │ (flat index)  │     │  (partitioned)   │
  └──────────────┘     └──────────────┘     └──────────────────┘
    durability          low-latency           high-quality
    append-only         brute-force scan      graph navigation
    O(1) write          O(N_hot) search       O(log N) search

Read path:
  Query → [search Hot Buffer] ∪ [search Compacted HNSW] → merge → top-k
```

#### Components

**Hot Buffer**: A flat (non-graph) index for recent points. Queries scan all points in the buffer via brute-force. This is fast because the buffer is small (bounded by compaction interval).

```rust
pub struct HotBuffer {
    /// Points not yet in the HNSW graph
    points: Vec<BufferedPoint>,
    /// Maximum size before triggering compaction
    capacity: usize,
    /// RwLock-free concurrent access via epoch-based reclamation
    epoch: AtomicU64,
}

pub struct BufferedPoint {
    pub entity_id: u64,
    pub timestamp: i64,
    pub vector: Vec<f32>,
    pub node_id: u32,  // pre-assigned for global consistency
}
```

**Compaction**: Periodically (or when buffer reaches capacity), flush the buffer into the active HNSW partition:

```rust
impl StreamingIndex {
    pub fn compact(&mut self) {
        // 1. Snapshot the hot buffer
        let points = self.hot_buffer.drain();

        // 2. Bulk-insert into active partition
        for point in points {
            self.partitioned_index.insert(
                point.entity_id,
                point.timestamp,
                &point.vector,
            );
        }

        // 3. Update metadata
        self.last_compaction = Instant::now();
    }
}
```

**Write-Ahead Log (WAL)**: For durability, writes go to the WAL before the hot buffer. On crash recovery, replay the WAL to reconstruct the buffer.

```rust
pub struct WriteAheadLog {
    pub path: PathBuf,
    pub writer: BufWriter<File>,
    pub sequence: u64,
}

impl WriteAheadLog {
    pub fn append(&mut self, point: &BufferedPoint) -> io::Result<u64>;
    pub fn replay(&self) -> io::Result<Vec<BufferedPoint>>;
    pub fn truncate_before(&mut self, sequence: u64) -> io::Result<()>;
}
```

#### Query Integration

```rust
pub struct StreamingTemporalHnsw<D: DistanceMetric> {
    hot_buffer: HotBuffer,
    partitioned_index: PartitionedTemporalHnsw<D>,
    wal: WriteAheadLog,
    compaction_config: CompactionConfig,
}

impl<D: DistanceMetric> StreamingTemporalHnsw<D> {
    pub fn insert(&mut self, entity_id: u64, timestamp: i64, vector: &[f32]) -> u32 {
        // 1. Write to WAL (durability)
        self.wal.append(&BufferedPoint { entity_id, timestamp, vector: vector.to_vec(), node_id });

        // 2. Insert into hot buffer (visibility)
        self.hot_buffer.push(entity_id, timestamp, vector);

        // 3. Trigger compaction if buffer full
        if self.hot_buffer.len() >= self.compaction_config.buffer_capacity {
            self.compact();
        }

        node_id
    }

    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        alpha: f32,
        filter: TemporalFilter,
    ) -> Vec<ScoredResult> {
        // Search both hot buffer and compacted index
        let mut results = self.partitioned_index.search(query, k, alpha, filter.clone());
        let buffer_results = self.hot_buffer.brute_force_search(query, k, alpha, &filter);

        // Merge
        results.extend(buffer_results);
        results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        results.truncate(k);
        results
    }
}
```

#### Configuration

```rust
pub struct CompactionConfig {
    pub buffer_capacity: usize,      // default: 10_000 points
    pub compaction_interval_secs: u64, // default: 60 (1 minute)
    pub wal_sync_mode: WalSyncMode,   // Fsync | Periodic(Duration) | None
}

pub enum WalSyncMode {
    Fsync,                    // fsync after every write (safest, slowest)
    Periodic(Duration),       // fsync every N seconds (balanced)
    None,                     // OS-level buffering only (fastest, risk of data loss)
}
```

#### Expected Improvement

| Metric | Current (ConcurrentTemporalHnsw) | Streaming |
|--------|----------------------------------|-----------|
| Write throughput | ~800 pts/sec (SQ8+ef=25, write lock) | ~50K pts/sec (WAL + buffer, no lock) |
| Write-to-visibility latency | Immediate (after lock) | Immediate (buffer is searchable) |
| Query latency (recent data) | Same as full index | Buffer scan (~0.1ms for 10K points) + partition search |
| Crash recovery | Rebuild from storage | WAL replay (~seconds) |

#### References

- LSM-tree: O'Neil et al. (1996). The log-structured merge-tree. *Acta Informatica*, 33(4).
- RocksDB compaction strategies (Facebook Engineering, 2013).
- DiskANN FreshDiskANN: streaming updates with tiered index (Singh et al., 2021).
- Milvus growing/sealed segment architecture (Zilliz, 2022).

---

## Implementation Plan

### Phase 1: `feat/time-partitions` (P0)

| Task | Crate | Details |
|------|-------|---------|
| `Partition<D>` struct | cvx-index | HNSW + time range + state |
| `PartitionedTemporalHnsw<D>` | cvx-index | Insert routing, query fan-out, partition lifecycle |
| `global_entity_index` | cvx-index | Cross-partition entity lookup |
| Partition pruning in search | cvx-index | Skip partitions that don't overlap temporal filter |
| Parallel partition query (rayon) | cvx-index | Fan-out + merge |
| Cold partition serialization | cvx-index | Reuse existing `save()`/`load()` per partition |
| Tests: multi-partition insert/search/trajectory | cvx-index | Verify correctness equals single-index |
| Migration: single→partitioned index | cvx-index | Load existing index into one partition |

### Phase 2: `feat/temporal-lsh` (P1)

| Task | Crate | Details |
|------|-------|---------|
| `TemporalLSH` struct | cvx-index | Hash tables + random projections |
| Build from existing data | cvx-index | Scan vectors + timestamps, hash into tables |
| Multi-probe query | cvx-index | Primary + neighboring buckets |
| Integration with `PartitionedTemporalHnsw` | cvx-index | Route α < 1.0 queries to T-LSH |
| Incremental insert | cvx-index | Hash new point into all tables |
| Benchmark: recall@10 vs over-fetch baseline | cvx-index | Vary n_tables, n_probes |
| Tests: T-LSH recall ≥ over-fetch recall at same latency | cvx-index | — |

### Phase 3: `feat/streaming-index` (P2)

| Task | Crate | Details |
|------|-------|---------|
| `HotBuffer` with brute-force search | cvx-index | Lock-free append + scan |
| `WriteAheadLog` (append + replay + truncate) | cvx-index | CRC32 validation per entry |
| `StreamingTemporalHnsw<D>` | cvx-index | Unified insert/search over buffer + partitions |
| Compaction pipeline | cvx-index | Drain buffer → bulk insert into partition |
| Crash recovery test | cvx-index | Kill mid-write, verify WAL replay correctness |
| Throughput benchmark | cvx-index | Sustained write rate with concurrent readers |

---

## Impact on Existing Code

| Component | Change |
|-----------|--------|
| `cvx-index` | Major: new `partitioned.rs`, `temporal_lsh.rs`, `streaming.rs` modules |
| `cvx-index/hnsw/temporal.rs` | Minor: add `timestamp_of(node_id)` and `vector_of(node_id)` accessors |
| `cvx-core/traits` | Extend `TemporalIndexAccess` to support partitioned index transparently |
| `cvx-api` | Transparent — handler code uses trait, doesn't change |
| `cvx-python` | Transparent — wraps trait implementation |

**Migration path**: Existing `TemporalHnsw` becomes a single-partition `PartitionedTemporalHnsw`. All tests pass without modification. Partitioning is opt-in via configuration.

---

## Verification

### Correctness Invariants

1. **Partition transparency**: `PartitionedTemporalHnsw::search()` returns identical results to `TemporalHnsw::search()` for the same data (single partition = single index).
2. **Trajectory completeness**: `trajectory(entity_id)` across partitions returns the same points as single-index trajectory.
3. **Insert ordering**: Points inserted out of temporal order are routed to the correct partition.
4. **WAL consistency**: After crash + replay, the index contains exactly the points that were WAL-committed.

### Benchmark Targets

| Operation | Single Index (current) | Partitioned (target) |
|-----------|----------------------|---------------------|
| Insert (10M pts) | ~3.5 hours (800 pts/sec) | Same (per-partition unchanged) |
| Search, last 24h (10M total, 7-day partitions) | ~5ms (full graph) | ~1ms (1 partition) |
| Search, last year (10M total) | ~5ms | ~5ms (52 partitions, parallel) |
| Filter bitmap build | 10M iterations | ~30K iterations (1 partition) |
| Trajectory retrieval | O(log N) | O(log N) via global_entity_index |

---

## References

1. Indyk, P. & Motwani, R. (1998). Approximate nearest neighbors. *STOC*.
2. Andoni, A. & Indyk, P. (2006). Near-optimal hashing. *FOCS*.
3. Lv, Q. et al. (2007). Multi-probe LSH. *VLDB*.
4. Zheng, B. et al. (2020). PM-LSH. *VLDB*.
5. O'Neil, P. et al. (1996). Log-structured merge-tree. *Acta Informatica*, 33(4).
6. Subramanya, S. J. et al. (2019). DiskANN: fast accurate billion-point NN search. *NeurIPS*.
7. Singh, A. et al. (2021). FreshDiskANN. *arXiv:2105.09613*.
8. Timescale (2017). TimescaleDB: an adaptive system for time-series storage.
9. Milvus architecture: growing/sealed segments (Zilliz, 2022).
