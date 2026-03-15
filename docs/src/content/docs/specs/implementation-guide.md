---
title: "Implementation Decisions"
description: "Key technical choices: concurrency, SIMD, serialization, storage, and testing strategy"
---

This page summarizes the 10 Implementation Decision Records (IDRs) that define *how* ChronosVector is built — the tools, patterns, and trade-offs at the code level. Each decision is presented with the choice made, the reasoning, and the key code pattern a contributor should know.

---

## IDR-001: Concurrency Model

**Choice:** Message passing (tokio channels) for data flow, `parking_lot::RwLock` for concurrent index access.

**Why:** CVX has two fundamentally different concurrency patterns. The ingestion pipeline is a linear flow of data through stages — channels provide natural backpressure without explicit rate limiting. The HNSW index, on the other hand, is read-heavy: dozens of concurrent searches need simultaneous access, with occasional writes for insertions. An actor model would serialize all searches through a single thread; `RwLock` allows N readers concurrently.

**Key pattern:** The ingestion pipeline is a chain of bounded `mpsc` channels, where each stage runs as an independent tokio task. Backpressure is automatic — if a downstream stage is slow, its channel fills and the upstream stage waits.

```
receive → validate → delta_encode → WAL → index_insert → store → ack
   tx→rx     tx→rx      tx→rx      tx→rx     tx→rx      tx→rx
```

**The rule:** *"If the data flows, use channels. If the data is queried concurrently, use RwLock."*

---

## IDR-002: Compute Parallelism

**Choice:** Rayon thread pool for all CPU-bound work, bridged from Tokio via `spawn_blocking`.

**Why:** CPU-bound work (SIMD distance computation, HNSW graph traversal, PELT, delta encoding) must never block Tokio's async runtime, or all I/O tasks on that thread stall. Rayon provides work-stealing parallelism *within* a single task — `par_iter()` distributes thousands of distance computations across all cores automatically.

**Key pattern:** The Tokio-to-Rayon bridge:

```rust
pub async fn search_index(
    index: Arc<RwLock<HnswGraph>>,
    query: QueryVector,
    k: usize,
) -> Result<Vec<SearchResult>, CvxError> {
    task::spawn_blocking(move || {
        let graph = index.read();
        graph.search(&query, k) // uses par_iter internally
    }).await?
}
```

Inside `graph.search()`, Rayon's `par_iter` distributes distance computations across cores with automatic granularity control.

---

## IDR-003: Serialization

**Choice:** rkyv for the HNSW graph (zero-copy, mmap-ready), postcard for everything else (compact, schema-evolvable).

**Why:** The HNSW graph can be 84 MB to 8.4 GB. Zero-copy deserialization via rkyv enables instant startup by memory-mapping the graph file. For RocksDB values, WAL entries, and deltas, postcard's variable-length integer encoding saves space (a `u32` with value 42 is 1 byte, not 4), and `#[serde(default)]` enables forward-compatible schema evolution.

**Key pattern:**

| Data | Format | Reason |
|------|--------|--------|
| HNSW graph | rkyv | Zero-copy via mmap, instant startup |
| Vectors in RocksDB | postcard | RocksDB copies data internally, zero-copy adds no value |
| Deltas in RocksDB | postcard | Variable-length encoding, efficient for sparse data |
| WAL entries | postcard | Schema evolution via `serde(default)` |
| Config/metadata | TOML/JSON | Human readable and editable |
| Network (gRPC) | protobuf | Interoperability |
| Network (REST) | JSON | Standard |

**Trade-off:** rkyv does *not* support schema evolution. Changing the HNSW struct requires re-indexing. This is acceptable because the source data lives in RocksDB and the index can always be rebuilt.

---

## IDR-004: Global Allocator

**Choice:** jemalloc via `tikv-jemallocator`, set only in `cvx-server` (the binary crate).

**Why:** The system allocator uses a global lock or limited arenas. Under concurrent load (many Rayon threads and Tokio tasks allocating simultaneously), this creates contention. jemalloc uses per-thread arenas, eliminating cross-thread contention. In Qdrant's benchmarks, switching to jemalloc improved search throughput by ~15% under high concurrency.

**Key pattern:** A single line in `cvx-server/src/main.rs`:

```rust
#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
```

Library crates are unaffected — the allocator is transparent to them.

---

## IDR-005: SIMD Strategy

**Choice:** `pulp` as the primary SIMD abstraction.

**Why:** Distance computation is CVX's hot path — each HNSW search computes hundreds to thousands of distances. pulp provides safe, cross-platform SIMD with automatic runtime dispatch (AVX-512, AVX2, NEON, scalar fallback) from a single code path. It runs on stable Rust (no nightly required) and is the SIMD engine behind `faer`, the fastest linear algebra library in Rust.

**Key pattern:** Implement `pulp::WithSimd` for each kernel, then dispatch:

```rust
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    pulp::Arch::new().dispatch(DotProduct { a, b })
}
```

The same code compiles to AVX-512 on Zen 4/Sapphire Rapids, AVX2 on most x86_64 servers, and NEON on Apple Silicon. If profiling shows pulp leaves performance on the table for a specific function, hand-written intrinsics can be used as a targeted fallback.

---

## IDR-006: RocksDB Key Encoding

**Choice:** Big-endian keys with sign-bit flip for `i64` timestamps, separate column families per data type.

**Why:** RocksDB keys are sorted lexicographically. Big-endian encoding makes byte comparison match numeric comparison. The sign-bit flip (XOR first byte with `0x80`) converts two's complement ordering to correct lexicographic ordering for signed integers, enabling range scans that work correctly with negative timestamps (pre-1970 historical data).

**Key format:** `entity_id (8B BE) + space_id (4B BE) + timestamp (8B BE with sign flip)` = 20 bytes fixed.

**Column families** isolate data types with independent tuning:

| Column Family | Compression | Bloom Filter | Block Cache |
|---------------|-------------|--------------|-------------|
| `vectors` | None | Prefix (12B) | 256 MB |
| `deltas` | LZ4 | Prefix (12B) | 64 MB |
| `timelines` | LZ4 | Full | 32 MB |
| `metadata` | LZ4 | Full | 32 MB |
| `changepoints` | Zstd | None | 16 MB |
| `views` | Zstd | None | 16 MB |
| `system` | None | None | 8 MB |

The 12-byte prefix bloom filter covers `entity_id + space_id`, allowing RocksDB to skip entire SST files that do not contain the queried entity.

---

## IDR-007: Error Handling

**Choice:** `thiserror` in all library crates, `anyhow` only in `cvx-server`.

**Why:** Library crates need structured errors so the API layer can pattern-match to HTTP status codes (400 for dimension mismatch, 404 for entity not found, 500 for internal errors). The binary crate needs ergonomic error propagation with `.context()` for startup/shutdown diagnostics, where pattern matching is unnecessary.

**Key pattern:** Each subsystem defines its own error enum with `thiserror`. A root `CvxError` enum wraps them all via `#[from]`. Internal errors are *never* exposed to the API consumer — they are logged with `tracing::error!` and return a generic 500.

```
cvx-core:      CvxError (wraps all subsystem errors)
cvx-index:     IndexError
cvx-storage:   StorageError
cvx-analytics: AnalyticsError
cvx-server:    anyhow::Result (startup/shutdown only)
```

---

## IDR-008: Testing Strategy

**Choice:** 5 levels of testing, with property-based testing as the core strategy for mathematical invariants.

**Why:** CVX has diverse code — from SIMD arithmetic (where off-by-one corrupts results) to HTTP APIs (where ergonomics matter more than numerical precision). Each type needs appropriate testing.

| Level | Tool | What it tests | CI frequency |
|-------|------|--------------|-------------|
| Unit | `#[test]` | Pure logic, specific examples | Every push |
| Property | `proptest` | Mathematical invariants for *any* input | Every push (subset), nightly (full) |
| Integration | `#[test]` + tempdir | Full flow with real RocksDB | Every push |
| E2E | `TestServer` | HTTP/gRPC stack end-to-end | PRs + nightly |
| Benchmarks | `criterion` | Performance regression detection | PRs touching perf-sensitive code |

**Key property tests:** distance metric symmetry/non-negativity/triangle inequality, key encoding roundtrip and ordering preservation, delta encode/decode roundtrip, HNSW graph connectivity (all nodes reachable from entry point), serialization roundtrips.

---

## IDR-009: Unsafe Policy

**Choice:** `#![deny(unsafe_code)]` by default, `#![allow(unsafe_code)]` in only 2 crates: `cvx-index` and `cvx-storage`.

**Why:** Rust's safety guarantees are a key value proposition. 7 of 9 crates are 100% safe Rust. The two exceptions need `unsafe` for SIMD intrinsic fallbacks (if pulp falls short on a specific function), rkyv zero-copy access, and `memmap2` file mapping.

**Rules:**
1. Every new crate starts with `deny(unsafe_code)`
2. If another crate needs `unsafe`, evaluate moving the functionality to `cvx-index` or `cvx-storage` first
3. Every `unsafe` block requires a `// SAFETY:` comment documenting the invariants the programmer guarantees
4. All `unsafe` is encapsulated in safe public wrappers — consumers never see `unsafe`
5. CI runs Miri on pure-Rust crates to detect undefined behavior

---

## IDR-010: Index Persistence

**Choice:** Progressive strategy — read-into-memory now, mmap with background prefetch later.

**Why:** For $\leq 1$M vectors (~84 MB graph), reading the full graph into memory at startup takes ~200ms — perfectly acceptable. For $\geq 10$M vectors (~840 MB+), startup delay becomes problematic. The progressive strategy defers the complex mmap implementation until the simpler approach hits its limits.

| Phase | When | Strategy | Startup time (1M vectors) |
|-------|------|----------|--------------------------|
| Phase 1 | Layers 2-4 | `rkyv::deserialize` into memory | ~200ms |
| Phase 2 | Layer 11+ | mmap + background prefetch | Instant (queries served during prefetch) |

In Phase 2, the server memory-maps the graph file and immediately starts serving queries. A background thread touches every 4KB page with `madvise(SEQUENTIAL)` to warm the OS page cache. Early queries may incur page faults (slightly higher latency), but the server is available immediately.

The rest of the codebase accesses the index through an `IndexStorage` enum that abstracts the difference between in-memory and mmap'd backends.

---

## Dependency Summary

All dependencies chosen across the IDRs and architecture decisions:

```toml
# Concurrency & Async
tokio = { version = "1", features = ["full"] }
rayon = "1"
parking_lot = "0.12"

# Serialization
rkyv = { version = "0.8", features = ["validation"] }
postcard = { version = "1", features = ["alloc"] }
serde = { version = "1", features = ["derive"] }

# SIMD
pulp = { version = "0.22", features = ["macro"] }

# Collections
smallvec = { version = "1", features = ["union"] }

# Allocator (cvx-server only)
tikv-jemallocator = "0.6"

# Error Handling
thiserror = "2"
anyhow = "1"  # cvx-server only

# Testing
proptest = "1"
criterion = { version = "0.5", features = ["html_reports"] }

# Observability
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Storage (added progressively)
# rocksdb = "0.22"   (Layer 3)
# memmap2 = "0.9"    (Layer 11)
```
