# ChronosVector (CVX)

[![CI](https://github.com/manucouto1/chronos-vector/actions/workflows/ci.yml/badge.svg)](https://github.com/manucouto1/chronos-vector/actions/workflows/ci.yml)
[![License: Elastic-2.0](https://img.shields.io/badge/license-Elastic--2.0-blue.svg)](LICENSE)
[![Rust 1.88+](https://img.shields.io/badge/rust-1.88%2B-orange.svg)](https://www.rust-lang.org)

A high-performance temporal vector database that treats time as a geometric dimension of embedding space. ChronosVector answers not just "what is similar?" but "what *was* similar, what *changed*, and what *will be* similar?"

## Key Features

- **Spatiotemporal kNN** — composite distance `d_ST = α·d_semantic + (1-α)·d_temporal`
- **SIMD-accelerated** — auto-dispatched AVX2/NEON via `pulp` (cosine, L2, dot product)
- **Temporal analytics** — velocity, drift, volatility, Hurst exponent, ADF stationarity
- **Change point detection** — PELT (offline, O(N)) and online EWMA detector
- **Interpretability** — drift attribution, PCA trajectory projection, dimension heatmaps
- **Tiered storage** — hot (in-memory/RocksDB) → warm (file-based) with automatic compaction
- **Crash safety** — WAL with CRC32 validation and segment rotation
- **REST API** — axum-based with ingest, query, trajectory, health endpoints

## Performance

| Metric | Value |
|--------|-------|
| HNSW recall@10 (1K, D=32) | **1.000** |
| HNSW recall@10 (10K, D=128) | **0.956** |
| Graph reachability (10K) | **100%** |
| PELT F1 (3 planted CPs) | **1.000** |
| Online detector FPR | **0.000** |
| Bitmap memory (100K vectors) | **0.16 bytes/vector** |

## Architecture

```
                    ┌──────────┐
                    │ cvx-api  │  REST endpoints (axum)
                    └────┬─────┘
                         │
              ┌──────────┼──────────┐
              │          │          │
        ┌─────┴──┐ ┌─────┴──┐ ┌────┴──────┐
        │ ingest │ │ query  │ │ analytics │
        └────┬───┘ └────┬───┘ └────┬──────┘
             │          │          │
        ┌────┴──────────┴──────────┴────┐
        │           cvx-index           │
        │  ST-HNSW + SIMD + Roaring BM  │
        └──────────────┬────────────────┘
                       │
        ┌──────────────┴────────────────┐
        │          cvx-storage          │
        │  Hot → Warm + WAL + RocksDB   │
        └──────────────┬────────────────┘
                       │
        ┌──────────────┴────────────────┐
        │           cvx-core            │
        │  Types, Traits, Config, Error │
        └───────────────────────────────┘
```

8 crates, ~11K lines of Rust, 280+ tests.

| Crate | Lines | Tests | Description |
|-------|------:|------:|-------------|
| `cvx-core` | 1,601 | 39 | Types, traits, config, error handling |
| `cvx-index` | 2,606 | 78 | ST-HNSW, SIMD distance kernels, Roaring Bitmaps |
| `cvx-storage` | 2,349 | 63 | Hot/warm tiers, WAL, RocksDB, tiered routing |
| `cvx-analytics` | 3,020 | 71 | Calculus, CPD, ODE solver, temporal ML features |
| `cvx-ingest` | 588 | 20 | Delta encoding, input validation |
| `cvx-api` | 398 | 9 | REST handlers, router, state |
| `cvx-server` | 68 | — | Binary with graceful shutdown |
| `cvx-query` | 11 | — | Query engine (planned) |

## Quick Start

```bash
# Build
cargo build --workspace

# Run tests
cargo test --workspace

# Run benchmarks
cargo bench -p cvx-index

# Start server
cargo run --release -p cvx-server

# Or with Docker
docker build -t cvx .
docker run -p 3000:3000 cvx
```

### Ingest vectors

```bash
curl -X POST http://localhost:3000/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"points": [{"entity_id": 1, "timestamp": 1000000, "vector": [0.1, 0.2, 0.3]}]}'
```

### Query

```bash
curl -X POST http://localhost:3000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, 0.3], "k": 5, "filter": {"type": "all"}, "alpha": 1.0}'
```

### Trajectory

```bash
curl http://localhost:3000/v1/entities/1/trajectory
```

## Design Documents

See `design/` for 13 specification documents covering architecture, storage layout,
stochastic analytics, interpretability, multi-scale alignment, temporal ML, and more.

## License

Licensed under the [Elastic License 2.0](LICENSE).

Free to use for research, education, and internal purposes. Commercial use as a managed service requires a separate license. See [LICENSE](LICENSE) for details.
