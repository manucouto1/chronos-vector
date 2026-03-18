# ChronosVector (CVX)

[![CI](https://github.com/manucouto1/chronos-vector/actions/workflows/ci.yml/badge.svg)](https://github.com/manucouto1/chronos-vector/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/manucouto1/chronos-vector/branch/main/graph/badge.svg)](https://codecov.io/gh/manucouto1/chronos-vector)
[![Docs](https://img.shields.io/badge/docs-live-green.svg)](https://manucouto1.github.io/chronos-vector/)
[![crates.io](https://img.shields.io/crates/v/chronos-vector.svg)](https://crates.io/crates/chronos-vector)
[![PyPI](https://img.shields.io/pypi/v/chronos-vector.svg)](https://pypi.org/project/chronos-vector/)
[![License: Elastic-2.0](https://img.shields.io/badge/license-Elastic--2.0-blue.svg)](LICENSE)
[![Rust 1.88+](https://img.shields.io/badge/rust-1.88%2B-orange.svg)](https://www.rust-lang.org)

A high-performance temporal vector database that treats time as a geometric dimension of embedding space. ChronosVector answers not just "what is similar?" but "what *was* similar, what *changed*, and what *will be* similar?"

## Documentation

| Resource | Link |
|----------|------|
| **Full Documentation** | [manucouto1.github.io/chronos-vector](https://manucouto1.github.io/chronos-vector/) |
| White Paper | [Research Overview](https://manucouto1.github.io/chronos-vector/research/white-paper/) |
| Temporal Analytics API (19 functions) | [Toolkit Reference](https://manucouto1.github.io/chronos-vector/specs/temporal-analytics/) |
| Rust API (cargo doc) | [API Reference](https://manucouto1.github.io/chronos-vector/api/cvx_core/) |
| RFC-006: Anchor Projection | [RFC](https://manucouto1.github.io/chronos-vector/rfc/rfc-006/) |

### Tutorials (with interactive Plotly visualizations)

| Tutorial | Domain | Key Result |
|----------|--------|------------|
| [Mental Health Explorer](https://manucouto1.github.io/chronos-vector/tutorials/b1-explorer/) | Clinical NLP | 13 CVX features → F1=0.600 |
| [Clinical Anchoring](https://manucouto1.github.io/chronos-vector/tutorials/b2-clinical-anchoring/) | Clinical NLP | DSM-5 anchors → F1=0.744, AUC=0.886 |
| [Political Rhetoric](https://manucouto1.github.io/chronos-vector/tutorials/trump-impact/) | Political NLP | Trump tweets + S&P 500 alignment |
| [Market Regimes](https://manucouto1.github.io/chronos-vector/tutorials/finance-regimes/) | Finance | 11 changepoints, Hurst=0.74 |
| [Anomaly Detection](https://manucouto1.github.io/chronos-vector/tutorials/nab-anomaly/) | Time Series | NAB benchmark, 4 detection strategies |
| [MAP-Elites](https://manucouto1.github.io/chronos-vector/tutorials/map-elites/) | Quality-Diversity | HNSW as adaptive niche discovery |
| [MLOps Drift](https://manucouto1.github.io/chronos-vector/tutorials/mlops-drift/) | Production ML | 5 independent drift signals |

## Key Features

- **Spatiotemporal kNN** — composite distance `d_ST = α·d_semantic + (1-α)·d_temporal`
- **19 analytical functions** — velocity, drift, Hurst, changepoints, path signatures, topology, anchor projection
- **Anchor projection** — project trajectories from ℝᴰ to interpretable ℝᴷ coordinates (RFC-006)
- **SIMD-accelerated** — auto-dispatched AVX2/NEON via `pulp` (cosine, L2, dot product)
- **Index persistence** — save/load HNSW graph via postcard binary serialization
- **Tiered storage** — hot (in-memory/RocksDB) → warm (file-based) with automatic compaction
- **Crash safety** — WAL with CRC32 validation and segment rotation
- **Python bindings** — `pip install chronos-vector` (PyO3/maturin)

## Python Quick Start

```python
import chronos_vector as cvx
import numpy as np

# Create and populate index
index = cvx.TemporalIndex(m=16, ef_construction=200)
index.bulk_insert(entity_ids, timestamps, vectors)
index.save("index.cvx")  # persist for fast reload

# Trajectory analysis
traj = index.trajectory(entity_id=1)
vel = cvx.velocity(traj, timestamp=t)
h = cvx.hurst_exponent(traj)
cps = cvx.detect_changepoints(1, traj)

# Anchor projection — measure relative to reference points
projected = cvx.project_to_anchors(traj, anchors, metric='cosine')
summary = cvx.anchor_summary(projected)  # {mean, min, trend, last}

# All analytics work on projected trajectory too
vel_anchor = cvx.velocity(projected, timestamp=t)
sig = cvx.path_signature(projected, depth=2)
```

## Performance

| Metric | Value |
|--------|-------|
| HNSW recall@10 (1K, D=32) | **1.000** |
| HNSW recall@10 (10K, D=128) | **0.956** |
| Graph reachability (10K) | **100%** |
| PELT F1 (3 planted CPs) | **1.000** |
| Bitmap memory (100K vectors) | **0.16 bytes/vector** |
| Index save/load (225K, D=768) | **< 1s** (vs 500s rebuild) |

## Architecture

```
                    ┌──────────┐
                    │ cvx-api  │  REST + gRPC endpoints
                    └────┬─────┘
                         │
              ┌──────────┼──────────┐
              │          │          │
        ┌─────┴──┐ ┌─────┴──┐ ┌────┴──────┐
        │ ingest │ │ query  │ │ analytics │  19 functions
        └────┬───┘ └────┬───┘ └────┬──────┘  + anchor projection
             │          │          │
        ┌────┴──────────┴──────────┴────┐
        │           cvx-index           │
        │  ST-HNSW + SIMD + Roaring BM  │  save/load persistence
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

8 crates + Python bindings, 280+ tests.

| Crate | Description |
|-------|-------------|
| `cvx-core` | Types, traits, config, error handling |
| `cvx-index` | ST-HNSW with temporal filtering, scalar quantization, persistence |
| `cvx-analytics` | 19 analytical functions: calculus, signatures, topology, anchors |
| `cvx-storage` | Hot/warm tiers, WAL, RocksDB, tiered routing |
| `cvx-ingest` | Delta encoding, input validation |
| `cvx-query` | Query engine |
| `cvx-api` | REST + gRPC handlers |
| `cvx-server` | Server binary with graceful shutdown |
| `cvx-python` | Python bindings via PyO3/maturin |

## Rust Quick Start

```bash
# Build
cargo build --workspace

# Run tests
cargo test --workspace

# Generate API docs
cargo doc --workspace --no-deps --open

# Start server
cargo run --release -p cvx-server
```

## Python Installation

```bash
# From source (requires Rust toolchain)
cd crates/cvx-python
maturin develop --release

# Or with pip (when published)
pip install chronos-vector
```

## Cross-Domain Research

CVX has been validated across 7 investigations in 6 domains. See the [White Paper](https://manucouto1.github.io/chronos-vector/research/white-paper/) for details.

| Domain | Dataset | CVX Contribution |
|--------|---------|-----------------|
| Clinical NLP | eRisk 2017-2022 | DSM-5 anchor projection: F1=0.744, AUC=0.886 |
| Political NLP | Trump Twitter Archive | Rhetorical regime detection via changepoints + signatures |
| Finance | S&P 500 Sector ETFs | 11 regime changepoints, Hurst=0.74, path signatures |
| Anomaly Detection | Numenta NAB | Trajectory-geometric detection (velocity, topology) |
| Fraud Detection | IEEE-CIS | Transaction trajectory fingerprinting |
| Cybersecurity | CERT CMU | Behavioral regime shift detection |
| Quality-Diversity | MAP-Elites | HNSW as adaptive niche discovery |

## License

Licensed under the [Elastic License 2.0](LICENSE).

Free to use for research, education, and internal purposes. Commercial use as a managed service requires a separate license. See [LICENSE](LICENSE) for details.
