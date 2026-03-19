---
title: "Rust API Reference"
description: "Auto-generated API documentation for all CVX crates"
---

The complete Rust API documentation is auto-generated from source code with `cargo doc` and deployed alongside this site.

## Browse the API

<a href="/chronos-vector/api/cvx_core/index.html" target="_blank" class="sl-link-button">cvx-core</a> — Core types, traits, and configuration
<a href="/chronos-vector/api/cvx_index/index.html" target="_blank" class="sl-link-button">cvx-index</a> — ST-HNSW temporal index engine
<a href="/chronos-vector/api/cvx_analytics/index.html" target="_blank" class="sl-link-button">cvx-analytics</a> — Temporal analytics (calculus, signatures, topology, anchors)
<a href="/chronos-vector/api/cvx_storage/index.html" target="_blank" class="sl-link-button">cvx-storage</a> — Tiered storage (WAL, hot/warm/cold)
<a href="/chronos-vector/api/cvx_ingest/index.html" target="_blank" class="sl-link-button">cvx-ingest</a> — Ingestion pipeline
<a href="/chronos-vector/api/cvx_query/index.html" target="_blank" class="sl-link-button">cvx-query</a> — Query engine
<a href="/chronos-vector/api/cvx_server/index.html" target="_blank" class="sl-link-button">cvx-server</a> — gRPC server

## Crate Architecture

```
cvx-core          Shared types, traits (DistanceMetric, TemporalFilter, TemporalPoint)
  ├── cvx-index   ST-HNSW graph with roaring bitmap filtering
  ├── cvx-analytics  19 analytical functions (calculus, signatures, topology, anchors)
  ├── cvx-storage    WAL + RocksDB (hot) + file-based (warm) tiered storage
  ├── cvx-ingest     Batch and streaming ingestion
  ├── cvx-query      Query planning and execution
  └── cvx-server     gRPC API server
```

## Key Modules

### cvx-analytics

| Module | Functions | Description |
|--------|-----------|-------------|
| `calculus` | `velocity`, `acceleration`, `drift_report`, `hurst_exponent`, `realized_volatility` | Vector differential calculus |
| `anchor` | `project_to_anchors`, `anchor_summary` | Reference frame projection (RFC-006) |
| `signatures` | `compute_signature`, `compute_log_signature` | Path signatures from rough path theory |
| `pelt` | `detect` | PELT offline change point detection |
| `point_process` | `extract_event_features` | Temporal event pattern analysis |
| `topology` | `topological_summary` | Persistent homology (Betti curves) |
| `wasserstein` | `wasserstein_drift` | Sliced Wasserstein optimal transport |
| `fisher_rao` | `fisher_rao_distance`, `hellinger_distance` | Riemannian distributional distances |
| `trajectory` | `discrete_frechet_temporal` | Frechet distance between trajectories |
| `cohort` | `cohort_drift` | Cohort-level drift, convergence, outliers (RFC-007) |
| `temporal_join` | `temporal_join`, `group_temporal_join` | Convergence window detection (RFC-007) |
| `motifs` | `discover_motifs`, `discover_discords` | Matrix Profile recurring patterns (RFC-007) |
| `granger` | `granger_causality` | VAR-based Granger causality testing (RFC-007) |
| `counterfactual` | `counterfactual_trajectory` | Pre-change extrapolation + divergence (RFC-007) |
| `anchor_index` | `AnchorSpaceIndex` | Cross-model anchor-space index (RFC-011) |

### cvx-index

| Module | Types | Description |
|--------|-------|-------------|
| `hnsw` | `HnswGraph`, `HnswConfig` | Vanilla HNSW with scalar quantization |
| `hnsw::temporal` | `TemporalHnsw` | Spatiotemporal HNSW with entity tracking |
| `hnsw::concurrent` | `ConcurrentTemporalHnsw` | Thread-safe concurrent index |
| `hnsw::partitioned` | `PartitionedTemporalHnsw` | Time-partitioned sharding (RFC-008) |
| `hnsw::streaming` | `StreamingTemporalHnsw` | Hot buffer + compaction (RFC-008) |
| `hnsw::temporal_lsh` | `TemporalLSH` | Spatiotemporal locality hashing (RFC-008) |
| `hnsw::temporal_edges` | `TemporalEdgeLayer` | Temporal successor edges (RFC-010) |
| `hnsw::temporal_graph` | `TemporalGraphIndex` | Hybrid causal search (RFC-010) |
| `metrics` | `L2Distance`, `CosineDistance`, `DotProductDistance` | Distance metrics with SIMD |

### cvx-mcp

| Module | Types | Description |
|--------|-------|-------------|
| `server` | `McpServer` | MCP JSON-RPC server with 8 LLM tools (RFC-009) |
| `embedder` | `MockEmbedder`, `OnnxEmbedder` | Inline text→vector conversion (RFC-009) |

### cvx-core

| Type | Description |
|------|-------------|
| `TemporalPoint` | Vector with entity_id, timestamp, metadata |
| `TemporalFilter` | Query constraints (Snapshot, Range, Before, After, All) |
| `DistanceMetric` | Trait for distance functions |
| `CvxConfig` | System configuration (TOML-serializable) |

## Generating Locally

```bash
cargo doc --workspace --no-deps --open
```

This opens the full API documentation in your browser, including all private items and cross-linked source code.

## Python Bindings

The Python API (`chronos_vector` module) is a subset of the Rust API exposed via PyO3. See the [Temporal Analytics Toolkit](/chronos-vector/specs/temporal-analytics) for the Python function reference.
