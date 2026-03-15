# ChronosVector (CVX)

[![CI](https://github.com/manucouto1/chronos-vector/actions/workflows/ci.yml/badge.svg)](https://github.com/manucouto1/chronos-vector/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

A temporal vector database that tracks how embeddings evolve over time. ChronosVector answers not just "what is similar?" but "what *was* similar, what *changed*, and what *will be* similar?"

## Architecture

ChronosVector is organized as a Cargo workspace with 8 crates:

| Crate | Description |
|-------|-------------|
| `cvx-core` | Core types, traits, config, and error handling |
| `cvx-index` | ST-HNSW temporal index with time-decay edges and SIMD distance kernels |
| `cvx-storage` | Tiered storage engine (hot/warm/cold) with WAL and compaction |
| `cvx-ingest` | Ingestion pipeline with delta encoding and validation |
| `cvx-analytics` | Neural ODE, change point detection (PELT/BOCPD), vector calculus |
| `cvx-query` | Query engine supporting 8 temporal query types |
| `cvx-api` | Dual-protocol API gateway (REST + gRPC) |
| `cvx-server` | Server binary — bootstrap, config, runtime |

## Quick Start

```bash
# Build all crates
cargo build --workspace

# Run tests
cargo test --workspace

# Run the server
cargo run -p cvx-server
```

## Project Layout

```
crates/          Rust workspace crates
design/          Architecture & design documents
docs/            Documentation site (Starlight)
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
