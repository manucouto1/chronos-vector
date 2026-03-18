//! # ChronosVector
//!
//! High-performance temporal vector database that treats time as a geometric
//! dimension of embedding space.
//!
//! This crate re-exports all ChronosVector components for convenient single-dependency usage:
//!
//! ```toml
//! [dependencies]
//! chronos-vector = "0.1"
//! ```
//!
//! ## Crate Architecture
//!
//! | Crate | Re-export | Description |
//! |-------|-----------|-------------|
//! | [`cvx_core`] | `core` | Types, traits, configuration |
//! | [`cvx_index`] | `index` | ST-HNSW temporal index with SIMD |
//! | [`cvx_analytics`] | `analytics` | 19 analytical functions |
//! | [`cvx_storage`] | `storage` | Tiered storage (WAL + hot/warm/cold) |
//! | [`cvx_ingest`] | `ingest` | Delta encoding, validation |
//! | [`cvx_query`] | `query` | Query engine |
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use chronos_vector::index::TemporalHnsw;
//! use chronos_vector::index::metrics::L2Distance;
//! use chronos_vector::core::TemporalFilter;
//!
//! let config = chronos_vector::index::HnswConfig::default();
//! let mut index = TemporalHnsw::new(config, L2Distance);
//!
//! // Insert temporal vectors
//! index.insert(/*entity_id=*/1, /*timestamp=*/1000, &[0.1, 0.2, 0.3]);
//! index.insert(1, 2000, &[0.15, 0.25, 0.35]);
//!
//! // Search with temporal filtering
//! let results = index.search(&[0.1, 0.2, 0.3], 5, TemporalFilter::All, 1.0, 0);
//!
//! // Retrieve trajectory
//! let traj = index.trajectory(1, TemporalFilter::All);
//! ```

/// Core types, traits, and configuration.
pub use cvx_core as core;

/// Temporal HNSW index with SIMD-accelerated distance metrics.
pub use cvx_index as index;

/// 19 analytical functions: calculus, signatures, topology, anchors.
pub use cvx_analytics as analytics;

/// Tiered storage engine: WAL, RocksDB (hot), file-based (warm).
pub use cvx_storage as storage;

/// Ingestion pipeline: delta encoding, validation.
pub use cvx_ingest as ingest;

/// Query engine.
pub use cvx_query as query;
