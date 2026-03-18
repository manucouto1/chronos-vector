//! # `cvx-index` — Temporal Index Engine for ChronosVector.
//!
//! This crate provides distance computation and index structures for
//! temporal vector search.
//!
//! ## Layer 1: Distance Kernels
//!
//! Three SIMD-accelerated distance metrics via [`pulp`]:
//! - [`metrics::CosineDistance`] — angular distance, range `[0.0, 2.0]`
//! - [`metrics::L2Distance`] — Euclidean distance squared
//! - [`metrics::DotProductDistance`] — negative dot product (for max inner product search)
//!
//! All implement [`cvx_core::DistanceMetric`] and automatically dispatch to
//! the best SIMD instruction set available at runtime (AVX-512, AVX2, NEON, or scalar).
//!
//! ## Layer 4: Spatiotemporal Index (ST-HNSW)
//!
//! [`hnsw::TemporalHnsw`] — temporal-aware HNSW with Roaring Bitmap pre-filtering,
//! composite distance $d_{ST} = \alpha \cdot d_{sem} + (1-\alpha) \cdot d_{time}$,
//! and trajectory retrieval.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod hnsw;
pub mod metrics;
