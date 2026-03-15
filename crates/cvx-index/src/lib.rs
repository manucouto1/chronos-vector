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

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod hnsw;
pub mod metrics;
