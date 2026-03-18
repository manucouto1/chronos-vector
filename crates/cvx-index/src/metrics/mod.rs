//! SIMD-accelerated distance metrics.
//!
//! All metrics use [`pulp`] for portable SIMD: a single implementation
//! dispatches to AVX-512, AVX2, NEON, or scalar at runtime.
//!
//! # Example
//!
//! ```
//! use cvx_core::DistanceMetric;
//! use cvx_index::metrics::CosineDistance;
//!
//! let metric = CosineDistance;
//! let a = vec![1.0, 0.0, 0.0];
//! let b = vec![0.0, 1.0, 0.0];
//!
//! let dist = metric.distance(&a, &b);
//! assert!((dist - 1.0).abs() < 1e-5); // orthogonal → cosine distance = 1.0
//! ```

mod cosine;
mod dot_product;
mod l2;
mod simd_ops;

pub use cosine::CosineDistance;
pub use dot_product::DotProductDistance;
pub use l2::L2Distance;
