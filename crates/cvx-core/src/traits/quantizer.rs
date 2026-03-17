//! Distance acceleration through vector quantization.
//!
//! The [`Quantizer`] trait abstracts over different strategies for accelerating
//! distance computations in high-dimensional spaces. The HNSW graph stores both
//! full-precision vectors (for exact operations like trajectories and signatures)
//! and compact codes (for fast approximate distance during graph construction/search).
//!
//! # Two-Phase Distance Computation
//!
//! 1. **Candidate selection**: Use fast approximate distance on compact codes
//! 2. **Final ranking**: Use exact distance on full vectors
//!
//! This mirrors production systems like Qdrant (Scalar Quantization),
//! Faiss (Product Quantization), and Weaviate (Binary Quantization).
//!
//! # Available Strategies
//!
//! | Strategy | Code size (D=768) | Speedup | Recall impact |
//! |----------|-------------------|---------|---------------|
//! | [`NoQuantizer`] | 0 bytes | 1× | None |
//! | Scalar (SQ8) | 768 bytes | ~4× | < 1% |
//! | Product (PQ96) | 96 bytes | ~8× | 1-3% |
//! | Binary (BQ) | 96 bytes | ~32× | 5-10% |
//!
//! # Example
//!
//! ```
//! use cvx_core::traits::quantizer::{Quantizer, NoQuantizer};
//! use cvx_core::DistanceMetric;
//!
//! // No acceleration (default, exact distances)
//! let q = NoQuantizer::new(cvx_core::traits::L2Fn);
//! let code = q.encode(&[1.0, 2.0, 3.0]);
//! ```

use super::DistanceMetric;

/// Acceleration strategy for distance computations.
///
/// Implementations encode vectors into compact codes and provide
/// fast approximate distance between codes.
///
/// The graph uses `distance_approx` for candidate exploration (hot path)
/// and `distance_exact` for final neighbor selection (quality-critical).
pub trait Quantizer: Send + Sync {
    /// Compact representation of a vector.
    ///
    /// - `NoQuantizer`: `()` (zero overhead)
    /// - Scalar: `Vec<u8>` (D bytes)
    /// - Product: `Vec<u8>` (M bytes, where M = D/subvector_dim)
    /// - Binary: `Vec<u64>` (D/64 words)
    type Code: Clone + Send + Sync;

    /// Encode a full-precision vector into a compact code.
    ///
    /// Called once per insert. The code is stored alongside the vector.
    fn encode(&self, vector: &[f32]) -> Self::Code;

    /// Fast approximate distance between two codes.
    ///
    /// This is the hot path: called O(ef_construction × log N) times per insert.
    /// Must be significantly faster than `distance_exact` for the acceleration
    /// to be worthwhile.
    fn distance_approx(&self, a: &Self::Code, b: &Self::Code) -> f32;

    /// Exact distance between full-precision vectors.
    ///
    /// Used for final neighbor selection (heuristic pruning) where
    /// distance quality matters more than speed.
    fn distance_exact(&self, a: &[f32], b: &[f32]) -> f32;

    /// Whether this quantizer provides actual acceleration.
    ///
    /// When `false`, `distance_approx` is unused and the graph
    /// calls `distance_exact` directly. This avoids storing codes.
    fn is_accelerated(&self) -> bool;

    /// Whether this quantizer needs training on a data sample before use.
    ///
    /// Product Quantization requires training a codebook; Scalar and Binary don't.
    fn needs_training(&self) -> bool {
        false
    }

    /// Train the quantizer on a sample of vectors.
    ///
    /// Only called when `needs_training()` returns true.
    /// For PQ: trains the codebook via k-means on subvectors.
    fn train(&mut self, _sample: &[&[f32]]) {}

    /// Human-readable name of this strategy.
    fn name(&self) -> &str;
}

// ─── NoQuantizer (default: exact distances) ─────────────────────

/// Identity quantizer — no acceleration, exact distances only.
///
/// This is the default. Codes are zero-sized (no storage overhead).
/// All distance computations use the underlying [`DistanceMetric`].
#[derive(Clone)]
pub struct NoQuantizer<D: DistanceMetric> {
    metric: D,
}

impl<D: DistanceMetric> NoQuantizer<D> {
    /// Create a no-acceleration quantizer wrapping the given metric.
    pub fn new(metric: D) -> Self {
        Self { metric }
    }

    /// Access the underlying metric.
    pub fn metric(&self) -> &D {
        &self.metric
    }
}

impl<D: DistanceMetric> Quantizer for NoQuantizer<D> {
    type Code = ();

    fn encode(&self, _vector: &[f32]) -> Self::Code {}

    fn distance_approx(&self, _a: &Self::Code, _b: &Self::Code) -> f32 {
        // Never called when is_accelerated() returns false
        0.0
    }

    fn distance_exact(&self, a: &[f32], b: &[f32]) -> f32 {
        self.metric.distance(a, b)
    }

    fn is_accelerated(&self) -> bool {
        false
    }

    fn name(&self) -> &str {
        "none"
    }
}

// ─── ScalarQuantizer (uint8, ~4× speedup) ───────────────────────

/// Scalar Quantization: compress each float32 dimension to uint8.
///
/// For each dimension, maps the value range [min, max] → [0, 255].
/// Distances are computed on uint8 values using integer arithmetic.
///
/// **Pros**: Simple, no training needed, ~4× distance speedup, <1% recall loss.
/// **Cons**: Requires knowing the value range (uses [-1, 1] for normalized vectors).
///
/// This is what Qdrant uses by default for HNSW construction.
#[derive(Clone)]
pub struct ScalarQuantizer<D: DistanceMetric> {
    metric: D,
    /// Min value per dimension (for denormalization). Default: -1.0
    min_val: f32,
    /// Scale factor: 255.0 / (max_val - min_val)
    scale: f32,
}

impl<D: DistanceMetric> ScalarQuantizer<D> {
    /// Create a scalar quantizer for L2-normalized vectors (range [-1, 1]).
    pub fn new(metric: D) -> Self {
        Self {
            metric,
            min_val: -1.0,
            scale: 255.0 / 2.0, // maps [-1, 1] → [0, 255]
        }
    }

    /// Create a scalar quantizer with custom value range.
    pub fn with_range(metric: D, min_val: f32, max_val: f32) -> Self {
        let range = max_val - min_val;
        Self {
            metric,
            min_val,
            scale: if range > 0.0 { 255.0 / range } else { 1.0 },
        }
    }
}

impl<D: DistanceMetric> Quantizer for ScalarQuantizer<D> {
    type Code = Vec<u8>;

    fn encode(&self, vector: &[f32]) -> Self::Code {
        vector
            .iter()
            .map(|&v| {
                let normalized = (v - self.min_val) * self.scale;
                normalized.clamp(0.0, 255.0) as u8
            })
            .collect()
    }

    fn distance_approx(&self, a: &Self::Code, b: &Self::Code) -> f32 {
        // L2 distance on uint8 values (integer arithmetic, auto-vectorized by LLVM)
        let mut sum: u32 = 0;
        for i in 0..a.len() {
            let diff = a[i] as i32 - b[i] as i32;
            sum += (diff * diff) as u32;
        }
        // Scale back to float distance (approximate)
        (sum as f32).sqrt() / self.scale
    }

    fn distance_exact(&self, a: &[f32], b: &[f32]) -> f32 {
        self.metric.distance(a, b)
    }

    fn is_accelerated(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "scalar_u8"
    }
}

// ─── Helper for L2 distance function (used in NoQuantizer default) ──

/// Simple L2 distance function for use with quantizers.
#[derive(Clone, Copy)]
pub struct L2Fn;

impl DistanceMetric for L2Fn {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    fn name(&self) -> &str {
        "l2"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_quantizer_exact() {
        let q = NoQuantizer::new(L2Fn);
        assert!(!q.is_accelerated());
        let d = q.distance_exact(&[1.0, 0.0], &[0.0, 1.0]);
        assert!((d - std::f32::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn scalar_quantizer_encode_decode() {
        let q = ScalarQuantizer::new(L2Fn);

        // Normalized vector
        let v = [0.5, -0.3, 0.0, 1.0, -1.0];
        let code = q.encode(&v);
        assert_eq!(code.len(), 5);
        assert_eq!(code[4], 0);   // -1.0 → 0
        assert_eq!(code[3], 255); // 1.0 → 255

        // Distance between identical codes should be ~0
        let d = q.distance_approx(&code, &code);
        assert!(d < 1e-6);
    }

    #[test]
    fn scalar_quantizer_preserves_order() {
        let q = ScalarQuantizer::new(L2Fn);

        let a = [0.5, 0.3, 0.0];
        let b = [0.6, 0.3, 0.0]; // close to a
        let c = [-0.5, -0.3, 0.9]; // far from a

        let code_a = q.encode(&a);
        let code_b = q.encode(&b);
        let code_c = q.encode(&c);

        let d_ab = q.distance_approx(&code_a, &code_b);
        let d_ac = q.distance_approx(&code_a, &code_c);

        // Approximate distance should preserve ordering
        assert!(d_ab < d_ac, "d(a,b)={d_ab} should be < d(a,c)={d_ac}");

        // Exact distances for comparison
        let exact_ab = q.distance_exact(&a, &b);
        let exact_ac = q.distance_exact(&a, &c);
        assert!(exact_ab < exact_ac);
    }
}
