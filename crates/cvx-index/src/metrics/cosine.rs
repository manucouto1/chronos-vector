//! Cosine distance metric.
//!
//! Cosine distance = $1 - \text{cosine\_similarity}(a, b)$
//!
//! Range: `[0.0, 2.0]` where 0.0 = identical direction, 1.0 = orthogonal, 2.0 = opposite.

use cvx_core::DistanceMetric;

use super::simd_ops::{dot_product_simd, norm_squared_simd};

/// Cosine distance: $d(a, b) = 1 - \frac{a \cdot b}{\|a\| \cdot \|b\|}$.
///
/// # Example
///
/// ```
/// use cvx_core::DistanceMetric;
/// use cvx_index::metrics::CosineDistance;
///
/// let d = CosineDistance;
/// let a = vec![1.0, 0.0];
/// let b = vec![1.0, 0.0];
/// assert!(d.distance(&a, &b) < 1e-5); // same direction → 0.0
/// ```
#[derive(Clone, Copy)]
pub struct CosineDistance;

impl DistanceMetric for CosineDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot = dot_product_simd(a, b);
        let norm_a = norm_squared_simd(a).sqrt();
        let norm_b = norm_squared_simd(b).sqrt();

        let denom = norm_a * norm_b;
        if denom < f32::EPSILON {
            // At least one zero vector — define distance as 1.0 (orthogonal)
            return 1.0;
        }

        let cosine_sim = dot / denom;
        // Clamp to [-1, 1] for numerical stability (FP rounding can exceed)
        1.0 - cosine_sim.clamp(-1.0, 1.0)
    }

    fn name(&self) -> &str {
        "cosine"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_distance_zero() {
        let d = CosineDistance;
        let v = vec![1.0, 2.0, 3.0];
        assert!(d.distance(&v, &v) < 1e-5);
    }

    #[test]
    fn opposite_vectors_distance_two() {
        let d = CosineDistance;
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert!((d.distance(&a, &b) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn orthogonal_vectors_distance_one() {
        let d = CosineDistance;
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((d.distance(&a, &b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn zero_vector_returns_one() {
        let d = CosineDistance;
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!((d.distance(&a, &b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn d768_works() {
        let d = CosineDistance;
        let a: Vec<f32> = (0..768).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..768).map(|i| ((768 - i) as f32) * 0.001).collect();
        let result = d.distance(&a, &b);
        assert!(result >= 0.0 && result <= 2.0);
    }

    #[test]
    fn name_is_cosine() {
        assert_eq!(CosineDistance.name(), "cosine");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Cosine distance is symmetric
        #[test]
        fn symmetric(
            a in prop::collection::vec(0.01f32..10.0, 32..=32),
            b in prop::collection::vec(0.01f32..10.0, 32..=32),
        ) {
            let d = CosineDistance;
            let ab = d.distance(&a, &b);
            let ba = d.distance(&b, &a);
            prop_assert!((ab - ba).abs() < 1e-5, "d(a,b)={ab}, d(b,a)={ba}");
        }

        /// Cosine distance is in [0.0, 2.0]
        #[test]
        fn in_range(
            a in prop::collection::vec(0.01f32..10.0, 32..=32),
            b in prop::collection::vec(0.01f32..10.0, 32..=32),
        ) {
            let d = CosineDistance;
            let dist = d.distance(&a, &b);
            prop_assert!(dist >= -1e-5 && dist <= 2.0 + 1e-5, "dist={dist}");
        }

        /// Distance to self is zero
        #[test]
        fn identity(
            a in prop::collection::vec(0.01f32..10.0, 32..=32),
        ) {
            let d = CosineDistance;
            prop_assert!(d.distance(&a, &a) < 1e-4);
        }
    }
}
