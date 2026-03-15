//! Dot product distance metric (for maximum inner product search).
//!
//! Returns **negative** dot product so that smaller values = more similar
//! (consistent with other distance metrics where lower = better).

use cvx_core::DistanceMetric;

use super::simd_ops::dot_product_simd;

/// Negative dot product distance: $d(a, b) = -a \cdot b$.
///
/// Used for Maximum Inner Product Search (MIPS). The negation ensures that
/// the most similar vectors (highest dot product) have the smallest distance.
///
/// # Example
///
/// ```
/// use cvx_core::DistanceMetric;
/// use cvx_index::metrics::DotProductDistance;
///
/// let d = DotProductDistance;
/// let a = vec![1.0, 0.0];
/// let b = vec![1.0, 0.0];
/// assert!(d.distance(&a, &b) < 0.0); // same direction → negative (= similar)
/// ```
pub struct DotProductDistance;

impl DistanceMetric for DotProductDistance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        -dot_product_simd(a, b)
    }

    fn name(&self) -> &str {
        "dot"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_direction_is_negative() {
        let d = DotProductDistance;
        let a = vec![1.0, 0.0];
        let b = vec![2.0, 0.0];
        assert!(d.distance(&a, &b) < 0.0);
    }

    #[test]
    fn orthogonal_is_zero() {
        let d = DotProductDistance;
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(d.distance(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn opposite_is_positive() {
        let d = DotProductDistance;
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!(d.distance(&a, &b) > 0.0);
    }

    #[test]
    fn more_similar_has_lower_distance() {
        let d = DotProductDistance;
        let query = vec![1.0, 0.0];
        let close = vec![0.9, 0.1];
        let far = vec![0.1, 0.9];
        assert!(d.distance(&query, &close) < d.distance(&query, &far));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Dot product distance is anti-symmetric in sign: d(a,b) == d(b,a)
        /// (because dot product is commutative, negation preserves equality)
        #[test]
        fn symmetric(
            a in prop::collection::vec(-100.0f32..100.0, 32..=32),
            b in prop::collection::vec(-100.0f32..100.0, 32..=32),
        ) {
            let ab = DotProductDistance.distance(&a, &b);
            let ba = DotProductDistance.distance(&b, &a);
            prop_assert!((ab - ba).abs() < 1e-3);
        }
    }
}
