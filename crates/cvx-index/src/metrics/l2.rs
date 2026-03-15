//! L2 (Euclidean) squared distance metric.
//!
//! Returns the squared Euclidean distance to avoid the sqrt cost in comparisons.
//! Range: `[0.0, ∞)`.

use cvx_core::DistanceMetric;

use super::simd_ops::l2_squared_simd;

/// Squared Euclidean distance: $d(a, b) = \sum_i (a_i - b_i)^2$.
///
/// Returns the **squared** distance (no sqrt) because ranking is preserved
/// and sqrt is unnecessary for nearest-neighbor comparisons.
///
/// # Example
///
/// ```
/// use cvx_core::DistanceMetric;
/// use cvx_index::metrics::L2Distance;
///
/// let d = L2Distance;
/// let a = vec![1.0, 0.0];
/// let b = vec![0.0, 1.0];
/// assert!((d.distance(&a, &b) - 2.0).abs() < 1e-5); // (1-0)^2 + (0-1)^2 = 2
/// ```
pub struct L2Distance;

impl DistanceMetric for L2Distance {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        l2_squared_simd(a, b)
    }

    fn name(&self) -> &str {
        "l2"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_vectors_zero() {
        let d = L2Distance;
        let v = vec![1.0, 2.0, 3.0];
        assert!(d.distance(&v, &v).abs() < 1e-6);
    }

    #[test]
    fn unit_vectors_known_distance() {
        let d = L2Distance;
        let a = vec![3.0, 0.0];
        let b = vec![0.0, 4.0];
        assert!((d.distance(&a, &b) - 25.0).abs() < 1e-5); // 9 + 16 = 25
    }

    #[test]
    fn d768_works() {
        let d = L2Distance;
        let a: Vec<f32> = (0..768).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..768).map(|i| ((768 - i) as f32) * 0.001).collect();
        let result = d.distance(&a, &b);
        assert!(result >= 0.0);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn non_negative(
            a in prop::collection::vec(-100.0f32..100.0, 32..=32),
            b in prop::collection::vec(-100.0f32..100.0, 32..=32),
        ) {
            prop_assert!(L2Distance.distance(&a, &b) >= -1e-5);
        }

        #[test]
        fn symmetric(
            a in prop::collection::vec(-100.0f32..100.0, 32..=32),
            b in prop::collection::vec(-100.0f32..100.0, 32..=32),
        ) {
            let ab = L2Distance.distance(&a, &b);
            let ba = L2Distance.distance(&b, &a);
            prop_assert!((ab - ba).abs() < 1e-3);
        }

        #[test]
        fn identity(
            a in prop::collection::vec(-100.0f32..100.0, 32..=32),
        ) {
            prop_assert!(L2Distance.distance(&a, &a).abs() < 1e-3);
        }

        /// Triangle inequality: d(a,c) <= d(a,b) + d(b,c) (using sqrt for true metric)
        #[test]
        fn triangle_inequality(
            a in prop::collection::vec(-10.0f32..10.0, 16..=16),
            b in prop::collection::vec(-10.0f32..10.0, 16..=16),
            c in prop::collection::vec(-10.0f32..10.0, 16..=16),
        ) {
            let d = L2Distance;
            let ab = d.distance(&a, &b).sqrt();
            let bc = d.distance(&b, &c).sqrt();
            let ac = d.distance(&a, &c).sqrt();
            prop_assert!(ac <= ab + bc + 1e-3, "triangle inequality: {ac} <= {ab} + {bc}");
        }
    }
}
