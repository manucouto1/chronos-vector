//! Low-level SIMD operations using pulp.
//!
//! These are the building blocks for all distance metrics. Each function
//! dispatches to the best SIMD ISA available at runtime.

use pulp::Arch;

/// Compute the dot product of two float slices using SIMD.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have equal length");

    let arch = Arch::new();
    arch.dispatch(DotProduct(a, b))
}

/// Compute the squared L2 norm of a float slice using SIMD.
pub fn norm_squared_simd(a: &[f32]) -> f32 {
    dot_product_simd(a, a)
}

/// Compute the sum of squared differences between two slices using SIMD.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
pub fn l2_squared_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have equal length");

    let arch = Arch::new();
    arch.dispatch(L2Squared(a, b))
}

struct DotProduct<'a>(&'a [f32], &'a [f32]);

impl pulp::WithSimd for DotProduct<'_> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> f32 {
        let (a_head, a_tail) = S::as_simd_f32s(self.0);
        let (b_head, b_tail) = S::as_simd_f32s(self.1);

        let mut acc = simd.splat_f32s(0.0);
        for (&a_chunk, &b_chunk) in a_head.iter().zip(b_head.iter()) {
            acc = simd.mul_add_e_f32s(a_chunk, b_chunk, acc);
        }

        let mut sum = simd.reduce_sum_f32s(acc);
        for (&a_val, &b_val) in a_tail.iter().zip(b_tail.iter()) {
            sum = f32::mul_add(a_val, b_val, sum);
        }
        sum
    }
}

struct L2Squared<'a>(&'a [f32], &'a [f32]);

impl pulp::WithSimd for L2Squared<'_> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, simd: S) -> f32 {
        let (a_head, a_tail) = S::as_simd_f32s(self.0);
        let (b_head, b_tail) = S::as_simd_f32s(self.1);

        // Compute ||a - b||² = ||a||² + ||b||² - 2(a·b)
        // This avoids needing a SIMD subtract operation.
        let mut acc_aa = simd.splat_f32s(0.0);
        let mut acc_bb = simd.splat_f32s(0.0);
        let mut acc_ab = simd.splat_f32s(0.0);

        for (&a_chunk, &b_chunk) in a_head.iter().zip(b_head.iter()) {
            acc_aa = simd.mul_add_e_f32s(a_chunk, a_chunk, acc_aa);
            acc_bb = simd.mul_add_e_f32s(b_chunk, b_chunk, acc_bb);
            acc_ab = simd.mul_add_e_f32s(a_chunk, b_chunk, acc_ab);
        }

        let mut sum_aa = simd.reduce_sum_f32s(acc_aa);
        let mut sum_bb = simd.reduce_sum_f32s(acc_bb);
        let mut sum_ab = simd.reduce_sum_f32s(acc_ab);

        for (&a_val, &b_val) in a_tail.iter().zip(b_tail.iter()) {
            sum_aa = f32::mul_add(a_val, a_val, sum_aa);
            sum_bb = f32::mul_add(b_val, b_val, sum_bb);
            sum_ab = f32::mul_add(a_val, b_val, sum_ab);
        }

        // ||a-b||² = ||a||² + ||b||² - 2(a·b)
        // Clamp to 0.0 for numerical stability (can be slightly negative due to FP)
        (sum_aa + sum_bb - 2.0 * sum_ab).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_product_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let result = dot_product_simd(&a, &b);
        assert!((result - 32.0).abs() < 1e-5); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn dot_product_orthogonal() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let result = dot_product_simd(&a, &b);
        assert!(result.abs() < 1e-6);
    }

    #[test]
    fn l2_squared_identical() {
        let a = [1.0, 2.0, 3.0];
        assert!(l2_squared_simd(&a, &a).abs() < 1e-6);
    }

    #[test]
    fn l2_squared_known() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        let result = l2_squared_simd(&a, &b);
        assert!((result - 2.0).abs() < 1e-5); // (1-0)^2 + (0-1)^2 = 2
    }

    #[test]
    fn norm_squared_unit_vector() {
        let a = [1.0, 0.0, 0.0];
        assert!((norm_squared_simd(&a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn handles_d768() {
        let a: Vec<f32> = (0..768).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..768).map(|i| ((768 - i) as f32) * 0.001).collect();
        let _result = dot_product_simd(&a, &b);
        let _result = l2_squared_simd(&a, &b);
    }

    #[test]
    #[should_panic(expected = "vectors must have equal length")]
    fn dot_product_panics_on_mismatch() {
        dot_product_simd(&[1.0, 2.0], &[1.0]);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Dot product is commutative: a·b == b·a
        #[test]
        fn dot_product_commutative(
            a in prop::collection::vec(-100.0f32..100.0, 64..=64),
            b in prop::collection::vec(-100.0f32..100.0, 64..=64),
        ) {
            let ab = dot_product_simd(&a, &b);
            let ba = dot_product_simd(&b, &a);
            prop_assert!((ab - ba).abs() < 1e-3, "a·b={ab}, b·a={ba}");
        }

        /// L2 squared is non-negative
        #[test]
        fn l2_squared_non_negative(
            a in prop::collection::vec(-100.0f32..100.0, 64..=64),
            b in prop::collection::vec(-100.0f32..100.0, 64..=64),
        ) {
            prop_assert!(l2_squared_simd(&a, &b) >= -1e-6);
        }

        /// L2 squared is symmetric: d(a,b) == d(b,a)
        #[test]
        fn l2_squared_symmetric(
            a in prop::collection::vec(-100.0f32..100.0, 64..=64),
            b in prop::collection::vec(-100.0f32..100.0, 64..=64),
        ) {
            let ab = l2_squared_simd(&a, &b);
            let ba = l2_squared_simd(&b, &a);
            prop_assert!((ab - ba).abs() < 1e-3, "d(a,b)={ab}, d(b,a)={ba}");
        }

        /// L2 squared of identical vectors is zero
        #[test]
        fn l2_squared_identity(
            a in prop::collection::vec(-100.0f32..100.0, 64..=64),
        ) {
            prop_assert!(l2_squared_simd(&a, &a).abs() < 1e-3);
        }

        /// Norm squared is non-negative
        #[test]
        fn norm_squared_non_negative(
            a in prop::collection::vec(-100.0f32..100.0, 64..=64),
        ) {
            prop_assert!(norm_squared_simd(&a) >= -1e-6);
        }
    }
}
