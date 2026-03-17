//! Trajectory distance measures for comparing entity evolutions.
//!
//! Provides measures that compare *sequences* of vectors, not individual points.
//! This is a fundamentally different operation from kNN search.
//!
//! # Available Measures
//!
//! | Measure | Complexity | Properties |
//! |---------|-----------|-----------|
//! | [`discrete_frechet`] | O(n×m) | Respects ordering, handles unequal lengths |
//! | Signature distance | O(K²) | Universal features, very fast (see [`signatures`]) |
//!
//! Signature distance is recommended as the default: it's O(K²) per comparison
//! and captures all order-dependent temporal dynamics. Fréchet is offered as
//! the exact alternative when ordering-sensitive geometry matters.
//!
//! # References
//!
//! - Eiter, T. & Mannila, H. (1994). Computing discrete Fréchet distance.
//! - Toohey, K. & Duckham, M. (2015). Trajectory Similarity Measures. ACM SIGSPATIAL.

/// Compute the discrete Fréchet distance between two trajectories.
///
/// The Fréchet distance measures the maximum minimum distance between
/// corresponding points of two curves when traversed monotonically.
/// Intuitively: the shortest leash needed to walk a dog along path B
/// while you walk along path A, both moving only forward.
///
/// # Arguments
///
/// * `a` - First trajectory: sequence of vectors (timestamps ignored).
/// * `b` - Second trajectory.
///
/// # Complexity
///
/// O(n × m) time and space, where n = |a|, m = |b|.
pub fn discrete_frechet(a: &[&[f32]], b: &[&[f32]]) -> f64 {
    let n = a.len();
    let m = b.len();
    if n == 0 || m == 0 {
        return f64::INFINITY;
    }

    // DP table: dp[i][j] = Fréchet distance considering a[0..=i] and b[0..=j]
    let mut dp = vec![vec![f64::NEG_INFINITY; m]; n];

    for i in 0..n {
        for j in 0..m {
            let d = l2_dist(a[i], b[j]);
            if i == 0 && j == 0 {
                dp[i][j] = d;
            } else if i == 0 {
                dp[i][j] = dp[i][j - 1].max(d);
            } else if j == 0 {
                dp[i][j] = dp[i - 1][j].max(d);
            } else {
                let prev = dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
                dp[i][j] = prev.max(d);
            }
        }
    }

    dp[n - 1][m - 1]
}

/// Compute discrete Fréchet distance from timestamped trajectories.
///
/// Convenience wrapper that extracts vectors from (timestamp, vector) pairs.
pub fn discrete_frechet_temporal(
    a: &[(i64, &[f32])],
    b: &[(i64, &[f32])],
) -> f64 {
    let va: Vec<&[f32]> = a.iter().map(|(_, v)| *v).collect();
    let vb: Vec<&[f32]> = b.iter().map(|(_, v)| *v).collect();
    discrete_frechet(&va, &vb)
}

/// L2 distance between two vectors.
#[inline]
fn l2_dist(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = *x as f64 - *y as f64;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frechet_identical_paths() {
        let a: Vec<&[f32]> = vec![&[0.0, 0.0], &[1.0, 0.0], &[1.0, 1.0]];
        let b: Vec<&[f32]> = vec![&[0.0, 0.0], &[1.0, 0.0], &[1.0, 1.0]];
        let d = discrete_frechet(&a, &b);
        assert!(d < 1e-10, "identical paths should have distance ~0, got {d}");
    }

    #[test]
    fn frechet_shifted_paths() {
        let a: Vec<&[f32]> = vec![&[0.0, 0.0], &[1.0, 0.0], &[2.0, 0.0]];
        let b: Vec<&[f32]> = vec![&[0.0, 1.0], &[1.0, 1.0], &[2.0, 1.0]];
        let d = discrete_frechet(&a, &b);
        // Parallel paths offset by 1.0 → Fréchet = 1.0
        assert!((d - 1.0).abs() < 1e-10, "parallel offset should be 1.0, got {d}");
    }

    #[test]
    fn frechet_different_lengths() {
        let a: Vec<&[f32]> = vec![&[0.0], &[1.0], &[2.0]];
        let b: Vec<&[f32]> = vec![&[0.0], &[2.0]];
        let d = discrete_frechet(&a, &b);
        // b jumps from 0→2 while a goes 0→1→2. Fréchet = 1.0 (at a[1], b[0] or b[1])
        assert!(d <= 1.0 + 1e-10, "should handle unequal lengths, got {d}");
    }

    #[test]
    fn frechet_preserves_order() {
        // Path A: right then up
        let a: Vec<&[f32]> = vec![&[0.0, 0.0], &[1.0, 0.0], &[1.0, 1.0]];
        // Path B: up then right (reversed order)
        let b: Vec<&[f32]> = vec![&[0.0, 0.0], &[0.0, 1.0], &[1.0, 1.0]];
        let d = discrete_frechet(&a, &b);
        // Same endpoints but different paths → nonzero Fréchet
        assert!(d > 0.5, "different-order paths should have positive distance, got {d}");
    }

    #[test]
    fn frechet_empty_paths() {
        let a: Vec<&[f32]> = vec![];
        let b: Vec<&[f32]> = vec![&[0.0]];
        assert!(discrete_frechet(&a, &b).is_infinite());
    }

    #[test]
    fn frechet_temporal_wrapper() {
        let a: Vec<(i64, &[f32])> = vec![(0, &[0.0f32, 0.0] as &[f32]), (1, &[1.0f32, 1.0])];
        let b: Vec<(i64, &[f32])> = vec![(0, &[0.0f32, 0.0] as &[f32]), (1, &[1.0f32, 1.0])];
        let d = discrete_frechet_temporal(&a, &b);
        assert!(d < 1e-10);
    }
}
