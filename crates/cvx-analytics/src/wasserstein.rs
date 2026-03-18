//! Wasserstein (optimal transport) distance between distributions.
//!
//! Measures the "cost" of transforming one probability distribution into another,
//! respecting the geometry of the underlying space. Unlike L2 distance between
//! histograms, Wasserstein accounts for which bins are *close* vs *far*.
//!
//! # Why Wasserstein for CVX
//!
//! Region distributions at two timestamps are histograms over K regions.
//! L2 treats all regions as equally distant. Wasserstein uses the actual
//! distances between region centroids — shifting mass between neighboring
//! regions costs less than between distant ones.
//!
//! # Implementations
//!
//! - [`sliced_wasserstein`]: Fast approximation via random 1D projections. O(K × n_proj × K log K).
//! - [`wasserstein_1d`]: Exact W₁ on 1D distributions. O(K log K).
//! - [`emd_1d`]: Earth Mover's Distance on sorted 1D values.
//!
//! # References
//!
//! - Villani, C. (2008). *Optimal Transport: Old and New*. Springer.
//! - Bonneel, N. et al. (2015). Sliced and Radon Wasserstein barycenters. *JMIV*.

/// Exact Wasserstein-1 (Earth Mover's Distance) between two 1D distributions.
///
/// Both distributions must be non-negative and sum to the same total
/// (typically 1.0 for probability distributions).
///
/// # Complexity
///
/// O(K log K) for sorting + O(K) for the sweep.
pub fn wasserstein_1d(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "distributions must have equal length");
    let k = a.len();
    if k == 0 {
        return 0.0;
    }

    // W₁ on 1D = integral of |CDF_a - CDF_b|
    // For discrete distributions on the same support: sum of |cumsum(a) - cumsum(b)|
    let mut cum_a = 0.0;
    let mut cum_b = 0.0;
    let mut distance = 0.0;

    for i in 0..k {
        cum_a += a[i];
        cum_b += b[i];
        distance += (cum_a - cum_b).abs();
    }

    distance
}

/// Sliced Wasserstein distance between two distributions in K dimensions.
///
/// Approximates the true Wasserstein distance by projecting both distributions
/// onto random 1D lines and computing the exact W₁ on each projection.
/// The average over projections converges to the Sliced Wasserstein distance.
///
/// # Arguments
///
/// * `a` - First distribution: weights over K bins (must sum to ~1.0).
/// * `b` - Second distribution: same K bins.
/// * `centroids` - K centroid vectors (one per bin). Used for projection.
/// * `n_projections` - Number of random 1D projections (more = more accurate).
/// * `seed` - Random seed for reproducibility.
///
/// # Complexity
///
/// O(n_proj × K × (D + K log K)) where D = centroid dimensionality.
pub fn sliced_wasserstein(
    a: &[f64],
    b: &[f64],
    centroids: &[&[f32]],
    n_projections: usize,
    seed: u64,
) -> f64 {
    let k = a.len();
    assert_eq!(k, b.len(), "distributions must have equal length");
    assert_eq!(k, centroids.len(), "must have one centroid per bin");
    if k == 0 {
        return 0.0;
    }

    let dim = centroids[0].len();
    let mut total = 0.0;

    // Simple PRNG (xorshift64) for random projections
    let mut rng_state = seed;
    let mut next_rand = || -> f64 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        // Map to [-1, 1]
        (rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0
    };

    for _ in 0..n_projections {
        // Generate random unit direction
        let mut direction: Vec<f64> = (0..dim).map(|_| next_rand()).collect();
        let norm: f64 = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-10 {
            continue;
        }
        for d in &mut direction {
            *d /= norm;
        }

        // Project centroids onto this direction
        let mut projections: Vec<(f64, f64, f64)> = centroids
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let proj: f64 = c
                    .iter()
                    .zip(direction.iter())
                    .map(|(&cv, &dv)| cv as f64 * dv)
                    .sum();
                (proj, a[i], b[i])
            })
            .collect();

        // Sort by projection value
        projections.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());

        // Extract sorted distributions and compute 1D Wasserstein
        let sorted_a: Vec<f64> = projections.iter().map(|p| p.1).collect();
        let sorted_b: Vec<f64> = projections.iter().map(|p| p.2).collect();
        total += wasserstein_1d(&sorted_a, &sorted_b);
    }

    total / n_projections as f64
}

/// Compute Wasserstein drift between two region distributions.
///
/// Convenience function that wraps [`sliced_wasserstein`] for the common
/// use case of comparing region distributions at two time points.
///
/// # Arguments
///
/// * `dist_t1` - Region distribution at time T₁ (K floats summing to ~1.0).
/// * `dist_t2` - Region distribution at time T₂.
/// * `centroids` - Region centroid vectors from `index.regions(level)`.
/// * `n_projections` - Number of projections (default: 50).
pub fn wasserstein_drift(
    dist_t1: &[f32],
    dist_t2: &[f32],
    centroids: &[&[f32]],
    n_projections: usize,
) -> f64 {
    let a: Vec<f64> = dist_t1.iter().map(|&v| v as f64).collect();
    let b: Vec<f64> = dist_t2.iter().map(|&v| v as f64).collect();
    sliced_wasserstein(&a, &b, centroids, n_projections, 42)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w1d_identical_distributions() {
        let a = vec![0.25, 0.25, 0.25, 0.25];
        let b = vec![0.25, 0.25, 0.25, 0.25];
        assert!((wasserstein_1d(&a, &b)).abs() < 1e-10);
    }

    #[test]
    fn w1d_shifted_mass() {
        // All mass in first bin vs all in last
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0, 1.0];
        let d = wasserstein_1d(&a, &b);
        // Mass must travel through 3 bins: cost = 1+1+1 = 3
        assert!((d - 3.0).abs() < 1e-10, "expected 3.0, got {d}");
    }

    #[test]
    fn w1d_adjacent_shift() {
        // Mass shifts one bin to the right
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let d = wasserstein_1d(&a, &b);
        assert!((d - 1.0).abs() < 1e-10, "expected 1.0, got {d}");
    }

    #[test]
    fn w1d_symmetric() {
        let a = vec![0.5, 0.3, 0.2];
        let b = vec![0.1, 0.4, 0.5];
        assert!((wasserstein_1d(&a, &b) - wasserstein_1d(&b, &a)).abs() < 1e-10);
    }

    #[test]
    fn w1d_non_negative() {
        let a = vec![0.7, 0.2, 0.1];
        let b = vec![0.1, 0.3, 0.6];
        assert!(wasserstein_1d(&a, &b) >= 0.0);
    }

    #[test]
    fn w1d_triangle_inequality() {
        let a = vec![0.5, 0.3, 0.2];
        let b = vec![0.1, 0.4, 0.5];
        let c = vec![0.3, 0.3, 0.4];
        let d_ab = wasserstein_1d(&a, &b);
        let d_bc = wasserstein_1d(&b, &c);
        let d_ac = wasserstein_1d(&a, &c);
        assert!(
            d_ac <= d_ab + d_bc + 1e-10,
            "triangle inequality: d(a,c)={d_ac} > d(a,b)+d(b,c)={}",
            d_ab + d_bc
        );
    }

    #[test]
    fn sliced_identical_zero() {
        let a = vec![0.5, 0.3, 0.2];
        let b = vec![0.5, 0.3, 0.2];
        let centroids: Vec<&[f32]> =
            vec![&[1.0f32, 0.0] as &[f32], &[0.0f32, 1.0], &[-1.0f32, 0.0]];
        let d = sliced_wasserstein(&a, &b, &centroids, 100, 42);
        assert!(
            d < 1e-10,
            "identical distributions should have distance ~0, got {d}"
        );
    }

    #[test]
    fn sliced_different_positive() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 1.0];
        let centroids: Vec<&[f32]> =
            vec![&[1.0f32, 0.0] as &[f32], &[0.0f32, 0.0], &[-1.0f32, 0.0]];
        let d = sliced_wasserstein(&a, &b, &centroids, 100, 42);
        assert!(
            d > 0.1,
            "different distributions should have positive distance, got {d}"
        );
    }

    #[test]
    fn sliced_symmetric() {
        let a = vec![0.6, 0.3, 0.1];
        let b = vec![0.1, 0.2, 0.7];
        let centroids: Vec<&[f32]> =
            vec![&[1.0f32, 0.0] as &[f32], &[0.0f32, 1.0], &[-1.0f32, -1.0]];
        let d_ab = sliced_wasserstein(&a, &b, &centroids, 200, 42);
        let d_ba = sliced_wasserstein(&b, &a, &centroids, 200, 42);
        assert!(
            (d_ab - d_ba).abs() < 0.05,
            "should be approximately symmetric: {d_ab} vs {d_ba}"
        );
    }

    #[test]
    fn drift_convenience() {
        let dist_t1: Vec<f32> = vec![0.5, 0.3, 0.2];
        let dist_t2: Vec<f32> = vec![0.2, 0.3, 0.5];
        let centroids: Vec<&[f32]> =
            vec![&[1.0f32, 0.0] as &[f32], &[0.0f32, 1.0], &[-1.0f32, 0.0]];
        let d = wasserstein_drift(&dist_t1, &dist_t2, &centroids, 100);
        assert!(d > 0.0);
    }
}
