//! Fisher-Rao distance on the statistical manifold.
//!
//! The Fisher-Rao metric is the unique Riemannian metric (up to scale)
//! on the space of probability distributions that is invariant under
//! sufficient statistics (Chentsov's theorem). It is the mathematically
//! correct way to measure distance between distributions.
//!
//! For categorical distributions (region proportions), the Fisher-Rao
//! distance has a closed-form solution via the Bhattacharyya angle:
//!
//! $$d_{FR}(p, q) = 2 \arccos\left(\sum_i \sqrt{p_i \cdot q_i}\right)$$
//!
//! # Why Fisher-Rao for CVX
//!
//! Region distributions at different timestamps are points on a statistical
//! manifold. The Fisher-Rao distance is the geodesic distance on this manifold —
//! more principled than L2, KL divergence, or even Wasserstein for measuring
//! distributional change.
//!
//! # Comparison with Other Distances
//!
//! | Distance | Symmetric | Metric | Invariant | Bounded |
//! |----------|:---------:|:------:|:---------:|:-------:|
//! | KL divergence | No | No | Yes | No |
//! | L2 | Yes | Yes | No | No |
//! | Wasserstein | Yes | Yes | No | No |
//! | **Fisher-Rao** | **Yes** | **Yes** | **Yes** | **Yes** [0, π] |
//!
//! # References
//!
//! - Rao, C.R. (1945). Information and accuracy attainable in estimation.
//! - Chentsov, N.N. (1982). *Statistical Decision Rules and Optimal Inference*.

/// Fisher-Rao distance between two categorical distributions.
///
/// Uses the Bhattacharyya angle: d(p, q) = 2 × arccos(Σ √(p_i × q_i)).
/// Both inputs must be non-negative and should sum to ~1.0.
///
/// Returns a value in [0, π]:
/// - 0 = identical distributions
/// - π = completely orthogonal (disjoint support)
pub fn fisher_rao_distance(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "distributions must have equal length");

    let bhattacharyya_coeff: f64 = p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| (pi.max(0.0) * qi.max(0.0)).sqrt())
        .sum();

    // Clamp to [0, 1] for numerical stability before arccos
    2.0 * bhattacharyya_coeff.clamp(0.0, 1.0).acos()
}

/// Fisher-Rao distance for f32 distributions (convenience wrapper).
pub fn fisher_rao_distance_f32(p: &[f32], q: &[f32]) -> f64 {
    let p64: Vec<f64> = p.iter().map(|&v| v as f64).collect();
    let q64: Vec<f64> = q.iter().map(|&v| v as f64).collect();
    fisher_rao_distance(&p64, &q64)
}

/// Bhattacharyya coefficient: BC(p, q) = Σ √(p_i × q_i).
///
/// BC = 1 for identical distributions, BC = 0 for disjoint support.
/// Related to Fisher-Rao: d_FR = 2 × arccos(BC).
pub fn bhattacharyya_coefficient(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len());
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| (pi.max(0.0) * qi.max(0.0)).sqrt())
        .sum()
}

/// Hellinger distance: H(p, q) = √(1 - BC(p, q)) / √2.
///
/// Closely related to Fisher-Rao but maps to [0, 1].
/// Often used when a [0, 1] range is preferred over [0, π].
pub fn hellinger_distance(p: &[f64], q: &[f64]) -> f64 {
    let bc = bhattacharyya_coefficient(p, q);
    ((1.0 - bc).max(0.0) / 2.0).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn identical_distributions_zero() {
        let p = vec![0.3, 0.5, 0.2];
        assert!(fisher_rao_distance(&p, &p) < 1e-10);
    }

    #[test]
    fn disjoint_distributions_pi() {
        let p = vec![1.0, 0.0, 0.0];
        let q = vec![0.0, 1.0, 0.0];
        let d = fisher_rao_distance(&p, &q);
        assert!((d - PI).abs() < 1e-10, "disjoint should be π, got {d}");
    }

    #[test]
    fn symmetric() {
        let p = vec![0.4, 0.3, 0.3];
        let q = vec![0.1, 0.6, 0.3];
        let d_pq = fisher_rao_distance(&p, &q);
        let d_qp = fisher_rao_distance(&q, &p);
        assert!((d_pq - d_qp).abs() < 1e-10);
    }

    #[test]
    fn triangle_inequality() {
        let p = vec![0.5, 0.3, 0.2];
        let q = vec![0.1, 0.4, 0.5];
        let r = vec![0.3, 0.3, 0.4];
        let d_pq = fisher_rao_distance(&p, &q);
        let d_qr = fisher_rao_distance(&q, &r);
        let d_pr = fisher_rao_distance(&p, &r);
        assert!(
            d_pr <= d_pq + d_qr + 1e-10,
            "triangle: d(p,r)={d_pr} > d(p,q)+d(q,r)={}",
            d_pq + d_qr
        );
    }

    #[test]
    fn bounded_zero_to_pi() {
        let cases: Vec<(Vec<f64>, Vec<f64>)> = vec![
            (vec![0.5, 0.5], vec![0.5, 0.5]),
            (vec![1.0, 0.0], vec![0.0, 1.0]),
            (vec![0.9, 0.1], vec![0.1, 0.9]),
            (vec![0.25, 0.25, 0.25, 0.25], vec![0.7, 0.1, 0.1, 0.1]),
        ];
        for (p, q) in &cases {
            let d = fisher_rao_distance(p, q);
            assert!(d >= 0.0 && d <= PI + 1e-10, "d={d} out of [0, π]");
        }
    }

    #[test]
    fn bhattacharyya_coefficient_range() {
        let p = vec![0.3, 0.7];
        let q = vec![0.6, 0.4];
        let bc = bhattacharyya_coefficient(&p, &q);
        assert!(bc >= 0.0 && bc <= 1.0, "BC={bc} out of [0, 1]");
    }

    #[test]
    fn hellinger_range() {
        let p = vec![0.3, 0.7];
        let q = vec![0.6, 0.4];
        let h = hellinger_distance(&p, &q);
        assert!(h >= 0.0 && h <= 1.0, "H={h} out of [0, 1]");
    }

    #[test]
    fn f32_wrapper_matches() {
        let p32 = vec![0.3f32, 0.5, 0.2];
        let q32 = vec![0.1f32, 0.6, 0.3];
        let p64: Vec<f64> = p32.iter().map(|&v| v as f64).collect();
        let q64: Vec<f64> = q32.iter().map(|&v| v as f64).collect();
        let d32 = fisher_rao_distance_f32(&p32, &q32);
        let d64 = fisher_rao_distance(&p64, &q64);
        assert!((d32 - d64).abs() < 1e-6);
    }
}
