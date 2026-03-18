//! Anchor-relative trajectory analysis.
//!
//! Projects trajectories from absolute embedding space (ℝᴰ) into a reference
//! frame defined by K anchor vectors, producing trajectories in ℝᴷ where each
//! dimension is the cosine distance to an anchor.
//!
//! This enables all existing CVX analytics (velocity, Hurst, changepoints,
//! signatures) to operate on clinically or semantically meaningful coordinates
//! instead of opaque embedding dimensions.
//!
//! # Example
//!
//! ```
//! use cvx_analytics::anchor::{project_to_anchors, AnchorMetric};
//!
//! let trajectory = vec![
//!     (1000_i64, vec![1.0_f32, 0.0, 0.0]),
//!     (2000, vec![0.9, 0.1, 0.0]),
//!     (3000, vec![0.5, 0.5, 0.0]),
//! ];
//! let anchors = vec![
//!     vec![1.0_f32, 0.0, 0.0],  // anchor A
//!     vec![0.0, 1.0, 0.0],      // anchor B
//! ];
//!
//! let traj_refs: Vec<(i64, &[f32])> = trajectory.iter().map(|(t, v)| (*t, v.as_slice())).collect();
//! let anchor_refs: Vec<&[f32]> = anchors.iter().map(|a| a.as_slice()).collect();
//!
//! let projected = project_to_anchors(&traj_refs, &anchor_refs, AnchorMetric::Cosine);
//! // projected[0] = (1000, [0.0, 1.0])   -- close to A, far from B
//! // projected[2] = (3000, [~0.3, ~0.3]) -- equidistant
//! ```

use crate::calculus;

/// Distance metric for anchor projection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnchorMetric {
    /// Cosine distance: 1 - cos(v, anchor). Range [0, 2], typically [0, 1].
    Cosine,
    /// L2 (Euclidean) distance.
    L2,
}

/// Project a trajectory into anchor-relative coordinates.
///
/// For a trajectory of N points in ℝᴰ and K anchors, produces N points in ℝᴷ
/// where dimension k = distance(point, anchor_k).
///
/// The resulting trajectory can be fed into any CVX analytics function
/// (velocity, Hurst, changepoints, signatures) for anchor-relative analysis.
pub fn project_to_anchors(
    trajectory: &[(i64, &[f32])],
    anchors: &[&[f32]],
    metric: AnchorMetric,
) -> Vec<(i64, Vec<f32>)> {
    trajectory
        .iter()
        .map(|&(ts, vec)| {
            let distances: Vec<f32> = anchors
                .iter()
                .map(|anchor| match metric {
                    AnchorMetric::Cosine => calculus::drift_magnitude_cosine(vec, anchor),
                    AnchorMetric::L2 => calculus::drift_magnitude_l2(vec, anchor),
                })
                .collect();
            (ts, distances)
        })
        .collect()
}

/// Summary statistics for anchor proximity over a trajectory.
#[derive(Debug, Clone)]
pub struct AnchorSummary {
    /// Mean distance to each anchor over the trajectory.
    pub mean: Vec<f32>,
    /// Minimum distance (closest approach) to each anchor.
    pub min: Vec<f32>,
    /// Linear trend (slope) of distance to each anchor.
    /// Negative = approaching the anchor over time.
    pub trend: Vec<f32>,
    /// Distance at the last time point.
    pub last: Vec<f32>,
}

/// Compute summary statistics of anchor proximity over a trajectory.
pub fn anchor_summary(projected: &[(i64, Vec<f32>)]) -> AnchorSummary {
    if projected.is_empty() {
        return AnchorSummary {
            mean: vec![],
            min: vec![],
            trend: vec![],
            last: vec![],
        };
    }

    let k = projected[0].1.len();
    let n = projected.len();

    let mut mean = vec![0.0f32; k];
    let mut min = vec![f32::INFINITY; k];
    let mut last = vec![0.0f32; k];

    for (_, dists) in projected {
        for (j, &d) in dists.iter().enumerate() {
            mean[j] += d;
            if d < min[j] {
                min[j] = d;
            }
        }
    }

    for j in 0..k {
        mean[j] /= n as f32;
    }

    if let Some((_, d)) = projected.last() {
        last = d.clone();
    }

    // Linear trend via least squares: slope = (Σ(x-x̄)(y-ȳ)) / Σ(x-x̄)²
    let x_mean = (n as f64 - 1.0) / 2.0;
    let x_var: f64 = (0..n).map(|i| (i as f64 - x_mean).powi(2)).sum();

    let trend = if x_var > 0.0 {
        (0..k)
            .map(|j| {
                let y_mean = mean[j] as f64;
                let covar: f64 = projected
                    .iter()
                    .enumerate()
                    .map(|(i, (_, dists))| (i as f64 - x_mean) * (dists[j] as f64 - y_mean))
                    .sum();
                (covar / x_var) as f32
            })
            .collect()
    } else {
        vec![0.0; k]
    };

    AnchorSummary {
        mean,
        min,
        trend,
        last,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_cosine_identity() {
        // A vector projected to itself should have distance 0
        let traj = vec![(0i64, [1.0f32, 0.0, 0.0].as_slice())];
        let anchors = vec![[1.0f32, 0.0, 0.0].as_slice()];
        let result = project_to_anchors(&traj, &anchors, AnchorMetric::Cosine);
        assert!(result[0].1[0] < 1e-6, "self-distance should be ~0");
    }

    #[test]
    fn project_cosine_orthogonal() {
        let traj = vec![(0i64, [1.0f32, 0.0, 0.0].as_slice())];
        let anchors = vec![[0.0f32, 1.0, 0.0].as_slice()];
        let result = project_to_anchors(&traj, &anchors, AnchorMetric::Cosine);
        assert!(
            (result[0].1[0] - 1.0).abs() < 1e-6,
            "orthogonal should be 1.0"
        );
    }

    #[test]
    fn project_multiple_anchors() {
        let traj = vec![(0i64, [1.0f32, 0.0].as_slice()), (1, [0.0, 1.0].as_slice())];
        let anchors = vec![[1.0f32, 0.0].as_slice(), [0.0, 1.0].as_slice()];
        let result = project_to_anchors(&traj, &anchors, AnchorMetric::Cosine);
        // t=0: close to anchor 0, far from anchor 1
        assert!(result[0].1[0] < 0.1);
        assert!(result[0].1[1] > 0.9);
        // t=1: far from anchor 0, close to anchor 1
        assert!(result[1].1[0] > 0.9);
        assert!(result[1].1[1] < 0.1);
    }

    #[test]
    fn summary_trend_approaching() {
        // Trajectory approaching anchor over time
        let projected = vec![
            (0i64, vec![1.0f32]),
            (1, vec![0.8]),
            (2, vec![0.6]),
            (3, vec![0.4]),
            (4, vec![0.2]),
        ];
        let summary = anchor_summary(&projected);
        assert!(
            summary.trend[0] < 0.0,
            "should have negative trend (approaching)"
        );
        assert!((summary.mean[0] - 0.6).abs() < 0.01);
        assert!((summary.min[0] - 0.2).abs() < 0.01);
    }

    #[test]
    fn summary_empty() {
        let summary = anchor_summary(&[]);
        assert!(summary.mean.is_empty());
    }
}
