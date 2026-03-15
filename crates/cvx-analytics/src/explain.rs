//! Interpretability layer: drift attribution, trajectory projection, dimension heatmaps.
//!
//! Provides human-readable explanations of temporal vector evolution:
//! - **DriftAttribution**: per-dimension drift analysis with Pareto ranking
//! - **TrajectoryProjector**: PCA-based 2D/3D projection
//! - **DimensionHeatmap**: time × dimension change intensity matrix

use cvx_core::error::AnalyticsError;

// ─── Drift Attribution ──────────────────────────────────────────────

/// Per-dimension drift attribution.
#[derive(Debug, Clone)]
pub struct DriftAttribution {
    /// Total L2 drift magnitude.
    pub total_magnitude: f32,
    /// Per-dimension: `(dim_index, absolute_change, relative_contribution)`.
    pub dimensions: Vec<DimensionContribution>,
}

/// A single dimension's contribution to overall drift.
#[derive(Debug, Clone)]
pub struct DimensionContribution {
    /// Dimension index.
    pub index: usize,
    /// Absolute change in this dimension.
    pub absolute_change: f32,
    /// Fraction of total L2² explained by this dimension.
    pub relative_contribution: f32,
}

/// Compute drift attribution between two vectors.
///
/// Returns dimensions sorted by contribution (highest first).
pub fn drift_attribution(v1: &[f32], v2: &[f32], top_k: usize) -> DriftAttribution {
    assert_eq!(v1.len(), v2.len(), "dimension mismatch");

    let changes: Vec<f32> = v1.iter().zip(v2.iter()).map(|(a, b)| b - a).collect();
    let total_sq: f32 = changes.iter().map(|c| c * c).sum();
    let total_magnitude = total_sq.sqrt();

    let mut dims: Vec<DimensionContribution> = changes
        .iter()
        .enumerate()
        .map(|(i, &c)| DimensionContribution {
            index: i,
            absolute_change: c.abs(),
            relative_contribution: if total_sq > 0.0 {
                (c * c) / total_sq
            } else {
                0.0
            },
        })
        .collect();

    dims.sort_by(|a, b| b.absolute_change.partial_cmp(&a.absolute_change).unwrap());
    dims.truncate(top_k);

    DriftAttribution {
        total_magnitude,
        dimensions: dims,
    }
}

// ─── Trajectory Projection (PCA) ───────────────────────────────────

/// A 2D projected point.
#[derive(Debug, Clone)]
pub struct ProjectedPoint {
    /// Original timestamp.
    pub timestamp: i64,
    /// First principal component coordinate.
    pub x: f32,
    /// Second principal component coordinate.
    pub y: f32,
}

/// Project a trajectory to 2D using PCA.
///
/// Computes the two principal components of the trajectory vectors
/// and projects each point onto them.
pub fn project_trajectory_2d(
    trajectory: &[(i64, &[f32])],
) -> Result<Vec<ProjectedPoint>, AnalyticsError> {
    if trajectory.len() < 2 {
        return Err(AnalyticsError::InsufficientData {
            needed: 2,
            have: trajectory.len(),
        });
    }

    let n = trajectory.len();
    let dim = trajectory[0].1.len();

    // Compute mean
    let mut mean = vec![0.0f64; dim];
    for (_, v) in trajectory {
        for d in 0..dim {
            mean[d] += v[d] as f64;
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }

    // Center the data
    let centered: Vec<Vec<f64>> = trajectory
        .iter()
        .map(|(_, v)| (0..dim).map(|d| v[d] as f64 - mean[d]).collect())
        .collect();

    // Power iteration to find top 2 eigenvectors of covariance matrix
    let pc1 = power_iteration(&centered, dim, None);
    let pc2 = power_iteration(&centered, dim, Some(&pc1));

    // Project
    let projected: Vec<ProjectedPoint> = trajectory
        .iter()
        .zip(centered.iter())
        .map(|((ts, _), c)| {
            let x: f64 = c.iter().zip(pc1.iter()).map(|(a, b)| a * b).sum();
            let y: f64 = c.iter().zip(pc2.iter()).map(|(a, b)| a * b).sum();
            ProjectedPoint {
                timestamp: *ts,
                x: x as f32,
                y: y as f32,
            }
        })
        .collect();

    Ok(projected)
}

/// Power iteration for finding a principal component.
///
/// If `deflate_against` is provided, the component is orthogonalized
/// against it (for finding the second PC).
fn power_iteration(data: &[Vec<f64>], dim: usize, deflate_against: Option<&[f64]>) -> Vec<f64> {
    let max_iter = 200;
    let tol = 1e-10;

    // Initialize with a vector not parallel to the deflation direction
    let mut v: Vec<f64> = if let Some(prev) = deflate_against {
        // Start with a vector orthogonal to the previous PC
        // Use the second data point or a unit vector in a different direction
        let mut init = if data.len() > 1 {
            data[1].clone()
        } else {
            let mut u = vec![0.0; dim];
            if dim > 1 {
                u[1] = 1.0;
            }
            u
        };
        // Orthogonalize against prev
        let dot: f64 = init.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
        for d in 0..dim {
            init[d] -= dot * prev[d];
        }
        let norm: f64 = init.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > tol {
            init.iter().map(|x| x / norm).collect()
        } else {
            // All data points parallel — try unit vectors
            let mut fallback = vec![0.0; dim];
            for d in 0..dim {
                fallback[d] = 1.0;
                let dot: f64 = fallback.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
                for dd in 0..dim {
                    fallback[dd] -= dot * prev[dd];
                }
                let norm: f64 = fallback.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > tol {
                    return fallback.iter().map(|x| x / norm).collect();
                }
                fallback = vec![0.0; dim];
            }
            fallback
        }
    } else if !data.is_empty() && data[0].iter().any(|&x| x != 0.0) {
        let norm: f64 = data[0].iter().map(|x| x * x).sum::<f64>().sqrt();
        data[0].iter().map(|x| x / norm).collect()
    } else {
        let mut init = vec![0.0; dim];
        if dim > 0 {
            init[0] = 1.0;
        }
        init
    };

    for _ in 0..max_iter {
        // Compute Cov * v = (1/n) Σ (x_i * (x_i · v))
        let mut new_v = vec![0.0f64; dim];
        for row in data {
            let dot: f64 = row.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            for d in 0..dim {
                new_v[d] += row[d] * dot;
            }
        }

        // Deflate against previous component if needed
        if let Some(prev) = deflate_against {
            let dot: f64 = new_v.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
            for d in 0..dim {
                new_v[d] -= dot * prev[d];
            }
        }

        // Normalize
        let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < tol {
            return v;
        }
        for x in &mut new_v {
            *x /= norm;
        }

        // Check convergence
        let diff: f64 = v
            .iter()
            .zip(new_v.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();

        v = new_v;
        if diff < tol {
            break;
        }
    }

    v
}

// ─── Dimension Heatmap ──────────────────────────────────────────────

/// Time × dimension change intensity matrix.
#[derive(Debug, Clone)]
pub struct DimensionHeatmap {
    /// Number of time bins.
    pub time_bins: usize,
    /// Number of dimensions.
    pub dimensions: usize,
    /// Timestamps for each time bin.
    pub timestamps: Vec<i64>,
    /// Flattened matrix: `[time_bin * dimensions + dim]` = change intensity.
    pub data: Vec<f32>,
}

impl DimensionHeatmap {
    /// Get the change intensity at a specific time bin and dimension.
    pub fn get(&self, time_bin: usize, dim: usize) -> f32 {
        self.data[time_bin * self.dimensions + dim]
    }
}

/// Build a dimension heatmap from a trajectory.
///
/// Computes the absolute per-dimension change between consecutive points.
pub fn dimension_heatmap(trajectory: &[(i64, &[f32])]) -> Result<DimensionHeatmap, AnalyticsError> {
    if trajectory.len() < 2 {
        return Err(AnalyticsError::InsufficientData {
            needed: 2,
            have: trajectory.len(),
        });
    }

    let dim = trajectory[0].1.len();
    let time_bins = trajectory.len() - 1;
    let mut data = Vec::with_capacity(time_bins * dim);
    let mut timestamps = Vec::with_capacity(time_bins);

    for w in trajectory.windows(2) {
        timestamps.push(w[1].0);
        for d in 0..dim {
            data.push((w[1].1[d] - w[0].1[d]).abs());
        }
    }

    Ok(DimensionHeatmap {
        time_bins,
        dimensions: dim,
        timestamps,
        data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trajectory(points: &[(i64, Vec<f32>)]) -> Vec<(i64, &[f32])> {
        points.iter().map(|(t, v)| (*t, v.as_slice())).collect()
    }

    // ─── Drift Attribution ──────────────────────────────────────────

    #[test]
    fn attribution_identifies_top_dimensions() {
        let v1 = vec![0.0; 10];
        let mut v2 = vec![0.0; 10];
        v2[3] = 5.0; // dim 3 changed most
        v2[7] = 3.0; // dim 7 second

        let attr = drift_attribution(&v1, &v2, 5);
        assert_eq!(attr.dimensions[0].index, 3);
        assert_eq!(attr.dimensions[1].index, 7);
        assert!((attr.dimensions[0].absolute_change - 5.0).abs() < 1e-6);
    }

    #[test]
    fn attribution_relative_contributions_sum_to_1() {
        let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let v2 = vec![2.0, 3.0, 1.0, 6.0, 4.0];

        let attr = drift_attribution(&v1, &v2, 5);
        let total: f32 = attr
            .dimensions
            .iter()
            .map(|d| d.relative_contribution)
            .sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "contributions should sum to 1.0, got {total}"
        );
    }

    #[test]
    fn attribution_zero_drift() {
        let v = vec![1.0, 2.0, 3.0];
        let attr = drift_attribution(&v, &v, 3);
        assert!(attr.total_magnitude < 1e-6);
    }

    // ─── PCA Projection ─────────────────────────────────────────────

    #[test]
    fn pca_projection_produces_2d() {
        let points: Vec<(i64, Vec<f32>)> = (0..100)
            .map(|i| {
                let t = i as f32 * 0.1;
                (i as i64 * 1000, vec![t.cos(), t.sin(), t * 0.1])
            })
            .collect();
        let traj = make_trajectory(&points);

        let projected = project_trajectory_2d(&traj).unwrap();
        assert_eq!(projected.len(), 100);

        // All points should have finite coordinates
        for p in &projected {
            assert!(p.x.is_finite(), "x should be finite");
            assert!(p.y.is_finite(), "y should be finite");
        }
    }

    #[test]
    fn pca_projection_insufficient_data() {
        let points = vec![(0i64, vec![1.0, 2.0])];
        let traj = make_trajectory(&points);
        assert!(project_trajectory_2d(&traj).is_err());
    }

    #[test]
    fn pca_preserves_variance() {
        // 2D signal + noise: PC1 should capture the main direction
        let points: Vec<(i64, Vec<f32>)> = (0..50)
            .map(|i| {
                let t = i as f32;
                // Strong signal in dim 0, weak noise in dim 1, zero in dim 2
                (i as i64 * 1000, vec![t * 2.0, t * 0.1, 0.0])
            })
            .collect();
        let traj = make_trajectory(&points);

        let projected = project_trajectory_2d(&traj).unwrap();
        let var_x: f64 =
            projected.iter().map(|p| (p.x as f64).powi(2)).sum::<f64>() / projected.len() as f64;
        let var_y: f64 =
            projected.iter().map(|p| (p.y as f64).powi(2)).sum::<f64>() / projected.len() as f64;

        // PC1 should capture most variance
        assert!(
            var_x > var_y * 5.0,
            "PC1 variance ({var_x:.2}) should dominate PC2 ({var_y:.2})"
        );
    }

    // ─── Dimension Heatmap ──────────────────────────────────────────

    #[test]
    fn heatmap_dimensions_correct() {
        let points: Vec<(i64, Vec<f32>)> = (0..10)
            .map(|i| (i as i64 * 1000, vec![i as f32; 5]))
            .collect();
        let traj = make_trajectory(&points);

        let heatmap = dimension_heatmap(&traj).unwrap();
        assert_eq!(heatmap.time_bins, 9); // 10 points → 9 transitions
        assert_eq!(heatmap.dimensions, 5);
        assert_eq!(heatmap.data.len(), 9 * 5);
        assert_eq!(heatmap.timestamps.len(), 9);
    }

    #[test]
    fn heatmap_stationary_is_zero() {
        let points: Vec<(i64, Vec<f32>)> = (0..5)
            .map(|i| (i as i64 * 1000, vec![1.0, 2.0, 3.0]))
            .collect();
        let traj = make_trajectory(&points);

        let heatmap = dimension_heatmap(&traj).unwrap();
        for &v in &heatmap.data {
            assert!(v < 1e-6, "stationary entity heatmap should be zero");
        }
    }

    #[test]
    fn heatmap_detects_change_in_specific_dim() {
        let points = vec![
            (0i64, vec![0.0, 0.0, 0.0]),
            (1000, vec![0.0, 5.0, 0.0]), // only dim 1 changes
        ];
        let traj = make_trajectory(&points);

        let heatmap = dimension_heatmap(&traj).unwrap();
        assert!(heatmap.get(0, 0) < 1e-6); // dim 0 unchanged
        assert!((heatmap.get(0, 1) - 5.0).abs() < 1e-6); // dim 1 changed by 5
        assert!(heatmap.get(0, 2) < 1e-6); // dim 2 unchanged
    }

    #[test]
    fn heatmap_insufficient_data() {
        let points = vec![(0i64, vec![1.0])];
        let traj = make_trajectory(&points);
        assert!(dimension_heatmap(&traj).is_err());
    }
}
