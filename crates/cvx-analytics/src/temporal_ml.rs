//! Differentiable temporal feature extraction.
//!
//! Defines the `TemporalOps` trait for computing fixed-size feature vectors
//! from variable-length embedding trajectories. The `AnalyticBackend`
//! implementation uses closed-form computations (no ML framework needed).
//!
//! Future backends (BurnBackend, TorchBackend) will implement the same
//! trait with learnable parameters for end-to-end training.
//!
//! # Feature Vector Layout
//!
//! The output is a fixed-size feature vector regardless of trajectory length:
//!
//! | Section | Size | Description |
//! |---------|------|-------------|
//! | Mean velocity | D | Average velocity per dimension |
//! | Drift magnitude | 1 | L2 drift from first to last |
//! | Volatility | D | Per-dimension realized volatility |
//! | Soft change point count | 1 | Smoothed number of regime changes |
//! | Multi-scale drift | S | Drift at S temporal scales |
//!
//! Total: `2*D + 2 + S` features.

use cvx_core::error::AnalyticsError;

/// Temporal feature extraction operations.
///
/// All implementations must produce the same feature vector layout
/// for the same input, differing only in differentiability.
pub trait TemporalOps {
    /// Extract a fixed-size feature vector from a trajectory.
    ///
    /// Returns a vector of size `feature_dim()`.
    fn extract_features(&self, trajectory: &[(i64, &[f32])]) -> Result<Vec<f32>, AnalyticsError>;

    /// Output feature dimensionality.
    fn feature_dim(&self, input_dim: usize) -> usize;

    /// Backend name.
    fn name(&self) -> &str;
}

/// Configuration for the analytic backend.
#[derive(Debug, Clone)]
pub struct AnalyticConfig {
    /// Temporal scales (bucket widths in microseconds) for multi-scale drift.
    pub scales: Vec<i64>,
    /// Threshold for soft change point detection (z-score).
    pub cp_threshold: f32,
    /// Smoothing window for soft change point count.
    pub cp_smoothing: usize,
}

impl Default for AnalyticConfig {
    fn default() -> Self {
        Self {
            scales: vec![1_000_000, 10_000_000, 100_000_000], // 1s, 10s, 100s
            cp_threshold: 3.0,
            cp_smoothing: 5,
        }
    }
}

/// Pure-Rust analytic feature extractor (non-differentiable).
///
/// Computes all temporal features using closed-form expressions.
/// This is the baseline and fallback when no ML backend is available.
pub struct AnalyticBackend {
    config: AnalyticConfig,
}

impl AnalyticBackend {
    /// Create with default config.
    pub fn new() -> Self {
        Self {
            config: AnalyticConfig::default(),
        }
    }

    /// Create with custom config.
    pub fn with_config(config: AnalyticConfig) -> Self {
        Self { config }
    }
}

impl Default for AnalyticBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalOps for AnalyticBackend {
    fn extract_features(&self, trajectory: &[(i64, &[f32])]) -> Result<Vec<f32>, AnalyticsError> {
        if trajectory.len() < 2 {
            return Err(AnalyticsError::InsufficientData {
                needed: 2,
                have: trajectory.len(),
            });
        }

        let dim = trajectory[0].1.len();
        let n = trajectory.len();
        let n_scales = self.config.scales.len();
        let total_features = 2 * dim + 2 + n_scales;
        let mut features = Vec::with_capacity(total_features);

        // 1. Mean velocity (D features)
        let total_dt = (trajectory[n - 1].0 - trajectory[0].0).max(1) as f64;
        for d in 0..dim {
            let total_displacement = trajectory[n - 1].1[d] as f64 - trajectory[0].1[d] as f64;
            features.push((total_displacement / total_dt) as f32);
        }

        // 2. Drift magnitude (1 feature)
        let drift: f32 = (0..dim)
            .map(|d| {
                let diff = trajectory[n - 1].1[d] - trajectory[0].1[d];
                diff * diff
            })
            .sum::<f32>()
            .sqrt();
        features.push(drift);

        // 3. Volatility (D features)
        if n > 2 {
            let mut vol = vec![0.0f64; dim];
            let mut means = vec![0.0f64; dim];

            // Compute returns
            let returns: Vec<Vec<f64>> = trajectory
                .windows(2)
                .map(|w| {
                    let dt = (w[1].0 - w[0].0).max(1) as f64;
                    (0..dim)
                        .map(|d| (w[1].1[d] as f64 - w[0].1[d] as f64) / dt)
                        .collect()
                })
                .collect();

            for ret in &returns {
                for d in 0..dim {
                    means[d] += ret[d];
                }
            }
            let nr = returns.len() as f64;
            for m in &mut means {
                *m /= nr;
            }

            for ret in &returns {
                for d in 0..dim {
                    let diff = ret[d] - means[d];
                    vol[d] += diff * diff;
                }
            }

            for v in &vol {
                features.push((v / (nr - 1.0).max(1.0)).sqrt() as f32);
            }
        } else {
            features.extend(vec![0.0f32; dim]);
        }

        // 4. Soft change point count (1 feature)
        let cp_count = soft_change_point_count(
            trajectory,
            self.config.cp_threshold,
            self.config.cp_smoothing,
        );
        features.push(cp_count);

        // 5. Multi-scale drift (S features)
        for &scale in &self.config.scales {
            let scale_drift = compute_scale_drift(trajectory, scale);
            features.push(scale_drift);
        }

        Ok(features)
    }

    fn feature_dim(&self, input_dim: usize) -> usize {
        2 * input_dim + 2 + self.config.scales.len()
    }

    fn name(&self) -> &str {
        "analytic"
    }
}

/// Soft change point count: number of points where the local deviation
/// exceeds `threshold` standard deviations, smoothed by a running window.
fn soft_change_point_count(trajectory: &[(i64, &[f32])], threshold: f32, smoothing: usize) -> f32 {
    if trajectory.len() < 3 {
        return 0.0;
    }

    let dim = trajectory[0].1.len();
    // Compute per-step L2 displacement
    let displacements: Vec<f32> = trajectory
        .windows(2)
        .map(|w| {
            (0..dim)
                .map(|d| {
                    let diff = w[1].1[d] - w[0].1[d];
                    diff * diff
                })
                .sum::<f32>()
                .sqrt()
        })
        .collect();

    // Running mean and std of displacements
    let mean: f32 = displacements.iter().sum::<f32>() / displacements.len() as f32;
    let var: f32 = displacements
        .iter()
        .map(|d| (d - mean) * (d - mean))
        .sum::<f32>()
        / displacements.len() as f32;
    let std = var.sqrt().max(1e-10);

    // Count z-score exceedances with smoothing
    let mut count = 0.0f32;
    let mut cooldown = 0usize;

    for &d in &displacements {
        if cooldown > 0 {
            cooldown -= 1;
            continue;
        }
        let z = (d - mean) / std;
        if z > threshold {
            count += 1.0;
            cooldown = smoothing;
        }
    }

    count
}

/// Average drift magnitude at a given temporal scale.
fn compute_scale_drift(trajectory: &[(i64, &[f32])], bucket_width: i64) -> f32 {
    if trajectory.len() < 2 || bucket_width <= 0 {
        return 0.0;
    }

    let dim = trajectory[0].1.len();
    let t_min = trajectory[0].0;
    let t_max = trajectory.last().unwrap().0;

    let mut total_drift = 0.0f32;
    let mut n_buckets = 0;

    let mut bucket_start = t_min;
    let mut prev_last: Option<&[f32]> = None;

    while bucket_start <= t_max {
        let bucket_end = bucket_start + bucket_width;

        // Find last point in bucket
        let last_in_bucket = trajectory
            .iter()
            .rev()
            .find(|(t, _)| *t >= bucket_start && *t < bucket_end);

        if let Some((_, v)) = last_in_bucket {
            if let Some(prev) = prev_last {
                let drift: f32 = (0..dim)
                    .map(|d| {
                        let diff = v[d] - prev[d];
                        diff * diff
                    })
                    .sum::<f32>()
                    .sqrt();
                total_drift += drift;
                n_buckets += 1;
            }
            prev_last = Some(v);
        }

        bucket_start = bucket_end;
    }

    if n_buckets > 0 {
        total_drift / n_buckets as f32
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trajectory(points: &[(i64, Vec<f32>)]) -> Vec<(i64, &[f32])> {
        points.iter().map(|(t, v)| (*t, v.as_slice())).collect()
    }

    // ─── Feature extraction ─────────────────────────────────────────

    #[test]
    fn feature_dim_correct() {
        let backend = AnalyticBackend::new();
        // D=3, S=3 → 2*3 + 2 + 3 = 11
        assert_eq!(backend.feature_dim(3), 11);
    }

    #[test]
    fn feature_vector_has_correct_size() {
        let backend = AnalyticBackend::new();
        let points: Vec<(i64, Vec<f32>)> = (0..20)
            .map(|i| (i as i64 * 1_000_000, vec![i as f32; 4]))
            .collect();
        let traj = make_trajectory(&points);

        let features = backend.extract_features(&traj).unwrap();
        assert_eq!(features.len(), backend.feature_dim(4));
    }

    #[test]
    fn feature_vector_is_fixed_size() {
        let backend = AnalyticBackend::new();

        // Different trajectory lengths → same feature size
        for n in [5, 10, 50, 100] {
            let points: Vec<(i64, Vec<f32>)> = (0..n)
                .map(|i| (i as i64 * 1_000_000, vec![i as f32; 3]))
                .collect();
            let traj = make_trajectory(&points);
            let features = backend.extract_features(&traj).unwrap();
            assert_eq!(features.len(), backend.feature_dim(3), "n={n}");
        }
    }

    #[test]
    fn stationary_has_zero_velocity_and_drift() {
        let backend = AnalyticBackend::new();
        let points: Vec<(i64, Vec<f32>)> = (0..50)
            .map(|i| (i as i64 * 1_000_000, vec![1.0, 2.0, 3.0]))
            .collect();
        let traj = make_trajectory(&points);

        let features = backend.extract_features(&traj).unwrap();

        // Mean velocity (first 3 features) should be ~0
        for d in 0..3 {
            assert!(features[d].abs() < 1e-6, "velocity[{d}] = {}", features[d]);
        }
        // Drift magnitude (feature 3) should be ~0
        assert!(features[3].abs() < 1e-6, "drift = {}", features[3]);
    }

    #[test]
    fn linear_trend_has_constant_velocity() {
        let backend = AnalyticBackend::new();
        let points: Vec<(i64, Vec<f32>)> = (0..50)
            .map(|i| (i as i64 * 1_000_000, vec![i as f32 * 2.0, i as f32]))
            .collect();
        let traj = make_trajectory(&points);

        let features = backend.extract_features(&traj).unwrap();

        // Mean velocity should be [2/1e6, 1/1e6]
        assert!(
            (features[0] - 2e-6).abs() < 1e-8,
            "vel[0] = {}",
            features[0]
        );
        assert!(
            (features[1] - 1e-6).abs() < 1e-8,
            "vel[1] = {}",
            features[1]
        );

        // Drift should be > 0
        assert!(features[2] > 0.0);
    }

    #[test]
    fn volatile_trajectory_has_high_volatility() {
        let backend = AnalyticBackend::new();

        // Alternating values → high volatility
        let points: Vec<(i64, Vec<f32>)> = (0..50)
            .map(|i| {
                let v = if i % 2 == 0 { 10.0 } else { -10.0 };
                (i as i64 * 1_000_000, vec![v])
            })
            .collect();
        let traj = make_trajectory(&points);

        let features = backend.extract_features(&traj).unwrap();
        // Volatility is at index D+1 = 2 (after vel[0] and drift)
        let vol = features[2];
        assert!(
            vol > 0.0,
            "volatile trajectory should have positive volatility"
        );
    }

    #[test]
    fn soft_cp_detects_regime_change() {
        let backend = AnalyticBackend::with_config(AnalyticConfig {
            cp_threshold: 2.0,
            cp_smoothing: 3,
            ..Default::default()
        });

        let mut points: Vec<(i64, Vec<f32>)> = Vec::new();
        for i in 0..50 {
            points.push((i as i64 * 1_000_000, vec![0.0]));
        }
        for i in 50..100 {
            points.push((i as i64 * 1_000_000, vec![10.0]));
        }
        let traj = make_trajectory(&points);

        let features = backend.extract_features(&traj).unwrap();
        // Soft CP count is at index 2*D + 1 = 3
        let cp_count = features[3];
        assert!(
            cp_count >= 1.0,
            "should detect at least 1 change point, got {cp_count}"
        );
    }

    #[test]
    fn insufficient_data() {
        let backend = AnalyticBackend::new();
        let points = vec![(0i64, vec![1.0f32])];
        let traj = make_trajectory(&points);
        assert!(backend.extract_features(&traj).is_err());
    }

    #[test]
    fn backend_name() {
        let backend = AnalyticBackend::new();
        assert_eq!(backend.name(), "analytic");
    }

    // ─── Multi-scale drift ──────────────────────────────────────────

    #[test]
    fn multiscale_drift_stationary_is_zero() {
        let points: Vec<(i64, Vec<f32>)> = (0..100)
            .map(|i| (i as i64 * 1_000_000, vec![1.0, 2.0]))
            .collect();
        let traj = make_trajectory(&points);

        let drift = compute_scale_drift(&traj, 10_000_000);
        assert!(drift < 1e-6, "stationary drift = {drift}");
    }

    #[test]
    fn multiscale_drift_detects_movement() {
        let points: Vec<(i64, Vec<f32>)> = (0..100)
            .map(|i| (i as i64 * 1_000_000, vec![i as f32]))
            .collect();
        let traj = make_trajectory(&points);

        let drift = compute_scale_drift(&traj, 10_000_000);
        assert!(drift > 0.0, "moving trajectory should have positive drift");
    }
}
