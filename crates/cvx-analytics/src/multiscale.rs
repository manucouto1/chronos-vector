//! Multi-scale temporal analysis: resampling and per-scale drift.
//!
//! Enables analysis at different temporal resolutions (hourly → daily → weekly)
//! and cross-scale drift comparison.

use cvx_core::error::AnalyticsError;

// ─── Resampling ─────────────────────────────────────────────────────

/// Resampling strategy for temporal downsampling.
#[derive(Debug, Clone, Copy)]
pub enum ResampleMethod {
    /// Use the last value in each bucket (sample-and-hold).
    LastValue,
    /// Linear interpolation at bucket midpoints.
    Linear,
}

/// Resample a trajectory to a coarser temporal resolution.
///
/// Groups points into buckets of `bucket_width` microseconds,
/// then applies the chosen resampling method.
///
/// Returns `(bucket_timestamp, vector)` pairs.
pub fn resample(
    trajectory: &[(i64, &[f32])],
    bucket_width: i64,
    method: ResampleMethod,
) -> Result<Vec<(i64, Vec<f32>)>, AnalyticsError> {
    if trajectory.is_empty() {
        return Ok(Vec::new());
    }
    if bucket_width <= 0 {
        return Err(AnalyticsError::InsufficientData { needed: 1, have: 0 });
    }

    let t_min = trajectory[0].0;
    let t_max = trajectory.last().unwrap().0;

    let mut result = Vec::new();
    let mut bucket_start = t_min;

    while bucket_start <= t_max {
        let bucket_end = bucket_start + bucket_width;
        let bucket_mid = bucket_start + bucket_width / 2;

        match method {
            ResampleMethod::LastValue => {
                // Find the last point in this bucket
                let last_in_bucket = trajectory
                    .iter()
                    .rev()
                    .find(|(t, _)| *t >= bucket_start && *t < bucket_end);

                if let Some((_, v)) = last_in_bucket {
                    result.push((bucket_mid, v.to_vec()));
                }
            }
            ResampleMethod::Linear => {
                // Interpolate at bucket midpoint
                if let Some(vec) = interpolate_at(trajectory, bucket_mid) {
                    result.push((bucket_mid, vec));
                }
            }
        }

        bucket_start = bucket_end;
    }

    Ok(result)
}

/// Linear interpolation at a specific timestamp.
fn interpolate_at(trajectory: &[(i64, &[f32])], target: i64) -> Option<Vec<f32>> {
    if trajectory.is_empty() {
        return None;
    }

    // Find bracketing points
    let idx = trajectory.partition_point(|(t, _)| *t <= target);

    if idx == 0 {
        // Before first point — extrapolate with first value
        return Some(trajectory[0].1.to_vec());
    }
    if idx >= trajectory.len() {
        // After last point — extrapolate with last value
        return Some(trajectory.last().unwrap().1.to_vec());
    }

    let (t0, v0) = &trajectory[idx - 1];
    let (t1, v1) = &trajectory[idx];

    if t1 == t0 {
        return Some(v0.to_vec());
    }

    let alpha = (target - t0) as f64 / (t1 - t0) as f64;
    let dim = v0.len();
    let interpolated: Vec<f32> = (0..dim)
        .map(|d| (v0[d] as f64 * (1.0 - alpha) + v1[d] as f64 * alpha) as f32)
        .collect();

    Some(interpolated)
}

// ─── Per-scale drift analysis ───────────────────────────────────────

/// Per-scale drift: L2 displacement at a given temporal resolution.
///
/// Returns `(bucket_timestamp, drift_magnitude)` pairs.
pub fn scale_drift(
    trajectory: &[(i64, &[f32])],
    bucket_width: i64,
) -> Result<Vec<(i64, f32)>, AnalyticsError> {
    let resampled = resample(trajectory, bucket_width, ResampleMethod::LastValue)?;

    if resampled.len() < 2 {
        return Ok(Vec::new());
    }

    let drifts: Vec<(i64, f32)> = resampled
        .windows(2)
        .map(|w| {
            let drift: f32 = w[0]
                .1
                .iter()
                .zip(w[1].1.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            (w[1].0, drift)
        })
        .collect();

    Ok(drifts)
}

// ─── Behavioral alignment ───────────────────────────────────────────

/// Behavioral alignment: Pearson correlation between two drift series.
///
/// Returns a value in `[-1, 1]`:
/// - `1.0`: perfectly correlated drift
/// - `0.0`: uncorrelated
/// - `-1.0`: perfectly anti-correlated
pub fn behavioral_alignment(drift_a: &[f32], drift_b: &[f32]) -> f32 {
    let n = drift_a.len().min(drift_b.len());
    if n < 2 {
        return 0.0;
    }

    let a = &drift_a[..n];
    let b = &drift_b[..n];

    let mean_a: f64 = a.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let mean_b: f64 = b.iter().map(|&x| x as f64).sum::<f64>() / n as f64;

    let mut cov = 0.0f64;
    let mut var_a = 0.0f64;
    let mut var_b = 0.0f64;

    for i in 0..n {
        let da = a[i] as f64 - mean_a;
        let db = b[i] as f64 - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom == 0.0 {
        return 0.0;
    }

    (cov / denom) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trajectory(points: &[(i64, Vec<f32>)]) -> Vec<(i64, &[f32])> {
        points.iter().map(|(t, v)| (*t, v.as_slice())).collect()
    }

    // ─── Resampling ─────────────────────────────────────────────────

    #[test]
    fn resample_hourly_to_daily() {
        // 24 hourly points → should produce ~1 daily bucket
        let hour = 3_600_000_000i64; // 1 hour in µs
        let day = 24 * hour;
        let points: Vec<(i64, Vec<f32>)> =
            (0..24).map(|i| (i as i64 * hour, vec![i as f32])).collect();
        let traj = make_trajectory(&points);

        let resampled = resample(&traj, day, ResampleMethod::LastValue).unwrap();
        assert_eq!(resampled.len(), 1); // one daily bucket
    }

    #[test]
    fn resample_preserves_count_same_resolution() {
        // 10 points at 1000µs intervals, bucket width = 1000µs → 10 buckets
        let points: Vec<(i64, Vec<f32>)> =
            (0..10).map(|i| (i as i64 * 1000, vec![i as f32])).collect();
        let traj = make_trajectory(&points);

        let resampled = resample(&traj, 1000, ResampleMethod::LastValue).unwrap();
        assert_eq!(resampled.len(), 10);
    }

    #[test]
    fn resample_linear_interpolation() {
        // Points spanning [0, 999] → two 500µs buckets: [0,500), [500,1000)
        let points = vec![(0i64, vec![0.0f32, 0.0]), (999, vec![10.0, 20.0])];
        let traj = make_trajectory(&points);

        let resampled = resample(&traj, 500, ResampleMethod::Linear).unwrap();
        assert_eq!(resampled.len(), 2);
        // First bucket midpoint at 250: ~25% of [0,999]
        assert!((resampled[0].1[0] - 2.5).abs() < 0.5);
        assert!((resampled[0].1[1] - 5.0).abs() < 1.0);
    }

    #[test]
    fn resample_empty() {
        let traj: Vec<(i64, &[f32])> = Vec::new();
        let result = resample(&traj, 1000, ResampleMethod::LastValue).unwrap();
        assert!(result.is_empty());
    }

    // ─── Scale drift ────────────────────────────────────────────────

    #[test]
    fn scale_drift_stationary() {
        let points: Vec<(i64, Vec<f32>)> =
            (0..10).map(|i| (i as i64 * 1000, vec![1.0, 2.0])).collect();
        let traj = make_trajectory(&points);

        let drifts = scale_drift(&traj, 1000).unwrap();
        for (_, d) in &drifts {
            assert!(*d < 1e-6, "stationary entity should have zero drift");
        }
    }

    #[test]
    fn scale_drift_linear() {
        let points: Vec<(i64, Vec<f32>)> =
            (0..10).map(|i| (i as i64 * 1000, vec![i as f32])).collect();
        let traj = make_trajectory(&points);

        let drifts = scale_drift(&traj, 1000).unwrap();
        // Each step should have drift = 1.0
        for (_, d) in &drifts {
            assert!((*d - 1.0).abs() < 1e-6);
        }
    }

    // ─── Behavioral alignment ───────────────────────────────────────

    #[test]
    fn alignment_identical_series() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let corr = behavioral_alignment(&a, &b);
        assert!(
            (corr - 1.0).abs() < 1e-5,
            "identical series should have corr ≈ 1.0"
        );
    }

    #[test]
    fn alignment_scaled_series() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b: Vec<f32> = a.iter().map(|x| x * 3.0).collect();
        let corr = behavioral_alignment(&a, &b);
        assert!(
            (corr - 1.0).abs() < 1e-5,
            "scaled series should have corr ≈ 1.0"
        );
    }

    #[test]
    fn alignment_anticorrelated() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b: Vec<f32> = a.iter().map(|x| -x).collect();
        let corr = behavioral_alignment(&a, &b);
        assert!(
            (corr + 1.0).abs() < 1e-5,
            "anti-correlated should be ≈ -1.0"
        );
    }

    #[test]
    fn alignment_constant_is_zero() {
        let a = vec![1.0, 1.0, 1.0, 1.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let corr = behavioral_alignment(&a, &b);
        assert!(corr.abs() < 1e-5, "constant vs trend should be ≈ 0.0");
    }
}
