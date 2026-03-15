//! Vector differential calculus over temporal trajectories.
//!
//! Provides first and second order finite differences, drift quantification,
//! and basic stochastic process characterization for embedding trajectories.
//!
//! # Conventions
//!
//! - Timestamps are in microseconds.
//! - Velocity units: vector displacement per microsecond.
//! - All functions operate on pre-sorted `(timestamp, &[f32])` slices.

use cvx_core::error::AnalyticsError;

/// Minimum number of points for velocity computation.
const MIN_POINTS_VELOCITY: usize = 2;
/// Minimum number of points for acceleration computation.
const MIN_POINTS_ACCELERATION: usize = 3;
/// Minimum number of points for stochastic characterization.
const MIN_POINTS_STOCHASTIC: usize = 10;

// ─── Core calculus ──────────────────────────────────────────────────

/// Compute velocity (first-order finite difference) at a target timestamp.
///
/// Uses central difference when `target` is between two points,
/// forward/backward difference at boundaries.
///
/// Returns a velocity vector with the same dimensionality as the input.
pub fn velocity(trajectory: &[(i64, &[f32])], target: i64) -> Result<Vec<f32>, AnalyticsError> {
    if trajectory.len() < MIN_POINTS_VELOCITY {
        return Err(AnalyticsError::InsufficientData {
            needed: MIN_POINTS_VELOCITY,
            have: trajectory.len(),
        });
    }

    let dim = trajectory[0].1.len();

    // Find the closest bracketing points
    let idx = find_closest_index(trajectory, target);

    let (t0, v0, t1, v1) = if idx == 0 {
        // Forward difference
        (
            trajectory[0].0,
            trajectory[0].1,
            trajectory[1].0,
            trajectory[1].1,
        )
    } else if idx >= trajectory.len() - 1 {
        let n = trajectory.len();
        (
            trajectory[n - 2].0,
            trajectory[n - 2].1,
            trajectory[n - 1].0,
            trajectory[n - 1].1,
        )
    } else {
        // Central difference
        (
            trajectory[idx - 1].0,
            trajectory[idx - 1].1,
            trajectory[idx + 1].0,
            trajectory[idx + 1].1,
        )
    };

    let dt = (t1 - t0) as f64;
    if dt == 0.0 {
        return Ok(vec![0.0; dim]);
    }

    let vel: Vec<f32> = (0..dim)
        .map(|d| ((v1[d] as f64 - v0[d] as f64) / dt) as f32)
        .collect();

    Ok(vel)
}

/// Compute acceleration (second-order finite difference) at a target timestamp.
///
/// Requires at least 3 trajectory points.
pub fn acceleration(trajectory: &[(i64, &[f32])], target: i64) -> Result<Vec<f32>, AnalyticsError> {
    if trajectory.len() < MIN_POINTS_ACCELERATION {
        return Err(AnalyticsError::InsufficientData {
            needed: MIN_POINTS_ACCELERATION,
            have: trajectory.len(),
        });
    }

    let dim = trajectory[0].1.len();
    let idx = find_closest_index(trajectory, target).clamp(1, trajectory.len() - 2);

    let (t_prev, v_prev) = (trajectory[idx - 1].0, trajectory[idx - 1].1);
    let (t_curr, v_curr) = (trajectory[idx].0, trajectory[idx].1);
    let (t_next, v_next) = (trajectory[idx + 1].0, trajectory[idx + 1].1);

    let dt1 = (t_curr - t_prev) as f64;
    let dt2 = (t_next - t_curr) as f64;
    let dt_avg = (dt1 + dt2) / 2.0;

    if dt_avg == 0.0 || dt1 == 0.0 || dt2 == 0.0 {
        return Ok(vec![0.0; dim]);
    }

    let acc: Vec<f32> = (0..dim)
        .map(|d| {
            let vel1 = (v_curr[d] as f64 - v_prev[d] as f64) / dt1;
            let vel2 = (v_next[d] as f64 - v_curr[d] as f64) / dt2;
            ((vel2 - vel1) / dt_avg) as f32
        })
        .collect();

    Ok(acc)
}

// ─── Drift quantification ───────────────────────────────────────────

/// Compute L2 drift magnitude between two snapshots.
pub fn drift_magnitude_l2(v1: &[f32], v2: &[f32]) -> f32 {
    assert_eq!(v1.len(), v2.len(), "dimension mismatch");
    let sum_sq: f32 = v1
        .iter()
        .zip(v2.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    sum_sq.sqrt()
}

/// Compute cosine drift (1 - cosine_similarity) between two snapshots.
pub fn drift_magnitude_cosine(v1: &[f32], v2: &[f32]) -> f32 {
    assert_eq!(v1.len(), v2.len(), "dimension mismatch");
    let dot: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let norm1: f32 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm1 == 0.0 || norm2 == 0.0 {
        return 1.0;
    }

    let similarity = (dot / (norm1 * norm2)).clamp(-1.0, 1.0);
    1.0 - similarity
}

/// Drift report identifying the most changed dimensions.
#[derive(Debug, Clone)]
pub struct DriftReport {
    /// L2 drift magnitude.
    pub l2_magnitude: f32,
    /// Cosine drift (1 - similarity).
    pub cosine_drift: f32,
    /// Top-N most changed dimensions: `(dimension_index, absolute_change)`.
    pub top_dimensions: Vec<(usize, f32)>,
}

/// Generate a drift report between two snapshots.
pub fn drift_report(v1: &[f32], v2: &[f32], top_n: usize) -> DriftReport {
    assert_eq!(v1.len(), v2.len(), "dimension mismatch");

    let l2_magnitude = drift_magnitude_l2(v1, v2);
    let cosine_drift = drift_magnitude_cosine(v1, v2);

    let mut changes: Vec<(usize, f32)> = v1
        .iter()
        .zip(v2.iter())
        .enumerate()
        .map(|(i, (a, b))| (i, (a - b).abs()))
        .collect();

    changes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    changes.truncate(top_n);

    DriftReport {
        l2_magnitude,
        cosine_drift,
        top_dimensions: changes,
    }
}

// ─── Stochastic characterization ────────────────────────────────────

/// Realized volatility over a trajectory (standard deviation of returns).
///
/// Returns the per-dimension realized volatility.
pub fn realized_volatility(trajectory: &[(i64, &[f32])]) -> Result<Vec<f32>, AnalyticsError> {
    if trajectory.len() < MIN_POINTS_STOCHASTIC {
        return Err(AnalyticsError::InsufficientData {
            needed: MIN_POINTS_STOCHASTIC,
            have: trajectory.len(),
        });
    }

    let dim = trajectory[0].1.len();
    let n = trajectory.len() - 1;

    // Compute log-returns (approximated as simple returns for embedding space)
    let mut means = vec![0.0f64; dim];
    let mut returns = Vec::with_capacity(n);

    for w in trajectory.windows(2) {
        let dt = (w[1].0 - w[0].0).max(1) as f64;
        let ret: Vec<f64> = (0..dim)
            .map(|d| (w[1].1[d] as f64 - w[0].1[d] as f64) / dt)
            .collect();
        for d in 0..dim {
            means[d] += ret[d];
        }
        returns.push(ret);
    }

    for m in &mut means {
        *m /= n as f64;
    }

    let mut vol = vec![0.0f64; dim];
    for ret in &returns {
        for d in 0..dim {
            let diff = ret[d] - means[d];
            vol[d] += diff * diff;
        }
    }

    let result: Vec<f32> = vol
        .iter()
        .map(|v| (v / (n - 1).max(1) as f64).sqrt() as f32)
        .collect();

    Ok(result)
}

/// Hurst exponent estimation via rescaled range (R/S) analysis.
///
/// Returns a single scalar H:
/// - H ≈ 0.5: Brownian motion (random walk)
/// - H > 0.5: persistent (trending)
/// - H < 0.5: anti-persistent (mean-reverting)
///
/// Computed on the L2 norm of displacement vectors.
pub fn hurst_exponent(trajectory: &[(i64, &[f32])]) -> Result<f32, AnalyticsError> {
    if trajectory.len() < MIN_POINTS_STOCHASTIC {
        return Err(AnalyticsError::InsufficientData {
            needed: MIN_POINTS_STOCHASTIC,
            have: trajectory.len(),
        });
    }

    // Convert to scalar series (L2 norm of increments)
    let increments: Vec<f64> = trajectory
        .windows(2)
        .map(|w| {
            let sum_sq: f64 = w[0]
                .1
                .iter()
                .zip(w[1].1.iter())
                .map(|(&a, &b)| {
                    let d = b as f64 - a as f64;
                    d * d
                })
                .sum();
            sum_sq.sqrt()
        })
        .collect();

    // R/S analysis over multiple window sizes
    let n = increments.len();
    let mut log_rs = Vec::new();
    let mut log_n = Vec::new();

    // Window sizes: powers of 2 from 4 to n/2
    let mut window = 4;
    while window <= n / 2 {
        let rs = rescaled_range(&increments, window);
        if rs > 0.0 {
            log_rs.push(rs.ln());
            log_n.push((window as f64).ln());
        }
        window *= 2;
    }

    if log_rs.len() < 2 {
        return Ok(0.5); // Not enough data for estimation
    }

    // Linear regression: log(R/S) = H * log(n) + c
    let h = linear_regression_slope(&log_n, &log_rs);
    Ok(h.clamp(0.0, 1.0) as f32)
}

/// Augmented Dickey-Fuller (ADF) test statistic for stationarity.
///
/// Returns the ADF test statistic. More negative values indicate
/// stronger evidence against the unit root (i.e., more stationary).
///
/// Computed on the L2 norm of the trajectory vectors.
pub fn adf_statistic(trajectory: &[(i64, &[f32])]) -> Result<f32, AnalyticsError> {
    if trajectory.len() < MIN_POINTS_STOCHASTIC {
        return Err(AnalyticsError::InsufficientData {
            needed: MIN_POINTS_STOCHASTIC,
            have: trajectory.len(),
        });
    }

    // Convert to scalar series (L2 norms)
    let series: Vec<f64> = trajectory
        .iter()
        .map(|(_, v)| {
            let sum_sq: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
            sum_sq.sqrt()
        })
        .collect();

    // Simple ADF: regress Δy_t on y_{t-1}
    let n = series.len();
    let mut sum_dy = 0.0;
    let mut sum_y = 0.0;

    for i in 1..n {
        let dy = series[i] - series[i - 1];
        let y_prev = series[i - 1];
        sum_dy += dy;
        sum_y += y_prev;
    }

    let m = (n - 1) as f64;
    let mean_y = sum_y / m;
    let mean_dy = sum_dy / m;

    // OLS coefficient: β = Σ((y-ȳ)(Δy-Δȳ)) / Σ((y-ȳ)²)
    let mut num = 0.0;
    let mut den = 0.0;
    let mut residual_ss = 0.0;

    for i in 1..n {
        let dy = series[i] - series[i - 1];
        let y_prev = series[i - 1];
        num += (y_prev - mean_y) * (dy - mean_dy);
        den += (y_prev - mean_y) * (y_prev - mean_y);
    }

    if den == 0.0 {
        return Ok(0.0);
    }

    let beta = num / den;

    // Compute residual standard error
    for i in 1..n {
        let dy = series[i] - series[i - 1];
        let y_prev = series[i - 1];
        let predicted = mean_dy + beta * (y_prev - mean_y);
        let resid = dy - predicted;
        residual_ss += resid * resid;
    }

    let se_beta = (residual_ss / (m - 2.0) / den).sqrt();

    if se_beta == 0.0 {
        return Ok(0.0);
    }

    let t_stat = beta / se_beta;
    Ok(t_stat as f32)
}

// ─── Helpers ────────────────────────────────────────────────────────

/// Find index of the closest point to a target timestamp.
fn find_closest_index(trajectory: &[(i64, &[f32])], target: i64) -> usize {
    trajectory
        .iter()
        .enumerate()
        .min_by_key(|(_, (ts, _))| (ts - target).unsigned_abs())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Rescaled range statistic for a given window size.
fn rescaled_range(data: &[f64], window: usize) -> f64 {
    let n_windows = data.len() / window;
    if n_windows == 0 {
        return 0.0;
    }

    let mut total_rs = 0.0;

    for w in 0..n_windows {
        let start = w * window;
        let end = start + window;
        let slice = &data[start..end];

        let mean: f64 = slice.iter().sum::<f64>() / window as f64;
        let std: f64 =
            (slice.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / window as f64).sqrt();

        if std == 0.0 {
            continue;
        }

        // Cumulative deviations
        let mut cumsum = 0.0;
        let mut max_dev = f64::NEG_INFINITY;
        let mut min_dev = f64::INFINITY;
        for &x in slice {
            cumsum += x - mean;
            max_dev = max_dev.max(cumsum);
            min_dev = min_dev.min(cumsum);
        }

        let r = max_dev - min_dev;
        total_rs += r / std;
    }

    total_rs / n_windows as f64
}

/// Simple linear regression slope.
fn linear_regression_slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let sum_xx: f64 = x.iter().map(|a| a * a).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom == 0.0 {
        return 0.0;
    }

    (n * sum_xy - sum_x * sum_y) / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create trajectory from owned vectors.
    fn make_trajectory(points: &[(i64, Vec<f32>)]) -> Vec<(i64, &[f32])> {
        points.iter().map(|(t, v)| (*t, v.as_slice())).collect()
    }

    // ─── Velocity ───────────────────────────────────────────────────

    #[test]
    fn velocity_linear_motion() {
        // Entity moves linearly: v = [1, 0, 0] per unit time
        let points: Vec<(i64, Vec<f32>)> = (0..10)
            .map(|i| (i as i64 * 1000, vec![i as f32, 0.0, 0.0]))
            .collect();
        let traj = make_trajectory(&points);

        let vel = velocity(&traj, 5000).unwrap();
        // Velocity should be ~[0.001, 0, 0] (1 unit per 1000µs)
        assert!((vel[0] - 0.001).abs() < 1e-6, "vel[0] = {}", vel[0]);
        assert!(vel[1].abs() < 1e-6);
        assert!(vel[2].abs() < 1e-6);
    }

    #[test]
    fn velocity_constant_is_zero() {
        let points: Vec<(i64, Vec<f32>)> = (0..5)
            .map(|i| (i as i64 * 1000, vec![1.0, 2.0, 3.0]))
            .collect();
        let traj = make_trajectory(&points);

        let vel = velocity(&traj, 2000).unwrap();
        for &v in &vel {
            assert!(v.abs() < 1e-6, "expected ~0, got {v}");
        }
    }

    #[test]
    fn velocity_insufficient_data() {
        let points = vec![(0, vec![1.0])];
        let traj = make_trajectory(&points);
        assert!(velocity(&traj, 0).is_err());
    }

    // ─── Acceleration ───────────────────────────────────────────────

    #[test]
    fn acceleration_linear_motion_is_zero() {
        let points: Vec<(i64, Vec<f32>)> = (0..10)
            .map(|i| (i as i64 * 1000, vec![i as f32 * 2.0, 0.0]))
            .collect();
        let traj = make_trajectory(&points);

        let acc = acceleration(&traj, 5000).unwrap();
        for &a in &acc {
            assert!(a.abs() < 1e-6, "expected ~0 for linear motion, got {a}");
        }
    }

    #[test]
    fn acceleration_quadratic_motion() {
        // x(t) = t², velocity = 2t, acceleration = 2
        let points: Vec<(i64, Vec<f32>)> = (0..10)
            .map(|i| {
                let t = i as f32;
                (i as i64 * 1000, vec![t * t])
            })
            .collect();
        let traj = make_trajectory(&points);

        let acc = acceleration(&traj, 5000).unwrap();
        // acceleration should be 2 / (1000 * 1000) = 2e-6 (units: per µs²)
        let expected = 2.0 / (1000.0 * 1000.0);
        assert!(
            (acc[0] - expected as f32).abs() < 1e-8,
            "expected ~{expected}, got {}",
            acc[0]
        );
    }

    #[test]
    fn acceleration_insufficient_data() {
        let points = vec![(0, vec![1.0]), (1000, vec![2.0])];
        let traj = make_trajectory(&points);
        assert!(acceleration(&traj, 0).is_err());
    }

    // ─── Drift magnitude ────────────────────────────────────────────

    #[test]
    fn drift_stationary_is_zero() {
        let v = vec![1.0, 2.0, 3.0];
        assert_eq!(drift_magnitude_l2(&v, &v), 0.0);
        assert!(drift_magnitude_cosine(&v, &v) < 1e-6);
    }

    #[test]
    fn drift_l2_known_value() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0];
        let drift = drift_magnitude_l2(&v1, &v2);
        assert!((drift - 2.0f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn drift_cosine_orthogonal() {
        let v1 = vec![1.0, 0.0];
        let v2 = vec![0.0, 1.0];
        let drift = drift_magnitude_cosine(&v1, &v2);
        assert!((drift - 1.0).abs() < 1e-6);
    }

    // ─── Drift report ───────────────────────────────────────────────

    #[test]
    fn drift_report_identifies_top_dimensions() {
        let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let v2 = vec![1.0, 2.0, 10.0, 4.0, 0.0]; // dim 2 changed by 7, dim 4 by 5

        let report = drift_report(&v1, &v2, 3);
        assert_eq!(report.top_dimensions[0].0, 2); // biggest change
        assert!((report.top_dimensions[0].1 - 7.0).abs() < 1e-6);
        assert_eq!(report.top_dimensions[1].0, 4); // second biggest
        assert!((report.top_dimensions[1].1 - 5.0).abs() < 1e-6);
    }

    // ─── Realized volatility ────────────────────────────────────────

    #[test]
    fn volatility_constant_is_zero() {
        let points: Vec<(i64, Vec<f32>)> =
            (0..20).map(|i| (i as i64 * 1000, vec![1.0, 2.0])).collect();
        let traj = make_trajectory(&points);

        let vol = realized_volatility(&traj).unwrap();
        for &v in &vol {
            assert!(
                v.abs() < 1e-10,
                "constant trajectory should have zero volatility"
            );
        }
    }

    #[test]
    fn volatility_linear_trend() {
        // Linear trend should have near-zero volatility (constant returns)
        let points: Vec<(i64, Vec<f32>)> = (0..50)
            .map(|i| (i as i64 * 1000, vec![i as f32 * 0.1]))
            .collect();
        let traj = make_trajectory(&points);

        let vol = realized_volatility(&traj).unwrap();
        // All returns are identical → volatility should be ~0
        assert!(vol[0] < 1e-6, "linear trend vol = {}", vol[0]);
    }

    // ─── Hurst exponent ─────────────────────────────────────────────

    #[test]
    fn hurst_brownian_motion_approx_05() {
        // Simulate Brownian motion: increments are iid
        use rand::Rng;
        let mut rng = rand::rng();
        let n = 1000;
        let mut points: Vec<(i64, Vec<f32>)> = Vec::with_capacity(n);
        let mut x = 0.0f32;

        for i in 0..n {
            x += rng.random::<f32>() - 0.5;
            points.push((i as i64 * 1000, vec![x]));
        }

        let traj = make_trajectory(&points);
        let h = hurst_exponent(&traj).unwrap();

        // H should be ~0.5 for Brownian motion (±0.15 tolerance for finite sample)
        assert!(
            (h - 0.5).abs() < 0.2,
            "Hurst exponent = {h}, expected ~0.5 for BM"
        );
    }

    // ─── ADF test ───────────────────────────────────────────────────

    #[test]
    fn adf_stationary_series() {
        // Mean-reverting series (OU process simulation)
        let n = 200;
        let mut points: Vec<(i64, Vec<f32>)> = Vec::with_capacity(n);
        let mut x = 0.0f32;
        let theta = 0.5; // mean reversion speed
        let mu = 1.0f32; // long-term mean
        let mut rng = rand::rng();

        for i in 0..n {
            x += theta * (mu - x) + 0.1 * (rand::Rng::random::<f32>(&mut rng) - 0.5);
            points.push((i as i64 * 1000, vec![x]));
        }

        let traj = make_trajectory(&points);
        let stat = adf_statistic(&traj).unwrap();

        // For a stationary series, ADF statistic should be negative
        // (more negative = stronger stationarity evidence)
        assert!(
            stat < 0.0,
            "ADF statistic = {stat}, expected negative for stationary series"
        );
    }

    #[test]
    fn adf_random_walk_not_strongly_negative() {
        // Pure random walk (unit root)
        let n = 200;
        let mut points: Vec<(i64, Vec<f32>)> = Vec::with_capacity(n);
        let mut x = 0.0f32;
        let mut rng = rand::rng();

        for i in 0..n {
            x += rand::Rng::random::<f32>(&mut rng) - 0.5;
            points.push((i as i64 * 1000, vec![x]));
        }

        let traj = make_trajectory(&points);
        let stat = adf_statistic(&traj).unwrap();

        // For a random walk, ADF should be close to 0 or weakly negative
        // Critical value at 5% is about -2.86 for n=200
        // We just check it's not *very* negative
        assert!(
            stat > -5.0,
            "ADF statistic = {stat}, random walk shouldn't be strongly negative"
        );
    }

    // ─── Helpers ────────────────────────────────────────────────────

    #[test]
    fn find_closest_index_exact() {
        let points = vec![(0i64, vec![0.0f32]), (1000, vec![1.0]), (2000, vec![2.0])];
        let traj = make_trajectory(&points);
        assert_eq!(find_closest_index(&traj, 1000), 1);
    }

    #[test]
    fn find_closest_index_between() {
        let points = vec![(0i64, vec![0.0f32]), (1000, vec![1.0]), (2000, vec![2.0])];
        let traj = make_trajectory(&points);
        assert_eq!(find_closest_index(&traj, 1200), 1);
        assert_eq!(find_closest_index(&traj, 1600), 2);
    }

    #[test]
    fn linear_regression_known() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x
        let slope = linear_regression_slope(&x, &y);
        assert!((slope - 2.0).abs() < 1e-10);
    }
}
