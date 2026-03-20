//! Counterfactual trajectory analysis.
//!
//! Given a detected change point, extrapolates the **pre-change trajectory**
//! beyond the change point and compares with the actual post-change trajectory.
//! Answers: "What *would* have happened if the change hadn't occurred?"
//!
//! # References
//!
//! - Brodersen, K. H. et al. (2015). Causal impact. *Annals of Applied Statistics*.
//! - Abadie, A. (2021). Synthetic controls. *JEL*, 59(2).

use crate::calculus::drift_magnitude_l2;
use cvx_core::error::AnalyticsError;

// ─── Types ──────────────────────────────────────────────────────────

/// Counterfactual analysis result.
#[derive(Debug, Clone)]
pub struct CounterfactualResult {
    /// The change point timestamp.
    pub change_point: i64,
    /// Actual post-change trajectory.
    pub actual: Vec<(i64, Vec<f32>)>,
    /// Counterfactual (extrapolated pre-change) trajectory.
    pub counterfactual: Vec<(i64, Vec<f32>)>,
    /// Divergence between actual and counterfactual over time.
    pub divergence_curve: Vec<(i64, f32)>,
    /// Total divergence (area under curve, via trapezoidal rule).
    pub total_divergence: f64,
    /// Timestamp of maximum divergence.
    pub max_divergence_time: i64,
    /// Maximum divergence value.
    pub max_divergence_value: f32,
    /// Method used for extrapolation.
    pub method: CounterfactualMethod,
}

/// Method used for counterfactual extrapolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CounterfactualMethod {
    /// OLS linear extrapolation per dimension.
    LinearExtrapolation,
}

// ─── Core function ──────────────────────────────────────────────────

/// Compute a counterfactual trajectory analysis.
///
/// Splits the trajectory at `change_point`, fits a linear trend to the
/// pre-change segment, extrapolates beyond the change point, and compares
/// with the actual post-change data.
///
/// # Arguments
///
/// * `pre_change` — Trajectory before the change point (sorted by timestamp)
/// * `post_change` — Trajectory after the change point (sorted by timestamp)
/// * `change_point` — Timestamp of the detected change
///
/// # Errors
///
/// Returns [`AnalyticsError::InsufficientData`] if pre_change has < 2 points
/// or post_change is empty.
pub fn counterfactual_trajectory(
    pre_change: &[(i64, &[f32])],
    post_change: &[(i64, &[f32])],
    change_point: i64,
) -> Result<CounterfactualResult, AnalyticsError> {
    if pre_change.len() < 2 {
        return Err(AnalyticsError::InsufficientData {
            needed: 2,
            have: pre_change.len(),
        });
    }
    if post_change.is_empty() {
        return Err(AnalyticsError::InsufficientData { needed: 1, have: 0 });
    }

    let dim = pre_change[0].1.len();

    // ── Fit linear trend per dimension on pre-change data ──

    let (slopes, intercepts) = fit_linear_per_dim(pre_change);

    // ── Extrapolate at post-change timestamps ──

    let counterfactual: Vec<(i64, Vec<f32>)> = post_change
        .iter()
        .map(|&(t, _)| {
            let t_f = t as f64;
            let vec: Vec<f32> = (0..dim)
                .map(|d| (slopes[d] * t_f + intercepts[d]) as f32)
                .collect();
            (t, vec)
        })
        .collect();

    // ── Actual post-change trajectory ──

    let actual: Vec<(i64, Vec<f32>)> = post_change.iter().map(|&(t, v)| (t, v.to_vec())).collect();

    // ── Divergence curve ──

    let divergence_curve: Vec<(i64, f32)> = actual
        .iter()
        .zip(counterfactual.iter())
        .map(|((t, act), (_, cf))| {
            let dist = drift_magnitude_l2(act, cf);
            (*t, dist)
        })
        .collect();

    // ── Aggregate metrics ──

    let (max_time, max_val) = divergence_curve
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|&(t, v)| (t, v))
        .unwrap_or((change_point, 0.0));

    let total_divergence = trapezoidal_integral(&divergence_curve);

    Ok(CounterfactualResult {
        change_point,
        actual,
        counterfactual,
        divergence_curve,
        total_divergence,
        max_divergence_time: max_time,
        max_divergence_value: max_val,
        method: CounterfactualMethod::LinearExtrapolation,
    })
}

// ─── Helpers ────────────────────────────────────────────────────────

/// Fit a linear model y_d = slope_d * t + intercept_d for each dimension.
fn fit_linear_per_dim(trajectory: &[(i64, &[f32])]) -> (Vec<f64>, Vec<f64>) {
    let n = trajectory.len() as f64;
    let dim = trajectory[0].1.len();

    let t_vals: Vec<f64> = trajectory.iter().map(|(t, _)| *t as f64).collect();
    let t_mean: f64 = t_vals.iter().sum::<f64>() / n;

    let mut slopes = vec![0.0f64; dim];
    let mut intercepts = vec![0.0f64; dim];

    let t_var: f64 = t_vals.iter().map(|t| (t - t_mean) * (t - t_mean)).sum();

    if t_var < 1e-15 {
        // All same timestamp — use last values as constant
        let last = trajectory.last().unwrap().1;
        for d in 0..dim {
            intercepts[d] = last[d] as f64;
        }
        return (slopes, intercepts);
    }

    for d in 0..dim {
        let y_vals: Vec<f64> = trajectory.iter().map(|(_, v)| v[d] as f64).collect();
        let y_mean: f64 = y_vals.iter().sum::<f64>() / n;

        let covar: f64 = t_vals
            .iter()
            .zip(y_vals.iter())
            .map(|(t, y)| (t - t_mean) * (y - y_mean))
            .sum();

        slopes[d] = covar / t_var;
        intercepts[d] = y_mean - slopes[d] * t_mean;
    }

    (slopes, intercepts)
}

/// Trapezoidal rule integration of a (timestamp, value) curve.
fn trapezoidal_integral(curve: &[(i64, f32)]) -> f64 {
    if curve.len() < 2 {
        return 0.0;
    }

    let mut total = 0.0f64;
    for w in curve.windows(2) {
        let dt = (w[1].0 - w[0].0) as f64;
        let avg_val = (w[0].1 as f64 + w[1].1 as f64) / 2.0;
        total += dt * avg_val;
    }

    total
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn as_refs(points: &[(i64, Vec<f32>)]) -> Vec<(i64, &[f32])> {
        points.iter().map(|(t, v)| (*t, v.as_slice())).collect()
    }

    // ─── fit_linear_per_dim ─────────────────────────────────────

    #[test]
    fn linear_fit_perfect_line() {
        // y = 2*t + 1 (1-dim)
        let owned = vec![
            (0i64, vec![1.0f32]),
            (1, vec![3.0]),
            (2, vec![5.0]),
            (3, vec![7.0]),
        ];
        let traj = as_refs(&owned);
        let (slopes, intercepts) = fit_linear_per_dim(&traj);
        assert!(
            (slopes[0] - 2.0).abs() < 1e-6,
            "slope should be 2.0, got {}",
            slopes[0]
        );
        assert!(
            (intercepts[0] - 1.0).abs() < 1e-6,
            "intercept should be 1.0, got {}",
            intercepts[0]
        );
    }

    #[test]
    fn linear_fit_multidim() {
        // dim 0: y = t, dim 1: y = -t + 10
        let owned = vec![
            (0i64, vec![0.0f32, 10.0]),
            (5, vec![5.0, 5.0]),
            (10, vec![10.0, 0.0]),
        ];
        let traj = as_refs(&owned);
        let (slopes, intercepts) = fit_linear_per_dim(&traj);
        assert!((slopes[0] - 1.0).abs() < 1e-6);
        assert!((slopes[1] - (-1.0)).abs() < 1e-6);
        assert!(intercepts[0].abs() < 1e-6);
        assert!((intercepts[1] - 10.0).abs() < 1e-6);
    }

    // ─── trapezoidal_integral ───────────────────────────────────

    #[test]
    fn integral_constant() {
        let curve = vec![(0i64, 5.0f32), (100, 5.0)];
        let area = trapezoidal_integral(&curve);
        assert!((area - 500.0).abs() < 1e-6);
    }

    #[test]
    fn integral_triangle() {
        // Triangle from (0,0) to (100, 10): area = 0.5 * 100 * 10 = 500
        let curve = vec![(0i64, 0.0f32), (100, 10.0)];
        let area = trapezoidal_integral(&curve);
        assert!((area - 500.0).abs() < 1e-6);
    }

    #[test]
    fn integral_single_point() {
        assert_eq!(trapezoidal_integral(&[(0, 5.0)]), 0.0);
    }

    // ─── counterfactual_trajectory ──────────────────────────────

    #[test]
    fn counterfactual_insufficient_pre() {
        let pre_owned = vec![(100i64, vec![1.0f32])];
        let post_owned = vec![(200i64, vec![2.0f32])];
        let pre = as_refs(&pre_owned);
        let post = as_refs(&post_owned);
        assert!(counterfactual_trajectory(&pre, &post, 150).is_err());
    }

    #[test]
    fn counterfactual_empty_post() {
        let pre_owned = vec![(100i64, vec![1.0f32]), (200, vec![2.0])];
        let post: Vec<(i64, &[f32])> = vec![];
        let pre = as_refs(&pre_owned);
        assert!(counterfactual_trajectory(&pre, &post, 250).is_err());
    }

    #[test]
    fn counterfactual_linear_continuation() {
        // Pre: y = t * 0.1 (linear, slope = 0.1)
        // Post (actual): y jumps to 100 (major change)
        // Counterfactual should continue the linear trend
        let pre_owned: Vec<(i64, Vec<f32>)> = (0..10)
            .map(|i| (i as i64 * 1000, vec![i as f32 * 0.1]))
            .collect();
        let post_owned: Vec<(i64, Vec<f32>)> = (10..15)
            .map(|i| (i as i64 * 1000, vec![100.0])) // sudden jump
            .collect();

        let pre = as_refs(&pre_owned);
        let post = as_refs(&post_owned);

        let result = counterfactual_trajectory(&pre, &post, 10000).unwrap();

        assert_eq!(result.change_point, 10000);
        assert_eq!(result.actual.len(), 5);
        assert_eq!(result.counterfactual.len(), 5);
        assert_eq!(result.divergence_curve.len(), 5);

        // Counterfactual at t=10000 should be ~1.0 (continuing slope 0.0001 * 10000)
        let cf_at_cp = &result.counterfactual[0].1[0];
        assert!(
            (*cf_at_cp - 1.0).abs() < 0.1,
            "counterfactual at change point should be ~1.0, got {cf_at_cp}"
        );

        // Actual is 100.0, so divergence should be large
        assert!(
            result.max_divergence_value > 90.0,
            "divergence should be large, got {}",
            result.max_divergence_value
        );

        assert!(result.total_divergence > 0.0);
    }

    #[test]
    fn counterfactual_no_change() {
        // Pre and post follow the same linear trend — divergence should be ~0
        let pre_owned: Vec<(i64, Vec<f32>)> = (0..10)
            .map(|i| (i as i64 * 1000, vec![i as f32 * 0.1]))
            .collect();
        let post_owned: Vec<(i64, Vec<f32>)> = (10..15)
            .map(|i| (i as i64 * 1000, vec![i as f32 * 0.1]))
            .collect();

        let pre = as_refs(&pre_owned);
        let post = as_refs(&post_owned);

        let result = counterfactual_trajectory(&pre, &post, 10000).unwrap();

        // Counterfactual should match actual (same linear trend)
        assert!(
            result.max_divergence_value < 0.1,
            "no change should have ~0 divergence, got {}",
            result.max_divergence_value
        );
    }

    #[test]
    fn counterfactual_multidim() {
        let pre_owned: Vec<(i64, Vec<f32>)> = (0..10)
            .map(|i| (i as i64 * 1000, vec![i as f32, -(i as f32)]))
            .collect();
        let post_owned: Vec<(i64, Vec<f32>)> = (10..15)
            .map(|i| (i as i64 * 1000, vec![50.0, 50.0])) // abrupt change in both dims
            .collect();

        let pre = as_refs(&pre_owned);
        let post = as_refs(&post_owned);

        let result = counterfactual_trajectory(&pre, &post, 10000).unwrap();

        assert_eq!(result.counterfactual[0].1.len(), 2);
        assert!(result.max_divergence_value > 10.0);
    }

    #[test]
    fn counterfactual_method_is_linear() {
        let pre_owned = vec![(0i64, vec![0.0f32]), (1000, vec![1.0])];
        let post_owned = vec![(2000i64, vec![10.0f32])];
        let pre = as_refs(&pre_owned);
        let post = as_refs(&post_owned);

        let result = counterfactual_trajectory(&pre, &post, 1500).unwrap();
        assert_eq!(result.method, CounterfactualMethod::LinearExtrapolation);
    }
}
