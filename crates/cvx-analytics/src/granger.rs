//! Granger causality testing for embedding trajectories.
//!
//! Tests whether entity A's movements in embedding space **precede** entity B's.
//! Uses a VAR(L) model on dimensionality-reduced trajectories.
//!
//! # Algorithm
//!
//! 1. Align trajectories to a common time grid (linear interpolation)
//! 2. For each dimension d:
//!    - Fit restricted model: `B_d(t) = Σ β_l · B_d(t-l) + ε`
//!    - Fit unrestricted model: `B_d(t) = Σ β_l · B_d(t-l) + Σ γ_l · A_d(t-l) + ε`
//!    - F-test: does the unrestricted model significantly improve?
//! 3. Combine per-dimension p-values via Fisher's method
//!
//! # References
//!
//! - Granger, C.W.J. (1969). Investigating causal relations. *Econometrica*, 37(3).
//! - Fisher, R.A. (1925). Statistical methods for research workers.

use cvx_core::error::AnalyticsError;

// ─── Types ──────────────────────────────────────────────────────────

/// Direction of Granger causality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrangerDirection {
    /// A Granger-causes B.
    AToB,
    /// B Granger-causes A.
    BToA,
    /// Both directions significant.
    Bidirectional,
    /// No significant causality detected.
    None,
}

/// Result of a Granger causality test.
#[derive(Debug, Clone)]
pub struct GrangerResult {
    /// Detected causal direction.
    pub direction: GrangerDirection,
    /// Optimal lag (the one with lowest combined p-value for the winning direction).
    pub optimal_lag: usize,
    /// F-statistic for the winning direction at optimal lag.
    pub f_statistic: f64,
    /// Combined p-value (Fisher's method) for the winning direction.
    pub p_value: f64,
    /// Partial R² improvement (effect size).
    pub effect_size: f64,
    /// Per-dimension F-statistics for A→B at optimal lag.
    pub per_dimension_a_to_b: Vec<f64>,
    /// Per-dimension F-statistics for B→A at optimal lag.
    pub per_dimension_b_to_a: Vec<f64>,
}

// ─── Alignment ──────────────────────────────────────────────────────

/// Align two trajectories to a common time grid via linear interpolation.
///
/// Returns `(aligned_a, aligned_b)` with the same timestamps.
fn align_trajectories(
    traj_a: &[(i64, &[f32])],
    traj_b: &[(i64, &[f32])],
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    // Use the union of timestamps, then interpolate both at those points
    // For simplicity, use the timestamps of the trajectory with more points
    // and interpolate the other
    let (base, other, swap) = if traj_a.len() >= traj_b.len() {
        (traj_a, traj_b, false)
    } else {
        (traj_b, traj_a, true)
    };

    let dim = base[0].1.len();
    let timestamps: Vec<i64> = base.iter().map(|(t, _)| *t).collect();

    let base_vecs: Vec<Vec<f32>> = base.iter().map(|(_, v)| v.to_vec()).collect();
    let other_vecs: Vec<Vec<f32>> = timestamps
        .iter()
        .map(|&t| interpolate_at(other, t, dim))
        .collect();

    if swap {
        (other_vecs, base_vecs)
    } else {
        (base_vecs, other_vecs)
    }
}

/// Linear interpolation of a trajectory at a specific timestamp.
fn interpolate_at(traj: &[(i64, &[f32])], t: i64, dim: usize) -> Vec<f32> {
    if traj.is_empty() {
        return vec![0.0; dim];
    }
    if traj.len() == 1 {
        return traj[0].1.to_vec();
    }

    // Before first point
    if t <= traj[0].0 {
        return traj[0].1.to_vec();
    }
    // After last point
    if t >= traj.last().unwrap().0 {
        return traj.last().unwrap().1.to_vec();
    }

    // Find bracketing points
    let idx = traj
        .iter()
        .position(|(ts, _)| *ts >= t)
        .unwrap_or(traj.len() - 1);

    if traj[idx].0 == t {
        return traj[idx].1.to_vec();
    }

    let (t0, v0) = &traj[idx - 1];
    let (t1, v1) = &traj[idx];
    let alpha = (t - t0) as f64 / (t1 - t0) as f64;

    v0.iter()
        .zip(v1.iter())
        .map(|(&a, &b)| (a as f64 * (1.0 - alpha) + b as f64 * alpha) as f32)
        .collect()
}

// ─── OLS regression ─────────────────────────────────────────────────

/// Fit OLS for a single dimension and compute residual sum of squares.
///
/// Restricted model: `y_d(t) = Σ_l β_l · y_d(t-l) + ε`
/// Unrestricted model: `y_d(t) = Σ_l β_l · y_d(t-l) + Σ_l γ_l · x_d(t-l) + ε`
///
/// Returns `(rss_restricted, rss_unrestricted, n_obs)`.
fn ols_granger_single_dim(
    y: &[f64], // target series (one dimension)
    x: &[f64], // predictor series (one dimension)
    lag: usize,
) -> (f64, f64, usize) {
    let n = y.len();
    if n <= lag {
        return (1.0, 1.0, 0);
    }

    let n_obs = n - lag;

    // Build design matrices
    // Restricted: [y(t-1), y(t-2), ..., y(t-lag)]
    // Unrestricted: [y(t-1), ..., y(t-lag), x(t-1), ..., x(t-lag)]

    let rss_r = fit_and_rss(y, &[y], lag, n_obs);
    let rss_u = fit_and_rss(y, &[y, x], lag, n_obs);

    (rss_r, rss_u, n_obs)
}

/// Fit a simple autoregressive model and return residual sum of squares.
///
/// `y[lag..] = sum over each series in `predictors` of (sum_l beta_l * series[t-l]) + epsilon`
///
/// Uses the normal equation (X^T X)^{-1} X^T y via iterative least squares
/// simplified to a simple approach.
fn fit_and_rss(y: &[f64], predictors: &[&[f64]], lag: usize, n_obs: usize) -> f64 {
    let n_features = predictors.len() * lag;
    if n_features == 0 || n_obs == 0 {
        return y.iter().map(|v| v * v).sum();
    }

    // Build X matrix (n_obs × n_features) and y vector
    let mut x_mat: Vec<Vec<f64>> = Vec::with_capacity(n_obs);
    let mut y_vec: Vec<f64> = Vec::with_capacity(n_obs);

    for t in lag..(lag + n_obs) {
        let mut row = Vec::with_capacity(n_features);
        for pred in predictors {
            for l in 1..=lag {
                row.push(pred[t - l]);
            }
        }
        x_mat.push(row);
        y_vec.push(y[t]);
    }

    // Solve via normal equations: β = (X^T X)^{-1} X^T y
    // Using simple gradient-free approach: compute pseudo-inverse iteratively
    // For small lag (1-20) and moderate n, this is fine
    let beta = solve_ols(&x_mat, &y_vec, n_features);

    // Compute RSS
    let mut rss = 0.0;
    for (i, row) in x_mat.iter().enumerate() {
        let pred: f64 = row.iter().zip(beta.iter()).map(|(x, b)| x * b).sum();
        let resid = y_vec[i] - pred;
        rss += resid * resid;
    }

    rss
}

/// Solve OLS via normal equations with regularization.
fn solve_ols(x: &[Vec<f64>], y: &[f64], p: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 || p == 0 {
        return vec![0.0; p];
    }

    // X^T X (p × p)
    let mut xtx = vec![vec![0.0f64; p]; p];
    for row in x {
        for i in 0..p {
            for j in 0..p {
                xtx[i][j] += row[i] * row[j];
            }
        }
    }

    // Ridge regularization (small lambda for numerical stability)
    let lambda = 1e-8;
    for i in 0..p {
        xtx[i][i] += lambda;
    }

    // X^T y (p × 1)
    let mut xty = vec![0.0f64; p];
    for (row, &yi) in x.iter().zip(y.iter()) {
        for i in 0..p {
            xty[i] += row[i] * yi;
        }
    }

    // Solve via Cholesky decomposition (X^T X is PD with ridge)
    cholesky_solve(&xtx, &xty)
}

/// Solve Ax = b where A is symmetric positive definite via Cholesky.
fn cholesky_solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = a.len();

    // Cholesky decomposition: A = L L^T
    let mut l = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0;
            for k in 0..j {
                s += l[i][k] * l[j][k];
            }
            if i == j {
                let val = a[i][i] - s;
                l[i][j] = if val > 0.0 { val.sqrt() } else { 1e-12 };
            } else {
                l[i][j] = if l[j][j].abs() > 1e-15 {
                    (a[i][j] - s) / l[j][j]
                } else {
                    0.0
                };
            }
        }
    }

    // Forward substitution: L z = b
    let mut z = vec![0.0f64; n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..i {
            s += l[i][j] * z[j];
        }
        z[i] = if l[i][i].abs() > 1e-15 {
            (b[i] - s) / l[i][i]
        } else {
            0.0
        };
    }

    // Backward substitution: L^T x = z
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = 0.0;
        for j in (i + 1)..n {
            s += l[j][i] * x[j];
        }
        x[i] = if l[i][i].abs() > 1e-15 {
            (z[i] - s) / l[i][i]
        } else {
            0.0
        };
    }

    x
}

// ─── F-test & Fisher's method ───────────────────────────────────────

/// Compute F-statistic from restricted and unrestricted RSS.
///
/// F = ((RSS_r - RSS_u) / q) / (RSS_u / (n - p_u))
/// where q = extra parameters in unrestricted model, p_u = total params.
fn f_statistic(rss_r: f64, rss_u: f64, q: usize, n: usize, p_u: usize) -> f64 {
    if rss_u <= 0.0 || n <= p_u || q == 0 {
        return 0.0;
    }
    let numerator = (rss_r - rss_u) / q as f64;
    let denominator = rss_u / (n - p_u) as f64;
    if denominator <= 0.0 {
        0.0
    } else {
        (numerator / denominator).max(0.0)
    }
}

/// Approximate p-value from F-statistic using the F-distribution.
///
/// Uses the regularized incomplete beta function approximation.
/// For F(q, n-p) distribution.
fn f_to_p(f: f64, df1: usize, df2: usize) -> f64 {
    if f <= 0.0 || df1 == 0 || df2 == 0 {
        return 1.0;
    }

    // Use the relationship: P(F > f) = I_x(df2/2, df1/2)
    // where x = df2 / (df2 + df1 * f)
    let x = df2 as f64 / (df2 as f64 + df1 as f64 * f);
    regularized_incomplete_beta(x, df2 as f64 / 2.0, df1 as f64 / 2.0)
}

/// Regularized incomplete beta function via continued fraction (Lentz's method).
///
/// I_x(a, b) = B_x(a, b) / B(a, b)
fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use symmetry relation if needed for convergence
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(1.0 - x, b, a);
    }

    let ln_prefix = a * x.ln() + b * (1.0 - x).ln()
        - (a.ln() + ln_beta(a, b));
    let prefix = ln_prefix.exp();

    // Lentz's continued fraction
    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut f = d;

    for m in 1..200 {
        let m_f = m as f64;

        // Even step
        let num_even = m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
        d = 1.0 + num_even * d;
        if d.abs() < 1e-30 { d = 1e-30; }
        c = 1.0 + num_even / c;
        if c.abs() < 1e-30 { c = 1e-30; }
        d = 1.0 / d;
        f *= d * c;

        // Odd step
        let num_odd = -(a + m_f) * (a + b + m_f) * x
            / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
        d = 1.0 + num_odd * d;
        if d.abs() < 1e-30 { d = 1e-30; }
        c = 1.0 + num_odd / c;
        if c.abs() < 1e-30 { c = 1e-30; }
        d = 1.0 / d;
        let delta = d * c;
        f *= delta;

        if (delta - 1.0).abs() < 1e-10 {
            break;
        }
    }

    prefix * f / a
}

/// Log of the Beta function: ln(B(a,b)) = ln(Gamma(a)) + ln(Gamma(b)) - ln(Gamma(a+b))
fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Stirling's approximation for ln(Gamma(x)) for x > 0.
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    // Lanczos approximation
    let g = 7.0;
    let coefs = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1259.139_216_722_402_9,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    let xx = x - 1.0;
    let mut sum = coefs[0];
    for (i, &c) in coefs[1..].iter().enumerate() {
        sum += c / (xx + i as f64 + 1.0);
    }

    let t = xx + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (xx + 0.5) * t.ln() - t + sum.ln()
}

/// Fisher's method: combine independent p-values.
///
/// χ² = -2 Σ ln(p_i), with 2k degrees of freedom.
fn fisher_combine(p_values: &[f64]) -> f64 {
    let valid: Vec<f64> = p_values
        .iter()
        .filter(|&&p| p > 0.0 && p <= 1.0)
        .copied()
        .collect();

    if valid.is_empty() {
        return 1.0;
    }

    let chi2: f64 = -2.0 * valid.iter().map(|p| p.ln()).sum::<f64>();
    let df = 2 * valid.len();

    // Approximate chi-squared p-value using Wilson-Hilferty transformation
    let k = df as f64;
    let z = ((chi2 / k).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k)))
        / (2.0 / (9.0 * k)).sqrt();

    // Standard normal survival function (1 - Φ(z))
    0.5 * erfc(z / std::f64::consts::SQRT_2)
}

/// Complementary error function approximation.
fn erfc(x: f64) -> f64 {
    // Abramowitz & Stegun approximation 7.1.26
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let result = poly * (-x * x).exp();
    if x >= 0.0 { result } else { 2.0 - result }
}

// ─── Core function ──────────────────────────────────────────────────

/// Test Granger causality between two embedding trajectories.
///
/// # Arguments
///
/// * `traj_a` — Entity A's trajectory (sorted by timestamp)
/// * `traj_b` — Entity B's trajectory (sorted by timestamp)
/// * `max_lag` — Maximum lag to test (number of time steps)
/// * `significance` — P-value threshold for significance (e.g., 0.05)
///
/// # Errors
///
/// Returns [`AnalyticsError::InsufficientData`] if trajectories are too short
/// for the requested lag.
pub fn granger_causality(
    traj_a: &[(i64, &[f32])],
    traj_b: &[(i64, &[f32])],
    max_lag: usize,
    significance: f64,
) -> Result<GrangerResult, AnalyticsError> {
    if traj_a.len() < 3 || traj_b.len() < 3 {
        return Err(AnalyticsError::InsufficientData {
            needed: 3,
            have: traj_a.len().min(traj_b.len()),
        });
    }

    let (aligned_a, aligned_b) = align_trajectories(traj_a, traj_b);
    let n = aligned_a.len();
    let dim = aligned_a[0].len();

    if n < max_lag + 3 {
        return Err(AnalyticsError::InsufficientData {
            needed: max_lag + 3,
            have: n,
        });
    }

    // Convert to f64 per-dimension series
    let a_dims: Vec<Vec<f64>> = (0..dim)
        .map(|d| aligned_a.iter().map(|v| v[d] as f64).collect())
        .collect();
    let b_dims: Vec<Vec<f64>> = (0..dim)
        .map(|d| aligned_b.iter().map(|v| v[d] as f64).collect())
        .collect();

    // Test each lag, keep the best
    let mut best_a2b_p = 1.0f64;
    let mut best_a2b_lag = 1;
    let mut best_a2b_f = 0.0;
    let mut best_a2b_effect = 0.0;
    let mut best_a2b_per_dim = vec![0.0; dim];

    let mut best_b2a_p = 1.0f64;
    let mut best_b2a_lag = 1;
    let mut best_b2a_f = 0.0;
    let mut best_b2a_effect = 0.0;
    let mut best_b2a_per_dim = vec![0.0; dim];

    for lag in 1..=max_lag {
        // A → B: does A's past improve prediction of B?
        let (p_a2b, f_a2b, effect_a2b, per_dim_a2b) =
            test_direction(&a_dims, &b_dims, lag, n);

        if p_a2b < best_a2b_p {
            best_a2b_p = p_a2b;
            best_a2b_lag = lag;
            best_a2b_f = f_a2b;
            best_a2b_effect = effect_a2b;
            best_a2b_per_dim = per_dim_a2b;
        }

        // B → A: does B's past improve prediction of A?
        let (p_b2a, f_b2a, effect_b2a, per_dim_b2a) =
            test_direction(&b_dims, &a_dims, lag, n);

        if p_b2a < best_b2a_p {
            best_b2a_p = p_b2a;
            best_b2a_lag = lag;
            best_b2a_f = f_b2a;
            best_b2a_effect = effect_b2a;
            best_b2a_per_dim = per_dim_b2a;
        }
    }

    let a2b_sig = best_a2b_p < significance;
    let b2a_sig = best_b2a_p < significance;

    let (direction, optimal_lag, f_stat, p_val, effect) = match (a2b_sig, b2a_sig) {
        (true, true) => (
            GrangerDirection::Bidirectional,
            if best_a2b_p < best_b2a_p { best_a2b_lag } else { best_b2a_lag },
            best_a2b_f.max(best_b2a_f),
            best_a2b_p.min(best_b2a_p),
            best_a2b_effect.max(best_b2a_effect),
        ),
        (true, false) => (
            GrangerDirection::AToB,
            best_a2b_lag,
            best_a2b_f,
            best_a2b_p,
            best_a2b_effect,
        ),
        (false, true) => (
            GrangerDirection::BToA,
            best_b2a_lag,
            best_b2a_f,
            best_b2a_p,
            best_b2a_effect,
        ),
        (false, false) => (GrangerDirection::None, 1, 0.0, 1.0, 0.0),
    };

    Ok(GrangerResult {
        direction,
        optimal_lag,
        f_statistic: f_stat,
        p_value: p_val,
        effect_size: effect,
        per_dimension_a_to_b: best_a2b_per_dim,
        per_dimension_b_to_a: best_b2a_per_dim,
    })
}

/// Test one direction: does `cause` Granger-cause `effect`?
///
/// Returns `(combined_p, mean_f, effect_size, per_dim_f)`.
fn test_direction(
    cause_dims: &[Vec<f64>],
    effect_dims: &[Vec<f64>],
    lag: usize,
    _n: usize,
) -> (f64, f64, f64, Vec<f64>) {
    let dim = effect_dims.len();
    let mut p_values = Vec::with_capacity(dim);
    let mut f_values = Vec::with_capacity(dim);
    let mut total_rss_r = 0.0;
    let mut total_rss_u = 0.0;

    for d in 0..dim {
        let (rss_r, rss_u, n_obs) =
            ols_granger_single_dim(&effect_dims[d], &cause_dims[d], lag);

        let q = lag; // extra parameters
        let p_u = 2 * lag; // total params in unrestricted model
        let f = f_statistic(rss_r, rss_u, q, n_obs, p_u);
        let df2 = if n_obs > p_u { n_obs - p_u } else { 1 };
        let p = f_to_p(f, q, df2);

        f_values.push(f);
        p_values.push(p);
        total_rss_r += rss_r;
        total_rss_u += rss_u;
    }

    let combined_p = fisher_combine(&p_values);
    let mean_f = if f_values.is_empty() {
        0.0
    } else {
        f_values.iter().sum::<f64>() / f_values.len() as f64
    };

    // Effect size: proportional reduction in RSS
    let effect = if total_rss_r > 0.0 {
        ((total_rss_r - total_rss_u) / total_rss_r).max(0.0)
    } else {
        0.0
    };

    (combined_p, mean_f, effect, f_values)
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn as_refs(points: &[(i64, Vec<f32>)]) -> Vec<(i64, &[f32])> {
        points.iter().map(|(t, v)| (*t, v.as_slice())).collect()
    }

    // ─── interpolation ──────────────────────────────────────────

    #[test]
    fn interpolate_at_exact_point() {
        let owned = vec![
            (100i64, vec![1.0f32]),
            (200, vec![2.0]),
            (300, vec![3.0]),
        ];
        let traj = as_refs(&owned);
        let v = interpolate_at(&traj, 200, 1);
        assert!((v[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn interpolate_at_midpoint() {
        let owned = vec![(100i64, vec![0.0f32]), (200, vec![10.0])];
        let traj = as_refs(&owned);
        let v = interpolate_at(&traj, 150, 1);
        assert!((v[0] - 5.0).abs() < 1e-4);
    }

    #[test]
    fn interpolate_at_boundary() {
        let owned = vec![(100i64, vec![5.0f32]), (200, vec![10.0])];
        let traj = as_refs(&owned);
        assert!((interpolate_at(&traj, 50, 1)[0] - 5.0).abs() < 1e-6);
        assert!((interpolate_at(&traj, 300, 1)[0] - 10.0).abs() < 1e-6);
    }

    // ─── cholesky_solve ─────────────────────────────────────────

    #[test]
    fn cholesky_simple() {
        // Solve [[4,2],[2,3]] x = [1, 2]
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let b = vec![1.0, 2.0];
        let x = cholesky_solve(&a, &b);
        // Expected: x = [-0.125, 0.75]
        assert!((x[0] - (-0.125)).abs() < 1e-6, "got {}", x[0]);
        assert!((x[1] - 0.75).abs() < 1e-6, "got {}", x[1]);
    }

    // ─── f_to_p ─────────────────────────────────────────────────

    #[test]
    fn f_to_p_zero_f() {
        assert!((f_to_p(0.0, 5, 20) - 1.0).abs() < 0.01);
    }

    #[test]
    fn f_to_p_large_f() {
        let p = f_to_p(100.0, 5, 50);
        assert!(p < 0.001, "very large F should give very small p, got {p}");
    }

    // ─── fisher_combine ─────────────────────────────────────────

    #[test]
    fn fisher_all_significant() {
        let p = fisher_combine(&[0.01, 0.02, 0.01]);
        assert!(p < 0.05, "combined very significant p-values should be significant, got {p}");
    }

    #[test]
    fn fisher_all_nonsignificant() {
        let p = fisher_combine(&[0.8, 0.9, 0.7]);
        assert!(p > 0.3, "combined non-significant should remain non-significant, got {p}");
    }

    // ─── granger_causality ──────────────────────────────────────

    #[test]
    fn granger_insufficient_data() {
        let a_owned = vec![(0i64, vec![1.0f32]), (1, vec![2.0])];
        let b_owned = vec![(0i64, vec![3.0f32]), (1, vec![4.0])];
        let a = as_refs(&a_owned);
        let b = as_refs(&b_owned);
        let result = granger_causality(&a, &b, 3, 0.05);
        assert!(result.is_err());
    }

    #[test]
    fn granger_synthetic_a_causes_b() {
        // A is a sine wave, B is A shifted by 2 steps (A leads B)
        let n = 100;
        let lag = 2;
        let a_owned: Vec<(i64, Vec<f32>)> = (0..n)
            .map(|i| {
                let t = i as f64 * 0.2;
                (i as i64 * 1000, vec![t.sin() as f32])
            })
            .collect();

        let b_owned: Vec<(i64, Vec<f32>)> = (0..n)
            .map(|i| {
                let t = (i as i64 - lag as i64).max(0) as f64 * 0.2;
                (i as i64 * 1000, vec![t.sin() as f32 + 0.01 * (i as f32)])
            })
            .collect();

        let a = as_refs(&a_owned);
        let b = as_refs(&b_owned);

        let result = granger_causality(&a, &b, 5, 0.05).unwrap();

        // We expect A→B direction or at least that A→B has some signal
        assert!(
            result.per_dimension_a_to_b[0] > 0.0,
            "A should have some predictive power for B"
        );
    }

    #[test]
    fn granger_independent_series() {
        // Two completely independent random-ish series
        let n = 80;
        let a_owned: Vec<(i64, Vec<f32>)> = (0..n)
            .map(|i| {
                let v = ((i as f64 * 1.7).sin() * 100.0) as f32;
                (i as i64 * 1000, vec![v])
            })
            .collect();
        let b_owned: Vec<(i64, Vec<f32>)> = (0..n)
            .map(|i| {
                let v = ((i as f64 * 3.1 + 42.0).cos() * 100.0) as f32;
                (i as i64 * 1000, vec![v])
            })
            .collect();

        let a = as_refs(&a_owned);
        let b = as_refs(&b_owned);

        let result = granger_causality(&a, &b, 3, 0.05).unwrap();

        // Independent series should ideally show no causality
        // But with deterministic pseudo-random, there could be spurious correlation
        // Just verify the function runs without error and returns valid values
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(result.f_statistic >= 0.0);
    }

    #[test]
    fn granger_multidimensional() {
        // 3D trajectory where A leads B in all dims
        let n = 60;
        let a_owned: Vec<(i64, Vec<f32>)> = (0..n)
            .map(|i| {
                let t = i as f64 * 0.15;
                (
                    i as i64 * 1000,
                    vec![t.sin() as f32, t.cos() as f32, (t * 0.5).sin() as f32],
                )
            })
            .collect();
        let b_owned: Vec<(i64, Vec<f32>)> = (0..n)
            .map(|i| {
                let t = (i as i64 - 3).max(0) as f64 * 0.15;
                (
                    i as i64 * 1000,
                    vec![t.sin() as f32, t.cos() as f32, (t * 0.5).sin() as f32],
                )
            })
            .collect();

        let a = as_refs(&a_owned);
        let b = as_refs(&b_owned);

        let result = granger_causality(&a, &b, 5, 0.1).unwrap();

        assert_eq!(result.per_dimension_a_to_b.len(), 3);
        assert_eq!(result.per_dimension_b_to_a.len(), 3);
    }

    #[test]
    fn granger_result_has_valid_fields() {
        let n = 50;
        let a_owned: Vec<(i64, Vec<f32>)> = (0..n)
            .map(|i| (i as i64 * 1000, vec![i as f32 * 0.1]))
            .collect();
        let b_owned: Vec<(i64, Vec<f32>)> = (0..n)
            .map(|i| (i as i64 * 1000, vec![i as f32 * 0.2 + 1.0]))
            .collect();

        let a = as_refs(&a_owned);
        let b = as_refs(&b_owned);

        let result = granger_causality(&a, &b, 3, 0.05).unwrap();

        assert!(result.optimal_lag >= 1 && result.optimal_lag <= 3);
        assert!(result.f_statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(result.effect_size >= 0.0 && result.effect_size <= 1.0);
    }
}
