//! ODE solver and prediction engine.
//!
//! ## RK45 (Dormand-Prince) Adaptive Solver
//!
//! A 4th/5th order adaptive Runge-Kutta solver for systems of ODEs.
//! Used to integrate learned dynamics $\frac{dy}{dt} = f(t, y)$.
//!
//! ## Linear Extrapolation Fallback
//!
//! When no Neural ODE model is available, uses linear extrapolation
//! from the last two observations as a simple baseline.

use cvx_core::error::AnalyticsError;

/// ODE system: dy/dt = f(t, y).
///
/// Implementations define the right-hand side of the ODE system.
pub trait OdeSystem {
    /// Evaluate the derivative at `(t, y)`.
    fn derivative(&self, t: f64, y: &[f64]) -> Vec<f64>;
}

/// RK45 solver configuration.
#[derive(Debug, Clone)]
pub struct Rk45Config {
    /// Relative tolerance for adaptive step size.
    pub rtol: f64,
    /// Absolute tolerance.
    pub atol: f64,
    /// Initial step size.
    pub h_init: f64,
    /// Minimum step size.
    pub h_min: f64,
    /// Maximum step size.
    pub h_max: f64,
    /// Maximum number of steps.
    pub max_steps: usize,
}

impl Default for Rk45Config {
    fn default() -> Self {
        Self {
            rtol: 1e-6,
            atol: 1e-9,
            h_init: 0.01,
            h_min: 1e-12,
            h_max: 1.0,
            max_steps: 100_000,
        }
    }
}

/// Result of an ODE integration.
#[derive(Debug, Clone)]
pub struct OdeResult {
    /// Final time.
    pub t: f64,
    /// Final state.
    pub y: Vec<f64>,
    /// Number of steps taken.
    pub steps: usize,
}

/// Integrate an ODE system from `(t0, y0)` to `t_end` using Dormand-Prince RK45.
///
/// The Dormand-Prince method uses 7 function evaluations per step with an
/// embedded error estimate for adaptive step-size control.
pub fn rk45_integrate(
    system: &dyn OdeSystem,
    t0: f64,
    y0: &[f64],
    t_end: f64,
    config: &Rk45Config,
) -> Result<OdeResult, AnalyticsError> {
    let dim = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut h = config.h_init.min(config.h_max);
    let direction = if t_end >= t0 { 1.0 } else { -1.0 };
    h *= direction;

    let mut steps = 0;

    // Dormand-Prince coefficients
    let a2 = 1.0 / 5.0;
    let a3 = 3.0 / 10.0;
    let a4 = 4.0 / 5.0;
    let a5 = 8.0 / 9.0;

    let b21 = 1.0 / 5.0;
    let b31 = 3.0 / 40.0;
    let b32 = 9.0 / 40.0;
    let b41 = 44.0 / 45.0;
    let b42 = -56.0 / 15.0;
    let b43 = 32.0 / 9.0;
    let b51 = 19372.0 / 6561.0;
    let b52 = -25360.0 / 2187.0;
    let b53 = 64448.0 / 6561.0;
    let b54 = -212.0 / 729.0;
    let b61 = 9017.0 / 3168.0;
    let b62 = -355.0 / 33.0;
    let b63 = 46732.0 / 5247.0;
    let b64 = 49.0 / 176.0;
    let b65 = -5103.0 / 18656.0;

    // 5th order weights
    let c1 = 35.0 / 384.0;
    let c3 = 500.0 / 1113.0;
    let c4 = 125.0 / 192.0;
    let c5 = -2187.0 / 6784.0;
    let c6 = 11.0 / 84.0;

    // 4th order weights (for error estimation)
    let e1 = 71.0 / 57600.0;
    let e3 = -71.0 / 16695.0;
    let e4 = 71.0 / 1920.0;
    let e5 = -17253.0 / 339200.0;
    let e6 = 22.0 / 525.0;
    let e7 = -1.0 / 40.0;

    while (t_end - t) * direction > 1e-15 {
        if steps >= config.max_steps {
            return Err(AnalyticsError::SolverDiverged { step: steps });
        }

        // Clamp step to not overshoot
        if (t + h - t_end) * direction > 0.0 {
            h = t_end - t;
        }

        // k1 = f(t, y)
        let k1 = system.derivative(t, &y);

        // k2
        let y2: Vec<f64> = (0..dim).map(|i| y[i] + h * b21 * k1[i]).collect();
        let k2 = system.derivative(t + a2 * h, &y2);

        // k3
        let y3: Vec<f64> = (0..dim)
            .map(|i| y[i] + h * (b31 * k1[i] + b32 * k2[i]))
            .collect();
        let k3 = system.derivative(t + a3 * h, &y3);

        // k4
        let y4: Vec<f64> = (0..dim)
            .map(|i| y[i] + h * (b41 * k1[i] + b42 * k2[i] + b43 * k3[i]))
            .collect();
        let k4 = system.derivative(t + a4 * h, &y4);

        // k5
        let y5: Vec<f64> = (0..dim)
            .map(|i| y[i] + h * (b51 * k1[i] + b52 * k2[i] + b53 * k3[i] + b54 * k4[i]))
            .collect();
        let k5 = system.derivative(t + a5 * h, &y5);

        // k6
        let y6: Vec<f64> = (0..dim)
            .map(|i| {
                y[i] + h * (b61 * k1[i] + b62 * k2[i] + b63 * k3[i] + b64 * k4[i] + b65 * k5[i])
            })
            .collect();
        let k6 = system.derivative(t + h, &y6);

        // 5th order solution
        let y_new: Vec<f64> = (0..dim)
            .map(|i| y[i] + h * (c1 * k1[i] + c3 * k3[i] + c4 * k4[i] + c5 * k5[i] + c6 * k6[i]))
            .collect();

        // k7 (for error estimate)
        let k7 = system.derivative(t + h, &y_new);

        // Error estimate
        let mut err = 0.0;
        for i in 0..dim {
            let ei =
                h * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] + e6 * k6[i] + e7 * k7[i]);
            let scale = config.atol + config.rtol * y_new[i].abs().max(y[i].abs());
            err += (ei / scale) * (ei / scale);
        }
        err = (err / dim as f64).sqrt();

        if err <= 1.0 {
            // Accept step
            t += h;
            y = y_new;
            steps += 1;
        }

        // Adjust step size
        let safety = 0.9;
        let factor = if err > 0.0 {
            safety * err.powf(-0.2)
        } else {
            5.0
        };
        h *= factor.clamp(0.2, 5.0);
        h = h.abs().clamp(config.h_min, config.h_max) * direction;
    }

    Ok(OdeResult { t, y, steps })
}

// ─── Linear Extrapolation Fallback ──────────────────────────────────

/// Predict a future vector using linear extrapolation.
///
/// Uses the last two observations to estimate velocity and extrapolate.
pub fn linear_extrapolate(
    trajectory: &[(i64, &[f32])],
    target_timestamp: i64,
) -> Result<Vec<f32>, AnalyticsError> {
    if trajectory.len() < 2 {
        return Err(AnalyticsError::InsufficientData {
            needed: 2,
            have: trajectory.len(),
        });
    }

    let n = trajectory.len();
    let (t1, v1) = &trajectory[n - 2];
    let (t2, v2) = &trajectory[n - 1];

    let dt = (*t2 - *t1) as f64;
    if dt == 0.0 {
        return Ok(v2.to_vec());
    }

    let dt_target = (target_timestamp - *t2) as f64;
    let ratio = dt_target / dt;

    let predicted: Vec<f32> = v1
        .iter()
        .zip(v2.iter())
        .map(|(&a, &b)| {
            let vel = b as f64 - a as f64;
            (b as f64 + vel * ratio) as f32
        })
        .collect();

    Ok(predicted)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Simple ODE systems for testing ─────────────────────────────

    /// dy/dt = -y (exponential decay: y(t) = y0 * e^{-t})
    struct ExponentialDecay;
    impl OdeSystem for ExponentialDecay {
        fn derivative(&self, _t: f64, y: &[f64]) -> Vec<f64> {
            vec![-y[0]]
        }
    }

    /// dx/dt = v, dv/dt = -x (simple harmonic oscillator: x(t) = cos(t))
    struct HarmonicOscillator;
    impl OdeSystem for HarmonicOscillator {
        fn derivative(&self, _t: f64, y: &[f64]) -> Vec<f64> {
            vec![y[1], -y[0]] // [dx/dt, dv/dt]
        }
    }

    /// dy/dt = y (exponential growth)
    struct ExponentialGrowth;
    impl OdeSystem for ExponentialGrowth {
        fn derivative(&self, _t: f64, y: &[f64]) -> Vec<f64> {
            vec![y[0]]
        }
    }

    // ─── RK45 tests ─────────────────────────────────────────────────

    #[test]
    fn exponential_decay_matches_analytical() {
        let system = ExponentialDecay;
        let y0 = [1.0];
        let t_end = 5.0;
        let config = Rk45Config {
            rtol: 1e-8,
            ..Default::default()
        };

        let result = rk45_integrate(&system, 0.0, &y0, t_end, &config).unwrap();
        let analytical = (-t_end).exp();

        assert!(
            (result.y[0] - analytical).abs() < 1e-5,
            "RK45: {}, analytical: {analytical}",
            result.y[0]
        );
    }

    #[test]
    fn harmonic_oscillator_matches_analytical() {
        let system = HarmonicOscillator;
        let y0 = [1.0, 0.0]; // x=1, v=0 → x(t) = cos(t)
        let t_end = 2.0 * std::f64::consts::PI;
        let config = Rk45Config {
            rtol: 1e-8,
            ..Default::default()
        };

        let result = rk45_integrate(&system, 0.0, &y0, t_end, &config).unwrap();
        // After one full period, should return to (1, 0)
        assert!(
            (result.y[0] - 1.0).abs() < 1e-5,
            "x after full period: {}, expected 1.0",
            result.y[0]
        );
        assert!(
            result.y[1].abs() < 1e-5,
            "v after full period: {}, expected 0.0",
            result.y[1]
        );
    }

    #[test]
    fn rk45_adaptive_step_count() {
        // Simple decay should need few steps with good tolerance
        let system = ExponentialDecay;
        let config = Rk45Config {
            rtol: 1e-6,
            ..Default::default()
        };

        let result = rk45_integrate(&system, 0.0, &[1.0], 1.0, &config).unwrap();
        assert!(
            result.steps < 100,
            "should need few steps, got {}",
            result.steps
        );
    }

    #[test]
    fn rk45_backward_integration() {
        let system = ExponentialDecay;
        let config = Rk45Config::default();

        // Integrate backward: from t=1 to t=0
        let y_at_1 = (-1.0f64).exp();
        let result = rk45_integrate(&system, 1.0, &[y_at_1], 0.0, &config).unwrap();
        assert!(
            (result.y[0] - 1.0).abs() < 1e-4,
            "backward: {}, expected 1.0",
            result.y[0]
        );
    }

    #[test]
    fn rk45_exponential_growth() {
        let system = ExponentialGrowth;
        let config = Rk45Config {
            rtol: 1e-6,
            h_max: 0.5,
            ..Default::default()
        };

        let result = rk45_integrate(&system, 0.0, &[1.0], 3.0, &config).unwrap();
        let analytical = 3.0f64.exp();
        assert!(
            (result.y[0] - analytical).abs() / analytical < 1e-5,
            "growth: {}, analytical: {analytical}",
            result.y[0]
        );
    }

    // ─── Linear extrapolation ───────────────────────────────────────

    #[test]
    fn linear_extrapolate_constant() {
        let points = vec![(0i64, vec![1.0f32, 2.0, 3.0]), (1000, vec![1.0, 2.0, 3.0])];
        let traj: Vec<(i64, &[f32])> = points.iter().map(|(t, v)| (*t, v.as_slice())).collect();

        let pred = linear_extrapolate(&traj, 5000).unwrap();
        assert_eq!(pred, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn linear_extrapolate_trend() {
        let points = vec![(0i64, vec![0.0f32, 0.0]), (1000, vec![1.0, 2.0])];
        let traj: Vec<(i64, &[f32])> = points.iter().map(|(t, v)| (*t, v.as_slice())).collect();

        let pred = linear_extrapolate(&traj, 2000).unwrap();
        // After 1 more unit: [2.0, 4.0]
        assert!((pred[0] - 2.0).abs() < 1e-5);
        assert!((pred[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn linear_extrapolate_backward() {
        let points = vec![(0i64, vec![0.0f32]), (1000, vec![10.0])];
        let traj: Vec<(i64, &[f32])> = points.iter().map(|(t, v)| (*t, v.as_slice())).collect();

        let pred = linear_extrapolate(&traj, -1000).unwrap();
        // 2 units before last: 10 + 10 * (-2) = -10
        assert!((pred[0] - (-10.0)).abs() < 1e-5);
    }

    #[test]
    fn linear_extrapolate_insufficient_data() {
        let points = vec![(0i64, vec![1.0f32])];
        let traj: Vec<(i64, &[f32])> = points.iter().map(|(t, v)| (*t, v.as_slice())).collect();
        assert!(linear_extrapolate(&traj, 1000).is_err());
    }
}
