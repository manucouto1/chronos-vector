//! Implementation of the `AnalyticsBackend` trait.
//!
//! Wires together calculus, PELT, and ODE modules into the core trait contract.

use cvx_core::error::AnalyticsError;
use cvx_core::traits::AnalyticsBackend;
use cvx_core::types::{ChangePoint, CpdMethod, TemporalPoint};

use crate::calculus;
use crate::ode;
use crate::pelt::{self, PeltConfig};

/// Default analytics backend using pure-Rust implementations.
pub struct DefaultAnalytics {
    pelt_config: PeltConfig,
}

impl DefaultAnalytics {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self {
            pelt_config: PeltConfig::default(),
        }
    }

    /// Create with custom PELT configuration.
    pub fn with_pelt_config(pelt_config: PeltConfig) -> Self {
        Self { pelt_config }
    }
}

impl Default for DefaultAnalytics {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a slice of `TemporalPoint` to the `(i64, &[f32])` format used internally.
fn to_trajectory(points: &[TemporalPoint]) -> Vec<(i64, &[f32])> {
    points.iter().map(|p| (p.timestamp(), p.vector())).collect()
}

impl AnalyticsBackend for DefaultAnalytics {
    fn predict(
        &self,
        trajectory: &[TemporalPoint],
        target_timestamp: i64,
    ) -> Result<TemporalPoint, AnalyticsError> {
        if trajectory.len() < 2 {
            return Err(AnalyticsError::InsufficientData {
                needed: 2,
                have: trajectory.len(),
            });
        }

        let traj = to_trajectory(trajectory);
        let predicted = ode::linear_extrapolate(&traj, target_timestamp)?;
        let entity_id = trajectory.last().unwrap().entity_id();

        Ok(TemporalPoint::new(entity_id, target_timestamp, predicted))
    }

    fn detect_changepoints(
        &self,
        trajectory: &[TemporalPoint],
        method: CpdMethod,
    ) -> Result<Vec<ChangePoint>, AnalyticsError> {
        if trajectory.is_empty() {
            return Ok(Vec::new());
        }

        let entity_id = trajectory[0].entity_id();
        let traj = to_trajectory(trajectory);

        match method {
            CpdMethod::Pelt => Ok(pelt::detect(entity_id, &traj, &self.pelt_config)),
            CpdMethod::Bocpd => {
                // Use online detector in batch mode
                let mut detector = crate::bocpd::BocpdDetector::new(
                    entity_id,
                    crate::bocpd::BocpdConfig::default(),
                );
                let mut cps = Vec::new();
                for p in trajectory {
                    if let Some(cp) = detector.observe(p.timestamp(), p.vector()) {
                        cps.push(cp);
                    }
                }
                Ok(cps)
            }
        }
    }

    fn velocity(
        &self,
        trajectory: &[TemporalPoint],
        timestamp: i64,
    ) -> Result<Vec<f32>, AnalyticsError> {
        let traj = to_trajectory(trajectory);
        calculus::velocity(&traj, timestamp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trajectory(n: usize, entity_id: u64) -> Vec<TemporalPoint> {
        (0..n)
            .map(|i| TemporalPoint::new(entity_id, (i as i64) * 1000, vec![i as f32 * 0.1; 3]))
            .collect()
    }

    #[test]
    fn predict_returns_point_at_target() {
        let backend = DefaultAnalytics::new();
        let traj = make_trajectory(20, 1);
        let result = backend.predict(&traj, 25_000).unwrap();
        assert_eq!(result.entity_id(), 1);
        assert_eq!(result.timestamp(), 25_000);
        assert_eq!(result.dim(), 3);
    }

    #[test]
    fn predict_insufficient_data() {
        let backend = DefaultAnalytics::new();
        let traj = make_trajectory(1, 1);
        assert!(backend.predict(&traj, 5000).is_err());
    }

    #[test]
    fn detect_changepoints_pelt_near_linear() {
        let backend = DefaultAnalytics::new();
        let traj = make_trajectory(100, 1);
        let cps = backend.detect_changepoints(&traj, CpdMethod::Pelt).unwrap();
        // Near-linear trajectory may have some CPs due to slope, but not many
        assert!(
            cps.len() <= 10,
            "too many CPs on near-linear data: {}",
            cps.len()
        );
    }

    #[test]
    fn detect_changepoints_pelt_with_change() {
        let backend = DefaultAnalytics::new();
        let mut traj = Vec::new();
        for i in 0..50 {
            traj.push(TemporalPoint::new(1, i * 1000, vec![0.0, 0.0]));
        }
        for i in 50..100 {
            traj.push(TemporalPoint::new(1, i * 1000, vec![10.0, 10.0]));
        }
        let cps = backend.detect_changepoints(&traj, CpdMethod::Pelt).unwrap();
        assert!(!cps.is_empty(), "should detect the planted change");
    }

    #[test]
    fn detect_changepoints_bocpd() {
        let backend = DefaultAnalytics::new();
        let mut traj = Vec::new();
        for i in 0..50 {
            traj.push(TemporalPoint::new(1, i * 1000, vec![0.0]));
        }
        for i in 50..100 {
            traj.push(TemporalPoint::new(1, i * 1000, vec![10.0]));
        }
        let cps = backend
            .detect_changepoints(&traj, CpdMethod::Bocpd)
            .unwrap();
        // BOCPD should detect at least 1 change
        assert!(!cps.is_empty());
    }

    #[test]
    fn velocity_linear_trajectory() {
        let backend = DefaultAnalytics::new();
        let traj = make_trajectory(20, 1);
        let vel = backend.velocity(&traj, 10_000).unwrap();
        assert_eq!(vel.len(), 3);
        // Linear trajectory → constant velocity
        for &v in &vel {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn velocity_insufficient_data() {
        let backend = DefaultAnalytics::new();
        let traj = make_trajectory(1, 1);
        assert!(backend.velocity(&traj, 0).is_err());
    }

    #[test]
    fn is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DefaultAnalytics>();
    }
}
