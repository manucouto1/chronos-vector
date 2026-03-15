//! Detected change point in an entity's trajectory.

use serde::{Deserialize, Serialize};

use super::CpdMethod;

/// A detected structural break in an entity's embedding trajectory.
///
/// Produced by PELT (offline) or BOCPD (online) change point detection.
/// Stored in the `changepoints` column family.
///
/// # Example
///
/// ```
/// use cvx_core::{ChangePoint, CpdMethod};
///
/// let cp = ChangePoint::new(42, 1_700_000_000, 0.87, vec![0.1, -0.2, 0.05], CpdMethod::Pelt);
/// assert!(cp.severity() > 0.5);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChangePoint {
    entity_id: u64,
    timestamp: i64,
    severity: f64,
    drift_vector: Vec<f32>,
    method: CpdMethod,
}

impl ChangePoint {
    /// Create a new change point.
    pub fn new(
        entity_id: u64,
        timestamp: i64,
        severity: f64,
        drift_vector: Vec<f32>,
        method: CpdMethod,
    ) -> Self {
        Self {
            entity_id,
            timestamp,
            severity,
            drift_vector,
            method,
        }
    }

    /// The entity this change point belongs to.
    pub fn entity_id(&self) -> u64 {
        self.entity_id
    }

    /// When the change was detected.
    pub fn timestamp(&self) -> i64 {
        self.timestamp
    }

    /// Severity of the change (0.0 = negligible, 1.0 = extreme).
    pub fn severity(&self) -> f64 {
        self.severity
    }

    /// Direction and magnitude of the drift at this change point.
    pub fn drift_vector(&self) -> &[f32] {
        &self.drift_vector
    }

    /// Which detection method found this change point.
    pub fn method(&self) -> CpdMethod {
        self.method
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creation_and_accessors() {
        let cp = ChangePoint::new(1, 5000, 0.75, vec![0.1, -0.3], CpdMethod::Bocpd);
        assert_eq!(cp.entity_id(), 1);
        assert_eq!(cp.timestamp(), 5000);
        assert!((cp.severity() - 0.75).abs() < f64::EPSILON);
        assert_eq!(cp.method(), CpdMethod::Bocpd);
    }

    #[test]
    fn postcard_roundtrip() {
        let cp = ChangePoint::new(42, -1000, 0.9, vec![0.5; 768], CpdMethod::Pelt);
        let bytes = postcard::to_allocvec(&cp).unwrap();
        let recovered: ChangePoint = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(cp, recovered);
    }
}
