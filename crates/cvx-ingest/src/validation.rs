//! Input validation for the ingestion pipeline.
//!
//! Validates incoming temporal points before they enter the storage and index layers.
//! Checks: dimension consistency, timestamp sanity, vector norm (no NaN/Inf, not zero).

use cvx_core::error::IngestError;
use cvx_core::types::TemporalPoint;

/// Validation configuration.
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Expected vector dimensionality. 0 = accept any.
    pub expected_dim: usize,
    /// Minimum allowed timestamp (microseconds).
    pub min_timestamp: i64,
    /// Maximum allowed timestamp (microseconds).
    pub max_timestamp: i64,
    /// Whether to reject zero-norm vectors.
    pub reject_zero_vectors: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            expected_dim: 0,
            min_timestamp: i64::MIN,
            max_timestamp: i64::MAX,
            reject_zero_vectors: true,
        }
    }
}

/// Validate a single temporal point.
pub fn validate_point(point: &TemporalPoint, config: &ValidationConfig) -> Result<(), IngestError> {
    // Dimension check
    if config.expected_dim > 0 && point.dim() != config.expected_dim {
        return Err(IngestError::DimensionMismatch {
            entity_id: point.entity_id(),
            expected: config.expected_dim,
            got: point.dim(),
        });
    }

    // Empty vector check
    if point.dim() == 0 {
        return Err(IngestError::ValidationFailed {
            reason: "vector must have at least one dimension".into(),
        });
    }

    // Timestamp range check
    if point.timestamp() < config.min_timestamp || point.timestamp() > config.max_timestamp {
        return Err(IngestError::ValidationFailed {
            reason: format!(
                "timestamp {} outside allowed range [{}, {}]",
                point.timestamp(),
                config.min_timestamp,
                config.max_timestamp
            ),
        });
    }

    // NaN/Inf check
    for (i, &v) in point.vector().iter().enumerate() {
        if v.is_nan() {
            return Err(IngestError::ValidationFailed {
                reason: format!("NaN at dimension {i}"),
            });
        }
        if v.is_infinite() {
            return Err(IngestError::ValidationFailed {
                reason: format!("Infinity at dimension {i}"),
            });
        }
    }

    // Zero vector check
    if config.reject_zero_vectors {
        let norm_sq: f32 = point.vector().iter().map(|v| v * v).sum();
        if norm_sq == 0.0 {
            return Err(IngestError::ValidationFailed {
                reason: "zero vector not allowed".into(),
            });
        }
    }

    Ok(())
}

/// Validate a batch of points. Returns the index of the first invalid point.
pub fn validate_batch(
    points: &[TemporalPoint],
    config: &ValidationConfig,
) -> Result<(), (usize, IngestError)> {
    for (i, point) in points.iter().enumerate() {
        validate_point(point, config).map_err(|e| (i, e))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_point() -> TemporalPoint {
        TemporalPoint::new(1, 1000, vec![0.1, 0.2, 0.3])
    }

    fn config_dim3() -> ValidationConfig {
        ValidationConfig {
            expected_dim: 3,
            ..Default::default()
        }
    }

    #[test]
    fn valid_point_passes() {
        assert!(validate_point(&valid_point(), &config_dim3()).is_ok());
    }

    #[test]
    fn wrong_dimension_rejected() {
        let point = TemporalPoint::new(1, 1000, vec![0.1, 0.2]);
        let err = validate_point(&point, &config_dim3()).unwrap_err();
        assert!(matches!(
            err,
            IngestError::DimensionMismatch {
                expected: 3,
                got: 2,
                ..
            }
        ));
    }

    #[test]
    fn empty_vector_rejected() {
        let point = TemporalPoint::new(1, 1000, vec![]);
        let config = ValidationConfig::default();
        let err = validate_point(&point, &config).unwrap_err();
        assert!(matches!(err, IngestError::ValidationFailed { .. }));
    }

    #[test]
    fn nan_rejected() {
        let point = TemporalPoint::new(1, 1000, vec![0.1, f32::NAN, 0.3]);
        let err = validate_point(&point, &config_dim3()).unwrap_err();
        match err {
            IngestError::ValidationFailed { reason } => assert!(reason.contains("NaN")),
            _ => panic!("expected ValidationFailed"),
        }
    }

    #[test]
    fn infinity_rejected() {
        let point = TemporalPoint::new(1, 1000, vec![0.1, f32::INFINITY, 0.3]);
        let err = validate_point(&point, &config_dim3()).unwrap_err();
        match err {
            IngestError::ValidationFailed { reason } => assert!(reason.contains("Infinity")),
            _ => panic!("expected ValidationFailed"),
        }
    }

    #[test]
    fn zero_vector_rejected_by_default() {
        let point = TemporalPoint::new(1, 1000, vec![0.0, 0.0, 0.0]);
        let err = validate_point(&point, &config_dim3()).unwrap_err();
        match err {
            IngestError::ValidationFailed { reason } => assert!(reason.contains("zero")),
            _ => panic!("expected ValidationFailed"),
        }
    }

    #[test]
    fn zero_vector_allowed_when_configured() {
        let point = TemporalPoint::new(1, 1000, vec![0.0, 0.0, 0.0]);
        let config = ValidationConfig {
            expected_dim: 3,
            reject_zero_vectors: false,
            ..Default::default()
        };
        assert!(validate_point(&point, &config).is_ok());
    }

    #[test]
    fn timestamp_out_of_range() {
        let config = ValidationConfig {
            min_timestamp: 0,
            max_timestamp: 10_000,
            ..Default::default()
        };
        let point = TemporalPoint::new(1, -100, vec![1.0]);
        assert!(validate_point(&point, &config).is_err());

        let point2 = TemporalPoint::new(1, 20_000, vec![1.0]);
        assert!(validate_point(&point2, &config).is_err());

        let point3 = TemporalPoint::new(1, 5000, vec![1.0]);
        assert!(validate_point(&point3, &config).is_ok());
    }

    #[test]
    fn any_dim_accepted_when_expected_dim_is_zero() {
        let config = ValidationConfig::default();
        let p1 = TemporalPoint::new(1, 100, vec![1.0]);
        let p2 = TemporalPoint::new(1, 100, vec![1.0; 768]);
        assert!(validate_point(&p1, &config).is_ok());
        assert!(validate_point(&p2, &config).is_ok());
    }

    #[test]
    fn batch_validation() {
        let config = config_dim3();
        let points = vec![
            TemporalPoint::new(1, 100, vec![1.0, 2.0, 3.0]),
            TemporalPoint::new(2, 200, vec![4.0, 5.0, 6.0]),
            TemporalPoint::new(3, 300, vec![7.0, 8.0]), // wrong dim
        ];
        let (idx, err) = validate_batch(&points, &config).unwrap_err();
        assert_eq!(idx, 2);
        assert!(matches!(err, IngestError::DimensionMismatch { .. }));
    }

    #[test]
    fn batch_all_valid() {
        let config = config_dim3();
        let points = vec![
            TemporalPoint::new(1, 100, vec![1.0, 2.0, 3.0]),
            TemporalPoint::new(2, 200, vec![4.0, 5.0, 6.0]),
        ];
        assert!(validate_batch(&points, &config).is_ok());
    }
}
