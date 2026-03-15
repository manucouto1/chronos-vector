//! Error types for ChronosVector.
//!
//! Each subsystem defines its own error type. All convert into [`CvxError`]
//! via `From` implementations, enabling uniform error handling at the API boundary.

use thiserror::Error;

/// Top-level error type wrapping all subsystem errors.
///
/// Library crates use their specific error types internally.
/// At the API boundary, errors are converted to `CvxError` for uniform handling.
#[derive(Debug, Error)]
pub enum CvxError {
    /// Error from the storage subsystem.
    #[error(transparent)]
    Storage(#[from] StorageError),

    /// Error from the index subsystem.
    #[error(transparent)]
    Index(#[from] IndexError),

    /// Error from the query engine.
    #[error(transparent)]
    Query(#[from] QueryError),

    /// Error from the ingestion pipeline.
    #[error(transparent)]
    Ingest(#[from] IngestError),

    /// Error from the analytics engine.
    #[error(transparent)]
    Analytics(#[from] AnalyticsError),

    /// Error from the interpretability layer.
    #[error(transparent)]
    Explain(#[from] ExplainError),

    /// Configuration error.
    #[error("configuration error: {0}")]
    Config(String),
}

/// Convenience alias for `Result<T, CvxError>`.
pub type CvxResult<T> = Result<T, CvxError>;

/// Storage subsystem errors.
#[derive(Debug, Error)]
pub enum StorageError {
    /// Entity not found in the specified space.
    #[error("entity {entity_id} not found in space {space_id}")]
    EntityNotFound {
        /// The requested entity ID.
        entity_id: u64,
        /// The embedding space ID.
        space_id: u32,
    },

    /// Vector dimensionality does not match existing data.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Expected dimensionality.
        expected: usize,
        /// Actual dimensionality received.
        got: usize,
    },

    /// Write-ahead log is corrupted.
    #[error("WAL corrupted at offset {offset}")]
    WalCorrupted {
        /// Byte offset where corruption was detected.
        offset: u64,
    },

    /// Tier migration failed.
    #[error("tier migration failed: {reason}")]
    TierMigration {
        /// Description of the migration failure.
        reason: String,
    },

    /// I/O error from the underlying storage engine.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Index subsystem errors.
#[derive(Debug, Error)]
pub enum IndexError {
    /// Referenced node does not exist in the graph.
    #[error("node {0} not found in graph")]
    NodeNotFound(u32),

    /// Graph structure is inconsistent.
    #[error("graph corrupted: {reason}")]
    GraphCorrupted {
        /// Description of the corruption.
        reason: String,
    },

    /// Vector dimensionality does not match index configuration.
    #[error("dimension mismatch: index expects {expected}, got {got}")]
    DimensionMismatch {
        /// Dimensionality configured for the index.
        expected: usize,
        /// Dimensionality of the provided vector.
        got: usize,
    },

    /// Insertion into the index failed.
    #[error("insert failed: {reason}")]
    InsertFailed {
        /// Description of the failure.
        reason: String,
    },
}

/// Query engine errors.
#[derive(Debug, Error)]
pub enum QueryError {
    /// The query references an entity that does not exist.
    #[error("entity {0} not found")]
    EntityNotFound(u64),

    /// Not enough historical data to perform the requested operation.
    #[error("insufficient data: need {needed} points, have {have}")]
    InsufficientData {
        /// Minimum number of points required.
        needed: usize,
        /// Number of points available.
        have: usize,
    },

    /// The query exceeded the configured timeout.
    #[error("query timeout exceeded")]
    Timeout,

    /// The query planner could not produce a valid plan.
    #[error("planning failed: {reason}")]
    PlanningFailed {
        /// Description of the planning failure.
        reason: String,
    },
}

/// Ingestion pipeline errors.
#[derive(Debug, Error)]
pub enum IngestError {
    /// Input data failed validation checks.
    #[error("validation failed: {reason}")]
    ValidationFailed {
        /// Description of the validation failure.
        reason: String,
    },

    /// Vector dimensionality does not match the entity's existing data.
    #[error("dimension mismatch for entity {entity_id}: expected {expected}, got {got}")]
    DimensionMismatch {
        /// The entity ID.
        entity_id: u64,
        /// Expected dimensionality.
        expected: usize,
        /// Actual dimensionality received.
        got: usize,
    },

    /// The WAL is full and cannot accept more writes.
    #[error("WAL full, backpressure active")]
    WalFull,

    /// Ingestion rate exceeded configured backpressure limits.
    #[error("backpressure threshold exceeded")]
    BackpressureExceeded,
}

/// Analytics engine errors.
#[derive(Debug, Error)]
pub enum AnalyticsError {
    /// The ODE/SDE solver diverged during integration.
    #[error("solver diverged at step {step}")]
    SolverDiverged {
        /// The integration step where divergence was detected.
        step: usize,
    },

    /// Not enough data points to perform the requested analysis.
    #[error("insufficient data: need {needed} points, have {have}")]
    InsufficientData {
        /// Minimum number of points required.
        needed: usize,
        /// Number of points available.
        have: usize,
    },

    /// The required model has not been loaded or trained.
    #[error("model not loaded: {name}")]
    ModelNotLoaded {
        /// Name of the model that was expected.
        name: String,
    },
}

/// Interpretability layer errors.
#[derive(Debug, Error)]
pub enum ExplainError {
    /// The requested projection method is not available.
    #[error("unsupported projection method: {0}")]
    UnsupportedProjection(String),

    /// Not enough data to produce the requested explanation.
    #[error("insufficient data for explanation: {reason}")]
    InsufficientData {
        /// Description of what data is missing.
        reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_error_converts_to_cvx_error() {
        let err = StorageError::EntityNotFound {
            entity_id: 42,
            space_id: 0,
        };
        let cvx_err: CvxError = err.into();
        assert!(matches!(cvx_err, CvxError::Storage(_)));
        assert!(cvx_err.to_string().contains("entity 42"));
    }

    #[test]
    fn index_error_converts_to_cvx_error() {
        let err = IndexError::NodeNotFound(99);
        let cvx_err: CvxError = err.into();
        assert!(matches!(cvx_err, CvxError::Index(_)));
    }

    #[test]
    fn query_error_displays_correctly() {
        let err = QueryError::InsufficientData { needed: 5, have: 2 };
        assert_eq!(err.to_string(), "insufficient data: need 5 points, have 2");
    }

    #[test]
    fn all_error_types_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CvxError>();
        assert_send_sync::<StorageError>();
        assert_send_sync::<IndexError>();
        assert_send_sync::<QueryError>();
        assert_send_sync::<IngestError>();
        assert_send_sync::<AnalyticsError>();
        assert_send_sync::<ExplainError>();
    }
}
