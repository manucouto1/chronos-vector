//! Query type definitions.

use cvx_core::types::{ChangePoint, TemporalFilter, TemporalPoint};
use serde::{Deserialize, Serialize};

/// A temporal query request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TemporalQuery {
    /// k-nearest neighbors at a specific timestamp.
    SnapshotKnn {
        /// Query vector.
        vector: Vec<f32>,
        /// Exact timestamp.
        timestamp: i64,
        /// Number of results.
        k: usize,
    },
    /// k-nearest neighbors over a time range.
    RangeKnn {
        /// Query vector.
        vector: Vec<f32>,
        /// Start timestamp (inclusive).
        start: i64,
        /// End timestamp (inclusive).
        end: i64,
        /// Number of results.
        k: usize,
        /// Semantic vs temporal weight.
        alpha: f32,
    },
    /// Full trajectory for an entity.
    Trajectory {
        /// Entity identifier.
        entity_id: u64,
        /// Temporal filter.
        filter: TemporalFilter,
    },
    /// Velocity at a given timestamp.
    Velocity {
        /// Entity identifier.
        entity_id: u64,
        /// Timestamp to compute velocity at.
        timestamp: i64,
    },
    /// Predict future vector state.
    Prediction {
        /// Entity identifier.
        entity_id: u64,
        /// Target timestamp for prediction.
        target_timestamp: i64,
    },
    /// Detect change points in a time window.
    ChangePointDetect {
        /// Entity identifier.
        entity_id: u64,
        /// Start timestamp.
        start: i64,
        /// End timestamp.
        end: i64,
    },
    /// Drift magnitude between two timestamps.
    DriftQuant {
        /// Entity identifier.
        entity_id: u64,
        /// Start timestamp.
        t1: i64,
        /// End timestamp.
        t2: i64,
        /// Number of top dimensions to report.
        top_n: usize,
    },
    /// Temporal analogy: "entity A at t1 is to A at t2 as B at t3 is to ?"
    Analogy {
        /// Source entity.
        entity_a: u64,
        /// Source timestamp 1.
        t1: i64,
        /// Source timestamp 2.
        t2: i64,
        /// Target entity.
        entity_b: u64,
        /// Target timestamp.
        t3: i64,
    },
}

/// Query result types.
#[derive(Debug, Clone)]
pub enum QueryResult {
    /// kNN results with scores.
    Knn(Vec<KnnResult>),
    /// Trajectory points.
    Trajectory(Vec<TemporalPoint>),
    /// Velocity vector.
    Velocity(Vec<f32>),
    /// Predicted vector.
    Prediction(PredictionResult),
    /// Detected change points.
    ChangePoints(Vec<ChangePoint>),
    /// Drift report.
    Drift(DriftResult),
    /// Analogy result.
    Analogy(Vec<f32>),
}

/// A single kNN result.
#[derive(Debug, Clone)]
pub struct KnnResult {
    /// Entity identifier.
    pub entity_id: u64,
    /// Timestamp of the matched point.
    pub timestamp: i64,
    /// Combined spatiotemporal score.
    pub score: f32,
}

/// Prediction result with confidence.
#[derive(Debug, Clone)]
pub struct PredictionResult {
    /// Predicted vector.
    pub vector: Vec<f32>,
    /// Target timestamp.
    pub timestamp: i64,
    /// Prediction method used.
    pub method: PredictionMethod,
}

/// Method used for prediction.
#[derive(Debug, Clone, Copy)]
pub enum PredictionMethod {
    /// Linear extrapolation from last two points.
    Linear,
    /// Neural ODE integration.
    NeuralOde,
}

/// Drift quantification result.
#[derive(Debug, Clone)]
pub struct DriftResult {
    /// L2 drift magnitude.
    pub l2_magnitude: f32,
    /// Cosine drift.
    pub cosine_drift: f32,
    /// Top changed dimensions: (index, absolute_change).
    pub top_dimensions: Vec<(usize, f32)>,
}
