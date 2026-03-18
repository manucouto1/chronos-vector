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
    /// Granger causality test between two entities.
    GrangerCausality {
        /// First entity (potential cause).
        entity_a: u64,
        /// Second entity (potential effect).
        entity_b: u64,
        /// Maximum lag to test.
        max_lag: usize,
        /// Significance threshold (e.g., 0.05).
        significance: f64,
    },
    /// Discover recurring motifs in an entity's trajectory.
    DiscoverMotifs {
        /// Entity identifier.
        entity_id: u64,
        /// Subsequence window size (number of time steps).
        window: usize,
        /// Maximum number of motifs to return.
        max_motifs: usize,
    },
    /// Discover anomalous subsequences (discords) in an entity's trajectory.
    DiscoverDiscords {
        /// Entity identifier.
        entity_id: u64,
        /// Subsequence window size (number of time steps).
        window: usize,
        /// Maximum number of discords to return.
        max_discords: usize,
    },
    /// Temporal join: find convergence windows between two entities.
    TemporalJoin {
        /// First entity.
        entity_a: u64,
        /// Second entity.
        entity_b: u64,
        /// Distance threshold for convergence.
        epsilon: f32,
        /// Window size in microseconds.
        window_us: i64,
    },
    /// Cohort drift analysis across multiple entities.
    CohortDrift {
        /// Entity identifiers in the cohort.
        entity_ids: Vec<u64>,
        /// Start timestamp.
        t1: i64,
        /// End timestamp.
        t2: i64,
        /// Number of top dimensions to report.
        top_n: usize,
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
    /// Granger causality result.
    Granger(GrangerCausalityResult),
    /// Discovered motifs.
    Motifs(Vec<MotifResult>),
    /// Discovered discords.
    Discords(Vec<DiscordResult>),
    /// Temporal join results.
    TemporalJoin(Vec<TemporalJoinResultEntry>),
    /// Cohort drift report.
    CohortDrift(CohortDriftResult),
}

/// A convergence window from a temporal join query.
#[derive(Debug, Clone)]
pub struct TemporalJoinResultEntry {
    /// Start of the convergence window.
    pub start: i64,
    /// End of the convergence window.
    pub end: i64,
    /// Mean distance during convergence.
    pub mean_distance: f32,
    /// Minimum distance during convergence.
    pub min_distance: f32,
    /// Points from entity A in window.
    pub points_a: usize,
    /// Points from entity B in window.
    pub points_b: usize,
}

/// Granger causality test result.
#[derive(Debug, Clone)]
pub struct GrangerCausalityResult {
    /// Detected direction.
    pub direction: String,
    /// Optimal lag.
    pub optimal_lag: usize,
    /// F-statistic.
    pub f_statistic: f64,
    /// Combined p-value.
    pub p_value: f64,
    /// Effect size (partial R²).
    pub effect_size: f64,
    /// Per-dimension F-statistics for A→B.
    pub per_dimension_a_to_b: Vec<f64>,
    /// Per-dimension F-statistics for B→A.
    pub per_dimension_b_to_a: Vec<f64>,
}

/// A discovered motif result.
#[derive(Debug, Clone)]
pub struct MotifResult {
    /// Index of the canonical occurrence.
    pub canonical_index: usize,
    /// All occurrences with timestamps and distances.
    pub occurrences: Vec<MotifOccurrenceResult>,
    /// Detected period (None if aperiodic).
    pub period: Option<usize>,
    /// Mean match distance.
    pub mean_match_distance: f32,
}

/// A single motif occurrence.
#[derive(Debug, Clone)]
pub struct MotifOccurrenceResult {
    /// Start index in trajectory.
    pub start_index: usize,
    /// Timestamp.
    pub timestamp: i64,
    /// Distance to canonical.
    pub distance: f32,
}

/// A discovered discord result.
#[derive(Debug, Clone)]
pub struct DiscordResult {
    /// Start index in trajectory.
    pub start_index: usize,
    /// Timestamp.
    pub timestamp: i64,
    /// Nearest-neighbor distance (higher = more anomalous).
    pub nn_distance: f32,
}

/// Cohort drift analysis result.
#[derive(Debug, Clone)]
pub struct CohortDriftResult {
    /// Number of entities analyzed.
    pub n_entities: usize,
    /// Mean L2 drift across the cohort.
    pub mean_drift_l2: f32,
    /// Median L2 drift.
    pub median_drift_l2: f32,
    /// Standard deviation of drift magnitudes.
    pub std_drift_l2: f32,
    /// Centroid L2 drift magnitude.
    pub centroid_l2_magnitude: f32,
    /// Centroid cosine drift.
    pub centroid_cosine_drift: f32,
    /// Dispersion at t1.
    pub dispersion_t1: f32,
    /// Dispersion at t2.
    pub dispersion_t2: f32,
    /// Dispersion change (positive = diverging).
    pub dispersion_change: f32,
    /// Convergence score (0 = random, 1 = same direction).
    pub convergence_score: f32,
    /// Top changed dimensions: (index, absolute_change).
    pub top_dimensions: Vec<(usize, f32)>,
    /// Outlier entities.
    pub outliers: Vec<CohortOutlierResult>,
}

/// An outlier entity in cohort drift analysis.
#[derive(Debug, Clone)]
pub struct CohortOutlierResult {
    /// Entity identifier.
    pub entity_id: u64,
    /// Individual drift magnitude.
    pub drift_magnitude: f32,
    /// Z-score relative to cohort.
    pub z_score: f32,
    /// Alignment with cohort mean drift direction.
    pub drift_direction_alignment: f32,
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
