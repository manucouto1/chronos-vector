//! REST endpoint handlers for ChronosVector.
//!
//! All handlers extract `State<SharedState>` and return JSON responses.
//! OpenAPI documentation is auto-generated via `utoipa` annotations.

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use cvx_core::{TemporalFilter, TemporalPoint};
use cvx_ingest::validation::validate_point;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::state::SharedState;

// ─── Request / Response types ───────────────────────────────────────

/// Ingest request: a single temporal point.
#[derive(Debug, Deserialize, ToSchema)]
pub struct IngestRequest {
    /// Entity identifier.
    pub entity_id: u64,
    /// Timestamp in microseconds.
    pub timestamp: i64,
    /// Embedding vector.
    pub vector: Vec<f32>,
}

/// Batch ingest request.
#[derive(Debug, Deserialize, ToSchema)]
pub struct BatchIngestRequest {
    /// List of points to ingest.
    pub points: Vec<IngestRequest>,
}

/// Ingest receipt for a single point.
#[derive(Debug, Serialize, ToSchema)]
pub struct IngestReceipt {
    /// Internal node ID assigned to this point.
    pub node_id: u32,
    /// Entity identifier.
    pub entity_id: u64,
    /// Timestamp.
    pub timestamp: i64,
}

/// Batch ingest response.
#[derive(Debug, Serialize, ToSchema)]
pub struct BatchIngestResponse {
    /// Number of points successfully ingested.
    pub ingested: usize,
    /// Receipts for each ingested point.
    pub receipts: Vec<IngestReceipt>,
}

/// Query request.
#[derive(Debug, Deserialize, ToSchema)]
pub struct QueryRequest {
    /// Query vector.
    pub vector: Vec<f32>,
    /// Number of results.
    #[serde(default = "default_k")]
    pub k: usize,
    /// Temporal filter.
    #[serde(default)]
    pub filter: QueryFilter,
    /// Semantic vs temporal weight (1.0 = pure semantic).
    #[serde(default = "default_alpha")]
    pub alpha: f32,
    /// Reference timestamp for temporal distance.
    #[serde(default)]
    pub query_timestamp: i64,
}

fn default_k() -> usize {
    10
}

fn default_alpha() -> f32 {
    1.0
}

/// Temporal filter for queries.
#[derive(Debug, Deserialize, Default, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum QueryFilter {
    /// No temporal constraint.
    #[default]
    All,
    /// Exact timestamp match.
    Snapshot {
        /// Target timestamp.
        timestamp: i64,
    },
    /// Time range.
    Range {
        /// Start timestamp (inclusive).
        start: i64,
        /// End timestamp (inclusive).
        end: i64,
    },
    /// Before timestamp.
    Before {
        /// Maximum timestamp (inclusive).
        timestamp: i64,
    },
    /// After timestamp.
    After {
        /// Minimum timestamp (inclusive).
        timestamp: i64,
    },
}

impl From<QueryFilter> for TemporalFilter {
    fn from(f: QueryFilter) -> Self {
        match f {
            QueryFilter::All => TemporalFilter::All,
            QueryFilter::Snapshot { timestamp } => TemporalFilter::Snapshot(timestamp),
            QueryFilter::Range { start, end } => TemporalFilter::Range(start, end),
            QueryFilter::Before { timestamp } => TemporalFilter::Before(timestamp),
            QueryFilter::After { timestamp } => TemporalFilter::After(timestamp),
        }
    }
}

/// A single query result.
#[derive(Debug, Serialize, ToSchema)]
pub struct QueryResult {
    /// Internal node ID.
    pub node_id: u32,
    /// Entity identifier.
    pub entity_id: u64,
    /// Timestamp of the matched point.
    pub timestamp: i64,
    /// Combined distance score.
    pub score: f32,
}

/// Query response.
#[derive(Debug, Serialize, ToSchema)]
pub struct QueryResponse {
    /// Search results ordered by score.
    pub results: Vec<QueryResult>,
}

/// Trajectory entry.
#[derive(Debug, Serialize, ToSchema)]
pub struct TrajectoryEntry {
    /// Timestamp.
    pub timestamp: i64,
    /// Node ID.
    pub node_id: u32,
}

/// Trajectory response.
#[derive(Debug, Serialize, ToSchema)]
pub struct TrajectoryResponse {
    /// Entity identifier.
    pub entity_id: u64,
    /// Trajectory points ordered by timestamp.
    pub points: Vec<TrajectoryEntry>,
}

/// Health response.
#[derive(Debug, Serialize, ToSchema)]
pub struct HealthResponse {
    /// Server status.
    pub status: String,
    /// Server version.
    pub version: String,
    /// Uptime in seconds.
    pub uptime_secs: u64,
    /// Number of indexed vectors.
    pub index_size: usize,
}

/// Error response body.
#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorResponse {
    /// Error message.
    pub error: String,
}

/// Velocity request params.
#[derive(Debug, Deserialize, ToSchema)]
pub struct VelocityParams {
    /// Timestamp to compute velocity at.
    pub timestamp: i64,
}

/// Velocity response.
#[derive(Debug, Serialize, ToSchema)]
pub struct VelocityResponse {
    /// Entity identifier.
    pub entity_id: u64,
    /// Timestamp.
    pub timestamp: i64,
    /// Velocity vector per dimension.
    pub velocity: Vec<f32>,
}

/// Drift request params.
#[derive(Debug, Deserialize, ToSchema)]
pub struct DriftParams {
    /// Start timestamp.
    pub t1: i64,
    /// End timestamp.
    pub t2: i64,
    /// Number of top dimensions (default 5).
    #[serde(default = "default_top_n")]
    pub top_n: usize,
}

fn default_top_n() -> usize {
    5
}

/// Drift response.
#[derive(Debug, Serialize, ToSchema)]
pub struct DriftResponse {
    /// Entity identifier.
    pub entity_id: u64,
    /// L2 drift magnitude.
    pub l2_magnitude: f32,
    /// Cosine drift (1 - similarity).
    pub cosine_drift: f32,
    /// Top changed dimensions.
    pub top_dimensions: Vec<DimensionChange>,
}

/// A single dimension change.
#[derive(Debug, Serialize, ToSchema)]
pub struct DimensionChange {
    /// Dimension index.
    pub index: usize,
    /// Absolute change.
    pub change: f32,
}

/// Changepoint request params.
#[derive(Debug, Deserialize, ToSchema)]
pub struct ChangepointParams {
    /// Start timestamp.
    pub start: i64,
    /// End timestamp.
    pub end: i64,
}

/// Changepoint response.
#[derive(Debug, Serialize, ToSchema)]
pub struct ChangepointResponse {
    /// Entity identifier.
    pub entity_id: u64,
    /// Detected change points.
    pub changepoints: Vec<ChangepointEntry>,
}

/// A detected change point.
#[derive(Debug, Serialize, ToSchema)]
pub struct ChangepointEntry {
    /// Timestamp of the change.
    pub timestamp: i64,
    /// Severity [0, 1].
    pub severity: f64,
}

/// Prediction request params.
#[derive(Debug, Deserialize, ToSchema)]
pub struct PredictionParams {
    /// Target timestamp for prediction.
    pub target_timestamp: i64,
}

/// Prediction response.
#[derive(Debug, Serialize, ToSchema)]
pub struct PredictionResponse {
    /// Entity identifier.
    pub entity_id: u64,
    /// Predicted vector.
    pub vector: Vec<f32>,
    /// Target timestamp.
    pub timestamp: i64,
    /// Method used.
    pub method: String,
}

/// Counterfactual request params.
#[derive(Debug, Deserialize, ToSchema)]
pub struct CounterfactualParams {
    /// Change point timestamp.
    pub change_point: i64,
}

/// Divergence point in the counterfactual analysis.
#[derive(Debug, Serialize, ToSchema)]
pub struct DivergenceEntry {
    /// Timestamp.
    pub timestamp: i64,
    /// Distance between actual and counterfactual.
    pub distance: f32,
}

/// Counterfactual response.
#[derive(Debug, Serialize, ToSchema)]
pub struct CounterfactualResponse {
    /// Entity identifier.
    pub entity_id: u64,
    /// Change point timestamp.
    pub change_point: i64,
    /// Total divergence (area under curve).
    pub total_divergence: f64,
    /// Timestamp of maximum divergence.
    pub max_divergence_time: i64,
    /// Maximum divergence value.
    pub max_divergence_value: f32,
    /// Divergence curve.
    pub divergence_curve: Vec<DivergenceEntry>,
    /// Method used.
    pub method: String,
}

/// Granger causality request.
#[derive(Debug, Deserialize, ToSchema)]
pub struct GrangerRequest {
    /// First entity (potential cause).
    pub entity_a: u64,
    /// Second entity (potential effect).
    pub entity_b: u64,
    /// Maximum lag to test (default 5).
    #[serde(default = "default_top_n")]
    pub max_lag: usize,
    /// Significance threshold (default 0.05).
    #[serde(default = "default_significance")]
    pub significance: f64,
}

fn default_significance() -> f64 {
    0.05
}

/// Granger causality response.
#[derive(Debug, Serialize, ToSchema)]
pub struct GrangerResponse {
    /// Detected direction: "a_to_b", "b_to_a", "bidirectional", or "none".
    pub direction: String,
    /// Optimal lag (time steps).
    pub optimal_lag: usize,
    /// F-statistic.
    pub f_statistic: f64,
    /// Combined p-value.
    pub p_value: f64,
    /// Effect size (partial R²).
    pub effect_size: f64,
}

/// Motif discovery request params.
#[derive(Debug, Deserialize, ToSchema)]
pub struct MotifParams {
    /// Subsequence window size (number of time steps).
    pub window: usize,
    /// Maximum motifs to return (default 5).
    #[serde(default = "default_top_n")]
    pub max_motifs: usize,
}

/// A motif occurrence entry.
#[derive(Debug, Serialize, ToSchema)]
pub struct MotifOccurrenceEntry {
    /// Start index in trajectory.
    pub start_index: usize,
    /// Timestamp.
    pub timestamp: i64,
    /// Distance to canonical.
    pub distance: f32,
}

/// A discovered motif.
#[derive(Debug, Serialize, ToSchema)]
pub struct MotifEntry {
    /// Index of canonical occurrence.
    pub canonical_index: usize,
    /// All occurrences.
    pub occurrences: Vec<MotifOccurrenceEntry>,
    /// Detected period (null if aperiodic).
    pub period: Option<usize>,
    /// Mean match distance.
    pub mean_match_distance: f32,
}

/// Motifs response.
#[derive(Debug, Serialize, ToSchema)]
pub struct MotifsResponse {
    /// Entity identifier.
    pub entity_id: u64,
    /// Discovered motifs.
    pub motifs: Vec<MotifEntry>,
}

/// Discord discovery request params.
#[derive(Debug, Deserialize, ToSchema)]
pub struct DiscordParams {
    /// Subsequence window size.
    pub window: usize,
    /// Maximum discords to return (default 5).
    #[serde(default = "default_top_n")]
    pub max_discords: usize,
}

/// A discovered discord.
#[derive(Debug, Serialize, ToSchema)]
pub struct DiscordEntry {
    /// Start index in trajectory.
    pub start_index: usize,
    /// Timestamp.
    pub timestamp: i64,
    /// Nearest-neighbor distance (higher = more anomalous).
    pub nn_distance: f32,
}

/// Discords response.
#[derive(Debug, Serialize, ToSchema)]
pub struct DiscordsResponse {
    /// Entity identifier.
    pub entity_id: u64,
    /// Discovered discords.
    pub discords: Vec<DiscordEntry>,
}

/// Temporal join request.
#[derive(Debug, Deserialize, ToSchema)]
pub struct TemporalJoinRequest {
    /// First entity.
    pub entity_a: u64,
    /// Second entity.
    pub entity_b: u64,
    /// Distance threshold for convergence.
    pub epsilon: f32,
    /// Window size in days.
    #[serde(default = "default_window_days")]
    pub window_days: f64,
}

fn default_window_days() -> f64 {
    7.0
}

/// A convergence window entry.
#[derive(Debug, Serialize, ToSchema)]
pub struct TemporalJoinEntry {
    /// Start timestamp.
    pub start: i64,
    /// End timestamp.
    pub end: i64,
    /// Mean distance during convergence.
    pub mean_distance: f32,
    /// Minimum distance.
    pub min_distance: f32,
    /// Points from entity A.
    pub points_a: usize,
    /// Points from entity B.
    pub points_b: usize,
}

/// Temporal join response.
#[derive(Debug, Serialize, ToSchema)]
pub struct TemporalJoinResponse {
    /// Entity A.
    pub entity_a: u64,
    /// Entity B.
    pub entity_b: u64,
    /// Convergence windows.
    pub windows: Vec<TemporalJoinEntry>,
}

/// Cohort drift request.
#[derive(Debug, Deserialize, ToSchema)]
pub struct CohortDriftRequest {
    /// Entity identifiers in the cohort.
    pub entity_ids: Vec<u64>,
    /// Start timestamp.
    pub t1: i64,
    /// End timestamp.
    pub t2: i64,
    /// Number of top dimensions (default 5).
    #[serde(default = "default_top_n")]
    pub top_n: usize,
}

/// Cohort drift response.
#[derive(Debug, Serialize, ToSchema)]
pub struct CohortDriftResponse {
    /// Number of entities analyzed.
    pub n_entities: usize,
    /// Mean L2 drift magnitude.
    pub mean_drift_l2: f32,
    /// Median L2 drift magnitude.
    pub median_drift_l2: f32,
    /// Standard deviation of drift magnitudes.
    pub std_drift_l2: f32,
    /// Centroid L2 drift.
    pub centroid_l2_magnitude: f32,
    /// Centroid cosine drift.
    pub centroid_cosine_drift: f32,
    /// Dispersion at t1.
    pub dispersion_t1: f32,
    /// Dispersion at t2.
    pub dispersion_t2: f32,
    /// Dispersion change (positive = diverging).
    pub dispersion_change: f32,
    /// Convergence score (0 = random, 1 = all same direction).
    pub convergence_score: f32,
    /// Top changed dimensions.
    pub top_dimensions: Vec<DimensionChange>,
    /// Outlier entities.
    pub outliers: Vec<CohortOutlierEntry>,
}

/// An outlier entity in cohort drift analysis.
#[derive(Debug, Serialize, ToSchema)]
pub struct CohortOutlierEntry {
    /// Entity identifier.
    pub entity_id: u64,
    /// Individual drift magnitude.
    pub drift_magnitude: f32,
    /// Z-score relative to cohort.
    pub z_score: f32,
    /// Alignment with cohort drift direction.
    pub drift_direction_alignment: f32,
}

/// Analogy request.
#[derive(Debug, Deserialize, ToSchema)]
pub struct AnalogyRequest {
    /// Source entity A.
    pub entity_a: u64,
    /// Source timestamp 1.
    pub t1: i64,
    /// Source timestamp 2.
    pub t2: i64,
    /// Target entity B.
    pub entity_b: u64,
    /// Target timestamp 3.
    pub t3: i64,
}

/// Analogy response.
#[derive(Debug, Serialize, ToSchema)]
pub struct AnalogyResponse {
    /// Resulting vector: B@t3 + (A@t2 - A@t1).
    pub vector: Vec<f32>,
}

// ─── Handlers ───────────────────────────────────────────────────────

/// Batch ingest temporal points.
#[utoipa::path(
    post,
    path = "/v1/ingest",
    request_body = BatchIngestRequest,
    responses(
        (status = 200, description = "Points ingested successfully", body = BatchIngestResponse),
        (status = 400, description = "Validation error", body = ErrorResponse),
    ),
    tag = "ingestion"
)]
pub async fn ingest(
    State(state): State<SharedState>,
    Json(req): Json<BatchIngestRequest>,
) -> Result<Json<BatchIngestResponse>, (StatusCode, Json<ErrorResponse>)> {
    if req.points.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "empty batch".into(),
            }),
        ));
    }

    let mut receipts = Vec::with_capacity(req.points.len());

    for p in &req.points {
        let point = TemporalPoint::new(p.entity_id, p.timestamp, p.vector.clone());

        if let Err(e) = validate_point(&point, &state.validation) {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            ));
        }

        let node_id = state.index.insert(p.entity_id, p.timestamp, &p.vector);

        state.store.put(0, &point).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            )
        })?;

        receipts.push(IngestReceipt {
            node_id,
            entity_id: p.entity_id,
            timestamp: p.timestamp,
        });
    }

    Ok(Json(BatchIngestResponse {
        ingested: receipts.len(),
        receipts,
    }))
}

/// Spatiotemporal kNN search.
#[utoipa::path(
    post,
    path = "/v1/query",
    request_body = QueryRequest,
    responses(
        (status = 200, description = "Search results", body = QueryResponse),
        (status = 400, description = "Invalid query", body = ErrorResponse),
    ),
    tag = "query"
)]
pub async fn query(
    State(state): State<SharedState>,
    Json(req): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, (StatusCode, Json<ErrorResponse>)> {
    if req.vector.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "query vector must not be empty".into(),
            }),
        ));
    }

    let filter: TemporalFilter = req.filter.into();
    let raw_results =
        state
            .index
            .search(&req.vector, req.k, filter, req.alpha, req.query_timestamp);

    let results = raw_results
        .into_iter()
        .map(|(node_id, score)| QueryResult {
            node_id,
            entity_id: state.index.entity_id(node_id),
            timestamp: state.index.timestamp(node_id),
            score,
        })
        .collect();

    Ok(Json(QueryResponse { results }))
}

/// Retrieve entity trajectory.
#[utoipa::path(
    get,
    path = "/v1/entities/{id}/trajectory",
    params(("id" = u64, Path, description = "Entity identifier")),
    responses(
        (status = 200, description = "Entity trajectory", body = TrajectoryResponse),
    ),
    tag = "query"
)]
pub async fn trajectory(
    State(state): State<SharedState>,
    Path(entity_id): Path<u64>,
) -> Json<TrajectoryResponse> {
    let points = state.index.trajectory(entity_id, TemporalFilter::All);

    let entries = points
        .into_iter()
        .map(|(timestamp, node_id)| TrajectoryEntry { timestamp, node_id })
        .collect();

    Json(TrajectoryResponse {
        entity_id,
        points: entries,
    })
}

/// Compute velocity at timestamp.
#[utoipa::path(
    get,
    path = "/v1/entities/{id}/velocity",
    params(
        ("id" = u64, Path, description = "Entity identifier"),
        ("timestamp" = i64, Query, description = "Timestamp to compute velocity at"),
    ),
    responses(
        (status = 200, description = "Velocity vector", body = VelocityResponse),
        (status = 400, description = "Insufficient data", body = ErrorResponse),
    ),
    tag = "analytics"
)]
pub async fn velocity(
    State(state): State<SharedState>,
    Path(entity_id): Path<u64>,
    axum::extract::Query(params): axum::extract::Query<VelocityParams>,
) -> Result<Json<VelocityResponse>, (StatusCode, Json<ErrorResponse>)> {
    let result = cvx_query::engine::execute_query(
        &state.index,
        cvx_query::types::TemporalQuery::Velocity {
            entity_id,
            timestamp: params.timestamp,
        },
    )
    .map_err(query_err)?;

    if let cvx_query::types::QueryResult::Velocity(vel) = result {
        Ok(Json(VelocityResponse {
            entity_id,
            timestamp: params.timestamp,
            velocity: vel,
        }))
    } else {
        unreachable!()
    }
}

/// Drift quantification between two timestamps.
#[utoipa::path(
    get,
    path = "/v1/entities/{id}/drift",
    params(
        ("id" = u64, Path, description = "Entity identifier"),
        ("t1" = i64, Query, description = "Start timestamp"),
        ("t2" = i64, Query, description = "End timestamp"),
        ("top_n" = Option<usize>, Query, description = "Number of top dimensions (default 5)"),
    ),
    responses(
        (status = 200, description = "Drift report", body = DriftResponse),
        (status = 404, description = "Entity not found", body = ErrorResponse),
    ),
    tag = "analytics"
)]
pub async fn drift(
    State(state): State<SharedState>,
    Path(entity_id): Path<u64>,
    axum::extract::Query(params): axum::extract::Query<DriftParams>,
) -> Result<Json<DriftResponse>, (StatusCode, Json<ErrorResponse>)> {
    let result = cvx_query::engine::execute_query(
        &state.index,
        cvx_query::types::TemporalQuery::DriftQuant {
            entity_id,
            t1: params.t1,
            t2: params.t2,
            top_n: params.top_n,
        },
    )
    .map_err(query_err)?;

    if let cvx_query::types::QueryResult::Drift(drift) = result {
        Ok(Json(DriftResponse {
            entity_id,
            l2_magnitude: drift.l2_magnitude,
            cosine_drift: drift.cosine_drift,
            top_dimensions: drift
                .top_dimensions
                .into_iter()
                .map(|(index, change)| DimensionChange { index, change })
                .collect(),
        }))
    } else {
        unreachable!()
    }
}

/// Detect change points in a time window.
#[utoipa::path(
    get,
    path = "/v1/entities/{id}/changepoints",
    params(
        ("id" = u64, Path, description = "Entity identifier"),
        ("start" = i64, Query, description = "Start timestamp"),
        ("end" = i64, Query, description = "End timestamp"),
    ),
    responses(
        (status = 200, description = "Detected change points", body = ChangepointResponse),
    ),
    tag = "analytics"
)]
pub async fn changepoints(
    State(state): State<SharedState>,
    Path(entity_id): Path<u64>,
    axum::extract::Query(params): axum::extract::Query<ChangepointParams>,
) -> Result<Json<ChangepointResponse>, (StatusCode, Json<ErrorResponse>)> {
    let result = cvx_query::engine::execute_query(
        &state.index,
        cvx_query::types::TemporalQuery::ChangePointDetect {
            entity_id,
            start: params.start,
            end: params.end,
        },
    )
    .map_err(query_err)?;

    if let cvx_query::types::QueryResult::ChangePoints(cps) = result {
        Ok(Json(ChangepointResponse {
            entity_id,
            changepoints: cps
                .into_iter()
                .map(|cp| ChangepointEntry {
                    timestamp: cp.timestamp(),
                    severity: cp.severity(),
                })
                .collect(),
        }))
    } else {
        unreachable!()
    }
}

/// Predict future vector state via linear extrapolation.
#[utoipa::path(
    get,
    path = "/v1/entities/{id}/prediction",
    params(
        ("id" = u64, Path, description = "Entity identifier"),
        ("target_timestamp" = i64, Query, description = "Target timestamp for prediction"),
    ),
    responses(
        (status = 200, description = "Predicted vector", body = PredictionResponse),
        (status = 400, description = "Insufficient data", body = ErrorResponse),
    ),
    tag = "analytics"
)]
pub async fn prediction(
    State(state): State<SharedState>,
    Path(entity_id): Path<u64>,
    axum::extract::Query(params): axum::extract::Query<PredictionParams>,
) -> Result<Json<PredictionResponse>, (StatusCode, Json<ErrorResponse>)> {
    let result = cvx_query::engine::execute_query(
        &state.index,
        cvx_query::types::TemporalQuery::Prediction {
            entity_id,
            target_timestamp: params.target_timestamp,
        },
    )
    .map_err(query_err)?;

    if let cvx_query::types::QueryResult::Prediction(pred) = result {
        Ok(Json(PredictionResponse {
            entity_id,
            vector: pred.vector,
            timestamp: pred.timestamp,
            method: format!("{:?}", pred.method),
        }))
    } else {
        unreachable!()
    }
}

/// Temporal analogy: B@t3 + (A@t2 - A@t1).
#[utoipa::path(
    post,
    path = "/v1/analogy",
    request_body = AnalogyRequest,
    responses(
        (status = 200, description = "Analogy result vector", body = AnalogyResponse),
        (status = 404, description = "Entity not found", body = ErrorResponse),
    ),
    tag = "query"
)]
pub async fn analogy(
    State(state): State<SharedState>,
    Json(req): Json<AnalogyRequest>,
) -> Result<Json<AnalogyResponse>, (StatusCode, Json<ErrorResponse>)> {
    let result = cvx_query::engine::execute_query(
        &state.index,
        cvx_query::types::TemporalQuery::Analogy {
            entity_a: req.entity_a,
            t1: req.t1,
            t2: req.t2,
            entity_b: req.entity_b,
            t3: req.t3,
        },
    )
    .map_err(query_err)?;

    if let cvx_query::types::QueryResult::Analogy(vec) = result {
        Ok(Json(AnalogyResponse { vector: vec }))
    } else {
        unreachable!()
    }
}

/// Counterfactual trajectory analysis.
#[utoipa::path(
    get,
    path = "/v1/entities/{id}/counterfactual",
    params(
        ("id" = u64, Path, description = "Entity identifier"),
        ("change_point" = i64, Query, description = "Change point timestamp"),
    ),
    responses(
        (status = 200, description = "Counterfactual analysis", body = CounterfactualResponse),
        (status = 400, description = "Insufficient data", body = ErrorResponse),
    ),
    tag = "analytics"
)]
pub async fn counterfactual(
    State(state): State<SharedState>,
    Path(entity_id): Path<u64>,
    axum::extract::Query(params): axum::extract::Query<CounterfactualParams>,
) -> Result<Json<CounterfactualResponse>, (StatusCode, Json<ErrorResponse>)> {
    let result = cvx_query::engine::execute_query(
        &state.index,
        cvx_query::types::TemporalQuery::Counterfactual {
            entity_id,
            change_point: params.change_point,
        },
    )
    .map_err(query_err)?;

    if let cvx_query::types::QueryResult::Counterfactual(cf) = result {
        Ok(Json(CounterfactualResponse {
            entity_id,
            change_point: cf.change_point,
            total_divergence: cf.total_divergence,
            max_divergence_time: cf.max_divergence_time,
            max_divergence_value: cf.max_divergence_value,
            divergence_curve: cf
                .divergence_curve
                .into_iter()
                .map(|(t, d)| DivergenceEntry {
                    timestamp: t,
                    distance: d,
                })
                .collect(),
            method: cf.method,
        }))
    } else {
        unreachable!()
    }
}

/// Granger causality test between two entities.
#[utoipa::path(
    post,
    path = "/v1/granger",
    request_body = GrangerRequest,
    responses(
        (status = 200, description = "Granger causality result", body = GrangerResponse),
        (status = 400, description = "Insufficient data", body = ErrorResponse),
        (status = 404, description = "Entity not found", body = ErrorResponse),
    ),
    tag = "analytics"
)]
pub async fn granger(
    State(state): State<SharedState>,
    Json(req): Json<GrangerRequest>,
) -> Result<Json<GrangerResponse>, (StatusCode, Json<ErrorResponse>)> {
    let result = cvx_query::engine::execute_query(
        &state.index,
        cvx_query::types::TemporalQuery::GrangerCausality {
            entity_a: req.entity_a,
            entity_b: req.entity_b,
            max_lag: req.max_lag,
            significance: req.significance,
        },
    )
    .map_err(query_err)?;

    if let cvx_query::types::QueryResult::Granger(g) = result {
        Ok(Json(GrangerResponse {
            direction: g.direction,
            optimal_lag: g.optimal_lag,
            f_statistic: g.f_statistic,
            p_value: g.p_value,
            effect_size: g.effect_size,
        }))
    } else {
        unreachable!()
    }
}

/// Discover recurring motifs in an entity's trajectory.
#[utoipa::path(
    get,
    path = "/v1/entities/{id}/motifs",
    params(
        ("id" = u64, Path, description = "Entity identifier"),
        ("window" = usize, Query, description = "Subsequence window size (time steps)"),
        ("max_motifs" = Option<usize>, Query, description = "Maximum motifs to return (default 5)"),
    ),
    responses(
        (status = 200, description = "Discovered motifs", body = MotifsResponse),
        (status = 400, description = "Insufficient data", body = ErrorResponse),
    ),
    tag = "analytics"
)]
pub async fn motifs(
    State(state): State<SharedState>,
    Path(entity_id): Path<u64>,
    axum::extract::Query(params): axum::extract::Query<MotifParams>,
) -> Result<Json<MotifsResponse>, (StatusCode, Json<ErrorResponse>)> {
    let result = cvx_query::engine::execute_query(
        &state.index,
        cvx_query::types::TemporalQuery::DiscoverMotifs {
            entity_id,
            window: params.window,
            max_motifs: params.max_motifs,
        },
    )
    .map_err(query_err)?;

    if let cvx_query::types::QueryResult::Motifs(found) = result {
        Ok(Json(MotifsResponse {
            entity_id,
            motifs: found
                .into_iter()
                .map(|m| MotifEntry {
                    canonical_index: m.canonical_index,
                    occurrences: m
                        .occurrences
                        .into_iter()
                        .map(|o| MotifOccurrenceEntry {
                            start_index: o.start_index,
                            timestamp: o.timestamp,
                            distance: o.distance,
                        })
                        .collect(),
                    period: m.period,
                    mean_match_distance: m.mean_match_distance,
                })
                .collect(),
        }))
    } else {
        unreachable!()
    }
}

/// Discover anomalous subsequences (discords) in an entity's trajectory.
#[utoipa::path(
    get,
    path = "/v1/entities/{id}/discords",
    params(
        ("id" = u64, Path, description = "Entity identifier"),
        ("window" = usize, Query, description = "Subsequence window size (time steps)"),
        ("max_discords" = Option<usize>, Query, description = "Maximum discords to return (default 5)"),
    ),
    responses(
        (status = 200, description = "Discovered discords", body = DiscordsResponse),
        (status = 400, description = "Insufficient data", body = ErrorResponse),
    ),
    tag = "analytics"
)]
pub async fn discords(
    State(state): State<SharedState>,
    Path(entity_id): Path<u64>,
    axum::extract::Query(params): axum::extract::Query<DiscordParams>,
) -> Result<Json<DiscordsResponse>, (StatusCode, Json<ErrorResponse>)> {
    let result = cvx_query::engine::execute_query(
        &state.index,
        cvx_query::types::TemporalQuery::DiscoverDiscords {
            entity_id,
            window: params.window,
            max_discords: params.max_discords,
        },
    )
    .map_err(query_err)?;

    if let cvx_query::types::QueryResult::Discords(found) = result {
        Ok(Json(DiscordsResponse {
            entity_id,
            discords: found
                .into_iter()
                .map(|d| DiscordEntry {
                    start_index: d.start_index,
                    timestamp: d.timestamp,
                    nn_distance: d.nn_distance,
                })
                .collect(),
        }))
    } else {
        unreachable!()
    }
}

/// Temporal join: find convergence windows between two entities.
#[utoipa::path(
    post,
    path = "/v1/temporal-join",
    request_body = TemporalJoinRequest,
    responses(
        (status = 200, description = "Convergence windows", body = TemporalJoinResponse),
        (status = 404, description = "Entity not found", body = ErrorResponse),
    ),
    tag = "analytics"
)]
pub async fn temporal_join(
    State(state): State<SharedState>,
    Json(req): Json<TemporalJoinRequest>,
) -> Result<Json<TemporalJoinResponse>, (StatusCode, Json<ErrorResponse>)> {
    let window_us = (req.window_days * 86_400.0 * 1_000_000.0) as i64;
    let result = cvx_query::engine::execute_query(
        &state.index,
        cvx_query::types::TemporalQuery::TemporalJoin {
            entity_a: req.entity_a,
            entity_b: req.entity_b,
            epsilon: req.epsilon,
            window_us,
        },
    )
    .map_err(query_err)?;

    if let cvx_query::types::QueryResult::TemporalJoin(joins) = result {
        Ok(Json(TemporalJoinResponse {
            entity_a: req.entity_a,
            entity_b: req.entity_b,
            windows: joins
                .into_iter()
                .map(|j| TemporalJoinEntry {
                    start: j.start,
                    end: j.end,
                    mean_distance: j.mean_distance,
                    min_distance: j.min_distance,
                    points_a: j.points_a,
                    points_b: j.points_b,
                })
                .collect(),
        }))
    } else {
        unreachable!()
    }
}

/// Cohort drift analysis across multiple entities.
#[utoipa::path(
    post,
    path = "/v1/cohort/drift",
    request_body = CohortDriftRequest,
    responses(
        (status = 200, description = "Cohort drift report", body = CohortDriftResponse),
        (status = 400, description = "Insufficient data", body = ErrorResponse),
    ),
    tag = "analytics"
)]
pub async fn cohort_drift(
    State(state): State<SharedState>,
    Json(req): Json<CohortDriftRequest>,
) -> Result<Json<CohortDriftResponse>, (StatusCode, Json<ErrorResponse>)> {
    let result = cvx_query::engine::execute_query(
        &state.index,
        cvx_query::types::TemporalQuery::CohortDrift {
            entity_ids: req.entity_ids,
            t1: req.t1,
            t2: req.t2,
            top_n: req.top_n,
        },
    )
    .map_err(query_err)?;

    if let cvx_query::types::QueryResult::CohortDrift(report) = result {
        Ok(Json(CohortDriftResponse {
            n_entities: report.n_entities,
            mean_drift_l2: report.mean_drift_l2,
            median_drift_l2: report.median_drift_l2,
            std_drift_l2: report.std_drift_l2,
            centroid_l2_magnitude: report.centroid_l2_magnitude,
            centroid_cosine_drift: report.centroid_cosine_drift,
            dispersion_t1: report.dispersion_t1,
            dispersion_t2: report.dispersion_t2,
            dispersion_change: report.dispersion_change,
            convergence_score: report.convergence_score,
            top_dimensions: report
                .top_dimensions
                .into_iter()
                .map(|(index, change)| DimensionChange { index, change })
                .collect(),
            outliers: report
                .outliers
                .into_iter()
                .map(|o| CohortOutlierEntry {
                    entity_id: o.entity_id,
                    drift_magnitude: o.drift_magnitude,
                    z_score: o.z_score,
                    drift_direction_alignment: o.drift_direction_alignment,
                })
                .collect(),
        }))
    } else {
        unreachable!()
    }
}

/// Health check with server info.
#[utoipa::path(
    get,
    path = "/v1/health",
    responses(
        (status = 200, description = "Server health", body = HealthResponse),
    ),
    tag = "system"
)]
pub async fn health(State(state): State<SharedState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        uptime_secs: state.started_at.elapsed().as_secs(),
        index_size: state.index.len(),
    })
}

/// Readiness probe.
#[utoipa::path(
    get,
    path = "/v1/ready",
    responses(
        (status = 200, description = "Server is ready"),
        (status = 503, description = "Server is not ready"),
    ),
    tag = "system"
)]
pub async fn ready(State(state): State<SharedState>) -> Result<StatusCode, StatusCode> {
    if state.ready.load(std::sync::atomic::Ordering::Relaxed) {
        Ok(StatusCode::OK)
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}

// ─── Helper ────────────────────────────────────────────────────

fn query_err(e: cvx_core::error::QueryError) -> (StatusCode, Json<ErrorResponse>) {
    let status = match &e {
        cvx_core::error::QueryError::EntityNotFound(_) => StatusCode::NOT_FOUND,
        cvx_core::error::QueryError::InsufficientData { .. } => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (status, Json(ErrorResponse { error: e.to_string() }))
}
