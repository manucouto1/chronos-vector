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
