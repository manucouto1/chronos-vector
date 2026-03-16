//! REST endpoint handlers for ChronosVector.
//!
//! All handlers extract `State<SharedState>` and return JSON responses.

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use cvx_core::{TemporalFilter, TemporalPoint};
use cvx_ingest::validation::validate_point;
use serde::{Deserialize, Serialize};

use crate::state::SharedState;

// ─── Request / Response types ───────────────────────────────────────

/// Ingest request: a single temporal point.
#[derive(Debug, Deserialize)]
pub struct IngestRequest {
    /// Entity identifier.
    pub entity_id: u64,
    /// Timestamp in microseconds.
    pub timestamp: i64,
    /// Embedding vector.
    pub vector: Vec<f32>,
}

/// Batch ingest request.
#[derive(Debug, Deserialize)]
pub struct BatchIngestRequest {
    /// List of points to ingest.
    pub points: Vec<IngestRequest>,
}

/// Ingest receipt for a single point.
#[derive(Debug, Serialize)]
pub struct IngestReceipt {
    /// Internal node ID assigned to this point.
    pub node_id: u32,
    /// Entity identifier.
    pub entity_id: u64,
    /// Timestamp.
    pub timestamp: i64,
}

/// Batch ingest response.
#[derive(Debug, Serialize)]
pub struct BatchIngestResponse {
    /// Number of points successfully ingested.
    pub ingested: usize,
    /// Receipts for each ingested point.
    pub receipts: Vec<IngestReceipt>,
}

/// Query request.
#[derive(Debug, Deserialize)]
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
#[derive(Debug, Deserialize, Default)]
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
#[derive(Debug, Serialize)]
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
#[derive(Debug, Serialize)]
pub struct QueryResponse {
    /// Search results ordered by score.
    pub results: Vec<QueryResult>,
}

/// Trajectory entry.
#[derive(Debug, Serialize)]
pub struct TrajectoryEntry {
    /// Timestamp.
    pub timestamp: i64,
    /// Node ID.
    pub node_id: u32,
}

/// Trajectory response.
#[derive(Debug, Serialize)]
pub struct TrajectoryResponse {
    /// Entity identifier.
    pub entity_id: u64,
    /// Trajectory points ordered by timestamp.
    pub points: Vec<TrajectoryEntry>,
}

/// Health response.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    /// Server status.
    pub status: &'static str,
    /// Server version.
    pub version: &'static str,
    /// Uptime in seconds.
    pub uptime_secs: u64,
    /// Number of indexed vectors.
    pub index_size: usize,
}

/// Error response body.
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    /// Error message.
    pub error: String,
}

// ─── Handlers ───────────────────────────────────────────────────────

/// POST /v1/ingest — Batch ingest temporal points.
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

        // Validate
        if let Err(e) = validate_point(&point, &state.validation) {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: e.to_string(),
                }),
            ));
        }

        // Insert into index
        let node_id = state.index.insert(p.entity_id, p.timestamp, &p.vector);

        // Store the full point
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

/// POST /v1/query — Spatiotemporal kNN search.
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

/// GET /v1/entities/:id/trajectory — Retrieve entity trajectory.
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

// ─── Analytics endpoints ────────────────────────────────────────

/// Velocity request params.
#[derive(Debug, Deserialize)]
pub struct VelocityParams {
    /// Timestamp to compute velocity at.
    pub timestamp: i64,
}

/// Velocity response.
#[derive(Debug, Serialize)]
pub struct VelocityResponse {
    /// Entity identifier.
    pub entity_id: u64,
    /// Timestamp.
    pub timestamp: i64,
    /// Velocity vector per dimension.
    pub velocity: Vec<f32>,
}

/// GET /v1/entities/:id/velocity?timestamp=T — Compute velocity at timestamp.
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

/// Drift request params.
#[derive(Debug, Deserialize)]
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
#[derive(Debug, Serialize)]
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
#[derive(Debug, Serialize)]
pub struct DimensionChange {
    /// Dimension index.
    pub index: usize,
    /// Absolute change.
    pub change: f32,
}

/// GET /v1/entities/:id/drift?t1=T1&t2=T2 — Drift quantification.
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

/// Changepoint request params.
#[derive(Debug, Deserialize)]
pub struct ChangepointParams {
    /// Start timestamp.
    pub start: i64,
    /// End timestamp.
    pub end: i64,
}

/// Changepoint response.
#[derive(Debug, Serialize)]
pub struct ChangepointResponse {
    /// Entity identifier.
    pub entity_id: u64,
    /// Detected change points.
    pub changepoints: Vec<ChangepointEntry>,
}

/// A detected change point.
#[derive(Debug, Serialize)]
pub struct ChangepointEntry {
    /// Timestamp of the change.
    pub timestamp: i64,
    /// Severity [0, 1].
    pub severity: f64,
}

/// GET /v1/entities/:id/changepoints?start=S&end=E — Detect change points.
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

// ─── Prediction & Analogy endpoints ────────────────────────────

/// Prediction request params.
#[derive(Debug, Deserialize)]
pub struct PredictionParams {
    /// Target timestamp for prediction.
    pub target_timestamp: i64,
}

/// Prediction response.
#[derive(Debug, Serialize)]
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

/// GET /v1/entities/:id/prediction?target_timestamp=T — Predict future vector.
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

/// Analogy request.
#[derive(Debug, Deserialize)]
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
#[derive(Debug, Serialize)]
pub struct AnalogyResponse {
    /// Resulting vector: B@t3 + (A@t2 - A@t1).
    pub vector: Vec<f32>,
}

/// POST /v1/analogy — Temporal analogy query.
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

// ─── Helper ────────────────────────────────────────────────────

/// Convert QueryError to HTTP error response.
fn query_err(e: cvx_core::error::QueryError) -> (StatusCode, Json<ErrorResponse>) {
    let status = match &e {
        cvx_core::error::QueryError::EntityNotFound(_) => StatusCode::NOT_FOUND,
        cvx_core::error::QueryError::InsufficientData { .. } => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (status, Json(ErrorResponse { error: e.to_string() }))
}

/// GET /v1/health — Health check with server info.
pub async fn health(State(state): State<SharedState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        uptime_secs: state.started_at.elapsed().as_secs(),
        index_size: state.index.len(),
    })
}

/// GET /v1/ready — Readiness probe.
pub async fn ready(State(state): State<SharedState>) -> Result<StatusCode, StatusCode> {
    if state.ready.load(std::sync::atomic::Ordering::Relaxed) {
        Ok(StatusCode::OK)
    } else {
        Err(StatusCode::SERVICE_UNAVAILABLE)
    }
}
