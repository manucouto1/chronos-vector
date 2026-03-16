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
    let traj_data = state.index.trajectory(entity_id, TemporalFilter::All);
    if traj_data.len() < 2 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: format!(
                    "entity {entity_id} has insufficient data ({} points, need 2)",
                    traj_data.len()
                ),
            }),
        ));
    }

    let vectors: Vec<Vec<f32>> = traj_data
        .iter()
        .map(|&(_, node_id)| state.index.vector(node_id))
        .collect();
    let traj: Vec<(i64, &[f32])> = traj_data
        .iter()
        .zip(vectors.iter())
        .map(|(&(ts, _), v)| (ts, v.as_slice()))
        .collect();

    let vel = cvx_analytics::calculus::velocity(&traj, params.timestamp).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    Ok(Json(VelocityResponse {
        entity_id,
        timestamp: params.timestamp,
        velocity: vel,
    }))
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
    let traj = state.index.trajectory(entity_id, TemporalFilter::All);
    if traj.is_empty() {
        return Err((
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("entity {entity_id} not found"),
            }),
        ));
    }

    // Find nearest points to t1 and t2
    let find_nearest = |target: i64| -> u32 {
        traj.iter()
            .min_by_key(|&&(ts, _)| (ts - target).unsigned_abs())
            .unwrap()
            .1
    };

    let node1 = find_nearest(params.t1);
    let node2 = find_nearest(params.t2);
    let v1 = state.index.vector(node1);
    let v2 = state.index.vector(node2);

    let report = cvx_analytics::calculus::drift_report(&v1, &v2, params.top_n);

    Ok(Json(DriftResponse {
        entity_id,
        l2_magnitude: report.l2_magnitude,
        cosine_drift: report.cosine_drift,
        top_dimensions: report
            .top_dimensions
            .into_iter()
            .map(|(index, change)| DimensionChange { index, change })
            .collect(),
    }))
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
) -> Json<ChangepointResponse> {
    let traj_data = state
        .index
        .trajectory(entity_id, TemporalFilter::Range(params.start, params.end));

    let changepoints = if traj_data.len() >= 4 {
        let vectors: Vec<Vec<f32>> = traj_data
            .iter()
            .map(|&(_, node_id)| state.index.vector(node_id))
            .collect();
        let traj: Vec<(i64, &[f32])> = traj_data
            .iter()
            .zip(vectors.iter())
            .map(|(&(ts, _), v)| (ts, v.as_slice()))
            .collect();

        cvx_analytics::pelt::detect(
            entity_id,
            &traj,
            &cvx_analytics::pelt::PeltConfig::default(),
        )
        .into_iter()
        .map(|cp| ChangepointEntry {
            timestamp: cp.timestamp(),
            severity: cp.severity(),
        })
        .collect()
    } else {
        Vec::new()
    };

    Json(ChangepointResponse {
        entity_id,
        changepoints,
    })
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
