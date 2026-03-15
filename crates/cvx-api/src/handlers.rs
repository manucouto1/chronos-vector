//! REST endpoint handlers for ChronosVector.
//!
//! All handlers extract `State<SharedState>` and return JSON responses.

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use cvx_core::{StorageBackend, TemporalFilter, TemporalPoint};
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
    /// Number of stored points.
    pub store_size: usize,
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

/// GET /v1/health — Health check with server info.
pub async fn health(State(state): State<SharedState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        uptime_secs: state.started_at.elapsed().as_secs(),
        index_size: state.index.len(),
        store_size: state.store.len(),
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
