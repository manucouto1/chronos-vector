//! Protobuf message types (manual prost definitions, no .proto codegen).

/// A temporal point for ingestion.
#[derive(Clone, prost::Message)]
pub struct IngestPoint {
    #[prost(uint64, tag = "1")]
    pub entity_id: u64,
    #[prost(int64, tag = "2")]
    pub timestamp: i64,
    #[prost(float, repeated, tag = "3")]
    pub vector: Vec<f32>,
}

/// Acknowledgement for a single ingested point.
#[derive(Clone, prost::Message)]
pub struct IngestAck {
    #[prost(uint32, tag = "1")]
    pub node_id: u32,
    #[prost(uint64, tag = "2")]
    pub entity_id: u64,
    #[prost(int64, tag = "3")]
    pub timestamp: i64,
}

/// Query request.
#[derive(Clone, prost::Message)]
pub struct QueryRequest {
    #[prost(float, repeated, tag = "1")]
    pub vector: Vec<f32>,
    #[prost(uint32, tag = "2")]
    pub k: u32,
    #[prost(float, tag = "3")]
    pub alpha: f32,
    #[prost(int64, tag = "4")]
    pub query_timestamp: i64,
    #[prost(int64, tag = "5")]
    pub filter_start: i64,
    #[prost(int64, tag = "6")]
    pub filter_end: i64,
}

/// A single search result.
#[derive(Clone, prost::Message)]
pub struct SearchResult {
    #[prost(uint64, tag = "1")]
    pub entity_id: u64,
    #[prost(int64, tag = "2")]
    pub timestamp: i64,
    #[prost(float, tag = "3")]
    pub score: f32,
}

/// Query response.
#[derive(Clone, prost::Message)]
pub struct QueryResponse {
    #[prost(message, repeated, tag = "1")]
    pub results: Vec<SearchResult>,
}

/// Drift event emitted by the WatchDrift stream.
#[derive(Clone, prost::Message)]
pub struct DriftEvent {
    #[prost(uint64, tag = "1")]
    pub entity_id: u64,
    #[prost(int64, tag = "2")]
    pub timestamp: i64,
    #[prost(float, tag = "3")]
    pub severity: f32,
    #[prost(float, repeated, tag = "4")]
    pub drift_vector: Vec<f32>,
}

/// WatchDrift subscription request.
#[derive(Clone, prost::Message)]
pub struct WatchDriftRequest {
    /// Entity to watch (0 = all entities).
    #[prost(uint64, tag = "1")]
    pub entity_id: u64,
    /// Minimum severity to report.
    #[prost(float, tag = "2")]
    pub min_severity: f32,
}

/// Empty message.
#[derive(Clone, prost::Message)]
pub struct Empty {}

/// Health response.
#[derive(Clone, prost::Message)]
pub struct HealthResponse {
    #[prost(string, tag = "1")]
    pub status: String,
    #[prost(string, tag = "2")]
    pub version: String,
    #[prost(uint64, tag = "3")]
    pub uptime_secs: u64,
    #[prost(uint64, tag = "4")]
    pub index_size: u64,
}
