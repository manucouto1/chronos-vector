//! gRPC service implementation for ChronosVector.

use std::pin::Pin;
use std::sync::Arc;

use cvx_core::{TemporalFilter, TemporalPoint};
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::{Stream, StreamExt};
use tonic::{Request, Response, Status};

use super::proto::*;
use crate::state::AppState;

/// gRPC service for ChronosVector.
pub struct CvxGrpcService {
    state: Arc<AppState>,
    drift_tx: broadcast::Sender<DriftEvent>,
}

impl CvxGrpcService {
    /// Create a new gRPC service.
    pub fn new(state: Arc<AppState>, drift_tx: broadcast::Sender<DriftEvent>) -> Self {
        Self { state, drift_tx }
    }

    /// Get the drift event sender (for the ingest pipeline to publish events).
    pub fn drift_sender(&self) -> broadcast::Sender<DriftEvent> {
        self.drift_tx.clone()
    }
}

/// The gRPC service trait (manual, not codegen).
#[tonic::async_trait]
pub trait CvxService: Send + Sync + 'static {
    /// Ingest a stream of points.
    async fn ingest_stream(
        &self,
        request: Request<tonic::Streaming<IngestPoint>>,
    ) -> Result<Response<IngestAck>, Status>;

    /// Execute a query.
    async fn query(
        &self,
        request: Request<QueryRequest>,
    ) -> Result<Response<QueryResponse>, Status>;

    /// Stream type for drift events.
    type WatchDriftStream: Stream<Item = Result<DriftEvent, Status>> + Send + 'static;
    /// Server-streaming: watch for drift events.
    async fn watch_drift(
        &self,
        request: Request<WatchDriftRequest>,
    ) -> Result<Response<Self::WatchDriftStream>, Status>;

    /// Health check.
    async fn health(&self, request: Request<Empty>) -> Result<Response<HealthResponse>, Status>;
}

#[tonic::async_trait]
impl CvxService for CvxGrpcService {
    async fn ingest_stream(
        &self,
        request: Request<tonic::Streaming<IngestPoint>>,
    ) -> Result<Response<IngestAck>, Status> {
        let mut stream = request.into_inner();
        let mut last_ack = IngestAck::default();
        let mut count = 0u64;

        while let Some(point) = stream.message().await? {
            let node_id = self
                .state
                .index
                .insert(point.entity_id, point.timestamp, &point.vector);

            let temporal_point =
                TemporalPoint::new(point.entity_id, point.timestamp, point.vector.clone());
            self.state
                .store
                .put(0, &temporal_point)
                .map_err(|e| Status::internal(e.to_string()))?;

            last_ack = IngestAck {
                node_id,
                entity_id: point.entity_id,
                timestamp: point.timestamp,
            };
            count += 1;
        }

        tracing::info!("Ingested {count} points via gRPC stream");
        Ok(Response::new(last_ack))
    }

    async fn query(
        &self,
        request: Request<QueryRequest>,
    ) -> Result<Response<QueryResponse>, Status> {
        let req = request.into_inner();

        let filter = if req.filter_start == 0 && req.filter_end == 0 {
            TemporalFilter::All
        } else {
            TemporalFilter::Range(req.filter_start, req.filter_end)
        };

        let raw_results = self.state.index.search(
            &req.vector,
            req.k as usize,
            filter,
            req.alpha,
            req.query_timestamp,
        );

        let results = raw_results
            .into_iter()
            .map(|(node_id, score)| SearchResult {
                entity_id: self.state.index.entity_id(node_id),
                timestamp: self.state.index.timestamp(node_id),
                score,
            })
            .collect();

        Ok(Response::new(QueryResponse { results }))
    }

    type WatchDriftStream =
        Pin<Box<dyn Stream<Item = Result<DriftEvent, Status>> + Send + 'static>>;

    async fn watch_drift(
        &self,
        request: Request<WatchDriftRequest>,
    ) -> Result<Response<Self::WatchDriftStream>, Status> {
        let params = request.into_inner();
        let rx = self.drift_tx.subscribe();
        let entity_filter = params.entity_id;
        let severity_filter = params.min_severity;

        let stream = BroadcastStream::new(rx).filter_map(move |result| match result {
            Ok(event) => {
                if entity_filter != 0 && event.entity_id != entity_filter {
                    return None;
                }
                if event.severity < severity_filter {
                    return None;
                }
                Some(Ok(event))
            }
            Err(_) => None,
        });

        Ok(Response::new(Box::pin(stream)))
    }

    async fn health(&self, _request: Request<Empty>) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse {
            status: "ok".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            uptime_secs: self.state.started_at.elapsed().as_secs(),
            index_size: self.state.index.len() as u64,
        }))
    }
}
