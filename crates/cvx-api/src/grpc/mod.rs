//! gRPC service for ChronosVector.
//!
//! Provides streaming ingestion (`IngestStream`), query, and drift watching
//! via tonic gRPC.

#[allow(missing_docs)]
pub mod proto;
pub mod service;

pub use service::CvxGrpcService;
