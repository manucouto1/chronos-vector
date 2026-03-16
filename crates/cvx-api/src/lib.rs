//! `cvx-api` — Dual-protocol API gateway for ChronosVector.
//!
//! ## REST Endpoints (axum)
//!
//! | Method | Path | Description |
//! |--------|------|-------------|
//! | POST | `/v1/ingest` | Batch ingest temporal points |
//! | POST | `/v1/query` | Spatiotemporal kNN search |
//! | GET | `/v1/entities/{id}/trajectory` | Entity trajectory retrieval |
//! | GET | `/v1/entities/{id}/velocity` | Velocity at timestamp |
//! | GET | `/v1/entities/{id}/drift` | Drift quantification |
//! | GET | `/v1/entities/{id}/changepoints` | Change point detection |
//! | GET | `/v1/health` | Health check with server info |
//! | GET | `/v1/ready` | Readiness probe |
//!
//! ## gRPC (tonic)
//!
//! - `IngestStream` — client-streaming ingestion
//! - `Query` — unary search
//! - `WatchDrift` — server-streaming drift events

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod grpc;
pub mod handlers;
pub mod router;
pub mod state;
