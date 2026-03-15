//! `cvx-api` — Dual-protocol API gateway for ChronosVector.
//!
//! Exposes ChronosVector functionality over REST and gRPC:
//! - **rest**: Axum HTTP handlers (ingest, query, entity lookup, health, admin)
//! - **grpc**: Tonic gRPC service (IngestStream, QueryStream, WatchDrift)
//! - **proto**: Generated protobuf types for wire format
