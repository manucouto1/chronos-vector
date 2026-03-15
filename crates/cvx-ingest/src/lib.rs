//! `cvx-ingest` — Data ingestion pipeline for ChronosVector.
//!
//! Orchestrates the full ingestion flow:
//! - **pipeline**: Ingestion orchestrator coordinating validation, delta encoding, indexing, and storage
//! - **delta**: Delta encoder/decoder with configurable threshold and keyframe intervals
//! - **validate**: Schema validation for incoming temporal vector data
