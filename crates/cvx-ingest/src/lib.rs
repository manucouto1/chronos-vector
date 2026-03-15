//! # `cvx-ingest` — Data ingestion pipeline for ChronosVector.
//!
//! ## Layer 3: Delta Encoding
//!
//! [`delta::DeltaEncoder`] compresses embedding trajectories by storing only
//! the sparse differences between consecutive vectors. [`delta::DeltaDecoder`]
//! reconstructs full vectors from keyframes + delta chains.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod delta;
pub mod validation;
