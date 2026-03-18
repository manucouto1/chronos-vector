//! # `cvx-core` — Core types, traits, configuration, and error handling for ChronosVector.
//!
//! This crate defines the foundational abstractions used across all other CVX crates.
//! It has **no dependencies on other workspace crates** — all other crates depend on it.
//!
//! ## Modules
//!
//! - [`types`] — Core data types: [`TemporalPoint`], [`DeltaEntry`], [`EntityTimeline`],
//!   [`ChangePoint`], [`ScoredResult`]
//! - [`traits`] — Trait definitions: [`DistanceMetric`], [`VectorSpace`], [`StorageBackend`],
//!   [`IndexBackend`], [`AnalyticsBackend`]
//! - [`config`] — Configuration: [`CvxConfig`] (deserializable from TOML)
//! - [`error`] — Error types: [`CvxError`], [`CvxResult`]
//!
//! ## Design Principles
//!
//! - **Domain-agnostic**: CVX knows about temporal vector trajectories, not about
//!   finance, NLP, or medicine. Domain-specific concepts are compositions of these primitives.
//! - **Typed errors**: Every subsystem has its own error type that converts into [`CvxError`].
//! - **Serializable**: All core types implement `serde::Serialize` + `serde::Deserialize`.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod config;
pub mod error;
pub mod traits;
pub mod types;

// Re-export commonly used types at crate root for ergonomics.
pub use config::CvxConfig;
pub use error::{CvxError, CvxResult};
pub use traits::{
    AnalyticsBackend, DistanceMetric, EmbedError, Embedder, IndexBackend, StorageBackend,
    TemporalIndexAccess, VectorSpace,
};
pub use types::{
    ChangePoint, CpdMethod, DeltaEntry, DenseVector, EntityTimeline, ScoredResult, TemporalFilter,
    TemporalPoint,
};
