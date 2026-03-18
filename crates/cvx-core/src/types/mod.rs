//! Core data types for ChronosVector.
//!
//! These types represent the fundamental data model — temporal points, deltas,
//! timelines, change points, and query results. All types are serializable
//! and designed to be domain-agnostic.

mod change_point;
mod delta_entry;
mod embedding_space;
mod entity_timeline;
mod scored_result;
mod temporal_filter;
mod temporal_point;
mod vector;

pub use change_point::ChangePoint;
pub use delta_entry::DeltaEntry;
pub use embedding_space::EmbeddingSpace;
pub use entity_timeline::EntityTimeline;
pub use scored_result::ScoredResult;
pub use temporal_filter::TemporalFilter;
pub use temporal_point::TemporalPoint;
pub use vector::DenseVector;

/// Method used for change point detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CpdMethod {
    /// Pruned Exact Linear Time — offline, exact, $O(N)$.
    Pelt,
    /// Bayesian Online Change Point Detection — online, streaming, $O(1)$ amortized.
    Bocpd,
}
