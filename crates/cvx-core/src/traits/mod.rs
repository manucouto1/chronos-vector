//! Core trait definitions for ChronosVector subsystems.
//!
//! These traits define the contracts between subsystems. Each crate implements
//! the relevant traits, enabling loose coupling and testability via mock implementations.

use crate::error::{AnalyticsError, IndexError, QueryError, StorageError};
use crate::types::{ChangePoint, CpdMethod, ScoredResult, TemporalFilter, TemporalPoint};

/// Operations on a vector space.
///
/// Defines the algebraic structure that embedding vectors inhabit.
/// Implementations are not required for Layer 0 — only signatures.
pub trait VectorSpace: Clone + Send + Sync {
    /// Dimensionality of vectors in this space.
    fn dim(&self) -> usize;

    /// The zero vector.
    fn zero(dim: usize) -> Self;

    /// Component-wise addition.
    fn add(&self, other: &Self) -> Self;

    /// Scalar multiplication.
    fn scale(&self, factor: f32) -> Self;

    /// View as a float slice.
    fn as_slice(&self) -> &[f32];
}

/// A distance metric over vectors.
///
/// Implementations must satisfy metric properties:
/// - Non-negativity: $d(a, b) \geq 0$
/// - Identity: $d(a, a) = 0$
/// - Symmetry: $d(a, b) = d(b, a)$
///
/// Triangle inequality is desired but not required (cosine distance violates it).
pub trait DistanceMetric: Send + Sync {
    /// Compute the distance between two vectors.
    ///
    /// # Panics
    ///
    /// Implementations should panic if `a.len() != b.len()`.
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;

    /// Human-readable name of this metric (e.g., `"cosine"`, `"l2"`).
    fn name(&self) -> &str;
}

/// Persistent storage backend for temporal points.
///
/// Abstracts over the underlying storage engine (in-memory, RocksDB, etc.).
pub trait StorageBackend: Send + Sync {
    /// Retrieve a single point by entity, space, and timestamp.
    fn get(
        &self,
        entity_id: u64,
        space_id: u32,
        timestamp: i64,
    ) -> Result<Option<TemporalPoint>, StorageError>;

    /// Store a temporal point.
    fn put(&self, space_id: u32, point: &TemporalPoint) -> Result<(), StorageError>;

    /// Retrieve all points for an entity in a time range, ordered by timestamp.
    fn range(
        &self,
        entity_id: u64,
        space_id: u32,
        start: i64,
        end: i64,
    ) -> Result<Vec<TemporalPoint>, StorageError>;

    /// Delete a specific point.
    fn delete(&self, entity_id: u64, space_id: u32, timestamp: i64) -> Result<(), StorageError>;
}

/// Index backend for approximate nearest neighbor search.
///
/// Abstracts over the indexing structure (HNSW, brute-force, etc.).
pub trait IndexBackend: Send + Sync {
    /// Insert a point into the index.
    fn insert(&self, point_id: u64, vector: &[f32], timestamp: i64) -> Result<(), IndexError>;

    /// Search for the k nearest neighbors with temporal filtering.
    ///
    /// `alpha` controls the semantic vs temporal weight:
    /// - `alpha = 1.0`: pure semantic distance
    /// - `alpha = 0.0`: pure temporal distance
    fn search(
        &self,
        query: &[f32],
        k: u32,
        filter: TemporalFilter,
        alpha: f32,
    ) -> Result<Vec<ScoredResult>, QueryError>;

    /// Remove a point from the index.
    fn remove(&self, point_id: u64) -> Result<(), IndexError>;
}

/// Analytics backend for temporal analysis operations.
///
/// Provides prediction, change point detection, and differential calculus.
pub trait AnalyticsBackend: Send + Sync {
    /// Predict a future vector state using the learned trajectory model.
    fn predict(
        &self,
        trajectory: &[TemporalPoint],
        target_timestamp: i64,
    ) -> Result<TemporalPoint, AnalyticsError>;

    /// Detect change points in a trajectory.
    fn detect_changepoints(
        &self,
        trajectory: &[TemporalPoint],
        method: CpdMethod,
    ) -> Result<Vec<ChangePoint>, AnalyticsError>;

    /// Compute the velocity vector at a given timestamp.
    fn velocity(
        &self,
        trajectory: &[TemporalPoint],
        timestamp: i64,
    ) -> Result<Vec<f32>, AnalyticsError>;
}
