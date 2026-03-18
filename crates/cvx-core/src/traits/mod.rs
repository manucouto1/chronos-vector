//! Core trait definitions for ChronosVector subsystems.
//!
//! These traits define the contracts between subsystems. Each crate implements
//! the relevant traits, enabling loose coupling and testability via mock implementations.

pub mod quantizer;

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

/// Low-level temporal index access for query engine orchestration.
///
/// Provides the methods that `QueryEngine` needs from a temporal index.
/// Implemented by both `TemporalHnsw` (single-threaded) and
/// `ConcurrentTemporalHnsw` (thread-safe).
pub trait TemporalIndexAccess: Send + Sync {
    /// Search with temporal filtering, returning (node_id, score) pairs.
    fn search_raw(
        &self,
        query: &[f32],
        k: usize,
        filter: TemporalFilter,
        alpha: f32,
        query_timestamp: i64,
    ) -> Vec<(u32, f32)>;

    /// Retrieve trajectory for an entity: (timestamp, node_id) pairs.
    fn trajectory(&self, entity_id: u64, filter: TemporalFilter) -> Vec<(i64, u32)>;

    /// Get the vector for a node. Returns owned vec for thread safety.
    fn vector(&self, node_id: u32) -> Vec<f32>;

    /// Get the entity_id for a node.
    fn entity_id(&self, node_id: u32) -> u64;

    /// Get the timestamp for a node.
    fn timestamp(&self, node_id: u32) -> i64;

    /// Number of points in the index.
    fn len(&self) -> usize;

    /// Whether the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get semantic regions at a given HNSW level (RFC-004).
    /// Returns `(hub_node_id, hub_vector, n_assigned)` per region.
    fn regions(&self, _level: usize) -> Vec<(u32, Vec<f32>, usize)> {
        Vec::new()
    }

    /// Get points belonging to a specific region, optionally time-filtered (RFC-005).
    /// Returns `(node_id, entity_id, timestamp)` per member.
    fn region_members(
        &self,
        _region_hub: u32,
        _level: usize,
        _filter: TemporalFilter,
    ) -> Vec<(u32, u64, i64)> {
        Vec::new()
    }

    /// Smoothed region-distribution trajectory for an entity (RFC-004).
    fn region_trajectory(
        &self,
        _entity_id: u64,
        _level: usize,
        _window_days: i64,
        _alpha: f32,
    ) -> Vec<(i64, Vec<f32>)> {
        Vec::new()
    }
}

/// Index backend for approximate nearest neighbor search.
///
/// Abstracts over the indexing structure (HNSW, brute-force, etc.).
pub trait IndexBackend: Send + Sync {
    /// Insert a point into the index.
    fn insert(&self, entity_id: u64, vector: &[f32], timestamp: i64) -> Result<u32, IndexError>;

    /// Search for the k nearest neighbors with temporal filtering.
    ///
    /// `alpha` controls the semantic vs temporal weight:
    /// - `alpha = 1.0`: pure semantic distance
    /// - `alpha = 0.0`: pure temporal distance
    ///
    /// `query_timestamp` is the reference time for temporal distance computation.
    fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: TemporalFilter,
        alpha: f32,
        query_timestamp: i64,
    ) -> Result<Vec<ScoredResult>, QueryError>;

    /// Remove a point from the index.
    fn remove(&self, point_id: u64) -> Result<(), IndexError>;

    /// Number of points in the index.
    fn len(&self) -> usize;

    /// Whether the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
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

// ─── Embedder trait (RFC-009) ───────────────────────────────────────

/// Error type for embedding operations.
#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    /// Model not loaded or unavailable.
    #[error("model not available: {0}")]
    ModelNotAvailable(String),
    /// Input text is empty or invalid.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// Backend-specific error.
    #[error("embedding error: {0}")]
    BackendError(String),
}

/// Trait for converting text to embedding vectors.
///
/// Implementations may use local models (ONNX, TorchScript) or
/// remote APIs (OpenAI, Cohere).
pub trait Embedder: Send + Sync {
    /// Embed a single text string into a vector.
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError>;

    /// Embed multiple texts in a batch (more efficient for APIs).
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Output dimensionality of the embedding model.
    fn dimension(&self) -> usize;

    /// Name of the embedding model.
    fn model_name(&self) -> &str;
}
