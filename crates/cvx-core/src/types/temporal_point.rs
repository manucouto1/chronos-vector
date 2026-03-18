//! A vector observation at a specific point in time.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A single vector observation for an entity at a specific timestamp.
///
/// This is the fundamental unit of data in ChronosVector. Every ingested
/// embedding becomes a `TemporalPoint`.
///
/// # Fields
///
/// - `entity_id` — Unique identifier for the entity this vector belongs to.
/// - `timestamp` — Microseconds since Unix epoch. Negative values represent pre-1970 dates.
/// - `vector` — The embedding vector (e.g., 768-dimensional BERT output).
/// - `metadata` — Optional key-value pairs (source, model version, tags, etc.).
///
/// # Example
///
/// ```
/// use cvx_core::TemporalPoint;
///
/// let point = TemporalPoint::new(42, 1_700_000_000, vec![0.1, 0.2, 0.3]);
/// assert_eq!(point.entity_id(), 42);
/// assert_eq!(point.dim(), 3);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalPoint {
    entity_id: u64,
    timestamp: i64,
    vector: Vec<f32>,
    #[serde(default)]
    metadata: HashMap<String, String>,
}

impl TemporalPoint {
    /// Create a new temporal point without metadata.
    pub fn new(entity_id: u64, timestamp: i64, vector: Vec<f32>) -> Self {
        Self {
            entity_id,
            timestamp,
            vector,
            metadata: HashMap::new(),
        }
    }

    /// Create a new temporal point with metadata.
    pub fn with_metadata(
        entity_id: u64,
        timestamp: i64,
        vector: Vec<f32>,
        metadata: HashMap<String, String>,
    ) -> Self {
        Self {
            entity_id,
            timestamp,
            vector,
            metadata,
        }
    }

    /// The entity this point belongs to.
    pub fn entity_id(&self) -> u64 {
        self.entity_id
    }

    /// Timestamp in microseconds since Unix epoch.
    pub fn timestamp(&self) -> i64 {
        self.timestamp
    }

    /// The embedding vector.
    pub fn vector(&self) -> &[f32] {
        &self.vector
    }

    /// Dimensionality of the vector.
    pub fn dim(&self) -> usize {
        self.vector.len()
    }

    /// Optional metadata.
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_point_without_metadata() {
        let p = TemporalPoint::new(1, 1000, vec![0.1, 0.2]);
        assert_eq!(p.entity_id(), 1);
        assert_eq!(p.timestamp(), 1000);
        assert_eq!(p.vector(), &[0.1, 0.2]);
        assert_eq!(p.dim(), 2);
        assert!(p.metadata().is_empty());
    }

    #[test]
    fn with_metadata_stores_kv_pairs() {
        let mut meta = HashMap::new();
        meta.insert("model".into(), "bert-v2".into());
        let p = TemporalPoint::with_metadata(1, 1000, vec![0.1], meta);
        assert_eq!(p.metadata().get("model").unwrap(), "bert-v2");
    }

    #[test]
    fn serde_json_roundtrip() {
        let p = TemporalPoint::new(42, -5000, vec![1.0, 2.0, 3.0]);
        let json = serde_json::to_string(&p).unwrap();
        let recovered: TemporalPoint = serde_json::from_str(&json).unwrap();
        assert_eq!(p, recovered);
    }

    #[test]
    fn postcard_roundtrip() {
        let p = TemporalPoint::new(42, 1_700_000_000, vec![0.5; 768]);
        let bytes = postcard::to_allocvec(&p).unwrap();
        let recovered: TemporalPoint = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(p, recovered);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn postcard_roundtrip_arbitrary(
            entity_id in any::<u64>(),
            timestamp in any::<i64>(),
            vector in prop::collection::vec(-1e6f32..1e6, 1..1024),
        ) {
            let p = TemporalPoint::new(entity_id, timestamp, vector);
            let bytes = postcard::to_allocvec(&p).unwrap();
            let recovered: TemporalPoint = postcard::from_bytes(&bytes).unwrap();
            prop_assert_eq!(p, recovered);
        }

        #[test]
        fn dim_matches_vector_length(
            vector in prop::collection::vec(-1.0f32..1.0, 0..2048),
        ) {
            let p = TemporalPoint::new(0, 0, vector.clone());
            prop_assert_eq!(p.dim(), vector.len());
        }
    }
}
