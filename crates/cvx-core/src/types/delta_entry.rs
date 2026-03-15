//! Compressed representation of a vector change between timestamps.

use serde::{Deserialize, Serialize};

/// A delta-encoded vector change for storage compression.
///
/// Instead of storing the full vector at every timestamp, CVX stores
/// keyframes (full vectors) at intervals and deltas (sparse changes) between them.
/// This is analogous to video compression with I-frames and P-frames.
///
/// # Sparse Representation
///
/// Only dimensions that changed beyond a threshold ε are stored,
/// using parallel `indices` and `values` arrays.
///
/// # Example
///
/// ```
/// use cvx_core::DeltaEntry;
///
/// // Only dimensions 2 and 5 changed
/// let delta = DeltaEntry::delta(42, 1000, 2000, vec![2, 5], vec![0.1, -0.3]);
/// assert!(!delta.is_keyframe());
/// assert_eq!(delta.nnz(), 2);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeltaEntry {
    entity_id: u64,
    base_timestamp: i64,
    delta_timestamp: i64,
    /// Indices of dimensions that changed (empty for keyframes).
    indices: Vec<u32>,
    /// Delta values for each changed dimension (empty for keyframes).
    values: Vec<f32>,
    /// If true, this is a keyframe (full vector stored separately).
    keyframe: bool,
}

impl DeltaEntry {
    /// Create a keyframe entry (references a full vector in storage).
    pub fn keyframe(entity_id: u64, timestamp: i64) -> Self {
        Self {
            entity_id,
            base_timestamp: timestamp,
            delta_timestamp: timestamp,
            indices: Vec::new(),
            values: Vec::new(),
            keyframe: true,
        }
    }

    /// Create a delta entry with sparse changes.
    ///
    /// # Panics
    ///
    /// Panics if `indices` and `values` have different lengths.
    pub fn delta(
        entity_id: u64,
        base_timestamp: i64,
        delta_timestamp: i64,
        indices: Vec<u32>,
        values: Vec<f32>,
    ) -> Self {
        assert_eq!(
            indices.len(),
            values.len(),
            "indices and values must have the same length"
        );
        Self {
            entity_id,
            base_timestamp,
            delta_timestamp,
            indices,
            values,
            keyframe: false,
        }
    }

    /// The entity this delta belongs to.
    pub fn entity_id(&self) -> u64 {
        self.entity_id
    }

    /// Timestamp of the base (keyframe or previous delta).
    pub fn base_timestamp(&self) -> i64 {
        self.base_timestamp
    }

    /// Timestamp of this delta.
    pub fn delta_timestamp(&self) -> i64 {
        self.delta_timestamp
    }

    /// Whether this is a keyframe (full vector reference).
    pub fn is_keyframe(&self) -> bool {
        self.keyframe
    }

    /// Indices of changed dimensions.
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Delta values for changed dimensions.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Number of non-zero (changed) dimensions.
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyframe_has_no_deltas() {
        let kf = DeltaEntry::keyframe(1, 1000);
        assert!(kf.is_keyframe());
        assert_eq!(kf.nnz(), 0);
        assert!(kf.indices().is_empty());
        assert!(kf.values().is_empty());
    }

    #[test]
    fn delta_stores_sparse_changes() {
        let d = DeltaEntry::delta(1, 1000, 2000, vec![0, 5, 10], vec![0.1, -0.2, 0.3]);
        assert!(!d.is_keyframe());
        assert_eq!(d.nnz(), 3);
        assert_eq!(d.base_timestamp(), 1000);
        assert_eq!(d.delta_timestamp(), 2000);
    }

    #[test]
    #[should_panic(expected = "indices and values must have the same length")]
    fn delta_panics_on_length_mismatch() {
        DeltaEntry::delta(1, 0, 1, vec![0, 1], vec![0.1]);
    }

    #[test]
    fn postcard_roundtrip() {
        let d = DeltaEntry::delta(42, 1000, 2000, vec![3, 7], vec![0.5, -0.5]);
        let bytes = postcard::to_allocvec(&d).unwrap();
        let recovered: DeltaEntry = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(d, recovered);
    }
}
