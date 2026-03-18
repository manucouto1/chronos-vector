//! In-memory storage backend.
//!
//! A non-persistent [`StorageBackend`] implementation using a `BTreeMap`
//! for ordered key access. Suitable for development, testing, and small datasets.
//!
//! Thread-safe via [`parking_lot::RwLock`].
//!
//! # Example
//!
//! ```
//! use cvx_core::{StorageBackend, TemporalPoint};
//! use cvx_storage::memory::InMemoryStore;
//!
//! let store = InMemoryStore::new();
//! let point = TemporalPoint::new(42, 1000, vec![0.1, 0.2, 0.3]);
//! store.put(0, &point).unwrap();
//!
//! let retrieved = store.get(42, 0, 1000).unwrap();
//! assert_eq!(retrieved.as_ref(), Some(&point));
//! ```

use std::collections::BTreeMap;

use cvx_core::StorageBackend;
use cvx_core::error::StorageError;
use cvx_core::types::TemporalPoint;
use parking_lot::RwLock;

/// Composite key for the in-memory store: `(entity_id, space_id, timestamp)`.
///
/// `BTreeMap` ordering on this tuple gives us:
/// - Prefix scan by `(entity_id,)` → all spaces and timestamps
/// - Prefix scan by `(entity_id, space_id)` → all timestamps in a space
/// - Range scan by `(entity_id, space_id, t1..=t2)` → time window
type StoreKey = (u64, u32, i64);

/// Non-persistent in-memory storage using a sorted `BTreeMap`.
///
/// Thread-safe: multiple readers or one writer via [`RwLock`].
/// All data is lost when the process exits.
pub struct InMemoryStore {
    data: RwLock<BTreeMap<StoreKey, TemporalPoint>>,
}

impl InMemoryStore {
    /// Create a new empty in-memory store.
    pub fn new() -> Self {
        Self {
            data: RwLock::new(BTreeMap::new()),
        }
    }

    /// Number of points currently stored.
    pub fn len(&self) -> usize {
        self.data.read().len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.data.read().is_empty()
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl StorageBackend for InMemoryStore {
    fn get(
        &self,
        entity_id: u64,
        space_id: u32,
        timestamp: i64,
    ) -> Result<Option<TemporalPoint>, StorageError> {
        let data = self.data.read();
        Ok(data.get(&(entity_id, space_id, timestamp)).cloned())
    }

    fn put(&self, space_id: u32, point: &TemporalPoint) -> Result<(), StorageError> {
        let key = (point.entity_id(), space_id, point.timestamp());
        let mut data = self.data.write();
        data.insert(key, point.clone());
        Ok(())
    }

    fn range(
        &self,
        entity_id: u64,
        space_id: u32,
        start: i64,
        end: i64,
    ) -> Result<Vec<TemporalPoint>, StorageError> {
        let data = self.data.read();
        let range_start = (entity_id, space_id, start);
        let range_end = (entity_id, space_id, end);
        let points: Vec<TemporalPoint> = data
            .range(range_start..=range_end)
            .map(|(_, v)| v.clone())
            .collect();
        Ok(points)
    }

    fn delete(&self, entity_id: u64, space_id: u32, timestamp: i64) -> Result<(), StorageError> {
        let mut data = self.data.write();
        data.remove(&(entity_id, space_id, timestamp));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_point(entity_id: u64, timestamp: i64) -> TemporalPoint {
        TemporalPoint::new(entity_id, timestamp, vec![0.1, 0.2, 0.3])
    }

    #[test]
    fn put_and_get() {
        let store = InMemoryStore::new();
        let p = sample_point(1, 1000);
        store.put(0, &p).unwrap();

        let result = store.get(1, 0, 1000).unwrap();
        assert_eq!(result, Some(p));
    }

    #[test]
    fn get_nonexistent_returns_none() {
        let store = InMemoryStore::new();
        assert_eq!(store.get(999, 0, 0).unwrap(), None);
    }

    #[test]
    fn put_overwrites() {
        let store = InMemoryStore::new();
        let p1 = TemporalPoint::new(1, 1000, vec![1.0]);
        let p2 = TemporalPoint::new(1, 1000, vec![2.0]);
        store.put(0, &p1).unwrap();
        store.put(0, &p2).unwrap();

        let result = store.get(1, 0, 1000).unwrap().unwrap();
        assert_eq!(result.vector(), &[2.0]);
    }

    #[test]
    fn delete_removes_point() {
        let store = InMemoryStore::new();
        store.put(0, &sample_point(1, 1000)).unwrap();
        assert_eq!(store.len(), 1);

        store.delete(1, 0, 1000).unwrap();
        assert_eq!(store.len(), 0);
        assert_eq!(store.get(1, 0, 1000).unwrap(), None);
    }

    #[test]
    fn delete_nonexistent_is_noop() {
        let store = InMemoryStore::new();
        store.delete(999, 0, 0).unwrap(); // should not panic
    }

    #[test]
    fn range_returns_ordered_subset() {
        let store = InMemoryStore::new();
        for ts in [100, 200, 300, 400, 500] {
            store.put(0, &sample_point(1, ts)).unwrap();
        }

        let results = store.range(1, 0, 200, 400).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].timestamp(), 200);
        assert_eq!(results[1].timestamp(), 300);
        assert_eq!(results[2].timestamp(), 400);
    }

    #[test]
    fn range_empty_window() {
        let store = InMemoryStore::new();
        store.put(0, &sample_point(1, 100)).unwrap();

        let results = store.range(1, 0, 200, 300).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn range_does_not_cross_entities() {
        let store = InMemoryStore::new();
        store.put(0, &sample_point(1, 100)).unwrap();
        store.put(0, &sample_point(2, 100)).unwrap();

        let results = store.range(1, 0, 0, i64::MAX).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entity_id(), 1);
    }

    #[test]
    fn range_does_not_cross_spaces() {
        let store = InMemoryStore::new();
        store.put(0, &sample_point(1, 100)).unwrap();
        store.put(1, &sample_point(1, 100)).unwrap();

        let results = store.range(1, 0, 0, i64::MAX).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn insert_100k_and_retrieve() {
        let store = InMemoryStore::new();
        for i in 0..100_000u64 {
            let p = TemporalPoint::new(i / 100, (i % 100) as i64 * 1000, vec![i as f32; 8]);
            store.put(0, &p).unwrap();
        }
        assert_eq!(store.len(), 100_000);

        // Retrieve specific entity
        let results = store.range(42, 0, 0, 100_000).unwrap();
        assert_eq!(results.len(), 100); // entity 42 has 100 points (i/100 == 42)

        // Verify ordering
        for window in results.windows(2) {
            assert!(window[0].timestamp() < window[1].timestamp());
        }
    }

    #[test]
    fn negative_timestamps_work() {
        let store = InMemoryStore::new();
        store.put(0, &sample_point(1, -5000)).unwrap();
        store.put(0, &sample_point(1, -1000)).unwrap();
        store.put(0, &sample_point(1, 0)).unwrap();
        store.put(0, &sample_point(1, 1000)).unwrap();

        let results = store.range(1, 0, -3000, 0).unwrap();
        assert_eq!(results.len(), 2); // -1000 and 0
        assert_eq!(results[0].timestamp(), -1000);
        assert_eq!(results[1].timestamp(), 0);
    }

    #[test]
    fn len_and_is_empty() {
        let store = InMemoryStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);

        store.put(0, &sample_point(1, 100)).unwrap();
        assert!(!store.is_empty());
        assert_eq!(store.len(), 1);
    }
}
