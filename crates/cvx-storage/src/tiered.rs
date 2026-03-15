//! Tiered storage: routes reads across hot → warm tiers transparently.
//!
//! Queries check the hot tier first, then fall through to warm storage.
//! Writes always go to the hot tier.

use cvx_core::StorageBackend;
use cvx_core::error::StorageError;
use cvx_core::types::TemporalPoint;

use crate::memory::InMemoryStore;
use crate::warm::WarmStore;

/// Composite storage that reads from hot first, then warm.
pub struct TieredStorage {
    /// Hot tier (fast, in-memory or RocksDB).
    hot: InMemoryStore,
    /// Warm tier (file-based, partitioned).
    warm: WarmStore,
}

impl TieredStorage {
    /// Create a new tiered storage with the given hot and warm stores.
    pub fn new(hot: InMemoryStore, warm: WarmStore) -> Self {
        Self { hot, warm }
    }

    /// Access the hot tier directly.
    pub fn hot(&self) -> &InMemoryStore {
        &self.hot
    }

    /// Access the warm tier directly.
    pub fn warm(&self) -> &WarmStore {
        &self.warm
    }

    /// Migrate points from hot to warm tier.
    ///
    /// Moves all points for the given entity+space with timestamp ≤ `cutoff`
    /// from hot to warm storage.
    pub fn compact(
        &self,
        entity_id: u64,
        space_id: u32,
        cutoff: i64,
    ) -> Result<usize, StorageError> {
        // Read from hot
        let points = self.hot.range(entity_id, space_id, i64::MIN, cutoff)?;
        if points.is_empty() {
            return Ok(0);
        }

        let count = points.len();

        // Write to warm
        self.warm.write_batch(space_id, &points)?;

        // Delete from hot
        for p in &points {
            self.hot.delete(entity_id, space_id, p.timestamp())?;
        }

        Ok(count)
    }
}

impl StorageBackend for TieredStorage {
    fn get(
        &self,
        entity_id: u64,
        space_id: u32,
        timestamp: i64,
    ) -> Result<Option<TemporalPoint>, StorageError> {
        // Try hot first
        if let Some(point) = self.hot.get(entity_id, space_id, timestamp)? {
            return Ok(Some(point));
        }
        // Fall through to warm
        self.warm.get(entity_id, space_id, timestamp)
    }

    fn put(&self, space_id: u32, point: &TemporalPoint) -> Result<(), StorageError> {
        // Always write to hot
        self.hot.put(space_id, point)
    }

    fn range(
        &self,
        entity_id: u64,
        space_id: u32,
        start: i64,
        end: i64,
    ) -> Result<Vec<TemporalPoint>, StorageError> {
        let mut hot_results = self.hot.range(entity_id, space_id, start, end)?;
        let warm_results = self.warm.range(entity_id, space_id, start, end)?;

        // Merge and deduplicate by timestamp
        hot_results.extend(warm_results);
        hot_results.sort_by_key(|p| p.timestamp());
        hot_results.dedup_by_key(|p| p.timestamp());

        Ok(hot_results)
    }

    fn delete(&self, entity_id: u64, space_id: u32, timestamp: i64) -> Result<(), StorageError> {
        self.hot.delete(entity_id, space_id, timestamp)?;
        self.warm.delete(entity_id, space_id, timestamp)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_point(entity_id: u64, timestamp: i64) -> TemporalPoint {
        TemporalPoint::new(entity_id, timestamp, vec![0.1, 0.2, 0.3])
    }

    fn make_tiered() -> (TieredStorage, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let hot = InMemoryStore::new();
        let warm = WarmStore::open(&dir.path().join("warm")).unwrap();
        (TieredStorage::new(hot, warm), dir)
    }

    #[test]
    fn write_to_hot_read_from_hot() {
        let (tiered, _dir) = make_tiered();
        let p = sample_point(1, 1000);
        tiered.put(0, &p).unwrap();

        let result = tiered.get(1, 0, 1000).unwrap();
        assert_eq!(result, Some(p));
    }

    #[test]
    fn compact_moves_to_warm() {
        let (tiered, _dir) = make_tiered();

        // Insert into hot
        for i in 0..10 {
            tiered.put(0, &sample_point(1, i * 1000)).unwrap();
        }
        assert_eq!(tiered.hot().len(), 10);

        // Compact: move timestamps ≤ 5000 to warm
        let moved = tiered.compact(1, 0, 5000).unwrap();
        assert_eq!(moved, 6); // 0, 1000, 2000, 3000, 4000, 5000

        // Hot should have remaining
        assert_eq!(tiered.hot().len(), 4);

        // Warm should have compacted points
        assert_eq!(tiered.warm().len(), 6);
    }

    #[test]
    fn get_finds_in_warm_after_compaction() {
        let (tiered, _dir) = make_tiered();

        tiered.put(0, &sample_point(1, 1000)).unwrap();
        tiered.compact(1, 0, 1000).unwrap();

        // Point is now in warm, not hot
        assert_eq!(tiered.hot().len(), 0);
        let result = tiered.get(1, 0, 1000).unwrap();
        assert!(result.is_some(), "should find point in warm tier");
    }

    #[test]
    fn range_merges_hot_and_warm() {
        let (tiered, _dir) = make_tiered();

        // Insert 10 points, compact first 5 to warm
        for i in 0..10 {
            tiered.put(0, &sample_point(1, i * 1000)).unwrap();
        }
        tiered.compact(1, 0, 4000).unwrap();

        // Range should span both tiers
        let results = tiered.range(1, 0, 0, 9000).unwrap();
        assert_eq!(results.len(), 10);

        // Verify ordering
        for w in results.windows(2) {
            assert!(w[0].timestamp() < w[1].timestamp());
        }
    }

    #[test]
    fn range_deduplicates() {
        let (tiered, _dir) = make_tiered();

        let p = sample_point(1, 1000);
        tiered.put(0, &p).unwrap();
        // Manually put in warm too (simulating partial compaction)
        tiered.warm().put(0, &p).unwrap();

        let results = tiered.range(1, 0, 0, 2000).unwrap();
        assert_eq!(results.len(), 1, "should deduplicate across tiers");
    }

    #[test]
    fn compact_empty_is_noop() {
        let (tiered, _dir) = make_tiered();
        let moved = tiered.compact(999, 0, i64::MAX).unwrap();
        assert_eq!(moved, 0);
    }

    #[test]
    fn compact_and_query_large() {
        let (tiered, _dir) = make_tiered();

        // Insert 1000 points
        for i in 0..1000 {
            tiered
                .put(0, &TemporalPoint::new(1, i * 100, vec![i as f32; 4]))
                .unwrap();
        }

        // Compact first 500
        let moved = tiered.compact(1, 0, 499 * 100).unwrap();
        assert_eq!(moved, 500);

        // All 1000 should still be findable
        let all = tiered.range(1, 0, 0, 100_000).unwrap();
        assert_eq!(all.len(), 1000);

        // Hot has 500, warm has 500
        assert_eq!(tiered.hot().len(), 500);
        assert_eq!(tiered.warm().len(), 500);
    }
}
