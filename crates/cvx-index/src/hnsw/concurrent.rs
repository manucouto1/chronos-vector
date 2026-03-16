//! Thread-safe concurrent HNSW index.
//!
//! Wraps [`TemporalHnsw`] with a [`parking_lot::RwLock`] for concurrent access:
//! - Multiple readers can search simultaneously (read lock)
//! - A single writer can insert (write lock)
//!
//! This is the main entry point for production use.
//!
//! # Example
//!
//! ```
//! use cvx_index::hnsw::{ConcurrentTemporalHnsw, HnswConfig};
//! use cvx_index::metrics::L2Distance;
//! use cvx_core::TemporalFilter;
//! use std::sync::Arc;
//!
//! let config = HnswConfig::default();
//! let index = Arc::new(ConcurrentTemporalHnsw::new(config, L2Distance));
//!
//! // Insert (takes write lock)
//! index.insert(1, 1000, &[1.0, 0.0, 0.0]);
//!
//! // Search (takes read lock) — can run from multiple threads
//! let results = index.search(&[1.0, 0.0, 0.0], 5, TemporalFilter::All, 1.0, 1000);
//! assert_eq!(results.len(), 1);
//! ```

use cvx_core::{DistanceMetric, TemporalFilter};
use parking_lot::{Mutex, RwLock};

use super::HnswConfig;
use super::temporal::TemporalHnsw;

/// A pending insert waiting in the queue.
struct PendingInsert {
    entity_id: u64,
    timestamp: i64,
    vector: Vec<f32>,
}

/// Thread-safe spatiotemporal HNSW index with insert queue (RFC-002-04).
///
/// Uses a two-tier approach to reduce write lock contention:
/// - Inserts are queued into a `Mutex<Vec<...>>` (sub-microsecond)
/// - `flush_inserts()` drains the queue under a single write lock
/// - Searches always acquire a read lock (concurrent, unblocked during queue drain)
///
/// For immediate visibility, use `insert()` which still takes the write lock directly.
/// For high-throughput ingestion, use `queue_insert()` + `flush_inserts()`.
pub struct ConcurrentTemporalHnsw<D: DistanceMetric> {
    inner: RwLock<TemporalHnsw<D>>,
    /// Insert queue for batched commits (RFC-002-04, Option A).
    insert_queue: Mutex<Vec<PendingInsert>>,
}

impl<D: DistanceMetric> ConcurrentTemporalHnsw<D> {
    /// Create a new empty concurrent index.
    pub fn new(config: HnswConfig, metric: D) -> Self {
        Self {
            inner: RwLock::new(TemporalHnsw::new(config, metric)),
            insert_queue: Mutex::new(Vec::new()),
        }
    }

    /// Number of points in the index.
    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.read().is_empty()
    }

    /// Insert a temporal point (exclusive write lock).
    pub fn insert(&self, entity_id: u64, timestamp: i64, vector: &[f32]) -> u32 {
        self.inner.write().insert(entity_id, timestamp, vector)
    }

    /// Search with temporal filtering (shared read lock).
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: TemporalFilter,
        alpha: f32,
        query_timestamp: i64,
    ) -> Vec<(u32, f32)> {
        self.inner
            .read()
            .search(query, k, filter, alpha, query_timestamp)
    }

    /// Retrieve trajectory for an entity (shared read lock).
    pub fn trajectory(&self, entity_id: u64, filter: TemporalFilter) -> Vec<(i64, u32)> {
        self.inner.read().trajectory(entity_id, filter)
    }

    /// Get timestamp for a node (shared read lock).
    pub fn timestamp(&self, node_id: u32) -> i64 {
        self.inner.read().timestamp(node_id)
    }

    /// Get entity_id for a node (shared read lock).
    pub fn entity_id(&self, node_id: u32) -> u64 {
        self.inner.read().entity_id(node_id)
    }

    /// Get vector for a node (shared read lock).
    pub fn vector(&self, node_id: u32) -> Vec<f32> {
        self.inner.read().vector(node_id).to_vec()
    }

    /// Queue an insert for batched processing (RFC-002-04).
    ///
    /// This only takes a `Mutex` (sub-microsecond), NOT the write lock.
    /// The insert becomes visible after `flush_inserts()` is called.
    pub fn queue_insert(&self, entity_id: u64, timestamp: i64, vector: Vec<f32>) {
        self.insert_queue.lock().push(PendingInsert {
            entity_id,
            timestamp,
            vector,
        });
    }

    /// Number of pending inserts in the queue.
    pub fn pending_inserts(&self) -> usize {
        self.insert_queue.lock().len()
    }

    /// Flush all queued inserts, applying them under a single write lock.
    ///
    /// Returns the number of inserts applied.
    pub fn flush_inserts(&self) -> usize {
        let pending: Vec<PendingInsert> = {
            let mut queue = self.insert_queue.lock();
            std::mem::take(&mut *queue)
        };

        if pending.is_empty() {
            return 0;
        }

        let count = pending.len();
        let mut inner = self.inner.write();
        for p in pending {
            inner.insert(p.entity_id, p.timestamp, &p.vector);
        }
        count
    }
}

impl<D: DistanceMetric> cvx_core::TemporalIndexAccess for ConcurrentTemporalHnsw<D> {
    fn search_raw(
        &self,
        query: &[f32],
        k: usize,
        filter: TemporalFilter,
        alpha: f32,
        query_timestamp: i64,
    ) -> Vec<(u32, f32)> {
        self.inner.read().search(query, k, filter, alpha, query_timestamp)
    }

    fn trajectory(&self, entity_id: u64, filter: TemporalFilter) -> Vec<(i64, u32)> {
        self.inner.read().trajectory(entity_id, filter)
    }

    fn vector(&self, node_id: u32) -> Vec<f32> {
        self.inner.read().vector(node_id).to_vec()
    }

    fn entity_id(&self, node_id: u32) -> u64 {
        self.inner.read().entity_id(node_id)
    }

    fn timestamp(&self, node_id: u32) -> i64 {
        self.inner.read().timestamp(node_id)
    }

    fn len(&self) -> usize {
        self.inner.read().len()
    }

    fn regions(&self, level: usize) -> Vec<(u32, Vec<f32>, usize)> {
        self.inner.read().regions(level)
    }

    fn region_trajectory(
        &self,
        entity_id: u64,
        level: usize,
        window_days: i64,
        alpha: f32,
    ) -> Vec<(i64, Vec<f32>)> {
        self.inner.read().region_trajectory(entity_id, level, window_days, alpha)
    }
}

impl<D: DistanceMetric> cvx_core::IndexBackend for ConcurrentTemporalHnsw<D> {
    fn insert(
        &self,
        entity_id: u64,
        vector: &[f32],
        timestamp: i64,
    ) -> Result<u32, cvx_core::error::IndexError> {
        Ok(self.inner.write().insert(entity_id, timestamp, vector))
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: TemporalFilter,
        alpha: f32,
        query_timestamp: i64,
    ) -> Result<Vec<cvx_core::ScoredResult>, cvx_core::error::QueryError> {
        let inner = self.inner.read();
        let raw_results = inner.search(query, k, filter, alpha, query_timestamp);

        let results = raw_results
            .into_iter()
            .map(|(node_id, combined_score)| {
                let entity_id = inner.entity_id(node_id);
                let timestamp = inner.timestamp(node_id);
                let vector = inner.vector(node_id).to_vec();
                let point = cvx_core::TemporalPoint::new(entity_id, timestamp, vector);

                // Decompose combined score into semantic and temporal components
                let temporal_dist = inner.temporal_distance_normalized(timestamp, query_timestamp);
                let semantic_dist = if alpha > 0.0 {
                    (combined_score - (1.0 - alpha) * temporal_dist) / alpha
                } else {
                    0.0
                };

                cvx_core::ScoredResult::new(point, semantic_dist, temporal_dist, combined_score)
            })
            .collect();

        Ok(results)
    }

    fn remove(&self, _point_id: u64) -> Result<(), cvx_core::error::IndexError> {
        // Removal not yet supported in HNSW — mark as tombstone in future
        Ok(())
    }

    fn len(&self) -> usize {
        self.inner.read().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::L2Distance;
    use std::sync::Arc;
    use std::thread;

    fn make_concurrent_index() -> Arc<ConcurrentTemporalHnsw<L2Distance>> {
        let config = HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            ..Default::default()
        };
        Arc::new(ConcurrentTemporalHnsw::new(config, L2Distance))
    }

    #[test]
    fn single_thread_basic() {
        let index = make_concurrent_index();
        index.insert(1, 1000, &[1.0, 0.0, 0.0]);
        index.insert(2, 2000, &[0.0, 1.0, 0.0]);

        let results = index.search(&[1.0, 0.0, 0.0], 2, TemporalFilter::All, 1.0, 0);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // closest
    }

    #[test]
    fn concurrent_readers() {
        let index = make_concurrent_index();

        // Insert some data first
        for i in 0..100u64 {
            index.insert(i, (i * 100) as i64, &[i as f32, 0.0, 0.0]);
        }

        // Spawn 8 reader threads
        let n_threads = 8;
        let mut handles = Vec::new();

        for t in 0..n_threads {
            let idx = Arc::clone(&index);
            handles.push(thread::spawn(move || {
                let query = [t as f32, 0.0, 0.0];
                for _ in 0..100 {
                    let results = idx.search(&query, 5, TemporalFilter::All, 1.0, 0);
                    assert_eq!(results.len(), 5);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn concurrent_readers_and_writer() {
        let index = make_concurrent_index();

        // Pre-populate with some data
        for i in 0..50u64 {
            index.insert(i, (i * 100) as i64, &[i as f32, 0.0, 0.0]);
        }

        let idx_writer = Arc::clone(&index);
        let idx_readers: Vec<_> = (0..8).map(|_| Arc::clone(&index)).collect();

        // Writer thread
        let writer = thread::spawn(move || {
            for i in 50..150u64 {
                idx_writer.insert(i, (i * 100) as i64, &[i as f32, 0.0, 0.0]);
            }
        });

        // Reader threads
        let readers: Vec<_> = idx_readers
            .into_iter()
            .map(|idx| {
                thread::spawn(move || {
                    let query = [50.0, 0.0, 0.0];
                    for _ in 0..50 {
                        let results = idx.search(&query, 5, TemporalFilter::All, 1.0, 0);
                        // Results should always be non-empty since we pre-populated
                        assert!(!results.is_empty());
                    }
                })
            })
            .collect();

        writer.join().unwrap();
        for r in readers {
            r.join().unwrap();
        }

        // Verify final state
        assert_eq!(index.len(), 150);
    }

    #[test]
    fn concurrent_search_with_temporal_filter() {
        let index = make_concurrent_index();

        for i in 0..200u64 {
            index.insert(i % 10, (i * 100) as i64, &[i as f32, 0.0]);
        }

        let mut handles = Vec::new();
        for t in 0..8 {
            let idx = Arc::clone(&index);
            handles.push(thread::spawn(move || {
                let filter = TemporalFilter::Range(1000, 5000);
                for _ in 0..50 {
                    let results = idx.search(&[t as f32 * 10.0, 0.0], 5, filter, 0.5, 3000);
                    // All results should have timestamps in range
                    for &(id, _) in &results {
                        let ts = idx.timestamp(id);
                        assert!(
                            ts >= 1000 && ts <= 5000,
                            "timestamp {ts} out of [1000, 5000]"
                        );
                    }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn queue_insert_and_flush() {
        let index = make_concurrent_index();

        // Queue 100 inserts (no write lock held)
        for i in 0..100u64 {
            index.queue_insert(i, (i * 100) as i64, vec![i as f32, 0.0, 0.0]);
        }

        // Not yet visible
        assert_eq!(index.len(), 0);
        assert_eq!(index.pending_inserts(), 100);

        // Flush applies them all under a single write lock
        let flushed = index.flush_inserts();
        assert_eq!(flushed, 100);
        assert_eq!(index.len(), 100);
        assert_eq!(index.pending_inserts(), 0);

        // Searchable after flush
        let results = index.search(&[50.0, 0.0, 0.0], 5, TemporalFilter::All, 1.0, 0);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn queue_insert_concurrent_with_search() {
        let index = make_concurrent_index();

        // Pre-populate so searches always return results
        for i in 0..50u64 {
            index.insert(i, (i * 100) as i64, &[i as f32, 0.0, 0.0]);
        }

        let idx_queue = Arc::clone(&index);
        let idx_search: Vec<_> = (0..4).map(|_| Arc::clone(&index)).collect();

        // Queue thread: queues inserts without blocking searches
        let queue_thread = thread::spawn(move || {
            for i in 50..200u64 {
                idx_queue.queue_insert(i, (i * 100) as i64, vec![i as f32, 0.0, 0.0]);
            }
        });

        // Search threads: run concurrently with queue thread
        let search_threads: Vec<_> = idx_search
            .into_iter()
            .map(|idx| {
                thread::spawn(move || {
                    for _ in 0..50 {
                        let results = idx.search(&[25.0, 0.0, 0.0], 5, TemporalFilter::All, 1.0, 0);
                        assert!(!results.is_empty());
                    }
                })
            })
            .collect();

        queue_thread.join().unwrap();
        for t in search_threads {
            t.join().unwrap();
        }

        // Flush and verify
        let flushed = index.flush_inserts();
        assert_eq!(flushed, 150);
        assert_eq!(index.len(), 200);
    }

    #[test]
    fn trajectory_concurrent() {
        let index = make_concurrent_index();

        // Insert trajectory for entity 1
        for i in 0..50u32 {
            index.insert(1, (i as i64) * 100, &[i as f32]);
        }

        let mut handles = Vec::new();
        for _ in 0..4 {
            let idx = Arc::clone(&index);
            handles.push(thread::spawn(move || {
                let traj = idx.trajectory(1, TemporalFilter::All);
                assert_eq!(traj.len(), 50);
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }
}
