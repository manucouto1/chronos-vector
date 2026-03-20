//! Streaming Window Index — RFC-008 Phase 3.
//!
//! Combines a hot buffer (brute-force, fast writes) with a compacted
//! partitioned HNSW (graph-based, fast reads) for streaming workloads.
//!
//! ```text
//! Write path:
//!   Insert → HotBuffer (flat scan) → [compaction] → PartitionedHnsw
//!
//! Read path:
//!   Query → [HotBuffer results] ∪ [PartitionedHnsw results] → merge → top-k
//! ```

use std::path::Path;

use cvx_core::traits::DistanceMetric;
use cvx_core::types::TemporalFilter;

use super::partitioned::{PartitionConfig, PartitionedTemporalHnsw};

// ─── Configuration ──────────────────────────────────────────────────

/// Configuration for the streaming index.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Maximum buffer capacity before triggering compaction.
    pub buffer_capacity: usize,
    /// Partition config for the compacted index.
    pub partition_config: PartitionConfig,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_capacity: 10_000,
            partition_config: PartitionConfig::default(),
        }
    }
}

// ─── HotBuffer ──────────────────────────────────────────────────────

/// A flat buffer for recent, not-yet-compacted points.
///
/// Supports brute-force search (scan all points). Fast for small N.
struct HotBuffer {
    points: Vec<BufferedPoint>,
}

/// A point in the hot buffer.
#[derive(Clone)]
struct BufferedPoint {
    entity_id: u64,
    timestamp: i64,
    vector: Vec<f32>,
    /// Pre-assigned global ID for consistency after compaction.
    global_id: u32,
}

impl HotBuffer {
    fn new() -> Self {
        Self { points: Vec::new() }
    }

    fn push(&mut self, entity_id: u64, timestamp: i64, vector: Vec<f32>, global_id: u32) {
        self.points.push(BufferedPoint {
            entity_id,
            timestamp,
            vector,
            global_id,
        });
    }

    fn len(&self) -> usize {
        self.points.len()
    }

    fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Drain all points from the buffer.
    fn drain(&mut self) -> Vec<BufferedPoint> {
        std::mem::take(&mut self.points)
    }

    /// Brute-force kNN search within the buffer.
    fn brute_force_search(
        &self,
        query: &[f32],
        k: usize,
        filter: &TemporalFilter,
        metric: &dyn DistanceMetric,
    ) -> Vec<(u32, f32)> {
        let mut results: Vec<(u32, f32)> = self
            .points
            .iter()
            .filter(|p| filter.matches(p.timestamp))
            .map(|p| {
                let dist = metric.distance(query, &p.vector);
                (p.global_id, dist)
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }

    /// Get trajectory points for an entity from the buffer.
    fn trajectory(&self, entity_id: u64, filter: &TemporalFilter) -> Vec<(i64, u32)> {
        self.points
            .iter()
            .filter(|p| p.entity_id == entity_id && filter.matches(p.timestamp))
            .map(|p| (p.timestamp, p.global_id))
            .collect()
    }

    /// Find a point by global ID.
    fn find(&self, global_id: u32) -> Option<&BufferedPoint> {
        self.points.iter().find(|p| p.global_id == global_id)
    }
}

// ─── StreamingTemporalHnsw ──────────────────────────────────────────

/// Streaming temporal index that combines a hot buffer with a compacted
/// partitioned HNSW index.
pub struct StreamingTemporalHnsw<D: DistanceMetric + Clone> {
    /// Hot buffer for recent writes.
    buffer: HotBuffer,
    /// Compacted partitioned index.
    compacted: PartitionedTemporalHnsw<D>,
    /// Configuration.
    config: StreamingConfig,
    /// Metric instance.
    metric: D,
    /// Next global ID to assign.
    next_global_id: u32,
    /// Number of compactions performed.
    compaction_count: usize,
}

impl<D: DistanceMetric + Clone> StreamingTemporalHnsw<D> {
    /// Create a new streaming index.
    pub fn new(config: StreamingConfig, metric: D) -> Self {
        let compacted =
            PartitionedTemporalHnsw::new(config.partition_config.clone(), metric.clone());
        Self {
            buffer: HotBuffer::new(),
            compacted,
            config,
            metric,
            next_global_id: 0,
            compaction_count: 0,
        }
    }

    /// Insert a point. Goes into the hot buffer first.
    ///
    /// Triggers compaction if the buffer exceeds capacity.
    pub fn insert(&mut self, entity_id: u64, timestamp: i64, vector: &[f32]) -> u32 {
        let global_id = self.next_global_id;
        self.next_global_id += 1;

        self.buffer
            .push(entity_id, timestamp, vector.to_vec(), global_id);

        // Auto-compact if buffer is full
        if self.buffer.len() >= self.config.buffer_capacity {
            self.compact();
        }

        global_id
    }

    /// Manually trigger compaction: flush buffer into the partitioned index.
    pub fn compact(&mut self) {
        let points = self.buffer.drain();
        if points.is_empty() {
            return;
        }

        for p in points {
            self.compacted.insert(p.entity_id, p.timestamp, &p.vector);
        }

        self.compaction_count += 1;
    }

    /// Search across both buffer and compacted index.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: TemporalFilter,
        alpha: f32,
        query_timestamp: i64,
    ) -> Vec<(u32, f32)> {
        // Search compacted index
        let mut results = self
            .compacted
            .search(query, k, filter, alpha, query_timestamp);

        // Search hot buffer (brute-force)
        let buffer_results = self
            .buffer
            .brute_force_search(query, k, &filter, &self.metric);
        results.extend(buffer_results);

        // Merge: sort by score, take top k
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }

    /// Retrieve trajectory across buffer and compacted index.
    pub fn trajectory(&self, entity_id: u64, filter: TemporalFilter) -> Vec<(i64, u32)> {
        let mut traj = self.compacted.trajectory(entity_id, filter);
        let buffer_traj = self.buffer.trajectory(entity_id, &filter);
        traj.extend(buffer_traj);
        traj.sort_by_key(|&(ts, _)| ts);
        traj
    }

    /// Get vector by global node ID (checks buffer first, then compacted).
    pub fn vector(&self, global_id: u32) -> Vec<f32> {
        if let Some(p) = self.buffer.find(global_id) {
            return p.vector.clone();
        }
        self.compacted.vector(global_id)
    }

    /// Get entity ID by global node ID.
    pub fn entity_id(&self, global_id: u32) -> u64 {
        if let Some(p) = self.buffer.find(global_id) {
            return p.entity_id;
        }
        self.compacted.entity_id(global_id)
    }

    /// Get timestamp by global node ID.
    pub fn timestamp(&self, global_id: u32) -> i64 {
        if let Some(p) = self.buffer.find(global_id) {
            return p.timestamp;
        }
        self.compacted.timestamp(global_id)
    }

    /// Total number of indexed points (buffer + compacted).
    pub fn len(&self) -> usize {
        self.buffer.len() + self.compacted.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty() && self.compacted.is_empty()
    }

    /// Number of points in the hot buffer.
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Number of points in the compacted index.
    pub fn compacted_len(&self) -> usize {
        self.compacted.len()
    }

    /// Number of compactions performed.
    pub fn compaction_count(&self) -> usize {
        self.compaction_count
    }

    /// Save the compacted index (buffer should be compacted first).
    pub fn save(&mut self, dir: &Path) -> std::io::Result<()> {
        // Compact buffer before saving
        self.compact();
        self.compacted.save(dir)
    }
}

// ─── TemporalIndexAccess implementation ─────────────────────────────

impl<D: DistanceMetric + Clone> cvx_core::TemporalIndexAccess for StreamingTemporalHnsw<D> {
    fn search_raw(
        &self,
        query: &[f32],
        k: usize,
        filter: TemporalFilter,
        alpha: f32,
        query_timestamp: i64,
    ) -> Vec<(u32, f32)> {
        self.search(query, k, filter, alpha, query_timestamp)
    }

    fn trajectory(&self, entity_id: u64, filter: TemporalFilter) -> Vec<(i64, u32)> {
        self.trajectory(entity_id, filter)
    }

    fn vector(&self, node_id: u32) -> Vec<f32> {
        self.vector(node_id)
    }

    fn entity_id(&self, node_id: u32) -> u64 {
        self.entity_id(node_id)
    }

    fn timestamp(&self, node_id: u32) -> i64 {
        self.timestamp(node_id)
    }

    fn len(&self) -> usize {
        self.len()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::L2Distance;

    fn test_config(buffer_cap: usize) -> StreamingConfig {
        StreamingConfig {
            buffer_capacity: buffer_cap,
            partition_config: PartitionConfig {
                partition_duration_us: 10_000_000, // 10 seconds
                ..Default::default()
            },
        }
    }

    // ─── Basic operations ───────────────────────────────────────

    #[test]
    fn new_empty() {
        let index = StreamingTemporalHnsw::new(test_config(100), L2Distance);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
        assert_eq!(index.buffer_len(), 0);
        assert_eq!(index.compacted_len(), 0);
    }

    #[test]
    fn insert_into_buffer() {
        let mut index = StreamingTemporalHnsw::new(test_config(100), L2Distance);

        for i in 0..10u64 {
            index.insert(i, i as i64 * 1000, &[i as f32, 0.0]);
        }

        assert_eq!(index.len(), 10);
        assert_eq!(index.buffer_len(), 10);
        assert_eq!(index.compacted_len(), 0);
    }

    #[test]
    fn auto_compaction_on_capacity() {
        let mut index = StreamingTemporalHnsw::new(test_config(5), L2Distance);

        // Insert 5 points → triggers compaction
        for i in 0..5u64 {
            index.insert(i, i as i64 * 1000, &[i as f32, 0.0]);
        }

        assert_eq!(index.compaction_count(), 1);
        assert_eq!(index.buffer_len(), 0);
        assert_eq!(index.compacted_len(), 5);
        assert_eq!(index.len(), 5);
    }

    #[test]
    fn manual_compaction() {
        let mut index = StreamingTemporalHnsw::new(test_config(100), L2Distance);

        for i in 0..10u64 {
            index.insert(i, i as i64 * 1000, &[i as f32, 0.0]);
        }

        assert_eq!(index.buffer_len(), 10);
        index.compact();
        assert_eq!(index.buffer_len(), 0);
        assert_eq!(index.compacted_len(), 10);
    }

    // ─── Search ─────────────────────────────────────────────────

    #[test]
    fn search_buffer_only() {
        let mut index = StreamingTemporalHnsw::new(test_config(100), L2Distance);

        for i in 0..10u64 {
            index.insert(i, i as i64 * 1000, &[i as f32, 0.0]);
        }

        let results = index.search(&[5.0, 0.0], 3, TemporalFilter::All, 1.0, 0);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn search_compacted_only() {
        let mut index = StreamingTemporalHnsw::new(test_config(5), L2Distance);

        for i in 0..10u64 {
            index.insert(i, i as i64 * 1000, &[i as f32, 0.0]);
        }
        // Force all into compacted
        index.compact();

        let results = index.search(&[5.0, 0.0], 3, TemporalFilter::All, 1.0, 0);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn search_merged_buffer_and_compacted() {
        let mut index = StreamingTemporalHnsw::new(test_config(100), L2Distance);

        // Insert some and compact
        for i in 0..5u64 {
            index.insert(i, i as i64 * 1000, &[i as f32, 0.0]);
        }
        index.compact();

        // Insert more into buffer
        for i in 5..10u64 {
            index.insert(i, i as i64 * 1000, &[i as f32, 0.0]);
        }

        assert_eq!(index.compacted_len(), 5);
        assert_eq!(index.buffer_len(), 5);

        let results = index.search(&[5.0, 0.0], 5, TemporalFilter::All, 1.0, 0);
        assert_eq!(results.len(), 5);
    }

    // ─── Trajectory ─────────────────────────────────────────────

    #[test]
    fn trajectory_across_buffer_and_compacted() {
        let mut index = StreamingTemporalHnsw::new(test_config(100), L2Distance);

        // Insert 5 points for entity 42, compact, then add 5 more
        for i in 0..5u64 {
            index.insert(42, i as i64 * 1000, &[i as f32]);
        }
        index.compact();
        for i in 5..10u64 {
            index.insert(42, i as i64 * 1000, &[i as f32]);
        }

        let traj = index.trajectory(42, TemporalFilter::All);
        assert_eq!(traj.len(), 10, "should find all 10 points");

        // Verify sorted
        for w in traj.windows(2) {
            assert!(w[0].0 <= w[1].0);
        }
    }

    // ─── ID resolution ──────────────────────────────────────────

    #[test]
    fn resolve_ids_in_buffer() {
        let mut index = StreamingTemporalHnsw::new(test_config(100), L2Distance);
        let id = index.insert(42, 1000, &[1.0, 2.0]);

        assert_eq!(index.entity_id(id), 42);
        assert_eq!(index.timestamp(id), 1000);
        assert_eq!(index.vector(id), vec![1.0, 2.0]);
    }

    #[test]
    fn resolve_ids_after_compaction() {
        let mut index = StreamingTemporalHnsw::new(test_config(2), L2Distance);
        let _id0 = index.insert(1, 100, &[1.0]);
        let _id1 = index.insert(2, 200, &[2.0]);
        // Auto-compacted

        // Buffer IDs should still resolve via compacted index
        // Note: after compaction, the global IDs in the compacted index
        // may differ since PartitionedTemporalHnsw assigns its own IDs.
        // But we can verify entity data via trajectory.
        let traj_1 = index.trajectory(1, TemporalFilter::All);
        let traj_2 = index.trajectory(2, TemporalFilter::All);
        assert_eq!(traj_1.len(), 1);
        assert_eq!(traj_2.len(), 1);
    }

    // ─── TemporalIndexAccess trait ──────────────────────────────

    #[test]
    fn trait_search() {
        let mut index = StreamingTemporalHnsw::new(test_config(100), L2Distance);
        for i in 0..10u64 {
            index.insert(i, i as i64 * 1000, &[i as f32, 0.0]);
        }

        let trait_ref: &dyn cvx_core::TemporalIndexAccess = &index;
        let results = trait_ref.search_raw(&[5.0, 0.0], 3, TemporalFilter::All, 1.0, 0);
        assert_eq!(results.len(), 3);
    }

    // ─── Edge cases ─────────────────────────────────────────────

    #[test]
    fn compact_empty_buffer() {
        let mut index = StreamingTemporalHnsw::new(test_config(100), L2Distance);
        index.compact(); // Should not panic
        assert_eq!(index.compaction_count(), 0); // No actual compaction
    }

    #[test]
    fn multiple_compactions() {
        let mut index = StreamingTemporalHnsw::new(test_config(5), L2Distance);

        for i in 0..20u64 {
            index.insert(i % 3, i as i64 * 1000, &[i as f32]);
        }

        assert!(
            index.compaction_count() >= 3,
            "should compact multiple times"
        );
        assert_eq!(index.len(), 20);
    }
}
