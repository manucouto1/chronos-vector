//! Spatiotemporal HNSW index (ST-HNSW).
//!
//! Wraps the vanilla [`HnswGraph`] with temporal awareness:
//! - **Roaring Bitmaps** for O(1) temporal pre-filtering
//! - **Composite distance**: $d_{ST} = \alpha \cdot d_{sem} + (1 - \alpha) \cdot d_{time}$
//! - **TemporalFilter** integration: snapshot kNN, range kNN
//! - **Trajectory retrieval**: all points for an entity ordered by time
//!
//! # Example
//!
//! ```
//! use cvx_index::hnsw::{HnswConfig, TemporalHnsw};
//! use cvx_index::metrics::L2Distance;
//! use cvx_core::TemporalFilter;
//!
//! let config = HnswConfig::default();
//! let mut index = TemporalHnsw::new(config, L2Distance);
//!
//! // Insert vectors with entity_id and timestamp
//! index.insert(1, 1000, &[1.0, 0.0, 0.0]);
//! index.insert(1, 2000, &[0.9, 0.1, 0.0]);
//! index.insert(2, 1500, &[0.0, 1.0, 0.0]);
//!
//! // Temporal range search with alpha=0.5
//! let results = index.search(
//!     &[1.0, 0.0, 0.0],
//!     2,
//!     TemporalFilter::Range(900, 1600),
//!     0.5,
//!     1000, // query timestamp for temporal distance
//! );
//! assert_eq!(results.len(), 2); // only 2 points in [900, 1600]
//! ```

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::Path;

use cvx_core::{DistanceMetric, TemporalFilter};
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

use super::{HnswConfig, HnswGraph, HnswSnapshot};

/// Spatiotemporal HNSW index.
///
/// Each inserted point has an `entity_id`, a `timestamp`, and a vector.
/// Internal node IDs are assigned sequentially.
pub struct TemporalHnsw<D: DistanceMetric> {
    graph: HnswGraph<D>,
    /// node_id → timestamp
    timestamps: Vec<i64>,
    /// node_id → entity_id
    entity_ids: Vec<u64>,
    /// entity_id → sorted vec of (timestamp, node_id)
    entity_index: BTreeMap<u64, Vec<(i64, u32)>>,
    /// Global temporal range for normalization
    min_timestamp: i64,
    max_timestamp: i64,
}

impl<D: DistanceMetric> TemporalHnsw<D> {
    /// Create a new empty spatiotemporal index.
    pub fn new(config: HnswConfig, metric: D) -> Self {
        Self {
            graph: HnswGraph::new(config, metric),
            timestamps: Vec::new(),
            entity_ids: Vec::new(),
            entity_index: BTreeMap::new(),
            min_timestamp: i64::MAX,
            max_timestamp: i64::MIN,
        }
    }

    /// Number of points in the index.
    pub fn len(&self) -> usize {
        self.graph.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.graph.is_empty()
    }

    /// Get the last (most recent) node_id for an entity, or None if not found.
    pub fn entity_last_node(&self, entity_id: u64) -> Option<u32> {
        self.entity_index
            .get(&entity_id)
            .and_then(|pts| pts.last().map(|&(_, nid)| nid))
    }

    /// Set ef_construction at runtime (lower for bulk load, higher for quality).
    pub fn set_ef_construction(&mut self, ef: usize) {
        self.graph.set_ef_construction(ef);
    }

    /// Set ef_search at runtime.
    pub fn set_ef_search(&mut self, ef: usize) {
        self.graph.set_ef_search(ef);
    }

    /// Get the current configuration.
    pub fn config(&self) -> &HnswConfig {
        self.graph.config()
    }

    /// Enable scalar quantization for faster distance computation.
    pub fn enable_scalar_quantization(&mut self, min_val: f32, max_val: f32) {
        self.graph.enable_scalar_quantization(min_val, max_val);
    }

    /// Disable scalar quantization.
    pub fn disable_scalar_quantization(&mut self) {
        self.graph.disable_scalar_quantization();
    }

    /// Whether scalar quantization is active.
    pub fn is_quantized(&self) -> bool {
        self.graph.is_quantized()
    }

    /// Insert a temporal point into the index.
    ///
    /// Returns the internal node_id assigned to this point.
    pub fn insert(&mut self, entity_id: u64, timestamp: i64, vector: &[f32]) -> u32 {
        let node_id = self.graph.len() as u32;
        self.graph.insert(node_id, vector);
        self.timestamps.push(timestamp);
        self.entity_ids.push(entity_id);

        // Update entity index
        self.entity_index
            .entry(entity_id)
            .or_default()
            .push((timestamp, node_id));

        // Update temporal range
        self.min_timestamp = self.min_timestamp.min(timestamp);
        self.max_timestamp = self.max_timestamp.max(timestamp);

        node_id
    }

    /// Build a Roaring Bitmap of node IDs matching the temporal filter.
    pub fn build_filter_bitmap(&self, filter: &TemporalFilter) -> RoaringBitmap {
        let mut bitmap = RoaringBitmap::new();
        for (i, &ts) in self.timestamps.iter().enumerate() {
            if filter.matches(ts) {
                bitmap.insert(i as u32);
            }
        }
        bitmap
    }

    /// Compute normalized temporal distance between two timestamps.
    ///
    /// Returns a value in `[0.0, 1.0]` where 0 = same timestamp, 1 = max range.
    pub fn temporal_distance_normalized(&self, t1: i64, t2: i64) -> f32 {
        let range = (self.max_timestamp - self.min_timestamp).max(1) as f64;
        let diff = (t1 as f64 - t2 as f64).abs();
        (diff / range) as f32
    }

    /// Search for the k nearest neighbors with temporal filtering and composite scoring.
    ///
    /// - `query`: the query vector
    /// - `k`: number of results
    /// - `filter`: temporal constraint (Snapshot, Range, Before, After, All)
    /// - `alpha`: weight for semantic distance (1.0 = pure semantic, 0.0 = pure temporal)
    /// - `query_timestamp`: reference timestamp for temporal distance computation
    ///
    /// Returns `(node_id, combined_score)` sorted by combined score ascending.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: TemporalFilter,
        alpha: f32,
        query_timestamp: i64,
    ) -> Vec<(u32, f32)> {
        if self.is_empty() {
            return Vec::new();
        }

        // Build bitmap of temporally valid nodes
        let bitmap = self.build_filter_bitmap(&filter);
        if bitmap.is_empty() {
            return Vec::new();
        }

        if alpha >= 1.0 {
            // Pure semantic: just filter, no re-ranking needed
            return self
                .graph
                .search_filtered(query, k, |id| bitmap.contains(id));
        }

        // Get more candidates than needed for re-ranking
        let over_fetch = k * 4;
        let candidates = self
            .graph
            .search_filtered(query, over_fetch, |id| bitmap.contains(id));

        // Re-rank with composite distance
        let mut scored: Vec<(u32, f32)> = candidates
            .into_iter()
            .map(|(id, sem_dist)| {
                let t_dist = self
                    .temporal_distance_normalized(self.timestamps[id as usize], query_timestamp);
                let combined = alpha * sem_dist + (1.0 - alpha) * t_dist;
                (id, combined)
            })
            .collect();

        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        scored.truncate(k);
        scored
    }

    /// Retrieve the full trajectory for an entity within a time range.
    ///
    /// Returns `(timestamp, node_id)` pairs sorted by timestamp ascending.
    pub fn trajectory(&self, entity_id: u64, filter: TemporalFilter) -> Vec<(i64, u32)> {
        let Some(points) = self.entity_index.get(&entity_id) else {
            return Vec::new();
        };

        let mut result: Vec<(i64, u32)> = points
            .iter()
            .filter(|&&(ts, _)| filter.matches(ts))
            .copied()
            .collect();

        result.sort_by_key(|&(ts, _)| ts);
        result
    }

    /// Get the timestamp for a node.
    pub fn timestamp(&self, node_id: u32) -> i64 {
        self.timestamps[node_id as usize]
    }

    /// Get the entity_id for a node.
    pub fn entity_id(&self, node_id: u32) -> u64 {
        self.entity_ids[node_id as usize]
    }

    /// Get the vector for a node.
    pub fn vector(&self, node_id: u32) -> &[f32] {
        self.graph.vector(node_id)
    }

    /// Approximate memory usage of the Roaring Bitmaps for a full-index filter.
    ///
    /// Useful for verifying the < 1 byte/vector target.
    pub fn bitmap_memory_bytes(&self) -> usize {
        let bitmap = self.build_filter_bitmap(&TemporalFilter::All);
        bitmap.serialized_size()
    }

    /// Access the underlying HNSW graph (for recall comparisons, etc.).
    pub fn graph(&self) -> &HnswGraph<D> {
        &self.graph
    }

    // ─── Semantic Regions (RFC-004) ────────────────────────────────

    /// Get semantic regions at a given HNSW level.
    ///
    /// Returns `(hub_node_id, hub_vector, n_assigned_nodes)` for each region.
    /// Use level 2-3 for interpretable granularity (~N/M^L regions).
    pub fn regions(&self, level: usize) -> Vec<(u32, Vec<f32>, usize)> {
        let hubs = self.graph.nodes_at_level(level);
        let n = self.graph.len();

        // Count assignments: for each node, find nearest hub
        let mut counts = vec![0usize; hubs.len()];
        let hub_set: std::collections::HashMap<u32, usize> =
            hubs.iter().enumerate().map(|(i, &h)| (h, i)).collect();

        for node_id in 0..n as u32 {
            if let Some(hub) = self.graph.assign_region(self.graph.vector(node_id), level) {
                if let Some(&idx) = hub_set.get(&hub) {
                    counts[idx] += 1;
                }
            }
        }

        hubs.iter()
            .enumerate()
            .map(|(i, &hub_id)| (hub_id, self.graph.vector(hub_id).to_vec(), counts[i]))
            .collect()
    }

    /// Compute smoothed region-distribution trajectory for an entity (RFC-004).
    ///
    /// - `level`: HNSW level for region granularity
    /// - `window_days`: sliding window in timestamp units (same scale as ingested timestamps)
    /// - `alpha`: EMA smoothing factor (0.3 typical, higher = more reactive)
    ///
    /// Returns `(timestamp, distribution)` where distribution is a Vec<f32> of length K
    /// (number of regions) that sums to ~1.0.
    pub fn region_trajectory(
        &self,
        entity_id: u64,
        level: usize,
        window_days: i64,
        alpha: f32,
    ) -> Vec<(i64, Vec<f32>)> {
        let hubs = self.graph.nodes_at_level(level);
        let k = hubs.len();
        if k == 0 {
            return Vec::new();
        }

        // Map hub_node_id → region index
        let hub_index: std::collections::HashMap<u32, usize> =
            hubs.iter().enumerate().map(|(i, &h)| (h, i)).collect();

        // Get entity's posts sorted by time
        let posts = self.trajectory(entity_id, TemporalFilter::All);
        if posts.is_empty() {
            return Vec::new();
        }

        // Assign each post to a region
        let assignments: Vec<(i64, usize)> = posts
            .iter()
            .filter_map(|&(ts, node_id)| {
                let vec = self.graph.vector(node_id);
                self.graph
                    .assign_region(vec, level)
                    .and_then(|hub| hub_index.get(&hub).map(|&idx| (ts, idx)))
            })
            .collect();

        if assignments.is_empty() {
            return Vec::new();
        }

        // Group by time windows
        let t_start = assignments[0].0;
        let t_end = assignments.last().unwrap().0;
        let mut result = Vec::new();
        let mut ema_state: Vec<f32> = vec![0.0; k];
        let mut first = true;

        let mut window_start = t_start;
        while window_start <= t_end {
            let window_end = window_start + window_days;

            // Count posts per region in this window
            let mut counts = vec![0.0f32; k];
            let mut n_in_window = 0.0f32;
            for &(ts, region_idx) in &assignments {
                if ts >= window_start && ts < window_end {
                    counts[region_idx] += 1.0;
                    n_in_window += 1.0;
                }
            }

            if n_in_window > 0.0 {
                // Normalize to distribution
                for c in &mut counts {
                    *c /= n_in_window;
                }

                // EMA smoothing
                if first {
                    ema_state = counts;
                    first = false;
                } else {
                    for i in 0..k {
                        ema_state[i] = alpha * counts[i] + (1.0 - alpha) * ema_state[i];
                    }
                }

                let mid_ts = window_start + window_days / 2;
                result.push((mid_ts, ema_state.clone()));
            }

            window_start = window_end;
        }

        result
    }

    /// Get points assigned to a specific region, optionally time-filtered (RFC-004, RFC-005).
    ///
    /// Returns `(node_id, entity_id, timestamp)` for all points in the region.
    /// This is the "SELECT * FROM points WHERE region = R" equivalent.
    pub fn region_members(
        &self,
        region_hub: u32,
        level: usize,
        filter: TemporalFilter,
    ) -> Vec<(u32, u64, i64)> {
        let mut members = Vec::new();
        for node_id in 0..self.graph.len() as u32 {
            let ts = self.timestamps[node_id as usize];
            if !filter.matches(ts) {
                continue;
            }
            let vec = self.graph.vector(node_id);
            if let Some(assigned_hub) = self.graph.assign_region(vec, level) {
                if assigned_hub == region_hub {
                    let eid = self.entity_ids[node_id as usize];
                    members.push((node_id, eid, ts));
                }
            }
        }
        members
    }
}

/// Serializable snapshot of a TemporalHnsw index.
#[derive(Serialize, Deserialize)]
struct TemporalSnapshot {
    graph: HnswSnapshot,
    timestamps: Vec<i64>,
    entity_ids: Vec<u64>,
    entity_index: BTreeMap<u64, Vec<(i64, u32)>>,
    min_timestamp: i64,
    max_timestamp: i64,
}

impl<D: DistanceMetric> TemporalHnsw<D> {
    /// Save the index to a file using postcard binary encoding.
    ///
    /// The distance metric is NOT serialized (it's stateless).
    /// On load, you must provide the same metric type.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let snapshot = TemporalSnapshot {
            graph: self.graph.to_snapshot(),
            timestamps: self.timestamps.clone(),
            entity_ids: self.entity_ids.clone(),
            entity_index: self.entity_index.clone(),
            min_timestamp: self.min_timestamp,
            max_timestamp: self.max_timestamp,
        };

        let bytes = postcard::to_allocvec(&snapshot).map_err(std::io::Error::other)?;

        let mut file = std::fs::File::create(path)?;
        file.write_all(&bytes)?;
        Ok(())
    }

    /// Load an index from a file, providing the distance metric.
    ///
    /// The metric must match the one used during construction.
    pub fn load(path: &Path, metric: D) -> std::io::Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;

        let snapshot: TemporalSnapshot = postcard::from_bytes(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        Ok(Self {
            graph: HnswGraph::from_snapshot(snapshot.graph, metric),
            timestamps: snapshot.timestamps,
            entity_ids: snapshot.entity_ids,
            entity_index: snapshot.entity_index,
            min_timestamp: snapshot.min_timestamp,
            max_timestamp: snapshot.max_timestamp,
        })
    }
}

impl<D: DistanceMetric> cvx_core::TemporalIndexAccess for TemporalHnsw<D> {
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
        self.graph.vector(node_id).to_vec()
    }

    fn entity_id(&self, node_id: u32) -> u64 {
        self.entity_ids[node_id as usize]
    }

    fn timestamp(&self, node_id: u32) -> i64 {
        self.timestamps[node_id as usize]
    }

    fn len(&self) -> usize {
        self.graph.len()
    }

    fn regions(&self, level: usize) -> Vec<(u32, Vec<f32>, usize)> {
        self.regions(level)
    }

    fn region_members(
        &self,
        region_hub: u32,
        level: usize,
        filter: TemporalFilter,
    ) -> Vec<(u32, u64, i64)> {
        self.region_members(region_hub, level, filter)
    }

    fn region_trajectory(
        &self,
        entity_id: u64,
        level: usize,
        window_days: i64,
        alpha: f32,
    ) -> Vec<(i64, Vec<f32>)> {
        self.region_trajectory(entity_id, level, window_days, alpha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{CosineDistance, L2Distance};

    fn make_temporal_index() -> TemporalHnsw<L2Distance> {
        let config = HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            ..Default::default()
        };
        TemporalHnsw::new(config, L2Distance)
    }

    // ─── Basic functionality ────────────────────────────────────────────

    #[test]
    fn empty_index() {
        let index = make_temporal_index();
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        let results = index.search(&[1.0, 0.0], 5, TemporalFilter::All, 1.0, 0);
        assert!(results.is_empty());
    }

    #[test]
    fn insert_and_metadata() {
        let mut index = make_temporal_index();
        let id = index.insert(42, 1000, &[1.0, 0.0, 0.0]);
        assert_eq!(id, 0);
        assert_eq!(index.len(), 1);
        assert_eq!(index.timestamp(0), 1000);
        assert_eq!(index.entity_id(0), 42);
        assert_eq!(index.vector(0), &[1.0, 0.0, 0.0]);
    }

    // ─── Snapshot kNN ───────────────────────────────────────────────────

    #[test]
    fn snapshot_knn_returns_only_matching_timestamp() {
        let mut index = make_temporal_index();
        // Entity 1 at t=1000
        index.insert(1, 1000, &[1.0, 0.0]);
        // Entity 2 at t=2000
        index.insert(2, 2000, &[0.9, 0.1]);
        // Entity 3 at t=1000
        index.insert(3, 1000, &[0.8, 0.2]);

        let results = index.search(&[1.0, 0.0], 10, TemporalFilter::Snapshot(1000), 1.0, 1000);
        assert_eq!(results.len(), 2);
        // Both results should have timestamp 1000
        for &(id, _) in &results {
            assert_eq!(index.timestamp(id), 1000);
        }
    }

    #[test]
    fn snapshot_knn_no_match_returns_empty() {
        let mut index = make_temporal_index();
        index.insert(1, 1000, &[1.0, 0.0]);
        index.insert(2, 2000, &[0.9, 0.1]);

        let results = index.search(&[1.0, 0.0], 10, TemporalFilter::Snapshot(5000), 1.0, 5000);
        assert!(results.is_empty());
    }

    // ─── Range kNN ──────────────────────────────────────────────────────

    #[test]
    fn range_knn_returns_only_in_range() {
        let mut index = make_temporal_index();
        index.insert(1, 1000, &[1.0, 0.0]);
        index.insert(2, 2000, &[0.9, 0.1]);
        index.insert(3, 3000, &[0.8, 0.2]);
        index.insert(4, 4000, &[0.7, 0.3]);

        let results = index.search(
            &[1.0, 0.0],
            10,
            TemporalFilter::Range(1500, 3500),
            1.0,
            2000,
        );

        // Only t=2000 and t=3000 should match
        assert_eq!(results.len(), 2);
        for &(id, _) in &results {
            let ts = index.timestamp(id);
            assert!(ts >= 1500 && ts <= 3500, "timestamp {ts} out of range");
        }
    }

    // ─── Composite distance ─────────────────────────────────────────────

    #[test]
    fn alpha_1_is_pure_semantic() {
        let mut index = make_temporal_index();
        // Same vector, different times
        index.insert(1, 1000, &[1.0, 0.0]);
        index.insert(2, 5000, &[0.99, 0.01]);
        index.insert(3, 100, &[0.0, 1.0]);

        let results = index.search(&[1.0, 0.0], 3, TemporalFilter::All, 1.0, 1000);
        // Pure semantic: [1.0, 0.0] is closest to itself, then [0.99, 0.01], then [0.0, 1.0]
        assert_eq!(results[0].0, 0); // entity 1
        assert_eq!(results[1].0, 1); // entity 2
        assert_eq!(results[2].0, 2); // entity 3
    }

    #[test]
    fn alpha_0_5_prefers_temporally_closer() {
        let mut index = make_temporal_index();
        // Two vectors equidistant semantically but at different timestamps
        index.insert(1, 1000, &[1.0, 0.0, 0.0]); // far in time
        index.insert(2, 5000, &[1.0, 0.0, 0.0]); // close in time

        let query_ts = 4900;
        let results = index.search(&[1.0, 0.0, 0.0], 2, TemporalFilter::All, 0.5, query_ts);

        // With alpha=0.5 and equal semantic distance, the temporally closer one wins
        assert_eq!(results[0].0, 1); // entity 2 at t=5000 is closer to query_ts=4900
        assert_eq!(results[1].0, 0); // entity 1 at t=1000 is farther
    }

    #[test]
    fn alpha_0_5_returns_temporally_closer_than_alpha_1() {
        let mut index = make_temporal_index();
        let dim = 8;
        let mut rng = rand::rng();

        // Insert 100 points at various timestamps
        for i in 0..100u64 {
            let ts = (i as i64) * 1000;
            let v: Vec<f32> = (0..dim)
                .map(|_| rand::Rng::random::<f32>(&mut rng))
                .collect();
            index.insert(i, ts, &v);
        }

        let query: Vec<f32> = (0..dim)
            .map(|_| rand::Rng::random::<f32>(&mut rng))
            .collect();
        let query_ts = 50_000; // middle of the range
        let k = 10;

        let results_pure = index.search(&query, k, TemporalFilter::All, 1.0, query_ts);
        let results_mixed = index.search(&query, k, TemporalFilter::All, 0.5, query_ts);

        // Average temporal distance of results
        let avg_tdist_pure: f64 = results_pure
            .iter()
            .map(|&(id, _)| (index.timestamp(id) - query_ts).unsigned_abs() as f64)
            .sum::<f64>()
            / k as f64;
        let avg_tdist_mixed: f64 = results_mixed
            .iter()
            .map(|&(id, _)| (index.timestamp(id) - query_ts).unsigned_abs() as f64)
            .sum::<f64>()
            / k as f64;

        assert!(
            avg_tdist_mixed <= avg_tdist_pure,
            "alpha=0.5 avg temporal dist ({avg_tdist_mixed:.0}) should be <= alpha=1.0 ({avg_tdist_pure:.0})"
        );
    }

    // ─── Alpha=1.0 parity with vanilla HNSW ────────────────────────────

    #[test]
    fn alpha_1_matches_vanilla_recall() {
        let dim = 32;
        let n = 1000u32;
        let k = 10;
        let config = HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            ..Default::default()
        };

        let mut temporal = TemporalHnsw::new(config, L2Distance);
        let mut rng = rand::rng();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| rand::Rng::random::<f32>(&mut rng))
                    .collect()
            })
            .collect();

        for (i, v) in vectors.iter().enumerate() {
            temporal.insert(i as u64, (i as i64) * 100, v);
        }

        // Compare temporal search (alpha=1.0, All) with brute force
        let n_queries = 50;
        let mut total_recall = 0.0;

        for _ in 0..n_queries {
            let query: Vec<f32> = (0..dim)
                .map(|_| rand::Rng::random::<f32>(&mut rng))
                .collect();
            let temporal_results = temporal.search(&query, k, TemporalFilter::All, 1.0, 0);
            let truth = temporal.graph().brute_force_knn(&query, k);
            let recall = super::super::recall_at_k(&temporal_results, &truth);
            total_recall += recall;
        }

        let avg_recall = total_recall / n_queries as f64;
        assert!(
            avg_recall >= 0.90,
            "alpha=1.0 recall = {avg_recall:.3}, expected >= 0.90 (vanilla parity)"
        );
    }

    // ─── Temporal filtering recall ──────────────────────────────────────

    #[test]
    fn range_knn_recall() {
        let dim = 32;
        let n = 1000u32;
        let k = 10;
        let config = HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 200,
            ..Default::default()
        };

        let mut index = TemporalHnsw::new(config, L2Distance);
        let mut rng = rand::rng();

        for i in 0..n {
            let ts = (i as i64) * 100;
            let v: Vec<f32> = (0..dim)
                .map(|_| rand::Rng::random::<f32>(&mut rng))
                .collect();
            index.insert(i as u64, ts, &v);
        }

        // Filter to middle 50% of timestamps
        let filter = TemporalFilter::Range(25_000, 75_000);
        let bitmap = index.build_filter_bitmap(&filter);

        let n_queries = 50;
        let mut total_recall = 0.0;

        for _ in 0..n_queries {
            let query: Vec<f32> = (0..dim)
                .map(|_| rand::Rng::random::<f32>(&mut rng))
                .collect();
            let results = index.search(&query, k, filter, 1.0, 50_000);

            // Brute-force ground truth within filter
            let mut truth: Vec<(u32, f32)> = (0..n)
                .filter(|&i| bitmap.contains(i))
                .map(|i| {
                    (
                        i,
                        index
                            .graph()
                            .brute_force_knn(&query, n as usize)
                            .iter()
                            .find(|&&(id, _)| id == i)
                            .map(|&(_, d)| d)
                            .unwrap_or(f32::INFINITY),
                    )
                })
                .collect();
            truth.sort_by(|a, b| a.1.total_cmp(&b.1));
            truth.truncate(k);

            total_recall += super::super::recall_at_k(&results, &truth);
        }

        let avg_recall = total_recall / n_queries as f64;
        assert!(
            avg_recall >= 0.90,
            "range kNN recall = {avg_recall:.3}, expected >= 0.90"
        );
    }

    // ─── Trajectory retrieval ───────────────────────────────────────────

    #[test]
    fn trajectory_returns_all_entity_points_ordered() {
        let mut index = make_temporal_index();

        // Insert 100 points for entity 1, interleaved with other entities
        for i in 0..100u32 {
            index.insert(1, (i as i64) * 1000, &[i as f32, 0.0]);
            index.insert(2, (i as i64) * 1000 + 500, &[0.0, i as f32]);
        }

        let traj = index.trajectory(1, TemporalFilter::All);
        assert_eq!(traj.len(), 100);

        // Verify ordering
        for window in traj.windows(2) {
            assert!(
                window[0].0 <= window[1].0,
                "trajectory not ordered: {} > {}",
                window[0].0,
                window[1].0
            );
        }

        // Verify all belong to entity 1
        for &(_, node_id) in &traj {
            assert_eq!(index.entity_id(node_id), 1);
        }
    }

    #[test]
    fn trajectory_with_range_filter() {
        let mut index = make_temporal_index();

        for i in 0..50u32 {
            index.insert(1, (i as i64) * 100, &[i as f32]);
        }

        let traj = index.trajectory(1, TemporalFilter::Range(1000, 3000));

        // timestamps 1000, 1100, ..., 3000 → 21 points
        assert_eq!(traj.len(), 21);
        for &(ts, _) in &traj {
            assert!(ts >= 1000 && ts <= 3000);
        }
    }

    #[test]
    fn trajectory_unknown_entity_returns_empty() {
        let mut index = make_temporal_index();
        index.insert(1, 1000, &[1.0]);
        assert!(index.trajectory(999, TemporalFilter::All).is_empty());
    }

    // ─── Roaring Bitmap memory ──────────────────────────────────────────

    #[test]
    fn bitmap_memory_under_1_byte_per_vector() {
        let mut index = make_temporal_index();

        // Insert 10K points
        for i in 0..10_000u32 {
            index.insert(i as u64, i as i64, &[i as f32]);
        }

        let mem = index.bitmap_memory_bytes();
        let bytes_per_vector = mem as f64 / 10_000.0;
        assert!(
            bytes_per_vector < 1.0,
            "bitmap uses {bytes_per_vector:.2} bytes/vector, expected < 1.0"
        );
    }

    // ─── Before/After filters ───────────────────────────────────────────

    #[test]
    fn before_filter() {
        let mut index = make_temporal_index();
        index.insert(1, 1000, &[1.0, 0.0]);
        index.insert(2, 2000, &[0.9, 0.1]);
        index.insert(3, 3000, &[0.8, 0.2]);

        let results = index.search(&[1.0, 0.0], 10, TemporalFilter::Before(2000), 1.0, 1000);
        assert_eq!(results.len(), 2);
        for &(id, _) in &results {
            assert!(index.timestamp(id) <= 2000);
        }
    }

    #[test]
    fn after_filter() {
        let mut index = make_temporal_index();
        index.insert(1, 1000, &[1.0, 0.0]);
        index.insert(2, 2000, &[0.9, 0.1]);
        index.insert(3, 3000, &[0.8, 0.2]);

        let results = index.search(&[1.0, 0.0], 10, TemporalFilter::After(2000), 1.0, 3000);
        assert_eq!(results.len(), 2);
        for &(id, _) in &results {
            assert!(index.timestamp(id) >= 2000);
        }
    }

    // ─── Cosine metric works ────────────────────────────────────────────

    #[test]
    fn works_with_cosine_metric() {
        let config = HnswConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 50,
            ..Default::default()
        };
        let mut index = TemporalHnsw::new(config, CosineDistance);

        index.insert(1, 1000, &[1.0, 0.0, 0.0]);
        index.insert(2, 2000, &[0.99, 0.01, 0.0]);
        index.insert(3, 3000, &[0.0, 0.0, 1.0]);

        let results = index.search(&[1.0, 0.0, 0.0], 2, TemporalFilter::All, 1.0, 0);
        assert_eq!(results[0].0, 0);
        assert_eq!(results[1].0, 1);
    }
}
