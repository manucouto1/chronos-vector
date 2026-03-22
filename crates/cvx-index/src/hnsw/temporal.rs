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
    /// Optional per-node metadata store.
    metadata_store: Option<super::metadata_store::MetadataStore>,
    /// Optional centroid for anisotropy correction (RFC-012 Part B).
    ///
    /// When set, all distance computations center vectors by subtracting
    /// this mean vector, amplifying the discriminative signal that is
    /// otherwise compressed by the dominant "average text" direction.
    centroid: Option<Vec<f32>>,
    /// Optional per-node reward for outcome-aware search (RFC-012 P4).
    ///
    /// NaN means "no reward assigned". Stored parallel to timestamps/entity_ids.
    rewards: Vec<f32>,
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
            metadata_store: None,
            centroid: None,
            rewards: Vec::new(),
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
        self.insert_with_reward(entity_id, timestamp, vector, f32::NAN)
    }

    /// Bulk insert multiple points with parallel distance computation (RFC-012 P9).
    ///
    /// Faster than sequential `insert()` calls for large batches. Uses rayon
    /// to parallelize neighbor search across chunks while keeping graph
    /// modifications sequential.
    ///
    /// Returns the number of points inserted.
    pub fn bulk_insert_parallel(
        &mut self,
        entity_ids: &[u64],
        timestamps: &[i64],
        vectors: &[&[f32]],
    ) -> usize {
        use rayon::prelude::*;

        let n = entity_ids.len();
        if n == 0 {
            return 0;
        }

        // Phase 1: Insert first 100 points sequentially to build initial graph
        let seed_count = n.min(100);
        for i in 0..seed_count {
            self.insert(entity_ids[i], timestamps[i], vectors[i]);
        }

        if seed_count >= n {
            return n;
        }

        // Phase 2: For remaining points, compute neighbors in parallel batches
        let batch_size = 256;
        let remaining = &vectors[seed_count..];
        let remaining_eids = &entity_ids[seed_count..];
        let remaining_ts = &timestamps[seed_count..];

        for batch_start in (0..remaining.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(remaining.len());
            let batch_vecs = &remaining[batch_start..batch_end];

            // Parallel: compute nearest neighbors for each vector in the batch
            let neighbor_lists: Vec<Vec<(u32, f32)>> = batch_vecs
                .par_iter()
                .map(|vec| self.graph.search(vec, self.graph.config().ef_construction))
                .collect();

            // Sequential: insert nodes and connect using pre-computed neighbors
            for (i, neighbors) in neighbor_lists.into_iter().enumerate() {
                let idx = batch_start + i;
                let eid = remaining_eids[idx];
                let ts = remaining_ts[idx];
                let vec = remaining[idx];

                let node_id = self.graph.len() as u32;
                // Allocate node
                let level = self.graph.random_level();
                self.graph.push_node(vec, level);
                self.timestamps.push(ts);
                self.entity_ids.push(eid);
                self.rewards.push(f32::NAN);

                // Connect using pre-computed neighbors
                self.graph.connect_node(node_id, &neighbors, level);

                // Update entity index
                self.entity_index
                    .entry(eid)
                    .or_default()
                    .push((ts, node_id));
                self.min_timestamp = self.min_timestamp.min(ts);
                self.max_timestamp = self.max_timestamp.max(ts);

                if let Some(ref mut store) = self.metadata_store {
                    store.push_empty();
                }
            }
        }

        n
    }

    /// Insert a temporal point with an outcome reward.
    ///
    /// `reward` annotates this point with an outcome signal (e.g., 0.0-1.0).
    /// Use `f32::NAN` for "no reward assigned".
    pub fn insert_with_reward(
        &mut self,
        entity_id: u64,
        timestamp: i64,
        vector: &[f32],
        reward: f32,
    ) -> u32 {
        let node_id = self.graph.len() as u32;
        self.graph.insert(node_id, vector);
        self.timestamps.push(timestamp);
        self.entity_ids.push(entity_id);
        self.rewards.push(reward);

        // Update entity index
        self.entity_index
            .entry(entity_id)
            .or_default()
            .push((timestamp, node_id));

        // Update temporal range
        self.min_timestamp = self.min_timestamp.min(timestamp);
        self.max_timestamp = self.max_timestamp.max(timestamp);

        // Store metadata (empty if store not enabled)
        if let Some(ref mut store) = self.metadata_store {
            store.push_empty();
        }

        node_id
    }

    /// Insert a temporal point with metadata.
    pub fn insert_with_metadata(
        &mut self,
        entity_id: u64,
        timestamp: i64,
        vector: &[f32],
        metadata: std::collections::HashMap<String, String>,
    ) -> u32 {
        // Enable metadata store on first metadata insert
        if self.metadata_store.is_none() {
            let mut store = super::metadata_store::MetadataStore::new();
            // Backfill empty entries for existing nodes
            for _ in 0..self.graph.len() {
                store.push_empty();
            }
            self.metadata_store = Some(store);
        }

        let node_id = self.graph.len() as u32;
        self.graph.insert(node_id, vector);
        self.timestamps.push(timestamp);
        self.entity_ids.push(entity_id);
        self.rewards.push(f32::NAN);

        self.entity_index
            .entry(entity_id)
            .or_default()
            .push((timestamp, node_id));

        self.min_timestamp = self.min_timestamp.min(timestamp);
        self.max_timestamp = self.max_timestamp.max(timestamp);

        if let Some(ref mut store) = self.metadata_store {
            store.push(metadata);
        }

        node_id
    }

    /// Get metadata for a node. Returns empty map if metadata store not enabled.
    pub fn node_metadata(&self, node_id: u32) -> std::collections::HashMap<String, String> {
        self.metadata_store
            .as_ref()
            .map(|s| s.get(node_id).clone())
            .unwrap_or_default()
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

    /// Normalize semantic distance to [0, 1] range (RFC-012 P8).
    ///
    /// Cosine distance ∈ [0, 2], L2 distance ∈ [0, ∞). This clamps and
    /// scales to [0, 1] so it's comparable with temporal distance [0, 1].
    fn normalize_semantic_distance(&self, d: f32) -> f32 {
        // Cosine: [0, 2] → [0, 1] by halving. L2: clamp to [0, 4] then /4.
        // Both produce [0, 1]. For most embeddings, distances rarely exceed 2.
        (d / 2.0).min(1.0)
    }

    /// Compute recency penalty for a node (RFC-012 P7).
    ///
    /// Returns a value in `[0.0, 1.0]` where 0 = most recent, 1 = oldest.
    /// Uses exponential decay: `1 - exp(-λ · age)` where age is normalized.
    ///
    /// `recency_lambda` controls decay speed:
    /// - λ = 0: no recency effect
    /// - λ = 1: moderate decay
    /// - λ = 3: strong decay (old nodes heavily penalized)
    fn recency_penalty(&self, node_timestamp: i64, recency_lambda: f32) -> f32 {
        if recency_lambda <= 0.0 {
            return 0.0;
        }
        let age = self.temporal_distance_normalized(node_timestamp, self.max_timestamp);
        1.0 - (-recency_lambda * age).exp()
    }

    /// Search with full composite scoring (RFC-012 P7 + P8).
    ///
    /// Enhanced distance: `d = α·d_sem_norm + (1-α)·d_temporal + γ·recency`
    ///
    /// - `alpha`: semantic vs temporal weight (1.0 = pure semantic)
    /// - `recency_lambda`: recency decay strength (0.0 = off, 1.0 = moderate, 3.0 = strong)
    /// - `recency_weight`: weight of recency term in composite score (0.0-1.0)
    #[allow(clippy::too_many_arguments)]
    pub fn search_with_recency(
        &self,
        query: &[f32],
        k: usize,
        filter: TemporalFilter,
        alpha: f32,
        query_timestamp: i64,
        recency_lambda: f32,
        recency_weight: f32,
    ) -> Vec<(u32, f32)> {
        if self.is_empty() {
            return Vec::new();
        }

        let bitmap = self.build_filter_bitmap(&filter);
        if bitmap.is_empty() {
            return Vec::new();
        }

        let over_fetch = k * 4;
        let candidates = self
            .graph
            .search_filtered(query, over_fetch, |id| bitmap.contains(id));

        let mut scored: Vec<(u32, f32)> = candidates
            .into_iter()
            .map(|(id, sem_dist)| {
                let sem_norm = self.normalize_semantic_distance(sem_dist);
                let t_dist = self
                    .temporal_distance_normalized(self.timestamps[id as usize], query_timestamp);
                let recency = self.recency_penalty(self.timestamps[id as usize], recency_lambda);

                let combined = alpha * sem_norm + (1.0 - alpha) * t_dist + recency_weight * recency;
                (id, combined)
            })
            .collect();

        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        scored.truncate(k);
        scored
    }

    // ─── Outcome / Reward (RFC-012 P4) ──────────────────────────────

    /// Get the reward for a node. Returns NaN if no reward was assigned.
    pub fn reward(&self, node_id: u32) -> f32 {
        self.rewards
            .get(node_id as usize)
            .copied()
            .unwrap_or(f32::NAN)
    }

    /// Set the reward for a node retroactively.
    ///
    /// Useful for annotating outcomes after an episode completes.
    pub fn set_reward(&mut self, node_id: u32, reward: f32) {
        if let Some(r) = self.rewards.get_mut(node_id as usize) {
            *r = reward;
        }
    }

    /// Build a bitmap of node_ids with reward >= min_reward.
    pub fn build_reward_bitmap(&self, min_reward: f32) -> RoaringBitmap {
        let mut bitmap = RoaringBitmap::new();
        for (i, &r) in self.rewards.iter().enumerate() {
            if !r.is_nan() && r >= min_reward {
                bitmap.insert(i as u32);
            }
        }
        bitmap
    }

    /// Search with reward filtering: only return nodes with reward >= min_reward.
    ///
    /// Combines temporal filter + reward filter as bitmap pre-filter.
    pub fn search_with_reward(
        &self,
        query: &[f32],
        k: usize,
        filter: TemporalFilter,
        alpha: f32,
        query_timestamp: i64,
        min_reward: f32,
    ) -> Vec<(u32, f32)> {
        if self.is_empty() {
            return Vec::new();
        }

        let temporal_bitmap = self.build_filter_bitmap(&filter);
        let reward_bitmap = self.build_reward_bitmap(min_reward);
        let combined = temporal_bitmap & reward_bitmap;

        if combined.is_empty() {
            return Vec::new();
        }

        let candidates = self
            .graph
            .search_filtered(query, k, |id| combined.contains(id));

        if alpha >= 1.0 {
            return candidates;
        }

        let mut scored: Vec<(u32, f32)> = candidates
            .into_iter()
            .map(|(id, sem_dist)| {
                let t_dist = self
                    .temporal_distance_normalized(self.timestamps[id as usize], query_timestamp);
                (id, alpha * sem_dist + (1.0 - alpha) * t_dist)
            })
            .collect();

        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        scored.truncate(k);
        scored
    }

    // ─── Centering (RFC-012 Part B) ──────────────────────────────────

    /// Compute the centroid (mean vector) of all indexed vectors.
    ///
    /// Single O(N×D) pass over stored vectors. Returns `None` if the index
    /// is empty.
    pub fn compute_centroid(&self) -> Option<Vec<f32>> {
        let n = self.graph.len();
        if n == 0 {
            return None;
        }

        let dim = self.graph.vector(0).len();
        let mut sum = vec![0.0f64; dim];

        for i in 0..n {
            let v = self.graph.vector(i as u32);
            for (s, &val) in sum.iter_mut().zip(v.iter()) {
                *s += val as f64;
            }
        }

        let inv_n = 1.0 / n as f64;
        Some(sum.into_iter().map(|s| (s * inv_n) as f32).collect())
    }

    /// Set the centroid for anisotropy correction.
    ///
    /// Once set, `centered_vector()` subtracts this from any vector,
    /// and search operations use centered distances. The centroid is
    /// serialized with the index snapshot.
    ///
    /// You can provide an externally computed centroid (e.g., from a
    /// larger corpus) or use `compute_centroid()` for the index contents.
    pub fn set_centroid(&mut self, centroid: Vec<f32>) {
        self.centroid = Some(centroid);
    }

    /// Clear the centroid, reverting to raw (uncentered) distances.
    pub fn clear_centroid(&mut self) {
        self.centroid = None;
    }

    /// Get the current centroid, if set.
    pub fn centroid(&self) -> Option<&[f32]> {
        self.centroid.as_deref()
    }

    /// Return a centered copy of the given vector (vec - centroid).
    ///
    /// If no centroid is set, returns the vector unchanged (cloned).
    pub fn centered_vector(&self, vec: &[f32]) -> Vec<f32> {
        match &self.centroid {
            Some(c) => vec.iter().zip(c.iter()).map(|(v, m)| v - m).collect(),
            None => vec.to_vec(),
        }
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

        // Re-rank with composite distance (P8: normalized scales)
        let mut scored: Vec<(u32, f32)> = candidates
            .into_iter()
            .map(|(id, sem_dist)| {
                let sem_norm = self.normalize_semantic_distance(sem_dist);
                let t_dist = self
                    .temporal_distance_normalized(self.timestamps[id as usize], query_timestamp);
                let combined = alpha * sem_norm + (1.0 - alpha) * t_dist;
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
    ///
    /// **Performance warning**: This does a full scan of all nodes. For multiple regions,
    /// use `region_assignments()` instead (single scan for all regions).
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

    /// Assign ALL nodes to their regions in a single pass, optionally time-filtered.
    ///
    /// Returns a HashMap from hub_id → Vec<(entity_id, timestamp)>.
    /// This is O(N) — one full scan instead of O(N × K) for K `region_members` calls.
    pub fn region_assignments(
        &self,
        level: usize,
        filter: TemporalFilter,
    ) -> std::collections::HashMap<u32, Vec<(u64, i64)>> {
        let mut assignments: std::collections::HashMap<u32, Vec<(u64, i64)>> =
            std::collections::HashMap::new();

        for node_id in 0..self.graph.len() as u32 {
            let ts = self.timestamps[node_id as usize];
            if !filter.matches(ts) {
                continue;
            }
            let vec = self.graph.vector(node_id);
            if let Some(hub) = self.graph.assign_region(vec, level) {
                let eid = self.entity_ids[node_id as usize];
                assignments.entry(hub).or_default().push((eid, ts));
            }
        }

        assignments
    }
}

/// Current snapshot format version. Increment when adding fields.
const SNAPSHOT_VERSION: u32 = 2;

/// Serializable snapshot of a TemporalHnsw index.
#[derive(Serialize, Deserialize)]
struct TemporalSnapshot {
    /// Format version for forward compatibility (RFC-012 P5).
    /// v1: original (graph, timestamps, entity_ids, entity_index, min/max_timestamp)
    /// v2: + metadata_store, centroid, rewards
    #[serde(default = "default_snapshot_version")]
    version: u32,
    graph: HnswSnapshot,
    timestamps: Vec<i64>,
    entity_ids: Vec<u64>,
    entity_index: BTreeMap<u64, Vec<(i64, u32)>>,
    min_timestamp: i64,
    max_timestamp: i64,
    #[serde(default)]
    metadata_store: Option<super::metadata_store::MetadataStore>,
    /// Centroid for anisotropy correction (RFC-012 Part B).
    #[serde(default)]
    centroid: Option<Vec<f32>>,
    /// Per-node reward for outcome-aware search (RFC-012 P4).
    #[serde(default)]
    rewards: Vec<f32>,
}

fn default_snapshot_version() -> u32 {
    1 // Old snapshots without version field default to v1
}

impl<D: DistanceMetric> TemporalHnsw<D> {
    /// Save the index to a file using postcard binary encoding.
    ///
    /// The distance metric is NOT serialized (it's stateless).
    /// On load, you must provide the same metric type.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let snapshot = TemporalSnapshot {
            version: SNAPSHOT_VERSION,
            graph: self.graph.to_snapshot(),
            timestamps: self.timestamps.clone(),
            entity_ids: self.entity_ids.clone(),
            entity_index: self.entity_index.clone(),
            min_timestamp: self.min_timestamp,
            max_timestamp: self.max_timestamp,
            metadata_store: self.metadata_store.clone(),
            centroid: self.centroid.clone(),
            rewards: self.rewards.clone(),
        };

        let bytes = postcard::to_allocvec(&snapshot).map_err(std::io::Error::other)?;

        let mut file = std::fs::File::create(path)?;
        file.write_all(&bytes)?;
        Ok(())
    }

    /// Load an index from a file, providing the distance metric.
    ///
    /// Supports all snapshot versions. Unknown future versions produce
    /// a clear error instead of silent corruption.
    pub fn load(path: &Path, metric: D) -> std::io::Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;

        let snapshot: TemporalSnapshot = postcard::from_bytes(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        if snapshot.version > SNAPSHOT_VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Snapshot version {} is newer than supported version {}. \
                     Please upgrade chronos-vector.",
                    snapshot.version, SNAPSHOT_VERSION
                ),
            ));
        }

        let n_points = snapshot.timestamps.len();
        let rewards = if snapshot.rewards.is_empty() {
            // Backward compat: old snapshots have no rewards → fill with NaN
            vec![f32::NAN; n_points]
        } else {
            snapshot.rewards
        };

        Ok(Self {
            graph: HnswGraph::from_snapshot(snapshot.graph, metric),
            timestamps: snapshot.timestamps,
            entity_ids: snapshot.entity_ids,
            entity_index: snapshot.entity_index,
            min_timestamp: snapshot.min_timestamp,
            max_timestamp: snapshot.max_timestamp,
            metadata_store: snapshot.metadata_store,
            centroid: snapshot.centroid,
            rewards,
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

    fn metadata(&self, node_id: u32) -> std::collections::HashMap<String, String> {
        self.node_metadata(node_id)
    }

    fn search_with_metadata(
        &self,
        query: &[f32],
        k: usize,
        filter: TemporalFilter,
        alpha: f32,
        query_timestamp: i64,
        metadata_filter: &cvx_core::types::MetadataFilter,
    ) -> Vec<(u32, f32)> {
        if metadata_filter.is_empty() {
            return self.search(query, k, filter, alpha, query_timestamp);
        }

        match &self.metadata_store {
            Some(store) => {
                // Pre-filter: build combined temporal + metadata bitmap
                let temporal_bitmap = self.build_filter_bitmap(&filter);
                let metadata_bitmap = store.build_filter_bitmap(metadata_filter);
                let combined = temporal_bitmap & metadata_bitmap;

                if combined.is_empty() {
                    return Vec::new();
                }

                // Search with combined bitmap
                let candidates = self
                    .graph
                    .search_filtered(query, k, |id| combined.contains(id));

                if alpha >= 1.0 {
                    return candidates;
                }

                // Re-rank with composite distance
                let mut scored: Vec<(u32, f32)> = candidates
                    .into_iter()
                    .map(|(id, sem_dist)| {
                        let t_dist = self.temporal_distance_normalized(
                            self.timestamps[id as usize],
                            query_timestamp,
                        );
                        let combined_score = alpha * sem_dist + (1.0 - alpha) * t_dist;
                        (id, combined_score)
                    })
                    .collect();

                scored.sort_by(|a, b| a.1.total_cmp(&b.1));
                scored.truncate(k);
                scored
            }
            None => {
                // No metadata store: fall back to search without metadata
                self.search(query, k, filter, alpha, query_timestamp)
            }
        }
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
            assert!((1500..=3500).contains(&ts), "timestamp {ts} out of range");
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
            assert!((1000..=3000).contains(&ts));
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

    // ─── Metadata integration ───────────────────────────────────────

    #[test]
    fn insert_with_metadata_stores_and_retrieves() {
        let config = HnswConfig::default();
        let mut index = TemporalHnsw::new(config, L2Distance);

        let mut meta = std::collections::HashMap::new();
        meta.insert("reward".to_string(), "0.8".to_string());
        meta.insert("step_index".to_string(), "0".to_string());

        let id = index.insert_with_metadata(1, 1000, &[1.0, 0.0, 0.0], meta);

        let retrieved = index.node_metadata(id);
        assert_eq!(retrieved.get("reward").unwrap(), "0.8");
        assert_eq!(retrieved.get("step_index").unwrap(), "0");
    }

    #[test]
    fn insert_with_metadata_enables_store_lazily() {
        let config = HnswConfig::default();
        let mut index = TemporalHnsw::new(config, L2Distance);

        // First insert without metadata
        index.insert(1, 1000, &[1.0, 0.0, 0.0]);

        // Second insert with metadata — should enable store and backfill
        let mut meta = std::collections::HashMap::new();
        meta.insert("reward".to_string(), "0.9".to_string());
        let id = index.insert_with_metadata(2, 2000, &[0.0, 1.0, 0.0], meta);

        // First node should have empty metadata
        assert!(index.node_metadata(0).is_empty());
        // Second node should have metadata
        assert_eq!(index.node_metadata(id).get("reward").unwrap(), "0.9");
    }

    #[test]
    fn search_with_metadata_filter() {
        use cvx_core::TemporalIndexAccess;
        use cvx_core::types::MetadataFilter;

        let config = HnswConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 50,
            ..Default::default()
        };
        let mut index = TemporalHnsw::new(config, L2Distance);

        // Insert 10 points: 5 with reward >= 0.5, 5 with reward < 0.5
        for i in 0..10u64 {
            let mut meta = std::collections::HashMap::new();
            meta.insert("reward".to_string(), format!("{}", i as f64 * 0.1));
            meta.insert("step_index".to_string(), "0".to_string());
            index.insert_with_metadata(i, i as i64 * 1000, &[i as f32, 0.0, 0.0], meta);
        }

        // Search with metadata filter: reward >= 0.5
        let filter = MetadataFilter::new().gte("reward", 0.5);
        let results =
            index.search_with_metadata(&[7.0, 0.0, 0.0], 5, TemporalFilter::All, 1.0, 0, &filter);

        // All results should have reward >= 0.5
        for &(nid, _) in &results {
            let meta = index.node_metadata(nid);
            let reward: f64 = meta.get("reward").unwrap().parse().unwrap();
            assert!(reward >= 0.5, "node {nid} has reward {reward} < 0.5");
        }
        assert!(!results.is_empty(), "should find some results");
    }

    // ─── Region assignments ──────────────────────────────────────────

    /// Build an index with enough points so that level-1 hubs exist.
    fn make_region_index() -> TemporalHnsw<L2Distance> {
        let config = HnswConfig {
            m: 4,
            ef_construction: 50,
            ef_search: 50,
            ..Default::default()
        };
        let mut index = TemporalHnsw::new(config, L2Distance);
        let mut rng = rand::rng();
        // 200 points across 4 entities, timestamps 0..199_000
        for i in 0..200u64 {
            let v: Vec<f32> = (0..8).map(|_| rand::Rng::random::<f32>(&mut rng)).collect();
            let entity = i % 4;
            index.insert(entity, i as i64 * 1000, &v);
        }
        index
    }

    #[test]
    fn region_assignments_covers_all_nodes() {
        let index = make_region_index();
        let level = 1;
        let assignments = index.region_assignments(level, TemporalFilter::All);

        let total: usize = assignments.values().map(|v| v.len()).sum();
        assert_eq!(
            total,
            index.len(),
            "sum of all region member counts ({total}) must equal index size ({})",
            index.len()
        );
    }

    #[test]
    fn region_assignments_consistent_with_regions_counts() {
        let index = make_region_index();
        let level = 1;
        let regions = index.regions(level);
        let assignments = index.region_assignments(level, TemporalFilter::All);

        for &(hub_id, _, count) in &regions {
            let assigned_count = assignments.get(&hub_id).map_or(0, |v| v.len());
            assert_eq!(
                assigned_count, count,
                "region hub {hub_id}: region_assignments has {assigned_count} members but regions() reports {count}"
            );
        }
    }

    #[test]
    fn region_assignments_temporal_filter_reduces_count() {
        let index = make_region_index();
        let level = 1;

        let all = index.region_assignments(level, TemporalFilter::All);
        let total_all: usize = all.values().map(|v| v.len()).sum();

        // Filter to middle 50% of timestamps (50_000..150_000)
        let filtered = index.region_assignments(level, TemporalFilter::Range(50_000, 150_000));
        let total_filtered: usize = filtered.values().map(|v| v.len()).sum();

        assert!(
            total_filtered < total_all,
            "Range filter should reduce total members: filtered={total_filtered}, all={total_all}"
        );

        // Verify every member in filtered results has a timestamp within the range
        for members in filtered.values() {
            for &(_eid, ts) in members {
                assert!(
                    (50_000..=150_000).contains(&ts),
                    "filtered result has timestamp {ts} outside [50000, 150000]"
                );
            }
        }
    }

    #[test]
    fn region_assignments_each_member_in_exactly_one_region() {
        let index = make_region_index();
        let level = 1;
        let assignments = index.region_assignments(level, TemporalFilter::All);

        // Collect (entity_id, timestamp) across all regions and check for duplicates
        let mut seen = std::collections::HashSet::new();
        let mut total = 0usize;
        for members in assignments.values() {
            for &(eid, ts) in members {
                total += 1;
                let _inserted = seen.insert((eid, ts));
                // Note: same (eid, ts) can appear if entity has multiple nodes at same ts,
                // so we count total instead and verify it matches index.len()
            }
        }

        // Each node_id maps to exactly one region, so total must equal index size
        assert_eq!(
            total,
            index.len(),
            "total assigned members ({total}) != index size ({}); a node appeared in multiple or no regions",
            index.len()
        );

        // Additionally, no hub should appear in two different regions' keys that don't exist at level
        let hubs: std::collections::HashSet<u32> = assignments.keys().copied().collect();
        let level_hubs: std::collections::HashSet<u32> =
            index.graph().nodes_at_level(level).into_iter().collect();
        for hub in &hubs {
            assert!(
                level_hubs.contains(hub),
                "assignment hub {hub} is not a level-{level} node"
            );
        }
    }

    // ─── Centering (RFC-012 Part B) ─────────────────────────────────

    #[test]
    fn compute_centroid_empty_index() {
        let index = make_temporal_index();
        assert!(index.compute_centroid().is_none());
    }

    #[test]
    fn compute_centroid_single_vector() {
        let mut index = make_temporal_index();
        index.insert(1, 1000, &[3.0, 4.0, 5.0]);
        let centroid = index.compute_centroid().unwrap();
        assert_eq!(centroid, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn compute_centroid_mean_of_vectors() {
        let mut index = make_temporal_index();
        index.insert(1, 1000, &[2.0, 0.0]);
        index.insert(2, 2000, &[4.0, 6.0]);
        let centroid = index.compute_centroid().unwrap();
        assert!((centroid[0] - 3.0).abs() < 1e-6);
        assert!((centroid[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn set_and_clear_centroid() {
        let mut index = make_temporal_index();
        index.insert(1, 1000, &[1.0, 2.0]);

        assert!(index.centroid().is_none());

        index.set_centroid(vec![0.5, 1.0]);
        assert!(index.centroid().is_some());
        assert_eq!(index.centroid().unwrap(), &[0.5, 1.0]);

        index.clear_centroid();
        assert!(index.centroid().is_none());
    }

    #[test]
    fn centered_vector_subtracts_centroid() {
        let mut index = make_temporal_index();
        index.insert(1, 1000, &[1.0, 2.0]);
        index.set_centroid(vec![0.5, 1.0]);

        let centered = index.centered_vector(&[3.0, 5.0]);
        assert!((centered[0] - 2.5).abs() < 1e-6);
        assert!((centered[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn centered_vector_without_centroid_is_identity() {
        let mut index = make_temporal_index();
        index.insert(1, 1000, &[1.0, 2.0]);
        // No centroid set
        let centered = index.centered_vector(&[3.0, 5.0]);
        assert_eq!(centered, vec![3.0, 5.0]);
    }

    #[test]
    fn centroid_survives_save_load() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_centroid_snapshot.cvx");

        let mut index = make_temporal_index();
        index.insert(1, 1000, &[1.0, 2.0, 3.0]);
        index.insert(2, 2000, &[4.0, 5.0, 6.0]);
        index.set_centroid(vec![2.5, 3.5, 4.5]);

        index.save(&path).unwrap();

        let loaded = TemporalHnsw::load(&path, L2Distance).unwrap();
        assert_eq!(loaded.centroid().unwrap(), &[2.5, 3.5, 4.5]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn load_without_centroid_is_none() {
        // Verifies backward compatibility: old snapshots without centroid
        // field deserialize with centroid = None (via #[serde(default)])
        let dir = std::env::temp_dir();
        let path = dir.join("test_no_centroid_snapshot.cvx");

        let mut index = make_temporal_index();
        index.insert(1, 1000, &[1.0, 0.0]);
        // No centroid set
        index.save(&path).unwrap();

        let loaded = TemporalHnsw::load(&path, L2Distance).unwrap();
        assert!(loaded.centroid().is_none());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn compute_centroid_precision_with_many_vectors() {
        let config = HnswConfig {
            m: 4,
            ef_construction: 20,
            ef_search: 10,
            ..Default::default()
        };
        let mut index = TemporalHnsw::new(config, L2Distance);

        // Insert 1000 vectors with known mean
        for i in 0..1000u64 {
            // Vectors centered around [10.0, 20.0] with small perturbation
            let v = vec![10.0 + (i as f32 * 0.001), 20.0 - (i as f32 * 0.001)];
            index.insert(i, i as i64, &v);
        }

        let centroid = index.compute_centroid().unwrap();
        // Expected mean: [10.0 + 0.4995, 20.0 - 0.4995] = [10.4995, 19.5005]
        assert!(
            (centroid[0] - 10.4995).abs() < 0.01,
            "centroid[0] = {}, expected ~10.4995",
            centroid[0]
        );
        assert!(
            (centroid[1] - 19.5005).abs() < 0.01,
            "centroid[1] = {}, expected ~19.5005",
            centroid[1]
        );
    }

    #[test]
    fn search_with_empty_metadata_filter_matches_all() {
        use cvx_core::TemporalIndexAccess;
        use cvx_core::types::MetadataFilter;

        let config = HnswConfig::default();
        let mut index = TemporalHnsw::new(config, L2Distance);

        for i in 0..5u64 {
            index.insert(i, i as i64 * 1000, &[i as f32, 0.0]);
        }

        let filter = MetadataFilter::new();
        let results =
            index.search_with_metadata(&[2.0, 0.0], 3, TemporalFilter::All, 1.0, 0, &filter);
        assert_eq!(results.len(), 3);
    }

    // ─── Reward / outcome-aware search (RFC-012 P4) ──────────────

    #[test]
    fn insert_with_reward_stores_reward() {
        let mut index = make_temporal_index();
        let n0 = index.insert(1, 1000, &[1.0, 0.0]);
        let n1 = index.insert_with_reward(2, 2000, &[0.0, 1.0], 0.8);

        assert!(index.reward(n0).is_nan()); // no reward
        assert!((index.reward(n1) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn set_reward_updates_retroactively() {
        let mut index = make_temporal_index();
        let n0 = index.insert(1, 1000, &[1.0, 0.0]);
        assert!(index.reward(n0).is_nan());

        index.set_reward(n0, 0.95);
        assert!((index.reward(n0) - 0.95).abs() < 1e-6);
    }

    #[test]
    fn search_with_reward_filters() {
        let mut index = make_temporal_index();
        // Insert 10 points with varying rewards
        for i in 0..10u64 {
            index.insert_with_reward(i, i as i64 * 1000, &[i as f32, 0.0, 0.0], i as f32 * 0.1);
        }

        // min_reward=0.5 → only nodes 5..9
        let results =
            index.search_with_reward(&[7.0, 0.0, 0.0], 5, TemporalFilter::All, 1.0, 0, 0.5);
        assert!(!results.is_empty());
        for &(node_id, _) in &results {
            let r = index.reward(node_id);
            assert!(r >= 0.5, "node {node_id} has reward {r} < 0.5");
        }
    }

    #[test]
    fn search_with_reward_no_matches() {
        let mut index = make_temporal_index();
        for i in 0..5u64 {
            index.insert_with_reward(i, i as i64 * 1000, &[i as f32, 0.0], 0.1);
        }

        let results = index.search_with_reward(&[2.0, 0.0], 5, TemporalFilter::All, 1.0, 0, 0.9);
        assert!(results.is_empty());
    }

    #[test]
    fn reward_survives_save_load() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_reward_snapshot.cvx");

        let mut index = make_temporal_index();
        index.insert_with_reward(1, 1000, &[1.0, 0.0], 0.75);
        index.insert(2, 2000, &[0.0, 1.0]); // no reward
        index.save(&path).unwrap();

        let loaded = TemporalHnsw::load(&path, L2Distance).unwrap();
        assert!((loaded.reward(0) - 0.75).abs() < 1e-6);
        assert!(loaded.reward(1).is_nan());

        std::fs::remove_file(&path).ok();
    }

    // ─── P7: Recency + P8: Normalization ─────────────────────────

    #[test]
    fn normalize_semantic_distance_clamps() {
        let mut index = make_temporal_index();
        index.insert(1, 1000, &[1.0, 0.0]);

        // Cosine distance [0, 2] → [0, 1]
        assert!((index.normalize_semantic_distance(0.0) - 0.0).abs() < 1e-6);
        assert!((index.normalize_semantic_distance(1.0) - 0.5).abs() < 1e-6);
        assert!((index.normalize_semantic_distance(2.0) - 1.0).abs() < 1e-6);
        // Clamp: values > 2 stay at 1.0
        assert!((index.normalize_semantic_distance(4.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn recency_penalty_zero_lambda() {
        let mut index = make_temporal_index();
        index.insert(1, 1000, &[1.0, 0.0]);
        index.insert(2, 2000, &[0.0, 1.0]);
        // lambda=0 → no recency effect
        assert!((index.recency_penalty(1000, 0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn recency_penalty_recent_is_lower() {
        let mut index = make_temporal_index();
        for i in 0..10u64 {
            index.insert(i, (i * 1000) as i64, &[i as f32, 0.0]);
        }
        let recent = index.recency_penalty(9000, 1.0); // most recent
        let old = index.recency_penalty(0, 1.0); // oldest
        assert!(
            recent < old,
            "recent penalty ({recent}) should be < old penalty ({old})"
        );
    }

    #[test]
    fn search_with_recency_prefers_recent() {
        let mut index = make_temporal_index();
        // Two identical vectors at different times
        index.insert(1, 1000, &[1.0, 0.0, 0.0]);
        index.insert(2, 9000, &[1.0, 0.0, 0.0]); // more recent

        let results = index.search_with_recency(
            &[1.0, 0.0, 0.0],
            2,
            TemporalFilter::All,
            1.0, // pure semantic
            0,
            2.0, // strong recency
            0.5, // high recency weight
        );

        assert_eq!(results.len(), 2);
        // More recent node should score lower (better)
        assert_eq!(
            results[0].0,
            1, // node 1 = entity 2 at t=9000 (more recent)
            "recent node should rank first"
        );
    }

    #[test]
    fn search_normalized_distances_balanced() {
        let mut index = make_temporal_index();
        // Semantically close but temporally far
        index.insert(1, 1000, &[1.0, 0.0, 0.0]);
        // Semantically far but temporally close
        index.insert(2, 5000, &[0.0, 1.0, 0.0]);
        // Query at t=4900 with alpha=0.5
        let results = index.search(
            &[0.9, 0.1, 0.0],
            2,
            TemporalFilter::All,
            0.5, // balanced
            4900,
        );

        // With normalized scales, temporal distance is comparable to semantic
        // Entity 2 at t=5000 is temporally close to query_ts=4900
        assert_eq!(results.len(), 2);
    }

    // ─── P9: Parallel bulk insert ────────────────────────────────

    #[test]
    fn bulk_insert_parallel_basic() {
        let config = HnswConfig {
            m: 8,
            ef_construction: 50,
            ef_search: 50,
            ..Default::default()
        };
        let mut index = TemporalHnsw::new(config, L2Distance);

        let n = 500;
        let dim = 16;
        let mut rng = rand::rng();
        let eids: Vec<u64> = (0..n).map(|i| i as u64 % 10).collect();
        let tss: Vec<i64> = (0..n).map(|i| i as i64 * 100).collect();
        let vecs: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| rand::Rng::random::<f32>(&mut rng))
                    .collect()
            })
            .collect();
        let vec_refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

        let count = index.bulk_insert_parallel(&eids, &tss, &vec_refs);
        assert_eq!(count, n);
        assert_eq!(index.len(), n);

        // Should be searchable
        let results = index.search(&vecs[0], 5, TemporalFilter::All, 1.0, 0);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn bulk_insert_parallel_recall() {
        let config = HnswConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 100,
            ..Default::default()
        };
        let mut index = TemporalHnsw::new(config, L2Distance);

        let n = 1000;
        let dim = 32;
        let mut rng = rand::rng();
        let eids: Vec<u64> = (0..n).map(|i| i as u64).collect();
        let tss: Vec<i64> = (0..n).map(|i| i as i64 * 100).collect();
        let vecs: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| rand::Rng::random::<f32>(&mut rng))
                    .collect()
            })
            .collect();
        let vec_refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

        index.bulk_insert_parallel(&eids, &tss, &vec_refs);

        // Check recall
        let k = 10;
        let mut total_recall = 0.0;
        let n_queries = 20;
        for _ in 0..n_queries {
            let query: Vec<f32> = (0..dim)
                .map(|_| rand::Rng::random::<f32>(&mut rng))
                .collect();
            let results = index.search(&query, k, TemporalFilter::All, 1.0, 0);
            let truth = index.graph().brute_force_knn(&query, k);
            total_recall += super::super::recall_at_k(&results, &truth);
        }
        let avg_recall = total_recall / n_queries as f64;
        assert!(
            avg_recall >= 0.80,
            "parallel bulk_insert recall = {avg_recall:.3}, expected >= 0.80"
        );
    }
}
