//! Hierarchical Navigable Small World (HNSW) graph index.
//!
//! A multi-layer graph structure for approximate nearest neighbor search.
//! Based on Malkov & Yashunin (2018) with single-threaded insert/search.
//!
//! This is a **vanilla HNSW** — no temporal filtering, decay, or timestamp graph.
//! Those are added in Layer 4+.
//!
//! # Algorithm Overview
//!
//! - **Insert**: assign random level, greedily descend from top to target level,
//!   then do beam search at each level to find neighbors
//! - **Search**: greedy descend from top to level 0, then beam search on level 0
//! - **Levels**: higher levels are sparser (fewer nodes), lower levels are denser
//!
//! # Example
//!
//! ```
//! use cvx_index::hnsw::{HnswGraph, HnswConfig};
//! use cvx_index::metrics::CosineDistance;
//!
//! let config = HnswConfig { m: 16, ef_construction: 200, ef_search: 50, ..Default::default() };
//! let mut graph = HnswGraph::new(config, CosineDistance);
//!
//! // Insert vectors
//! graph.insert(0, &[1.0, 0.0, 0.0]);
//! graph.insert(1, &[0.9, 0.1, 0.0]);
//! graph.insert(2, &[0.0, 1.0, 0.0]);
//!
//! // Search
//! let results = graph.search(&[1.0, 0.0, 0.0], 2);
//! assert_eq!(results[0].0, 0); // closest is itself
//! assert_eq!(results[1].0, 1); // second closest
//! ```

pub mod concurrent;
pub mod optimized;
pub mod partitioned;
pub mod streaming;
pub mod temporal;
pub mod temporal_lsh;

pub use concurrent::ConcurrentTemporalHnsw;
pub use temporal::TemporalHnsw;

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use cvx_core::DistanceMetric;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// HNSW index configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Maximum connections per node per layer (except layer 0 which gets 2*M).
    pub m: usize,
    /// Search width during construction.
    pub ef_construction: usize,
    /// Default search width during queries.
    pub ef_search: usize,
    /// Maximum level (auto-calculated if 0).
    pub max_level: usize,
    /// Level generation multiplier: 1 / ln(M).
    pub level_mult: f64,
    /// Use heuristic neighbor selection (Malkov §4.2) for better connectivity.
    /// When false, uses simple closest-M selection.
    pub use_heuristic: bool,
}

impl Default for HnswConfig {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            ef_construction: 200,
            ef_search: 50,
            max_level: 0,
            level_mult: 1.0 / (m as f64).ln(),
            use_heuristic: true,
        }
    }
}

/// Neighbor list: inline up to M entries (no heap allocation).
type NeighborList = SmallVec<[u32; 16]>;

/// A node in the HNSW graph.
#[derive(Serialize, Deserialize)]
pub(crate) struct HnswNode {
    /// The vector data.
    vector: Vec<f32>,
    /// Neighbors at each level this node participates in.
    /// `neighbors[0]` = layer 0 (max 2*M neighbors), `neighbors[1]` = layer 1 (max M), etc.
    neighbors: Vec<NeighborList>,
}

impl optimized::NodeVectors for [HnswNode] {
    fn get_vector(&self, id: u32) -> &[f32] {
        &self[id as usize].vector
    }
}

/// HNSW graph index for approximate nearest neighbor search.
///
/// Optionally stores scalar-quantized codes (uint8) for accelerated distance
/// computation during construction and search. When enabled, candidate exploration
/// uses fast integer distances on codes, while final neighbor selection uses exact
/// float32 distances for quality. See RFC-005 §3.
pub struct HnswGraph<D: DistanceMetric> {
    config: HnswConfig,
    metric: D,
    nodes: Vec<HnswNode>,
    entry_point: Option<u32>,
    max_level: usize,
    rng: SmallRng,
    /// Scalar-quantized codes: node_id → uint8 code (same dim as vectors).
    /// When Some, `distance_fast` uses integer arithmetic (~4× faster).
    sq_codes: Option<Vec<Vec<u8>>>,
    /// Quantization parameters: (min_val, scale) for encoding/decoding.
    sq_params: (f32, f32),
}

impl<D: DistanceMetric> HnswGraph<D> {
    /// Create a new empty HNSW graph.
    pub fn new(config: HnswConfig, metric: D) -> Self {
        Self {
            config,
            metric,
            nodes: Vec::new(),
            entry_point: None,
            max_level: 0,
            rng: SmallRng::from_os_rng(),
            sq_codes: None,
            sq_params: (-1.0, 127.5), // default for L2-normalized vectors: [-1,1]→[0,255]
        }
    }

    /// Enable scalar quantization for accelerated distance computation.
    ///
    /// When enabled, each inserted vector is also encoded as uint8 and
    /// candidate distances during `search_layer` use fast integer arithmetic.
    /// Final neighbor selection still uses exact float32 distances.
    ///
    /// For L2-normalized embeddings (range [-1, 1]), use default parameters.
    /// For unnormalized data, provide the expected min/max range.
    pub fn enable_scalar_quantization(&mut self, min_val: f32, max_val: f32) {
        let range = max_val - min_val;
        self.sq_params = (min_val, if range > 0.0 { 255.0 / range } else { 1.0 });

        // Encode existing nodes
        let codes: Vec<Vec<u8>> = self.nodes.iter()
            .map(|node| Self::encode_sq(&node.vector, self.sq_params.0, self.sq_params.1))
            .collect();
        self.sq_codes = Some(codes);
    }

    /// Disable scalar quantization (revert to exact distances only).
    pub fn disable_scalar_quantization(&mut self) {
        self.sq_codes = None;
    }

    /// Whether scalar quantization is active.
    pub fn is_quantized(&self) -> bool {
        self.sq_codes.is_some()
    }

    /// Encode a vector to uint8 using scalar quantization.
    #[inline]
    fn encode_sq(vector: &[f32], min_val: f32, scale: f32) -> Vec<u8> {
        vector.iter()
            .map(|&v| ((v - min_val) * scale).clamp(0.0, 255.0) as u8)
            .collect()
    }

    /// Fast L2 distance on uint8 codes (auto-vectorized by LLVM).
    #[inline]
    fn distance_sq(a: &[u8], b: &[u8]) -> f32 {
        let mut sum: u32 = 0;
        for i in 0..a.len() {
            let diff = a[i] as i32 - b[i] as i32;
            sum += (diff * diff) as u32;
        }
        sum as f32 // skip sqrt — monotonic, preserves ordering
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Set ef_construction at runtime (e.g., lower for bulk load, higher for incremental).
    pub fn set_ef_construction(&mut self, ef: usize) {
        self.config.ef_construction = ef;
    }

    /// Set ef_search at runtime.
    pub fn set_ef_search(&mut self, ef: usize) {
        self.config.ef_search = ef;
    }

    /// Generate a random level for a new node.
    ///
    /// Capped at 32 (supports up to M^32 ≈ 10^38 nodes). See RFC-002-07.
    fn random_level(&mut self) -> usize {
        let r: f64 = self.rng.random();
        let level = (-r.ln() * self.config.level_mult).floor() as usize;
        level.min(32)
    }

    /// Max neighbors allowed at a given level.
    fn max_neighbors(&self, level: usize) -> usize {
        if level == 0 {
            self.config.m * 2
        } else {
            self.config.m
        }
    }

    /// Insert a vector into the index.
    ///
    /// The `id` should be a unique sequential identifier starting from 0.
    /// Panics if `id != self.len()` (must insert in order).
    pub fn insert(&mut self, id: u32, vector: &[f32]) {
        assert_eq!(
            id as usize,
            self.nodes.len(),
            "must insert sequentially: expected id {}, got {id}",
            self.nodes.len()
        );

        let level = self.random_level();
        let node = HnswNode {
            vector: vector.to_vec(),
            neighbors: (0..=level).map(|_| NeighborList::new()).collect(),
        };
        self.nodes.push(node);

        // Store SQ code if quantization is enabled
        if let Some(ref mut codes) = self.sq_codes {
            codes.push(Self::encode_sq(vector, self.sq_params.0, self.sq_params.1));
        }

        // First node
        if self.nodes.len() == 1 {
            self.entry_point = Some(0);
            self.max_level = level;
            return;
        }

        let entry = self.entry_point.unwrap();
        let mut current = entry;

        // Phase 1: greedy descend from top level to node's level + 1
        for lev in (level + 1..=self.max_level).rev() {
            current = self.greedy_closest(current, vector, lev);
        }

        // Phase 2: insert at each level from min(level, max_level) down to 0
        let insert_from = level.min(self.max_level);
        for lev in (0..=insert_from).rev() {
            let neighbors = self.search_layer(current, vector, self.config.ef_construction, lev);

            // Select best M neighbors
            let max_n = self.max_neighbors(lev);
            let mut selected: Vec<u32> = if self.config.use_heuristic {
                optimized::select_neighbors_heuristic(
                    &self.metric,
                    &neighbors,
                    self.nodes.as_slice(),
                    max_n,
                    false,
                )
            } else {
                neighbors.iter().take(max_n).map(|&(n, _)| n).collect()
            };

            // Safety: ensure at least one connection at every level
            if selected.is_empty() {
                selected.push(current);
            }

            // Add bidirectional connections
            for &neighbor_id in &selected {
                // Avoid self-loops
                if neighbor_id == id {
                    continue;
                }
                // Avoid duplicate edges
                if !self.nodes[id as usize].neighbors[lev].contains(&neighbor_id) {
                    self.nodes[id as usize].neighbors[lev].push(neighbor_id);
                }
                if !self.nodes[neighbor_id as usize].neighbors[lev].contains(&id) {
                    self.nodes[neighbor_id as usize].neighbors[lev].push(id);
                }

                // Prune neighbor's list if over capacity
                let neighbor_count = self.nodes[neighbor_id as usize].neighbors[lev].len();
                if neighbor_count > max_n {
                    self.prune_neighbors(neighbor_id, lev, max_n);
                }
            }

            // Use closest as entry for next lower level
            if let Some(&(closest, _)) = neighbors.first() {
                current = closest;
            }
        }

        // Ensure the new node has at least one connection at level 0.
        // This prevents disconnected components from forming.
        if self.nodes[id as usize].neighbors[0].is_empty() {
            // Find closest node via brute force scan (rare case, only for disconnected nodes)
            let mut best_id = entry;
            let mut best_dist = self.distance(entry, vector);
            for i in 0..self.nodes.len() - 1 {
                let d = self.distance(i as u32, vector);
                if d < best_dist {
                    best_dist = d;
                    best_id = i as u32;
                }
            }
            self.nodes[id as usize].neighbors[0].push(best_id);
            self.nodes[best_id as usize].neighbors[0].push(id);
        }

        // Update entry point if new node has higher level
        if level > self.max_level {
            self.entry_point = Some(id);
            self.max_level = level;
        }
    }

    /// Greedy search for the single closest node at a given level.
    fn greedy_closest(&self, start: u32, query: &[f32], level: usize) -> u32 {
        let query_code = self.sq_codes.as_ref().map(|_|
            Self::encode_sq(query, self.sq_params.0, self.sq_params.1));
        let qc = query_code.as_deref();

        let mut current = start;
        let mut current_dist = self.distance_fast(current, qc, query);

        loop {
            let mut changed = false;
            let neighbors = self.neighbors_at(current, level);
            for &neighbor in neighbors {
                let dist = self.distance_fast(neighbor, qc, query);
                if dist < current_dist {
                    current = neighbor;
                    current_dist = dist;
                    changed = true;
                }
            }
            if !changed {
                return current;
            }
        }
    }

    /// Beam search at a single level. Returns candidates sorted by distance (ascending).
    ///
    /// When scalar quantization is enabled, candidate exploration uses fast uint8
    /// distances. Final results are re-ranked with exact float32 distances.
    fn search_layer(&self, entry: u32, query: &[f32], ef: usize, level: usize) -> Vec<(u32, f32)> {
        // Pre-encode query for SQ if enabled
        let query_code = self.sq_codes.as_ref().map(|_|
            Self::encode_sq(query, self.sq_params.0, self.sq_params.1));
        let qc = query_code.as_deref();

        let entry_dist = self.distance_fast(entry, qc, query);

        // Min-heap for candidates to explore (closest first)
        let mut candidates: BinaryHeap<Reverse<OrdF32Entry>> = BinaryHeap::new();
        // Max-heap for results (farthest first, so we can evict)
        let mut results: BinaryHeap<OrdF32Entry> = BinaryHeap::new();
        // Visited set
        let mut visited = vec![false; self.nodes.len()];

        candidates.push(Reverse(OrdF32Entry(entry_dist, entry)));
        results.push(OrdF32Entry(entry_dist, entry));
        visited[entry as usize] = true;

        while let Some(Reverse(OrdF32Entry(c_dist, c_id))) = candidates.pop() {
            // If closest candidate is farther than farthest result, stop
            let farthest_result = results.peek().map(|e| e.0).unwrap_or(f32::INFINITY);
            if c_dist > farthest_result {
                break;
            }

            let neighbors = self.neighbors_at(c_id, level);
            for &neighbor in neighbors {
                if visited[neighbor as usize] {
                    continue;
                }
                visited[neighbor as usize] = true;

                let dist = self.distance_fast(neighbor, qc, query);
                let farthest_result = results.peek().map(|e| e.0).unwrap_or(f32::INFINITY);

                if dist < farthest_result || results.len() < ef {
                    candidates.push(Reverse(OrdF32Entry(dist, neighbor)));
                    results.push(OrdF32Entry(dist, neighbor));
                    if results.len() > ef {
                        results.pop(); // remove farthest
                    }
                }
            }
        }

        // Re-rank with exact distances when SQ was used (quality matters for final results)
        let mut result_vec: Vec<(u32, f32)> = if self.sq_codes.is_some() {
            results.into_iter()
                .map(|e| (e.1, self.distance(e.1, query)))
                .collect()
        } else {
            results.into_iter().map(|e| (e.1, e.0)).collect()
        };
        result_vec.sort_by(|a, b| a.1.total_cmp(&b.1));
        result_vec
    }

    /// Prune a node's neighbor list to keep only the best `max_n`.
    ///
    /// Uses heuristic selection when enabled (diverse directions),
    /// otherwise keeps the closest `max_n` by distance.
    fn prune_neighbors(&mut self, node_id: u32, level: usize, max_n: usize) {
        let node_vec = self.nodes[node_id as usize].vector.clone();
        let scored: Vec<(u32, f32)> = self.nodes[node_id as usize].neighbors[level]
            .iter()
            .map(|&n| {
                (
                    n,
                    self.metric
                        .distance(&node_vec, &self.nodes[n as usize].vector),
                )
            })
            .collect();

        let pruned = if self.config.use_heuristic {
            optimized::select_neighbors_heuristic(
                &self.metric,
                &scored,
                self.nodes.as_slice(),
                max_n,
                false,
            )
        } else {
            let mut s = scored;
            s.sort_by(|a, b| a.1.total_cmp(&b.1));
            s.truncate(max_n);
            s.iter().map(|&(n, _)| n).collect()
        };

        self.nodes[node_id as usize].neighbors[level] = pruned.into_iter().collect();
    }

    /// Search for the k nearest neighbors of `query`.
    ///
    /// Returns a Vec of `(node_id, distance)` sorted by distance ascending.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        let entry = self.entry_point.unwrap();
        let mut current = entry;

        // Greedy descend from top to level 1
        for lev in (1..=self.max_level).rev() {
            current = self.greedy_closest(current, query, lev);
        }

        // Beam search on level 0
        let mut results = self.search_layer(current, query, self.config.ef_search.max(k), 0);
        results.truncate(k);
        results
    }

    /// Search with a predicate filter.
    ///
    /// Like [`search`](Self::search), but only returns nodes where `filter(node_id)` is true.
    /// The HNSW graph is still traversed through filtered-out nodes (they act as bridges),
    /// but only matching nodes appear in the results.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: impl Fn(u32) -> bool,
    ) -> Vec<(u32, f32)> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        let entry = self.entry_point.unwrap();
        let mut current = entry;

        // Greedy descend from top to level 1
        for lev in (1..=self.max_level).rev() {
            current = self.greedy_closest(current, query, lev);
        }

        // Beam search on level 0, collecting all candidates
        let ef = self.config.ef_search.max(k * 4); // over-fetch to compensate for filtering
        let all_candidates = self.search_layer(current, query, ef, 0);

        // Apply filter and take top-k
        let mut results: Vec<(u32, f32)> = all_candidates
            .into_iter()
            .filter(|&(id, _)| filter(id))
            .collect();
        results.truncate(k);
        results
    }

    /// Get the stored vector for a node.
    pub fn vector(&self, node_id: u32) -> &[f32] {
        &self.nodes[node_id as usize].vector
    }

    /// Get the configuration.
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// Get the entry point node ID.
    pub fn entry_point(&self) -> Option<u32> {
        self.entry_point
    }

    /// Get the maximum level in the graph.
    pub fn max_level(&self) -> usize {
        self.max_level
    }

    /// Get all neighbor lists for a node (one per level).
    pub fn all_neighbors(&self, node_id: u32) -> Vec<Vec<u32>> {
        let node = &self.nodes[node_id as usize];
        node.neighbors.iter().map(|n| n.to_vec()).collect()
    }

    /// Return node IDs present at the given HNSW level (RFC-004).
    ///
    /// These are the natural "hub" nodes of the graph hierarchy.
    /// Level 0 = all nodes, higher levels = fewer, more connected hubs.
    /// Count follows geometric distribution: ~N/M^level.
    pub fn nodes_at_level(&self, level: usize) -> Vec<u32> {
        (0..self.nodes.len() as u32)
            .filter(|&id| self.nodes[id as usize].neighbors.len() > level)
            .collect()
    }

    /// Assign a vector to its nearest hub at the given level (RFC-004).
    ///
    /// Uses greedy descent from the entry point — O(log N).
    /// Returns the node_id of the nearest hub at that level.
    pub fn assign_region(&self, vector: &[f32], level: usize) -> Option<u32> {
        if self.nodes.is_empty() {
            return None;
        }

        let entry = self.entry_point.unwrap();
        let mut current = entry;

        // Greedy descend from top to target level + 1
        for lev in (level + 1..=self.max_level).rev() {
            current = self.greedy_closest(current, vector, lev);
        }

        // At the target level, find the closest hub
        if level <= self.max_level {
            current = self.greedy_closest(current, vector, level);
        }

        // Ensure result is actually at the target level
        if self.nodes[current as usize].neighbors.len() > level {
            Some(current)
        } else {
            // Fallback: search among known hubs at this level
            let hubs = self.nodes_at_level(level);
            hubs.into_iter()
                .min_by(|&a, &b| {
                    self.distance(a, vector)
                        .total_cmp(&self.distance(b, vector))
                })
        }
    }

    /// Compute distance between a stored node and a query vector.
    ///
    /// When scalar quantization is enabled, uses fast uint8 distances
    /// for candidate exploration (~4× faster). Falls back to exact
    /// float32 distance when SQ is disabled.
    #[inline]
    fn distance(&self, node_id: u32, query: &[f32]) -> f32 {
        self.metric
            .distance(&self.nodes[node_id as usize].vector, query)
    }

    /// Fast approximate distance using scalar-quantized codes.
    ///
    /// Returns the exact distance if SQ is not enabled.
    #[inline]
    fn distance_fast(&self, node_id: u32, query_code: Option<&[u8]>, query: &[f32]) -> f32 {
        if let (Some(codes), Some(qc)) = (&self.sq_codes, query_code) {
            Self::distance_sq(&codes[node_id as usize], qc)
        } else {
            self.distance(node_id, query)
        }
    }

    /// Get the neighbor list for a node at a given level.
    #[inline]
    fn neighbors_at(&self, node_id: u32, level: usize) -> &[u32] {
        let node = &self.nodes[node_id as usize];
        if level < node.neighbors.len() {
            &node.neighbors[level]
        } else {
            &[]
        }
    }

    /// Check graph invariant: all nodes are reachable from entry point at level 0.
    ///
    /// Returns the number of reachable nodes. Should equal `self.len()`.
    pub fn count_reachable(&self) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }
        let entry = self.entry_point.unwrap();
        let mut visited = vec![false; self.nodes.len()];
        let mut stack = vec![entry];
        visited[entry as usize] = true;
        let mut count = 1usize;

        while let Some(node) = stack.pop() {
            for &neighbor in self.neighbors_at(node, 0) {
                if !visited[neighbor as usize] {
                    visited[neighbor as usize] = true;
                    count += 1;
                    stack.push(neighbor);
                }
            }
        }
        count
    }

    /// Brute-force kNN for ground truth comparison.
    ///
    /// Returns `(node_id, distance)` sorted ascending. O(N) per query.
    pub fn brute_force_knn(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let mut all: Vec<(u32, f32)> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, node)| (i as u32, self.metric.distance(&node.vector, query)))
            .collect();
        all.sort_by(|a, b| a.1.total_cmp(&b.1));
        all.truncate(k);
        all
    }
}

/// Serializable snapshot of an HNSW graph (excludes metric + RNG).
///
/// Used by `TemporalHnsw::save` / `TemporalHnsw::load` for index persistence.
#[derive(Serialize, Deserialize)]
pub(crate) struct HnswSnapshot {
    pub(crate) config: HnswConfig,
    pub(crate) nodes: Vec<HnswNode>,
    pub(crate) entry_point: Option<u32>,
    pub(crate) max_level: usize,
    pub(crate) sq_codes: Option<Vec<Vec<u8>>>,
    pub(crate) sq_params: (f32, f32),
}

impl<D: DistanceMetric> HnswGraph<D> {
    /// Create a serializable snapshot (excludes metric and RNG).
    pub(crate) fn to_snapshot(&self) -> HnswSnapshot {
        HnswSnapshot {
            config: self.config.clone(),
            nodes: self.nodes.iter().map(|n| HnswNode {
                vector: n.vector.clone(),
                neighbors: n.neighbors.clone(),
            }).collect(),
            entry_point: self.entry_point,
            max_level: self.max_level,
            sq_codes: self.sq_codes.clone(),
            sq_params: self.sq_params,
        }
    }

    /// Restore from a snapshot, providing the metric.
    pub(crate) fn from_snapshot(snapshot: HnswSnapshot, metric: D) -> Self {
        Self {
            config: snapshot.config,
            metric,
            nodes: snapshot.nodes,
            entry_point: snapshot.entry_point,
            max_level: snapshot.max_level,
            rng: SmallRng::from_os_rng(),
            sq_codes: snapshot.sq_codes,
            sq_params: snapshot.sq_params,
        }
    }
}

/// Wrapper for f32 ordering in BinaryHeap (total order via bit comparison).
#[derive(Clone, Copy)]
struct OrdF32Entry(f32, u32);

impl PartialEq for OrdF32Entry {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits() && self.1 == other.1
    }
}

impl Eq for OrdF32Entry {}

impl PartialOrd for OrdF32Entry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF32Entry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0).then(self.1.cmp(&other.1))
    }
}

/// Compute recall@k: fraction of true kNN found by approximate search.
pub fn recall_at_k(approximate: &[(u32, f32)], ground_truth: &[(u32, f32)]) -> f64 {
    let truth_set: std::collections::HashSet<u32> =
        ground_truth.iter().map(|&(id, _)| id).collect();
    let found = approximate
        .iter()
        .filter(|&&(id, _)| truth_set.contains(&id))
        .count();
    found as f64 / ground_truth.len().max(1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{CosineDistance, L2Distance};

    fn make_graph(m: usize, ef_c: usize, ef_s: usize) -> HnswGraph<L2Distance> {
        let config = HnswConfig {
            m,
            ef_construction: ef_c,
            ef_search: ef_s,
            ..Default::default()
        };
        HnswGraph::new(config, L2Distance)
    }

    #[test]
    fn empty_graph() {
        let graph = make_graph(16, 200, 50);
        assert!(graph.is_empty());
        assert_eq!(graph.len(), 0);
        assert_eq!(graph.search(&[1.0, 2.0], 5), vec![]);
    }

    #[test]
    fn single_insert_and_search() {
        let mut graph = make_graph(16, 200, 50);
        graph.insert(0, &[1.0, 0.0, 0.0]);

        let results = graph.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 < 1e-5); // exact match
    }

    #[test]
    fn three_vectors_correct_order() {
        let mut graph = make_graph(16, 200, 50);
        graph.insert(0, &[1.0, 0.0]);
        graph.insert(1, &[0.9, 0.1]);
        graph.insert(2, &[0.0, 1.0]);

        let results = graph.search(&[1.0, 0.0], 3);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 0); // exact match
        assert_eq!(results[1].0, 1); // close
        assert_eq!(results[2].0, 2); // far
    }

    #[test]
    fn all_nodes_reachable_100() {
        let mut graph = make_graph(16, 200, 50);
        for i in 0..100u32 {
            graph.insert(i, &[i as f32, (100 - i) as f32]);
        }
        assert_eq!(graph.count_reachable(), 100);
    }

    #[test]
    fn all_nodes_reachable_1000() {
        let mut graph = make_graph(16, 200, 50);
        for i in 0..1000u32 {
            let angle = (i as f32) * 0.1;
            graph.insert(i, &[angle.cos(), angle.sin()]);
        }
        assert_eq!(graph.count_reachable(), 1000);
    }

    #[test]
    fn recall_at_10_random_1k_d32() {
        let dim = 32;
        let n = 1000u32;
        let mut graph = make_graph(16, 200, 50);

        // Insert random vectors
        let mut rng = rand::rng();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
            .collect();

        for (i, v) in vectors.iter().enumerate() {
            graph.insert(i as u32, v);
        }

        // Test recall on 100 random queries
        let k = 10;
        let n_queries = 100;
        let mut total_recall = 0.0;

        for _ in 0..n_queries {
            let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            let approx = graph.search(&query, k);
            let truth = graph.brute_force_knn(&query, k);
            total_recall += recall_at_k(&approx, &truth);
        }

        let avg_recall = total_recall / n_queries as f64;
        assert!(
            avg_recall >= 0.90,
            "recall@10 = {avg_recall:.3}, expected >= 0.90"
        );
    }

    #[test]
    fn recall_at_10_random_10k_d128() {
        let dim = 128;
        let n = 10_000u32;
        let k = 10;
        let mut graph = HnswGraph::new(
            HnswConfig {
                m: 16,
                ef_construction: 200,
                ef_search: 200, // higher ef_search for better recall
                ..Default::default()
            },
            L2Distance,
        );

        let mut rng = rand::rng();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
            .collect();

        for (i, v) in vectors.iter().enumerate() {
            graph.insert(i as u32, v);
        }

        let reachable = graph.count_reachable();
        // Reachability may not be 100% due to pruning creating components.
        // This is a known HNSW limitation that improves with higher M and ef_construction.
        // We check that at least 98% of nodes are reachable.
        assert!(
            reachable >= (n as usize) * 98 / 100,
            "reachable: {reachable} / {n}, expected >= 98%"
        );

        let n_queries = 50;
        let mut total_recall = 0.0;

        for _ in 0..n_queries {
            let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            let approx = graph.search(&query, k);
            let truth = graph.brute_force_knn(&query, k);
            total_recall += recall_at_k(&approx, &truth);
        }

        let avg_recall = total_recall / n_queries as f64;
        assert!(
            avg_recall >= 0.85,
            "recall@10 on 10K D=128 = {avg_recall:.3}, expected >= 0.85"
        );
    }

    #[test]
    fn works_with_cosine_distance() {
        let config = HnswConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 50,
            ..Default::default()
        };
        let mut graph = HnswGraph::new(config, CosineDistance);

        graph.insert(0, &[1.0, 0.0, 0.0]);
        graph.insert(1, &[0.99, 0.01, 0.0]);
        graph.insert(2, &[0.0, 0.0, 1.0]);

        let results = graph.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results[0].0, 0);
        assert_eq!(results[1].0, 1);
    }

    #[test]
    fn search_k_larger_than_n() {
        let mut graph = make_graph(16, 200, 50);
        graph.insert(0, &[1.0, 0.0]);
        graph.insert(1, &[0.0, 1.0]);

        let results = graph.search(&[1.0, 0.0], 10);
        assert_eq!(results.len(), 2); // only 2 nodes exist
    }

    #[test]
    fn recall_helper_correct() {
        let approx = vec![(0, 0.1), (1, 0.2), (2, 0.3)];
        let truth = vec![(0, 0.1), (1, 0.2), (3, 0.25)];
        assert!((recall_at_k(&approx, &truth) - 2.0 / 3.0).abs() < 1e-10);
    }

    /// 100K vectors D=128, recall@10 ≥ 0.95 (Layer 2 exit criterion)
    #[test]
    #[ignore] // slow: ~10s, run with `cargo test -- --ignored`
    fn recall_100k_d128() {
        let dim = 128;
        let n = 100_000u32;
        let k = 10;
        let mut graph = HnswGraph::new(
            HnswConfig {
                m: 16,
                ef_construction: 200,
                ef_search: 100,
                ..Default::default()
            },
            L2Distance,
        );

        let mut rng = rand::rng();
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
            .collect();

        for (i, v) in vectors.iter().enumerate() {
            graph.insert(i as u32, v);
        }

        assert_eq!(
            graph.count_reachable(),
            n as usize,
            "not all nodes reachable"
        );

        let n_queries = 100;
        let mut total_recall = 0.0;

        for _ in 0..n_queries {
            let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
            let approx = graph.search(&query, k);
            let truth = graph.brute_force_knn(&query, k);
            total_recall += recall_at_k(&approx, &truth);
        }

        let avg_recall = total_recall / n_queries as f64;
        assert!(
            avg_recall >= 0.95,
            "recall@10 on 100K D=128 = {avg_recall:.3}, expected >= 0.95"
        );
    }
}
