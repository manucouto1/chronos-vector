//! Temporal Graph Index — HNSW extended with temporal successor edges (RFC-010).
//!
//! Composites `TemporalHnsw` with `TemporalEdgeLayer` to enable:
//! 1. **Causal search**: find semantic neighbors, then walk temporal edges for context
//! 2. **Hybrid search**: beam search that explores both semantic AND temporal edges

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::path::Path;

use cvx_core::traits::DistanceMetric;
use cvx_core::types::TemporalFilter;

use super::HnswConfig;
use super::temporal::TemporalHnsw;
use super::temporal_edges::TemporalEdgeLayer;
use super::typed_edges::{EdgeType, TypedEdgeStore};

// ─── Types ──────────────────────────────────────────────────────────

/// A search result with causal temporal context.
#[derive(Debug, Clone)]
pub struct CausalSearchResult {
    /// The semantically matched node.
    pub node_id: u32,
    /// Distance score.
    pub score: f32,
    /// Entity that owns this node.
    pub entity_id: u64,
    /// Temporal successors: what happened NEXT to this entity.
    pub successors: Vec<(u32, i64)>,
    /// Temporal predecessors: what happened BEFORE.
    pub predecessors: Vec<(u32, i64)>,
}

// ─── TemporalGraphIndex ─────────────────────────────────────────────

/// Temporal Graph Index: HNSW with temporal successor/predecessor edges.
///
/// Wraps `TemporalHnsw` (untouched) with a `TemporalEdgeLayer` for
/// causal navigation and hybrid search.
pub struct TemporalGraphIndex<D: DistanceMetric> {
    /// The underlying spatiotemporal HNSW index.
    inner: TemporalHnsw<D>,
    /// Temporal edge layer (successor/predecessor per entity).
    edges: TemporalEdgeLayer,
    /// Typed relational edges (RFC-013 Part B).
    typed_edges: TypedEdgeStore,
}

impl<D: DistanceMetric + Clone> TemporalGraphIndex<D> {
    /// Create a new empty temporal graph index.
    pub fn new(config: HnswConfig, metric: D) -> Self {
        Self {
            inner: TemporalHnsw::new(config, metric),
            edges: TemporalEdgeLayer::new(),
            typed_edges: TypedEdgeStore::new(),
        }
    }

    /// Create from an existing TemporalHnsw (migration path).
    ///
    /// Rebuilds the temporal edge layer from the entity_index.
    pub fn from_temporal_hnsw(inner: TemporalHnsw<D>) -> Self {
        let mut edges = TemporalEdgeLayer::with_capacity(inner.len());

        // We need to register all nodes in order.
        // The entity_index has (timestamp, node_id) sorted by timestamp per entity.
        // But we must register in node_id order (0, 1, 2, ...).

        // Build a mapping: node_id → its predecessor in the entity chain
        let mut pred_map: Vec<Option<u32>> = vec![None; inner.len()];

        for nid in 0..inner.len() as u32 {
            let eid = inner.entity_id(nid);
            // Find the previous node for this entity (node with closest earlier timestamp)
            let traj = inner.trajectory(eid, TemporalFilter::All);
            let my_ts = inner.timestamp(nid);

            let prev = traj
                .iter()
                .filter(|&&(ts, id)| ts < my_ts || (ts == my_ts && id < nid))
                .max_by_key(|&&(ts, _)| ts)
                .map(|&(_, id)| id);

            pred_map[nid as usize] = prev;
        }

        for nid in 0..inner.len() as u32 {
            edges.register(nid, pred_map[nid as usize]);
        }

        Self {
            inner,
            edges,
            typed_edges: TypedEdgeStore::new(),
        }
    }

    /// Insert a temporal point.
    pub fn insert(&mut self, entity_id: u64, timestamp: i64, vector: &[f32]) -> u32 {
        let last_node = self.inner.entity_last_node(entity_id);
        let node_id = self.inner.insert(entity_id, timestamp, vector);
        self.edges.register(node_id, last_node);
        node_id
    }

    /// Insert a temporal point with an outcome reward.
    pub fn insert_with_reward(
        &mut self,
        entity_id: u64,
        timestamp: i64,
        vector: &[f32],
        reward: f32,
    ) -> u32 {
        let last_node = self.inner.entity_last_node(entity_id);
        let node_id = self
            .inner
            .insert_with_reward(entity_id, timestamp, vector, reward);
        self.edges.register(node_id, last_node);
        node_id
    }

    /// Standard search (delegates to inner TemporalHnsw).
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: TemporalFilter,
        alpha: f32,
        query_timestamp: i64,
    ) -> Vec<(u32, f32)> {
        self.inner.search(query, k, filter, alpha, query_timestamp)
    }

    /// Causal search: semantic search + temporal edge context.
    ///
    /// Phase 1: Standard HNSW search.
    /// Phase 2: For each result, walk temporal edges to get what happened
    /// before and after.
    ///
    /// Answers: "Find similar entities, and show me what happened to them next."
    pub fn causal_search(
        &self,
        query: &[f32],
        k: usize,
        filter: TemporalFilter,
        alpha: f32,
        query_timestamp: i64,
        temporal_context: usize,
    ) -> Vec<CausalSearchResult> {
        let results = self.inner.search(query, k, filter, alpha, query_timestamp);

        results
            .into_iter()
            .map(|(node_id, score)| {
                let entity_id = self.inner.entity_id(node_id);

                let succ_ids = self.edges.walk_forward(node_id, temporal_context);
                let successors: Vec<(u32, i64)> = succ_ids
                    .into_iter()
                    .map(|nid| (nid, self.inner.timestamp(nid)))
                    .collect();

                let pred_ids = self.edges.walk_backward(node_id, temporal_context);
                let predecessors: Vec<(u32, i64)> = pred_ids
                    .into_iter()
                    .map(|nid| (nid, self.inner.timestamp(nid)))
                    .collect();

                CausalSearchResult {
                    node_id,
                    score,
                    entity_id,
                    successors,
                    predecessors,
                }
            })
            .collect()
    }

    /// Hybrid search: beam search exploring both semantic AND temporal edges.
    ///
    /// At each step of the beam search on level 0, when visiting a node,
    /// also adds its temporal neighbors to the candidate set with a
    /// distance penalty controlled by `beta`.
    ///
    /// - `beta = 0.0`: pure semantic HNSW (ignores temporal edges)
    /// - `beta = 1.0`: always follow temporal edges (aggressive temporal exploration)
    pub fn hybrid_search(
        &self,
        query: &[f32],
        k: usize,
        filter: TemporalFilter,
        alpha: f32,
        beta: f32,
        query_timestamp: i64,
    ) -> Vec<(u32, f32)> {
        let graph = self.inner.graph();

        if graph.is_empty() {
            return Vec::new();
        }

        let entry = match graph.entry_point() {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        let bitmap = self.inner.build_filter_bitmap(&filter);
        let ef = graph.config().ef_search.max(k);

        // Phase 1: greedy descent from entry point to level 0
        let max_level = graph.max_level();
        let mut current = entry;
        let mut current_dist = graph.distance_to(current, query);

        for level in (1..=max_level).rev() {
            let mut improved = true;
            while improved {
                improved = false;
                for &neighbor in graph.neighbors_at_level(current, level) {
                    let d = graph.distance_to(neighbor, query);
                    if d < current_dist {
                        current = neighbor;
                        current_dist = d;
                        improved = true;
                    }
                }
            }
        }

        // Phase 2: hybrid beam search on level 0
        // candidates: min-heap (closest first to explore)
        // results: max-heap (farthest first to evict)
        let mut candidates: BinaryHeap<Reverse<(OrderedF32, u32)>> = BinaryHeap::new();
        let mut results: BinaryHeap<(OrderedF32, u32)> = BinaryHeap::new();
        let mut visited: HashSet<u32> = HashSet::new();

        let entry_dist = graph.distance_to(current, query);
        candidates.push(Reverse((OrderedF32(entry_dist), current)));
        if bitmap.contains(current) {
            results.push((OrderedF32(entry_dist), current));
        }
        visited.insert(current);

        while let Some(Reverse((OrderedF32(c_dist), c_id))) = candidates.pop() {
            let farthest_dist = results
                .peek()
                .map(|(OrderedF32(d), _)| *d)
                .unwrap_or(f32::MAX);
            if c_dist > farthest_dist && results.len() >= ef {
                break;
            }

            // Explore semantic neighbors
            let semantic_neighbors = graph.neighbors_at_level(c_id, 0);

            // Explore temporal neighbors (weighted by beta)
            let temporal_neighbors: Vec<u32> = if beta > 0.0 {
                self.edges.temporal_neighbors(c_id).collect()
            } else {
                Vec::new()
            };

            // Process all neighbors
            for &neighbor in semantic_neighbors.iter().chain(temporal_neighbors.iter()) {
                if !visited.insert(neighbor) {
                    continue;
                }

                // Skip if not in temporal filter
                if !bitmap.contains(neighbor) {
                    continue;
                }

                let mut dist = graph.distance_to(neighbor, query);

                // Apply temporal component if alpha < 1.0
                if alpha < 1.0 {
                    let t_dist = self.inner.temporal_distance_normalized(
                        self.inner.timestamp(neighbor),
                        query_timestamp,
                    );
                    dist = alpha * dist + (1.0 - alpha) * t_dist;
                }

                let farthest = results
                    .peek()
                    .map(|(OrderedF32(d), _)| *d)
                    .unwrap_or(f32::MAX);
                if dist < farthest || results.len() < ef {
                    candidates.push(Reverse((OrderedF32(dist), neighbor)));
                    results.push((OrderedF32(dist), neighbor));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        // Collect and sort by distance
        let mut final_results: Vec<(u32, f32)> = results
            .into_iter()
            .map(|(OrderedF32(d), nid)| (nid, d))
            .collect();
        final_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        final_results.truncate(k);
        final_results
    }

    // ─── Accessors ──────────────────────────────────────────────

    /// Access the underlying TemporalHnsw.
    pub fn inner(&self) -> &TemporalHnsw<D> {
        &self.inner
    }

    /// Access the temporal edge layer.
    pub fn edges(&self) -> &TemporalEdgeLayer {
        &self.edges
    }

    /// Get trajectory for an entity.
    pub fn trajectory(&self, entity_id: u64, filter: TemporalFilter) -> Vec<(i64, u32)> {
        self.inner.trajectory(entity_id, filter)
    }

    /// Get vector by node ID.
    pub fn vector(&self, node_id: u32) -> &[f32] {
        self.inner.vector(node_id)
    }

    /// Get entity ID by node ID.
    pub fn entity_id(&self, node_id: u32) -> u64 {
        self.inner.entity_id(node_id)
    }

    /// Get timestamp by node ID.
    pub fn timestamp(&self, node_id: u32) -> i64 {
        self.inner.timestamp(node_id)
    }

    /// Total number of points.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    // ─── Delegated configuration ──────────────────────────────────

    /// Mutable access to the underlying TemporalHnsw.
    pub fn inner_mut(&mut self) -> &mut TemporalHnsw<D> {
        &mut self.inner
    }

    /// Get HNSW config.
    pub fn config(&self) -> &super::HnswConfig {
        self.inner.config()
    }

    /// Set ef_construction at runtime.
    pub fn set_ef_construction(&mut self, ef: usize) {
        self.inner.set_ef_construction(ef);
    }

    /// Set ef_search at runtime.
    pub fn set_ef_search(&mut self, ef: usize) {
        self.inner.set_ef_search(ef);
    }

    /// Enable scalar quantization.
    pub fn enable_scalar_quantization(&mut self, min_val: f32, max_val: f32) {
        self.inner.enable_scalar_quantization(min_val, max_val);
    }

    /// Disable scalar quantization.
    pub fn disable_scalar_quantization(&mut self) {
        self.inner.disable_scalar_quantization();
    }

    // ─── Delegated recency search (RFC-012 P7+P8) ──────────────────

    /// Search with recency bias and normalized distances.
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
        self.inner.search_with_recency(
            query,
            k,
            filter,
            alpha,
            query_timestamp,
            recency_lambda,
            recency_weight,
        )
    }

    // ─── Delegated outcome / reward (RFC-012 P4) ───────────────────

    /// Get the reward for a node.
    pub fn reward(&self, node_id: u32) -> f32 {
        self.inner.reward(node_id)
    }

    /// Set the reward for a node retroactively.
    pub fn set_reward(&mut self, node_id: u32, reward: f32) {
        self.inner.set_reward(node_id, reward);
    }

    /// Search with reward pre-filtering.
    pub fn search_with_reward(
        &self,
        query: &[f32],
        k: usize,
        filter: TemporalFilter,
        alpha: f32,
        query_timestamp: i64,
        min_reward: f32,
    ) -> Vec<(u32, f32)> {
        self.inner
            .search_with_reward(query, k, filter, alpha, query_timestamp, min_reward)
    }

    // ─── Delegated centering (RFC-012 Part B) ─────────────────────

    /// Compute the centroid of all vectors.
    pub fn compute_centroid(&self) -> Option<Vec<f32>> {
        self.inner.compute_centroid()
    }

    /// Set centroid for anisotropy correction.
    pub fn set_centroid(&mut self, centroid: Vec<f32>) {
        self.inner.set_centroid(centroid);
    }

    /// Clear centroid.
    pub fn clear_centroid(&mut self) {
        self.inner.clear_centroid();
    }

    /// Get current centroid.
    pub fn centroid(&self) -> Option<&[f32]> {
        self.inner.centroid()
    }

    /// Center a vector by subtracting the centroid.
    pub fn centered_vector(&self, vec: &[f32]) -> Vec<f32> {
        self.inner.centered_vector(vec)
    }

    // ─── Delegated region operations ──────────────────────────────

    /// Get semantic regions at a given HNSW level.
    pub fn regions(&self, level: usize) -> Vec<(u32, Vec<f32>, usize)> {
        self.inner.regions(level)
    }

    /// O(N) single-pass region assignments.
    pub fn region_assignments(
        &self,
        level: usize,
        filter: TemporalFilter,
    ) -> std::collections::HashMap<u32, Vec<(u64, i64)>> {
        self.inner.region_assignments(level, filter)
    }

    /// Smoothed region distribution trajectory for an entity.
    pub fn region_trajectory(
        &self,
        entity_id: u64,
        level: usize,
        window_days: i64,
        alpha: f32,
    ) -> Vec<(i64, Vec<f32>)> {
        self.inner
            .region_trajectory(entity_id, level, window_days, alpha)
    }

    // ─── Typed edges (RFC-013 Part B) ───────────────────────────

    /// Access the typed edge store.
    pub fn typed_edges(&self) -> &TypedEdgeStore {
        &self.typed_edges
    }

    /// Mutable access to the typed edge store.
    pub fn typed_edges_mut(&mut self) -> &mut TypedEdgeStore {
        &mut self.typed_edges
    }

    /// Add a typed edge between two nodes.
    pub fn add_typed_edge(&mut self, source: u32, target: u32, edge_type: EdgeType, weight: f32) {
        self.typed_edges.add_edge(source, target, edge_type, weight);
    }

    /// Get the success score of a node based on typed edges.
    ///
    /// Uses Beta prior: P(success) = (1 + n_success) / (2 + n_total).
    pub fn success_score(&self, node_id: u32) -> f32 {
        self.typed_edges.success_score(node_id)
    }

    /// Save to directory (index + temporal edges + typed edges).
    pub fn save(&self, dir: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        self.inner.save(&dir.join("index.bin"))?;
        let edge_bytes = postcard::to_allocvec(&self.edges).map_err(std::io::Error::other)?;
        std::fs::write(dir.join("temporal_edges.bin"), edge_bytes)?;
        let typed_bytes =
            postcard::to_allocvec(&self.typed_edges).map_err(std::io::Error::other)?;
        std::fs::write(dir.join("typed_edges.bin"), typed_bytes)?;
        Ok(())
    }

    /// Load from directory.
    pub fn load(dir: &Path, metric: D) -> std::io::Result<Self> {
        let inner = TemporalHnsw::load(&dir.join("index.bin"), metric)?;
        let edge_bytes = std::fs::read(dir.join("temporal_edges.bin"))?;
        let edges: TemporalEdgeLayer = postcard::from_bytes(&edge_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        // Typed edges are optional (backward compat)
        let typed_edges = if dir.join("typed_edges.bin").exists() {
            let typed_bytes = std::fs::read(dir.join("typed_edges.bin"))?;
            postcard::from_bytes(&typed_bytes)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
        } else {
            TypedEdgeStore::new()
        };
        Ok(Self {
            inner,
            edges,
            typed_edges,
        })
    }
}

// ─── TemporalIndexAccess ────────────────────────────────────────────

impl<D: DistanceMetric + Clone> cvx_core::TemporalIndexAccess for TemporalGraphIndex<D> {
    fn search_raw(
        &self,
        query: &[f32],
        k: usize,
        filter: TemporalFilter,
        alpha: f32,
        query_timestamp: i64,
    ) -> Vec<(u32, f32)> {
        // Use hybrid search with moderate beta
        self.hybrid_search(query, k, filter, alpha, 0.3, query_timestamp)
    }

    fn trajectory(&self, entity_id: u64, filter: TemporalFilter) -> Vec<(i64, u32)> {
        self.inner.trajectory(entity_id, filter)
    }

    fn vector(&self, node_id: u32) -> Vec<f32> {
        self.inner.vector(node_id).to_vec()
    }

    fn entity_id(&self, node_id: u32) -> u64 {
        self.inner.entity_id(node_id)
    }

    fn timestamp(&self, node_id: u32) -> i64 {
        self.inner.timestamp(node_id)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

// ─── Ordered float helper ───────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedF32(f32);

impl Eq for OrderedF32 {}

impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::L2Distance;

    fn setup_index(
        n_entities: u64,
        points_per_entity: usize,
        dim: usize,
    ) -> TemporalGraphIndex<L2Distance> {
        let config = HnswConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 50,
            ..Default::default()
        };
        let mut index = TemporalGraphIndex::new(config, L2Distance);

        for e in 0..n_entities {
            for i in 0..points_per_entity {
                let ts = (i as i64) * 1_000_000;
                let v: Vec<f32> = (0..dim)
                    .map(|d| (e as f32 * 10.0) + (i as f32 * 0.1) + (d as f32 * 0.01))
                    .collect();
                index.insert(e, ts, &v);
            }
        }

        index
    }

    // ─── Basic insert + edges ───────────────────────────────────

    #[test]
    fn insert_creates_temporal_edges() {
        let index = setup_index(1, 5, 3);

        assert_eq!(index.len(), 5);
        assert_eq!(index.edges().len(), 5);

        // Chain: 0 → 1 → 2 → 3 → 4
        assert_eq!(index.edges().successor(0), Some(1));
        assert_eq!(index.edges().successor(3), Some(4));
        assert_eq!(index.edges().predecessor(4), Some(3));
        assert_eq!(index.edges().successor(4), None);
    }

    #[test]
    fn multi_entity_edges_isolated() {
        let index = setup_index(3, 5, 3);

        // Entity 0: nodes 0-4, Entity 1: nodes 5-9, Entity 2: nodes 10-14
        // (assuming sequential insert order)
        for i in 0..4u32 {
            let succ = index.edges().successor(i);
            assert!(succ.is_some());
            // Successor should be same entity
            let succ_entity = index.entity_id(succ.unwrap());
            let my_entity = index.entity_id(i);
            assert_eq!(
                succ_entity, my_entity,
                "edge from node {i} crosses entities"
            );
        }
    }

    // ─── Causal search ──────────────────────────────────────────

    #[test]
    fn causal_search_returns_context() {
        let index = setup_index(3, 10, 4);

        let results = index.causal_search(
            &[0.5, 0.05, 0.005, 0.001],
            3,
            TemporalFilter::All,
            1.0,
            5_000_000,
            3, // 3 steps of temporal context
        );

        assert_eq!(results.len(), 3);

        for r in &results {
            // Each result should have temporal context
            // (unless it's at the very end of its entity's timeline)
            assert!(
                !r.successors.is_empty() || !r.predecessors.is_empty(),
                "node {} should have some temporal context",
                r.node_id
            );

            // Verify successors are temporally ordered
            for w in r.successors.windows(2) {
                assert!(w[0].1 <= w[1].1, "successors should be time-ordered");
            }
        }
    }

    #[test]
    fn causal_search_successors_same_entity() {
        let index = setup_index(5, 10, 3);

        let results = index.causal_search(&[0.0, 0.0, 0.0], 5, TemporalFilter::All, 1.0, 0, 5);

        for r in &results {
            for &(succ_id, _) in &r.successors {
                assert_eq!(
                    index.entity_id(succ_id),
                    r.entity_id,
                    "successor should be same entity"
                );
            }
            for &(pred_id, _) in &r.predecessors {
                assert_eq!(
                    index.entity_id(pred_id),
                    r.entity_id,
                    "predecessor should be same entity"
                );
            }
        }
    }

    // ─── Hybrid search ──────────────────────────────────────────

    #[test]
    fn hybrid_search_beta_zero_matches_standard() {
        let index = setup_index(5, 20, 4);
        let query = [5.0f32, 0.05, 0.005, 0.001];

        let standard = index.search(&query, 10, TemporalFilter::All, 1.0, 0);
        let hybrid = index.hybrid_search(&query, 10, TemporalFilter::All, 1.0, 0.0, 0);

        // With beta=0, hybrid should produce the same top results as standard
        // (may differ slightly due to beam search implementation differences)
        assert_eq!(standard.len(), hybrid.len());

        // At least the top result should match
        assert_eq!(
            standard[0].0, hybrid[0].0,
            "top result should match between standard and hybrid (beta=0)"
        );
    }

    #[test]
    fn hybrid_search_with_temporal_edges() {
        let index = setup_index(3, 20, 4);
        let query = [0.5f32, 0.05, 0.005, 0.001];

        let results = index.hybrid_search(
            &query,
            10,
            TemporalFilter::All,
            1.0,
            0.5, // moderate temporal edge exploration
            5_000_000,
        );

        assert!(!results.is_empty());
        assert!(results.len() <= 10);

        // Verify all results are valid nodes
        for &(nid, score) in &results {
            assert!((nid as usize) < index.len());
            assert!(score >= 0.0);
            assert!(score.is_finite());
        }
    }

    #[test]
    fn hybrid_search_respects_temporal_filter() {
        let index = setup_index(3, 20, 4);
        let query = [1.0f32, 0.1, 0.01, 0.001];

        let results = index.hybrid_search(
            &query,
            10,
            TemporalFilter::Range(5_000_000, 15_000_000),
            1.0,
            0.5,
            10_000_000,
        );

        for &(nid, _) in &results {
            let ts = index.timestamp(nid);
            assert!(
                (5_000_000..=15_000_000).contains(&ts),
                "ts {ts} outside filter range"
            );
        }
    }

    // ─── TemporalIndexAccess trait ──────────────────────────────

    #[test]
    fn trait_search_works() {
        let index = setup_index(3, 10, 4);
        let trait_ref: &dyn cvx_core::TemporalIndexAccess = &index;

        let results = trait_ref.search_raw(&[0.0; 4], 5, TemporalFilter::All, 1.0, 0);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn trait_trajectory_works() {
        let index = setup_index(3, 10, 4);
        let trait_ref: &dyn cvx_core::TemporalIndexAccess = &index;

        let traj = trait_ref.trajectory(0, TemporalFilter::All);
        assert_eq!(traj.len(), 10);
    }

    // ─── from_temporal_hnsw migration ───────────────────────────

    #[test]
    fn from_temporal_hnsw_preserves_edges() {
        let config = HnswConfig::default();
        let mut hnsw = TemporalHnsw::new(config, L2Distance);

        for i in 0..10u64 {
            hnsw.insert(i % 3, i as i64 * 1000, &[i as f32, 0.0]);
        }

        let graph_index = TemporalGraphIndex::from_temporal_hnsw(hnsw);

        assert_eq!(graph_index.len(), 10);
        assert_eq!(graph_index.edges().len(), 10);

        // Verify temporal chains don't cross entities
        for nid in 0..10u32 {
            if let Some(succ) = graph_index.edges().successor(nid) {
                assert_eq!(
                    graph_index.entity_id(succ),
                    graph_index.entity_id(nid),
                    "edge from {nid} crosses entities after migration"
                );
            }
        }
    }

    // ─── Save/Load ──────────────────────────────────────────────

    #[test]
    fn save_load_roundtrip() {
        let index = setup_index(3, 10, 3);

        let dir = tempfile::tempdir().unwrap();
        index.save(dir.path()).unwrap();

        let loaded = TemporalGraphIndex::load(dir.path(), L2Distance).unwrap();

        assert_eq!(loaded.len(), 30);
        assert_eq!(loaded.edges().len(), 30);

        // Verify edges match
        for nid in 0..30u32 {
            assert_eq!(
                loaded.edges().successor(nid),
                index.edges().successor(nid),
                "successor mismatch at node {nid}"
            );
        }
    }

    // ─── Edge cases ─────────────────────────────────────────────

    #[test]
    fn empty_index() {
        let config = HnswConfig::default();
        let index = TemporalGraphIndex::new(config, L2Distance);

        assert!(index.is_empty());
        let results = index.hybrid_search(&[0.0; 3], 5, TemporalFilter::All, 1.0, 0.5, 0);
        assert!(results.is_empty());

        let causal = index.causal_search(&[0.0; 3], 5, TemporalFilter::All, 1.0, 0, 3);
        assert!(causal.is_empty());
    }

    #[test]
    fn single_point() {
        let config = HnswConfig::default();
        let mut index = TemporalGraphIndex::new(config, L2Distance);
        index.insert(1, 1000, &[1.0, 2.0, 3.0]);

        let causal = index.causal_search(&[1.0, 2.0, 3.0], 1, TemporalFilter::All, 1.0, 0, 5);
        assert_eq!(causal.len(), 1);
        assert!(causal[0].successors.is_empty());
        assert!(causal[0].predecessors.is_empty());
    }

    // ─── Delegated method coverage ───────────────────────────────

    #[test]
    fn config_and_ef_delegation() {
        let config = HnswConfig {
            m: 8,
            ef_construction: 100,
            ef_search: 50,
            ..Default::default()
        };
        let mut index = TemporalGraphIndex::new(config, L2Distance);
        assert_eq!(index.config().m, 8);
        assert_eq!(index.config().ef_construction, 100);

        index.set_ef_construction(150);
        assert_eq!(index.config().ef_construction, 150);

        index.set_ef_search(200);
        assert_eq!(index.config().ef_search, 200);
    }

    #[test]
    fn centering_delegation() {
        let config = HnswConfig::default();
        let mut index = TemporalGraphIndex::new(config, L2Distance);
        index.insert(1, 1000, &[2.0, 4.0]);
        index.insert(2, 2000, &[4.0, 6.0]);

        let centroid = index.compute_centroid().unwrap();
        assert!((centroid[0] - 3.0).abs() < 1e-6);

        index.set_centroid(vec![3.0, 5.0]);
        assert_eq!(index.centroid().unwrap(), &[3.0, 5.0]);

        let centered = index.centered_vector(&[5.0, 8.0]);
        assert!((centered[0] - 2.0).abs() < 1e-6);
        assert!((centered[1] - 3.0).abs() < 1e-6);

        index.clear_centroid();
        assert!(index.centroid().is_none());
    }

    #[test]
    fn reward_delegation() {
        let config = HnswConfig::default();
        let mut index = TemporalGraphIndex::new(config, L2Distance);
        let n0 = index.insert(1, 1000, &[1.0, 0.0]);
        let n1 = index.insert_with_reward(2, 2000, &[0.0, 1.0], 0.8);

        assert!(index.reward(n0).is_nan());
        assert!((index.reward(n1) - 0.8).abs() < 1e-6);

        index.set_reward(n0, 0.95);
        assert!((index.reward(n0) - 0.95).abs() < 1e-6);
    }

    #[test]
    fn search_with_reward_delegation() {
        let config = HnswConfig::default();
        let mut index = TemporalGraphIndex::new(config, L2Distance);
        for i in 0..10u64 {
            index.insert_with_reward(i, i as i64 * 1000, &[i as f32, 0.0, 0.0], i as f32 * 0.1);
        }

        let results =
            index.search_with_reward(&[7.0, 0.0, 0.0], 5, TemporalFilter::All, 1.0, 0, 0.5);
        assert!(!results.is_empty());
        for &(node_id, _) in &results {
            assert!(
                index.reward(node_id) >= 0.5,
                "node {node_id} reward {} < 0.5",
                index.reward(node_id)
            );
        }
    }

    #[test]
    fn region_delegation() {
        let config = HnswConfig {
            m: 4,
            ef_construction: 50,
            ef_search: 50,
            ..Default::default()
        };
        let mut index = TemporalGraphIndex::new(config, L2Distance);
        let mut rng = rand::rng();
        for i in 0..200u64 {
            let v: Vec<f32> = (0..8).map(|_| rand::Rng::random::<f32>(&mut rng)).collect();
            index.insert(i % 4, i as i64 * 1000, &v);
        }

        let regions = index.regions(1);
        assert!(!regions.is_empty());

        let assignments = index.region_assignments(1, TemporalFilter::All);
        let total: usize = assignments.values().map(|v| v.len()).sum();
        assert_eq!(total, 200);
    }

    #[test]
    fn scalar_quantization_delegation() {
        let config = HnswConfig::default();
        let mut index = TemporalGraphIndex::new(config, L2Distance);
        index.insert(1, 1000, &[1.0, 0.0]);
        index.insert(2, 2000, &[0.0, 1.0]);

        index.enable_scalar_quantization(-1.0, 1.0);
        let results = index.search(&[1.0, 0.0], 2, TemporalFilter::All, 1.0, 0);
        assert_eq!(results.len(), 2);

        index.disable_scalar_quantization();
        let results = index.search(&[1.0, 0.0], 2, TemporalFilter::All, 1.0, 0);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn insert_with_reward_creates_temporal_edges() {
        let config = HnswConfig::default();
        let mut index = TemporalGraphIndex::new(config, L2Distance);

        // Insert 5 steps for same entity with rewards
        for i in 0..5u32 {
            index.insert_with_reward(1, i as i64 * 100, &[i as f32, 0.0], i as f32 * 0.2);
        }

        // Check temporal edges exist
        let edges = index.edges();
        assert!(edges.successor(0).is_some());
        assert!(edges.predecessor(4).is_some());

        // Causal search should return continuations
        let results = index.causal_search(&[0.0, 0.0], 1, TemporalFilter::All, 1.0, 0, 3);
        assert_eq!(results.len(), 1);
        assert!(!results[0].successors.is_empty());
    }
}
