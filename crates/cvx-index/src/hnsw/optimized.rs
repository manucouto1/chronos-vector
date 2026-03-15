//! Advanced HNSW optimizations.
//!
//! ## Heuristic Neighbor Selection (Malkov §4.2)
//!
//! Instead of keeping the M closest neighbors, selects neighbors that provide
//! good graph connectivity by preferring diverse directions over raw proximity.
//!
//! ## Time-Decay Edge Weights
//!
//! Edge weights decay exponentially with age: `w(e, t) = w0 * exp(-λ * age)`.
//! During search, decayed weights penalize stale connections.
//!
//! ## Backup Neighbors
//!
//! Each node maintains a secondary neighbor list used when primary neighbors
//! are removed (e.g., node expiration in streaming scenarios).
//!
//! ## Index Persistence
//!
//! Serialize/deserialize the HNSW graph for crash recovery and snapshots.

use cvx_core::DistanceMetric;

use super::HnswGraph;

/// Select neighbors using the heuristic from Malkov & Yashunin (2018) §4.2.
///
/// From a candidate set, greedily selects neighbors that are closer to the
/// target than to any already-selected neighbor. This produces a more diverse
/// neighbor set with better graph connectivity.
///
/// Returns at most `m` neighbor IDs.
/// Trait for accessing node vectors by ID, avoiding full-collection clones.
pub trait NodeVectors {
    /// Get the vector for a given node ID.
    fn get_vector(&self, id: u32) -> &[f32];
}

/// Implementation for a slice of Vec<f32>.
impl NodeVectors for [Vec<f32>] {
    fn get_vector(&self, id: u32) -> &[f32] {
        &self[id as usize]
    }
}

/// Select neighbors using the heuristic from Malkov & Yashunin (2018) §4.2.
///
/// Greedily selects neighbors closer to target than to any already-selected
/// neighbor, producing a diverse set with better graph connectivity.
///
/// Returns at most `m` neighbor IDs.
pub fn select_neighbors_heuristic<D: DistanceMetric, N: NodeVectors + ?Sized>(
    metric: &D,
    candidates: &[(u32, f32)], // (node_id, distance_to_target)
    node_vectors: &N,
    m: usize,
    extend_candidates: bool,
) -> Vec<u32> {
    if candidates.is_empty() || m == 0 {
        return Vec::new();
    }

    // Working set sorted by distance (closest first)
    let mut working: Vec<(u32, f32)> = candidates.to_vec();
    working.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut selected: Vec<u32> = Vec::with_capacity(m);
    let mut selected_vectors: Vec<&[f32]> = Vec::with_capacity(m);

    for &(cand_id, cand_dist) in &working {
        if selected.len() >= m {
            break;
        }

        // Check if candidate is closer to target than to any selected neighbor
        let cand_vec = node_vectors.get_vector(cand_id);
        let is_good = selected_vectors.iter().all(|&sel_vec| {
            let dist_to_selected = metric.distance(cand_vec, sel_vec);
            cand_dist <= dist_to_selected
        });

        if is_good || (extend_candidates && selected.len() < m / 2) {
            selected.push(cand_id);
            selected_vectors.push(cand_vec);
        }
    }

    // If we didn't fill up, add closest remaining candidates
    if selected.len() < m {
        for &(cand_id, _) in &working {
            if selected.len() >= m {
                break;
            }
            if !selected.contains(&cand_id) {
                selected.push(cand_id);
            }
        }
    }

    selected
}

/// Time-decay weight for an edge.
///
/// Returns `exp(-lambda * age)` where age is in the same units as timestamps.
pub fn time_decay_weight(edge_timestamp: i64, current_time: i64, lambda: f64) -> f32 {
    let age = (current_time - edge_timestamp).max(0) as f64;
    (-lambda * age).exp() as f32
}

/// Apply time-decay to a distance score.
///
/// The effective distance increases for stale edges:
/// `d_effective = d_raw / decay_weight`
pub fn decay_adjusted_distance(
    raw_distance: f32,
    edge_timestamp: i64,
    current_time: i64,
    lambda: f64,
) -> f32 {
    let weight = time_decay_weight(edge_timestamp, current_time, lambda);
    if weight > 1e-10 {
        raw_distance / weight
    } else {
        f32::INFINITY
    }
}

/// Backup neighbor storage for handling node expiration.
#[derive(Debug, Clone)]
pub struct BackupNeighbors {
    /// Primary neighbors (from HNSW construction).
    pub primary: Vec<u32>,
    /// Backup neighbors (next-best candidates).
    pub backup: Vec<u32>,
}

impl BackupNeighbors {
    /// Create with primary and backup lists.
    pub fn new(primary: Vec<u32>, backup: Vec<u32>) -> Self {
        Self { primary, backup }
    }

    /// Get active neighbors, replacing expired primaries with backups.
    pub fn active_neighbors(&self, is_expired: &dyn Fn(u32) -> bool) -> Vec<u32> {
        let mut active: Vec<u32> = self
            .primary
            .iter()
            .copied()
            .filter(|&id| !is_expired(id))
            .collect();

        // Fill with backups if primaries were expired
        let needed = self.primary.len().saturating_sub(active.len());
        for &b in self.backup.iter().take(needed) {
            if !is_expired(b) && !active.contains(&b) {
                active.push(b);
            }
        }

        active
    }
}

/// Serialized HNSW graph for persistence.
///
/// Contains all data needed to reconstruct the graph.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SerializedGraph {
    /// Configuration.
    pub m: usize,
    /// Number of nodes.
    pub num_nodes: usize,
    /// Entry point node ID.
    pub entry_point: Option<u32>,
    /// Maximum level in the graph.
    pub max_level: usize,
    /// All vectors (flattened: `[node_0_dim_0, node_0_dim_1, ..., node_1_dim_0, ...]`).
    pub vectors: Vec<f32>,
    /// Vector dimensionality.
    pub dim: usize,
    /// Neighbor lists: `[node_0_level_0, node_0_level_1, ..., node_1_level_0, ...]`
    /// Encoded as: for each node, number of levels, then for each level, count + neighbor IDs.
    pub neighbors: Vec<u32>,
}

/// Serialize an HNSW graph to a portable format.
pub fn serialize_graph<D: DistanceMetric>(graph: &HnswGraph<D>) -> SerializedGraph {
    let num_nodes = graph.len();
    let dim = if num_nodes > 0 {
        graph.vector(0).len()
    } else {
        0
    };

    let mut vectors = Vec::with_capacity(num_nodes * dim);
    let mut neighbors = Vec::new();

    for i in 0..num_nodes {
        let id = i as u32;
        vectors.extend_from_slice(graph.vector(id));

        // Encode neighbor lists for this node
        let node_neighbors = graph.all_neighbors(id);
        neighbors.push(node_neighbors.len() as u32); // number of levels
        for level_neighbors in &node_neighbors {
            neighbors.push(level_neighbors.len() as u32); // count
            neighbors.extend(level_neighbors.iter());
        }
    }

    SerializedGraph {
        m: graph.config().m,
        num_nodes,
        entry_point: graph.entry_point(),
        max_level: graph.max_level(),
        vectors,
        dim,
        neighbors,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::L2Distance;

    // ─── Heuristic neighbor selection ───────────────────────────────

    #[test]
    fn heuristic_selects_diverse_neighbors() {
        let metric = L2Distance;
        let target = vec![0.0, 0.0];
        let node_vectors = vec![
            vec![1.0, 0.0],   // 0: right, dist=1.0
            vec![0.95, 0.05], // 1: almost same as 0, dist≈0.91
            vec![0.0, 1.0],   // 2: up, dist=1.0
            vec![-1.0, 0.0],  // 3: left, dist=1.0
        ];

        let candidates = vec![
            (0, metric.distance(&target, &node_vectors[0])),
            (1, metric.distance(&target, &node_vectors[1])),
            (2, metric.distance(&target, &node_vectors[2])),
            (3, metric.distance(&target, &node_vectors[3])),
        ];

        let selected =
            select_neighbors_heuristic(&metric, &candidates, node_vectors.as_slice(), 3, false);
        assert_eq!(selected.len(), 3);

        // The heuristic should exclude one of {0, 1} since they're in the same direction
        // Node 1 is closest, so it gets selected first. Then 2 and 3 add diversity.
        assert!(
            selected.contains(&2),
            "should include node 2 (up direction)"
        );
        assert!(
            selected.contains(&3),
            "should include node 3 (left direction)"
        );
    }

    #[test]
    fn heuristic_empty_candidates() {
        let metric = L2Distance;
        let empty: &[Vec<f32>] = &[];
        let result = select_neighbors_heuristic(&metric, &[], empty, 5, false);
        assert!(result.is_empty());
    }

    #[test]
    fn heuristic_fewer_candidates_than_m() {
        let metric = L2Distance;
        let vectors = vec![vec![1.0], vec![2.0]];
        let candidates = vec![(0, 1.0), (1, 2.0)];
        let selected =
            select_neighbors_heuristic(&metric, &candidates, vectors.as_slice(), 5, false);
        assert_eq!(selected.len(), 2); // only 2 available
    }

    // ─── Time-decay weights ─────────────────────────────────────────

    #[test]
    fn decay_weight_same_time_is_one() {
        let w = time_decay_weight(1000, 1000, 0.001);
        assert!((w - 1.0).abs() < 1e-6);
    }

    #[test]
    fn decay_weight_decreases_with_age() {
        let w1 = time_decay_weight(900, 1000, 0.01);
        let w2 = time_decay_weight(500, 1000, 0.01);
        assert!(w1 > w2, "newer edge should have higher weight");
    }

    #[test]
    fn decay_weight_high_lambda_fast_decay() {
        let w_slow = time_decay_weight(0, 1000, 0.001);
        let w_fast = time_decay_weight(0, 1000, 0.01);
        assert!(w_slow > w_fast, "higher lambda should decay faster");
    }

    #[test]
    fn decay_adjusted_distance_increases_with_age() {
        let d_new = decay_adjusted_distance(1.0, 900, 1000, 0.01);
        let d_old = decay_adjusted_distance(1.0, 0, 1000, 0.01);
        assert!(
            d_old > d_new,
            "older edges should have larger effective distance"
        );
    }

    // ─── Backup neighbors ───────────────────────────────────────────

    #[test]
    fn backup_fills_expired_primary() {
        let bn = BackupNeighbors::new(vec![1, 2, 3], vec![10, 11]);

        // Node 2 expired
        let active = bn.active_neighbors(&|id| id == 2);
        assert_eq!(active.len(), 3); // 1, 3 from primary + 10 from backup
        assert!(active.contains(&1));
        assert!(active.contains(&3));
        assert!(active.contains(&10));
        assert!(!active.contains(&2));
    }

    #[test]
    fn backup_no_expired() {
        let bn = BackupNeighbors::new(vec![1, 2, 3], vec![10]);
        let active = bn.active_neighbors(&|_| false);
        assert_eq!(active, vec![1, 2, 3]);
    }

    #[test]
    fn backup_all_expired_uses_all_backups() {
        let bn = BackupNeighbors::new(vec![1, 2], vec![10, 11, 12]);
        let active = bn.active_neighbors(&|id| id <= 2);
        assert_eq!(active, vec![10, 11]);
    }

    // ─── Serialization ──────────────────────────────────────────────

    #[test]
    fn serialize_empty_graph() {
        let config = super::super::HnswConfig::default();
        let graph = HnswGraph::new(config, L2Distance);
        let serialized = serialize_graph(&graph);
        assert_eq!(serialized.num_nodes, 0);
        assert_eq!(serialized.entry_point, None);
    }

    #[test]
    fn serialize_graph_preserves_data() {
        let config = super::super::HnswConfig {
            m: 8,
            ef_construction: 100,
            ef_search: 50,
            ..Default::default()
        };
        let mut graph = HnswGraph::new(config, L2Distance);

        for i in 0..50u32 {
            graph.insert(i, &[i as f32, (50 - i) as f32]);
        }

        let serialized = serialize_graph(&graph);
        assert_eq!(serialized.num_nodes, 50);
        assert_eq!(serialized.dim, 2);
        assert_eq!(serialized.vectors.len(), 100); // 50 * 2
        assert!(serialized.entry_point.is_some());
        assert_eq!(serialized.m, 8);

        // Verify first vector
        assert!((serialized.vectors[0] - 0.0).abs() < 1e-6);
        assert!((serialized.vectors[1] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn serialized_graph_is_serde_roundtrip() {
        let config = super::super::HnswConfig {
            m: 4,
            ..Default::default()
        };
        let mut graph = HnswGraph::new(config, L2Distance);
        for i in 0..10u32 {
            graph.insert(i, &[i as f32]);
        }

        let serialized = serialize_graph(&graph);
        let json = serde_json::to_string(&serialized).unwrap();
        let recovered: SerializedGraph = serde_json::from_str(&json).unwrap();

        assert_eq!(recovered.num_nodes, 10);
        assert_eq!(recovered.vectors, serialized.vectors);
        assert_eq!(recovered.neighbors, serialized.neighbors);
    }
}
