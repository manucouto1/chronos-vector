//! Temporal edge layer — successor/predecessor links between entity time points.
//!
//! Stores temporal ordering edges SEPARATELY from the HNSW graph, preserving
//! the vanilla HNSW structure while adding causal navigation capabilities.
//!
//! # Design
//!
//! For each entity's consecutive points (ordered by timestamp), this layer
//! maintains bidirectional temporal edges:
//! - `predecessor[node_i+1] = node_i`
//! - `successor[node_i] = node_i+1`
//!
//! Memory: 8 bytes/node (Option<u32> for successor + predecessor).

use serde::{Deserialize, Serialize};

/// Temporal edge layer tracking successor/predecessor relationships.
///
/// Node IDs must be registered sequentially (0, 1, 2, ...).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalEdgeLayer {
    /// node_id → temporal successor (next point for same entity, or None).
    successors: Vec<Option<u32>>,
    /// node_id → temporal predecessor (previous point for same entity, or None).
    predecessors: Vec<Option<u32>>,
}

impl TemporalEdgeLayer {
    /// Create a new empty temporal edge layer.
    pub fn new() -> Self {
        Self {
            successors: Vec::new(),
            predecessors: Vec::new(),
        }
    }

    /// Create with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            successors: Vec::with_capacity(capacity),
            predecessors: Vec::with_capacity(capacity),
        }
    }

    /// Register a new node and link it to the entity's previous latest node.
    ///
    /// `entity_last_node` should be the node_id of the entity's most recent
    /// point before this insert (None if this is the entity's first point).
    ///
    /// # Panics
    ///
    /// Panics if `node_id != self.len()` (nodes must be registered sequentially).
    pub fn register(&mut self, node_id: u32, entity_last_node: Option<u32>) {
        assert_eq!(
            node_id as usize,
            self.successors.len(),
            "nodes must be registered sequentially"
        );

        // New node has no successor yet
        self.successors.push(None);
        // New node's predecessor is the entity's previous latest
        self.predecessors.push(entity_last_node);

        // Link the previous node's successor to this new node
        if let Some(prev) = entity_last_node {
            if (prev as usize) < self.successors.len() {
                self.successors[prev as usize] = Some(node_id);
            }
        }
    }

    /// Get the temporal successor of a node (next point for same entity).
    pub fn successor(&self, node_id: u32) -> Option<u32> {
        self.successors.get(node_id as usize).copied().flatten()
    }

    /// Get the temporal predecessor of a node (previous point for same entity).
    pub fn predecessor(&self, node_id: u32) -> Option<u32> {
        self.predecessors.get(node_id as usize).copied().flatten()
    }

    /// Iterate temporal neighbors (predecessor and/or successor).
    pub fn temporal_neighbors(&self, node_id: u32) -> impl Iterator<Item = u32> + '_ {
        let pred = self.predecessor(node_id);
        let succ = self.successor(node_id);
        pred.into_iter().chain(succ)
    }

    /// Walk forward in time from a node, returning up to `max_steps` successors.
    pub fn walk_forward(&self, start: u32, max_steps: usize) -> Vec<u32> {
        let mut path = Vec::with_capacity(max_steps);
        let mut current = start;
        for _ in 0..max_steps {
            match self.successor(current) {
                Some(next) => {
                    path.push(next);
                    current = next;
                }
                None => break,
            }
        }
        path
    }

    /// Walk backward in time from a node, returning up to `max_steps` predecessors.
    pub fn walk_backward(&self, start: u32, max_steps: usize) -> Vec<u32> {
        let mut path = Vec::with_capacity(max_steps);
        let mut current = start;
        for _ in 0..max_steps {
            match self.predecessor(current) {
                Some(prev) => {
                    path.push(prev);
                    current = prev;
                }
                None => break,
            }
        }
        path
    }

    /// Number of registered nodes.
    pub fn len(&self) -> usize {
        self.successors.len()
    }

    /// Whether no nodes have been registered.
    pub fn is_empty(&self) -> bool {
        self.successors.is_empty()
    }

    /// Approximate memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        // Each Vec<Option<u32>>: len * 8 bytes (Option<u32> = 8 bytes due to alignment)
        self.successors.len() * 8 + self.predecessors.len() * 8
    }
}

impl Default for TemporalEdgeLayer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Basic operations ───────────────────────────────────────

    #[test]
    fn new_empty() {
        let layer = TemporalEdgeLayer::new();
        assert_eq!(layer.len(), 0);
        assert!(layer.is_empty());
    }

    #[test]
    fn register_single_node_no_predecessor() {
        let mut layer = TemporalEdgeLayer::new();
        layer.register(0, None);

        assert_eq!(layer.len(), 1);
        assert_eq!(layer.successor(0), None);
        assert_eq!(layer.predecessor(0), None);
    }

    // ─── Single entity chain ────────────────────────────────────

    #[test]
    fn single_entity_chain() {
        let mut layer = TemporalEdgeLayer::new();

        // Entity A: 5 sequential points
        layer.register(0, None);    // first point
        layer.register(1, Some(0)); // second
        layer.register(2, Some(1)); // third
        layer.register(3, Some(2)); // fourth
        layer.register(4, Some(3)); // fifth

        // Forward chain
        assert_eq!(layer.successor(0), Some(1));
        assert_eq!(layer.successor(1), Some(2));
        assert_eq!(layer.successor(2), Some(3));
        assert_eq!(layer.successor(3), Some(4));
        assert_eq!(layer.successor(4), None); // end

        // Backward chain
        assert_eq!(layer.predecessor(0), None); // start
        assert_eq!(layer.predecessor(1), Some(0));
        assert_eq!(layer.predecessor(2), Some(1));
        assert_eq!(layer.predecessor(3), Some(2));
        assert_eq!(layer.predecessor(4), Some(3));
    }

    // ─── Multi-entity isolation ─────────────────────────────────

    #[test]
    fn multi_entity_edges_dont_cross() {
        let mut layer = TemporalEdgeLayer::new();

        // Interleaved inserts from 2 entities:
        // Entity A: nodes 0, 2, 4
        // Entity B: nodes 1, 3, 5
        layer.register(0, None);       // A first
        layer.register(1, None);       // B first
        layer.register(2, Some(0));    // A second (links to A's last = 0)
        layer.register(3, Some(1));    // B second (links to B's last = 1)
        layer.register(4, Some(2));    // A third (links to A's last = 2)
        layer.register(5, Some(3));    // B third (links to B's last = 3)

        // Entity A chain: 0 → 2 → 4
        assert_eq!(layer.successor(0), Some(2));
        assert_eq!(layer.successor(2), Some(4));
        assert_eq!(layer.successor(4), None);
        assert_eq!(layer.predecessor(4), Some(2));
        assert_eq!(layer.predecessor(2), Some(0));

        // Entity B chain: 1 → 3 → 5
        assert_eq!(layer.successor(1), Some(3));
        assert_eq!(layer.successor(3), Some(5));
        assert_eq!(layer.successor(5), None);
        assert_eq!(layer.predecessor(5), Some(3));
        assert_eq!(layer.predecessor(3), Some(1));

        // No cross-entity edges
        assert_ne!(layer.successor(0), Some(1)); // A doesn't link to B
        assert_ne!(layer.successor(1), Some(2)); // B doesn't link to A
    }

    // ─── Walk operations ────────────────────────────────────────

    #[test]
    fn walk_forward_full_chain() {
        let mut layer = TemporalEdgeLayer::new();
        for i in 0..5u32 {
            layer.register(i, if i == 0 { None } else { Some(i - 1) });
        }

        let path = layer.walk_forward(0, 10);
        assert_eq!(path, vec![1, 2, 3, 4]);
    }

    #[test]
    fn walk_forward_limited() {
        let mut layer = TemporalEdgeLayer::new();
        for i in 0..10u32 {
            layer.register(i, if i == 0 { None } else { Some(i - 1) });
        }

        let path = layer.walk_forward(0, 3);
        assert_eq!(path, vec![1, 2, 3]);
    }

    #[test]
    fn walk_forward_from_end() {
        let mut layer = TemporalEdgeLayer::new();
        layer.register(0, None);
        layer.register(1, Some(0));

        let path = layer.walk_forward(1, 5);
        assert!(path.is_empty());
    }

    #[test]
    fn walk_backward_full_chain() {
        let mut layer = TemporalEdgeLayer::new();
        for i in 0..5u32 {
            layer.register(i, if i == 0 { None } else { Some(i - 1) });
        }

        let path = layer.walk_backward(4, 10);
        assert_eq!(path, vec![3, 2, 1, 0]);
    }

    #[test]
    fn walk_backward_limited() {
        let mut layer = TemporalEdgeLayer::new();
        for i in 0..10u32 {
            layer.register(i, if i == 0 { None } else { Some(i - 1) });
        }

        let path = layer.walk_backward(9, 3);
        assert_eq!(path, vec![8, 7, 6]);
    }

    // ─── Temporal neighbors ─────────────────────────────────────

    #[test]
    fn temporal_neighbors_middle_node() {
        let mut layer = TemporalEdgeLayer::new();
        layer.register(0, None);
        layer.register(1, Some(0));
        layer.register(2, Some(1));

        let neighbors: Vec<u32> = layer.temporal_neighbors(1).collect();
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&0)); // predecessor
        assert!(neighbors.contains(&2)); // successor
    }

    #[test]
    fn temporal_neighbors_first_node() {
        let mut layer = TemporalEdgeLayer::new();
        layer.register(0, None);
        layer.register(1, Some(0));

        let neighbors: Vec<u32> = layer.temporal_neighbors(0).collect();
        assert_eq!(neighbors.len(), 1); // only successor
        assert_eq!(neighbors[0], 1);
    }

    #[test]
    fn temporal_neighbors_last_node() {
        let mut layer = TemporalEdgeLayer::new();
        layer.register(0, None);
        layer.register(1, Some(0));

        let neighbors: Vec<u32> = layer.temporal_neighbors(1).collect();
        assert_eq!(neighbors.len(), 1); // only predecessor
        assert_eq!(neighbors[0], 0);
    }

    #[test]
    fn temporal_neighbors_isolated_node() {
        let mut layer = TemporalEdgeLayer::new();
        layer.register(0, None); // no predecessor, no successor

        let neighbors: Vec<u32> = layer.temporal_neighbors(0).collect();
        assert!(neighbors.is_empty());
    }

    // ─── Edge cases ─────────────────────────────────────────────

    #[test]
    fn out_of_bounds_queries() {
        let layer = TemporalEdgeLayer::new();
        assert_eq!(layer.successor(999), None);
        assert_eq!(layer.predecessor(999), None);
    }

    #[test]
    fn walk_from_nonexistent() {
        let layer = TemporalEdgeLayer::new();
        assert!(layer.walk_forward(999, 5).is_empty());
        assert!(layer.walk_backward(999, 5).is_empty());
    }

    #[test]
    #[should_panic(expected = "nodes must be registered sequentially")]
    fn register_out_of_order_panics() {
        let mut layer = TemporalEdgeLayer::new();
        layer.register(5, None); // should be 0
    }

    // ─── Memory ─────────────────────────────────────────────────

    #[test]
    fn memory_scales_linearly() {
        let mut layer = TemporalEdgeLayer::new();
        for i in 0..1000u32 {
            layer.register(i, if i == 0 { None } else { Some(i - 1) });
        }
        let bytes = layer.memory_bytes();
        // 1000 nodes * 8 bytes * 2 (succ + pred) = 16000
        assert_eq!(bytes, 16000);
    }

    // ─── Serialization ──────────────────────────────────────────

    #[test]
    fn serialize_deserialize_roundtrip() {
        let mut layer = TemporalEdgeLayer::new();
        layer.register(0, None);
        layer.register(1, Some(0));
        layer.register(2, Some(1));

        let bytes = postcard::to_allocvec(&layer).unwrap();
        let restored: TemporalEdgeLayer = postcard::from_bytes(&bytes).unwrap();

        assert_eq!(restored.len(), 3);
        assert_eq!(restored.successor(0), Some(1));
        assert_eq!(restored.successor(1), Some(2));
        assert_eq!(restored.predecessor(2), Some(1));
    }
}
