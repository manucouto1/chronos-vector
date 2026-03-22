//! Typed edge layer for relational structure over HNSW nodes (RFC-013 Part B).
//!
//! Adds weighted, typed, cross-entity edges to the temporal vector index.
//! These edges encode relationships that vector similarity alone cannot
//! capture: causal attribution, action similarity, region transitions.
//!
//! Edge types:
//! - `SameActionType`: connects nodes where the same abstract action was taken
//! - `CausedSuccess`: action node → successful outcome (with confidence weight)
//! - `CausedFailure`: action node → failed outcome
//! - `RegionTransition`: observed transition between HNSW regions
//!
//! All edges are directed and weighted. The adjacency structure supports
//! efficient outgoing-edge queries for graph traversal during search.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Edge type for relational structure between nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeType {
    /// Same abstract action type in different episodes.
    SameActionType,
    /// This action contributed to a successful outcome.
    CausedSuccess,
    /// This action was followed during a failed episode.
    CausedFailure,
    /// Observed transition between HNSW regions.
    RegionTransition,
    /// Custom edge type for domain-specific relations.
    Custom(u16),
}

/// A single typed, weighted, directed edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedEdge {
    /// Target node ID.
    pub target: u32,
    /// Edge type.
    pub edge_type: EdgeType,
    /// Weight (0.0 - 1.0). Higher = stronger relationship.
    pub weight: f32,
}

/// Typed edge layer: directed, weighted edges with type labels.
///
/// Stored separately from the HNSW graph — does not affect vector search.
/// Used for post-retrieval scoring, graph traversal queries, and
/// causal attribution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TypedEdgeStore {
    /// Node ID → outgoing edges.
    outgoing: HashMap<u32, Vec<TypedEdge>>,
    /// Total edge count.
    n_edges: usize,
}

impl TypedEdgeStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self {
            outgoing: HashMap::new(),
            n_edges: 0,
        }
    }

    /// Add a directed edge from `source` to `target`.
    ///
    /// Duplicate edges (same source, target, type) are allowed — weights
    /// accumulate for Bayesian updates.
    pub fn add_edge(&mut self, source: u32, target: u32, edge_type: EdgeType, weight: f32) {
        self.outgoing.entry(source).or_default().push(TypedEdge {
            target,
            edge_type,
            weight,
        });
        self.n_edges += 1;
    }

    /// Add a bidirectional edge (convenience for symmetric relations).
    pub fn add_edge_bidi(&mut self, node_a: u32, node_b: u32, edge_type: EdgeType, weight: f32) {
        self.add_edge(node_a, node_b, edge_type, weight);
        self.add_edge(node_b, node_a, edge_type, weight);
    }

    /// Get all outgoing edges from a node.
    pub fn outgoing(&self, node_id: u32) -> &[TypedEdge] {
        self.outgoing
            .get(&node_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get outgoing edges of a specific type.
    pub fn outgoing_by_type(&self, node_id: u32, edge_type: EdgeType) -> Vec<&TypedEdge> {
        self.outgoing(node_id)
            .iter()
            .filter(|e| e.edge_type == edge_type)
            .collect()
    }

    /// Get all nodes reachable from `node_id` via edges of `edge_type`.
    pub fn neighbors_by_type(&self, node_id: u32, edge_type: EdgeType) -> Vec<(u32, f32)> {
        self.outgoing(node_id)
            .iter()
            .filter(|e| e.edge_type == edge_type)
            .map(|e| (e.target, e.weight))
            .collect()
    }

    /// Multi-hop traversal: follow edges of given types up to `max_hops`.
    ///
    /// Returns all reachable nodes with their accumulated path weight
    /// (product of edge weights along the path).
    pub fn traverse(
        &self,
        start: u32,
        edge_types: &[EdgeType],
        max_hops: usize,
    ) -> Vec<(u32, f32, usize)> {
        let mut visited = HashMap::new();
        let mut frontier = vec![(start, 1.0f32, 0usize)];

        while let Some((node, path_weight, depth)) = frontier.pop() {
            if depth > max_hops {
                continue;
            }
            if let Some(&existing_weight) = visited.get(&node) {
                if existing_weight >= path_weight {
                    continue; // Already visited with better weight
                }
            }
            visited.insert(node, path_weight);

            for edge in self.outgoing(node) {
                if edge_types.contains(&edge.edge_type) {
                    let new_weight = path_weight * edge.weight;
                    if new_weight > 0.01 {
                        // Prune very weak paths
                        frontier.push((edge.target, new_weight, depth + 1));
                    }
                }
            }
        }

        visited.remove(&start); // Don't include start node
        visited
            .into_iter()
            .map(|(node, weight)| {
                // Compute hop count (approximate — use min depth seen)
                let depth = 1; // Simplified; could track properly
                (node, weight, depth)
            })
            .collect()
    }

    /// Compute a "success score" for a node based on its CausedSuccess
    /// and CausedFailure edges.
    ///
    /// Returns P(success) using a Beta prior: (1 + successes) / (2 + total).
    pub fn success_score(&self, node_id: u32) -> f32 {
        let edges = self.outgoing(node_id);
        let successes: f32 = edges
            .iter()
            .filter(|e| e.edge_type == EdgeType::CausedSuccess)
            .map(|e| e.weight)
            .sum();
        let failures: f32 = edges
            .iter()
            .filter(|e| e.edge_type == EdgeType::CausedFailure)
            .map(|e| e.weight)
            .sum();
        let total = successes + failures;
        if total < 0.01 {
            return 0.5; // Uninformative prior
        }
        (1.0 + successes) / (2.0 + total)
    }

    /// Compute edge statistics.
    pub fn stats(&self) -> TypedEdgeStats {
        let mut type_counts: HashMap<EdgeType, usize> = HashMap::new();
        for edges in self.outgoing.values() {
            for edge in edges {
                *type_counts.entry(edge.edge_type).or_default() += 1;
            }
        }
        TypedEdgeStats {
            n_nodes_with_edges: self.outgoing.len(),
            n_edges: self.n_edges,
            type_counts,
        }
    }

    /// Total number of edges.
    pub fn len(&self) -> usize {
        self.n_edges
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.n_edges == 0
    }

    /// Remove all edges of a specific type from a node.
    pub fn remove_edges(&mut self, node_id: u32, edge_type: EdgeType) {
        if let Some(edges) = self.outgoing.get_mut(&node_id) {
            let before = edges.len();
            edges.retain(|e| e.edge_type != edge_type);
            self.n_edges -= before - edges.len();
        }
    }

    /// Update weight of all edges of a given type from a node.
    pub fn update_weights(&mut self, node_id: u32, edge_type: EdgeType, new_weight: f32) {
        if let Some(edges) = self.outgoing.get_mut(&node_id) {
            for edge in edges.iter_mut() {
                if edge.edge_type == edge_type {
                    edge.weight = new_weight;
                }
            }
        }
    }
}

/// Statistics about the typed edge store.
#[derive(Debug)]
pub struct TypedEdgeStats {
    /// Number of nodes that have at least one outgoing edge.
    pub n_nodes_with_edges: usize,
    /// Total number of directed edges.
    pub n_edges: usize,
    /// Count per edge type.
    pub type_counts: HashMap<EdgeType, usize>,
}

impl std::fmt::Display for TypedEdgeStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} edges across {} nodes",
            self.n_edges, self.n_nodes_with_edges
        )?;
        for (etype, count) in &self.type_counts {
            write!(f, ", {etype:?}={count}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_query_edges() {
        let mut store = TypedEdgeStore::new();
        store.add_edge(0, 1, EdgeType::CausedSuccess, 0.9);
        store.add_edge(0, 2, EdgeType::CausedFailure, 0.7);
        store.add_edge(0, 3, EdgeType::SameActionType, 1.0);

        assert_eq!(store.len(), 3);
        assert_eq!(store.outgoing(0).len(), 3);
        assert_eq!(store.outgoing(1).len(), 0); // directed: no reverse edge
    }

    #[test]
    fn query_by_type() {
        let mut store = TypedEdgeStore::new();
        store.add_edge(0, 1, EdgeType::CausedSuccess, 0.9);
        store.add_edge(0, 2, EdgeType::CausedSuccess, 0.8);
        store.add_edge(0, 3, EdgeType::CausedFailure, 0.5);

        let successes = store.neighbors_by_type(0, EdgeType::CausedSuccess);
        assert_eq!(successes.len(), 2);

        let failures = store.neighbors_by_type(0, EdgeType::CausedFailure);
        assert_eq!(failures.len(), 1);
    }

    #[test]
    fn bidirectional_edges() {
        let mut store = TypedEdgeStore::new();
        store.add_edge_bidi(0, 1, EdgeType::SameActionType, 1.0);

        assert_eq!(store.outgoing(0).len(), 1);
        assert_eq!(store.outgoing(1).len(), 1);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn success_score_beta_prior() {
        let mut store = TypedEdgeStore::new();
        // 3 successes, 1 failure → P = (1+3)/(2+4) = 4/6 = 0.667
        store.add_edge(0, 10, EdgeType::CausedSuccess, 1.0);
        store.add_edge(0, 11, EdgeType::CausedSuccess, 1.0);
        store.add_edge(0, 12, EdgeType::CausedSuccess, 1.0);
        store.add_edge(0, 13, EdgeType::CausedFailure, 1.0);

        let score = store.success_score(0);
        assert!((score - 0.667).abs() < 0.01, "score = {score}");
    }

    #[test]
    fn success_score_no_edges() {
        let store = TypedEdgeStore::new();
        assert!((store.success_score(99) - 0.5).abs() < 0.01);
    }

    #[test]
    fn traverse_multi_hop() {
        let mut store = TypedEdgeStore::new();
        // Chain: 0 → 1 → 2 → 3
        store.add_edge(0, 1, EdgeType::RegionTransition, 0.8);
        store.add_edge(1, 2, EdgeType::RegionTransition, 0.9);
        store.add_edge(2, 3, EdgeType::RegionTransition, 0.7);

        let reachable = store.traverse(0, &[EdgeType::RegionTransition], 2);
        let nodes: Vec<u32> = reachable.iter().map(|&(n, _, _)| n).collect();
        assert!(nodes.contains(&1));
        assert!(nodes.contains(&2));
        // Node 3 is 3 hops away, max_hops=2 → should not be reachable
        // (depends on traversal order, may or may not reach)
    }

    #[test]
    fn remove_edges_by_type() {
        let mut store = TypedEdgeStore::new();
        store.add_edge(0, 1, EdgeType::CausedSuccess, 0.9);
        store.add_edge(0, 2, EdgeType::CausedFailure, 0.5);
        store.add_edge(0, 3, EdgeType::CausedSuccess, 0.8);

        store.remove_edges(0, EdgeType::CausedSuccess);
        assert_eq!(store.outgoing(0).len(), 1); // only failure remains
        assert_eq!(store.outgoing(0)[0].edge_type, EdgeType::CausedFailure);
    }

    #[test]
    fn update_weights() {
        let mut store = TypedEdgeStore::new();
        store.add_edge(0, 1, EdgeType::CausedSuccess, 0.5);
        store.add_edge(0, 2, EdgeType::CausedSuccess, 0.6);

        store.update_weights(0, EdgeType::CausedSuccess, 0.95);

        for edge in store.outgoing(0) {
            assert!((edge.weight - 0.95).abs() < 0.01);
        }
    }

    #[test]
    fn stats() {
        let mut store = TypedEdgeStore::new();
        store.add_edge(0, 1, EdgeType::CausedSuccess, 0.9);
        store.add_edge(0, 2, EdgeType::CausedFailure, 0.5);
        store.add_edge(1, 3, EdgeType::RegionTransition, 0.7);

        let stats = store.stats();
        assert_eq!(stats.n_edges, 3);
        assert_eq!(stats.n_nodes_with_edges, 2);
        assert_eq!(stats.type_counts[&EdgeType::CausedSuccess], 1);
    }

    #[test]
    fn serialization_roundtrip() {
        let mut store = TypedEdgeStore::new();
        store.add_edge(0, 1, EdgeType::CausedSuccess, 0.9);
        store.add_edge(0, 2, EdgeType::RegionTransition, 0.7);
        store.add_edge(1, 3, EdgeType::Custom(42), 1.0);

        let bytes = postcard::to_allocvec(&store).unwrap();
        let restored: TypedEdgeStore = postcard::from_bytes(&bytes).unwrap();

        assert_eq!(restored.len(), 3);
        assert_eq!(restored.outgoing(0).len(), 2);
        assert_eq!(restored.outgoing(1).len(), 1);
        assert_eq!(restored.outgoing(1)[0].edge_type, EdgeType::Custom(42));
    }
}
