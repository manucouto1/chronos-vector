//! In-memory metadata store for HNSW nodes.
//!
//! Stores `HashMap<String, String>` per node_id, enabling metadata filtering
//! on search results without modifying the HNSW graph structure.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use cvx_core::types::metadata_filter::MetadataFilter;

/// Dense metadata store: node_id → metadata map.
///
/// Memory: ~200 bytes per node with 5 metadata fields (String overhead).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetadataStore {
    /// node_id → metadata. Empty HashMap for nodes without metadata.
    entries: Vec<HashMap<String, String>>,
}

impl MetadataStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Register metadata for a new node (must be called in order).
    pub fn push(&mut self, metadata: HashMap<String, String>) {
        self.entries.push(metadata);
    }

    /// Register an empty metadata entry.
    pub fn push_empty(&mut self) {
        self.entries.push(HashMap::new());
    }

    /// Get metadata for a node.
    pub fn get(&self, node_id: u32) -> &HashMap<String, String> {
        static EMPTY: std::sync::LazyLock<HashMap<String, String>> =
            std::sync::LazyLock::new(HashMap::new);
        self.entries.get(node_id as usize).unwrap_or(&EMPTY)
    }

    /// Check if a node passes a metadata filter.
    pub fn matches(&self, node_id: u32, filter: &MetadataFilter) -> bool {
        if filter.is_empty() {
            return true;
        }
        filter.matches(self.get(node_id))
    }

    /// Filter a list of (node_id, score) results by metadata.
    pub fn filter_results(
        &self,
        results: &[(u32, f32)],
        filter: &MetadataFilter,
    ) -> Vec<(u32, f32)> {
        if filter.is_empty() {
            return results.to_vec();
        }
        results
            .iter()
            .filter(|(nid, _)| self.matches(*nid, filter))
            .copied()
            .collect()
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_get() {
        let mut store = MetadataStore::new();
        let mut meta = HashMap::new();
        meta.insert("reward".into(), "0.8".into());
        meta.insert("step_index".into(), "0".into());
        store.push(meta);
        store.push_empty();

        assert_eq!(store.get(0).get("reward").unwrap(), "0.8");
        assert!(store.get(1).is_empty());
        assert!(store.get(999).is_empty()); // out of bounds → empty
    }

    #[test]
    fn filter_results_by_metadata() {
        let mut store = MetadataStore::new();
        for i in 0..5u32 {
            let mut m = HashMap::new();
            m.insert("reward".into(), format!("{}", i as f64 * 0.2));
            m.insert("step_index".into(), format!("{}", i));
            store.push(m);
        }

        let results: Vec<(u32, f32)> = (0..5).map(|i| (i, i as f32 * 0.1)).collect();

        // Filter: reward >= 0.5 → nodes 3 (0.6) and 4 (0.8)
        let filter = MetadataFilter::new().gte("reward", 0.5);
        let filtered = store.filter_results(&results, &filter);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].0, 3);
        assert_eq!(filtered[1].0, 4);
    }

    #[test]
    fn empty_filter_passes_all() {
        let mut store = MetadataStore::new();
        store.push_empty();
        store.push_empty();

        let results = vec![(0u32, 0.1f32), (1, 0.2)];
        let filtered = store.filter_results(&results, &MetadataFilter::new());
        assert_eq!(filtered.len(), 2);
    }
}
