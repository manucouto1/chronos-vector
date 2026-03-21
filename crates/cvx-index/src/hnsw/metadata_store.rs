//! In-memory metadata store for HNSW nodes.
//!
//! Stores `HashMap<String, String>` per node_id, enabling metadata filtering
//! on search results without modifying the HNSW graph structure.

use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use cvx_core::types::metadata_filter::{MetadataFilter, MetadataPredicate};

/// Dense metadata store with inverted index for O(1) pre-filtering.
///
/// Two data structures:
/// - `entries`: node_id → metadata map (for retrieval)
/// - `inverted`: key → value → RoaringBitmap of node_ids (for filtering)
///
/// The inverted index supports exact-match pre-filtering during HNSW
/// traversal, replacing the O(4k) post-filter with O(1) bitmap lookups.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetadataStore {
    /// node_id → metadata. Empty HashMap for nodes without metadata.
    entries: Vec<HashMap<String, String>>,
    /// Inverted index: key → value → bitmap of matching node_ids.
    /// Only populated for exact string values (not numeric ranges).
    #[serde(default)]
    inverted: HashMap<String, HashMap<String, RoaringBitmap>>,
}

impl MetadataStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            inverted: HashMap::new(),
        }
    }

    /// Register metadata for a new node (must be called in order).
    pub fn push(&mut self, metadata: HashMap<String, String>) {
        let node_id = self.entries.len() as u32;
        // Update inverted index
        for (key, value) in &metadata {
            self.inverted
                .entry(key.clone())
                .or_default()
                .entry(value.clone())
                .or_default()
                .insert(node_id);
        }
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

    /// Build a RoaringBitmap of node_ids matching the metadata filter.
    ///
    /// For `Equals` predicates: uses the inverted index for O(1) lookup.
    /// For other predicates (Gte, Lte, Contains, Exists): falls back to
    /// scanning entries.
    ///
    /// Multiple predicates are AND-combined (intersection).
    pub fn build_filter_bitmap(&self, filter: &MetadataFilter) -> RoaringBitmap {
        if filter.is_empty() {
            // No filter → all nodes match
            let mut all = RoaringBitmap::new();
            for i in 0..self.entries.len() as u32 {
                all.insert(i);
            }
            return all;
        }

        let mut result: Option<RoaringBitmap> = None;

        for (field, predicate) in &filter.predicates {
            let bitmap = match predicate {
                MetadataPredicate::Equals(value) => {
                    // Fast path: use inverted index
                    self.inverted
                        .get(field)
                        .and_then(|values| values.get(value))
                        .cloned()
                        .unwrap_or_default()
                }
                _ => {
                    // Slow path: scan entries
                    let mut bm = RoaringBitmap::new();
                    for (i, entry) in self.entries.iter().enumerate() {
                        if predicate.matches(entry.get(field)) {
                            bm.insert(i as u32);
                        }
                    }
                    bm
                }
            };

            result = Some(match result {
                Some(existing) => existing & bitmap, // AND intersection
                None => bitmap,
            });
        }

        result.unwrap_or_default()
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
            m.insert("step_index".into(), format!("{i}"));
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

    // ─── Inverted index tests ────────────────────────────────────

    #[test]
    fn inverted_index_built_on_push() {
        let mut store = MetadataStore::new();
        let mut m = HashMap::new();
        m.insert("goal".into(), "clean".into());
        m.insert("room".into(), "kitchen".into());
        store.push(m);

        let mut m2 = HashMap::new();
        m2.insert("goal".into(), "clean".into());
        m2.insert("room".into(), "bedroom".into());
        store.push(m2);

        let mut m3 = HashMap::new();
        m3.insert("goal".into(), "find".into());
        store.push(m3);

        // Check inverted index
        let goal_clean = &store.inverted["goal"]["clean"];
        assert!(goal_clean.contains(0));
        assert!(goal_clean.contains(1));
        assert!(!goal_clean.contains(2));

        let goal_find = &store.inverted["goal"]["find"];
        assert!(goal_find.contains(2));
        assert_eq!(goal_find.len(), 1);
    }

    #[test]
    fn build_filter_bitmap_equals_uses_inverted() {
        let mut store = MetadataStore::new();
        for i in 0..100u32 {
            let mut m = HashMap::new();
            m.insert(
                "goal".into(),
                if i % 3 == 0 { "clean" } else { "find" }.into(),
            );
            store.push(m);
        }

        let filter = MetadataFilter::new().equals("goal", "clean");
        let bitmap = store.build_filter_bitmap(&filter);
        assert_eq!(bitmap.len(), 34); // 0,3,6,...,99 → 34 values

        for id in bitmap.iter() {
            assert_eq!(id % 3, 0);
        }
    }

    #[test]
    fn build_filter_bitmap_gte_scans() {
        let mut store = MetadataStore::new();
        for i in 0..10u32 {
            let mut m = HashMap::new();
            m.insert("reward".into(), format!("{}", i as f64 * 0.1));
            store.push(m);
        }

        let filter = MetadataFilter::new().gte("reward", 0.5);
        let bitmap = store.build_filter_bitmap(&filter);
        // reward >= 0.5: nodes 5(0.5),6(0.6),7(0.7),8(0.8),9(0.9)
        assert_eq!(bitmap.len(), 5);
        for id in bitmap.iter() {
            assert!(id >= 5);
        }
    }

    #[test]
    fn build_filter_bitmap_combined_and() {
        let mut store = MetadataStore::new();
        for i in 0..20u32 {
            let mut m = HashMap::new();
            m.insert(
                "goal".into(),
                if i % 2 == 0 { "clean" } else { "find" }.into(),
            );
            m.insert("reward".into(), format!("{}", i as f64 * 0.05));
            store.push(m);
        }

        // goal=clean AND reward >= 0.5
        let filter = MetadataFilter::new()
            .equals("goal", "clean")
            .gte("reward", 0.5);
        let bitmap = store.build_filter_bitmap(&filter);

        // goal=clean: 0,2,4,6,8,10,12,14,16,18
        // reward >= 0.5: 10(0.5),11,12,...,19
        // AND: 10,12,14,16,18
        assert_eq!(bitmap.len(), 5);
        for id in bitmap.iter() {
            assert!(id >= 10);
            assert_eq!(id % 2, 0);
        }
    }

    #[test]
    fn build_filter_bitmap_empty_returns_all() {
        let mut store = MetadataStore::new();
        for _ in 0..10 {
            store.push_empty();
        }
        let bitmap = store.build_filter_bitmap(&MetadataFilter::new());
        assert_eq!(bitmap.len(), 10);
    }
}
