//! Anchor-space invariant index (RFC-011).
//!
//! Indexes entities in anchor-projected space (ℝᴷ) where K = number of anchors.
//! Vectors from DIFFERENT embedding models are directly comparable when projected
//! through the same anchor set, enabling cross-model search and trajectory analysis.
//!
//! # Why flat scan?
//!
//! K is typically 5-20 (number of clinical/semantic anchors). For 1M points in
//! ℝ¹⁰, a flat scan with SIMD L2 takes ~10ms — fast enough for most use cases.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::anchor::{AnchorMetric, project_to_anchors};
use crate::calculus::{drift_magnitude_l2, drift_report};
use cvx_core::types::TemporalFilter;

// ─── Configuration ──────────────────────────────────────────────────

/// Configuration for an anchor set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnchorSetConfig {
    /// Unique identifier for this anchor set.
    pub anchor_set_id: u32,
    /// Human-readable name (e.g., "clinical_anchors_v1").
    pub name: String,
    /// Distance metric for projection.
    pub metric: AnchorMetricSerde,
}

/// Serializable anchor metric (mirrors `AnchorMetric`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AnchorMetricSerde {
    /// Cosine distance.
    Cosine,
    /// L2 distance.
    L2,
}

impl From<AnchorMetricSerde> for AnchorMetric {
    fn from(m: AnchorMetricSerde) -> Self {
        match m {
            AnchorMetricSerde::Cosine => AnchorMetric::Cosine,
            AnchorMetricSerde::L2 => AnchorMetric::L2,
        }
    }
}

// ─── Drift report in anchor space ───────────────────────────────────

/// Drift report in anchor-projected space.
#[derive(Debug, Clone)]
pub struct AnchorDriftReport {
    /// Per-anchor distance change: positive = moved away, negative = approached.
    pub per_anchor_delta: Vec<f32>,
    /// L2 magnitude of the drift vector in anchor space.
    pub l2_magnitude: f32,
    /// Cosine drift in anchor space.
    pub cosine_drift: f32,
    /// Index of the anchor with the largest absolute change.
    pub dominant_anchor: usize,
    /// Source model ID at t1.
    pub model_t1: u32,
    /// Source model ID at t2.
    pub model_t2: u32,
}

// ─── AnchorSpaceIndex ───────────────────────────────────────────────

/// An index operating in anchor-projected space (ℝᴷ).
///
/// Stores pre-projected vectors from potentially multiple embedding models.
/// All comparable because they use the same anchor set.
pub struct AnchorSpaceIndex {
    /// Anchor set configuration.
    config: AnchorSetConfig,
    /// Number of anchors (= dimensionality of projected space).
    k: usize,
    /// Projected vectors: node_id → Vec<f32> of length K.
    projected_vectors: Vec<Vec<f32>>,
    /// Source model identifier per node.
    source_model: Vec<u32>,
    /// Entity ID per node.
    entity_ids: Vec<u64>,
    /// Timestamp per node.
    timestamps: Vec<i64>,
    /// Entity index: entity_id → sorted vec of (timestamp, node_id).
    entity_index: BTreeMap<u64, Vec<(i64, u32)>>,
}

impl AnchorSpaceIndex {
    /// Create a new empty anchor space index.
    pub fn new(config: AnchorSetConfig, k: usize) -> Self {
        Self {
            config,
            k,
            projected_vectors: Vec::new(),
            source_model: Vec::new(),
            entity_ids: Vec::new(),
            timestamps: Vec::new(),
            entity_index: BTreeMap::new(),
        }
    }

    /// Insert a raw vector, projecting it to anchor space.
    ///
    /// `model_anchors` are the anchor vectors embedded in the SAME model
    /// as the input vector. These may differ from anchors of other models.
    pub fn insert(
        &mut self,
        entity_id: u64,
        timestamp: i64,
        vector: &[f32],
        model_anchors: &[&[f32]],
        model_id: u32,
    ) -> u32 {
        // Project single point via project_to_anchors
        let traj = [(timestamp, vector)];
        let projected = project_to_anchors(&traj, model_anchors, self.config.metric.into());

        let proj_vec = projected.into_iter().next().unwrap().1;
        self.insert_projected(entity_id, timestamp, proj_vec, model_id)
    }

    /// Insert a pre-projected vector (already in ℝᴷ).
    pub fn insert_projected(
        &mut self,
        entity_id: u64,
        timestamp: i64,
        projected: Vec<f32>,
        model_id: u32,
    ) -> u32 {
        assert_eq!(
            projected.len(),
            self.k,
            "projected vector dim {} != anchor count {}",
            projected.len(),
            self.k
        );

        let node_id = self.projected_vectors.len() as u32;
        self.projected_vectors.push(projected);
        self.source_model.push(model_id);
        self.entity_ids.push(entity_id);
        self.timestamps.push(timestamp);

        self.entity_index
            .entry(entity_id)
            .or_default()
            .push((timestamp, node_id));

        node_id
    }

    /// Search in anchor space: flat scan kNN by L2 distance in ℝᴷ.
    ///
    /// Cross-model: results may come from ANY source model.
    pub fn search(
        &self,
        query_projected: &[f32],
        k: usize,
        filter: TemporalFilter,
    ) -> Vec<(u32, f32)> {
        let mut results: Vec<(u32, f32)> = self
            .projected_vectors
            .iter()
            .enumerate()
            .filter(|(i, _)| filter.matches(self.timestamps[*i]))
            .map(|(i, v)| (i as u32, drift_magnitude_l2(query_projected, v)))
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }

    /// Retrieve trajectory in anchor space for an entity.
    pub fn trajectory(&self, entity_id: u64, filter: TemporalFilter) -> Vec<(i64, Vec<f32>)> {
        let Some(entries) = self.entity_index.get(&entity_id) else {
            return Vec::new();
        };

        let mut result: Vec<(i64, Vec<f32>)> = entries
            .iter()
            .filter(|(ts, _)| filter.matches(*ts))
            .map(|&(ts, nid)| (ts, self.projected_vectors[nid as usize].clone()))
            .collect();

        result.sort_by_key(|&(ts, _)| ts);
        result
    }

    /// Cross-model trajectory: separate trajectories per source model.
    pub fn cross_model_trajectory(
        &self,
        entity_id: u64,
        filter: TemporalFilter,
    ) -> BTreeMap<u32, Vec<(i64, Vec<f32>)>> {
        let Some(entries) = self.entity_index.get(&entity_id) else {
            return BTreeMap::new();
        };

        let mut by_model: BTreeMap<u32, Vec<(i64, Vec<f32>)>> = BTreeMap::new();

        for &(ts, nid) in entries {
            if !filter.matches(ts) {
                continue;
            }
            let model = self.source_model[nid as usize];
            by_model
                .entry(model)
                .or_default()
                .push((ts, self.projected_vectors[nid as usize].clone()));
        }

        // Sort each model's trajectory by timestamp
        for traj in by_model.values_mut() {
            traj.sort_by_key(|&(ts, _)| ts);
        }

        by_model
    }

    /// Compute drift in anchor space between two timestamps.
    pub fn anchor_drift(&self, entity_id: u64, t1: i64, t2: i64) -> Option<AnchorDriftReport> {
        let entries = self.entity_index.get(&entity_id)?;

        // Find nearest point to t1 and t2
        let (_, nid1) = entries
            .iter()
            .min_by_key(|&&(ts, _)| (ts - t1).unsigned_abs())?;
        let (_, nid2) = entries
            .iter()
            .min_by_key(|&&(ts, _)| (ts - t2).unsigned_abs())?;

        let v1 = &self.projected_vectors[*nid1 as usize];
        let v2 = &self.projected_vectors[*nid2 as usize];

        let per_anchor_delta: Vec<f32> = v2.iter().zip(v1.iter()).map(|(a, b)| a - b).collect();
        let report = drift_report(v1, v2, self.k);

        let dominant_anchor = per_anchor_delta
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        Some(AnchorDriftReport {
            per_anchor_delta,
            l2_magnitude: report.l2_magnitude,
            cosine_drift: report.cosine_drift,
            dominant_anchor,
            model_t1: self.source_model[*nid1 as usize],
            model_t2: self.source_model[*nid2 as usize],
        })
    }

    /// Number of indexed points.
    pub fn len(&self) -> usize {
        self.projected_vectors.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.projected_vectors.is_empty()
    }

    /// Number of unique entities.
    pub fn n_entities(&self) -> usize {
        self.entity_index.len()
    }

    /// Get entity ID for a node.
    pub fn entity_id(&self, node_id: u32) -> u64 {
        self.entity_ids[node_id as usize]
    }

    /// Get timestamp for a node.
    pub fn timestamp(&self, node_id: u32) -> i64 {
        self.timestamps[node_id as usize]
    }

    /// Get source model for a node.
    pub fn source_model(&self, node_id: u32) -> u32 {
        self.source_model[node_id as usize]
    }

    /// Get projected vector for a node.
    pub fn projected_vector(&self, node_id: u32) -> &[f32] {
        &self.projected_vectors[node_id as usize]
    }

    /// Anchor set config.
    pub fn config(&self) -> &AnchorSetConfig {
        &self.config
    }

    /// Dimensionality of the projected space (= number of anchors).
    pub fn k(&self) -> usize {
        self.k
    }

    /// Save to file via postcard.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let snapshot = AnchorSpaceSnapshot {
            config: self.config.clone(),
            k: self.k,
            projected_vectors: self.projected_vectors.clone(),
            source_model: self.source_model.clone(),
            entity_ids: self.entity_ids.clone(),
            timestamps: self.timestamps.clone(),
            entity_index: self.entity_index.clone(),
        };
        let bytes = postcard::to_allocvec(&snapshot).map_err(std::io::Error::other)?;
        std::fs::write(path, bytes)
    }

    /// Load from file.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let snapshot: AnchorSpaceSnapshot = postcard::from_bytes(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(Self {
            config: snapshot.config,
            k: snapshot.k,
            projected_vectors: snapshot.projected_vectors,
            source_model: snapshot.source_model,
            entity_ids: snapshot.entity_ids,
            timestamps: snapshot.timestamps,
            entity_index: snapshot.entity_index,
        })
    }
}

#[derive(Serialize, Deserialize)]
struct AnchorSpaceSnapshot {
    config: AnchorSetConfig,
    k: usize,
    projected_vectors: Vec<Vec<f32>>,
    source_model: Vec<u32>,
    entity_ids: Vec<u64>,
    timestamps: Vec<i64>,
    entity_index: BTreeMap<u64, Vec<(i64, u32)>>,
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> AnchorSetConfig {
        AnchorSetConfig {
            anchor_set_id: 1,
            name: "test_anchors".to_string(),
            metric: AnchorMetricSerde::Cosine,
        }
    }

    // ─── Basic operations ───────────────────────────────────────

    #[test]
    fn new_empty() {
        let index = AnchorSpaceIndex::new(test_config(), 3);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
        assert_eq!(index.k(), 3);
    }

    #[test]
    fn insert_projected() {
        let mut index = AnchorSpaceIndex::new(test_config(), 3);
        let id = index.insert_projected(42, 1000, vec![0.1, 0.5, 0.3], 0);

        assert_eq!(index.len(), 1);
        assert_eq!(index.entity_id(id), 42);
        assert_eq!(index.timestamp(id), 1000);
        assert_eq!(index.source_model(id), 0);
        assert_eq!(index.projected_vector(id), &[0.1, 0.5, 0.3]);
    }

    #[test]
    fn insert_with_projection() {
        let mut index = AnchorSpaceIndex::new(test_config(), 2);

        let vector = [1.0f32, 0.0, 0.0];
        let anchors: Vec<Vec<f32>> = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let anchor_refs: Vec<&[f32]> = anchors.iter().map(|a| a.as_slice()).collect();

        let id = index.insert(42, 1000, &vector, &anchor_refs, 0);

        assert_eq!(index.len(), 1);
        let proj = index.projected_vector(id);
        assert_eq!(proj.len(), 2);
        // [1,0,0] is cosine distance 0 from anchor [1,0,0] and 1 from [0,1,0]
        assert!(
            proj[0] < 0.01,
            "should be close to anchor 0, got {}",
            proj[0]
        );
        assert!(
            (proj[1] - 1.0).abs() < 0.01,
            "should be far from anchor 1, got {}",
            proj[1]
        );
    }

    // ─── Search ─────────────────────────────────────────────────

    #[test]
    fn search_finds_nearest() {
        let mut index = AnchorSpaceIndex::new(test_config(), 3);

        // Insert 5 points at different positions in anchor space
        for i in 0..5u32 {
            index.insert_projected(i as u64, i as i64 * 1000, vec![i as f32, 0.0, 0.0], 0);
        }

        // Query near point 2
        let results = index.search(&[2.1, 0.0, 0.0], 3, TemporalFilter::All);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 2); // closest
    }

    #[test]
    fn search_with_temporal_filter() {
        let mut index = AnchorSpaceIndex::new(test_config(), 2);

        for i in 0..10u32 {
            index.insert_projected(i as u64, i as i64 * 1000, vec![i as f32, 0.0], 0);
        }

        let results = index.search(&[5.0, 0.0], 10, TemporalFilter::Range(3000, 7000));
        for &(nid, _) in &results {
            let ts = index.timestamp(nid);
            assert!((3000..=7000).contains(&ts), "ts {ts} outside range");
        }
    }

    // ─── Cross-model search ─────────────────────────────────────

    #[test]
    fn cross_model_search() {
        let mut index = AnchorSpaceIndex::new(test_config(), 2);

        // Model 0: entity 1 near [0.1, 0.9]
        index.insert_projected(1, 1000, vec![0.1, 0.9], 0);
        // Model 1: entity 2 near [0.1, 0.8]
        index.insert_projected(2, 1000, vec![0.1, 0.8], 1);
        // Model 0: entity 3 far away
        index.insert_projected(3, 1000, vec![5.0, 5.0], 0);

        let results = index.search(&[0.1, 0.85], 2, TemporalFilter::All);
        assert_eq!(results.len(), 2);

        // Top 2 should be entities 1 and 2 (from different models!)
        let model_0 = results.iter().any(|&(nid, _)| index.source_model(nid) == 0);
        let model_1 = results.iter().any(|&(nid, _)| index.source_model(nid) == 1);
        assert!(
            model_0 && model_1,
            "search should return results from both models"
        );
    }

    // ─── Trajectory ─────────────────────────────────────────────

    #[test]
    fn trajectory_in_anchor_space() {
        let mut index = AnchorSpaceIndex::new(test_config(), 2);

        for i in 0..5u64 {
            index.insert_projected(42, i as i64 * 1000, vec![i as f32 * 0.1, 0.5], 0);
        }

        let traj = index.trajectory(42, TemporalFilter::All);
        assert_eq!(traj.len(), 5);
        // Sorted by timestamp
        for w in traj.windows(2) {
            assert!(w[0].0 <= w[1].0);
        }
    }

    #[test]
    fn cross_model_trajectory() {
        let mut index = AnchorSpaceIndex::new(test_config(), 2);

        // Same entity from 2 models
        index.insert_projected(42, 1000, vec![0.1, 0.9], 0);
        index.insert_projected(42, 2000, vec![0.2, 0.8], 0);
        index.insert_projected(42, 1000, vec![0.15, 0.85], 1);
        index.insert_projected(42, 2000, vec![0.25, 0.75], 1);

        let by_model = index.cross_model_trajectory(42, TemporalFilter::All);
        assert_eq!(by_model.len(), 2); // 2 models
        assert_eq!(by_model[&0].len(), 2);
        assert_eq!(by_model[&1].len(), 2);
    }

    // ─── Anchor drift ───────────────────────────────────────────

    #[test]
    fn anchor_drift_approaching() {
        let mut index = AnchorSpaceIndex::new(test_config(), 3);

        // Entity moves closer to anchor 0 over time
        index.insert_projected(1, 1000, vec![1.0, 0.5, 0.5], 0);
        index.insert_projected(1, 2000, vec![0.5, 0.5, 0.5], 0);

        let report = index.anchor_drift(1, 1000, 2000).unwrap();
        assert!(
            report.per_anchor_delta[0] < 0.0,
            "should be approaching anchor 0"
        );
        assert_eq!(report.dominant_anchor, 0);
        assert!(report.l2_magnitude > 0.0);
    }

    #[test]
    fn anchor_drift_cross_model() {
        let mut index = AnchorSpaceIndex::new(test_config(), 2);

        // t1 from model 0, t2 from model 1
        index.insert_projected(1, 1000, vec![0.8, 0.2], 0);
        index.insert_projected(1, 2000, vec![0.3, 0.7], 1);

        let report = index.anchor_drift(1, 1000, 2000).unwrap();
        assert_eq!(report.model_t1, 0);
        assert_eq!(report.model_t2, 1);
        assert!(report.l2_magnitude > 0.0);
    }

    #[test]
    fn anchor_drift_unknown_entity() {
        let index = AnchorSpaceIndex::new(test_config(), 2);
        assert!(index.anchor_drift(999, 0, 1000).is_none());
    }

    // ─── Persistence ────────────────────────────────────────────

    #[test]
    fn save_load_roundtrip() {
        let mut index = AnchorSpaceIndex::new(test_config(), 3);

        for i in 0..10u32 {
            index.insert_projected(
                i as u64 % 3,
                i as i64 * 1000,
                vec![i as f32 * 0.1, 0.5, 0.3],
                i % 2,
            );
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("anchor_index.bin");
        index.save(&path).unwrap();

        let loaded = AnchorSpaceIndex::load(&path).unwrap();
        assert_eq!(loaded.len(), 10);
        assert_eq!(loaded.k(), 3);
        assert_eq!(loaded.n_entities(), 3);

        // Verify search results match
        let orig_results = index.search(&[0.5, 0.5, 0.3], 3, TemporalFilter::All);
        let loaded_results = loaded.search(&[0.5, 0.5, 0.3], 3, TemporalFilter::All);
        assert_eq!(orig_results.len(), loaded_results.len());
        for (a, b) in orig_results.iter().zip(loaded_results.iter()) {
            assert_eq!(a.0, b.0);
        }
    }

    // ─── Edge cases ─────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "projected vector dim")]
    fn insert_wrong_dim_panics() {
        let mut index = AnchorSpaceIndex::new(test_config(), 3);
        index.insert_projected(1, 1000, vec![0.1, 0.2], 0); // 2 dims, expected 3
    }

    #[test]
    fn trajectory_unknown_entity() {
        let index = AnchorSpaceIndex::new(test_config(), 2);
        assert!(index.trajectory(999, TemporalFilter::All).is_empty());
    }

    #[test]
    fn search_empty_index() {
        let index = AnchorSpaceIndex::new(test_config(), 3);
        let results = index.search(&[0.0, 0.0, 0.0], 5, TemporalFilter::All);
        assert!(results.is_empty());
    }
}
