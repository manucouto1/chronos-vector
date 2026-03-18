//! Time-partitioned temporal HNSW index (RFC-008).
//!
//! Divides the timeline into fixed-duration partitions, each with its own
//! HNSW graph. Queries are routed only to partitions that overlap the
//! temporal filter, achieving sub-linear latency on historical data.
//!
//! # Architecture
//!
//! ```text
//! Timeline: ──────────────────────────────────────────────────►
//!            │ Partition 0  │ Partition 1  │ Partition 2  │ ...
//!            │ [t0, t0+Δ)  │ [t0+Δ, t0+2Δ)│ [t0+2Δ, ...)│
//!            │  HNSW_0      │  HNSW_1      │  HNSW_2     │
//! ```
//!
//! Node IDs are globally unique: each partition has an `id_offset` such that
//! `global_id = partition.id_offset + local_id`.

use std::collections::BTreeMap;
use std::path::Path;

use cvx_core::traits::DistanceMetric;
use cvx_core::types::TemporalFilter;

use super::temporal::TemporalHnsw;
use super::HnswConfig;

// ─── Configuration ──────────────────────────────────────────────────

/// Configuration for time-partitioned sharding.
#[derive(Debug, Clone)]
pub struct PartitionConfig {
    /// Duration of each partition in microseconds.
    /// Default: 7 days (604_800_000_000 µs).
    pub partition_duration_us: i64,
    /// HNSW configuration applied to each partition.
    pub hnsw_config: HnswConfig,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            partition_duration_us: 7 * 24 * 3600 * 1_000_000, // 7 days
            hnsw_config: HnswConfig::default(),
        }
    }
}

// ─── Partition ──────────────────────────────────────────────────────

/// A single time partition containing an HNSW graph.
struct Partition<D: DistanceMetric> {
    /// Start of the time range (inclusive).
    start_us: i64,
    /// End of the time range (exclusive).
    end_us: i64,
    /// The HNSW graph for this partition.
    hnsw: TemporalHnsw<D>,
    /// Offset for mapping local node IDs to global IDs.
    /// global_id = id_offset + local_id
    id_offset: u32,
    /// Number of points in this partition.
    point_count: usize,
}

impl<D: DistanceMetric> Partition<D> {
    /// Whether this partition's time range overlaps with the given filter.
    fn overlaps(&self, filter: &TemporalFilter) -> bool {
        match filter {
            TemporalFilter::All => true,
            TemporalFilter::Snapshot(t) => *t >= self.start_us && *t < self.end_us,
            TemporalFilter::Range(start, end) => *start < self.end_us && *end >= self.start_us,
            TemporalFilter::Before(t) => *t >= self.start_us,
            TemporalFilter::After(t) => *t < self.end_us,
        }
    }

    /// Whether a timestamp falls within this partition's range.
    fn contains_timestamp(&self, timestamp: i64) -> bool {
        timestamp >= self.start_us && timestamp < self.end_us
    }
}

// ─── PartitionedTemporalHnsw ────────────────────────────────────────

/// Time-partitioned temporal HNSW index.
///
/// Routes inserts to the correct partition by timestamp, prunes queries
/// to only touch relevant partitions, and merges results globally.
pub struct PartitionedTemporalHnsw<D: DistanceMetric> {
    /// Partitions sorted by start time.
    partitions: Vec<Partition<D>>,
    /// Configuration.
    config: PartitionConfig,
    /// Global entity index: entity_id → vec of (partition_idx, local_node_id).
    global_entity_index: BTreeMap<u64, Vec<(usize, u32)>>,
    /// Total number of points across all partitions.
    total_points: usize,
    /// Next global ID to assign.
    next_global_id: u32,
    /// Metric (cloned to create new partitions).
    metric: D,
}

impl<D: DistanceMetric + Clone> PartitionedTemporalHnsw<D> {
    /// Create a new empty partitioned index.
    pub fn new(config: PartitionConfig, metric: D) -> Self {
        Self {
            partitions: Vec::new(),
            config,
            global_entity_index: BTreeMap::new(),
            total_points: 0,
            next_global_id: 0,
            metric,
        }
    }

    /// Create from an existing `TemporalHnsw` (migration path).
    ///
    /// The entire existing index becomes partition 0.
    pub fn from_single(index: TemporalHnsw<D>, config: PartitionConfig, metric: D) -> Self {
        let point_count = index.len();
        let (min_ts, max_ts) = if point_count > 0 {
            let mut min = i64::MAX;
            let mut max = i64::MIN;
            for i in 0..point_count {
                let ts = index.timestamp(i as u32);
                min = min.min(ts);
                max = max.max(ts);
            }
            (min, max)
        } else {
            (0, 0)
        };

        // Build global entity index from the single partition
        let mut global_entity_index = BTreeMap::new();
        for i in 0..point_count {
            let eid = index.entity_id(i as u32);
            global_entity_index
                .entry(eid)
                .or_insert_with(Vec::new)
                .push((0usize, i as u32));
        }

        let partition = Partition {
            start_us: min_ts,
            end_us: max_ts + 1,
            hnsw: index,
            id_offset: 0,
            point_count,
        };

        Self {
            partitions: vec![partition],
            config,
            global_entity_index,
            total_points: point_count,
            next_global_id: point_count as u32,
            metric,
        }
    }

    /// Insert a temporal point, routing to the correct partition.
    pub fn insert(&mut self, entity_id: u64, timestamp: i64, vector: &[f32]) -> u32 {
        let part_idx = self.ensure_partition_for(timestamp);

        let local_id = self.partitions[part_idx]
            .hnsw
            .insert(entity_id, timestamp, vector);

        let global_id = self.partitions[part_idx].id_offset + local_id;
        self.partitions[part_idx].point_count += 1;

        // Update global entity index
        self.global_entity_index
            .entry(entity_id)
            .or_default()
            .push((part_idx, local_id));

        self.total_points += 1;
        self.next_global_id = self.next_global_id.max(global_id + 1);

        global_id
    }

    /// Search across relevant partitions, merging results.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: TemporalFilter,
        alpha: f32,
        query_timestamp: i64,
    ) -> Vec<(u32, f32)> {
        let mut all_results: Vec<(u32, f32)> = Vec::new();

        for part in &self.partitions {
            if !part.overlaps(&filter) {
                continue;
            }

            let local_results = part.hnsw.search(query, k, filter.clone(), alpha, query_timestamp);

            // Map local IDs to global IDs
            for (local_id, score) in local_results {
                all_results.push((part.id_offset + local_id, score));
            }
        }

        // Sort by score and take top k
        all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        all_results.truncate(k);
        all_results
    }

    /// Retrieve trajectory for an entity across all partitions.
    pub fn trajectory(&self, entity_id: u64, filter: TemporalFilter) -> Vec<(i64, u32)> {
        let Some(entries) = self.global_entity_index.get(&entity_id) else {
            return Vec::new();
        };

        let mut result: Vec<(i64, u32)> = Vec::new();

        for &(part_idx, local_id) in entries {
            let part = &self.partitions[part_idx];
            let ts = part.hnsw.timestamp(local_id);
            if filter.matches(ts) {
                result.push((ts, part.id_offset + local_id));
            }
        }

        result.sort_by_key(|&(ts, _)| ts);
        result
    }

    /// Get vector by global node ID.
    pub fn vector(&self, global_id: u32) -> Vec<f32> {
        let (part, local_id) = self.resolve_global_id(global_id);
        part.hnsw.vector(local_id).to_vec()
    }

    /// Get entity ID by global node ID.
    pub fn entity_id(&self, global_id: u32) -> u64 {
        let (part, local_id) = self.resolve_global_id(global_id);
        part.hnsw.entity_id(local_id)
    }

    /// Get timestamp by global node ID.
    pub fn timestamp(&self, global_id: u32) -> i64 {
        let (part, local_id) = self.resolve_global_id(global_id);
        part.hnsw.timestamp(local_id)
    }

    /// Total number of indexed points.
    pub fn len(&self) -> usize {
        self.total_points
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.total_points == 0
    }

    /// Number of partitions.
    pub fn num_partitions(&self) -> usize {
        self.partitions.len()
    }

    /// Get partition info: `(start_us, end_us, point_count)`.
    pub fn partition_info(&self) -> Vec<(i64, i64, usize)> {
        self.partitions
            .iter()
            .map(|p| (p.start_us, p.end_us, p.point_count))
            .collect()
    }

    /// Save all partitions to a directory.
    pub fn save(&self, dir: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;

        // Save metadata
        let meta = PartitionMeta {
            partition_duration_us: self.config.partition_duration_us,
            num_partitions: self.partitions.len(),
            partitions: self
                .partitions
                .iter()
                .map(|p| PartitionMetaEntry {
                    start_us: p.start_us,
                    end_us: p.end_us,
                    id_offset: p.id_offset,
                    point_count: p.point_count,
                })
                .collect(),
        };
        let meta_bytes = postcard::to_allocvec(&meta)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(dir.join("partitions.meta"), meta_bytes)?;

        // Save each partition
        for (i, part) in self.partitions.iter().enumerate() {
            let path = dir.join(format!("partition_{i}.bin"));
            part.hnsw.save(&path)?;
        }

        Ok(())
    }

    /// Load partitioned index from a directory.
    pub fn load(dir: &Path, metric: D) -> std::io::Result<Self> {
        let meta_bytes = std::fs::read(dir.join("partitions.meta"))?;
        let meta: PartitionMeta = postcard::from_bytes(&meta_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let mut partitions = Vec::with_capacity(meta.num_partitions);
        let mut global_entity_index: BTreeMap<u64, Vec<(usize, u32)>> = BTreeMap::new();
        let mut total_points = 0;
        let mut next_global_id: u32 = 0;

        for (i, pm) in meta.partitions.iter().enumerate() {
            let path = dir.join(format!("partition_{i}.bin"));
            let hnsw = TemporalHnsw::load(&path, metric.clone())?;

            // Rebuild global entity index for this partition
            for local_id in 0..hnsw.len() as u32 {
                let eid = hnsw.entity_id(local_id);
                global_entity_index
                    .entry(eid)
                    .or_default()
                    .push((i, local_id));
            }

            total_points += pm.point_count;
            next_global_id = next_global_id.max(pm.id_offset + hnsw.len() as u32);

            partitions.push(Partition {
                start_us: pm.start_us,
                end_us: pm.end_us,
                hnsw,
                id_offset: pm.id_offset,
                point_count: pm.point_count,
            });
        }

        let config = PartitionConfig {
            partition_duration_us: meta.partition_duration_us,
            hnsw_config: HnswConfig::default(),
        };

        Ok(Self {
            partitions,
            config,
            global_entity_index,
            total_points,
            next_global_id,
            metric,
        })
    }

    // ─── Private helpers ────────────────────────────────────────

    /// Find or create the partition for a given timestamp.
    fn ensure_partition_for(&mut self, timestamp: i64) -> usize {
        // Check existing partitions
        for (i, part) in self.partitions.iter().enumerate() {
            if part.contains_timestamp(timestamp) {
                return i;
            }
        }

        // Create new partition
        let dur = self.config.partition_duration_us;
        let start = if dur > 0 {
            (timestamp / dur) * dur
        } else {
            timestamp
        };
        let end = start + dur;

        let id_offset = self.next_global_id;

        let partition = Partition {
            start_us: start,
            end_us: end,
            hnsw: TemporalHnsw::new(self.config.hnsw_config.clone(), self.metric.clone()),
            id_offset,
            point_count: 0,
        };

        self.partitions.push(partition);

        // Keep partitions sorted by start time
        let idx = self.partitions.len() - 1;
        self.partitions.sort_by_key(|p| p.start_us);

        // Return the index of the newly inserted partition
        self.partitions
            .iter()
            .position(|p| p.start_us == start)
            .unwrap_or(idx)
    }

    /// Resolve a global node ID to (partition, local_id).
    fn resolve_global_id(&self, global_id: u32) -> (&Partition<D>, u32) {
        for part in &self.partitions {
            if global_id >= part.id_offset
                && global_id < part.id_offset + part.hnsw.len() as u32
            {
                return (part, global_id - part.id_offset);
            }
        }
        panic!(
            "global_id {global_id} not found in any partition (total: {})",
            self.total_points
        );
    }
}

// ─── TemporalIndexAccess implementation ─────────────────────────────

impl<D: DistanceMetric + Clone> cvx_core::TemporalIndexAccess for PartitionedTemporalHnsw<D> {
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
        self.vector(node_id)
    }

    fn entity_id(&self, node_id: u32) -> u64 {
        self.entity_id(node_id)
    }

    fn timestamp(&self, node_id: u32) -> i64 {
        self.timestamp(node_id)
    }

    fn len(&self) -> usize {
        self.len()
    }
}

// ─── Serialization types ────────────────────────────────────────────

#[derive(serde::Serialize, serde::Deserialize)]
struct PartitionMeta {
    partition_duration_us: i64,
    num_partitions: usize,
    partitions: Vec<PartitionMetaEntry>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct PartitionMetaEntry {
    start_us: i64,
    end_us: i64,
    id_offset: u32,
    point_count: usize,
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::L2Distance;

    fn default_config() -> PartitionConfig {
        PartitionConfig {
            partition_duration_us: 7_000_000, // 7 seconds (small for testing)
            hnsw_config: HnswConfig::default(),
        }
    }

    // ─── Basic insert and search ────────────────────────────────

    #[test]
    fn insert_single_partition() {
        let mut index = PartitionedTemporalHnsw::new(default_config(), L2Distance);

        // All points within one partition
        for i in 0..10u64 {
            let ts = i as i64 * 100_000; // 0 to 0.9 seconds
            index.insert(i, ts, &[i as f32, 0.0, 0.0]);
        }

        assert_eq!(index.len(), 10);
        assert_eq!(index.num_partitions(), 1);
    }

    #[test]
    fn insert_multiple_partitions() {
        let mut index = PartitionedTemporalHnsw::new(default_config(), L2Distance);

        // Spread across 3 partitions (7s each)
        for i in 0..30u64 {
            let ts = i as i64 * 1_000_000; // 0 to 29 seconds
            index.insert(0, ts, &[i as f32, 0.0, 0.0]);
        }

        assert_eq!(index.len(), 30);
        assert!(
            index.num_partitions() >= 3,
            "expected >= 3 partitions, got {}",
            index.num_partitions()
        );
    }

    #[test]
    fn search_across_partitions() {
        let mut index = PartitionedTemporalHnsw::new(default_config(), L2Distance);

        for i in 0..30u64 {
            let ts = i as i64 * 1_000_000;
            index.insert(i, ts, &[i as f32, 0.0, 0.0]);
        }

        let results = index.search(&[5.0, 0.0, 0.0], 3, TemporalFilter::All, 1.0, 0);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn search_with_temporal_filter_prunes_partitions() {
        let mut index = PartitionedTemporalHnsw::new(default_config(), L2Distance);

        // Insert into partition 0 (t=0-6s) and partition 1 (t=7-13s)
        for i in 0..14u64 {
            let ts = i as i64 * 1_000_000;
            index.insert(i, ts, &[i as f32, 0.0, 0.0]);
        }

        // Search only in partition 0's time range
        let results = index.search(
            &[3.0, 0.0, 0.0],
            5,
            TemporalFilter::Range(0, 6_999_999),
            1.0,
            3_000_000,
        );

        // All results should be from partition 0 (timestamps < 7s)
        for &(global_id, _) in &results {
            let ts = index.timestamp(global_id);
            assert!(ts < 7_000_000, "got timestamp {ts} outside partition 0");
        }
    }

    // ─── Trajectory ─────────────────────────────────────────────

    #[test]
    fn trajectory_across_partitions() {
        let mut index = PartitionedTemporalHnsw::new(default_config(), L2Distance);

        // Entity 42 has points in 3 different partitions
        for i in 0..21u64 {
            let ts = i as i64 * 1_000_000;
            index.insert(42, ts, &[i as f32, 0.0]);
        }

        let traj = index.trajectory(42, TemporalFilter::All);
        assert_eq!(traj.len(), 21, "should get all 21 points");

        // Verify sorted by timestamp
        for w in traj.windows(2) {
            assert!(w[0].0 <= w[1].0, "trajectory should be sorted");
        }
    }

    #[test]
    fn trajectory_with_filter() {
        let mut index = PartitionedTemporalHnsw::new(default_config(), L2Distance);

        for i in 0..21u64 {
            let ts = i as i64 * 1_000_000;
            index.insert(42, ts, &[i as f32]);
        }

        let traj = index.trajectory(42, TemporalFilter::Range(5_000_000, 15_000_000));
        for &(ts, _) in &traj {
            assert!(
                ts >= 5_000_000 && ts <= 15_000_000,
                "ts {ts} outside filter range"
            );
        }
        assert!(!traj.is_empty());
    }

    #[test]
    fn trajectory_unknown_entity() {
        let index = PartitionedTemporalHnsw::new(default_config(), L2Distance);
        let traj = index.trajectory(999, TemporalFilter::All);
        assert!(traj.is_empty());
    }

    // ─── Global ID resolution ───────────────────────────────────

    #[test]
    fn vector_entity_timestamp_resolution() {
        let mut index = PartitionedTemporalHnsw::new(default_config(), L2Distance);

        let id0 = index.insert(1, 0, &[1.0, 2.0, 3.0]);
        let id1 = index.insert(2, 8_000_000, &[4.0, 5.0, 6.0]); // different partition

        assert_eq!(index.entity_id(id0), 1);
        assert_eq!(index.entity_id(id1), 2);
        assert_eq!(index.timestamp(id0), 0);
        assert_eq!(index.timestamp(id1), 8_000_000);
        assert_eq!(index.vector(id0), vec![1.0, 2.0, 3.0]);
        assert_eq!(index.vector(id1), vec![4.0, 5.0, 6.0]);
    }

    // ─── TemporalIndexAccess trait ──────────────────────────────

    #[test]
    fn trait_search_works() {
        let mut index = PartitionedTemporalHnsw::new(default_config(), L2Distance);
        for i in 0..20u64 {
            index.insert(i, i as i64 * 500_000, &[i as f32, 0.0]);
        }

        let trait_ref: &dyn cvx_core::TemporalIndexAccess = &index;
        let results = trait_ref.search_raw(&[10.0, 0.0], 3, TemporalFilter::All, 1.0, 0);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn trait_trajectory_works() {
        let mut index = PartitionedTemporalHnsw::new(default_config(), L2Distance);
        for i in 0..10u64 {
            index.insert(42, i as i64 * 1_000_000, &[i as f32]);
        }

        let trait_ref: &dyn cvx_core::TemporalIndexAccess = &index;
        let traj = trait_ref.trajectory(42, TemporalFilter::All);
        assert_eq!(traj.len(), 10);
    }

    // ─── Partition overlap logic ────────────────────────────────

    #[test]
    fn partition_overlap_all() {
        let part = Partition {
            start_us: 100,
            end_us: 200,
            hnsw: TemporalHnsw::new(HnswConfig::default(), L2Distance),
            id_offset: 0,
            point_count: 0,
        };
        assert!(part.overlaps(&TemporalFilter::All));
    }

    #[test]
    fn partition_overlap_range() {
        let part = Partition {
            start_us: 100,
            end_us: 200,
            hnsw: TemporalHnsw::new(HnswConfig::default(), L2Distance),
            id_offset: 0,
            point_count: 0,
        };
        // Overlapping range
        assert!(part.overlaps(&TemporalFilter::Range(150, 250)));
        // Non-overlapping
        assert!(!part.overlaps(&TemporalFilter::Range(200, 300)));
        assert!(!part.overlaps(&TemporalFilter::Range(0, 99)));
    }

    #[test]
    fn partition_overlap_before_after() {
        let part = Partition {
            start_us: 100,
            end_us: 200,
            hnsw: TemporalHnsw::new(HnswConfig::default(), L2Distance),
            id_offset: 0,
            point_count: 0,
        };
        assert!(part.overlaps(&TemporalFilter::Before(150)));
        assert!(!part.overlaps(&TemporalFilter::Before(99)));
        assert!(part.overlaps(&TemporalFilter::After(150)));
        assert!(!part.overlaps(&TemporalFilter::After(200)));
    }

    // ─── Save / Load ────────────────────────────────────────────

    #[test]
    fn save_and_load_roundtrip() {
        let mut index = PartitionedTemporalHnsw::new(default_config(), L2Distance);

        for i in 0..20u64 {
            let ts = i as i64 * 1_000_000;
            index.insert(i % 5, ts, &[i as f32, (i as f32).sin()]);
        }

        let dir = tempfile::tempdir().unwrap();
        index.save(dir.path()).unwrap();

        let loaded = PartitionedTemporalHnsw::load(dir.path(), L2Distance).unwrap();

        assert_eq!(loaded.len(), 20);
        assert_eq!(loaded.num_partitions(), index.num_partitions());

        // Verify trajectory is intact
        let traj_orig = index.trajectory(0, TemporalFilter::All);
        let traj_loaded = loaded.trajectory(0, TemporalFilter::All);
        assert_eq!(traj_orig.len(), traj_loaded.len());
    }

    // ─── from_single migration ──────────────────────────────────

    #[test]
    fn from_single_preserves_data() {
        let config = HnswConfig::default();
        let mut single = TemporalHnsw::new(config, L2Distance);

        for i in 0..15u64 {
            single.insert(i % 3, i as i64 * 1000, &[i as f32, 0.0]);
        }

        let partitioned = PartitionedTemporalHnsw::from_single(
            single,
            default_config(),
            L2Distance,
        );

        assert_eq!(partitioned.len(), 15);
        assert_eq!(partitioned.num_partitions(), 1);

        // Verify entity trajectories
        for eid in 0..3 {
            let traj = partitioned.trajectory(eid, TemporalFilter::All);
            assert_eq!(traj.len(), 5, "entity {eid} should have 5 points");
        }
    }

    // ─── Edge cases ─────────────────────────────────────────────

    #[test]
    fn empty_index() {
        let index = PartitionedTemporalHnsw::new(default_config(), L2Distance);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
        assert_eq!(index.num_partitions(), 0);
    }

    #[test]
    fn out_of_order_inserts() {
        let mut index = PartitionedTemporalHnsw::new(default_config(), L2Distance);

        // Insert in reverse temporal order
        index.insert(1, 20_000_000, &[20.0]);
        index.insert(1, 10_000_000, &[10.0]);
        index.insert(1, 0, &[0.0]);

        let traj = index.trajectory(1, TemporalFilter::All);
        assert_eq!(traj.len(), 3);
        // Should be sorted by timestamp
        assert!(traj[0].0 <= traj[1].0);
        assert!(traj[1].0 <= traj[2].0);
    }

    #[test]
    fn partition_info() {
        let mut index = PartitionedTemporalHnsw::new(default_config(), L2Distance);

        index.insert(1, 0, &[1.0]);
        index.insert(2, 8_000_000, &[2.0]);

        let info = index.partition_info();
        assert_eq!(info.len(), 2);
        assert_eq!(info[0].2, 1); // 1 point in partition 0
        assert_eq!(info[1].2, 1); // 1 point in partition 1
    }
}
