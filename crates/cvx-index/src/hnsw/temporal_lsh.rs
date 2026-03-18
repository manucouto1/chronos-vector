//! Temporal Locality-Sensitive Hashing (T-LSH) — RFC-008 Phase 2.
//!
//! An auxiliary index for composite spatiotemporal queries (α < 1.0).
//! Instead of over-fetching from the semantic HNSW and re-ranking,
//! T-LSH generates candidates that are naturally distributed in both
//! semantic and temporal space.
//!
//! # Hash function
//!
//! For a point `(vector, timestamp)`:
//! ```text
//! h_ST(v, t) = h_sem(v) ⊕ h_time(t)
//!
//! h_sem(v) = [sign(r₁·v), sign(r₂·v), ..., sign(rₖ·v)]  (random hyperplane LSH)
//! h_time(t) = floor(t / bucket_size)                       (temporal bucketing)
//!
//! Combined: concatenate semantic_bits + temporal_bits into a u64 hash
//! ```
//!
//! # References
//!
//! - Indyk & Motwani (1998). Approximate nearest neighbors. *STOC*.
//! - Lv et al. (2007). Multi-probe LSH. *VLDB*.

use std::collections::HashMap;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

// ─── Configuration ──────────────────────────────────────────────────

/// Configuration for the T-LSH index.
#[derive(Debug, Clone)]
pub struct TLSHConfig {
    /// Number of hash tables (more = higher recall, more memory).
    pub n_tables: usize,
    /// Number of semantic hash bits per table.
    pub semantic_bits: usize,
    /// Number of temporal hash bits per table.
    pub temporal_bits: usize,
    /// Temporal bucket size in microseconds.
    pub temporal_bucket_us: i64,
    /// Number of neighboring buckets to probe (multi-probe depth).
    pub n_probes: usize,
}

impl Default for TLSHConfig {
    fn default() -> Self {
        Self {
            n_tables: 16,
            semantic_bits: 12,
            temporal_bits: 4,
            temporal_bucket_us: 86_400_000_000, // 1 day
            n_probes: 3,
        }
    }
}

impl TLSHConfig {
    /// Create config tuned for a given alpha value.
    ///
    /// Higher alpha → more semantic bits, fewer temporal bits.
    pub fn for_alpha(alpha: f32, dim: usize) -> Self {
        let total_bits = 16usize;
        let sem_bits = ((alpha * total_bits as f32).round() as usize).clamp(2, total_bits - 2);
        let time_bits = total_bits - sem_bits;

        Self {
            n_tables: 16,
            semantic_bits: sem_bits,
            temporal_bits: time_bits,
            temporal_bucket_us: 86_400_000_000,
            n_probes: if dim > 100 { 5 } else { 3 },
        }
    }
}

// ─── T-LSH Index ────────────────────────────────────────────────────

/// Temporal Locality-Sensitive Hashing index.
///
/// Maintains multiple hash tables where each hash combines semantic
/// (random hyperplane) and temporal (bucket) components.
pub struct TemporalLSH {
    /// Hash tables: `tables[table_idx][hash] → vec of node_ids`.
    tables: Vec<HashMap<u64, Vec<u32>>>,
    /// Random hyperplanes for semantic hashing.
    /// Shape: `[n_tables][semantic_bits][dim]`.
    hyperplanes: Vec<Vec<Vec<f32>>>,
    /// Configuration.
    config: TLSHConfig,
    /// Dimensionality of vectors.
    dim: usize,
    /// Total number of indexed points.
    n_points: usize,
}

impl TemporalLSH {
    /// Create a new empty T-LSH index.
    pub fn new(dim: usize, config: TLSHConfig) -> Self {
        let mut rng = SmallRng::seed_from_u64(42);

        let hyperplanes: Vec<Vec<Vec<f32>>> = (0..config.n_tables)
            .map(|_| {
                (0..config.semantic_bits)
                    .map(|_| (0..dim).map(|_| rng.random::<f32>() - 0.5).collect())
                    .collect()
            })
            .collect();

        let tables = (0..config.n_tables)
            .map(|_| HashMap::new())
            .collect();

        Self {
            tables,
            hyperplanes,
            config,
            dim,
            n_points: 0,
        }
    }

    /// Build T-LSH from existing index data.
    pub fn build(
        vectors: &[&[f32]],
        timestamps: &[i64],
        config: TLSHConfig,
    ) -> Self {
        assert_eq!(vectors.len(), timestamps.len());
        if vectors.is_empty() {
            return Self::new(0, config);
        }

        let dim = vectors[0].len();
        let mut index = Self::new(dim, config);

        for (i, (v, &ts)) in vectors.iter().zip(timestamps.iter()).enumerate() {
            index.insert(i as u32, v, ts);
        }

        index
    }

    /// Insert a point into all hash tables.
    pub fn insert(&mut self, node_id: u32, vector: &[f32], timestamp: i64) {
        for table_idx in 0..self.config.n_tables {
            let hash = self.compute_hash(table_idx, vector, timestamp);
            self.tables[table_idx]
                .entry(hash)
                .or_default()
                .push(node_id);
        }
        self.n_points += 1;
    }

    /// Query: find candidate node IDs under spatiotemporal locality.
    ///
    /// Returns deduplicated candidate IDs from all tables + multi-probe.
    pub fn query(
        &self,
        vector: &[f32],
        timestamp: i64,
    ) -> Vec<u32> {
        let mut candidates = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for table_idx in 0..self.config.n_tables {
            let primary_hash = self.compute_hash(table_idx, vector, timestamp);

            // Primary bucket
            if let Some(ids) = self.tables[table_idx].get(&primary_hash) {
                for &id in ids {
                    if seen.insert(id) {
                        candidates.push(id);
                    }
                }
            }

            // Multi-probe: neighboring temporal buckets
            let temporal_bucket = self.temporal_bucket(timestamp);
            for delta in 1..=self.config.n_probes as i64 {
                for &dir in &[-1i64, 1] {
                    let neighbor_bucket = temporal_bucket + delta * dir;
                    let neighbor_hash = self.combine_hash(
                        table_idx,
                        &self.semantic_hash(table_idx, vector),
                        neighbor_bucket,
                    );
                    if let Some(ids) = self.tables[table_idx].get(&neighbor_hash) {
                        for &id in ids {
                            if seen.insert(id) {
                                candidates.push(id);
                            }
                        }
                    }
                }
            }

            // Multi-probe: flip one semantic bit
            let sem_hash = self.semantic_hash(table_idx, vector);
            for bit in 0..self.config.semantic_bits.min(3) {
                let mut flipped = sem_hash.clone();
                flipped[bit] = !flipped[bit];
                let flipped_hash = self.combine_hash(table_idx, &flipped, temporal_bucket);
                if let Some(ids) = self.tables[table_idx].get(&flipped_hash) {
                    for &id in ids {
                        if seen.insert(id) {
                            candidates.push(id);
                        }
                    }
                }
            }
        }

        candidates
    }

    /// Number of indexed points.
    pub fn len(&self) -> usize {
        self.n_points
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.n_points == 0
    }

    /// Memory usage estimate in bytes.
    pub fn memory_bytes(&self) -> usize {
        let hyperplane_mem = self.config.n_tables
            * self.config.semantic_bits
            * self.dim
            * std::mem::size_of::<f32>();

        let table_mem: usize = self.tables.iter().map(|t| {
            t.values().map(|v| v.len() * std::mem::size_of::<u32>() + 8).sum::<usize>()
                + t.len() * (std::mem::size_of::<u64>() + 24)
        }).sum();

        hyperplane_mem + table_mem
    }

    // ─── Private helpers ────────────────────────────────────────

    /// Compute the full hash for a point in a specific table.
    fn compute_hash(&self, table_idx: usize, vector: &[f32], timestamp: i64) -> u64 {
        let sem_bits = self.semantic_hash(table_idx, vector);
        let temp_bucket = self.temporal_bucket(timestamp);
        self.combine_hash(table_idx, &sem_bits, temp_bucket)
    }

    /// Compute semantic hash bits via random hyperplane LSH.
    fn semantic_hash(&self, table_idx: usize, vector: &[f32]) -> Vec<bool> {
        self.hyperplanes[table_idx]
            .iter()
            .map(|plane| {
                let dot: f32 = plane.iter().zip(vector.iter()).map(|(a, b)| a * b).sum();
                dot >= 0.0
            })
            .collect()
    }

    /// Compute temporal bucket index.
    fn temporal_bucket(&self, timestamp: i64) -> i64 {
        if self.config.temporal_bucket_us > 0 {
            timestamp / self.config.temporal_bucket_us
        } else {
            0
        }
    }

    /// Combine semantic bits and temporal bucket into a single u64 hash.
    fn combine_hash(&self, _table_idx: usize, sem_bits: &[bool], temp_bucket: i64) -> u64 {
        let mut hash: u64 = 0;

        // Pack semantic bits into lower bits
        for (i, &bit) in sem_bits.iter().enumerate() {
            if bit {
                hash |= 1u64 << i;
            }
        }

        // Pack temporal bucket into upper bits
        let temp_hash = temp_bucket as u64;
        hash |= temp_hash << self.config.semantic_bits;

        hash
    }
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> TLSHConfig {
        TLSHConfig {
            n_tables: 4,
            semantic_bits: 8,
            temporal_bits: 4,
            temporal_bucket_us: 1_000_000, // 1 second for testing
            n_probes: 2,
        }
    }

    // ─── Basic operations ───────────────────────────────────────

    #[test]
    fn new_empty() {
        let index = TemporalLSH::new(4, default_config());
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn insert_and_query_identical() {
        let mut index = TemporalLSH::new(3, default_config());
        let v = [1.0f32, 0.0, 0.0];
        let ts = 1_000_000;

        index.insert(0, &v, ts);
        let candidates = index.query(&v, ts);

        assert!(
            candidates.contains(&0),
            "query with identical vector+timestamp should find the point"
        );
    }

    #[test]
    fn insert_multiple_query_nearest() {
        let config = default_config();
        let mut index = TemporalLSH::new(3, config);

        // Insert 100 points at various positions and timestamps
        for i in 0..100u32 {
            let v = [i as f32 * 0.1, (i as f32 * 0.05).sin(), 0.0];
            let ts = i as i64 * 500_000; // spread over 50 seconds
            index.insert(i, &v, ts);
        }

        assert_eq!(index.len(), 100);

        // Query near point 50
        let query_v = [5.0, (50.0 * 0.05f32).sin(), 0.0];
        let query_ts = 25_000_000;
        let candidates = index.query(&query_v, query_ts);

        // Should find some candidates (may not be exact, it's LSH)
        assert!(
            !candidates.is_empty(),
            "should find at least one candidate"
        );
    }

    // ─── Temporal locality ──────────────────────────────────────

    #[test]
    fn temporal_neighbors_found_via_multiprobe() {
        let config = TLSHConfig {
            n_tables: 8,
            semantic_bits: 8,
            temporal_bits: 4,
            temporal_bucket_us: 1_000_000, // 1 second buckets
            n_probes: 3,
        };
        let mut index = TemporalLSH::new(2, config);

        // Insert point at t=0
        index.insert(0, &[1.0, 0.0], 0);
        // Insert point at t=2s (2 buckets away)
        index.insert(1, &[1.0, 0.0], 2_000_000);

        // Query at t=1s — should find both via multi-probe
        let candidates = index.query(&[1.0, 0.0], 1_000_000);

        // With n_probes=3, should probe buckets -3 to +3 around bucket 1
        // Bucket 0 (point 0) and bucket 2 (point 1) should be found
        let found_0 = candidates.contains(&0);
        let found_1 = candidates.contains(&1);
        assert!(
            found_0 || found_1,
            "multi-probe should find at least one temporal neighbor, got {:?}",
            candidates
        );
    }

    // ─── Semantic locality ──────────────────────────────────────

    #[test]
    fn similar_vectors_same_bucket() {
        let config = TLSHConfig {
            n_tables: 16,
            semantic_bits: 8,
            temporal_bits: 2,
            temporal_bucket_us: 1_000_000,
            n_probes: 1,
        };
        let mut index = TemporalLSH::new(4, config);

        // Two very similar vectors at the same time
        index.insert(0, &[1.0, 0.0, 0.0, 0.0], 0);
        index.insert(1, &[0.99, 0.01, 0.0, 0.0], 0);
        // One very different vector
        index.insert(2, &[-1.0, 0.0, 0.0, 0.0], 0);

        let candidates = index.query(&[1.0, 0.0, 0.0, 0.0], 0);

        // Point 0 should definitely be found (exact match)
        assert!(candidates.contains(&0));
        // Point 1 should likely be found (very similar)
        // Point 2 may or may not be found (opposite direction)
    }

    // ─── Build from data ────────────────────────────────────────

    #[test]
    fn build_from_vectors() {
        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![i as f32, 0.0])
            .collect();
        let timestamps: Vec<i64> = (0..50)
            .map(|i| i as i64 * 1_000_000)
            .collect();

        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let index = TemporalLSH::build(&refs, &timestamps, default_config());

        assert_eq!(index.len(), 50);
    }

    // ─── Config tuning ──────────────────────────────────────────

    #[test]
    fn config_for_alpha_high() {
        let config = TLSHConfig::for_alpha(0.9, 384);
        // High alpha → more semantic bits
        assert!(config.semantic_bits > config.temporal_bits);
    }

    #[test]
    fn config_for_alpha_balanced() {
        let config = TLSHConfig::for_alpha(0.5, 384);
        // Balanced alpha → roughly equal bits
        let diff = (config.semantic_bits as i32 - config.temporal_bits as i32).unsigned_abs();
        assert!(diff <= 2, "balanced alpha should give roughly equal bits");
    }

    #[test]
    fn config_for_alpha_low() {
        let config = TLSHConfig::for_alpha(0.2, 384);
        // Low alpha → more temporal bits
        assert!(config.temporal_bits > config.semantic_bits);
    }

    // ─── Memory estimate ────────────────────────────────────────

    #[test]
    fn memory_estimate_grows_with_data() {
        let config = default_config();
        let mut index = TemporalLSH::new(4, config);
        let mem_empty = index.memory_bytes();

        for i in 0..100u32 {
            index.insert(i, &[i as f32, 0.0, 0.0, 0.0], i as i64 * 1000);
        }
        let mem_full = index.memory_bytes();

        assert!(
            mem_full > mem_empty,
            "memory should grow with inserted points"
        );
    }

    // ─── Edge cases ─────────────────────────────────────────────

    #[test]
    fn query_empty_index() {
        let index = TemporalLSH::new(3, default_config());
        let candidates = index.query(&[1.0, 0.0, 0.0], 0);
        assert!(candidates.is_empty());
    }

    #[test]
    fn negative_timestamps() {
        let mut index = TemporalLSH::new(2, default_config());
        index.insert(0, &[1.0, 0.0], -5_000_000);
        index.insert(1, &[1.0, 0.0], -3_000_000);

        let candidates = index.query(&[1.0, 0.0], -4_000_000);
        assert!(!candidates.is_empty(), "should handle negative timestamps");
    }

    #[test]
    fn high_dimensional() {
        let dim = 384;
        let config = TLSHConfig {
            n_tables: 4,
            semantic_bits: 12,
            temporal_bits: 4,
            temporal_bucket_us: 1_000_000,
            n_probes: 2,
        };
        let mut index = TemporalLSH::new(dim, config);

        let v: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
        index.insert(0, &v, 0);

        let candidates = index.query(&v, 0);
        assert!(candidates.contains(&0));
    }

    // ─── Hash consistency ───────────────────────────────────────

    #[test]
    fn same_input_same_hash() {
        let index = TemporalLSH::new(3, default_config());
        let v = [1.0f32, 2.0, 3.0];
        let ts = 5_000_000;

        let h1 = index.compute_hash(0, &v, ts);
        let h2 = index.compute_hash(0, &v, ts);
        assert_eq!(h1, h2, "same input should produce same hash");
    }

    #[test]
    fn different_time_different_hash() {
        let index = TemporalLSH::new(3, default_config());
        let v = [1.0f32, 0.0, 0.0];

        // Different temporal buckets should give different hashes (usually)
        let h1 = index.compute_hash(0, &v, 0);
        let h2 = index.compute_hash(0, &v, 10_000_000); // 10 seconds later, different bucket

        // They might be the same if semantic bits dominate, but for 1s buckets
        // 10 seconds apart should differ
        assert_ne!(h1, h2, "different temporal buckets should usually give different hashes");
    }
}
