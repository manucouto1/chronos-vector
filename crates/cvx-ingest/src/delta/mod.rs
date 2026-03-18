//! Delta encoding and decoding for vector compression.
//!
//! Instead of storing every embedding in full, CVX stores **keyframes**
//! (full vectors) at regular intervals and **deltas** (sparse changes)
//! between them. This is analogous to video compression (I-frames + P-frames).
//!
//! A delta is sparse: only dimensions that changed by more than ε are stored.
//! For slowly-drifting embeddings, most dimensions don't change between updates,
//! yielding 3-10x compression.
//!
//! # Example
//!
//! ```
//! use cvx_ingest::delta::{DeltaEncoder, DeltaDecoder};
//!
//! let encoder = DeltaEncoder::new(10, 0.001); // keyframe every 10, threshold 0.001
//!
//! let v1 = vec![1.0, 2.0, 3.0];
//! let v2 = vec![1.0, 2.1, 3.0]; // only dim 1 changed above ε
//!
//! let entry1 = encoder.encode(0, 1000, &v1, None);
//! assert!(entry1.is_keyframe()); // first vector is always a keyframe
//!
//! let entry2 = encoder.encode(1, 2000, &v2, Some(&v1));
//! assert!(!entry2.is_keyframe());
//! assert_eq!(entry2.nnz(), 1); // only 1 dimension stored
//!
//! // Reconstruct v2 from keyframe + delta
//! let reconstructed = DeltaDecoder::apply(&v1, &entry2);
//! assert!((reconstructed[1] - 2.1).abs() < 1e-7);
//! ```

use cvx_core::DeltaEntry;

/// Encodes vectors into keyframe + delta sequences.
///
/// Configuration:
/// - `keyframe_interval`: store a full vector every K updates
/// - `threshold`: minimum absolute change per dimension to include in delta (ε)
pub struct DeltaEncoder {
    keyframe_interval: u32,
    threshold: f32,
}

impl DeltaEncoder {
    /// Create a new delta encoder.
    ///
    /// - `keyframe_interval` (K): store a full keyframe every K updates.
    ///   K=10 means every 10th vector is stored in full.
    /// - `threshold` (ε): minimum per-dimension change to include in delta.
    ///   ε=0.001 means changes smaller than 0.001 are discarded.
    pub fn new(keyframe_interval: u32, threshold: f32) -> Self {
        Self {
            keyframe_interval,
            threshold,
        }
    }

    /// Encode a vector, producing either a keyframe or a delta entry.
    ///
    /// - `sequence_index`: the index of this vector in the entity's sequence (0, 1, 2, ...)
    /// - `timestamp`: the timestamp for this vector
    /// - `vector`: the full vector
    /// - `previous`: the previous full vector (None for the first vector)
    ///
    /// Returns a `DeltaEntry`. If it's a keyframe, the full vector should be
    /// stored separately; if it's a delta, only the sparse changes are stored.
    pub fn encode(
        &self,
        sequence_index: u32,
        timestamp: i64,
        vector: &[f32],
        previous: Option<&[f32]>,
    ) -> DeltaEntry {
        // First vector or every K-th vector is a keyframe
        if previous.is_none() || sequence_index % self.keyframe_interval == 0 {
            return DeltaEntry::keyframe(0, timestamp);
        }

        let prev = previous.unwrap();
        assert_eq!(
            vector.len(),
            prev.len(),
            "vector dimensions must match: {} vs {}",
            vector.len(),
            prev.len()
        );

        // Compute sparse delta: only store dimensions that changed by more than ε
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (i, (&curr, &prev_val)) in vector.iter().zip(prev.iter()).enumerate() {
            let delta = curr - prev_val;
            if delta.abs() > self.threshold {
                indices.push(i as u32);
                values.push(delta);
            }
        }

        DeltaEntry::delta(0, timestamp - 1, timestamp, indices, values)
    }

    /// The configured keyframe interval.
    pub fn keyframe_interval(&self) -> u32 {
        self.keyframe_interval
    }

    /// The configured threshold (ε).
    pub fn threshold(&self) -> f32 {
        self.threshold
    }
}

/// Reconstructs full vectors from keyframes and delta chains.
pub struct DeltaDecoder;

impl DeltaDecoder {
    /// Apply a delta entry to a base vector to reconstruct the full vector.
    ///
    /// - `base`: the previous full vector (typically a keyframe or already-reconstructed vector)
    /// - `delta`: the delta entry to apply
    ///
    /// Returns the reconstructed full vector.
    ///
    /// # Panics
    ///
    /// Panics if any delta index is out of bounds for the base vector.
    pub fn apply(base: &[f32], delta: &DeltaEntry) -> Vec<f32> {
        let mut result = base.to_vec();
        for (&idx, &val) in delta.indices().iter().zip(delta.values().iter()) {
            result[idx as usize] += val;
        }
        result
    }

    /// Reconstruct a full vector from a keyframe and a chain of deltas.
    ///
    /// - `keyframe`: the base full vector
    /// - `deltas`: sequence of delta entries to apply in order
    ///
    /// Returns the final reconstructed vector.
    pub fn reconstruct(keyframe: &[f32], deltas: &[DeltaEntry]) -> Vec<f32> {
        let mut current = keyframe.to_vec();
        for delta in deltas {
            if delta.is_keyframe() {
                continue; // keyframes are stored separately, skip
            }
            current = Self::apply(&current, delta);
        }
        current
    }
}

/// Compute the compression ratio: full_size / delta_size.
///
/// Returns how many times smaller the delta representation is.
pub fn compression_ratio(dim: usize, deltas: &[DeltaEntry]) -> f64 {
    let full_size = deltas.len() * dim * 4; // f32 = 4 bytes per dimension
    let delta_size: usize = deltas
        .iter()
        .map(|d| {
            if d.is_keyframe() {
                dim * 4 // keyframes store full vector
            } else {
                d.nnz() * (4 + 4) // index (u32) + value (f32) per non-zero
            }
        })
        .sum();
    if delta_size == 0 {
        return 0.0;
    }
    full_size as f64 / delta_size as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_vector_is_keyframe() {
        let enc = DeltaEncoder::new(10, 0.001);
        let v = vec![1.0, 2.0, 3.0];
        let entry = enc.encode(0, 1000, &v, None);
        assert!(entry.is_keyframe());
    }

    #[test]
    fn keyframe_every_k() {
        let enc = DeltaEncoder::new(5, 0.001);
        let v = vec![1.0; 10];
        for i in 0..20u32 {
            let entry = enc.encode(i, i as i64 * 1000, &v, Some(&v));
            if i % 5 == 0 {
                assert!(entry.is_keyframe(), "index {i} should be keyframe");
            } else {
                assert!(!entry.is_keyframe(), "index {i} should be delta");
            }
        }
    }

    #[test]
    fn delta_only_stores_changed_dims() {
        let enc = DeltaEncoder::new(100, 0.01);
        let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let v2 = vec![1.0, 2.05, 3.0, 4.0, 5.1]; // dims 1 and 4 changed

        let entry = enc.encode(1, 2000, &v2, Some(&v1));
        assert!(!entry.is_keyframe());
        assert_eq!(entry.nnz(), 2);
        assert_eq!(entry.indices(), &[1, 4]);
    }

    #[test]
    fn small_changes_below_threshold_ignored() {
        let enc = DeltaEncoder::new(100, 0.01);
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![1.005, 2.005, 3.005]; // all changes < ε=0.01

        let entry = enc.encode(1, 2000, &v2, Some(&v1));
        assert_eq!(entry.nnz(), 0); // nothing stored
    }

    #[test]
    fn apply_reconstructs_correctly() {
        let base = vec![1.0, 2.0, 3.0, 4.0];
        let delta = DeltaEntry::delta(0, 0, 1, vec![1, 3], vec![0.5, -0.3]);

        let result = DeltaDecoder::apply(&base, &delta);
        assert!((result[0] - 1.0).abs() < 1e-7);
        assert!((result[1] - 2.5).abs() < 1e-7);
        assert!((result[2] - 3.0).abs() < 1e-7);
        assert!((result[3] - 3.7).abs() < 1e-7);
    }

    #[test]
    fn reconstruct_chain() {
        let keyframe = vec![1.0, 2.0, 3.0];
        let deltas = vec![
            DeltaEntry::keyframe(0, 0), // skipped
            DeltaEntry::delta(0, 0, 1, vec![0], vec![0.1]),
            DeltaEntry::delta(0, 1, 2, vec![1], vec![-0.2]),
            DeltaEntry::delta(0, 2, 3, vec![0, 2], vec![0.3, 0.5]),
        ];

        let result = DeltaDecoder::reconstruct(&keyframe, &deltas);
        assert!((result[0] - 1.4).abs() < 1e-6); // 1.0 + 0.1 + 0.3
        assert!((result[1] - 1.8).abs() < 1e-6); // 2.0 - 0.2
        assert!((result[2] - 3.5).abs() < 1e-6); // 3.0 + 0.5
    }

    #[test]
    fn encode_decode_roundtrip_precision() {
        let enc = DeltaEncoder::new(100, 1e-8); // very low threshold
        let dim = 768;
        let mut rng = 42u64; // simple PRNG for determinism
        let mut pseudo_random = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 33) as f32 / (u32::MAX >> 1) as f32
        };

        let v1: Vec<f32> = (0..dim).map(|_| pseudo_random()).collect();
        // Slowly drifting: add small perturbations
        let v2: Vec<f32> = v1.iter().map(|&x| x + pseudo_random() * 0.001).collect();

        let entry = enc.encode(1, 1000, &v2, Some(&v1));
        let reconstructed = DeltaDecoder::apply(&v1, &entry);

        for (i, (&original, &recovered)) in v2.iter().zip(reconstructed.iter()).enumerate() {
            assert!(
                (original - recovered).abs() < 1e-7,
                "dim {i}: original={original}, recovered={recovered}"
            );
        }
    }

    #[test]
    fn compression_ratio_slow_drift() {
        let enc = DeltaEncoder::new(10, 0.01);
        let dim = 768;
        let n_vectors = 100;

        let mut rng = 123u64;
        let mut pseudo_random = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 33) as f32 / (u32::MAX >> 1) as f32
        };

        let base: Vec<f32> = (0..dim).map(|_| pseudo_random()).collect();
        let mut prev = base.clone();
        let mut deltas = Vec::new();

        for i in 0..n_vectors {
            // Slow drift: ~5% of dimensions change per step
            let current: Vec<f32> = prev
                .iter()
                .map(|&x| {
                    if pseudo_random() < 0.05 {
                        x + pseudo_random() * 0.1
                    } else {
                        x + pseudo_random() * 0.001 // below threshold
                    }
                })
                .collect();
            let entry = enc.encode(i as u32, i as i64 * 1000, &current, Some(&prev));
            deltas.push(entry);
            prev = current;
        }

        let ratio = compression_ratio(dim, &deltas);
        assert!(
            ratio >= 3.0,
            "compression ratio {ratio:.1}x, expected >= 3x for slow drift"
        );
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Apply(base, encode(base, target)) ≈ target
        #[test]
        fn encode_decode_roundtrip(
            base in prop::collection::vec(-10.0f32..10.0, 32..=32),
            perturbation in prop::collection::vec(-0.1f32..0.1, 32..=32),
        ) {
            let target: Vec<f32> = base.iter().zip(perturbation.iter())
                .map(|(&b, &p)| b + p).collect();

            let enc = DeltaEncoder::new(100, 1e-8);
            let entry = enc.encode(1, 1000, &target, Some(&base));
            let recovered = DeltaDecoder::apply(&base, &entry);

            for (i, (&orig, &rec)) in target.iter().zip(recovered.iter()).enumerate() {
                prop_assert!((orig - rec).abs() < 1e-6,
                    "dim {i}: orig={orig}, rec={rec}");
            }
        }
    }
}
