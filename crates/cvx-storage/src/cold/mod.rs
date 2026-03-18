//! Cold storage tier with Product Quantization (PQ) compression.
//!
//! Vectors are encoded into compact PQ codes for massive storage reduction.
//! A codebook of centroids is trained via k-means on representative data,
//! then each vector is encoded as a sequence of centroid indices.
//!
//! ## Compression
//!
//! With M=8 subspaces and K=256 centroids per subspace:
//! - Original D=768 vector: 768 × 4 bytes = 3,072 bytes
//! - PQ code: 8 × 1 byte = 8 bytes
//! - **Compression ratio: 384×**
//!
//! ## Asymmetric Distance Computation (ADC)
//!
//! Query-to-code distance is computed without decoding:
//! precompute query-to-centroid distances, then sum lookup table entries.

/// Product Quantization codebook.
#[derive(Debug, Clone)]
pub struct PqCodebook {
    /// Number of subspaces.
    pub m: usize,
    /// Number of centroids per subspace.
    pub k: usize,
    /// Original vector dimensionality.
    pub dim: usize,
    /// Centroids: `[subspace][centroid][sub_dim]`.
    /// Flattened: length = m * k * (dim / m).
    pub centroids: Vec<f32>,
}

impl PqCodebook {
    /// Train a codebook from a set of vectors using k-means.
    ///
    /// - `vectors`: training data (each of length `dim`)
    /// - `m`: number of subspaces
    /// - `k`: centroids per subspace
    /// - `iterations`: k-means iterations
    pub fn train(vectors: &[&[f32]], m: usize, k: usize, iterations: usize) -> Self {
        assert!(!vectors.is_empty(), "need training data");
        let dim = vectors[0].len();
        assert!(dim % m == 0, "dim must be divisible by m");
        let sub_dim = dim / m;

        let mut centroids = vec![0.0f32; m * k * sub_dim];

        for sub in 0..m {
            let offset = sub * sub_dim;

            // k-means++ initialization (Arthur & Vassilvitskii, SODA 2007)
            // Sample centroids proportional to D²(x) for O(log k) approximation.
            // See RFC-002-09.
            {
                // First centroid: pick the first vector's subvector
                let src = vectors[0];
                for d in 0..sub_dim {
                    centroids[sub * k * sub_dim + d] = src[offset + d];
                }
                let mut rng_state: u64 = 42 + sub as u64;

                for c in 1..k {
                    // Compute D²(x): min squared distance to any existing centroid
                    let weights: Vec<f64> = vectors
                        .iter()
                        .map(|v| {
                            let sub_vec = &v[offset..offset + sub_dim];
                            (0..c)
                                .map(|ci| {
                                    let base = sub * k * sub_dim + ci * sub_dim;
                                    (0..sub_dim)
                                        .map(|d| {
                                            let diff = sub_vec[d] - centroids[base + d];
                                            (diff * diff) as f64
                                        })
                                        .sum::<f64>()
                                })
                                .fold(f64::INFINITY, f64::min)
                        })
                        .collect();

                    // Cumulative sum for weighted sampling
                    let total: f64 = weights.iter().sum();
                    if total <= 0.0 {
                        // All points coincide with existing centroids; just cycle
                        let src = vectors[c % vectors.len()];
                        for d in 0..sub_dim {
                            centroids[sub * k * sub_dim + c * sub_dim + d] = src[offset + d];
                        }
                        continue;
                    }

                    // Simple LCG for deterministic sampling
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let threshold = ((rng_state >> 33) as f64 / u32::MAX as f64) * total;

                    let mut cumulative = 0.0;
                    let mut selected = vectors.len() - 1;
                    for (i, w) in weights.iter().enumerate() {
                        cumulative += w;
                        if cumulative >= threshold {
                            selected = i;
                            break;
                        }
                    }

                    let src = vectors[selected];
                    for d in 0..sub_dim {
                        centroids[sub * k * sub_dim + c * sub_dim + d] = src[offset + d];
                    }
                }
            }

            // K-means iterations
            for _ in 0..iterations {
                let mut sums = vec![0.0f64; k * sub_dim];
                let mut counts = vec![0usize; k];

                // Assign
                for &v in vectors {
                    let sub_vec = &v[offset..offset + sub_dim];
                    let closest = find_closest_centroid(sub_vec, &centroids, sub, k, sub_dim);
                    counts[closest] += 1;
                    for d in 0..sub_dim {
                        sums[closest * sub_dim + d] += sub_vec[d] as f64;
                    }
                }

                // Update
                for c in 0..k {
                    if counts[c] > 0 {
                        for d in 0..sub_dim {
                            centroids[sub * k * sub_dim + c * sub_dim + d] =
                                (sums[c * sub_dim + d] / counts[c] as f64) as f32;
                        }
                    }
                }
            }
        }

        PqCodebook {
            m,
            k,
            dim,
            centroids,
        }
    }

    /// Subdimension size.
    pub fn sub_dim(&self) -> usize {
        self.dim / self.m
    }

    /// Encode a vector into PQ codes.
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        assert_eq!(vector.len(), self.dim);
        let sub_dim = self.sub_dim();
        let mut codes = Vec::with_capacity(self.m);

        for sub in 0..self.m {
            let offset = sub * sub_dim;
            let sub_vec = &vector[offset..offset + sub_dim];
            let closest = find_closest_centroid(sub_vec, &self.centroids, sub, self.k, sub_dim);
            codes.push(closest as u8);
        }

        codes
    }

    /// Decode PQ codes back to an approximate vector.
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        assert_eq!(codes.len(), self.m);
        let sub_dim = self.sub_dim();
        let mut vector = Vec::with_capacity(self.dim);

        for (sub, &code) in codes.iter().enumerate() {
            let base = sub * self.k * sub_dim + (code as usize) * sub_dim;
            vector.extend_from_slice(&self.centroids[base..base + sub_dim]);
        }

        vector
    }

    /// Build asymmetric distance table for a query vector.
    ///
    /// Returns a table of size `[m][k]` where `table[sub][code]` is the
    /// squared distance from the query subvector to that centroid.
    pub fn build_distance_table(&self, query: &[f32]) -> Vec<Vec<f32>> {
        assert_eq!(query.len(), self.dim);
        let sub_dim = self.sub_dim();

        (0..self.m)
            .map(|sub| {
                let q_offset = sub * sub_dim;
                (0..self.k)
                    .map(|c| {
                        let c_base = sub * self.k * sub_dim + c * sub_dim;
                        (0..sub_dim)
                            .map(|d| {
                                let diff = query[q_offset + d] - self.centroids[c_base + d];
                                diff * diff
                            })
                            .sum()
                    })
                    .collect()
            })
            .collect()
    }

    /// Compute asymmetric distance from query to a PQ code using precomputed table.
    pub fn asymmetric_distance(table: &[Vec<f32>], codes: &[u8]) -> f32 {
        codes
            .iter()
            .enumerate()
            .map(|(sub, &code)| table[sub][code as usize])
            .sum()
    }
}

fn find_closest_centroid(
    sub_vec: &[f32],
    centroids: &[f32],
    sub: usize,
    k: usize,
    sub_dim: usize,
) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f32::INFINITY;

    for c in 0..k {
        let base = sub * k * sub_dim + c * sub_dim;
        let dist: f32 = (0..sub_dim)
            .map(|d| {
                let diff = sub_vec[d] - centroids[base + d];
                diff * diff
            })
            .sum();
        if dist < best_dist {
            best_dist = dist;
            best_idx = c;
        }
    }

    best_idx
}

/// Cold store using PQ-encoded vectors.
pub struct ColdStore {
    codebook: PqCodebook,
    /// Encoded vectors: (entity_id, space_id, timestamp, pq_codes).
    entries: Vec<ColdEntry>,
}

/// A single entry in cold storage.
#[derive(Debug, Clone)]
struct ColdEntry {
    entity_id: u64,
    space_id: u32,
    timestamp: i64,
    codes: Vec<u8>,
}

impl ColdStore {
    /// Create a cold store with a trained codebook.
    pub fn new(codebook: PqCodebook) -> Self {
        Self {
            codebook,
            entries: Vec::new(),
        }
    }

    /// Store a vector (encodes it with PQ).
    pub fn put(&mut self, entity_id: u64, space_id: u32, timestamp: i64, vector: &[f32]) {
        let codes = self.codebook.encode(vector);
        self.entries.push(ColdEntry {
            entity_id,
            space_id,
            timestamp,
            codes,
        });
    }

    /// Retrieve and decode a vector.
    pub fn get(&self, entity_id: u64, space_id: u32, timestamp: i64) -> Option<Vec<f32>> {
        self.entries
            .iter()
            .find(|e| {
                e.entity_id == entity_id && e.space_id == space_id && e.timestamp == timestamp
            })
            .map(|e| self.codebook.decode(&e.codes))
    }

    /// Number of stored entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Storage size in bytes (codes only, no overhead).
    pub fn storage_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.codes.len()).sum()
    }

    /// Access the codebook.
    pub fn codebook(&self) -> &PqCodebook {
        &self.codebook
    }

    /// Search using asymmetric distance computation.
    ///
    /// Returns `(entity_id, timestamp, distance)` sorted by distance.
    pub fn search_adc(&self, query: &[f32], k: usize) -> Vec<(u64, i64, f32)> {
        let table = self.codebook.build_distance_table(query);
        let mut scored: Vec<(u64, i64, f32)> = self
            .entries
            .iter()
            .map(|e| {
                let dist = PqCodebook::asymmetric_distance(&table, &e.codes);
                (e.entity_id, e.timestamp, dist)
            })
            .collect();
        scored.sort_by(|a, b| a.2.total_cmp(&b.2));
        scored.truncate(k);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ((state >> 33) as f32) / (u32::MAX as f32) - 0.5
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn train_and_encode_decode() {
        let vectors = random_vectors(100, 32, 42);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let codebook = PqCodebook::train(&refs, 4, 16, 10);

        assert_eq!(codebook.m, 4);
        assert_eq!(codebook.k, 16);
        assert_eq!(codebook.dim, 32);
        assert_eq!(codebook.sub_dim(), 8);

        // Encode and decode
        let codes = codebook.encode(&vectors[0]);
        assert_eq!(codes.len(), 4);

        let decoded = codebook.decode(&codes);
        assert_eq!(decoded.len(), 32);
    }

    #[test]
    fn decode_approximates_original() {
        let vectors = random_vectors(500, 64, 42);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let codebook = PqCodebook::train(&refs, 8, 256, 20);

        // Measure reconstruction error
        let mut total_error = 0.0f64;
        for v in &vectors {
            let codes = codebook.encode(v);
            let decoded = codebook.decode(&codes);
            let error: f64 = v
                .iter()
                .zip(decoded.iter())
                .map(|(a, b)| ((*a - *b) as f64).powi(2))
                .sum();
            total_error += error;
        }
        let avg_error = total_error / vectors.len() as f64;

        // PQ should have reasonable reconstruction error
        assert!(
            avg_error < 10.0,
            "avg reconstruction error too high: {avg_error:.4}"
        );
    }

    #[test]
    fn compression_ratio() {
        let dim = 768;
        let m = 8;
        let n = 100;
        let vectors = random_vectors(n, dim, 42);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let codebook = PqCodebook::train(&refs, m, 256, 5);

        // Actually encode all vectors and measure real sizes
        let mut total_code_bytes = 0usize;
        let mut total_reconstruction_error = 0.0f64;

        for v in &vectors {
            let codes = codebook.encode(v);
            total_code_bytes += codes.len();

            let decoded = codebook.decode(&codes);
            let error: f64 = v
                .iter()
                .zip(decoded.iter())
                .map(|(a, b)| ((*a - *b) as f64).powi(2))
                .sum();
            total_reconstruction_error += error;
        }

        let original_bytes = n * dim * 4;
        let ratio = original_bytes as f64 / total_code_bytes as f64;
        let avg_error = total_reconstruction_error / n as f64;

        assert!(
            ratio >= 300.0,
            "compression ratio = {ratio:.0}x, expected >= 300x for D={dim} M={m}"
        );

        // Verify each code is M bytes (1 byte per subspace with K=256)
        assert_eq!(total_code_bytes, n * m);

        // Reconstruction error should be bounded (PQ is lossy but usable)
        assert!(
            avg_error < 50.0,
            "avg reconstruction error = {avg_error:.2}, expected < 50 for D={dim}"
        );
    }

    #[test]
    fn cold_store_put_get() {
        let vectors = random_vectors(50, 32, 42);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let codebook = PqCodebook::train(&refs, 4, 16, 10);
        let mut store = ColdStore::new(codebook);

        store.put(1, 0, 1000, &vectors[0]);
        store.put(1, 0, 2000, &vectors[1]);

        assert_eq!(store.len(), 2);

        let decoded = store.get(1, 0, 1000).unwrap();
        assert_eq!(decoded.len(), 32);
    }

    #[test]
    fn cold_store_get_nonexistent() {
        let vectors = random_vectors(10, 16, 42);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let codebook = PqCodebook::train(&refs, 2, 8, 5);
        let store = ColdStore::new(codebook);

        assert!(store.get(999, 0, 0).is_none());
    }

    #[test]
    fn adc_search() {
        let dim = 32;
        let vectors = random_vectors(200, dim, 42);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let codebook = PqCodebook::train(&refs, 4, 32, 10);
        let mut store = ColdStore::new(codebook);

        for (i, v) in vectors.iter().enumerate() {
            store.put(i as u64, 0, (i as i64) * 1000, v);
        }

        let results = store.search_adc(&vectors[0], 5);
        assert_eq!(results.len(), 5);

        // First result should be the query itself (or very close)
        assert_eq!(results[0].0, 0, "closest should be the query vector itself");
    }

    #[test]
    fn storage_bytes_compact() {
        let vectors = random_vectors(1000, 768, 42);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let codebook = PqCodebook::train(&refs, 8, 256, 5);
        let mut store = ColdStore::new(codebook);

        for (i, v) in vectors.iter().enumerate() {
            store.put(i as u64, 0, (i as i64) * 1000, v);
        }

        let original_bytes = 1000 * 768 * 4;
        let cold_bytes = store.storage_bytes();
        let ratio = original_bytes as f64 / cold_bytes as f64;

        assert!(
            ratio > 100.0,
            "cold storage ratio = {ratio:.0}x, expected > 100x"
        );
    }
}
