//! Path signatures for trajectory characterization.
//!
//! Implements truncated path signatures from rough path theory — a **universal**
//! and **order-aware** feature of sequential data. Any continuous function of a
//! path can be approximated by a linear function of its signature.
//!
//! # Key Properties
//!
//! - **Universality**: sufficient statistics for trajectory classification
//! - **Reparametrization invariance**: captures shape, not sampling rate
//! - **Chen's Identity**: `S(α * β) = S(α) ⊗ S(β)` — incremental updates in O(K²)
//! - **Hierarchical**: depth 1 = displacement, depth 2 = signed area (correlation/volatility)
//!
//! # Usage with CVX
//!
//! Path signatures operate on **region trajectories** (K~80 dims at L3), NOT on
//! raw embeddings (D=768). The HNSW graph hierarchy provides the dimensionality
//! reduction that makes signatures tractable:
//!
//! - Region trajectory at L3: K=80 → depth 2 signature = 80 + 6400 = 6,480 features
//! - Raw embeddings: D=768 → depth 2 = 768 + 589,824 → intractable
//!
//! # References
//!
//! - Lyons, T.J. (1998). Differential equations driven by rough signals.
//! - Chevyrev & Kormilitzin (2016). A primer on the signature method in ML.
//! - Kidger & Lyons (2021). Signatory: differentiable computations of the signature.

/// Configuration for signature computation.
#[derive(Debug, Clone)]
pub struct SignatureConfig {
    /// Truncation depth (1-3). Higher = more expressive but larger output.
    pub depth: usize,
    /// Whether to augment with time as an extra dimension.
    /// Breaks reparametrization invariance but captures speed information.
    pub time_augmentation: bool,
}

impl Default for SignatureConfig {
    fn default() -> Self {
        Self {
            depth: 2,
            time_augmentation: false,
        }
    }
}

/// Computed path signature result.
#[derive(Debug, Clone)]
pub struct PathSignatureResult {
    /// Truncation depth used.
    pub depth: usize,
    /// Input dimensionality (K for region trajectories).
    pub input_dim: usize,
    /// The signature vector (concatenation of all levels up to depth).
    pub signature: Vec<f64>,
    /// Total output dimensionality.
    pub output_dim: usize,
}

/// Compute the truncated path signature of a trajectory.
///
/// The trajectory is a sequence of (timestamp, vector) pairs. The signature
/// captures the ordered, multi-scale structure of the path.
///
/// # Arguments
///
/// * `trajectory` - Sequence of (timestamp, vector) pairs, ordered by time.
/// * `config` - Truncation depth and options.
///
/// # Returns
///
/// The concatenated signature up to the specified depth:
/// - Depth 1: K features (net displacement per dimension)
/// - Depth 2: K + K² features (displacement + signed areas)
/// - Depth 3: K + K² + K³ features
///
/// # Complexity
///
/// O(N × K^depth) where N = trajectory length, K = dimensionality.
pub fn compute_signature(
    trajectory: &[(i64, &[f32])],
    config: &SignatureConfig,
) -> Result<PathSignatureResult, SignatureError> {
    if trajectory.len() < 2 {
        return Err(SignatureError::InsufficientData {
            got: trajectory.len(),
            need: 2,
        });
    }

    let k = trajectory[0].1.len();
    if k == 0 {
        return Err(SignatureError::ZeroDimension);
    }

    // Optionally augment with time dimension
    let (points, dim) = if config.time_augmentation {
        let t0 = trajectory[0].0 as f64;
        let t_range = (trajectory.last().unwrap().0 - trajectory[0].0).max(1) as f64;
        let augmented: Vec<Vec<f64>> = trajectory
            .iter()
            .map(|(ts, vec)| {
                let mut p = Vec::with_capacity(k + 1);
                p.push((*ts as f64 - t0) / t_range); // normalized time [0, 1]
                p.extend(vec.iter().map(|&v| v as f64));
                p
            })
            .collect();
        (augmented, k + 1)
    } else {
        let points: Vec<Vec<f64>> = trajectory
            .iter()
            .map(|(_, vec)| vec.iter().map(|&v| v as f64).collect())
            .collect();
        (points, k)
    };

    // Compute increments: dx[i] = points[i+1] - points[i]
    let n = points.len();
    let increments: Vec<Vec<f64>> = (0..n - 1)
        .map(|i| (0..dim).map(|d| points[i + 1][d] - points[i][d]).collect())
        .collect();

    let mut signature = Vec::new();
    let mut output_dim = 0;

    // Depth 1: S^i = sum of increments = net displacement
    if config.depth >= 1 {
        let mut level1 = vec![0.0f64; dim];
        for dx in &increments {
            for d in 0..dim {
                level1[d] += dx[d];
            }
        }
        output_dim += dim;
        signature.extend_from_slice(&level1);
    }

    // Depth 2: S^{i,j} = sum_{s<t} dx_s^i * dx_t^j (iterated integral approximation)
    if config.depth >= 2 {
        let mut level2 = vec![0.0f64; dim * dim];
        // Running sum for Chen's identity-style incremental computation:
        // S^{i,j} = sum_t (cumsum_i[t] * dx_t^j)
        let mut cumsum = vec![0.0f64; dim];
        for dx in &increments {
            // Accumulate: level2[i*dim + j] += cumsum[i] * dx[j]
            for i in 0..dim {
                for j in 0..dim {
                    level2[i * dim + j] += cumsum[i] * dx[j];
                }
            }
            // Update cumulative sum
            for d in 0..dim {
                cumsum[d] += dx[d];
            }
        }
        output_dim += dim * dim;
        signature.extend_from_slice(&level2);
    }

    // Depth 3: S^{i,j,k} (optional, expensive for large dim)
    if config.depth >= 3 {
        let mut level3 = vec![0.0f64; dim * dim * dim];
        // S^{i,j,k} = sum_t cumsum2[i,j] * dx_t[k]
        // where cumsum2[i,j] = running sum of level2 contributions up to t
        let mut cumsum1 = vec![0.0f64; dim];
        let mut cumsum2 = vec![0.0f64; dim * dim];
        for dx in &increments {
            // Accumulate depth 3
            for i in 0..dim {
                for j in 0..dim {
                    for kk in 0..dim {
                        level3[i * dim * dim + j * dim + kk] += cumsum2[i * dim + j] * dx[kk];
                    }
                }
            }
            // Update cumsum2: cumsum2[i,j] += cumsum1[i] * dx[j]
            for i in 0..dim {
                for j in 0..dim {
                    cumsum2[i * dim + j] += cumsum1[i] * dx[j];
                }
            }
            // Update cumsum1
            for d in 0..dim {
                cumsum1[d] += dx[d];
            }
        }
        output_dim += dim * dim * dim;
        signature.extend_from_slice(&level3);
    }

    Ok(PathSignatureResult {
        depth: config.depth,
        input_dim: dim,
        signature,
        output_dim,
    })
}

/// Compute the log-signature (compact alternative via antisymmetric part).
///
/// For depth 2, the log-signature extracts the antisymmetric part of the
/// level-2 signature: `L^{i,j} = (S^{i,j} - S^{j,i}) / 2`.
/// This removes redundant symmetric components.
///
/// Dimension at depth 2: K + K(K-1)/2 (vs K + K² for full signature).
pub fn compute_log_signature(
    trajectory: &[(i64, &[f32])],
    config: &SignatureConfig,
) -> Result<PathSignatureResult, SignatureError> {
    // First compute full signature
    let full = compute_signature(trajectory, config)?;
    let dim = full.input_dim;

    let mut log_sig = Vec::new();

    // Level 1: same as signature (displacement)
    log_sig.extend_from_slice(&full.signature[..dim]);

    // Level 2: antisymmetric part only (upper triangle of S^{i,j} - S^{j,i})
    if config.depth >= 2 {
        let level2_start = dim;
        for i in 0..dim {
            for j in (i + 1)..dim {
                let s_ij = full.signature[level2_start + i * dim + j];
                let s_ji = full.signature[level2_start + j * dim + i];
                log_sig.push((s_ij - s_ji) / 2.0);
            }
        }
    }

    // Level 3 log-signature is more complex (Dynkin map / BCH formula).
    // For now, we include the full level 3 if requested.
    if config.depth >= 3 {
        let level3_start = dim + dim * dim;
        let level3_len = dim * dim * dim;
        if full.signature.len() >= level3_start + level3_len {
            log_sig.extend_from_slice(&full.signature[level3_start..level3_start + level3_len]);
        }
    }

    let output_dim = log_sig.len();
    Ok(PathSignatureResult {
        depth: config.depth,
        input_dim: dim,
        signature: log_sig,
        output_dim,
    })
}

/// Incrementally update a signature when a new point is appended.
///
/// Uses Chen's Identity: S(α * β) = S(α) ⊗ S(β)
/// where β is the single-segment path from the last point to the new point.
///
/// Cost: O(K²) for depth 2 (vs O(N × K²) for full recomputation).
///
/// # Arguments
///
/// * `existing` - The current signature (will be modified in-place).
/// * `last_point` - The last point of the existing trajectory.
/// * `new_point` - The new point being appended.
pub fn update_signature_incremental(
    existing: &mut PathSignatureResult,
    last_point: &[f32],
    new_point: &[f32],
) -> Result<(), SignatureError> {
    let dim = existing.input_dim;
    if last_point.len() != dim || new_point.len() != dim {
        return Err(SignatureError::DimensionMismatch {
            expected: dim,
            got: new_point.len(),
        });
    }

    // Increment: dx = new_point - last_point
    let dx: Vec<f64> = (0..dim)
        .map(|d| new_point[d] as f64 - last_point[d] as f64)
        .collect();

    // Chen's Identity for depth ≤ 2:
    // S(α * β)^i = S(α)^i + dx^i
    // S(α * β)^{i,j} = S(α)^{i,j} + S(α)^i * dx^j + dx^i * dx^j / 2
    //   (the last term is the self-integral of the single segment)

    // Update level 1: displacement
    for d in 0..dim {
        existing.signature[d] += dx[d];
    }

    // Update level 2: signed area
    if existing.depth >= 2 {
        let level2_start = dim;
        // Read current level 1 BEFORE the update we just did
        // Actually, we already updated level 1. We need the value BEFORE update.
        // So: old_s1[d] = existing.signature[d] - dx[d]
        for i in 0..dim {
            let old_s1_i = existing.signature[i] - dx[i];
            for j in 0..dim {
                existing.signature[level2_start + i * dim + j] += old_s1_i * dx[j];
            }
        }
    }

    Ok(())
}

/// Compute L2 distance between two signatures.
///
/// This is a fast trajectory similarity measure: O(output_dim) per comparison,
/// capturing all order-dependent temporal dynamics.
pub fn signature_distance(a: &PathSignatureResult, b: &PathSignatureResult) -> f64 {
    if a.signature.len() != b.signature.len() {
        return f64::INFINITY;
    }
    a.signature
        .iter()
        .zip(b.signature.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Error types for signature computation.
#[derive(Debug, thiserror::Error)]
pub enum SignatureError {
    /// Not enough data points.
    #[error("insufficient data: got {got} points, need at least {need}")]
    InsufficientData {
        /// Number of points provided.
        got: usize,
        /// Minimum required.
        need: usize,
    },
    /// Zero-dimensional vectors.
    #[error("zero-dimensional input vectors")]
    ZeroDimension,
    /// Dimension mismatch in incremental update.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        got: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trajectory<'a>(points: &'a [&'a [f32]]) -> Vec<(i64, &'a [f32])> {
        points
            .iter()
            .enumerate()
            .map(|(i, p)| (i as i64, *p))
            .collect()
    }

    #[test]
    fn depth1_is_displacement() {
        let traj = make_trajectory(&[&[0.0, 0.0], &[1.0, 0.0], &[1.0, 2.0]]);
        let config = SignatureConfig {
            depth: 1,
            time_augmentation: false,
        };
        let result = compute_signature(&traj, &config).unwrap();

        assert_eq!(result.output_dim, 2);
        // Net displacement: (1.0, 2.0)
        assert!((result.signature[0] - 1.0).abs() < 1e-10);
        assert!((result.signature[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn depth2_captures_rotation() {
        // Path going right then up: (0,0) → (1,0) → (1,1)
        let traj_a = make_trajectory(&[&[0.0, 0.0], &[1.0, 0.0], &[1.0, 1.0]]);
        // Path going up then right: (0,0) → (0,1) → (1,1)
        let traj_b = make_trajectory(&[&[0.0, 0.0], &[0.0, 1.0], &[1.0, 1.0]]);

        let config = SignatureConfig {
            depth: 2,
            time_augmentation: false,
        };
        let sig_a = compute_signature(&traj_a, &config).unwrap();
        let sig_b = compute_signature(&traj_b, &config).unwrap();

        // Same displacement (depth 1)
        assert!((sig_a.signature[0] - sig_b.signature[0]).abs() < 1e-10);
        assert!((sig_a.signature[1] - sig_b.signature[1]).abs() < 1e-10);

        // Different signed area (depth 2) — this is the key:
        // path A and B have opposite signed areas
        let area_a = sig_a.signature[2 + 0 * 2 + 1]; // S^{0,1} for path A
        let area_b = sig_b.signature[2 + 0 * 2 + 1]; // S^{0,1} for path B
        assert!(
            (area_a - area_b).abs() > 0.1,
            "Depth 2 should distinguish rotation: area_a={area_a}, area_b={area_b}"
        );
    }

    #[test]
    fn incremental_matches_full() {
        let points: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.5, -0.3],
            [1.5, 1.0, 0.2],
            [2.0, 0.8, 0.5],
        ];

        let config = SignatureConfig {
            depth: 2,
            time_augmentation: false,
        };

        // Full signature on all 4 points
        let traj_full: Vec<(i64, &[f32])> = points
            .iter()
            .enumerate()
            .map(|(i, p)| (i as i64, p.as_slice()))
            .collect();
        let full = compute_signature(&traj_full, &config).unwrap();

        // Incremental: compute on first 3, then update with 4th
        let traj_3: Vec<(i64, &[f32])> = points[..3]
            .iter()
            .enumerate()
            .map(|(i, p)| (i as i64, p.as_slice()))
            .collect();
        let mut incremental = compute_signature(&traj_3, &config).unwrap();
        update_signature_incremental(&mut incremental, &points[2], &points[3]).unwrap();

        // Should match within numerical precision
        for (i, (a, b)) in full
            .signature
            .iter()
            .zip(incremental.signature.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-8,
                "Mismatch at index {i}: full={a}, incremental={b}"
            );
        }
    }

    #[test]
    fn log_signature_is_smaller() {
        let traj = make_trajectory(&[&[0.0, 0.0, 0.0], &[1.0, 0.5, -0.3], &[1.5, 1.0, 0.2]]);
        let config = SignatureConfig {
            depth: 2,
            time_augmentation: false,
        };

        let full = compute_signature(&traj, &config).unwrap();
        let log = compute_log_signature(&traj, &config).unwrap();

        // K=3: full depth 2 = 3 + 9 = 12, log depth 2 = 3 + 3 = 6
        assert_eq!(full.output_dim, 12);
        assert_eq!(log.output_dim, 6);
    }

    #[test]
    fn signature_distance_works() {
        let traj_a = make_trajectory(&[&[0.0, 0.0], &[1.0, 0.0], &[1.0, 1.0]]);
        let traj_b = make_trajectory(&[&[0.0, 0.0], &[0.0, 1.0], &[1.0, 1.0]]);
        let traj_c = make_trajectory(&[&[0.0, 0.0], &[1.0, 0.0], &[1.0, 1.0]]); // same as A

        let config = SignatureConfig {
            depth: 2,
            time_augmentation: false,
        };
        let sig_a = compute_signature(&traj_a, &config).unwrap();
        let sig_b = compute_signature(&traj_b, &config).unwrap();
        let sig_c = compute_signature(&traj_c, &config).unwrap();

        assert!(signature_distance(&sig_a, &sig_c) < 1e-10); // identical
        assert!(signature_distance(&sig_a, &sig_b) > 0.1); // different paths
    }

    #[test]
    fn insufficient_data_error() {
        let traj = make_trajectory(&[&[1.0, 2.0]]);
        let config = SignatureConfig::default();
        assert!(compute_signature(&traj, &config).is_err());
    }
}
