#![allow(clippy::needless_range_loop)]
//! Procrustes alignment for cross-model embedding comparison (RFC-012 P10).
//!
//! When switching embedding models (e.g., MentalRoBERTa → all-MiniLM),
//! vectors from the old model are incompatible with the new model's space.
//! Procrustes alignment finds the optimal orthogonal rotation R that
//! minimizes ||A - BR||² where A is the target space and B is the source.
//!
//! # Algorithm
//!
//! Given N corresponding vector pairs (a_i, b_i):
//! 1. Center both sets: A' = A - mean(A), B' = B - mean(B)
//! 2. Compute cross-covariance: M = A'^T B'
//! 3. SVD: M = U Σ V^T
//! 4. Rotation: R = V U^T
//! 5. Scale: s = trace(Σ) / trace(B'^T B')
//! 6. Transform: b_aligned = s × (b - mean(B)) × R + mean(A)
//!
//! # Example
//!
//! ```
//! use cvx_analytics::procrustes::{ProcrustesTransform, fit_procrustes};
//!
//! let source = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
//! let target = vec![vec![0.0, 1.0], vec![-1.0, 0.0], vec![-1.0, 1.0]];
//!
//! let transform = fit_procrustes(&source, &target).unwrap();
//! let aligned = transform.apply(&[1.0, 0.0]);
//! // aligned should be close to [0.0, 1.0] (90° rotation)
//! ```

/// A fitted Procrustes transformation.
#[derive(Debug, Clone)]
pub struct ProcrustesTransform {
    /// Rotation matrix (D × D), stored row-major.
    pub rotation: Vec<Vec<f64>>,
    /// Scale factor.
    pub scale: f64,
    /// Source centroid (subtracted before rotation).
    pub source_mean: Vec<f64>,
    /// Target centroid (added after rotation).
    pub target_mean: Vec<f64>,
    /// Dimensionality.
    pub dim: usize,
    /// Alignment error (Frobenius norm of residual).
    pub error: f64,
}

impl ProcrustesTransform {
    /// Apply the transform to a source vector → target space.
    pub fn apply(&self, source_vec: &[f32]) -> Vec<f32> {
        let d = self.dim;
        assert_eq!(source_vec.len(), d);

        // Center
        let centered: Vec<f64> = source_vec
            .iter()
            .zip(&self.source_mean)
            .map(|(&v, &m)| v as f64 - m)
            .collect();

        // Rotate + scale
        let mut rotated = vec![0.0f64; d];
        for i in 0..d {
            for j in 0..d {
                rotated[i] += centered[j] * self.rotation[j][i];
            }
            rotated[i] *= self.scale;
        }

        // Translate to target space
        rotated
            .iter()
            .zip(&self.target_mean)
            .map(|(&r, &m)| (r + m) as f32)
            .collect()
    }

    /// Apply to a batch of vectors.
    pub fn apply_batch(&self, vectors: &[Vec<f32>]) -> Vec<Vec<f32>> {
        vectors.iter().map(|v| self.apply(v)).collect()
    }
}

/// Fit a Procrustes transformation from source to target vectors.
///
/// Both sets must have the same number of vectors and same dimensionality.
/// Vectors should be corresponding pairs (same entity in different models).
///
/// Returns `None` if inputs are empty or have mismatched dimensions.
pub fn fit_procrustes(source: &[Vec<f32>], target: &[Vec<f32>]) -> Option<ProcrustesTransform> {
    let n = source.len();
    if n == 0 || n != target.len() {
        return None;
    }
    let d = source[0].len();
    if d == 0 || target[0].len() != d {
        return None;
    }

    // 1. Compute centroids
    let mut src_mean = vec![0.0f64; d];
    let mut tgt_mean = vec![0.0f64; d];
    for i in 0..n {
        for j in 0..d {
            src_mean[j] += source[i][j] as f64;
            tgt_mean[j] += target[i][j] as f64;
        }
    }
    let inv_n = 1.0 / n as f64;
    for j in 0..d {
        src_mean[j] *= inv_n;
        tgt_mean[j] *= inv_n;
    }

    // 2. Center
    let mut a = vec![vec![0.0f64; d]; n]; // target centered
    let mut b = vec![vec![0.0f64; d]; n]; // source centered
    for i in 0..n {
        for j in 0..d {
            a[i][j] = target[i][j] as f64 - tgt_mean[j];
            b[i][j] = source[i][j] as f64 - src_mean[j];
        }
    }

    // 3. Cross-covariance M = A^T B (D × D)
    let mut m = vec![vec![0.0f64; d]; d];
    for i in 0..n {
        for j in 0..d {
            for k in 0..d {
                m[j][k] += a[i][j] * b[i][k];
            }
        }
    }

    // 4. SVD via Jacobi iterations (simple, works for moderate D)
    let (u, sigma, vt) = svd_jacobi(&m, d);

    // 5. Rotation R = V U^T (V is rows of vt transposed)
    let mut rotation = vec![vec![0.0f64; d]; d];
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                rotation[i][j] += vt[k][i] * u[j][k]; // V^T^T * U^T = V * U^T
            }
        }
    }

    // 6. Scale
    let trace_sigma: f64 = sigma.iter().sum();
    let mut trace_btb = 0.0f64;
    for i in 0..n {
        for j in 0..d {
            trace_btb += b[i][j] * b[i][j];
        }
    }
    let scale = if trace_btb > 1e-12 {
        trace_sigma / trace_btb
    } else {
        1.0
    };

    // 7. Compute error
    let mut error = 0.0f64;
    for i in 0..n {
        let aligned = apply_rotation(&b[i], &rotation, scale);
        for j in 0..d {
            let diff = a[i][j] - aligned[j];
            error += diff * diff;
        }
    }
    error = error.sqrt() / n as f64;

    Some(ProcrustesTransform {
        rotation,
        scale,
        source_mean: src_mean,
        target_mean: tgt_mean,
        dim: d,
        error,
    })
}

fn apply_rotation(vec: &[f64], rotation: &[Vec<f64>], scale: f64) -> Vec<f64> {
    let d = vec.len();
    let mut result = vec![0.0f64; d];
    for i in 0..d {
        for j in 0..d {
            result[i] += vec[j] * rotation[j][i];
        }
        result[i] *= scale;
    }
    result
}

/// Simple Jacobi SVD for small-to-moderate matrices.
/// Returns (U, singular_values, V^T).
fn svd_jacobi(m: &[Vec<f64>], d: usize) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    // Compute M^T M
    let mut mtm = vec![vec![0.0f64; d]; d];
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                mtm[i][j] += m[k][i] * m[k][j];
            }
        }
    }

    // Eigendecomposition of M^T M via Jacobi rotations
    let (eigenvalues, v) = jacobi_eigendecomposition(&mtm, d);

    // Singular values = sqrt(eigenvalues)
    let sigma: Vec<f64> = eigenvalues.iter().map(|&e| e.max(0.0).sqrt()).collect();

    // U = M V Σ^{-1}
    let mut u = vec![vec![0.0f64; d]; d];
    for i in 0..d {
        for j in 0..d {
            let mut sum = 0.0f64;
            for k in 0..d {
                sum += m[i][k] * v[k][j];
            }
            u[i][j] = if sigma[j] > 1e-12 {
                sum / sigma[j]
            } else {
                0.0
            };
        }
    }

    // V^T
    let mut vt = vec![vec![0.0f64; d]; d];
    for i in 0..d {
        for j in 0..d {
            vt[i][j] = v[j][i];
        }
    }

    (u, sigma, vt)
}

/// Jacobi eigendecomposition for symmetric matrices.
fn jacobi_eigendecomposition(a: &[Vec<f64>], d: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut mat = a.to_vec();
    let mut v = vec![vec![0.0f64; d]; d];
    for i in 0..d {
        v[i][i] = 1.0;
    }

    let max_iter = 100 * d * d;
    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..d {
            for j in (i + 1)..d {
                if mat[i][j].abs() > max_val {
                    max_val = mat[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-12 {
            break;
        }

        // Compute rotation angle
        let theta = if (mat[p][p] - mat[q][q]).abs() < 1e-12 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * mat[p][q] / (mat[p][p] - mat[q][q])).atan()
        };

        let (sin_t, cos_t) = theta.sin_cos();

        // Apply rotation to mat
        let mut new_mat = mat.clone();
        for i in 0..d {
            if i != p && i != q {
                new_mat[i][p] = cos_t * mat[i][p] + sin_t * mat[i][q];
                new_mat[p][i] = new_mat[i][p];
                new_mat[i][q] = -sin_t * mat[i][p] + cos_t * mat[i][q];
                new_mat[q][i] = new_mat[i][q];
            }
        }
        new_mat[p][p] =
            cos_t * cos_t * mat[p][p] + 2.0 * sin_t * cos_t * mat[p][q] + sin_t * sin_t * mat[q][q];
        new_mat[q][q] =
            sin_t * sin_t * mat[p][p] - 2.0 * sin_t * cos_t * mat[p][q] + cos_t * cos_t * mat[q][q];
        new_mat[p][q] = 0.0;
        new_mat[q][p] = 0.0;
        mat = new_mat;

        // Update eigenvectors
        for i in 0..d {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = cos_t * vip + sin_t * viq;
            v[i][q] = -sin_t * vip + cos_t * viq;
        }
    }

    let eigenvalues: Vec<f64> = (0..d).map(|i| mat[i][i]).collect();
    (eigenvalues, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_alignment() {
        // Same vectors → identity transform, zero error
        let vecs: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![-1.0, 0.5],
        ];
        let t = fit_procrustes(&vecs, &vecs).unwrap();
        assert!(t.error < 0.01, "error = {}", t.error);
        assert!((t.scale - 1.0).abs() < 0.01, "scale = {}", t.scale);

        let aligned = t.apply(&vecs[0]);
        assert!((aligned[0] - 1.0).abs() < 0.1);
        assert!((aligned[1] - 0.0).abs() < 0.1);
    }

    #[test]
    fn rotation_90_degrees() {
        // Source rotated 90° CCW from target
        let source = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
            vec![0.0, -1.0],
        ];
        let target = vec![
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
            vec![0.0, -1.0],
            vec![1.0, 0.0],
        ];

        let t = fit_procrustes(&source, &target).unwrap();
        assert!(t.error < 0.1, "error = {}", t.error);

        let aligned = t.apply(&[1.0, 0.0]);
        assert!(
            (aligned[0] - 0.0).abs() < 0.2 && (aligned[1] - 1.0).abs() < 0.2,
            "expected ~[0, 1], got {aligned:?}",
        );
    }

    #[test]
    fn higher_dimension() {
        let d = 8;
        let n = 20;
        let mut rng = 42u64;
        let next = |r: &mut u64| -> f32 {
            *r = r.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*r >> 33) as f32) / (u32::MAX as f32) - 0.5
        };

        let source: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..d).map(|_| next(&mut rng)).collect())
            .collect();
        // Target = source with small perturbation (should align well)
        let target: Vec<Vec<f32>> = source
            .iter()
            .map(|v| v.iter().map(|&x| x + next(&mut rng) * 0.01).collect())
            .collect();

        let t = fit_procrustes(&source, &target).unwrap();
        assert!(t.error < 0.1, "error = {} (d={d}, n={n})", t.error);
    }

    #[test]
    fn empty_input() {
        let empty: Vec<Vec<f32>> = vec![];
        assert!(fit_procrustes(&empty, &empty).is_none());
    }

    #[test]
    fn mismatched_sizes() {
        let a = vec![vec![1.0, 0.0]];
        let b = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert!(fit_procrustes(&a, &b).is_none());
    }
}
