//! Cohort-level temporal drift analytics.
//!
//! Measures how a **group** of entities evolves collectively in embedding space,
//! complementing the single-entity analytics in [`crate::calculus`].
//!
//! # Key metrics
//!
//! | Metric | What it captures |
//! |--------|-----------------|
//! | `centroid_drift` | How the group center moved |
//! | `dispersion_change` | Did the group spread or compress? |
//! | `convergence_score` | Are entities moving in the same direction? |
//! | `outliers` | Entities drifting abnormally vs the group |

use crate::calculus::{drift_magnitude_l2, drift_report, DriftReport};
use cvx_core::error::AnalyticsError;

// ─── Types ──────────────────────────────────────────────────────────

/// Full cohort drift analysis between two time points.
#[derive(Debug, Clone)]
pub struct CohortDriftReport {
    /// Number of entities successfully analyzed (with data at both t1 and t2).
    pub n_entities: usize,
    /// Mean individual L2 drift magnitude across the cohort.
    pub mean_drift_l2: f32,
    /// Median individual L2 drift magnitude.
    pub median_drift_l2: f32,
    /// Standard deviation of individual L2 drift magnitudes.
    pub std_drift_l2: f32,
    /// Drift of the cohort centroid between t1 and t2.
    pub centroid_drift: DriftReport,
    /// Mean distance from entities to centroid at t1.
    pub dispersion_t1: f32,
    /// Mean distance from entities to centroid at t2.
    pub dispersion_t2: f32,
    /// Change in dispersion: positive = diverging, negative = converging.
    pub dispersion_change: f32,
    /// Cosine alignment of individual drift vectors (0 = random, 1 = all same direction).
    pub convergence_score: f32,
    /// Top-N most changed dimensions aggregated across the cohort.
    pub top_dimensions: Vec<(usize, f32)>,
    /// Entities flagged as outliers (|z-score| > 2.0).
    pub outliers: Vec<CohortOutlier>,
}

/// An entity whose drift deviates significantly from the cohort.
#[derive(Debug, Clone)]
pub struct CohortOutlier {
    /// Entity identifier.
    pub entity_id: u64,
    /// Individual L2 drift magnitude.
    pub drift_magnitude: f32,
    /// Z-score relative to cohort distribution.
    pub z_score: f32,
    /// Cosine similarity between this entity's drift direction and the cohort mean direction.
    pub drift_direction_alignment: f32,
}

// ─── Helpers ────────────────────────────────────────────────────────

/// Find the vector closest in time to `target` within a trajectory.
///
/// Returns `None` if the trajectory is empty.
pub fn nearest_vector_at<'a>(
    trajectory: &'a [(i64, &'a [f32])],
    target: i64,
) -> Option<&'a [f32]> {
    if trajectory.is_empty() {
        return None;
    }
    let idx = trajectory
        .iter()
        .enumerate()
        .min_by_key(|(_, (ts, _))| (ts - target).unsigned_abs())
        .map(|(i, _)| i)?;
    Some(trajectory[idx].1)
}

/// Compute the centroid (element-wise mean) of a set of vectors.
fn centroid(vectors: &[&[f32]]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }
    let dim = vectors[0].len();
    let n = vectors.len() as f32;
    let mut result = vec![0.0f32; dim];
    for v in vectors {
        for (i, &val) in v.iter().enumerate() {
            result[i] += val;
        }
    }
    for val in &mut result {
        *val /= n;
    }
    result
}

/// Mean cosine similarity of all drift vectors against their mean direction.
///
/// Returns 0.0 if fewer than 2 drift vectors, or if the mean drift is zero.
fn compute_convergence_score(drift_vectors: &[Vec<f32>]) -> f32 {
    if drift_vectors.len() < 2 {
        return 0.0;
    }
    let dim = drift_vectors[0].len();
    let n = drift_vectors.len() as f32;

    // Mean drift direction
    let mut mean_dir = vec![0.0f32; dim];
    for dv in drift_vectors {
        for (i, &val) in dv.iter().enumerate() {
            mean_dir[i] += val;
        }
    }
    for val in &mut mean_dir {
        *val /= n;
    }

    let mean_norm: f32 = mean_dir.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mean_norm < 1e-12 {
        return 0.0;
    }

    // Average cosine similarity of each drift vector against mean direction
    let mut total_sim = 0.0f32;
    let mut valid = 0usize;
    for dv in drift_vectors {
        let dv_norm: f32 = dv.iter().map(|x| x * x).sum::<f32>().sqrt();
        if dv_norm < 1e-12 {
            continue;
        }
        let dot: f32 = dv.iter().zip(mean_dir.iter()).map(|(a, b)| a * b).sum();
        total_sim += (dot / (dv_norm * mean_norm)).clamp(-1.0, 1.0);
        valid += 1;
    }

    if valid == 0 {
        0.0
    } else {
        total_sim / valid as f32
    }
}

// ─── Core function ──────────────────────────────────────────────────

/// Compute cohort-level drift analysis.
///
/// Each trajectory in `trajectories` is `(entity_id, sorted_trajectory)` where
/// the trajectory uses the standard CVX format `&[(i64, &[f32])]`.
///
/// The function finds the nearest vector to `t1` and `t2` for each entity,
/// computes individual drift vectors, then aggregates cohort statistics.
///
/// # Errors
///
/// Returns [`AnalyticsError::InsufficientData`] if fewer than 2 entities have
/// data at both t1 and t2.
pub fn cohort_drift(
    trajectories: &[(u64, &[(i64, &[f32])])],
    t1: i64,
    t2: i64,
    top_n: usize,
) -> Result<CohortDriftReport, AnalyticsError> {
    // Collect per-entity data: (entity_id, vector_at_t1, vector_at_t2, drift_vector)
    let mut entity_data: Vec<(u64, Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::new();

    for &(entity_id, traj) in trajectories {
        let Some(v1) = nearest_vector_at(traj, t1) else {
            continue;
        };
        let Some(v2) = nearest_vector_at(traj, t2) else {
            continue;
        };
        if v1.len() != v2.len() {
            continue;
        }
        let drift_vec: Vec<f32> = v2.iter().zip(v1.iter()).map(|(a, b)| a - b).collect();
        entity_data.push((entity_id, v1.to_vec(), v2.to_vec(), drift_vec));
    }

    let n = entity_data.len();
    if n < 2 {
        return Err(AnalyticsError::InsufficientData {
            needed: 2,
            have: n,
        });
    }

    // ── Individual drift magnitudes ──

    let drift_magnitudes: Vec<f32> = entity_data
        .iter()
        .map(|(_, v1, v2, _)| drift_magnitude_l2(v1, v2))
        .collect();

    let mean_drift_l2 = drift_magnitudes.iter().sum::<f32>() / n as f32;

    let mut sorted_mags = drift_magnitudes.clone();
    sorted_mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_drift_l2 = if n % 2 == 0 {
        (sorted_mags[n / 2 - 1] + sorted_mags[n / 2]) / 2.0
    } else {
        sorted_mags[n / 2]
    };

    let variance: f32 = drift_magnitudes
        .iter()
        .map(|m| (m - mean_drift_l2) * (m - mean_drift_l2))
        .sum::<f32>()
        / (n - 1) as f32;
    let std_drift_l2 = variance.sqrt();

    // ── Centroid drift ──

    let vectors_t1: Vec<&[f32]> = entity_data.iter().map(|(_, v1, _, _)| v1.as_slice()).collect();
    let vectors_t2: Vec<&[f32]> = entity_data.iter().map(|(_, _, v2, _)| v2.as_slice()).collect();

    let centroid_t1 = centroid(&vectors_t1);
    let centroid_t2 = centroid(&vectors_t2);
    let centroid_drift = drift_report(&centroid_t1, &centroid_t2, top_n);

    // ── Dispersion ──

    let dispersion_t1 = vectors_t1
        .iter()
        .map(|v| drift_magnitude_l2(v, &centroid_t1))
        .sum::<f32>()
        / n as f32;

    let dispersion_t2 = vectors_t2
        .iter()
        .map(|v| drift_magnitude_l2(v, &centroid_t2))
        .sum::<f32>()
        / n as f32;

    let dispersion_change = dispersion_t2 - dispersion_t1;

    // ── Convergence score ──

    let drift_vectors: Vec<Vec<f32>> = entity_data.iter().map(|(_, _, _, dv)| dv.clone()).collect();
    let convergence_score = compute_convergence_score(&drift_vectors);

    // ── Top dimensions (aggregated) ──

    let dim = entity_data[0].3.len();
    let mut mean_delta = vec![0.0f32; dim];
    for (_, _, _, dv) in &entity_data {
        for (i, &val) in dv.iter().enumerate() {
            mean_delta[i] += val;
        }
    }
    for val in &mut mean_delta {
        *val /= n as f32;
    }

    let mut dim_changes: Vec<(usize, f32)> = mean_delta
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v.abs()))
        .collect();
    dim_changes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    dim_changes.truncate(top_n);

    // ── Outlier detection ──

    // Mean drift direction for alignment computation
    let mean_drift_dir: Vec<f32> = mean_delta.clone();
    let mean_dir_norm: f32 = mean_drift_dir.iter().map(|x| x * x).sum::<f32>().sqrt();

    let outliers: Vec<CohortOutlier> = entity_data
        .iter()
        .zip(drift_magnitudes.iter())
        .filter_map(|((entity_id, _, _, dv), &mag)| {
            let z = if std_drift_l2 > 1e-12 {
                (mag - mean_drift_l2) / std_drift_l2
            } else {
                0.0
            };

            if z.abs() <= 2.0 {
                return None;
            }

            let alignment = if mean_dir_norm > 1e-12 {
                let dv_norm: f32 = dv.iter().map(|x| x * x).sum::<f32>().sqrt();
                if dv_norm > 1e-12 {
                    let dot: f32 = dv.iter().zip(mean_drift_dir.iter()).map(|(a, b)| a * b).sum();
                    (dot / (dv_norm * mean_dir_norm)).clamp(-1.0, 1.0)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            Some(CohortOutlier {
                entity_id: *entity_id,
                drift_magnitude: mag,
                z_score: z,
                drift_direction_alignment: alignment,
            })
        })
        .collect();

    Ok(CohortDriftReport {
        n_entities: n,
        mean_drift_l2,
        median_drift_l2,
        std_drift_l2,
        centroid_drift,
        dispersion_t1,
        dispersion_t2,
        dispersion_change,
        convergence_score,
        top_dimensions: dim_changes,
        outliers,
    })
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: convert owned trajectory to borrowed format.
    fn as_refs(points: &[(i64, Vec<f32>)]) -> Vec<(i64, &[f32])> {
        points.iter().map(|(t, v)| (*t, v.as_slice())).collect()
    }

    // ─── nearest_vector_at ──────────────────────────────────────

    #[test]
    fn nearest_vector_empty_trajectory() {
        let traj: Vec<(i64, &[f32])> = vec![];
        assert!(nearest_vector_at(&traj, 100).is_none());
    }

    #[test]
    fn nearest_vector_exact_match() {
        let owned = vec![
            (100i64, vec![1.0f32, 0.0]),
            (200, vec![0.0, 1.0]),
            (300, vec![1.0, 1.0]),
        ];
        let traj = as_refs(&owned);
        let v = nearest_vector_at(&traj, 200).unwrap();
        assert_eq!(v, &[0.0, 1.0]);
    }

    #[test]
    fn nearest_vector_between_timestamps() {
        let owned = vec![
            (100i64, vec![1.0f32, 0.0]),
            (200, vec![0.0, 1.0]),
            (300, vec![1.0, 1.0]),
        ];
        let traj = as_refs(&owned);
        // 190 is closer to 200 than to 100
        let v = nearest_vector_at(&traj, 190).unwrap();
        assert_eq!(v, &[0.0, 1.0]);
    }

    #[test]
    fn nearest_vector_before_first() {
        let owned = vec![(100i64, vec![1.0f32, 2.0])];
        let traj = as_refs(&owned);
        let v = nearest_vector_at(&traj, 0).unwrap();
        assert_eq!(v, &[1.0, 2.0]);
    }

    // ─── centroid ───────────────────────────────────────────────

    #[test]
    fn centroid_single_vector() {
        let v = vec![2.0f32, 4.0, 6.0];
        let c = centroid(&[v.as_slice()]);
        assert_eq!(c, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn centroid_two_vectors() {
        let v1 = vec![0.0f32, 0.0];
        let v2 = vec![2.0, 4.0];
        let c = centroid(&[v1.as_slice(), v2.as_slice()]);
        assert!((c[0] - 1.0).abs() < 1e-6);
        assert!((c[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn centroid_empty() {
        let c = centroid(&[]);
        assert!(c.is_empty());
    }

    // ─── convergence_score ──────────────────────────────────────

    #[test]
    fn convergence_all_same_direction() {
        let drifts = vec![
            vec![1.0f32, 0.0, 0.0],
            vec![2.0, 0.0, 0.0],
            vec![0.5, 0.0, 0.0],
        ];
        let score = compute_convergence_score(&drifts);
        assert!((score - 1.0).abs() < 1e-6, "expected ~1.0, got {score}");
    }

    #[test]
    fn convergence_opposite_directions() {
        let drifts = vec![vec![1.0f32, 0.0], vec![-1.0, 0.0]];
        let score = compute_convergence_score(&drifts);
        // Mean direction is [0, 0], so score should be 0
        assert!(
            score.abs() < 1e-6,
            "expected ~0.0 for zero mean, got {score}"
        );
    }

    #[test]
    fn convergence_orthogonal_directions() {
        // 4 vectors pointing in 4 orthogonal-ish directions
        let drifts = vec![
            vec![1.0f32, 0.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
            vec![0.0, -1.0],
        ];
        let score = compute_convergence_score(&drifts);
        // Mean direction is ~[0, 0], score should be ~0
        assert!(score.abs() < 1e-6, "expected ~0.0, got {score}");
    }

    #[test]
    fn convergence_too_few_vectors() {
        let drifts = vec![vec![1.0f32]];
        assert_eq!(compute_convergence_score(&drifts), 0.0);
    }

    // ─── cohort_drift — basic functionality ─────────────────────

    #[test]
    fn cohort_drift_insufficient_data() {
        let traj1 = vec![(100i64, vec![1.0f32, 0.0])];
        let refs1 = as_refs(&traj1);

        let trajectories: Vec<(u64, &[(i64, &[f32])])> = vec![(1, &refs1)];
        let result = cohort_drift(&trajectories, 100, 200, 5);
        assert!(result.is_err());
        match result.unwrap_err() {
            AnalyticsError::InsufficientData { needed, have } => {
                assert_eq!(needed, 2);
                assert_eq!(have, 1);
            }
            other => panic!("expected InsufficientData, got {other:?}"),
        }
    }

    #[test]
    fn cohort_drift_uniform_shift() {
        // All entities shift by exactly [0.1, 0.0, 0.0]
        let dim = 3;
        let n_entities = 10;
        let shift = 0.1f32;

        let mut owned_trajs: Vec<Vec<(i64, Vec<f32>)>> = Vec::new();
        for i in 0..n_entities {
            let base: Vec<f32> = (0..dim).map(|d| (i * dim + d) as f32 * 0.1).collect();
            let shifted: Vec<f32> = base.iter().enumerate().map(|(d, &v)| {
                if d == 0 { v + shift } else { v }
            }).collect();
            owned_trajs.push(vec![(1000, base), (2000, shifted)]);
        }

        let ref_trajs: Vec<Vec<(i64, &[f32])>> = owned_trajs
            .iter()
            .map(|t| as_refs(t))
            .collect();
        let trajectories: Vec<(u64, &[(i64, &[f32])])> = ref_trajs
            .iter()
            .enumerate()
            .map(|(i, t)| (i as u64, t.as_slice()))
            .collect();

        let report = cohort_drift(&trajectories, 1000, 2000, 5).unwrap();

        assert_eq!(report.n_entities, n_entities);

        // All drifts should be equal to shift magnitude
        assert!(
            (report.mean_drift_l2 - shift).abs() < 1e-5,
            "expected mean drift ~{shift}, got {}",
            report.mean_drift_l2
        );
        assert!(
            (report.median_drift_l2 - shift).abs() < 1e-5,
            "expected median ~{shift}, got {}",
            report.median_drift_l2
        );
        assert!(
            report.std_drift_l2 < 1e-5,
            "expected std ~0 for uniform shift, got {}",
            report.std_drift_l2
        );

        // All moving in same direction → convergence ~1.0
        assert!(
            report.convergence_score > 0.99,
            "expected convergence ~1.0, got {}",
            report.convergence_score
        );

        // Top dimension should be dim 0
        assert_eq!(report.top_dimensions[0].0, 0);

        // No outliers (all identical drift)
        assert!(
            report.outliers.is_empty(),
            "expected no outliers, got {}",
            report.outliers.len()
        );
    }

    #[test]
    fn cohort_drift_convergence_detected() {
        // Entities start far apart, end close together → dispersion decreases
        let owned_trajs = vec![
            vec![(1000i64, vec![0.0f32, 0.0]), (2000, vec![0.5, 0.5])],
            vec![(1000, vec![2.0, 0.0]), (2000, vec![0.5, 0.5])],
            vec![(1000, vec![0.0, 2.0]), (2000, vec![0.5, 0.5])],
        ];

        let ref_trajs: Vec<Vec<(i64, &[f32])>> = owned_trajs
            .iter()
            .map(|t| as_refs(t))
            .collect();
        let trajectories: Vec<(u64, &[(i64, &[f32])])> = ref_trajs
            .iter()
            .enumerate()
            .map(|(i, t)| (i as u64, t.as_slice()))
            .collect();

        let report = cohort_drift(&trajectories, 1000, 2000, 2).unwrap();

        assert!(
            report.dispersion_change < 0.0,
            "expected negative dispersion change (convergence), got {}",
            report.dispersion_change
        );
        assert!(
            report.dispersion_t2 < report.dispersion_t1,
            "t2 dispersion ({}) should be less than t1 ({})",
            report.dispersion_t2,
            report.dispersion_t1
        );
    }

    #[test]
    fn cohort_drift_divergence_detected() {
        // Entities start close together, end far apart → dispersion increases
        let owned_trajs = vec![
            vec![(1000i64, vec![0.5f32, 0.5]), (2000, vec![0.0, 0.0])],
            vec![(1000, vec![0.5, 0.5]), (2000, vec![2.0, 0.0])],
            vec![(1000, vec![0.5, 0.5]), (2000, vec![0.0, 2.0])],
        ];

        let ref_trajs: Vec<Vec<(i64, &[f32])>> = owned_trajs
            .iter()
            .map(|t| as_refs(t))
            .collect();
        let trajectories: Vec<(u64, &[(i64, &[f32])])> = ref_trajs
            .iter()
            .enumerate()
            .map(|(i, t)| (i as u64, t.as_slice()))
            .collect();

        let report = cohort_drift(&trajectories, 1000, 2000, 2).unwrap();

        assert!(
            report.dispersion_change > 0.0,
            "expected positive dispersion change (divergence), got {}",
            report.dispersion_change
        );
    }

    #[test]
    fn cohort_drift_outlier_detection() {
        // 9 entities with tiny drift + 1 entity with massive drift
        let dim = 4;
        let mut owned_trajs: Vec<Vec<(i64, Vec<f32>)>> = Vec::new();

        // 9 normal entities: drift of 0.01 in dim 0
        for i in 0..9u64 {
            let base: Vec<f32> = vec![i as f32 * 0.1; dim];
            let shifted: Vec<f32> = base.iter().enumerate().map(|(d, &v)| {
                if d == 0 { v + 0.01 } else { v }
            }).collect();
            owned_trajs.push(vec![(1000, base), (2000, shifted)]);
        }

        // 1 outlier: drift of 10.0 in dim 0
        let base = vec![0.5f32; dim];
        let shifted: Vec<f32> = base.iter().enumerate().map(|(d, &v)| {
            if d == 0 { v + 10.0 } else { v }
        }).collect();
        owned_trajs.push(vec![(1000, base), (2000, shifted)]);

        let ref_trajs: Vec<Vec<(i64, &[f32])>> = owned_trajs
            .iter()
            .map(|t| as_refs(t))
            .collect();
        let trajectories: Vec<(u64, &[(i64, &[f32])])> = ref_trajs
            .iter()
            .enumerate()
            .map(|(i, t)| (i as u64, t.as_slice()))
            .collect();

        let report = cohort_drift(&trajectories, 1000, 2000, 3).unwrap();

        assert_eq!(report.n_entities, 10);
        assert!(
            !report.outliers.is_empty(),
            "expected at least one outlier"
        );

        // The outlier should be entity 9
        let outlier = report.outliers.iter().find(|o| o.entity_id == 9);
        assert!(outlier.is_some(), "entity 9 should be flagged as outlier");
        let outlier = outlier.unwrap();
        assert!(
            outlier.z_score > 2.0,
            "outlier z-score should be > 2.0, got {}",
            outlier.z_score
        );
        assert!(
            outlier.drift_magnitude > 9.0,
            "outlier drift should be large, got {}",
            outlier.drift_magnitude
        );
    }

    #[test]
    fn cohort_drift_centroid_drift_matches_manual() {
        // 2 entities, manually compute expected centroid drift
        let owned_trajs = vec![
            vec![(1000i64, vec![0.0f32, 0.0]), (2000, vec![1.0, 0.0])],
            vec![(1000, vec![2.0, 0.0]), (2000, vec![3.0, 0.0])],
        ];

        let ref_trajs: Vec<Vec<(i64, &[f32])>> = owned_trajs
            .iter()
            .map(|t| as_refs(t))
            .collect();
        let trajectories: Vec<(u64, &[(i64, &[f32])])> = ref_trajs
            .iter()
            .enumerate()
            .map(|(i, t)| (i as u64, t.as_slice()))
            .collect();

        let report = cohort_drift(&trajectories, 1000, 2000, 2).unwrap();

        // Centroid at t1 = [1.0, 0.0], centroid at t2 = [2.0, 0.0]
        // Centroid drift L2 = 1.0
        assert!(
            (report.centroid_drift.l2_magnitude - 1.0).abs() < 1e-5,
            "expected centroid drift 1.0, got {}",
            report.centroid_drift.l2_magnitude
        );
    }

    #[test]
    fn cohort_drift_no_data_at_one_timepoint() {
        // Entity 1 has data only at t1, entity 2 has data at both, entity 3 only at t2
        // Entity 1's nearest to t2=2000 will be its only point at t1=1000
        // All entities will actually be included since nearest_vector_at finds closest
        let owned_trajs = vec![
            vec![(1000i64, vec![1.0f32, 0.0])],
            vec![(1000, vec![2.0, 0.0]), (2000, vec![3.0, 0.0])],
            vec![(2000i64, vec![4.0, 0.0])],
        ];

        let ref_trajs: Vec<Vec<(i64, &[f32])>> = owned_trajs
            .iter()
            .map(|t| as_refs(t))
            .collect();
        let trajectories: Vec<(u64, &[(i64, &[f32])])> = ref_trajs
            .iter()
            .enumerate()
            .map(|(i, t)| (i as u64, t.as_slice()))
            .collect();

        // Should succeed with all 3 entities (nearest_vector_at finds closest available)
        let result = cohort_drift(&trajectories, 1000, 2000, 2);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().n_entities, 3);
    }

    #[test]
    fn cohort_drift_stationary_cohort() {
        // All entities stay in the same place
        let owned_trajs = vec![
            vec![(1000i64, vec![1.0f32, 2.0, 3.0]), (2000, vec![1.0, 2.0, 3.0])],
            vec![(1000, vec![4.0, 5.0, 6.0]), (2000, vec![4.0, 5.0, 6.0])],
            vec![(1000, vec![7.0, 8.0, 9.0]), (2000, vec![7.0, 8.0, 9.0])],
        ];

        let ref_trajs: Vec<Vec<(i64, &[f32])>> = owned_trajs
            .iter()
            .map(|t| as_refs(t))
            .collect();
        let trajectories: Vec<(u64, &[(i64, &[f32])])> = ref_trajs
            .iter()
            .enumerate()
            .map(|(i, t)| (i as u64, t.as_slice()))
            .collect();

        let report = cohort_drift(&trajectories, 1000, 2000, 3).unwrap();

        assert!(report.mean_drift_l2 < 1e-6, "stationary cohort should have ~0 drift");
        assert!(report.median_drift_l2 < 1e-6);
        assert!(report.centroid_drift.l2_magnitude < 1e-6);
        assert!((report.dispersion_change).abs() < 1e-6, "dispersion should not change");
        assert!(report.outliers.is_empty());
    }

    #[test]
    fn cohort_drift_high_dimensional() {
        // Sanity check with 128-dim vectors
        let dim = 128;
        let n_entities = 20;
        let mut owned_trajs = Vec::new();

        for i in 0..n_entities {
            let base: Vec<f32> = (0..dim).map(|d| ((i * dim + d) as f32 * 0.01).sin()).collect();
            let shifted: Vec<f32> = base.iter().map(|v| v + 0.05).collect();
            owned_trajs.push(vec![(1000i64, base), (2000, shifted)]);
        }

        let ref_trajs: Vec<Vec<(i64, &[f32])>> = owned_trajs
            .iter()
            .map(|t| as_refs(t))
            .collect();
        let trajectories: Vec<(u64, &[(i64, &[f32])])> = ref_trajs
            .iter()
            .enumerate()
            .map(|(i, t)| (i as u64, t.as_slice()))
            .collect();

        let report = cohort_drift(&trajectories, 1000, 2000, 10).unwrap();

        assert_eq!(report.n_entities, n_entities);
        assert!(report.mean_drift_l2 > 0.0);
        assert_eq!(report.top_dimensions.len(), 10);
        // Uniform shift → high convergence
        assert!(
            report.convergence_score > 0.95,
            "uniform shift should give high convergence, got {}",
            report.convergence_score
        );
    }
}
