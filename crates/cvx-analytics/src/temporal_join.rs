//! Temporal join — find time windows where entities are semantically close.
//!
//! A temporal join finds periods where two (or more) entities converge in
//! embedding space. Unlike point-level kNN, this captures sustained proximity.
//!
//! # Example
//!
//! Two users who sporadically post similar content won't trigger a join,
//! but users who consistently converge over a week-long window will.

use crate::calculus::drift_magnitude_l2;
use cvx_core::error::AnalyticsError;

// ─── Types ──────────────────────────────────────────────────────────

/// A time window during which two entities were semantically close.
#[derive(Debug, Clone)]
pub struct TemporalJoinResult {
    /// Start of the convergence window.
    pub start: i64,
    /// End of the convergence window.
    pub end: i64,
    /// Mean pairwise distance during the window.
    pub mean_distance: f32,
    /// Minimum pairwise distance during the window.
    pub min_distance: f32,
    /// Number of points from entity A in the window.
    pub points_a: usize,
    /// Number of points from entity B in the window.
    pub points_b: usize,
}

/// A time window during which a subset of a group converged.
#[derive(Debug, Clone)]
pub struct GroupJoinResult {
    /// Start of the convergence window.
    pub start: i64,
    /// End of the convergence window.
    pub end: i64,
    /// Number of entities within epsilon during this window.
    pub n_converged: usize,
    /// Entity IDs that converged.
    pub converged_entities: Vec<u64>,
    /// Mean pairwise distance among converged entities.
    pub mean_distance: f32,
}

// ─── Core: pairwise temporal join ───────────────────────────────────

/// Find time windows where entities A and B are within distance `epsilon`.
///
/// Uses a sliding window of size `window_us` microseconds over the merged
/// timeline. Within each window, computes the minimum pairwise distance
/// between A's and B's points.
///
/// # Arguments
///
/// * `traj_a` — Entity A's trajectory, sorted by timestamp
/// * `traj_b` — Entity B's trajectory, sorted by timestamp
/// * `epsilon` — Maximum distance threshold for convergence
/// * `window_us` — Sliding window size in microseconds
///
/// # Errors
///
/// Returns [`AnalyticsError::InsufficientData`] if either trajectory is empty.
pub fn temporal_join(
    traj_a: &[(i64, &[f32])],
    traj_b: &[(i64, &[f32])],
    epsilon: f32,
    window_us: i64,
) -> Result<Vec<TemporalJoinResult>, AnalyticsError> {
    if traj_a.is_empty() || traj_b.is_empty() {
        return Err(AnalyticsError::InsufficientData { needed: 1, have: 0 });
    }

    // Determine the global time range
    let global_start = traj_a[0].0.min(traj_b[0].0);
    let global_end = traj_a.last().unwrap().0.max(traj_b.last().unwrap().0);

    if global_end - global_start < window_us {
        // Single window covers everything
        let (min_dist, mean_dist, n_a, n_b) =
            window_distances(traj_a, traj_b, global_start, global_end);
        return if min_dist <= epsilon && n_a > 0 && n_b > 0 {
            Ok(vec![TemporalJoinResult {
                start: global_start,
                end: global_end,
                mean_distance: mean_dist,
                min_distance: min_dist,
                points_a: n_a,
                points_b: n_b,
            }])
        } else {
            Ok(vec![])
        };
    }

    // Slide window with step = window_us / 2 (50% overlap for smoother detection)
    let step = (window_us / 2).max(1);
    let mut results: Vec<TemporalJoinResult> = Vec::new();
    let mut current_start: Option<i64> = None;
    let mut acc_min_dist = f32::MAX;
    let mut acc_sum_dist = 0.0f32;
    let mut acc_count = 0usize;
    let mut acc_points_a = 0usize;
    let mut acc_points_b = 0usize;

    let mut t = global_start;
    while t <= global_end {
        let w_end = (t + window_us).min(global_end);
        let (min_dist, mean_dist, n_a, n_b) = window_distances(traj_a, traj_b, t, w_end);

        let is_close = min_dist <= epsilon && n_a > 0 && n_b > 0;

        if is_close {
            if current_start.is_none() {
                current_start = Some(t);
                acc_min_dist = f32::MAX;
                acc_sum_dist = 0.0;
                acc_count = 0;
                acc_points_a = 0;
                acc_points_b = 0;
            }
            acc_min_dist = acc_min_dist.min(min_dist);
            acc_sum_dist += mean_dist;
            acc_count += 1;
            acc_points_a = acc_points_a.max(n_a);
            acc_points_b = acc_points_b.max(n_b);
        } else if let Some(start) = current_start.take() {
            results.push(TemporalJoinResult {
                start,
                end: (t - step + window_us).min(global_end),
                mean_distance: acc_sum_dist / acc_count as f32,
                min_distance: acc_min_dist,
                points_a: acc_points_a,
                points_b: acc_points_b,
            });
        }

        t += step;
    }

    // Close any open interval
    if let Some(start) = current_start {
        results.push(TemporalJoinResult {
            start,
            end: global_end,
            mean_distance: if acc_count > 0 {
                acc_sum_dist / acc_count as f32
            } else {
                0.0
            },
            min_distance: acc_min_dist,
            points_a: acc_points_a,
            points_b: acc_points_b,
        });
    }

    Ok(results)
}

// ─── Core: group temporal join ──────────────────────────────────────

/// Find time windows where at least `min_entities` from the group converge.
///
/// For each window, computes all pairwise distances and finds the largest
/// subset within distance `epsilon` of each other.
///
/// # Arguments
///
/// * `trajectories` — `(entity_id, trajectory)` pairs
/// * `epsilon` — Maximum distance threshold
/// * `min_entities` — Minimum number of entities that must converge
/// * `window_us` — Sliding window size in microseconds
///
/// # Errors
///
/// Returns [`AnalyticsError::InsufficientData`] if fewer than `min_entities`
/// trajectories are provided.
#[allow(clippy::type_complexity)]
pub fn group_temporal_join(
    trajectories: &[(u64, &[(i64, &[f32])])],
    epsilon: f32,
    min_entities: usize,
    window_us: i64,
) -> Result<Vec<GroupJoinResult>, AnalyticsError> {
    if trajectories.len() < min_entities {
        return Err(AnalyticsError::InsufficientData {
            needed: min_entities,
            have: trajectories.len(),
        });
    }

    // Global time range
    let global_start = trajectories
        .iter()
        .filter_map(|(_, t)| t.first().map(|(ts, _)| *ts))
        .min()
        .unwrap_or(0);
    let global_end = trajectories
        .iter()
        .filter_map(|(_, t)| t.last().map(|(ts, _)| *ts))
        .max()
        .unwrap_or(0);

    if global_end <= global_start {
        return Ok(vec![]);
    }

    let step = (window_us / 2).max(1);
    let mut results: Vec<GroupJoinResult> = Vec::new();

    let mut t = global_start;
    while t <= global_end {
        let w_end = (t + window_us).min(global_end);

        // For each entity, find their representative vector in this window
        // (use the centroid of all points in window)
        let mut entity_reps: Vec<(u64, Vec<f32>)> = Vec::new();

        for &(eid, traj) in trajectories {
            let points_in_window: Vec<&[f32]> = traj
                .iter()
                .filter(|(ts, _)| *ts >= t && *ts <= w_end)
                .map(|(_, v)| *v)
                .collect();

            if points_in_window.is_empty() {
                continue;
            }

            // Centroid of points in window
            let dim = points_in_window[0].len();
            let n = points_in_window.len() as f32;
            let mut centroid = vec![0.0f32; dim];
            for v in &points_in_window {
                for (i, &val) in v.iter().enumerate() {
                    centroid[i] += val;
                }
            }
            for val in &mut centroid {
                *val /= n;
            }

            entity_reps.push((eid, centroid));
        }

        if entity_reps.len() >= min_entities {
            // Greedy clustering: find entities within epsilon of the first,
            // then try each as seed and keep the largest cluster
            let converged = find_largest_epsilon_cluster(&entity_reps, epsilon);

            if converged.len() >= min_entities {
                // Compute mean pairwise distance among converged
                let mean_dist = mean_pairwise_distance(&entity_reps, &converged);

                results.push(GroupJoinResult {
                    start: t,
                    end: w_end,
                    n_converged: converged.len(),
                    converged_entities: converged,
                    mean_distance: mean_dist,
                });
            }
        }

        t += step;
    }

    // Merge consecutive overlapping windows
    merge_group_results(&mut results, step);

    Ok(results)
}

// ─── Helpers ────────────────────────────────────────────────────────

/// Compute min and mean distances between A's and B's points in a time window.
fn window_distances(
    traj_a: &[(i64, &[f32])],
    traj_b: &[(i64, &[f32])],
    start: i64,
    end: i64,
) -> (f32, f32, usize, usize) {
    let a_in_window: Vec<&[f32]> = traj_a
        .iter()
        .filter(|(ts, _)| *ts >= start && *ts <= end)
        .map(|(_, v)| *v)
        .collect();
    let b_in_window: Vec<&[f32]> = traj_b
        .iter()
        .filter(|(ts, _)| *ts >= start && *ts <= end)
        .map(|(_, v)| *v)
        .collect();

    if a_in_window.is_empty() || b_in_window.is_empty() {
        return (f32::MAX, f32::MAX, a_in_window.len(), b_in_window.len());
    }

    let mut min_dist = f32::MAX;
    let mut sum_dist = 0.0f32;
    let mut count = 0;

    for a in &a_in_window {
        for b in &b_in_window {
            let d = drift_magnitude_l2(a, b);
            min_dist = min_dist.min(d);
            sum_dist += d;
            count += 1;
        }
    }

    let mean_dist = if count > 0 {
        sum_dist / count as f32
    } else {
        f32::MAX
    };

    (min_dist, mean_dist, a_in_window.len(), b_in_window.len())
}

/// Find the largest cluster of entities where all pairs are within epsilon.
///
/// Uses a greedy approach: for each entity as seed, collect all within epsilon,
/// then keep the largest set.
fn find_largest_epsilon_cluster(entity_reps: &[(u64, Vec<f32>)], epsilon: f32) -> Vec<u64> {
    let n = entity_reps.len();
    let mut best: Vec<u64> = Vec::new();

    for i in 0..n {
        let mut cluster: Vec<usize> = vec![i];

        for j in 0..n {
            if i == j {
                continue;
            }
            // Check if j is within epsilon of ALL current cluster members
            let all_close = cluster
                .iter()
                .all(|&c| drift_magnitude_l2(&entity_reps[c].1, &entity_reps[j].1) <= epsilon);
            if all_close {
                cluster.push(j);
            }
        }

        if cluster.len() > best.len() {
            best = cluster.into_iter().map(|idx| entity_reps[idx].0).collect();
        }
    }

    best
}

/// Mean pairwise distance among a subset of entities.
fn mean_pairwise_distance(entity_reps: &[(u64, Vec<f32>)], ids: &[u64]) -> f32 {
    let vecs: Vec<&Vec<f32>> = ids
        .iter()
        .filter_map(|id| {
            entity_reps
                .iter()
                .find(|(eid, _)| eid == id)
                .map(|(_, v)| v)
        })
        .collect();

    if vecs.len() < 2 {
        return 0.0;
    }

    let mut sum = 0.0f32;
    let mut count = 0;
    for i in 0..vecs.len() {
        for j in (i + 1)..vecs.len() {
            sum += drift_magnitude_l2(vecs[i], vecs[j]);
            count += 1;
        }
    }

    if count > 0 { sum / count as f32 } else { 0.0 }
}

/// Merge consecutive group join results that overlap in time.
fn merge_group_results(results: &mut Vec<GroupJoinResult>, step: i64) {
    if results.len() < 2 {
        return;
    }

    let mut merged: Vec<GroupJoinResult> = Vec::new();
    let mut current = results[0].clone();

    for r in &results[1..] {
        // If this window overlaps with the current merged window
        if r.start <= current.end + step {
            // Extend the window
            current.end = r.end;
            current.n_converged = current.n_converged.max(r.n_converged);
            current.mean_distance = (current.mean_distance + r.mean_distance) / 2.0;
            // Union of converged entities
            for eid in &r.converged_entities {
                if !current.converged_entities.contains(eid) {
                    current.converged_entities.push(*eid);
                }
            }
        } else {
            merged.push(current);
            current = r.clone();
        }
    }
    merged.push(current);

    *results = merged;
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::type_complexity,
    clippy::needless_range_loop,
    clippy::useless_vec
)]
mod tests {
    use super::*;

    fn as_refs(points: &[(i64, Vec<f32>)]) -> Vec<(i64, &[f32])> {
        points.iter().map(|(t, v)| (*t, v.as_slice())).collect()
    }

    // ─── temporal_join (pairwise) ───────────────────────────────

    #[test]
    fn join_empty_trajectories() {
        let a: Vec<(i64, &[f32])> = vec![];
        let b_owned = vec![(100i64, vec![1.0f32])];
        let b = as_refs(&b_owned);
        assert!(temporal_join(&a, &b, 1.0, 1000).is_err());
    }

    #[test]
    fn join_no_overlap() {
        // A and B are far apart in embedding space
        let a_owned = vec![
            (1000i64, vec![0.0f32, 0.0]),
            (2000, vec![0.0, 0.0]),
            (3000, vec![0.0, 0.0]),
        ];
        let b_owned = vec![
            (1000i64, vec![100.0f32, 100.0]),
            (2000, vec![100.0, 100.0]),
            (3000, vec![100.0, 100.0]),
        ];
        let a = as_refs(&a_owned);
        let b = as_refs(&b_owned);

        let results = temporal_join(&a, &b, 0.5, 2000).unwrap();
        assert!(
            results.is_empty(),
            "distant entities should produce no join"
        );
    }

    #[test]
    fn join_full_convergence() {
        // A and B are at the same point for the entire duration
        let a_owned = vec![
            (1000i64, vec![1.0f32, 1.0]),
            (2000, vec![1.0, 1.0]),
            (3000, vec![1.0, 1.0]),
        ];
        let b_owned = vec![
            (1000i64, vec![1.0f32, 1.0]),
            (2000, vec![1.0, 1.0]),
            (3000, vec![1.0, 1.0]),
        ];
        let a = as_refs(&a_owned);
        let b = as_refs(&b_owned);

        let results = temporal_join(&a, &b, 0.5, 2000).unwrap();
        assert!(
            !results.is_empty(),
            "identical trajectories should produce at least one join"
        );
        assert!(results[0].min_distance < 1e-6);
    }

    #[test]
    fn join_partial_convergence() {
        // A starts far from B, converges around t=5000, then diverges
        let a_owned = vec![
            (1000i64, vec![0.0f32, 0.0]),
            (2000, vec![0.0, 0.0]),
            (3000, vec![0.2, 0.0]),
            (4000, vec![0.8, 0.0]),
            (5000, vec![1.0, 0.0]), // close to B
            (6000, vec![1.0, 0.0]), // close to B
            (7000, vec![0.5, 0.0]),
            (8000, vec![0.0, 0.0]),
        ];
        let b_owned = vec![
            (1000i64, vec![1.0f32, 0.0]),
            (2000, vec![1.0, 0.0]),
            (3000, vec![1.0, 0.0]),
            (4000, vec![1.0, 0.0]),
            (5000, vec![1.0, 0.0]),
            (6000, vec![1.0, 0.0]),
            (7000, vec![1.0, 0.0]),
            (8000, vec![1.0, 0.0]),
        ];
        let a = as_refs(&a_owned);
        let b = as_refs(&b_owned);

        let results = temporal_join(&a, &b, 0.15, 2000).unwrap();
        assert!(
            !results.is_empty(),
            "should detect convergence around t=5000-6000"
        );

        // The convergence should be in the t=4000-7000 region
        let convergence = &results[0];
        assert!(
            convergence.start >= 3000 && convergence.start <= 5000,
            "convergence should start around t=4000-5000, got {}",
            convergence.start
        );
    }

    #[test]
    fn join_respects_epsilon() {
        let a_owned = vec![(1000i64, vec![0.0f32, 0.0])];
        let b_owned = vec![(1000i64, vec![0.5, 0.0])];
        let a = as_refs(&a_owned);
        let b = as_refs(&b_owned);

        // Distance is 0.5, epsilon=0.4 should not match
        let results = temporal_join(&a, &b, 0.4, 2000).unwrap();
        assert!(results.is_empty());

        // epsilon=0.6 should match
        let results = temporal_join(&a, &b, 0.6, 2000).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn join_single_window_covers_all() {
        // Window larger than total time range
        let a_owned = vec![(100i64, vec![1.0f32]), (200, vec![1.0])];
        let b_owned = vec![(150i64, vec![1.0f32]), (250, vec![1.0])];
        let a = as_refs(&a_owned);
        let b = as_refs(&b_owned);

        let results = temporal_join(&a, &b, 0.5, 1000).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn join_result_has_point_counts() {
        let a_owned = vec![
            (1000i64, vec![0.0f32]),
            (2000, vec![0.0]),
            (3000, vec![0.0]),
        ];
        let b_owned = vec![(1500i64, vec![0.0f32]), (2500, vec![0.0])];
        let a = as_refs(&a_owned);
        let b = as_refs(&b_owned);

        let results = temporal_join(&a, &b, 1.0, 4000).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].points_a > 0);
        assert!(results[0].points_b > 0);
    }

    // ─── group_temporal_join ────────────────────────────────────

    #[test]
    fn group_join_insufficient_entities() {
        let t1_owned = vec![(100i64, vec![1.0f32])];
        let t1 = as_refs(&t1_owned);
        let trajectories: Vec<(u64, &[(i64, &[f32])])> = vec![(1, &t1)];
        let result = group_temporal_join(&trajectories, 0.5, 3, 1000);
        assert!(result.is_err());
    }

    #[test]
    fn group_join_all_converge() {
        // 4 entities all at the same point
        let mut owned: Vec<Vec<(i64, Vec<f32>)>> = Vec::new();
        for _ in 0..4 {
            owned.push(vec![
                (1000i64, vec![1.0f32, 1.0]),
                (2000, vec![1.0, 1.0]),
                (3000, vec![1.0, 1.0]),
            ]);
        }
        let refs: Vec<Vec<(i64, &[f32])>> = owned.iter().map(|t| as_refs(t)).collect();
        let trajectories: Vec<(u64, &[(i64, &[f32])])> = refs
            .iter()
            .enumerate()
            .map(|(i, t)| (i as u64, t.as_slice()))
            .collect();

        let results = group_temporal_join(&trajectories, 0.5, 3, 3000).unwrap();
        assert!(
            !results.is_empty(),
            "all identical entities should converge"
        );
        assert!(results[0].n_converged >= 3);
    }

    #[test]
    fn group_join_partial_subset() {
        // 3 close entities + 1 far entity
        let owned = vec![
            vec![(1000i64, vec![1.0f32, 1.0]), (2000, vec![1.0, 1.0])],
            vec![(1000i64, vec![1.0f32, 1.1]), (2000, vec![1.0, 1.1])],
            vec![(1000i64, vec![1.1f32, 1.0]), (2000, vec![1.1, 1.0])],
            vec![(1000i64, vec![100.0f32, 100.0]), (2000, vec![100.0, 100.0])], // far
        ];
        let refs: Vec<Vec<(i64, &[f32])>> = owned.iter().map(|t| as_refs(t)).collect();
        let trajectories: Vec<(u64, &[(i64, &[f32])])> = refs
            .iter()
            .enumerate()
            .map(|(i, t)| (i as u64, t.as_slice()))
            .collect();

        let results = group_temporal_join(&trajectories, 0.5, 3, 2000).unwrap();
        assert!(!results.is_empty(), "3 close entities should converge");

        // Entity 3 (far) should NOT be in the converged set
        let converged = &results[0].converged_entities;
        assert!(
            !converged.contains(&3),
            "far entity should not be in cluster"
        );
        assert!(converged.len() >= 3, "at least 3 should converge");
    }

    #[test]
    fn group_join_no_convergence() {
        // All entities far apart
        let owned = [
            vec![(1000i64, vec![0.0f32, 0.0])],
            vec![(1000i64, vec![10.0, 0.0])],
            vec![(1000i64, vec![0.0, 10.0])],
        ];
        let refs: Vec<Vec<(i64, &[f32])>> = owned.iter().map(|t| as_refs(t)).collect();
        let trajectories: Vec<(u64, &[(i64, &[f32])])> = refs
            .iter()
            .enumerate()
            .map(|(i, t)| (i as u64, t.as_slice()))
            .collect();

        let results = group_temporal_join(&trajectories, 0.5, 2, 1000).unwrap();
        assert!(results.is_empty(), "distant entities should not converge");
    }

    // ─── helpers ────────────────────────────────────────────────

    #[test]
    fn window_distances_empty_window() {
        let a_owned = vec![(1000i64, vec![0.0f32])];
        let b_owned = vec![(5000i64, vec![0.0f32])];
        let a = as_refs(&a_owned);
        let b = as_refs(&b_owned);

        // Window that only contains A, not B
        let (min, _, n_a, n_b) = window_distances(&a, &b, 900, 1100);
        assert_eq!(n_a, 1);
        assert_eq!(n_b, 0);
        assert_eq!(min, f32::MAX);
    }

    #[test]
    fn find_cluster_all_close() {
        let reps = vec![
            (0u64, vec![0.0f32, 0.0]),
            (1, vec![0.1, 0.0]),
            (2, vec![0.0, 0.1]),
        ];
        let cluster = find_largest_epsilon_cluster(&reps, 0.5);
        assert_eq!(cluster.len(), 3);
    }

    #[test]
    fn find_cluster_with_outlier() {
        let reps = vec![
            (0u64, vec![0.0f32, 0.0]),
            (1, vec![0.1, 0.0]),
            (2, vec![0.0, 0.1]),
            (3, vec![100.0, 100.0]),
        ];
        let cluster = find_largest_epsilon_cluster(&reps, 0.5);
        assert_eq!(cluster.len(), 3);
        assert!(!cluster.contains(&3));
    }
}
