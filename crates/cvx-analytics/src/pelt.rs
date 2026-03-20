//! PELT (Pruned Exact Linear Time) offline change point detection.
//!
//! Detects structural breaks in embedding trajectories by finding
//! timestamps where the mean vector shifts significantly.
//!
//! # Algorithm
//!
//! 1. Define cost function as sum of squared deviations from segment mean.
//! 2. Dynamic programming with pruning: maintain set of candidate change points.
//! 3. Penalty controls sensitivity (BIC default: `dim * ln(n) / 2`).
//!
//! Complexity: O(N) amortized per Killick et al. (2012).

use cvx_core::types::{ChangePoint, CpdMethod};

/// PELT configuration.
#[derive(Debug, Clone)]
pub struct PeltConfig {
    /// Penalty per change point. If `None`, uses BIC: `dim * ln(n) / 2`.
    pub penalty: Option<f64>,
    /// Minimum segment length (avoids overfitting on short runs).
    pub min_segment_len: usize,
}

impl Default for PeltConfig {
    fn default() -> Self {
        Self {
            penalty: None,
            min_segment_len: 2,
        }
    }
}

/// Run PELT change point detection on a trajectory.
///
/// Input: `trajectory` of `(timestamp, vector)` pairs, sorted by timestamp.
/// Returns detected `ChangePoint`s.
pub fn detect(
    entity_id: u64,
    trajectory: &[(i64, &[f32])],
    config: &PeltConfig,
) -> Vec<ChangePoint> {
    let n = trajectory.len();
    if n < 2 * config.min_segment_len {
        return Vec::new();
    }

    let dim = trajectory[0].1.len();
    let penalty = config
        .penalty
        .unwrap_or_else(|| dim as f64 * (n as f64).ln() / 2.0);

    // Precompute cumulative sums for O(1) segment cost
    let cumsum = build_cumsum(trajectory);

    // DP: f[t] = min cost to segment [0..t]
    // last_cp[t] = last change point before t
    let mut f = vec![f64::INFINITY; n + 1];
    let mut last_cp: Vec<usize> = vec![0; n + 1];
    f[0] = -penalty; // so that f[0] + penalty + cost(0,t) = cost(0,t) for first segment

    // Candidate set for pruning
    let mut candidates: Vec<usize> = vec![0];

    for t in config.min_segment_len..=n {
        let mut best_cost = f64::INFINITY;
        let mut best_cp = 0;

        // Phase 1: find optimal cost f[t] over all candidates
        for &s in &candidates {
            if t - s < config.min_segment_len {
                continue;
            }

            let seg_cost = segment_cost(&cumsum, s, t, dim);
            let total = f[s] + seg_cost + penalty;

            if total < best_cost {
                best_cost = total;
                best_cp = s;
            }
        }

        f[t] = best_cost;
        last_cp[t] = best_cp;

        // Phase 2: prune candidates using the now-known f[t]
        // (RFC-002-06, Killick Theorem 3.1). Pruning AFTER f[t] is set
        // avoids the f[t]==INFINITY fallback that caused O(N) candidate growth.
        let mut new_candidates = Vec::new();
        for &s in &candidates {
            if t - s < config.min_segment_len {
                new_candidates.push(s);
                continue;
            }

            let seg_cost = segment_cost(&cumsum, s, t, dim);
            if f[s] + seg_cost <= f[t] + penalty {
                new_candidates.push(s);
            }
        }
        new_candidates.push(t);

        // Safety cap: prevent O(N²) worst case if pruning is ineffective.
        // Keep candidates with lowest f[s] values.
        if new_candidates.len() > n / 2 {
            new_candidates
                .sort_by(|&a, &b| f[a].partial_cmp(&f[b]).unwrap_or(std::cmp::Ordering::Equal));
            new_candidates.truncate(n / 4);
        }

        candidates = new_candidates;
    }

    // Backtrack to find change points
    let mut cps = Vec::new();
    let mut pos = n;
    while pos > 0 {
        let cp = last_cp[pos];
        if cp > 0 {
            cps.push(cp);
        }
        pos = cp;
    }
    cps.reverse();

    // Convert to ChangePoint structs
    cps.iter()
        .map(|&idx| {
            let ts = trajectory[idx].0;
            let (before_mean, after_mean) = compute_segment_means(trajectory, idx);
            let drift_vector: Vec<f32> = after_mean
                .iter()
                .zip(before_mean.iter())
                .map(|(a, b)| a - b)
                .collect();
            let severity = drift_vector
                .iter()
                .map(|d| (*d as f64) * (*d as f64))
                .sum::<f64>()
                .sqrt();
            // Normalize severity to [0, 1] using sigmoid-like mapping
            let normalized_severity = 1.0 - (-severity).exp();

            ChangePoint::new(
                entity_id,
                ts,
                normalized_severity,
                drift_vector,
                CpdMethod::Pelt,
            )
        })
        .collect()
}

/// Cumulative sums for O(1) segment cost computation.
struct CumulativeSums {
    /// Cumulative sum of vectors: `sum[i]` = Σ_{j=0}^{i-1} v[j]
    sum: Vec<Vec<f64>>,
    /// Cumulative sum of squared norms: `sq[i]` = Σ_{j=0}^{i-1} ||v[j]||²
    sq: Vec<f64>,
}

fn build_cumsum(trajectory: &[(i64, &[f32])]) -> CumulativeSums {
    let n = trajectory.len();
    let dim = trajectory[0].1.len();

    let mut sum = vec![vec![0.0f64; dim]; n + 1];
    let mut sq = vec![0.0f64; n + 1];

    for i in 0..n {
        let v = trajectory[i].1;
        for (d, &val) in v.iter().enumerate().take(dim) {
            sum[i + 1][d] = sum[i][d] + val as f64;
        }
        sq[i + 1] = sq[i] + v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();
    }

    CumulativeSums { sum, sq }
}

/// Cost of segment [start, end): sum of squared deviations from segment mean.
fn segment_cost(cs: &CumulativeSums, start: usize, end: usize, dim: usize) -> f64 {
    let n = (end - start) as f64;
    if n <= 0.0 {
        return 0.0;
    }

    // cost = Σ||x_i||² - (1/n)||Σx_i||²
    let total_sq = cs.sq[end] - cs.sq[start];
    let mut mean_sq = 0.0;
    for d in 0..dim {
        let s = cs.sum[end][d] - cs.sum[start][d];
        mean_sq += s * s;
    }

    total_sq - mean_sq / n
}

/// Compute mean vectors before and after a change point.
fn compute_segment_means(trajectory: &[(i64, &[f32])], cp_idx: usize) -> (Vec<f32>, Vec<f32>) {
    let dim = trajectory[0].1.len();

    let before_mean = segment_mean(&trajectory[..cp_idx], dim);
    let after_end = (cp_idx + 10).min(trajectory.len());
    let after_mean = segment_mean(&trajectory[cp_idx..after_end], dim);

    (before_mean, after_mean)
}

fn segment_mean(segment: &[(i64, &[f32])], dim: usize) -> Vec<f32> {
    if segment.is_empty() {
        return vec![0.0; dim];
    }
    let n = segment.len() as f64;
    let mut mean = vec![0.0f64; dim];
    for (_, v) in segment {
        for d in 0..dim {
            mean[d] += v[d] as f64;
        }
    }
    mean.iter().map(|m| (m / n) as f32).collect()
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    fn make_trajectory(points: &[(i64, Vec<f32>)]) -> Vec<(i64, &[f32])> {
        points.iter().map(|(t, v)| (*t, v.as_slice())).collect()
    }

    #[test]
    fn stationary_no_changepoints() {
        // Constant trajectory — should detect 0 change points
        let points: Vec<(i64, Vec<f32>)> = (0..100)
            .map(|i| (i as i64 * 1000, vec![1.0, 2.0, 3.0]))
            .collect();
        let traj = make_trajectory(&points);

        let cps = detect(1, &traj, &PeltConfig::default());
        assert!(
            cps.is_empty(),
            "stationary data should have 0 changepoints, got {}",
            cps.len()
        );
    }

    #[test]
    fn single_planted_change() {
        // Trajectory with one clear change at index 50
        let mut points: Vec<(i64, Vec<f32>)> = Vec::new();
        for i in 0..50 {
            points.push((i as i64 * 1000, vec![0.0, 0.0, 0.0]));
        }
        for i in 50..100 {
            points.push((i as i64 * 1000, vec![10.0, 10.0, 10.0]));
        }
        let traj = make_trajectory(&points);

        let cps = detect(1, &traj, &PeltConfig::default());
        assert!(!cps.is_empty(), "should detect at least 1 changepoint");

        // Check that a changepoint is near index 50 (ts = 50000)
        let near_50 = cps.iter().any(|cp| (cp.timestamp() - 50000).abs() < 5000);
        assert!(
            near_50,
            "changepoint should be near t=50000, got: {:?}",
            cps.iter().map(|cp| cp.timestamp()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn three_planted_changes() {
        // Segments: [0,25), [25,50), [50,75), [75,100) with different means
        let means = [
            vec![0.0, 0.0],
            vec![5.0, 5.0],
            vec![0.0, 10.0],
            vec![10.0, 0.0],
        ];
        let mut points: Vec<(i64, Vec<f32>)> = Vec::new();
        for seg in 0..4 {
            for i in 0..25 {
                let idx = seg * 25 + i;
                points.push((idx as i64 * 1000, means[seg].clone()));
            }
        }
        let traj = make_trajectory(&points);

        let cps = detect(1, &traj, &PeltConfig::default());

        // Should detect 3 change points (at 25, 50, 75)
        let expected_ts = [25000, 50000, 75000];
        let mut found = 0;
        for &expected in &expected_ts {
            if cps
                .iter()
                .any(|cp| (cp.timestamp() - expected).abs() < 5000)
            {
                found += 1;
            }
        }

        let precision = found as f64 / cps.len().max(1) as f64;
        let recall = found as f64 / 3.0;
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        assert!(
            f1 >= 0.85,
            "F1 = {f1:.2} (precision={precision:.2}, recall={recall:.2}), expected >= 0.85. \
             Detected: {:?}",
            cps.iter().map(|cp| cp.timestamp()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn changepoints_have_severity() {
        let mut points: Vec<(i64, Vec<f32>)> = Vec::new();
        for i in 0..50 {
            points.push((i as i64 * 1000, vec![0.0]));
        }
        for i in 50..100 {
            points.push((i as i64 * 1000, vec![100.0])); // large change
        }
        let traj = make_trajectory(&points);

        let cps = detect(1, &traj, &PeltConfig::default());
        assert!(!cps.is_empty());
        // Large change → high severity
        assert!(
            cps[0].severity() > 0.5,
            "severity should be high for large change, got {}",
            cps[0].severity()
        );
    }

    #[test]
    fn changepoints_have_drift_vector() {
        let mut points: Vec<(i64, Vec<f32>)> = Vec::new();
        for i in 0..50 {
            points.push((i as i64 * 1000, vec![0.0, 0.0]));
        }
        for i in 50..100 {
            points.push((i as i64 * 1000, vec![5.0, -3.0]));
        }
        let traj = make_trajectory(&points);

        let cps = detect(1, &traj, &PeltConfig::default());
        assert!(!cps.is_empty());
        let drift = cps[0].drift_vector();
        assert_eq!(drift.len(), 2);
        // Drift should be approximately [5.0, -3.0]
        assert!((drift[0] - 5.0).abs() < 1.0, "drift[0] = {}", drift[0]);
        assert!((drift[1] + 3.0).abs() < 1.0, "drift[1] = {}", drift[1]);
    }

    #[test]
    fn short_trajectory_returns_empty() {
        let points = vec![(0i64, vec![1.0]), (1000, vec![2.0])];
        let traj = make_trajectory(&points);
        let cps = detect(
            1,
            &traj,
            &PeltConfig {
                min_segment_len: 3,
                ..Default::default()
            },
        );
        assert!(cps.is_empty());
    }

    #[test]
    fn custom_penalty() {
        // Higher penalty → fewer changepoints
        let mut points: Vec<(i64, Vec<f32>)> = Vec::new();
        for i in 0..50 {
            points.push((i as i64 * 1000, vec![0.0]));
        }
        for i in 50..100 {
            points.push((i as i64 * 1000, vec![2.0]));
        }
        let traj = make_trajectory(&points);

        let low_penalty = detect(
            1,
            &traj,
            &PeltConfig {
                penalty: Some(1.0),
                ..Default::default()
            },
        );
        let high_penalty = detect(
            1,
            &traj,
            &PeltConfig {
                penalty: Some(1000.0),
                ..Default::default()
            },
        );

        assert!(
            low_penalty.len() >= high_penalty.len(),
            "higher penalty should find fewer or equal changepoints"
        );
    }

    #[test]
    fn all_changepoints_are_pelt_method() {
        let mut points: Vec<(i64, Vec<f32>)> = Vec::new();
        for i in 0..50 {
            points.push((i as i64 * 1000, vec![0.0]));
        }
        for i in 50..100 {
            points.push((i as i64 * 1000, vec![10.0]));
        }
        let traj = make_trajectory(&points);

        let cps = detect(1, &traj, &PeltConfig::default());
        for cp in &cps {
            assert_eq!(cp.method(), CpdMethod::Pelt);
        }
    }
}
