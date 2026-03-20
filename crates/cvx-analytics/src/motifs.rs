//! Temporal motif discovery via Matrix Profile.
//!
//! Detects **recurring patterns** (motifs) and **anomalous subsequences** (discords)
//! in entity trajectories. Adapted from the STOMP algorithm (Zhu et al., 2016)
//! to operate on multi-dimensional embedding trajectories.
//!
//! # Key insight
//!
//! Operating on region trajectories (K ≈ 60-80 dims) rather than raw embeddings
//! (D = 768) makes the Matrix Profile tractable: O(N² × K) for a 500-window
//! trajectory with K=80 is ~20M operations.
//!
//! # References
//!
//! - Yeh, C.-C. M. et al. (2016). Matrix Profile I. *IEEE ICDM*.
//! - Zhu, Y. et al. (2016). Matrix Profile II: STOMP. *IEEE ICDM*.

use cvx_core::error::AnalyticsError;

// ─── Types ──────────────────────────────────────────────────────────

/// A discovered recurring pattern in a trajectory.
#[derive(Debug, Clone)]
pub struct Motif {
    /// Index of the canonical occurrence in the trajectory.
    pub canonical_index: usize,
    /// All occurrences: `(start_index, distance_to_canonical)`.
    pub occurrences: Vec<MotifOccurrence>,
    /// Detected period in number of steps (if regular). `None` if aperiodic.
    pub period: Option<usize>,
    /// Mean distance across all occurrences to canonical.
    pub mean_match_distance: f32,
}

/// A single occurrence of a motif.
#[derive(Debug, Clone)]
pub struct MotifOccurrence {
    /// Start index in the trajectory.
    pub start_index: usize,
    /// Timestamp of the start of this occurrence.
    pub timestamp: i64,
    /// Distance to the canonical motif.
    pub distance: f32,
}

/// An anomalous subsequence (most unlike anything else in the trajectory).
#[derive(Debug, Clone)]
pub struct Discord {
    /// Start index in the trajectory.
    pub start_index: usize,
    /// Timestamp of the start.
    pub timestamp: i64,
    /// Matrix Profile value (nearest-neighbor distance — high = unusual).
    pub nn_distance: f32,
}

// ─── Matrix Profile computation ─────────────────────────────────────

/// Compute the Matrix Profile for a multi-dimensional trajectory.
///
/// Returns `(profile, profile_index)` where:
/// - `profile[i]` = distance to the nearest non-trivial match of subsequence i
/// - `profile_index[i]` = index of that nearest match
///
/// Uses the STOMP-like approach: for each subsequence, compute distances to
/// all other non-overlapping subsequences and record the minimum.
///
/// # Arguments
///
/// * `trajectory` — Time-ordered `(timestamp, vector)` pairs
/// * `window` — Subsequence length (number of time steps)
/// * `exclusion_zone` — Fraction of window size to exclude around each
///   subsequence to avoid trivial matches (default: 0.5)
///
/// # Errors
///
/// Returns [`AnalyticsError::InsufficientData`] if trajectory length < 2 * window.
pub fn matrix_profile(
    trajectory: &[(i64, &[f32])],
    window: usize,
    exclusion_zone: f32,
) -> Result<(Vec<f32>, Vec<usize>), AnalyticsError> {
    let n = trajectory.len();
    if n < 2 * window {
        return Err(AnalyticsError::InsufficientData {
            needed: 2 * window,
            have: n,
        });
    }

    let n_subs = n - window + 1;
    let excl = ((window as f32) * exclusion_zone).ceil() as usize;

    let mut profile = vec![f32::MAX; n_subs];
    let mut profile_index = vec![0usize; n_subs];

    for i in 0..n_subs {
        for j in 0..n_subs {
            // Skip if within exclusion zone
            if i.abs_diff(j) < excl {
                continue;
            }

            let dist = subsequence_distance(trajectory, i, j, window);

            if dist < profile[i] {
                profile[i] = dist;
                profile_index[i] = j;
            }
        }
    }

    Ok((profile, profile_index))
}

/// Euclidean distance between two subsequences of the trajectory.
fn subsequence_distance(trajectory: &[(i64, &[f32])], i: usize, j: usize, window: usize) -> f32 {
    let mut sum_sq = 0.0f64;
    for step in 0..window {
        let vi = trajectory[i + step].1;
        let vj = trajectory[j + step].1;
        for (a, b) in vi.iter().zip(vj.iter()) {
            let diff = (*a as f64) - (*b as f64);
            sum_sq += diff * diff;
        }
    }
    (sum_sq / window as f64).sqrt() as f32
}

// ─── Motif discovery ────────────────────────────────────────────────

/// Discover the top-k recurring motifs in a trajectory.
///
/// A motif is the subsequence with the smallest Matrix Profile value
/// (i.e., it has a very similar match elsewhere). We follow the Matrix
/// Profile Index chain to find all occurrences.
///
/// # Arguments
///
/// * `trajectory` — Time-ordered `(timestamp, vector)` pairs
/// * `window` — Subsequence length
/// * `max_motifs` — Maximum number of motifs to return
/// * `exclusion_zone` — Fraction of window for non-trivial matches (default: 0.5)
///
/// # Errors
///
/// Returns [`AnalyticsError::InsufficientData`] if trajectory too short.
pub fn discover_motifs(
    trajectory: &[(i64, &[f32])],
    window: usize,
    max_motifs: usize,
    exclusion_zone: f32,
) -> Result<Vec<Motif>, AnalyticsError> {
    let (profile, profile_index) = matrix_profile(trajectory, window, exclusion_zone)?;
    let n_subs = profile.len();
    let excl = ((window as f32) * exclusion_zone).ceil() as usize;

    // Sort by profile value (smallest = best motif)
    let mut indices: Vec<usize> = (0..n_subs).collect();
    indices.sort_by(|&a, &b| profile[a].partial_cmp(&profile[b]).unwrap());

    let mut motifs: Vec<Motif> = Vec::new();
    let mut used = vec![false; n_subs]; // prevent overlapping motifs

    for &candidate in &indices {
        if motifs.len() >= max_motifs {
            break;
        }
        if used[candidate] || profile[candidate] == f32::MAX {
            continue;
        }

        // Collect all occurrences: subsequences within 2× the motif distance
        let threshold = (profile[candidate] * 2.0).max(1e-6);
        let mut occurrences: Vec<MotifOccurrence> = Vec::new();

        // The canonical occurrence
        occurrences.push(MotifOccurrence {
            start_index: candidate,
            timestamp: trajectory[candidate].0,
            distance: 0.0,
        });

        // The nearest match
        let nn = profile_index[candidate];
        occurrences.push(MotifOccurrence {
            start_index: nn,
            timestamp: trajectory[nn].0,
            distance: profile[candidate],
        });

        // Find additional occurrences
        for k in 0..n_subs {
            if k == candidate || k == nn {
                continue;
            }
            if k.abs_diff(candidate) < excl || k.abs_diff(nn) < excl {
                continue;
            }
            let dist = subsequence_distance(trajectory, candidate, k, window);
            if dist <= threshold {
                occurrences.push(MotifOccurrence {
                    start_index: k,
                    timestamp: trajectory[k].0,
                    distance: dist,
                });
            }
        }

        // Mark all occurrences as used
        for occ in &occurrences {
            let start = occ.start_index.saturating_sub(excl);
            let end = (occ.start_index + excl).min(n_subs);
            for flag in used.iter_mut().take(end).skip(start) {
                *flag = true;
            }
        }

        // Sort occurrences by index (temporal order)
        occurrences.sort_by_key(|o| o.start_index);

        // Detect periodicity
        let period = detect_period(&occurrences);

        let mean_match_distance = if occurrences.len() > 1 {
            occurrences.iter().map(|o| o.distance).sum::<f32>() / occurrences.len() as f32
        } else {
            0.0
        };

        motifs.push(Motif {
            canonical_index: candidate,
            occurrences,
            period,
            mean_match_distance,
        });
    }

    Ok(motifs)
}

// ─── Discord discovery ──────────────────────────────────────────────

/// Discover the top-k anomalous subsequences (discords).
///
/// A discord is the subsequence with the **largest** Matrix Profile value
/// (i.e., it is most unlike any other subsequence).
///
/// # Arguments
///
/// * `trajectory` — Time-ordered `(timestamp, vector)` pairs
/// * `window` — Subsequence length
/// * `max_discords` — Maximum number of discords to return
///
/// # Errors
///
/// Returns [`AnalyticsError::InsufficientData`] if trajectory too short.
pub fn discover_discords(
    trajectory: &[(i64, &[f32])],
    window: usize,
    max_discords: usize,
) -> Result<Vec<Discord>, AnalyticsError> {
    let (profile, _) = matrix_profile(trajectory, window, 0.5)?;
    let n_subs = profile.len();
    let excl = ((window as f32) * 0.5).ceil() as usize;

    // Sort by profile value descending (largest = most anomalous)
    let mut indices: Vec<usize> = (0..n_subs).collect();
    indices.sort_by(|&a, &b| profile[b].partial_cmp(&profile[a]).unwrap());

    let mut discords: Vec<Discord> = Vec::new();
    let mut used = vec![false; n_subs];

    for &candidate in &indices {
        if discords.len() >= max_discords {
            break;
        }
        if used[candidate] || profile[candidate] == f32::MAX {
            continue;
        }

        discords.push(Discord {
            start_index: candidate,
            timestamp: trajectory[candidate].0,
            nn_distance: profile[candidate],
        });

        // Mark exclusion zone
        let start = candidate.saturating_sub(excl);
        let end = (candidate + excl).min(n_subs);
        for flag in used.iter_mut().take(end).skip(start) {
            *flag = true;
        }
    }

    Ok(discords)
}

// ─── Helpers ────────────────────────────────────────────────────────

/// Detect periodicity from occurrence indices.
///
/// If gaps between consecutive occurrences are within 20% of the median gap,
/// returns the median gap as the period. Otherwise returns `None`.
fn detect_period(occurrences: &[MotifOccurrence]) -> Option<usize> {
    if occurrences.len() < 3 {
        return None;
    }

    let gaps: Vec<usize> = occurrences
        .windows(2)
        .map(|w| w[1].start_index - w[0].start_index)
        .collect();

    if gaps.is_empty() {
        return None;
    }

    let mut sorted_gaps = gaps.clone();
    sorted_gaps.sort();
    let median_gap = sorted_gaps[sorted_gaps.len() / 2];

    if median_gap == 0 {
        return None;
    }

    // Check regularity: all gaps within 20% of median
    let tolerance = (median_gap as f64 * 0.2).max(1.0) as usize;
    let is_regular = gaps.iter().all(|&g| g.abs_diff(median_gap) <= tolerance);

    if is_regular { Some(median_gap) } else { None }
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    fn as_refs(points: &[(i64, Vec<f32>)]) -> Vec<(i64, &[f32])> {
        points.iter().map(|(t, v)| (*t, v.as_slice())).collect()
    }

    /// Build a trajectory with a planted motif that repeats at known positions.
    fn trajectory_with_planted_motif() -> Vec<(i64, Vec<f32>)> {
        let dim = 4;
        let n = 100;
        let mut traj = Vec::with_capacity(n);

        for i in 0..n {
            let t = i as i64 * 1000;
            // Base: slowly varying signal
            let base: Vec<f32> = (0..dim).map(|d| (i as f32 * 0.01 + d as f32)).collect();
            traj.push((t, base));
        }

        // Plant a motif at positions 10, 40, 70 (period = 30)
        let motif_pattern: Vec<Vec<f32>> = (0..5)
            .map(|step| {
                (0..dim)
                    .map(|d| 10.0 + (step as f32 * 0.5) + d as f32 * 0.1)
                    .collect()
            })
            .collect();

        for &start in &[10, 40, 70] {
            for (step, pattern) in motif_pattern.iter().enumerate() {
                traj[start + step].1 = pattern.clone();
            }
        }

        traj
    }

    // ─── matrix_profile ─────────────────────────────────────────

    #[test]
    fn matrix_profile_insufficient_data() {
        let owned = vec![(0i64, vec![1.0f32]), (1, vec![2.0])];
        let traj = as_refs(&owned);
        let result = matrix_profile(&traj, 3, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn matrix_profile_basic() {
        // Simple trajectory with 10 points
        let owned: Vec<(i64, Vec<f32>)> = (0..10)
            .map(|i| (i as i64 * 1000, vec![i as f32 * 0.1]))
            .collect();
        let traj = as_refs(&owned);

        let (profile, index) = matrix_profile(&traj, 3, 0.5).unwrap();
        assert_eq!(profile.len(), 8); // n - window + 1 = 10 - 3 + 1
        assert_eq!(index.len(), 8);

        // All profile values should be finite and positive
        for &val in &profile {
            assert!(val.is_finite());
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn matrix_profile_identical_subsequences() {
        // Trajectory where positions 0-2 and 5-7 are identical
        let mut owned: Vec<(i64, Vec<f32>)> =
            (0..10).map(|i| (i as i64 * 1000, vec![i as f32])).collect();

        // Make positions 5,6,7 identical to 0,1,2
        owned[5].1 = vec![0.0];
        owned[6].1 = vec![1.0];
        owned[7].1 = vec![2.0];

        let traj = as_refs(&owned);
        let (profile, index) = matrix_profile(&traj, 3, 0.5).unwrap();

        // The profile at index 0 should be very low (has a near-perfect match at 5)
        assert!(
            profile[0] < 0.01,
            "identical subsequences should have ~0 profile, got {}",
            profile[0]
        );
        assert_eq!(index[0], 5);
    }

    // ─── discover_motifs ────────────────────────────────────────

    #[test]
    fn motifs_planted_pattern() {
        let owned = trajectory_with_planted_motif();
        let traj = as_refs(&owned);

        let motifs = discover_motifs(&traj, 5, 3, 0.5).unwrap();

        assert!(!motifs.is_empty(), "should find at least one motif");

        let best = &motifs[0];
        assert!(
            best.occurrences.len() >= 2,
            "best motif should have at least 2 occurrences, got {}",
            best.occurrences.len()
        );

        // The canonical should be at one of the planted positions (10, 40, 70)
        let planted = [10usize, 40, 70];
        let found_planted = best
            .occurrences
            .iter()
            .any(|o| planted.iter().any(|&p| o.start_index.abs_diff(p) <= 2));
        assert!(
            found_planted,
            "should find at least one planted position, got indices: {:?}",
            best.occurrences
                .iter()
                .map(|o| o.start_index)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn motifs_period_detection() {
        let owned = trajectory_with_planted_motif();
        let traj = as_refs(&owned);

        let motifs = discover_motifs(&traj, 5, 1, 0.5).unwrap();
        assert!(!motifs.is_empty());

        let best = &motifs[0];
        if best.occurrences.len() >= 3 {
            // If 3 occurrences found, period should be detected as ~30
            if let Some(period) = best.period {
                assert!(
                    period.abs_diff(30) <= 6,
                    "expected period ~30, got {period}"
                );
            }
        }
    }

    #[test]
    fn motifs_stationary_trajectory() {
        // Constant trajectory — everything is a motif
        let owned: Vec<(i64, Vec<f32>)> =
            (0..30).map(|i| (i as i64 * 1000, vec![1.0, 2.0])).collect();
        let traj = as_refs(&owned);

        let motifs = discover_motifs(&traj, 5, 3, 0.5).unwrap();
        assert!(!motifs.is_empty());
        // Best motif should have very low match distance
        assert!(
            motifs[0].mean_match_distance < 0.01,
            "constant trajectory should have ~0 match distance"
        );
    }

    #[test]
    fn motifs_max_motifs_respected() {
        let owned: Vec<(i64, Vec<f32>)> = (0..50)
            .map(|i| (i as i64 * 1000, vec![(i as f32).sin()]))
            .collect();
        let traj = as_refs(&owned);

        let motifs = discover_motifs(&traj, 5, 2, 0.5).unwrap();
        assert!(motifs.len() <= 2);
    }

    // ─── discover_discords ──────────────────────────────────────

    #[test]
    fn discords_planted_anomaly() {
        let dim = 2;
        let mut owned: Vec<(i64, Vec<f32>)> =
            (0..50).map(|i| (i as i64 * 1000, vec![1.0; dim])).collect();

        // Plant an anomaly at position 25
        for step in 25..30 {
            owned[step].1 = vec![100.0; dim];
        }

        let traj = as_refs(&owned);
        let discords = discover_discords(&traj, 5, 3).unwrap();

        assert!(!discords.is_empty(), "should detect planted anomaly");

        // The top discord should be near position 25
        let top = &discords[0];
        assert!(
            top.start_index.abs_diff(25) <= 5,
            "top discord should be near position 25, got {}",
            top.start_index
        );
        assert!(
            top.nn_distance > 1.0,
            "discord should have high nn_distance, got {}",
            top.nn_distance
        );
    }

    #[test]
    fn discords_constant_trajectory() {
        // Constant trajectory — no real discords
        let owned: Vec<(i64, Vec<f32>)> = (0..30).map(|i| (i as i64 * 1000, vec![1.0])).collect();
        let traj = as_refs(&owned);

        let discords = discover_discords(&traj, 5, 3).unwrap();
        // All nn_distances should be very low
        for d in &discords {
            assert!(
                d.nn_distance < 0.01,
                "constant trajectory should have ~0 discord distance"
            );
        }
    }

    #[test]
    fn discords_max_discords_respected() {
        let owned: Vec<(i64, Vec<f32>)> = (0..50)
            .map(|i| (i as i64 * 1000, vec![(i as f32 * 0.1).sin()]))
            .collect();
        let traj = as_refs(&owned);

        let discords = discover_discords(&traj, 5, 1).unwrap();
        assert!(discords.len() <= 1);
    }

    // ─── detect_period ──────────────────────────────────────────

    #[test]
    fn period_regular_spacing() {
        let occs = vec![
            MotifOccurrence {
                start_index: 10,
                timestamp: 10000,
                distance: 0.0,
            },
            MotifOccurrence {
                start_index: 30,
                timestamp: 30000,
                distance: 0.1,
            },
            MotifOccurrence {
                start_index: 50,
                timestamp: 50000,
                distance: 0.1,
            },
            MotifOccurrence {
                start_index: 70,
                timestamp: 70000,
                distance: 0.05,
            },
        ];
        let period = detect_period(&occs);
        assert_eq!(period, Some(20));
    }

    #[test]
    fn period_irregular_spacing() {
        let occs = vec![
            MotifOccurrence {
                start_index: 5,
                timestamp: 5000,
                distance: 0.0,
            },
            MotifOccurrence {
                start_index: 10,
                timestamp: 10000,
                distance: 0.1,
            },
            MotifOccurrence {
                start_index: 50,
                timestamp: 50000,
                distance: 0.1,
            },
        ];
        let period = detect_period(&occs);
        assert!(period.is_none(), "irregular gaps should return None");
    }

    #[test]
    fn period_too_few_occurrences() {
        let occs = vec![
            MotifOccurrence {
                start_index: 0,
                timestamp: 0,
                distance: 0.0,
            },
            MotifOccurrence {
                start_index: 10,
                timestamp: 10000,
                distance: 0.1,
            },
        ];
        assert!(detect_period(&occs).is_none());
    }

    // ─── subsequence_distance ───────────────────────────────────

    #[test]
    fn subseq_distance_identical() {
        let owned: Vec<(i64, Vec<f32>)> = (0..10).map(|i| (i as i64, vec![1.0, 2.0])).collect();
        let traj = as_refs(&owned);
        let dist = subsequence_distance(&traj, 0, 5, 3);
        assert!(
            dist < 1e-6,
            "identical subsequences should have ~0 distance"
        );
    }

    #[test]
    fn subseq_distance_different() {
        let mut owned: Vec<(i64, Vec<f32>)> = (0..10).map(|i| (i as i64, vec![0.0])).collect();
        // Make positions 5,6,7 = [10.0]
        for i in 5..8 {
            owned[i].1 = vec![10.0];
        }
        let traj = as_refs(&owned);
        let dist = subsequence_distance(&traj, 0, 5, 3);
        assert!(
            dist > 1.0,
            "different subsequences should have large distance"
        );
    }
}
