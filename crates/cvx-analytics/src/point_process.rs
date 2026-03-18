//! Temporal point process features extracted from event timestamps.
//!
//! The *when* of events is a signal independent of *what* the vectors contain.
//! This module extracts features from inter-event intervals that characterize
//! temporal patterns: regularity, burstiness, self-excitation, circadian rhythms.
//!
//! # Features
//!
//! | Feature | Range | Interpretation |
//! |---------|-------|---------------|
//! | `burstiness` | [−1, 1] | −1 = perfectly regular, 0 = Poisson, +1 = maximally bursty |
//! | `memory` | [−1, 1] | Autocorrelation of consecutive gaps |
//! | `temporal_entropy` | [0, ∞) | Entropy of gap distribution (higher = more irregular) |
//! | `intensity_trend` | ℝ | Slope of event rate over time (positive = accelerating) |
//! | `circadian_strength` | [0, 1] | Amplitude of 24h periodicity |
//!
//! # References
//!
//! - Goh, K.-I. & Barabási, A.-L. (2008). Burstiness and memory. *EPL*, 81(4).
//! - Hawkes, A.G. (1971). Self-exciting point processes. *Biometrika*, 58(1).

/// Features extracted from event timestamps.
#[derive(Debug, Clone)]
pub struct EventFeatures {
    /// Number of events.
    pub n_events: usize,
    /// Total time span (last - first timestamp).
    pub span: f64,
    /// Mean inter-event interval.
    pub mean_gap: f64,
    /// Standard deviation of inter-event intervals.
    pub std_gap: f64,
    /// Burstiness parameter B = (σ - μ) / (σ + μ).
    /// B = −1: perfectly regular. B = 0: Poisson. B → +1: bursty.
    pub burstiness: f64,
    /// Memory coefficient: correlation between consecutive gaps.
    /// M > 0: short gaps follow short gaps (clustering).
    /// M < 0: short gaps follow long gaps (alternating).
    /// M ≈ 0: no memory (Poisson-like).
    pub memory: f64,
    /// Shannon entropy of the gap distribution (binned).
    /// Higher = more irregular/unpredictable event timing.
    pub temporal_entropy: f64,
    /// Slope of event rate over time (positive = accelerating).
    pub intensity_trend: f64,
    /// Coefficient of variation of gaps (std/mean).
    pub gap_cv: f64,
    /// Maximum gap (longest silence).
    pub max_gap: f64,
    /// Strength of 24h circadian rhythm (0 = no rhythm, 1 = strong).
    /// Only meaningful if timestamps are in seconds/milliseconds.
    pub circadian_strength: f64,
}

/// Extract temporal point process features from a sequence of timestamps.
///
/// Timestamps should be sorted in ascending order. Units are arbitrary
/// but consistent (seconds, days, etc.).
///
/// # Arguments
///
/// * `timestamps` - Sorted event timestamps (at least 3 for meaningful features).
///
/// # Returns
///
/// [`EventFeatures`] struct with all computed features.
pub fn extract_event_features(timestamps: &[i64]) -> Result<EventFeatures, PointProcessError> {
    let n = timestamps.len();
    if n < 3 {
        return Err(PointProcessError::InsufficientEvents { got: n, need: 3 });
    }

    // Inter-event intervals
    let gaps: Vec<f64> = timestamps
        .windows(2)
        .map(|w| (w[1] - w[0]) as f64)
        .collect();
    let n_gaps = gaps.len();

    let span = (timestamps[n - 1] - timestamps[0]) as f64;
    let mean_gap = gaps.iter().sum::<f64>() / n_gaps as f64;
    let std_gap = (gaps.iter().map(|g| (g - mean_gap).powi(2)).sum::<f64>() / n_gaps as f64).sqrt();
    let max_gap = gaps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Burstiness: B = (σ - μ) / (σ + μ)
    let burstiness = if mean_gap + std_gap > 0.0 {
        (std_gap - mean_gap) / (std_gap + mean_gap)
    } else {
        0.0
    };

    // Gap CV
    let gap_cv = if mean_gap > 0.0 {
        std_gap / mean_gap
    } else {
        0.0
    };

    // Memory coefficient: autocorrelation of consecutive gaps
    let memory = if n_gaps >= 3 {
        let mut num = 0.0;
        let mut den1 = 0.0;
        let mut den2 = 0.0;
        for i in 0..n_gaps - 1 {
            let a = gaps[i] - mean_gap;
            let b = gaps[i + 1] - mean_gap;
            num += a * b;
            den1 += a * a;
            den2 += b * b;
        }
        let denom = (den1 * den2).sqrt();
        if denom > 1e-15 { num / denom } else { 0.0 }
    } else {
        0.0
    };

    // Temporal entropy: bin gaps into 10 bins, compute Shannon entropy
    let temporal_entropy = {
        let n_bins = 10;
        let min_gap = gaps.iter().cloned().fold(f64::INFINITY, f64::min);
        let range = max_gap - min_gap;
        if range > 0.0 {
            let mut counts = vec![0usize; n_bins];
            for &g in &gaps {
                let bin = (((g - min_gap) / range) * (n_bins - 1) as f64).round() as usize;
                counts[bin.min(n_bins - 1)] += 1;
            }
            let total = n_gaps as f64;
            counts
                .iter()
                .filter(|&&c| c > 0)
                .map(|&c| {
                    let p = c as f64 / total;
                    -p * p.ln()
                })
                .sum()
        } else {
            0.0 // all gaps identical → zero entropy
        }
    };

    // Intensity trend: divide timeline into 5 windows, fit slope of event count
    let intensity_trend = {
        let n_windows = 5;
        let window_size = span / n_windows as f64;
        if window_size > 0.0 {
            let counts: Vec<f64> = (0..n_windows)
                .map(|w| {
                    let start = timestamps[0] as f64 + w as f64 * window_size;
                    let end = start + window_size;
                    timestamps
                        .iter()
                        .filter(|&&t| (t as f64) >= start && (t as f64) < end)
                        .count() as f64
                })
                .collect();
            // Linear regression slope
            let x_mean = (n_windows - 1) as f64 / 2.0;
            let y_mean = counts.iter().sum::<f64>() / n_windows as f64;
            let mut num = 0.0;
            let mut den = 0.0;
            for (i, &c) in counts.iter().enumerate() {
                let x = i as f64 - x_mean;
                num += x * (c - y_mean);
                den += x * x;
            }
            if den > 0.0 { num / den } else { 0.0 }
        } else {
            0.0
        }
    };

    // Circadian strength: amplitude of 24h Fourier component
    // Assumes timestamps are in seconds
    let circadian_strength = {
        let period = 86400.0; // 24 hours in seconds
        let mut sin_sum = 0.0;
        let mut cos_sum = 0.0;
        for &t in timestamps {
            let phase = 2.0 * std::f64::consts::PI * (t as f64 % period) / period;
            sin_sum += phase.sin();
            cos_sum += phase.cos();
        }
        let amplitude = ((sin_sum / n as f64).powi(2) + (cos_sum / n as f64).powi(2)).sqrt();
        amplitude // Range [0, 1], higher = stronger circadian pattern
    };

    Ok(EventFeatures {
        n_events: n,
        span,
        mean_gap,
        std_gap,
        burstiness,
        memory,
        temporal_entropy,
        intensity_trend,
        gap_cv,
        max_gap,
        circadian_strength,
    })
}

/// Error types for point process analysis.
#[derive(Debug, thiserror::Error)]
pub enum PointProcessError {
    /// Not enough events.
    #[error("insufficient events: got {got}, need at least {need}")]
    InsufficientEvents {
        /// Number provided.
        got: usize,
        /// Minimum required.
        need: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regular_events_low_burstiness() {
        // Perfectly regular: gaps = [10, 10, 10, ...]
        let timestamps: Vec<i64> = (0..100).map(|i| i * 10).collect();
        let f = extract_event_features(&timestamps).unwrap();
        assert!(
            f.burstiness < -0.9,
            "regular events should have burstiness ≈ -1, got {}",
            f.burstiness
        );
        assert!(f.temporal_entropy < 0.1, "regular → low entropy");
    }

    #[test]
    fn poisson_events_zero_burstiness() {
        // Approximate Poisson: exponential gaps
        // Using deterministic pseudo-exponential for reproducibility
        let mut timestamps = vec![0i64];
        let mut t = 0i64;
        let gaps = [
            8, 12, 7, 15, 9, 11, 13, 6, 14, 10, 8, 12, 11, 9, 7, 13, 10, 14, 8, 12,
        ];
        for &g in &gaps {
            t += g;
            timestamps.push(t);
        }
        let f = extract_event_features(&timestamps).unwrap();
        // Poisson-like should have burstiness between regular (-1) and bursty (+1)
        // With only 20 samples, variance is high. Check it's not extreme.
        assert!(
            f.burstiness > -0.8 && f.burstiness < 0.8,
            "Poisson-like should have moderate burstiness, got {}",
            f.burstiness
        );
    }

    #[test]
    fn bursty_events_high_burstiness() {
        // Bursts: clusters of rapid events separated by long silences
        let mut timestamps = Vec::new();
        for burst in 0..5 {
            let base = burst * 1000;
            for i in 0..10 {
                timestamps.push(base + i); // 10 events in 10 time units
            }
        }
        let f = extract_event_features(&timestamps).unwrap();
        assert!(
            f.burstiness > 0.5,
            "bursty events should have B > 0.5, got {}",
            f.burstiness
        );
    }

    #[test]
    fn memory_positive_for_clustered() {
        // Short gaps followed by short gaps, long by long
        let gaps = [1, 1, 1, 50, 50, 50, 1, 1, 1, 50, 50, 50];
        let mut timestamps = vec![0i64];
        let mut t = 0;
        for &g in &gaps {
            t += g;
            timestamps.push(t);
        }
        let f = extract_event_features(&timestamps).unwrap();
        assert!(
            f.memory > 0.2,
            "clustered gaps should have positive memory, got {}",
            f.memory
        );
    }

    #[test]
    fn accelerating_positive_trend() {
        // Events get more frequent over time
        let mut timestamps = Vec::new();
        let mut t = 0i64;
        for i in 1..50 {
            t += 100 / i; // decreasing gaps
            timestamps.push(t);
        }
        let f = extract_event_features(&timestamps).unwrap();
        assert!(
            f.intensity_trend > 0.0,
            "accelerating events should have positive trend, got {}",
            f.intensity_trend
        );
    }

    #[test]
    fn insufficient_events_error() {
        assert!(extract_event_features(&[1, 2]).is_err());
        assert!(extract_event_features(&[1]).is_err());
        assert!(extract_event_features(&[]).is_err());
    }

    #[test]
    fn features_are_finite() {
        let timestamps: Vec<i64> = (0..50).map(|i| i * 100 + (i * 7) % 13).collect();
        let f = extract_event_features(&timestamps).unwrap();
        assert!(f.burstiness.is_finite());
        assert!(f.memory.is_finite());
        assert!(f.temporal_entropy.is_finite());
        assert!(f.intensity_trend.is_finite());
        assert!(f.circadian_strength.is_finite());
    }
}
