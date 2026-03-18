//! Online change point detection via exponentially weighted statistics.
//!
//! A streaming detector that maintains an exponentially weighted mean and
//! variance. When a new observation deviates significantly from the expected
//! distribution (Mahalanobis-like distance exceeds threshold), a change
//! point is emitted.
//!
//! This is a practical simplification of BOCPD that works well for
//! embedding trajectories where the distribution shifts are typically large.

use cvx_core::types::{ChangePoint, CpdMethod};

/// Online detector configuration.
#[derive(Debug, Clone)]
pub struct BocpdConfig {
    /// Smoothing factor for exponential moving average (0 < alpha < 1).
    /// Lower values = more smoothing = less sensitive.
    pub alpha: f64,
    /// Number of standard deviations for change detection threshold.
    pub threshold_sigmas: f64,
    /// Minimum observations before detection starts (warm-up period).
    pub min_observations: usize,
    /// Cooldown: minimum observations between consecutive detections.
    pub cooldown: usize,
}

impl Default for BocpdConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            threshold_sigmas: 3.0,
            min_observations: 5,
            cooldown: 5,
        }
    }
}

/// Online streaming change point detector.
pub struct BocpdDetector {
    config: BocpdConfig,
    entity_id: u64,
    /// Exponentially weighted mean per dimension.
    ew_mean: Vec<f64>,
    /// Exponentially weighted variance per dimension.
    ew_var: Vec<f64>,
    /// Previous mean (for drift vector computation).
    prev_mean: Vec<f32>,
    /// Number of observations processed.
    count: usize,
    /// Observations since last detection (cooldown counter).
    since_last_detection: usize,
    /// Dimensionality.
    dim: Option<usize>,
}

impl BocpdDetector {
    /// Create a new detector for the given entity.
    pub fn new(entity_id: u64, config: BocpdConfig) -> Self {
        Self {
            config,
            entity_id,
            ew_mean: Vec::new(),
            ew_var: Vec::new(),
            prev_mean: Vec::new(),
            count: 0,
            since_last_detection: usize::MAX / 2,
            dim: None,
        }
    }

    /// Process a single observation.
    ///
    /// Returns `Some(ChangePoint)` if a change is detected.
    pub fn observe(&mut self, timestamp: i64, vector: &[f32]) -> Option<ChangePoint> {
        let dim = *self.dim.get_or_insert(vector.len());
        assert_eq!(vector.len(), dim, "dimension mismatch");

        self.count += 1;
        self.since_last_detection += 1;

        // Initialize on first observation
        if self.ew_mean.is_empty() {
            self.ew_mean = vector.iter().map(|&x| x as f64).collect();
            self.ew_var = vec![1.0; dim]; // initial variance
            self.prev_mean = vector.to_vec();
            return None;
        }

        // Compute deviation from expected distribution
        let deviation = self.mahalanobis_like(vector);

        // Update exponential moving statistics
        let alpha = self.config.alpha;
        #[allow(clippy::needless_range_loop)]
        for d in 0..dim {
            let x = vector[d] as f64;
            let diff = x - self.ew_mean[d];
            self.ew_mean[d] += alpha * diff;
            self.ew_var[d] = (1.0 - alpha) * (self.ew_var[d] + alpha * diff * diff);
        }

        // Check for change point
        let is_change = self.count > self.config.min_observations
            && self.since_last_detection >= self.config.cooldown
            && deviation > self.config.threshold_sigmas;

        if is_change {
            let current_mean: Vec<f32> = self.ew_mean.iter().map(|&x| x as f32).collect();
            let drift_vector: Vec<f32> = current_mean
                .iter()
                .zip(self.prev_mean.iter())
                .map(|(a, b)| a - b)
                .collect();

            let severity = drift_vector
                .iter()
                .map(|d| (*d as f64) * (*d as f64))
                .sum::<f64>()
                .sqrt();
            let normalized_severity = 1.0 - (-severity).exp();

            self.prev_mean = current_mean;
            self.since_last_detection = 0;

            Some(ChangePoint::new(
                self.entity_id,
                timestamp,
                normalized_severity,
                drift_vector,
                CpdMethod::Bocpd,
            ))
        } else {
            None
        }
    }

    /// Compute a Mahalanobis-like distance (average z-score across dimensions).
    fn mahalanobis_like(&self, vector: &[f32]) -> f64 {
        let dim = self.ew_mean.len();
        let mut total_z2 = 0.0;

        for (d, &v) in vector.iter().enumerate().take(dim) {
            let x = v as f64;
            let diff = x - self.ew_mean[d];
            let var = self.ew_var[d].max(1e-10);
            total_z2 += diff * diff / var;
        }

        (total_z2 / dim as f64).sqrt()
    }

    /// Current number of observations processed.
    pub fn count(&self) -> usize {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stationary_no_false_positives() {
        let config = BocpdConfig {
            alpha: 0.05,
            threshold_sigmas: 3.0,
            min_observations: 10,
            cooldown: 5,
        };
        let mut detector = BocpdDetector::new(1, config);

        let mut false_positives = 0;
        for i in 0..200 {
            let v = vec![1.0, 2.0, 3.0];
            if detector.observe(i as i64 * 1000, &v).is_some() {
                false_positives += 1;
            }
        }

        let fpr = false_positives as f64 / 200.0;
        assert!(
            fpr < 0.05,
            "false positive rate = {fpr:.3}, expected < 0.05"
        );
    }

    #[test]
    fn detects_change_within_window() {
        let config = BocpdConfig {
            alpha: 0.1,
            threshold_sigmas: 3.0,
            min_observations: 5,
            cooldown: 3,
        };
        let mut detector = BocpdDetector::new(1, config);

        let change_at = 50;
        let window = 10;

        let mut detected_at = None;
        for i in 0..100 {
            let v = if i < change_at {
                vec![0.0, 0.0, 0.0]
            } else {
                vec![10.0, 10.0, 10.0]
            };

            if let Some(cp) = detector.observe(i as i64 * 1000, &v) {
                if detected_at.is_none() && i >= change_at {
                    detected_at = Some(i);
                    assert_eq!(cp.method(), CpdMethod::Bocpd);
                }
            }
        }

        let det = detected_at.expect("should detect the change");
        assert!(
            det <= change_at + window,
            "detected at {det}, expected within {window} of {change_at}"
        );
    }

    #[test]
    fn changepoint_has_drift_vector() {
        let config = BocpdConfig {
            alpha: 0.1,
            threshold_sigmas: 3.0,
            min_observations: 5,
            cooldown: 3,
        };
        let mut detector = BocpdDetector::new(1, config);

        let mut cp_found = None;
        for i in 0..100 {
            let v = if i < 30 {
                vec![0.0, 0.0]
            } else {
                vec![5.0, -3.0]
            };
            if let Some(cp) = detector.observe(i as i64 * 1000, &v) {
                if cp_found.is_none() && i >= 30 {
                    cp_found = Some(cp);
                }
            }
        }

        let cp = cp_found.expect("should detect change");
        assert_eq!(cp.drift_vector().len(), 2);
        assert!(cp.severity() > 0.0);
    }

    #[test]
    fn multiple_changes() {
        let config = BocpdConfig {
            alpha: 0.1,
            threshold_sigmas: 3.0,
            min_observations: 5,
            cooldown: 10,
        };
        let mut detector = BocpdDetector::new(1, config);

        let mut change_detections = 0;
        for i in 0..300 {
            let v = if i < 100 {
                vec![0.0]
            } else if i < 200 {
                vec![10.0]
            } else {
                vec![-5.0]
            };
            if detector.observe(i as i64 * 1000, &v).is_some() {
                change_detections += 1;
            }
        }

        assert!(
            change_detections >= 2,
            "should detect at least 2 changes, got {change_detections}"
        );
    }

    #[test]
    fn count_tracks_observations() {
        let mut detector = BocpdDetector::new(1, BocpdConfig::default());
        for i in 0..10 {
            detector.observe(i * 1000, &[1.0]);
        }
        assert_eq!(detector.count(), 10);
    }
}
