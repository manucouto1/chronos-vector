//! Bayesian retrieval scoring for multi-factor candidate ranking (RFC-013 Part C).
//!
//! Replaces flat cosine scoring with a weighted composite:
//!
//! ```text
//! score = w_sim * similarity
//!       + w_recency * recency_factor
//!       + w_reward * reward
//!       + w_success * success_score
//!       + w_region * region_match
//! ```
//!
//! Weights are configurable and can be learned from online feedback
//! (logistic regression on outcome data).

use serde::{Deserialize, Serialize};

/// Scoring weights for Bayesian retrieval ranking.
///
/// Each weight controls the contribution of a factor to the final score.
/// Higher score = less relevant (distance-like). Factors are normalized
/// to [0, 1] before weighting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringWeights {
    /// Weight for semantic similarity (HNSW distance, normalized).
    pub similarity: f32,
    /// Weight for recency factor (1 - exp(-λ·age)).
    pub recency: f32,
    /// Weight for reward (1 - reward, so higher reward = lower score).
    pub reward: f32,
    /// Weight for typed-edge success score (1 - P(success)).
    pub success: f32,
    /// Weight for region match (0 if same region, 1 if different).
    pub region_match: f32,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            similarity: 1.0,
            recency: 0.0,
            reward: 0.0,
            success: 0.0,
            region_match: 0.0,
        }
    }
}

impl ScoringWeights {
    /// Create weights with all factors active at equal strength.
    pub fn balanced() -> Self {
        Self {
            similarity: 1.0,
            recency: 0.3,
            reward: 0.5,
            success: 0.4,
            region_match: 0.2,
        }
    }

    /// Create weights optimized for agent memory retrieval.
    ///
    /// Prioritizes reward and success over recency.
    pub fn agent_memory() -> Self {
        Self {
            similarity: 1.0,
            recency: 0.1,
            reward: 0.6,
            success: 0.5,
            region_match: 0.2,
        }
    }
}

/// Features for a single retrieval candidate.
#[derive(Debug, Clone)]
pub struct CandidateFeatures {
    /// Node ID in the index.
    pub node_id: u32,
    /// Raw semantic distance from HNSW (unnormalized).
    pub raw_distance: f32,
    /// Normalized semantic distance [0, 1].
    pub similarity: f32,
    /// Recency factor [0, 1] (0 = most recent, 1 = oldest).
    pub recency: f32,
    /// Reward annotation [0, 1] (NaN → 0.5 default).
    pub reward: f32,
    /// Typed-edge success score [0, 1] from Beta prior.
    pub success_score: f32,
    /// Whether candidate is in the same HNSW region as the query.
    pub region_match: bool,
}

/// Compute the Bayesian composite score for a candidate.
///
/// Lower score = more relevant (distance-like convention).
pub fn score_candidate(candidate: &CandidateFeatures, weights: &ScoringWeights) -> f32 {
    let reward_factor = if candidate.reward.is_nan() {
        0.5 // uninformative
    } else {
        1.0 - candidate.reward // higher reward → lower (better) score
    };

    let success_factor = 1.0 - candidate.success_score; // higher success → lower score
    let region_factor = if candidate.region_match { 0.0 } else { 1.0 };

    weights.similarity * candidate.similarity
        + weights.recency * candidate.recency
        + weights.reward * reward_factor
        + weights.success * success_factor
        + weights.region_match * region_factor
}

/// Re-rank a list of candidates using Bayesian scoring.
///
/// Takes pre-computed features for each candidate, scores them, and
/// returns the top-k sorted by composite score (ascending = best).
pub fn rerank(
    candidates: &[CandidateFeatures],
    weights: &ScoringWeights,
    k: usize,
) -> Vec<(u32, f32)> {
    let mut scored: Vec<(u32, f32)> = candidates
        .iter()
        .map(|c| (c.node_id, score_candidate(c, weights)))
        .collect();

    scored.sort_by(|a, b| a.1.total_cmp(&b.1));
    scored.truncate(k);
    scored
}

/// Online weight learning from outcome feedback.
///
/// Simple gradient update: if the retrieval led to success, decrease the
/// score (make it more likely to be retrieved again). If failure, increase.
///
/// Uses a learning rate to control update speed.
pub struct WeightLearner {
    /// Current weights.
    pub weights: ScoringWeights,
    /// Learning rate for gradient updates.
    pub learning_rate: f32,
    /// Number of updates applied.
    pub n_updates: usize,
}

impl WeightLearner {
    /// Create a new learner with initial weights.
    pub fn new(weights: ScoringWeights, learning_rate: f32) -> Self {
        Self {
            weights,
            learning_rate,
            n_updates: 0,
        }
    }

    /// Update weights based on outcome feedback.
    ///
    /// `candidates`: the features of candidates that were retrieved.
    /// `outcome`: 1.0 for success, 0.0 for failure.
    ///
    /// Adjusts weights to make factors that correlated with success
    /// stronger, and factors that correlated with failure weaker.
    pub fn update(&mut self, candidates: &[CandidateFeatures], outcome: f32) {
        if candidates.is_empty() {
            return;
        }

        // Direction: if success (outcome=1), we want to decrease score
        // (make these candidates rank higher). If failure, increase.
        let direction = if outcome > 0.5 { -1.0 } else { 1.0 };
        let lr = self.learning_rate / (1.0 + self.n_updates as f32 * 0.01); // Decay LR

        // Average features across candidates
        let n = candidates.len() as f32;
        let avg_sim: f32 = candidates.iter().map(|c| c.similarity).sum::<f32>() / n;
        let avg_rec: f32 = candidates.iter().map(|c| c.recency).sum::<f32>() / n;
        let avg_rew: f32 = candidates
            .iter()
            .map(|c| {
                if c.reward.is_nan() {
                    0.5
                } else {
                    1.0 - c.reward
                }
            })
            .sum::<f32>()
            / n;
        let avg_suc: f32 = candidates
            .iter()
            .map(|c| 1.0 - c.success_score)
            .sum::<f32>()
            / n;
        let avg_reg: f32 = candidates
            .iter()
            .map(|c| if c.region_match { 0.0 } else { 1.0 })
            .sum::<f32>()
            / n;

        // Gradient step on each weight
        self.weights.similarity += direction * lr * avg_sim;
        self.weights.recency += direction * lr * avg_rec;
        self.weights.reward += direction * lr * avg_rew;
        self.weights.success += direction * lr * avg_suc;
        self.weights.region_match += direction * lr * avg_reg;

        // Clamp weights to [0, 2]
        self.weights.similarity = self.weights.similarity.clamp(0.0, 2.0);
        self.weights.recency = self.weights.recency.clamp(0.0, 2.0);
        self.weights.reward = self.weights.reward.clamp(0.0, 2.0);
        self.weights.success = self.weights.success.clamp(0.0, 2.0);
        self.weights.region_match = self.weights.region_match.clamp(0.0, 2.0);

        self.n_updates += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(
        sim: f32,
        recency: f32,
        reward: f32,
        success: f32,
        region: bool,
    ) -> CandidateFeatures {
        CandidateFeatures {
            node_id: 0,
            raw_distance: sim * 2.0,
            similarity: sim,
            recency,
            reward,
            success_score: success,
            region_match: region,
        }
    }

    #[test]
    fn default_weights_pure_similarity() {
        let w = ScoringWeights::default();
        let c = make_candidate(0.3, 0.8, 0.9, 0.7, true);
        let score = score_candidate(&c, &w);
        // Only similarity matters with default weights
        assert!((score - 0.3).abs() < 0.01, "score = {score}");
    }

    #[test]
    fn reward_lowers_score() {
        let w = ScoringWeights::balanced();
        let high_reward = make_candidate(0.5, 0.5, 0.9, 0.5, true);
        let low_reward = make_candidate(0.5, 0.5, 0.1, 0.5, true);

        let s_high = score_candidate(&high_reward, &w);
        let s_low = score_candidate(&low_reward, &w);
        assert!(
            s_high < s_low,
            "high reward ({s_high}) should score lower (better) than low ({s_low})"
        );
    }

    #[test]
    fn success_score_lowers_score() {
        let w = ScoringWeights::balanced();
        let high_success = make_candidate(0.5, 0.5, 0.5, 0.9, true);
        let low_success = make_candidate(0.5, 0.5, 0.5, 0.1, true);

        let s_high = score_candidate(&high_success, &w);
        let s_low = score_candidate(&low_success, &w);
        assert!(s_high < s_low);
    }

    #[test]
    fn region_match_lowers_score() {
        let w = ScoringWeights::balanced();
        let same_region = make_candidate(0.5, 0.5, 0.5, 0.5, true);
        let diff_region = make_candidate(0.5, 0.5, 0.5, 0.5, false);

        let s_same = score_candidate(&same_region, &w);
        let s_diff = score_candidate(&diff_region, &w);
        assert!(s_same < s_diff);
    }

    #[test]
    fn nan_reward_uses_default() {
        let w = ScoringWeights::balanced();
        let nan_reward = make_candidate(0.5, 0.5, f32::NAN, 0.5, true);
        let mid_reward = make_candidate(0.5, 0.5, 0.5, 0.5, true);

        let s_nan = score_candidate(&nan_reward, &w);
        let s_mid = score_candidate(&mid_reward, &w);
        assert!(
            (s_nan - s_mid).abs() < 0.01,
            "NaN should behave like 0.5 reward"
        );
    }

    #[test]
    fn rerank_sorts_by_composite() {
        let w = ScoringWeights::agent_memory();
        let candidates = vec![
            CandidateFeatures {
                node_id: 1,
                raw_distance: 1.0,
                similarity: 0.5,
                recency: 0.5,
                reward: 0.1,
                success_score: 0.2,
                region_match: false,
            },
            CandidateFeatures {
                node_id: 2,
                raw_distance: 0.8,
                similarity: 0.4,
                recency: 0.3,
                reward: 0.9,
                success_score: 0.8,
                region_match: true,
            },
            CandidateFeatures {
                node_id: 3,
                raw_distance: 0.6,
                similarity: 0.3,
                recency: 0.8,
                reward: 0.5,
                success_score: 0.5,
                region_match: true,
            },
        ];

        let ranked = rerank(&candidates, &w, 2);
        assert_eq!(ranked.len(), 2);
        // Node 2 should rank first (best reward + success + region match)
        assert_eq!(ranked[0].0, 2);
    }

    #[test]
    fn weight_learner_success_decreases_weights() {
        let mut learner = WeightLearner::new(ScoringWeights::balanced(), 0.1);
        let initial_sim = learner.weights.similarity;

        let candidates = vec![make_candidate(0.5, 0.3, 0.8, 0.7, true)];
        learner.update(&candidates, 1.0); // success

        // Similarity weight should decrease (lower score = better for these candidates)
        assert!(learner.weights.similarity < initial_sim);
    }

    #[test]
    fn weight_learner_failure_increases_weights() {
        let mut learner = WeightLearner::new(ScoringWeights::balanced(), 0.1);
        let initial_sim = learner.weights.similarity;

        let candidates = vec![make_candidate(0.5, 0.3, 0.2, 0.3, false)];
        learner.update(&candidates, 0.0); // failure

        // Similarity weight should increase (push these candidates down)
        assert!(learner.weights.similarity > initial_sim);
    }

    #[test]
    fn weight_learner_clamps() {
        let mut learner = WeightLearner::new(ScoringWeights::default(), 10.0); // huge LR
        let candidates = vec![make_candidate(0.9, 0.9, 0.9, 0.9, true)];

        for _ in 0..100 {
            learner.update(&candidates, 0.0);
        }

        assert!(learner.weights.similarity <= 2.0);
        assert!(learner.weights.recency <= 2.0);
    }
}
