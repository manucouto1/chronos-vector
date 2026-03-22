//! Region MDP: Markov Decision Process over HNSW semantic regions (RFC-013 Part A).
//!
//! HNSW regions define a discrete state space. Episode trajectories define
//! transitions. Rewards define outcomes. This module learns P(success | region,
//! action_type) from observed trajectories.
//!
//! States are HNSW hub node IDs at a given level. Actions are abstract types
//! represented as string labels (e.g., "navigate", "take", "place").

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A transition model over HNSW regions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RegionMdp {
    /// (region, action_type) → {next_region: count}.
    transitions: HashMap<(u32, String), HashMap<u32, u32>>,
    /// (region, action_type) → [rewards].
    rewards: HashMap<(u32, String), Vec<f32>>,
    /// region → [episode_rewards].
    region_quality: HashMap<u32, Vec<f32>>,
    /// Total transitions learned.
    n_transitions: usize,
}

impl RegionMdp {
    /// Create an empty MDP.
    pub fn new() -> Self {
        Self::default()
    }

    /// Learn from a single episode trajectory.
    ///
    /// `regions`: sequence of region IDs the episode traversed.
    /// `actions`: corresponding action types (same length as regions).
    /// `reward`: episode outcome (1.0 = success, 0.0 = failure).
    pub fn learn_trajectory(&mut self, regions: &[u32], actions: &[String], reward: f32) {
        let n = regions.len().min(actions.len());
        for i in 0..n.saturating_sub(1) {
            let s = regions[i];
            let a = actions[i].clone();
            let s_next = regions[i + 1];

            *self
                .transitions
                .entry((s, a.clone()))
                .or_default()
                .entry(s_next)
                .or_default() += 1;

            self.rewards.entry((s, a)).or_default().push(reward);
            self.n_transitions += 1;
        }

        // Track per-region quality
        for &s in regions {
            self.region_quality.entry(s).or_default().push(reward);
        }
    }

    /// P(success | region, action_type) using Beta prior.
    ///
    /// Returns (1 + n_successes) / (2 + n_total) for Bayesian smoothing.
    /// Falls back to region-level quality if no data for this (region, action).
    pub fn action_success_rate(&self, region: u32, action_type: &str) -> f32 {
        let key = (region, action_type.to_string());
        if let Some(rewards) = self.rewards.get(&key) {
            if !rewards.is_empty() {
                let successes: f32 = rewards.iter().filter(|&&r| r > 0.5).count() as f32;
                return (1.0 + successes) / (2.0 + rewards.len() as f32);
            }
        }
        // Fallback: region-level quality
        self.region_quality_score(region)
    }

    /// Overall quality of a region (mean reward of episodes passing through).
    pub fn region_quality_score(&self, region: u32) -> f32 {
        self.region_quality
            .get(&region)
            .filter(|r| !r.is_empty())
            .map(|r| r.iter().sum::<f32>() / r.len() as f32)
            .unwrap_or(0.5)
    }

    /// Rank action types by success rate in a given region.
    ///
    /// Returns `(action_type, success_rate)` sorted descending.
    pub fn best_actions(&self, region: u32) -> Vec<(String, f32)> {
        let mut scores: HashMap<String, f32> = HashMap::new();
        for (key, rewards) in &self.rewards {
            if key.0 == region && !rewards.is_empty() {
                let successes = rewards.iter().filter(|&&r| r > 0.5).count() as f32;
                let rate = (1.0 + successes) / (2.0 + rewards.len() as f32);
                scores.insert(key.1.clone(), rate);
            }
        }
        let mut sorted: Vec<_> = scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
        sorted
    }

    /// P(s' | s, a) — transition probability.
    pub fn transition_probability(&self, region: u32, action: &str, next_region: u32) -> f32 {
        let key = (region, action.to_string());
        let counts = match self.transitions.get(&key) {
            Some(c) => c,
            None => return 0.0,
        };
        let total: u32 = counts.values().sum();
        if total == 0 {
            return 0.0;
        }
        *counts.get(&next_region).unwrap_or(&0) as f32 / total as f32
    }

    /// Context-aware decay factor based on region quality.
    ///
    /// High-quality region → small decay (0.95).
    /// Low-quality region → large decay (0.70).
    pub fn decay_factor(&self, region: u32) -> f32 {
        let q = self.region_quality_score(region);
        0.70 + 0.25 * q
    }

    /// Format action hints for a region as a string.
    pub fn format_hints(&self, region: u32, top_n: usize) -> String {
        let best = self.best_actions(region);
        if best.is_empty() {
            return String::new();
        }
        let hints: Vec<String> = best
            .iter()
            .take(top_n)
            .map(|(a, s)| format!("{a}({:.0}%)", s * 100.0))
            .collect();
        format!("Action success rates: {}", hints.join(", "))
    }

    /// Total transitions learned.
    pub fn n_transitions(&self) -> usize {
        self.n_transitions
    }

    /// Number of distinct (region, action) pairs observed.
    pub fn n_state_actions(&self) -> usize {
        self.rewards.len()
    }

    /// Number of distinct regions observed.
    pub fn n_regions(&self) -> usize {
        self.region_quality.len()
    }

    /// Summary statistics.
    pub fn stats(&self) -> String {
        format!(
            "{} transitions, {} state-action pairs, {} regions",
            self.n_transitions,
            self.n_state_actions(),
            self.n_regions()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learn_and_query() {
        let mut mdp = RegionMdp::new();
        // Episode 1: regions [0, 1, 2], actions [navigate, take, place], success
        mdp.learn_trajectory(
            &[0, 1, 2],
            &["navigate".into(), "take".into(), "place".into()],
            1.0,
        );
        // Episode 2: same regions, different actions, failure
        mdp.learn_trajectory(
            &[0, 1, 3],
            &["navigate".into(), "open".into(), "take".into()],
            0.0,
        );

        assert_eq!(mdp.n_transitions(), 4);

        // navigate from region 0: 1 success + 1 failure → P = (1+1)/(2+2) = 0.5
        let rate = mdp.action_success_rate(0, "navigate");
        assert!((rate - 0.5).abs() < 0.01, "rate = {rate}");

        // take from region 1: 1 success → P = (1+1)/(2+1) = 0.67
        let rate = mdp.action_success_rate(1, "take");
        assert!((rate - 0.667).abs() < 0.02, "rate = {rate}");
    }

    #[test]
    fn best_actions_sorted() {
        let mut mdp = RegionMdp::new();
        // 3 successes with navigate, 1 failure with open
        for _ in 0..3 {
            mdp.learn_trajectory(&[0, 1], &["navigate".into(), "done".into()], 1.0);
        }
        mdp.learn_trajectory(&[0, 2], &["open".into(), "fail".into()], 0.0);

        let best = mdp.best_actions(0);
        assert!(!best.is_empty());
        assert_eq!(best[0].0, "navigate");
        assert!(best[0].1 > best.last().unwrap().1);
    }

    #[test]
    fn transition_probability() {
        let mut mdp = RegionMdp::new();
        mdp.learn_trajectory(&[0, 1], &["go".into(), "x".into()], 1.0);
        mdp.learn_trajectory(&[0, 1], &["go".into(), "x".into()], 1.0);
        mdp.learn_trajectory(&[0, 2], &["go".into(), "x".into()], 0.0);

        let p1 = mdp.transition_probability(0, "go", 1);
        let p2 = mdp.transition_probability(0, "go", 2);
        assert!((p1 - 0.667).abs() < 0.02, "p1 = {p1}");
        assert!((p2 - 0.333).abs() < 0.02, "p2 = {p2}");
        assert!((p1 + p2 - 1.0).abs() < 0.01);
    }

    #[test]
    fn region_quality_and_decay() {
        let mut mdp = RegionMdp::new();
        // Region 0: 3 successes, 1 failure → quality ~0.75
        for _ in 0..3 {
            mdp.learn_trajectory(&[0, 1], &["a".into(), "b".into()], 1.0);
        }
        mdp.learn_trajectory(&[0, 1], &["a".into(), "b".into()], 0.0);

        let q = mdp.region_quality_score(0);
        assert!((q - 0.75).abs() < 0.01, "quality = {q}");

        let d = mdp.decay_factor(0);
        // 0.70 + 0.25 * 0.75 = 0.8875
        assert!((d - 0.8875).abs() < 0.01, "decay = {d}");
    }

    #[test]
    fn unknown_region_defaults() {
        let mdp = RegionMdp::new();
        assert!((mdp.action_success_rate(99, "x") - 0.5).abs() < 0.01);
        assert!((mdp.region_quality_score(99) - 0.5).abs() < 0.01);
        assert!((mdp.decay_factor(99) - 0.825).abs() < 0.01);
    }

    #[test]
    fn format_hints() {
        let mut mdp = RegionMdp::new();
        for _ in 0..5 {
            mdp.learn_trajectory(&[0, 1], &["navigate".into(), "done".into()], 1.0);
        }
        mdp.learn_trajectory(&[0, 2], &["open".into(), "fail".into()], 0.0);

        let hints = mdp.format_hints(0, 3);
        assert!(hints.contains("navigate"), "hints = {hints}");
        assert!(hints.contains("%"));
    }

    #[test]
    fn serialization_roundtrip() {
        let mut mdp = RegionMdp::new();
        mdp.learn_trajectory(&[0, 1, 2], &["a".into(), "b".into(), "c".into()], 1.0);

        let bytes = postcard::to_allocvec(&mdp).unwrap();
        let restored: RegionMdp = postcard::from_bytes(&bytes).unwrap();

        assert_eq!(restored.n_transitions(), 2);
        assert!(
            (restored.action_success_rate(0, "a") - mdp.action_success_rate(0, "a")).abs() < 0.01
        );
    }
}
