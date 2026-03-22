//! Conditional Probability Tables (CPTs) for Bayesian network nodes.
//!
//! A CPT stores P(X | parents(X)) as a multi-dimensional table.
//! For a variable with K states and parents with N1, N2, ... states,
//! the table has K × N1 × N2 × ... entries.
//!
//! CPTs can be learned from data (counting + Laplace smoothing) or
//! specified manually.

use serde::{Deserialize, Serialize};

use crate::VariableId;

/// Conditional probability table: P(variable | parents).
///
/// Stored as a flat array with row-major layout.
/// Index: parent_config * n_states + state_index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cpt {
    /// The variable this CPT belongs to.
    pub variable_id: VariableId,
    /// Parent variable IDs (ordered).
    pub parent_ids: Vec<VariableId>,
    /// Number of states for this variable.
    pub n_states: usize,
    /// Number of states per parent (ordered same as parent_ids).
    pub parent_sizes: Vec<usize>,
    /// Probability table (flat, row-major).
    /// Length = n_states × product(parent_sizes).
    pub table: Vec<f64>,
    /// Observation counts for learning (same layout as table).
    counts: Vec<f64>,
    /// Laplace smoothing parameter (pseudo-count per cell).
    pub smoothing: f64,
}

impl Cpt {
    /// Create a CPT with uniform prior.
    ///
    /// All probabilities initialized to 1/n_states (maximum entropy).
    pub fn new(
        variable_id: VariableId,
        n_states: usize,
        parent_ids: Vec<VariableId>,
        parent_sizes: Vec<usize>,
    ) -> Self {
        let n_configs: usize = parent_sizes.iter().product::<usize>().max(1);
        let total_cells = n_states * n_configs;
        let uniform = 1.0 / n_states as f64;

        Self {
            variable_id,
            parent_ids,
            n_states,
            parent_sizes,
            table: vec![uniform; total_cells],
            counts: vec![0.0; total_cells],
            smoothing: 1.0, // Laplace smoothing
        }
    }

    /// Create a CPT with no parents (prior probability).
    pub fn prior(variable_id: VariableId, n_states: usize) -> Self {
        Self::new(variable_id, n_states, vec![], vec![])
    }

    /// Create a CPT from explicit probabilities.
    ///
    /// `probs` must have length n_states × product(parent_sizes).
    /// Each row (n_states consecutive values) must sum to ~1.0.
    pub fn from_probs(
        variable_id: VariableId,
        n_states: usize,
        parent_ids: Vec<VariableId>,
        parent_sizes: Vec<usize>,
        probs: Vec<f64>,
    ) -> Self {
        let n_configs: usize = parent_sizes.iter().product::<usize>().max(1);
        assert_eq!(
            probs.len(),
            n_states * n_configs,
            "probs length mismatch: expected {}, got {}",
            n_states * n_configs,
            probs.len()
        );

        Self {
            variable_id,
            parent_ids: parent_ids.clone(),
            n_states,
            parent_sizes: parent_sizes.clone(),
            table: probs,
            counts: vec![0.0; n_states * n_configs],
            smoothing: 1.0,
        }
    }

    /// Get P(state | parent_config).
    ///
    /// `parent_config`: index into the parent configuration space.
    /// For a single parent with 3 states, parent_config ∈ {0, 1, 2}.
    /// For two parents with 3 and 2 states, parent_config ∈ {0..5}.
    pub fn probability(&self, state: usize, parent_config: usize) -> f64 {
        let idx = parent_config * self.n_states + state;
        self.table.get(idx).copied().unwrap_or(0.0)
    }

    /// Get the probability distribution P(· | parent_config) as a slice.
    pub fn distribution(&self, parent_config: usize) -> &[f64] {
        let start = parent_config * self.n_states;
        let end = start + self.n_states;
        &self.table[start..end.min(self.table.len())]
    }

    /// Compute parent configuration index from individual parent states.
    pub fn parent_config_index(&self, parent_states: &[usize]) -> usize {
        if parent_states.is_empty() {
            return 0;
        }
        let mut idx = 0;
        let mut stride = 1;
        for i in (0..parent_states.len()).rev() {
            idx += parent_states[i] * stride;
            stride *= self.parent_sizes[i];
        }
        idx
    }

    /// Observe a data point for learning.
    ///
    /// `state`: observed state of this variable.
    /// `parent_states`: observed states of parent variables.
    pub fn observe(&mut self, state: usize, parent_states: &[usize]) {
        let config = self.parent_config_index(parent_states);
        let idx = config * self.n_states + state;
        if idx < self.counts.len() {
            self.counts[idx] += 1.0;
        }
    }

    /// Update probabilities from accumulated observations.
    ///
    /// Uses Laplace smoothing: P(x | parents) = (count + α) / (total + α·K)
    /// where α is the smoothing parameter and K is the number of states.
    pub fn update_from_counts(&mut self) {
        let n_configs: usize = self.parent_sizes.iter().product::<usize>().max(1);

        for config in 0..n_configs {
            let start = config * self.n_states;
            let total: f64 = (0..self.n_states)
                .map(|s| self.counts[start + s])
                .sum::<f64>()
                + self.smoothing * self.n_states as f64;

            for s in 0..self.n_states {
                let idx = start + s;
                self.table[idx] = (self.counts[idx] + self.smoothing) / total;
            }
        }
    }

    /// Number of parent configurations.
    pub fn n_configs(&self) -> usize {
        self.parent_sizes.iter().product::<usize>().max(1)
    }

    /// Total number of cells in the table.
    pub fn n_cells(&self) -> usize {
        self.table.len()
    }

    /// Reset observation counts (for re-learning).
    pub fn reset_counts(&mut self) {
        self.counts.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prior_uniform() {
        let cpt = Cpt::prior(0, 3);
        let dist = cpt.distribution(0);
        assert_eq!(dist.len(), 3);
        for &p in dist {
            assert!((p - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn conditional_with_one_parent() {
        // P(success | task_type) with 3 task types and 2 outcomes
        let cpt = Cpt::from_probs(
            0,
            2,       // success: {true, false}
            vec![1], // parent: task_type
            vec![3], // 3 task types
            vec![
                0.8, 0.2, // P(success | task=0): 80% success
                0.5, 0.5, // P(success | task=1): 50% success
                0.3, 0.7, // P(success | task=2): 30% success
            ],
        );

        assert_eq!(cpt.n_configs(), 3);
        assert!((cpt.probability(0, 0) - 0.8).abs() < 1e-6); // P(true | task=0)
        assert!((cpt.probability(0, 2) - 0.3).abs() < 1e-6); // P(true | task=2)
    }

    #[test]
    fn parent_config_index() {
        // Two parents: task (3 states) × region (4 states)
        let cpt = Cpt::new(0, 2, vec![1, 2], vec![3, 4]);
        assert_eq!(cpt.n_configs(), 12); // 3 × 4

        assert_eq!(cpt.parent_config_index(&[0, 0]), 0);
        assert_eq!(cpt.parent_config_index(&[0, 1]), 1);
        assert_eq!(cpt.parent_config_index(&[1, 0]), 4);
        assert_eq!(cpt.parent_config_index(&[2, 3]), 11);
    }

    #[test]
    fn learn_from_observations() {
        let mut cpt = Cpt::prior(0, 2); // binary, no parents

        // Observe 8 successes, 2 failures
        for _ in 0..8 {
            cpt.observe(0, &[]); // state 0 = success
        }
        for _ in 0..2 {
            cpt.observe(1, &[]); // state 1 = failure
        }

        cpt.update_from_counts();

        // With Laplace smoothing (α=1): P = (count+1)/(total+2)
        // P(success) = (8+1)/(10+2) = 9/12 = 0.75
        let p = cpt.probability(0, 0);
        assert!((p - 0.75).abs() < 0.01, "P(success) = {p}");
    }

    #[test]
    fn learn_conditional() {
        // P(success | action) where action has 2 types
        let mut cpt = Cpt::new(0, 2, vec![1], vec![2]);

        // action=0 (navigate): 9 success, 1 failure
        for _ in 0..9 {
            cpt.observe(0, &[0]);
        }
        cpt.observe(1, &[0]);

        // action=1 (take): 3 success, 7 failure
        for _ in 0..3 {
            cpt.observe(0, &[1]);
        }
        for _ in 0..7 {
            cpt.observe(1, &[1]);
        }

        cpt.update_from_counts();

        // P(success | navigate) = (9+1)/(10+2) = 10/12 ≈ 0.833
        let p_nav = cpt.probability(0, 0);
        assert!((p_nav - 0.833).abs() < 0.01, "P(s|nav) = {p_nav}");

        // P(success | take) = (3+1)/(10+2) = 4/12 ≈ 0.333
        let p_take = cpt.probability(0, 1);
        assert!((p_take - 0.333).abs() < 0.01, "P(s|take) = {p_take}");
    }

    #[test]
    fn serialization() {
        let mut cpt = Cpt::prior(0, 3);
        cpt.observe(0, &[]);
        cpt.observe(0, &[]);
        cpt.update_from_counts();

        let bytes = postcard::to_allocvec(&cpt).unwrap();
        let restored: Cpt = postcard::from_bytes(&bytes).unwrap();

        assert_eq!(restored.n_states, 3);
        assert!((restored.probability(0, 0) - cpt.probability(0, 0)).abs() < 1e-6);
    }
}
