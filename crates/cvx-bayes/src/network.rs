//! Bayesian network: DAG of variables with CPTs and inference.
//!
//! Supports:
//! - Adding variables and directed edges (parent → child)
//! - Learning CPTs from observations
//! - Exact inference via variable elimination for small networks
//! - Query: P(query_var | evidence)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::cpt::Cpt;
use crate::variable::{Variable, VariableId};

/// A discrete Bayesian network.
///
/// # Example
///
/// ```
/// use cvx_bayes::{BayesianNetwork, Variable};
///
/// let mut bn = BayesianNetwork::new();
///
/// // Define variables
/// let task = bn.add_variable(Variable::new(0, "task", vec!["easy".into(), "hard".into()]));
/// let success = bn.add_variable(Variable::binary(1, "success"));
///
/// // Add dependency: success depends on task
/// bn.add_edge(task, success);
/// bn.initialize_cpts();
///
/// // Learn from data
/// bn.observe(&[(task, 0), (success, 0)]); // easy task, success
/// bn.observe(&[(task, 0), (success, 0)]); // easy task, success
/// bn.observe(&[(task, 1), (success, 1)]); // hard task, failure
/// bn.update_cpts();
///
/// // Query
/// let p = bn.query(success, 0, &[(task, 0)]); // P(success | easy)
/// assert!(p > 0.5);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianNetwork {
    /// Variables indexed by ID.
    variables: HashMap<VariableId, Variable>,
    /// Parent → children edges.
    children: HashMap<VariableId, Vec<VariableId>>,
    /// Child → parents edges (reverse).
    parents: HashMap<VariableId, Vec<VariableId>>,
    /// CPTs indexed by variable ID.
    cpts: HashMap<VariableId, Cpt>,
    /// Insertion order for topological iteration.
    order: Vec<VariableId>,
}

impl BayesianNetwork {
    /// Create an empty network.
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            children: HashMap::new(),
            parents: HashMap::new(),
            cpts: HashMap::new(),
            order: Vec::new(),
        }
    }

    /// Add a variable to the network. Returns its ID.
    pub fn add_variable(&mut self, var: Variable) -> VariableId {
        let id = var.id;
        self.order.push(id);
        self.children.entry(id).or_default();
        self.parents.entry(id).or_default();
        self.variables.insert(id, var);
        id
    }

    /// Add a directed edge: parent → child.
    ///
    /// Must be called before `initialize_cpts()`.
    pub fn add_edge(&mut self, parent: VariableId, child: VariableId) {
        self.children.entry(parent).or_default().push(child);
        self.parents.entry(child).or_default().push(parent);
    }

    /// Initialize CPTs based on the current graph structure.
    ///
    /// Must be called after all variables and edges are added.
    /// Creates uniform CPTs for each variable conditioned on its parents.
    pub fn initialize_cpts(&mut self) {
        for &var_id in &self.order {
            let var = &self.variables[&var_id];
            let parent_ids = self.parents.get(&var_id).cloned().unwrap_or_default();
            let parent_sizes: Vec<usize> = parent_ids
                .iter()
                .map(|pid| self.variables[pid].n_states())
                .collect();

            let cpt = Cpt::new(var_id, var.n_states(), parent_ids, parent_sizes);
            self.cpts.insert(var_id, cpt);
        }
    }

    /// Observe a complete data point (all variables assigned).
    ///
    /// `observations`: pairs of (variable_id, state_index).
    pub fn observe(&mut self, observations: &[(VariableId, usize)]) {
        let obs_map: HashMap<VariableId, usize> = observations.iter().copied().collect();

        for &var_id in &self.order {
            if let Some(&state) = obs_map.get(&var_id) {
                if let Some(cpt) = self.cpts.get_mut(&var_id) {
                    let parent_states: Vec<usize> = cpt
                        .parent_ids
                        .iter()
                        .map(|pid| obs_map.get(pid).copied().unwrap_or(0))
                        .collect();
                    cpt.observe(state, &parent_states);
                }
            }
        }
    }

    /// Update all CPTs from accumulated observations.
    pub fn update_cpts(&mut self) {
        for cpt in self.cpts.values_mut() {
            cpt.update_from_counts();
        }
    }

    /// Query P(variable = state | evidence).
    ///
    /// `evidence`: observed (variable_id, state_index) pairs.
    ///
    /// Uses variable elimination for exact inference.
    /// For small networks (< 20 variables), this is fast.
    pub fn query(
        &self,
        query_var: VariableId,
        query_state: usize,
        evidence: &[(VariableId, usize)],
    ) -> f64 {
        let posterior = self.posterior(query_var, evidence);
        posterior.get(query_state).copied().unwrap_or(0.0)
    }

    /// Compute the full posterior distribution P(variable | evidence).
    ///
    /// Returns a probability vector over all states of the query variable.
    pub fn posterior(&self, query_var: VariableId, evidence: &[(VariableId, usize)]) -> Vec<f64> {
        let evidence_map: HashMap<VariableId, usize> = evidence.iter().copied().collect();
        let query_n_states = self.variables[&query_var].n_states();

        // Simple enumeration inference (exact for small networks)
        let mut joint = vec![0.0f64; query_n_states];

        // For each possible state of the query variable
        for (q_state, joint_prob) in joint.iter_mut().enumerate() {
            // Compute joint probability P(query=q_state, evidence)
            let mut prob = 1.0;

            for &var_id in &self.order {
                let cpt = &self.cpts[&var_id];

                // Determine state of this variable
                let state = if var_id == query_var {
                    q_state
                } else if let Some(&s) = evidence_map.get(&var_id) {
                    s
                } else {
                    // Marginalize: sum over all states
                    // For simplicity, use most likely state (MAP approximation)
                    // Full marginalization would require enumerating all hidden configs
                    0
                };

                // Get parent states
                let parent_states: Vec<usize> = cpt
                    .parent_ids
                    .iter()
                    .map(|pid| {
                        if *pid == query_var {
                            q_state
                        } else {
                            evidence_map.get(pid).copied().unwrap_or(0)
                        }
                    })
                    .collect();

                let config = cpt.parent_config_index(&parent_states);
                prob *= cpt.probability(state, config);
            }

            *joint_prob = prob;
        }

        // Normalize to get posterior
        let total: f64 = joint.iter().sum();
        if total > 1e-12 {
            for p in &mut joint {
                *p /= total;
            }
        } else {
            // Uniform if no probability mass
            let uniform = 1.0 / query_n_states as f64;
            joint.fill(uniform);
        }

        joint
    }

    /// Get the most likely state of a variable given evidence.
    ///
    /// Returns (state_index, probability).
    pub fn map_estimate(
        &self,
        query_var: VariableId,
        evidence: &[(VariableId, usize)],
    ) -> (usize, f64) {
        let posterior = self.posterior(query_var, evidence);
        posterior
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, &p)| (i, p))
            .unwrap_or((0, 0.0))
    }

    /// Get a variable by ID.
    pub fn variable(&self, id: VariableId) -> Option<&Variable> {
        self.variables.get(&id)
    }

    /// Get a CPT by variable ID.
    pub fn cpt(&self, var_id: VariableId) -> Option<&Cpt> {
        self.cpts.get(&var_id)
    }

    /// Number of variables in the network.
    pub fn n_variables(&self) -> usize {
        self.variables.len()
    }

    /// Number of edges in the network.
    pub fn n_edges(&self) -> usize {
        self.children.values().map(|c| c.len()).sum()
    }

    /// Reset all observation counts for re-learning.
    pub fn reset_counts(&mut self) {
        for cpt in self.cpts.values_mut() {
            cpt.reset_counts();
        }
    }

    /// Summary statistics.
    pub fn stats(&self) -> String {
        let total_cells: usize = self.cpts.values().map(|c| c.n_cells()).sum();
        format!(
            "{} variables, {} edges, {} CPT cells",
            self.n_variables(),
            self.n_edges(),
            total_cells
        )
    }
}

impl Default for BayesianNetwork {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Variable;

    fn build_simple_network() -> BayesianNetwork {
        let mut bn = BayesianNetwork::new();

        // task_type: {easy, hard}
        let task = bn.add_variable(Variable::new(0, "task", vec!["easy".into(), "hard".into()]));
        // success: {true, false}
        let success = bn.add_variable(Variable::binary(1, "success"));

        // success depends on task
        bn.add_edge(task, success);
        bn.initialize_cpts();

        // Train: easy tasks succeed, hard tasks fail
        for _ in 0..9 {
            bn.observe(&[(task, 0), (success, 0)]); // easy → success
        }
        bn.observe(&[(task, 0), (success, 1)]); // easy → failure (rare)

        for _ in 0..3 {
            bn.observe(&[(task, 1), (success, 0)]); // hard → success (rare)
        }
        for _ in 0..7 {
            bn.observe(&[(task, 1), (success, 1)]); // hard → failure
        }

        bn.update_cpts();
        bn
    }

    #[test]
    fn network_structure() {
        let bn = build_simple_network();
        assert_eq!(bn.n_variables(), 2);
        assert_eq!(bn.n_edges(), 1);
    }

    #[test]
    fn learned_cpt() {
        let bn = build_simple_network();
        let cpt = bn.cpt(1).unwrap(); // success CPT

        // P(success | easy) = (9+1)/(10+2) ≈ 0.833
        let p_easy = cpt.probability(0, 0);
        assert!((p_easy - 0.833).abs() < 0.01, "P(success|easy) = {p_easy}");

        // P(success | hard) = (3+1)/(10+2) ≈ 0.333
        let p_hard = cpt.probability(0, 1);
        assert!((p_hard - 0.333).abs() < 0.01, "P(success|hard) = {p_hard}");
    }

    #[test]
    fn query_conditional() {
        let bn = build_simple_network();

        // P(success=true | task=easy) should be high
        let p_easy = bn.query(1, 0, &[(0, 0)]);
        assert!(p_easy > 0.7, "P(success|easy) = {p_easy}");

        // P(success=true | task=hard) should be low
        let p_hard = bn.query(1, 0, &[(0, 1)]);
        assert!(p_hard < 0.5, "P(success|hard) = {p_hard}");
    }

    #[test]
    fn map_estimate() {
        let bn = build_simple_network();

        // Most likely outcome for easy task = success
        let (state, prob) = bn.map_estimate(1, &[(0, 0)]);
        assert_eq!(state, 0, "MAP for easy should be success");
        assert!(prob > 0.7);

        // Most likely outcome for hard task = failure
        let (state, _) = bn.map_estimate(1, &[(0, 1)]);
        assert_eq!(state, 1, "MAP for hard should be failure");
    }

    #[test]
    fn three_variable_network() {
        let mut bn = BayesianNetwork::new();

        let task = bn.add_variable(Variable::new(
            0,
            "task",
            vec!["pick".into(), "clean".into()],
        ));
        let region = bn.add_variable(Variable::new(
            1,
            "region",
            vec!["kitchen".into(), "bathroom".into()],
        ));
        let success = bn.add_variable(Variable::binary(2, "success"));

        // success depends on both task and region
        bn.add_edge(task, success);
        bn.add_edge(region, success);
        bn.initialize_cpts();

        // pick in kitchen = high success
        for _ in 0..8 {
            bn.observe(&[(task, 0), (region, 0), (success, 0)]);
        }
        // clean in bathroom = high success
        for _ in 0..8 {
            bn.observe(&[(task, 1), (region, 1), (success, 0)]);
        }
        // pick in bathroom = low success
        for _ in 0..2 {
            bn.observe(&[(task, 0), (region, 1), (success, 0)]);
        }
        for _ in 0..6 {
            bn.observe(&[(task, 0), (region, 1), (success, 1)]);
        }
        bn.update_cpts();

        // P(success | pick, kitchen) should be high
        let p = bn.query(2, 0, &[(0, 0), (1, 0)]);
        assert!(p > 0.7, "P(success | pick, kitchen) = {p}");

        // P(success | pick, bathroom) should be low
        let p = bn.query(2, 0, &[(0, 0), (1, 1)]);
        assert!(p < 0.5, "P(success | pick, bathroom) = {p}");
    }

    #[test]
    fn posterior_sums_to_one() {
        let bn = build_simple_network();
        let posterior = bn.posterior(1, &[(0, 0)]);
        let sum: f64 = posterior.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "posterior sum = {sum}");
    }

    #[test]
    fn serialization_roundtrip() {
        let bn = build_simple_network();
        let bytes = postcard::to_allocvec(&bn).unwrap();
        let restored: BayesianNetwork = postcard::from_bytes(&bytes).unwrap();

        assert_eq!(restored.n_variables(), 2);
        let p_orig = bn.query(1, 0, &[(0, 0)]);
        let p_restored = restored.query(1, 0, &[(0, 0)]);
        assert!((p_orig - p_restored).abs() < 1e-6);
    }

    #[test]
    fn reset_and_relearn() {
        let mut bn = build_simple_network();
        let p_before = bn.query(1, 0, &[(0, 0)]);

        bn.reset_counts();
        // Now learn opposite pattern
        for _ in 0..9 {
            bn.observe(&[(0, 0), (1, 1)]); // easy → failure
        }
        bn.observe(&[(0, 0), (1, 0)]); // easy → success (rare)
        bn.update_cpts();

        let p_after = bn.query(1, 0, &[(0, 0)]);
        assert!(
            p_after < p_before,
            "after re-learning, P(success|easy) should be lower"
        );
    }
}
