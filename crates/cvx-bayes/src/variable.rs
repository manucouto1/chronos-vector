//! Discrete random variables for Bayesian network nodes.

use serde::{Deserialize, Serialize};

/// Unique identifier for a variable in the network.
pub type VariableId = u32;

/// A discrete random variable with named states.
///
/// # Example
///
/// ```
/// use cvx_bayes::Variable;
///
/// let task = Variable::new(0, "task_type", vec![
///     "pick_and_place".into(),
///     "heat_then_place".into(),
///     "clean_then_place".into(),
/// ]);
/// assert_eq!(task.n_states(), 3);
/// assert_eq!(task.state_index("heat_then_place"), Some(1));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variable {
    /// Unique ID in the network.
    pub id: VariableId,
    /// Human-readable name.
    pub name: String,
    /// Named states (e.g., ["success", "failure"]).
    pub states: Vec<String>,
}

impl Variable {
    /// Create a new variable with named states.
    pub fn new(id: VariableId, name: impl Into<String>, states: Vec<String>) -> Self {
        assert!(!states.is_empty(), "variable must have at least one state");
        Self {
            id,
            name: name.into(),
            states,
        }
    }

    /// Create a binary variable (two states: "true", "false").
    pub fn binary(id: VariableId, name: impl Into<String>) -> Self {
        Self::new(id, name, vec!["true".into(), "false".into()])
    }

    /// Number of states.
    pub fn n_states(&self) -> usize {
        self.states.len()
    }

    /// Get state index by name.
    pub fn state_index(&self, state_name: &str) -> Option<usize> {
        self.states.iter().position(|s| s == state_name)
    }

    /// Get state name by index.
    pub fn state_name(&self, index: usize) -> Option<&str> {
        self.states.get(index).map(|s| s.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variable_creation() {
        let v = Variable::new(
            0,
            "color",
            vec!["red".into(), "blue".into(), "green".into()],
        );
        assert_eq!(v.n_states(), 3);
        assert_eq!(v.state_index("blue"), Some(1));
        assert_eq!(v.state_name(2), Some("green"));
        assert_eq!(v.state_index("yellow"), None);
    }

    #[test]
    fn binary_variable() {
        let v = Variable::binary(0, "success");
        assert_eq!(v.n_states(), 2);
        assert_eq!(v.state_index("true"), Some(0));
        assert_eq!(v.state_index("false"), Some(1));
    }

    #[test]
    #[should_panic]
    fn empty_states_panics() {
        Variable::new(0, "empty", vec![]);
    }
}
