//! Metadata predicates for filtering search results.
//!
//! Composable with `TemporalFilter` for combined temporal + metadata queries.
//! Metadata is stored as `HashMap<String, String>` on `TemporalPoint`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single predicate on a metadata field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetadataPredicate {
    /// Exact string match: `field == value`.
    Equals(String),
    /// Numeric greater-than-or-equal: `field.parse::<f64>() >= value`.
    Gte(f64),
    /// Numeric less-than-or-equal: `field.parse::<f64>() <= value`.
    Lte(f64),
    /// String contains substring.
    Contains(String),
    /// Field exists (any value).
    Exists,
}

impl MetadataPredicate {
    /// Test whether a metadata value satisfies this predicate.
    pub fn matches(&self, value: Option<&String>) -> bool {
        match self {
            MetadataPredicate::Exists => value.is_some(),
            MetadataPredicate::Equals(expected) => value.is_some_and(|v| v == expected),
            MetadataPredicate::Contains(substr) => value.is_some_and(|v| v.contains(substr.as_str())),
            MetadataPredicate::Gte(threshold) => {
                value.and_then(|v| v.parse::<f64>().ok()).is_some_and(|n| n >= *threshold)
            }
            MetadataPredicate::Lte(threshold) => {
                value.and_then(|v| v.parse::<f64>().ok()).is_some_and(|n| n <= *threshold)
            }
        }
    }
}

/// A set of metadata predicates (all must match — AND semantics).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetadataFilter {
    /// Field name → predicate. All predicates must match.
    pub predicates: HashMap<String, MetadataPredicate>,
}

impl MetadataFilter {
    /// Create an empty filter (matches everything).
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an equals predicate.
    pub fn equals(mut self, field: impl Into<String>, value: impl Into<String>) -> Self {
        self.predicates.insert(field.into(), MetadataPredicate::Equals(value.into()));
        self
    }

    /// Add a >= predicate.
    pub fn gte(mut self, field: impl Into<String>, value: f64) -> Self {
        self.predicates.insert(field.into(), MetadataPredicate::Gte(value));
        self
    }

    /// Add a <= predicate.
    pub fn lte(mut self, field: impl Into<String>, value: f64) -> Self {
        self.predicates.insert(field.into(), MetadataPredicate::Lte(value));
        self
    }

    /// Add a contains predicate.
    pub fn contains(mut self, field: impl Into<String>, substr: impl Into<String>) -> Self {
        self.predicates.insert(field.into(), MetadataPredicate::Contains(substr.into()));
        self
    }

    /// Add an exists predicate.
    pub fn exists(mut self, field: impl Into<String>) -> Self {
        self.predicates.insert(field.into(), MetadataPredicate::Exists);
        self
    }

    /// Test whether a metadata map satisfies ALL predicates.
    pub fn matches(&self, metadata: &HashMap<String, String>) -> bool {
        self.predicates.iter().all(|(field, predicate)| {
            predicate.matches(metadata.get(field))
        })
    }

    /// Whether this filter has any predicates.
    pub fn is_empty(&self) -> bool {
        self.predicates.is_empty()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn empty_filter_matches_everything() {
        let f = MetadataFilter::new();
        assert!(f.matches(&HashMap::new()));
        assert!(f.matches(&meta(&[("a", "b")])));
    }

    #[test]
    fn equals_match() {
        let f = MetadataFilter::new().equals("step_index", "0");
        assert!(f.matches(&meta(&[("step_index", "0")])));
        assert!(!f.matches(&meta(&[("step_index", "1")])));
        assert!(!f.matches(&HashMap::new()));
    }

    #[test]
    fn gte_match() {
        let f = MetadataFilter::new().gte("reward", 0.5);
        assert!(f.matches(&meta(&[("reward", "0.7")])));
        assert!(f.matches(&meta(&[("reward", "0.5")])));
        assert!(!f.matches(&meta(&[("reward", "0.3")])));
        assert!(!f.matches(&HashMap::new()));
    }

    #[test]
    fn lte_match() {
        let f = MetadataFilter::new().lte("step_index", 3.0);
        assert!(f.matches(&meta(&[("step_index", "0")])));
        assert!(f.matches(&meta(&[("step_index", "3")])));
        assert!(!f.matches(&meta(&[("step_index", "5")])));
    }

    #[test]
    fn contains_match() {
        let f = MetadataFilter::new().contains("action", "pick");
        assert!(f.matches(&meta(&[("action", "pick_up_apple")])));
        assert!(!f.matches(&meta(&[("action", "navigate_to")])));
    }

    #[test]
    fn exists_match() {
        let f = MetadataFilter::new().exists("episode_id");
        assert!(f.matches(&meta(&[("episode_id", "ep_42")])));
        assert!(!f.matches(&HashMap::new()));
    }

    #[test]
    fn multiple_predicates_and_semantics() {
        let f = MetadataFilter::new()
            .equals("step_index", "0")
            .gte("reward", 0.5);

        assert!(f.matches(&meta(&[("step_index", "0"), ("reward", "0.8")])));
        assert!(!f.matches(&meta(&[("step_index", "0"), ("reward", "0.2")])));
        assert!(!f.matches(&meta(&[("step_index", "3"), ("reward", "0.8")])));
    }

    #[test]
    fn non_numeric_gte_fails_gracefully() {
        let f = MetadataFilter::new().gte("reward", 0.5);
        assert!(!f.matches(&meta(&[("reward", "not_a_number")])));
    }
}
