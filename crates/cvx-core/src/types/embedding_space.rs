//! Embedding space metadata.

use serde::{Deserialize, Serialize};

/// Metadata describing an embedding space.
///
/// Each space has its own dimensionality, distance metric, and temporal
/// frequency. Entities can have vectors in multiple spaces simultaneously.
///
/// # Example
///
/// ```
/// use cvx_core::types::EmbeddingSpace;
///
/// let space = EmbeddingSpace::new(0, "bert-base".into(), 768)
///     .with_metric("cosine".into())
///     .with_frequency("daily".into());
/// assert_eq!(space.dim(), 768);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingSpace {
    space_id: u32,
    name: String,
    dimensionality: usize,
    #[serde(default = "default_metric")]
    metric: String,
    #[serde(default)]
    normalization: Option<String>,
    #[serde(default)]
    frequency: Option<String>,
}

fn default_metric() -> String {
    "cosine".into()
}

impl EmbeddingSpace {
    /// Create a new embedding space with required fields.
    pub fn new(space_id: u32, name: String, dimensionality: usize) -> Self {
        Self {
            space_id,
            name,
            dimensionality,
            metric: default_metric(),
            normalization: None,
            frequency: None,
        }
    }

    /// Set the distance metric.
    pub fn with_metric(mut self, metric: String) -> Self {
        self.metric = metric;
        self
    }

    /// Set the normalization method.
    pub fn with_normalization(mut self, normalization: String) -> Self {
        self.normalization = Some(normalization);
        self
    }

    /// Set the temporal frequency.
    pub fn with_frequency(mut self, frequency: String) -> Self {
        self.frequency = Some(frequency);
        self
    }

    /// Space identifier.
    pub fn space_id(&self) -> u32 {
        self.space_id
    }

    /// Human-readable name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Vector dimensionality.
    pub fn dim(&self) -> usize {
        self.dimensionality
    }

    /// Distance metric name.
    pub fn metric(&self) -> &str {
        &self.metric
    }

    /// Normalization method, if set.
    pub fn normalization(&self) -> Option<&str> {
        self.normalization.as_deref()
    }

    /// Temporal frequency, if set.
    pub fn frequency(&self) -> Option<&str> {
        self.frequency.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creation_and_accessors() {
        let space = EmbeddingSpace::new(0, "test".into(), 128)
            .with_metric("l2".into())
            .with_normalization("l2_norm".into())
            .with_frequency("hourly".into());

        assert_eq!(space.space_id(), 0);
        assert_eq!(space.name(), "test");
        assert_eq!(space.dim(), 128);
        assert_eq!(space.metric(), "l2");
        assert_eq!(space.normalization(), Some("l2_norm"));
        assert_eq!(space.frequency(), Some("hourly"));
    }

    #[test]
    fn defaults() {
        let space = EmbeddingSpace::new(1, "bert".into(), 768);
        assert_eq!(space.metric(), "cosine");
        assert_eq!(space.normalization(), None);
        assert_eq!(space.frequency(), None);
    }

    #[test]
    fn serde_roundtrip() {
        let space = EmbeddingSpace::new(0, "test".into(), 128).with_frequency("daily".into());
        let json = serde_json::to_string(&space).unwrap();
        let recovered: EmbeddingSpace = serde_json::from_str(&json).unwrap();
        assert_eq!(space, recovered);
    }
}
