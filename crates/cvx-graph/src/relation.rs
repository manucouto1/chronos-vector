//! Typed relations (edges) in the knowledge graph.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::EntityId;

/// Relation type label.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationType {
    /// Step A must precede step B in a plan.
    Precedes,
    /// Task requires this action/step.
    Requires,
    /// Object is located at a location.
    LocatedAt,
    /// Action uses an appliance.
    Uses,
    /// Entity A is similar to entity B.
    SimilarTo,
    /// Entity A is a sub-type of entity B.
    IsA,
    /// Task produces/achieves a state.
    Produces,
    /// Custom relation.
    Custom(String),
}

/// A directed relation (edge) between two entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    /// Source entity.
    pub source: EntityId,
    /// Target entity.
    pub target: EntityId,
    /// Relation type.
    pub relation_type: RelationType,
    /// Weight / confidence (0.0 - 1.0).
    pub weight: f32,
    /// Arbitrary properties.
    pub properties: HashMap<String, String>,
}

impl Relation {
    /// Create a new relation.
    pub fn new(
        source: EntityId,
        target: EntityId,
        relation_type: RelationType,
        weight: f32,
    ) -> Self {
        Self {
            source,
            target,
            relation_type,
            weight,
            properties: HashMap::new(),
        }
    }

    /// Add a property.
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relation_creation() {
        let r = Relation::new(1, 2, RelationType::Precedes, 1.0).with_property("order", "1");

        assert_eq!(r.source, 1);
        assert_eq!(r.target, 2);
        assert_eq!(r.relation_type, RelationType::Precedes);
    }
}
