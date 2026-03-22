//! Typed entities (nodes) in the knowledge graph.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique entity identifier.
pub type EntityId = u64;

/// Entity type label.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityType {
    /// A task type (e.g., "heat_then_place").
    Task,
    /// An abstract action (e.g., "navigate", "take").
    Action,
    /// A step in a plan (ordered).
    Step,
    /// A physical object (e.g., "tomato", "mug").
    Object,
    /// A location (e.g., "countertop 1", "fridge").
    Location,
    /// A tool or appliance (e.g., "microwave", "sinkbasin").
    Appliance,
    /// Custom type.
    Custom(String),
}

/// An entity (node) in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Unique identifier.
    pub id: EntityId,
    /// Entity type.
    pub entity_type: EntityType,
    /// Name / label.
    pub name: String,
    /// Arbitrary key-value properties.
    pub properties: HashMap<String, String>,
}

impl Entity {
    /// Create a new entity.
    pub fn new(id: EntityId, entity_type: EntityType, name: impl Into<String>) -> Self {
        Self {
            id,
            entity_type,
            name: name.into(),
            properties: HashMap::new(),
        }
    }

    /// Add a property.
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Get a property value.
    pub fn property(&self, key: &str) -> Option<&str> {
        self.properties.get(key).map(|s| s.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_creation() {
        let e = Entity::new(1, EntityType::Task, "heat_then_place")
            .with_property("n_steps", "7")
            .with_property("difficulty", "hard");

        assert_eq!(e.id, 1);
        assert_eq!(e.entity_type, EntityType::Task);
        assert_eq!(e.name, "heat_then_place");
        assert_eq!(e.property("n_steps"), Some("7"));
        assert_eq!(e.property("missing"), None);
    }
}
