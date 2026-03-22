//! # `cvx-graph` — Knowledge Graph for ChronosVector
//!
//! Provides a typed property graph for compositional reasoning over
//! entities and their relationships. Designed for:
//!
//! - Task structure: "heat_then_place requires steps: find→take→heat→take→put"
//! - Shared sub-plans: "clean and heat share find→take prefix"
//! - Constraint validation: "after take, valid next actions are go/use/put"
//! - Cross-entity transfer: "entity A is-similar-to entity B"
//!
//! ## Graph Model
//!
//! - **Nodes**: typed entities (task, action, object, location, step)
//! - **Edges**: typed relations (requires, precedes, located_at, similar_to)
//! - **Properties**: key-value attributes on nodes and edges
//!
//! ## References
//!
//! - Hogan et al. (2021). *Knowledge Graphs*. ACM Computing Surveys.
//! - Ji et al. (2022). *A Survey on Knowledge Graphs*. Expert Systems.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod entity;
pub mod graph;
pub mod relation;

pub use entity::{Entity, EntityId, EntityType};
pub use graph::KnowledgeGraph;
pub use relation::{Relation, RelationType};
