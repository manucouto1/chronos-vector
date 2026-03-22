//! Knowledge graph: typed property graph with traversal and query.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::entity::{Entity, EntityId, EntityType};
use crate::relation::{Relation, RelationType};

/// A knowledge graph with typed entities and relations.
///
/// # Example
///
/// ```
/// use cvx_graph::{KnowledgeGraph, Entity, EntityType, Relation, RelationType};
///
/// let mut kg = KnowledgeGraph::new();
///
/// // Define a task plan
/// let task = kg.add_entity(Entity::new(1, EntityType::Task, "heat_then_place"));
/// let find = kg.add_entity(Entity::new(2, EntityType::Action, "find"));
/// let take = kg.add_entity(Entity::new(3, EntityType::Action, "take"));
/// let heat = kg.add_entity(Entity::new(4, EntityType::Action, "heat"));
///
/// kg.add_relation(Relation::new(task, find, RelationType::Requires, 1.0));
/// kg.add_relation(Relation::new(find, take, RelationType::Precedes, 1.0));
/// kg.add_relation(Relation::new(take, heat, RelationType::Precedes, 1.0));
///
/// // Query: what steps does heat_then_place require?
/// let steps = kg.neighbors(task, Some(RelationType::Requires));
/// assert_eq!(steps.len(), 1);
///
/// // Multi-hop: what comes after find?
/// let chain = kg.traverse(find, &[RelationType::Precedes], 3);
/// assert!(chain.len() >= 2); // take, heat
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KnowledgeGraph {
    /// Entities indexed by ID.
    entities: HashMap<EntityId, Entity>,
    /// Outgoing relations: source → [relations].
    outgoing: HashMap<EntityId, Vec<Relation>>,
    /// Incoming relations: target → [relations].
    incoming: HashMap<EntityId, Vec<Relation>>,
    /// Index by entity type.
    type_index: HashMap<EntityType, Vec<EntityId>>,
}

impl KnowledgeGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an entity. Returns its ID.
    pub fn add_entity(&mut self, entity: Entity) -> EntityId {
        let id = entity.id;
        self.type_index
            .entry(entity.entity_type.clone())
            .or_default()
            .push(id);
        self.entities.insert(id, entity);
        id
    }

    /// Add a directed relation.
    pub fn add_relation(&mut self, relation: Relation) {
        self.incoming
            .entry(relation.target)
            .or_default()
            .push(relation.clone());
        self.outgoing
            .entry(relation.source)
            .or_default()
            .push(relation);
    }

    /// Get an entity by ID.
    pub fn entity(&self, id: EntityId) -> Option<&Entity> {
        self.entities.get(&id)
    }

    /// Get all entities of a given type.
    pub fn entities_by_type(&self, entity_type: &EntityType) -> Vec<&Entity> {
        self.type_index
            .get(entity_type)
            .map(|ids| ids.iter().filter_map(|id| self.entities.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get outgoing neighbors, optionally filtered by relation type.
    pub fn neighbors(
        &self,
        entity_id: EntityId,
        relation_type: Option<RelationType>,
    ) -> Vec<(&Entity, &Relation)> {
        let relations = self.outgoing.get(&entity_id);
        match relations {
            Some(rels) => rels
                .iter()
                .filter(|r| {
                    relation_type
                        .as_ref()
                        .map(|rt| r.relation_type == *rt)
                        .unwrap_or(true)
                })
                .filter_map(|r| self.entities.get(&r.target).map(|e| (e, r)))
                .collect(),
            None => vec![],
        }
    }

    /// Get incoming neighbors (reverse edges).
    pub fn incoming_neighbors(
        &self,
        entity_id: EntityId,
        relation_type: Option<RelationType>,
    ) -> Vec<(&Entity, &Relation)> {
        let relations = self.incoming.get(&entity_id);
        match relations {
            Some(rels) => rels
                .iter()
                .filter(|r| {
                    relation_type
                        .as_ref()
                        .map(|rt| r.relation_type == *rt)
                        .unwrap_or(true)
                })
                .filter_map(|r| self.entities.get(&r.source).map(|e| (e, r)))
                .collect(),
            None => vec![],
        }
    }

    /// Multi-hop traversal following specified relation types.
    ///
    /// Returns all reachable entities with their hop distance.
    pub fn traverse(
        &self,
        start: EntityId,
        relation_types: &[RelationType],
        max_hops: usize,
    ) -> Vec<(EntityId, usize)> {
        let mut visited = HashMap::new();
        let mut frontier = vec![(start, 0usize)];

        while let Some((node, depth)) = frontier.pop() {
            if depth > max_hops {
                continue;
            }
            if visited.contains_key(&node) {
                continue;
            }
            visited.insert(node, depth);

            if let Some(rels) = self.outgoing.get(&node) {
                for rel in rels {
                    if relation_types.contains(&rel.relation_type)
                        && !visited.contains_key(&rel.target)
                    {
                        frontier.push((rel.target, depth + 1));
                    }
                }
            }
        }

        visited.remove(&start);
        let mut result: Vec<_> = visited.into_iter().collect();
        result.sort_by_key(|&(_, d)| d);
        result
    }

    /// Find a path between two entities following given relation types.
    ///
    /// Returns the path as a sequence of entity IDs, or None if no path.
    pub fn find_path(
        &self,
        from: EntityId,
        to: EntityId,
        relation_types: &[RelationType],
        max_hops: usize,
    ) -> Option<Vec<EntityId>> {
        let mut visited = HashMap::new();
        let mut frontier = vec![(from, vec![from])];

        while let Some((node, path)) = frontier.pop() {
            if node == to {
                return Some(path);
            }
            if path.len() > max_hops + 1 {
                continue;
            }
            if visited.contains_key(&node) {
                continue;
            }
            visited.insert(node, true);

            if let Some(rels) = self.outgoing.get(&node) {
                for rel in rels {
                    if relation_types.contains(&rel.relation_type)
                        && !visited.contains_key(&rel.target)
                    {
                        let mut new_path = path.clone();
                        new_path.push(rel.target);
                        frontier.push((rel.target, new_path));
                    }
                }
            }
        }

        None
    }

    /// Get the ordered sequence of steps for a task.
    ///
    /// Follows `Requires` from the task, then `Precedes` between steps.
    pub fn task_plan(&self, task_id: EntityId) -> Vec<EntityId> {
        // Find the first step (required by task, not preceded by anything else in this task)
        let required = self.neighbors(task_id, Some(RelationType::Requires));
        if required.is_empty() {
            return vec![];
        }

        // Find the step that has no predecessor in this set
        let step_ids: Vec<EntityId> = required.iter().map(|(e, _)| e.id).collect();
        let mut first = step_ids[0];
        for &sid in &step_ids {
            let predecessors = self.incoming_neighbors(sid, Some(RelationType::Precedes));
            if predecessors.is_empty()
                || predecessors.iter().all(|(e, _)| !step_ids.contains(&e.id))
            {
                first = sid;
                break;
            }
        }

        // Walk the Precedes chain
        let mut plan = vec![first];
        let mut current = first;
        for _ in 0..100 {
            // safety limit
            let next = self.neighbors(current, Some(RelationType::Precedes));
            if let Some((entity, _)) = next.first() {
                plan.push(entity.id);
                current = entity.id;
            } else {
                break;
            }
        }

        plan
    }

    // ─── Validation (RFC-014 Opción 4) ─────────────────────────

    /// Get preconditions of an action entity.
    ///
    /// Returns entities connected via `HasPrecondition` edges.
    /// Each precondition represents a state that must hold before the action.
    pub fn preconditions(&self, action_id: EntityId) -> Vec<(&Entity, f32)> {
        self.neighbors(action_id, Some(RelationType::HasPrecondition))
            .into_iter()
            .map(|(e, r)| (e, r.weight))
            .collect()
    }

    /// Get effects of an action entity.
    ///
    /// Returns entities connected via `HasEffect` edges.
    /// Each effect represents a state change produced by the action.
    pub fn effects(&self, action_id: EntityId) -> Vec<(&Entity, f32)> {
        self.neighbors(action_id, Some(RelationType::HasEffect))
            .into_iter()
            .map(|(e, r)| (e, r.weight))
            .collect()
    }

    /// Validate whether an action is applicable given current state.
    ///
    /// Checks all `HasPrecondition` edges of the action entity.
    /// `current_state`: set of entity IDs that are currently true/present.
    ///
    /// Returns `(is_valid, missing_preconditions)`.
    pub fn validate_action(
        &self,
        action_id: EntityId,
        current_state: &std::collections::HashSet<EntityId>,
    ) -> (bool, Vec<EntityId>) {
        let preconditions = self.preconditions(action_id);
        let mut missing = Vec::new();

        for (precond_entity, _weight) in &preconditions {
            if !current_state.contains(&precond_entity.id) {
                missing.push(precond_entity.id);
            }
        }

        (missing.is_empty(), missing)
    }

    /// Filter a list of candidate actions to only valid ones.
    ///
    /// Returns action IDs that have all preconditions satisfied.
    pub fn filter_valid_actions(
        &self,
        candidate_actions: &[EntityId],
        current_state: &std::collections::HashSet<EntityId>,
    ) -> Vec<EntityId> {
        candidate_actions
            .iter()
            .filter(|&&action_id| {
                let (valid, _) = self.validate_action(action_id, current_state);
                valid
            })
            .copied()
            .collect()
    }

    /// Compute the state after applying an action (forward simulation).
    ///
    /// Adds the action's effects to the current state.
    /// Does NOT remove preconditions (effects are additive).
    pub fn apply_action(
        &self,
        action_id: EntityId,
        current_state: &std::collections::HashSet<EntityId>,
    ) -> std::collections::HashSet<EntityId> {
        let mut new_state = current_state.clone();
        for (effect_entity, _weight) in self.effects(action_id) {
            new_state.insert(effect_entity.id);
        }
        new_state
    }

    /// Number of entities.
    pub fn n_entities(&self) -> usize {
        self.entities.len()
    }

    /// Number of relations.
    pub fn n_relations(&self) -> usize {
        self.outgoing.values().map(|r| r.len()).sum()
    }

    /// Summary statistics.
    pub fn stats(&self) -> String {
        let mut type_counts: HashMap<&EntityType, usize> = HashMap::new();
        for e in self.entities.values() {
            *type_counts.entry(&e.entity_type).or_default() += 1;
        }
        let types: Vec<String> = type_counts
            .iter()
            .map(|(t, c)| format!("{t:?}={c}"))
            .collect();
        format!(
            "{} entities ({}), {} relations",
            self.n_entities(),
            types.join(", "),
            self.n_relations()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_task_graph() -> KnowledgeGraph {
        let mut kg = KnowledgeGraph::new();

        // Task: heat_then_place
        kg.add_entity(Entity::new(1, EntityType::Task, "heat_then_place"));

        // Steps in order
        kg.add_entity(Entity::new(10, EntityType::Action, "find"));
        kg.add_entity(Entity::new(11, EntityType::Action, "take"));
        kg.add_entity(Entity::new(12, EntityType::Action, "go_microwave"));
        kg.add_entity(Entity::new(13, EntityType::Action, "heat"));
        kg.add_entity(Entity::new(14, EntityType::Action, "take_heated"));
        kg.add_entity(Entity::new(15, EntityType::Action, "go_target"));
        kg.add_entity(Entity::new(16, EntityType::Action, "put"));

        // Task requires first step
        kg.add_relation(Relation::new(1, 10, RelationType::Requires, 1.0));

        // Step chain
        kg.add_relation(Relation::new(10, 11, RelationType::Precedes, 1.0));
        kg.add_relation(Relation::new(11, 12, RelationType::Precedes, 1.0));
        kg.add_relation(Relation::new(12, 13, RelationType::Precedes, 1.0));
        kg.add_relation(Relation::new(13, 14, RelationType::Precedes, 1.0));
        kg.add_relation(Relation::new(14, 15, RelationType::Precedes, 1.0));
        kg.add_relation(Relation::new(15, 16, RelationType::Precedes, 1.0));

        kg
    }

    #[test]
    fn graph_structure() {
        let kg = build_task_graph();
        assert_eq!(kg.n_entities(), 8); // 1 task + 7 steps
        assert_eq!(kg.n_relations(), 7); // 1 requires + 6 precedes
    }

    #[test]
    fn neighbors_filtered() {
        let kg = build_task_graph();

        let required = kg.neighbors(1, Some(RelationType::Requires));
        assert_eq!(required.len(), 1);
        assert_eq!(required[0].0.name, "find");

        let all = kg.neighbors(10, None);
        assert_eq!(all.len(), 1); // only Precedes to take
    }

    #[test]
    fn traverse_chain() {
        let kg = build_task_graph();

        let reachable = kg.traverse(10, &[RelationType::Precedes], 10);
        assert_eq!(reachable.len(), 6); // take, go_microwave, heat, take_heated, go_target, put

        // Check hop distances
        let find_take = reachable.iter().find(|&&(id, _)| id == 11);
        assert_eq!(find_take.unwrap().1, 1);

        let find_put = reachable.iter().find(|&&(id, _)| id == 16);
        assert_eq!(find_put.unwrap().1, 6);
    }

    #[test]
    fn traverse_limited_hops() {
        let kg = build_task_graph();
        let reachable = kg.traverse(10, &[RelationType::Precedes], 2);
        assert_eq!(reachable.len(), 2); // only take and go_microwave
    }

    #[test]
    fn find_path() {
        let kg = build_task_graph();

        let path = kg.find_path(10, 16, &[RelationType::Precedes], 10);
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.first(), Some(&10)); // find
        assert_eq!(path.last(), Some(&16)); // put
        assert_eq!(path.len(), 7);
    }

    #[test]
    fn find_path_no_route() {
        let kg = build_task_graph();
        let path = kg.find_path(16, 10, &[RelationType::Precedes], 10);
        assert!(path.is_none()); // can't go backwards
    }

    #[test]
    fn task_plan() {
        let kg = build_task_graph();
        let plan = kg.task_plan(1);
        assert_eq!(plan.len(), 7);
        assert_eq!(plan[0], 10); // find
        assert_eq!(plan[6], 16); // put
    }

    #[test]
    fn entities_by_type() {
        let kg = build_task_graph();
        let actions = kg.entities_by_type(&EntityType::Action);
        assert_eq!(actions.len(), 7);

        let tasks = kg.entities_by_type(&EntityType::Task);
        assert_eq!(tasks.len(), 1);
    }

    #[test]
    fn incoming_neighbors() {
        let kg = build_task_graph();
        let predecessors = kg.incoming_neighbors(13, Some(RelationType::Precedes));
        assert_eq!(predecessors.len(), 1);
        assert_eq!(predecessors[0].0.name, "go_microwave");
    }

    #[test]
    fn shared_sub_plans() {
        let mut kg = build_task_graph();

        // Add clean_then_place — shares find→take prefix
        kg.add_entity(Entity::new(2, EntityType::Task, "clean_then_place"));
        kg.add_entity(Entity::new(20, EntityType::Action, "go_sink"));
        kg.add_entity(Entity::new(21, EntityType::Action, "clean"));

        // clean_then_place requires same find step
        kg.add_relation(Relation::new(2, 10, RelationType::Requires, 1.0));

        // After take, clean diverges from heat
        kg.add_relation(Relation::new(11, 20, RelationType::Precedes, 1.0));
        kg.add_relation(Relation::new(20, 21, RelationType::Precedes, 1.0));

        // find→take is shared
        let find_reachable = kg.traverse(10, &[RelationType::Precedes], 1);
        assert_eq!(find_reachable.len(), 1); // take
        assert_eq!(find_reachable[0].0, 11);

        // From take, two paths diverge
        let take_neighbors = kg.neighbors(11, Some(RelationType::Precedes));
        assert_eq!(take_neighbors.len(), 2); // go_microwave AND go_sink
    }

    #[test]
    fn serialization() {
        let kg = build_task_graph();
        let bytes = postcard::to_allocvec(&kg).unwrap();
        let restored: KnowledgeGraph = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(restored.n_entities(), 8);
        assert_eq!(restored.n_relations(), 7);
    }

    // ─── Validation (RFC-014) ────────────────────────────────────

    fn build_validation_graph() -> KnowledgeGraph {
        let mut kg = KnowledgeGraph::new();

        // States (things that can be true)
        kg.add_entity(Entity::new(
            100,
            EntityType::Custom("state".into()),
            "object_visible",
        ));
        kg.add_entity(Entity::new(
            101,
            EntityType::Custom("state".into()),
            "holding_object",
        ));
        kg.add_entity(Entity::new(
            102,
            EntityType::Custom("state".into()),
            "at_appliance",
        ));
        kg.add_entity(Entity::new(
            103,
            EntityType::Custom("state".into()),
            "object_heated",
        ));

        // Actions
        kg.add_entity(Entity::new(10, EntityType::Action, "take"));
        kg.add_entity(Entity::new(11, EntityType::Action, "heat"));
        kg.add_entity(Entity::new(12, EntityType::Action, "put"));

        // take: requires object_visible, produces holding_object
        kg.add_relation(Relation::new(10, 100, RelationType::HasPrecondition, 1.0));
        kg.add_relation(Relation::new(10, 101, RelationType::HasEffect, 1.0));

        // heat: requires holding_object + at_appliance, produces object_heated
        kg.add_relation(Relation::new(11, 101, RelationType::HasPrecondition, 1.0));
        kg.add_relation(Relation::new(11, 102, RelationType::HasPrecondition, 1.0));
        kg.add_relation(Relation::new(11, 103, RelationType::HasEffect, 1.0));

        // put: requires holding_object, no special effect
        kg.add_relation(Relation::new(12, 101, RelationType::HasPrecondition, 1.0));

        kg
    }

    #[test]
    fn preconditions_and_effects() {
        let kg = build_validation_graph();

        let take_pre = kg.preconditions(10);
        assert_eq!(take_pre.len(), 1);
        assert_eq!(take_pre[0].0.name, "object_visible");

        let take_eff = kg.effects(10);
        assert_eq!(take_eff.len(), 1);
        assert_eq!(take_eff[0].0.name, "holding_object");

        let heat_pre = kg.preconditions(11);
        assert_eq!(heat_pre.len(), 2); // holding + at_appliance
    }

    #[test]
    fn validate_action_success() {
        let kg = build_validation_graph();
        let mut state = std::collections::HashSet::new();
        state.insert(100); // object_visible

        let (valid, missing) = kg.validate_action(10, &state); // take
        assert!(valid, "take should be valid when object is visible");
        assert!(missing.is_empty());
    }

    #[test]
    fn validate_action_failure() {
        let kg = build_validation_graph();
        let state = std::collections::HashSet::new(); // empty state

        let (valid, missing) = kg.validate_action(10, &state); // take
        assert!(!valid, "take should fail with empty state");
        assert_eq!(missing, vec![100]); // missing: object_visible
    }

    #[test]
    fn validate_action_partial_preconditions() {
        let kg = build_validation_graph();
        let mut state = std::collections::HashSet::new();
        state.insert(101); // holding_object but NOT at_appliance

        let (valid, missing) = kg.validate_action(11, &state); // heat
        assert!(!valid, "heat needs holding + at_appliance");
        assert_eq!(missing, vec![102]); // missing: at_appliance
    }

    #[test]
    fn filter_valid_actions() {
        let kg = build_validation_graph();
        let mut state = std::collections::HashSet::new();
        state.insert(101); // holding_object

        let candidates = vec![10, 11, 12]; // take, heat, put
        let valid = kg.filter_valid_actions(&candidates, &state);

        // take needs object_visible (not in state) → invalid
        // heat needs holding + at_appliance → invalid (missing at_appliance)
        // put needs holding → valid
        assert_eq!(valid, vec![12]);
    }

    #[test]
    fn apply_action_adds_effects() {
        let kg = build_validation_graph();
        let mut state = std::collections::HashSet::new();
        state.insert(100); // object_visible

        // Apply take → should add holding_object
        let new_state = kg.apply_action(10, &state);
        assert!(new_state.contains(&100)); // object_visible preserved
        assert!(new_state.contains(&101)); // holding_object added
    }

    #[test]
    fn forward_simulation() {
        let kg = build_validation_graph();
        let mut state = std::collections::HashSet::new();
        state.insert(100); // object_visible

        // take → holding
        state = kg.apply_action(10, &state);
        assert!(state.contains(&101));

        // add at_appliance manually (from navigation)
        state.insert(102);

        // heat → object_heated
        let (valid, _) = kg.validate_action(11, &state);
        assert!(valid, "heat should now be valid");
        state = kg.apply_action(11, &state);
        assert!(state.contains(&103)); // object_heated
    }
}
