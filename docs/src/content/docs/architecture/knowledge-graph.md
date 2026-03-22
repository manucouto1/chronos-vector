---
title: "Knowledge Graph (cvx-graph)"
description: "Typed property graph for compositional reasoning over task structure, shared plans, and entity relations"
---

## Motivation

Vector retrieval finds "similar states" but cannot reason about **structure**: "what step comes next?", "what do these two tasks share?", "is this action valid here?". A knowledge graph encodes these relations explicitly.

## Theoretical Foundation

A **property graph** extends the standard graph with:

- **Typed nodes**: entities have a semantic type (Task, Action, Object, Location)
- **Typed edges**: relations have a label (Precedes, Requires, LocatedAt)
- **Properties**: key-value attributes on both nodes and edges

This enables **compositional queries**: starting from a task entity, follow `Requires` edges to find its steps, then follow `Precedes` edges to get the ordered plan.

### Task Structure as a Graph

A multi-step task like `heat_then_place` encodes as:

```
[heat_then_place] --Requires--> [find] --Precedes--> [take] --Precedes--> [go_microwave]
                                                                          --Precedes--> [heat]
                                                                          --Precedes--> [take_heated]
                                                                          --Precedes--> [go_target]
                                                                          --Precedes--> [put]
```

The agent can query: "I just completed `take`. What comes next?" → follow `Precedes` → `go_microwave`.

### Shared Sub-Plans

Tasks share structure:

```
heat_then_place:  find → take → go_microwave → heat → take → go_target → put
clean_then_place: find → take → go_sinkbasin → clean → go_target → put
                  ^^^^^^^^^^^^
                  shared prefix
```

The graph captures this: both tasks' `Requires` edge points to the same `find` entity, and `find → take` is a shared sub-graph. Knowledge transfer becomes graph traversal.

## Architecture

```
cvx-graph
├── Entity        — typed node (Task, Action, Object, Location, Appliance)
├── Relation      — typed directed edge (Precedes, Requires, LocatedAt, Uses, SimilarTo)
└── KnowledgeGraph — adjacency + type index + traversal + path finding + task_plan
```

## API

### Build a Task Graph

```rust
use cvx_graph::{KnowledgeGraph, Entity, EntityType, Relation, RelationType};

let mut kg = KnowledgeGraph::new();

let task = kg.add_entity(Entity::new(1, EntityType::Task, "heat_then_place"));
let find = kg.add_entity(Entity::new(2, EntityType::Action, "find"));
let take = kg.add_entity(Entity::new(3, EntityType::Action, "take"));
let heat = kg.add_entity(Entity::new(4, EntityType::Action, "heat"));

kg.add_relation(Relation::new(task, find, RelationType::Requires, 1.0));
kg.add_relation(Relation::new(find, take, RelationType::Precedes, 1.0));
kg.add_relation(Relation::new(take, heat, RelationType::Precedes, 1.0));
```

### Query

```rust
// What does this task require?
let steps = kg.neighbors(task, Some(RelationType::Requires));

// Ordered plan extraction
let plan = kg.task_plan(task);
// plan = [find, take, heat, ...]

// Multi-hop traversal: what's reachable from find?
let reachable = kg.traverse(find, &[RelationType::Precedes], 5);

// Path finding: how to get from find to put?
let path = kg.find_path(find, put, &[RelationType::Precedes], 10);
```

### Entities by Type

```rust
let all_actions = kg.entities_by_type(&EntityType::Action);
let all_locations = kg.entities_by_type(&EntityType::Location);
```

## Integration with CVX

The knowledge graph provides **structural context** for retrieval:

1. Agent identifies current task type
2. `task_plan()` gives the expected step sequence
3. Agent determines current step position (via action matching)
4. Next expected step guides retrieval: bias `causal_search` toward episodes that match the expected next action
5. Constraint validation: reject retrieved actions that violate the `Precedes` chain

## References

1. Hogan et al. (2021). *Knowledge Graphs*. ACM Computing Surveys.
2. Ji et al. (2022). *A Survey on Knowledge Graphs*. Expert Systems.
