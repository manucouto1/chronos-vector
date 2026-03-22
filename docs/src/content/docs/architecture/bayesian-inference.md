---
title: "Bayesian Inference (cvx-bayes)"
description: "Discrete Bayesian networks for conditional probability reasoning over temporal episode data"
---

## Motivation

Vector similarity answers "what is close?" but not "what is likely to succeed?". A Bayesian network models the **conditional dependencies** between variables (task type, region, action, outcome) to compute:

$$P(\text{success} \mid \text{task\_type}, \text{region}, \text{action\_type})$$

This captures interactions that linear scoring cannot: the same action has different success rates depending on the context.

## Theoretical Foundation

A Bayesian network is a directed acyclic graph (DAG) where:

- **Nodes** are random variables with discrete states
- **Edges** encode conditional dependencies (parent → child)
- **CPTs** (Conditional Probability Tables) store $P(X \mid \text{parents}(X))$

Inference computes the posterior $P(\text{query} \mid \text{evidence})$ by propagating beliefs through the graph.

### Laplace Smoothing

CPTs are learned from observations with Laplace smoothing to prevent zero probabilities:

$$P(x \mid \text{parents}) = \frac{\text{count}(x, \text{parents}) + \alpha}{\sum_x \text{count}(x, \text{parents}) + \alpha \cdot K}$$

where $\alpha$ is the pseudo-count (default 1.0) and $K$ is the number of states.

## Architecture

```
cvx-bayes
├── Variable       — discrete random variable with named states
├── Cpt            — conditional probability table with online learning
└── BayesianNetwork — DAG structure + inference
```

## API

### Define Variables

```rust
use cvx_bayes::{BayesianNetwork, Variable};

let mut bn = BayesianNetwork::new();

let task = bn.add_variable(Variable::new(0, "task_type", vec![
    "pick_and_place".into(),
    "heat_then_place".into(),
    "clean_then_place".into(),
]));

let region = bn.add_variable(Variable::new(1, "region", vec![
    "kitchen".into(), "bathroom".into(), "bedroom".into(),
]));

let success = bn.add_variable(Variable::binary(2, "success"));
```

### Define Dependencies

```rust
// Success depends on both task type and region
bn.add_edge(task, success);
bn.add_edge(region, success);
bn.initialize_cpts();
```

### Learn from Observations

```rust
// After each episode, observe the outcome
bn.observe(&[(task, 0), (region, 0), (success, 0)]); // pick in kitchen → success
bn.observe(&[(task, 1), (region, 1), (success, 1)]); // heat in bathroom → failure

// Update CPTs with accumulated counts
bn.update_cpts();
```

### Query

```rust
// P(success | task=heat, region=kitchen)
let p = bn.query(success, 0, &[(task, 1), (region, 0)]);

// Most likely outcome
let (state, prob) = bn.map_estimate(success, &[(task, 0), (region, 0)]);

// Full posterior distribution
let posterior = bn.posterior(success, &[(task, 2)]);
// posterior = [P(true), P(false)]
```

## Integration with CVX

The Bayesian network augments `scored_search` by providing context-aware probability estimates instead of the fixed-weight linear scorer:

```
1. CVX retrieves k candidates via HNSW
2. For each candidate, extract (task_type, region, action_type)
3. BN computes P(success | task, region, action) per candidate
4. Re-rank by posterior probability
```

## References

1. Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems*
2. Koller & Friedman (2009). *Probabilistic Graphical Models*
3. Murphy, K. (2012). *Machine Learning: A Probabilistic Perspective*
