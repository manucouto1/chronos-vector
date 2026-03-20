---
title: "AI Agent Long-Term Memory"
description: "Using CVX as episodic memory for AI agents: storing and retrieving successful action sequences dependent on context"
---

## The Vision

Standard RAG gives LLM agents access to **facts** — but not to **how a problem was solved before**. This maps to cognitive science's separation between **semantic memory** (what you know) and **episodic memory** (what you've done and how it went).

Current vector stores are unordered — they cannot return the *sequence* of states an episode traversed. They have no temporal structure, no concept of episode identity. CVX has all three: entity-level trajectories, temporal ordering, and causal search.

## What CVX Already Provides

| Capability | Standard Vector Store | CVX |
|-----------|----------------------|-----|
| Find similar states | Cosine similarity | Cosine + temporal distance (alpha-weighted) |
| Return ordered trajectory | Not possible | `trajectory(entity_id)` — ordered by timestamp |
| Episode identity | Not native | `entity_id = episode_id << 16 \| step_index` |
| "What happened next?" | Not possible | `causal_search()` — successor edges (RFC-010) |
| Trajectory shape matching | Not possible | `path_signature()` — reparametrization-invariant |

### Episode Data Model

Each reasoning step is a `TemporalPoint`:

```
entity_id:  encode(episode_id, step_index)  — up to 281T episodes x 65K steps
timestamp:  monotonic within episode         — step ordering
vector:     embed(state + action)            — what was the situation + what was done
metadata:   {reward, action_type, outcome}   — optional annotations
```

## Experimental Results

### E1: Code Generation (MBPP to HumanEval)

- **Task**: Use solved MBPP problems as episodic memory to solve HumanEval
- **Memory**: 384 episodes x 3 steps (problem, plan, solution)
- **Result**: 77.8% pass@1 with episodic retrieval

### E3: Interactive ALFWorld (Embodied RL)

- **Task**: LLM agent navigating household environments, querying CVX at each step
- **Memory**: 336 expert trajectories (3-35 steps each)
- **Result**: 3.3% -> 20.0% task completion (6x improvement)
- **Key insight**: Step-by-step retrieval from current observation outperforms full-plan retrieval

### E4: Iterative Debugging (APPS Benchmark)

- **Task**: Agent retries failed code using CVX error-to-fix memory
- **Memory**: 172 multi-step debug traces (error -> attempt -> fix)
- **Result**: 28% -> 31% (3 additional problems rescued)

### Summary of Evidence

The signal is consistent: CVX-based episodic retrieval improves agent performance across domains. The improvements are moderate (not dramatic) because the current implementation lacks key capabilities — see gaps below.

## Identified Architectural Gaps (RFC-012 Part C)

What's needed to make CVX a first-class agent memory system:

### Gap 1: Outcome Awareness

CVX stores vectors but doesn't know if an episode was successful. An agent retrieves ALL experiences — successful and failed — without distinction.

**Needed**: `index.search(query, k=5, min_reward=0.5)` — filter by outcome.

### Gap 2: Causal Continuation Not Exposed

The most valuable pattern: "given a similar state, what steps came AFTER?". `TemporalGraphIndex` (RFC-010) implements this with predecessor/successor edges, but it's not available in the Python API.

**Needed**: `index.causal_search(query, k=5, continuation_steps=10)`

### Gap 3: No Structured Context Filtering

An agent needs: "in situations similar to X **when the goal was Y**". Currently, context is mixed into the embedding with no way to filter by goal or task type.

**Needed**: Indexed metadata with pre-filtering (not post-filtering).

### Gap 4: No Memory Consolidation

Biological memory consolidates: repeated similar experiences merge into abstract prototypes. CVX only accumulates, degrading retrieval quality at scale.

**Needed**: `index.consolidate()` — merge similar successful episodes into prototypes.

### Gap 5: No Recency Weighting

More recent experiences should be more accessible by default.

**Needed**: Optional recency bias in composite distance scoring.

## Future Directions

### Auxiliary Structures (Under Investigation)

Some agent memory needs may require structures beyond HNSW:

| Structure | What it adds | Potential use |
|-----------|-------------|---------------|
| **Knowledge graph** | Typed entities + relations | "Tool X requires skill Y" — compositional planning |
| **Bayesian network** | Conditional probability tables | P(success \| region, context) — decision under uncertainty |
| **Causal DAG** | Directed cause-effect | Counterfactual reasoning across episodes |

These are deferred until Gaps 1-4 are resolved. See [RFC-012 Part D](/chronos-vector/rfc/rfc-012) for the full analysis.

## Notebooks

| Notebook | Focus | Key result |
|----------|-------|------------|
| E1_episodic_coding | Code generation with episodic retrieval | 77.8% HumanEval |
| E2_episodic_alfworld | Plan quality from episodic retrieval | 0.709 semantic similarity |
| E3_interactive_alfworld | Step-by-step agent with CVX queries | 3.3% -> 20% completion |
| E4_iterative_coding | Debug retry memory | 28% -> 31% |

## Related

- [Episodic Trace Memory — Concept](/chronos-vector/research/episodic-trace-memory)
- [Episodic Memory — Full Experimental Report](/chronos-vector/research/episodic-memory-experiments)
- [RFC-010: Temporal Graph Extension](/chronos-vector/rfc/rfc-010) (causal search)
- [RFC-012 Part C: Architecture Gaps](/chronos-vector/rfc/rfc-012) (agent memory roadmap)
