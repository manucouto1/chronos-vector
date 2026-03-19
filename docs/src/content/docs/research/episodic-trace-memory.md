---
title: "Episodic Trace Memory for LLM Agents"
description: "Using CVX as a temporal reasoning backend for agent procedural memory"
---

## The Problem

Standard RAG gives LLM agents access to **facts** — but not to **how a problem was solved before**. This maps to cognitive science's separation between **semantic memory** (what you know) and **episodic memory** (what you've done and how it went).

Current vector stores are unordered — they cannot return the *sequence* of states an episode traversed. They have no temporal structure, no concept of episode identity. CVX has all three: entity-level trajectories, temporal ordering, and causal search.

## Core Idea: Reward-Filtered Temporal Analogy

> *"Find past reasoning episodes that began in a state similar to my current state, and return the path they took — ranked by whether they succeeded."*

| Capability | Standard Vector Store | CVX |
|-----------|----------------------|-----|
| Find similar states | Cosine similarity | Cosine + temporal distance (α-weighted) |
| Return ordered trajectory | Not possible | `trajectory(entity_id)` — ordered by timestamp |
| Filter by outcome | Post-hoc filtering | `MetadataFilter::gte("reward", 0.5)` |
| Show what happened next | Not possible | `causal_search()` — successor edges |
| Episode identity | Not native | `entity_id = episode_id << 16 | step_index` |

## Data Model

Each reasoning step is a `TemporalPoint`:

```
vector:     embed(state ⊕ action)     — what was the situation + what was done
entity_id:  encode(episode_id, step)  — episode:step encoded as u64
timestamp:  monotonic within episode  — step ordering
metadata:   {episode_id, step_index, reward, action_type, outcome_text}
```

## Retrieval Pipeline

1. **Phase 1** — Analogous episode discovery: `snapshot_knn` with `metadata_filter(step_index==0, reward>=0.5)` to find initial states of successful episodes
2. **Phase 2** — Full trajectory retrieval: `trajectory(entity_id)` for each candidate episode
3. **Phase 3** — Context injection: format retrieved traces into structured LLM prompt with success/failure annotations

## CVX API Surface

| CVX Feature | Used for |
|-------------|----------|
| `snapshot_knn` + `MetadataFilter` | Finding initial steps of successful episodes |
| `trajectory()` | Reconstructing full episode |
| `causal_search()` | "What happened next to similar states?" |
| `velocity()` | Filtering out high-variance noisy episodes |
| `detect_changepoints()` | Identifying episodes with unexpected pivots |
| Episode encoding (`encode_entity_id`) | Mapping episode:step → u64 entity_id |

## Proposed Benchmarks

| Benchmark | Domain | Baseline | CVX Expected |
|-----------|--------|----------|-------------|
| ALFWorld task completion | Embodied | 39% (GPT-4o no memory) | 60-75% (reward-filtered retrieval) |
| Procedural memory MAP | Retrieval quality | BM25/SentenceBERT | Higher on unseen vocabulary queries |
| LoCoMo QA F1 | Long-term conversation | 29.6 (standard RAG) | Improvement on temporal/multi-hop QA |

## References

- Fang et al. (2025). Memp: Exploring Agent Procedural Memory. arXiv:2508.06433
- Maharana et al. (2024). LoCoMo: Evaluating Very Long-Term Conversational Memory. ACL 2024
- Xu et al. (2025). A-MEM: Agentic Memory for LLM Agents. arXiv:2502.12110
- Position paper (2025). Episodic Memory is the Missing Piece. arXiv:2502.06975

See the [full implementation guide](/chronos-vector/architecture/llm-integration) for MCP tool definitions and agentic workflow patterns.
