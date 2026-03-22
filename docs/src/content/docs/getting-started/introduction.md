---
title: Introduction
description: What is ChronosVector and why does it exist
---

**Project:** High-Performance Temporal Vector Analytics Platform in Rust
**Author:** Manuel Couto Pintos

---

ChronosVector (CVX) is a **temporal vector database** built in Rust that treats **time as a geometric dimension** in the embedding space. It doesn't just store and retrieve vectors — it models the **evolution of semantics over time**: analyzing drift velocity, detecting change points, projecting trajectories onto clinical or domain-specific anchors, and serving as episodic memory for AI agents.

## What Makes CVX Different

Standard vector databases (Qdrant, Milvus, Pinecone) treat vectors as static snapshots. CVX treats them as **trajectories** — ordered sequences of embeddings that evolve over time. This enables operations that static stores cannot provide:

| Capability | Static Vector DB | CVX |
|-----------|-----------------|-----|
| kNN search | Cosine/L2 at a point in time | Composite semantic + temporal distance |
| Trajectory extraction | Not possible | `trajectory(entity_id)` — ordered by timestamp |
| Drift measurement | Not possible | `velocity()`, `drift()`, `hurst_exponent()` |
| Change point detection | Not possible | `detect_changepoints()` (PELT/BOCPD) |
| Anchor projection | Not possible | `project_to_anchors()` — map to interpretable dimensions |
| Semantic regions | Not possible | `regions()`, `region_assignments()` — HNSW hierarchy as clustering |
| Episodic memory | Not possible | Episode encoding, causal search, temporal continuation |
| Path signatures | Not possible | Reparametrization-invariant trajectory descriptors |

## The Six Layers

CVX implements a [unified theory](/chronos-vector/research/unified-theory/) across six layers:

1. **Temporal vector index** — ST-HNSW with time-decay edges, SIMD distance kernels, RoaringBitmap filtering, centering for anisotropy correction
2. **Differential calculus** — Velocity, drift, Hurst exponent, change point detection (PELT/BOCPD) on embedding trajectories
3. **Algebraic invariants** — Path signatures (reparametrization-invariant), persistent homology, distributional distances (Fisher-Rao, Wasserstein)
4. **Temporal causality** — Episode encoding, causal search (temporal edges), typed edges (success/failure attribution), Granger causality
5. **Probabilistic reasoning** — Bayesian networks (`cvx-bayes`) for P(success | context), Region MDP, Bayesian multi-factor scoring with online weight learning
6. **Structural knowledge** — Knowledge graph (`cvx-graph`) for task plans, shared sub-plans, compositional reasoning, constraint validation

## Validated Applications

| Domain | Dataset | Key Result |
|--------|---------|------------|
| **Mental health detection** | eRisk (1.36M Reddit posts, 2,285 users) | F1=0.744 with DSM-5 anchor projection |
| **Political discourse** | ParlaMint-ES (32K speeches, 841 MPs) | F1=0.94 predicting speaker gender from rhetoric |
| **AI agent memory** | HumanEval, ALFWorld, APPS | 6x improvement in task completion (E3) |
| **Embedding anisotropy** | MentalRoBERTa D=768 | 30x discriminative signal improvement via centering |

## Who is this for?

| Persona | Use Case |
|---------|----------|
| **Clinical NLP Researcher** | Track symptom drift in social media using DSM-5 anchors |
| **Political Scientist** | Analyze rhetorical evolution and polarization over time |
| **AI/Agent Researcher** | Build episodic memory for LLM agents — store and retrieve action sequences |
| **ML Engineer** | Monitor embedding drift in production models |
| **NLP Researcher** | Study diachronic semantic evolution with path signatures |

## What CVX is NOT

- **Not a general-purpose vector database.** If you don't need temporal queries, use Qdrant.
- **Not a streaming platform.** CVX receives vectors, it doesn't produce them.
- **Not a model training framework.** Neural ODE is trained externally (TorchScript); CVX is inference + storage + analytics.
- **Not distributed.** Single-node architecture. Designed for research-scale datasets (up to ~10M vectors).

## Tech Stack

- **Language**: Rust (edition 2024), with Python bindings via PyO3
- **Index**: HNSW with temporal extensions, SIMD distance kernels (AVX2/NEON via `pulp`)
- **Storage**: RocksDB (hot tier), file-based partitions (warm tier), postcard serialization
- **Analytics**: 20+ modules in `cvx-analytics` — calculus, signatures, topology, ODE, Granger, Procrustes
- **Reasoning**: `cvx-bayes` (Bayesian networks), `cvx-graph` (knowledge graphs)
- **API**: Python bindings (primary), REST (axum), gRPC (tonic), MCP server for LLM integration
