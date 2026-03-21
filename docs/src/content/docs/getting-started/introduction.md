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

## The Four Pillars

1. **Temporal vector index** — ST-HNSW with time-decay edges, SIMD distance kernels, and RoaringBitmap temporal filtering
2. **27+ analytical functions** — Vector calculus, path signatures, topology, distributional distances, Granger causality, motif discovery
3. **Anchor-based interpretability** — Project high-dimensional trajectories onto clinically or domain-meaningful reference vectors
4. **Episodic memory for AI** — Episode encoding, temporal edges, causal search for agent long-term memory

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
- **Analytics**: 19+ modules in `cvx-analytics` — calculus, signatures, topology, ODE solver, Granger causality
- **API**: Python bindings (primary), REST (axum), gRPC (tonic), MCP server for LLM integration
