# ChronosVector (CVX)

[![CI](https://github.com/manucouto1/chronos-vector/actions/workflows/ci.yml/badge.svg)](https://github.com/manucouto1/chronos-vector/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-live-green.svg)](https://manucouto1.github.io/chronos-vector/)
[![crates.io](https://img.shields.io/crates/v/chronos-vector.svg)](https://crates.io/crates/chronos-vector)
[![PyPI](https://img.shields.io/pypi/v/chronos-vector.svg)](https://pypi.org/project/chronos-vector/)
[![License: Elastic-2.0](https://img.shields.io/badge/license-Elastic--2.0-blue.svg)](LICENSE)
[![Rust 1.88+](https://img.shields.io/badge/rust-1.88%2B-orange.svg)](https://www.rust-lang.org)

A temporal vector database that treats **time as a geometric dimension** of embedding space. CVX stores vectors as trajectories, not snapshots — enabling velocity analysis, change point detection, causal retrieval, probabilistic reasoning, and structural knowledge over embedding evolution.

## What Makes CVX Different

| Capability | Static Vector DB | CVX |
|-----------|-----------------|-----|
| kNN search | Cosine/L2 at a point | Composite semantic + temporal distance |
| Trajectory extraction | Not possible | `trajectory(entity_id)` — ordered by time |
| "What happened next?" | Not possible | `causal_search()` — temporal edge traversal |
| "Was it successful?" | Not possible | `search_with_reward(min_reward=0.5)` — bitmap pre-filter |
| Drift / velocity | Not possible | `velocity()`, `drift()`, `hurst_exponent()` |
| Task structure | Not possible | `cvx-graph` — knowledge graph with task plans |
| P(success \| context) | Not possible | `cvx-bayes` — Bayesian network inference |

## Documentation

| Resource | Link |
|----------|------|
| **Full Documentation** (101 pages) | [manucouto1.github.io/chronos-vector](https://manucouto1.github.io/chronos-vector/) |
| **Unified Theory** | [6-layer framework](https://manucouto1.github.io/chronos-vector/research/unified-theory/) |
| **Python API** (52 functions) | [Reference](https://manucouto1.github.io/chronos-vector/specs/python-api/) |
| **Rust API** | [Tutorial](https://manucouto1.github.io/chronos-vector/tutorials/guides/rust-api/) |

### Tutorials

| Tutorial | What it covers |
|----------|---------------|
| [Quick Start](https://manucouto1.github.io/chronos-vector/tutorials/guides/quick-start/) | Install, insert, search, save/load |
| [Temporal Analytics](https://manucouto1.github.io/chronos-vector/tutorials/guides/temporal-analytics/) | Velocity, drift, changepoints, signatures, topology |
| [Anchor Projection](https://manucouto1.github.io/chronos-vector/tutorials/guides/anchor-projection/) | Centering, anchor projection, anisotropy correction |
| [Semantic Regions](https://manucouto1.github.io/chronos-vector/tutorials/guides/semantic-regions/) | HNSW hierarchy as clustering, distributional distances |
| [Episodic Memory](https://manucouto1.github.io/chronos-vector/tutorials/guides/episodic-memory/) | Causal search, reward filtering, agent memory |
| [Rust API Guide](https://manucouto1.github.io/chronos-vector/tutorials/guides/rust-api/) | Full Rust-facing API |

### Applications

| Domain | Dataset | Key Result |
|--------|---------|------------|
| [Mental Health](https://manucouto1.github.io/chronos-vector/applications/mental-health/overview/) | eRisk (1.36M posts) | F1=0.744 with DSM-5 anchor projection |
| [Political Discourse](https://manucouto1.github.io/chronos-vector/applications/political-discourse/overview/) | ParlaMint-ES (32K speeches) | F1=0.94 predicting speaker gender |
| [AI Agent Memory](https://manucouto1.github.io/chronos-vector/applications/agent-memory/overview/) | ALFWorld | 2× improvement with causal memory |

## Python Quick Start

```python
import chronos_vector as cvx
import numpy as np

# Create and populate
index = cvx.TemporalIndex(m=16, ef_construction=200)
index.bulk_insert(entity_ids, timestamps, vectors)
index.save("my_index")

# Centering (30× signal improvement for anisotropic embeddings)
centroid = index.compute_centroid()
index.set_centroid(centroid)

# Trajectory analysis
traj = index.trajectory(entity_id=1)
vel = cvx.velocity(traj, timestamp=t)
cps = cvx.detect_changepoints(1, traj)

# Causal search: "what happened next in similar situations?"
results = index.causal_search(query_vec, k=5, temporal_context=5)

# Reward-filtered: only successful experiences
results = index.search_with_reward(query_vec, k=5, min_reward=0.5)

# Bayesian scored search: multi-factor ranking
results = index.scored_search(query_vec, k=5,
    w_similarity=1.0, w_reward=0.5, w_success=0.4)

# Anchor projection
projected = cvx.project_to_anchors(traj, anchors, metric='cosine')
summary = cvx.anchor_summary(projected)
```

## Architecture

```
                    ┌──────────────────────────┐
                    │        cvx-api           │  REST + gRPC
                    └─────────┬────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
        ┌─────┴──┐    ┌──────┴──┐    ┌───────┴──────┐
        │ ingest │    │  query  │    │  analytics   │  27+ functions
        └────┬───┘    └────┬────┘    └───────┬──────┘
             │             │                 │
        ┌────┴─────────────┴─────────────────┴────┐
        │              cvx-index                  │
        │  ST-HNSW · temporal edges · typed edges │
        │  bayesian scorer · region MDP           │
        └────────────────┬────────────────────────┘
                         │
        ┌────────────────┴────────────────────────┐
        │             cvx-core                    │
        │  Types · Traits · Config · Error        │
        └─────────────────────────────────────────┘

        ┌──────────┐  ┌───────────┐
        │ cvx-bayes│  │ cvx-graph │  Companion crates
        │ Bayesian │  │ Knowledge │
        │ networks │  │ graph     │
        └──────────┘  └───────────┘
```

14 crates, 300+ tests.

| Crate | Purpose |
|-------|---------|
| `cvx-core` | Types, traits, config |
| `cvx-index` | ST-HNSW, temporal/typed edges, bayesian scorer, region MDP, persistence |
| `cvx-analytics` | 27+ functions: calculus, signatures, topology, anchors, Procrustes |
| `cvx-storage` | Hot/warm tiers, WAL, RocksDB |
| `cvx-query` | Query engine (15 query types) |
| `cvx-api` | REST (axum) + gRPC (tonic) |
| `cvx-mcp` | MCP server for LLM integration |
| `cvx-bayes` | Bayesian networks: variables, CPTs, inference |
| `cvx-graph` | Knowledge graph: entities, relations, task plans |
| `cvx-python` | Python bindings (PyO3) |

## Installation

```bash
# Python
pip install chronos-vector

# Rust
cargo add chronos-vector

# From source
cargo build --workspace
cargo test --workspace
```

## License

Licensed under the [Elastic License 2.0](LICENSE).
