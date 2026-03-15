---
title: Introduction
description: What is ChronosVector and why does it exist
---

**Project:** High-Performance Temporal Vector Data Platform in Rust
**Author:** Manuel Couto Pintos
**Version:** 1.0
**Status:** Design Phase

---

ChronosVector (CVX) is a **high-performance temporal vector database** built in Rust that treats **time as a geometric dimension** in the embedding space. It doesn't just store and retrieve vectors — it models the **curvature of semantics over time**: predicting future states, analyzing semantic drift velocity, and detecting change points in the evolution of vector representations.

## The Three Pillars

The project combines three disciplines that are rarely integrated in a single system:

1. **High-performance vector storage** (at the level of Qdrant, DiskANN)
2. **Continuous temporal modeling** (Neural ODEs, diachronic embeddings)
3. **Drift detection and analysis** (BOCPD, PELT, semantic drift)

CVX positions Rust as the ideal language for this convergence: byte-level memory control, native SIMD, and concurrency without data races.

## Who is this for?

| Persona | Use Case |
|---------|----------|
| **ML Engineer** | Monitor embedding drift in production |
| **NLP Researcher** | Study diachronic semantic evolution |
| **RecSys Developer** | Predict user interest trajectories |
| **KG Engineer** | Temporal knowledge graph completion |
| **Data Scientist** | Exploratory analysis of embedding time series |

## What CVX is NOT

- **Not a general-purpose vector database.** If you don't need temporal queries, use Qdrant.
- **Not a streaming platform.** CVX receives vectors, it doesn't produce them.
- **Not a model training framework.** The Neural ODE is trained externally; CVX is inference + storage infrastructure.
- **Not distributed (initially).** Phases 1-4 are single-node.
