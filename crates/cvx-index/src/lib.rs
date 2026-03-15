//! `cvx-index` — Temporal Index Engine (ST-HNSW) for ChronosVector.
//!
//! Implements a Spatio-Temporal Hierarchical Navigable Small World graph with:
//! - **hnsw**: Core graph structure with navigable layers (L0–L3)
//! - **timestamp_graph**: TANNS integration for timestamp-aware neighbor tracking
//! - **bitmap**: Roaring bitmap-based temporal filtering per time range
//! - **decay**: Time-decay edge manager (exponential decay)
//! - **metrics**: SIMD distance kernels (cosine, L2, dot product, Poincare)
