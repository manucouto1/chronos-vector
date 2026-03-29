# Changelog

All notable changes to this project will be documented in this file.

## [0.1.2](https://github.com/manucouto1/chronos-vector/compare/cvx-index-v0.1.1...cvx-index-v0.1.2) - 2026-03-22

### Added

- KG validation + trajectory search (RFC-014 Opciones 3+4)
- *(index)* scored_search wiring A+B+C into search pipeline (RFC-013 Part D)
- *(index)* Bayesian scorer + Region MDP in Rust (RFC-013 Parts A+C)
- *(index)* typed edge store for relational memory (RFC-013 Part B)
- *(index)* parallel bulk_insert with rayon (RFC-012 P9)
- *(index)* recency-weighted search + distance normalization (RFC-012 P7+P8)
- RFC-013 integrated design, RegionMDP module, snapshot versioning, results
- *(index)* outcome-aware search with native reward field (RFC-012 P4)
- *(index)* inverted metadata index for O(1) pre-filtering (RFC-012 P3)
- *(python)* expose causal_search and hybrid_search (RFC-012 P2)
- *(index)* native centroid for anisotropy correction (RFC-012 Part B)

### Other

- *(index)* close coverage gaps — 18 new tests for accessors, threading, delegations

## [0.1.1](https://github.com/manucouto1/chronos-vector/compare/cvx-index-v0.1.0...cvx-index-v0.1.1) - 2026-03-20

### Added

- *(core)* region_assignments O(N) + clinical dashboard + ParlaMint B8
- *(index)* integrate MetadataStore into TemporalHnsw
- episodic trace memory infrastructure (metadata filtering, episode encoding, MCP causal search)
- merge develop — RFC-010 temporal graph + RFC-011 anchor-space index
- *(index)* add TemporalGraphIndex with hybrid and causal search (RFC-010 Phase 2)
- *(index)* add TemporalEdgeLayer for successor/predecessor edges (RFC-010 Phase 1)
- *(index)* add T-LSH and Streaming Window Index (RFC-008 Phase 2-3)
- *(index)* add time-partitioned HNSW index (RFC-008 Phase 1)

### Other

- *(release)* add CHANGELOG.md to all workspace crates

## [0.1.0] - 2025-12-01

### Added

- Initial release with temporal HNSW index with SIMD distance kernels and semantic regions
