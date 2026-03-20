# Changelog

All notable changes to this project will be documented in this file.

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
