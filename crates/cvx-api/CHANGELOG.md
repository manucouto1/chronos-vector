# Changelog

All notable changes to this project will be documented in this file.

## [0.1.1](https://github.com/manucouto1/chronos-vector/compare/cvx-api-v0.1.0...cvx-api-v0.1.1) - 2026-03-20

### Added

- *(core)* region_assignments O(N) + clinical dashboard + ParlaMint B8
- merge develop — RFC-010 temporal graph + RFC-011 anchor-space index
- *(query,api)* add causal search endpoint with temporal context (RFC-010 Phase 3)
- *(api)* add LLM-optimized composite endpoints (RFC-009 Phase 2)
- *(analytics)* add counterfactual trajectory analysis (RFC-007 Phase 5)
- *(analytics)* add Granger causality testing for embedding trajectories (RFC-007 Phase 4)
- *(analytics)* add temporal motif and discord discovery via Matrix Profile (RFC-007 Phase 3)
- *(analytics)* add temporal join for pairwise and group convergence detection (RFC-007 Phase 2)
- *(query,api)* integrate cohort drift into query engine and REST API

### Other

- *(release)* add CHANGELOG.md to all workspace crates

## [0.1.0] - 2025-12-01

### Added

- Initial release with REST (axum) and gRPC (tonic) API gateway
