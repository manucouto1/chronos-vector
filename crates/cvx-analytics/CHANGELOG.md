# Changelog

All notable changes to this project will be documented in this file.

## [0.1.2](https://github.com/manucouto1/chronos-vector/compare/cvx-analytics-v0.1.1...cvx-analytics-v0.1.2) - 2026-03-22

### Added

- parallel bulk_insert (P9) + Procrustes alignment (P10)

## [0.1.1](https://github.com/manucouto1/chronos-vector/compare/cvx-analytics-v0.1.0...cvx-analytics-v0.1.1) - 2026-03-20

### Added

- *(core)* region_assignments O(N) + clinical dashboard + ParlaMint B8
- merge develop — RFC-010 temporal graph + RFC-011 anchor-space index
- *(analytics)* add AnchorSpaceIndex for cross-model invariant indexing (RFC-011 Phase 1)
- *(analytics)* add counterfactual trajectory analysis (RFC-007 Phase 5)
- *(analytics)* add Granger causality testing for embedding trajectories (RFC-007 Phase 4)
- *(analytics)* add temporal motif and discord discovery via Matrix Profile (RFC-007 Phase 3)
- *(analytics)* add temporal join for pairwise and group convergence detection (RFC-007 Phase 2)
- *(analytics)* add cohort drift analysis module (RFC-007 Phase 1)

### Other

- *(release)* add CHANGELOG.md to all workspace crates

## [0.1.0] - 2025-12-01

### Added

- Initial release with 19+ temporal analytics functions (calculus, signatures, topology, changepoints, ODE)
