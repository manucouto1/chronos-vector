# Changelog

All notable changes to this project will be documented in this file.

## [0.1.1](https://github.com/manucouto1/chronos-vector/compare/cvx-mcp-v0.1.0...cvx-mcp-v0.1.1) - 2026-03-20

### Added

- *(core)* region_assignments O(N) + clinical dashboard + ParlaMint B8
- episodic trace memory infrastructure (metadata filtering, episode encoding, MCP causal search)
- *(core,mcp)* add Embedder trait and embedding backends (RFC-009 Phase 3)
- *(mcp)* add MCP server crate with 8 LLM tools (RFC-009 Phase 1)

### Fixed

- *(cvx-mcp)* use workspace dependencies and add CHANGELOG.md for release

## [0.1.0] - 2026-03-20

### Added

- Initial MCP server implementation with JSON-RPC 2.0 protocol
- Tools: cvx_search, cvx_entity_summary, cvx_drift_report, cvx_detect_anomalies,
  cvx_compare_entities, cvx_cohort_analysis, cvx_forecast, cvx_causal_search, cvx_ingest
