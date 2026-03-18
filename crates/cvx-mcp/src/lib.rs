//! `cvx-mcp` — MCP (Model Context Protocol) server for ChronosVector.
//!
//! Exposes CVX temporal analytics as tools for LLMs via the MCP protocol.
//! Communicates over JSON-RPC 2.0 on stdio.
//!
//! # Tools
//!
//! | Tool | Purpose |
//! |------|---------|
//! | `cvx_search` | Temporal RAG — find similar content within time windows |
//! | `cvx_entity_summary` | High-level temporal overview of an entity |
//! | `cvx_drift_report` | Quantify semantic change between two time points |
//! | `cvx_detect_anomalies` | Scan entities for unusual changes |
//! | `cvx_compare_entities` | Cross-entity temporal analysis |
//! | `cvx_cohort_analysis` | Group-level drift and convergence |
//! | `cvx_forecast` | Trajectory prediction |
//! | `cvx_ingest` | Add new temporal data points |

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod protocol;
pub mod server;
pub mod tools;
