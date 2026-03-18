//! `cvx-query` — Query execution engine for ChronosVector.
//!
//! Provides a unified interface over index, storage, and analytics to execute
//! eight temporal query types:
//!
//! - **SnapshotKnn**: k-nearest neighbors at instant t
//! - **RangeKnn**: k-nearest neighbors over time range [t1, t2]
//! - **Trajectory**: Path extraction for an entity over time
//! - **Velocity**: Rate of change (dv/dt) calculation
//! - **Prediction**: Future vector estimation via ODE/linear extrapolation
//! - **ChangePoint**: Change detection within a time window
//! - **DriftQuant**: Drift magnitude measurement
//! - **Analogy**: Temporal analogy queries

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod engine;
pub mod types;
