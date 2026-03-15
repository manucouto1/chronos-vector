//! `cvx-query` — Query execution engine for ChronosVector.
//!
//! Supports eight temporal query types:
//! - **SnapshotKnn**: k-nearest neighbors at instant t
//! - **RangeKnn**: k-nearest neighbors over time range [t1, t2]
//! - **Trajectory**: Path extraction for an entity over time
//! - **Velocity**: Rate of change (dv/dt) calculation
//! - **Prediction**: Future vector estimation via Neural ODE
//! - **ChangePoint**: Change detection within a time window
//! - **DriftQuant**: Drift magnitude measurement
//! - **Analogy**: Temporal analogy queries
