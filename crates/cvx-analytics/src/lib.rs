//! `cvx-analytics` — Advanced temporal analytics for ChronosVector.
//!
//! Provides analytical capabilities for understanding vector evolution over time:
//! - **calculus**: Vector differential calculus (velocity, acceleration, drift, volatility)
//! - **ode**: Neural ODE solver (future)
//! - **pelt**: PELT offline change point detection (future)
//! - **bocpd**: BOCPD online streaming change point detection (future)

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod backend;
pub mod bocpd;
pub mod calculus;
pub mod explain;
pub mod multiscale;
pub mod ode;
pub mod pelt;
pub mod point_process;
pub mod signatures;
pub mod temporal_ml;
pub mod topology;
pub mod trajectory;
pub mod wasserstein;

/// TorchScript Neural ODE model (requires `torch-backend` feature).
#[cfg(feature = "torch-backend")]
pub mod torch_ode;

/// Neural ODE training in Rust (requires `torch-backend` feature).
#[cfg(feature = "torch-backend")]
pub mod torch_train;
