//! `cvx-analytics` — Advanced temporal analytics for ChronosVector.
//!
//! Provides analytical capabilities for understanding vector evolution over time:
//! - **calculus**: Vector differential calculus (velocity, acceleration, drift, volatility)
//! - **ode**: Neural ODE solver (future)
//! - **pelt**: PELT offline change point detection (future)
//! - **bocpd**: BOCPD online streaming change point detection (future)

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod bocpd;
pub mod calculus;
pub mod explain;
pub mod multiscale;
pub mod ode;
pub mod pelt;
