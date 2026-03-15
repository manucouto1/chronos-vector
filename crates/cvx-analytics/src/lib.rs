//! `cvx-analytics` — Advanced temporal analytics for ChronosVector.
//!
//! Provides analytical capabilities for understanding vector evolution over time:
//! - **ode**: Neural ODE solver using Dormand-Prince RK45 with ODE-RNN encoder/decoder
//! - **pelt**: PELT offline change point detection with O(N) complexity
//! - **bocpd**: BOCPD online streaming change point detection with O(1) amortized complexity
//! - **diffcalc**: Vector differential calculus (velocity, acceleration, curvature, geodesic distance)
