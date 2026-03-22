//! # `cvx-bayes` — Bayesian Network Inference for ChronosVector
//!
//! Provides discrete Bayesian networks for probabilistic reasoning over
//! temporal episode data. Designed to answer questions that vector
//! similarity alone cannot:
//!
//! - P(success | task_type=clean, region=R5, action=navigate)
//! - "How confident am I in this prediction?" (posterior variance)
//! - "What's the most likely action given partial observations?"
//!
//! ## Theoretical Foundation
//!
//! A Bayesian network is a directed acyclic graph (DAG) where:
//! - **Nodes** are random variables (task_type, region, action, success)
//! - **Edges** encode conditional dependencies
//! - **CPTs** (Conditional Probability Tables) store P(X | parents(X))
//!
//! Inference computes the posterior P(query | evidence) by propagating
//! beliefs through the graph. For small networks (< 20 variables),
//! exact inference via variable elimination is tractable.
//!
//! ## References
//!
//! - Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems*
//! - Koller & Friedman (2009). *Probabilistic Graphical Models*
//! - Murphy, K. (2012). *Machine Learning: A Probabilistic Perspective*

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod cpt;
pub mod network;
pub mod variable;

pub use cpt::Cpt;
pub use network::BayesianNetwork;
pub use variable::{Variable, VariableId};
