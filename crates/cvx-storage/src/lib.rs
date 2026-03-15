//! # `cvx-storage` — Tiered storage architecture for ChronosVector.
//!
//! ## Layer 1: In-Memory Store
//!
//! [`memory::InMemoryStore`] provides a non-persistent storage backend for
//! development and testing. Implements [`cvx_core::StorageBackend`].

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod memory;
