//! # `cvx-storage` — Tiered storage architecture for ChronosVector.
//!
//! ## Layer 1: In-Memory Store
//! [`memory::InMemoryStore`] — non-persistent storage for development and testing.
//!
//! ## Layer 3: Hot Store (RocksDB)
//! [`hot::HotStore`] — persistent storage with column families, prefix bloom filters,
//! and per-CF compression. Requires the `hot-storage` feature flag.
//!
//! ## Layer 5: Write-Ahead Log
//! [`wal`] — Append-only, CRC32-validated log with segment rotation and crash recovery.
//!
//! ## Shared
//! [`keys`] — Big-endian key encoding with sign-bit flip for correct timestamp ordering.

#![deny(unsafe_code)]
#![warn(missing_docs)]

#[cfg(feature = "hot-storage")]
pub mod hot;
pub mod keys;
pub mod memory;
pub mod wal;
