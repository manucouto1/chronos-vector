//! `cvx-storage` — Tiered storage architecture for ChronosVector.
//!
//! Manages hot/warm/cold storage tiers with automatic compaction:
//! - **hot**: RocksDB wrapper with column families for vectors, deltas, metadata, timelines
//! - **warm**: Parquet read/write with Arrow RecordBatch and dictionary encoding
//! - **cold**: Object store (S3/MinIO/GCS) with Zarr format and PQ-encoded vectors
//! - **compactor**: Tier migration logic (hot -> warm -> cold)
//! - **wal**: Write-ahead log with segment-based append-only design
