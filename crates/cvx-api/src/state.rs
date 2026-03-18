//! Shared application state for the API server.

use std::sync::Arc;
use std::time::Instant;

use cvx_core::StorageBackend;
use cvx_index::hnsw::{ConcurrentTemporalHnsw, HnswConfig};
use cvx_index::metrics::L2Distance;
use cvx_ingest::validation::ValidationConfig;
use cvx_storage::memory::InMemoryStore;

/// Shared application state injected into all handlers.
pub struct AppState {
    /// The temporal vector index.
    pub index: ConcurrentTemporalHnsw<L2Distance>,
    /// The storage backend (hot tier by default).
    pub store: Box<dyn StorageBackend>,
    /// Validation config for ingestion.
    pub validation: ValidationConfig,
    /// Server start time (for uptime reporting).
    pub started_at: Instant,
    /// Whether the server is ready to serve requests.
    pub ready: std::sync::atomic::AtomicBool,
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

impl AppState {
    /// Create a new application state with default configuration (in-memory store).
    pub fn new() -> Self {
        Self::with_config(HnswConfig::default(), ValidationConfig::default())
    }

    /// Create with custom HNSW and validation configs.
    pub fn with_config(hnsw_config: HnswConfig, validation_config: ValidationConfig) -> Self {
        Self {
            index: ConcurrentTemporalHnsw::new(hnsw_config, L2Distance),
            store: Box::new(InMemoryStore::new()),
            validation: validation_config,
            ready: std::sync::atomic::AtomicBool::new(true),
            started_at: Instant::now(),
        }
    }

    /// Create with a custom storage backend (e.g., TieredStorage).
    pub fn with_store(
        hnsw_config: HnswConfig,
        validation_config: ValidationConfig,
        store: Box<dyn StorageBackend>,
    ) -> Self {
        Self {
            index: ConcurrentTemporalHnsw::new(hnsw_config, L2Distance),
            store,
            validation: validation_config,
            ready: std::sync::atomic::AtomicBool::new(true),
            started_at: Instant::now(),
        }
    }
}

/// Type alias for the shared state used in axum extractors.
pub type SharedState = Arc<AppState>;
