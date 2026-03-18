//! Configuration types for ChronosVector.
//!
//! The primary configuration is [`CvxConfig`], which is deserialized from a TOML file.
//! All fields have sensible defaults for development use.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Top-level ChronosVector configuration.
///
/// Deserialized from `config.toml`. All sections are optional with defaults.
///
/// # Example
///
/// ```
/// use std::path::PathBuf;
/// use cvx_core::CvxConfig;
///
/// let config: CvxConfig = toml::from_str(r#"
///     [server]
///     host = "0.0.0.0"
///     port = 8080
///
///     [storage]
///     data_dir = "./data"
///
///     [logging]
///     level = "info"
/// "#).unwrap();
///
/// assert_eq!(config.server.port, 8080);
/// assert_eq!(config.storage.data_dir, PathBuf::from("./data"));
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct CvxConfig {
    /// Server network configuration.
    pub server: ServerConfig,
    /// Storage engine configuration.
    pub storage: StorageConfig,
    /// Index (ST-HNSW) configuration.
    pub index: IndexConfig,
    /// Analytics engine configuration.
    pub analytics: AnalyticsConfig,
    /// Logging and observability configuration.
    pub logging: LoggingConfig,
}

impl CvxConfig {
    /// Load configuration from a TOML file.
    ///
    /// Returns `CvxError::Config` if the file cannot be read or parsed.
    pub fn from_file(path: &std::path::Path) -> Result<Self, crate::CvxError> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            crate::CvxError::Config(format!("cannot read config file {}: {e}", path.display()))
        })?;
        toml::from_str(&content).map_err(|e| {
            crate::CvxError::Config(format!("cannot parse config file {}: {e}", path.display()))
        })
    }

    /// Parse configuration from a TOML string.
    pub fn parse(s: &str) -> Result<Self, crate::CvxError> {
        toml::from_str(s).map_err(|e| crate::CvxError::Config(format!("cannot parse config: {e}")))
    }
}

/// Network server configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ServerConfig {
    /// Bind address for the HTTP server.
    pub host: String,
    /// HTTP port.
    pub port: u16,
    /// Optional gRPC port. If `None`, gRPC is disabled.
    pub grpc_port: Option<u16>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".into(),
            port: 8080,
            grpc_port: None,
        }
    }
}

/// Storage engine configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct StorageConfig {
    /// Root directory for all storage data.
    pub data_dir: PathBuf,
    /// Hot tier (RocksDB) configuration.
    pub hot: HotTierConfig,
    /// Warm tier (Parquet) configuration.
    pub warm: WarmTierConfig,
    /// Cold tier (object store) configuration.
    pub cold: ColdTierConfig,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./data"),
            hot: HotTierConfig::default(),
            warm: WarmTierConfig::default(),
            cold: ColdTierConfig::default(),
        }
    }
}

/// Hot tier (RocksDB) configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct HotTierConfig {
    /// Maximum size of the hot tier in megabytes.
    pub max_size_mb: u64,
}

impl Default for HotTierConfig {
    fn default() -> Self {
        Self { max_size_mb: 1024 }
    }
}

/// Warm tier (Parquet) configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct WarmTierConfig {
    /// Whether the warm tier is enabled.
    pub enabled: bool,
}

/// Cold tier (object store) configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ColdTierConfig {
    /// Whether the cold tier is enabled.
    pub enabled: bool,
    /// Object store endpoint (e.g., `s3://bucket/prefix`).
    pub endpoint: Option<String>,
}

/// ST-HNSW index configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct IndexConfig {
    /// Maximum connections per node per layer.
    pub m: usize,
    /// Search width during index construction.
    pub ef_construction: usize,
    /// Search width during queries.
    pub ef_search: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
        }
    }
}

/// Analytics engine configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AnalyticsConfig {
    /// Enable Neural ODE prediction (requires `torch-backend` feature).
    pub neural_ode: bool,
    /// Path to TorchScript Neural ODE model file (`.pt`).
    /// Required when `neural_ode = true` and `torch-backend` feature is enabled.
    pub model_path: Option<PathBuf>,
    /// Change point detection method.
    pub change_detection: String,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            neural_ode: false,
            model_path: None,
            change_detection: "pelt".into(),
        }
    }
}

/// Logging and observability configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct LoggingConfig {
    /// Log level (`trace`, `debug`, `info`, `warn`, `error`).
    pub level: String,
    /// Log format (`text` or `json`).
    pub format: String,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".into(),
            format: "text".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let config = CvxConfig::default();
        assert_eq!(config.server.port, 8080);
        assert_eq!(config.index.m, 16);
        assert_eq!(config.logging.level, "info");
    }

    #[test]
    fn deserialize_minimal_toml() {
        let config: CvxConfig = toml::from_str("").unwrap();
        assert_eq!(config, CvxConfig::default());
    }

    #[test]
    fn deserialize_partial_toml() {
        let config: CvxConfig = toml::from_str(
            r#"
            [server]
            port = 9090

            [index]
            m = 32
        "#,
        )
        .unwrap();
        assert_eq!(config.server.port, 9090);
        assert_eq!(config.index.m, 32);
        // Rest should be defaults
        assert_eq!(config.storage.data_dir, PathBuf::from("./data"));
    }

    #[test]
    fn deserialize_full_config_example() {
        let toml_str = r#"
            [server]
            host = "0.0.0.0"
            port = 8080

            [storage]
            data_dir = "./data"

            [storage.hot]
            max_size_mb = 2048

            [storage.warm]
            enabled = true

            [storage.cold]
            enabled = true
            endpoint = "s3://my-bucket/cvx"

            [index]
            m = 16
            ef_construction = 200
            ef_search = 50

            [analytics]
            neural_ode = false
            change_detection = "pelt"

            [logging]
            level = "info"
            format = "json"
        "#;
        let config: CvxConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.storage.hot.max_size_mb, 2048);
        assert!(config.storage.warm.enabled);
        assert!(config.storage.cold.enabled);
        assert_eq!(
            config.storage.cold.endpoint.as_deref(),
            Some("s3://my-bucket/cvx")
        );
        assert_eq!(config.logging.format, "json");
    }

    #[test]
    fn from_str_returns_error_on_invalid_toml() {
        let result = CvxConfig::parse("[invalid toml %%");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, crate::CvxError::Config(_)));
    }

    #[test]
    fn config_serialization_roundtrip() {
        let config = CvxConfig::default();
        let toml_str = toml::to_string_pretty(&config).unwrap();
        let recovered: CvxConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(config, recovered);
    }
}
