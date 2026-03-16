//! `cvx-server` — ChronosVector server binary.
//!
//! Entry point responsible for:
//! - Configuration loading from file or environment
//! - Dependency injection and service wiring
//! - Tokio runtime bootstrap
//! - Graceful shutdown on SIGTERM/SIGINT

use std::path::PathBuf;
use std::sync::Arc;

use cvx_api::router::build_router;
use cvx_api::state::AppState;
use cvx_core::CvxConfig;
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    // Load configuration
    let config = load_config()?;
    let host = config.server.host.clone();
    let port = config.server.port;
    let addr = format!("{host}:{port}");

    tracing::info!("Configuration loaded");
    tracing::debug!(?config);

    // Build application state
    let state = Arc::new(AppState::new());
    let app = build_router(state);

    // Start server
    let listener = TcpListener::bind(&addr).await?;
    tracing::info!(
        "ChronosVector v{} listening on {addr}",
        env!("CARGO_PKG_VERSION")
    );

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("Server shut down gracefully");
    Ok(())
}

/// Load configuration from file or environment, with defaults.
fn load_config() -> anyhow::Result<CvxConfig> {
    // Check for config file path
    let config_path = std::env::var("CVX_CONFIG")
        .map(PathBuf::from)
        .ok()
        .or_else(|| {
            let default = PathBuf::from("config.toml");
            default.exists().then_some(default)
        });

    let mut config = if let Some(path) = config_path {
        let content = std::fs::read_to_string(&path)?;
        tracing::info!("Loading config from {}", path.display());
        CvxConfig::parse(&content)?
    } else {
        tracing::info!("Using default configuration");
        CvxConfig::default()
    };

    // Environment overrides
    if let Ok(host) = std::env::var("CVX_HOST") {
        config.server.host = host;
    }
    if let Ok(port) = std::env::var("CVX_PORT") {
        config.server.port = port.parse().unwrap_or(3000);
    }

    Ok(config)
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => tracing::info!("Received Ctrl+C"),
        () = terminate => tracing::info!("Received SIGTERM"),
    }
}
