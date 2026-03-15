//! `cvx-server` — ChronosVector server binary.
//!
//! Entry point responsible for:
//! - Configuration loading and validation
//! - Dependency injection and service wiring
//! - Tokio runtime bootstrap
//! - Graceful shutdown on SIGTERM/SIGINT

use std::sync::Arc;

use cvx_api::router::build_router;
use cvx_api::state::AppState;
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

    let host = std::env::var("CVX_HOST").unwrap_or_else(|_| "0.0.0.0".into());
    let port = std::env::var("CVX_PORT").unwrap_or_else(|_| "3000".into());
    let addr = format!("{host}:{port}");

    let state = Arc::new(AppState::new());
    let app = build_router(state);

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
