//! Axum router configuration.

use axum::Router;
use axum::routing::{get, post};
use tower_http::trace::TraceLayer;

use crate::handlers;
use crate::state::SharedState;

/// Build the ChronosVector REST API router.
pub fn build_router(state: SharedState) -> Router {
    Router::new()
        .route("/v1/ingest", post(handlers::ingest))
        .route("/v1/query", post(handlers::query))
        .route("/v1/entities/{id}/trajectory", get(handlers::trajectory))
        .route("/v1/health", get(handlers::health))
        .route("/v1/ready", get(handlers::ready))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
