//! Axum router configuration.

use axum::Router;
use axum::routing::{get, post};
use tower_http::trace::TraceLayer;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use crate::handlers;
use crate::openapi::ApiDoc;
use crate::state::SharedState;

/// Build the ChronosVector REST API router.
///
/// Includes Swagger UI at `/swagger-ui` and OpenAPI JSON at `/api-docs/openapi.json`.
pub fn build_router(state: SharedState) -> Router {
    Router::new()
        // Ingestion & query
        .route("/v1/ingest", post(handlers::ingest))
        .route("/v1/query", post(handlers::query))
        // Entity endpoints
        .route("/v1/entities/{id}/trajectory", get(handlers::trajectory))
        .route("/v1/entities/{id}/velocity", get(handlers::velocity))
        .route("/v1/entities/{id}/drift", get(handlers::drift))
        .route(
            "/v1/entities/{id}/changepoints",
            get(handlers::changepoints),
        )
        .route(
            "/v1/entities/{id}/prediction",
            get(handlers::prediction),
        )
        .route("/v1/analogy", post(handlers::analogy))
        .route("/v1/granger", post(handlers::granger))
        .route("/v1/entities/{id}/motifs", get(handlers::motifs))
        .route("/v1/entities/{id}/discords", get(handlers::discords))
        .route("/v1/temporal-join", post(handlers::temporal_join))
        .route("/v1/cohort/drift", post(handlers::cohort_drift))
        // System
        .route("/v1/health", get(handlers::health))
        .route("/v1/ready", get(handlers::ready))
        // OpenAPI documentation
        .merge(SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
