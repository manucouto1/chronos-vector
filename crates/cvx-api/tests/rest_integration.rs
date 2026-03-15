//! Integration tests for the REST API.
//!
//! Uses axum's test utilities to send requests without starting a real server.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use cvx_api::router::build_router;
use cvx_api::state::AppState;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

fn app() -> axum::Router {
    let state = Arc::new(AppState::new());
    build_router(state)
}

#[tokio::test]
async fn health_returns_ok() {
    let app = app();
    let req = Request::get("/v1/health").body(Body::empty()).unwrap();
    let resp = app.oneshot(req).await.unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "ok");
    assert_eq!(json["version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(json["index_size"], 0);
}

#[tokio::test]
async fn ready_returns_ok() {
    let app = app();
    let req = Request::get("/v1/ready").body(Body::empty()).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn ingest_single_point() {
    let app = app();
    let body = json!({
        "points": [{
            "entity_id": 42,
            "timestamp": 1000,
            "vector": [0.1, 0.2, 0.3]
        }]
    });

    let req = Request::post("/v1/ingest")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["ingested"], 1);
    assert_eq!(json["receipts"][0]["entity_id"], 42);
    assert_eq!(json["receipts"][0]["timestamp"], 1000);
}

#[tokio::test]
async fn ingest_batch_1000_and_query() {
    let state = Arc::new(AppState::new());
    let app = build_router(state);

    // Ingest 1000 vectors
    let mut points = Vec::new();
    for i in 0..1000u64 {
        points.push(json!({
            "entity_id": i,
            "timestamp": (i as i64) * 100,
            "vector": [i as f64 * 0.001, 1.0 - i as f64 * 0.001, 0.5]
        }));
    }

    let body = json!({ "points": points });
    let req = Request::post("/v1/ingest")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap();

    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let resp_json: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(resp_json["ingested"], 1000);

    // Query
    let query = json!({
        "vector": [0.5, 0.5, 0.5],
        "k": 5,
        "filter": { "type": "all" },
        "alpha": 1.0,
        "query_timestamp": 0
    });

    let req = Request::post("/v1/query")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_string(&query).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let resp_json: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(resp_json["results"].as_array().unwrap().len(), 5);
}

#[tokio::test]
async fn query_with_temporal_filter() {
    let state = Arc::new(AppState::new());
    let app = build_router(state);

    // Ingest points at different timestamps
    let points: Vec<Value> = (0..10)
        .map(|i| {
            json!({
                "entity_id": i as u64,
                "timestamp": (i as i64) * 1000,
                "vector": [1.0, 0.0, 0.0]
            })
        })
        .collect();

    let req = Request::post("/v1/ingest")
        .header("content-type", "application/json")
        .body(Body::from(json!({ "points": points }).to_string()))
        .unwrap();
    app.clone().oneshot(req).await.unwrap();

    // Query with range filter
    let query = json!({
        "vector": [1.0, 0.0, 0.0],
        "k": 10,
        "filter": { "type": "range", "start": 2000, "end": 5000 },
        "alpha": 1.0,
        "query_timestamp": 3000
    });

    let req = Request::post("/v1/query")
        .header("content-type", "application/json")
        .body(Body::from(query.to_string()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    let results = json["results"].as_array().unwrap();

    // Only timestamps 2000, 3000, 4000, 5000 should match
    assert_eq!(results.len(), 4);
    for r in results {
        let ts = r["timestamp"].as_i64().unwrap();
        assert!(ts >= 2000 && ts <= 5000, "timestamp {ts} out of range");
    }
}

#[tokio::test]
async fn trajectory_endpoint() {
    let state = Arc::new(AppState::new());
    let app = build_router(state);

    // Ingest points for entity 1
    let points: Vec<Value> = (0..5)
        .map(|i| {
            json!({
                "entity_id": 1,
                "timestamp": (i as i64) * 100,
                "vector": [i as f64 * 0.1, 1.0]
            })
        })
        .collect();

    let req = Request::post("/v1/ingest")
        .header("content-type", "application/json")
        .body(Body::from(json!({ "points": points }).to_string()))
        .unwrap();
    app.clone().oneshot(req).await.unwrap();

    // Get trajectory
    let req = Request::get("/v1/entities/1/trajectory")
        .body(Body::empty())
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["entity_id"], 1);
    assert_eq!(json["points"].as_array().unwrap().len(), 5);
}

#[tokio::test]
async fn ingest_empty_batch_returns_400() {
    let app = app();
    let body = json!({ "points": [] });
    let req = Request::post("/v1/ingest")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn ingest_nan_vector_returns_400() {
    let app = app();
    let body = json!({
        "points": [{
            "entity_id": 1,
            "timestamp": 100,
            "vector": [1.0, null, 0.0]
        }]
    });

    let req = Request::post("/v1/ingest")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    // null in vector should fail deserialization → 422
    assert!(
        resp.status() == StatusCode::BAD_REQUEST
            || resp.status() == StatusCode::UNPROCESSABLE_ENTITY
    );
}

#[tokio::test]
async fn query_empty_vector_returns_400() {
    let app = app();
    let query = json!({
        "vector": [],
        "k": 5
    });

    let req = Request::post("/v1/query")
        .header("content-type", "application/json")
        .body(Body::from(query.to_string()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}
