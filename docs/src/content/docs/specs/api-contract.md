---
title: "API Contract"
description: "REST and gRPC API specification for ChronosVector"
---

This document extracts the API contract from the ChronosVector Technical PRD, covering REST endpoints, gRPC services, error codes, timestamp semantics, and supported vector types.

---

## Supported Vector Types

| Type | Storage Size (D=768) | Use Case |
|---|---|---|
| FP32 | 3072 bytes | Default. Full precision for hot tier. |
| FP16 | 1536 bytes | Warm tier option for 2x compression with <1% recall loss. |
| INT8 (Scalar Quantized) | 768 bytes | 4x compression. Good for large-scale with moderate accuracy needs. |
| PQ (Product Quantized) | 8-64 bytes (configurable) | Cold tier. 50-400x compression. Lossy. |

---

## Timestamp Semantics

- Timestamps are `i64` representing **microseconds since Unix epoch**.
- Negative timestamps are valid (pre-1970 data for historical corpora).
- Timestamp resolution: microsecond. Sub-microsecond events at the same entity_id are rejected (collision).
- Timestamps must be monotonically increasing per entity. Out-of-order ingestion returns error with option to force (overwrite).

---

## REST Endpoints

| Method | Path | Description | Request Body | Response |
|---|---|---|---|---|
| POST | `/v1/ingest` | Batch insert | `{ points: [TemporalPoint] }` | `{ receipts: [WriteReceipt] }` |
| POST | `/v1/query` | Execute query | `QueryRequest` | `QueryResponse` |
| GET | `/v1/entities/{id}` | Entity timeline info | — | `EntityTimeline` |
| GET | `/v1/entities/{id}/trajectory` | Fetch trajectory | `?t1=&t2=` | `[TemporalPoint]` |
| GET | `/v1/entities/{id}/velocity` | Vector velocity | `?t=` | `{ velocity: [f32], magnitude: f32 }` |
| GET | `/v1/entities/{id}/changepoints` | List change points | `?t1=&t2=&method=` | `[ChangePoint]` |
| GET | `/v1/health` | Health check | — | `{ status, uptime, version }` |
| GET | `/v1/ready` | Readiness probe | — | 200 or 503 |
| POST | `/v1/admin/compact` | Trigger compaction | `{ tier: "hot_to_warm" }` | `{ status: "started" }` |
| GET | `/v1/admin/stats` | System statistics | — | `SystemStats` |

---

## gRPC Services

```
service ChronosVector {
  rpc IngestStream (stream TemporalPoint) returns (stream WriteReceipt);
  rpc Query (QueryRequest) returns (QueryResponse);
  rpc QueryStream (QueryRequest) returns (stream ScoredResult);
  rpc WatchDrift (WatchRequest) returns (stream DriftEvent);
}
```

---

## Error Codes

| HTTP | gRPC | Meaning |
|---|---|---|
| 400 | INVALID_ARGUMENT | Malformed request, dimension mismatch, invalid timestamp |
| 404 | NOT_FOUND | Entity not found |
| 409 | ALREADY_EXISTS | Duplicate (entity_id, timestamp) with different vector hash |
| 422 | FAILED_PRECONDITION | Insufficient data for requested operation (e.g., prediction with <5 points) |
| 429 | RESOURCE_EXHAUSTED | Rate limit exceeded |
| 500 | INTERNAL | Unexpected error |
| 503 | UNAVAILABLE | Not ready (WAL recovery in progress, index loading) |
