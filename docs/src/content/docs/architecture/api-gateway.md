---
title: API Gateway
description: REST and gRPC API endpoints, protobuf schema definitions, and protocol translation for ChronosVector's API Gateway.
---

## 11. API Gateway

### 11.1 API Endpoints

```mermaid
graph TB
    subgraph REST["REST API (Axum)"]
        direction TB
        POST_INGEST["POST /v1/ingest<br/>Batch insert vectors"]
        POST_QUERY["POST /v1/query<br/>Execute query"]
        GET_ENTITY["GET /v1/entities/:id<br/>Entity timeline info"]
        GET_TRAJ["GET /v1/entities/:id/trajectory?t1=&t2=<br/>Fetch trajectory"]
        GET_HEALTH["GET /v1/health<br/>Health check"]
        POST_ADMIN["POST /v1/admin/compact<br/>Trigger compaction"]
    end

    subgraph GRPC["gRPC API (Tonic)"]
        direction TB
        INGEST_STREAM["IngestStream (bidirectional)<br/>Stream vectors → receipts"]
        QUERY_STREAM["QueryStream (server-stream)<br/>Request → stream of results"]
        WATCH_DRIFT["WatchDrift (server-stream)<br/>Subscribe to drift events"]
    end

    subgraph Proto["Protobuf Definitions"]
        direction TB
        POINT_PROTO["TemporalPoint message"]
        QUERY_PROTO["QueryRequest message"]
        RESULT_PROTO["QueryResult message"]
        EVENT_PROTO["DriftEvent message"]
    end

    REST --> Proto
    GRPC --> Proto
```

### 11.2 Protobuf Schema (Simplified)

```protobuf
// cvx_api.proto

service ChronosVector {
  // Ingestion
  rpc IngestBatch (IngestRequest) returns (IngestResponse);
  rpc IngestStream (stream TemporalPoint) returns (stream WriteReceipt);

  // Queries
  rpc Query (QueryRequest) returns (QueryResponse);
  rpc QueryStream (QueryRequest) returns (stream ScoredResult);

  // Monitoring
  rpc WatchDrift (WatchRequest) returns (stream DriftEvent);
}

message TemporalPoint {
  uint64 entity_id = 1;
  int64  timestamp  = 2;
  repeated float vector = 3;
  map<string, string> metadata = 4;
}

message QueryRequest {
  QueryType type = 1;
  repeated float query_vector = 2;
  TemporalFilter temporal = 3;
  uint32 k = 4;
  float  alpha = 5;           // semantic vs temporal weight
  string metric = 6;          // "cosine" | "l2" | "dot" | "poincare"
  PredictionParams prediction = 7;
}

message TemporalFilter {
  oneof filter {
    int64 at_timestamp = 1;           // snapshot
    TimeRange range = 2;              // range query
    int64 predict_to = 3;             // extrapolation target
  }
}
```
