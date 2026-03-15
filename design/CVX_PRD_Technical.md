# ChronosVector — Technical Product Requirements Document (PRD)

**Version:** 1.0  
**Author:** Manuel Couto Pintos  
**Date:** March 2026  
**Status:** Draft

---

## 1. Product Definition

### 1.1 One-Line Description

ChronosVector es una base de datos vectorial temporal en Rust que trata el tiempo como una dimensión geométrica del espacio de embeddings, permitiendo búsquedas espacio-temporales, análisis de drift semántico y predicción de trayectorias vectoriales.

### 1.2 Target Users

| User Persona | Use Case | Valor Principal |
|---|---|---|
| **ML Engineer** | Monitorizar drift de embeddings en producción | Alertas cuando un modelo se degrada por cambio en distribución de datos |
| **NLP Researcher** | Estudiar evolución semántica diacrónica | Queries de trayectoria, analogía temporal, velocidad de cambio |
| **Recommender System Developer** | Predecir hacia dónde evolucionan los intereses de usuarios | Extrapolación de vectores de usuario via Neural ODE |
| **Knowledge Graph Engineer** | Temporal knowledge graph completion | Cuadrupletos (entity, relation, entity, time) indexados nativamente |
| **Data Scientist** | Análisis exploratorio de series de embeddings | Detección de change points, drift quantification, cohort divergence |

### 1.3 Non-Goals (Explicit Exclusions)

- **Not a general-purpose vector database.** CVX no compite con Qdrant/Milvus/Pinecone en kNN puro sin componente temporal. Si el usuario no necesita tiempo, debería usar Qdrant.
- **Not a streaming platform.** CVX recibe vectores, no los produce. No incluye modelos de embedding.
- **Not a model training framework.** El Neural ODE se entrena externamente o con un módulo auxiliar; CVX es primariamente infraestructura de servicio (inference + storage).
- **Not a distributed database (initially).** Phase 1-4 son single-node. La distribución es Phase 5.

---

## 2. Functional Requirements

### 2.1 Core Operations

#### FR-01: Temporal Vector Ingestion

| Attribute | Specification |
|---|---|
| **Input** | `(entity_id: u64, timestamp: i64, vector: [f32; D], metadata: Map<String, Value>)` |
| **Protocols** | REST (batch), gRPC (bidirectional stream) |
| **Throughput target** | ≥ 50,000 vectors/second (single node, D=768) |
| **Latency target** | p99 < 5ms per vector (streaming mode) |
| **Durability** | Write-ahead log fsync antes de ack. No data loss on crash. |
| **Validation** | Dimension consistency per entity, timestamp monotonically increasing per entity, vector norm finite |
| **Delta encoding** | Automatic. Keyframe every K=10 updates. Configurable threshold ε. |
| **Idempotency** | Re-insert same (entity_id, timestamp) is no-op if vector hash matches; update if different. |

#### FR-02: Snapshot kNN Query

| Attribute | Specification |
|---|---|
| **Input** | `(query_vector, k, timestamp, metric, alpha)` |
| **Output** | Top-k results sorted by combined spatiotemporal distance |
| **Latency target** | p99 < 10ms (1M vectors, D=768) |
| **Recall target** | ≥ 95% at 10-recall@10 |
| **Metrics supported** | Cosine, L2, Dot Product, Poincaré (hyperbolic) |
| **Alpha range** | `[0.0, 1.0]` — 0.0 = pure temporal, 1.0 = pure semantic |

#### FR-03: Range kNN Query

| Attribute | Specification |
|---|---|
| **Input** | `(query_vector, k, time_range: [t1, t2], metric, alpha)` |
| **Output** | Top-k results within time range |
| **Behavior** | Pre-filter by Roaring Bitmap, then search within valid set |

#### FR-04: Trajectory Retrieval

| Attribute | Specification |
|---|---|
| **Input** | `(entity_id, time_range: [t1, t2])` |
| **Output** | Ordered sequence of `TemporalPoint` for that entity within range |
| **Reconstruction** | Transparent delta decoding. Caller receives full vectors. |
| **Latency** | Proportional to trajectory length. Target: < 1ms + 0.1ms per point. |

#### FR-05: Vector Velocity & Acceleration

| Attribute | Specification |
|---|---|
| **Input** | `(entity_id, timestamp)` |
| **Output** | Velocity vector `∂v/∂t` and optionally acceleration `∂²v/∂t²` |
| **Method** | Finite differences over stored deltas (first-order: central difference; second-order: second central difference) |
| **Edge case** | If entity has < 2 points: return error `InsufficientData` |

#### FR-06: Trajectory Prediction (Extrapolation)

| Attribute | Specification |
|---|---|
| **Input** | `(entity_id, target_timestamp, confidence_level)` |
| **Output** | `PredictedPoint { vector, confidence_interval, uncertainty_per_dimension }` |
| **Method** | Neural ODE solver (Dormand-Prince RK45) with learned f_θ |
| **Fallback** | If Neural ODE not available/trained: linear extrapolation from last two points |
| **Cold start** | Entity needs ≥ 5 historical points for Neural ODE. Below that: linear only. |

#### FR-07: Change Point Detection

| Attribute | Specification |
|---|---|
| **Input (offline)** | `(entity_id, time_range, method: PELT, sensitivity)` |
| **Input (online)** | Automatic during ingestion. Configurable per-entity or global. |
| **Output** | `Vec<ChangePoint { timestamp, severity, drift_vector, method }>` |
| **PELT specifics** | Penalty: BIC (default) or AIC. Minimum segment length configurable. |
| **BOCPD specifics** | Hazard function: constant (default). Prior: Normal-InverseGamma. Threshold configurable. |

#### FR-08: Temporal Analogy Query

| Attribute | Specification |
|---|---|
| **Input** | `(reference_entity, t_reference, t_target, k)` |
| **Semantics** | "What entities at t_target occupied the same semantic role as reference at t_reference?" |
| **Method** | Compute displacement, project query, run snapshot kNN at t_target |

#### FR-09: Drift Quantification

| Attribute | Specification |
|---|---|
| **Input** | `(entity_id, t1, t2)` |
| **Output** | `DriftReport { magnitude, direction_cosines, affected_dimensions, rate_per_unit_time }` |
| **Metrics** | Cosine distance, L2 distance, angular displacement |

#### FR-10: Cohort Divergence Detection

| Attribute | Specification |
|---|---|
| **Input** | `(entity_a, entity_b, time_range)` |
| **Output** | Pairwise distance time series + detected divergence points (via PELT on the distance series) |

### 2.2 Administrative Operations

#### FR-11: Tier Management

- Manual trigger for compaction: `POST /v1/admin/compact`
- View tier statistics: `GET /v1/admin/stats`
- Configure tier thresholds at runtime: `PUT /v1/admin/config`

#### FR-12: Index Management

- Rebuild index from storage: `POST /v1/admin/reindex`
- Get index statistics (node count, edge count, layer distribution): `GET /v1/admin/index/stats`

#### FR-13: Health & Readiness

- Health check: `GET /v1/health` → `{ status: "ok", uptime, version }`
- Readiness probe: `GET /v1/ready` → 200 when WAL recovery complete and index loaded

### 2.3 Monitoring

#### FR-14: Drift Watch (Subscription)

| Attribute | Specification |
|---|---|
| **Protocol** | gRPC server-streaming |
| **Input** | `WatchRequest { entity_filter: Option<Vec<u64>>, min_severity: f64 }` |
| **Output** | Stream of `DriftEvent { entity_id, timestamp, severity, drift_vector }` |
| **Behavior** | Client subscribes; server pushes events as BOCPD detects them |

---

## 3. Non-Functional Requirements

### 3.1 Performance

| Metric | Target | Measurement Method |
|---|---|---|
| Ingest throughput | ≥ 50K vectors/sec (D=768, single node) | Benchmark with synthetic stream |
| Snapshot kNN latency p50 | < 2ms (1M vectors) | Benchmark with random queries |
| Snapshot kNN latency p99 | < 10ms (1M vectors) | Same as above |
| Trajectory retrieval | < 1ms + 0.1ms/point | Benchmark with varying trajectory lengths |
| Prediction latency | < 50ms (single entity) | Benchmark with trained Neural ODE |
| CPD (PELT) offline | < 1s for 100K-point trajectory | Benchmark on synthetic + real data |
| Cold start (empty → serving) | < 5s (1M vectors pre-loaded) | Measure from process start to first query |

### 3.2 Scalability

| Dimension | Phase 1-4 Target | Phase 5 Target |
|---|---|---|
| Total vectors | 100M (single node) | 10B (distributed) |
| Entities | 10M | 1B |
| Vector dimensions | Up to 4096 | Same |
| Concurrent queries | 1000 QPS | 10,000 QPS (cluster) |
| Concurrent ingest streams | 100 | 1000 (cluster) |

### 3.3 Durability & Consistency

- **Durability:** WAL fsync before ack. Crash recovery replays WAL from last committed offset.
- **Consistency model:** Single-node: linearizable (single writer to index + storage). Distributed: eventual consistency for reads from followers; linearizable for writes via Raft leader.
- **Data loss window:** Zero (WAL is authoritative).

### 3.4 Availability

- **Single node:** Process crash → automatic restart via supervisor (systemd). Recovery time < cold start time.
- **Distributed (Phase 5):** Raft-based replication. Tolerate 1 node failure per shard (3-replica minimum).

### 3.5 Operability

- **Configuration:** Single TOML file + environment variable overrides.
- **Observability:** Prometheus metrics endpoint, OpenTelemetry traces, structured JSON logging.
- **Upgrade:** Graceful shutdown drains in-flight requests. Storage format versioned for backward compatibility.
- **Backup:** Hot backup via RocksDB checkpoint. Cold tier already on object store (inherently backed up).

---

## 4. Data Requirements

### 4.1 Supported Vector Types

| Type | Storage Size (D=768) | Use Case |
|---|---|---|
| FP32 | 3072 bytes | Default. Full precision for hot tier. |
| FP16 | 1536 bytes | Warm tier option for 2x compression with <1% recall loss. |
| INT8 (Scalar Quantized) | 768 bytes | 4x compression. Good for large-scale with moderate accuracy needs. |
| PQ (Product Quantized) | 8-64 bytes (configurable) | Cold tier. 50-400x compression. Lossy. |

### 4.2 Metadata Schema

Metadata is schemaless (arbitrary string-to-value map). Common expected fields:

```
{
  "source": "model_v3",
  "domain": "medical",
  "language": "en",
  "confidence": 0.95,
  "tags": ["cardiology", "ecg"]
}
```

Metadata is stored but **not indexed** in Phase 1-4. Metadata-based filtering (e.g., "kNN where domain=medical") is deferred.

### 4.3 Timestamp Semantics

- Timestamps are `i64` representing **microseconds since Unix epoch**.
- Negative timestamps are valid (pre-1970 data for historical corpora).
- Timestamp resolution: microsecond. Sub-microsecond events at the same entity_id are rejected (collision).
- Timestamps must be monotonically increasing per entity. Out-of-order ingestion returns error with option to force (overwrite).

---

## 5. API Contract Summary

### 5.1 REST Endpoints

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

### 5.2 gRPC Services

```
service ChronosVector {
  rpc IngestStream (stream TemporalPoint) returns (stream WriteReceipt);
  rpc Query (QueryRequest) returns (QueryResponse);
  rpc QueryStream (QueryRequest) returns (stream ScoredResult);
  rpc WatchDrift (WatchRequest) returns (stream DriftEvent);
}
```

### 5.3 Error Codes

| HTTP | gRPC | Meaning |
|---|---|---|
| 400 | INVALID_ARGUMENT | Malformed request, dimension mismatch, invalid timestamp |
| 404 | NOT_FOUND | Entity not found |
| 409 | ALREADY_EXISTS | Duplicate (entity_id, timestamp) with different vector hash |
| 422 | FAILED_PRECONDITION | Insufficient data for requested operation (e.g., prediction with <5 points) |
| 429 | RESOURCE_EXHAUSTED | Rate limit exceeded |
| 500 | INTERNAL | Unexpected error |
| 503 | UNAVAILABLE | Not ready (WAL recovery in progress, index loading) |

---

## 6. Constraints & Assumptions

### 6.1 Constraints

1. **Single developer initially.** Architecture must be implementable incrementally by one person.
2. **No cloud vendor lock-in.** Object store interface via `object_store` crate abstracts S3/GCS/Azure/local.
3. **Rust stable only (no nightly).** Except for explicit SIMD if `std::simd` stabilizes.
4. **No Python dependency in runtime.** `burn` and `candle` are pure Rust. No PyTorch/ONNX runtime.

### 6.2 Assumptions

1. Embedding dimensions are fixed per entity after first insertion. Changing dimensions requires re-ingestion.
2. Timestamps are provider-assigned (not server-assigned). The system trusts the producer's clock.
3. Vectors are normalized or unnormalized depending on the metric. CVX does not auto-normalize.
4. The Neural ODE model ($f_\theta$) is trained offline and loaded at startup. Online training is out of scope for Phase 1-4.

---

## 7. Success Criteria

### 7.1 Portfolio Value (Primary Goal)

| Criterion | Evidence |
|---|---|
| Demonstrates advanced Rust proficiency | Unsafe SIMD kernels, async concurrency, trait-based architecture, zero-copy serialization |
| Solves a real, unmet need | No existing VDB treats time as a geometric dimension with drift analysis |
| Publishable as technical work | Sufficient novelty for a systems paper (ST-HNSW + delta encoding + Neural ODE integration) |
| Usable by others | Clean API, documentation, runnable benchmarks |

### 7.2 Technical Milestones

| Milestone | Definition of Done |
|---|---|
| M1: First kNN query | Ingest 1M vectors, execute snapshot kNN, recall ≥ 90% |
| M2: Temporal queries work | Range kNN, trajectory retrieval, velocity computation all passing integration tests |
| M3: Delta encoding saves storage | Measurable ≥ 3x storage reduction on real embedding dataset (e.g., Wikipedia temporal) |
| M4: PELT detects known change points | On synthetic data with planted change points, F1 ≥ 0.85 |
| M5: Neural ODE predicts | On held-out trajectory data, prediction error < linear extrapolation baseline |
| M6: API is production-ready | REST + gRPC serving, health checks, graceful shutdown, structured logging |

---

## 8. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| ST-HNSW composite distance hurts recall | Medium | High | Benchmark α=1.0 (pure semantic) against vanilla HNSW as baseline. If recall drops >5%, revisit ADR-002. |
| Delta encoding reconstruction too slow | Low | Medium | Tune keyframe interval K. Worst case: disable deltas and store full vectors. |
| Neural ODE training data insufficient for useful predictions | Medium | Low | Linear extrapolation as fallback is always available. Neural ODE is a bonus, not a requirement. |
| RocksDB write amplification causes SSD wear | Low | Medium | Monitor write amplification ratio. Tune compaction. Use leveled compaction for read-heavy workloads. |
| Scope creep into distributed mode too early | High | High | Strict phase gating. No distributed code until single-node milestones M1-M6 are met. |
