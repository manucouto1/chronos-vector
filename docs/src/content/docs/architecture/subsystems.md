---
title: Subsystem Decomposition
description: Detailed decomposition of ChronosVector's 6 principal subsystems, their responsibilities, and interface contracts.
---

## 4. Subsystem Decomposition

ChronosVector se descompone en 6 subsistemas principales, cada uno con responsabilidades claras y contratos de interfaz definidos.

```mermaid
graph TB
    subgraph S1["S1: API Gateway"]
        S1A[Request Routing]
        S1B[Auth & Rate Limiting]
        S1C[Protocol Translation<br/>REST ↔ gRPC ↔ Internal]
        S1D[Response Serialization]
    end

    subgraph S2["S2: Ingestion Engine"]
        S2A[Stream Demux]
        S2B[Schema Validation]
        S2C[Delta Computation]
        S2D[Write-Ahead Log]
        S2E[BOCPD Monitor]
    end

    subgraph S3["S3: Temporal Index"]
        S3A[ST-HNSW Graph]
        S3B[Timestamp Graph]
        S3C[Roaring Bitmap Index]
        S3D[Time-Decay Manager]
        S3E[Neighbor Compressor HNT]
    end

    subgraph S4["S4: Tiered Storage"]
        S4A[Hot Store - RocksDB]
        S4B[Warm Store - Parquet]
        S4C[Cold Store - Object Store]
        S4D[Compactor / Migrator]
        S4E[PQ Codebook Manager]
    end

    subgraph S5["S5: Query Engine"]
        S5A[Query Planner]
        S5B[Snapshot Executor]
        S5C[Range Executor]
        S5D[Trajectory Executor]
        S5E[Analytic Executor]
    end

    subgraph S6["S6: Analytics Engine"]
        S6A[Neural ODE Solver]
        S6B[PELT - Offline CPD]
        S6C[BOCPD - Online CPD]
        S6D[Vector Differential Calculus]
        S6E[Drift Quantifier]
    end

    S1 --> S2
    S1 --> S5
    S2 --> S3
    S2 --> S4
    S5 --> S3
    S5 --> S4
    S5 --> S6
    S6 --> S4
    S3 --> S4
```

### Responsabilidades por Subsistema

| Subsistema | Responsabilidad Principal | Interfaces Expuestas |
|---|---|---|
| **S1: API Gateway** | Punto de entrada único. Traduce protocolos externos a comandos internos | `IngestCommand`, `QueryRequest`, `AdminCommand` |
| **S2: Ingestion Engine** | Valida, normaliza, computa deltas, persiste y actualiza el índice de forma atómica | `ingest(batch)` → `WriteReceipt` |
| **S3: Temporal Index** | Estructura de indexación espacio-temporal. Resuelve kNN con constraints temporales | `search(query, temporal_filter)` → `Vec<ScoredResult>` |
| **S4: Tiered Storage** | Almacenamiento multi-temperatura. Gestiona ciclo de vida de datos | `get(id, t)`, `put(id, t, vec)`, `range(id, t1..t2)` |
| **S5: Query Engine** | Planifica y ejecuta queries complejas componiendo operaciones de S3, S4 y S6 | `execute(QueryPlan)` → `QueryResult` |
| **S6: Analytics Engine** | Operaciones analíticas: predicción, CPD, cálculo diferencial vectorial | `predict(id, t_future)`, `detect_changes(id, window)` |
