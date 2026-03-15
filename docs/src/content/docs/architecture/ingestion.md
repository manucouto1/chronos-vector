---
title: Ingestion Pipeline
description: Detailed design of ChronosVector's ingestion pipeline including delta encoding strategy, WAL architecture, and BOCPD online monitoring.
---

## 6. Ingestion Pipeline

### 6.1 Pipeline Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as API Gateway
    participant VAL as Validator
    participant WAL as Write-Ahead Log
    participant DELTA as Delta Encoder
    participant BOCPD as BOCPD Monitor
    participant IDX as ST-HNSW Index
    participant STORE as Hot Store

    Client->>API: IngestRequest(entity_id, timestamp, vector, metadata)
    API->>VAL: validate(request)

    alt Validation Fails
        VAL-->>API: Err(ValidationError)
        API-->>Client: 400 Bad Request
    end

    VAL->>WAL: append(validated_point)
    WAL-->>VAL: WALOffset

    par Parallel Processing
        VAL->>DELTA: compute_delta(entity_id, new_vector)
        Note over DELTA: Fetch previous vector<br/>Compute Δv = v_new - v_prev<br/>If ‖Δv‖ < ε → skip storage<br/>Every K updates → store keyframe

        VAL->>BOCPD: update(entity_id, new_vector, timestamp)
        Note over BOCPD: Update run-length posterior<br/>If P(changepoint) > threshold<br/>→ emit ChangePointEvent
    end

    DELTA->>STORE: put(entity_id, timestamp, vector_or_delta)
    DELTA->>IDX: insert(point_id, vector, timestamp)

    IDX-->>DELTA: IndexReceipt
    STORE-->>DELTA: StorageReceipt

    DELTA-->>API: WriteReceipt(wal_offset, indexed, stored)
    API-->>Client: 200 OK (WriteReceipt)
```

### 6.2 Delta Encoding Strategy

```mermaid
graph LR
    subgraph Timeline["Entity Timeline on Disk"]
        KF0["KF₀<br/>Keyframe<br/>Full Vector<br/>t=0"]
        D1["Δ₁<br/>Sparse Delta<br/>t=1"]
        D2["Δ₂<br/>Sparse Delta<br/>t=2"]
        D3["Δ₃<br/>Sparse Delta<br/>t=3"]
        KF1["KF₁<br/>Keyframe<br/>Full Vector<br/>t=4"]
        D4["Δ₄<br/>Sparse Delta<br/>t=5"]
        D5["Δ₅<br/>Sparse Delta<br/>t=6"]
    end

    KF0 --> D1 --> D2 --> D3 --> KF1 --> D4 --> D5

    subgraph Reconstruction["Reconstruction v(t=3)"]
        R1["Read KF₀"]
        R2["+ Δ₁"]
        R3["+ Δ₂"]
        R4["+ Δ₃"]
        R5["= v(t=3)"]
    end

    R1 --> R2 --> R3 --> R4 --> R5
```

**Reglas del Delta Encoder:**

1. **Threshold ε:** Si `‖Δv‖ < ε`, el punto se descarta (el concepto no ha cambiado significativamente). Configurable por entity o globalmente.
2. **Keyframe Interval K:** Cada K deltas, se almacena el vector completo. Limita la cadena de reconstrucción a máximo K lecturas.
3. **Sparse Encoding:** Los deltas se almacenan como `(indices[], values[])` — solo las dimensiones que cambiaron por encima de un micro-threshold.
4. **Content Hash:** Cada delta se hashea (xxhash) para deduplicación. Si dos entidades producen el mismo delta, se almacena una sola vez.

### 6.3 Write-Ahead Log (WAL)

```mermaid
graph LR
    subgraph WAL["Write-Ahead Log (Append-Only)"]
        SEG1["Segment 0<br/>0..64MB"]
        SEG2["Segment 1<br/>64..128MB"]
        SEG3["Segment 2<br/>128..192MB<br/>(active)"]
    end

    subgraph State["WAL State"]
        HEAD["Head Offset<br/>(last written)"]
        COMMIT["Commit Offset<br/>(last flushed to store)"]
        TAIL["Tail Offset<br/>(safe to truncate)"]
    end

    SEG1 -.->|truncated after commit| TAIL
    SEG3 -->|append| HEAD
    SEG2 -.->|flushing| COMMIT
```

El WAL garantiza durabilidad: si el proceso se cae entre la escritura al WAL y la actualización del índice/store, el recovery re-aplica las entradas pendientes desde `COMMIT` hasta `HEAD`.
