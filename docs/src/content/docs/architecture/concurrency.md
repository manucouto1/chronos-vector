---
title: Concurrency Model
description: Thread architecture, index concurrency with concurrent readers/single writer, and the Tokio runtime layout for ChronosVector.
---

## 12. Concurrency Model

### 12.1 Thread Architecture

```mermaid
graph TB
    subgraph TokioRuntime["Tokio Runtime (async)"]
        subgraph IO["I/O Threads"]
            NET1["Network I/O<br/>Accept connections"]
            NET2["Network I/O<br/>Accept connections"]
            DISK_IO["Disk I/O Pool<br/>RocksDB / Parquet reads"]
        end

        subgraph Workers["Worker Threads"]
            W1["Query Worker 1"]
            W2["Query Worker 2"]
            W3["Query Worker N"]
        end

        subgraph Background["Background Tasks"]
            COMPACTOR["Compaction Task<br/>(periodic)"]
            DRIFT_MON["Drift Monitor Task<br/>(per-entity BOCPD)"]
            METRIC_FLUSH["Metrics Flush<br/>(periodic)"]
            CODEBOOK["Codebook Retrain<br/>(periodic)"]
        end
    end

    subgraph Dedicated["Dedicated Threads (non-Tokio)"]
        INGEST_BATCH["Ingest Batch Processor<br/>(CPU-bound: delta encoding)"]
        INDEX_WRITER["Index Writer<br/>(single writer, RwLock)"]
        ODE_COMPUTE["ODE Compute Pool<br/>(CPU-bound: SIMD math)"]
    end

    NET1 --> W1
    NET2 --> W2
    W1 -->|spawn_blocking| INGEST_BATCH
    W2 -->|spawn_blocking| ODE_COMPUTE
    INGEST_BATCH --> INDEX_WRITER
```

### 12.2 Index Concurrency

El índice ST-HNSW usa un esquema de **concurrent readers, single writer**:

```mermaid
graph TB
    subgraph Readers["Concurrent Readers (N threads)"]
        R1["Search Thread 1<br/>RwLock::read()"]
        R2["Search Thread 2<br/>RwLock::read()"]
        R3["Search Thread N<br/>RwLock::read()"]
    end

    subgraph Writer["Single Writer"]
        IW["Index Writer Thread<br/>RwLock::write()"]
    end

    subgraph Index["ST-HNSW Graph"]
        NODES["Node Array<br/>(append-only Vec)"]
        EDGES["Edge Lists<br/>(per-node RwLock)"]
        ENTRY["Entry Point<br/>(AtomicU64)"]
    end

    R1 -->|read lock| EDGES
    R2 -->|read lock| EDGES
    R3 -->|read lock| EDGES
    IW -->|write lock on single node| EDGES
    IW -->|CAS| ENTRY
    IW -->|push| NODES
```

**Estrategia:** Las inserciones adquieren write lock solo en las listas de vecinos de los nodos afectados (no del grafo completo), permitiendo que las búsquedas continúen en paralelo sobre el resto del grafo. El entry point se actualiza atómicamente vía CAS (Compare-and-Swap).
