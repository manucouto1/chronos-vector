---
title: System Overview & Architecture Principles
description: ChronosVector system overview, architecture principles, and high-level system architecture diagram showing all layers and their interactions.
---

import { Aside } from '@astrojs/starlight/components';

## 1. System Overview

ChronosVector es una plataforma de datos vectoriales donde el tiempo es un ciudadano de primera clase. El sistema recibe streams de embeddings con marca temporal, los indexa en una estructura espacio-temporal, los almacena en capas de temperatura variable, y expone un motor de queries que permite desde búsquedas kNN clásicas hasta predicción de trayectorias futuras y detección de drift semántico.

```mermaid
graph TB
    subgraph External["External Systems"]
        P1[Embedding Producers]
        P2[ML Pipelines]
        P3[Application Clients]
    end

    subgraph CVX["ChronosVector Core"]
        API[API Gateway<br/>REST + gRPC]
        IE[Ingestion Engine]
        QE[Query Engine]
        TI[Temporal Index<br/>ST-HNSW]
        TS[Tiered Storage<br/>Hot / Warm / Cold]
        AE[Analytics Engine<br/>Neural ODE + CPD]
        OBS[Observability<br/>Metrics + Traces]
    end

    P1 -->|gRPC Stream| API
    P2 -->|Batch Insert| API
    P3 -->|Query| API

    API --> IE
    API --> QE
    IE --> TI
    IE --> TS
    QE --> TI
    QE --> TS
    QE --> AE
    AE --> TS
    TI --> TS

    OBS -.->|monitors| IE
    OBS -.->|monitors| QE
    OBS -.->|monitors| TI
    OBS -.->|monitors| TS
```

---

## 2. Architecture Principles

| Principio | Descripción | Impacto en Diseño |
|---|---|---|
| **Time as Geometry** | El tiempo no es un filtro; es una dimensión del espacio de búsqueda | El índice combina distancia semántica y temporal nativamente |
| **Zero-Copy Pipeline** | Los datos atraviesan el sistema con cero copias innecesarias | `rkyv` para serialización zero-copy, `bytes::Bytes` para buffers compartidos |
| **Tiered by Temperature** | Los datos migran automáticamente según su "calor" (recency de acceso) | Hot (RAM+LSM), Warm (Parquet), Cold (Object Store + PQ) |
| **Compute Near Data** | Las operaciones analíticas se ejecutan donde residen los datos | SIMD en hot path, polars en warm, chunked reads en cold |
| **Separation of Index & Storage** | El índice (grafo) y los vectores viven en subsistemas independientes | Permite reindexar sin mover datos y viceversa |
| **Pluggable Metrics** | La métrica de distancia es un trait, no hardcoded | Soporta coseno, L2, dot product, Poincaré hiperbólico |
| **Fail Loud, Recover Gracefully** | Los errores se propagan explícitamente; la recuperación es automática | `Result<T, CvxError>` en todas las interfaces internas |

---

## 3. High-Level System Architecture

```mermaid
graph LR
    subgraph Ingest["Ingestion Layer"]
        direction TB
        GW_IN[gRPC Ingest<br/>Stream Receiver]
        REST_IN[REST Ingest<br/>Batch Receiver]
        VAL[Validator &<br/>Normalizer]
        DELTA[Delta<br/>Encoder]
        DRIFT_IN[Online Drift<br/>Monitor BOCPD]
    end

    subgraph Index["Index Layer"]
        direction TB
        STHNSW[ST-HNSW<br/>Temporal Graph]
        TSGRAPH[Timestamp<br/>Graph Manager]
        BITMAP[Roaring Bitmap<br/>Temporal Filter]
        DECAY[Time-Decay<br/>Edge Manager]
    end

    subgraph Storage["Storage Layer"]
        direction TB
        HOT[Hot Store<br/>RocksDB + RAM]
        WARM[Warm Store<br/>Parquet / Arrow]
        COLD[Cold Store<br/>Object Store + PQ]
        COMPACT[Compaction &<br/>Tier Migration]
    end

    subgraph Query["Query Layer"]
        direction TB
        PARSER[Query Parser<br/>& Planner]
        SNAP[Snapshot kNN<br/>Executor]
        TRAJ[Trajectory<br/>Executor]
        PRED[Prediction<br/>Executor]
        CPD[Change Point<br/>Executor]
    end

    subgraph Analytics["Analytics Layer"]
        direction TB
        ODE[Neural ODE<br/>Solver RK45]
        PELT[PELT Offline<br/>Change Point]
        BOCPD[BOCPD Online<br/>Change Point]
        DERIV[Vector Calculus<br/>Velocity / Accel]
    end

    subgraph API_Layer["API Layer"]
        direction TB
        AXUM[Axum HTTP<br/>REST API]
        TONIC[Tonic gRPC<br/>Streaming API]
        PROTO[Protobuf<br/>Schema]
    end

    AXUM --> PARSER
    TONIC --> GW_IN
    TONIC --> PARSER
    REST_IN --> VAL
    GW_IN --> VAL
    VAL --> DELTA
    DELTA --> HOT
    DELTA --> STHNSW
    VAL --> DRIFT_IN

    PARSER --> SNAP
    PARSER --> TRAJ
    PARSER --> PRED
    PARSER --> CPD

    SNAP --> STHNSW
    SNAP --> HOT
    SNAP --> WARM
    TRAJ --> HOT
    TRAJ --> WARM
    PRED --> ODE
    CPD --> PELT
    CPD --> BOCPD

    STHNSW --> TSGRAPH
    STHNSW --> BITMAP
    STHNSW --> DECAY

    HOT --> WARM
    WARM --> COLD
    COMPACT --> HOT
    COMPACT --> WARM
    COMPACT --> COLD

    ODE --> HOT
    PELT --> WARM
    DERIV --> HOT
```
