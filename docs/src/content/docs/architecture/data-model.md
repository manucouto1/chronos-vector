---
title: Data Model
description: Core entities, key schema, and quantized representations used in ChronosVector's temporal vector data model.
---

## 5. Data Model

### 5.1 Core Entities

```mermaid
classDiagram
    class TemporalPoint {
        +PointId: u64
        +EntityId: u64
        +Timestamp: i64
        +Vector: Vec~f32~
        +Metadata: HashMap~String, Value~
        +version(): u32
        +dim(): usize
    }

    class DeltaEntry {
        +PointId: u64
        +EntityId: u64
        +BaseTimestamp: i64
        +DeltaTimestamp: i64
        +DeltaVector: SparseVec~f32~
        +is_keyframe(): bool
    }

    class EntityTimeline {
        +EntityId: u64
        +FirstSeen: i64
        +LastSeen: i64
        +PointCount: u32
        +KeyframeInterval: u32
        +get_at(t: i64): TemporalPoint
        +get_range(t1: i64, t2: i64): Vec~TemporalPoint~
        +velocity_at(t: i64): Vec~f32~
    }

    class ChangePoint {
        +EntityId: u64
        +Timestamp: i64
        +Severity: f64
        +DriftVector: Vec~f32~
        +DetectionMethod: CPDMethod
    }

    class ScoredResult {
        +Point: TemporalPoint
        +SemanticDistance: f32
        +TemporalDistance: f32
        +CombinedScore: f32
    }

    TemporalPoint "1..*" --o EntityTimeline : composes
    DeltaEntry "0..*" --o EntityTimeline : compresses
    ChangePoint "0..*" --o EntityTimeline : annotates
    TemporalPoint --> ScoredResult : wraps in query results
```

### 5.2 Key Schema

Las claves en RocksDB siguen un esquema compuesto que permite range scans eficientes:

```
Column Family: vectors
  Key:   [entity_id: u64][timestamp: i64 BE]
  Value: [vector_data: [f32; D]] (rkyv serialized)

Column Family: deltas
  Key:   [entity_id: u64][delta_timestamp: i64 BE]
  Value: [base_timestamp: i64][sparse_indices: Vec<u32>][sparse_values: Vec<f32>]

Column Family: metadata
  Key:   [entity_id: u64][timestamp: i64 BE]
  Value: [metadata_map: HashMap<String, Value>] (rkyv serialized)

Column Family: timelines
  Key:   [entity_id: u64]
  Value: [first_seen: i64][last_seen: i64][point_count: u32][keyframe_interval: u32]

Column Family: changepoints
  Key:   [entity_id: u64][timestamp: i64 BE]
  Value: [severity: f64][drift_vector: Vec<f32>][method: u8]
```

Las claves usan Big-Endian para timestamps para que el orden lexicográfico de bytes coincida con el orden temporal (crucial para range scans eficientes en RocksDB).

### 5.3 Quantized Representations

```mermaid
classDiagram
    class QuantizationLevel {
        <<enumeration>>
        Full_FP32
        Half_FP16
        Scalar_INT8
        ProductQuantized_PQ
        BinaryQuantized_BQ
    }

    class PQCodebook {
        +NumSubspaces: u32
        +CentroidsPerSubspace: u32
        +Codebooks: Vec~Vec~Vec~f32~~~
        +encode(vector: Vec~f32~): PQCode
        +decode(code: PQCode): Vec~f32~
        +asymmetric_distance(query: Vec~f32~, code: PQCode): f32
    }

    class PQCode {
        +Indices: Vec~u8~
        +bytes(): usize
    }

    PQCodebook --> PQCode : produces
    QuantizationLevel --> PQCodebook : configures
```
