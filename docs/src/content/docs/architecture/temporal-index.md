---
title: Temporal Index Engine (ST-HNSW)
description: Architecture of the Spatiotemporal HNSW index engine, including the search algorithm, combined distance function, and Timestamp Graph (TANNS) integration.
---

## 7. Temporal Index Engine (ST-HNSW)

### 7.1 Index Architecture

```mermaid
graph TB
    subgraph STHNSW["ST-HNSW: Spatiotemporal HNSW"]
        subgraph Layers["HNSW Layers"]
            L3["Layer 3 (sparse)<br/>~0.1% of nodes<br/>Long-range links"]
            L2["Layer 2<br/>~1% of nodes"]
            L1["Layer 1<br/>~10% of nodes"]
            L0["Layer 0 (dense)<br/>100% of nodes<br/>Short-range links"]
        end

        subgraph Temporal["Temporal Extensions"]
            TSGM["Timestamp Graph<br/>Manager"]
            BITMAP_IDX["Roaring Bitmap<br/>per Time Range"]
            DECAY_MGR["Edge Decay<br/>Manager"]
            HNT["Historic Neighbor<br/>Tree Compressor"]
        end

        subgraph Distance["Distance Computation"]
            METRIC["Metric Dispatcher<br/>(trait object)"]
            COSINE["Cosine SIMD"]
            L2D["L2 SIMD"]
            DOT["Dot Product SIMD"]
            POINCARE["Poincaré Distance"]
        end
    end

    L3 --> L2 --> L1 --> L0
    L0 --> TSGM
    L0 --> BITMAP_IDX
    L0 --> DECAY_MGR
    TSGM --> HNT

    L0 --> METRIC
    METRIC --> COSINE
    METRIC --> L2D
    METRIC --> DOT
    METRIC --> POINCARE
```

### 7.2 Search Algorithm

```mermaid
sequenceDiagram
    participant QE as Query Engine
    participant HNSW as ST-HNSW
    participant BM as Roaring Bitmap
    participant DECAY as Decay Manager
    participant METRIC as Distance Metric
    participant STORE as Storage

    QE->>HNSW: search(query_vec, k, temporal_filter, alpha)

    HNSW->>BM: get_valid_set(temporal_filter)
    BM-->>HNSW: RoaringBitmap(valid_point_ids)

    Note over HNSW: Enter at top layer entry point

    loop For each layer L (top → bottom)
        Note over HNSW: Greedy search on layer L
        loop For each candidate neighbor
            HNSW->>BM: contains(neighbor_id)?
            alt neighbor is temporally valid
                HNSW->>METRIC: combined_distance(query, neighbor, alpha)
                Note over METRIC: d_combined = α·d_semantic + (1-α)·d_temporal
                METRIC->>STORE: load_vector(neighbor_id)
                STORE-->>METRIC: vector_data
                METRIC-->>HNSW: distance_score
                HNSW->>DECAY: get_edge_weight(current_node, neighbor, t_query)
                DECAY-->>HNSW: decay_factor
                Note over HNSW: Adjust score by decay factor
            end
        end
        Note over HNSW: Best candidate → entry point for next layer
    end

    Note over HNSW: Beam search on Layer 0 with ef_search candidates
    HNSW-->>QE: Vec<ScoredResult>[k]
```

### 7.3 Combined Distance Function

La distancia combinada es el corazón de ST-HNSW:

```
d_ST(query, candidate) = α · d_semantic(q.vector, c.vector)
                       + (1 - α) · d_temporal(q.timestamp, c.timestamp)
                       × decay(c.age)
```

Donde:
- `α ∈ [0, 1]` — peso relativo semántico vs temporal (configurable por query)
- `d_semantic` — coseno, L2, dot product o Poincaré según configuración
- `d_temporal` — `|q.timestamp - c.timestamp| / temporal_scale`
- `decay(age) = e^(-λ · age)` — penaliza conexiones antiguas

### 7.4 Timestamp Graph (TANNS Integration)

```mermaid
graph TB
    subgraph TSGraph["Timestamp Graph"]
        subgraph T1["t=1"]
            N1_1["Node A<br/>neighbors: B,C,D"]
        end
        subgraph T2["t=2"]
            N2_1["Node A<br/>neighbors: B,C,E"]
            N2_2["Node E (new)<br/>neighbors: A,B"]
        end
        subgraph T3["t=3"]
            N3_1["Node A<br/>neighbors: C,E,F"]
            N3_3["Node F (new)<br/>neighbors: A,E"]
        end

        subgraph HNT_Detail["Historic Neighbor Tree for Node A"]
            ROOT["[t=1] B,C,D"]
            L_CHILD["[t=2] B→E"]
            R_CHILD["[t=3] B→F, D→∅"]
        end
    end

    T1 --> T2 --> T3
    ROOT --> L_CHILD
    ROOT --> R_CHILD

    style HNT_Detail fill:#f0f0ff,stroke:#333
```

El HNT almacena los cambios en las listas de vecinos como un árbol binario de diffs, evitando duplicación de vecinos estables.
