---
title: Deployment Topologies
description: Single-node and distributed deployment topologies for ChronosVector, including sharding strategy and Raft-based replication.
---

## 18. Deployment Topologies

### 18.1 Single-Node (Development / Small Scale)

```mermaid
graph TB
    subgraph SingleNode["Single Node"]
        PROC["cvx-server process"]
        ROCKS["RocksDB<br/>(local SSD)"]
        PARQUET_DIR["Parquet files<br/>(local disk)"]
        LOCAL_S3["MinIO<br/>(local cold storage)"]
    end

    CLIENT["Clients"] --> PROC
    PROC --> ROCKS
    PROC --> PARQUET_DIR
    PROC --> LOCAL_S3
```

### 18.2 Distributed (Production)

```mermaid
graph TB
    subgraph LB["Load Balancer"]
        NGINX["NGINX / Envoy"]
    end

    subgraph Cluster["CVX Cluster"]
        subgraph Shard1["Shard 1 (entities 0-N/3)"]
            S1_LEADER["Leader"]
            S1_FOLLOW1["Follower 1"]
            S1_FOLLOW2["Follower 2"]
        end
        subgraph Shard2["Shard 2 (entities N/3-2N/3)"]
            S2_LEADER["Leader"]
            S2_FOLLOW1["Follower 1"]
            S2_FOLLOW2["Follower 2"]
        end
        subgraph Shard3["Shard 3 (entities 2N/3-N)"]
            S3_LEADER["Leader"]
            S3_FOLLOW1["Follower 1"]
            S3_FOLLOW2["Follower 2"]
        end
    end

    subgraph SharedStorage["Shared Storage"]
        S3["S3 / MinIO<br/>(Cold Tier)"]
        ETCD["etcd<br/>(Cluster Metadata)"]
    end

    NGINX --> S1_LEADER
    NGINX --> S2_LEADER
    NGINX --> S3_LEADER

    S1_LEADER --> S1_FOLLOW1
    S1_LEADER --> S1_FOLLOW2
    S2_LEADER --> S2_FOLLOW1
    S2_LEADER --> S2_FOLLOW2
    S3_LEADER --> S3_FOLLOW1
    S3_LEADER --> S3_FOLLOW2

    S1_LEADER --> S3
    S2_LEADER --> S3
    S3_LEADER --> S3

    S1_LEADER -.-> ETCD
    S2_LEADER -.-> ETCD
    S3_LEADER -.-> ETCD
```

**Sharding Strategy:** Por `entity_id` hash (consistent hashing). Cada shard posee un rango de entidades y mantiene su propia instancia de ST-HNSW + tiered storage. Los queries cross-shard (e.g., "global kNN") requieren scatter-gather coordinado por el load balancer.

**Replicación:** Raft (vía `openraft`) dentro de cada shard para durabilidad. Las lecturas se sirven desde followers; las escrituras se routean al leader.
