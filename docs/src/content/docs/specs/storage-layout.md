---
title: "Storage Layout Specification"
description: "Exact on-disk data layout for every storage tier in ChronosVector"
---

**Version:** 1.0
**Author:** Manuel Couto Pintos
**Date:** March 2026
**Status:** Draft
**Companion:** `ChronosVector_Architecture.md` §8, `CVX_RFC_001` ADR-004, ADR-005, ADR-006

---

## 1. Overview

Este documento especifica el layout exacto de datos en disco para cada tier de almacenamiento de ChronosVector. Sirve como referencia para la implementación de `cvx-storage` y como contrato para la compatibilidad entre versiones.

---

## 2. Directory Structure

```
$CVX_DATA_DIR/
├── config.toml                    # System configuration
├── wal/                           # Write-Ahead Log
│   ├── segment-000000000000.wal   # WAL segments (64MB each)
│   ├── segment-000000000001.wal
│   └── wal.meta                   # Head/commit/tail offsets
├── hot/                           # Hot Tier (RocksDB)
│   ├── CURRENT
│   ├── MANIFEST-*
│   ├── OPTIONS-*
│   ├── *.sst                      # SSTable files
│   └── *.log                      # RocksDB WAL (separate from CVX WAL)
├── warm/                          # Warm Tier (Parquet)
│   ├── manifests/
│   │   └── warm_manifest.json     # Partition → file mapping
│   └── data/
│       ├── entity_prefix=00/
│       │   ├── 2026-01.parquet
│       │   └── 2026-02.parquet
│       └── entity_prefix=01/
│           └── 2026-01.parquet
├── cold/                          # Cold Tier (local cache + remote refs)
│   ├── manifests/
│   │   └── cold_manifest.json     # Partition → object_store key mapping
│   ├── codebooks/
│   │   ├── pq_codebook_v001.bin   # PQ codebook (rkyv serialized)
│   │   └── pq_codebook_latest -> pq_codebook_v001.bin
│   └── cache/                     # Local LRU cache of fetched cold objects
│       └── *.pqvec                # PQ-encoded vector blocks
├── index/                         # Index persistence
│   ├── sthnsw_graph.bin           # Serialized ST-HNSW graph (rkyv)
│   ├── sthnsw_meta.json           # Graph metadata (human-readable)
│   ├── timestamp_bitmaps/
│   │   ├── bitmap_range_*.roaring # Roaring bitmap files per time range
│   │   └── bitmap_index.json      # Range → file mapping
│   └── hnt/
│       └── hnt_data.bin           # Historic Neighbor Tree (rkyv)
└── snapshots/                     # Point-in-time snapshots
    └── snap_20260315T120000/
        ├── hot_checkpoint/        # RocksDB checkpoint (hardlinks)
        └── index_snapshot.bin     # Index graph copy
```

---

## 3. Write-Ahead Log (WAL)

### 3.1 Segment Format

Cada segmento WAL es un archivo append-only de tamaño máximo 64MB.

```
┌─────────────────────────────────────────────────────┐
│ Segment Header (32 bytes)                           │
├─────────────────────────────────────────────────────┤
│ Magic:         [u8; 4]  = b"CVXW"                   │
│ Version:       u16      = 1                          │
│ SegmentId:     u64      = monotonic segment number   │
│ CreatedAt:     i64      = timestamp (µs)             │
│ Reserved:      [u8; 6]  = zeroed                     │
├─────────────────────────────────────────────────────┤
│ Entry 0                                             │
├─────────────────────────────────────────────────────┤
│ Entry 1                                             │
├─────────────────────────────────────────────────────┤
│ ...                                                 │
├─────────────────────────────────────────────────────┤
│ Entry N                                             │
└─────────────────────────────────────────────────────┘
```

### 3.2 WAL Entry Format

```
┌─────────────────────────────────────────────────────┐
│ Entry Header (24 bytes)                             │
├─────────────────────────────────────────────────────┤
│ EntryLength:   u32      (total bytes including hdr) │
│ SequenceNum:   u64      (global monotonic)          │
│ EntryType:     u8       (0=Insert, 1=Delete,        │
│                          2=Update, 3=Checkpoint)     │
│ Flags:         u8       (bit 0: is_keyframe)        │
│ Reserved:      [u8; 2]                              │
│ CRC32:         u32      (over payload only)         │
├─────────────────────────────────────────────────────┤
│ Payload (variable length)                           │
├─────────────────────────────────────────────────────┤
│ For Insert/Update:                                  │
│   EntityId:    u64                                  │
│   Timestamp:   i64                                  │
│   Dimension:   u16                                  │
│   VectorData:  [f32; Dimension]                     │
│   MetadataLen: u32                                  │
│   MetadataBytes: [u8; MetadataLen] (rkyv)           │
├─────────────────────────────────────────────────────┤
│ For Delete:                                         │
│   EntityId:    u64                                  │
│   Timestamp:   i64                                  │
├─────────────────────────────────────────────────────┤
│ For Checkpoint:                                     │
│   CheckpointId: u64                                 │
│   HotLSN:      u64 (RocksDB sequence number)        │
└─────────────────────────────────────────────────────┘
```

### 3.3 WAL Meta File

```json
{
  "format_version": 1,
  "head_segment": 42,
  "head_offset": 16384,
  "commit_segment": 41,
  "commit_offset": 65000,
  "tail_segment": 38,
  "last_sequence": 1234567890
}
```

### 3.4 WAL Recovery Protocol

1. Read `wal.meta` to find `commit_offset`.
2. Scan from `commit_offset` to `head_offset`, validating CRC32 of each entry.
3. For each valid uncommitted entry, replay into hot store + index.
4. If CRC fails mid-entry, truncate at that point (partial write).
5. Update `commit_offset` = `head_offset` and fsync `wal.meta`.

---

## 4. Hot Tier (RocksDB)

### 4.1 Column Families

| CF Name | Key Format | Value Format | Purpose |
|---|---|---|---|
| `vectors` | `[entity_id: u64 BE][timestamp: i64 BE]` | `[rkyv(Vec<f32>)]` | Full vectors + keyframes |
| `deltas` | `[entity_id: u64 BE][timestamp: i64 BE]` | Delta Encoding (§4.3) | Sparse delta vectors |
| `metadata` | `[entity_id: u64 BE][timestamp: i64 BE]` | `[rkyv(HashMap<String,Value>)]` | Arbitrary key-value metadata |
| `timelines` | `[entity_id: u64 BE]` | Timeline Record (§4.4) | Per-entity lifecycle tracking |
| `changepoints` | `[entity_id: u64 BE][timestamp: i64 BE]` | ChangePoint Record (§4.5) | Detected change points |
| `system` | `[ascii key]` | `[rkyv(value)]` | Internal state (e.g., "global_dim", "vector_count") |

### 4.2 Key Encoding

Todas las claves usan **Big-Endian byte order** para que el orden lexicográfico de bytes coincida con el orden numérico. Esto permite usar `prefix_iterator` de RocksDB para range scans eficientes.

```rust
fn encode_key(entity_id: u64, timestamp: i64) -> [u8; 16] {
    let mut key = [0u8; 16];
    key[0..8].copy_from_slice(&entity_id.to_be_bytes());
    // Flip sign bit for correct ordering of signed timestamps
    let ts_sortable = (timestamp as u64) ^ (1u64 << 63);
    key[8..16].copy_from_slice(&ts_sortable.to_be_bytes());
    key
}
```

El XOR con `1 << 63` en el timestamp convierte la representación signed a un formato donde el orden lexicográfico de bytes coincide con el orden numérico signed (i.e., -1 < 0 < 1 en bytes).

### 4.3 Delta Value Format

```
┌──────────────────────────────────────────┐
│ Delta Header (17 bytes)                  │
├──────────────────────────────────────────┤
│ FormatVersion:     u8  = 1               │
│ BaseTimestamp:     i64 (reference point)  │
│ NumNonZero:        u32 (sparse count)    │
│ ContentHash:       u32 (xxhash32)        │
├──────────────────────────────────────────┤
│ Sparse Indices: [u16; NumNonZero]        │
│   (dimensions that changed)              │
├──────────────────────────────────────────┤
│ Sparse Values: [f32; NumNonZero]         │
│   (delta values for those dimensions)    │
└──────────────────────────────────────────┘
```

**Size estimation:** Para D=768 con 10% of dimensions changing, NumNonZero ≈ 77. Delta size = 17 + 77×2 + 77×4 = 479 bytes vs. 3072 bytes for full FP32 vector. **6.4x compression**.

### 4.4 Timeline Value Format

```
┌──────────────────────────────────────────┐
│ Timeline Record (36 bytes)               │
├──────────────────────────────────────────┤
│ FormatVersion:     u8  = 1               │
│ Flags:             u8                    │
│   bit 0: has_neural_ode_model            │
│   bit 1: bocpd_monitoring_active         │
│ FirstSeen:         i64                   │
│ LastSeen:          i64                   │
│ PointCount:        u32                   │
│ KeyframeInterval:  u16                   │
│ Dimension:         u16                   │
│ LastKeyframeTs:    i64                   │
│ DeltasSinceKF:     u16 (counter 0..K)   │
└──────────────────────────────────────────┘
```

### 4.5 ChangePoint Value Format

```
┌──────────────────────────────────────────┐
│ ChangePoint Record                       │
├──────────────────────────────────────────┤
│ FormatVersion:     u8  = 1               │
│ DetectionMethod:   u8  (0=PELT, 1=BOCPD)│
│ Severity:          f64                   │
│ DriftMagnitude:    f32                   │
│ NumDriftDims:      u16                   │
│ DriftDimIndices:   [u16; NumDriftDims]   │
│   (top-N dimensions contributing most)   │
│ DriftDimValues:    [f32; NumDriftDims]   │
└──────────────────────────────────────────┘
```

### 4.6 RocksDB Tuning

```toml
[storage.hot.rocksdb]
# Write performance
write_buffer_size_mb = 64           # Per CF memtable size
max_write_buffer_number = 3         # Memtables before stall
min_write_buffer_number_to_merge = 1

# Read performance
block_cache_size_mb = 512           # Shared block cache
bloom_filter_bits_per_key = 10      # For prefix bloom

# Compaction
compaction_style = "level"          # Level compaction for read-heavy
max_bytes_for_level_base_mb = 256
target_file_size_base_mb = 64
max_background_compactions = 4
max_background_flushes = 2

# Compression
compression_type = "lz4"           # Fast compression for hot data
bottommost_compression = "zstd"    # Better ratio for bottom level
```

---

## 5. Warm Tier (Parquet)

### 5.1 Partitioning Scheme

Archivos Parquet particionados por dos niveles:
1. **entity_prefix:** Los primeros 2 hex chars del entity_id (256 partitions). Distribuye I/O uniformemente.
2. **month:** `YYYY-MM`. Alinea con la granularidad típica de tier migration.

```
warm/data/entity_prefix=a7/2026-01.parquet
warm/data/entity_prefix=a7/2026-02.parquet
warm/data/entity_prefix=f3/2026-01.parquet
```

### 5.2 Parquet Schema

```
message TemporalVectorRecord {
  required int64  entity_id;
  required int64  timestamp (TIMESTAMP(MICROS, false));
  required binary vector (LIST<FLOAT>);        // D floats
  optional int64  keyframe_ref_timestamp;       // null = this IS the keyframe
  optional binary delta_indices (LIST<INT32>);  // null = full vector stored
  optional binary delta_values (LIST<FLOAT>);   // null = full vector stored
  optional binary metadata (JSON);              // schemaless metadata
  optional double change_severity;              // null = no change point here
  optional int32  change_method;                // 0=PELT, 1=BOCPD
}
```

### 5.3 Parquet Settings

```toml
[storage.warm.parquet]
row_group_size = 65536              # Rows per row group
data_page_size_bytes = 1048576      # 1MB data pages
compression = "zstd"                # Good ratio + speed balance
compression_level = 3               # zstd level (1-22, 3 is fast)
write_batch_size = 10000            # Buffer this many rows before writing
dictionary_encoding = true          # For metadata column
statistics = true                   # Enable min/max stats for predicate pushdown
sorting_columns = ["entity_id", "timestamp"]  # Sort within row groups
```

### 5.4 Warm Manifest

```json
{
  "format_version": 1,
  "last_updated": "2026-03-15T12:00:00Z",
  "partitions": [
    {
      "entity_prefix": "a7",
      "month": "2026-01",
      "file": "data/entity_prefix=a7/2026-01.parquet",
      "row_count": 487293,
      "min_timestamp": 1704067200000000,
      "max_timestamp": 1706745600000000,
      "size_bytes": 14238912,
      "entities_count": 12847
    }
  ],
  "total_row_count": 52847293,
  "total_size_bytes": 1572864000
}
```

---

## 6. Cold Tier (Object Store + PQ)

### 6.1 Object Key Layout

```
s3://<bucket>/cvx/v1/
├── codebooks/
│   ├── pq_cb_v001.bin
│   └── pq_cb_v002.bin
├── data/
│   ├── entity_prefix=a7/
│   │   ├── 2025-Q1.pqvec      # Quarter-level granularity
│   │   └── 2025-Q2.pqvec
│   └── entity_prefix=f3/
│       └── 2025-Q1.pqvec
└── manifests/
    └── cold_manifest_v042.json
```

### 6.2 PQ Vector Block Format (.pqvec)

```
┌──────────────────────────────────────────────────┐
│ File Header (64 bytes)                           │
├──────────────────────────────────────────────────┤
│ Magic:             [u8; 4]  = b"PQVX"            │
│ FormatVersion:     u16      = 1                   │
│ CodebookVersion:   u32      (references codebook) │
│ NumVectors:        u64                            │
│ OriginalDimension: u16                            │
│ NumSubspaces:      u16      (M in PQ)             │
│ CentroidsPerSub:   u16      (K in PQ, usually 256)│
│ BytesPerCode:      u16      (M × ceil(log2(K))/8) │
│ MinTimestamp:       i64                            │
│ MaxTimestamp:       i64                            │
│ Reserved:          [u8; 10]                        │
├──────────────────────────────────────────────────┤
│ Index Section                                    │
│   For each vector i (0..NumVectors):             │
│     EntityId:      u64                           │
│     Timestamp:     i64                           │
│   (sorted by entity_id, then timestamp)          │
├──────────────────────────────────────────────────┤
│ PQ Codes Section                                 │
│   For each vector i (0..NumVectors):             │
│     Code:          [u8; BytesPerCode]             │
│   (same order as index)                          │
├──────────────────────────────────────────────────┤
│ Footer (16 bytes)                                │
│   IndexOffset:     u64  (byte offset of index)   │
│   CodesOffset:     u64  (byte offset of codes)   │
└──────────────────────────────────────────────────┘
```

**Design decisions:**
- Index and codes are separate sections to allow reading the index without loading all codes (useful for filtering by entity/time before decompression).
- Footer at end allows streaming writes (offsets known only at completion).
- Sorted by entity_id + timestamp enables binary search on the index section.

### 6.3 PQ Codebook Format (.bin)

```
┌──────────────────────────────────────────────────┐
│ Codebook Header (32 bytes)                       │
├──────────────────────────────────────────────────┤
│ Magic:             [u8; 4]  = b"PQCB"            │
│ FormatVersion:     u16      = 1                   │
│ CodebookVersion:   u32      (monotonic)           │
│ OriginalDimension: u16                            │
│ NumSubspaces:      u16      (M)                   │
│ CentroidsPerSub:   u16      (K)                   │
│ SubspaceDim:       u16      (D / M)               │
│ TrainingVectors:   u64      (how many used)       │
│ TrainedAt:         i64      (timestamp)            │
├──────────────────────────────────────────────────┤
│ Centroids Data                                   │
│   For subspace m (0..M):                         │
│     For centroid k (0..K):                       │
│       Values: [f32; SubspaceDim]                 │
│   Total: M × K × SubspaceDim × 4 bytes          │
├──────────────────────────────────────────────────┤
│ Lookup Tables (precomputed, optional)            │
│   ADC tables for common query patterns           │
└──────────────────────────────────────────────────┘
```

**Size estimation:** M=8, K=256, D=768 → SubspaceDim=96. Codebook = 8 × 256 × 96 × 4 = 786,432 bytes ≈ 768KB. Fits easily in L2 cache.

### 6.4 Cold Manifest

```json
{
  "format_version": 1,
  "manifest_version": 42,
  "last_updated": "2026-03-15T12:00:00Z",
  "codebook_ref": "codebooks/pq_cb_v002.bin",
  "blocks": [
    {
      "entity_prefix": "a7",
      "quarter": "2025-Q1",
      "object_key": "data/entity_prefix=a7/2025-Q1.pqvec",
      "num_vectors": 1284700,
      "min_timestamp": 1704067200000000,
      "max_timestamp": 1711929600000000,
      "size_bytes": 24389120,
      "codebook_version": 2,
      "entities_count": 52340
    }
  ],
  "total_vectors": 284700000,
  "total_blocks": 1024
}
```

---

## 7. Index Persistence

### 7.1 ST-HNSW Graph Serialization

El grafo se serializa usando `rkyv` para permitir memory-mapping sin deserialización.

```
┌──────────────────────────────────────────────────┐
│ Graph Header (48 bytes)                          │
├──────────────────────────────────────────────────┤
│ Magic:             [u8; 4]  = b"HSTN"            │
│ FormatVersion:     u16      = 1                   │
│ NumNodes:          u64                            │
│ NumLayers:         u8                             │
│ M (max neighbors): u16                            │
│ EfConstruction:    u16                            │
│ EntryPointId:      u64                            │
│ MetricType:        u8  (0=cosine, 1=l2, 2=dot,   │
│                         3=poincare)               │
│ Dimension:         u16                            │
│ Reserved:          [u8; 8]                        │
├──────────────────────────────────────────────────┤
│ Node Array (rkyv serialized)                     │
│   For each node:                                 │
│     NodeId:         u64                          │
│     EntityId:       u64                          │
│     Timestamp:      i64                          │
│     MaxLayer:       u8                           │
│     Neighbors per layer:                         │
│       For layer l (0..MaxLayer+1):               │
│         NumNeighbors: u16                        │
│         NeighborIds:  [u64; NumNeighbors]        │
├──────────────────────────────────────────────────┤
│ Footer                                           │
│   NodeArrayOffset:  u64                          │
│   TotalSize:        u64                          │
│   CRC64:            u64  (of entire file)        │
└──────────────────────────────────────────────────┘
```

### 7.2 Roaring Bitmap Files

Cada archivo bitmap cubre un rango temporal (e.g., 1 hora). El bitmap contiene los `point_id`s válidos en ese rango.

```json
// bitmap_index.json
{
  "format_version": 1,
  "granularity_micros": 3600000000,
  "ranges": [
    {
      "start": 1710460800000000,
      "end":   1710464400000000,
      "file": "bitmap_range_1710460800.roaring",
      "cardinality": 48293
    }
  ]
}
```

Los archivos `.roaring` usan el formato estándar de serialización de Roaring Bitmaps (compatible con `croaring` C library).

---

## 8. Tier Migration Data Flows

### 8.1 Hot → Warm Migration

```
Trigger: age(entry) > hot_ttl OR hot_store_size > threshold

1. Scan hot CF "vectors" for entries where timestamp < (now - hot_ttl)
2. Batch read: collect entries grouped by entity_prefix
3. For each group:
   a. Read existing warm Parquet file for that partition (if exists)
   b. Merge new entries with existing (sort by entity_id, timestamp)
   c. Write new Parquet file (atomic rename)
   d. Update warm manifest
4. Delete migrated entries from hot store
5. Update Roaring Bitmaps (remove hot bitmap entries, add warm range entries)
```

### 8.2 Warm → Cold Migration

```
Trigger: age(partition) > warm_ttl

1. Read Parquet partition file
2. For each row:
   a. Reconstruct full vector (if delta, apply to keyframe)
   b. Encode with PQ using latest codebook
3. Build .pqvec block file
4. Upload to object store
5. Update cold manifest
6. Delete Parquet partition file
7. Update warm manifest
```

### 8.3 Codebook Retraining

```
Trigger: periodic (e.g., weekly) OR cold_vectors_since_last_train > threshold

1. Sample N vectors from warm tier (representative recent data)
2. Run k-means per subspace (M runs, each on D/M dimensions)
3. Produce new codebook with incremented version
4. Upload to object store
5. Re-encode oldest cold blocks using new codebook (background, non-blocking)
6. Update cold manifest to reference new codebook for new blocks
7. Old codebook retained until all blocks referencing it are re-encoded
```

---

## 9. Snapshot & Backup

### 9.1 Snapshot Creation

```
1. Create RocksDB checkpoint (hardlinks, instant, no copy)
2. Serialize ST-HNSW graph to snapshot directory
3. Copy current WAL meta
4. Record snapshot metadata:
   {
     "snapshot_id": "snap_20260315T120000",
     "created_at": "2026-03-15T12:00:00Z",
     "wal_sequence": 1234567890,
     "hot_lsn": 987654321,
     "index_nodes": 1000000,
     "warm_manifest_version": 42,
     "cold_manifest_version": 42
   }
```

### 9.2 Restore from Snapshot

```
1. Stop cvx-server
2. Replace hot/ directory with snapshot's hot_checkpoint/
3. Replace index/ graph with snapshot's index_snapshot.bin
4. Set WAL commit_offset to snapshot's wal_sequence
5. Start cvx-server → WAL recovery replays entries after snapshot point
```

---

## 10. Format Versioning & Evolution

Cada formato de archivo incluye un `FormatVersion` field en su header. Las reglas de evolución:

| Change Type | Version Bump | Compatibility |
|---|---|---|
| New optional field at end | Minor (no bump needed) | Readers ignore unknown trailing bytes |
| Change field type/order | Major (increment version) | Reader dispatches to version-specific deserializer |
| New column family in RocksDB | No bump | Old CFs unaffected; new CF created on first write |
| New Parquet column | No bump | Parquet is self-describing; old readers ignore new columns |

**Invariant:** CVX version N can always read data written by version N-1. Forward compatibility (N reading N+1) is not guaranteed.
