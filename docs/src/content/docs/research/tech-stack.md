---
title: "Tech Stack"
description: "Rust crate ecosystem and technology choices for CVX"
---

## 6. Core Rust Tech Stack (2026 Ready)

| Componente | Crate(s) | Justificación |
|---|---|---|
| **Tensor Compute** | `burn` + `candle` | `burn` para training/inference de Neural ODE; `candle` para inferencia ligera de modelos pre-entrenados |
| **Storage Engine** | `rocksdb` (rust-rocksdb) | Versionado por prefijo de timestamp, column families, bloom filters |
| **Vector SIMD** | `pulp` + auto-vectorización | Dispatch dinámico SSE4.2/AVX2/AVX-512/NEON; `simsimd` como fallback |
| **Columnar Analytics** | `arrow-rs` + `polars` | Análisis de series temporales sobre warm storage |
| **HTTP/gRPC API** | `axum` + `tonic` | API REST con axum, gRPC streaming con tonic para ingesta en tiempo real |
| **Async Runtime** | `tokio` | Orquestación de I/O async sin bloquear búsquedas |
| **Serialización** | `serde` + `rkyv` | `serde` para API; `rkyv` (zero-copy deserialization) para datos internos |
| **Bitmaps** | `roaring` | Roaring Bitmaps para filtrado por rango de tiempo en O(1) |
| **Object Storage** | `object_store` | Abstracción sobre S3/GCS/Azure/local filesystem |
| **Consensus** | `raft-rs` (openraft) | Consistencia de shards temporales en modo distribuido |
| **Hashing** | `xxhash-rust` | Hash rápido para content-addressed delta storage |
| **ODE Solver** | Custom (Dormand-Prince RK45) | Solver adaptativo con SIMD, evaluación de $f_\theta$ vía burn |
| **Change Point** | Custom (PELT + BOCPD) | Implementación nativa en Rust de ambos algoritmos |

---

## 9. Rust Skills Development Map

Este proyecto te obligará a dominar aspectos avanzados de Rust en cada fase:

| Fase | Skill de Rust | Concepto |
|------|--------------|----------|
| Phase 1 | **Generics & Traits** | `VectorSpace`, `DistanceMetric`, `Timestamp` como traits genéricos con bounds complejos |
| Phase 1 | **Unsafe Rust** | Kernels SIMD para distancias, acceso raw a buffers de vectores |
| Phase 2 | **Lifetime management** | Referencias a slices temporales sin copiar datos |
| Phase 2 | **Iterator patterns** | Lazy evaluation sobre series de deltas |
| Phase 3 | **Async/Concurrency** | `tokio` para orquestar ingesta masiva sin bloquear búsquedas; `RwLock` para índices concurrentes |
| Phase 3 | **FFI** | Bindings a RocksDB (C++) vía `rust-rocksdb` |
| Phase 4 | **Trait objects vs enums** | Dispatch dinámico para diferentes métricas y solvers |
| Phase 4 | **Error handling** | `thiserror`/`anyhow` para propagación elegante en pipelines complejos |
| Phase 5 | **Memory management** | Arenas, memory pools, direct memory access para buffers fuera del heap |
| Phase 5 | **Procedural macros** | DSL para definir queries temporales de forma declarativa |
| Phase 6 | **WASM compilation** | Compilar kernels de distancia para browser-based demos |
