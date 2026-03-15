# ChronosVector — Implementation Decisions & Technical Guidelines

**Version:** 1.0
**Author:** Manuel Couto Pintos
**Date:** March 2026
**Status:** Draft
**Dependencies:** Architecture Doc, RFC-001, Iterative Roadmap

---

## 1. Overview

Este documento registra 10 decisiones de implementacion tomadas durante la fase de diseno de ChronosVector (CVX). Cubren concurrencia, serializacion, SIMD, storage engine, error handling, testing, unsafe policy, e index persistence. Cada decision se documenta como un IDR (Implementation Decision Record) independiente, siguiendo el formato Context/Decision/Alternatives/Consequences del RFC-001.

Estas decisiones son complementarias a las decisiones arquitecturales (ADRs) del RFC-001. Mientras los ADRs definen *que* construimos, los IDRs definen *como* lo construimos — las herramientas, patrones, y trade-offs concretos a nivel de codigo.

Las 10 decisiones:

| IDR | Topic | Key Choice |
|-----|-------|------------|
| 001 | Concurrency Model | Message passing + selective locks |
| 002 | Compute Parallelism | Rayon thread pool |
| 003 | Serialization | Hybrid rkyv + postcard |
| 004 | Global Allocator | jemalloc |
| 005 | SIMD Strategy | pulp |
| 006 | RocksDB Key Encoding | Big-endian + sign-bit flip |
| 007 | Error Handling | thiserror (libs) + anyhow (binary) |
| 008 | Testing Strategy | 5 levels, property-based core |
| 009 | Unsafe Policy | deny by default, allow in 2 crates |
| 010 | Index Persistence | Progressive (read-into-memory → mmap) |

---

## 2. IDR-001 — Concurrency Model: Message Passing + Selective Locks

### Context

CVX tiene dos tipos fundamentales de trabajo concurrente:

- **I/O-bound:** red (HTTP/gRPC), disco (RocksDB, WAL writes), conectores de fuentes externas.
- **CPU-bound:** SIMD distance computation, graph traversal en HNSW, PELT/BOCPD, delta encoding.

La pregunta clasica: locks vs message passing. Rust ofrece ambos con garantias de seguridad, pero cada uno tiene trade-offs diferentes segun el patron de acceso.

### Decision

Usar **ambos paradigmas**, cada uno donde encaja naturalmente.

**Message passing (tokio channels)** para flujo de datos:

- Ingestion pipeline: cada etapa es un task conectado por `tokio::sync::mpsc` bounded channels.
- Background tasks: compaction scheduler, BOCPD streaming, tier migration.
- Event bus: drift events, alertas de monitors, model version change notifications.
- Source connector sync: cada conector envia batches via channel al pipeline.
- Monitor evaluation: timer envia ticks, evaluator recibe y ejecuta.

**RwLock para estructuras de lectura concurrente:**

- ST-HNSW graph: `parking_lot::RwLock<HnswGraph>`. Multiples readers concurrentes (searches) + single writer (inserts/deletes).
- Entry point del indice: `AtomicU64` — lock-free para el hot path mas critico.
- Metrics counters: `AtomicU64` / `AtomicF64` via atomics.
- Config hot reload: `ArcSwap<Config>` — readers nunca bloquean, writer swaps atomicamente.

**La regla:** *"Si el dato fluye, channels. Si el dato se consulta concurrentemente, RwLock."*

### Por que no actor model para el indice

Un patron actor para el ST-HNSW significaria que cada busqueda enviaria un mensaje al actor del indice, esperaria respuesta, y el actor procesaria busquedas secuencialmente. Con 100 busquedas concurrentes, todas se serializarian a traves de un unico thread — un desastre para throughput.

Con `RwLock`, 100 busquedas concurrentes toman el read lock simultaneamente sin contention. Solo cuando el writer necesita insertar, espera a que los readers terminen. Este patron es ideal para workloads read-heavy (que es el caso comun: muchas queries, pocas inserciones en comparacion).

### Pipeline de ingesta como cadena de channels

```
receive → validate → delta_encode → WAL → index_insert → store → ack
   tx→rx     tx→rx      tx→rx      tx→rx     tx→rx      tx→rx
```

Cada flecha es un `mpsc::channel` bounded. Cada etapa corre como un tokio task independiente. Si `delta_encode` es mas lento que `validate`, el channel se llena y `validate` espera — backpressure natural sin logica adicional.

```rust
// Ejemplo simplificado del pipeline
let (validate_tx, validate_rx) = mpsc::channel::<ValidatedBatch>(PIPELINE_BUFFER);
let (delta_tx, delta_rx) = mpsc::channel::<DeltaBatch>(PIPELINE_BUFFER);
let (wal_tx, wal_rx) = mpsc::channel::<WalEntry>(PIPELINE_BUFFER);

// Cada stage es un tokio::spawn independiente
tokio::spawn(async move {
    while let Some(batch) = validate_rx.recv().await {
        let deltas = delta_encode(batch).await;
        delta_tx.send(deltas).await.expect("delta stage alive");
    }
});
```

### Consequences

- (+) Backpressure natural via bounded channels — no se necesita rate limiting explicito.
- (+) No deadlocks en el pipeline path — los channels son unidireccionales.
- (+) Read-heavy index mantiene paralelismo completo (N readers simultaneos).
- (+) Patron validado: Qdrant usa `parking_lot::RwLock` para su indice HNSW.
- (-) Dos paradigmas de concurrencia que entender y mantener.
- (-) `RwLock` en el indice puede starve al writer si los readers son continuos (mitigado por la fairness policy de `parking_lot`).
- (-) Los channels bounded requieren elegir un buffer size — demasiado pequeno causa stalls, demasiado grande usa memoria.

### Future

Si profiling muestra contention en el `RwLock` del indice, considerar lock-free concurrent HNSW (como lo que Qdrant esta explorando con segmented indices). Pero es optimizacion prematura ahora — el read-heavy workload de CVX favorece `RwLock`.

---

## 3. IDR-002 — Compute Parallelism: Rayon

### Context

El trabajo CPU-bound (SIMD distances, graph traversal, PELT, delta encoding) **no debe bloquear** el runtime async de Tokio. Si una tarea CPU-bound ocupa un thread del runtime Tokio, todas las tareas I/O que comparten ese thread se detienen — incluyendo HTTP responses, gRPC streams, y WAL writes.

### Decision

Rayon thread pool dedicado para todo el trabajo CPU-bound, conectado desde Tokio via `spawn_blocking`.

**Arquitectura de threads:**

```
Tokio runtime (I/O)
├── HTTP server (axum)
├── gRPC server (tonic)
├── Background tasks (compaction scheduler, monitor timer)
├── Channel pipeline (ingestion stages)
└── spawn_blocking → Rayon pool (compute)
    ├── HNSW search (graph traversal + distance computation)
    ├── Distance computation (SIMD via pulp)
    ├── PELT / BOCPD (change point detection)
    ├── Delta encode/decode (sparse vector ops)
    └── Neural ODE inference (burn)
```

**Bridge pattern de spawn_blocking a Rayon:**

```rust
use rayon::prelude::*;
use tokio::task;

/// Ejecuta trabajo CPU-bound en el pool de Rayon, sin bloquear Tokio.
pub async fn search_index(
    index: Arc<RwLock<HnswGraph>>,
    query: QueryVector,
    k: usize,
) -> Result<Vec<SearchResult>, CvxError> {
    task::spawn_blocking(move || {
        let graph = index.read(); // parking_lot read lock
        graph.search(&query, k)   // runs on Rayon internally
    })
    .await
    .map_err(|e| CvxError::Internal(format!("spawn_blocking failed: {e}")))?
}
```

Dentro de `graph.search()`, Rayon puede usar `par_iter` para computar distancias a multiples candidatos en paralelo:

```rust
impl HnswGraph {
    pub fn search(&self, query: &QueryVector, k: usize) -> Result<Vec<SearchResult>, IndexError> {
        // ... graph traversal ...

        // Compute distances to candidates in parallel
        let distances: Vec<f32> = candidates
            .par_iter()
            .map(|candidate| {
                SimdKernels::cosine_distance(&query.values, &candidate.values)
            })
            .collect();

        // ... select top-k ...
    }
}
```

### Por que Rayon y no solo spawn_blocking

`spawn_blocking` ejecuta **una tarea por thread**. Si necesitas computar 100 distancias, lanzas 1 tarea que computa 100 distancias secuencialmente.

Rayon ofrece **work-stealing y paralelismo DENTRO de una tarea**. Con `par_iter()`, esas 100 distancias se distribuyen automaticamente entre todos los cores del pool. Si un core termina antes, roba trabajo de otro — maximizando utilizacion.

Ademas, Rayon maneja automaticamente la granularidad: para 10 distancias, probablemente no paraleliza (overhead > beneficio). Para 10,000, paraleliza agresivamente.

**Dependency:**

```toml
rayon = "1"
```

### Consequences

- (+) Work-stealing maximiza utilizacion de cores.
- (+) `par_iter` habilita data-parallel distance computation dentro de una sola busqueda.
- (+) Qdrant usa el mismo patron (Tokio + Rayon) — validado en produccion a escala.
- (+) Separacion clara: Tokio para I/O, Rayon para compute.
- (-) Dos thread pools (Tokio + Rayon) a configurar y monitorear.
- (-) El bridge `spawn_blocking` anade ~1us de overhead por llamada (negligible para operaciones de ms).
- (-) Si Rayon pool esta saturado, nuevas tareas esperan — necesita monitoring de queue depth.

---

## 4. IDR-003 — Serialization: Hybrid rkyv + postcard

### Context

CVX serializa datos en multiples puntos con requisitos diferentes:

- El grafo HNSW se persiste a disco y se carga al arranque. Puede ser de 84MB a 8.4GB. La velocidad de carga impacta directamente el tiempo de startup.
- Los vectores y deltas en RocksDB se serializan/deserializan en cada read/write.
- El WAL necesita forward-compatibility para schema changes entre versiones.
- Config y metadata deben ser legibles por humanos.

### Decision

**Hybrid approach:** rkyv para el grafo HNSW (zero-copy, mmap), postcard para todo lo demas (compacto, schema-evolvable).

| Data | Format | Why |
|------|--------|-----|
| HNSW graph (save/load) | rkyv | Zero-copy via mmap, startup instantaneo para grafos grandes |
| Vectors en RocksDB | postcard | RocksDB ya copia los datos internamente, zero-copy no aporta ventaja |
| Deltas en RocksDB | postcard | Codificacion variable-length, eficiente para sparse data |
| WAL entries | postcard | Schema evolution via `serde(default)`, compacto |
| Config/metadata | TOML/JSON via serde | Human readable, editable |
| Network (gRPC) | protobuf | Decidido en ADR-011 |
| Network (REST) | JSON | Decidido en ADR-011 |

### Por que postcard sobre bincode

Ambos son formatos binarios basados en serde. La diferencia clave:

- **postcard** usa variable-length integer encoding (VarInt). Un `u32` con valor 42 ocupa 1 byte, no 4. Para datos sparse (deltas con muchos indices pequenos), esto reduce el tamano significativamente.
- **bincode** usa fixed-width encoding por defecto. Mas rapido para deserializar (no hay que decodificar VarInts), pero la diferencia es negligible para los tamanos involucrados (<1KB por vector).
- **postcard** es `no_std` compatible — util si en el futuro queremos compilar `cvx-core` para WASM.
- **postcard** esta activamente mantenido por James Munns (Ferrous Systems), con un track record solido.

### Schema evolution

postcard + serde soporta `#[serde(default)]` para backward compatibility:

```rust
#[derive(Serialize, Deserialize)]
pub struct WalEntry {
    pub entity_id: u64,
    pub timestamp: i64,
    pub vector: Vec<f32>,
    #[serde(default)]          // v2: nuevo campo, default = None
    pub space_id: Option<u32>,
    #[serde(default)]          // v3: nuevo campo, default = 0
    pub schema_version: u32,
}
```

Datos escritos por v1 (sin `space_id`) se deserializan correctamente en v2 — `space_id` toma el valor default `None`.

**rkyv NO soporta schema evolution.** Cambiar un campo en el struct del HNSW graph requiere re-indexar. Esto es aceptable porque:
1. Re-index es una operacion esperada (ocurre al cambiar parametros del indice).
2. Los datos fuente estan en RocksDB — el indice se puede reconstruir siempre.
3. La frecuencia de schema changes en el grafo es muy baja.

### Dependencies

```toml
rkyv = { version = "0.8", features = ["validation"] }
postcard = { version = "1", features = ["alloc"] }
```

### Consequences

- (+) Startup instantaneo para grafos grandes via rkyv + mmap.
- (+) postcard compacto para el alto volumen de reads/writes en RocksDB.
- (+) Schema evolution donde importa (WAL, RocksDB values).
- (+) Cada formato usado donde aporta mas valor.
- (-) Dos formatos de serializacion a mantener y entender.
- (-) rkyv requiere `#[derive(Archive, Serialize, Deserialize)]` — duplicacion de derives.
- (-) No hay un unico formato para depurar todos los datos.

---

## 5. IDR-004 — Global Allocator: jemalloc

### Context

CVX es un sistema data-intensive con muchos threads concurrentes allocando/deallocando memoria:
- Threads de busqueda en Rayon allocando temporales para distancias, candidatos, resultados.
- Stages del pipeline de ingesta allocando buffers para batches, deltas, WAL entries.
- Tokio tasks allocando buffers de I/O.

### Decision

jemalloc como global allocator via `tikv-jemallocator`.

### El problema con system malloc

El allocator del sistema (glibc malloc, macOS malloc) usa un lock global o un numero limitado de arenas. Cuando muchos threads allocan simultaneamente, compiten por el lock — contention que escala linealmente con el numero de threads.

En benchmarks de Qdrant, cambiar de system malloc a jemalloc mejoro throughput de busqueda en ~15% bajo carga concurrente alta.

### La solucion de jemalloc

jemalloc usa **per-thread arenas**: cada thread tiene su propia arena de memoria, eliminando contention entre threads. Ademas:
- Mejor fragmentation handling para patrones de alloc/dealloc repetitivos (como los temporales de busqueda).
- Thread-local caches para allocaciones pequenas.
- Introspection via `malloc_stats` para debugging de memoria.

### Implementation

Solo en `cvx-server` (el binary), **no** en library crates:

```rust
// cvx-server/src/main.rs

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
```

El `#[cfg(not(target_env = "msvc"))]` excluye Windows MSVC. CVX no soporta Windows nativo — produccion es Docker (Linux), desarrollo es macOS/Linux.

**Dependency (solo en cvx-server):**

```toml
[dependencies]
tikv-jemallocator = "0.6"
```

### Quien lo usa

- **Qdrant:** jemalloc para su vector database.
- **TiKV:** jemalloc (mismo crate `tikv-jemallocator` — ellos lo mantienen).
- **Firefox:** jemalloc como allocator principal.

### Consequences

- (+) Elimina contention de allocator bajo carga concurrente.
- (+) Mejor fragmentation handling para workloads repetitivos.
- (+) Zero code changes en library crates — es transparente.
- (+) Battle-tested en produccion por TiKV, Qdrant, Firefox.
- (-) Anade ~1MB al binary size.
- (-) No disponible en Windows MSVC (no es un problema para CVX).
- (-) Debugging de memory leaks requiere herramientas jemalloc-aware (jemalloc's `prof` feature).

---

## 6. IDR-005 — SIMD Strategy: pulp

### Context

Distance computation (cosine, L2, dot product) es el **hot path** de CVX. Cada busqueda HNSW computa cientos a miles de distancias. SIMD (Single Instruction, Multiple Data) es obligatorio para rendimiento competitivo.

Requisitos:
1. Funcionar en **stable Rust** (no nightly).
2. Runtime dispatch: detectar automaticamente AVX2/AVX-512/NEON y usar la mejor ISA.
3. **Safe**: minimizar `unsafe` en el codigo de distancias.
4. Cross-platform: x86_64 (servers) y aarch64 (macOS dev, ARM servers).

### Decision

`pulp` como abstraccion SIMD primaria.

### Ejemplo completo: dot_product con pulp

```rust
use pulp::Simd;

/// Kernel SIMD para dot product.
struct DotProduct<'a> {
    a: &'a [f32],
    b: &'a [f32],
}

impl pulp::WithSimd for DotProduct<'_> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> f32 {
        let (a_head, a_tail) = S::f32s_as_simd(self.a);
        let (b_head, b_tail) = S::f32s_as_simd(self.b);

        let mut acc = simd.f32s_splat(0.0);
        for (&a, &b) in a_head.iter().zip(b_head) {
            acc = simd.f32s_mul_add(a, b, acc);
        }

        let mut sum = simd.f32s_reduce_sum(acc);
        for (&a, &b) in a_tail.iter().zip(b_tail) {
            sum += a * b;
        }
        sum
    }
}

/// Public safe API.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    pulp::Arch::new().dispatch(DotProduct { a, b })
}
```

El mismo codigo compila a:
- **AVX-512** en servers con Zen 4, Sapphire Rapids.
- **AVX2** en la mayoria de servidores x86_64.
- **NEON** en macOS Apple Silicon y ARM servers.
- **Scalar fallback** si no hay SIMD disponible.

La decision de que ISA usar se toma en runtime via `pulp::Arch::new()`.

### Por que pulp

| Criterio | pulp | std::arch intrinsics | packed_simd / std::simd |
|----------|------|---------------------|------------------------|
| Stable Rust | Si | Si | No (nightly) |
| Runtime dispatch | Automatico | Manual (is_x86_feature_detected!) | No |
| Safety | Safe (no unsafe) | Unsafe | Safe |
| Cross-platform | Una implementacion | Una por ISA | Una implementacion |
| Usado por | faer (fastest Rust LA library) | - | - |
| Downloads/month | ~2M | N/A | ~200K |

pulp es el motor SIMD de `faer`, la library de algebra lineal mas rapida en Rust. Esto nos da confianza de que el modelo de abstraccion no deja rendimiento en la mesa.

### SimdKernels trait

Las tres funciones criticas de CVX encapsuladas:

```rust
/// Trait que agrupa los kernels SIMD criticos para distance computation.
pub trait SimdKernels {
    /// Producto punto: sum(a[i] * b[i])
    fn dot_product(a: &[f32], b: &[f32]) -> f32;

    /// Distancia L2 al cuadrado: sum((a[i] - b[i])^2)
    fn l2_squared(a: &[f32], b: &[f32]) -> f32;

    /// Distancia coseno: 1 - (a . b) / (||a|| * ||b||)
    fn cosine_distance(a: &[f32], b: &[f32]) -> f32;
}

pub struct PulpKernels;

impl SimdKernels for PulpKernels {
    fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        pulp::Arch::new().dispatch(DotProduct { a, b })
    }

    fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
        pulp::Arch::new().dispatch(L2Squared { a, b })
    }

    fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
        let dot = Self::dot_product(a, b);
        let norm_a = Self::dot_product(a, a).sqrt();
        let norm_b = Self::dot_product(b, b).sqrt();
        1.0 - dot / (norm_a * norm_b + f32::EPSILON)
    }
}
```

### Fallback strategy

Si un benchmark especifico muestra que pulp deja rendimiento en la mesa (posible con AVX-512 masking o specific shuffle patterns), escribir intrinsics manuales **solo para esa funcion**. Pero pulp es el default para todo.

**Dependency:**

```toml
pulp = { version = "0.22", features = ["macro"] }
```

### Consequences

- (+) Un solo codigo para AVX2, AVX-512, NEON, y scalar.
- (+) Safe — no `unsafe` en el codigo de distancias.
- (+) Runtime dispatch automatico sin boilerplate.
- (+) Validated: faer usa pulp y logra rendimiento competitivo con MKL.
- (-) Abstraccion puede no exponer instrucciones muy especificas de AVX-512.
- (-) Dependency adicional (~compile time).
- (-) Debugging SIMD es inherentemente mas dificil que scalar.

---

## 7. IDR-006 — RocksDB Key Encoding & Column Families

### Context

Las keys de RocksDB se ordenan lexicograficamente. El diseno de las keys determina la eficiencia de los scans — un prefix scan con key bien disenada es O(matching keys), mientras un scan con key mal disenada requiere full iteration + filtering.

CVX necesita scans eficientes para:
- Todos los timestamps de una entidad en un espacio: `(entity_id, space_id, *)`
- Rango temporal de una entidad: `(entity_id, space_id, t_start..t_end)`
- Ultimo timestamp de una entidad: `(entity_id, space_id, MAX)` → seek + prev

### Decision

**Big-endian encoding** con sign-bit flip para timestamps, **column families separadas** por tipo de dato.

### Key format

```
entity_id (8 bytes BE u64) + space_id (4 bytes BE u32) + timestamp (8 bytes BE i64 con sign flip)
Total: 20 bytes
```

**Big-endian** porque la comparacion lexicografica de bytes coincide con la comparacion numerica cuando los enteros se codifican en big-endian.

**Sign-bit flip** para timestamps `i64`: XOR del primer byte con `0x80`. Esto convierte el ordenamiento de complemento a dos en ordenamiento lexicografico correcto:

```
i64 value    →  bytes (BE)           →  after XOR 0x80 on first byte
-2           →  FF FF FF FF FF FF FF FE  →  7F FF FF FF FF FF FF FE
-1           →  FF FF FF FF FF FF FF FF  →  7F FF FF FF FF FF FF FF
 0           →  00 00 00 00 00 00 00 00  →  80 00 00 00 00 00 00 00
 1           →  00 00 00 00 00 00 00 01  →  80 00 00 00 00 00 00 01
 2           →  00 00 00 00 00 00 00 02  →  80 00 00 00 00 00 00 02
```

Despues del flip, el orden lexicografico de los bytes coincide con el orden numerico de los `i64`, incluyendo valores negativos.

### Funciones encode/decode

```rust
/// Codifica una key para RocksDB.
/// entity_id + space_id + timestamp → 20 bytes, lexicographically ordered.
pub fn encode_key(entity_id: u64, space_id: u32, timestamp: i64) -> [u8; 20] {
    let mut key = [0u8; 20];
    key[0..8].copy_from_slice(&entity_id.to_be_bytes());
    key[8..12].copy_from_slice(&space_id.to_be_bytes());

    let mut ts_bytes = timestamp.to_be_bytes();
    ts_bytes[0] ^= 0x80; // Sign-bit flip for correct lexicographic ordering
    key[12..20].copy_from_slice(&ts_bytes);

    key
}

/// Decodifica una key de RocksDB.
pub fn decode_key(key: &[u8; 20]) -> (u64, u32, i64) {
    let entity_id = u64::from_be_bytes(key[0..8].try_into().unwrap());
    let space_id = u32::from_be_bytes(key[8..12].try_into().unwrap());

    let mut ts_bytes: [u8; 8] = key[12..20].try_into().unwrap();
    ts_bytes[0] ^= 0x80; // Reverse sign-bit flip
    let timestamp = i64::from_be_bytes(ts_bytes);

    (entity_id, space_id, timestamp)
}
```

### Column Families

Cada column family tiene su propia configuracion de compresion, bloom filters, y block cache, optimizada para su patron de acceso.

| Column Family | Compression | Bloom Filter | Block Cache | Notes |
|---------------|-------------|--------------|-------------|-------|
| `vectors` | None | Prefix bloom (12 bytes) | 256 MB | Hot path, no compression overhead |
| `deltas` | LZ4 | Prefix bloom (12 bytes) | 64 MB | Sparse data compresses well |
| `timelines` | LZ4 | Full bloom | 32 MB | Metadata por entidad |
| `metadata` | LZ4 | Full bloom | 32 MB | Entity metadata, space definitions |
| `changepoints` | Zstd | None | 16 MB | Infrequent access, max compression |
| `views` | Zstd | None | 16 MB | Materialized views, read-heavy but infrequent |
| `system` | None | None | 8 MB | Internal state, WAL checkpoints |

**Prefix bloom (12 bytes)** en `vectors` y `deltas`: cubre `entity_id (8) + space_id (4)`, permitiendo que RocksDB descarte SST files que no contienen la entidad+espacio buscados sin leer ningun bloque.

### Configuracion por Column Family

```rust
use rocksdb::{Options, BlockBasedOptions, SliceTransform};

fn configure_cf(name: &str) -> Options {
    let mut opts = Options::default();
    let mut block_opts = BlockBasedOptions::default();

    match name {
        "vectors" => {
            // Hot path: no compression, large cache, prefix bloom
            opts.set_compression_type(rocksdb::DBCompressionType::None);
            block_opts.set_block_cache(&rocksdb::Cache::new_lru_cache(256 * 1024 * 1024));
            block_opts.set_bloom_filter(10.0, false);
            opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(12));
        }
        "deltas" => {
            opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
            block_opts.set_block_cache(&rocksdb::Cache::new_lru_cache(64 * 1024 * 1024));
            block_opts.set_bloom_filter(10.0, false);
            opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(12));
        }
        "timelines" | "metadata" => {
            opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
            block_opts.set_block_cache(&rocksdb::Cache::new_lru_cache(32 * 1024 * 1024));
            block_opts.set_bloom_filter(10.0, false);
        }
        "changepoints" | "views" => {
            opts.set_compression_type(rocksdb::DBCompressionType::Zstd);
            block_opts.set_block_cache(&rocksdb::Cache::new_lru_cache(16 * 1024 * 1024));
            // No bloom filter — infrequent access
        }
        "system" => {
            opts.set_compression_type(rocksdb::DBCompressionType::None);
            block_opts.set_block_cache(&rocksdb::Cache::new_lru_cache(8 * 1024 * 1024));
        }
        _ => {}
    }

    opts.set_block_based_table_factory(&block_opts);
    opts
}
```

### Consequences

- (+) Prefix scan para una entidad es O(matching keys) — RocksDB salta SST files via prefix bloom.
- (+) Sign-bit flip permite range scans correctos con timestamps negativos (pre-epoch).
- (+) Column families aisladas permiten tuning independiente (compresion, cache, compaction).
- (+) Key fija de 20 bytes — no overhead de length-prefix.
- (-) Key de 20 bytes por entry (vs variable-length encoding que podria ser ~12 bytes para entities pequenas).
- (-) 7 column families = 7x file descriptors abiertos. Necesita `ulimit` adecuado.
- (-) Sign-bit flip requiere documentacion clara — es un truco no obvio.

---

## 8. IDR-007 — Error Handling: thiserror (libs) + anyhow (binary)

### Context

CVX necesita:
1. **Errores estructurados** en las libraries: el API layer debe hacer pattern matching para mapear errores a HTTP status codes.
2. **Propagacion ergonomica** en el binary: startup, shutdown, y orchestracion no necesitan errores tipados — solo contexto para debugging.

### Decision

`thiserror` en todas las library crates, `anyhow` solo en `cvx-server`.

### Error hierarchy

```
cvx-core:    CvxError    (enum que wrappea todos los errores de subsistemas)
cvx-index:   IndexError
cvx-storage: StorageError
cvx-analytics: AnalyticsError
cvx-query:   QueryError
cvx-ingest:  IngestError
cvx-explain: ExplainError
cvx-api:     CvxError → HTTP status codes + ApiError (safe, sin detalles internos)
cvx-server:  anyhow::Result (startup/shutdown, context chaining)
```

### CvxError — el error raiz

```rust
// cvx-core/src/error.rs

use thiserror::Error;

#[derive(Error, Debug)]
pub enum CvxError {
    #[error("Index error: {0}")]
    Index(#[from] IndexError),

    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),

    #[error("Analytics error: {0}")]
    Analytics(#[from] AnalyticsError),

    #[error("Query error: {0}")]
    Query(#[from] QueryError),

    #[error("Ingest error: {0}")]
    Ingest(#[from] IngestError),

    #[error("Explain error: {0}")]
    Explain(#[from] ExplainError),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Invalid configuration: {0}")]
    Config(String),
}
```

### Ejemplo de subsistema: StorageError

```rust
// cvx-storage/src/error.rs

use thiserror::Error;

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("RocksDB error: {0}")]
    RocksDb(#[from] rocksdb::Error),

    #[error("WAL corruption at offset {offset}: {reason}")]
    WalCorruption { offset: u64, reason: String },

    #[error("Key not found: entity={entity_id} space={space_id} ts={timestamp}")]
    KeyNotFound {
        entity_id: u64,
        space_id: u32,
        timestamp: i64,
    },

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Column family '{0}' not found")]
    CfNotFound(String),

    #[error("Tier migration failed: {reason}")]
    TierMigration { reason: String },
}
```

### Mapping CvxError → HTTP

```rust
// cvx-api/src/error.rs

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

#[derive(Serialize)]
pub struct ApiError {
    pub code: u16,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

impl IntoResponse for CvxError {
    fn into_response(self) -> Response {
        let (status, api_error) = match &self {
            // User errors — specific message
            CvxError::Query(QueryError::InvalidDimension { expected, got }) => (
                StatusCode::BAD_REQUEST,
                ApiError {
                    code: 400,
                    message: format!("Vector dimension mismatch: expected {expected}, got {got}"),
                    details: None,
                },
            ),
            CvxError::Storage(StorageError::KeyNotFound { entity_id, .. }) => (
                StatusCode::NOT_FOUND,
                ApiError {
                    code: 404,
                    message: format!("Entity {entity_id} not found"),
                    details: None,
                },
            ),
            CvxError::Ingest(IngestError::InvalidBatch { reason }) => (
                StatusCode::BAD_REQUEST,
                ApiError {
                    code: 400,
                    message: reason.clone(),
                    details: None,
                },
            ),

            // Internal errors — generic message, log the real error
            _ => {
                tracing::error!(?self, "Internal error");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    ApiError {
                        code: 500,
                        message: "Internal server error".to_string(),
                        details: None,
                    },
                )
            }
        };

        (status, Json(api_error)).into_response()
    }
}
```

**Principio clave:** los errores internos (RocksDB corruption, serialization failures) NUNCA se exponen al usuario. Se loguean con `tracing::error!` y se devuelve un mensaje generico. Los errores de usuario (dimension mismatch, entity not found, invalid batch) devuelven detalles especificos.

### anyhow en cvx-server

```rust
// cvx-server/src/main.rs

use anyhow::{Context, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let config = load_config()
        .context("Failed to load configuration")?;

    let storage = Storage::open(&config.storage)
        .await
        .context("Failed to open storage engine")?;

    let index = Index::load(&config.index)
        .await
        .context("Failed to load HNSW index")?;

    serve(config, storage, index)
        .await
        .context("Server terminated with error")
}
```

anyhow's `.context()` crea una cadena de errores legible para diagnostico:

```
Error: Server terminated with error

Caused by:
    0: Failed to open storage engine
    1: RocksDB error: IO error: No such file or directory: /data/cvx/db/LOCK
```

### Consequences

- (+) Pattern matching preciso en API layer para HTTP status codes.
- (+) Errores seguros: nunca se leak detalles internos al usuario.
- (+) anyhow ergonomico para startup/shutdown (donde no se necesita pattern matching).
- (+) thiserror genera implementaciones de `Display` y `From` automaticamente.
- (-) Conversion boilerplate entre errores de subsistemas (mitigado por `#[from]`).
- (-) Dos paradigmas de error handling en el codebase.

---

## 9. IDR-008 — Testing Strategy

### Context

CVX tiene codigo diverso: desde aritmetica SIMD (donde un off-by-one corrompe resultados) hasta HTTP APIs (donde la ergonomia importa mas que la precision numerica). Cada tipo necesita testing apropiado.

### Decision

**5 niveles de testing**, mas fuzzing como nivel futuro.

### Level 1: Unit Tests (#[test])

Tests estandar para logica pura. Nada especial — `assert_eq!`, `assert!`, `#[should_panic]`.

```rust
#[test]
fn delta_encode_sparse() {
    let prev = vec![1.0, 2.0, 3.0, 4.0];
    let curr = vec![1.0, 2.1, 3.0, 4.2];
    let delta = delta_encode(&prev, &curr, 0.05);
    assert_eq!(delta.indices, vec![1, 3]);
    assert_approx_eq!(delta.values, vec![0.1, 0.2], 1e-6);
}
```

### Level 2: Property-Based Testing (proptest)

Para invariantes matematicas que deben cumplirse para CUALQUIER input, no solo los ejemplos que imaginamos.

**Distance metrics — symmetry, non-negativity, identity, triangle inequality:**

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn cosine_distance_is_symmetric(
        a in prop::collection::vec(-1.0f32..1.0, 128),
        b in prop::collection::vec(-1.0f32..1.0, 128),
    ) {
        let d_ab = PulpKernels::cosine_distance(&a, &b);
        let d_ba = PulpKernels::cosine_distance(&b, &a);
        prop_assert!((d_ab - d_ba).abs() < 1e-6);
    }

    #[test]
    fn l2_distance_non_negative(
        a in prop::collection::vec(-100.0f32..100.0, 128),
        b in prop::collection::vec(-100.0f32..100.0, 128),
    ) {
        let d = PulpKernels::l2_squared(&a, &b);
        prop_assert!(d >= 0.0);
    }

    #[test]
    fn l2_distance_identity(
        a in prop::collection::vec(-100.0f32..100.0, 128),
    ) {
        let d = PulpKernels::l2_squared(&a, &a);
        prop_assert!(d.abs() < 1e-6);
    }

    #[test]
    fn l2_triangle_inequality(
        a in prop::collection::vec(-10.0f32..10.0, 64),
        b in prop::collection::vec(-10.0f32..10.0, 64),
        c in prop::collection::vec(-10.0f32..10.0, 64),
    ) {
        let d_ab = PulpKernels::l2_squared(&a, &b).sqrt();
        let d_bc = PulpKernels::l2_squared(&b, &c).sqrt();
        let d_ac = PulpKernels::l2_squared(&a, &c).sqrt();
        prop_assert!(d_ac <= d_ab + d_bc + 1e-4); // epsilon for float imprecision
    }
}
```

**Key encoding — roundtrip + ordering preservation:**

```rust
proptest! {
    #[test]
    fn key_encoding_roundtrip(
        entity_id in any::<u64>(),
        space_id in any::<u32>(),
        timestamp in any::<i64>(),
    ) {
        let encoded = encode_key(entity_id, space_id, timestamp);
        let (e, s, t) = decode_key(&encoded);
        prop_assert_eq!(e, entity_id);
        prop_assert_eq!(s, space_id);
        prop_assert_eq!(t, timestamp);
    }

    #[test]
    fn key_ordering_preserves_timestamp_order(
        entity_id in any::<u64>(),
        space_id in any::<u32>(),
        t1 in any::<i64>(),
        t2 in any::<i64>(),
    ) {
        let k1 = encode_key(entity_id, space_id, t1);
        let k2 = encode_key(entity_id, space_id, t2);
        // Same entity+space: key ordering must match timestamp ordering
        prop_assert_eq!(k1.cmp(&k2), t1.cmp(&t2));
    }
}
```

**Delta encoding — roundtrip:**

```rust
proptest! {
    #[test]
    fn delta_encode_decode_roundtrip(
        base in prop::collection::vec(-100.0f32..100.0, 128),
        perturbation in prop::collection::vec(-0.1f32..0.1, 128),
    ) {
        let current: Vec<f32> = base.iter()
            .zip(&perturbation)
            .map(|(b, p)| b + p)
            .collect();
        let delta = delta_encode(&base, &current, 0.001);
        let reconstructed = delta_decode(&base, &delta);
        for (orig, recon) in current.iter().zip(&reconstructed) {
            prop_assert!((orig - recon).abs() < 0.002); // threshold tolerance
        }
    }
}
```

**HNSW — connectivity invariant:**

```rust
proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))] // Expensive test
    #[test]
    fn hnsw_all_nodes_reachable(
        vectors in prop::collection::vec(
            prop::collection::vec(-1.0f32..1.0, 32),
            10..100
        ),
    ) {
        let mut graph = HnswGraph::new(HnswParams::default());
        for (i, v) in vectors.iter().enumerate() {
            graph.insert(i as u64, v);
        }
        // BFS from entry point must reach all nodes
        let reachable = graph.bfs_from_entry();
        prop_assert_eq!(reachable.len(), vectors.len());
    }
}
```

**Serialization — postcard/rkyv roundtrip:**

```rust
proptest! {
    #[test]
    fn postcard_roundtrip_wal_entry(
        entity_id in any::<u64>(),
        timestamp in any::<i64>(),
        vector in prop::collection::vec(-100.0f32..100.0, 128),
    ) {
        let entry = WalEntry {
            entity_id,
            timestamp,
            vector: vector.clone(),
            space_id: None,
            schema_version: 1,
        };
        let bytes = postcard::to_allocvec(&entry).unwrap();
        let decoded: WalEntry = postcard::from_bytes(&bytes).unwrap();
        prop_assert_eq!(decoded.entity_id, entity_id);
        prop_assert_eq!(decoded.timestamp, timestamp);
        prop_assert_eq!(decoded.vector, vector);
    }
}
```

### Level 3: Integration Tests

Tests con RocksDB real usando `tempdir`, in-process. Validan el flujo completo: write → read → scan → delete.

```rust
#[test]
fn storage_write_and_range_scan() {
    let dir = tempdir().unwrap();
    let storage = Storage::open(dir.path()).unwrap();

    // Insert 100 timestamps for entity 42, space 0
    for t in 0..100i64 {
        let vector = vec![t as f32; 128];
        storage.put_vector(42, 0, t, &vector).unwrap();
    }

    // Range scan [10, 20)
    let results = storage.range_scan(42, 0, 10, 20).unwrap();
    assert_eq!(results.len(), 10);
    assert_eq!(results[0].timestamp, 10);
    assert_eq!(results[9].timestamp, 19);
}
```

### Level 4: E2E Tests

Pocos tests con `TestServer` real (HTTP/gRPC stack completo). Para validar que el API layer funciona end-to-end.

```rust
#[tokio::test]
async fn e2e_insert_and_search() {
    let server = TestServer::start().await;

    // Insert via REST
    let resp = server.client()
        .post("/api/v1/vectors")
        .json(&json!({
            "entity_id": 42,
            "vector": vec![1.0f32; 128],
            "timestamp": 1000
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);

    // Search via REST
    let resp = server.client()
        .post("/api/v1/search")
        .json(&json!({
            "vector": vec![1.0f32; 128],
            "k": 5
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let results: SearchResponse = resp.json().await.unwrap();
    assert_eq!(results.results[0].entity_id, 42);
}
```

### Level 5: Benchmarks

`criterion` con regression detection en CI.

```rust
fn bench_cosine_distance(c: &mut Criterion) {
    let a: Vec<f32> = (0..768).map(|i| (i as f32).sin()).collect();
    let b: Vec<f32> = (0..768).map(|i| (i as f32).cos()).collect();

    c.bench_function("cosine_768d", |bencher| {
        bencher.iter(|| {
            black_box(PulpKernels::cosine_distance(
                black_box(&a),
                black_box(&b),
            ))
        })
    });
}
```

### Level 6 (Future): Fuzzing

`cargo-fuzz` para ingestion API — input arbitrario no debe causar panics ni undefined behavior.

### Coverage por layer

| Test Level | Crates | Frecuencia en CI |
|------------|--------|-----------------|
| Unit (#[test]) | Todos | Cada push |
| Property (proptest) | cvx-core, cvx-index, cvx-storage | Cada push (subset), full en nightly |
| Integration | cvx-storage, cvx-ingest | Cada push |
| E2E | cvx-api | PRs + nightly |
| Benchmarks (criterion) | cvx-index, cvx-core | PRs que tocan codigo perf-sensitive |
| Fuzzing | cvx-api | Nightly (cuando este implementado) |

### CI Configuration

- **Cada push:** `cargo test` (unit + integration + proptest con cases reducidos).
- **Proptests exhaustivos:** Marcados con `#[ignore]`, ejecutados en full build nightly: `cargo test -- --ignored`.
- **Criterion benchmarks:** En PRs que tocan `cvx-index` o `cvx-core`, con comparacion contra baseline.
- **Miri:** En crates sin `unsafe` (`cvx-core`, `cvx-analytics`) para detectar undefined behavior.

### Consequences

- (+) Coverage exhaustiva sin sacrificar velocidad de CI en el dia a dia.
- (+) Property-based tests capturan edge cases que los unit tests manuales no cubren.
- (+) Criterion detecta regresiones de performance automaticamente.
- (-) proptest anade tiempo de compilacion (~10s).
- (-) Los proptests exhaustivos tardan minutos — solo viables en nightly.
- (-) E2E tests son fragiles y lentos — mantener pocos pero significativos.

---

## 10. IDR-009 — Unsafe Policy

### Context

Las safety guarantees de Rust son una propuesta de valor clave. `unsafe` debe minimizarse y aislarse para mantener la confianza en el sistema.

### Decision

`#![deny(unsafe_code)]` por defecto, `#![allow(unsafe_code)]` solo en 2 crates.

### Mapa de crates

```
cvx-core        #![deny(unsafe_code)]
cvx-index       #![allow(unsafe_code)]  — SIMD fallback, mmap
cvx-storage     #![allow(unsafe_code)]  — mmap (WAL/index files)
cvx-ingest      #![deny(unsafe_code)]
cvx-analytics   #![deny(unsafe_code)]
cvx-query       #![deny(unsafe_code)]
cvx-explain     #![deny(unsafe_code)]
cvx-api         #![deny(unsafe_code)]
cvx-server      #![deny(unsafe_code)]
```

### Reglas

1. **`#![deny(unsafe_code)]` como default.** Cada crate nuevo empieza con deny.
2. **Solo `cvx-index` y `cvx-storage` pueden usar `unsafe`.** Si otro crate lo necesita, primero evaluar si la funcionalidad puede moverse a uno de estos dos.
3. **Cada bloque `unsafe` requiere un comentario `// SAFETY:`** documentando los invariantes que el programador garantiza.
4. **Unsafe siempre encapsulado en wrappers publicos safe.** El consumidor nunca ve `unsafe`.
5. **CI ejecuta Miri en crates pure-Rust** (`cvx-core`, `cvx-analytics`) para detectar undefined behavior.

### Ejemplo: SAFETY comment y safe wrapper

```rust
// cvx-index/src/simd/fallback.rs

/// Computes dot product using AVX2 intrinsics.
/// Only used if pulp benchmark shows >10% gap for this specific function.
///
/// # Safety invariants documented below.
pub fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have same length");
    assert!(a.len() >= 8, "vectors must have at least 8 elements for AVX2");

    // SAFETY:
    // - We verified a.len() == b.len() above.
    // - We verified len >= 8, so the SIMD loads won't read past the slice.
    // - AVX2 is available (checked by caller via is_x86_feature_detected!).
    // - The pointers are valid because they come from valid slices.
    unsafe {
        dot_product_avx2_inner(a.as_ptr(), b.as_ptr(), a.len())
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_product_avx2_inner(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::x86_64::*;
    // ... AVX2 implementation ...
    todo!()
}
```

El usuario de `cvx-index` llama `dot_product_avx2(&a, &b)` — una funcion safe. Los asserts en el wrapper garantizan los invariantes antes de entrar en `unsafe`.

### Ejemplo: mmap safe wrapper

```rust
// cvx-storage/src/mmap.rs

use memmap2::MmapOptions;
use std::fs::File;

pub struct MmappedFile {
    mmap: memmap2::Mmap,
}

impl MmappedFile {
    /// Opens a file as a read-only memory map.
    ///
    /// The caller must ensure the file is not modified externally
    /// while the MmappedFile is alive.
    pub fn open(path: &std::path::Path) -> Result<Self, StorageError> {
        let file = File::open(path)
            .map_err(|e| StorageError::Io(e))?;

        // SAFETY:
        // - The file is opened read-only.
        // - We document that external modification is undefined behavior.
        // - The Mmap is dropped before the File (Rust drop order: fields in declaration order).
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        Ok(MmappedFile { mmap })
    }

    /// Returns the memory-mapped contents as a byte slice.
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }
}
```

### Donde se necesita unsafe

| Uso | Crate | Motivo |
|-----|-------|--------|
| SIMD intrinsics (fallback) | cvx-index | Si pulp no cubre un caso especifico |
| rkyv zero-copy access | cvx-index | `archived_root` requiere unsafe para acceso directo |
| mmap file mapping | cvx-storage | `memmap2::Mmap::map()` es inherently unsafe |
| mmap for WAL | cvx-storage | Write-ahead log memory-mapped |

### Consequences

- (+) 7 de 9 crates son 100% safe Rust — alta confianza en correctness.
- (+) Los 2 crates con unsafe son los mas testeados (distance metrics con proptest, storage con integration tests).
- (+) SAFETY comments fuerzan al autor a pensar y documentar invariantes.
- (+) Miri en CI detecta problemas que los tests normales no ven.
- (-) deny(unsafe_code) puede requerir workarounds verbosos en algunos casos.
- (-) No se puede usar Miri en crates que dependen de FFI (RocksDB).

---

## 11. IDR-010 — Index Persistence: Progressive Strategy

### Context

El grafo ST-HNSW debe sobrevivir restarts. El tamano varia segun el numero de vectores:

| Vectors | Graph Size (approx) |
|---------|-------------------|
| 100K | ~8.4 MB |
| 1M | ~84 MB |
| 10M | ~840 MB |
| 100M | ~8.4 GB |

Para 1M vectores, cargar 84MB desde disco es rapido. Para 100M vectores, cargar 8.4GB introduce un startup delay significativo.

### Decision

**Estrategia progresiva:** read-into-memory ahora, mmap + prefetch despues.

### Phase 1 (Layer 2-4): Read-into-memory con rkyv

Simple y predecible. El servidor no acepta queries hasta que el indice esta completamente cargado.

```rust
// Save
pub fn save_index(graph: &HnswGraph, path: &Path) -> Result<(), StorageError> {
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(graph)
        .map_err(|e| StorageError::Serialization(e.to_string()))?;

    // Atomic write: write to .tmp, then rename
    let tmp_path = path.with_extension("tmp");
    std::fs::write(&tmp_path, &bytes)?;
    std::fs::rename(&tmp_path, path)?;

    tracing::info!(
        path = %path.display(),
        size_mb = bytes.len() / (1024 * 1024),
        "Index saved"
    );
    Ok(())
}

// Load
pub fn load_index(path: &Path) -> Result<HnswGraph, StorageError> {
    let bytes = std::fs::read(path)?;

    let archived = rkyv::access::<rkyv::Archived<HnswGraph>, rkyv::rancor::Error>(&bytes)
        .map_err(|e| StorageError::Serialization(e.to_string()))?;

    let graph: HnswGraph = rkyv::deserialize::<HnswGraph, rkyv::rancor::Error>(archived)
        .map_err(|e| StorageError::Serialization(e.to_string()))?;

    tracing::info!(
        path = %path.display(),
        nodes = graph.len(),
        "Index loaded"
    );
    Ok(graph)
}
```

**Rendimiento estimado:** ~200ms para 1M vectores (84MB) en SSD. Aceptable para startup.

### Phase 2 (Layer 11+): mmap + background prefetch

Cuando el indice excede ~1GB, el startup delay se vuelve problematico. La solucion:

1. **mmap** el archivo — el servidor esta "ready" inmediatamente (el OS mapea paginas on-demand).
2. **Background thread** prefetches paginas secuencialmente con `madvise(SEQUENTIAL)` + touch every 4KB page.
3. El servidor puede servir queries **mientras el prefetch corre**. Las primeras queries pueden ser mas lentas (page faults), pero el servidor esta disponible.

```rust
pub struct MmappedIndex {
    mmap: memmap2::Mmap,
    prefetch_complete: Arc<AtomicBool>,
}

impl MmappedIndex {
    pub fn open(path: &Path) -> Result<Self, StorageError> {
        let file = std::fs::File::open(path)?;

        // SAFETY: file is read-only, not modified externally while mapped.
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };

        // Advise the OS for sequential prefetch
        #[cfg(unix)]
        mmap.advise(memmap2::Advice::Sequential)?;

        let prefetch_complete = Arc::new(AtomicBool::new(false));

        // Background prefetch: touch every page to bring it into RAM
        let mmap_ptr = mmap.as_ptr() as usize;
        let mmap_len = mmap.len();
        let complete = prefetch_complete.clone();

        std::thread::spawn(move || {
            let mut dummy: u8 = 0;
            let ptr = mmap_ptr as *const u8;
            for offset in (0..mmap_len).step_by(4096) {
                // SAFETY: offset < mmap_len, so ptr.add(offset) is within the mapping.
                dummy = dummy.wrapping_add(unsafe { *ptr.add(offset) });
            }
            // Prevent optimizer from removing the reads
            std::hint::black_box(dummy);
            complete.store(true, std::sync::atomic::Ordering::Release);
            tracing::info!(size_mb = mmap_len / (1024 * 1024), "Index prefetch complete");
        });

        Ok(MmappedIndex {
            mmap,
            prefetch_complete,
        })
    }

    pub fn is_warm(&self) -> bool {
        self.prefetch_complete.load(std::sync::atomic::Ordering::Acquire)
    }
}
```

### Abstraccion: IndexStorage enum

El resto del codigo no sabe si el indice esta en memoria o mmap'd:

```rust
pub enum IndexStorage {
    /// Full graph deserialized in memory (Phase 1).
    InMemory(HnswGraph),
    /// Memory-mapped rkyv archive (Phase 2).
    Mmap(MmappedIndex),
}

impl IndexStorage {
    /// Access neighbors of a node. Works identically for both backends.
    pub fn neighbors(&self, node_id: u64, layer: usize) -> &[u64] {
        match self {
            IndexStorage::InMemory(graph) => graph.neighbors(node_id, layer),
            IndexStorage::Mmap(mmap) => {
                let archived = rkyv::access::<rkyv::Archived<HnswGraph>, rkyv::rancor::Error>(
                    mmap.as_bytes()
                ).expect("index file validated at open time");
                // Access neighbors from archived data without deserialization
                &archived.nodes[&node_id].layers[layer].neighbors
            }
        }
    }

    /// Number of nodes in the index.
    pub fn len(&self) -> usize {
        match self {
            IndexStorage::InMemory(graph) => graph.len(),
            IndexStorage::Mmap(mmap) => {
                let archived = rkyv::access::<rkyv::Archived<HnswGraph>, rkyv::rancor::Error>(
                    mmap.as_bytes()
                ).expect("index file validated at open time");
                archived.nodes.len()
            }
        }
    }
}
```

### Consequences

- (+) Phase 1 es simple, predecible, y suficiente hasta ~1M vectores.
- (+) Phase 2 elimina startup delay — server ready en milliseconds independiente del tamano del indice.
- (+) El `IndexStorage` enum permite migrar sin cambiar el resto del codigo.
- (+) Background prefetch permite servir queries inmediatamente (con posibles page faults iniciales).
- (-) Phase 2 depende de que el OS no evict las paginas bajo memory pressure.
- (-) mmap con rkyv requiere que el formato del archivo no cambie (no schema evolution).
- (-) Dos code paths a mantener y testear.

---

## 12. Dependency Summary

Todas las dependencias decididas en este documento y en el RFC-001, consolidadas:

```toml
# ─── Concurrency & Async ───────────────────────────────
tokio = { version = "1", features = ["full"] }
rayon = "1"
parking_lot = "0.12"

# ─── Serialization ─────────────────────────────────────
rkyv = { version = "0.8", features = ["validation"] }
postcard = { version = "1", features = ["alloc"] }
serde = { version = "1", features = ["derive"] }

# ─── SIMD ──────────────────────────────────────────────
pulp = { version = "0.22", features = ["macro"] }

# ─── Collections ───────────────────────────────────────
smallvec = { version = "1", features = ["union"] }

# ─── Allocator (solo en cvx-server) ───────────────────
tikv-jemallocator = "0.6"

# ─── Error Handling ────────────────────────────────────
thiserror = "2"
anyhow = "1"  # solo en cvx-server

# ─── Testing ──────────────────────────────────────────
proptest = "1"
criterion = { version = "0.5", features = ["html_reports"] }

# ─── Observability ────────────────────────────────────
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# ─── Storage ──────────────────────────────────────────
# rocksdb = "0.22"  (added in Layer 3)
# memmap2 = "0.9"   (added in Layer 11)

# ─── API ──────────────────────────────────────────────
# axum = "0.7"      (added in Layer 5)
# tonic = "0.12"    (added in Layer 5)
# prost = "0.13"    (added in Layer 5)

# ─── Analytics ────────────────────────────────────────
# burn = "0.15"     (added in Layer 10)
```

Cada dependencia se introduce en el Layer del roadmap donde se necesita por primera vez. Las dependencias comentadas se agregan en layers posteriores.

---

## 13. Future Considerations

Estas decisiones no son permanentes. Cada una puede evolucionar si las condiciones cambian:

| Decision | Possible Evolution | Trigger |
|----------|-------------------|---------|
| Rayon | Custom thread pool con priority scheduling | Si la latencia de busqueda sufre bajo ingesta pesada |
| pulp | Manual intrinsics para funciones especificas | Si benchmark muestra >10% gap vs hand-tuned |
| postcard | Considerar rkyv para RocksDB hot-path values | Si profiling muestra deserialization como bottleneck |
| RwLock on index | Lock-free concurrent HNSW | Si write contention se vuelve medible |
| jemalloc | mimalloc | Si p99 latency es mas critico que throughput |
| Read-into-memory | mmap + prefetch | Cuando el indice excede ~1GB |
| Column families | Merge CFs pequenas | Si file descriptor pressure en entornos constrained |
| DashMap | Reemplazar `RwLock<HashMap>` especificos | Si profiling muestra contention en un mapa particular |
| anyhow in binary | Structured errors everywhere | Si necesitamos error telemetry con categorias en startup |
| proptest cases | Aumentar cases default | Si encontramos bugs que solo aparecen con mas iteraciones |

**Principio general:** no optimizar hasta que el profiler lo pida. Cada IDR documenta la *siguiente* opcion si la decision actual resulta insuficiente — asi el futuro yo no parte de cero.

---

## Appendix A: Decision Timeline

| IDR | Decided | First Used In | Revisable From |
|-----|---------|---------------|----------------|
| 001 Concurrency | Design phase | Layer 1 (cvx-core) | Layer 5 (bajo carga real) |
| 002 Rayon | Design phase | Layer 2 (cvx-index) | Layer 5 (benchmarks) |
| 003 Serialization | Design phase | Layer 2 (rkyv), Layer 3 (postcard) | Layer 6 (con datos reales) |
| 004 jemalloc | Design phase | Layer 5 (cvx-server) | Layer 5 (benchmarks A/B) |
| 005 pulp | Design phase | Layer 2 (cvx-index) | Layer 2 (criterion benchmarks) |
| 006 RocksDB keys | Design phase | Layer 3 (cvx-storage) | Layer 6 (scan benchmarks) |
| 007 Error handling | Design phase | Layer 1 (cvx-core) | Stable — unlikely to change |
| 008 Testing | Design phase | Layer 1 onwards | Continuous improvement |
| 009 Unsafe policy | Design phase | Layer 1 (crate attributes) | Stable — unlikely to relax |
| 010 Index persistence | Design phase | Layer 2 (Phase 1), Layer 11 (Phase 2) | Layer 8+ (size benchmarks) |
