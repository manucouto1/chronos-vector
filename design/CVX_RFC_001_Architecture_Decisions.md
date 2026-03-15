# RFC-001: ChronosVector Architecture Decisions

**Status:** Proposed  
**Author:** Manuel Couto Pintos  
**Date:** March 2026  
**Reviewers:** (Open for comments)

---

## 1. Summary

Este RFC documenta las decisiones arquitecturales fundamentales de ChronosVector (CVX), sus alternativas consideradas, y la justificación para cada elección. Cada decisión se registra como un ADR (Architecture Decision Record) independiente, facilitando la trazabilidad y la revisión futura.

---

## 2. Motivation

CVX combina tres dominios (vector search, temporal modeling, drift analytics) en un solo sistema. Muchas de las decisiones arquitecturales no tienen precedente directo porque ningún sistema existente integra estos tres dominios. Este RFC sirve como registro permanente del *por qué* detrás de cada elección, no solo del *qué*.

---

## 3. Architecture Decision Records

### ADR-001: Rust as Implementation Language

**Context:** CVX necesita control de memoria a nivel de byte para SIMD, zero-copy data paths, y concurrencia segura para acceso compartido al índice.

**Decision:** Implementar en Rust (edition 2021, MSRV 1.75+).

**Alternatives Considered:**
- **C++:** Máximo control, pero sin safety guarantees. El modelo de ownership de Rust previene data races en el índice concurrente sin runtime cost.
- **Go:** Excelente concurrencia, pero el garbage collector introduce latencia impredecible en el hot path de búsqueda. Qdrant (Rust) demuestra ventaja sobre Milvus (Go/C++) en latencia p99.
- **Zig:** Prometedor, pero ecosistema de crates inmaduro para las dependencias que necesitamos (RocksDB bindings, Arrow, protobuf).

**Consequences:**
- (+) SIMD explícito sin overhead de runtime.
- (+) Concurrencia fearless para índice shared-nothing por nodo.
- (+) Ecosystem: `rocksdb`, `arrow-rs`, `tonic`, `axum`, `burn` disponibles y maduros.
- (-) Curva de aprendizaje para contribuidores.
- (-) Compile times significativos en workspace multi-crate.

---

### ADR-002: Composite Spatiotemporal Distance

**Context:** El enfoque convencional (filtrar por tiempo, luego buscar por distancia semántica) descarta la relación continua entre proximidad temporal y semántica.

**Decision:** Usar distancia compuesta $d_{ST} = \alpha \cdot d_{sem} + (1-\alpha) \cdot d_{time} \cdot decay$ como función nativa del índice, con $\alpha$ configurable por query.

**Alternatives Considered:**
- **Post-filtering:** Buscar kNN semántico, luego filtrar por rango temporal. Simple, pero ignora vectores semánticamente lejanos pero temporalmente relevantes.
- **Pre-filtering con Roaring Bitmaps:** Calcular primero el set de IDs válidos temporalmente, luego buscar kNN solo dentro de ese set. Eficiente, pero no permite *gradación* temporal (un vector de ayer y uno de hace un año tienen el mismo peso).
- **Separate indices per time bucket:** Construir un HNSW por bucket temporal. Alto consumo de memoria, O(T) índices.

**Consequences:**
- (+) Queries expresivas: el usuario controla el trade-off temporal/semántico via α.
- (+) Un solo índice para todas las queries temporales.
- (-) La función de distancia compuesta es más costosa (~2x) que distancia pura.
- (-) α óptimo depende del use case; requiere experimentación por dominio.

**Open Question:** ¿La combinación lineal es suficiente? Podría necesitarse un modelo aprendido que combine ambas distancias de forma no-lineal. Deferred to Phase 5.

---

### ADR-003: HNSW + Timestamp Graph (Hybrid Indexing)

**Context:** HNSW es el estado del arte para ANN search pero asume conjuntos estáticos. Wang et al. (ICDE 2025) proponen Timestamp Graphs que explotan la localidad temporal.

**Decision:** Implementar ST-HNSW como un HNSW estándar aumentado con:
1. Roaring Bitmaps para filtrado temporal rápido (pre-filter).
2. Timestamp Graph overlay para queries de snapshot exacto (TANNS).
3. Time-decay edges para degradación gradual de conexiones antiguas.

**Alternatives Considered:**
- **HNSW puro + metadata filtering:** Lo que hace Qdrant. Funciona, pero no es nativo; cada query requiere cruzar filtro con grafo.
- **DiskANN temporal partitioned:** Un índice Vamana por partición temporal. Bueno para cold data, pero excesivo para hot data en RAM.
- **Timestamp Graph solo (sin HNSW):** El Timestamp Graph de Wang et al. resuelve TANNS pero no soporta nuestras queries de distancia compuesta.

**Consequences:**
- (+) Soporta snapshot kNN, range kNN, y distancia compuesta en un solo índice.
- (+) Roaring Bitmaps son O(1) para membership test, eficientes en memoria.
- (-) Complejidad de implementación: tres estructuras superpuestas.
- (-) El Timestamp Graph requiere manejo cuidadoso de expiración de nodos (backup neighbors).

---

### ADR-004: Delta-Embedding Storage with Keyframes

**Context:** Las entidades generan embeddings nuevos frecuentemente, pero entre dos timestamps consecutivos, la mayoría de las dimensiones cambian mínimamente.

**Decision:** Almacenar solo deltas $\Delta v = v(t_i) - v(t_{i-1})$ entre timestamps, con keyframes (vector completo) cada K updates. Los deltas se almacenan como sparse vectors (indices + values).

**Alternatives Considered:**
- **Full vector every time:** Simple, pero desperdicia ~90% de storage para las dimensiones que no cambiaron.
- **Full vector + compression (zstd):** Mejor que raw, pero sigue almacenando datos redundantes. No permite reconstrucción parcial.
- **Snapshot + WAL log:** Similar a databases. Pero los "logs" no tienen estructura sparse aprovechable.

**Consequences:**
- (+) Reducción de storage estimada: 5-10x para series densas (updates frecuentes con cambios pequeños).
- (+) Los deltas sparse se comprimen aún más con codificación run-length.
- (-) Lectura de un punto arbitrario requiere reconstrucción: leer keyframe + acumular deltas (max K reads).
- (-) Necesita keyframe interval tuning: K muy grande → reconstrucción lenta; K muy pequeño → poco ahorro.

**Mitigation:** K=10 como default. Benchmark para encontrar el sweet spot por workload.

---

### ADR-005: Tiered Storage (Hot/Warm/Cold)

**Context:** Los datos temporales tienen un patrón de acceso claro: los recientes se consultan frecuentemente, los históricos raramente.

**Decision:** Tres tiers con tecnologías diferentes:
- **Hot:** RocksDB (LSM-tree) con datos en RAM/SSD local. Vectores FP32 completos.
- **Warm:** Parquet files (arrow-rs). Datos de la última semana/mes. Formato columnar para analytics.
- **Cold:** Object Store (S3/MinIO) con Product Quantization. Datos históricos comprimidos.

**Alternatives Considered:**
- **Single store (RocksDB for everything):** Simple, pero RocksDB no es eficiente para analytics columnar ni para almacenamiento masivo barato.
- **Two tiers (Hot + Cold):** Viable, pero pierde la capacidad de analytics eficiente sobre datos medianamente recientes.
- **SQLite + extension:** Demasiado limitado para vectores de alta dimensión a escala.
- **Lance format (LanceDB):** Prometedor como evolución de Parquet para vectores, pero demasiado inmaduro para depender de él como único warm store. Podría reemplazar Parquet en el futuro.

**Consequences:**
- (+) Cada tier está optimizado para su patrón de acceso.
- (+) Cost-efficient: cold storage en object store es ~10x más barato que SSD.
- (-) Tres subsistemas de storage a mantener.
- (-) Compaction/migration entre tiers añade complejidad operativa.

---

### ADR-006: RocksDB as Hot Store Engine

**Context:** El hot store necesita writes de baja latencia, range scans por prefix (entity_id + timestamp), y column families para separar vectores/deltas/metadata.

**Decision:** Usar RocksDB vía `rust-rocksdb` bindings.

**Alternatives Considered:**
- **LMDB (via `lmdb-rkv`):** Excelente latencia de lectura, pero writes single-writer limitan throughput de ingesta.
- **Sled (pure Rust):** No production-ready para nuestro volumen. Benchmarks muestran 2-5x más lento que RocksDB.
- **Custom LSM-tree:** Control total, pero años de engineering. RocksDB es battle-tested.
- **FoundationDB:** Excelente para distributed, pero overhead innecesario para single-node hot store.

**Consequences:**
- (+) Column families permiten separar datos con diferentes compaction strategies.
- (+) Bloom filters por prefix reducen lecturas innecesarias.
- (+) Tunable write buffer, block cache, compaction.
- (-) FFI boundary (C++ → Rust) — no podemos usar unsafe-free bindings.
- (-) Write amplification inherente a LSM-trees.

---

### ADR-007: rkyv for Internal Serialization

**Context:** Los vectores y metadata se serializan/deserializan millones de veces por segundo en el hot path.

**Decision:** Usar `rkyv` (zero-copy deserialization) para datos internos. `serde` + `serde_json`/`prost` solo en la API boundary.

**Alternatives Considered:**
- **serde + bincode:** Rápido pero no zero-copy. Cada deserialización copia los datos.
- **FlatBuffers:** Zero-copy, pero la API es ergonómicamente pobre en Rust y requiere schema compilation.
- **Cap'n Proto:** Similar a FlatBuffers. Bueno, pero `rkyv` está más integrado en el ecosystem Rust.
- **Raw bytes (manual layout):** Máximo rendimiento, pero propenso a errores y no portable entre architectures.

**Consequences:**
- (+) Deserialización en O(0) — el buffer se usa directamente como struct.
- (+) `rkyv` soporta `#[derive(Archive)]` nativo en Rust.
- (-) Los datos rkyv no son human-readable; debugging requiere herramientas específicas.
- (-) Schema evolution más limitado que protobuf (no hay field numbers).

**Mitigation:** Versionado explícito en los primeros bytes de cada valor almacenado. Si la versión no coincide, se deserializa con el deserializer legacy y se re-serializa.

---

### ADR-008: Neural ODE for Trajectory Prediction

**Context:** Queremos predecir posiciones futuras de vectores basándose en su historial. Las alternativas son interpolación lineal, modelos autorregresivos, o Neural ODEs.

**Decision:** Implementar un solver Dormand-Prince (RK45) con $f_\theta$ como MLP ligero entrenado por entity o por cluster de entities. Usar `burn` como backend de tensores.

**Alternatives Considered:**
- **Linear extrapolation:** $v(t+\Delta t) = v(t) + \Delta t \cdot \frac{dv}{dt}$. Trivial de implementar pero poor para trayectorias no lineales.
- **ARIMA / exponential smoothing:** Clásicos para time series, pero operan dimension-by-dimension, ignorando la geometría del espacio vectorial.
- **Transformer-based:** Potente pero costoso en inferencia. Overkill para un servicio de baja latencia.
- **Neural CDE (Controlled Differential Equations):** Más expresivo que Neural ODE para entradas irregulares, pero más complejo de implementar. Deferred to Phase 6.

**Consequences:**
- (+) Neural ODEs modelan flujos continuos — natural para embeddings que evolucionan suavemente.
- (+) El solver adaptativo ajusta step size automáticamente (preciso donde la trayectoria curva, rápido donde es recta).
- (+) Memoria constante vía adjoint method para training.
- (-) Training requiere trayectorias históricas suficientes (cold start problem).
- (-) Predicción degrada rápidamente para horizontes largos (fundamentally chaotic systems).

---

### ADR-009: PELT + BOCPD Dual Change Point Detection

**Context:** Necesitamos detectar tanto cambios bruscos históricos (análisis offline) como cambios en tiempo real (monitoreo online).

**Decision:** Implementar ambos algoritmos:
- **PELT** (offline): Para análisis batch de trayectorias completas. Resultado exacto, O(N).
- **BOCPD** (online): Para monitoreo streaming durante ingesta. Per-entity state, O(1) amortizado.

**Alternatives Considered:**
- **Solo PELT:** Excelente offline, pero no puede operar en streaming.
- **Solo BOCPD:** Puede operar online, pero para análisis histórico es menos preciso que PELT.
- **CUSUM (Cumulative Sum):** Clásico y simple, pero solo detecta cambios en media, no en distribución completa.
- **Deep learning CPD:** Redes neuronales para CPD (e.g., Transformers). Potente pero latencia inaceptable para online monitoring per-entity.

**Consequences:**
- (+) Cobertura completa: offline preciso + online rápido.
- (+) BOCPD per-entity permite miles de streams simultáneos con poco overhead.
- (-) Dos implementaciones a mantener.
- (-) BOCPD requiere tuning del hazard function y prior por dominio.

---

### ADR-010: Workspace Multi-Crate Architecture

**Context:** El sistema tiene 6 subsistemas con dependencias claras. Necesitamos compilation units independientes para builds incrementales rápidos.

**Decision:** Cargo workspace con 8 crates:
```
cvx-core → cvx-index, cvx-storage, cvx-ingest, cvx-analytics
cvx-query → cvx-index, cvx-storage, cvx-analytics
cvx-api → cvx-query, cvx-ingest
cvx-server → cvx-api
```

**Alternatives Considered:**
- **Monolithic crate:** Compile times inmanejables a medida que crece. No permite feature-gate subsistemas.
- **Many small crates (one per module):** Over-engineering. Overhead de boilerplate y versioning excesivo para un solo desarrollador.
- **Dynamic linking (dylib):** Permitiría hot-reload, pero añade complejidad de deployment y pierde LTO optimizations.

**Consequences:**
- (+) Cada crate compila independientemente; cambios en `cvx-analytics` no recompilan `cvx-index`.
- (+) Feature flags por crate permiten builds mínimos (e.g., server sin analytics para testing).
- (+) Tests por crate son rápidos y focalizados.
- (-) Dependency management entre crates requiere disciplina.
- (-) Circular dependencies entre crates son compile errors, no warnings — requiere diseño upfront.

---

### ADR-011: gRPC for Streaming, REST for Request-Response

**Context:** La ingesta necesita bidirectional streaming de alto throughput. Las queries son request-response con posible streaming de resultados.

**Decision:** Dual API:
- `tonic` (gRPC) para ingesta streaming y WatchDrift subscriptions.
- `axum` (REST) para queries, admin operations, y health checks.

**Alternatives Considered:**
- **gRPC only:** Eficiente pero pobre developer experience para queries ad-hoc (no hay curl/Postman equivalente simple).
- **REST only:** Familiar pero no soporta bidirectional streaming nativamente. WebSockets como workaround añade complejidad.
- **GraphQL:** Expresivo para queries complejas, pero overhead de parsing innecesario para nuestro modelo de query tipado.

**Consequences:**
- (+) Mejor herramienta para cada job: gRPC para throughput, REST para usabilidad.
- (+) Protobuf como schema compartido entre ambos (gRPC nativo, REST vía serde mapping).
- (-) Dos servers escuchando en puertos diferentes.
- (-) Duplicación parcial de request/response types entre proto y REST models.

---

### ADR-008: Interpretability as a Separate Crate (`cvx-explain`)

**Context:** CVX produce datos analíticos ricos (drift reports, change points, velocidades, predicciones) pero son vectores crudos de alta dimensionalidad. Los usuarios necesitan artefactos interpretables: qué dimensiones cambiaron, proyecciones 2D de trayectorias, timelines anotadas.

**Decision:** Crear `cvx-explain` como crate separado que consume outputs de `cvx-analytics` y `cvx-query` y los transforma en artefactos interpretables (drift attribution, trajectory projection, heatmaps, narratives). El principio es "datos para interpretar, no gráficos" — CVX produce JSON estructurado, la renderización es del consumidor.

**Alternatives Considered:**
- **Integrar en `cvx-analytics`:** Menor overhead de crates, pero mezcla cómputo y presentación. Viola separation of concerns.
- **Dashboard integrado (web UI):** Más completo para el usuario, pero añade dependencias frontend, aumenta superficie de mantenimiento, y limita flexibilidad (Grafana, Jupyter, custom UI son mejores opciones para renderización).
- **Solo exportar a formatos estándar (CSV/Parquet):** Simple, pero pierde la semántica de los artefactos. El consumidor tendría que re-interpretar campos.

**Consequences:**
- (+) Separation of concerns: analytics computa, explain interpreta.
- (+) Cualquier frontend puede consumir los endpoints (Grafana, Jupyter, React).
- (+) Testable independientemente: los artefactos tienen schemas bien definidos.
- (-) Un crate adicional en el workspace.
- (-) Latencia añadida para transformaciones (mitigable: <5ms para la mayoría de operaciones).

**Full Spec:** `CVX_Explain_Interpretability_Spec.md`

---

### ADR-009: Multi-Space Embedding Support

**Context:** En producción, las entidades tienen múltiples representaciones (texto D=768, imagen D=512, usuario D=128) que evolucionan a diferentes escalas temporales. CVX actualmente asume un solo espacio vectorial por entidad.

**Decision:** Extender el data model con `EmbeddingSpace` como concepto first-class. La tupla fundamental pasa de `(entity_id, timestamp, vector)` a `(entity_id, space_id, timestamp, vector)`. Cada espacio tiene su propio ST-HNSW index. Se proporcionan métodos de alignment (Structural, Behavioral, Procrustes, CCA) para medir coherencia cross-space.

**Alternatives Considered:**
- **Namespaces por colección (como Qdrant named vectors):** Más simple pero no modela la relación entre espacios. No permite cross-space queries.
- **Concatenar todos los embeddings en un super-vector:** Pierde la semántica de cada espacio. Dimensionalidades diferentes impiden concatenación directa.
- **Espacios separados sin relación:** Funcional, pero desaprovecha la oportunidad única de CVX de analizar evolución cross-modal.

**Consequences:**
- (+) Habilita cross-modal drift analysis — capacidad única de CVX.
- (+) Backward compatible: sin space_id se usa default space (space_id=0).
- (+) Cada espacio se indexa independientemente — no degrada rendimiento de single-space.
- (-) Storage key encoding se extiende en 4 bytes por entry.
- (-) O(S) índices ST-HNSW donde S = número de espacios (típicamente 2-10).
- (-) Alignment methods requieren álgebra lineal (CCA, Procrustes) — nuevas dependencias.

**Full Spec:** `CVX_MultiScale_Alignment_Spec.md`

---

### ADR-010: Multi-Scale Temporal Analysis

**Context:** Diferentes fuentes de datos actualizan embeddings a frecuencias distintas (real-time, horario, diario, semanal). El análisis de drift y change points puede variar significativamente según la escala temporal elegida. Señales ruidosas a escala fina pueden enmascarar tendencias reales, y viceversa.

**Decision:** Implementar resampling temporal (LastValue, Linear, Slerp, NeuralODE) y análisis multi-escala que ejecuta drift analysis y CPD a múltiples escalas simultáneamente. Los change points que persisten a través de escalas se clasifican como "robustos" (high-confidence).

**Alternatives Considered:**
- **Dejar al usuario elegir la escala manualmente:** Simple, pero el usuario no sabe cuál es la escala óptima a priori. Probablemente ejecutaría múltiples queries y haría el cruce manualmente.
- **Escala adaptativa automática:** Elegiría la "mejor" escala automáticamente, pero pierde la riqueza del análisis multi-escala. No hay una única "mejor" escala.

**Consequences:**
- (+) Reducción de falsos positivos: change points deben persistir a múltiples escalas.
- (+) Optimal analysis scale detection: el sistema puede recomendar la escala con mejor SNR.
- (+) Habilita comparación cross-space con diferentes frecuencias de actualización.
- (-) Cómputo multiplicativo: O(S) × coste de análisis single-scale, donde S = número de escalas.
- (-) Slerp interpolation asume esfera unitaria — no universal para todos los espacios.

**Full Spec:** `CVX_MultiScale_Alignment_Spec.md` §4

---

### ADR-011: Dual Backend for Temporal Features (Analytic + Differentiable)

**Context:** CVX extrae features temporales (velocity, drift, change points) de trayectorias de embeddings. Para análisis e interpretación, estas features no necesitan ser diferenciables. Pero para training end-to-end (e.g., fine-tuning BERT para detección de trastornos desde redes sociales), los gradientes deben fluir desde el loss del clasificador a través de las features temporales hasta el modelo de embeddings.

**Decision:** Implementar un trait `TemporalOps` con tres backends:
1. `AnalyticBackend`: Rust puro + SIMD, sin autograd. Para serving, API, explain.
2. `BurnBackend`: burn tensors con autograd + CUDA. Para training 100% Rust.
3. `TorchBackend`: tch-rs (libtorch bindings). Para interop con PyTorch — los tensores comparten memoria y autograd graph con Python, permitiendo backpropagation cross-language.

La misma lógica matemática se escribe una vez en el trait; cada backend la ejecuta en su contexto de tensores.

**Alternatives Considered:**
- **Solo burn (sin tch-rs):** Obliga a los usuarios Python a convertir tensores burn ↔ PyTorch, rompiendo el autograd graph. No viable para fine-tuning desde Python.
- **Solo Python (PyTorch puro):** Duplica la lógica de features temporales fuera de CVX. Pierde las optimizaciones SIMD de Rust. Mezcla lenguajes en el codebase.
- **CVX no diferenciable + cvx-torch separado en Python:** Funcional, pero el usuario tiene dos implementaciones divergentes de las mismas features. Mantenimiento doble.

**Consequences:**
- (+) Una sola lógica matemática, tres contextos de ejecución.
- (+) Usuarios Rust puros: burn con CUDA, sin dependencias Python.
- (+) Usuarios Python: tch-rs preserva gradientes transparentemente.
- (+) El camino analítico (serving) no paga el overhead de autograd.
- (-) tch-rs añade dependencia en libtorch (~2GB) para quienes activen el feature flag.
- (-) Tres backends que testear y mantener sincronizados.

**Full Spec:** `CVX_Temporal_ML_Spec.md`

---

### ADR-012: Soft Relaxations for Non-Differentiable Features

**Context:** PELT, BOCPD, top-K dimension selection y conteos son operaciones discretas no diferenciables. Para que las features que dependen de ellas participen en backpropagation, se necesitan aproximaciones continuas.

**Decision:** Usar relaxaciones sigmoid/softmax con temperatura configurable (τ):
- Soft change point count: `Σ σ((deviation - μ) / τ)` ≈ número de change points.
- Soft top-K: `gumbel_softmax(|delta|, τ)` ≈ indicador de dimensiones top.
- Soft counting: `Σ σ((gap - θ) / τ)` ≈ conteo de silencios.

La temperatura τ puede ser un parámetro learnable que el modelo optimiza durante training.

**Consequences:**
- (+) Features que capturan la misma información que PELT/BOCPD pero son diferenciables.
- (+) τ → 0 converge a la versión discreta (validable contra PELT analítico).
- (+) τ learnable permite que el modelo encuentre la "sensibilidad" óptima.
- (-) Las relaxaciones son aproximaciones — no idénticas a PELT/BOCPD.
- (-) Gradientes pueden ser ruidosos con τ muy pequeño (sigmoid saturado).

**Validation:** Tras training, comparar soft_cp_count con PELT count analítico. Spearman > 0.8 indica que la relaxación captura la misma señal.

---

### ADR-013: Selective Data Virtualization (Ingestion, Not Federation)

**Context:** In production ML, embeddings are scattered across systems (S3, Kafka, model serving APIs, other vector DBs). Full data virtualization (query-time federation like Denodo) would let CVX query remote sources directly. However, CVX's temporal analytics (velocity, change points, trajectories) require the full history of each entity — latency from remote fetches would be prohibitive.

**Decision:** Adopt selective data virtualization concepts:
1. **Source connectors** for declarative ingestion (not query-time federation)
2. **Model version alignment** to handle retraining discontinuities
3. **Temporal materialized views** for caching repeated analytics
4. **Provenance/lineage** tracking per embedding
5. **Monitors** for declarative alerting on temporal patterns

Reject full query federation, distributed query optimizer, SQL interface, and row/column security.

**Alternatives Considered:**
- **Full data virtualization (Denodo-style):** Query remote sources at query time. Rejected because temporal analytics need complete history — can't compute velocity from a single remote snapshot.
- **Pure ETL (no CVX involvement):** Leave ingestion entirely to user scripts. Works but creates friction and loses provenance.
- **Embedding-specific lakehouse (like LanceDB):** Merges storage and compute but doesn't address temporal analytics or model versioning.

**Consequences:**
- (+) Reduces ingestion friction dramatically (declare sources, CVX syncs).
- (+) Model version alignment is a unique differentiator — no other VDB does this.
- (+) Materialized views eliminate redundant computation for iterative research.
- (+) Provenance enables "is this drift real or a model artifact?" analysis.
- (-) Source connectors add maintenance burden (one per source type).
- (-) Model alignment quality depends on overlap data between versions.

**Full Spec:** `CVX_Data_Virtualization_Spec.md`

---

### ADR-014: Stochastic Analytics Layer

**Context:** CVX's trajectory analytics (velocity, acceleration, drift) are deterministic measures. The quantitative finance and stochastic processes literature provides richer tools: drift significance testing, volatility modeling (GARCH), mean reversion analysis (Ornstein-Uhlenbeck), path signatures (rough path theory), regime detection (HMM), and Neural SDEs. These tools characterize not just how embeddings move, but the statistical nature of their movement.

**Decision:** Extend `cvx-analytics` with a `stochastic/` module implementing per-entity process characterization (drift tests, GARCH, mean reversion, Hurst exponent), path signatures for trajectory descriptors, regime detection via HMM, and Neural SDE extension of the existing Neural ODE. Cross-entity analysis includes DCC, co-integration, and Granger causality.

**Alternatives Considered:**
- **External analytics only (export to R/Python):** CVX exports trajectories, user runs statsmodels/R. Viable but loses the value of integrated analysis and real-time computation.
- **Only neural approaches (Neural SDE, no classical stats):** Misses the diagnostic value of simple statistical tests (ADF, GARCH) that are interpretable and well-understood.

**Consequences:**
- (+) CVX becomes a temporal analytics platform, not just a temporal database.
- (+) Path signatures enable a new query type: trajectory similarity search.
- (+) Neural SDE adds uncertainty quantification to predictions.
- (+) Stochastic characterization feeds directly into the interpretability layer (cvx-explain).
- (-) Significant implementation effort across multiple mathematical domains.
- (-) Some methods (GARCH, ADF) require numerical optimization — need robust implementations.

**Full Spec:** `CVX_Stochastic_Analytics_Spec.md`

---

### ADR-015: Domain-Agnostic Core, Domain-Specific Analytics as Composition

**Context:** CVX incorpora conceptos de múltiples dominios — NLP (drift semántico, embeddings diacrónicos), finanzas cuantitativas (GARCH, mean reversion, path signatures, flow imbalance à la López de Prado), y clinical NLP (detección temprana desde redes sociales). Existe el riesgo de que el core de CVX se "contamine" con asunciones de un dominio específico, limitando su generalidad.

**Decision:** Separación estricta en tres capas:

1. **Core** (`cvx-core`, `cvx-index`, `cvx-storage`): completamente agnóstico al dominio. Solo sabe de `(entity_id, space_id, timestamp, vector)`. No conoce conceptos como "volatilidad financiera", "drift semántico" ni "publicaciones en redes sociales".

2. **Primitivas analíticas** (`cvx-analytics`): operaciones genéricas sobre trayectorias — velocity, drift, change points, stationarity tests, path signatures, regime detection. Los nombres y APIs son genéricos (e.g., "neighborhood drift coherence", no "order flow imbalance").

3. **Composición de dominio**: los conceptos de dominio específico son **combinaciones de primitivas**, no features del core. Ejemplos:
   - "Flow imbalance" (finanzas) = kNN + velocity + coherencia direccional
   - "Adaptive delta threshold" = ingestion policy configurable, no hardcoded
   - "Fractional differentiation" = función analítica sobre trajectory, no transformación del storage
   - "Meta-labeling" = regime confidence × prediction, no un módulo separado

**Alternatives Considered:**
- **Vertical financiero primero:** Optimizar CVX para quant finance y luego generalizar. Riesgo: las asunciones financieras (GBM, log-returns) se embeben en el core y son difíciles de extraer después.
- **Plugins por dominio:** Crates separados (`cvx-finance`, `cvx-nlp`, `cvx-clinical`). Overhead de mantenimiento alto, y la mayoría de las "features de dominio" son simplemente composiciones de 2-3 primitivas — no justifican un crate entero.

**Consequences:**
- (+) CVX es general: cualquier dato temporal vectorial, de cualquier dominio.
- (+) Las primitivas son reutilizables: un test de mean reversion sirve para finanzas, NLP y clinical.
- (+) Nuevos dominios no requieren cambios en el core — solo nuevas composiciones.
- (+) El naming genérico evita alienar a usuarios de otros campos.
- (-) Los usuarios de un dominio específico deben componer las primitivas ellos mismos (mitigado por documentación de use cases y ejemplos).
- (-) Algunas optimizaciones domain-specific (e.g., log-returns para datos financieros) no se implementan "out of the box".

**Principio rector:** *"CVX no sabe de finanzas, NLP, ni medicina. Sabe de trayectorias vectoriales en el tiempo. Los dominios son configuraciones y composiciones, no features."*

---

## 3.5 Implementation Decisions

The following implementation-level decisions are documented in `CVX_Implementation_Decisions.md`:

| IDR | Decision | Choice |
|-----|---------|--------|
| IDR-001 | Concurrency model | Message passing (channels) + RwLock (index) |
| IDR-002 | Compute parallelism | Rayon thread pool with work-stealing |
| IDR-003 | Serialization | rkyv (HNSW graph) + postcard (storage/WAL) |
| IDR-004 | Global allocator | jemalloc (per-thread arenas) |
| IDR-005 | SIMD strategy | pulp (safe, stable, runtime dispatch) |
| IDR-006 | RocksDB key encoding | BE + sign-bit flip, separate CFs |
| IDR-007 | Error handling | thiserror (libs) + anyhow (binary) |
| IDR-008 | Testing strategy | proptest + criterion + cargo-fuzz |
| IDR-009 | Unsafe policy | deny by default, allow in 2 crates |
| IDR-010 | Index persistence | Read-into-memory → mmap + prefetch |

These are implementation-level decisions (HOW), complementing the architectural decisions above (WHAT/WHY).

---

## 4. Decisions Deferred

| Decision | Deferred To | Reason |
|---|---|---|
| Hyperbolic metric as default | Phase 5 | Need empirical evidence on real workloads |
| Neural CDE vs Neural ODE | Phase 6 | ODE is simpler; CDE only if ODE proves insufficient |
| Learned distance combination (replacing linear α) | Phase 5 | Linear is baseline; optimize later |
| Sharding strategy (hash vs range) | Phase 5 (distributed) | Single-node first |
| Lance format replacing Parquet | Continuous evaluation | Depends on Lance crate maturity |
| UMAP implementation (custom vs linfa-reduction) | Layer 7.5 | PCA first; UMAP if demand warrants the dependency |
| nalgebra vs ndarray for CCA/Procrustes | Layer 7.5 implementation | Both viable; evaluate ergonomics during implementation |
| Cross-modal prediction viability | Post Layer 10 | Requires trained Neural ODE + alignment data; test Granger causality first |
| Natural language narratives via LLM | Post Layer 12 | Explore generating human-readable explanations from structured explain data |
| Projection cache strategy for UMAP | Layer 7.5 implementation | LRU cache by (entity_id, time_range, method) — evaluate if hit rate justifies memory |
| burn vs candle for BERT encoder loading | Layer 10.5 | candle has better HuggingFace integration; burn has better training ergonomics. May coexist. |
| Learnable temperature schedule for soft CPD | Layer 10.5 | Fixed τ, learnable τ, or annealing schedule during training? Empirical evaluation needed. |
| Source connector priority (S3 vs Kafka vs pgvector first) | Layer 9 | Depends on user demand; S3-Parquet likely first (most common in data lakes) |
| Alignment quality threshold for canonical trajectories | Layer 7.5+ | Need empirical data on Procrustes residuals across real model retrains |

---

## 5. How to Propose Changes

1. Copy the ADR template below.
2. Add it to this RFC with the next sequential number.
3. Set status to `Proposed`.
4. After review and implementation, change status to `Accepted` and date.

```markdown
### ADR-NNN: [Title]

**Context:** [What is the problem?]
**Decision:** [What did we decide?]
**Alternatives Considered:** [What else was evaluated?]
**Consequences:** [What are the trade-offs?]
```
