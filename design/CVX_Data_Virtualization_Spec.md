# ChronosVector — Data Virtualization Layer Specification

**Version:** 1.0
**Author:** Manuel Couto Pintos
**Date:** March 2026
**Status:** Draft
**Dependencies:** Architecture Doc §8 (Tiered Storage), §11 (API Gateway), MultiScale Alignment Spec (multi-space), Temporal ML Spec

---

## 1. Motivation

En producción, los embeddings de un sistema ML no viven en un solo lugar. Están **dispersos** a través de múltiples sistemas:

- **Model serving APIs** (Triton Inference Server, TorchServe) — generan embeddings en tiempo real.
- **Data lakes** (S3/Parquet) — archivan embeddings históricos en batch.
- **Streaming platforms** (Kafka) — transportan embeddings como eventos.
- **PostgreSQL + pgvector** — almacenan embeddings con metadata relacional.
- **Otros VDBs** (Qdrant, Pinecone) — almacenan snapshots sin contexto temporal.

CVX actualmente requiere ingesta manual vía `POST /v1/ingest` — cada embedding debe ser empujado explícitamente. Esto genera **fricción significativa**: pipelines de ETL ad-hoc, scripts de sincronización frágiles, y — lo más problemático — **pérdida de contexto**. ¿De qué modelo vino este embedding? ¿De qué versión? ¿Cuándo fue producido vs. cuándo fue ingerido?

### 1.1 Problemas Concretos

| Problema | Impacto |
|----------|---------|
| **Ingesta manual** | Fricción en onboarding, scripts de sync frágiles |
| **Retraining de modelos** | BERT v1 → v2 crea discontinuidades artificiales en trayectorias. Change point detection reporta falsos positivos. |
| **Analytics repetidos** | Mismo investigador computa drift, velocity, change points sobre las mismas trayectorias múltiples veces durante iteración |
| **Sin provenance** | No se puede responder: "¿qué modelo produjo el embedding que triggeó este change point?" |
| **Sin alerting** | No hay forma declarativa de decir "avísame cuando el drift de esta entidad exceda X" |

### 1.2 Principio de Diseño: Adoptar, No Replicar

CVX **NO** es una plataforma de data virtualization. No competimos con Denodo, Dremio, o Trino. Adoptamos **conceptos específicos** del mundo de data virtualization que resuelven problemas reales para temporal embedding analytics:

| Concepto DV | Adaptación CVX | Valor |
|-------------|----------------|-------|
| Source connectors | Declarative ingestion de embedding sources | Eliminar scripts de sync ad-hoc |
| Semantic layer | Model version alignment + canonical spaces | Resolver discontinuidades de retrain |
| Materialized views | Temporal feature cache con invalidation | Eliminar recomputation durante iteración |
| Data lineage | Embedding provenance metadata | Root cause analysis de anomalías |
| Data quality monitors | Declarative alerting sobre métricas temporales | Proactive monitoring sin código |

---

## 2. Source Connectors (Declarative Ingestion)

### 2.1 Design Decision: Ingestion, NOT Federation

Los sistemas de data virtualization tradicionais (Denodo, Dremio) ejecutan queries **en tiempo de consulta** contra fuentes remotas (query-time federation). Esto **no funciona** para temporal analytics por una razón fundamental: el análisis temporal requiere **la historia completa**. Calcular drift, change points, o velocity requiere acceso a toda la trayectoria — no puedes computar PELT sobre una ventana parcial obtenida en query-time.

Por lo tanto, los source connectors de CVX son **mecanismos de ingesta declarativa**: sincronizan datos de fuentes externas **hacia CVX**, donde se almacenan y se indexan normalmente. La diferencia con la ingesta manual es que la sincronización es **declarativa, incremental, y monitoreable**.

### 2.2 EmbeddingSource Trait

```rust
/// Base trait for all embedding source connectors.
/// Each source knows how to fetch temporal embeddings from an external system.
pub trait EmbeddingSource: Send + Sync {
    /// Human-readable name for this source (used in provenance).
    fn name(&self) -> &str;

    /// Target CVX space for embeddings from this source.
    fn space(&self) -> &str;

    /// Poll the source for new embeddings since the given timestamp.
    /// If `since` is None, fetch all available data (initial sync).
    fn poll(&self, since: Option<i64>) -> Result<Vec<TemporalPoint>>;

    /// Whether this source supports incremental sync (delta fetching).
    /// If false, each poll() returns the full dataset.
    fn supports_incremental(&self) -> bool;
}
```

### 2.3 Source Types

| Source Type | Protocol | Use Case | Push-down Filtering |
|-------------|----------|----------|---------------------|
| **S3/Parquet** | AWS SDK + Arrow | Batch-produced embeddings archivados en data lake | Partition pruning (date partitions), Parquet predicate pushdown |
| **Kafka** | Consumer group | Real-time embedding streams | Topic + partition filtering |
| **REST API** | HTTP polling | Model serving APIs (Triton, TorchServe) | Query params (entity_ids, since_timestamp) |
| **gRPC stream** | gRPC client | High-throughput streaming sources | Stream metadata filtering |
| **PostgreSQL + pgvector** | SQL client | Embeddings almacenados con metadata relacional | SQL WHERE clause pushdown |

### 2.4 Sync Modes

- **Incremental sync:** solo fetch deltas desde `last_sync` timestamp. Requiere que la fuente soporte consultas por rango temporal. La mayoría de fuentes lo soportan (Parquet por partición de fecha, SQL por `WHERE updated_at > $last_sync`, Kafka por offset).
- **Push-down filtering:** filtrar en la fuente (entity_ids, time range) para evitar transferir todo. Reduce I/O y latencia de sincronización.
- **Schedule-based polling:** cron expressions para polling periódico (`"0 */6 * * *"` = cada 6 horas).
- **Event-driven:** Kafka consumer groups para ingesta en tiempo real.

### 2.5 Configuration

Las fuentes se definen declarativamente en `config.toml`:

```toml
[[sources]]
name = "daily-user-embeddings"
type = "s3_parquet"
space = "bert-v2"
schedule = "0 2 * * *"   # daily at 2am
incremental = true

[sources.s3_parquet]
bucket = "ml-embeddings"
prefix = "user-embeddings/bert-v2/"
partition_format = "dt=%Y-%m-%d"
entity_id_column = "user_id"
vector_column = "embedding"
timestamp_column = "produced_at"
region = "us-east-1"

[[sources]]
name = "realtime-content-embeddings"
type = "kafka"
space = "content-clip-512"

[sources.kafka]
brokers = ["kafka-1:9092", "kafka-2:9092"]
topic = "content-embeddings"
group_id = "cvx-ingest"
entity_id_field = "content_id"
vector_field = "embedding"
timestamp_field = "created_at"

[[sources]]
name = "triton-user-encoder"
type = "rest_api"
space = "user-encoder-v3"
schedule = "*/30 * * * *"   # every 30 minutes
incremental = true

[sources.rest_api]
url = "https://ml-serving.internal/v2/models/user-encoder/infer"
method = "POST"
entity_ids_param = "user_ids"
batch_size = 1000
auth = { type = "bearer", token_env = "TRITON_API_TOKEN" }
```

### 2.6 Module Structure

```
crates/cvx-ingest/src/sources/
    ├── mod.rs            // EmbeddingSource trait, SourceManager
    ├── s3_parquet.rs     // S3ParquetSource
    ├── kafka.rs          // KafkaSource
    ├── rest_api.rs       // RestApiSource
    ├── grpc_stream.rs    // GrpcStreamSource
    └── pgvector.rs       // PgVectorSource
```

### 2.7 Feature Flags

```toml
[features]
source-s3 = ["aws-sdk-s3", "arrow", "parquet"]
source-kafka = ["rdkafka"]
source-rest = ["reqwest"]
source-grpc = ["tonic"]
source-pgvector = ["sqlx", "tokio-postgres"]
```

### 2.8 Error Handling & Observability

- **Retry with backoff:** errores transientes (network, rate limiting) se reintentan con exponential backoff (1s, 2s, 4s, ..., max 5 min).
- **Dead letter:** registros que fallan parsing/validación se envían a un dead letter log para inspección manual.
- **Métricas expuestas:**

| Metric | Type | Description |
|--------|------|-------------|
| `cvx_source_sync_lag_seconds` | Gauge | Tiempo desde la última sincronización exitosa |
| `cvx_source_records_synced_total` | Counter | Total de registros sincronizados por fuente |
| `cvx_source_sync_errors_total` | Counter | Errores de sincronización por fuente y tipo |
| `cvx_source_sync_duration_seconds` | Histogram | Duración de cada ciclo de sincronización |

---

## 3. Model Version Alignment

Este es el concepto **más valioso** que adoptamos de data virtualization: la idea de que diferentes "vistas" de los mismos datos (en nuestro caso, diferentes versiones de un modelo) pueden y deben ser alineadas en un espacio semántico unificado.

### 3.1 El Problema

Cuando un equipo de ML retrain su modelo de embeddings (BERT v1 → BERT v2), los nuevos embeddings viven en un **espacio semántico diferente** aunque tengan la misma dimensionalidad. Los vectores no son directamente comparables:

- La trayectoria de un usuario muestra un **salto artificial** en la frontera v1 → v2.
- Change point detection (PELT, BOCPD) reporta un **falso positivo** en el momento del retrain.
- Drift analytics se vuelven **inútiles** a través de fronteras de modelo.
- El ML engineer tiene que mantener **pipelines separados** para cada versión.

Esto es exactamente el problema que la capa semántica de un sistema de data virtualization resuelve: proveer una **vista unificada** sobre datos heterogéneos.

### 3.2 ModelVersion Type

```rust
/// Represents a specific version of an embedding model registered in CVX.
/// Each model version maps to a CVX space (space_id).
pub struct ModelVersion {
    /// The CVX space_id that holds embeddings from this model version.
    pub space_id: u32,
    /// Model family name (e.g., "bert-base", "user-encoder").
    pub model_name: String,
    /// Version identifier (e.g., "v2", "2024-03-01", "abc123").
    pub version: String,
    /// When this model version was registered.
    pub created_at: i64,
    /// The space_id of the predecessor version (if any).
    pub predecessor: Option<u32>,
    /// Computed alignment transform from predecessor space to this space.
    pub alignment_transform: Option<AlignmentTransform>,
}

/// A transform that projects embeddings from one model version's space
/// to another, enabling cross-version trajectory continuity.
pub enum AlignmentTransform {
    /// Orthogonal Procrustes: rotation matrix R + scale factor s.
    /// Projects v_old → s * R * v_old ≈ v_new.
    Procrustes(Matrix),
    /// Canonical Correlation Analysis: projects both spaces to a common subspace.
    Cca(CcaProjection),
    /// Learned alignment model (neural network or similar).
    /// Path to serialized model artifact.
    Learned(ModelPath),
}
```

### 3.3 Automatic Alignment Computation

Cuando se registra una nueva versión de modelo con un `predecessor`, CVX ejecuta automáticamente el siguiente protocolo:

1. **Identify overlap entities:** Encontrar entidades que tienen embeddings en AMBOS espacios (old y new). Esto ocurre naturalmente durante el período de transición donde ambas versiones del modelo están produciendo embeddings.
2. **Compute Procrustes alignment:** Usando los pares de embeddings de overlap entities, calcular la transformación óptima (rotación + escala) que minimiza `||V_old_projected - V_new||²_F`.
3. **Store transform:** Guardar la transformación como parte del `ModelVersion`.
4. **Create canonical space (optional):** Generar un espacio virtual "canónico" que unifica todas las versiones de un modelo, proyectando cada embedding a un marco de referencia común.

**Requisito mínimo de overlap:** al menos 100 entidades con embeddings en ambos espacios. Si hay menos, CVX emite un warning y marca la alineación como `low_confidence`.

### 3.4 Canonical Trajectory

La abstracción central: una **vista virtual** que proyecta todos los embeddings de una entidad — independientemente de qué versión de modelo los produjo — a un espacio canónico común.

```rust
/// A virtual trajectory that unifies embeddings across model versions
/// by projecting each point to a canonical space.
pub struct CanonicalTrajectory {
    /// The entity whose trajectory we're viewing.
    pub entity_id: u64,
    /// The canonical space name (e.g., "bert-canonical").
    pub canonical_space: String,
    /// Points projected to the canonical space, in chronological order.
    pub points: Vec<CanonicalPoint>,
}

/// A single point in a canonical trajectory.
pub struct CanonicalPoint {
    /// Original timestamp.
    pub timestamp: i64,
    /// Vector projected to the canonical space.
    pub vector: Vec<f32>,
    /// Which model version (space_id) originally produced this embedding.
    pub original_space_id: u32,
    /// How much the alignment transform shifted this point (L2 distance
    /// between original and projected vector). High residuals indicate
    /// poor alignment quality for this specific point.
    pub alignment_residual: f64,
}
```

### 3.5 Change Point Enhancement: Model Artifact Detection

Con model version alignment, podemos mejorar significativamente la detección de change points. Un change point que coincide con una frontera de versión de modelo Y tiene un alignment residual alto es probablemente un **artefacto del retrain**, no un cambio semántico real.

| Señal | Clasificación |
|-------|---------------|
| Change point en frontera de modelo + alignment residual alto | `ModelArtifact` — falso positivo por retrain |
| Change point en frontera de modelo + alignment residual bajo | `RealChange` — coincidencia con retrain, pero cambio real |
| Change point lejos de frontera de modelo | `RealChange` — cambio semántico genuino |

```rust
pub enum ChangePointClassification {
    /// Genuine semantic change in the entity's trajectory.
    RealChange,
    /// Artifact of model retraining — the change point disappears
    /// when viewing the canonical trajectory.
    ModelArtifact {
        model_boundary: i64,
        alignment_residual: f64,
    },
}
```

### 3.6 API Endpoints

#### `POST /v1/models/versions`

Registrar una nueva versión de modelo con su predecesor.

**Request:**
```json
{
  "model_name": "user-bert",
  "version": "v2",
  "space_id": 3,
  "predecessor_space_id": 1,
  "auto_align": true
}
```

**Response:**
```json
{
  "model_name": "user-bert",
  "version": "v2",
  "space_id": 3,
  "predecessor_space_id": 1,
  "alignment_status": "computing",
  "overlap_entities": 4521
}
```

#### `POST /v1/models/versions/{id}/align`

Trigger manual de alignment computation (o re-computation con más overlap data).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `method` | `string` | No | `"procrustes"` | `"procrustes"`, `"cca"`, `"learned"` |
| `min_overlap` | `usize` | No | `100` | Minimum overlap entities required |

#### `GET /v1/entities/{id}/canonical-trajectory`

Obtener la trayectoria canónica unificada de una entidad.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_name` | `string` | Yes | — | Model family name |
| `from` | `i64` | No | first_seen | Start timestamp |
| `to` | `i64` | No | last_seen | End timestamp |

**Response:**
```json
{
  "entity_id": 42,
  "canonical_space": "user-bert-canonical",
  "points": [
    {
      "timestamp": 1640000000,
      "vector": [0.12, -0.34, ...],
      "original_space_id": 1,
      "alignment_residual": 0.0
    },
    {
      "timestamp": 1650000000,
      "vector": [0.15, -0.31, ...],
      "original_space_id": 3,
      "alignment_residual": 0.023
    }
  ]
}
```

#### `GET /v1/models/versions/{id}/alignment-quality`

Estadísticas de calidad de la alineación.

**Response:**
```json
{
  "model_name": "user-bert",
  "from_version": "v1",
  "to_version": "v2",
  "overlap_entities": 4521,
  "mean_residual": 0.031,
  "p95_residual": 0.087,
  "p99_residual": 0.142,
  "alignment_method": "procrustes",
  "confidence": "high"
}
```

### 3.7 NFR

| Requirement | Target |
|-------------|--------|
| Alignment computation (Procrustes, 10K overlap entities, D=768) | < 30s |
| Canonical projection per point | < 0.1ms |
| Alignment quality query | < 5ms |
| Canonical trajectory retrieval (1K points) | < 50ms |

---

## 4. Temporal Materialized Views

### 4.1 El Problema

Computar temporal features (velocity, drift, change points) sobre trayectorias completas es **caro**. Un researcher iterando sobre un modelo de clasificación de riesgo (el caso de uso de social media del Temporal ML Spec) necesita:

1. Extraer features temporales de 100K usuarios.
2. Entrenar un clasificador.
3. Evaluar, ajustar hiperparámetros, repetir.

En cada iteración, las mismas trayectorias se recomputan. Con PELT (O(n²)), esto es prohibitivamente lento. El researcher termina exportando features a un CSV y trabajando offline — perdiendo la capacidad de CVX de actualizar features cuando llegan nuevos datos.

### 4.2 MaterializedView Definition

```rust
/// A pre-computed, cached temporal aggregation with automatic invalidation.
pub struct MaterializedView {
    /// Unique name for this view (e.g., "user_drift_daily").
    pub name: String,
    /// The temporal query that defines what this view computes.
    pub query: TemporalQuery,
    /// When to refresh the view.
    pub refresh_policy: RefreshPolicy,
    /// How long to keep computed results before eviction.
    pub retention: Duration,
    /// Timestamp of last successful computation.
    pub last_computed: Option<i64>,
    /// Current status of the view.
    pub status: ViewStatus,
}

pub enum RefreshPolicy {
    /// Recompute immediately when new data arrives for affected entities.
    OnIngest,
    /// Recompute on a schedule (cron expression).
    Scheduled(String),
    /// Only recompute on explicit API request.
    Manual,
}

pub enum ViewStatus {
    /// View data is up-to-date with all ingested embeddings.
    Fresh,
    /// New data has arrived since last computation.
    Stale { since: i64 },
    /// View is currently being recomputed.
    Computing,
    /// Last computation failed.
    Error(String),
}
```

### 4.3 Built-in View Types

| View Type | Description | Compute Cost | Invalidation Granularity |
|-----------|-------------|-------------|--------------------------|
| `drift_summary` | Drift rate, trend, and statistics per entity per time window | O(T) per entity | Per entity |
| `temporal_features` | Feature vectors para ML: velocity, acceleration, drift entropy, stability score | O(T) per entity | Per entity |
| `changepoint_cache` | PELT results cached. Invalidated when new data arrives for an entity. | O(T²) per entity (PELT) | Per entity |
| `cohort_snapshot` | Periodic cohort divergence matrices (all-pairs drift in a cohort) | O(N² · T) per cohort | Per cohort |

### 4.4 Invalidation Protocol

Cuando nuevos embeddings llegan para la entidad X:

1. **Identify affected views:** buscar en un índice invertido `entity_id → [view_names]`.
2. **Mark as Stale:** para cada vista afectada, actualizar `status = Stale { since: now }`.
3. **Trigger refresh (if OnIngest):** encolar una tarea de recomputation solo para los entities afectados. **No se recomputa toda la vista**, solo los entries que corresponden a los entities con nuevos datos.
4. **Scheduled views:** se recomputan en su próximo ciclo de schedule. El estado `Stale` se preserva hasta entonces.

**Costo de invalidation:** O(1) por entity (hash lookup), O(V) total donde V = número de vistas afectadas.

### 4.5 Storage

Las materialized views se almacenan en un **dedicated RocksDB column family** `views`:

```
Key:    view_name (string) ++ entity_id (BE u64)
Value:  serialized view result (MessagePack or bincode)
```

Metadata de la vista (definition, status, schedule) se almacena en el column family `view_metadata`:

```
Key:    view_name (string)
Value:  serialized MaterializedView definition
```

### 4.6 Caso de Uso: Social Media Classifier

Del Temporal ML Spec, el caso del clasificador de riesgo en redes sociales:

1. **Crear vista:** `POST /v1/views` con tipo `temporal_features` para los 100K usuarios.
2. **Durante training:** query `GET /v1/views/user_risk_features/data?entity_id=42` — respuesta instantánea desde cache.
3. **Nuevos posts llegan:** solo los usuarios afectados se marcan como `Stale`. Con `OnIngest`, sus features se recomputan inmediatamente.
4. **Siguiente epoch de training:** features actualizadas disponibles sin recomputation global.

**Impacto:** training loop iteration baja de ~minutes (recompute features) a ~seconds (cache lookup).

### 4.7 API Endpoints

#### `POST /v1/views`

Crear una materialized view.

**Request:**
```json
{
  "name": "user_risk_features",
  "type": "temporal_features",
  "space": "bert-v2",
  "entity_filter": { "tag": "social_media_user" },
  "refresh_policy": "on_ingest",
  "retention": "30d",
  "config": {
    "features": ["velocity", "acceleration", "drift_entropy", "stability_score"],
    "window": "7d"
  }
}
```

**Response:**
```json
{
  "name": "user_risk_features",
  "status": "computing",
  "estimated_entities": 102341,
  "created_at": "2026-03-15T10:00:00Z"
}
```

#### `GET /v1/views`

Listar todas las vistas materializadas con su status.

**Response:**
```json
{
  "views": [
    {
      "name": "user_risk_features",
      "type": "temporal_features",
      "status": "fresh",
      "entities": 102341,
      "last_computed": "2026-03-15T10:05:00Z",
      "stale_entities": 0
    }
  ]
}
```

#### `GET /v1/views/{name}/data`

Query datos materializados.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `entity_id` | `u64` | No | — | Specific entity (if omitted, returns all) |
| `entity_ids` | `string` | No | — | Comma-separated entity IDs |
| `include_stale` | `bool` | No | `true` | Whether to include stale entries |

#### `POST /v1/views/{name}/refresh`

Manual refresh de una vista.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `entity_ids` | `array` | No | all stale | Specific entities to refresh |
| `force` | `bool` | No | `false` | Re-compute even if Fresh |

#### `DELETE /v1/views/{name}`

Eliminar una vista y sus datos cacheados.

### 4.8 NFR

| Requirement | Target |
|-------------|--------|
| View data lookup (single entity) | < 1ms |
| Invalidation marking (per entity) | < 0.1ms |
| Refresh scheduling overhead | < 100us |
| View creation (100K entities, temporal_features) | < 5 min |
| Storage overhead per cached entity | < 1KB (feature vectors) |

---

## 5. Provenance & Lineage

### 5.1 Motivation

Cada embedding en CVX debería poder responder: **"¿De dónde vienes?"** — qué modelo lo produjo, qué versión, qué input procesó, cuándo fue creado, y cuándo llegó a CVX.

Sin provenance, las anomalías son cajas negras. Con provenance, un ML engineer puede trazar un change point hasta el pipeline que produjo los embeddings anómalos.

### 5.2 EmbeddingProvenance Type

```rust
/// Provenance metadata for a single embedding in CVX.
/// Answers: who produced this embedding, when, and how?
pub struct EmbeddingProvenance {
    /// Name of the source that ingested this embedding.
    /// Maps to a configured source name (e.g., "bert-v2-triton", "s3-daily-sync").
    pub source_name: String,
    /// Model version identifier (if known).
    pub model_version: Option<String>,
    /// Hash of the model weights used to produce this embedding.
    pub model_hash: Option<String>,
    /// Hash of the input that was embedded (for reproducibility checks).
    pub input_hash: Option<String>,
    /// LabChain pipeline hash (if this embedding was produced by a CVX experiment).
    pub pipeline_hash: Option<String>,
    /// When CVX received this embedding (ingestion timestamp).
    pub ingested_at: i64,
    /// When the source system produced this embedding (if known).
    /// Can differ from ingested_at for batch sources.
    pub produced_at: Option<i64>,
}
```

### 5.3 Storage

Provenance se almacena en el column family `metadata`, keyed por la misma tupla que los vectores:

```
Key:    entity_id (BE u64) ++ space_id (BE u32) ++ timestamp (BE i64)
Value:  serialized EmbeddingProvenance
```

**Trade-off de storage:** provenance añade ~100-200 bytes por embedding. Para un dataset de 100M embeddings, esto es ~10-20GB. Aceptable dado el valor diagnóstico.

### 5.4 Provenance Queries

| Query | Use Case | Implementation |
|-------|----------|----------------|
| "¿Todos los embeddings de esta trayectoria son del mismo modelo?" | Consistency check antes de drift analysis | Scan provenance by entity_id, check `model_version` uniformity |
| "¿Qué pipeline produjo el embedding que triggeó este change point?" | Root cause analysis | Lookup provenance at change point timestamp |
| "¿Cuándo fue el último embedding de la fuente X?" | Freshness monitoring | Reverse scan by source_name |
| "¿Hay embeddings con model_hash diferente al registrado?" | Data quality check | Scan + compare against ModelVersion registry |

### 5.5 API

#### `GET /v1/entities/{id}/provenance`

Obtener la cadena de provenance para una entidad.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `space` | `string` | No | default | Embedding space |
| `from` | `i64` | No | first_seen | Start timestamp |
| `to` | `i64` | No | last_seen | End timestamp |
| `source_name` | `string` | No | all | Filter by source |

**Response:**
```json
{
  "entity_id": 42,
  "provenance": [
    {
      "timestamp": 1640000000,
      "source_name": "s3-daily-sync",
      "model_version": "bert-v1",
      "model_hash": "abc123",
      "input_hash": "def456",
      "pipeline_hash": null,
      "ingested_at": 1640003600,
      "produced_at": 1640000000
    },
    {
      "timestamp": 1650000000,
      "source_name": "triton-realtime",
      "model_version": "bert-v2",
      "model_hash": "ghi789",
      "input_hash": "jkl012",
      "pipeline_hash": "labchain-exp-42",
      "ingested_at": 1650000005,
      "produced_at": 1650000000
    }
  ]
}
```

### 5.6 Integration con LabChain

Cuando un pipeline de LabChain produce embeddings que se ingestan en CVX, el `pipeline_hash` de LabChain se propaga como provenance. Esto cierra el loop:

```
LabChain Pipeline → Embedding → CVX Ingest (with pipeline_hash)
                                     ↓
CVX Analytics → Change Point Detected
                                     ↓
Provenance Query → pipeline_hash → LabChain → Full experiment context
```

### 5.7 Lineage en cvx-explain

Para el módulo de interpretabilidad (cvx-explain), provenance enriquece las narrativas:

- **Trajectory projections:** cada punto anotado con su fuente y modelo.
- **Change point narratives:** "Change point detectado en t=1650000000. Este embedding fue producido por el pipeline `labchain-exp-42` usando `bert-v2` (model_hash: `ghi789`). El embedding anterior era de `bert-v1` — ver §3 Model Version Alignment para determinar si es artefacto de retrain."

---

## 6. Semantic Layer & Monitors

### 6.1 Semantic Entities

Las consultas en CVX operan sobre entity_ids numéricos y space_ids. En producción, los usuarios piensan en **conceptos**: "usuarios de alto riesgo", "productos del catálogo premium", "contenido viral reciente". Una capa semántica mapea conceptos con nombre a filtros de CVX.

```rust
/// A named semantic concept that maps to a set of CVX entities and spaces.
pub struct SemanticEntity {
    /// Human-readable name (e.g., "user_risk_trajectory").
    pub name: String,
    /// Description of what this concept represents.
    pub description: String,
    /// Filter that selects which entities belong to this concept.
    pub entity_filter: EntityFilter,
    /// Which embedding spaces are relevant for this concept.
    pub spaces: Vec<String>,
    /// Default time range for queries on this concept.
    pub default_time_range: Option<TimeRange>,
}
```

### 6.2 Monitors: Declarative Alerting

Los monitors son reglas de alerting declarativas que evalúan condiciones sobre métricas temporales y ejecutan acciones cuando se cumplen.

```rust
/// A declarative alerting rule that monitors temporal metrics.
pub struct Monitor {
    /// Unique name for this monitor.
    pub name: String,
    /// Human-readable description of what this monitor watches.
    pub description: String,
    /// Condition that triggers the alert.
    pub condition: MonitorCondition,
    /// Action to take when the condition is met.
    pub action: MonitorAction,
    /// How often to evaluate the condition.
    pub check_interval: Duration,
    /// Alert severity level.
    pub severity: Severity,
}

pub enum MonitorCondition {
    /// Alert when drift rate exceeds threshold within the given window.
    DriftExceeds { threshold: f64, window: Duration },
    /// Alert when a change point is detected with at least this severity.
    ChangePointDetected { min_severity: f64 },
    /// Alert when velocity exceeds threshold (rapid embedding evolution).
    VelocityExceeds { threshold: f64 },
    /// Alert when an entity has no new embeddings for longer than duration.
    SilenceExceeds { duration: Duration },
    /// Alert based on a custom temporal query expression.
    CustomQuery(String),
}

pub enum MonitorAction {
    /// Push alert to the WatchDrift gRPC stream (existing real-time API).
    EmitEvent,
    /// POST alert payload to an external webhook URL.
    WebhookPost(String),
    /// Log the alert at the specified level.
    Log { level: String },
}

pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}
```

### 6.3 Config-Driven Monitors

Monitors se pueden definir en `config.toml` para deployment declarativo:

```toml
[[monitors]]
name = "high_risk_drift"
description = "Alert when user language shifts toward risk pattern"
condition = { type = "drift_exceeds", threshold = 0.3, window = "7d" }
action = { type = "webhook_post", url = "https://alerts.example.com/cvx" }
check_interval = "1h"
severity = "high"

[[monitors]]
name = "model_silence"
description = "Alert when no embeddings arrive from production model"
condition = { type = "silence_exceeds", duration = "6h" }
action = { type = "emit_event" }
check_interval = "30m"
severity = "critical"

[[monitors]]
name = "changepoint_alert"
description = "Notify on significant change points in user trajectories"
condition = { type = "change_point_detected", min_severity = 0.8 }
action = { type = "webhook_post", url = "https://slack-webhook.example.com/cvx-alerts" }
check_interval = "15m"
severity = "medium"
```

### 6.4 Implementation Strategy

- **Periodic monitors** (`DriftExceeds`, `VelocityExceeds`, `SilenceExceeds`): evaluados por un scheduler que ejecuta las queries temporales correspondientes en cada `check_interval`. Se apoyan en **materialized views** cuando existen (e.g., si hay una vista `drift_summary`, el monitor la consulta en lugar de recomputar).
- **Event-driven monitors** (`ChangePointDetected`): se apoyan en **BOCPD (online)** para detección en tiempo real. Cuando BOCPD emite un run-length reset, el monitor evalúa si la severidad excede el threshold.
- **Deduplication:** para evitar alertas repetidas, cada monitor mantiene un `last_alert_at` por entity. No se re-alerta sobre la misma condición dentro de un `cooldown` configurable (default: 1 hora).

### 6.5 API Endpoints

#### `POST /v1/monitors`

Crear un monitor.

**Request:**
```json
{
  "name": "high_risk_drift",
  "description": "Alert when user language shifts toward risk pattern",
  "condition": {
    "type": "drift_exceeds",
    "threshold": 0.3,
    "window": "7d"
  },
  "action": {
    "type": "webhook_post",
    "url": "https://alerts.example.com/cvx"
  },
  "check_interval": "1h",
  "severity": "high"
}
```

#### `GET /v1/monitors`

Listar todos los monitors con su estado actual.

#### `GET /v1/monitors/{name}/history`

Historial de alertas emitidas por este monitor.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `from` | `i64` | No | 24h ago | Start timestamp |
| `to` | `i64` | No | now | End timestamp |
| `limit` | `usize` | No | 100 | Max alerts to return |

#### `DELETE /v1/monitors/{name}`

Eliminar un monitor.

---

## 7. Architecture Integration

### 7.1 New Modules

No se introducen nuevos crates. Toda la funcionalidad se integra en crates existentes:

```
crates/cvx-ingest/src/
    └── sources/
        ├── mod.rs            // EmbeddingSource trait, SourceManager, scheduler
        ├── s3_parquet.rs     // S3ParquetSource
        ├── kafka.rs          // KafkaSource
        ├── rest_api.rs       // RestApiSource
        ├── grpc_stream.rs    // GrpcStreamSource
        └── pgvector.rs       // PgVectorSource

crates/cvx-core/src/
    ├── provenance.rs         // EmbeddingProvenance type
    ├── views.rs              // MaterializedView, RefreshPolicy, ViewStatus types
    └── monitors.rs           // Monitor, MonitorCondition, MonitorAction, Severity types

crates/cvx-analytics/src/
    └── alignment/
        └── versioning.rs     // ModelVersion, AlignmentTransform, CanonicalTrajectory

crates/cvx-storage/src/
    └── views.rs              // View storage engine, invalidation index, RocksDB CF management

crates/cvx-api/src/rest/
    ├── views.rs              // CRUD endpoints for materialized views
    ├── monitors.rs           // CRUD endpoints for monitors
    └── sources.rs            // Source management + sync status endpoints
```

### 7.2 Dependency Graph

```
cvx-core        → (none)                          [+ provenance.rs, views.rs, monitors.rs]
cvx-index       → cvx-core                        [sin cambios]
cvx-storage     → cvx-core                        [+ views.rs, views CF, view_metadata CF]
cvx-ingest      → cvx-core, cvx-index, cvx-storage [+ sources/]
cvx-analytics   → cvx-core, cvx-storage           [+ alignment/versioning.rs]
cvx-query       → cvx-core, cvx-index, cvx-storage, cvx-analytics
cvx-explain     → cvx-core, cvx-analytics, cvx-query, cvx-storage
cvx-api         → cvx-core, cvx-query, cvx-ingest, cvx-explain [+ views.rs, monitors.rs, sources.rs]
cvx-server      → cvx-api
```

No se introducen dependencias cíclicas. Cada nuevo módulo se integra en el crate donde semánticamente pertenece.

### 7.3 Feature Flags

```toml
[features]
# Source connectors
source-s3 = ["aws-sdk-s3", "arrow", "parquet"]
source-kafka = ["rdkafka"]
source-rest = ["reqwest"]
source-grpc = ["tonic"]
source-pgvector = ["sqlx", "tokio-postgres"]

# Temporal analytics features
materialized-views = []
monitors = ["materialized-views"]  # monitors leverage views for periodic checks
model-versioning = ["multi-space"]  # requires multi-space from MultiScale Alignment Spec
```

---

## 8. Roadmap Position

Cada componente de este spec se ubica en el roadmap iterativo existente (CVX_Iterative_Roadmap.md) según sus dependencias técnicas:

| Component | Layer | Rationale |
|-----------|-------|-----------|
| **Provenance** | Layer 6 | Solo requiere API + metadata storage. Sin dependencias pesadas. |
| **Model Version Alignment** | Layer 7.5+ | Requiere multi-space (Layer 7) + alignment (MultiScale Spec). |
| **Materialized Views** | Layer 8+ | Requiere PELT (Layer 8) para changepoint_cache, analytics para temporal_features. |
| **Monitors** | Layer 8 | Requiere BOCPD (Layer 8) para event-driven monitors. |
| **Source Connectors** | Layer 9 | Requiere tiered storage con S3 (Layer 9) para el source-s3 connector. Kafka y REST pueden adelantarse. |

**Nota:** provenance puede (y debería) implementarse temprano. Es metadata pura — no tiene costo computacional y habilita diagnóstico desde el día uno.

---

## 9. What We Explicitly Do NOT Do

CVX adopta conceptos selectos de data virtualization. Estos son los conceptos que **explícitamente rechazamos** y por qué:

| Concepto Denodo/DV | Status | Razón del rechazo |
|---------------------|--------|-------------------|
| **Query-time federation** | Rechazado | Temporal analytics requieren historia completa. No puedes computar PELT sobre datos que viven en S3 — necesitas materializarlos primero. Federation sería O(T) en latencia por query. |
| **Distributed query optimizer** | Rechazado | CVX no es un query engine SQL. Las queries temporales (drift, velocity, change points) son operaciones analíticas sobre series, no joins relacionales. Un optimizer SQL no tiene valor. |
| **Row/column level security** | Rechazado | Scope de CVX. Security se maneja en la capa de API Gateway (Architecture Doc §11). Si se necesita, se implementa como middleware, no como feature del storage layer. |
| **Full SQL interface** | Rechazado | Los embeddings temporales no son tabulares. Las operaciones fundamentales (drift, alignment, change point detection) no se expresan naturalmente en SQL. CVX tiene su propio query model optimizado para series temporales de vectores. |
| **Data catalog & discovery** | Rechazado | Scope demasiado amplio. CVX registra spaces y model versions — eso es suficiente. Un catálogo completo con search, tags, y governance es un producto separado (Apache Atlas, DataHub, etc). |

---

## 10. Non-Functional Requirements

Resumen consolidado de targets de rendimiento para todos los componentes de este spec:

| Component | Operation | Latency Target | Throughput Target |
|-----------|-----------|----------------|-------------------|
| **Source Connectors** | Incremental sync (S3 Parquet, 10K records) | < 30s | 10K records/s |
| **Source Connectors** | Kafka consumer throughput | — | 50K records/s |
| **Source Connectors** | Sync lag (time since last successful sync) | < 2x schedule interval | — |
| **Model Version Alignment** | Procrustes computation (10K entities, D=768) | < 30s | — |
| **Model Version Alignment** | Canonical projection per point | < 0.1ms | 10M points/s |
| **Model Version Alignment** | Alignment quality query | < 5ms | — |
| **Materialized Views** | View data lookup (single entity) | < 1ms | — |
| **Materialized Views** | Invalidation marking (per entity) | < 0.1ms | — |
| **Materialized Views** | Refresh scheduling overhead | < 100us | — |
| **Materialized Views** | Full view creation (100K entities) | < 5 min | — |
| **Provenance** | Provenance lookup (single entity, full trajectory) | < 5ms | — |
| **Provenance** | Provenance write (per embedding) | < 0.05ms overhead | — |
| **Monitors** | Monitor evaluation (single condition) | < 10ms | — |
| **Monitors** | Alert dispatch (webhook) | < 500ms (async) | — |
| **Monitors** | Deduplication check | < 0.1ms | — |

---

## 11. Open Questions

1. **¿Cuál es el threshold mínimo de alignment quality para considerar válida una alineación de versiones de modelo?** Un alignment residual de 0.1 podría ser aceptable para algunos use cases pero no para otros. ¿Deberíamos definir un threshold global o dejarlo configurable por model family? Recomendación: configurable, con un default conservador (mean_residual < 0.05 para `high` confidence).

2. **¿Cuál es el overhead aceptable de evaluación de monitors?** Con 100 monitors activos evaluándose cada 15 minutos, el costo total de evaluación no debería exceder el 5% del CPU disponible. ¿Es esto realista? Depende de si los monitors pueden apoyarse en materialized views (O(1) lookup) o necesitan recomputar queries (O(T) per entity). Recomendación: requerir que monitors periódicos se apoyen en vistas materializadas.

3. **¿Qué source connector priorizar?** S3/Parquet y Kafka son los más demandados en producción ML. REST API es el más genérico. pgvector es nicho pero atrae usuarios de PostgreSQL. Recomendación: Kafka primero (event-driven, se alinea con real-time ingestion), S3/Parquet segundo (batch historical data), REST API tercero.

4. **¿Cómo manejar la retención de provenance?** Provenance crece linealmente con el número de embeddings. ¿Deberíamos aplicar la misma política de tiered storage (hot/warm/cold) a provenance, o mantenerlo todo en hot storage dado su tamaño relativamente pequeño? Recomendación: seguir la misma tier que el vector — si el vector se mueve a cold, su provenance también.
