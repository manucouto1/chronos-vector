# ChronosVector — Multi-Modal & Multi-Scale Alignment Specification

**Version:** 1.0
**Author:** Manuel Couto Pintos
**Date:** March 2026
**Status:** Draft
**Dependencies:** Architecture Doc §5 (Data Model), PRD §2.1 (FR-01), RFC-001 (ADR-002 Composite Distance)

---

## 1. Motivation

Los embeddings del mundo real no son homogéneos. Un sistema de producción puede necesitar almacenar y correlacionar:

- **Embeddings de texto** (D=768, modelo BERT) actualizados diariamente.
- **Embeddings de imagen** (D=512, modelo CLIP) actualizados por hora.
- **Embeddings de usuario** (D=128, modelo de recomendación) actualizados en tiempo real.
- **Embeddings de grafo** (D=64, TransE) actualizados semanalmente.

Estos embeddings viven en **espacios diferentes** (dimensionalidad, escala, frecuencia de actualización, métrica de distancia óptima) pero representan **las mismas entidades o entidades relacionadas**. Un ML Engineer quiere preguntar: "¿Cuándo divergieron las representaciones de texto e imagen de este producto?" Un researcher quiere: "¿La evolución semántica textual predice la evolución visual?"

ChronosVector, al ser un **VDB temporal**, está en una posición única para resolver este problema: no solo almacena múltiples representaciones, sino que puede analizar cómo evolucionan *con respecto a las demás* a lo largo del tiempo.

### 1.1 Diferenciación Competitiva

| Capacidad | Qdrant / Milvus | CVX (actual) | CVX (con Multi-Scale) |
|-----------|----------------|--------------|----------------------|
| Múltiples vectores por entidad | Named vectors (Qdrant) | Un vector por (entity, timestamp) | Múltiples espacios por entidad |
| Correlación cross-modal | No | No | Alignment Score temporal |
| Escala temporal heterogénea | N/A | Asume frecuencia uniforme | Interpolación + resampling |
| Análisis de coherencia | No | No | Coherence Drift detection |
| Projección cross-space | No | No | Alignment projection |

### 1.2 Casos de Uso

| Caso | Descripción | Query Ejemplo |
|------|-------------|---------------|
| **Multi-modal drift** | Detectar cuándo las representaciones de texto e imagen de un producto divergen | "¿Cuándo dejó de ser coherente la representación visual y textual del producto X?" |
| **Cross-modal prediction** | Usar la evolución textual para predecir la evolución visual | "Si el embedding textual de X cambió así, ¿cómo cambiará su embedding visual?" |
| **Scale-aware analytics** | Comparar drift a diferentes escalas temporales | "El drift diario es ruidoso, pero ¿hay una tendencia semanal consistente?" |
| **Embedding alignment monitoring** | Monitorizar que múltiples modelos mantienen coherencia | "¿Mis modelos v1 y v2 siguen produciendo representaciones alineadas?" |
| **Temporal knowledge fusion** | Combinar información de múltiples fuentes temporales | "¿Qué dice la evidencia combinada (texto + imagen + grafo) sobre la evolución de X?" |

---

## 2. Conceptos Fundamentales

### 2.1 Embedding Space

Un **Embedding Space** es un espacio vectorial registrado en CVX con propiedades definidas:

```rust
pub struct EmbeddingSpace {
    /// Unique identifier for this space.
    pub space_id: u32,
    /// Human-readable name (e.g., "text-bert-768", "image-clip-512").
    pub name: String,
    /// Vector dimensionality in this space.
    pub dimensionality: u32,
    /// Default distance metric for this space.
    pub metric: DistanceMetricType,
    /// Typical update frequency (informational, for resampling hints).
    pub typical_frequency: Option<TemporalFrequency>,
    /// Normalization applied to vectors in this space.
    pub normalization: Normalization,
    /// Optional: description of what this space represents.
    pub description: Option<String>,
}

pub enum TemporalFrequency {
    RealTime,        // sub-second updates
    Minutely,
    Hourly,
    Daily,
    Weekly,
    Monthly,
    Irregular,       // no fixed frequency
}

pub enum Normalization {
    UnitNorm,        // L2 normalized to ||v|| = 1
    None,
    Custom(String),  // named normalization scheme
}
```

### 2.2 Multi-Space Entity

Una entidad puede tener vectores en **múltiples espacios**, cada uno con su propia trayectoria temporal:

```rust
pub struct MultiSpaceEntity {
    pub entity_id: u64,
    /// Map from space_id to the entity's timeline in that space.
    pub spaces: HashMap<u32, EntityTimeline>,
}
```

La tupla fundamental pasa de `(entity_id, timestamp, vector)` a `(entity_id, space_id, timestamp, vector)`.

### 2.3 Alignment Function

Una **Alignment Function** mide la coherencia entre dos espacios para una misma entidad a lo largo del tiempo. No compara vectores directamente (están en espacios diferentes) — compara *comportamientos*:

```rust
pub trait AlignmentFunction: Send + Sync {
    /// Compute alignment score between two trajectories.
    /// Returns 1.0 for perfectly aligned evolution, 0.0 for uncorrelated.
    fn alignment_score(
        &self,
        trajectory_a: &[TemporalPoint],
        trajectory_b: &[TemporalPoint],
        window: TimeRange,
    ) -> AlignmentResult;
}
```

---

## 3. Alignment Methods

### 3.1 Structural Alignment (Topology Preservation)

Compara si las **relaciones de vecindario** se preservan entre espacios. Si los vecinos de X en el espacio textual son los mismos que en el espacio visual, los espacios están alineados para X.

| Property | Specification |
|----------|---------------|
| **Algorithm** | Compute kNN(entity, space_a, t) and kNN(entity, space_b, t). Measure Jaccard similarity of neighbor sets. |
| **Output** | `structural_alignment ∈ [0.0, 1.0]` per timestamp |
| **Advantage** | No requiere que los espacios tengan la misma dimensionalidad |
| **Cost** | O(k² · T) where T = timestamps, k = neighbor count |

```rust
pub struct StructuralAlignment;

impl AlignmentFunction for StructuralAlignment {
    fn alignment_score(
        &self,
        trajectory_a: &[TemporalPoint],
        trajectory_b: &[TemporalPoint],
        window: TimeRange,
    ) -> AlignmentResult {
        // For each timestamp in window:
        //   1. Get kNN of entity in space A
        //   2. Get kNN of entity in space B
        //   3. Compute Jaccard(neighbors_A, neighbors_B)
        // Return mean Jaccard over timestamps
        todo!()
    }
}
```

### 3.2 Behavioral Alignment (Drift Correlation)

Compara si los **patrones de cambio** son similares entre espacios. Si X cambia rápido en texto, ¿también cambia rápido en imagen?

| Property | Specification |
|----------|---------------|
| **Algorithm** | Compute per-step drift magnitude in each space. Correlate the two drift time series (Pearson or Spearman). |
| **Output** | `behavioral_alignment ∈ [-1.0, 1.0]` (correlation coefficient) |
| **Advantage** | Scale-invariant — different dimensionalities and magnitudes don't matter |
| **Cost** | O(T · max(D_a, D_b)) |

```rust
pub struct BehavioralAlignment {
    pub correlation_method: CorrelationMethod,
}

pub enum CorrelationMethod {
    Pearson,
    Spearman,
    KendallTau,
}
```

### 3.3 Procrustes Alignment (Geometric)

Cuando los espacios tienen la misma dimensionalidad (o se proyectan a una común), el **análisis de Procrustes** encuentra la mejor transformación (rotación + escala) que alinea las trayectorias.

| Property | Specification |
|----------|---------------|
| **Algorithm** | Orthogonal Procrustes: find rotation R minimizing `||A - BR||²_F`. After alignment, residual error measures misalignment. |
| **Prerequisite** | Same dimensionality, or projected to common dim via PCA/CCA |
| **Output** | `procrustes_distance ∈ [0.0, ∞)` (0 = perfect alignment) |
| **Cost** | O(T · D²) for SVD |

### 3.4 Canonical Correlation Analysis (CCA)

Para espacios de diferente dimensionalidad, CCA encuentra subespacios donde las correlaciones son máximas.

| Property | Specification |
|----------|---------------|
| **Algorithm** | Classical CCA: find projection matrices W_a, W_b that maximize correlation between projected spaces |
| **Output** | - Canonical correlations (sorted, highest first)<br>- Projection matrices for cross-space comparison<br>- Effective alignment dimensionality (how many dimensions are correlated) |
| **Cost** | O(T · (D_a + D_b)²) |
| **Use** | As a preprocessing step for cross-space kNN or as a standalone alignment measure |

---

## 4. Multi-Scale Temporal Analysis

### 4.1 The Problem

Embeddings from different sources update at different frequencies. Text embeddings might update daily, image embeddings hourly, and user behavior embeddings in real-time. To analyze cross-space alignment, we need to bring these timelines to a **common temporal scale**.

### 4.2 Temporal Resampling

```rust
pub struct ResamplingConfig {
    /// Target frequency for alignment analysis.
    pub target_frequency: TemporalFrequency,
    /// How to handle gaps when upsampling.
    pub interpolation: InterpolationMethod,
    /// How to handle multiple values when downsampling.
    pub aggregation: AggregationMethod,
}

pub enum InterpolationMethod {
    /// Use the last known value (zero-order hold).
    LastValue,
    /// Linear interpolation between known points.
    Linear,
    /// Spherical linear interpolation (preserves unit norm on sphere).
    Slerp,
    /// Use Neural ODE to interpolate continuous trajectory.
    NeuralOde,
}

pub enum AggregationMethod {
    /// Use the last value in the bin.
    Last,
    /// Average vectors in the bin (after normalization).
    Mean,
    /// Use the value with highest confidence/freshness.
    MostRecent,
}
```

**Interpolation Details:**

- **LastValue:** Simple, no assumptions. Good for sparse updates. Introduces staircase artifacts.
- **Linear:** Assumes linear path between snapshots. Good for short gaps. Can produce non-unit-norm vectors (re-normalize after).
- **Slerp:** Spherical linear interpolation. Preserves unit sphere geometry. Ideal for cosine-metric spaces.
- **NeuralOde:** Uses trained Neural ODE to interpolate. Most accurate but expensive. Only available after Layer 10.

### 4.3 Multi-Scale Drift Analysis

Drift can appear different at different temporal scales. Daily noise might mask weekly trends, or amplify them.

```rust
pub struct MultiScaleDriftAnalysis {
    pub entity_id: u64,
    pub space_id: u32,
    pub scales: Vec<ScaleDriftReport>,
}

pub struct ScaleDriftReport {
    pub scale: TemporalFrequency,
    pub mean_drift_rate: f64,
    pub drift_variance: f64,
    /// Trend: is drift accelerating at this scale?
    pub trend: DriftTrend,
    /// Change points detected at this scale.
    pub change_points: Vec<ChangePoint>,
    /// Signal-to-noise: drift signal vs measurement noise at this scale.
    pub snr: f64,
}

pub enum DriftTrend {
    Accelerating,
    Decelerating,
    Stable,
    Oscillating,
}
```

**Analysis Protocol:**
1. Resample trajectory to each target scale (hourly → daily → weekly → monthly).
2. At each scale, compute drift time series.
3. Apply change point detection at each scale.
4. Compare results across scales: change points that persist across scales are high-confidence. Change points that appear only at fine scales are likely noise.

### 4.4 Cross-Scale Coherence

```rust
pub struct CrossScaleCoherence {
    /// Does the drift pattern at scale_fine predict drift at scale_coarse?
    pub fine_to_coarse_correlation: f64,
    /// Change points that appear at multiple scales.
    pub robust_change_points: Vec<RobustChangePoint>,
    /// Scale at which signal-to-noise ratio is maximized.
    pub optimal_analysis_scale: TemporalFrequency,
}

pub struct RobustChangePoint {
    pub timestamp: i64,
    pub severity: f64,
    /// At how many scales was this change point detected?
    pub scale_count: usize,
    pub scales_detected: Vec<TemporalFrequency>,
}
```

---

## 5. Data Model Extensions

### 5.1 Storage Layout

La tupla fundamental se extiende:

```
Antes:  Key = entity_id (BE u64) + timestamp (BE i64)
Ahora:  Key = entity_id (BE u64) + space_id (BE u32) + timestamp (BE i64)
```

**Impact on RocksDB Column Families:**

| CF | Key antes | Key ahora | Notes |
|----|-----------|-----------|-------|
| `vectors` | `entity_id ++ ts` | `entity_id ++ space_id ++ ts` | Prefix scan by entity still works (gets all spaces) |
| `deltas` | `entity_id ++ ts` | `entity_id ++ space_id ++ ts` | Delta chain is per-space |
| `timelines` | `entity_id` | `entity_id ++ space_id` | Timeline metadata per entity per space |
| `metadata` | `entity_id` | `entity_id` | Entity-level metadata (shared across spaces) |
| `spaces` | (new) | `space_id` | Space definitions |

**Backward Compatibility:** Si `space_id` no se proporciona, se usa `space_id = 0` (default space). Esto mantiene compatibilidad con el API existente.

### 5.2 Index Impact

Cada espacio tiene su **propio ST-HNSW index** (diferentes dimensionalidades impiden un índice compartido):

```rust
pub struct MultiSpaceIndex {
    /// One ST-HNSW index per embedding space.
    pub indices: HashMap<u32, StHnsw>,
}
```

El coste es O(S) índices donde S = número de espacios. En la práctica, S es pequeño (2-10 espacios típicamente).

### 5.3 Ingestion Extension

```
Antes:  POST /v1/ingest { entity_id, timestamp, vector, metadata }
Ahora:  POST /v1/ingest { entity_id, space: "text-bert-768", timestamp, vector, metadata }
```

Si `space` no se proporciona, se usa el espacio default. Nuevo endpoint para registrar espacios:

```
POST /v1/spaces { name, dimensionality, metric, typical_frequency, normalization }
GET  /v1/spaces
GET  /v1/spaces/{name}
```

---

## 6. API Endpoints

### 6.1 Space Management

#### `POST /v1/spaces`

Registrar un nuevo embedding space.

**Request:**
```json
{
  "name": "text-bert-768",
  "dimensionality": 768,
  "metric": "cosine",
  "typical_frequency": "daily",
  "normalization": "unit_norm",
  "description": "BERT base embeddings for article text"
}
```

**Response:**
```json
{
  "space_id": 1,
  "name": "text-bert-768",
  "created_at": "2026-03-15T00:00:00Z"
}
```

#### `GET /v1/spaces`

Listar todos los espacios registrados.

#### `GET /v1/spaces/{name}`

Obtener detalles de un espacio.

### 6.2 Cross-Space Alignment Analysis

#### `GET /v1/alignment/entities/{id}`

Medir alignment entre dos espacios para una entidad.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `space_a` | `string` | Yes | — | First embedding space name |
| `space_b` | `string` | Yes | — | Second embedding space name |
| `from` | `i64` | No | entity first_seen | Start timestamp |
| `to` | `i64` | No | entity last_seen | End timestamp |
| `method` | `string` | No | `"behavioral"` | `"structural"`, `"behavioral"`, `"procrustes"`, `"cca"` |
| `resample` | `string` | No | auto | Resampling frequency for alignment |

**Response:**
```json
{
  "entity_id": 42,
  "space_a": "text-bert-768",
  "space_b": "image-clip-512",
  "method": "behavioral",
  "alignment_score": 0.73,
  "alignment_over_time": [
    { "timestamp": 1640000000, "score": 0.85 },
    { "timestamp": 1640100000, "score": 0.71 },
    { "timestamp": 1640200000, "score": 0.42 }
  ],
  "divergence_points": [
    { "timestamp": 1640200000, "severity": 0.43 }
  ]
}
```

#### `POST /v1/alignment/cohort`

Alignment analysis para múltiples entidades a la vez.

**Request:**
```json
{
  "entity_ids": [42, 43, 44],
  "space_a": "text-bert-768",
  "space_b": "image-clip-512",
  "from": 1640000000,
  "to": 1700000000,
  "method": "behavioral"
}
```

**Response:** Lista de alignment results por entidad + aggregate statistics.

#### `GET /v1/alignment/entities/{id}/cross-prediction`

Predicción cross-modal: usar la evolución en un espacio para predecir la evolución en otro.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source_space` | `string` | Yes | — | Space with known trajectory |
| `target_space` | `string` | Yes | — | Space to predict |
| `target_timestamp` | `i64` | Yes | — | Timestamp to predict |

**Response:** Predicted vector in target space + confidence + alignment-based uncertainty.

### 6.3 Multi-Scale Analysis

#### `GET /v1/multiscale/entities/{id}/drift`

Drift analysis at multiple temporal scales.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `space` | `string` | No | default | Embedding space |
| `from` | `i64` | No | entity first_seen | Start timestamp |
| `to` | `i64` | No | entity last_seen | End timestamp |
| `scales` | `string` | No | `"hourly,daily,weekly"` | Comma-separated scales |

**Response:** `MultiScaleDriftAnalysis` (JSON) — drift report at each requested scale.

#### `GET /v1/multiscale/entities/{id}/robust-changepoints`

Change points that persist across multiple temporal scales.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `space` | `string` | No | default | Embedding space |
| `min_scales` | `usize` | No | `2` | Minimum scales for a change point to be "robust" |

**Response:** `CrossScaleCoherence` (JSON).

---

## 7. Architecture Integration

### 7.1 New Modules

```
crates/cvx-core/src/
    ├── spaces.rs          // EmbeddingSpace, TemporalFrequency, Normalization types
    └── alignment.rs       // AlignmentFunction trait, AlignmentResult

crates/cvx-analytics/src/
    ├── alignment/
    │   ├── mod.rs
    │   ├── structural.rs  // StructuralAlignment (Jaccard kNN)
    │   ├── behavioral.rs  // BehavioralAlignment (drift correlation)
    │   ├── procrustes.rs  // ProcrustesAlignment (geometric)
    │   └── cca.rs         // CCA alignment
    ├── multiscale/
    │   ├── mod.rs
    │   ├── resample.rs    // Temporal resampling (LastValue, Linear, Slerp)
    │   ├── scale_drift.rs // Per-scale drift analysis
    │   └── coherence.rs   // Cross-scale coherence
```

### 7.2 Dependency Graph Update

```
cvx-core        → (none)                          [+ spaces.rs, alignment.rs]
cvx-index       → cvx-core                        [MultiSpaceIndex wrapper]
cvx-storage     → cvx-core                        [extended key encoding]
cvx-ingest      → cvx-core, cvx-index, cvx-storage [space-aware ingestion]
cvx-analytics   → cvx-core, cvx-storage           [+ alignment/, multiscale/]
cvx-query       → cvx-core, cvx-index, cvx-storage, cvx-analytics
cvx-explain     → cvx-core, cvx-analytics, cvx-query, cvx-storage
cvx-api         → cvx-core, cvx-query, cvx-ingest, cvx-explain
cvx-server      → cvx-api
```

No se introducen nuevas dependencias cíclicas. La nueva funcionalidad vive en crates existentes + `cvx-explain`.

### 7.3 Roadmap Position

| Layer | Multi-Scale / Alignment Component |
|-------|-----------------------------------|
| **L0** | `EmbeddingSpace` type en `cvx-core` |
| **L1** | `space_id` en key encoding |
| **L3** | RocksDB storage con `spaces` CF, extended key format |
| **L4** | Per-space ST-HNSW indices |
| **L6** | Space management API endpoints |
| **L7** | Behavioral alignment, temporal resampling |
| **L7.5** | Alignment explain artifacts |
| **L8** | Multi-scale change point analysis |
| **L10** | Neural ODE interpolation for resampling, cross-modal prediction |
| **L12** | Cross-space benchmarks, alignment quality evaluation |

### 7.4 Feature Flags

```toml
[features]
# Multi-space support
multi-space = []          # Core multi-space types and storage

# Alignment methods (require multi-space)
alignment-structural = ["multi-space"]
alignment-behavioral = ["multi-space"]
alignment-procrustes = ["multi-space"]   # Requires linear algebra
alignment-cca = ["multi-space"]          # Requires linear algebra

# Multi-scale analysis
multiscale = []
```

---

## 8. Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| **Space registration** | < 1ms |
| **Behavioral alignment** (2 spaces, 1K timestamps) | < 50ms |
| **Structural alignment** (2 spaces, 1K timestamps, k=10) | < 500ms |
| **Procrustes alignment** (D=768, 1K timestamps) | < 200ms |
| **CCA** (D_a=768, D_b=512, 1K timestamps) | < 1s |
| **Temporal resampling** (10K points → 1K points) | < 10ms |
| **Multi-scale drift** (3 scales, 10K points) | < 500ms |
| **Storage overhead per space** | < 1KB metadata + standard vector storage |
| **Index overhead per space** | One ST-HNSW instance (same as single-space) |

---

## 9. Open Questions

1. **¿Cuántos espacios soportar?** Diseñamos para 2-10 espacios por entidad. ¿Deberíamos optimizar para más? Recomendación: no optimizar prematuramente; 10 es suficiente para todos los use cases identificados.

2. **¿CCA requiere `nalgebra` o `ndarray`?** Ambos son candidatos para álgebra lineal. `nalgebra` es más idiomático en Rust; `ndarray` es más familiar para gente de ML. Decisión diferida a implementación.

3. **¿Slerp para todos los espacios?** Slerp asume vectores en la esfera unitaria. Para espacios no normalizados, ¿usar interpolación lineal + renormalización o implementar interpolación geodésica en el espacio correspondiente?

4. **¿Cross-modal prediction es viable?** La predicción cross-modal asume que la evolución en un espacio informa la evolución en otro. Esto puede no ser cierto en todos los casos. ¿Deberíamos incluir un test de predictibilidad (Granger causality) antes de ofrecer predicciones?

5. **¿Impacto en rendimiento del key encoding extendido?** El campo `space_id` (4 bytes) extiende cada key. Para single-space workloads con `space_id=0`, ¿hay overhead medible? Benchmark pendiente.

---

## 10. Benchmarks Específicos

Estos benchmarks se añaden al plan general (CVX_Benchmark_Plan.md):

| Benchmark | Category | Description |
|-----------|----------|-------------|
| **MS-1** | Unique (A) | Cross-space alignment accuracy on CLIP text-image pairs |
| **MS-2** | Unique (A) | Multi-scale change point robustness (F1 at different scales) |
| **MS-3** | Unique (A) | Cross-modal prediction accuracy (text → image drift) |
| **MS-4** | Parity (B) | Single-space performance with multi-space key encoding (overhead test) |
| **MS-5** | Storage (C) | Storage overhead of N spaces vs 1 space |

### MS-1: Cross-Space Alignment Dataset

| Property | Value |
|----------|-------|
| **Source** | MS-COCO images + captions, monthly CLIP re-embeddings |
| **Spaces** | text-clip-512, image-clip-512 |
| **Ground Truth** | Same-entity text/image pairs are naturally aligned |
| **Metric** | Alignment score distribution for matched vs unmatched pairs |
