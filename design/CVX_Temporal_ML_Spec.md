# ChronosVector — Differentiable Temporal ML & Social Media Use Case

**Version:** 1.0
**Author:** Manuel Couto Pintos
**Date:** March 2026
**Status:** Draft
**Dependencies:** Architecture Doc §10 (Analytics), §16 (Feature Flags), Roadmap Layer 10 (Neural ODE)

---

## 1. Motivation

CVX extrae features temporales ricas de trayectorias de embeddings: velocidad, aceleración, drift, change points, volatilidad. Estas features son extremadamente útiles para tareas downstream de clasificación — por ejemplo, detección temprana de trastornos psicológicos a partir de historiales de publicaciones en redes sociales.

Sin embargo, si estas features no son **diferenciables**, el gradiente del clasificador no puede propagarse hasta el modelo de embeddings base (BERT, sentence-transformers). Esto impide el fine-tuning end-to-end y limita la potencia del sistema.

### 1.1 El Problema de la Backpropagation

```
Texto → Embedding Model → v(t) → CVX features → Clasificador → loss
         (BERT)                    ────────────
                                   Si esto no es diferenciable,
                                   el gradiente muere aquí y
                                   BERT no puede ajustarse
```

### 1.2 La Solución: Dual Path

CVX ofrece **dos caminos** para features temporales — mismo cálculo matemático, diferente contexto de ejecución:

| Camino | Implementación | Diferenciable | Propósito |
|--------|---------------|---------------|-----------|
| **Analítico** | Rust puro, SIMD | No | Análisis, interpretación, API, explain |
| **ML** | burn / tch-rs con autograd | Sí | Training end-to-end, fine-tuning |

Ambos viven en `cvx-analytics`, comparten la misma lógica matemática vía un trait `TemporalOps`, y producen los mismos resultados numéricos. La diferencia es que el camino ML registra las operaciones en un grafo de autograd.

### 1.3 Non-Goal

CVX **no es un framework de entrenamiento general**. No reemplaza a PyTorch, burn ni ningún framework ML. Lo que CVX ofrece es:
- **Storage temporal** de embeddings (core de CVX)
- **Features temporales diferenciables** que participan en training loops externos
- **Analytics no diferenciables** para interpretación (PELT, BOCPD, drift attribution)

El training loop, el optimizador, el scheduler — eso lo gestiona el usuario o su framework.

---

## 2. Architecture

### 2.1 Dual Backend via Trait Abstraction

```rust
/// Trait que abstrae operaciones temporales sobre tensores.
/// Implementado para backend analítico (Rust puro) y ML (burn/tch-rs).
pub trait TemporalOps {
    type Tensor;

    /// Velocity: finite differences (v[t+1] - v[t]) / dt
    fn velocity(embeddings: &Self::Tensor, timestamps: &Self::Tensor) -> Self::Tensor;

    /// Acceleration: second-order finite differences
    fn acceleration(embeddings: &Self::Tensor, timestamps: &Self::Tensor) -> Self::Tensor;

    /// Total drift vector: v[t_last] - v[t_first]
    fn drift(embeddings: &Self::Tensor) -> Self::Tensor;

    /// Per-dimension volatility: std of consecutive deltas
    fn volatility(embeddings: &Self::Tensor, timestamps: &Self::Tensor) -> Self::Tensor;

    /// Soft change point detection (differentiable relaxation of PELT)
    fn soft_changepoints(
        embeddings: &Self::Tensor,
        timestamps: &Self::Tensor,
        temperature: f64,
    ) -> Self::Tensor;

    /// Velocity magnitudes over time
    fn velocity_magnitudes(embeddings: &Self::Tensor, timestamps: &Self::Tensor) -> Self::Tensor;

    /// Temporal exponential moving average
    fn temporal_ema(embeddings: &Self::Tensor, timestamps: &Self::Tensor, decay: f64) -> Self::Tensor;

    /// Concatenate all features into a single vector
    fn extract_all(
        embeddings: &Self::Tensor,
        timestamps: &Self::Tensor,
        config: &TemporalFeaturesConfig,
    ) -> Self::Tensor;
}
```

### 2.2 Backends

#### Backend Analítico (Rust puro, SIMD)

```rust
pub struct AnalyticBackend;

impl TemporalOps for AnalyticBackend {
    type Tensor = Vec<Vec<f32>>; // o ndarray si se prefiere

    fn velocity(embeddings: &Self::Tensor, timestamps: &Self::Tensor) -> Self::Tensor {
        // Implementación directa con SIMD auto-vectorization
        // No registra operaciones en ningún grafo de autograd
        // Rápida, para serving y API /explain
    }
    // ...
}
```

#### Backend burn (Rust puro, autograd, CUDA)

```rust
use burn::prelude::*;

pub struct BurnBackend<B: burn::tensor::backend::AutodiffBackend> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B: AutodiffBackend> TemporalOps for BurnBackend<B> {
    type Tensor = burn::tensor::Tensor<B, 2>;

    fn velocity(embeddings: &Self::Tensor, timestamps: &Self::Tensor) -> Self::Tensor {
        // Operaciones sobre burn tensors — autograd habilitado
        // Gradientes fluyen a través de estas operaciones
        let dt = timestamps.clone().slice([1..]).sub(timestamps.slice([..(-1)]));
        embeddings.clone().slice([1..])
            .sub(embeddings.clone().slice([..(-1)]))
            .div(dt.unsqueeze_dim(1))
    }
    // ...
}
```

#### Backend tch-rs (libtorch, interop PyTorch)

```rust
use tch::Tensor;

pub struct TorchBackend;

impl TemporalOps for TorchBackend {
    type Tensor = tch::Tensor;

    fn velocity(embeddings: &Self::Tensor, timestamps: &Self::Tensor) -> Self::Tensor {
        // Operaciones sobre tensores de libtorch
        // Comparten autograd graph con PyTorch de Python
        // Zero-copy via PyO3 → gradientes cruzan la frontera Rust↔Python
        let dt = timestamps.slice(0, 1, -1, 1) - timestamps.slice(0, 0, -2, 1);
        (embeddings.slice(0, 1, -1, 1) - embeddings.slice(0, 0, -2, 1))
            / dt.unsqueeze(-1)
    }
    // ...
}
```

### 2.3 Cuándo usar cada backend

| Contexto | Backend | Razón |
|----------|---------|-------|
| API REST/gRPC (serving) | `AnalyticBackend` | Rápido, sin overhead de autograd |
| `cvx-explain` endpoints | `AnalyticBackend` | No necesita gradientes |
| Training 100% Rust | `BurnBackend` | Autograd + CUDA, sin Python |
| Wrapper Python con fine-tuning PyTorch | `TorchBackend` | Gradientes compatibles con PyTorch |
| Neural ODE (ya planificado) | `BurnBackend` | Ya usa burn |

---

## 3. Feature Extraction: Differentiable vs Non-Differentiable

### 3.1 Features Naturalmente Diferenciables

Estas features se computan igual en todos los backends. En el backend analítico son cálculos puros; en burn/tch-rs, registran operaciones en el autograd graph.

| Feature | Fórmula | ∂feature/∂embedding |
|---------|---------|---------------------|
| **Velocity** | `(v[t+1] - v[t]) / dt` | `-1/dt` y `+1/dt` (lineal) |
| **Acceleration** | `(v[t+2] - 2v[t+1] + v[t]) / dt²` | Lineal |
| **Drift total** | `v[t_last] - v[t_first]` | `±1` (trivial) |
| **Per-dim delta** | `\|v[t2][d] - v[t1][d]\|` | `sign(delta)` |
| **Volatilidad** | `std(‖v[t+1] - v[t]‖)` | Derivada de std, bien definida |
| **Cosine distance** | `1 - cos(v[t1], v[t2])` | Estándar, diferenciable |
| **EMA** | `Σ e^{-λ·age} · v[t]` | Pesos exponenciales fijos |
| **Mean velocity magnitude** | `mean(‖velocity‖)` | Composición de norma + media |
| **Neural ODE latent state** | `z(t) = ODESolve(f_θ, z_0, t)` | Método adjunto (Chen 2018) |

### 3.2 Soft Relaxations (Diferenciables ≈ Features Discretas)

Features que en el camino analítico son discretas (PELT, conteos) pero tienen aproximaciones continuas diferenciables:

| Feature discreta | Relaxación diferenciable | Parámetro |
|-----------------|--------------------------|-----------|
| **Número de change points** | `Σ σ((deviation - μ) / τ)` — suma de sigmoids | τ (temperatura) |
| **Severidad máxima** | `softmax(severity) · severity` — smooth max | τ |
| **Top-K dimensiones** | `gumbel_softmax(\|delta\|, τ)` | τ |
| **Conteo de silencios** | `Σ σ((gap - θ) / τ)` | θ (umbral), τ |
| **Indicador de cambio** | `σ(CUSUM(‖velocity‖))` — soft CUSUM | — |

A medida que τ → 0, las relaxaciones convergen a las versiones discretas. Durante training se usa τ > 0 para permitir gradientes; en inference se puede usar τ → 0 para resultados exactos.

### 3.3 Features No Diferenciables (Solo Camino Analítico)

Estas features solo existen en el camino analítico. No participan en backpropagation.

| Feature | Por qué no es diferenciable |
|---------|----------------------------|
| PELT (segmentación exacta) | Optimización combinatoria discreta |
| BOCPD posterior | Run-length discreto |
| Timestamp exacto de change point | Argmin discreto |
| Rank de dimensiones | Operación ordinal |
| Change point narrative | Genera texto/estructura, no numérica |
| Drift attribution (Pareto) | Sorting + cumsum con threshold discreto |

Estas features son para **interpretación humana**, no para training. El investigador las usa para entender qué aprendió el modelo, no para enseñarle.

---

## 4. Learnable Components

El `TemporalFeatureExtractor` no solo computa features fijas — incluye componentes **entrenables** que se optimizan junto con el clasificador:

### 4.1 Dimension Attention

Un módulo lineal que aprende **qué dimensiones del embedding importan** para la tarea:

```rust
#[derive(Module, Debug)]
pub struct DimensionAttention<B: Backend> {
    projection: Linear<B>,
}

impl<B: AutodiffBackend> DimensionAttention<B> {
    pub fn forward(&self, drift: Tensor<B, 1>) -> Tensor<B, 1> {
        let weights = activation::sigmoid(self.projection.forward(drift.clone()));
        drift * weights  // Soft masking — gradientes fluyen
    }
}
```

Esto permite que el modelo aprenda, por ejemplo, que las dimensiones asociadas a "afecto negativo" son más relevantes que las de "sintaxis" para detección de depresión. Es equivalente a la drift attribution de CVX, pero aprendido end-to-end.

### 4.2 Temporal Scale Weights

Pesos aprendidos para la importancia relativa de cada escala temporal:

```rust
#[derive(Module, Debug)]
pub struct MultiScaleAggregator<B: Backend> {
    scale_weights: Param<Tensor<B, 1>>, // n_scales weights, learnable
}
```

El modelo aprende si la señal está más en la escala diaria, semanal o mensual — análogo al multi-scale analysis de CVX, pero optimizado para la tarea específica.

### 4.3 Soft Change Point Temperature

La temperatura τ del soft change point detector puede ser aprendible:

```rust
pub struct SoftChangePointDetector<B: Backend> {
    temperature: Param<Tensor<B, 1>>, // learnable temperature
}
```

El modelo aprende la "sensibilidad" óptima a cambios para la tarea.

---

## 5. Temporal Feature Extractor (Módulo Completo)

```rust
#[derive(Module, Debug)]
pub struct TemporalFeatureExtractor<B: Backend> {
    /// Learnable: qué dimensiones del embedding importan
    dim_attention: DimensionAttention<B>,
    /// Learnable: peso por escala temporal
    scale_aggregator: MultiScaleAggregator<B>,
    /// Learnable: sensibilidad a change points
    cpd: SoftChangePointDetector<B>,
}

pub struct TemporalFeaturesConfig {
    pub embed_dim: usize,
    pub n_scales: usize,
    pub initial_temperature: f64,
}

impl<B: AutodiffBackend> TemporalFeatureExtractor<B> {
    pub fn forward(
        &self,
        embeddings: Tensor<B, 2>,  // (seq_len, embed_dim)
        timestamps: Tensor<B, 1>,  // (seq_len,)
    ) -> Tensor<B, 1> {
        // Todas las operaciones registradas en autograd graph

        // 1. Velocity & acceleration (diferenciable)
        let velocity = BurnBackend::<B>::velocity(&embeddings, &timestamps);
        let acceleration = BurnBackend::<B>::acceleration(&embeddings, &timestamps);
        let vel_magnitudes = BurnBackend::<B>::velocity_magnitudes(&embeddings, &timestamps);

        // 2. Drift con dimension attention (learnable)
        let drift = BurnBackend::<B>::drift(&embeddings);
        let attended_drift = self.dim_attention.forward(drift.clone());

        // 3. Volatilidad (diferenciable)
        let volatility = BurnBackend::<B>::volatility(&embeddings, &timestamps);

        // 4. Soft change points (diferenciable, temperatura learnable)
        let soft_cp = self.cpd.forward(&vel_magnitudes);

        // 5. Multi-scale features (pesos learnable)
        let scale_features = self.scale_aggregator.forward(&vel_magnitudes);

        // 6. Trajectory curvature (diferenciable)
        let curvature = velocity_to_curvature(&velocity);

        // Concatenar → feature vector
        Tensor::cat(vec![
            velocity.mean_dim(0),            // (D,)
            attended_drift,                   // (D,)
            volatility,                       // (D,)
            acceleration.mean_dim(0),         // (D,)
            vel_magnitudes.mean().unsqueeze(), // (1,)
            vel_magnitudes.max().unsqueeze(),  // (1,) - soft approximation
            soft_cp.sum().unsqueeze(),         // (1,) ~n_changepoints
            soft_cp.max().unsqueeze(),         // (1,) ~max_severity
            curvature.mean().unsqueeze(),      // (1,)
            scale_features,                    // (n_scales * 2,)
        ], 0)
    }
}
```

### 5.1 Feature Vector Dimensionality

Para D=768, n_scales=3:

| Component | Dims |
|-----------|------|
| Mean velocity | 768 |
| Attended drift | 768 |
| Volatility | 768 |
| Mean acceleration | 768 |
| Scalar features (5) | 5 |
| Scale features | 6 |

**Total: 3077** features — un vector de tamaño fijo independiente de la longitud de la secuencia. Esto resuelve el problema de secuencias de longitud variable (usuarios con 10 posts vs 500 posts producen el mismo feature vector).

---

## 6. Use Case: Social Media Temporal Classification

### 6.1 Estructura del Problema

```
Dataset:
  - N usuarios (entidades)
  - Cada usuario tiene M_i posts (secuencia temporal, M_i variable)
  - Cada post tiene: texto, timestamp
  - Label binario por usuario (positivo/negativo)
  - Train/test split a nivel de usuario

Objetivo:
  - Clasificar usuarios a partir de la evolución temporal de sus posts
  - Interpretabilidad: ¿qué cambió, cuándo, en qué dimensiones?
  - Early detection: ¿se puede detectar antes de tener todos los posts?
```

### 6.2 Pipeline End-to-End

```
                         TRAINING (diferenciable, burn/tch-rs)
                         ════════════════════════════════════
Phase 1: Embed       Texto → BERT (entrenable) → embeddings por post
Phase 2: Store       embeddings → CVX storage (temporal, por usuario)
Phase 3: Features    CVX TemporalFeatureExtractor → feature vector
Phase 4: Classify    feature vector → Linear → logits → loss
Phase 5: Backprop    loss.backward() → gradientes hasta BERT

                         ANALYSIS (no diferenciable, cvx-analytics)
                         ════════════════════════════════════════
Phase 6: Interpret   CVX PELT → change points por usuario
                     CVX drift attribution → dimensiones responsables
                     CVX trajectory projection → visualización 2D
                     CVX cohort divergence → ¿cuándo divergen pos/neg?
```

### 6.3 Código (burn, 100% Rust)

```rust
use burn::prelude::*;
use cvx_analytics::temporal_ml::{TemporalFeatureExtractor, TemporalFeaturesConfig};

#[derive(Module, Debug)]
pub struct SocialMediaClassifier<B: Backend> {
    /// Pre-trained encoder (e.g., BERT via candle o burn)
    encoder: TextEncoder<B>,
    /// Temporal feature extraction (differentiable)
    temporal: TemporalFeatureExtractor<B>,
    /// Classification head
    classifier: Linear<B>,
}

impl<B: AutodiffBackend> SocialMediaClassifier<B> {
    pub fn forward(
        &self,
        post_tokens: Vec<Tensor<B, 2>>,  // tokens por post
        timestamps: Tensor<B, 1>,         // timestamp por post
    ) -> Tensor<B, 1> {
        // 1. Embed cada post (diferenciable → BERT se fine-tunea)
        let embeddings: Vec<Tensor<B, 1>> = post_tokens.iter()
            .map(|tokens| self.encoder.forward(tokens))
            .collect();
        let embeddings = Tensor::stack(embeddings, 0); // (seq_len, D)

        // 2. Features temporales (diferenciable)
        let features = self.temporal.forward(embeddings, timestamps);

        // 3. Clasificar
        self.classifier.forward(features)
    }
}

// Training loop
fn train_step<B: AutodiffBackend>(
    model: &SocialMediaClassifier<B>,
    batch: UserBatch<B>,
    optimizer: &mut impl Optimizer<B>,
) -> f64 {
    let logits = model.forward(batch.posts, batch.timestamps);
    let loss = binary_cross_entropy(logits, batch.label);

    // Gradientes fluyen: loss → classifier → temporal features → BERT
    let grads = loss.backward();
    optimizer.step(model, grads);

    loss.into_scalar()
}
```

### 6.4 Código (tch-rs, wrapper Python)

```rust
// En cvx-python (crate con PyO3)
use pyo3::prelude::*;
use tch::Tensor;
use cvx_analytics::temporal_ml::TorchBackend;

#[pyfunction]
fn temporal_features(embeddings: &Tensor, timestamps: &Tensor) -> Tensor {
    // Misma lógica que BurnBackend, pero sobre tensores libtorch
    // Los gradientes se preservan en el autograd graph de PyTorch
    TorchBackend::extract_all(embeddings, timestamps, &default_config())
}

#[pymodule]
fn cvx_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(temporal_features, m)?)?;
    Ok(())
}
```

```python
# Python — gradientes cruzan la frontera transparentemente
import torch
from transformers import AutoModel
import cvx_python  # Rust compiled extension

bert = AutoModel.from_pretrained("bert-base-uncased")
classifier = torch.nn.Linear(3077, 2)

for batch in dataloader:
    embeddings = bert(batch.tokens).last_hidden_state[:, 0]  # (seq, 768)
    features = cvx_python.temporal_features(embeddings, batch.timestamps)  # Rust, autograd ✓
    logits = classifier(features)
    loss = F.cross_entropy(logits, batch.label)
    loss.backward()  # gradientes llegan hasta BERT ✓
```

### 6.5 Interpretability (Post-Training)

Una vez entrenado el modelo, CVX proporciona interpretación no diferenciable:

```rust
// Tras el training, analizar un usuario clasificado como positivo
let user_id = 42;

// ¿Cuándo cambió?
let changepoints = cvx.changepoints(user_id, method: Pelt, sensitivity: 0.5);
// → [ChangePoint { timestamp: 2024-03-15, severity: 0.87 }]

// ¿Qué dimensiones cambiaron?
let attribution = cvx.drift_attribution(user_id, t1: first_post, t2: last_post, top_k: 10);
// → [dim 42: 0.23, dim 157: 0.19, dim 384: 0.15, ...]

// ¿Se parece a otros usuarios positivos?
let trajectory = cvx.trajectory(user_id, t1: first_post, t2: last_post);
let similar = cvx.temporal_knn(trajectory.last(), k: 5, at: trajectory.last_timestamp());
// → usuarios con trayectorias similares (mayoría positivos)

// ¿Se podría haber detectado antes?
for week in 1..total_weeks {
    let partial_features = cvx.temporal_features(user_id, t1: first_post, t2: week_end(week));
    let prediction = classifier.predict(partial_features);
    // → detecta en semana 8 con 85% confidence
}
```

### 6.6 Validación: ¿El Learnable Dimension Attention Converge a lo Esperado?

Tras el training, podemos comparar los pesos aprendidos del `DimensionAttention` con la drift attribution analítica de CVX:

```rust
// Pesos aprendidos por el modelo (end-to-end)
let learned_weights = model.temporal.dim_attention.get_weights();

// Drift attribution analítica (CVX, no diferenciable)
let analytical_attribution = cvx.drift_attribution(user_id, t1, t2);

// ¿Correlacionan?
let correlation = spearman_rank_correlation(learned_weights, analytical_attribution);
// Si correlation alta → el modelo aprendió lo que CVX ya sabía
// Si correlation baja → el modelo descubrió señales que CVX no captura
//                       (las relaxaciones diferenciables capturaron algo nuevo)
```

Esto es una **validación cruzada entre los dos caminos**: el analítico y el ML. Si convergen, aumenta la confianza en ambos. Si divergen, el investigador tiene algo interesante que investigar.

---

## 7. Module Structure

```
crates/cvx-analytics/src/
├── temporal_ml/
│   ├── mod.rs              // TemporalOps trait, TemporalFeaturesConfig
│   ├── ops.rs              // Trait definition + AnalyticBackend implementation
│   ├── burn_backend.rs     // BurnBackend (feature-gated: temporal-ml-burn)
│   ├── torch_backend.rs    // TorchBackend (feature-gated: temporal-ml-torch)
│   ├── features.rs         // TemporalFeatureExtractor (burn Module)
│   ├── attention.rs        // DimensionAttention, MultiScaleAggregator
│   ├── soft_cpd.rs         // SoftChangePointDetector
│   └── classifier.rs       // Example classifier (SocialMediaClassifier)
```

---

## 8. Feature Flags

```toml
# In cvx-analytics/Cargo.toml

[features]
# Analytic temporal features (always available, no extra deps)
# — uses AnalyticBackend, no autograd

# Differentiable temporal features via burn (pure Rust, CUDA)
temporal-ml-burn = ["burn"]

# Differentiable temporal features via tch-rs (PyTorch interop)
temporal-ml-torch = ["tch"]
```

---

## 9. Specific Features for Social Media Data

### 9.1 Handling Irregular Timestamps

Los posts en redes sociales no son periódicos. La implementación debe manejar:

- **Gaps largos**: un usuario que no publica en 3 semanas. La velocity se calcula con `dt` real, no asumiendo intervalos uniformes.
- **Ráfagas**: 10 posts en una hora. Opcionalmente agregar por ventana temporal antes de computar features.
- **Silencios como feature**: el patrón de actividad temporal (cuándo publica, gaps, frecuencia) es información en sí mismo.

```rust
pub struct PostingPatternFeatures {
    pub mean_gap: f64,
    pub max_gap: f64,
    pub gap_variance: f64,
    pub posting_frequency: f64,           // posts per day
    pub burst_count: usize,               // periods with >N posts/hour
    pub silence_count: usize,             // gaps > threshold
    pub soft_silence_count: Tensor,       // differentiable approximation
    pub temporal_entropy: f64,            // uniformity of posting distribution
}
```

### 9.2 Sequence Length Normalization

Usuarios con 10 posts y usuarios con 500 posts deben producir features comparables. Las features basadas en medias (mean velocity, mean acceleration) son naturalmente normalizadas. Las basadas en sumas (n_changepoints) necesitan normalización:

```rust
// Normalización por duración de la ventana temporal
let normalized_cp_count = soft_cp_count / (t_last - t_first);

// O por número de posts
let normalized_cp_count = soft_cp_count / n_posts as f64;
```

### 9.3 Early Detection Mode

Para evaluar detección temprana, el feature extractor debe operar sobre **prefijos** de la trayectoria:

```rust
impl<B: AutodiffBackend> TemporalFeatureExtractor<B> {
    /// Extract features from only the first `n` posts.
    pub fn forward_prefix(
        &self,
        embeddings: Tensor<B, 2>,
        timestamps: Tensor<B, 1>,
        n_posts: usize,
    ) -> Tensor<B, 1> {
        let prefix_emb = embeddings.slice([..n_posts]);
        let prefix_ts = timestamps.slice([..n_posts]);
        self.forward(prefix_emb, prefix_ts)
    }
}
```

Esto permite evaluar: "¿Con cuántos posts podemos clasificar correctamente al 90% de los usuarios?"

---

## 10. Benchmarks Específicos

Estos benchmarks se añaden al plan general (CVX_Benchmark_Plan.md):

| Benchmark | Description |
|-----------|-------------|
| **SM-1** | Classification accuracy: temporal features vs mean pooling vs last embedding |
| **SM-2** | Early detection curve: accuracy vs number of posts used |
| **SM-3** | Feature importance: ¿qué features temporales aportan más? (ablation) |
| **SM-4** | End-to-end vs frozen: ¿fine-tuning BERT con backprop mejora sobre BERT congelado? |
| **SM-5** | Dimension attention convergence: ¿correlaciona con drift attribution analítica? |
| **SM-6** | Interpretability quality: ¿los change points detectados coinciden con eventos conocidos? |
| **SM-7** | Computational overhead: temporal features vs raw embeddings (latency, memory) |
| **SM-8** | Multi-scale robustness: ¿a qué escala temporal está la señal? |

### Dataset para Benchmarks

| Dataset | Description | Source |
|---------|-------------|--------|
| **eRisk (CLEF)** | Early risk detection on social media | CLEF shared task datasets |
| **CLPsych** | Mental health on Reddit | CLPsych shared task |
| **Synthetic** | Planted trajectory patterns with known change points | Generated |

---

## 11. Non-Functional Requirements

| Requirement | Target |
|-------------|--------|
| Feature extraction (100 posts, D=768, burn CPU) | < 1ms |
| Feature extraction (100 posts, D=768, burn CUDA) | < 0.5ms |
| Feature extraction (100 posts, D=768, tch-rs) | < 0.5ms |
| Batch extraction (1000 users × 100 posts, CUDA) | < 100ms |
| Backward pass overhead vs forward-only | < 2x |
| Memory per user trajectory (in autograd graph) | < 10MB |
| Feature vector size (fixed, independent of seq length) | 3077 (for D=768, 3 scales) |

---

## 12. Relationship with Existing CVX Components

```
cvx-analytics
├── diffcalc/          Analytical velocity, acceleration, drift
│                      (Rust puro, SIMD, no autograd)
│                      → Usado por API, explain, serving
│
├── temporal_ml/       Differentiable temporal features
│                      (burn/tch-rs, autograd, CUDA)
│                      → Usado para training end-to-end
│                      → MISMAS FÓRMULAS que diffcalc
│
├── pelt/              Change point detection (discrete, exact)
│                      → Interpretación post-hoc
│
├── bocpd/             Online change point detection
│                      → Monitorización en tiempo real
│
├── ode/               Neural ODE (burn, differentiable)
│                      → Predicción de trayectorias
│                      → Comparte backend burn con temporal_ml
│
├── alignment/         Cross-space alignment
│                      → Analítico
│
└── multiscale/        Multi-scale analysis
    └── temporal_ml puede usar multi-scale features
        como componentes entrenables
```

### 12.1 Roadmap Integration

| Layer | Temporal ML Component |
|-------|----------------------|
| **L7** | `diffcalc` (analítico) → base para temporal_ml |
| **L7.5** | `TemporalOps` trait definition |
| **L10** | `burn` backend completo (Neural ODE ya usa burn) |
| **L10.5** | `TemporalFeatureExtractor` con componentes learnable |
| **L12** | `tch-rs` backend para wrapper Python, social media benchmarks |

---

## 13. Open Questions

1. **¿burn o candle para el encoder (BERT)?** `candle` tiene mejor soporte para cargar modelos de HuggingFace. `burn` es más ergonómico para definir training loops. Podrían coexistir: candle para cargar pesos, burn para el training loop.

2. **¿Soft change points convergen a PELT?** La relajación sigmoid es una aproximación. ¿Es suficientemente buena para que las features aprendidas capturen los mismos patrones que PELT encuentra analíticamente? Requiere validación empírica.

3. **¿Qué proporción del feature vector debería ser learnable?** El `DimensionAttention` añade parámetros entrenables. ¿Debería haber más componentes learnable (e.g., learned temporal kernels) o menos (para evitar overfitting en datasets pequeños)?

4. **¿El training loop debería vivir en CVX o en el código del usuario?** Recomendación: CVX proporciona el `TemporalFeatureExtractor` (nn.Module / burn Module), pero el training loop es responsabilidad del usuario. CVX no es un framework de training.
