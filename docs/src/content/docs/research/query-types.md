---
title: "Query Types"
description: "The 10 temporal query types supported by ChronosVector"
---

## 5. Query Types (The Research Frontier)

| Query Type | Definición | Contexto Matemático | Complejidad |
|---|---|---|---|
| **Snapshot kNN** | Vecinos más cercanos en un instante exacto | $kNN(v, t_0)$ | $O(\log N)$ via ST-HNSW |
| **Range kNN** | kNN sobre una ventana temporal | $kNN(v, [t_1, t_2])$ | $O(\log N \cdot W)$ con W = timestamps en rango |
| **Evolutionary Path** | Trayectoria de un ID a través del tiempo | $\{v(t) \mid t \in [t_1, t_2]\}$ | $O(P)$ con P = puntos en path |
| **Vector Velocity** | Rapidez de cambio semántico | $\frac{\partial v}{\partial t} \approx \frac{v(t+\delta) - v(t)}{\delta}$ | $O(1)$ con deltas pre-computados |
| **Vector Acceleration** | Cambio de la velocidad de drift | $\frac{\partial^2 v}{\partial t^2}$ | $O(1)$ con deltas de segundo orden |
| **Extrapolation** | Predicción de posición futura | $v(t + \Delta t) \approx v(t) + \int_t^{t+\Delta t} f_\theta(v, \tau) d\tau$ | Depende del solver ODE |
| **Change Point Detection** | Detectar cuándo un concepto sufre drift brusco | PELT offline / BOCPD online | $O(N)$ / $O(1)$ amortizado |
| **Drift Quantification** | Magnitud y dirección del cambio | $\|v(t_2) - v(t_1)\|$ con métricas múltiples | $O(D)$ |
| **Temporal Analogy** | "¿Qué era en 2020 lo que X es en 2024?" | $v_{2020} + (v_X^{2024} - v_X^{2020})$ | $O(\log N)$ + kNN |
| **Cohort Divergence** | Cuándo dos conceptos empiezan a divergir | $d(v_A(t), v_B(t))$ como serie temporal → CPD | $O(T \cdot D)$ |

---

## 4. Advanced System Components

### 4.1 Neural Ingestion Engine (The Pipeline)

En lugar de un Message Queue pasivo, proponemos un **Stream Processor con normalización temporal y delta-compression**.

- **Tech:** Integración de **Fluvio** (alternativa moderna a Kafka escrita en Rust) para procesamiento *zero-copy*. Alternativamente, **Redpanda** vía cliente Rust para compatibilidad con ecosistemas Kafka existentes.
- **Feature: Delta-Embeddings.** Solo se almacena el cambio $\Delta v = v(t_{new}) - v(t_{prev})$ si $\|\Delta v\| > \epsilon$, optimizando almacenamiento. La reconstrucción del vector completo es $v(t_n) = v(t_0) + \sum_{i=1}^{n} \Delta v_i$ — similar a video coding con keyframes + deltas.
- **Keyframes periódicos:** Cada $K$ updates (configurable), se almacena el vector completo para limitar la acumulación de error y acelerar la reconstrucción.

### 4.2 Temporal Vector Index: ST-HNSW

Aquí convergen las ideas de HNSW, TANNS y nuestra innovación.

#### 4.2.1 Spatiotemporal HNSW (ST-HNSW)

La distancia combinada entre dos nodos incorpora proximidad semántica y temporal:

$$d_{ST}(u, v) = \alpha \cdot d_{sem}(u, v) + (1 - \alpha) \cdot d_{time}(t_u, t_v)$$

donde $\alpha$ es configurable por query, $d_{sem}$ puede ser coseno, euclídea o hiperbólica, y $d_{time}$ es una función de decaimiento temporal.

#### 4.2.2 Time-Decay Graphs

Implementación de un factor de decaimiento $\lambda$ donde la fuerza de las conexiones decae exponencialmente:

$$w(e_{uv}, t) = w_0(e_{uv}) \cdot e^{-\lambda(t - t_{creation})}$$

Las conexiones antiguas se debilitan pero no desaparecen, permitiendo búsquedas históricas rápidas. Las conexiones "tibias" actúan como "puentes temporales" que conectan épocas diferentes del espacio semántico.

#### 4.2.3 Timestamp Graph Integration

Adoptamos las ideas de Wang et al. (ICDE 2025):

- **Grafo unificado** con listas de vecinos versionadas por timestamp.
- **Backup neighbors** para mantener conectividad cuando los nodos expiran.
- **Historic Neighbor Tree (HNT)** para comprimir el historial de vecinos.

#### 4.2.4 DiskANN with Time-Partitioning

Para datasets que exceden la RAM, particionamos por rangos temporales y construimos índices Vamana por partición, mergeándolos en un índice global siguiendo la estrategia divide-and-conquer de DiskANN. Las particiones temporales recientes residen en RAM (hot), las históricas en SSD con PQ comprimido.

### 4.3 Storage Layer: Tiered Temporal Storage

| Tier | Technology | Datos | Latencia |
|------|-----------|-------|----------|
| **Hot** | LSM-Tree + RAM (RocksDB) | Vectores última hora/día | < 1ms |
| **Warm** | Parquet columnar (arrow-rs/polars) | Vectores última semana/mes | < 10ms |
| **Cold** | Object Store (S3/MinIO) + Zarr | Archivo histórico con PQ agresiva | < 100ms |

**Detalles del Hot Layer:**
- RocksDB con prefijo de timestamp como key prefix para range scans eficientes.
- Column families separadas para vectores completos, deltas, y metadata.
- Bloom filters por rango temporal para evitar lecturas innecesarias.

**Detalles del Warm Layer:**
- Formato columnar permite análisis de series temporales eficiente (e.g., calcular velocidad de drift sobre ventanas).
- Parquet con encoding dictionary para metadata repetitiva.

**Detalles del Cold Layer:**
- **Product Quantization adaptativa:** PQ con codebooks re-entrenados periódicamente para reflejar la distribución actual de los datos.
- **Formato Zarr** para lectura paralela masiva en pipelines de investigación (compatible con NumPy/Dask).

### 4.4 Compute Engine

#### 4.4.1 Distancias SIMD

La situación de SIMD en Rust (2025) ofrece varias opciones:

- **Auto-vectorización del compilador:** Funciona bien para código bien estructurado, pero no toca operaciones float sin flags explícitos.
- **`pulp`** (por el autor de `faer`): Soporta SSE, AVX2, AVX-512, NEON, WASM. API genérica sobre instruction sets.
- **`gemm`/`faer`**: Para operaciones matriciales de alto rendimiento.
- **`simsimd`**: Librería C con bindings Rust, optimizada específicamente para distancias vectoriales.

Para CVX, implementaremos **kernels de distancia especializados** para cada métrica soportada (coseno, L2, dot product, Poincaré) con dispatch dinámico basado en las capacidades del CPU detectadas en runtime.

#### 4.4.2 Tensor Operations

- **`burn`**: Framework de deep learning en Rust con backend-agnostic (Wgpu, NdArray, Candle, CUDA). Ideal para implementar los componentes de Neural ODE.
- **`candle`** (HuggingFace): Minimalist ML framework, más ligero que burn, con soporte para modelos pre-entrenados (LLaMA, CLIP, T5) y CUDA nativo.

Para el solver ODE, usaremos `burn` para la forward pass de $f_\theta$ y un solver adaptativo Dormand-Prince (RK45) implementado en Rust puro con soporte SIMD.
