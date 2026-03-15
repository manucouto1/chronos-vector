# ChronosVector — Stochastic Analytics & Quantitative Finance Layer

**Version:** 1.0
**Author:** Manuel Couto Pintos
**Date:** March 2026
**Status:** Draft
**Dependencies:** Architecture Doc §10 (Analytics Engine), Temporal ML Spec, Roadmap Layer 7-10

---

## 1. Motivation

### 1.1 Beyond First-Order Analytics

CVX ya computa un conjunto robusto de analticas temporales de "primer orden": velocidad (drift rate), aceleracin (cambio del drift), deteccin de change points (PELT, BOCPD), y divergencia de cohorte. Estas herramientas responden preguntas fundamentales: "cunto cambi?", "cundo cambi?", "se est alejando de su grupo?"

Sin embargo, estas mtricas son *descriptivas* y *deterministas*. No distinguen entre cambio real y ruido estadstico, no modelan la *incertidumbre* de la trayectoria, y no capturan patrones complejos como clustering de volatilidad, reversin a la media, o dependencia temporal de largo alcance.

The quantitative finance and stochastic processes literature provides a much richer mathematical toolkit for analyzing temporal trajectories. Decades of research in modeling financial time series, estimating volatility, detecting regime changes, and characterizing path-dependent behavior translate directly to the problem of analyzing embedding trajectories in a temporal vector database.

### 1.2 The Stochastic Process Framework

La insight clave: una trayectoria de embedding $v(t) \in \mathbb{R}^d$ puede modelarse como un proceso estocstico. En la formulacin ms general, un proceso de difusin:

$$
dv(t) = \mu(v, t) \, dt + \sigma(v, t) \, dW(t)
$$

donde:

- $\mu(v, t)$ es la **funcin de drift** — el cambio sistemtico, la "seal" en la trayectoria.
- $\sigma(v, t)$ es la **funcin de difusin/volatilidad** — la fluctuacin estocstica, el "ruido" en la trayectoria.
- $W(t)$ es un **proceso de Wiener $d$-dimensional** (movimiento browniano estndar).

This formulation is not merely an analogy. Embedding trajectories exhibit many of the same statistical properties as financial time series: they show periods of stability and turbulence, their volatility clusters, they may revert to equilibria or trend persistently, and their increments are rarely i.i.d. Gaussian.

### 1.3 What This Framework Connects

Este marco terico conecta el anlisis de trayectorias de CVX con una rica familia de herramientas matemticas:

| Framework | Connection to CVX | Key Insight |
|-----------|-------------------|-------------|
| **Brownian motion theory** | Drift vs noise separation | Is the observed velocity statistically significant, or just a random walk? |
| **Financial volatility modeling** (GARCH) | Volatility characterization | Embedding volatility clusters — periods of rapid change beget more rapid change |
| **Mean reversion** (Ornstein-Uhlenbeck) | Trajectory equilibrium analysis | Does this entity revert to an attractor, or does it wander freely? |
| **Path signatures** (rough path theory) | Universal trajectory descriptors | A fixed-dimensional vector that completely characterizes the shape of a trajectory |
| **Regime detection** (Markov switching) | Regime labeling | Not just *when* a change happens, but *what kind* of regime the entity occupies |
| **Optimal transport** (Wasserstein) | Distribution evolution | How does an entity's neighborhood change over time, measured rigorously? |
| **Neural SDEs** | Stochastic extension of Neural ODE | Prediction with calibrated uncertainty, not just point estimates |

### 1.4 Target Users

| Persona | Use Case | Feature |
|---------|----------|---------|
| Quant Researcher | Market regime matching — find historical periods similar to current market state | Temporal kNN + regime detection |
| Risk Manager | Factor decay monitoring — detect alpha erosion before it becomes critical | Mean reversion test + GARCH volatility |
| Clinical NLP Researcher | Trajectory pattern classification — classify patient note evolution patterns | Path signatures + regime labels |
| ML Engineer | Model drift characterization — determine if observed drift is signal or noise | Stationarity tests + drift significance |
| Data Scientist | Cross-entity temporal correlation — discover causal/co-moving relationships | DCC, co-integration, Granger causality |
| Portfolio Manager | Regime-conditional allocation — adjust weights based on detected regime | HMM regime detection + regime-aware prediction |
| Compliance Officer | Behavioral anomaly detection — flag trajectory deviations from regulatory norms | Cohort divergence + stochastic process classification |

### 1.5 Competitive Differentiation

Ningn vector database existente ofrece estas capacidades. Las bases de datos de series temporales (InfluxDB, TimescaleDB) operan sobre escalares, no vectores de alta dimensionalidad. Las plataformas de ML (MLflow, Weights & Biases) rastrean mtricas pero no modelan trayectorias como procesos estocsticos. CVX, al almacenar trayectorias vectoriales completas con resolucin temporal, es el nico sistema capaz de aplicar estas herramientas a embeddings.

| Capability | Traditional VDB | Time-Series DB | CVX (Current) | CVX (Stochastic Layer) |
|------------|----------------|----------------|---------------|------------------------|
| Drift significance | No | No | Velocity (unsigned) | t-test with p-value |
| Volatility model | No | Simple std | No | GARCH with forecasting |
| Mean reversion test | No | No | No | ADF + KPSS + OU params |
| Path signatures | No | No | No | Truncated signatures for trajectory similarity |
| Regime detection | No | No | Change points only | Full HMM regime labeling |
| Cross-entity causality | No | Granger (scalar) | Cohort divergence | Granger + co-integration + DCC |
| Stochastic prediction | No | ARIMA (scalar) | Neural ODE (deterministic) | Neural SDE (distributional) |

---

## 2. Stochastic Process Characterization (Per-Entity Features)

This section defines the per-entity stochastic analytics that CVX computes on individual embedding trajectories. Cada subseccin describe el contexto terico, el mtodo de estimacin, el tipo Rust, y el endpoint API.

### 2.1 Drift Significance Test

#### Context

CVX computes velocity (drift rate) as a first-class analytic. But a critical question remains unanswered: **is the observed velocity statistically significant, or could it arise from a pure random walk?**

Un entity con drift rate de 0.01 por timestep podra estar experimentando un cambio real y direccional, o simplemente fluctuando aleatoriamente. La diferencia tiene implicaciones profundas para la interpretacin y la accin.

#### Method

Under the null hypothesis $H_0$: no drift (pure random walk), the increments $\Delta v_i = v(t_{i+1}) - v(t_i)$ have zero mean. The test statistic is:

$$
t = \frac{\bar{\Delta v}}{\hat{\sigma}_{\Delta v} / \sqrt{n}}
$$

where $\bar{\Delta v}$ is the mean increment magnitude, $\hat{\sigma}_{\Delta v}$ is the sample standard deviation, and $n$ is the number of increments. This follows a t-distribution with $n-1$ degrees of freedom under $H_0$.

For multivariate drift, we use the Hotelling $T^2$ test on the vector of mean increments, which reduces to the scalar t-test when applied to drift magnitudes.

#### Rust Type

```rust
/// Result of testing whether an entity's drift is statistically significant.
/// A significant drift indicates directional movement beyond random fluctuation.
pub struct DriftSignificance {
    /// Entity under analysis.
    pub entity_id: u64,
    /// Time window of analysis.
    pub time_range: (i64, i64),
    /// Mean drift vector (direction of movement).
    pub drift_vector: Vec<f32>,
    /// Magnitude of the mean drift vector.
    pub drift_magnitude: f64,
    /// t-statistic for the drift magnitude.
    pub t_statistic: f64,
    /// p-value from the t-test (two-tailed).
    pub p_value: f64,
    /// Whether drift is significant at the given threshold.
    pub is_significant: bool, // p_value < threshold (default 0.05)
    /// Number of increments used in the test.
    pub n_increments: usize,
    /// Effect size (Cohen's d) for practical significance.
    pub effect_size: f64,
}
```

#### API

```
GET /v1/stochastic/entities/{id}/drift-significance?from=&to=&threshold=0.05
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `id` | `u64` | Yes | — | Entity ID |
| `from` | `i64` | No | first_seen | Start timestamp |
| `to` | `i64` | No | last_seen | End timestamp |
| `threshold` | `f64` | No | `0.05` | Significance level (alpha) |

**Response:**
```json
{
  "entity_id": 42,
  "time_range": [1640000000, 1700000000],
  "drift_magnitude": 0.0134,
  "t_statistic": 3.42,
  "p_value": 0.0007,
  "is_significant": true,
  "n_increments": 365,
  "effect_size": 0.18
}
```

### 2.2 Realized Volatility

#### Context

En finanzas, la volatilidad (desviacin estndar de los retornos) es quizs la mtrica ms importante despus del retorno mismo. Para trayectorias de embeddings, la volatilidad mide la *variabilidad* del cambio — no cunto cambi en promedio, sino cun *errtico* fue el cambio.

Financial volatility = std of log-returns. Embedding volatility = std of per-step drift magnitudes. Una volatilidad alta indica que la trayectoria es impredecible; una volatilidad baja indica movimiento suave y consistente.

#### Estimators

Multiple estimators are provided, each with different properties:

| Estimator | Formula | Property |
|-----------|---------|----------|
| Simple realized volatility | $\hat{\sigma} = \text{std}(\lVert v(t_{i+1}) - v(t_i) \rVert)$ | Overall trajectory roughness |
| Per-dimension volatility | $\hat{\sigma}_d = \text{std}(v_d(t_{i+1}) - v_d(t_i))$ for each dimension $d$ | Identifies which dimensions are volatile |
| Annualized/normalized | $\hat{\sigma}_{\text{ann}} = \hat{\sigma} \cdot \sqrt{T / \Delta t}$ | Comparable across different sampling frequencies |
| Parkinson (high-low) | Uses max/min within windows | More efficient estimator when available |

#### Rust Type

```rust
/// Realized volatility estimates for an entity's embedding trajectory.
pub struct RealizedVolatility {
    /// Entity under analysis.
    pub entity_id: u64,
    /// Time window of analysis.
    pub time_range: (i64, i64),
    /// Overall scalar volatility (std of increment magnitudes).
    pub scalar_volatility: f64,
    /// Per-dimension volatility vector.
    pub per_dimension_volatility: Vec<f64>,
    /// Volatility of volatility — meta-stability measure.
    /// High vol-of-vol indicates volatility itself is unstable.
    pub volatility_of_volatility: f64,
    /// Number of increments used.
    pub n_increments: usize,
    /// Annualized volatility (if time units are specified).
    pub annualized_volatility: Option<f64>,
}
```

#### API

```
GET /v1/stochastic/entities/{id}/volatility?from=&to=
```

**Response:**
```json
{
  "entity_id": 42,
  "time_range": [1640000000, 1700000000],
  "scalar_volatility": 0.0087,
  "per_dimension_volatility": [0.012, 0.008, 0.011, "..."],
  "volatility_of_volatility": 0.003,
  "n_increments": 365
}
```

### 2.3 GARCH Volatility Model

#### Context

La volatilidad no es constante en el tiempo. Un fenmeno bien documentado en finanzas (y observable en trayectorias de embeddings) es el **clustering de volatilidad**: periodos de alta volatilidad tienden a seguir a periodos de alta volatilidad, y viceversa.

The GARCH(1,1) model (Generalized Autoregressive Conditional Heteroskedasticity) captures this clustering:

$$
\sigma^2(t) = \omega + \alpha \cdot \varepsilon^2(t-1) + \beta \cdot \sigma^2(t-1)
$$

donde:
- $\omega$ es el peso de varianza de largo plazo
- $\alpha$ mide la reaccin a shocks recientes (innovation coefficient)
- $\beta$ mide la persistencia de la volatilidad (lag coefficient)
- $\varepsilon(t) = \Delta v(t) / \sigma(t)$ son los residuos estandarizados

#### Interpretation

La **persistencia** $\alpha + \beta$ es el parmetro ms informativo:

| Persistence $(\alpha + \beta)$ | Interpretation |
|-------------------------------|----------------|
| $> 0.95$ | Integrated GARCH — volatility shocks are nearly permanent |
| $0.8 - 0.95$ | High persistence — shocks decay slowly |
| $0.5 - 0.8$ | Moderate — shocks decay at medium speed |
| $< 0.5$ | Low persistence — volatility reverts quickly to long-run level |

The **half-life** of a volatility shock is:

$$
h = \frac{\ln 2}{\ln(\alpha + \beta)^{-1}} = \frac{-\ln 2}{\ln(\alpha + \beta)}
$$

#### Rust Type

```rust
/// GARCH(1,1) model estimated on embedding trajectory increments.
pub struct GarchEstimate {
    /// Entity under analysis.
    pub entity_id: u64,
    /// Time window of analysis.
    pub time_range: (i64, i64),
    /// Long-run variance weight (intercept).
    pub omega: f64,
    /// Reaction coefficient — sensitivity to recent shocks.
    pub alpha: f64,
    /// Persistence coefficient — memory of past volatility.
    pub beta: f64,
    /// Total persistence (alpha + beta). Must be < 1 for stationarity.
    pub persistence: f64,
    /// Current conditional volatility estimate sigma(t_now).
    pub current_vol: f64,
    /// One-step-ahead forecast sigma(t_now + 1).
    pub forecast_vol: f64,
    /// Half-life of volatility shocks (in time units).
    pub half_life: f64,
    /// Long-run (unconditional) volatility: sqrt(omega / (1 - alpha - beta)).
    pub long_run_vol: f64,
    /// Log-likelihood of the fitted model.
    pub log_likelihood: f64,
    /// AIC for model comparison.
    pub aic: f64,
}
```

#### Estimation

El modelo se estima por mxima verosimilitud (MLE). Los incrementos escalares $r_t = \lVert v(t+1) - v(t) \rVert$ se modelan como $r_t = \sigma_t \cdot z_t$ con $z_t \sim \mathcal{N}(0,1)$. La verosimilitud gaussiana condicional es:

$$
\ell(\theta) = -\frac{1}{2} \sum_{t=1}^{T} \left[ \ln(2\pi) + \ln(\sigma_t^2) + \frac{r_t^2}{\sigma_t^2} \right]
$$

Optimized with L-BFGS-B subject to constraints $\omega > 0$, $\alpha \geq 0$, $\beta \geq 0$, $\alpha + \beta < 1$.

#### API

```
GET /v1/stochastic/entities/{id}/garch?from=&to=
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `id` | `u64` | Yes | — | Entity ID |
| `from` | `i64` | No | first_seen | Start timestamp |
| `to` | `i64` | No | last_seen | End timestamp |
| `forecast_steps` | `usize` | No | `1` | Number of forward volatility forecasts |

**Response:**
```json
{
  "entity_id": 42,
  "omega": 0.000012,
  "alpha": 0.08,
  "beta": 0.89,
  "persistence": 0.97,
  "current_vol": 0.012,
  "forecast_vol": 0.0115,
  "half_life": 23.1,
  "long_run_vol": 0.0063,
  "log_likelihood": 1234.5,
  "aic": -2465.0
}
```

### 2.4 Mean Reversion Test

#### Context

Una pregunta fundamental sobre cualquier trayectoria de embedding: **revierte a un equilibrio, o vaga libremente (random walk), o tiene una tendencia persistente?**

This classification has profound implications:

| Classification | Meaning | Action |
|---------------|---------|--------|
| Mean-reverting | Current position will revert to an equilibrium | Deviations are temporary — the entity "wants" to return |
| Random walk | No equilibrium — position is unpredictable | Past trajectory does not inform future position |
| Trending | Persistent directional movement | Momentum — current direction likely to continue |

#### Methods

Two complementary tests provide robust classification:

**Augmented Dickey-Fuller (ADF) test:**
- $H_0$: unit root (random walk)
- Rejection $\Rightarrow$ mean-reverting (stationary)
- Test regression: $\Delta v_t = \alpha + \beta v_{t-1} + \sum_{j=1}^{p} \gamma_j \Delta v_{t-j} + \varepsilon_t$
- Test statistic on $\beta$. Critical values from Dickey-Fuller distribution (non-standard).

**KPSS test (Kwiatkowski-Phillips-Schmidt-Shin):**
- $H_0$: stationary (mean-reverting)
- Rejection $\Rightarrow$ unit root or trend
- Complementary to ADF: using both provides a 2x2 classification matrix

**Combined interpretation:**

| ADF rejects? | KPSS rejects? | Classification |
|-------------|--------------|----------------|
| Yes | No | **Mean-Reverting** (stationary) |
| No | Yes | **Random Walk** (unit root) |
| Yes | Yes | **Trending** (trend-stationary) |
| No | No | **Inconclusive** |

#### Ornstein-Uhlenbeck Parameters

Cuando se detecta mean reversion, estimamos los parmetros del proceso de Ornstein-Uhlenbeck:

$$
dv = \theta(\mu - v) \, dt + \sigma \, dW
$$

- $\theta$ = velocidad de reversin a la media
- $\mu$ = posicin de equilibrio (attractor)
- $\sigma$ = difusin (volatilidad residual)
- Half-life = $\ln(2) / \theta$ — cunto tiempo tarda en revertir la mitad de la desviacin

#### Rust Types

```rust
/// Complete mean reversion analysis for an entity's trajectory.
pub struct MeanReversionAnalysis {
    /// Entity under analysis.
    pub entity_id: u64,
    /// Time window of analysis.
    pub time_range: (i64, i64),
    /// Classification based on ADF + KPSS combination.
    pub classification: TrajectoryClass,
    /// ADF test statistic.
    pub adf_statistic: f64,
    /// ADF p-value (interpolated from Dickey-Fuller tables).
    pub adf_p_value: f64,
    /// KPSS test statistic.
    pub kpss_statistic: f64,
    /// KPSS p-value.
    pub kpss_p_value: f64,
    /// Ornstein-Uhlenbeck parameters (populated if mean-reverting).
    pub ou_params: Option<OrnsteinUhlenbeckParams>,
}

/// Classification of trajectory dynamics based on stationarity tests.
pub enum TrajectoryClass {
    /// ADF rejects, KPSS doesn't -> stationary process, reverts to equilibrium.
    MeanReverting,
    /// ADF doesn't reject, KPSS rejects -> unit root, no equilibrium.
    RandomWalk,
    /// Both reject -> deterministic trend with stationary residuals.
    Trending,
    /// Neither rejects -> insufficient evidence for classification.
    Inconclusive,
}

/// Parameters of a fitted Ornstein-Uhlenbeck process.
pub struct OrnsteinUhlenbeckParams {
    /// Mean-reversion speed. Higher theta = faster reversion.
    pub theta: f64,
    /// Equilibrium position (long-run mean).
    pub mu: Vec<f32>,
    /// Diffusion coefficient (residual volatility).
    pub sigma: f64,
    /// Half-life of mean reversion in time units: ln(2)/theta.
    pub half_life: f64,
    /// R-squared of the OU regression.
    pub r_squared: f64,
}
```

#### API

```
GET /v1/stochastic/entities/{id}/mean-reversion?from=&to=
```

**Response:**
```json
{
  "entity_id": 42,
  "classification": "MeanReverting",
  "adf_statistic": -3.87,
  "adf_p_value": 0.002,
  "kpss_statistic": 0.14,
  "kpss_p_value": 0.52,
  "ou_params": {
    "theta": 0.03,
    "mu": [0.12, -0.05, 0.33, "..."],
    "sigma": 0.008,
    "half_life": 23.1,
    "r_squared": 0.67
  }
}
```

### 2.5 Hurst Exponent

#### Context

El exponente de Hurst $H$ mide la "rugosidad" o "memoria" de una trayectoria. Es una medida fundamental que distingue tres regmenes de comportamiento:

| Hurst Value | Classification | Meaning |
|-------------|---------------|---------|
| $H = 0.5$ | Random (Brownian) | Pure random walk — no memory, increments are i.i.d. |
| $H > 0.5$ | Persistent (trending) | Momentum — past direction predicts future direction |
| $H < 0.5$ | Anti-persistent (rough) | Mean-reverting at small scales — past direction predicts *reversal* |

Un hallazgo notable en finanzas: la volatilidad realizada tiene $H \approx 0.1$ (muy rugosa), lo que motiv la teora de volatilidad rugosa (Gatheral et al., 2018). Las trayectorias de embeddings pueden exhibir rugosidad similar, lo que tendra implicaciones para la prediccin y el modelado.

#### Method: Detrended Fluctuation Analysis (DFA)

1. Compute the cumulative deviation from mean: $Y(i) = \sum_{k=1}^{i} (x_k - \bar{x})$
2. Divide $Y$ into windows of size $s$
3. In each window, fit a polynomial trend and compute residual variance $F(s)$
4. The Hurst exponent satisfies: $F(s) \sim s^H$
5. Estimate $H$ from the slope of $\log F(s)$ vs $\log s$

Alternative: R/S analysis (rescaled range), which is simpler but less robust to trends.

#### Rust Type

```rust
/// Hurst exponent estimation for an entity's trajectory.
pub struct HurstExponent {
    /// Entity under analysis.
    pub entity_id: u64,
    /// Time window of analysis.
    pub time_range: (i64, i64),
    /// Estimated Hurst exponent.
    pub hurst: f64,
    /// Classification based on H value.
    pub classification: HurstClass,
    /// 95% confidence interval for H.
    pub confidence_interval: (f64, f64),
    /// Method used for estimation.
    pub method: HurstMethod,
    /// R-squared of the log-log regression (goodness of fit).
    pub r_squared: f64,
}

/// Classification of trajectory memory based on Hurst exponent.
pub enum HurstClass {
    /// H < 0.5 — anti-persistent, mean-reverting at small scales.
    AntiPersistent,
    /// H ~ 0.5 — Brownian, no memory.
    Random,
    /// H > 0.5 — persistent, trending/momentum.
    Persistent,
}

/// Method used to estimate the Hurst exponent.
pub enum HurstMethod {
    /// Detrended Fluctuation Analysis (recommended).
    DFA,
    /// Rescaled Range analysis (classic, simpler).
    RescaledRange,
}
```

#### API

```
GET /v1/stochastic/entities/{id}/hurst?from=&to=&method=dfa
```

**Response:**
```json
{
  "entity_id": 42,
  "hurst": 0.63,
  "classification": "Persistent",
  "confidence_interval": [0.58, 0.68],
  "method": "DFA",
  "r_squared": 0.97
}
```

### 2.6 Stationarity Classification (Unified Report)

#### Context

Las secciones anteriores proporcionan piezas individuales del rompecabezas. La Stationarity Classification las combina en un **reporte unificado** que clasifica el proceso estocstico de la trayectoria de una entidad.

#### Combined Classification Logic

```rust
/// Unified stochastic process report for a single entity.
pub struct StationarityReport {
    /// Entity under analysis.
    pub entity_id: u64,
    /// Time window of analysis.
    pub time_range: (i64, i64),
    /// Drift significance analysis.
    pub drift: DriftSignificance,
    /// Mean reversion analysis (ADF + KPSS + OU).
    pub mean_reversion: MeanReversionAnalysis,
    /// Hurst exponent analysis.
    pub hurst: HurstExponent,
    /// Realized volatility.
    pub volatility: RealizedVolatility,
    /// GARCH volatility model.
    pub garch: GarchEstimate,
    /// Overall process classification derived from all components.
    pub overall_classification: ProcessClassification,
    /// Human-readable summary of the classification.
    pub summary: String,
}

/// High-level classification of the stochastic process governing a trajectory.
pub enum ProcessClassification {
    /// Mean-reverting, low volatility, H < 0.5.
    /// The entity fluctuates around a stable equilibrium.
    StableEquilibrium,
    /// No significant drift, no reversion, H ~ 0.5.
    /// The entity executes a random walk — unpredictable.
    RandomWalk,
    /// Significant drift, H > 0.5.
    /// The entity is moving persistently in a direction.
    TrendingWithMomentum,
    /// Mean-reverting but high GARCH persistence.
    /// The entity cycles around equilibrium with episodic volatility bursts.
    VolatileCycling,
    /// Stochastic characteristics changed during the analysis window.
    /// The entity transitioned between regimes — needs regime detection.
    RegimeTransition,
}
```

#### Classification Decision Tree

```
1. Is drift significant? (p < 0.05)
   ├─ Yes: Is trajectory trending? (ADF+KPSS → Trending, Hurst > 0.5)
   │       ├─ Yes → TrendingWithMomentum
   │       └─ No:  Is it mean-reverting despite drift?
   │               ├─ Yes → StableEquilibrium (drift exists but is small relative to reversion)
   │               └─ No  → Check for regime change
   └─ No:  Is trajectory mean-reverting? (ADF rejects)
           ├─ Yes: Is GARCH persistence high? (α + β > 0.9)
           │       ├─ Yes → VolatileCycling
           │       └─ No  → StableEquilibrium
           └─ No:  Is Hurst ≈ 0.5?
                   ├─ Yes → RandomWalk
                   └─ No  → RegimeTransition (mixed signals suggest regime change)
```

#### API

```
GET /v1/stochastic/entities/{id}/classification?from=&to=
```

**Response:**
```json
{
  "entity_id": 42,
  "overall_classification": "StableEquilibrium",
  "summary": "Entity 42 exhibits mean-reverting behavior (ADF p=0.002) with low volatility (sigma=0.008) and anti-persistent dynamics (H=0.42). The trajectory fluctuates around a stable equilibrium with half-life of 23 time units.",
  "drift": { "...": "..." },
  "mean_reversion": { "...": "..." },
  "hurst": { "...": "..." },
  "volatility": { "...": "..." },
  "garch": { "...": "..." }
}
```

---

## 3. Path Signatures

### 3.1 Mathematical Foundation

Las **path signatures** (firmas de trayectoria) son una herramienta de la teora de caminos rugosos (rough path theory) que proporciona un **descriptor universal** de la forma de una trayectoria. Cualquier funcin continua de una trayectoria puede aproximarse como una funcin lineal de su signature.

Formally, the path signature of $X: [0,T] \to \mathbb{R}^d$ is the sequence of iterated integrals:

$$
S(X)^{i_1, \ldots, i_k} = \int_{0 < u_1 < \cdots < u_k < T} dX^{i_1}(u_1) \otimes \cdots \otimes dX^{i_k}(u_k)
$$

The full signature is an infinite series:

$$
S(X) = (1, S^1(X), S^{1,1}(X), S^{1,2}(X), \ldots)
$$

#### Key Properties

| Property | Description | Implication for CVX |
|----------|-------------|---------------------|
| **Universality** | Any continuous function on paths $\approx$ linear function of signature | Signatures are sufficient statistics for trajectory classification |
| **Reparametrization invariance** | Speed doesn't matter, only shape | Robust to different sampling rates |
| **Hierarchical structure** | Level 1 = displacement; level 2 = "signed area" (encodes rotation/volatility); level 3+ = complex patterns | Can truncate at desired detail level |
| **Uniqueness** | (Up to tree-like equivalences) the signature uniquely determines the path | No information loss (at infinite depth) |
| **Multiplicativity** | $S(X|_{[0,T]}) = S(X|_{[0,s]}) \otimes S(X|_{[s,T]})$ (Chen's identity) | Can update incrementally as new data arrives |

#### Intuition by Level

- **Level 1:** $S^i(X) = X^i(T) - X^i(0)$ — net displacement in each dimension. "Where did the entity end up relative to where it started?"
- **Level 2:** $S^{i,j}(X) = \int_0^T (X^i(t) - X^i(0)) \, dX^j(t)$ — signed area between dimensions $i$ and $j$. Encodes rotation, correlation structure, and quadratic variation (volatility).
- **Level 3+:** Higher-order interactions capturing complex path geometry. Analogous to higher moments of a distribution.

### 3.2 Practical Computation for CVX

#### The Dimensionality Challenge

For $D=768$ dimensional embeddings, computing the full signature is intractable:

| Depth $k$ | Number of terms $d^k$ (with $d=768$) | Feasible? |
|-----------|--------------------------------------|-----------|
| 1 | 768 | Yes |
| 2 | 589,824 | Marginal |
| 3 | 452,984,832 | No |

#### Solution: PCA Reduction + Truncated Signature

1. **Project** the trajectory to principal components: $d_{\text{reduced}} = 5{-}10$
2. **Compute** truncated signature at depth 2-3 on the reduced trajectory
3. **Result:** fixed-dimensional vector of manageable size

| $d_{\text{reduced}}$ | Depth | Signature dimensions | Log-signature dimensions |
|----------------------|-------|---------------------|-------------------------|
| 5 | 2 | 5 + 25 = 30 | 5 + 10 = 15 |
| 5 | 3 | 5 + 25 + 125 = 155 | 5 + 10 + 20 = 35 |
| 10 | 2 | 10 + 100 = 110 | 10 + 45 = 55 |
| 10 | 3 | 10 + 100 + 1000 = 1110 | 10 + 45 + 120 = 175 |

The **log-signature** is a more compact alternative that removes redundant terms via the Baker-Campbell-Hausdorff formula. It contains the same information in fewer dimensions.

#### Time Augmentation

Adding time as an extra dimension ($\tilde{X}(t) = (t, X(t))$) breaks reparametrization invariance but captures speed information. Useful when the *timing* of movement matters, not just its shape.

#### Rust Types

```rust
/// Computed path signature for an entity's trajectory.
pub struct PathSignature {
    /// Entity whose trajectory was analyzed.
    pub entity_id: u64,
    /// Time window of the trajectory.
    pub time_range: (i64, i64),
    /// Truncation depth used.
    pub depth: usize,
    /// Number of PCA dimensions used before computing signature.
    pub reduced_dims: usize,
    /// The signature vector (concatenation of all levels).
    pub signature: Vec<f64>,
    /// The log-signature (more compact alternative).
    pub log_signature: Vec<f64>,
    /// Variance explained by each PCA component.
    pub pca_variance_explained: Vec<f64>,
    /// Total variance explained by the PCA projection.
    pub total_variance_explained: f64,
}

/// Configuration for signature computation.
pub struct SignatureConfig {
    /// Truncation depth (2-4). Higher = more expressive but larger.
    pub depth: usize,
    /// Number of PCA dimensions before signature computation (5-10).
    pub reduced_dims: usize,
    /// Whether to return log-signature instead of full signature.
    pub use_log_signature: bool,
    /// Whether to add time as an extra dimension (breaks reparametrization invariance).
    pub time_augmentation: bool,
    /// Minimum variance explained by PCA (default 0.9).
    pub min_variance_explained: f64,
}

impl Default for SignatureConfig {
    fn default() -> Self {
        Self {
            depth: 3,
            reduced_dims: 5,
            use_log_signature: true,
            time_augmentation: false,
            min_variance_explained: 0.9,
        }
    }
}
```

#### Computation Algorithm

```rust
/// Compute the truncated path signature.
///
/// Steps:
/// 1. Extract trajectory: [(t1, v1), (t2, v2), ...] for entity in time_range
/// 2. Stack vectors into matrix X of shape (T, D)
/// 3. Fit PCA on X, project to X_reduced of shape (T, d_reduced)
/// 4. Optionally prepend time column: X_aug of shape (T, d_reduced + 1)
/// 5. Compute iterated integrals up to specified depth
/// 6. (Optional) Convert to log-signature via BCH formula
pub fn compute_path_signature(
    trajectory: &[(i64, &[f32])],
    config: &SignatureConfig,
) -> PathSignature {
    todo!()
}
```

#### API

```
GET /v1/stochastic/entities/{id}/signature?from=&to=&depth=3&dims=5&log=true&time_aug=false
```

**Response:**
```json
{
  "entity_id": 42,
  "depth": 3,
  "reduced_dims": 5,
  "signature_dims": 155,
  "log_signature_dims": 35,
  "log_signature": [0.12, -0.03, 0.45, "..."],
  "pca_variance_explained": [0.42, 0.21, 0.13, 0.08, 0.06],
  "total_variance_explained": 0.90
}
```

### 3.3 Signature-Based Trajectory Similarity

#### The Killer Application

This is the most transformative capability that path signatures bring to CVX: **trajectory similarity search**. Traditional kNN retrieves entities with similar *positions* at a given time. Signature-based kNN retrieves entities with similar *evolution patterns*.

```
Traditional CVX:  kNN(query_vector, timestamp) → similar positions at time t
Signature CVX:    kNN(query_signature, signature_space) → similar TRAJECTORIES
```

This enables a fundamentally new type of query: **"Find entities that evolved the same way as this one."**

| Query Type | What It Finds | Example |
|-----------|---------------|---------|
| Position kNN | Entities near entity X at time t | "What concepts are similar to 'AI' right now?" |
| Trajectory kNN | Entities that followed a similar path | "What concepts evolved like 'AI' did over the last year?" |

Two entities can be far apart in embedding space but have identical signatures — they underwent the same *type* of evolution (same direction, speed pattern, volatility structure) in different regions of the space. Conversely, two nearby entities might have very different signatures.

#### Query Type

```rust
/// Query for trajectory similarity using path signatures.
pub struct TrajectoryQuery {
    /// Reference entity whose trajectory pattern we want to match.
    pub reference_entity: u64,
    /// Time range of the reference trajectory.
    pub time_range: (i64, i64),
    /// Number of similar trajectories to return.
    pub k: usize,
    /// Signature configuration.
    pub signature_config: SignatureConfig,
    /// Distance metric for signature comparison.
    pub metric: SignatureMetric,
}

/// Distance metric for comparing signatures.
pub enum SignatureMetric {
    /// Euclidean distance in signature space (default).
    Euclidean,
    /// Cosine similarity in signature space.
    Cosine,
    /// Signature kernel (Kiraly & Oberhauser, 2019) — a proper kernel
    /// on unparameterized paths. More theoretically grounded but expensive.
    SignatureKernel { truncation: usize },
}

/// Result of trajectory similarity search.
pub struct TrajectorySimilarityResult {
    /// The reference entity and its trajectory.
    pub reference: TrajectoryDescriptor,
    /// K most similar trajectories, sorted by distance.
    pub matches: Vec<TrajectoryMatch>,
}

pub struct TrajectoryMatch {
    pub entity_id: u64,
    pub time_range: (i64, i64),
    pub signature_distance: f64,
    pub trajectory_class: ProcessClassification,
}

pub struct TrajectoryDescriptor {
    pub entity_id: u64,
    pub time_range: (i64, i64),
    pub signature: PathSignature,
    pub trajectory_class: ProcessClassification,
}
```

#### API

```
POST /v1/query
{
  "type": "trajectory_similarity",
  "entity_id": 42,
  "time_range": [1640000000, 1700000000],
  "k": 10,
  "signature_config": {
    "depth": 3,
    "reduced_dims": 5,
    "use_log_signature": true
  }
}
```

**Response:**
```json
{
  "reference": {
    "entity_id": 42,
    "trajectory_class": "TrendingWithMomentum"
  },
  "matches": [
    { "entity_id": 107, "signature_distance": 0.034, "trajectory_class": "TrendingWithMomentum" },
    { "entity_id": 293, "signature_distance": 0.051, "trajectory_class": "TrendingWithMomentum" },
    { "entity_id": 58,  "signature_distance": 0.089, "trajectory_class": "VolatileCycling" }
  ]
}
```

### 3.4 Signature as Materialized View

Las signatures pueden pre-computarse y cachearse como vistas materializadas, permitiendo bsquedas rpidas sin recomputacin:

```toml
[[views]]
name = "entity_signatures"
query = "path_signature(entity_id, t_first, t_last, depth=3, dims=5)"
refresh = "on_ingest"
index = "hnsw"  # Build HNSW index on signature vectors for fast kNN

[[views]]
name = "monthly_signatures"
query = "path_signature(entity_id, month_start, month_end, depth=2, dims=5)"
refresh = "on_schedule"
schedule = "0 0 1 * *"  # First day of each month
```

#### Incremental Updates via Chen's Identity

Thanks to Chen's identity ($S(X|_{[0,T]}) = S(X|_{[0,s]}) \otimes S(X|_{[s,T]})$), signatures can be updated incrementally when new data arrives, without recomputing from scratch. This makes `refresh = "on_ingest"` efficient.

### 3.5 References

- Lyons, T. (1998). "Differential equations driven by rough signals." *Revista Matemtica Iberoamericana*, 14(2), 215-310.
- Chevyrev, I., & Kormilitzin, A. (2016). "A Primer on the Signature Method in Machine Learning." *arXiv:1603.03788*.
- Kidger, P., & Lyons, T. (2021). "Signatory: differentiable computations of the signature and logsignature transforms, on both CPU and GPU." *ICLR 2021*.
- Arribas, I. P., Salvi, C., & Sherbet, L. (2020). "Sig-SDEs model for quantitative finance." *arXiv:2006.00218*.
- Ni, H., Szpruch, L., Wiese, M., Liao, S., & Xiao, B. (2021). "Sig-Wasserstein GANs for time series generation." *ICAIF 2021*.
- Kiraly, F. J., & Oberhauser, H. (2019). "Kernels for sequentially ordered data." *JMLR*, 20(31), 1-45.

---

## 4. Regime Detection (Markov Switching)

### 4.1 Concept

La deteccin de rgimen complementa al BOCPD (Bayesian Online Change Point Detection) que CVX ya planea implementar. Mientras BOCPD detecta **CUNDO** ocurre un cambio, los Markov Switching Models caracterizan **QU** rgimen ocupa la entidad.

| Approach | Answers | Output |
|----------|---------|--------|
| BOCPD | "When did the trajectory change?" | Change point timestamps + severity |
| Regime Detection | "What kind of behavior is the entity exhibiting?" | Regime labels + transition probabilities |

Los regmenes para trayectorias de embeddings son patrones dinmicos recurrentes:

| Regime | Drift | Volatility | Hurst | Typical Meaning |
|--------|-------|-----------|-------|-----------------|
| **Stable** | Low | Low | $< 0.5$ | Entity is in equilibrium, fluctuating around a fixed position |
| **Transitioning** | High | Moderate | $> 0.5$ | Entity is moving purposefully to a new position |
| **Turbulent** | Low | High | $\approx 0.5$ | Entity is being buffeted by noise, no clear direction |
| **Accelerating** | Increasing | Increasing | $> 0.5$ | Entity is gaining momentum — often precedes a regime change |

### 4.2 Hidden Markov Model for Trajectories

The Hidden Markov Model (HMM) treats the regime as a latent (hidden) state that generates observed trajectory features. The model is defined by:

- **States:** $K$ regimes (typically $K = 2{-}5$)
- **Transition matrix:** $A_{ij} = P(\text{regime}_j | \text{regime}_i)$ — probability of switching
- **Emission model:** per-regime parameters (drift, volatility, etc.)
- **Inference:** Baum-Welch (EM) for parameter estimation, Viterbi for most likely state sequence, forward-backward for state probabilities

#### Rust Types

```rust
/// Complete regime detection result for an entity.
pub struct RegimeDetection {
    /// Entity under analysis.
    pub entity_id: u64,
    /// Time window of analysis.
    pub time_range: (i64, i64),
    /// Number of regimes identified.
    pub n_regimes: usize,
    /// Current most likely regime.
    pub current_regime: usize,
    /// Probability distribution over regimes at current time.
    pub regime_probabilities: Vec<f64>,
    /// K x K transition probability matrix.
    pub transition_matrix: Vec<Vec<f64>>,
    /// Stochastic parameters characterizing each regime.
    pub regime_params: Vec<RegimeParams>,
    /// Temporal history of regime assignments.
    pub regime_history: Vec<RegimeSegment>,
    /// Model selection criterion.
    pub bic: f64,
    /// Log-likelihood of the fitted model.
    pub log_likelihood: f64,
}

/// Stochastic parameters characterizing a single regime.
pub struct RegimeParams {
    /// Human-readable label (auto-generated from parameters).
    pub label: String,
    /// Mean drift magnitude in this regime.
    pub mean_drift_magnitude: f64,
    /// Mean volatility in this regime.
    pub mean_volatility: f64,
    /// Hurst exponent in this regime.
    pub hurst_exponent: f64,
    /// Mean-reversion speed (if applicable).
    pub mean_reversion_speed: Option<f64>,
    /// Stationary probability of being in this regime.
    pub stationary_probability: f64,
    /// Expected duration in this regime (from transition matrix).
    pub expected_duration: f64,
}

/// A contiguous time segment assigned to a single regime.
pub struct RegimeSegment {
    /// Start timestamp of this segment.
    pub from_timestamp: i64,
    /// End timestamp of this segment.
    pub to_timestamp: i64,
    /// Assigned regime index.
    pub regime: usize,
    /// Average posterior probability of this regime assignment.
    pub confidence: f64,
}
```

### 4.3 Regime Selection

Cmo elegir el nmero de regmenes $K$? Dos enfoques:

1. **User-specified:** El usuario indica $K$ basndose en conocimiento del dominio (e.g., "risk-on/risk-off" $\Rightarrow K=2$)
2. **Auto-detection:** Fit models for $K = 2, 3, \ldots, K_{\max}$ and select by BIC (Bayesian Information Criterion)

```rust
/// Configuration for regime detection.
pub struct RegimeConfig {
    /// Number of regimes. If None, auto-detect via BIC.
    pub n_regimes: Option<usize>,
    /// Maximum regimes to consider in auto-detection.
    pub max_regimes: usize, // default: 5
    /// Features to use for regime classification.
    pub features: RegimeFeatures,
    /// Minimum segment length (avoid too-frequent switching).
    pub min_segment_length: usize, // default: 5 time steps
}

/// Which trajectory features to use for regime classification.
pub struct RegimeFeatures {
    pub use_drift_magnitude: bool,  // default: true
    pub use_volatility: bool,       // default: true
    pub use_hurst: bool,            // default: false (expensive per window)
    pub use_direction: bool,        // default: false
}
```

### 4.4 Regime-Aware Prediction

El Neural ODE/SDE puede condicionar en el rgimen actual, usando diferentes parmetros por rgimen:

$$
\text{In regime } k: \quad dv = f_{\theta_k}(v, t) \, dt + g_{\phi_k}(v, t) \, dW
$$

Esto permite predicciones que respetan la dinmica del rgimen actual. Si el entity est en rgimen "Stable", la prediccin ser de bajo drift. Si est en rgimen "Transitioning", la prediccin capturar el momentum.

### 4.5 Regime-Based Similarity

Un nuevo tipo de query: **"Find entities currently in the same regime."** Esto identifica entidades co-movindose, tiles para:

- Portfolio construction (entities in same regime may be correlated)
- Anomaly detection (entity in different regime from its peers)
- Cohort analysis (group entities by dynamic behavior, not static position)

```rust
/// Query for entities in a specific regime.
pub struct RegimeQuery {
    /// Target regime index (from a reference entity's regime model).
    pub regime: usize,
    /// Reference entity (whose regime model defines the regime).
    pub reference_entity: u64,
    /// Time at which to evaluate regime membership.
    pub timestamp: i64,
    /// Maximum results.
    pub k: usize,
    /// Minimum probability of being in the target regime.
    pub min_probability: f64, // default: 0.5
}
```

#### API

```
GET /v1/stochastic/entities/{id}/regime?n_regimes=3&from=&to=
```

**Response:**
```json
{
  "entity_id": 42,
  "n_regimes": 3,
  "current_regime": 1,
  "regime_probabilities": [0.05, 0.90, 0.05],
  "transition_matrix": [
    [0.95, 0.04, 0.01],
    [0.03, 0.94, 0.03],
    [0.02, 0.05, 0.93]
  ],
  "regime_params": [
    { "label": "Stable", "mean_drift_magnitude": 0.002, "mean_volatility": 0.004, "expected_duration": 20.0 },
    { "label": "Transitioning", "mean_drift_magnitude": 0.025, "mean_volatility": 0.012, "expected_duration": 16.7 },
    { "label": "Turbulent", "mean_drift_magnitude": 0.005, "mean_volatility": 0.031, "expected_duration": 14.3 }
  ],
  "regime_history": [
    { "from_timestamp": 1640000000, "to_timestamp": 1650000000, "regime": 0, "confidence": 0.92 },
    { "from_timestamp": 1650000000, "to_timestamp": 1680000000, "regime": 1, "confidence": 0.88 },
    { "from_timestamp": 1680000000, "to_timestamp": 1700000000, "regime": 0, "confidence": 0.95 }
  ],
  "bic": -4521.3
}
```

### 4.6 References

- Hamilton, J. D. (1989). "A new approach to the economic analysis of nonstationary time series and the business cycle." *Econometrica*, 57(2), 357-384.
- Kim, C. J., & Nelson, C. R. (1999). *State-Space Models with Regime Switching.* MIT Press.
- Nystrup, P., Madsen, H., & Lindstrm, E. (2020). "Learning hidden Markov models with persistent states by penalizing jumps." *Expert Systems with Applications*, 150, 113307.

---

## 5. Neural SDE Extension

### 5.1 From Neural ODE to Neural SDE

#### Current State

CVX's Temporal ML Spec defines a Neural ODE predictor:

$$
\frac{dz}{dt} = f_\theta(z, t)
$$

This produces a **deterministic** trajectory prediction: given initial conditions, there is exactly one predicted future. But embedding trajectories are inherently stochastic — the same entity in the same state could evolve differently depending on unpredictable factors (new data, external events, model updates).

#### Extension

La extensin natural es el Neural SDE:

$$
dz = f_\theta(z, t) \, dt + g_\phi(z, t) \, dW(t)
$$

This produces a **distribution** of predicted futures. Multiple forward passes with different Brownian motion samples generate a fan of possible trajectories, providing:

- **Point estimate:** mean of sampled trajectories
- **Uncertainty quantification:** width of the trajectory fan
- **Tail risk:** probability of extreme deviations
- **Prediction intervals:** per-timestamp confidence bands

### 5.2 Architecture

```rust
/// Neural SDE predictor extending the Neural ODE with learned diffusion.
pub struct NeuralSdePredictor<B: Backend> {
    /// Drift network f_theta: R^d x R -> R^d.
    /// Learned systematic dynamics.
    pub drift_net: MLP<B>,
    /// Diffusion network g_phi: R^d x R -> R^{d x m}.
    /// Learned volatility structure. m = noise_dim.
    pub diffusion_net: MLP<B>,
    /// SDE solver for forward integration.
    pub solver: SdeSolver,
    /// Noise dimensionality (can be < d for low-rank diffusion).
    pub noise_dim: usize,
}

/// SDE numerical solver.
pub enum SdeSolver {
    /// Euler-Maruyama: simplest SDE solver.
    /// Strong order 0.5, weak order 1.0.
    /// dz = f dt + g dW ≈ f*dt + g*sqrt(dt)*N(0,1)
    EulerMaruyama { dt: f64 },
    /// Milstein: improved SDE solver using Ito-Taylor expansion.
    /// Strong order 1.0, weak order 1.0.
    /// Requires computing dg/dz (Jacobian of diffusion).
    Milstein { dt: f64 },
    /// Stochastic Runge-Kutta (Rssler, 2010).
    /// Higher order but more expensive per step.
    SRK { dt: f64, order: usize },
}
```

### 5.3 Prediction with Uncertainty

```rust
/// Stochastic prediction result: a distribution of future trajectories.
pub struct StochasticPrediction {
    /// Entity for which the prediction was made.
    pub entity_id: u64,
    /// Target timestamp for prediction.
    pub target_timestamp: i64,
    /// Mean trajectory (average of all samples).
    pub mean_trajectory: Vec<Vec<f32>>,
    /// Individual sampled trajectories.
    /// Each is a sequence of vectors from current time to target_timestamp.
    pub sampled_trajectories: Vec<Vec<Vec<f32>>>,
    /// Per-timestamp confidence intervals.
    pub confidence_intervals: Vec<ConfidenceInterval>,
    /// Prediction entropy — overall uncertainty measure.
    /// Higher entropy = less confident prediction.
    pub prediction_entropy: f64,
    /// Probability of exceeding a distance threshold from mean.
    pub tail_probability: Option<f64>,
}

/// Confidence interval at a specific future timestamp.
pub struct ConfidenceInterval {
    /// Timestamp of this interval.
    pub timestamp: i64,
    /// Per-dimension lower bound of the interval.
    pub lower: Vec<f32>,
    /// Per-dimension upper bound of the interval.
    pub upper: Vec<f32>,
    /// Confidence level (e.g., 0.95 for 95% CI).
    pub level: f64,
}
```

### 5.4 Gradient Computation: Adjoint SDE Method

El backpropagation a travs de solvers SDE usa el **mtodo del SDE adjunto** (Li et al., 2020). Este es una extensin estocstica del mtodo ODE adjunto que CVX ya planea usar.

The key insight: instead of backpropagating through the solver steps (which requires storing all intermediate states), we solve an *adjoint SDE* backwards in time. This is memory-efficient (constant memory regardless of number of solver steps) but requires careful handling of the stochastic integral.

The adjoint dynamics:

$$
d\mathbf{a}(t) = -\mathbf{a}(t)^T \frac{\partial f}{\partial z} \, dt - \mathbf{a}(t)^T \frac{\partial g}{\partial z} \, dW(t)
$$

where $\mathbf{a}(t) = \partial L / \partial z(t)$ is the adjoint state (gradient of the loss with respect to the hidden state at time $t$).

```rust
/// Configuration for adjoint SDE gradient computation.
pub struct AdjointSdeConfig {
    /// Whether to use the adjoint method (memory-efficient)
    /// or direct backpropagation (faster but O(T) memory).
    pub use_adjoint: bool,
    /// Number of Brownian motion samples for gradient estimation.
    pub gradient_samples: usize, // default: 1 (stochastic gradient)
    /// Whether to reuse the forward Brownian path in the adjoint
    /// (required for correct gradients — Brownian reconstruction).
    pub brownian_reconstruction: bool, // default: true
}
```

### 5.5 Jump Processes: Neural Jump SDE

Los Neural Jump SDEs (Jia & Benson, 2019) aaden saltos discontinuos a la dinmica:

$$
dv = f_\theta(v,t) \, dt + g_\phi(v,t) \, dW + h_\psi(v,t) \, dN(t)
$$

where $N(t)$ is a counting process (e.g., Poisson process with learned intensity). The jump function $h_\psi$ determines the size and direction of discontinuities.

**Connection to BOCPD:** Los saltos corresponden a change points detectados por BOCPD. Esto **unifica** prediccin y deteccin de change points en un solo modelo:

| Component | Role |
|-----------|------|
| $f_\theta$ (drift) | Smooth evolution between change points |
| $g_\phi$ (diffusion) | Stochastic fluctuation |
| $h_\psi$ (jump) | Discontinuous transitions at change points |
| $\lambda(v, t)$ (intensity) | Probability of a jump (change point) occurring |

```rust
/// Neural Jump SDE predictor — unified prediction + change point detection.
pub struct NeuralJumpSdePredictor<B: Backend> {
    /// Drift network f_theta.
    pub drift_net: MLP<B>,
    /// Diffusion network g_phi.
    pub diffusion_net: MLP<B>,
    /// Jump size network h_psi.
    pub jump_net: MLP<B>,
    /// Jump intensity network lambda(v, t) -> R+.
    pub intensity_net: MLP<B>,
    /// SDE solver with jump handling.
    pub solver: JumpSdeSolver,
}

/// SDE solver that handles both continuous dynamics and jumps.
pub struct JumpSdeSolver {
    /// Base continuous solver.
    pub continuous_solver: SdeSolver,
    /// Method for simulating jump times.
    pub jump_method: JumpSimulation,
}

pub enum JumpSimulation {
    /// Thinning algorithm (Lewis & Shedler, 1979).
    /// Exact but requires upper bound on intensity.
    Thinning { max_intensity: f64 },
    /// Time-discretized: at each dt, sample Bernoulli(lambda * dt).
    /// Approximate but simpler.
    Discretized,
}
```

### 5.6 Training Pipeline

El entrenamiento de Neural SDEs requiere:

1. **Data:** Trajectory pairs $(v(t_0), v(t_T))$ for training prediction
2. **Loss:** KL divergence between predicted and observed path distributions, or maximum likelihood of observed endpoints
3. **Optimization:** Stochastic gradient descent with adjoint SDE gradients
4. **Regularization:** KL penalty on diffusion (prevents collapse to deterministic model or infinite noise)

```rust
/// Training configuration for Neural SDE.
pub struct NeuralSdeTrainingConfig {
    /// Learning rate.
    pub lr: f64,
    /// Number of training epochs.
    pub epochs: usize,
    /// Number of SDE samples per training step (for gradient estimation).
    pub n_samples: usize,
    /// KL regularization weight on diffusion term.
    pub kl_weight: f64,
    /// Batch size (number of entity trajectories per batch).
    pub batch_size: usize,
    /// Whether to use adjoint method for memory efficiency.
    pub adjoint: bool,
}
```

### 5.7 API

```
POST /v1/query
{
  "type": "stochastic_prediction",
  "entity_id": 42,
  "target_timestamp": 1710000000,
  "n_samples": 100,
  "confidence_levels": [0.50, 0.80, 0.95]
}
```

**Response:**
```json
{
  "entity_id": 42,
  "target_timestamp": 1710000000,
  "mean_prediction": [0.12, -0.05, 0.33, "..."],
  "prediction_entropy": 2.34,
  "confidence_intervals": [
    {
      "timestamp": 1710000000,
      "level": 0.95,
      "lower": [0.05, -0.12, 0.25, "..."],
      "upper": [0.19, 0.02, 0.41, "..."]
    }
  ],
  "n_samples": 100
}
```

### 5.8 References

- Li, X., Wong, T. K. L., Chen, R. T. Q., & Duvenaud, D. (2020). "Scalable Gradients for Stochastic Differential Equations." *AISTATS 2020*.
- Kidger, P., Foster, J., Li, X., & Lyons, T. (2021). "Neural SDEs as Infinite-Dimensional GANs." *ICML 2021*.
- Jia, J., & Benson, A. R. (2019). "Neural Jump Stochastic Differential Equations." *NeurIPS 2019*.
- Gierjatowicz, P., Sabate-Vidales, M., Siska, D., Szpruch, L., & Zuric, Z. (2022). "Robust pricing and hedging via Neural SDEs." *Journal of Computational Finance*, 26(3).
- Morrill, J., Salvi, C., Kidger, P., & Foster, J. (2021). "Neural Rough Differential Equations." *arXiv:2009.08295*.
- Rssler, A. (2010). "Runge-Kutta Methods for the Strong Approximation of Solutions of Stochastic Differential Equations." *SIAM Journal on Numerical Analysis*, 48(3), 922-952.

---

## 6. Cross-Entity Stochastic Analysis

Las secciones anteriores se centran en el anlisis de entidades individuales. Esta seccin extiende las herramientas estocsticas a **relaciones entre entidades** — correlacin, co-integracin, causalidad y transporte ptimo.

### 6.1 Dynamic Conditional Correlation (DCC)

#### Context

CVX ya ofrece "cohort divergence" como mtrica de cmo una entidad se aleja de su grupo. Pero la divergencia de cohorte es una medida esttica en cada timestamp. DCC (Engle, 2002) modela las **correlaciones variables en el tiempo** entre los drift rates de dos entidades.

Si dos entidades tenan correlacin 0.8 hace un mes y ahora tienen 0.3, esa decorrelacin es una seal analtica importante — indica que los factores que las movan juntas ya no operan.

#### Rust Type

```rust
/// Dynamic Conditional Correlation between two entities' drift rates.
pub struct DynamicCorrelation {
    /// First entity.
    pub entity_a: u64,
    /// Second entity.
    pub entity_b: u64,
    /// Time window of analysis.
    pub time_range: (i64, i64),
    /// Time series of correlations: (timestamp, correlation).
    pub correlation_series: Vec<(i64, f64)>,
    /// Current (most recent) correlation.
    pub current_correlation: f64,
    /// Average correlation over the window.
    pub mean_correlation: f64,
    /// Change points in the correlation series.
    pub correlation_change_points: Vec<CorrelationChangePoint>,
    /// DCC model parameters.
    pub dcc_params: DccParams,
}

/// DCC model parameters (Engle 2002).
pub struct DccParams {
    /// DCC alpha — reaction of correlation to shocks.
    pub alpha: f64,
    /// DCC beta — persistence of correlation.
    pub beta: f64,
    /// Unconditional correlation.
    pub unconditional_correlation: f64,
}

/// A detected change in the correlation between two entities.
pub struct CorrelationChangePoint {
    /// Timestamp of the change.
    pub timestamp: i64,
    /// Correlation before the change.
    pub correlation_before: f64,
    /// Correlation after the change.
    pub correlation_after: f64,
    /// Severity of the change.
    pub severity: f64,
}
```

#### API

```
GET /v1/stochastic/correlation?entity_a=42&entity_b=43&from=&to=
```

**Response:**
```json
{
  "entity_a": 42,
  "entity_b": 43,
  "current_correlation": 0.34,
  "mean_correlation": 0.72,
  "correlation_change_points": [
    { "timestamp": 1680000000, "correlation_before": 0.81, "correlation_after": 0.35, "severity": 0.46 }
  ]
}
```

### 6.2 Co-integration Test

#### Context

Dos entidades pueden tener trayectorias individuales que son random walks (no estacionarias), pero su **diferencia** (spread) puede ser estacionaria. Esto se llama co-integracin: comparten un equilibrio de largo plazo.

Ejemplo: "AI" y "machine learning" pueden vagar libremente en el espacio de embeddings, pero su distancia relativa se mantiene acotada. Si divergen ms all de lo normal, es probable que converjan de vuelta.

Implications:
- Co-integrated entities are fundamentally linked
- Current divergence from equilibrium is likely temporary
- The speed of convergence (half-life) tells you how quickly they'll reconnect
- Breaking co-integration is a major structural event

#### Rust Type

```rust
/// Co-integration analysis between two entities.
pub struct CointegrationResult {
    /// First entity.
    pub entity_a: u64,
    /// Second entity.
    pub entity_b: u64,
    /// Time window of analysis.
    pub time_range: (i64, i64),
    /// Whether the entities are co-integrated.
    pub is_cointegrated: bool,
    /// Johansen trace statistic.
    pub johansen_statistic: f64,
    /// p-value of the co-integration test.
    pub p_value: f64,
    /// Long-run equilibrium spread (mean distance when co-integrated).
    pub equilibrium_spread: f64,
    /// Current deviation from the equilibrium spread.
    pub current_deviation: f64,
    /// Half-life of convergence back to equilibrium.
    pub half_life_of_convergence: f64,
    /// Number of co-integrating vectors found (Johansen).
    pub n_cointegrating_vectors: usize,
    /// Z-score: how many standard deviations from equilibrium.
    pub z_score: f64,
}
```

Application: "These two concepts are historically linked — their current divergence is likely temporary and will revert within ~15 time units."

#### API

```
GET /v1/stochastic/cointegration?entity_a=42&entity_b=43&from=&to=
```

**Response:**
```json
{
  "entity_a": 42,
  "entity_b": 43,
  "is_cointegrated": true,
  "johansen_statistic": 18.7,
  "p_value": 0.003,
  "equilibrium_spread": 0.45,
  "current_deviation": 0.12,
  "half_life_of_convergence": 14.8,
  "z_score": 1.2
}
```

### 6.3 Granger Causality

#### Context

Does entity A's movement **predict** entity B's movement? This is Granger causality (Granger, 1969): entity A "Granger-causes" entity B if including A's past improves the prediction of B's future, beyond what B's own past provides.

Example: "Does the evolution of 'AI' Granger-cause the evolution of 'automation'?" If yes, changes in the 'AI' embedding precede and predict changes in the 'automation' embedding.

**Important caveat:** Granger causality is *predictive* causality, not true causal inference. It captures lead-lag relationships but cannot distinguish true causation from common causation by an unobserved third factor.

#### Method

For scalar drift magnitudes $x_t$ (entity A) and $y_t$ (entity B):

$$
y_t = \alpha + \sum_{j=1}^{p} \beta_j y_{t-j} + \sum_{j=1}^{p} \gamma_j x_{t-j} + \varepsilon_t
$$

Test $H_0: \gamma_1 = \cdots = \gamma_p = 0$ (A does not Granger-cause B) using an F-test.

#### Rust Type

```rust
/// Granger causality test between two entities.
pub struct GrangerCausalityResult {
    /// Potential cause entity.
    pub cause_entity: u64,
    /// Potential effect entity.
    pub effect_entity: u64,
    /// Time window of analysis.
    pub time_range: (i64, i64),
    /// Lags tested.
    pub lags_tested: Vec<usize>,
    /// F-statistic at each lag.
    pub f_statistics: Vec<f64>,
    /// p-value at each lag.
    pub p_values: Vec<f64>,
    /// Optimal lag (minimum p-value or BIC).
    pub optimal_lag: usize,
    /// Whether A Granger-causes B at the optimal lag.
    pub is_causal: bool,
    /// Bidirectional test: does B also Granger-cause A?
    pub is_bidirectional: Option<bool>,
}
```

#### API

```
GET /v1/stochastic/granger?cause=42&effect=43&max_lag=10
```

**Response:**
```json
{
  "cause_entity": 42,
  "effect_entity": 43,
  "lags_tested": [1, 2, 3, 4, 5],
  "f_statistics": [1.2, 3.8, 8.1, 5.2, 3.1],
  "p_values": [0.27, 0.05, 0.001, 0.02, 0.08],
  "optimal_lag": 3,
  "is_causal": true,
  "is_bidirectional": false
}
```

### 6.4 Wasserstein Neighborhood Drift

#### Context

El drift de una entidad individual mide cmo cambia su posicin. Pero cmo cambia su **vecindario**? Los vecinos ms cercanos en $t_1$ pueden ser completamente diferentes en $t_2$.

The Wasserstein distance (optimal transport distance) measures the "cost" of transforming one distribution into another. Applied to neighborhoods, it quantifies how much the local structure around an entity has changed.

#### Method

1. At time $t_1$, compute the $k$ nearest neighbors of entity $e$: distribution $P_1$
2. At time $t_2$, compute the $k$ nearest neighbors: distribution $P_2$
3. Compute the Wasserstein distance $W_p(P_1, P_2)$

For computational tractability in high dimensions ($d = 768$), we use the **sliced Wasserstein distance**: project distributions onto random 1D directions and average the 1D Wasserstein distances (which have closed-form solutions).

$$
SW_p(P_1, P_2) = \left( \int_{\mathbb{S}^{d-1}} W_p^p(P_1^\theta, P_2^\theta) \, d\theta \right)^{1/p}
$$

where $P^\theta$ is the projection of distribution $P$ onto direction $\theta$.

#### Rust Type

```rust
/// Wasserstein distance measuring how an entity's neighborhood evolved.
pub struct WassersteinDrift {
    /// Entity under analysis.
    pub entity_id: u64,
    /// First timestamp.
    pub t1: i64,
    /// Second timestamp.
    pub t2: i64,
    /// Number of neighbors considered.
    pub k_neighbors: usize,
    /// Wasserstein distance between neighborhood distributions.
    pub wasserstein_distance: f64,
    /// Total transport cost (sum of all pairwise transport).
    pub transport_cost: f64,
    /// Entities that became neighbors between t1 and t2.
    pub neighbors_gained: Vec<u64>,
    /// Entities that were neighbors at t1 but not at t2.
    pub neighbors_lost: Vec<u64>,
    /// Entities that remained neighbors but changed rank.
    pub neighbors_reranked: Vec<(u64, i32)>, // (entity_id, rank_change)
    /// Number of random projections used (for sliced Wasserstein).
    pub n_projections: usize,
}
```

#### API

```
GET /v1/stochastic/entities/{id}/wasserstein-drift?t1=&t2=&k=50&projections=100
```

**Response:**
```json
{
  "entity_id": 42,
  "t1": 1640000000,
  "t2": 1700000000,
  "k_neighbors": 50,
  "wasserstein_distance": 0.234,
  "transport_cost": 11.7,
  "neighbors_gained": [107, 293, 58],
  "neighbors_lost": [15, 88, 201],
  "n_projections": 100
}
```

### 6.5 Cross-Entity Analysis Summary

| Analysis | What It Measures | Time Complexity | Use Case |
|----------|-----------------|-----------------|----------|
| DCC | Time-varying correlation between drift rates | $O(T)$ per pair | Detect decorrelation events |
| Co-integration | Shared long-run equilibrium | $O(T \cdot p)$ per pair | Identify fundamentally linked entities |
| Granger causality | Predictive lead-lag relationship | $O(T \cdot p^2)$ per pair | Discover causal influence chains |
| Wasserstein drift | Neighborhood distribution change | $O(k^2 + k \cdot n_{\text{proj}})$ per entity | Measure local structure evolution |

---

## 7. Quantitative Finance Use Cases

Esta seccin detalla los casos de uso en finanzas cuantitativa — el dominio donde estas herramientas estocsticas tienen mayor madurez y donde CVX puede capturar valor inmediato.

### 7.1 Detailed Use Case Table

| Use Case | Industry | CVX Feature | Example | Reference |
|----------|----------|-------------|---------|-----------|
| Market regime matching | Hedge funds (Citadel, Renaissance) | Temporal kNN + regime detection | "Find historical periods where macro indicator embeddings were in the same regime as today" | Hamilton (1989), Nystrup et al. (2020) |
| Factor decay detection | Quant funds (Two Sigma, AQR) | Mean reversion + GARCH on factor embeddings | "Is the momentum factor's alpha decaying? ADF test on factor return embedding trajectory" | Two Sigma factor embedding research |
| Alt-data anomaly detection | Systematic funds (Man AHL, Winton) | BOCPD + change point narrative on satellite/social embeddings | "Satellite image embeddings of Chinese ports shifted — trade signal?" | Bloomberg TKG alt-data embeddings |
| Derivatives pricing | Banks (Goldman Sachs, JP Morgan) | Path signatures as pricing features | "Signature of the underlying's trajectory $\to$ exotic option price via signature regression" | Arribas et al. (2020), Gierjatowicz et al. (2022) |
| Portfolio regime switching | Macro funds (Bridgewater, Millennium) | HMM regime detection on macro indicator embeddings | "Risk-on vs risk-off regime identification from multi-asset embedding trajectories" | Nystrup et al. (2020) |
| Fraud detection | Banks, fintechs (Stripe, Revolut) | Cohort divergence + trajectory anomaly | "Client's transaction embedding trajectory diverges from peer group — signature dissimilarity flags anomaly" | — |
| Supply chain risk | Corporates (Apple, Toyota) | Drift attribution + early detection on supplier embeddings | "Supplier's embedding drifting away from industry cluster — early warning of financial trouble" | — |
| Credit risk | Banks (Moody's, S&P) | Mean reversion + co-integration on counterparty embeddings | "Counterparty's embedding breaking co-integration with its sector — credit event signal" | — |
| Order book dynamics | Market makers (Virtu, Citadel Securities) | Path signatures on order book embeddings | "Signature of limit order book state evolution $\to$ short-term price prediction" | JP Morgan LOB embedding research |
| ESG trend analysis | Asset managers (BlackRock, Vanguard) | Hurst exponent + drift significance on ESG embeddings | "ESG narrative embeddings have H=0.7 (persistent trend) with significant drift — structural shift in corporate behavior" | — |

### 7.2 End-to-End Workflow: Market Regime Matching

Un ejemplo detallado de cmo un quant researcher usara CVX's stochastic analytics:

**Step 1: Ingest macro indicator embeddings**
```bash
# Daily embeddings of macro indicators (GDP, unemployment, inflation, etc.)
# Embedded via a financial language model
POST /v1/ingest { entity_id: "macro_state", timestamp: ..., vector: [...] }
```

**Step 2: Classify current regime**
```bash
GET /v1/stochastic/entities/macro_state/regime?n_regimes=3
# Returns: current regime = "Transitioning" with 87% probability
```

**Step 3: Find historical matches**
```bash
POST /v1/query {
  "type": "trajectory_similarity",
  "entity_id": "macro_state",
  "time_range": ["2025-01-01", "2026-03-01"],
  "k": 5,
  "signature_config": { "depth": 3, "dims": 5 }
}
# Returns: 5 historical periods with similar macro evolution patterns
```

**Step 4: Analyze what happened after those historical matches**
```bash
# For each historical match, query what followed
GET /v1/stochastic/entities/macro_state/classification?from=<match_end>&to=<match_end+6months>
# Returns: 3/5 matches transitioned to "Stable", 2/5 to "Turbulent"
```

**Step 5: Stochastic prediction**
```bash
POST /v1/query {
  "type": "stochastic_prediction",
  "entity_id": "macro_state",
  "target_timestamp": "2026-09-01",
  "n_samples": 1000
}
# Returns: distribution of possible futures with confidence intervals
```

### 7.3 End-to-End Workflow: Factor Alpha Decay

**Step 1: Ingest factor return embeddings**
```bash
# Weekly embeddings of factor portfolio characteristics
POST /v1/ingest { entity_id: "momentum_factor", timestamp: ..., vector: [...] }
```

**Step 2: Test mean reversion of alpha signal**
```bash
GET /v1/stochastic/entities/momentum_factor/mean-reversion?from=2020-01-01&to=2026-03-01
# Returns: classification = "Trending", drift is significant, H = 0.62
# Interpretation: alpha is persistently declining, not mean-reverting
```

**Step 3: Estimate volatility structure**
```bash
GET /v1/stochastic/entities/momentum_factor/garch
# Returns: persistence = 0.94, half_life = 11.5 weeks
# Interpretation: volatility shocks in the factor persist for ~3 months
```

**Step 4: Check co-integration with market factor**
```bash
GET /v1/stochastic/cointegration?entity_a=momentum_factor&entity_b=market_factor
# Returns: is_cointegrated = false, p_value = 0.42
# Interpretation: momentum has decoupled from the market — structural change
```

---

## 8. Module Structure

```
crates/cvx-analytics/src/
├── stochastic/
│   ├── mod.rs              // StochasticAnalytics trait, ProcessClassification
│   ├── drift_test.rs       // DriftSignificance, t-test, Hotelling T²
│   ├── volatility.rs       // RealizedVolatility, multiple estimators
│   ├── garch.rs            // GARCH(1,1) estimation via MLE
│   ├── mean_reversion.rs   // ADF test, KPSS test, OU parameter estimation
│   ├── hurst.rs            // Hurst exponent via DFA and R/S
│   ├── classification.rs   // StationarityReport, ProcessClassification, decision tree
│   └── regime.rs           // HMM-based regime detection, Baum-Welch, Viterbi
├── signatures/
│   ├── mod.rs              // PathSignature, SignatureConfig
│   ├── compute.rs          // Signature computation (iterated integrals)
│   ├── log_signature.rs    // Log-signature via Baker-Campbell-Hausdorff
│   ├── pca.rs              // PCA reduction before signature computation
│   └── similarity.rs       // Signature-based trajectory similarity search
├── cross_entity/
│   ├── mod.rs              // Cross-entity analysis trait
│   ├── dcc.rs              // Dynamic Conditional Correlation (Engle 2002)
│   ├── cointegration.rs    // Johansen trace test
│   ├── granger.rs          // Granger causality (VAR-based F-test)
│   └── wasserstein.rs      // Sliced Wasserstein neighborhood drift
├── neural_sde/
│   ├── mod.rs              // NeuralSdePredictor
│   ├── solver.rs           // Euler-Maruyama, Milstein, SRK SDE solvers
│   ├── adjoint.rs          // Adjoint SDE method for backpropagation
│   ├── jump.rs             // Neural Jump SDE (discontinuities + intensity)
│   └── training.rs         // Training loop, loss functions, regularization
```

### 8.1 Trait Definition

```rust
/// Core trait for stochastic analytics on entity trajectories.
pub trait StochasticAnalytics {
    /// Test whether an entity's drift is statistically significant.
    fn drift_significance(
        &self,
        entity_id: u64,
        time_range: (i64, i64),
        threshold: f64,
    ) -> Result<DriftSignificance>;

    /// Compute realized volatility for an entity's trajectory.
    fn realized_volatility(
        &self,
        entity_id: u64,
        time_range: (i64, i64),
    ) -> Result<RealizedVolatility>;

    /// Fit GARCH(1,1) model to an entity's trajectory.
    fn garch_estimate(
        &self,
        entity_id: u64,
        time_range: (i64, i64),
    ) -> Result<GarchEstimate>;

    /// Test mean reversion and estimate OU parameters.
    fn mean_reversion(
        &self,
        entity_id: u64,
        time_range: (i64, i64),
    ) -> Result<MeanReversionAnalysis>;

    /// Estimate Hurst exponent.
    fn hurst_exponent(
        &self,
        entity_id: u64,
        time_range: (i64, i64),
        method: HurstMethod,
    ) -> Result<HurstExponent>;

    /// Full stationarity classification report.
    fn classify(
        &self,
        entity_id: u64,
        time_range: (i64, i64),
    ) -> Result<StationarityReport>;

    /// Compute path signature of an entity's trajectory.
    fn path_signature(
        &self,
        entity_id: u64,
        time_range: (i64, i64),
        config: &SignatureConfig,
    ) -> Result<PathSignature>;

    /// Detect regimes in an entity's trajectory.
    fn detect_regimes(
        &self,
        entity_id: u64,
        time_range: (i64, i64),
        config: &RegimeConfig,
    ) -> Result<RegimeDetection>;
}

/// Trait for cross-entity stochastic analysis.
pub trait CrossEntityAnalytics {
    /// Dynamic conditional correlation between two entities.
    fn dynamic_correlation(
        &self,
        entity_a: u64,
        entity_b: u64,
        time_range: (i64, i64),
    ) -> Result<DynamicCorrelation>;

    /// Co-integration test between two entities.
    fn cointegration(
        &self,
        entity_a: u64,
        entity_b: u64,
        time_range: (i64, i64),
    ) -> Result<CointegrationResult>;

    /// Granger causality test.
    fn granger_causality(
        &self,
        cause: u64,
        effect: u64,
        max_lag: usize,
        time_range: (i64, i64),
    ) -> Result<GrangerCausalityResult>;

    /// Wasserstein neighborhood drift.
    fn wasserstein_drift(
        &self,
        entity_id: u64,
        t1: i64,
        t2: i64,
        k: usize,
    ) -> Result<WassersteinDrift>;
}
```

---

## 9. Feature Flags

```toml
[features]
# Stochastic process characterization (per-entity)
stochastic-basic = []          # drift test, realized volatility, mean reversion, Hurst
stochastic-garch = []          # GARCH(1,1) volatility model (requires MLE optimizer)
stochastic-regime = []         # HMM regime detection (Baum-Welch + Viterbi)

# Path signatures
signatures = []                # truncated path signatures + log-signatures
signatures-index = ["signatures"] # HNSW index on materialized signatures

# Neural SDE (extends Neural ODE from Temporal ML Spec)
neural-sde = ["burn"]         # stochastic prediction with uncertainty
neural-jump-sde = ["neural-sde"]  # + jump processes (unified prediction + CPD)

# Cross-entity analysis
cross-entity = []              # DCC, co-integration, Granger causality
cross-entity-wasserstein = ["cross-entity"]  # + sliced Wasserstein (more expensive)

# Full stochastic layer
stochastic-full = [
    "stochastic-basic",
    "stochastic-garch",
    "stochastic-regime",
    "signatures",
    "cross-entity",
]
```

### 9.1 Dependency Notes

| Feature | External Dependencies | Internal Dependencies |
|---------|----------------------|----------------------|
| `stochastic-basic` | None (pure Rust math) | `cvx-analytics` diffcalc |
| `stochastic-garch` | None (custom MLE) | `stochastic-basic` |
| `stochastic-regime` | None (custom HMM) | `stochastic-basic` |
| `signatures` | None (custom implementation) | `cvx-analytics` PCA |
| `neural-sde` | `burn` | Neural ODE infrastructure |
| `neural-jump-sde` | `burn` | `neural-sde` + BOCPD |
| `cross-entity` | None | `stochastic-basic` |
| `cross-entity-wasserstein` | None | `cross-entity` + kNN index |

---

## 10. Relationship with Existing CVX Analytics

Esta seccin muestra cmo la nueva capa estocstica se integra con y extiende los componentes analticos existentes de CVX. Cada herramienta existente tiene una extensin natural en el marco estocstico.

```
EXISTING                          NEW (Stochastic Layer)
════════                          ════════════════════════
diffcalc/velocity           →→→   drift_test (is velocity significant?)
                                  "You have drift = 0.01. But is it real? p = 0.002 — yes."

diffcalc/acceleration       →→→   GARCH (is acceleration clustering?)
                                  "Acceleration spikes cluster. GARCH persistence = 0.94."

PELT change points          →→→   regime detection (what regime before/after?)
                                  "Change point at t=500. Before: Stable. After: Transitioning."

BOCPD online detection      →→→   Neural Jump SDE (unified prediction + detection)
                                  "Predict trajectory AND detect jumps in one model."

Neural ODE prediction       →→→   Neural SDE (prediction with uncertainty)
                                  "Not just where it will be, but the distribution of where."

cohort divergence           →→→   DCC, co-integration, Granger causality
                                  "Not just 'diverging' — but is it decorrelation, broken
                                   co-integration, or causal lead-lag?"

trajectory retrieval        →→→   path signatures (trajectory descriptor)
                                  "A compact vector that captures the SHAPE of evolution."

temporal kNN                →→→   signature kNN (trajectory similarity search)
                                  "Find entities that EVOLVED the same way, not just near now."

drift attribution           →→→   drift significance test (is it real?)
                                  "Drift attributed to dimension 42. But p = 0.3 — not significant."

multi-scale analysis        →→→   Hurst exponent (at what scale is the signal?)
                                  "H = 0.3 at hourly scale (noise), H = 0.7 at weekly (trend)."
```

### 10.1 Integration Points

| Existing Module | Integration | Data Flow |
|----------------|-------------|-----------|
| `diffcalc` | Provides increments for all stochastic tests | `diffcalc::velocity()` $\to$ `drift_test::test()` |
| `change_points` (PELT) | Change points segment trajectory for per-regime analysis | `pelt::detect()` $\to$ `regime::label_segments()` |
| `change_points` (BOCPD) | Online change points feed jump intensity estimation | `bocpd::update()` $\to$ `jump_sde::update_intensity()` |
| `neural_ode` | Drift network architecture reused for Neural SDE drift | `neural_ode::MLP` $\to$ `neural_sde::drift_net` |
| `cohort` | Cohort entity lists feed cross-entity analysis | `cohort::members()` $\to$ `cross_entity::pairwise()` |
| `explain` | Stochastic classifications become explain artifacts | `classification::classify()` $\to$ `explain::StochasticExplanation` |

---

## 11. Roadmap Integration

La capa estocstica se distribuye a lo largo del roadmap existente de CVX, integrndose con las capas de implementacin ya definidas.

| Layer | Stochastic Component | Dependency | Effort Estimate |
|-------|---------------------|------------|-----------------|
| **L7** (Vector Calculus) | `drift_test`, `realized_volatility`, `mean_reversion`, `hurst` | Only needs `diffcalc` output — pure math on increment sequences | 2-3 weeks |
| **L7.5** (Explain) | Stationarity classification as explain artifact | `drift_test` + `mean_reversion` + `hurst` combined | 1 week |
| **L8** (PELT + BOCPD) | Regime detection (HMM) — complements BOCPD | BOCPD provides change points, HMM adds regime labels and transition probabilities | 2-3 weeks |
| **L8+** | Path signatures: compute + similarity search | Trajectory retrieval from storage, PCA infrastructure | 3-4 weeks |
| **L8+** | GARCH volatility model | `stochastic-basic` for increment sequences | 1-2 weeks |
| **L10** (Neural ODE) | Extend to Neural SDE + Neural Jump SDE | `burn`, existing ODE infrastructure, adjoint method | 4-6 weeks |
| **L11+** | Cross-entity: DCC, co-integration, Granger, Wasserstein | Multiple trajectory access, kNN index for Wasserstein | 3-4 weeks |
| **L12** (Benchmarks) | Evaluation of all stochastic analytics on real data | All stochastic components | 2-3 weeks |

**Total estimated effort:** 18-26 weeks (partially parallelizable with other roadmap work).

### 11.1 Incremental Value Delivery

Each layer delivers standalone value — no need to wait for the full stack:

- **L7 alone:** Stationarity classification answers "is this drift real?" — immediately useful for any ML monitoring use case.
- **L7 + L8:** Regime detection adds "what kind of state is this entity in?" — enables regime-conditional dashboards.
- **L8+:** Path signatures enable trajectory similarity search — a completely novel query type.
- **L10:** Neural SDE provides prediction with uncertainty — essential for production risk management.
- **L11+:** Cross-entity analysis enables relationship discovery at scale.

---

## 12. Non-Functional Requirements

### 12.1 Latency Targets

Todos los targets asumen una trayectoria de 1,000 puntos temporales con $D = 768$ dimensiones, ejecutndose en un solo ncleo.

| Operation | Target Latency | Notes |
|-----------|---------------|-------|
| Drift significance test | < 1ms | Simple mean/std computation on increments |
| Realized volatility | < 1ms | Same complexity as drift test |
| GARCH estimation | < 10ms | MLE optimization (L-BFGS-B, ~50 iterations) |
| ADF test | < 5ms | OLS regression + test statistic |
| KPSS test | < 5ms | Cumulative sum + kernel estimator |
| Mean reversion (ADF + KPSS + OU) | < 15ms | Combined |
| Hurst exponent (DFA) | < 10ms | Multi-scale fluctuation analysis |
| Full stationarity report | < 30ms | All of the above combined |
| Path signature (d=5, depth=3) | < 5ms | PCA + iterated integrals |
| Path signature (d=10, depth=3) | < 20ms | Larger signature space |
| Regime detection (3 regimes) | < 50ms | Baum-Welch EM (10-20 iterations) |
| Neural SDE prediction (100 samples) | < 500ms per entity | 100 forward SDE integrations |
| DCC correlation | < 20ms per pair | Bivariate GARCH + DCC estimation |
| Co-integration test (Johansen) | < 10ms per pair | VAR estimation + trace test |
| Granger causality | < 15ms per pair | VAR estimation + F-test |
| Sliced Wasserstein (k=50, 100 proj) | < 50ms per entity | kNN retrieval + 100 random projections |

### 12.2 Accuracy Targets

| Test | Requirement |
|------|-------------|
| Drift significance | Type I error rate $\leq 0.05$ on simulated random walks |
| ADF test | Critical values match MacKinnon (1996) tables within 1% |
| KPSS test | Critical values match Kwiatkowski et al. (1992) tables within 1% |
| Hurst exponent | Recovers $H = 0.5$ on simulated Brownian motion ($\pm 0.05$) |
| GARCH | Recovers known parameters on simulated GARCH(1,1) ($\pm 10\%$) |
| Regime detection | Correctly labels $> 90\%$ of time steps on simulated 2-regime data |
| Path signature | Matches reference implementation (signatory) within $10^{-6}$ |

### 12.3 Memory Requirements

| Component | Memory Usage |
|-----------|-------------|
| Per-entity stochastic analysis | $O(T \cdot D)$ — stores trajectory in memory |
| GARCH estimation | $O(T)$ additional for conditional variances |
| Path signature | $O(T \cdot d_{\text{red}})$ for reduced trajectory + $O(d_{\text{red}}^k)$ for signature |
| Regime detection (HMM) | $O(T \cdot K)$ for forward-backward probabilities |
| Neural SDE (adjoint) | $O(D)$ — constant memory regardless of trajectory length |
| Neural SDE (direct) | $O(T \cdot D)$ — stores all intermediate states |
| Cross-entity (per pair) | $O(T)$ — stores paired drift series |
| Sliced Wasserstein | $O(k \cdot D + n_{\text{proj}})$ — neighbor vectors + projections |

### 12.4 Scalability

| Scenario | Requirement |
|----------|-------------|
| Batch stationarity report | 10,000 entities in < 5 minutes (parallelized) |
| Signature materialization | 100,000 entities in < 1 hour |
| Pairwise cross-entity | 1,000 entity pairs in < 30 seconds |
| Online regime update | < 1ms per new data point (forward algorithm) |

---

## 13. Bibliography

### Stochastic Processes & Time Series

- Hamilton, J. D. (1989). "A new approach to the economic analysis of nonstationary time series and the business cycle." *Econometrica*, 57(2), 357-384.
- Engle, R. F. (1982). "Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation." *Econometrica*, 50(4), 987-1007.
- Bollerslev, T. (1986). "Generalized autoregressive conditional heteroskedasticity." *Journal of Econometrics*, 31(3), 307-327.
- Uhlenbeck, G. E., & Ornstein, L. S. (1930). "On the theory of the Brownian motion." *Physical Review*, 36(5), 823-841.
- Dickey, D. A., & Fuller, W. A. (1979). "Distribution of the estimators for autoregressive time series with a unit root." *Journal of the American Statistical Association*, 74(366a), 427-431.
- Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y. (1992). "Testing the null hypothesis of stationarity against the alternative of a unit root." *Journal of Econometrics*, 54(1-3), 159-178.
- MacKinnon, J. G. (1996). "Numerical distribution functions for unit root and cointegration tests." *Journal of Applied Econometrics*, 11(6), 601-618.
- Granger, C. W. J. (1969). "Investigating causal relations by econometric models and cross-spectral methods." *Econometrica*, 37(3), 424-438.
- Johansen, S. (1991). "Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models." *Econometrica*, 59(6), 1551-1580.
- Hurst, H. E. (1951). "Long-term storage capacity of reservoirs." *Transactions of the American Society of Civil Engineers*, 116, 770-799.
- Peng, C. K., Buldyrev, S. V., Havlin, S., Simons, M., Stanley, H. E., & Goldberger, A. L. (1994). "Mosaic organization of DNA nucleotides." *Physical Review E*, 49(2), 1685-1689.
- Engle, R. (2002). "Dynamic conditional correlation: A simple class of multivariate generalized autoregressive conditional heteroskedasticity models." *Journal of Business & Economic Statistics*, 20(3), 339-350.

### Path Signatures & Rough Path Theory

- Lyons, T. (1998). "Differential equations driven by rough signals." *Revista Matemtica Iberoamericana*, 14(2), 215-310.
- Chevyrev, I., & Kormilitzin, A. (2016). "A Primer on the Signature Method in Machine Learning." *arXiv:1603.03788*.
- Kidger, P., & Lyons, T. (2021). "Signatory: differentiable computations of the signature and logsignature transforms, on both CPU and GPU." *ICLR 2021*.
- Arribas, I. P., Salvi, C., & Sherbet, L. (2020). "Sig-SDEs model for quantitative finance." *arXiv:2006.00218*.
- Ni, H., Szpruch, L., Wiese, M., Liao, S., & Xiao, B. (2021). "Sig-Wasserstein GANs for time series generation." *ICAIF 2021*.
- Kiraly, F. J., & Oberhauser, H. (2019). "Kernels for sequentially ordered data." *JMLR*, 20(31), 1-45.
- Chen, K. T. (1958). "Integration of paths — a faithful representation of paths by noncommutative formal power series." *Transactions of the American Mathematical Society*, 89(2), 395-407.

### Neural SDEs & Differential Equations

- Li, X., Wong, T. K. L., Chen, R. T. Q., & Duvenaud, D. (2020). "Scalable Gradients for Stochastic Differential Equations." *AISTATS 2020*.
- Kidger, P., Foster, J., Li, X., & Lyons, T. (2021). "Neural SDEs as Infinite-Dimensional GANs." *ICML 2021*.
- Jia, J., & Benson, A. R. (2019). "Neural Jump Stochastic Differential Equations." *NeurIPS 2019*.
- Gierjatowicz, P., Sabate-Vidales, M., Siska, D., Szpruch, L., & Zuric, Z. (2022). "Robust pricing and hedging via Neural SDEs." *Journal of Computational Finance*, 26(3).
- Morrill, J., Salvi, C., Kidger, P., & Foster, J. (2021). "Neural Rough Differential Equations." *arXiv:2009.08295*.
- Rssler, A. (2010). "Runge-Kutta Methods for the Strong Approximation of Solutions of Stochastic Differential Equations." *SIAM Journal on Numerical Analysis*, 48(3), 922-952.
- Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). "Neural Ordinary Differential Equations." *NeurIPS 2018*.
- Lewis, P. A. W., & Shedler, G. S. (1979). "Simulation of nonhomogeneous Poisson processes by thinning." *Naval Research Logistics Quarterly*, 26(3), 403-413.

### Embeddings & Stochastic Processes

- Bamler, R., & Mandt, S. (2017). "Dynamic Word Embeddings." *ICML 2017*. — Models word embedding evolution as a diffusion process.
- Rosenfeld, A., & Erk, K. (2018). "Deep Neural Models of Semantic Shift." *NAACL 2018*. — Semantic shift as stochastic diffusion in embedding space.
- Arora, S., Li, Y., Liang, Y., Ma, T., & Risteski, A. (2016). "A Latent Variable Model Approach to PMI-based Word Embeddings." *TACL*, 4, 385-399. — Word embeddings as random walk on a discourse model.
- Rudolph, M., & Blei, D. (2018). "Dynamic Embeddings for Language Evolution." *WWW 2018*. — Kalman filter model for embedding trajectories.

### Financial Applications & Stylized Facts

- Cont, R. (2001). "Empirical properties of asset returns: stylized facts and statistical issues." *Quantitative Finance*, 1(2), 223-236.
- Nystrup, P., Madsen, H., & Lindstrm, E. (2020). "Learning hidden Markov models with persistent states by penalizing jumps." *Expert Systems with Applications*, 150, 113307.
- Villani, C. (2008). *Optimal Transport: Old and New.* Springer. — Mathematical foundations of Wasserstein distance.
- Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). "Volatility is rough." *Quantitative Finance*, 18(6), 933-949. — Hurst exponent $H \approx 0.1$ for realized volatility.
- Ramaswamy, V., DeMiguel, V., & Nogales, F. J. (2021). "Convex optimization for portfolio allocation with transaction costs." *SIAM Journal on Financial Mathematics*, 12(4), 1405-1437.

### Change Point Detection

- Matteson, D. S., & James, N. A. (2014). "A nonparametric approach for multiple change point analysis of multivariate data." *Journal of the American Statistical Association*, 109(505), 334-345.
- Fan, Z., & Mackey, L. (2017). "Multi-Sequence BOCPD." *arXiv:1710.07269*.
- Adams, R. P., & MacKay, D. J. C. (2007). "Bayesian Online Changepoint Detection." *arXiv:0710.3742*.

---

## 14. Open Questions

### 14.1 Path Signature Depth

**Question:** Depth 2 vs 3 vs 4?

Higher depth = more expressive but exponentially more features. For $d_{\text{reduced}} = 5$:

| Depth | Signature dims | Log-signature dims |
|-------|---------------|-------------------|
| 2 | 30 | 15 |
| 3 | 155 | 35 |
| 4 | 780 | 75 |

**Recommendation:** Start with depth 3 (log-signature) as the default. This gives 35 features — manageable and expressive. Empirical evaluation on real embedding trajectories is needed to determine if depth 4 adds meaningful discriminative power. The `SignatureConfig` makes this user-configurable.

### 14.2 GARCH in High Dimensions

**Question:** Fit GARCH on scalar volatility (norm of increments) or multivariate DCC-GARCH on per-dimension increments?

Multivariate DCC-GARCH is $O(d^2)$ — prohibitively expensive for $d = 768$. Options:
1. **Scalar GARCH** on $\lVert \Delta v \rVert$ — simple, captures overall volatility clustering. Loses directional information.
2. **Per-component GARCH** on PCA-reduced increments (top 5-10 PCs) — captures directional volatility structure in a tractable way.
3. **Factor GARCH** — use PCA factors, fit GARCH on each factor independently.

**Recommendation:** Scalar GARCH as default (fast, interpretable). Per-component GARCH on PCA-reduced increments as an advanced option behind a configuration flag.

### 14.3 Regime Count

**Question:** How many regimes? Auto-detection via BIC/AIC? Or user-specified?

**Recommendation:** Support both. Default to auto-detection with $K_{\max} = 5$ and BIC selection. Allow user override for domain-specific regime counts (e.g., $K=2$ for risk-on/risk-off). Add a minimum segment length constraint to prevent degenerate solutions with too-frequent switching.

### 14.4 SDE Solver Choice

**Question:** Euler-Maruyama is simple but $O(\Delta t^{0.5})$ convergence (strong order). Milstein is $O(\Delta t)$ but requires computing the Jacobian $\partial g / \partial z$ of the diffusion network. For stiff problems, implicit solvers may be needed.

**Recommendation:** Euler-Maruyama as default (simpler, sufficient for most applications). Milstein as an option when higher accuracy is needed. Automatic step size adaptation based on estimated local error. Monitor for stiffness via eigenvalue estimation and warn users if an implicit solver may be needed.

### 14.5 Signature Computation in Rust

**Question:** Implement from scratch or port/wrap signatory (Python/C++)?

A pure Rust implementation would be:
- Consistent with CVX's no-Python-dependency philosophy
- Potentially faster (no FFI overhead)
- Novel (no production-quality Rust signature library exists as of 2026)
- Full control over optimizations (SIMD, incremental updates via Chen's identity)

**Recommendation:** Implement from scratch in Rust. The core algorithm (iterated integrals) is mathematically well-defined and not excessively complex. The PCA step can use `nalgebra`. This would be a valuable open-source contribution independent of CVX.

### 14.6 E-Divisive as Alternative to PELT

**Question:** E-Divisive (Matteson & James, 2014) works directly on multivariate data without distributional assumptions — potentially better for high-dimensional embeddings than PELT (which typically operates on univariate cost functions).

Advantages of E-Divisive:
- Nonparametric — no assumption on distribution of increments
- Natively multivariate — uses energy statistics on the full $d$-dimensional vectors
- Detects changes in *any* aspect of the distribution (mean, variance, shape)

Disadvantages:
- $O(n^2)$ time complexity (vs $O(n)$ for PELT)
- Not naturally online (vs BOCPD)

**Recommendation:** Support both PELT and E-Divisive. PELT for online/fast detection on scalar projections. E-Divisive for offline, thorough multivariate analysis. The regime detection (HMM) provides a third complementary approach. The choice depends on the use case:

| Use Case | Recommended Method |
|----------|-------------------|
| Real-time alerting | BOCPD |
| Fast offline detection | PELT |
| Thorough multivariate analysis | E-Divisive |
| Regime characterization | HMM |

### 14.7 Statistical Testing Framework

**Question:** Should CVX implement its own statistical testing framework (t-distributions, F-distributions, chi-squared, critical value tables) or depend on an external crate?

**Recommendation:** Implement core statistical functions (t-distribution CDF, F-distribution CDF, normal CDF) using well-known numerical approximations (e.g., Abramowitz & Stegun). This avoids external dependencies for a small amount of code. For Dickey-Fuller critical values specifically, embed the MacKinnon (1996) response surface coefficients.

### 14.8 Interaction with Multi-Scale Alignment

**Question:** The Multi-Scale Alignment Spec defines cross-space alignment (text vs image embeddings). The Stochastic Analytics Spec defines cross-entity analysis (entity A vs entity B). How do they interact?

**Recommendation:** Stochastic analytics should be applicable *within* each embedding space. Cross-space alignment provides the multi-modal view; stochastic analytics provides the temporal characterization. The natural combination: "In space 'text-bert-768', entities A and B are co-integrated. In space 'image-clip-512', they are not. The alignment between spaces for entity A shows behavioral divergence — the text and image trajectories are decorrelating."

---

## 15. Glossary

| Term | Definition |
|------|-----------|
| **ADF** | Augmented Dickey-Fuller test — tests for unit roots (random walk) |
| **BOCPD** | Bayesian Online Change Point Detection |
| **DCC** | Dynamic Conditional Correlation (Engle, 2002) |
| **DFA** | Detrended Fluctuation Analysis — method for estimating Hurst exponent |
| **GARCH** | Generalized Autoregressive Conditional Heteroskedasticity |
| **HMM** | Hidden Markov Model |
| **KPSS** | Kwiatkowski-Phillips-Schmidt-Shin test — tests for stationarity |
| **MLE** | Maximum Likelihood Estimation |
| **Neural SDE** | Stochastic differential equation with neural network drift and diffusion |
| **OU** | Ornstein-Uhlenbeck process — mean-reverting diffusion |
| **Path signature** | Sequence of iterated integrals characterizing a path (Lyons, 1998) |
| **PELT** | Pruned Exact Linear Time — change point detection algorithm |
| **Realized volatility** | Volatility estimated from observed trajectory increments |
| **Regime** | A distinct dynamical state of a stochastic process |
| **Sliced Wasserstein** | Computationally tractable approximation of Wasserstein distance |
| **Wiener process** | Standard Brownian motion $W(t)$ — continuous-time random walk |
