---
title: "Stochastic Processes for Embeddings"
description: "Mathematical framework: Brownian motion, GARCH, mean reversion, and Hurst exponents applied to embedding trajectories"
---

## Why Stochastic Processes?

ChronosVector already computes first-order temporal analytics: velocity, acceleration, change points, and cohort divergence. These answer *what* changed and *when*. But they are deterministic and descriptive — they cannot distinguish real signal from noise, model uncertainty, or capture complex phenomena like volatility clustering or long-range dependence.

The key insight is that an embedding trajectory $v(t) \in \mathbb{R}^d$ can be modeled as a **stochastic process**. In the most general diffusion formulation:

$$
dv(t) = \mu(v, t) \, dt + \sigma(v, t) \, dW(t)
$$

where:

- $\mu(v, t)$ is the **drift function** — the systematic, directional component of change. This is the "signal" in the trajectory.
- $\sigma(v, t)$ is the **diffusion/volatility function** — the stochastic fluctuation, the "noise" in the trajectory.
- $W(t)$ is a **$d$-dimensional Wiener process** (standard Brownian motion).

This is not merely an analogy. Embedding trajectories exhibit many of the same statistical properties as financial time series: periods of stability and turbulence, volatility clustering, mean reversion toward equilibria, and increments that are rarely i.i.d. Gaussian. The quantitative finance literature provides decades of battle-tested tools for exactly these phenomena (Bamler & Mandt, 2017; Hamilton, 1989).

---

## Drift Significance Test

CVX computes velocity (drift rate) as a first-class analytic. But a critical question remains: **is the observed velocity statistically significant, or could it arise from a pure random walk?**

An entity with a drift rate of 0.01 per timestep could be undergoing genuine directional change, or simply fluctuating randomly. The distinction has profound implications for interpretation and action.

### Method

Under the null hypothesis $H_0$: no drift (pure random walk), the increments $\Delta v_i = v(t_{i+1}) - v(t_i)$ have zero mean. The test statistic is:

$$
t = \frac{\bar{\Delta v}}{\hat{\sigma}_{\Delta v} / \sqrt{n}}
$$

where $\bar{\Delta v}$ is the mean increment magnitude, $\hat{\sigma}_{\Delta v}$ is the sample standard deviation of increments, and $n$ is the number of increments. Under $H_0$, this follows a $t$-distribution with $n - 1$ degrees of freedom.

For multivariate drift, CVX uses the **Hotelling $T^2$ test** on the vector of mean increments, which reduces to the scalar $t$-test when applied to drift magnitudes.

The result includes both statistical significance ($p$-value) and **practical significance** (Cohen's $d$ effect size), because a statistically significant but tiny drift may not be actionable.

---

## Realized Volatility

In finance, volatility (the standard deviation of returns) is arguably the most important metric after the return itself. For embedding trajectories, volatility measures the **variability** of change — not how much the entity changed on average, but how *erratic* that change was.

### Estimators

CVX provides multiple volatility estimators, each with different properties:

| Estimator | Formula | What it captures |
|-----------|---------|-----------------|
| Scalar realized volatility | $\hat{\sigma} = \text{std}(\lVert v(t_{i+1}) - v(t_i) \rVert)$ | Overall trajectory roughness |
| Per-dimension volatility | $\hat{\sigma}_d = \text{std}(v_d(t_{i+1}) - v_d(t_i))$ for each $d$ | Which dimensions are volatile |
| Annualized/normalized | $\hat{\sigma}_{\text{ann}} = \hat{\sigma} \cdot \sqrt{T / \Delta t}$ | Comparable across sampling frequencies |

The **volatility of volatility** (vol-of-vol) measures meta-stability: high vol-of-vol indicates that the volatility itself is unstable, suggesting regime transitions or structural breaks in the trajectory's dynamics.

---

## GARCH Volatility Model

Volatility is not constant over time. A well-documented phenomenon in finance — and observable in embedding trajectories — is **volatility clustering**: periods of high volatility tend to follow periods of high volatility, and vice versa (Engle, 1982; Bollerslev, 1986).

The GARCH(1,1) model (Generalized Autoregressive Conditional Heteroskedasticity) captures this clustering:

$$
\sigma^2(t) = \omega + \alpha \cdot \varepsilon^2(t-1) + \beta \cdot \sigma^2(t-1)
$$

where:

- $\omega$ is the long-run variance weight (intercept)
- $\alpha$ measures the reaction to recent shocks — the **innovation coefficient**
- $\beta$ measures the persistence of past volatility — the **lag coefficient**
- $\varepsilon(t) = \Delta v(t) / \sigma(t)$ are the standardized residuals

### Interpreting GARCH Parameters

The **persistence** $\alpha + \beta$ is the most informative parameter:

| Persistence $(\alpha + \beta)$ | Interpretation |
|-------------------------------|----------------|
| $> 0.95$ | Integrated GARCH — volatility shocks are nearly permanent |
| $0.8 - 0.95$ | High persistence — shocks decay slowly |
| $0.5 - 0.8$ | Moderate — shocks decay at medium speed |
| $< 0.5$ | Low persistence — volatility reverts quickly to long-run level |

The **half-life** of a volatility shock tells you how long a perturbation lasts:

$$
h = \frac{-\ln 2}{\ln(\alpha + \beta)}
$$

The **long-run (unconditional) volatility** is $\sqrt{\omega / (1 - \alpha - \beta)}$, which gives the equilibrium volatility level the process reverts to.

### Estimation

The model is estimated by maximum likelihood (MLE). Scalar increments $r_t = \lVert v(t+1) - v(t) \rVert$ are modeled as $r_t = \sigma_t \cdot z_t$ with $z_t \sim \mathcal{N}(0,1)$. The conditional Gaussian log-likelihood:

$$
\ell(\theta) = -\frac{1}{2} \sum_{t=1}^{T} \left[ \ln(2\pi) + \ln(\sigma_t^2) + \frac{r_t^2}{\sigma_t^2} \right]
$$

is optimized with L-BFGS-B subject to constraints $\omega > 0$, $\alpha \geq 0$, $\beta \geq 0$, $\alpha + \beta < 1$.

---

## Mean Reversion and the Ornstein-Uhlenbeck Process

A fundamental question about any embedding trajectory: **does it revert to an equilibrium, wander freely (random walk), or trend persistently?**

| Classification | Meaning | Implication |
|---------------|---------|-------------|
| Mean-reverting | Current position will revert to an equilibrium | Deviations are temporary — the entity "wants" to return |
| Random walk | No equilibrium — position is unpredictable | Past trajectory does not inform future position |
| Trending | Persistent directional movement | Momentum — current direction likely to continue |

### Stationarity Tests

Two complementary tests provide robust classification:

**Augmented Dickey-Fuller (ADF) test** — tests whether the series has a unit root:
- $H_0$: unit root (random walk)
- Rejection implies mean-reverting (stationary)
- Test regression: $\Delta v_t = \alpha + \beta v_{t-1} + \sum_{j=1}^{p} \gamma_j \Delta v_{t-j} + \varepsilon_t$

**KPSS test** (Kwiatkowski-Phillips-Schmidt-Shin) — tests the opposite null:
- $H_0$: stationary (mean-reverting)
- Rejection implies unit root or trend

Using both tests together yields a 2x2 classification matrix:

| ADF rejects? | KPSS rejects? | Classification |
|-------------|--------------|----------------|
| Yes | No | **Mean-Reverting** (stationary) |
| No | Yes | **Random Walk** (unit root) |
| Yes | Yes | **Trending** (trend-stationary) |
| No | No | **Inconclusive** |

### Ornstein-Uhlenbeck Parameters

When mean reversion is detected, CVX estimates the parameters of the Ornstein-Uhlenbeck (OU) process:

$$
dv = \theta(\mu - v) \, dt + \sigma \, dW
$$

- $\theta$ = speed of mean reversion (higher $\theta$ means faster reversion)
- $\mu$ = equilibrium position (the attractor)
- $\sigma$ = diffusion coefficient (residual volatility after accounting for reversion)
- **Half-life** $= \ln(2) / \theta$ — how long it takes to revert halfway to equilibrium

The half-life is particularly actionable: a concept with a half-life of 5 time units reverts quickly to its semantic center, while one with a half-life of 500 is effectively a random walk over practical horizons.

---

## Hurst Exponent

The Hurst exponent $H$ measures the **roughness** or **memory** of a trajectory. It is a fundamental quantity that distinguishes three regimes:

| Hurst Value | Classification | Meaning |
|-------------|---------------|---------|
| $H = 0.5$ | Random (Brownian) | Pure random walk — no memory, increments are i.i.d. |
| $H > 0.5$ | Persistent (trending) | Momentum — past direction predicts future direction |
| $H < 0.5$ | Anti-persistent (rough) | Mean-reverting at small scales — past direction predicts *reversal* |

A notable finding in finance: realized volatility has $H \approx 0.1$ (very rough), which motivated the rough volatility theory (Gatheral et al., 2018). Embedding trajectories may exhibit similar roughness, with implications for prediction and modeling.

### Estimation: Detrended Fluctuation Analysis (DFA)

1. Compute the cumulative deviation from the mean: $Y(i) = \sum_{k=1}^{i} (x_k - \bar{x})$
2. Divide $Y$ into windows of size $s$
3. In each window, fit a polynomial trend and compute the residual variance $F(s)$
4. The Hurst exponent satisfies the scaling law: $F(s) \sim s^H$
5. Estimate $H$ from the slope of $\log F(s)$ vs $\log s$

CVX also supports **R/S (rescaled range) analysis** as a simpler alternative, though DFA is more robust to trends.

---

## Unified Stationarity Classification

CVX combines all the above analyses into a unified **process classification** that categorizes each entity's trajectory:

| Classification | Conditions | Description |
|---------------|------------|-------------|
| **StableEquilibrium** | Mean-reverting, low volatility, $H < 0.5$ | Fluctuates around a stable attractor |
| **RandomWalk** | No significant drift, no reversion, $H \approx 0.5$ | Unpredictable — past does not inform future |
| **TrendingWithMomentum** | Significant drift, $H > 0.5$ | Persistent directional movement |
| **VolatileCycling** | Mean-reverting but GARCH persistence $> 0.9$ | Cycles with episodic volatility bursts |
| **RegimeTransition** | Mixed signals across tests | Stochastic character changed during the window |

The classification follows a decision tree that combines drift significance ($p$-value), ADF/KPSS results, Hurst exponent, and GARCH persistence into a single actionable label with a human-readable summary.

---

## Why Financial Tools Apply to Embeddings

The application of quantitative finance tools to embedding trajectories is not a forced analogy. Both domains share structural properties:

1. **Non-stationarity.** Both financial returns and embedding increments exhibit time-varying statistical properties.
2. **Volatility clustering.** Periods of rapid semantic change (e.g., during major events) cluster together, just as market volatility clusters around crises.
3. **Mean reversion.** Many concepts revert to semantic equilibria after perturbations, analogous to mean-reverting assets.
4. **Heavy tails.** Embedding increments, like financial returns, exhibit heavier tails than the Gaussian distribution.
5. **Long-range dependence.** Hurst exponents different from 0.5 indicate that embedding trajectories have memory, just as financial series do.

The key difference is interpretive: in finance, these tools inform trading decisions; in CVX, they inform understanding of semantic evolution, drift detection, and predictive modeling.

---

## References

- Bamler, R. & Mandt, S. (2017). *Dynamic Word Embeddings*. ICML 2017.
- Rosenfeld, A. & Erk, K. (2018). *Deep Neural Models of Semantic Shift*. NAACL 2018.
- Hamilton, J.D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle*. Econometrica.
- Engle, R.F. (1982). *Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation*. Econometrica.
- Bollerslev, T. (1986). *Generalized Autoregressive Conditional Heteroskedasticity*. Journal of Econometrics.
- Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). *Volatility is Rough*. Quantitative Finance.
- Dickey, D.A. & Fuller, W.A. (1979). *Distribution of the Estimators for Autoregressive Time Series with a Unit Root*. JASA.
- Kwiatkowski, D., Phillips, P.C.B., Schmidt, P., & Shin, Y. (1992). *Testing the Null Hypothesis of Stationarity*. Journal of Econometrics.
