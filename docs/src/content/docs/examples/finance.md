---
title: "Quantitative Finance"
description: "Market regime detection, factor decay analysis, and path-dependent pricing with temporal vectors"
---

Quantitative finance operates on high-dimensional, temporally structured data: factor returns evolve, correlations shift, and market regimes emerge and dissolve. ChronosVector (CVX) provides the temporal embedding infrastructure to reason about these dynamics natively.

## Problem

Quant researchers need to:

- **Match current market conditions to historical periods** across hundreds of correlated features, accounting for both similarity and temporal proximity.
- **Detect factor decay early** by monitoring how factor embeddings drift over time, before traditional performance metrics reveal the degradation.
- **Characterize market trajectory shapes** — not just where the market is, but how it got there — for path-dependent pricing and strategy selection.

All of these require working in high-dimensional embedding spaces with first-class temporal semantics, which is exactly what CVX provides.

## Data Model

| Finance Concept       | CVX Primitive        |
| --------------------- | -------------------- |
| Market state          | Vector (factor returns, correlations) |
| Trading day           | Timestamp            |
| Asset / strategy      | Entity               |
| Market regime         | Region               |
| Regime transition     | Change point         |
| Factor trajectory     | Entity trajectory    |

A single market state vector might concatenate sector returns, cross-asset correlations, volatility surface features, and macro indicator embeddings. Each trading day produces a new observation, and CVX indexes them for both semantic and temporal retrieval.

## Code Examples

### Market Regime Matching via Temporal kNN

Find historical market states that resemble current conditions, weighted by both semantic similarity and temporal proximity:

```python
# Find similar historical market states
results = index.search(vector=current_state, k=10, alpha=0.3, query_timestamp=today)
# alpha weighs temporal proximity vs semantic similarity
```

An `alpha` of 0.3 means the search is 70% driven by vector similarity and 30% by recency. Lower alpha values favor pure similarity regardless of when the state occurred; higher values prefer recent matches.

### Factor Trajectory Analysis

Track how a factor embedding evolves and determine its statistical character:

```python
traj = index.trajectory(entity_id=factor_id)
h = cvx.hurst_exponent(traj)
# H > 0.5: trending market → momentum strategies
# H < 0.5: mean-reverting → contrarian strategies
```

The Hurst exponent computed over the embedding trajectory reveals whether the factor is trending (`H > 0.5`), mean-reverting (`H < 0.5`), or following a random walk (`H ≈ 0.5`). This directly informs strategy selection.

### Regime Transition Detection

Identify structural breaks in factor behavior:

```python
changepoints = cvx.detect_changepoints(factor_id, traj)
# Each changepoint = potential regime shift
```

Change points mark moments where the underlying data-generating process shifts. In financial terms, these correspond to regime transitions — from low-vol to high-vol, from risk-on to risk-off, or from one correlation structure to another.

### Velocity for Regime Transition Speed

Measure how fast the market is moving through embedding space:

```python
vel = cvx.velocity(traj, timestamp=transition_ts)
# Velocity spikes indicate fast regime transitions
```

Velocity spikes at a detected change point indicate an abrupt regime shift (e.g., a flash crash or sudden policy change), while gradual velocity increases suggest a slow rotation.

### Sector Rotation via Distributional Distances

Track how portfolio composition shifts across hierarchical market sectors:

```python
reg_traj = index.region_trajectory(entity_id=portfolio_id, level=2)
wd = cvx.wasserstein_drift(dist_t1, dist_t2, centroids)
# Wasserstein respects sector distance: tech→fintech < tech→energy
```

The Wasserstein (earth mover's) distance captures that rotating from technology into fintech is a smaller shift than rotating from technology into energy. This geometry-aware measure of distributional change is critical for understanding true portfolio drift.

### Path Signatures for Trajectory Classification

Compute fixed-dimensional descriptors of market trajectories:

```python
sig = cvx.path_signature(factor_traj, depth=2, time_augmentation=True)
# Fixed-dimensional descriptor invariant to reparametrization
# "Shape" of market move matters more than exact timing
```

Path signatures from rough path theory produce a canonical description of the trajectory shape. A V-shaped recovery and a slow grind-down have different signatures even if they start and end at the same point. Time augmentation preserves information about the speed of moves.

### Portfolio Path Comparison

Compare how two portfolios moved through embedding space:

```python
fd = cvx.frechet_distance(portfolio_traj_a, portfolio_traj_b)
sd = cvx.signature_distance(sig_a, sig_b)
# Signature distance O(K²) vs Fréchet O(nm)
```

Frechet distance measures the maximum deviation between two paths (the "dog-walking" distance), while signature distance compares their canonical shape descriptors. For long trajectories, signature distance is substantially more efficient.

### Event Features for Trading Patterns

Characterize the temporal structure of trade arrival times:

```python
ef = cvx.event_features(trade_timestamps)
# burstiness > 0: bursty trading → possible algorithmic activity
# memory > 0: trades cluster → herding behavior
```

Burstiness and memory coefficients reveal whether trading activity is uniform, bursty (algorithmic), or clustered (herding). These features complement volume analysis with temporal microstructure information.

### Market Structure Topology

Analyze the topological features of sector-level embeddings:

```python
topo = cvx.topological_features(sector_centroids)
# Fragmenting β₀ → market decoupling
# Converging β₀ → correlation spike (risk-off)
```

The zeroth Betti number (`β₀`) counts connected components in the sector embedding space. When `β₀` increases, sectors are decoupling — correlations are breaking down and diversification is working. When `β₀` drops toward 1, everything is converging — a classic risk-off correlation spike.

## Example Workflow: Regime Detection Pipeline

A complete regime detection pipeline chains these primitives together:

1. **Embed daily market states.** Concatenate factor returns, rolling correlations, and macro indicators into a state vector and insert daily:
   ```python
   index.bulk_insert(vectors=state_vectors, timestamps=trading_days, entity_ids=strategy_ids)
   ```

2. **Query similar historical periods.** Search for past states that resemble today, balancing similarity and recency:
   ```python
   matches = index.search(vector=today_state, k=20, alpha=0.3, query_timestamp=today)
   ```

3. **Retrieve forward trajectories.** For each historical match, pull the trajectory of what happened next:
   ```python
   forward_trajs = [index.trajectory(entity_id=m.entity_id) for m in matches]
   ```

4. **Compute path signatures.** Reduce each forward trajectory to a fixed-dimensional shape descriptor:
   ```python
   sigs = [cvx.path_signature(t, depth=2, time_augmentation=True) for t in forward_trajs]
   ```

5. **Classify dominant post-match pattern.** Cluster signatures to identify the most common forward regime:
   ```python
   # Compare signature distances to identify dominant pattern
   distances = [[cvx.signature_distance(a, b) for b in sigs] for a in sigs]
   ```

6. **Inform position sizing.** Use the dominant forward pattern and its frequency to calibrate conviction and position size.

This pipeline answers the question: *"Given that the market looks like this today, what usually happened next, and how confident should we be?"*

## References

- **Lopez de Prado, M.** — Information-driven sampling methods for financial machine learning, including triple-barrier labeling and entropy-based feature selection for non-stationary financial data.
- **Hamilton, J. D., Waggoner, D. F., & Zha, T. (2016)** — Regime-switching models for identifying structural breaks in financial time series and macroeconomic data.
