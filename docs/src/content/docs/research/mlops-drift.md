---
title: "MLOps Drift Detection"
description: "Production model drift detection with 5 independent CVX signals"
---

## Abstract

ML models in production silently degrade as input distributions shift. Traditional monitoring relies on single-metric thresholds (e.g., PSI > 0.2). CVX provides 5 independent drift signals computed natively from the embedding trajectory: velocity magnitude, distributional distance (Wasserstein/Fisher-Rao), change point detection, Hurst exponent, and topological changes. On a synthetic 3-phase benchmark (stable → gradual drift → sudden shift, D=64, 18K embeddings over 90 days), CVX detects both gradual and sudden shifts with complementary signals that single metrics miss.

## Related Work

- **Data drift detection** (Rabanser et al., 2019): Statistical tests for distribution shift
- **Population Stability Index (PSI)**: Industry standard for production monitoring
- **Evidently AI, NannyML**: Open-source drift detection tools (threshold-based)
- **CVX advantage**: Trajectory-based analysis captures drift *dynamics* (speed, persistence, topology) not just magnitude

## Methodology

| Component | Details |
|-----------|---------|
| Data | Synthetic: D=64, 200 points/day, 90 days, 3 phases |
| Phases | Stable (days 1-30), Gradual drift (31-60), Sudden shift (61-90) |
| Entities | Daily embedding batches |
| CVX Functions | `velocity`, `drift`, `wasserstein_drift`, `fisher_rao_distance`, `detect_changepoints`, `hurst_exponent`, `topological_features` |

### 5 Independent Drift Signals

1. **Velocity**: `cvx.velocity()` — rate of change in embedding space. Spikes at sudden shifts.
2. **Distributional distance**: `cvx.wasserstein_drift()` + `cvx.fisher_rao_distance()` — optimal transport between daily distributions.
3. **Change points**: `cvx.detect_changepoints()` — PELT identifies structural breaks.
4. **Hurst exponent**: `cvx.hurst_exponent()` — persistent drift (H>0.5) vs noise (H≈0.5).
5. **Topology**: `cvx.topological_features()` — Betti curve changes when cluster structure shifts.

## Key Results

- All 5 signals independently detect the sudden shift at day 60
- Gradual drift (days 30-60) is best captured by Wasserstein distance and Hurst exponent
- Combined signal provides earlier detection than any single metric
- Topology detects structural changes invisible to distance-based metrics

## Running the Notebook

```bash
conda activate cvx
jupyter notebook notebooks/T_MLOps_Drift.ipynb
```

No external data required — generates synthetic data.

See the [full tutorial with Plotly plots](/chronos-vector/tutorials/mlops-drift).

---

[← Back to White Paper](/chronos-vector/research/white-paper)
