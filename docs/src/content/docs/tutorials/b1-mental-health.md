---
title: "B1: Linguistic Trajectory Analysis for Mental Health"
description: "Detecting psychological distress through temporal embedding analysis — velocity, change points, drift attribution, and stochastic characterization"
---

> **Interactive notebook:** [`notebooks/B1_linguistic_trajectory_mental_health.ipynb`](https://github.com/manucouto1/chronos-vector/blob/develop/notebooks/B1_linguistic_trajectory_mental_health.ipynb)

## Abstract

Early detection of psychological distress from social media requires tracking how a user's language *evolves* over time, not just what they say at a single point. We demonstrate how ChronosVector enables continuous monitoring of linguistic embedding trajectories, using temporal velocity, drift attribution, change point detection, and stochastic characterization to identify early warning signals of mental health deterioration.

## Methods Used

| CVX Feature | Scientific Basis | Purpose |
|------------|-----------------|---------|
| `cvx.velocity()` | Finite differences (dv/dt) | Quantify rate of linguistic change |
| `cvx.detect_changepoints()` | PELT (Killick et al., 2012) | Detect regime transitions |
| `cvx.drift()` | L2 + per-dimension decomposition | Explain *what* changed |
| `cvx.hurst_exponent()` | DFA / R/S analysis | Classify trajectory dynamics |
| `cvx.predict()` | Linear / Neural ODE | Anticipate future states |
| `cvx.temporal_features()` | Fixed-size feature extraction | Enable downstream ML |

## User Archetypes

| Archetype | Description | Expected Signals |
|-----------|-------------|-----------------|
| Stable | Healthy baseline | Low velocity, H ≈ 0.5 |
| Gradual decline | Slow drift toward distress | Rising velocity, H > 0.5 |
| Acute crisis | Sudden shift | Change point, high severity |
| Cyclic | Recurring episodes | Mean-reverting, H < 0.5 |
| Recovery | Decline → improvement | Two change points |

## Key Visualizations

The notebook produces:
- **PCA trajectory plots** — 2D projection of 32D embedding paths
- **Velocity time series** — Rate of change over time per user
- **Change point overlay** — Detected vs ground truth transitions
- **Dimension heatmap** — Semantic dimensions × time (crisis user)
- **Hurst exponent bar chart** — Stochastic classification
- **Prediction plots** — 30-day forecast per semantic dimension
- **Feature matrix heatmap** — Fixed-size features across archetypes

## References

1. Couto, M. et al. (2025). Temporal word embeddings for psychological disorder early detection. *JHIR*.
2. Coppersmith, G. et al. (2018). NLP of social media as screening for suicide risk. *BMI Insights*.
3. De Choudhury, M. et al. (2013). Predicting depression via social media. *ICWSM*.
4. Killick, R. et al. (2012). Optimal detection of changepoints. *JASA*.
5. Chen, R.T.Q. et al. (2018). Neural ODEs. *NeurIPS*.
