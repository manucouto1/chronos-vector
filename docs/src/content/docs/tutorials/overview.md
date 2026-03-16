---
title: "Tutorials Overview"
description: "Interactive Jupyter notebooks demonstrating ChronosVector's analytical capabilities"
---

The ChronosVector tutorial series consists of interactive Jupyter notebooks that demonstrate practical applications of temporal vector analytics. Each tutorial is structured as a scientific paper with executable code, rich visualizations, and measurable metrics.

## Tutorial Families

### Family B: Early Risk Detection in Social Media

| # | Tutorial | Key Methods | Status |
|---|---------|-------------|--------|
| **B1** | [Linguistic Trajectory Analysis for Mental Health](/tutorials/b1-mental-health) | Velocity, CPD, drift attribution, Hurst | Available |
| B2 | Early Warning Signals: Change Point Detection in User Behavior | PELT, BOCPD, prefix features | Planned |
| B3 | Drift Attribution: Explaining What Changed and When | Per-dimension analysis, Pareto | Planned |
| B4 | Stochastic Characterization of User Trajectories | Hurst, ADF/KPSS, OU process | Planned |
| B5 | Predictive Modeling: Neural ODE for Crisis Anticipation | Neural ODE, TorchScript | Planned |
| B6 | Cohort Divergence: When At-Risk Groups Separate | Pairwise distance, CPD | Planned |
| B7 | Multi-Scale Temporal Analysis | Multi-scale resampling | Planned |

### Family A: Quantitative Finance

| # | Tutorial | Key Methods | Status |
|---|---------|-------------|--------|
| A1 | Market Regime Detection via Temporal kNN | GARCH, temporal features | Planned |
| A2 | Factor Decay & Mean Reversion Analysis | ADF/KPSS, Ornstein-Uhlenbeck | Planned |
| A3 | Volatility Clustering in Embedding Space | GARCH(1,1), rough volatility | Planned |
| A4 | Path Signatures for Portfolio Classification | Rough path theory | Planned |
| A5 | Change Point Detection for Regime Shifts | PELT, BOCPD | Planned |
| A6 | Temporal Analogy Queries | Diachronic semantics | Planned |

### Transversal

| # | Tutorial | Key Methods | Status |
|---|---------|-------------|--------|
| T1 | Model Monitoring: Detecting Embedding Drift | MLOps, BOCPD | Planned |
| T2 | Interpretability: From Vectors to Narratives | Attribution, PCA, heatmaps | Planned |

## Running the Notebooks

```bash
# Install dependencies
pip install chronos-vector matplotlib numpy seaborn plotly pandas scipy scikit-learn

# Launch Jupyter
cd notebooks/
jupyter notebook
```

## Structure

Each notebook follows scientific paper structure:
1. **Abstract** — Problem statement and key results
2. **Introduction** — Scientific context and motivation
3. **Setup** — Dependencies and data preparation
4. **Methodology** — Step-by-step analysis with executable code
5. **Results** — Visualizations, metrics, tables
6. **Discussion** — Interpretation, limitations, clinical/financial implications
7. **References** — Academic citations
