---
title: "Domain Showcase"
description: "Exploratory applications of CVX across finance, anomaly detection, quality-diversity, and more"
---

## Overview

These applications demonstrate CVX's generality across domains beyond the primary research areas. They are **exploratory** — proof-of-concept demonstrations using synthetic or benchmark data, not production-validated pipelines.

## Available Showcases

### Quantitative Finance

Market regime detection using temporal trajectory analysis. Embeddings of price/volume features track regime transitions (bull/bear/sideways). CVX detects change points and measures drift between regimes.

**CVX features**: `detect_changepoints()`, `drift()`, `velocity()`, `region_trajectory()`

### Anomaly Detection (NAB)

Evaluation on the Numenta Anomaly Benchmark. CVX's change point detection (PELT/BOCPD) applied to time-series embedding trajectories for unsupervised anomaly detection.

**CVX features**: `detect_changepoints()`, `bocpd_observe()`, `event_features()`

### Quality-Diversity (MAP-Elites)

Using CVX as a behavioral archive for MAP-Elites evolutionary optimization. Embedding-based behavioral descriptors enable continuous behavioral spaces with HNSW-based niche assignment.

**CVX features**: `regions()`, `region_assignments()`, `search()`, `path_signature()`

### MLOps Drift Detection

Monitoring embedding model drift in production ML systems. CVX tracks how model outputs evolve across retraining cycles, detecting concept drift and measuring distribution shift.

**CVX features**: `cohort_drift()`, `wasserstein_drift()`, `fisher_rao_distance()`

### Molecular Dynamics & Drug Discovery

Trajectory analysis for molecular conformational dynamics and binding site evolution. Conceptual application of CVX's temporal primitives to molecular embedding spaces.

**CVX features**: `trajectory()`, `velocity()`, `frechet_distance()`, `topological_features()`

### Fraud Detection & Insider Threat

Behavioral trajectory analysis for detecting anomalous patterns in user activity embeddings. Change point detection identifies behavioral regime changes.

**CVX features**: `detect_changepoints()`, `discover_discords()`, `event_features()`

## Notebooks

| Notebook | Domain | Data |
|----------|--------|------|
| T_Finance_Regimes | Market regimes | Synthetic |
| T_NAB_Anomaly | Anomaly detection | NAB benchmark |
| T_MAP_Elites | Quality-diversity | Synthetic |
| T_MLOps_Drift | Model monitoring | Synthetic |
| T_Fraud_Detection | Fraud detection | Synthetic |
| T_Insider_Threat | Insider threat | Synthetic |
