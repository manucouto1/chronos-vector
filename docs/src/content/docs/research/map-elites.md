---
title: "MAP-Elites Quality-Diversity Archive"
description: "HNSW as adaptive niche discovery for quality-diversity optimization"
---

## Abstract

Quality-diversity (QD) algorithms like MAP-Elites maintain archives of diverse, high-performing solutions. Traditional implementations use fixed grid discretizations that scale poorly with behavior space dimensionality. We demonstrate that CVX's HNSW graph naturally discovers adaptive niches via its hierarchical region structure, replacing rigid grids with data-driven clusters. On a synthetic D=20 benchmark with 10K solutions over 200 generations, CVX provides O(log N) novelty scoring, topology-aware archive analysis, and trajectory-based generation tracking.

## Related Work

- **MAP-Elites** (Mouret & Clune, 2015): Quality-diversity via behavior-performance grid
- **CVT-MAP-Elites** (Vassiliades et al., 2018): Centroidal Voronoi tessellation for continuous behavior spaces
- **HNSW for QD**: Novel application — HNSW's hierarchical structure provides natural multi-resolution niches

## Methodology

| Component | Details |
|-----------|---------|
| Data | Synthetic: D=20, 10K solutions, 200 generations |
| Entities | Individual solutions in the archive |
| Vectors | D=20 behavior descriptors |
| CVX Functions | `regions`, `region_members`, `trajectory`, `topological_features`, `path_signature` |

### CVX Pipeline

1. Insert solutions as temporal points (generation = timestamp)
2. HNSW regions at multiple levels → adaptive niches (replaces CVT)
3. Novelty scoring via `search()` — O(log N) per query
4. Track archive evolution via `region_trajectory()` and `topological_features()`

## Key Results

- HNSW regions match CVT-style niches without pre-specification
- Topology reveals archive fragmentation and convergence phases
- Path signatures capture archive evolution dynamics

## Running the Notebook

```bash
conda activate cvx
jupyter notebook notebooks/T_MAP_Elites.ipynb
```

No external data required — generates synthetic data.

See the [full tutorial with Plotly plots](/tutorials/map-elites).

---

[← Back to White Paper](/research/white-paper)
