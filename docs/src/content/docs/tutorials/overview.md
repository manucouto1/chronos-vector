---
title: "Tutorials"
description: "Interactive notebooks demonstrating how ChronosVector captures, analyzes, and quantifies transformation in high-dimensional spaces"
---

## The Dimension of Change

Every domain has entities that **transform over time** — a patient's language evolves, a market regime shifts, a molecule folds, an ML model drifts. Traditional vector databases store snapshots. ChronosVector stores *trajectories* and provides the mathematical tools to understand their shape.

The temporal dimension is not just "when" — it is the **dimension of change, evolution, and transformation**. CVX's 17 analytical functions form a layered framework for understanding how entities move through embedding spaces:

| Level | Question | CVX Functions | What It Reveals |
|-------|----------|---------------|-----------------|
| 1 | Where has the entity been? | `trajectory`, `search` | The raw path through embedding space |
| 2 | How fast is it changing? | `velocity`, `drift` | Rate and direction of transformation |
| 3 | Is change persistent or erratic? | `hurst_exponent` | Long-range dependence: trending vs oscillating |
| 4 | When did regime transitions happen? | `detect_changepoints` | Structural breaks in behavior |
| 5 | How does the distribution transform? | `region_trajectory`, `wasserstein_drift`, `fisher_rao_distance` | Semantic migration between topics/clusters |
| 6 | What is the *shape* of the transformation? | `path_signature`, `signature_distance` | Universal nonlinear trajectory fingerprint |
| 7 | How does the topology evolve? | `topological_features` | Fragmentation, convergence, structural change |

Each tutorial applies this framework to a different domain, demonstrating that the **same mathematical tools** reveal transformation across fundamentally different data.

---

## Available Tutorials

| Tutorial | Domain | Data | Key Insight | Status |
|----------|--------|------|-------------|--------|
| [Mental Health Explorer](/chronos-vector/tutorials/b1-explorer) | Clinical NLP | eRisk (D=768) | 13 CVX temporal features → F1=0.60 (AUC=0.64); early detection at 10% of posts | Available |
| [Clinical Anchoring](/chronos-vector/tutorials/b2-clinical-anchoring) | Clinical NLP | eRisk (D=768) | DSM-5 anchor projection: F1=0.744 (AUC=0.886) on temporal split, early detection at 10% | Available |
| [MAP-Elites Archive](/chronos-vector/tutorials/map-elites) | Quality-Diversity | Synthetic (D=20) | HNSW replaces CVT with adaptive niches; archive topology reveals exploration structure | Available |
| [MLOps Drift Detection](/chronos-vector/tutorials/mlops-drift) | Production ML | Synthetic (D=64) | 5 independent drift signals detect gradual and sudden distribution shifts | Available |

### Planned

| Tutorial | Domain | Focus |
|----------|--------|-------|
| Molecular Dynamics | Computational Chemistry | Conformational clustering via graph regions, trajectory comparison |
| Drug Discovery | Medicinal Chemistry | Campaign navigation through chemical space |
| Quantitative Finance | Trading | Market regime detection, path-dependent analysis |

---

## Running the Tutorials

Each tutorial is a self-contained Jupyter notebook in the `notebooks/` directory.

```bash
# Setup
conda activate cvx
cd crates/cvx-python && maturin develop --release && cd ../..

# Launch
jupyter notebook notebooks/T_MAP_Elites.ipynb
```

The Mental Health tutorial requires the eRisk dataset (see [Research Protocol 001](https://github.com/manucouto1/chronos-vector/blob/develop/design/CVX_Research_Protocol_001_Mental_Health.md)). The MAP-Elites and MLOps tutorials generate synthetic data and require no external files.

---

## From Examples to Tutorials

The [Examples](/chronos-vector/examples/overview) section provides a **quick API reference** for each of CVX's 17 functions with concise code snippets. The tutorials below go deeper: each is a complete, executable analysis that follows the 7-level framework above, showing real outputs, interactive visualizations, and domain-specific interpretation.

| Need | Go to |
|------|-------|
| "How do I call `path_signature()`?" | [Examples → API Reference](/chronos-vector/examples/overview) |
| "How does path signature analysis work on real data?" | [Tutorials](/chronos-vector/tutorials/overview) |
| "What functions does CVX have for distributional analysis?" | [Temporal Analytics Toolkit](/chronos-vector/specs/temporal-analytics) |
