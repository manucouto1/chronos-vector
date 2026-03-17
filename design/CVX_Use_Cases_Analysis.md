# ChronosVector: Cross-Domain Use Case Analysis

**Status**: Research document
**Created**: 2026-03-17
**Authors**: Manuel Couto Pintos
**Purpose**: Identify real computational bottlenecks across domains that a temporal vector database can resolve, and evaluate which capabilities justify implementation.

---

## 1. The Core Abstraction

ChronosVector stores **(entity, timestamp, vector)** tuples in a hierarchical navigable graph (HNSW). The temporal dimension is first-class: queries can filter by time, entities have trajectories, and the graph provides natural multi-scale clustering.

The question is: **which real-world problems map to this abstraction and have computational bottlenecks that the graph structure resolves?**

---

## 2. Domain Analysis

### 2.1 Quality-Diversity Optimization (MAP-Elites)

#### The Problem

MAP-Elites and its variants (CVT-MAP-Elites, CMA-ME) maintain an **archive** of diverse, high-performing solutions. The archive is indexed by a **behavioral descriptor** — a vector that characterizes *how* a solution behaves, not just *how well*.

The critical operation at every generation is:

```
For each new solution:
  1. Compute behavioral descriptor (vector)
  2. Find which archive niche it belongs to → kNN in descriptor space
  3. Compare fitness with current niche occupant
  4. Replace if better (or add to unstructured archive)
```

**Current bottleneck:** Step 2 is O(N) for unstructured archives (linear scan) or O(k) for CVT archives (nearest centroid). For archives with >100K solutions in high-dimensional descriptor spaces (>20D), this dominates computation time.

CVT-MAP-Elites pre-computes k centroids, but: (a) centroid positions are fixed and may not match the actual solution distribution, (b) the number of cells must be chosen a priori, (c) cell volume grows exponentially with dimensionality (Vassiliades et al., 2018).

#### How CVX Maps

| MAP-Elites concept | CVX equivalent |
|-------------------|----------------|
| Archive | HNSW index |
| Behavioral descriptor | Vector |
| Generation | Timestamp |
| Find niche | `search(descriptor, k=1)` → O(log N) |
| Novelty score | `search(descriptor, k=15)` → mean distance |
| Archive regions | `regions(level)` → natural niches |
| Niche occupant | `region_members(niche_id)` |
| Solution lineage | `trajectory(entity_id)` → evolution over generations |

**What CVX provides today:**
- kNN in O(log N) instead of O(N) ✓
- Hierarchical regions without pre-specifying K ✓
- Temporal tracking of solutions across generations ✓

**What CVX needs:**
- **Batch insert** — each generation produces hundreds of new solutions
- **Region members** — to find the current occupant(s) of a niche
- **Conditional insert** — "insert only if better than current occupant" (application logic, but batch insert enables it efficiently)

**Quantified value:**
- Archive of 1M solutions, 50D descriptors, 1000 generations
- Current: ~2s per generation for novelty scoring (O(N) × population_size)
- With HNSW: ~2ms per generation (O(log N) × population_size)
- **~1000× speedup on the archive operation**

**References:**
- Vassiliades, V. et al. (2018). Using Centroidal Voronoi Tessellations to Scale Up MAP-Elites. *IEEE Trans. Evol. Comp.*
- Mouret, J.-B. & Clune, J. (2015). Illuminating search spaces by mapping elites. *arXiv:1504.04909*.
- Fontaine, M. et al. (2020). Covariance Matrix Adaptation MAP-Elites. *GECCO*.

---

### 2.2 Molecular Dynamics Trajectory Analysis

#### The Problem

MD simulations generate trajectories of millions of conformations. Each conformation is a high-dimensional vector (e.g., pairwise Cα distances: D ~ 1000 for a 50-residue protein). Key analyses:

1. **Conformational clustering** — identify stable states (folded, unfolded, intermediate)
2. **Transition detection** — when does the molecule change state?
3. **Trajectory comparison** — do two simulations visit the same states?

**Current bottleneck:** Pairwise distance matrix is O(N²). For a 100ns simulation with 10⁶ frames, this is 10¹² distance computations. Even with downsampling, it remains the dominant cost.

Recent work (MOSCITO, 2024; fast conformational clustering, 2023) addresses this with temporal-aware methods that exploit the sequential nature of MD trajectories — **exactly the setting CVX is designed for**.

#### How CVX Maps

| MD concept | CVX equivalent |
|-----------|----------------|
| Conformation | Vector (fingerprint) |
| Simulation timestep | Timestamp |
| Molecule/simulation run | Entity |
| Conformational state | Region at level L |
| State transition | Change point on region trajectory |
| Metastable state | Region with high self-transition probability |

**What CVX provides today:**
- Hierarchical clustering via graph regions — O(N log N) construction vs O(N²) pairwise ✓
- Temporal ordering is native ✓
- Change point detection on trajectories ✓
- Region trajectory shows state transitions over time ✓

**What CVX needs:**
- **Batch insert** — millions of frames per simulation
- **Region members** — "which conformations belong to state X?"
- **Trajectory similarity** — "which simulations visited similar states?"

**Quantified value:**
- 10⁶ frames, D=1000
- Pairwise clustering: O(N²) = 10¹² operations, hours on GPU
- HNSW region discovery: O(N log N) ≈ 2×10⁷ operations, minutes on CPU
- **~50,000× reduction in clustering cost**

**References:**
- Molgedey, L. & Schuster, H.G. (1994). Separation of a mixture of independent signals. *Physical Review Letters*.
- MOSCITO (2024). Temporal Subspace Clustering for MD Data. *arXiv:2408.00056*.
- Shao, J. et al. (2007). Clustering MD Trajectories. *J. Chem. Theory Comput.*
- Fast conformational clustering (2023). *J. Chem. Phys.* 158(14).

---

### 2.3 Surrogate-Assisted Expensive Optimization

#### The Problem

When fitness evaluation is expensive (CFD, finite element, wet lab), surrogate models approximate fitness from previously evaluated solutions. **kNN is one of the most popular surrogates** because it's non-parametric and doesn't require training:

```
fitness_approx(x) = weighted_mean(fitness(x_i) for x_i in kNN(x, archive))
```

**Current bottleneck:** Each surrogate query is O(N) where N is the number of previously evaluated solutions. In high-dimensional design spaces (>50D), this grows with every generation. For multi-objective problems, archive sizes of 10K-100K are common.

#### How CVX Maps

| SAEO concept | CVX equivalent |
|-------------|----------------|
| Evaluated solution | Point (vector + fitness as metadata) |
| Optimization iteration | Timestamp |
| kNN surrogate query | `search(candidate, k)` → O(log N) |
| Model management region | `regions(level)` → subregions for local surrogates |
| Convergence tracking | `velocity()`, `hurst_exponent()` on best-fitness trajectory |

**What CVX provides today:**
- O(log N) kNN for surrogate queries ✓
- Temporal tracking of evaluations ✓
- Velocity/Hurst for convergence diagnosis ✓

**What CVX needs:**
- **Batch insert** — add all new evaluations per generation at once
- **Region members** — build local surrogates per subregion of the search space

**Quantified value:**
- Archive of 50K evaluations, 100D design space, 500 generations
- kNN query per candidate: O(N)=50K distance computations → O(log N)≈17 with HNSW
- **~3,000× speedup per surrogate query**

**References:**
- Jin, Y. (2011). Surrogate-Assisted Evolutionary Computation: Recent Advances and Future Challenges. *Swarm & Evolutionary Computation*.
- Survey of SAEAs for Expensive Combinatorial Optimization (2024). *Complex & Intelligent Systems*.

---

### 2.4 Concept Drift Detection in Production ML

#### The Problem

Deployed ML models degrade when the data distribution shifts (concept drift). Monitoring requires comparing incoming embedding distributions against reference distributions over time.

**Current approaches:**
- DriftLens (2024): extract embeddings → PCA → fit Gaussian → detect shift
- Population Stability Index: histogram comparison between windows
- MMD (Maximum Mean Discrepancy): kernel-based distribution comparison

**Bottleneck:** Computing drift statistics requires comparing each new batch against the reference. For streaming data with millions of embeddings, this is expensive.

#### How CVX Maps

| Drift detection concept | CVX equivalent |
|------------------------|----------------|
| Embedding stream | Insert with current timestamp |
| Reference distribution | Region distribution at T₀ |
| Current distribution | Region distribution at T_now |
| Drift metric | `drift()` between distributions |
| Drift onset | `detect_changepoints()` on region trajectory |
| Affected subpopulation | `region_members(changed_region, time_range)` |

**What CVX provides today:**
- Continuous ingestion with timestamps ✓
- Region distributions over time (region_trajectory) ✓
- Change point detection ✓
- Drift quantification ✓

**What CVX needs:**
- **Batch insert** — streaming batches of embeddings
- **Region members** — inspect which data points are in the drifting cluster

**This is one of the strongest use cases** because it uses nearly every CVX feature in its intended way, and drift detection is a growing market need (MLOps).

**References:**
- DriftLens (2024). Unsupervised Real-time Concept Drift Detection. *arXiv:2406.17813*.
- Gama, J. et al. (2014). A Survey on Concept Drift Adaptation. *ACM Computing Surveys*.

---

### 2.5 Drug Discovery Campaigns

#### The Problem

A drug discovery campaign is an iterative optimization process:
1. Screen library of compounds (molecular fingerprints as vectors)
2. Select hits based on activity, selectivity, ADMET properties
3. Optimize hits → lead compounds → candidates
4. Each iteration narrows the chemical space explored

**Bottleneck:** Similarity search in large compound libraries (10⁶-10⁹ molecules). Also: tracking which regions of chemical space have been explored, and how the campaign navigated through them.

#### How CVX Maps

| Drug discovery concept | CVX equivalent |
|-----------------------|----------------|
| Compound | Vector (fingerprint) |
| Screening round | Timestamp |
| Campaign | Entity trajectory |
| Chemical cluster | Region |
| Hit-to-lead optimization | Trajectory in chemical space |
| Structure-Activity Relationship | Region distribution evolution |

**What CVX provides:**
- Similarity search in chemical space ✓
- Tracking campaign navigation over time ✓
- Regions = chemical series or scaffolds ✓
- "When did the campaign shift focus?" = change point ✓

**What CVX needs:**
- **Batch insert** — loading compound libraries
- **Trajectory similarity** — "which campaigns followed a similar path?"

**References:**
- Chemical Space Navigation with Generative AI (2025). *PNAS*.
- Molecular Fingerprint Similarity Search in Virtual Screening (2016). *Expert Opinion on Drug Discovery*.

---

## 3. Cross-Domain Capability Matrix

| Capability | MAP-Elites | MD Trajectories | SAEOs | Drift Detection | Drug Discovery |
|-----------|:----------:|:--------------:|:-----:|:---------------:|:--------------:|
| **kNN search** | Critical | Important | Critical | Important | Critical |
| **Batch insert** | Critical | Critical | Important | Critical | Critical |
| **Region members** | Critical | Critical | Important | Important | Useful |
| **Trajectory similarity** | Useful | Critical | Useful | Useful | Important |
| **Region trajectory** | Useful | Critical | Useful | Critical | Important |
| **Change point detection** | — | Critical | Useful | Critical | Useful |
| **Velocity/Hurst** | — | Useful | Important | Useful | — |
| **Temporal filter** | Useful | Important | Useful | Critical | Important |

### Capabilities ranked by cross-domain impact

| Rank | Capability | Domains that need it | Already in CVX? |
|------|-----------|---------------------|-----------------|
| **1** | **Batch insert** | ALL 5 | No (P0 in RFC-005) |
| **2** | **Region members query** | 4 of 5 (critical in 2) | No (P1 in RFC-005) |
| **3** | **Trajectory similarity** | 4 of 5 (critical in 1) | No (P2 in RFC-005) |
| 4 | kNN search | ALL 5 | Yes ✓ |
| 5 | Region trajectory | 4 of 5 | Yes ✓ |
| 6 | Change point detection | 3 of 5 | Yes ✓ |
| 7 | Temporal filter | ALL 5 | Yes ✓ |
| 8 | Velocity/Hurst | 3 of 5 | Yes ✓ |

---

## 4. Evaluation: What Justifies Implementation

### Must implement (justified by ≥4 domains, critical in ≥2)

**Batch insert** — Universal need. Without it, CVX is a research prototype, not a tool. Every domain listed here involves loading thousands to millions of vectors. There is no workaround.

**Region members query** — The graph provides natural clustering. But if you can't ask "which points are in cluster X?", the clustering is useless for downstream analysis. This is the equivalent of a `SELECT * FROM table WHERE cluster = X` — a basic database operation.

### Should implement (justified by ≥3 domains, critical in ≥1)

**Trajectory similarity search** — The most differentiating capability. No existing vector database offers this. MD trajectory comparison alone is a real market: the paper "Fast conformational clustering" (J. Chem. Phys., 2023) addresses exactly this problem with custom algorithms. CVX could provide it as a built-in operation. Fréchet distance on region trajectories (K~80 dims instead of D~768) makes this tractable.

### Could implement later (useful but not blocking)

**Temporal neighbors** — Derivable from existing `search()` with per-timestamp queries. Convenience API, not a new capability. Implement when a specific user needs it.

**Conditional insert ("insert if better")** — Application logic for MAP-Elites. Can be built on `region_members()` + `insert()`. Not a DB-level operation.

### Should NOT implement

**Novelty score** — It's `mean(distances from search(x, k=15))`. One line of Python. Not a DB operation.

**Surrogate model** — Application logic. The DB provides kNN; the user builds the surrogate.

**Visualization** — Client concern. Plotly, D3, whatever. The DB returns data.

**Domain-specific processing** — Text analysis, molecular parsing, financial indicators. Preprocessing, not storage.

---

## 5. Implementation Priority (Updated)

Based on cross-domain analysis, the RFC-005 priorities are confirmed and reinforced:

| Priority | Capability | Justification |
|----------|-----------|---------------|
| **P0** | Batch insert + configurable ef | Blocks ALL domains. No adoption without this. |
| **P1** | Region members query | Blocks MAP-Elites and MD trajectory analysis. Enables "inspect cluster contents". |
| **P2** | Trajectory similarity (Fréchet) | Unique differentiator. No competitor has this. High value in MD, drug discovery, MLOps. |

---

## 6. What This Means for Tutorials

Each tutorial family should demonstrate CVX solving a **real computational bottleneck**:

| Tutorial | Domain | CVX solves... |
|----------|--------|---------------|
| **B1** | Mental health | Temporal trajectory analysis of social media users |
| **A1** | Finance | Regime detection via region trajectory + change points |
| **E1** (new) | MAP-Elites/QD | Archive as HNSW index, O(log N) novelty scoring |
| **M1** (new) | Molecular dynamics | Conformational clustering via graph regions |
| **D1** (new) | MLOps drift | Embedding drift detection via region distributions |

The existing tutorials (B1, A1) are valid. The new ones (E1, M1, D1) would demonstrate CVX's value proposition in domains with clear computational bottlenecks.

---

## References

1. Vassiliades, V. et al. (2018). Using Centroidal Voronoi Tessellations to Scale Up MAP-Elites. *IEEE Trans. Evol. Comp.*, 22(4).
2. Mouret, J.-B. & Clune, J. (2015). Illuminating search spaces by mapping elites. *arXiv:1504.04909*.
3. Fontaine, M. et al. (2020). CMA-ME. *GECCO*.
4. MOSCITO (2024). Temporal Subspace Clustering for Molecular Dynamics. *arXiv:2408.00056*.
5. Shao, J. et al. (2007). Clustering Molecular Dynamics Trajectories. *JCTC*, 3(6).
6. Fast conformational clustering (2023). *J. Chem. Phys.*, 158(14), 144109.
7. Jin, Y. (2011). Surrogate-Assisted Evolutionary Computation. *Swarm & Evolutionary Computation*, 1(2).
8. SAEAs for Expensive Combinatorial Optimization (2024). *Complex & Intelligent Systems*.
9. DriftLens (2024). Unsupervised Concept Drift Detection. *arXiv:2406.17813*.
10. Chemical Space Navigation (2025). *PNAS*.
11. Molecular Fingerprint Similarity Search (2016). *Expert Opin. Drug Discovery*, 11(2).
12. Malkov, Y.A. & Yashunin, D.A. (2018). HNSW graphs. *IEEE TPAMI*, 42(4).
13. Salvador, S. & Chan, P. (2007). FastDTW. *Intelligent Data Analysis*, 11(5).
14. Toohey, K. & Duckham, M. (2015). Trajectory Similarity Measures. *ACM SIGSPATIAL*.
15. Spatio-Temporal Trajectory Similarity Survey (2023). *arXiv:2303.05012*.
