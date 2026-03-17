---
title: "Drug Discovery"
description: "Tracking chemical space exploration, campaign comparison, and SAR analysis with temporal vectors"
---

Drug discovery campaigns are **iterative temporal processes**. A team screens a compound library, selects hits, optimizes leads, and progressively narrows focus in chemical space across multiple screening rounds. Understanding *how* a campaign navigated chemical space is as important as where it ended up. Campaign comparison, chemical series tracking, and structure-activity relationship (SAR) evolution all require temporal-aware analysis that standard fingerprint databases lack.

ChronosVector (CVX) treats each compound as a vector (molecular fingerprint), each screening round as a timestamp, and each campaign as an entity trajectory. The result is a system that can track chemical space exploration, detect when a campaign shifted focus, compare campaigns quantitatively, and reveal the topological structure of active chemical space -- all with O(log N) similarity search over libraries of 10^6 to 10^9 molecules via HNSW.

---

## Data Model

| Drug Discovery Concept | CVX Abstraction |
|---|---|
| Compound | Vector (molecular fingerprint, D~1024-2048) |
| Screening round | Timestamp |
| Campaign | Entity trajectory |
| Chemical cluster / series | HNSW region |
| Hit-to-lead optimization | Trajectory through chemical space |
| SAR evolution | Region distribution change over time |

Molecular fingerprints (ECFP4, Morgan, MACCS) map directly to CVX's float32 vectors. A 2048-bit ECFP4 fingerprint becomes a 2048-dimensional vector; CVX's scalar quantization (SQ8) compresses this to ~2 KB per compound while preserving Tanimoto-correlated distance structure.

---

## Ingesting a Compound Library

Each compound enters the index with its campaign ID as the entity, the screening round as the timestamp, and its fingerprint as the vector.

```python
import chronos_vector as cvx
import numpy as np

# Create index sized for molecular fingerprints
index = cvx.TemporalIndex(m=32, ef_construction=200, ef_search=64)
index.enable_quantization(vmin=0.0, vmax=1.0)  # fingerprints are binary/count vectors

# compound_df columns: campaign_id, screening_round (unix ts), fingerprint (np.float32)
n = index.bulk_insert(
    entity_ids=compound_df["campaign_id"].values,
    timestamps=compound_df["screening_round"].values,
    vectors=np.stack(compound_df["fingerprint"].values),
    ef_construction=50,
)
# => "Ingested 1,247,803 points in 8.4s (148,548 pts/sec)"
```

With `m=32` and SQ8, a 2M-compound library fits in ~5 GB of RAM with full HNSW connectivity.

---

## Similarity Search in Chemical Space

Find the 10 nearest neighbors to a query compound in fingerprint space. HNSW provides O(log N) search, making this practical even over billion-scale virtual libraries.

```python
# Query: a hit compound from HTS
query_fp = compute_ecfp4(hit_smiles)  # => np.float32, shape (2048,)

results = index.search(query_fp, k=10)
# => [(campaign_id, screening_round, distance), ...]
# Distances correlate with 1 - Tanimoto similarity for binary fingerprints
```

---

## Chemical Series Discovery via HNSW Regions

HNSW's hierarchical graph naturally clusters chemically similar compounds. Higher levels yield coarser groupings that correspond to chemical series or scaffolds.

```python
# Discover chemical clusters at different granularities
for level in [1, 2, 3]:
    regions = index.regions(level=level)
    print(f"Level {level}: {len(regions)} chemical clusters")
# Level 1: 12,481 clusters (individual scaffolds)
# Level 2:  1,847 clusters (chemical series)
# Level 3:    142 clusters (broad chemotypes)

# Inspect a chemical series
regions_l2 = index.regions(level=2)
# => list of (region_id, centroid_vec, n_members)

members = index.region_members(region_id=42, level=2)
# => list of (campaign_id, screening_round) tuples in this series

# The centroid fingerprint represents the "average" compound in the series
# Use it for scaffold analysis or to seed further library enumeration
```

---

## Campaign Trajectory

Retrieve the full path a campaign traced through chemical space across screening rounds.

```python
traj = index.trajectory(entity_id="campaign_AZ_2024_kinase")
# => [(round_1_ts, fp_vec_1), (round_2_ts, fp_vec_2), ...]
# Sorted chronologically: each vector is the centroid of compounds screened in that round

# Trajectory length and span
print(f"Rounds: {len(traj)}, span: {traj[-1][0] - traj[0][0]:.0f} seconds")
```

---

## Detecting Campaign Focus Shifts

Change point detection identifies screening rounds where the campaign underwent a statistically significant shift in chemical space -- for example, when a team pivoted from one scaffold to another.

```python
changepoints = cvx.detect_changepoints(
    entity_id="campaign_AZ_2024_kinase",
    trajectory=traj,
    penalty=None,       # auto-calibrate via BIC
    min_segment_len=3,  # at least 3 rounds per segment
)
# => [(round_ts, severity), ...]
# severity: magnitude of the shift in embedding space

for ts, severity in changepoints:
    print(f"Focus shift at round {ts}: severity={severity:.4f}")
# Focus shift at round 1710288000: severity=0.3421  (pivot from aminopyridines to indazoles)
# Focus shift at round 1718064000: severity=0.1892  (narrowing within indazole series)
```

---

## Chemical Space Drift

Measure how much the campaign's chemical footprint has migrated between early and late screening rounds using optimal transport.

```python
# Region trajectory: probability distribution over chemical clusters per time window
reg_traj = index.region_trajectory(
    entity_id="campaign_AZ_2024_kinase",
    level=2,
    window_days=30,
    alpha=0.3,
)

p_early = reg_traj[0][1]   # distribution over clusters, first window
q_late  = reg_traj[-1][1]  # distribution over clusters, last window
centroids = [c for _, c, _ in index.regions(level=2)]

# Wasserstein drift (respects chemical space geometry)
wd = cvx.wasserstein_drift(list(p_early), list(q_late), centroids, n_projections=50)
print(f"Chemical space drift (Wasserstein): {wd:.4f}")
# => 0.2847 — substantial migration from broad screening to focused optimization
```

---

## Campaign Comparison via Path Signatures

Path signatures provide a universal, order-aware fingerprint of a campaign's trajectory. Comparing two campaigns reduces to comparing their signatures.

```python
# Compute signatures on region trajectories (Level 3 for tractable dimensions)
rt_a = index.region_trajectory(
    entity_id="campaign_AZ_2024_kinase", level=3, window_days=30, alpha=0.3
)
rt_b = index.region_trajectory(
    entity_id="campaign_PF_2023_kinase", level=3, window_days=30, alpha=0.3
)

sig_a = cvx.path_signature(
    [(t, [float(x) for x in d]) for t, d in rt_a],
    depth=2, time_augmentation=True,
)
sig_b = cvx.path_signature(
    [(t, [float(x) for x in d]) for t, d in rt_b],
    depth=2, time_augmentation=True,
)

d = cvx.signature_distance(sig_a, sig_b)
print(f"Campaign similarity (signature distance): {d:.4f}")
# => 0.0731 — these two kinase campaigns followed similar chemical space strategies
```

A low signature distance means the campaigns explored chemical space in a similar order and at a similar pace -- regardless of whether they screened the same compounds.

---

## Fisher-Rao Distance Between Activity Profiles

Fisher-Rao distance measures the geodesic distance on the statistical manifold of chemical activity distributions. It is reparameterization-invariant, making it robust to differences in library size or screening format.

```python
# Compare activity profiles between two campaigns at the same time point
p_activity = reg_traj_a[-1][1]  # campaign A's final chemical distribution
q_activity = reg_traj_b[-1][1]  # campaign B's final chemical distribution

fr = cvx.fisher_rao_distance(list(p_activity), list(q_activity))
print(f"Fisher-Rao distance: {fr:.4f}")  # d in [0, pi]
# => 0.4102 — moderate divergence in chemical focus areas
```

---

## Topological Analysis of Active Chemical Space

Persistent homology reveals the shape of the active chemical space: how many disconnected series exist, how well-separated they are, and whether the space is fragmenting or converging over the course of a campaign.

```python
centroids = [c for _, c, _ in index.regions(level=2)]

topo = cvx.topological_features(
    centroids, n_radii=30, persistence_threshold=0.05
)
print(f"Chemical clusters (connected components): {topo['n_components']}")
print(f"Max persistence: {topo['max_persistence']:.4f}")
print(f"Persistence entropy: {topo['persistence_entropy']:.4f}")
# Chemical clusters: 7
# Max persistence: 0.4231  (dominant chemotype is well-separated)
# Persistence entropy: 1.8934 (moderate diversity across chemotypes)

# The Betti curve shows how chemical space fragments as you tighten
# the similarity threshold — useful for deciding cluster cutoffs
# topo['betti_curve'], topo['radii']
```

A campaign that converges toward a single lead series will show decreasing `n_components` and `persistence_entropy` over time. A diverging campaign (exploring multiple scaffolds in parallel) shows the opposite.

---

## Full Campaign Analysis Pipeline

Putting it all together: load a compound library, track a campaign, detect pivots, compare with historical successes, and monitor topology.

```python
import chronos_vector as cvx
import numpy as np

# 1. Create index and ingest compound library
index = cvx.TemporalIndex(m=32, ef_construction=200, ef_search=64)
index.enable_quantization(vmin=0.0, vmax=1.0)
index.bulk_insert(entity_ids, timestamps, fingerprints)

# 2. Track campaign over screening rounds
traj = index.trajectory(entity_id="campaign_AZ_2024_kinase")

# 3. Identify when focus shifted
changepoints = cvx.detect_changepoints(
    "campaign_AZ_2024_kinase", traj, min_segment_len=3
)

# 4. Measure chemical space drift (early vs. late)
reg_traj = index.region_trajectory(
    entity_id="campaign_AZ_2024_kinase", level=2, window_days=30, alpha=0.3
)
centroids = [c for _, c, _ in index.regions(level=2)]
drift = cvx.wasserstein_drift(
    list(reg_traj[0][1]), list(reg_traj[-1][1]), centroids
)

# 5. Compare with a historically successful campaign
rt_ref = index.region_trajectory(
    entity_id="campaign_SUCCESS_2022_kinase", level=3, window_days=30, alpha=0.3
)
sig_current = cvx.path_signature(
    [(t, [float(x) for x in d]) for t, d in
     index.region_trajectory("campaign_AZ_2024_kinase", level=3, window_days=30, alpha=0.3)],
    depth=2, time_augmentation=True,
)
sig_ref = cvx.path_signature(
    [(t, [float(x) for x in d]) for t, d in rt_ref],
    depth=2, time_augmentation=True,
)
similarity = cvx.signature_distance(sig_current, sig_ref)

# 6. Monitor chemical space topology
topo = cvx.topological_features(centroids, n_radii=30, persistence_threshold=0.05)

print(f"Campaign: campaign_AZ_2024_kinase")
print(f"  Screening rounds:      {len(traj)}")
print(f"  Focus shifts detected: {len(changepoints)}")
print(f"  Chemical space drift:  {drift:.4f}")
print(f"  Similarity to ref:     {similarity:.4f}")
print(f"  Active chemotypes:     {topo['n_components']}")
print(f"  Persistence entropy:   {topo['persistence_entropy']:.4f}")
```

---

## CVX Functions Used

| Function | Role in Drug Discovery |
|---|---|
| `bulk_insert` | Ingest compound library with screening-round timestamps |
| `search` | k-NN similarity search in fingerprint space |
| `regions` | Discover chemical clusters / series from HNSW hierarchy |
| `region_members` | List compounds belonging to a chemical series |
| `trajectory` | Full path of a campaign through chemical space |
| `detect_changepoints` | Identify screening rounds where focus shifted |
| `region_trajectory` | Campaign's distribution over chemical clusters over time |
| `wasserstein_drift` | Chemical space migration between time windows |
| `fisher_rao_distance` | Geodesic distance between activity profiles |
| `path_signature` | Universal trajectory fingerprint for campaign comparison |
| `signature_distance` | Distance between campaign signatures |
| `topological_features` | Shape of active chemical space (clusters, persistence) |

---

## References

1. Vogt, M. & Bajorath, J. (2016). Molecular Fingerprint Similarity Search. *Chemoinformatics*, Humana Press.
2. Gao, W. et al. (2025). Chemical Space Navigation with Generative AI. *Proceedings of the National Academy of Sciences*.
