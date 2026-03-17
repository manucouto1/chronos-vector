---
title: "Molecular Dynamics"
description: "Conformational clustering, state transition detection, and trajectory comparison for MD simulations"
---

Molecular Dynamics (MD) simulations produce trajectories of millions of conformations, each represented as a high-dimensional vector derived from pairwise C-alpha distances (D ~ 1000). Three analyses are fundamental to every MD study: **conformational clustering**, **state transition detection**, and **trajectory comparison across simulation runs**. All three hit the same computational wall: pairwise distance calculations scale O(N^2), and with 10^6 frames that means 10^12 operations -- hours even on a modern GPU.

ChronosVector (CVX) sidesteps this bottleneck entirely. HNSW-based region discovery provides O(N log N) conformational clustering, yielding roughly a **50,000x reduction** in computation compared to brute-force pairwise methods. Because CVX treats every conformation as a timestamped vector, the full temporal analysis toolkit -- velocity, change points, path signatures, topological features -- applies directly.

## Data Model

The mapping from MD concepts to CVX primitives is direct:

| MD concept | CVX equivalent |
|---|---|
| Conformation (C-alpha fingerprint) | Vector |
| Simulation timestep | Timestamp |
| Molecule / simulation run | Entity |
| Conformational state | Region at level L |
| State transition | Change point on region trajectory |
| Metastable state | Region with high self-transition probability |

Each frame of the simulation becomes a single `(entity_id, timestamp, vector)` tuple. Multiple simulation runs of the same molecule are separate entities, enabling direct trajectory comparison.

## Ingesting a Trajectory

```python
import chronos_vector as cvx
import numpy as np

# Load C-alpha distance matrices from trajectory frames
# conformations: np.ndarray of shape (n_frames, d) where d ~ 1000
conformations = np.load("calpha_distances.npy")
n_frames = conformations.shape[0]

# One entity per simulation run
entity_ids = np.zeros(n_frames, dtype=np.uint64)       # run 0
timesteps = np.arange(n_frames, dtype=np.uint64)        # frame index as timestamp

# Build the temporal index
index = cvx.TemporalIndex(m=16, ef_construction=200)

# Enable scalar quantization for memory efficiency at D=1000
min_val = conformations.min()
max_val = conformations.max()
index.enable_quantization(min_val, max_val)

# Bulk insert all frames
index.bulk_insert(entity_ids, timesteps, conformations, ef_construction=50)
```

With scalar quantization, a 10^6-frame trajectory at D=1000 occupies roughly 1 GB instead of 4 GB in full float32.

## Conformational Clustering via HNSW Regions

Traditional conformational clustering (k-means, GROMOS, Daura) requires an all-pairs distance matrix. CVX regions exploit the HNSW graph structure to identify dense conformational neighborhoods in O(N log N) time.

```python
# Extract regions at hierarchical level 2
regions = index.regions(level=2)

for rid, centroid, n_members in regions:
    members = index.region_members(rid, level=2)
    # Each region corresponds to a conformational state
    print(f"State {rid}: {n_members} frames, centroid norm = {np.linalg.norm(centroid):.2f}")
```

Higher levels yield coarser states (e.g., folded vs. unfolded), while lower levels resolve substates within a conformational basin.

## State Transition Detection

Change point detection on the entity trajectory identifies the exact frames where the molecule transitions between conformational states.

```python
# Retrieve the full trajectory for simulation run 0
traj = index.trajectory(entity_id=0)

# Detect conformational transitions
# min_segment_len prevents spurious transitions from thermal noise
changepoints = cvx.detect_changepoints(0, traj, min_segment_len=100)

for cp in changepoints:
    print(f"Transition at frame {cp.timestamp}, "
          f"magnitude = {cp.magnitude:.4f}")
```

Setting `min_segment_len=100` ensures that short-lived fluctuations (thermal noise) are not flagged as genuine transitions. For a 1 ns simulation at 1 ps timestep resolution, `min_segment_len=100` corresponds to a minimum dwell time of 100 ps.

## Transition Speed via Velocity

The velocity at a transition frame quantifies how rapidly the conformational change occurs -- distinguishing fast barrier crossings from slow diffusive transitions.

```python
transition_ts = changepoints[0].timestamp

vel = cvx.velocity(traj, timestamp=transition_ts)
speed = np.linalg.norm(vel)

print(f"Transition speed at frame {transition_ts}: {speed:.4f}")
# High speed → fast barrier crossing (two-state folding)
# Low speed  → gradual diffusive transition
```

## Region Trajectory: State Occupancy Over Time

The region trajectory tracks which conformational state the molecule occupies at each point in the simulation, producing a time-resolved state occupancy profile.

```python
# Compute region trajectory with exponential smoothing
reg_traj = index.region_trajectory(
    entity_id=0,
    level=2,
    window_days=1000,   # window size in timestamp units
    alpha=0.3           # smoothing factor
)

# reg_traj: list of (timestamp, region_id, probability) tuples
# Visualize as a heatmap of state occupancy over simulation time
```

A metastable state appears as a region with high self-transition probability -- the molecule remains in the same HNSW region across many consecutive frames.

## Trajectory Comparison Between Simulations

Path signatures provide a global descriptor for an entire trajectory, enabling comparison of simulation runs without frame-by-frame alignment.

```python
# Two independent simulations of the same protein
traj_sim1 = index.trajectory(entity_id=0)
traj_sim2 = index.trajectory(entity_id=1)

# Path signature comparison (truncated at depth 2)
sig_a = cvx.path_signature(traj_sim1, depth=2)
sig_b = cvx.path_signature(traj_sim2, depth=2)
similarity = cvx.signature_distance(sig_a, sig_b)

print(f"Signature distance: {similarity:.6f}")
# Low distance → simulations explore similar conformational pathways

# For exact geometric comparison (sensitive to speed and ordering):
fd = cvx.frechet_distance(traj_sim1, traj_sim2)
print(f"Fréchet distance: {fd:.4f}")
```

Path signatures capture the sequential structure of conformational exploration: two simulations that visit the same states in different order will have different signatures. Frechet distance additionally penalizes differences in transition timing.

## Topological Analysis: Conformational Basins

Topological features (persistent homology) detect the formation and merging of conformational basins over the course of a simulation.

```python
# Extract region centroids as a point cloud
centroids = np.array([c for _, c, _ in regions])

# Compute topological features across filtration radii
topo = cvx.topological_features(centroids, n_radii=30)

# topo contains Betti numbers at each radius:
# β₀ = number of connected components (conformational basins)
# β₁ = number of loops (circular transition pathways)

# Increasing β₀ over time → conformational landscape developing new basins
# Persistent β₁ features → stable cyclic transition pathways
```

Tracking topological features at different simulation timepoints reveals how the conformational landscape evolves -- whether new basins appear, existing ones merge, or cyclic pathways form between states.

## Wasserstein Drift: State Redistribution

Wasserstein drift quantifies how the population distribution across conformational states changes between two timepoints, accounting for the geometric relationship between states.

```python
# State distributions at two timepoints
dist_t1 = np.array([0.6, 0.3, 0.1])  # 60% folded, 30% intermediate, 10% unfolded
dist_t2 = np.array([0.2, 0.3, 0.5])  # shifted toward unfolded

wd = cvx.wasserstein_drift(dist_t1, dist_t2, centroids, n_projections=50)
print(f"Wasserstein drift: {wd:.4f}")
```

Unlike KL divergence, Wasserstein drift is geometry-aware: redistributing population between neighboring conformational states costs less than shifting between distant states. This makes it the natural metric for detecting gradual unfolding or ligand-induced conformational shifts.

## Complete Analysis Pipeline

Putting it all together for a typical MD analysis workflow:

```python
import chronos_vector as cvx
import numpy as np

# --- 1. Ingest trajectories from two simulation runs ---
index = cvx.TemporalIndex(m=16, ef_construction=200)

for run_id, path in enumerate(["run1.npy", "run2.npy"]):
    frames = np.load(path)
    n = frames.shape[0]
    min_v, max_v = frames.min(), frames.max()
    index.enable_quantization(min_v, max_v)
    index.bulk_insert(
        np.full(n, run_id, dtype=np.uint64),
        np.arange(n, dtype=np.uint64),
        frames,
        ef_construction=50,
    )

# --- 2. Conformational clustering ---
regions = index.regions(level=2)
print(f"Identified {len(regions)} conformational states")

# --- 3. Transition detection for each run ---
for run_id in range(2):
    traj = index.trajectory(entity_id=run_id)
    cps = cvx.detect_changepoints(run_id, traj, min_segment_len=100)
    print(f"Run {run_id}: {len(cps)} transitions detected")

    for cp in cps:
        vel = cvx.velocity(traj, timestamp=cp.timestamp)
        print(f"  Frame {cp.timestamp}: speed = {np.linalg.norm(vel):.4f}")

# --- 4. Trajectory comparison ---
traj_0 = index.trajectory(entity_id=0)
traj_1 = index.trajectory(entity_id=1)

sig_dist = cvx.signature_distance(
    cvx.path_signature(traj_0, depth=2),
    cvx.path_signature(traj_1, depth=2),
)
print(f"Trajectory similarity (signature distance): {sig_dist:.6f}")

# --- 5. Topological analysis ---
centroids = np.array([c for _, c, _ in regions])
topo = cvx.topological_features(centroids, n_radii=30)
print(f"Conformational basins (β₀): {topo.betti[0]}")
print(f"Cyclic pathways (β₁): {topo.betti[1]}")
```

## Performance

For a typical MD dataset of 10^6 frames at D=1000:

| Operation | Brute-force | CVX |
|---|---|---|
| Conformational clustering | O(N^2) -- hours on GPU | O(N log N) -- minutes on CPU |
| Transition detection | Requires pre-computed clusters | Direct on trajectory |
| Trajectory comparison (2 runs) | DTW: O(N^2) | Signature: O(N * depth) |
| Topological features | Rips complex: O(N^3) | On region centroids: O(K^3), K << N |

The key insight is that HNSW regions replace the expensive all-pairs distance matrix with a graph-based neighborhood structure that is built incrementally during insertion. Clustering is essentially free once the index is constructed.

## References

- MOSCITO MD simulation package, Boresch et al. (2024)
- Shao, J. et al. "Clustering molecular dynamics trajectories." *J. Chem. Theory Comput.* 3(6), 2312--2334 (2007)
- Fast conformational clustering of extensive molecular dynamics simulation data. *J. Chem. Inf. Model.* (2023)
- Molgedey, L. & Schuster, H. G. "Separation of a mixture of independent signals using time delayed correlations." *Phys. Rev. Lett.* 72, 3634 (1994)
