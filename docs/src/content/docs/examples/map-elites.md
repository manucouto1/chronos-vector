---
title: "Quality-Diversity (MAP-Elites)"
description: "Using HNSW as a MAP-Elites archive for O(log N) novelty scoring and natural niche discovery"
---

MAP-Elites is a quality-diversity (QD) algorithm that maintains an archive of diverse, high-performing solutions indexed by **behavioral descriptor** vectors. The central bottleneck in large-scale QD is archive operations: finding which niche a new solution belongs to and computing novelty scores. Both reduce to nearest-neighbor search over the archive.

ChronosVector replaces the $O(N)$ linear scan with $O(\log N)$ HNSW search, yielding ~1000x speedup at million-solution scale. Its graph hierarchy provides **natural niche discovery** without the need to pre-specify cell counts, and its temporal analytics track how exploration evolves across generations.

---

## Problem

Three limitations constrain MAP-Elites at scale:

1. **Archive operations dominate computation.** Every offspring requires a nearest-neighbor lookup against the full archive to determine its niche and compute novelty. With a linear scan, this is $O(N)$ per evaluation — for a 1M-entry archive, this overwhelms the actual fitness evaluation.

2. **CVT-MAP-Elites requires pre-specified cell counts.** Centroidal Voronoi Tessellation discretizes the descriptor space into a fixed number of cells before the run begins. Choosing too few cells loses diversity; too many leaves most cells empty. The user must commit to a tessellation before seeing the data.

3. **High-dimensional descriptors break Voronoi tessellation.** Beyond ~20 dimensions, Voronoi cells become pathological — nearly all volume concentrates on cell boundaries, and the tessellation provides little meaningful structure.

HNSW resolves all three: $O(\log N)$ search eliminates the archive bottleneck, the graph hierarchy defines niches adaptively from the data, and navigable small-world properties remain effective in high dimensions.

---

## Data Model

| MAP-Elites Concept | CVX Equivalent | Notes |
|---|---|---|
| Archive | `TemporalIndex` | HNSW index stores all solutions |
| Behavioral descriptor | Vector | The embedding for each solution |
| Generation | Timestamp | Microsecond-precision `i64`; use generation number directly |
| Find niche | `index.search(vector=desc, k=1)` | Nearest archive member = niche assignment |
| Novelty score | `index.search(vector=desc, k=15)` then mean distance | Standard novelty metric (Lehman & Stanley, 2011) |
| Archive regions | `index.regions(level)` | HNSW hierarchy = adaptive niches |
| Niche occupant | `index.region_members(region_id, level)` | All solutions in a niche |
| Solution lineage | `index.trajectory(entity_id)` | How a lineage evolves across generations |

---

## Archive Setup

Create a temporal index configured for the descriptor dimensionality and expected archive size:

```python
import numpy as np
import chronos_vector as cvx

# HNSW parameters:
#   m=16        — 16 bidirectional links per node (good default)
#   ef_construction=100 — build-time beam width (higher = better recall, slower build)
#   ef_search=64        — query-time beam width (tunable per query)
index = cvx.TemporalIndex(m=16, ef_construction=100, ef_search=64)
```

---

## Core Operations

### Batch Insert Per Generation

After evaluating a batch of offspring, insert them all at once. The generation number serves as the timestamp:

```python
def insert_generation(index, offspring_ids, descriptors, generation):
    """Insert a generation of solutions into the archive.

    Args:
        offspring_ids: np.ndarray[uint64] of shape (N,)
        descriptors:   np.ndarray[float32] of shape (N, D)
        generation:    int, current generation number
    """
    timestamps = np.full(len(offspring_ids), generation, dtype=np.int64)
    n_inserted = index.bulk_insert(offspring_ids, timestamps, descriptors)
    return n_inserted
```

### Niche Assignment

Find the closest existing archive member — the niche this solution belongs to:

```python
def find_niche(index, descriptor):
    """Return (niche_entity_id, niche_timestamp, distance) for a new solution."""
    results = index.search(vector=descriptor.tolist(), k=1)
    if results:
        entity_id, timestamp, distance = results[0]
        return entity_id, timestamp, distance
    return None
```

### Novelty Scoring

Novelty is the mean distance to the $k$ nearest archive members. With HNSW this is $O(\log N)$ instead of $O(N)$:

```python
def novelty_score(index, descriptor, k=15):
    """Compute novelty as mean distance to k nearest archive members."""
    results = index.search(vector=descriptor.tolist(), k=k)
    if not results:
        return float('inf')  # empty archive = maximally novel
    distances = [distance for _, _, distance in results]
    return np.mean(distances)
```

### Natural Niche Discovery

HNSW's hierarchical graph defines niches at multiple granularities without pre-specifying cell counts:

```python
# Coarse niches (level 3): ~N/4096 regions
coarse_niches = index.regions(level=3)
for region_id, centroid, member_count in coarse_niches:
    print(f"Niche {region_id}: {member_count} solutions, centroid dim={len(centroid)}")

# Fine niches (level 2): ~N/256 regions
fine_niches = index.regions(level=2)

# Inspect a specific niche
members = index.region_members(region_id=coarse_niches[0][0], level=3)
for entity_id, timestamp, vector in members:
    print(f"  Solution {entity_id} from generation {timestamp}")
```

---

## Complete Generation Loop

The following replaces the traditional MAP-Elites archive with CVX. The key change: niche assignment and novelty scoring become $O(\log N)$ operations.

```python
import numpy as np
import chronos_vector as cvx

# --- Configuration ---
DIM = 50               # behavioral descriptor dimensionality
POP_SIZE = 256          # offspring per generation
MAX_GEN = 5000
NOVELTY_K = 15
NOVELTY_THRESHOLD = 0.1 # minimum novelty to keep

# --- Initialize ---
index = cvx.TemporalIndex(m=16, ef_construction=100, ef_search=64)
next_id = 0
fitness_log = {}        # entity_id -> best fitness

def generate_offspring(parent_ids, parent_descriptors):
    """Domain-specific: mutate parents to produce offspring."""
    # ... mutation/crossover logic ...
    return offspring_ids, offspring_descriptors, offspring_fitness

# Seed population
seed_descriptors = np.random.randn(POP_SIZE, DIM).astype(np.float32)
seed_ids = np.arange(POP_SIZE, dtype=np.uint64)
seed_fitness = evaluate_batch(seed_descriptors)  # domain-specific
index.bulk_insert(seed_ids, np.zeros(POP_SIZE, dtype=np.int64), seed_descriptors)
for sid, fit in zip(seed_ids, seed_fitness):
    fitness_log[int(sid)] = fit
next_id = POP_SIZE

# --- Main loop ---
for generation in range(1, MAX_GEN + 1):
    # 1. Generate offspring
    parent_ids = np.random.choice(list(fitness_log.keys()), size=POP_SIZE)
    offspring_desc = mutate(parent_ids)  # -> np.ndarray[float32] (POP_SIZE, DIM)
    offspring_fit = evaluate_batch(offspring_desc)

    # 2. Assign IDs
    offspring_ids = np.arange(next_id, next_id + POP_SIZE, dtype=np.uint64)
    next_id += POP_SIZE

    # 3. Batch insert with generation as timestamp
    timestamps = np.full(POP_SIZE, generation, dtype=np.int64)
    index.bulk_insert(offspring_ids, timestamps, offspring_desc)

    # 4. Novelty scoring in O(log N)
    for i in range(POP_SIZE):
        desc = offspring_desc[i].tolist()
        neighbors = index.search(vector=desc, k=NOVELTY_K)
        novelty = np.mean([dist for _, _, dist in neighbors])

        if novelty > NOVELTY_THRESHOLD:
            fitness_log[int(offspring_ids[i])] = offspring_fit[i]

    # 5. Periodic analytics (every 100 generations)
    if generation % 100 == 0:
        regions = index.regions(level=2)
        print(f"Gen {generation}: archive={len(index)}, niches={len(regions)}")
```

---

## Archive Analytics

CVX's temporal analytics provide insight into the evolutionary process itself, not just the solutions.

### Solution Lineage Analysis

Track how a lineage evolves through descriptor space across generations using path signatures:

```python
# Retrieve a solution's full trajectory across generations
traj = index.trajectory(entity_id=42)
# traj = [(gen0, vec0), (gen1, vec1), ...]

# Path signature captures the shape of the evolutionary path
sig = cvx.path_signature(traj, depth=2)

# Compare two lineages
sig_a = cvx.path_signature(index.trajectory(entity_id=42), depth=2)
sig_b = cvx.path_signature(index.trajectory(entity_id=99), depth=2)
similarity = cvx.signature_distance(sig_a, sig_b)
# Low distance = similar evolutionary dynamics
```

### Exploration Dynamics

Track how the search distributes across niches over time:

```python
# Region trajectory: how an entity's niche membership shifts over generations
region_traj = index.region_trajectory(entity_id=42, level=3, window_days=100, alpha=0.3)
# Returns distribution over regions at each time window

# Wasserstein drift between early and late exploration
early_dist = region_traj[0]    # niche distribution at start
late_dist = region_traj[-1]    # niche distribution now

centroids = [centroid for _, centroid, _ in index.regions(level=3)]
w_drift = cvx.wasserstein_drift(early_dist, late_dist, centroids)
# High drift = exploration shifted to new regions of descriptor space
```

### Archive Topology

Monitor whether the archive is well-connected or fragmenting into isolated clusters:

```python
# Extract region centroids
centroids = [centroid for _, centroid, _ in index.regions(level=2)]

# Topological features via persistent homology
topo = cvx.topological_features(centroids)
print(f"Connected components: {topo['n_components']}")
print(f"Max persistence:      {topo['max_persistence']:.3f}")
print(f"Persistence entropy:  {topo['persistence_entropy']:.3f}")

# Interpretation:
#   n_components = 1     → archive is well-connected
#   n_components >> 1    → fragmented into isolated niches
#   High persistence     → strong cluster separation
#   High entropy         → uniform cluster structure
```

### Convergence and Stagnation Detection

Detect when the evolutionary search is losing steam:

```python
# Hurst exponent on a solution lineage
traj = index.trajectory(entity_id=42)
h = cvx.hurst_exponent(traj)
# H > 0.5 → persistent (trending exploration in one direction)
# H ≈ 0.5 → random walk (search becoming undirected — convergence)
# H < 0.5 → anti-persistent (oscillating, possibly stuck)

# Velocity: how fast is exploration moving?
vel = cvx.velocity(traj, timestamp=generation)
speed = np.linalg.norm(vel)
# Declining speed across generations → stagnation
# Sudden spike → discovered a new promising region
```

---

## Performance

The asymptotic improvement from HNSW is dramatic at scale:

| Archive Size | Linear Scan (per query) | HNSW (per query) | Speedup |
|---|---|---|---|
| 1,000 | 1,000 ops | ~10 ops | 100x |
| 100,000 | 100,000 ops | ~17 ops | ~6,000x |
| 1,000,000 | 1,000,000 ops | ~20 ops | ~50,000x |

*50D descriptors, k=15, ef_search=64. HNSW query time scales as $O(\log N)$.*

For a typical QD run with 1M archive entries and 256 offspring per generation, the per-generation archive operations drop from ~256M distance computations (linear) to ~5,120 (HNSW), making novelty scoring negligible compared to fitness evaluation.

---

## Why Not CVT-MAP-Elites?

CVT-MAP-Elites (Vassiliades et al., 2018) pre-computes a Centroidal Voronoi Tessellation to discretize the descriptor space. This works well in low dimensions but has structural limitations:

| Aspect | CVT-MAP-Elites | CVX (HNSW) |
|---|---|---|
| Cell count | Fixed before run | Adaptive (grows with archive) |
| High dimensions | Voronoi degenerates >20D | HNSW effective to 1000D+ |
| Niche granularity | Single scale | Multi-scale via graph levels |
| Temporal tracking | Not built in | Native (generation = timestamp) |
| Analytics | External tooling | Integrated (signatures, topology, drift) |

CVX's HNSW hierarchy provides the niche structure that CVT pre-computes, but discovers it adaptively from the actual solutions rather than from a prior tessellation of the space.

---

## References

1. Mouret, J.-B. & Clune, J. (2015). Illuminating search spaces by mapping elites. *arXiv:1504.04909*.
2. Vassiliades, V., Chatzilygeroudis, K., & Mouret, J.-B. (2018). Using centroidal Voronoi tessellations to scale up the multidimensional archive of phenotypic elites algorithm. *IEEE Transactions on Evolutionary Computation*, 22(4), 623-630.
3. Fontaine, M.C., Togelius, J., Nikolaidis, S., & Hoover, A.K. (2020). Covariance matrix adaptation for the rapid illumination of behavior space. *GECCO*.
4. Lehman, J. & Stanley, K.O. (2011). Abandoning objectives: Evolution through the search for novelty alone. *Evolutionary Computation*, 19(2), 189-223.
5. Malkov, Y.A. & Yashunin, D.A. (2018). Efficient and robust approximate nearest neighbor using hierarchical navigable small world graphs. *IEEE TPAMI*, 42(4), 824-836.
