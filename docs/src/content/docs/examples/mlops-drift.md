---
title: "MLOps Drift Detection"
description: "Monitoring production ML models for embedding drift using temporal vector analytics"
---

Production ML models degrade silently. The embeddings a model produces shift over time as user behavior changes, new data distributions emerge, and the world moves on from the training snapshot. This is **concept drift**, and it is the leading cause of model failures in production.

Current monitoring tools track scalar metrics — accuracy on a labeled sample, input feature statistics, prediction confidence histograms. But the richest signal lives in the embeddings themselves: high-dimensional trajectories that encode exactly how a model's internal representation is changing. The problem is that no existing infrastructure treats embedding streams as first-class temporal objects.

ChronosVector was designed for exactly this. Every embedding is stored with its timestamp. Region distributions provide natural drift indicators. Change point detection identifies drift onset. Velocity quantifies drift rate. The Hurst exponent predicts whether drift will persist or self-correct. Topological features reveal structural changes in the embedding space. This page shows how to wire it all together.

## Data model

The mapping from drift detection concepts to CVX primitives is direct:

| Drift Detection Concept | CVX Primitive |
|---|---|
| Embedding stream | `insert()` with timestamp |
| Reference distribution | Region distribution at T₀ |
| Current distribution | Region distribution at T_now |
| Drift metric | `drift()`, `wasserstein_drift()`, `fisher_rao_distance()` |
| Drift onset | `detect_changepoints()` |
| Affected subpopulation | `region_members(changed_region, time_range)` or `region_assignments(level, start, end)` for all regions at once |

This is not an analogy. CVX's temporal HNSW index stores embedding trajectories natively, and its statistical toolkit operates on exactly the objects that drift detection requires.

## Ingesting production embeddings

Every inference request produces an embedding. Stream them into CVX with their timestamps:

```python
import chronos_vector as cvx
import numpy as np

# Initialize with production-grade parameters
index = cvx.TemporalIndex(m=16, ef_construction=200)

# Bulk insert from your inference pipeline
# request_ids: unique ID per inference request
# timestamps: when each request was served
# embeddings: the model's internal representation
index.bulk_insert(request_ids, timestamps, embeddings)
```

In production, this runs as a sidecar that consumes from your inference pipeline's embedding output stream. The index grows continuously — CVX handles the temporal bookkeeping.

## Computing the reference distribution

The reference distribution captures what "normal" looks like. Compute it from a window where the model was known to perform well:

```python
regions = index.regions(level=2)
ref_dist = index.region_trajectory(entity_id=0, level=2)
```

The region distribution at level 2 of the HNSW hierarchy provides a natural coarse-grained view of how embeddings are distributed across the space. Each region corresponds to a Voronoi cell around a cluster centroid. The reference distribution records how many embeddings fall into each cell during the baseline period.

## Drift detection over time windows

CVX provides three complementary distance measures between distributions. Each has different mathematical properties suited to different monitoring needs:

```python
# Fisher-Rao: the mathematically correct distance on the
# statistical manifold of distributions
fr = cvx.fisher_rao_distance(ref_distribution, current_distribution)

# Wasserstein (Earth Mover's Distance): respects the geometry
# of the embedding space, not just the probability simplex
wd = cvx.wasserstein_drift(ref_dist, curr_dist, centroids)

# Hellinger: bounded in [0, 1], ideal for threshold-based alerting
hd = cvx.hellinger_distance(ref_dist, curr_dist)
```

**When to use which:**
- **Fisher-Rao** is the information-geometric gold standard. It measures distance along geodesics of the statistical manifold, making it invariant to reparameterization. Use it when you need a principled, theoretically grounded metric.
- **Wasserstein** accounts for the ground distance between regions. If two distributions shift mass between nearby clusters, Wasserstein reports a small distance; Hellinger and Fisher-Rao may overreact. Use it when spatial relationships between clusters matter.
- **Hellinger** is bounded between 0 and 1, which makes threshold selection straightforward. Use it for production alerting where you need a simple "drift or no drift" decision.

## Detecting drift onset

Knowing that drift exists is not enough — you need to know *when* it started. CVX's change point detection identifies structural breaks in embedding trajectories:

```python
traj = index.trajectory(entity_id=model_version)
changepoints = cvx.detect_changepoints(model_version, traj)
# Each changepoint = potential drift onset
```

Each detected changepoint corresponds to a timestamp where the embedding trajectory's statistical properties shifted. This tells you exactly when the data distribution changed, which is critical for root cause analysis: you can correlate drift onset with deployments, upstream data pipeline changes, or external events.

## Diagnosing what drifted

Once you detect drift, the next question is *what changed*. CVX's region-level analysis lets you identify which subpopulations are affected:

```python
# Which regions changed?
# Option A: query one region at a time with region_members
for rid, centroid, n in regions:
    members_before = index.region_members(rid, level=2, start=t0, end=t1)
    members_after = index.region_members(rid, level=2, start=t1, end=t2)
    # Compare counts to identify growing/shrinking clusters

# Option B (faster): assign all nodes in a single O(N) pass
assignments_before = index.region_assignments(level=2, start=t0, end=t1)
assignments_after = index.region_assignments(level=2, start=t1, end=t2)
for rid in assignments_before:
    # Compare len(assignments_before[rid]) vs len(assignments_after.get(rid, []))
    pass
```

A region that doubles in size may indicate a new user behavior pattern the model was not trained on. A region that empties out may mean a previously common pattern has disappeared. Both require different remediation strategies.

## Drift rate via velocity

Not all drift is equally urgent. CVX computes the instantaneous velocity of an embedding trajectory, which quantifies *how fast* the distribution is changing right now:

```python
vel = cvx.velocity(traj, timestamp=now)
# High velocity = fast drift → urgent intervention
```

Velocity turns drift from a binary signal into a continuous one. A model that drifted slowly over six months may tolerate a scheduled retrain. A model whose embedding velocity spikes overnight needs immediate attention.

## Drift persistence via Hurst exponent

The most valuable question in drift monitoring is not "is the model drifting?" but "will the drift continue?". The Hurst exponent answers this:

```python
h = cvx.hurst_exponent(traj)
# H > 0.5: persistent drift (model will keep degrading)
# H ≈ 0.5: random walk (unpredictable)
# H < 0.5: mean-reverting (may self-correct)
```

A Hurst exponent above 0.5 indicates persistent drift — the trend will continue, and the model will keep degrading unless you intervene. Below 0.5 suggests anti-persistent behavior: the distribution oscillates and may return to baseline on its own. This distinction determines whether you retrain now or wait.

## Topological monitoring

Beyond distributional distance, the *structure* of the embedding space can change. CVX's topological features detect this:

```python
topo = cvx.topological_features(centroids)
# Increasing β₀ → embedding space fragmenting
# Decreasing β₀ → clusters merging (mode collapse?)
```

An increasing zeroth Betti number (β₀) means the embedding space is breaking into more connected components — the model is losing its ability to organize inputs into coherent clusters. A decreasing β₀ suggests clusters are merging, which may indicate mode collapse where the model maps distinct inputs to similar representations. Both are failure modes that scalar drift metrics miss entirely.

## Complete monitoring pipeline

Putting it all together into a production monitoring loop:

```python
import chronos_vector as cvx
import time

# --- Setup ---
index = cvx.TemporalIndex(m=16, ef_construction=200)

# Ingest baseline period (known-good model performance)
index.bulk_insert(baseline_ids, baseline_timestamps, baseline_embeddings)

# Compute reference distribution
regions = index.regions(level=2)
ref_dist = index.region_trajectory(entity_id=0, level=2)

# --- Monitoring loop ---
HELLINGER_THRESHOLD = 0.3
VELOCITY_THRESHOLD = 0.05
CHECK_INTERVAL = 300  # seconds

while True:
    # Ingest latest embeddings from inference pipeline
    new_ids, new_ts, new_embs = fetch_latest_embeddings()
    index.bulk_insert(new_ids, new_ts, new_embs)

    # Current distribution
    curr_dist = index.region_trajectory(entity_id=0, level=2)

    # Drift magnitude
    hd = cvx.hellinger_distance(ref_dist, curr_dist)

    # Drift rate
    traj = index.trajectory(entity_id=0)
    vel = cvx.velocity(traj, timestamp=time.time())

    # Drift persistence
    h = cvx.hurst_exponent(traj)

    # Structural health
    centroids = [c for _, c, _ in regions]
    topo = cvx.topological_features(centroids)

    # --- Alert logic ---
    if hd > HELLINGER_THRESHOLD:
        if vel > VELOCITY_THRESHOLD and h > 0.5:
            alert_critical(
                f"Persistent fast drift detected: "
                f"Hellinger={hd:.3f}, velocity={vel:.4f}, Hurst={h:.2f}"
            )
        else:
            alert_warning(
                f"Drift detected: Hellinger={hd:.3f}, "
                f"velocity={vel:.4f}, Hurst={h:.2f}"
            )

        # Diagnose affected regions
        changepoints = cvx.detect_changepoints(0, traj)
        if changepoints:
            drift_onset = changepoints[-1]
            for rid, centroid, n in regions:
                before = index.region_members(rid, level=2,
                    start=drift_onset - 3600, end=drift_onset)
                after = index.region_members(rid, level=2,
                    start=drift_onset, end=time.time())
                report_region_change(rid, len(before), len(after))

    time.sleep(CHECK_INTERVAL)
```

## Why CVX for drift detection

This use case exercises nearly every CVX feature:

- **Temporal HNSW index** stores the embedding stream with timestamps
- **Region distributions** provide natural reference and current distributions
- **Three drift distances** (Fisher-Rao, Wasserstein, Hellinger) cover different monitoring needs
- **Change point detection** identifies when drift started
- **Region members** diagnose which subpopulations are affected
- **Velocity** quantifies drift urgency
- **Hurst exponent** predicts drift persistence
- **Topological features** detect structural changes invisible to scalar metrics

Existing tools like DriftLens (Greco et al., 2024) address embedding drift monitoring but rely on external storage and batch processing. They compute drift metrics over snapshots rather than continuous trajectories. CVX provides the temporal-native storage layer that makes real-time, trajectory-aware drift detection possible — the embeddings, their timestamps, their hierarchical structure, and the full statistical toolkit live in a single system.

## References

- Greco, S., Vacanti, G., Prenkaj, B., & Gravina, G. (2024). DriftLens: Real-Time Unsupervised Concept Drift Detection by Leveraging Drift in the Embedding Space.
- Gama, J., Zliobaite, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM Computing Surveys*, 46(4), 1–37.
