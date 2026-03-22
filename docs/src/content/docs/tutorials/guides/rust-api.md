---
title: "Rust API Guide"
description: "Using ChronosVector directly from Rust — index construction, search, analytics, and persistence"
---

## Add to Your Project

```toml
# Cargo.toml
[dependencies]
cvx-index = { version = "0.1", features = [] }
cvx-analytics = { version = "0.1" }
cvx-core = { version = "0.1" }
```

## Create and Populate an Index

```rust
use cvx_index::hnsw::{HnswConfig, TemporalHnsw};
use cvx_index::metrics::CosineDistance;
use cvx_core::TemporalFilter;

// Configure HNSW parameters
let config = HnswConfig {
    m: 16,                    // connections per node
    ef_construction: 200,     // search width during build
    ef_search: 50,            // search width during queries
    ..Default::default()
};

let mut index = TemporalHnsw::new(config, CosineDistance);

// Insert points: (entity_id, timestamp, vector)
index.insert(1, 1000, &[0.1, 0.2, 0.3, 0.4]);
index.insert(1, 2000, &[0.15, 0.25, 0.28, 0.42]);
index.insert(2, 1500, &[0.9, 0.1, 0.05, 0.02]);

println!("Index has {} points", index.len());
```

### Distance Metrics

```rust
use cvx_index::metrics::{CosineDistance, L2Distance, DotProductDistance};

// Cosine: d(a,b) = 1 - cos(a,b), range [0, 2]
let cosine_index = TemporalHnsw::new(config.clone(), CosineDistance);

// L2 (Euclidean squared): d(a,b) = ||a-b||², range [0, ∞)
let l2_index = TemporalHnsw::new(config.clone(), L2Distance);

// Dot product (negative for MIPS): d(a,b) = -a·b
let dot_index = TemporalHnsw::new(config, DotProductDistance);
```

All metrics use SIMD acceleration (AVX2/NEON) via `pulp`.

## Search

### Composite Distance

Search combines semantic and temporal distance:

$$d_{ST} = \alpha \cdot d_{semantic} + (1 - \alpha) \cdot d_{temporal}$$

```rust
// Pure semantic search (alpha = 1.0)
let results = index.search(
    &[0.1, 0.2, 0.3, 0.4],  // query vector
    5,                         // k nearest
    TemporalFilter::All,       // no time filter
    1.0,                       // alpha: pure semantic
    0,                         // query_timestamp (unused when alpha=1.0)
);

for (node_id, score) in &results {
    println!("node={node_id}, score={score:.4}");
}
```

### Temporal Filtering

```rust
// Only results in time range [1000, 2000]
let results = index.search(
    &query, 5,
    TemporalFilter::Range(1000, 2000),
    0.5,    // balanced semantic + temporal
    1500,   // query timestamp for temporal distance
);

// Other filters
let _ = TemporalFilter::Before(2000);       // ts <= 2000
let _ = TemporalFilter::After(1000);        // ts >= 1000
let _ = TemporalFilter::Snapshot(1500);     // ts == 1500
```

### Recency-Weighted Search (P7)

```rust
let results = index.search_with_recency(
    &query, 5,
    TemporalFilter::All,
    1.0,          // alpha
    0,            // query_timestamp
    2.0,          // recency_lambda: decay speed (0=off, 3=strong)
    0.3,          // recency_weight: contribution to score
);
```

### Reward-Filtered Search (P4)

```rust
// Insert with reward
index.insert_with_reward(entity_id, timestamp, &vector, 0.85);

// Set reward retroactively
index.set_reward(node_id, 0.95);

// Only retrieve successful experiences
let results = index.search_with_reward(
    &query, 5, TemporalFilter::All, 1.0, 0,
    0.5,  // min_reward threshold
);
```

## Trajectory

```rust
let trajectory = index.trajectory(1, TemporalFilter::All);
for (timestamp, node_id) in &trajectory {
    let vec = index.vector(*node_id);
    println!("t={timestamp}, dim={}", vec.len());
}
```

## Centering (Anisotropy Correction)

```rust
// Compute and set centroid
if let Some(centroid) = index.compute_centroid() {
    index.set_centroid(centroid);
}

// Center any vector
let centered = index.centered_vector(&query);

// Centroid persists through save/load
```

## Semantic Regions

```rust
// Hub nodes at level 1 (unsupervised clusters)
let regions = index.regions(1);
for (hub_id, hub_vec, member_count) in &regions {
    println!("Hub {hub_id}: {member_count} members");
}

// O(N) single-pass assignment
let assignments = index.region_assignments(1, TemporalFilter::All);
for (hub_id, members) in &assignments {
    println!("Hub {hub_id}: {} members", members.len());
}
```

## Persistence

```rust
use std::path::Path;

// Save
index.save(Path::new("my_index.bin"))?;

// Load (must provide same metric type)
let loaded = TemporalHnsw::load(
    Path::new("my_index.bin"),
    CosineDistance,
)?;
```

## Causal Search (TemporalGraphIndex)

For episode-based retrieval with predecessor/successor edges:

```rust
use cvx_index::hnsw::TemporalGraphIndex;

let mut graph_index = TemporalGraphIndex::new(config, CosineDistance);

// Insert episode steps (same entity_id = same episode)
graph_index.insert(episode_id, step_0_ts, &step_0_vec);
graph_index.insert(episode_id, step_1_ts, &step_1_vec);
graph_index.insert(episode_id, step_2_ts, &step_2_vec);

// Causal search: find similar states + what happened next
let results = graph_index.causal_search(
    &query, 5,
    TemporalFilter::All,
    1.0, 0,
    5,  // walk 5 steps forward/backward
);

for result in &results {
    println!("Match: entity={}, score={:.3}", result.entity_id, result.score);
    println!("  {} successors, {} predecessors",
        result.successors.len(), result.predecessors.len());
}
```

## Analytics (cvx-analytics)

### Vector Calculus

```rust
use cvx_analytics::calculus;
use cvx_core::TemporalPoint;

let trajectory: Vec<TemporalPoint> = /* from index */;

// Velocity at timestamp
let velocity = calculus::velocity(&trajectory, target_ts)?;

// Drift between two vectors
let report = calculus::drift_report(&vec_a, &vec_b, 5)?;
println!("L2 drift: {}", report.l2_magnitude);
```

### Change Point Detection

```rust
use cvx_analytics::pelt::{PeltConfig, detect};

let config = PeltConfig {
    penalty: None,        // auto BIC
    min_segment_len: 5,
};

let changepoints = detect(&trajectory, &config)?;
for cp in &changepoints {
    println!("Change at t={}, severity={:.3}", cp.timestamp, cp.severity());
}
```

### Procrustes Alignment (P10)

```rust
use cvx_analytics::procrustes::fit_procrustes;

// Align source model's vectors to target model's space
let transform = fit_procrustes(&source_vecs, &target_vecs).unwrap();

// Transform any source vector to target space
let aligned = transform.apply(&new_source_vec);
println!("Alignment error: {:.4}", transform.error);
```

## Thread-Safe Access

```rust
use cvx_index::hnsw::ConcurrentTemporalHnsw;
use std::sync::Arc;

let index = Arc::new(ConcurrentTemporalHnsw::new(config, CosineDistance));

// Insert (write lock)
index.insert(entity_id, timestamp, &vector);

// Search (read lock — concurrent readers)
let results = index.search(&query, 5, TemporalFilter::All, 1.0, 0);

// Queue inserts for batch flush
index.queue_insert(entity_id, timestamp, vector);
let n_flushed = index.flush_inserts();
```

## Scalar Quantization

```rust
// Enable 4× faster distance computation (uint8 codes)
index.enable_scalar_quantization(-1.0, 1.0);

// Candidates use fast SQ distance; final results use exact float32
let results = index.search(&query, 10, TemporalFilter::All, 1.0, 0);

// Disable when precision is critical
index.disable_scalar_quantization();
```
