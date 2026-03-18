//! Query execution engine.
//!
//! Orchestrates index, storage, and analytics to execute temporal queries.

use cvx_core::error::QueryError;
use cvx_core::types::TemporalFilter;
use cvx_core::{StorageBackend, TemporalIndexAccess};

use cvx_analytics::calculus;
use cvx_analytics::cohort;
use cvx_analytics::ode;
use cvx_analytics::pelt::{self, PeltConfig};

use crate::types::*;

/// Query engine that coordinates index + storage + analytics.
///
/// Generic over the index `I` (any `TemporalIndexAccess` impl) and storage `S`.
/// This allows the same engine to work with both `TemporalHnsw` (single-threaded)
/// and `ConcurrentTemporalHnsw` (thread-safe).
pub struct QueryEngine<I: TemporalIndexAccess, S: StorageBackend> {
    index: I,
    store: S,
}

impl<I: TemporalIndexAccess, S: StorageBackend> QueryEngine<I, S> {
    /// Create a new query engine.
    pub fn new(index: I, store: S) -> Self {
        Self { index, store }
    }

    /// Execute a temporal query.
    pub fn execute(&self, query: TemporalQuery) -> Result<QueryResult, QueryError> {
        execute_query(&self.index, query)
    }

    /// Access the underlying index.
    pub fn index(&self) -> &I {
        &self.index
    }

    /// Access the underlying store.
    pub fn store(&self) -> &S {
        &self.store
    }
}

/// Execute a temporal query against any index without owning it.
///
/// This is the primary entry point for API handlers that share index/store
/// via `Arc<AppState>`.
pub fn execute_query(
    index: &dyn TemporalIndexAccess,
    query: TemporalQuery,
) -> Result<QueryResult, QueryError> {
    match query {
        TemporalQuery::SnapshotKnn {
            vector,
            timestamp,
            k,
        } => {
            let results = index.search_raw(
                &vector,
                k,
                TemporalFilter::Snapshot(timestamp),
                1.0,
                timestamp,
            );
            Ok(QueryResult::Knn(
                results
                    .into_iter()
                    .map(|(node_id, score)| KnnResult {
                        entity_id: index.entity_id(node_id),
                        timestamp: index.timestamp(node_id),
                        score,
                    })
                    .collect(),
            ))
        }

        TemporalQuery::RangeKnn {
            vector,
            start,
            end,
            k,
            alpha,
        } => {
            let mid = start + (end - start) / 2;
            let results =
                index.search_raw(&vector, k, TemporalFilter::Range(start, end), alpha, mid);
            Ok(QueryResult::Knn(
                results
                    .into_iter()
                    .map(|(node_id, score)| KnnResult {
                        entity_id: index.entity_id(node_id),
                        timestamp: index.timestamp(node_id),
                        score,
                    })
                    .collect(),
            ))
        }

        TemporalQuery::Trajectory { entity_id, filter } => {
            let traj = index.trajectory(entity_id, filter);
            let points = traj
                .iter()
                .map(|&(ts, node_id)| {
                    cvx_core::TemporalPoint::new(entity_id, ts, index.vector(node_id))
                })
                .collect();
            Ok(QueryResult::Trajectory(points))
        }

        TemporalQuery::Velocity {
            entity_id,
            timestamp,
        } => do_velocity(index, entity_id, timestamp),

        TemporalQuery::Prediction {
            entity_id,
            target_timestamp,
        } => do_prediction(index, entity_id, target_timestamp),

        TemporalQuery::ChangePointDetect {
            entity_id,
            start,
            end,
        } => do_change_points(index, entity_id, start, end),

        TemporalQuery::DriftQuant {
            entity_id,
            t1,
            t2,
            top_n,
        } => do_drift_quant(index, entity_id, t1, t2, top_n),

        TemporalQuery::Analogy {
            entity_a,
            t1,
            t2,
            entity_b,
            t3,
        } => do_analogy(index, entity_a, t1, t2, entity_b, t3),

        TemporalQuery::CohortDrift {
            entity_ids,
            t1,
            t2,
            top_n,
        } => do_cohort_drift(index, &entity_ids, t1, t2, top_n),
    }
}

// ─── Internal helpers ──────────────────────────────────────────────

fn build_traj(
    index: &dyn TemporalIndexAccess,
    entity_id: u64,
    filter: TemporalFilter,
) -> (Vec<(i64, u32)>, Vec<Vec<f32>>) {
    let traj_data = index.trajectory(entity_id, filter);
    let vectors: Vec<Vec<f32>> = traj_data
        .iter()
        .map(|&(_, node_id)| index.vector(node_id))
        .collect();
    (traj_data, vectors)
}

fn to_slices<'a>(
    traj_data: &'a [(i64, u32)],
    vectors: &'a [Vec<f32>],
) -> Vec<(i64, &'a [f32])> {
    traj_data
        .iter()
        .zip(vectors.iter())
        .map(|(&(ts, _), v)| (ts, v.as_slice()))
        .collect()
}

fn find_nearest(
    index: &dyn TemporalIndexAccess,
    entity_id: u64,
    timestamp: i64,
) -> Result<u32, QueryError> {
    let traj = index.trajectory(entity_id, TemporalFilter::All);
    if traj.is_empty() {
        return Err(QueryError::EntityNotFound(entity_id));
    }
    let (_, node_id) = traj
        .iter()
        .min_by_key(|&&(ts, _)| (ts - timestamp).unsigned_abs())
        .unwrap();
    Ok(*node_id)
}

fn do_velocity(
    index: &dyn TemporalIndexAccess,
    entity_id: u64,
    timestamp: i64,
) -> Result<QueryResult, QueryError> {
    let (traj_data, vectors) = build_traj(index, entity_id, TemporalFilter::All);
    if traj_data.len() < 2 {
        return Err(QueryError::InsufficientData {
            needed: 2,
            have: traj_data.len(),
        });
    }
    let traj = to_slices(&traj_data, &vectors);
    let vel = calculus::velocity(&traj, timestamp).map_err(|_| QueryError::InsufficientData {
        needed: 2,
        have: traj.len(),
    })?;
    Ok(QueryResult::Velocity(vel))
}

fn do_prediction(
    index: &dyn TemporalIndexAccess,
    entity_id: u64,
    target_timestamp: i64,
) -> Result<QueryResult, QueryError> {
    let (traj_data, vectors) = build_traj(index, entity_id, TemporalFilter::All);
    if traj_data.len() < 2 {
        return Err(QueryError::InsufficientData {
            needed: 2,
            have: traj_data.len(),
        });
    }
    let traj = to_slices(&traj_data, &vectors);
    let predicted = ode::linear_extrapolate(&traj, target_timestamp).map_err(|_| {
        QueryError::InsufficientData {
            needed: 2,
            have: traj.len(),
        }
    })?;
    Ok(QueryResult::Prediction(PredictionResult {
        vector: predicted,
        timestamp: target_timestamp,
        method: PredictionMethod::Linear,
    }))
}

fn do_change_points(
    index: &dyn TemporalIndexAccess,
    entity_id: u64,
    start: i64,
    end: i64,
) -> Result<QueryResult, QueryError> {
    let (traj_data, vectors) = build_traj(index, entity_id, TemporalFilter::Range(start, end));
    if traj_data.len() < 4 {
        return Ok(QueryResult::ChangePoints(Vec::new()));
    }
    let traj = to_slices(&traj_data, &vectors);
    let cps = pelt::detect(entity_id, &traj, &PeltConfig::default());
    Ok(QueryResult::ChangePoints(cps))
}

fn do_drift_quant(
    index: &dyn TemporalIndexAccess,
    entity_id: u64,
    t1: i64,
    t2: i64,
    top_n: usize,
) -> Result<QueryResult, QueryError> {
    let p1 = find_nearest(index, entity_id, t1)?;
    let p2 = find_nearest(index, entity_id, t2)?;
    let v1 = index.vector(p1);
    let v2 = index.vector(p2);
    let report = calculus::drift_report(&v1, &v2, top_n);
    Ok(QueryResult::Drift(DriftResult {
        l2_magnitude: report.l2_magnitude,
        cosine_drift: report.cosine_drift,
        top_dimensions: report.top_dimensions,
    }))
}

fn do_cohort_drift(
    index: &dyn TemporalIndexAccess,
    entity_ids: &[u64],
    t1: i64,
    t2: i64,
    top_n: usize,
) -> Result<QueryResult, QueryError> {
    // Build trajectories for all entities
    let mut traj_data: Vec<(Vec<(i64, u32)>, Vec<Vec<f32>>)> = Vec::new();
    let mut valid_ids: Vec<u64> = Vec::new();

    for &eid in entity_ids {
        let (td, vecs) = build_traj(index, eid, TemporalFilter::All);
        if !td.is_empty() {
            traj_data.push((td, vecs));
            valid_ids.push(eid);
        }
    }

    if valid_ids.len() < 2 {
        return Err(QueryError::InsufficientData {
            needed: 2,
            have: valid_ids.len(),
        });
    }

    // Build slice-based trajectories for the cohort function
    let slice_trajs: Vec<Vec<(i64, &[f32])>> = traj_data
        .iter()
        .map(|(td, vecs)| to_slices(td, vecs))
        .collect();

    let input: Vec<(u64, &[(i64, &[f32])])> = valid_ids
        .iter()
        .zip(slice_trajs.iter())
        .map(|(&eid, st)| (eid, st.as_slice()))
        .collect();

    let report = cohort::cohort_drift(&input, t1, t2, top_n)
        .map_err(|_| QueryError::InsufficientData {
            needed: 2,
            have: valid_ids.len(),
        })?;

    Ok(QueryResult::CohortDrift(CohortDriftResult {
        n_entities: report.n_entities,
        mean_drift_l2: report.mean_drift_l2,
        median_drift_l2: report.median_drift_l2,
        std_drift_l2: report.std_drift_l2,
        centroid_l2_magnitude: report.centroid_drift.l2_magnitude,
        centroid_cosine_drift: report.centroid_drift.cosine_drift,
        dispersion_t1: report.dispersion_t1,
        dispersion_t2: report.dispersion_t2,
        dispersion_change: report.dispersion_change,
        convergence_score: report.convergence_score,
        top_dimensions: report.top_dimensions,
        outliers: report
            .outliers
            .into_iter()
            .map(|o| CohortOutlierResult {
                entity_id: o.entity_id,
                drift_magnitude: o.drift_magnitude,
                z_score: o.z_score,
                drift_direction_alignment: o.drift_direction_alignment,
            })
            .collect(),
    }))
}

fn do_analogy(
    index: &dyn TemporalIndexAccess,
    entity_a: u64,
    t1: i64,
    t2: i64,
    entity_b: u64,
    t3: i64,
) -> Result<QueryResult, QueryError> {
    let a1 = find_nearest(index, entity_a, t1)?;
    let a2 = find_nearest(index, entity_a, t2)?;
    let b3 = find_nearest(index, entity_b, t3)?;
    let va1 = index.vector(a1);
    let va2 = index.vector(a2);
    let vb3 = index.vector(b3);
    let result: Vec<f32> = vb3
        .iter()
        .zip(va2.iter().zip(va1.iter()))
        .map(|(&b, (&a2, &a1))| b + (a2 - a1))
        .collect();
    Ok(QueryResult::Analogy(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cvx_core::TemporalPoint;
    use cvx_index::hnsw::HnswConfig;
    use cvx_index::hnsw::temporal::TemporalHnsw;
    use cvx_index::metrics::L2Distance;
    use cvx_storage::memory::InMemoryStore;

    fn setup_engine(
        n_entities: u64,
        points_per_entity: usize,
        dim: usize,
    ) -> QueryEngine<TemporalHnsw<L2Distance>, InMemoryStore> {
        let config = HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 100,
            ..Default::default()
        };
        let mut index = TemporalHnsw::new(config, L2Distance);
        let store = InMemoryStore::new();

        for e in 0..n_entities {
            for i in 0..points_per_entity {
                let ts = (i as i64) * 1_000_000;
                let v: Vec<f32> = (0..dim)
                    .map(|d| (e as f32 * 10.0) + (i as f32 * 0.1) + (d as f32 * 0.01))
                    .collect();
                index.insert(e, ts, &v);
                store.put(0, &TemporalPoint::new(e, ts, v)).unwrap();
            }
        }

        QueryEngine::new(index, store)
    }

    #[test]
    fn snapshot_knn_returns_at_timestamp() {
        let engine = setup_engine(5, 20, 4);
        let result = engine
            .execute(TemporalQuery::SnapshotKnn {
                vector: vec![0.0; 4],
                timestamp: 5_000_000,
                k: 3,
            })
            .unwrap();

        if let QueryResult::Knn(results) = result {
            for r in &results {
                assert_eq!(r.timestamp, 5_000_000);
            }
        } else {
            panic!("expected Knn result");
        }
    }

    #[test]
    fn range_knn_returns_in_range() {
        let engine = setup_engine(5, 20, 4);
        let result = engine
            .execute(TemporalQuery::RangeKnn {
                vector: vec![0.0; 4],
                start: 3_000_000,
                end: 7_000_000,
                k: 10,
                alpha: 1.0,
            })
            .unwrap();

        if let QueryResult::Knn(results) = result {
            assert!(!results.is_empty());
            for r in &results {
                assert!(
                    r.timestamp >= 3_000_000 && r.timestamp <= 7_000_000,
                    "ts {} out of range",
                    r.timestamp
                );
            }
        } else {
            panic!("expected Knn result");
        }
    }

    #[test]
    fn trajectory_returns_all_points_ordered() {
        let engine = setup_engine(3, 20, 4);
        let result = engine
            .execute(TemporalQuery::Trajectory {
                entity_id: 1,
                filter: TemporalFilter::All,
            })
            .unwrap();

        if let QueryResult::Trajectory(points) = result {
            assert_eq!(points.len(), 20);
            for w in points.windows(2) {
                assert!(w[0].timestamp() <= w[1].timestamp());
            }
            for p in &points {
                assert_eq!(p.entity_id(), 1);
            }
        } else {
            panic!("expected Trajectory result");
        }
    }

    #[test]
    fn trajectory_with_range_filter() {
        let engine = setup_engine(1, 20, 4);
        let result = engine
            .execute(TemporalQuery::Trajectory {
                entity_id: 0,
                filter: TemporalFilter::Range(5_000_000, 10_000_000),
            })
            .unwrap();

        if let QueryResult::Trajectory(points) = result {
            assert_eq!(points.len(), 6);
        } else {
            panic!("expected Trajectory result");
        }
    }

    #[test]
    fn velocity_returns_vector() {
        let engine = setup_engine(1, 20, 4);
        let result = engine
            .execute(TemporalQuery::Velocity {
                entity_id: 0,
                timestamp: 10_000_000,
            })
            .unwrap();

        if let QueryResult::Velocity(vel) = result {
            assert_eq!(vel.len(), 4);
            for &v in &vel {
                assert!(v.is_finite());
            }
        } else {
            panic!("expected Velocity result");
        }
    }

    #[test]
    fn velocity_insufficient_data() {
        let config = HnswConfig::default();
        let index = TemporalHnsw::new(config, L2Distance);
        let store = InMemoryStore::new();
        let engine = QueryEngine::new(index, store);

        let result = engine.execute(TemporalQuery::Velocity {
            entity_id: 999,
            timestamp: 0,
        });
        assert!(result.is_err());
    }

    #[test]
    fn prediction_linear_extrapolation() {
        let engine = setup_engine(1, 20, 4);
        let result = engine
            .execute(TemporalQuery::Prediction {
                entity_id: 0,
                target_timestamp: 25_000_000,
            })
            .unwrap();

        if let QueryResult::Prediction(pred) = result {
            assert_eq!(pred.vector.len(), 4);
            assert_eq!(pred.timestamp, 25_000_000);
            assert!(matches!(pred.method, PredictionMethod::Linear));
        } else {
            panic!("expected Prediction result");
        }
    }

    #[test]
    fn changepoint_on_stationary() {
        let engine = setup_engine(1, 50, 2);
        let result = engine
            .execute(TemporalQuery::ChangePointDetect {
                entity_id: 0,
                start: 0,
                end: 50_000_000,
            })
            .unwrap();

        if let QueryResult::ChangePoints(cps) = result {
            assert!(
                cps.len() <= 5,
                "too many CPs on near-linear data: {}",
                cps.len()
            );
        } else {
            panic!("expected ChangePoints result");
        }
    }

    #[test]
    fn drift_quant_returns_report() {
        let engine = setup_engine(1, 20, 4);
        let result = engine
            .execute(TemporalQuery::DriftQuant {
                entity_id: 0,
                t1: 0,
                t2: 19_000_000,
                top_n: 3,
            })
            .unwrap();

        if let QueryResult::Drift(drift) = result {
            assert!(drift.l2_magnitude > 0.0);
            assert!(drift.top_dimensions.len() <= 3);
        } else {
            panic!("expected Drift result");
        }
    }

    #[test]
    fn analogy_computes_displacement() {
        let engine = setup_engine(3, 20, 4);
        let result = engine
            .execute(TemporalQuery::Analogy {
                entity_a: 0,
                t1: 0,
                t2: 10_000_000,
                entity_b: 1,
                t3: 5_000_000,
            })
            .unwrap();

        if let QueryResult::Analogy(vec) = result {
            assert_eq!(vec.len(), 4);
            for &v in &vec {
                assert!(v.is_finite());
            }
        } else {
            panic!("expected Analogy result");
        }
    }

    #[test]
    fn cohort_drift_via_engine() {
        let engine = setup_engine(5, 20, 4);
        let result = engine
            .execute(TemporalQuery::CohortDrift {
                entity_ids: vec![0, 1, 2, 3, 4],
                t1: 0,
                t2: 19_000_000,
                top_n: 3,
            })
            .unwrap();

        if let QueryResult::CohortDrift(report) = result {
            assert_eq!(report.n_entities, 5);
            assert!(report.mean_drift_l2 > 0.0);
            assert!(report.top_dimensions.len() <= 3);
            assert!(
                report.convergence_score > 0.0,
                "entities with similar drift patterns should show some convergence"
            );
        } else {
            panic!("expected CohortDrift result");
        }
    }

    #[test]
    fn cohort_drift_insufficient_entities() {
        let engine = setup_engine(1, 10, 4);
        let result = engine.execute(TemporalQuery::CohortDrift {
            entity_ids: vec![0],
            t1: 0,
            t2: 9_000_000,
            top_n: 3,
        });
        assert!(result.is_err());
    }

    #[test]
    fn analogy_unknown_entity() {
        let engine = setup_engine(1, 10, 4);
        let result = engine.execute(TemporalQuery::Analogy {
            entity_a: 999,
            t1: 0,
            t2: 1000,
            entity_b: 0,
            t3: 0,
        });
        assert!(result.is_err());
    }
}
