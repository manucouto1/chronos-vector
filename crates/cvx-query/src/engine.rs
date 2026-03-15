//! Query execution engine.
//!
//! Orchestrates index, storage, and analytics to execute temporal queries.

use cvx_core::StorageBackend;
use cvx_core::error::QueryError;
use cvx_core::types::TemporalFilter;

use cvx_analytics::calculus;
use cvx_analytics::ode;
use cvx_analytics::pelt::{self, PeltConfig};

use cvx_index::hnsw::temporal::TemporalHnsw;
use cvx_index::metrics::L2Distance;

use crate::types::*;

/// Query engine that coordinates index + storage + analytics.
pub struct QueryEngine<S: StorageBackend> {
    index: TemporalHnsw<L2Distance>,
    store: S,
}

impl<S: StorageBackend> QueryEngine<S> {
    /// Create a new query engine.
    pub fn new(index: TemporalHnsw<L2Distance>, store: S) -> Self {
        Self { index, store }
    }

    /// Execute a temporal query.
    pub fn execute(&self, query: TemporalQuery) -> Result<QueryResult, QueryError> {
        match query {
            TemporalQuery::SnapshotKnn {
                vector,
                timestamp,
                k,
            } => self.snapshot_knn(&vector, timestamp, k),

            TemporalQuery::RangeKnn {
                vector,
                start,
                end,
                k,
                alpha,
            } => self.range_knn(&vector, start, end, k, alpha),

            TemporalQuery::Trajectory { entity_id, filter } => self.trajectory(entity_id, filter),

            TemporalQuery::Velocity {
                entity_id,
                timestamp,
            } => self.velocity(entity_id, timestamp),

            TemporalQuery::Prediction {
                entity_id,
                target_timestamp,
            } => self.prediction(entity_id, target_timestamp),

            TemporalQuery::ChangePointDetect {
                entity_id,
                start,
                end,
            } => self.change_points(entity_id, start, end),

            TemporalQuery::DriftQuant {
                entity_id,
                t1,
                t2,
                top_n,
            } => self.drift_quant(entity_id, t1, t2, top_n),

            TemporalQuery::Analogy {
                entity_a,
                t1,
                t2,
                entity_b,
                t3,
            } => self.analogy(entity_a, t1, t2, entity_b, t3),
        }
    }

    /// Snapshot kNN: nearest neighbors at exact timestamp.
    fn snapshot_knn(
        &self,
        vector: &[f32],
        timestamp: i64,
        k: usize,
    ) -> Result<QueryResult, QueryError> {
        let results = self.index.search(
            vector,
            k,
            TemporalFilter::Snapshot(timestamp),
            1.0,
            timestamp,
        );

        Ok(QueryResult::Knn(
            results
                .into_iter()
                .map(|(node_id, score)| KnnResult {
                    entity_id: self.index.entity_id(node_id),
                    timestamp: self.index.timestamp(node_id),
                    score,
                })
                .collect(),
        ))
    }

    /// Range kNN: nearest neighbors over time window with alpha blending.
    fn range_knn(
        &self,
        vector: &[f32],
        start: i64,
        end: i64,
        k: usize,
        alpha: f32,
    ) -> Result<QueryResult, QueryError> {
        let mid = start + (end - start) / 2;
        let results = self
            .index
            .search(vector, k, TemporalFilter::Range(start, end), alpha, mid);

        Ok(QueryResult::Knn(
            results
                .into_iter()
                .map(|(node_id, score)| KnnResult {
                    entity_id: self.index.entity_id(node_id),
                    timestamp: self.index.timestamp(node_id),
                    score,
                })
                .collect(),
        ))
    }

    /// Trajectory: all points for an entity.
    fn trajectory(
        &self,
        entity_id: u64,
        filter: TemporalFilter,
    ) -> Result<QueryResult, QueryError> {
        let traj = self.index.trajectory(entity_id, filter);

        let points: Result<Vec<_>, _> = traj
            .iter()
            .map(|&(ts, node_id)| {
                let vector = self.index.vector(node_id).to_vec();
                Ok(cvx_core::TemporalPoint::new(entity_id, ts, vector))
            })
            .collect();

        Ok(QueryResult::Trajectory(points?))
    }

    /// Velocity at a given timestamp.
    fn velocity(&self, entity_id: u64, timestamp: i64) -> Result<QueryResult, QueryError> {
        let traj_data = self.index.trajectory(entity_id, TemporalFilter::All);
        if traj_data.len() < 2 {
            return Err(QueryError::InsufficientData {
                needed: 2,
                have: traj_data.len(),
            });
        }

        let vectors: Vec<Vec<f32>> = traj_data
            .iter()
            .map(|&(_, node_id)| self.index.vector(node_id).to_vec())
            .collect();
        let traj: Vec<(i64, &[f32])> = traj_data
            .iter()
            .zip(vectors.iter())
            .map(|(&(ts, _), v)| (ts, v.as_slice()))
            .collect();

        let vel =
            calculus::velocity(&traj, timestamp).map_err(|_| QueryError::InsufficientData {
                needed: 2,
                have: traj.len(),
            })?;

        Ok(QueryResult::Velocity(vel))
    }

    /// Prediction: extrapolate future vector.
    fn prediction(&self, entity_id: u64, target_timestamp: i64) -> Result<QueryResult, QueryError> {
        let traj_data = self.index.trajectory(entity_id, TemporalFilter::All);
        if traj_data.len() < 2 {
            return Err(QueryError::InsufficientData {
                needed: 2,
                have: traj_data.len(),
            });
        }

        let vectors: Vec<Vec<f32>> = traj_data
            .iter()
            .map(|&(_, node_id)| self.index.vector(node_id).to_vec())
            .collect();
        let traj: Vec<(i64, &[f32])> = traj_data
            .iter()
            .zip(vectors.iter())
            .map(|(&(ts, _), v)| (ts, v.as_slice()))
            .collect();

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

    /// Change point detection on a time window.
    fn change_points(
        &self,
        entity_id: u64,
        start: i64,
        end: i64,
    ) -> Result<QueryResult, QueryError> {
        let traj_data = self
            .index
            .trajectory(entity_id, TemporalFilter::Range(start, end));
        if traj_data.len() < 4 {
            return Ok(QueryResult::ChangePoints(Vec::new()));
        }

        let vectors: Vec<Vec<f32>> = traj_data
            .iter()
            .map(|&(_, node_id)| self.index.vector(node_id).to_vec())
            .collect();
        let traj: Vec<(i64, &[f32])> = traj_data
            .iter()
            .zip(vectors.iter())
            .map(|(&(ts, _), v)| (ts, v.as_slice()))
            .collect();

        let cps = pelt::detect(entity_id, &traj, &PeltConfig::default());
        Ok(QueryResult::ChangePoints(cps))
    }

    /// Drift quantification between two timestamps.
    fn drift_quant(
        &self,
        entity_id: u64,
        t1: i64,
        t2: i64,
        top_n: usize,
    ) -> Result<QueryResult, QueryError> {
        let p1 = self.find_nearest_point(entity_id, t1)?;
        let p2 = self.find_nearest_point(entity_id, t2)?;

        let v1 = self.index.vector(p1);
        let v2 = self.index.vector(p2);

        let report = calculus::drift_report(v1, v2, top_n);
        Ok(QueryResult::Drift(DriftResult {
            l2_magnitude: report.l2_magnitude,
            cosine_drift: report.cosine_drift,
            top_dimensions: report.top_dimensions,
        }))
    }

    /// Temporal analogy: A@t1 → A@t2 displacement applied to B@t3.
    fn analogy(
        &self,
        entity_a: u64,
        t1: i64,
        t2: i64,
        entity_b: u64,
        t3: i64,
    ) -> Result<QueryResult, QueryError> {
        let a1 = self.find_nearest_point(entity_a, t1)?;
        let a2 = self.find_nearest_point(entity_a, t2)?;
        let b3 = self.find_nearest_point(entity_b, t3)?;

        let va1 = self.index.vector(a1);
        let va2 = self.index.vector(a2);
        let vb3 = self.index.vector(b3);

        // analogy = B@t3 + (A@t2 - A@t1)
        let result: Vec<f32> = vb3
            .iter()
            .zip(va2.iter().zip(va1.iter()))
            .map(|(&b, (&a2, &a1))| b + (a2 - a1))
            .collect();

        Ok(QueryResult::Analogy(result))
    }

    /// Find the node closest in time for an entity.
    fn find_nearest_point(&self, entity_id: u64, timestamp: i64) -> Result<u32, QueryError> {
        let traj = self.index.trajectory(entity_id, TemporalFilter::All);
        if traj.is_empty() {
            return Err(QueryError::EntityNotFound(entity_id));
        }

        let (_, node_id) = traj
            .iter()
            .min_by_key(|&&(ts, _)| (ts - timestamp).unsigned_abs())
            .unwrap();

        Ok(*node_id)
    }

    /// Access the underlying index.
    pub fn index(&self) -> &TemporalHnsw<L2Distance> {
        &self.index
    }

    /// Access the underlying store.
    pub fn store(&self) -> &S {
        &self.store
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cvx_core::TemporalPoint;
    use cvx_index::hnsw::HnswConfig;
    use cvx_storage::memory::InMemoryStore;

    fn setup_engine(
        n_entities: u64,
        points_per_entity: usize,
        dim: usize,
    ) -> QueryEngine<InMemoryStore> {
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

    // ─── SnapshotKnn ────────────────────────────────────────────────

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

    // ─── RangeKnn ───────────────────────────────────────────────────

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

    // ─── Trajectory ─────────────────────────────────────────────────

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
            assert_eq!(points.len(), 6); // 5M, 6M, 7M, 8M, 9M, 10M
        } else {
            panic!("expected Trajectory result");
        }
    }

    // ─── Velocity ───────────────────────────────────────────────────

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
            // Linear trajectory → velocity should be approximately constant
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

    // ─── Prediction ─────────────────────────────────────────────────

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

    // ─── ChangePoints ───────────────────────────────────────────────

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
            // Linear trajectory with small slope → might detect 0 or few
            // The important thing is it doesn't crash
            assert!(
                cps.len() <= 5,
                "too many CPs on near-linear data: {}",
                cps.len()
            );
        } else {
            panic!("expected ChangePoints result");
        }
    }

    // ─── DriftQuant ─────────────────────────────────────────────────

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

    // ─── Analogy ────────────────────────────────────────────────────

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
            // Result = B@t3 + (A@t2 - A@t1), should be finite
            for &v in &vec {
                assert!(v.is_finite());
            }
        } else {
            panic!("expected Analogy result");
        }
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
