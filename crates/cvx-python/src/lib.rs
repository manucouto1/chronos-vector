//! Python bindings for ChronosVector via PyO3.
//!
//! Exposes the core CVX functionality to Python:
//!
//! ```python
//! import chronos_vector as cvx
//!
//! # Create an index
//! index = cvx.TemporalIndex()
//!
//! # Insert vectors
//! index.insert(entity_id=1, timestamp=1000, vector=[0.1, 0.2, 0.3])
//!
//! # Search
//! results = index.search(vector=[0.1, 0.2, 0.3], k=5)
//!
//! # Trajectory
//! traj = index.trajectory(entity_id=1)
//!
//! # Analytics
//! vel = cvx.velocity(trajectory, timestamp=5000)
//! features = cvx.temporal_features(trajectory)
//! ```

use pyo3::prelude::*;

use cvx_analytics::calculus;
use cvx_analytics::pelt::{self, PeltConfig};
use cvx_analytics::temporal_ml::{AnalyticBackend, TemporalOps};
use cvx_core::TemporalFilter;
use cvx_index::hnsw::{HnswConfig, temporal::TemporalHnsw};
use cvx_index::metrics::L2Distance;

/// Python-exposed temporal vector index.
#[pyclass]
struct TemporalIndex {
    inner: TemporalHnsw<L2Distance>,
}

#[pymethods]
impl TemporalIndex {
    /// Create a new temporal index.
    #[new]
    #[pyo3(signature = (m=16, ef_construction=200, ef_search=50))]
    fn new(m: usize, ef_construction: usize, ef_search: usize) -> Self {
        let config = HnswConfig {
            m,
            ef_construction,
            ef_search,
            ..Default::default()
        };
        Self {
            inner: TemporalHnsw::new(config, L2Distance),
        }
    }

    /// Insert a temporal point.
    fn insert(&mut self, entity_id: u64, timestamp: i64, vector: Vec<f32>) -> u32 {
        self.inner.insert(entity_id, timestamp, &vector)
    }

    /// Search for k nearest neighbors.
    #[pyo3(signature = (vector, k=10, alpha=1.0, query_timestamp=0, filter_start=None, filter_end=None))]
    fn search(
        &self,
        vector: Vec<f32>,
        k: usize,
        alpha: f32,
        query_timestamp: i64,
        filter_start: Option<i64>,
        filter_end: Option<i64>,
    ) -> Vec<(u64, i64, f32)> {
        let filter = match (filter_start, filter_end) {
            (Some(start), Some(end)) => TemporalFilter::Range(start, end),
            _ => TemporalFilter::All,
        };

        self.inner
            .search(&vector, k, filter, alpha, query_timestamp)
            .into_iter()
            .map(|(node_id, score)| {
                let eid = self.inner.entity_id(node_id);
                let ts = self.inner.timestamp(node_id);
                (eid, ts, score)
            })
            .collect()
    }

    /// Get trajectory for an entity.
    #[pyo3(signature = (entity_id, start=None, end=None))]
    fn trajectory(
        &self,
        entity_id: u64,
        start: Option<i64>,
        end: Option<i64>,
    ) -> Vec<(i64, Vec<f32>)> {
        let filter = match (start, end) {
            (Some(s), Some(e)) => TemporalFilter::Range(s, e),
            _ => TemporalFilter::All,
        };

        self.inner
            .trajectory(entity_id, filter)
            .into_iter()
            .map(|(ts, node_id)| {
                let vec = self.inner.vector(node_id).to_vec();
                (ts, vec)
            })
            .collect()
    }

    /// Number of points in the index.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// String representation.
    fn __repr__(&self) -> String {
        format!("TemporalIndex(len={})", self.inner.len())
    }
}

/// Compute velocity at a timestamp from a trajectory.
///
/// Args:
///     trajectory: List of (timestamp, vector) tuples.
///     timestamp: Time at which to compute velocity.
///
/// Returns:
///     Velocity vector.
#[pyfunction]
fn velocity(trajectory: Vec<(i64, Vec<f32>)>, timestamp: i64) -> PyResult<Vec<f32>> {
    let traj: Vec<(i64, &[f32])> = trajectory.iter().map(|(t, v)| (*t, v.as_slice())).collect();
    calculus::velocity(&traj, timestamp)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Compute drift between two vectors.
///
/// Returns:
///     (l2_magnitude, cosine_drift, top_dimensions)
#[pyfunction]
#[pyo3(signature = (v1, v2, top_n=5))]
fn drift(v1: Vec<f32>, v2: Vec<f32>, top_n: usize) -> (f32, f32, Vec<(usize, f32)>) {
    let report = calculus::drift_report(&v1, &v2, top_n);
    (report.l2_magnitude, report.cosine_drift, report.top_dimensions)
}

/// Detect change points using PELT.
///
/// Returns:
///     List of (timestamp, severity) tuples.
#[pyfunction]
fn detect_changepoints(
    entity_id: u64,
    trajectory: Vec<(i64, Vec<f32>)>,
) -> Vec<(i64, f64)> {
    let traj: Vec<(i64, &[f32])> = trajectory.iter().map(|(t, v)| (*t, v.as_slice())).collect();
    pelt::detect(entity_id, &traj, &PeltConfig::default())
        .into_iter()
        .map(|cp| (cp.timestamp(), cp.severity()))
        .collect()
}

/// Extract fixed-size temporal features from a trajectory.
///
/// Returns:
///     Feature vector of size 2*D + 2 + 3.
#[pyfunction]
fn temporal_features(trajectory: Vec<(i64, Vec<f32>)>) -> PyResult<Vec<f32>> {
    let traj: Vec<(i64, &[f32])> = trajectory.iter().map(|(t, v)| (*t, v.as_slice())).collect();
    let backend = AnalyticBackend::new();
    backend
        .extract_features(&traj)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Hurst exponent estimation.
#[pyfunction]
fn hurst_exponent(trajectory: Vec<(i64, Vec<f32>)>) -> PyResult<f32> {
    let traj: Vec<(i64, &[f32])> = trajectory.iter().map(|(t, v)| (*t, v.as_slice())).collect();
    calculus::hurst_exponent(&traj)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// ChronosVector Python module.
#[pymodule]
fn chronos_vector(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TemporalIndex>()?;
    m.add_function(wrap_pyfunction!(velocity, m)?)?;
    m.add_function(wrap_pyfunction!(drift, m)?)?;
    m.add_function(wrap_pyfunction!(detect_changepoints, m)?)?;
    m.add_function(wrap_pyfunction!(temporal_features, m)?)?;
    m.add_function(wrap_pyfunction!(hurst_exponent, m)?)?;
    Ok(())
}
