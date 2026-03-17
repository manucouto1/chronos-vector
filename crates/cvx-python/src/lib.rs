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

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use cvx_analytics::calculus;
use cvx_analytics::ode;
use cvx_analytics::pelt::{self, PeltConfig};
use cvx_analytics::temporal_ml::{AnalyticBackend, TemporalOps};
use cvx_core::TemporalFilter;
use cvx_index::hnsw::{HnswConfig, temporal::TemporalHnsw};
use cvx_index::metrics::L2Distance;

/// Python-exposed temporal vector index.
///
/// Optionally loads a TorchScript Neural ODE model for prediction.
/// If no model is provided, prediction uses linear extrapolation.
#[pyclass]
struct TemporalIndex {
    inner: TemporalHnsw<L2Distance>,
    #[cfg(feature = "torch-backend")]
    torch_model: Option<std::sync::Arc<cvx_analytics::torch_ode::TorchOdeModel>>,
}

#[pymethods]
impl TemporalIndex {
    /// Create a new temporal index.
    ///
    /// Args:
    ///     m: HNSW connections per node (default 16).
    ///     ef_construction: Search width during construction (default 200).
    ///     ef_search: Search width during queries (default 50).
    ///     model_path: Optional path to TorchScript Neural ODE model (.pt).
    #[new]
    #[pyo3(signature = (m=16, ef_construction=200, ef_search=50, model_path=None))]
    fn new(m: usize, ef_construction: usize, ef_search: usize, model_path: Option<String>) -> PyResult<Self> {
        let config = HnswConfig {
            m,
            ef_construction,
            ef_search,
            ..Default::default()
        };

        #[cfg(feature = "torch-backend")]
        let torch_model = if let Some(path) = model_path {
            match cvx_analytics::torch_ode::TorchOdeModel::load(std::path::Path::new(&path)) {
                Ok(model) => Some(std::sync::Arc::new(model)),
                Err(e) => {
                    return Err(pyo3::exceptions::PyIOError::new_err(
                        format!("Failed to load model: {e}")
                    ));
                }
            }
        } else {
            None
        };

        #[cfg(not(feature = "torch-backend"))]
        if model_path.is_some() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "torch-backend feature not enabled. Build with: maturin develop --features torch-backend"
            ));
        }

        Ok(Self {
            inner: TemporalHnsw::new(config, L2Distance),
            #[cfg(feature = "torch-backend")]
            torch_model,
        })
    }

    /// Insert a temporal point.
    fn insert(&mut self, entity_id: u64, timestamp: i64, vector: Vec<f32>) -> u32 {
        self.inner.insert(entity_id, timestamp, &vector)
    }

    /// Bulk insert from numpy arrays.
    ///
    /// Significantly faster than individual insert() calls: avoids per-call
    /// Python↔Rust overhead and optionally lowers ef_construction during load.
    ///
    /// Args:
    ///     entity_ids: np.ndarray[uint64] of shape (N,)
    ///     timestamps: np.ndarray[int64] of shape (N,)
    ///     vectors: np.ndarray[float32] of shape (N, D)
    ///     ef_construction: Optional reduced ef for faster ingestion (restored after).
    ///
    /// Returns:
    ///     Number of points inserted.
    #[pyo3(signature = (entity_ids, timestamps, vectors, ef_construction=None))]
    fn bulk_insert(
        &mut self,
        entity_ids: PyReadonlyArray1<u64>,
        timestamps: PyReadonlyArray1<i64>,
        vectors: PyReadonlyArray2<f32>,
        ef_construction: Option<usize>,
    ) -> PyResult<usize> {
        let ids = entity_ids.as_slice()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("entity_ids: {e}")))?;
        let ts = timestamps.as_slice()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("timestamps: {e}")))?;
        let vecs = vectors.as_array();
        let n = ids.len();

        if ts.len() != n || vecs.nrows() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Length mismatch: entity_ids={n}, timestamps={}, vectors={}",
                        ts.len(), vecs.nrows())
            ));
        }

        // Optionally lower ef_construction for bulk load
        let original_ef = self.inner.config().ef_construction;
        if let Some(ef) = ef_construction {
            self.inner.set_ef_construction(ef);
        }

        for i in 0..n {
            let row = vecs.row(i);
            self.inner.insert(ids[i], ts[i], row.as_slice().unwrap());
        }

        // Restore original ef_construction
        if ef_construction.is_some() {
            self.inner.set_ef_construction(original_ef);
        }

        Ok(n)
    }

    /// Set ef_construction at runtime.
    ///
    /// Lower values (50-100) speed up ingestion at slight recall cost.
    /// Higher values (200+) improve construction quality.
    fn set_ef_construction(&mut self, ef: usize) {
        self.inner.set_ef_construction(ef);
    }

    /// Set ef_search at runtime.
    ///
    /// Trade search speed for accuracy per query.
    fn set_ef_search(&mut self, ef: usize) {
        self.inner.set_ef_search(ef);
    }

    /// Enable scalar quantization for ~4× faster distance computation.
    ///
    /// Encodes each vector dimension as uint8. Candidate distances use
    /// fast integer arithmetic; final results use exact float32.
    ///
    /// Args:
    ///     min_val: Expected minimum value per dimension (default -1.0 for normalized).
    ///     max_val: Expected maximum value per dimension (default 1.0 for normalized).
    #[pyo3(signature = (min_val=-1.0, max_val=1.0))]
    fn enable_quantization(&mut self, min_val: f32, max_val: f32) {
        self.inner.enable_scalar_quantization(min_val, max_val);
    }

    /// Disable scalar quantization.
    fn disable_quantization(&mut self) {
        self.inner.disable_scalar_quantization();
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

    /// Predict a future vector for an entity.
    ///
    /// Uses Neural ODE if a model was loaded, otherwise linear extrapolation.
    ///
    /// Args:
    ///     entity_id: Entity to predict for.
    ///     target_timestamp: Target time for prediction.
    ///
    /// Returns:
    ///     Tuple of (predicted_vector, method) where method is "neural_ode" or "linear".
    fn predict(&self, entity_id: u64, target_timestamp: i64) -> PyResult<(Vec<f32>, String)> {
        let traj_data = self.inner.trajectory(entity_id, TemporalFilter::All);
        if traj_data.len() < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("entity {entity_id} has insufficient data ({} points, need 2)", traj_data.len())
            ));
        }

        let vectors: Vec<Vec<f32>> = traj_data
            .iter()
            .map(|&(_, node_id)| self.inner.vector(node_id).to_vec())
            .collect();
        let traj: Vec<(i64, &[f32])> = traj_data
            .iter()
            .zip(vectors.iter())
            .map(|(&(ts, _), v)| (ts, v.as_slice()))
            .collect();

        // Try Neural ODE first
        #[cfg(feature = "torch-backend")]
        if let Some(ref model) = self.torch_model {
            match model.predict(&traj, target_timestamp) {
                Ok(predicted) => return Ok((predicted, "neural_ode".into())),
                Err(_) => {} // Fall through to linear
            }
        }

        // Fallback: linear extrapolation
        let predicted = ode::linear_extrapolate(&traj, target_timestamp)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok((predicted, "linear".into()))
    }

    /// Whether a Neural ODE model is loaded.
    fn has_neural_ode(&self) -> bool {
        #[cfg(feature = "torch-backend")]
        {
            self.torch_model.is_some()
        }
        #[cfg(not(feature = "torch-backend"))]
        {
            false
        }
    }

    // ─── Semantic Regions (RFC-004) ──────────────────────────────

    /// Get semantic regions from the HNSW graph hierarchy.
    ///
    /// Args:
    ///     level: HNSW level (higher = fewer, coarser regions).
    ///            Level 2 typically gives ~N/256 regions.
    ///
    /// Returns:
    ///     List of (region_id, centroid_vector, n_members).
    #[pyo3(signature = (level=2))]
    fn regions(&self, level: usize) -> Vec<(u32, Vec<f32>, usize)> {
        self.inner.regions(level)
    }

    /// Compute smoothed region-distribution trajectory for an entity.
    ///
    /// Tracks how the user's posts distribute across semantic regions
    /// over time, smoothed with Exponential Moving Average.
    ///
    /// Args:
    ///     entity_id: Entity to analyze.
    ///     level: HNSW level for region granularity (default 2).
    ///     window_days: Sliding window width in timestamp units (default 7).
    ///     alpha: EMA smoothing factor, 0-1 (default 0.3).
    ///
    /// Returns:
    ///     List of (timestamp, region_distribution) tuples.
    ///     Each distribution is a list of floats summing to ~1.0.
    #[pyo3(signature = (entity_id, level=2, window_days=7, alpha=0.3))]
    fn region_trajectory(
        &self,
        entity_id: u64,
        level: usize,
        window_days: i64,
        alpha: f32,
    ) -> Vec<(i64, Vec<f32>)> {
        self.inner.region_trajectory(entity_id, level, window_days, alpha)
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
/// Args:
///     entity_id: Entity identifier.
///     trajectory: List of (timestamp, vector) tuples.
///     penalty: Optional penalty per change point. If None, uses BIC (dim * ln(n) / 2).
///              For high-dimensional embeddings (D>100), consider using 3*ln(n) or lower.
///     min_segment_len: Minimum segment length (default 2).
///
/// Returns:
///     List of (timestamp, severity) tuples.
#[pyfunction]
#[pyo3(signature = (entity_id, trajectory, penalty=None, min_segment_len=2))]
fn detect_changepoints(
    entity_id: u64,
    trajectory: Vec<(i64, Vec<f32>)>,
    penalty: Option<f64>,
    min_segment_len: usize,
) -> Vec<(i64, f64)> {
    let traj: Vec<(i64, &[f32])> = trajectory.iter().map(|(t, v)| (*t, v.as_slice())).collect();
    let config = PeltConfig {
        penalty,
        min_segment_len,
    };
    pelt::detect(entity_id, &traj, &config)
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

/// Predict a future vector using linear extrapolation.
///
/// For Neural ODE prediction, use `TemporalIndex(model_path="model.pt").predict()`.
#[pyfunction]
fn predict(trajectory: Vec<(i64, Vec<f32>)>, target_timestamp: i64) -> PyResult<Vec<f32>> {
    let traj: Vec<(i64, &[f32])> = trajectory.iter().map(|(t, v)| (*t, v.as_slice())).collect();
    ode::linear_extrapolate(&traj, target_timestamp)
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
    m.add_function(wrap_pyfunction!(predict, m)?)?;
    Ok(())
}
