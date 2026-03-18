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
            // as_slice() can fail if the array isn't C-contiguous
            match row.as_slice() {
                Some(slice) => { self.inner.insert(ids[i], ts[i], slice); }
                None => {
                    let vec: Vec<f32> = row.iter().copied().collect();
                    self.inner.insert(ids[i], ts[i], &vec);
                }
            }
        }

        // Restore original ef_construction
        if ef_construction.is_some() {
            self.inner.set_ef_construction(original_ef);
        }

        Ok(n)
    }

    /// Save the index to a file for fast reload.
    ///
    /// Persists the full HNSW graph structure (nodes, edges, quantization,
    /// timestamps, entity mappings) so it can be loaded without rebuilding.
    ///
    /// Args:
    ///     path: File path to save to (e.g., "index.cvx").
    fn save(&self, path: String) -> PyResult<()> {
        self.inner.save(std::path::Path::new(&path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to save index: {e}")))
    }

    /// Load a previously saved index from file.
    ///
    /// Restores the full HNSW graph without rebuilding.
    /// Typically ~100× faster than bulk_insert for large indices.
    ///
    /// Args:
    ///     path: File path to load from.
    ///
    /// Returns:
    ///     A new TemporalIndex with all data and graph structure restored.
    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        let inner = TemporalHnsw::load(std::path::Path::new(&path), L2Distance)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to load index: {e}")))?;
        Ok(Self {
            inner,
            #[cfg(feature = "torch-backend")]
            torch_model: None,
        })
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

    /// Get all points in a specific region, optionally time-filtered (RFC-005).
    ///
    /// Args:
    ///     region_id: The hub node_id of the region (from regions()).
    ///     level: HNSW level (default 3 for coarsest).
    ///     start: Optional start timestamp filter.
    ///     end: Optional end timestamp filter.
    ///
    /// Returns:
    ///     List of (entity_id, timestamp) for each point in the region.
    #[pyo3(signature = (region_id, level=3, start=None, end=None))]
    fn region_members(
        &self,
        region_id: u32,
        level: usize,
        start: Option<i64>,
        end: Option<i64>,
    ) -> Vec<(u64, i64)> {
        let filter = match (start, end) {
            (Some(s), Some(e)) => TemporalFilter::Range(s, e),
            _ => TemporalFilter::All,
        };
        self.inner
            .region_members(region_id, level, filter)
            .into_iter()
            .map(|(_node_id, entity_id, timestamp)| (entity_id, timestamp))
            .collect()
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

/// Compute the truncated path signature of a trajectory.
///
/// The signature is a universal, order-aware feature of sequential data from
/// rough path theory. It captures the shape of the trajectory — not just where
/// it ends, but how it gets there.
///
/// Args:
///     trajectory: List of (timestamp, vector) tuples.
///     depth: Truncation depth (1-3). Depth 2 captures signed areas. Default 2.
///     time_augmentation: Add time as extra dimension (captures speed). Default false.
///
/// Returns:
///     Signature vector. Size depends on dim K and depth:
///     - Depth 1: K features
///     - Depth 2: K + K² features
///     - Depth 3: K + K² + K³ features
#[pyfunction]
#[pyo3(signature = (trajectory, depth=2, time_augmentation=false))]
fn path_signature(
    trajectory: Vec<(i64, Vec<f32>)>,
    depth: usize,
    time_augmentation: bool,
) -> PyResult<Vec<f64>> {
    let traj: Vec<(i64, &[f32])> = trajectory.iter().map(|(t, v)| (*t, v.as_slice())).collect();
    let config = cvx_analytics::signatures::SignatureConfig { depth, time_augmentation };
    cvx_analytics::signatures::compute_signature(&traj, &config)
        .map(|r| r.signature)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Compute the log-signature (compact version of path signature).
///
/// Removes redundant symmetric components at depth 2. Same information,
/// fewer dimensions: K + K(K-1)/2 instead of K + K².
#[pyfunction]
#[pyo3(signature = (trajectory, depth=2, time_augmentation=false))]
fn log_signature(
    trajectory: Vec<(i64, Vec<f32>)>,
    depth: usize,
    time_augmentation: bool,
) -> PyResult<Vec<f64>> {
    let traj: Vec<(i64, &[f32])> = trajectory.iter().map(|(t, v)| (*t, v.as_slice())).collect();
    let config = cvx_analytics::signatures::SignatureConfig { depth, time_augmentation };
    cvx_analytics::signatures::compute_log_signature(&traj, &config)
        .map(|r| r.signature)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Compute discrete Fréchet distance between two trajectories.
///
/// Measures the maximum minimum distance between corresponding points
/// when both paths are traversed monotonically (the "dog-walking" distance).
///
/// Args:
///     traj_a: List of (timestamp, vector) tuples.
///     traj_b: List of (timestamp, vector) tuples.
///
/// Returns:
///     Fréchet distance (float). Lower = more similar paths.
#[pyfunction]
fn frechet_distance(
    traj_a: Vec<(i64, Vec<f32>)>,
    traj_b: Vec<(i64, Vec<f32>)>,
) -> f64 {
    let a: Vec<(i64, &[f32])> = traj_a.iter().map(|(t, v)| (*t, v.as_slice())).collect();
    let b: Vec<(i64, &[f32])> = traj_b.iter().map(|(t, v)| (*t, v.as_slice())).collect();
    cvx_analytics::trajectory::discrete_frechet_temporal(&a, &b)
}

/// Compute distance between two path signatures.
///
/// Fast trajectory similarity: O(output_dim) per comparison.
/// Captures all order-dependent temporal dynamics.
#[pyfunction]
fn signature_distance(sig_a: Vec<f64>, sig_b: Vec<f64>) -> f64 {
    sig_a.iter().zip(sig_b.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}

/// Extract temporal point process features from event timestamps.
///
/// Analyzes the *when* of events as an independent signal from *what* the vectors
/// contain. Returns features characterizing temporal patterns: regularity,
/// burstiness, memory, and intensity trends.
///
/// Args:
///     timestamps: Sorted list of event timestamps (at least 3).
///
/// Returns:
///     Dict with keys: n_events, span, mean_gap, std_gap, burstiness, memory,
///     temporal_entropy, intensity_trend, gap_cv, max_gap, circadian_strength.
#[pyfunction]
fn event_features(timestamps: Vec<i64>) -> PyResult<std::collections::HashMap<String, f64>> {
    let f = cvx_analytics::point_process::extract_event_features(&timestamps)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let mut map = std::collections::HashMap::new();
    map.insert("n_events".into(), f.n_events as f64);
    map.insert("span".into(), f.span);
    map.insert("mean_gap".into(), f.mean_gap);
    map.insert("std_gap".into(), f.std_gap);
    map.insert("burstiness".into(), f.burstiness);
    map.insert("memory".into(), f.memory);
    map.insert("temporal_entropy".into(), f.temporal_entropy);
    map.insert("intensity_trend".into(), f.intensity_trend);
    map.insert("gap_cv".into(), f.gap_cv);
    map.insert("max_gap".into(), f.max_gap);
    map.insert("circadian_strength".into(), f.circadian_strength);
    Ok(map)
}

/// Fisher-Rao distance between two categorical distributions.
///
/// The unique Riemannian metric on the statistical manifold that is
/// invariant under sufficient statistics. Range: [0, π].
/// 0 = identical, π = completely orthogonal (disjoint support).
///
/// More principled than L2 or KL divergence for comparing region distributions.
///
/// Args:
///     p: First distribution (list of floats summing to ~1.0).
///     q: Second distribution.
///
/// Returns:
///     Fisher-Rao distance in [0, π].
#[pyfunction]
fn fisher_rao_distance(p: Vec<f64>, q: Vec<f64>) -> f64 {
    cvx_analytics::fisher_rao::fisher_rao_distance(&p, &q)
}

/// Hellinger distance between two distributions. Range: [0, 1].
///
/// Related to Fisher-Rao but bounded in [0, 1] for convenience.
#[pyfunction]
fn hellinger_distance(p: Vec<f64>, q: Vec<f64>) -> f64 {
    cvx_analytics::fisher_rao::hellinger_distance(&p, &q)
}

/// Compute topological features of a point cloud via persistent homology.
///
/// Tracks connected components (β₀) as filtration radius grows.
/// Detects cluster structure: fragmentation, convergence, prominent gaps.
///
/// Best applied on region centroids (from index.regions()), NOT raw points.
///
/// Args:
///     points: List of vectors (the point cloud).
///     n_radii: Number of radii for Betti curve sampling (default 20).
///     persistence_threshold: Min persistence to count as significant (default 0.1).
///
/// Returns:
///     Dict with: n_components, total_persistence, max_persistence,
///     mean_persistence, persistence_entropy, betti_curve, radii.
#[pyfunction]
#[pyo3(signature = (points, n_radii=20, persistence_threshold=0.1))]
fn topological_features(
    points: Vec<Vec<f32>>,
    n_radii: usize,
    persistence_threshold: f64,
) -> std::collections::HashMap<String, PyObject> {
    use pyo3::IntoPyObject;

    let point_refs: Vec<&[f32]> = points.iter().map(|p| p.as_slice()).collect();
    let feat = cvx_analytics::topology::topological_summary(&point_refs, n_radii, persistence_threshold);

    Python::with_gil(|py| {
        let mut map = std::collections::HashMap::new();
        map.insert("n_components".into(), feat.n_components.into_pyobject(py).unwrap().into_any().unbind());
        map.insert("total_persistence".into(), feat.total_persistence_h0.into_pyobject(py).unwrap().into_any().unbind());
        map.insert("max_persistence".into(), feat.max_persistence.into_pyobject(py).unwrap().into_any().unbind());
        map.insert("mean_persistence".into(), feat.mean_persistence.into_pyobject(py).unwrap().into_any().unbind());
        map.insert("persistence_entropy".into(), feat.persistence_entropy.into_pyobject(py).unwrap().into_any().unbind());
        map.insert("betti_curve".into(), feat.betti_curve.into_pyobject(py).unwrap().into_any().unbind());
        map.insert("radii".into(), feat.radii.into_pyobject(py).unwrap().into_any().unbind());
        map
    })
}

/// Wasserstein (optimal transport) drift between two region distributions.
///
/// Unlike L2, Wasserstein respects the geometry: shifting mass between
/// *neighboring* regions costs less than between *distant* ones.
///
/// Args:
///     dist_a: Region distribution at time T₁ (list of floats summing to ~1.0).
///     dist_b: Region distribution at time T₂.
///     centroids: List of centroid vectors (one per region, from index.regions()).
///     n_projections: Number of random projections for Sliced Wasserstein (default 50).
///
/// Returns:
///     Sliced Wasserstein distance (float). Lower = more similar.
#[pyfunction]
#[pyo3(signature = (dist_a, dist_b, centroids, n_projections=50))]
fn wasserstein_drift(
    dist_a: Vec<f32>,
    dist_b: Vec<f32>,
    centroids: Vec<Vec<f32>>,
    n_projections: usize,
) -> f64 {
    let centroid_refs: Vec<&[f32]> = centroids.iter().map(|c| c.as_slice()).collect();
    cvx_analytics::wasserstein::wasserstein_drift(&dist_a, &dist_b, &centroid_refs, n_projections)
}

/// Project a trajectory into anchor-relative coordinates.
///
/// For a trajectory in ℝᴰ and K anchor vectors, produces a new trajectory in ℝᴷ
/// where dimension k = distance(point, anchor_k).
///
/// The output can be fed into any CVX analytics function (velocity, hurst_exponent,
/// detect_changepoints, path_signature) for anchor-relative analysis.
///
/// Args:
///     trajectory: List of (timestamp, vector) tuples.
///     anchors: List of anchor vectors (same dimensionality as trajectory vectors).
///     metric: Distance metric — "cosine" (default, range [0,1]) or "l2".
///
/// Returns:
///     List of (timestamp, distances) where distances[k] = dist(point, anchor_k).
#[pyfunction]
#[pyo3(signature = (trajectory, anchors, metric="cosine"))]
fn project_to_anchors(
    trajectory: Vec<(i64, Vec<f32>)>,
    anchors: Vec<Vec<f32>>,
    metric: &str,
) -> PyResult<Vec<(i64, Vec<f32>)>> {
    let traj: Vec<(i64, &[f32])> = trajectory.iter().map(|(t, v)| (*t, v.as_slice())).collect();
    let anchor_refs: Vec<&[f32]> = anchors.iter().map(|a| a.as_slice()).collect();
    let m = match metric {
        "cosine" => cvx_analytics::anchor::AnchorMetric::Cosine,
        "l2" => cvx_analytics::anchor::AnchorMetric::L2,
        _ => return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Unknown metric '{metric}'. Use 'cosine' or 'l2'.")
        )),
    };
    Ok(cvx_analytics::anchor::project_to_anchors(&traj, &anchor_refs, m))
}

/// Compute summary statistics of anchor proximity over a projected trajectory.
///
/// Args:
///     projected: Output from project_to_anchors().
///
/// Returns:
///     Dict with keys: mean, min, trend, last (each a list of K floats).
///     trend[k] < 0 means the entity is approaching anchor k over time.
#[pyfunction]
fn anchor_summary(
    projected: Vec<(i64, Vec<f32>)>,
) -> std::collections::HashMap<String, Vec<f32>> {
    let summary = cvx_analytics::anchor::anchor_summary(&projected);
    let mut map = std::collections::HashMap::new();
    map.insert("mean".into(), summary.mean);
    map.insert("min".into(), summary.min);
    map.insert("trend".into(), summary.trend);
    map.insert("last".into(), summary.last);
    map
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
    m.add_function(wrap_pyfunction!(path_signature, m)?)?;
    m.add_function(wrap_pyfunction!(log_signature, m)?)?;
    m.add_function(wrap_pyfunction!(signature_distance, m)?)?;
    m.add_function(wrap_pyfunction!(frechet_distance, m)?)?;
    m.add_function(wrap_pyfunction!(wasserstein_drift, m)?)?;
    m.add_function(wrap_pyfunction!(event_features, m)?)?;
    m.add_function(wrap_pyfunction!(topological_features, m)?)?;
    m.add_function(wrap_pyfunction!(fisher_rao_distance, m)?)?;
    m.add_function(wrap_pyfunction!(hellinger_distance, m)?)?;
    m.add_function(wrap_pyfunction!(project_to_anchors, m)?)?;
    m.add_function(wrap_pyfunction!(anchor_summary, m)?)?;
    Ok(())
}
