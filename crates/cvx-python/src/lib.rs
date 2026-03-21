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
use cvx_core::{TemporalFilter, TemporalIndexAccess};
use cvx_index::hnsw::{HnswConfig, TemporalGraphIndex, temporal::TemporalHnsw};
use cvx_index::metrics::L2Distance;

/// Python-exposed temporal vector index.
///
/// Optionally loads a TorchScript Neural ODE model for prediction.
/// If no model is provided, prediction uses linear extrapolation.
#[pyclass]
struct TemporalIndex {
    inner: TemporalGraphIndex<L2Distance>,
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
    fn new(
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        model_path: Option<String>,
    ) -> PyResult<Self> {
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
                    return Err(pyo3::exceptions::PyIOError::new_err(format!(
                        "Failed to load model: {e}"
                    )));
                }
            }
        } else {
            None
        };

        #[cfg(not(feature = "torch-backend"))]
        if model_path.is_some() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "torch-backend feature not enabled. Build with: maturin develop --features torch-backend",
            ));
        }

        Ok(Self {
            inner: TemporalGraphIndex::new(config, L2Distance),
            #[cfg(feature = "torch-backend")]
            torch_model,
        })
    }

    /// Insert a temporal point, optionally with a reward.
    ///
    /// Args:
    ///     entity_id: Entity identifier.
    ///     timestamp: Unix timestamp.
    ///     vector: Embedding vector.
    ///     reward: Optional outcome reward (e.g., 0.0-1.0). None = no reward.
    #[pyo3(signature = (entity_id, timestamp, vector, reward=None))]
    fn insert(&mut self, entity_id: u64, timestamp: i64, vector: Vec<f32>, reward: Option<f32>) -> u32 {
        match reward {
            Some(r) => self.inner.insert_with_reward(entity_id, timestamp, &vector, r),
            None => self.inner.insert(entity_id, timestamp, &vector),
        }
    }

    /// Set the reward for a node retroactively.
    ///
    /// Useful for annotating outcomes after an episode completes.
    fn set_reward(&mut self, node_id: u32, reward: f32) {
        self.inner.set_reward(node_id, reward);
    }

    /// Get the reward for a node. Returns None if no reward was assigned.
    fn reward(&self, node_id: u32) -> Option<f32> {
        let r = self.inner.reward(node_id);
        if r.is_nan() { None } else { Some(r) }
    }

    /// Search with reward pre-filtering: only return nodes with reward >= min_reward.
    ///
    /// Args:
    ///     vector: Query embedding.
    ///     k: Number of results.
    ///     min_reward: Minimum reward threshold.
    ///     alpha: Semantic vs temporal weight (default 1.0).
    ///     query_timestamp: Reference timestamp (default 0).
    ///     filter_start: Optional temporal range start.
    ///     filter_end: Optional temporal range end.
    ///
    /// Returns:
    ///     List of (entity_id, timestamp, score) tuples.
    #[pyo3(signature = (vector, k=10, min_reward=0.0, alpha=1.0, query_timestamp=0, filter_start=None, filter_end=None))]
    fn search_with_reward(
        &self,
        vector: Vec<f32>,
        k: usize,
        min_reward: f32,
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
            .search_with_reward(&vector, k, filter, alpha, query_timestamp, min_reward)
            .into_iter()
            .map(|(node_id, score)| {
                let eid = self.inner.entity_id(node_id);
                let ts = self.inner.timestamp(node_id);
                (eid, ts, score)
            })
            .collect()
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
        let ids = entity_ids
            .as_slice()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("entity_ids: {e}")))?;
        let ts = timestamps
            .as_slice()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("timestamps: {e}")))?;
        let vecs = vectors.as_array();
        let n = ids.len();

        if ts.len() != n || vecs.nrows() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Length mismatch: entity_ids={n}, timestamps={}, vectors={}",
                ts.len(),
                vecs.nrows()
            )));
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
                Some(slice) => {
                    self.inner.insert(ids[i], ts[i], slice);
                }
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
        self.inner
            .save(std::path::Path::new(&path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to save index: {e}")))
    }

    /// Load a previously saved index.
    ///
    /// Supports both formats:
    /// - Directory format (new): contains index.bin + temporal_edges.bin
    /// - Single file format (legacy .cvx): auto-migrates to TemporalGraphIndex
    ///
    /// Args:
    ///     path: Path to directory or legacy .cvx file.
    ///
    /// Returns:
    ///     A new TemporalIndex with all data and graph structure restored.
    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        let p = std::path::Path::new(&path);

        let inner = if p.is_dir() {
            // New directory format with temporal edges
            TemporalGraphIndex::load(p, L2Distance)
        } else {
            // Legacy single-file format — migrate to TemporalGraphIndex
            TemporalHnsw::load(p, L2Distance).map(TemporalGraphIndex::from_temporal_hnsw)
        };

        let inner = inner.map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to load index: {e}"))
        })?;

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

    // ─── Centering (RFC-012 Part B) ──────────────────────────────────

    /// Compute the centroid (mean vector) of all indexed vectors.
    ///
    /// Returns the mean vector as a list of floats, or None if the index
    /// is empty. This is an O(N×D) operation.
    ///
    /// Use with `set_centroid()` to enable anisotropy correction.
    /// After centering, all distance-based analytics (`project_to_anchors`,
    /// `drift`, `velocity`, etc.) operate on centered vectors, amplifying
    /// the discriminative signal that embedding models compress.
    fn compute_centroid(&self) -> Option<Vec<f32>> {
        self.inner.compute_centroid()
    }

    /// Set the centroid for anisotropy correction.
    ///
    /// Once set, `centered_vector()` subtracts this from any vector.
    /// The centroid is persisted with `save()`.
    ///
    /// Args:
    ///     centroid: The mean vector to subtract. Can be computed via
    ///         `compute_centroid()` or provided externally (e.g., from
    ///         a larger corpus).
    fn set_centroid(&mut self, centroid: Vec<f32>) {
        self.inner.set_centroid(centroid);
    }

    /// Clear the centroid, reverting to raw (uncentered) distances.
    fn clear_centroid(&mut self) {
        self.inner.clear_centroid();
    }

    /// Get the current centroid, if set.
    fn centroid(&self) -> Option<Vec<f32>> {
        self.inner.centroid().map(|c| c.to_vec())
    }

    /// Return a centered copy of the vector (vec - centroid).
    ///
    /// If no centroid is set, returns the vector unchanged.
    fn centered_vector(&self, vec: Vec<f32>) -> Vec<f32> {
        self.inner.centered_vector(&vec)
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

    /// Causal search: find similar states and return what happened next.
    ///
    /// Phase 1: Standard HNSW search for k nearest neighbors.
    /// Phase 2: Walk temporal edges forward/backward from each match.
    ///
    /// This is the primary retrieval pattern for AI agent memory:
    /// "Find past states similar to mine, and show me the continuation."
    ///
    /// Args:
    ///     vector: Query embedding.
    ///     k: Number of matches.
    ///     temporal_context: Steps to walk forward/backward (default 5).
    ///     alpha: Semantic vs temporal weight (default 1.0).
    ///     query_timestamp: Reference timestamp (default 0).
    ///     filter_start: Optional temporal range start.
    ///     filter_end: Optional temporal range end.
    ///
    /// Returns:
    ///     List of dicts with keys: node_id, score, entity_id,
    ///     successors [(node_id, timestamp, vector)],
    ///     predecessors [(node_id, timestamp, vector)].
    #[pyo3(signature = (vector, k=5, temporal_context=5, alpha=1.0, query_timestamp=0, filter_start=None, filter_end=None))]
    fn causal_search(
        &self,
        vector: Vec<f32>,
        k: usize,
        temporal_context: usize,
        alpha: f32,
        query_timestamp: i64,
        filter_start: Option<i64>,
        filter_end: Option<i64>,
    ) -> Vec<pyo3::Py<pyo3::types::PyDict>> {
        let filter = match (filter_start, filter_end) {
            (Some(start), Some(end)) => TemporalFilter::Range(start, end),
            _ => TemporalFilter::All,
        };

        let results =
            self.inner
                .causal_search(&vector, k, filter, alpha, query_timestamp, temporal_context);

        Python::with_gil(|py| {
            results
                .into_iter()
                .map(|r| {
                    let dict = pyo3::types::PyDict::new(py);
                    dict.set_item("node_id", r.node_id).unwrap();
                    dict.set_item("score", r.score).unwrap();
                    dict.set_item("entity_id", r.entity_id).unwrap();

                    // Successors with vectors
                    let succ: Vec<(u32, i64, Vec<f32>)> = r
                        .successors
                        .iter()
                        .map(|&(nid, ts)| (nid, ts, self.inner.vector(nid).to_vec()))
                        .collect();
                    dict.set_item("successors", succ).unwrap();

                    // Predecessors with vectors
                    let pred: Vec<(u32, i64, Vec<f32>)> = r
                        .predecessors
                        .iter()
                        .map(|&(nid, ts)| (nid, ts, self.inner.vector(nid).to_vec()))
                        .collect();
                    dict.set_item("predecessors", pred).unwrap();

                    dict.into()
                })
                .collect()
        })
    }

    /// Hybrid search: beam search exploring both semantic and temporal neighbors.
    ///
    /// Unlike standard search, this also explores temporal edges (predecessor/
    /// successor) during HNSW traversal. Useful for finding semantically
    /// similar points that are also temporally connected.
    ///
    /// Args:
    ///     vector: Query embedding.
    ///     k: Number of results.
    ///     beta: Temporal edge exploration weight (0.0=pure semantic, 1.0=aggressive temporal).
    ///     alpha: Semantic vs temporal distance weight (default 1.0).
    ///     query_timestamp: Reference timestamp (default 0).
    ///     filter_start: Optional temporal range start.
    ///     filter_end: Optional temporal range end.
    ///
    /// Returns:
    ///     List of (entity_id, timestamp, score) tuples.
    #[pyo3(signature = (vector, k=10, beta=0.3, alpha=1.0, query_timestamp=0, filter_start=None, filter_end=None))]
    fn hybrid_search(
        &self,
        vector: Vec<f32>,
        k: usize,
        beta: f32,
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
            .hybrid_search(&vector, k, filter, alpha, beta, query_timestamp)
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
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "entity {entity_id} has insufficient data ({} points, need 2)",
                traj_data.len()
            )));
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

    /// Assign all nodes to their regions in a single O(N) pass.
    ///
    /// Much faster than calling `region_members()` per region — one scan
    /// instead of K scans (where K = number of regions).
    ///
    /// Args:
    ///     level: HNSW level for region granularity (default 3).
    ///     start: Optional start timestamp for temporal filtering.
    ///     end: Optional end timestamp for temporal filtering.
    ///
    /// Returns:
    ///     Dict mapping hub_id → list of (entity_id, timestamp).
    #[pyo3(signature = (level=3, start=None, end=None))]
    fn region_assignments(
        &self,
        level: usize,
        start: Option<i64>,
        end: Option<i64>,
    ) -> std::collections::HashMap<u32, Vec<(u64, i64)>> {
        let filter = match (start, end) {
            (Some(s), Some(e)) => TemporalFilter::Range(s, e),
            _ => TemporalFilter::All,
        };
        self.inner.region_assignments(level, filter)
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
        self.inner
            .region_trajectory(entity_id, level, window_days, alpha)
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
    (
        report.l2_magnitude,
        report.cosine_drift,
        report.top_dimensions,
    )
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
    let config = cvx_analytics::signatures::SignatureConfig {
        depth,
        time_augmentation,
    };
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
    let config = cvx_analytics::signatures::SignatureConfig {
        depth,
        time_augmentation,
    };
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
fn frechet_distance(traj_a: Vec<(i64, Vec<f32>)>, traj_b: Vec<(i64, Vec<f32>)>) -> f64 {
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
    sig_a
        .iter()
        .zip(sig_b.iter())
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
    let feat =
        cvx_analytics::topology::topological_summary(&point_refs, n_radii, persistence_threshold);

    Python::with_gil(|py| {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "n_components".into(),
            feat.n_components
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        map.insert(
            "total_persistence".into(),
            feat.total_persistence_h0
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        map.insert(
            "max_persistence".into(),
            feat.max_persistence
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        map.insert(
            "mean_persistence".into(),
            feat.mean_persistence
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        map.insert(
            "persistence_entropy".into(),
            feat.persistence_entropy
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        map.insert(
            "betti_curve".into(),
            feat.betti_curve
                .into_pyobject(py)
                .unwrap()
                .into_any()
                .unbind(),
        );
        map.insert(
            "radii".into(),
            feat.radii.into_pyobject(py).unwrap().into_any().unbind(),
        );
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
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown metric '{metric}'. Use 'cosine' or 'l2'."
            )));
        }
    };
    Ok(cvx_analytics::anchor::project_to_anchors(
        &traj,
        &anchor_refs,
        m,
    ))
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
fn anchor_summary(projected: Vec<(i64, Vec<f32>)>) -> std::collections::HashMap<String, Vec<f32>> {
    let summary = cvx_analytics::anchor::anchor_summary(&projected);
    let mut map = std::collections::HashMap::new();
    map.insert("mean".into(), summary.mean);
    map.insert("min".into(), summary.min);
    map.insert("trend".into(), summary.trend);
    map.insert("last".into(), summary.last);
    map
}

// ─── RFC-007: Advanced Temporal Primitives ──────────────────────────

/// Compute cohort-level drift analysis.
///
/// Args:
///     trajectories: List of (entity_id, trajectory) where trajectory = [(timestamp, vector), ...].
///     t1: Start timestamp.
///     t2: End timestamp.
///     top_n: Number of top dimensions to report (default 5).
///
/// Returns:
///     Dict with n_entities, mean_drift_l2, median_drift_l2, std_drift_l2,
///     centroid_l2, centroid_cosine, dispersion_t1, dispersion_t2,
///     dispersion_change, convergence_score, top_dimensions, outliers.
#[pyfunction]
#[pyo3(signature = (trajectories, t1, t2, top_n=5))]
fn cohort_drift(
    trajectories: Vec<(u64, Vec<(i64, Vec<f32>)>)>,
    t1: i64,
    t2: i64,
    top_n: usize,
) -> PyResult<std::collections::HashMap<String, pyo3::PyObject>> {
    use pyo3::IntoPyObject;

    let owned_refs: Vec<Vec<(i64, &[f32])>> = trajectories
        .iter()
        .map(|(_, traj)| traj.iter().map(|(t, v)| (*t, v.as_slice())).collect())
        .collect();

    let input: Vec<(u64, &[(i64, &[f32])])> = trajectories
        .iter()
        .zip(owned_refs.iter())
        .map(|((eid, _), refs)| (*eid, refs.as_slice()))
        .collect();

    let report = cvx_analytics::cohort::cohort_drift(&input, t1, t2, top_n)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Python::with_gil(|py| {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "n_entities".into(),
            report.n_entities.into_pyobject(py)?.into_any().unbind(),
        );
        map.insert(
            "mean_drift_l2".into(),
            report.mean_drift_l2.into_pyobject(py)?.into_any().unbind(),
        );
        map.insert(
            "median_drift_l2".into(),
            report
                .median_drift_l2
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        );
        map.insert(
            "std_drift_l2".into(),
            report.std_drift_l2.into_pyobject(py)?.into_any().unbind(),
        );
        map.insert(
            "centroid_l2".into(),
            report
                .centroid_drift
                .l2_magnitude
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        );
        map.insert(
            "centroid_cosine".into(),
            report
                .centroid_drift
                .cosine_drift
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        );
        map.insert(
            "dispersion_t1".into(),
            report.dispersion_t1.into_pyobject(py)?.into_any().unbind(),
        );
        map.insert(
            "dispersion_t2".into(),
            report.dispersion_t2.into_pyobject(py)?.into_any().unbind(),
        );
        map.insert(
            "dispersion_change".into(),
            report
                .dispersion_change
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        );
        map.insert(
            "convergence_score".into(),
            report
                .convergence_score
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        );
        map.insert(
            "top_dimensions".into(),
            report.top_dimensions.into_pyobject(py)?.into_any().unbind(),
        );
        let outliers: Vec<(u64, f32, f32, f32)> = report
            .outliers
            .into_iter()
            .map(|o| {
                (
                    o.entity_id,
                    o.drift_magnitude,
                    o.z_score,
                    o.drift_direction_alignment,
                )
            })
            .collect();
        map.insert(
            "outliers".into(),
            outliers.into_pyobject(py)?.into_any().unbind(),
        );
        Ok(map)
    })
}

/// Find time windows where two entities are semantically close.
///
/// Args:
///     traj_a: Trajectory of entity A as [(timestamp, vector), ...].
///     traj_b: Trajectory of entity B.
///     epsilon: Distance threshold for convergence.
///     window_us: Sliding window size in microseconds.
///
/// Returns:
///     List of (start, end, mean_distance, min_distance, points_a, points_b).
#[pyfunction]
fn temporal_join(
    traj_a: Vec<(i64, Vec<f32>)>,
    traj_b: Vec<(i64, Vec<f32>)>,
    epsilon: f32,
    window_us: i64,
) -> PyResult<Vec<(i64, i64, f32, f32, usize, usize)>> {
    let a: Vec<(i64, &[f32])> = traj_a.iter().map(|(t, v)| (*t, v.as_slice())).collect();
    let b: Vec<(i64, &[f32])> = traj_b.iter().map(|(t, v)| (*t, v.as_slice())).collect();

    cvx_analytics::temporal_join::temporal_join(&a, &b, epsilon, window_us)
        .map(|results| {
            results
                .into_iter()
                .map(|r| {
                    (
                        r.start,
                        r.end,
                        r.mean_distance,
                        r.min_distance,
                        r.points_a,
                        r.points_b,
                    )
                })
                .collect()
        })
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Discover recurring motifs (patterns) in a trajectory via Matrix Profile.
///
/// Args:
///     trajectory: List of (timestamp, vector) tuples.
///     window: Subsequence window size (number of time steps).
///     max_motifs: Maximum number of motifs to return (default 5).
///     exclusion_zone: Fraction of window for non-trivial matches (default 0.5).
///
/// Returns:
///     List of dicts with 'canonical_index', 'occurrences', 'period', 'mean_match_distance'.
#[pyfunction]
#[pyo3(signature = (trajectory, window, max_motifs=5, exclusion_zone=0.5))]
fn discover_motifs(
    trajectory: Vec<(i64, Vec<f32>)>,
    window: usize,
    max_motifs: usize,
    exclusion_zone: f32,
) -> PyResult<Vec<std::collections::HashMap<String, pyo3::PyObject>>> {
    use pyo3::IntoPyObject;

    let traj: Vec<(i64, &[f32])> = trajectory.iter().map(|(t, v)| (*t, v.as_slice())).collect();

    let motifs = cvx_analytics::motifs::discover_motifs(&traj, window, max_motifs, exclusion_zone)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Python::with_gil(|py| {
        motifs
            .into_iter()
            .map(|m| {
                let mut map = std::collections::HashMap::new();
                map.insert(
                    "canonical_index".into(),
                    m.canonical_index.into_pyobject(py)?.into_any().unbind(),
                );
                let occs: Vec<(usize, i64, f32)> = m
                    .occurrences
                    .into_iter()
                    .map(|o| (o.start_index, o.timestamp, o.distance))
                    .collect();
                map.insert(
                    "occurrences".into(),
                    occs.into_pyobject(py)?.into_any().unbind(),
                );
                map.insert(
                    "period".into(),
                    m.period.into_pyobject(py)?.into_any().unbind(),
                );
                map.insert(
                    "mean_match_distance".into(),
                    m.mean_match_distance.into_pyobject(py)?.into_any().unbind(),
                );
                Ok(map)
            })
            .collect()
    })
}

/// Discover anomalous subsequences (discords) via Matrix Profile.
///
/// Args:
///     trajectory: List of (timestamp, vector) tuples.
///     window: Subsequence window size.
///     max_discords: Maximum number of discords to return (default 5).
///
/// Returns:
///     List of (start_index, timestamp, nn_distance).
#[pyfunction]
#[pyo3(signature = (trajectory, window, max_discords=5))]
fn discover_discords(
    trajectory: Vec<(i64, Vec<f32>)>,
    window: usize,
    max_discords: usize,
) -> PyResult<Vec<(usize, i64, f32)>> {
    let traj: Vec<(i64, &[f32])> = trajectory.iter().map(|(t, v)| (*t, v.as_slice())).collect();

    cvx_analytics::motifs::discover_discords(&traj, window, max_discords)
        .map(|results| {
            results
                .into_iter()
                .map(|d| (d.start_index, d.timestamp, d.nn_distance))
                .collect()
        })
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Test Granger causality between two embedding trajectories.
///
/// Args:
///     traj_a: Trajectory of entity A as [(timestamp, vector), ...].
///     traj_b: Trajectory of entity B.
///     max_lag: Maximum lag to test (default 5).
///     significance: P-value threshold (default 0.05).
///
/// Returns:
///     Dict with 'direction', 'optimal_lag', 'f_statistic', 'p_value', 'effect_size'.
#[pyfunction]
#[pyo3(signature = (traj_a, traj_b, max_lag=5, significance=0.05))]
fn granger_causality(
    traj_a: Vec<(i64, Vec<f32>)>,
    traj_b: Vec<(i64, Vec<f32>)>,
    max_lag: usize,
    significance: f64,
) -> PyResult<std::collections::HashMap<String, pyo3::PyObject>> {
    use pyo3::IntoPyObject;

    let a: Vec<(i64, &[f32])> = traj_a.iter().map(|(t, v)| (*t, v.as_slice())).collect();
    let b: Vec<(i64, &[f32])> = traj_b.iter().map(|(t, v)| (*t, v.as_slice())).collect();

    let result = cvx_analytics::granger::granger_causality(&a, &b, max_lag, significance)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Python::with_gil(|py| {
        let mut map = std::collections::HashMap::new();
        let direction = match result.direction {
            cvx_analytics::granger::GrangerDirection::AToB => "a_to_b",
            cvx_analytics::granger::GrangerDirection::BToA => "b_to_a",
            cvx_analytics::granger::GrangerDirection::Bidirectional => "bidirectional",
            cvx_analytics::granger::GrangerDirection::None => "none",
        };
        map.insert(
            "direction".into(),
            direction.into_pyobject(py)?.into_any().unbind(),
        );
        map.insert(
            "optimal_lag".into(),
            result.optimal_lag.into_pyobject(py)?.into_any().unbind(),
        );
        map.insert(
            "f_statistic".into(),
            result.f_statistic.into_pyobject(py)?.into_any().unbind(),
        );
        map.insert(
            "p_value".into(),
            result.p_value.into_pyobject(py)?.into_any().unbind(),
        );
        map.insert(
            "effect_size".into(),
            result.effect_size.into_pyobject(py)?.into_any().unbind(),
        );
        Ok(map)
    })
}

/// Compute counterfactual trajectory analysis.
///
/// Given pre-change and post-change trajectory segments, extrapolates the
/// pre-change linear trend and measures divergence from actual post-change data.
///
/// Args:
///     pre_change: Trajectory before change point [(timestamp, vector), ...].
///     post_change: Trajectory after change point.
///     change_point: Timestamp of the detected change.
///
/// Returns:
///     Dict with 'total_divergence', 'max_divergence_value', 'max_divergence_time',
///     'divergence_curve' as list of (timestamp, distance).
#[pyfunction]
fn counterfactual_trajectory(
    pre_change: Vec<(i64, Vec<f32>)>,
    post_change: Vec<(i64, Vec<f32>)>,
    change_point: i64,
) -> PyResult<std::collections::HashMap<String, pyo3::PyObject>> {
    use pyo3::IntoPyObject;

    let pre: Vec<(i64, &[f32])> = pre_change.iter().map(|(t, v)| (*t, v.as_slice())).collect();
    let post: Vec<(i64, &[f32])> = post_change
        .iter()
        .map(|(t, v)| (*t, v.as_slice()))
        .collect();

    let result =
        cvx_analytics::counterfactual::counterfactual_trajectory(&pre, &post, change_point)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Python::with_gil(|py| {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "total_divergence".into(),
            result
                .total_divergence
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        );
        map.insert(
            "max_divergence_value".into(),
            result
                .max_divergence_value
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        );
        map.insert(
            "max_divergence_time".into(),
            result
                .max_divergence_time
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        );
        let curve: Vec<(i64, f32)> = result.divergence_curve;
        map.insert(
            "divergence_curve".into(),
            curve.into_pyobject(py)?.into_any().unbind(),
        );
        Ok(map)
    })
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
    // RFC-007: Advanced Temporal Primitives
    m.add_function(wrap_pyfunction!(cohort_drift, m)?)?;
    m.add_function(wrap_pyfunction!(temporal_join, m)?)?;
    m.add_function(wrap_pyfunction!(discover_motifs, m)?)?;
    m.add_function(wrap_pyfunction!(discover_discords, m)?)?;
    m.add_function(wrap_pyfunction!(granger_causality, m)?)?;
    m.add_function(wrap_pyfunction!(counterfactual_trajectory, m)?)?;
    Ok(())
}
