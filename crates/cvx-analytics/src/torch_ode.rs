//! TorchScript Neural ODE model loader and inference (RFC-003).
//!
//! Loads a pre-trained TorchScript model (`.pt`) exported from Python or Rust,
//! and runs inference to predict future embedding vectors from trajectories.
//!
//! # Model Contract
//!
//! The model must accept:
//! - `trajectory`: `Tensor[1, T, D+1]` — normalized time + vector per step
//! - `target_t`: `Tensor[1, 1]` — normalized target timestamp
//!
//! And return:
//! - `predicted`: `Tensor[1, D]` — predicted vector at target time
//!
//! # Example
//!
//! ```ignore
//! use cvx_analytics::torch_ode::TorchOdeModel;
//!
//! let model = TorchOdeModel::load("models/neural_ode.pt")?;
//! let trajectory = vec![(1000, vec![0.1, 0.2, 0.3].as_slice()), (2000, vec![0.4, 0.5, 0.6].as_slice())];
//! let predicted = model.predict(&trajectory, 3000)?;
//! ```

use std::path::Path;
use std::sync::Arc;

use tch::{CModule, Device, Kind, Tensor};

use cvx_core::error::AnalyticsError;

/// A loaded TorchScript Neural ODE model for trajectory prediction.
///
/// Thread-safe: `CModule` is `Send + Sync`, so this can live in `Arc<AppState>`.
pub struct TorchOdeModel {
    model: CModule,
    device: Device,
}

// Safety: tch::CModule is Send + Sync
unsafe impl Send for TorchOdeModel {}
unsafe impl Sync for TorchOdeModel {}

impl TorchOdeModel {
    /// Load a TorchScript model from a `.pt` file.
    pub fn load(path: &Path) -> Result<Self, AnalyticsError> {
        let device = if tch::Cuda::is_available() {
            tracing::info!("CUDA available, using GPU for Neural ODE inference");
            Device::Cuda(0)
        } else {
            tracing::info!("Using CPU for Neural ODE inference");
            Device::Cpu
        };

        let model = CModule::load_on_device(path, device).map_err(|e| {
            AnalyticsError::ModelNotLoaded {
                name: path.display().to_string(),
            }
        })?;

        tracing::info!("Loaded Neural ODE model from {}", path.display());
        Ok(Self { model, device })
    }

    /// Predict the vector at `target_timestamp` given a trajectory.
    ///
    /// Trajectory: slice of `(timestamp_micros, vector)` pairs, sorted by time.
    /// Returns the predicted D-dimensional vector.
    pub fn predict(
        &self,
        trajectory: &[(i64, &[f32])],
        target_timestamp: i64,
    ) -> Result<Vec<f32>, AnalyticsError> {
        if trajectory.len() < 2 {
            return Err(AnalyticsError::InsufficientData {
                needed: 2,
                have: trajectory.len(),
            });
        }

        let dim = trajectory[0].1.len();
        let t_len = trajectory.len();

        // Normalize timestamps to [0, 1] range
        let t_first = trajectory[0].0 as f64;
        let t_target = target_timestamp as f64;
        let t_range = (t_target - t_first).max(1.0);

        // Build input tensor [1, T, D+1]: normalized_time + vector
        let mut input_data = Vec::with_capacity(t_len * (dim + 1));
        for &(ts, vec) in trajectory {
            let t_norm = ((ts as f64 - t_first) / t_range) as f32;
            input_data.push(t_norm);
            input_data.extend_from_slice(vec);
        }

        let input_tensor = Tensor::from_slice(&input_data)
            .reshape([1, t_len as i64, (dim + 1) as i64])
            .to_device(self.device)
            .to_kind(Kind::Float);

        // Target time tensor [1, 1]
        let target_t_tensor = Tensor::from_slice(&[1.0f32]) // normalized to 1.0
            .reshape([1, 1])
            .to_device(self.device)
            .to_kind(Kind::Float);

        // Run inference
        let output = self
            .model
            .forward_ts(&[input_tensor, target_t_tensor])
            .map_err(|e| AnalyticsError::SolverDiverged {
                step: 0,
                error: format!("TorchScript inference failed: {e}"),
            })?;

        // Convert output tensor [1, D] to Vec<f32>
        let output = output.to_device(Device::Cpu).to_kind(Kind::Float);
        let output_size = output.size();

        if output_size.len() != 2 || output_size[1] != dim as i64 {
            return Err(AnalyticsError::SolverDiverged {
                step: 0,
                error: format!(
                    "model output shape {:?}, expected [1, {dim}]",
                    output_size
                ),
            });
        }

        let result: Vec<f32> = Vec::try_from(output.reshape([dim as i64])).map_err(|e| {
            AnalyticsError::SolverDiverged {
                step: 0,
                error: format!("failed to convert output tensor: {e}"),
            }
        })?;

        // Validate: reject NaN/Inf
        if result.iter().any(|v| !v.is_finite()) {
            return Err(AnalyticsError::SolverDiverged {
                step: 0,
                error: "model produced NaN/Inf values".into(),
            });
        }

        Ok(result)
    }
}

/// Wrap a `TorchOdeModel` in an `Arc` for shared ownership.
pub type SharedTorchModel = Arc<TorchOdeModel>;
