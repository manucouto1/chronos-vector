//! Rust-native Neural ODE training via tch-rs (RFC-003, Phase 6).
//!
//! Defines and trains the same GRU-ODE-Decoder architecture as the Python
//! training script, producing a TorchScript model loadable by `torch_ode.rs`.
//!
//! # Example
//!
//! ```ignore
//! use cvx_analytics::torch_train::{TrainConfig, train_neural_ode};
//!
//! let trajectories: Vec<Vec<(i64, Vec<f32>)>> = load_data();
//! let config = TrainConfig { dim: 128, epochs: 100, ..Default::default() };
//! let model = train_neural_ode(&trajectories, &config)?;
//! model.save("models/neural_ode.pt")?;
//! ```

use std::path::Path;

use tch::nn::{self, Module, OptimizerConfig};
use tch::{Device, Kind, Tensor};

use cvx_core::error::AnalyticsError;

/// Training configuration for the Neural ODE model.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Input vector dimensionality.
    pub dim: usize,
    /// Latent ODE state dimensionality.
    pub latent_dim: i64,
    /// Hidden layer size.
    pub hidden_dim: i64,
    /// Number of training epochs.
    pub epochs: usize,
    /// Learning rate.
    pub lr: f64,
    /// Number of Euler integration steps.
    pub ode_steps: i64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            latent_dim: 64,
            hidden_dim: 128,
            epochs: 100,
            lr: 1e-3,
            ode_steps: 20,
        }
    }
}

/// A trained Neural ODE model that can be saved as TorchScript.
pub struct TrainedModel {
    vs: nn::VarStore,
    encoder: nn::GRU,
    ode_net: nn::Sequential,
    decoder: nn::Linear,
    config: TrainConfig,
}

impl TrainedModel {
    /// Save the model as TorchScript via tracing.
    pub fn save(&self, path: &Path) -> Result<(), AnalyticsError> {
        // Create example inputs for tracing
        let device = self.vs.device();
        let dim = self.config.dim as i64;
        let example_traj = Tensor::randn([1, 10, dim + 1], (Kind::Float, device));
        let example_t = Tensor::ones([1, 1], (Kind::Float, device));

        // Build a traceable forward function using closure
        let traced = tch::CModule::create_by_tracing(
            "NeuralODEPredictor",
            "forward",
            &[example_traj, example_t],
            |inputs| {
                let traj = &inputs[0];
                let _target_t = &inputs[1];
                self.forward_impl(traj)
            },
        )
        .map_err(|e| AnalyticsError::ModelNotLoaded {
            name: format!("tracing failed: {e}"),
        })?;

        traced
            .save(path)
            .map_err(|e| AnalyticsError::ModelNotLoaded {
                name: format!("save failed: {e}"),
            })?;

        tracing::info!("Saved Neural ODE model to {}", path.display());
        Ok(())
    }

    fn forward_impl(&self, trajectory: &Tensor) -> Tensor {
        // Encode: GRU over trajectory → last hidden state
        let (_, h) = self.encoder.seq(trajectory);
        let h = h.squeeze_dim(0); // [B, latent]

        // Euler ODE integration (ode_steps steps in latent space)
        let steps = self.config.ode_steps;
        let dt = 1.0 / steps as f64;
        let mut state = h;
        for _ in 0..steps {
            let deriv = self.ode_net.forward(&state);
            state = &state + dt * deriv;
        }

        // Decode: latent → vector space
        self.decoder.forward(&state) // [B, D]
    }
}

/// Train a Neural ODE model from trajectory data.
///
/// Each trajectory is a `Vec<(timestamp, vector)>` for a single entity.
/// Returns a trained model that can be saved as TorchScript.
pub fn train_neural_ode(
    trajectories: &[Vec<(i64, Vec<f32>)>],
    config: &TrainConfig,
) -> Result<TrainedModel, AnalyticsError> {
    if trajectories.is_empty() {
        return Err(AnalyticsError::InsufficientData {
            needed: 1,
            have: 0,
        });
    }

    let device = if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    let dim = config.dim as i64;
    let vs = nn::VarStore::new(device);
    let root = vs.root();

    // Build model architecture (matches Python script)
    let encoder = nn::gru(
        &root / "encoder",
        dim + 1, // input: time + vector
        config.latent_dim,
        nn::GRUConfig {
            batch_first: true,
            ..Default::default()
        },
    );

    let ode_net = nn::seq()
        .add(nn::linear(
            &root / "ode_fc1",
            config.latent_dim,
            config.hidden_dim,
            Default::default(),
        ))
        .add_fn(|x| x.tanh())
        .add(nn::linear(
            &root / "ode_fc2",
            config.hidden_dim,
            config.hidden_dim,
            Default::default(),
        ))
        .add_fn(|x| x.tanh())
        .add(nn::linear(
            &root / "ode_fc3",
            config.hidden_dim,
            config.latent_dim,
            Default::default(),
        ));

    let decoder = nn::linear(
        &root / "decoder",
        config.latent_dim,
        dim,
        Default::default(),
    );

    let mut opt = nn::Adam::default().build(&vs, config.lr).map_err(|e| {
        AnalyticsError::ModelNotLoaded {
            name: format!("optimizer init failed: {e}"),
        }
    })?;

    // Prepare training data: split each trajectory into (input, target)
    let (inputs, targets) = prepare_training_data(trajectories, config.dim, device)?;

    let n_samples = inputs.size()[0];
    tracing::info!(
        "Training Neural ODE: dim={}, samples={}, epochs={}",
        config.dim,
        n_samples,
        config.epochs
    );

    let model = TrainedModel {
        vs,
        encoder,
        ode_net,
        decoder,
        config: config.clone(),
    };

    // Training loop
    for epoch in 0..config.epochs {
        let predicted = model.forward_impl(&inputs);
        let loss = predicted.mse_loss(&targets, tch::Reduction::Mean);

        opt.backward_step(&loss);

        if (epoch + 1) % 10 == 0 || epoch == 0 {
            let loss_val: f64 = loss.double_value(&[]);
            tracing::info!("  Epoch {}/{}, loss={:.6}", epoch + 1, config.epochs, loss_val);
        }
    }

    Ok(model)
}

/// Prepare training data from trajectories.
///
/// For each trajectory of length T, use first T-1 points as input
/// and the last point as target.
fn prepare_training_data(
    trajectories: &[Vec<(i64, Vec<f32>)>],
    dim: usize,
    device: Device,
) -> Result<(Tensor, Tensor), AnalyticsError> {
    let mut all_inputs = Vec::new();
    let mut all_targets = Vec::new();
    let mut max_len = 0usize;

    // Filter to trajectories with >= 3 points
    let valid: Vec<&Vec<(i64, Vec<f32>)>> = trajectories
        .iter()
        .filter(|t| t.len() >= 3 && t[0].1.len() == dim)
        .collect();

    if valid.is_empty() {
        return Err(AnalyticsError::InsufficientData {
            needed: 3,
            have: 0,
        });
    }

    // Find max sequence length (for padding)
    for traj in &valid {
        max_len = max_len.max(traj.len() - 1);
    }

    for traj in &valid {
        let t_first = traj[0].0 as f64;
        let t_last = traj.last().unwrap().0 as f64;
        let t_range = (t_last - t_first).max(1.0);

        // Input: first T-1 points with normalized time
        let input_len = traj.len() - 1;
        let mut input_data = vec![0.0f32; max_len * (dim + 1)];
        for (i, (ts, vec)) in traj[..input_len].iter().enumerate() {
            let t_norm = ((*ts as f64 - t_first) / t_range) as f32;
            input_data[i * (dim + 1)] = t_norm;
            input_data[i * (dim + 1) + 1..i * (dim + 1) + 1 + dim]
                .copy_from_slice(&vec[..dim]);
        }
        all_inputs.extend_from_slice(&input_data);

        // Target: last point's vector
        let target_vec = &traj.last().unwrap().1;
        all_targets.extend_from_slice(&target_vec[..dim]);
    }

    let n = valid.len() as i64;
    let inputs = Tensor::from_slice(&all_inputs)
        .reshape([n, max_len as i64, (dim + 1) as i64])
        .to_device(device)
        .to_kind(Kind::Float);

    let targets = Tensor::from_slice(&all_targets)
        .reshape([n, dim as i64])
        .to_device(device)
        .to_kind(Kind::Float);

    Ok((inputs, targets))
}
