# RFC-003: Neural ODE Prediction via TorchScript (tch-rs)

**Status**: Proposed
**Created**: 2026-03-16
**Authors**: Manuel Couto Pintos
**Related**: RFC-002-10 (ODE Stiffness Detection), Layer 10 (Neural ODE Prediction)

---

## Summary

This RFC proposes adding a PyTorch-based Neural ODE prediction backend to ChronosVector. Models can be trained in either Python (PyTorch) or Rust (`tch-rs`), exported as TorchScript (`.pt`), and used for inference. The implementation is gated behind an optional `torch-backend` feature flag — without libtorch, the system compiles and runs normally using linear extrapolation.

---

## Motivation

ChronosVector currently implements `PredictionMethod::NeuralOde` as an enum variant but always falls back to `linear_extrapolate()`. Linear extrapolation works for short-term smooth trajectories but fails for:

- **Non-linear dynamics**: Social media embeddings that evolve through regime changes
- **Multi-modal trajectories**: Entities whose embedding space behavior has multiple attractors
- **Long-horizon prediction**: Beyond 2-3 timesteps, linear extrapolation diverges

Neural ODEs (Chen et al., NeurIPS 2018) learn continuous-time dynamics $\frac{dy}{dt} = f_\theta(t, y)$ from data, enabling accurate prediction by integrating the learned function to any target time.

### Why tch-rs over burn

| Factor | tch-rs (libtorch) | burn |
|--------|-------------------|------|
| **Ecosystem** | Full PyTorch compatibility | Rust-native, smaller ecosystem |
| **Pre-trained models** | Load any `.pt` from HuggingFace/PyTorch Hub | Requires re-implementation |
| **Training** | Python + Rust | Rust only |
| **Community** | Millions of PyTorch users | Growing but small |
| **User's workflow** | Train in Python (familiar), deploy in Rust | Train + deploy in Rust |

For a temporal vector database targeting the ML community, PyTorch compatibility is essential.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Training (offline)                     │
│                                                          │
│   Python (PyTorch + torchdiffeq)    Rust (tch-rs)       │
│   ─────────────────────────────     ───────────────     │
│   1. Load trajectory data           1. Load data         │
│   2. Train NeuralODEPredictor       2. Train model       │
│   3. torch.jit.trace(model)         3. model.save(.pt)   │
│   4. model.save("model.pt")                              │
└──────────────────────┬──────────────────────────────────┘
                       │ model.pt (TorchScript)
┌──────────────────────▼──────────────────────────────────┐
│                   Inference (runtime)                     │
│                                                          │
│   cvx-analytics/src/torch_ode.rs                        │
│   ┌───────────────────────────────────────────────┐     │
│   │ TorchOdeModel::load("model.pt")               │     │
│   │   → CModule (thread-safe, cached)              │     │
│   │                                                │     │
│   │ TorchOdeModel::predict(trajectory, target_t)   │     │
│   │   1. Normalize timestamps to [0, 1]            │     │
│   │   2. Build input tensor [1, T, D+1]            │     │
│   │   3. model.forward([input, target_t])          │     │
│   │   4. Output tensor [1, D] → Vec<f32>           │     │
│   └───────────────────────────────────────────────┘     │
│                         │                                │
│   Fallback: if model unavailable → linear_extrapolate()  │
└─────────────────────────────────────────────────────────┘
```

---

## Model Specification

### Input Format

The TorchScript model receives two tensors:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `trajectory` | `[1, T, D+1]` | Batch=1, T timesteps, each row = `[normalized_time, v_1, v_2, ..., v_D]` |
| `target_t` | `[1, 1]` | Normalized target timestamp |

**Time normalization**: Timestamps are mapped to `[0, 1]` where `t=0` is the first observation and `t=1` is the target timestamp. This makes the model invariant to absolute time scales.

### Output Format

| Tensor | Shape | Description |
|--------|-------|-------------|
| `predicted` | `[1, D]` | Predicted D-dimensional vector at `target_t` |

### Model Architecture (Reference)

```
NeuralODEPredictor
├── Encoder: GRU(input_dim=D+1, hidden_dim=latent_dim)
│   → Encodes variable-length trajectory into fixed latent state h₀
├── ODE Dynamics: MLP(latent_dim → hidden → latent_dim) with tanh
│   → Defines dh/dt = f_θ(t, h), integrated via odeint
├── Integration: RK4/Dormand-Prince from t_last to t_target
│   → Produces h(t_target)
└── Decoder: Linear(latent_dim → D)
    → Maps latent state back to vector space
```

This is a **reference architecture** — users can train any model that respects the input/output tensor contract.

---

## Pre-trained Models

There is **no standard pre-trained model** for high-dimensional temporal vector trajectory prediction on HuggingFace. Existing time series foundation models (Chronos, TimesFM, Lag-Llama) operate on scalar series, not $\mathbb{R}^D$ embeddings.

### Strategy

1. **Phase 1**: Ship without a default model. Users train on their own data.
2. **Phase 2**: Train and publish a ChronosVector reference model on HuggingFace:
   - Trained on synthetic trajectories (random walk, OU process, regime-switching)
   - Dimensions: D=128, D=384, D=768 variants
   - Published as `manucouto1/cvx-neural-ode-{dim}`
3. **Phase 3**: Auto-download from HuggingFace if no local model and `neural_ode = true`:
   ```rust
   // Future: auto-fetch default model
   let model = TorchOdeModel::from_huggingface("manucouto1/cvx-neural-ode-128")?;
   ```

---

## Implementation Plan

### Phase 1: Feature Flags & Dependencies

**Files to modify:**

| File | Change |
|------|--------|
| `Cargo.toml` (workspace) | Add `tch = { version = "0.17", optional = true }` |
| `cvx-analytics/Cargo.toml` | Add feature `torch-backend = ["dep:tch"]`, optional tch dep |
| `cvx-query/Cargo.toml` | Add feature `torch-backend = ["cvx-analytics/torch-backend"]` |
| `cvx-api/Cargo.toml` | Add feature `torch-backend = ["cvx-query/torch-backend"]` |
| `cvx-server/Cargo.toml` | Add feature `torch-backend = ["cvx-api/torch-backend"]` |

Feature propagation chain: `cvx-server → cvx-api → cvx-query → cvx-analytics → tch`

### Phase 2: Configuration

**File**: `cvx-core/src/config/mod.rs`

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AnalyticsConfig {
    pub neural_ode: bool,
    pub change_detection: String,
    /// Path to TorchScript model file (.pt).
    /// Required when neural_ode = true and torch-backend is enabled.
    pub model_path: Option<PathBuf>,
}
```

**TOML example:**

```toml
[analytics]
neural_ode = true
model_path = "./models/neural_ode_d128.pt"
change_detection = "pelt"
```

### Phase 3: TorchScript Model Loader

**New file**: `cvx-analytics/src/torch_ode.rs`

```rust
#[cfg(feature = "torch-backend")]
pub struct TorchOdeModel {
    model: tch::CModule,
    device: tch::Device,
}

impl TorchOdeModel {
    /// Load a TorchScript model from disk.
    pub fn load(path: &Path) -> Result<Self, AnalyticsError>;

    /// Predict vector at target_timestamp from a trajectory.
    /// Falls back with Err if model fails, caller handles fallback.
    pub fn predict(
        &self,
        trajectory: &[(i64, &[f32])],
        target_timestamp: i64,
    ) -> Result<Vec<f32>, AnalyticsError>;
}
```

Key implementation details:

1. **Thread safety**: `tch::CModule` is `Send + Sync` — safe in `Arc<AppState>`
2. **Device selection**: Use CUDA if available, fallback to CPU
3. **Time normalization**: `t_norm = (t - t_first) / (t_target - t_first)`
4. **Validation**: Check output tensor shape, reject NaN/Inf
5. **Error handling**: Return `AnalyticsError::SolverDiverged` on failure

### Phase 4: Integration with Prediction Path

**Modify**: `cvx-query/src/engine.rs`

Add optional `AnalyticsBackend` parameter to `execute_query`:

```rust
pub fn execute_query(
    index: &dyn TemporalIndexAccess,
    query: TemporalQuery,
    analytics: Option<&dyn AnalyticsBackend>,  // NEW
) -> Result<QueryResult, QueryError> { ... }
```

In `do_prediction()`:
1. If `analytics` is `Some`, call `analytics.predict()`
2. If model succeeds, return `PredictionMethod::NeuralOde`
3. If model fails or `analytics` is `None`, fallback to `linear_extrapolate()`

**Modify**: `cvx-analytics/src/backend.rs`

```rust
pub struct DefaultAnalytics {
    pelt_config: PeltConfig,
    #[cfg(feature = "torch-backend")]
    torch_model: Option<Arc<TorchOdeModel>>,
}
```

**Modify**: `cvx-api/src/state.rs` — load model at startup from config

### Phase 5: Python Training Script

**New file**: `scripts/train_neural_ode.py`

Reference training script using PyTorch + `torchdiffeq`:

```python
class ODEFunc(nn.Module):
    """Learned dynamics f(t, y) for dy/dt = f(t, y)."""
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim),
        )
    def forward(self, t, y):
        return self.net(y)

class NeuralODEPredictor(nn.Module):
    """Encodes trajectory via GRU, integrates ODE, decodes to vector."""
    def __init__(self, input_dim, latent_dim=64, hidden=128):
        super().__init__()
        self.encoder = nn.GRU(input_dim + 1, latent_dim, batch_first=True)
        self.ode_func = ODEFunc(latent_dim, hidden)
        self.decoder = nn.Linear(latent_dim, input_dim)
```

Training produces `model.pt` via `torch.jit.trace()` or `torch.jit.script()`.

### Phase 6: Rust Training API

**New file**: `cvx-analytics/src/torch_train.rs` (optional, feature-gated)

Enables training directly in Rust without Python:

```rust
#[cfg(feature = "torch-backend")]
pub fn train_neural_ode(
    trajectories: &[Vec<(i64, Vec<f32>)>],
    config: TrainConfig,
) -> Result<TorchOdeModel, AnalyticsError>;
```

This uses `tch-rs` to define the same model architecture, run gradient descent, and save as TorchScript. Same model, two training languages.

### Phase 7: Python Bindings

**Modify**: `cvx-python/src/lib.rs`

```python
# Python API
import chronos_vector as cvx

# Load a pre-trained model
index = cvx.TemporalIndex(model_path="model.pt")

# Prediction uses Neural ODE if model loaded
prediction = index.predict(entity_id=1, target_timestamp=5_000_000)
```

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Feature `torch-backend` not compiled | All code elided at compile time, linear fallback always |
| `neural_ode = false` in config | Model not loaded, linear fallback |
| `model_path` not set or file missing | Warning at startup, linear fallback at runtime |
| libtorch not installed | Load error at startup, linear fallback |
| Model inference fails (bad dims, NaN) | Warning logged, linear fallback per-request |
| CUDA unavailable | Automatic CPU fallback (tch-rs handles this) |

Every failure mode falls back to `linear_extrapolate()` — the system never crashes.

---

## Performance Considerations

| Operation | Expected Latency |
|-----------|-----------------|
| Model load (one-time) | 100-500ms |
| CPU inference (D=128, T=50) | 1-5ms |
| GPU inference (D=768, T=100) | <1ms |
| Linear extrapolation fallback | <0.1ms |

The model is loaded once at startup and cached in `Arc`. Inference is thread-safe.

---

## Testing Strategy

1. **Unit tests** (`torch_ode.rs`): Load a small test model, verify predict() returns correct shape
2. **Integration test**: Synthetic trajectory → train tiny model → export → load in Rust → verify prediction ≠ linear
3. **Feature-gated tests**: `#[cfg(test)] #[cfg(feature = "torch-backend")]` — CI runs with and without
4. **Fallback test**: Verify that without torch-backend, prediction returns `PredictionMethod::Linear`
5. **Python roundtrip**: Train in Python, predict in Rust, compare outputs

---

## Dependencies

| Crate | Version | Purpose | Condition |
|-------|---------|---------|-----------|
| `tch` | 0.17 | libtorch Rust bindings | `torch-backend` feature |
| `torchdiffeq` | (Python) | ODE integration for training | Python only |
| `torch` | ≥2.0 | Model training | Python only |

---

## Implementation Phases & Effort

| Phase | Scope | Effort | Status |
|-------|-------|--------|--------|
| 1 | Feature flags & dependencies | Low | ✅ Done |
| 2 | Configuration (model_path) | Low | ✅ Done |
| 3 | TorchScript loader + inference | Medium | ✅ Done |
| 4 | Query engine integration | Medium | ✅ Done |
| 5 | Python training script | Low | ✅ Done |
| 6 | Rust training API | Medium | Proposed |
| 7 | Python bindings update | Low | Proposed |

Total estimated effort: ~2-3 weeks

---

## Future Extensions

- **Auto-download from HuggingFace**: Default model for common dimensions
- **Per-entity models**: Different dynamics for different entity types
- **Online fine-tuning**: Adapt model as new data arrives
- **Stiffness detection (RFC-002-10)**: Switch solver when dynamics become stiff
- **Ensemble prediction**: Average Neural ODE + linear for robustness

---

## References

1. Chen, R.T.Q. et al. "Neural Ordinary Differential Equations." NeurIPS 2018.
2. Kidger, P. "On Neural Differential Equations." PhD Thesis, Oxford, 2022.
3. Rubanova, Y. et al. "Latent ODEs for Irregularly-Sampled Time Series." NeurIPS 2019.
4. De Brouwer, E. et al. "GRU-ODE-Bayes: Continuous Modeling of Sporadically-Observed Time Series." NeurIPS 2019.
5. PyTorch TorchScript documentation: https://pytorch.org/docs/stable/jit.html
6. tch-rs: Rust bindings for the C++ API of PyTorch: https://github.com/LaurentMazare/tch-rs
