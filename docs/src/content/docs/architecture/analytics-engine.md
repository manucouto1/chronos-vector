---
title: Analytics Engine
description: Neural ODE prediction, change point detection (PELT & BOCPD), and vector differential calculus components of ChronosVector's analytics engine.
---

## 10. Analytics Engine

### 10.1 Component Overview

```mermaid
graph TB
    subgraph AnalyticsEngine["Analytics Engine"]
        subgraph NeuralODE["Neural ODE Module"]
            ENCODER["Trajectory Encoder<br/>(ODE-RNN)"]
            LATENT["Latent State z(t)"]
            SOLVER["RK45 Adaptive Solver<br/>with SIMD evaluation"]
            DECODER["State Decoder<br/>z(t) → v(t)"]
            FTHETA["f_θ Network<br/>(MLP via burn)"]
        end

        subgraph CPD["Change Point Detection"]
            PELT_MOD["PELT Module<br/>Offline, exact<br/>O(N) complexity"]
            BOCPD_MOD["BOCPD Module<br/>Online, streaming<br/>O(1) amortized"]
            SEG["Segmentation Result<br/>change_points[], segments[]"]
        end

        subgraph VectorCalc["Vector Differential Calculus"]
            VELOCITY["Velocity ∂v/∂t<br/>First-order finite diff"]
            ACCEL["Acceleration ∂²v/∂t²<br/>Second-order finite diff"]
            CURVATURE["Path Curvature<br/>κ(t) of trajectory"]
            GEODESIC["Geodesic Distance<br/>d(v(t1), v(t2)) along path"]
        end
    end

    ENCODER --> LATENT
    LATENT --> SOLVER
    SOLVER --> FTHETA
    FTHETA --> SOLVER
    SOLVER --> DECODER

    PELT_MOD --> SEG
    BOCPD_MOD --> SEG
```

### 10.2 Neural ODE Prediction Flow

```mermaid
graph LR
    subgraph Input["Input"]
        TRAJ["Historical Trajectory<br/>v(t₁), v(t₂), ..., v(tₙ)"]
    end

    subgraph Encode["Encode"]
        ODE_RNN["ODE-RNN Encoder<br/>(process backwards)"]
        Z0["Initial Latent State<br/>z(tₙ)"]
    end

    subgraph Integrate["Integrate Forward"]
        RK45["Dormand-Prince RK45<br/>dz/dt = f_θ(z, t)"]
        STEP1["z(tₙ + Δt₁)"]
        STEP2["z(tₙ + Δt₂)"]
        STEPN["z(t_future)"]
    end

    subgraph Decode["Decode"]
        DEC["Decoder MLP"]
        PRED["Predicted v(t_future)<br/>+ uncertainty estimate"]
    end

    TRAJ --> ODE_RNN --> Z0 --> RK45
    RK45 --> STEP1 --> STEP2 --> STEPN
    STEPN --> DEC --> PRED
```

### 10.3 BOCPD Online Monitor

```mermaid
stateDiagram-v2
    [*] --> Observing: Entity stream starts

    Observing --> Observing: New vector arrives<br/>Update run-length posterior<br/>P(cp) < threshold

    Observing --> ChangeDetected: P(cp) > threshold

    ChangeDetected --> EmitEvent: Create ChangePoint entity
    EmitEvent --> ResetPosterior: Reset run-length to 0
    ResetPosterior --> Observing: Continue monitoring

    Observing --> Dormant: Entity inactive > dormant_ttl
    Dormant --> Observing: New vector arrives
    Dormant --> [*]: Entity expired
```

Cada entidad monitorizada mantiene su propio estado BOCPD con complejidad O(run_length) por actualización, truncada a un máximo configurable de la ventana de run-length.
