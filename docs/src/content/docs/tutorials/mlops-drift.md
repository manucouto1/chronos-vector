---
title: "MLOps Drift Detection"
description: "CVX temporal analytics"
---

Production ML models degrade silently. The embeddings they produce shift over time --- new topics emerge,
user behavior changes, upstream data pipelines break --- and by the time accuracy metrics drop,
the damage is already done.

**ChronosVector (CVX)** treats embedding distributions as *trajectories on a statistical manifold*.
Instead of comparing snapshots, it tracks how the distribution *transforms* over time, using the
HNSW graph hierarchy as a natural set of distribution monitors.

### Three drift scenarios

| Phase | Days | Description |
|-------|------|-------------|
| **Stable** | 0--29 | Reference distribution (3-cluster Gaussian mixture) |
| **Gradual drift** | 30--59 | One cluster's mean shifts linearly over 30 days |
| **Sudden shift** | 60--89 | A cluster disappears and a new one appears far away |

### Five independent drift signals from CVX

| Signal | CVX Function | What It Captures |
|--------|-------------|------------------|
| Velocity | `velocity()` | Rate of change in embedding space |
| Distributional distances | `fisher_rao_distance()`, `hellinger_distance()`, `wasserstein_drift()` | How distributions diverge |
| Change points | `detect_changepoints()` | When regime transitions occur |
| Hurst exponent | `hurst_exponent()` | Whether drift is persistent or transient |
| Topology | `topological_features()` | Whether embedding space is fragmenting |

---
## 1. Setup + Simulate Production Embeddings

We simulate 90 days of production embeddings (D=64, 200 per day) from a model
that transitions through three phases: stable, gradual drift, and sudden shift.

```python
import chronos_vector as cvx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

np.random.seed(42)

D = 64            # embedding dimensionality
N_DAYS = 90
BATCH_SIZE = 200  # embeddings per day

# Color palette
C_STABLE  = '#2ecc71'  # green
C_GRADUAL = '#f39c12'  # orange
C_SUDDEN  = '#e74c3c'  # red
C_ACCENT  = '#3498db'  # blue

# Reference distribution: mixture of 3 Gaussians
ref_means = [np.random.randn(D) * 0.5 for _ in range(3)]
ref_covs  = [np.eye(D) * 0.1 for _ in range(3)]


def sample_reference(n):
    """Sample from reference distribution (stable model)."""
    samples = []
    for _ in range(n):
        cluster = np.random.randint(0, 3)
        samples.append(np.random.multivariate_normal(ref_means[cluster], ref_covs[cluster]))
    return np.array(samples, dtype=np.float32)


# --- Build the temporal index ---
index = cvx.TemporalIndex(m=16, ef_construction=100, ef_search=50)

# Fixed "new cluster" mean for the sudden-shift phase (outside the loop for consistency)
new_mean = np.random.randn(D) * 3.0

drift_schedule = {}
for day in range(N_DAYS):
    if day < 30:
        phase = 'stable'
        embeddings = sample_reference(BATCH_SIZE)
    elif day < 60:
        phase = 'gradual'
        shift_amount = (day - 30) / 30.0 * 2.0  # 0 -> 2.0
        shifted_mean = ref_means[0] + shift_amount * np.ones(D)
        embeddings = []
        for _ in range(BATCH_SIZE):
            cluster = np.random.randint(0, 3)
            if cluster == 0:
                embeddings.append(np.random.multivariate_normal(shifted_mean, ref_covs[0]))
            else:
                embeddings.append(np.random.multivariate_normal(ref_means[cluster], ref_covs[cluster]))
        embeddings = np.array(embeddings, dtype=np.float32)
    else:
        phase = 'sudden'
        embeddings = []
        for _ in range(BATCH_SIZE):
            cluster = np.random.choice([0, 1, 3])  # no cluster 2
            if cluster == 3:
                embeddings.append(np.random.multivariate_normal(new_mean, np.eye(D) * 0.1))
            else:
                embeddings.append(np.random.multivariate_normal(ref_means[cluster], ref_covs[cluster]))
        embeddings = np.array(embeddings, dtype=np.float32)

    drift_schedule[day] = phase

    # Insert into CVX: entity_id=0 (single model), timestamp=day
    entity_ids = np.zeros(BATCH_SIZE, dtype=np.uint64)
    timestamps = np.full(BATCH_SIZE, day, dtype=np.int64)
    index.bulk_insert(entity_ids, timestamps, embeddings)

print(f'Ingested {len(index):,} embeddings over {N_DAYS} days')
print(f'Phases: stable (0-29), gradual drift (30-59), sudden shift (60-89)')
```


```text
Ingested 18,000 embeddings over 90 days
Phases: stable (0-29), gradual drift (30-59), sudden shift (60-89)
```

---
## 2. HNSW Regions as Distribution Monitors

CVX's HNSW graph naturally creates a hierarchy of semantic regions.
Higher levels yield fewer, coarser regions --- each one acts as a bin
for tracking how the embedding distribution shifts over time.

```python
from sklearn.decomposition import PCA

regions = index.regions(level=2)
print(f'{len(regions)} regions at Level 2')
for rid, centroid, n_members in regions[:10]:
    print(f'  Region {rid}: {n_members:,} members, centroid norm={np.linalg.norm(centroid):.2f}')

# Visualize region centroids in PCA space
centroids = [c for _, c, _ in regions]
sizes = [n for _, _, n in regions]

if len(centroids) >= 3:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(np.array(centroids))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coords[:, 0], y=coords[:, 1],
        mode='markers',
        marker=dict(
            size=np.sqrt(np.array(sizes)) * 0.5,
            color=np.array(sizes),
            colorscale='Viridis',
            colorbar=dict(title='Members'),
            line=dict(width=1, color='white'),
        ),
        text=[f'Region {rid}: {n:,} members' for rid, _, n in regions],
        hovertemplate='%{text}<extra></extra>',
    ))
    fig.update_layout(
        title=f'HNSW Level 2 Regions ({len(regions)} clusters) in PCA Space',
        xaxis_title='PCA 1', yaxis_title='PCA 2',
        width=800, height=500, template='plotly_dark',
    )
    fig.show()
else:
    print('Too few regions for PCA visualization')
```


```text
69 regions at Level 2
  Region 319: 85 members, centroid norm=4.44
  Region 348: 138 members, centroid norm=4.65
  Region 465: 105 members, centroid norm=5.04
  Region 999: 86 members, centroid norm=3.89
  Region 1153: 41 members, centroid norm=4.12
  Region 1213: 168 members, centroid norm=4.70
  Region 1243: 28 members, centroid norm=4.70
  Region 1310: 283 members, centroid norm=4.62
  Region 1396: 256 members, centroid norm=4.74
  Region 1460: 1,214 members, centroid norm=4.02
```

<iframe src="/chronos-vector/plots/mlops-drift_fig_0.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

---
## 3. Region Trajectory: Watching the Distribution Transform

The region trajectory tracks how the model's embedding distribution spreads
across HNSW regions over time. Each row in the heatmap is a time step;
each column is a region. The pattern should be clear: stable, gradual change,
then a sudden reorganization.

```python
reg_traj = index.region_trajectory(entity_id=0, level=2, window_days=5, alpha=0.3)
print(f'Region trajectory: {len(reg_traj)} time steps, {len(reg_traj[0][1])} regions per step')

# Build heatmap
times = [t for t, _ in reg_traj]
dists = np.array([d for _, d in reg_traj])

# Show only active regions (max weight > 1%)
active_mask = dists.max(axis=0) > 0.01
dists_active = dists[:, active_mask]
active_ids = np.where(active_mask)[0]

fig = go.Figure(data=go.Heatmap(
    z=dists_active.T,
    x=times,  # integer days — avoids Plotly string-axis bugs
    y=[f'R{i}' for i in active_ids],
    colorscale='Hot',
    hovertemplate='Day: %{x}<br>Region: %{y}<br>Weight: %{z:.3f}<extra></extra>',
))

# Phase annotations (use integer x values)
fig.add_vline(x=30, line_dash='dash', line_color=C_GRADUAL, annotation_text='Gradual drift starts')
fig.add_vline(x=60, line_dash='dash', line_color=C_SUDDEN, annotation_text='Sudden shift')

fig.update_layout(
    title='Region Distribution Over Time (Level 2)',
    xaxis_title='Day', yaxis_title='Region',
    width=1000, height=500, template='plotly_dark',
)
fig.show()
```


```text
Region trajectory: 18 time steps, 69 regions per step
```

<iframe src="/chronos-vector/plots/mlops-drift_fig_1.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

---
## 4. Distributional Drift Signals

CVX provides three complementary distributional distances between consecutive
time steps:

- **Fisher-Rao**: The unique Riemannian metric on the probability simplex. Range [0, pi].
- **Hellinger**: Bounded [0, 1], related to Fisher-Rao but easier to threshold.
- **Wasserstein (Sliced)**: Geometry-aware optimal transport --- moving mass between
  *nearby* regions costs less than between distant ones.

```python
centroids = [c for _, c, _ in regions]

drift_signals = []
for i in range(1, len(reg_traj)):
    t_prev, d_prev = reg_traj[i - 1]
    t_curr, d_curr = reg_traj[i]

    p = [float(x) for x in d_prev]
    q = [float(x) for x in d_curr]

    fr = cvx.fisher_rao_distance(p, q)
    hd = cvx.hellinger_distance(p, q)
    wd = cvx.wasserstein_drift(list(d_prev), list(d_curr), centroids, 50)

    drift_signals.append({
        'day': t_curr,
        'fisher_rao': fr,
        'hellinger': hd,
        'wasserstein': wd,
    })

print(f'Computed {len(drift_signals)} consecutive-day drift measurements')

days = [s['day'] for s in drift_signals]
fr_vals = [s['fisher_rao'] for s in drift_signals]
hd_vals = [s['hellinger'] for s in drift_signals]
wd_vals = [s['wasserstein'] for s in drift_signals]

fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    subplot_titles=[
        'Fisher-Rao Distance (information-geometric, d in [0, pi])',
        'Hellinger Distance (bounded, d in [0, 1])',
        'Sliced Wasserstein Drift (geometry-aware optimal transport)',
    ],
    vertical_spacing=0.08,
)

fig.add_trace(go.Scatter(
    x=days, y=fr_vals, mode='lines+markers',
    line=dict(color='#e74c3c', width=2), marker=dict(size=3),
    name='Fisher-Rao',
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=days, y=hd_vals, mode='lines+markers',
    line=dict(color='#f39c12', width=2), marker=dict(size=3),
    name='Hellinger',
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=days, y=wd_vals, mode='lines+markers',
    line=dict(color='#9b59b6', width=2), marker=dict(size=3),
    name='Wasserstein',
), row=3, col=1)

# Phase annotations on all panels
for row in range(1, 4):
    fig.add_vrect(x0=0, x1=29, fillcolor=C_STABLE, opacity=0.08, line_width=0, row=row, col=1)
    fig.add_vrect(x0=30, x1=59, fillcolor=C_GRADUAL, opacity=0.08, line_width=0, row=row, col=1)
    fig.add_vrect(x0=60, x1=89, fillcolor=C_SUDDEN, opacity=0.08, line_width=0, row=row, col=1)

fig.add_annotation(x=15, y=max(fr_vals) * 0.9, text='STABLE', showarrow=False,
                   font=dict(color=C_STABLE, size=12), row=1, col=1)
fig.add_annotation(x=45, y=max(fr_vals) * 0.9, text='GRADUAL DRIFT', showarrow=False,
                   font=dict(color=C_GRADUAL, size=12), row=1, col=1)
fig.add_annotation(x=75, y=max(fr_vals) * 0.9, text='SUDDEN SHIFT', showarrow=False,
                   font=dict(color=C_SUDDEN, size=12), row=1, col=1)

fig.update_layout(
    title='Distributional Drift: Three Independent Distance Metrics Over Time',
    height=700, width=1000, showlegend=False, template='plotly_dark',
)
fig.update_xaxes(title_text='Day', row=3, col=1)
fig.show()
```


```text
Computed 17 consecutive-day drift measurements
```

<iframe src="/chronos-vector/plots/mlops-drift_fig_2.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

---
## 5. Change Point Detection: When Did Drift Start?

CVX's PELT algorithm detects regime transitions in the region trajectory.
It should find transitions near day 30 (gradual drift onset) and day 60
(sudden shift).

```python
# Prepare region trajectory for PELT: list of (timestamp, float_vector)
reg_traj_pelt = [(t, [float(x) for x in d]) for t, d in reg_traj]

changepoints = cvx.detect_changepoints(
    0,  # entity_id
    reg_traj_pelt,
    penalty=3.0 * np.log(len(reg_traj_pelt)),
    min_segment_len=5,
)

print(f'Detected {len(changepoints)} regime transitions:')
for ts, sev in changepoints:
    phase = 'stable' if ts < 30 else ('gradual' if ts < 60 else 'sudden')
    print(f'  Day {ts}: severity={sev:.4f}  (in {phase} phase)')

# Overlay on Fisher-Rao plot
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=days, y=fr_vals, mode='lines',
    line=dict(color='#e74c3c', width=2), name='Fisher-Rao',
))

for ts, sev in changepoints:
    fig.add_vline(
        x=ts, line_dash='dot', line_color=C_ACCENT,
        annotation_text=f'CP day {ts} (s={sev:.2f})',
        annotation_font_size=10,
    )

fig.add_vrect(x0=0, x1=29, fillcolor=C_STABLE, opacity=0.08, line_width=0)
fig.add_vrect(x0=30, x1=59, fillcolor=C_GRADUAL, opacity=0.08, line_width=0)
fig.add_vrect(x0=60, x1=89, fillcolor=C_SUDDEN, opacity=0.08, line_width=0)

fig.update_layout(
    title=f'Change Point Detection (PELT): {len(changepoints)} Regime Transitions Found',
    xaxis_title='Day', yaxis_title='Fisher-Rao Distance',
    width=1000, height=400, template='plotly_dark',
)
fig.show()
```


```text
Detected 0 regime transitions:
```

<iframe src="/chronos-vector/plots/mlops-drift_fig_3.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

---
## 6. Drift Persistence: Will It Self-Correct?

The **Hurst exponent** (H) characterizes long-range dependence in the trajectory:

- H > 0.5: **persistent** drift --- the model will keep degrading
- H = 0.5: random walk (no memory)
- H < 0.5: **anti-persistent** (mean-reverting) --- may self-correct

With our designed drift schedule, we expect H > 0.5.

```python
traj = index.trajectory(entity_id=0)
print(f'Raw trajectory: {len(traj)} points')

h = cvx.hurst_exponent(traj)
print(f'\nHurst exponent: {h:.3f}')
if h > 0.5:
    print('  -> Persistent drift: the model is trending away from its reference distribution.')
    print('     ACTION: Retrain or rollback. This will NOT self-correct.')
elif h < 0.5:
    print('  -> Anti-persistent: fluctuations may self-correct.')
    print('     ACTION: Monitor but do not panic.')
else:
    print('  -> Random walk: no long-range dependence detected.')
```


```text
Raw trajectory: 18000 points

Hurst exponent: 0.722
  -> Persistent drift: the model is trending away from its reference distribution.
     ACTION: Retrain or rollback. This will NOT self-correct.
```

---
## 7. Velocity: How Fast Is the Model Drifting?

Velocity measures the instantaneous rate of change in embedding space.
It should be low during the stable phase, increase during gradual drift,
and spike at the sudden shift boundary.

```python
velocities = []
for ts, vec in traj[1:-1]:
    try:
        vel = cvx.velocity(traj, timestamp=ts)
        velocities.append({'day': ts, 'velocity': float(np.linalg.norm(vel))})
    except Exception:
        pass

print(f'Computed velocity at {len(velocities)} time steps')

vel_days = [v['day'] for v in velocities]
vel_vals = [v['velocity'] for v in velocities]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=vel_days, y=vel_vals, mode='lines',
    line=dict(color='#3498db', width=1.5), name='Velocity |dv/dt|',
))

# Rolling average for smoothing
if len(vel_vals) > 10:
    window = min(20, len(vel_vals) // 3)
    vel_smooth = np.convolve(vel_vals, np.ones(window) / window, mode='valid')
    smooth_days = vel_days[window // 2 : window // 2 + len(vel_smooth)]
    fig.add_trace(go.Scatter(
        x=smooth_days, y=vel_smooth.tolist(), mode='lines',
        line=dict(color='yellow', width=2.5), name='Smoothed',
    ))

fig.add_vrect(x0=0, x1=29, fillcolor=C_STABLE, opacity=0.08, line_width=0)
fig.add_vrect(x0=30, x1=59, fillcolor=C_GRADUAL, opacity=0.08, line_width=0)
fig.add_vrect(x0=60, x1=89, fillcolor=C_SUDDEN, opacity=0.08, line_width=0)

fig.add_annotation(x=15, y=max(vel_vals) * 0.95, text='STABLE', showarrow=False,
                   font=dict(color=C_STABLE, size=12))
fig.add_annotation(x=45, y=max(vel_vals) * 0.95, text='GRADUAL', showarrow=False,
                   font=dict(color=C_GRADUAL, size=12))
fig.add_annotation(x=75, y=max(vel_vals) * 0.95, text='SUDDEN', showarrow=False,
                   font=dict(color=C_SUDDEN, size=12))

fig.update_layout(
    title='Embedding Velocity Over Time',
    xaxis_title='Day', yaxis_title='Velocity Magnitude |dv/dt|',
    width=1000, height=400, template='plotly_dark',
)
fig.show()
```


```text
Computed velocity at 17998 time steps
```

<iframe src="/chronos-vector/plots/mlops-drift_fig_4.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

---
## 8. Topological Monitoring: Is the Embedding Space Fragmenting?

Persistent homology tracks the **Betti number** beta_0(r) --- the number of
connected components at filtration radius r. The Betti curve reveals cluster
structure: how many distinct clusters exist and how separated they are.

Applied to region centroids, this reveals whether the embedding space
is fragmenting into disconnected islands.

```python
topo = cvx.topological_features(centroids, n_radii=20, persistence_threshold=0.05)

print(f'Topological features of {len(centroids)} region centroids:')
print(f'  Significant clusters (n_components): {topo["n_components"]}')
print(f'  Max persistence: {topo["max_persistence"]:.4f}')
print(f'  Mean persistence: {topo["mean_persistence"]:.4f}')
print(f'  Persistence entropy: {topo["persistence_entropy"]:.4f}')
print(f'  Total persistence: {topo["total_persistence"]:.4f}')

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=topo['radii'], y=topo['betti_curve'],
    mode='lines+markers',
    line=dict(color=C_ACCENT, width=2),
    marker=dict(size=5),
    name='beta_0(r)',
))

fig.update_layout(
    title=f'Betti Curve beta_0(r) --- {len(centroids)} Region Centroids',
    xaxis_title='Filtration Radius r',
    yaxis_title='Connected Components beta_0',
    width=800, height=400, template='plotly_dark',
    annotations=[dict(
        x=0.75, y=0.9, xref='paper', yref='paper',
        text=(
            f'Significant clusters: {topo["n_components"]}<br>'
            f'Max persistence: {topo["max_persistence"]:.3f}<br>'
            f'Persistence entropy: {topo["persistence_entropy"]:.3f}'
        ),
        showarrow=False, font=dict(size=11),
        bgcolor='rgba(0,0,0,0.6)', bordercolor='white',
    )],
)
fig.show()
```


```text
Topological features of 69 region centroids:
  Significant clusters (n_components): 69
  Max persistence: 25.0146
  Mean persistence: 3.6091
  Persistence entropy: 4.0929
  Total persistence: 245.4166
```

<iframe src="/chronos-vector/plots/mlops-drift_fig_5.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

---
## 9. Event Features: Temporal Patterns in Production Traffic

Even the **timing** of high-drift events carries information.
CVX analyzes event timestamps as a point process --- burstiness,
memory, and temporal entropy characterize the drift pattern's regularity.

```python
timestamps_list = [t for t, _ in traj]
ef = cvx.event_features(timestamps_list)

print('Event features from the production embedding stream:')
print(f'  Events (embeddings):    {ef["n_events"]:.0f}')
print(f'  Span (days):            {ef["span"]:.1f}')
print(f'  Mean gap:               {ef["mean_gap"]:.4f}')
print(f'  Burstiness:             {ef["burstiness"]:.4f}')
print(f'  Memory (lag-1 autocorr):{ef["memory"]:.4f}')
print(f'  Temporal entropy:       {ef["temporal_entropy"]:.4f}')
print(f'  Intensity trend:        {ef["intensity_trend"]:.6f}')
print(f'  Circadian strength:     {ef["circadian_strength"]:.4f}')

# In a real system, burstiness of high-drift events (e.g., days where
# Fisher-Rao > threshold) would indicate whether drift comes in bursts
# or is steady. Here we demonstrate the API on the full timestamp stream.
```


```text
Event features from the production embedding stream:
  Events (embeddings):    18000
  Span (days):            89.0
  Mean gap:               0.0049
  Burstiness:             0.8683
  Memory (lag-1 autocorr):-0.0050
  Temporal entropy:       0.0312
  Intensity trend:        -40.000000
  Circadian strength:     1.0000
```

---
## 10. Complete Monitoring Dashboard

Four aligned panels showing all drift signals on a common time axis.
This is the view an MLOps team would monitor in production.

```python
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True,
    subplot_titles=[
        'Fisher-Rao Distributional Drift',
        'Velocity Magnitude |dv/dt|',
        'Betti Curve beta_0 (topological clusters)',
        'Phase Labels',
    ],
    vertical_spacing=0.07,
    row_heights=[0.3, 0.3, 0.25, 0.15],
)

# Panel 1: Fisher-Rao drift
fig.add_trace(go.Scatter(
    x=days, y=fr_vals, mode='lines',
    line=dict(color='#e74c3c', width=2), name='Fisher-Rao',
), row=1, col=1)

# Mark change points
for ts, sev in changepoints:
    fig.add_vline(
        x=ts, line_dash='dot', line_color=C_ACCENT, opacity=0.6,
        row=1, col=1,
    )

# Panel 2: Velocity
fig.add_trace(go.Scatter(
    x=vel_days, y=vel_vals, mode='lines',
    line=dict(color='#3498db', width=1.5), name='Velocity',
), row=2, col=1)

if len(vel_vals) > 10:
    fig.add_trace(go.Scatter(
        x=smooth_days, y=vel_smooth.tolist(), mode='lines',
        line=dict(color='yellow', width=2), name='Vel. (smooth)',
    ), row=2, col=1)

# Panel 3: Betti curve (static topology, shown as reference)
fig.add_trace(go.Scatter(
    x=topo['radii'], y=topo['betti_curve'],
    mode='lines+markers',
    line=dict(color='#2ecc71', width=2),
    marker=dict(size=4), name='beta_0(r)',
), row=3, col=1)

# Panel 4: Phase bar
phase_colors = {'stable': C_STABLE, 'gradual': C_GRADUAL, 'sudden': C_SUDDEN}
for phase, start, end in [('stable', 0, 29), ('gradual', 30, 59), ('sudden', 60, 89)]:
    fig.add_trace(go.Bar(
        x=[(start + end) / 2], y=[1], width=[end - start],
        marker_color=phase_colors[phase], name=phase.capitalize(),
        text=[phase.upper()], textposition='inside',
        textfont=dict(size=14, color='white'),
        showlegend=False,
    ), row=4, col=1)

# Phase shading on panels 1--2
for row in [1, 2]:
    fig.add_vrect(x0=0, x1=29, fillcolor=C_STABLE, opacity=0.06, line_width=0, row=row, col=1)
    fig.add_vrect(x0=30, x1=59, fillcolor=C_GRADUAL, opacity=0.06, line_width=0, row=row, col=1)
    fig.add_vrect(x0=60, x1=89, fillcolor=C_SUDDEN, opacity=0.06, line_width=0, row=row, col=1)

fig.update_yaxes(title_text='d_FR', row=1, col=1)
fig.update_yaxes(title_text='|dv/dt|', row=2, col=1)
fig.update_yaxes(title_text='beta_0', row=3, col=1)
fig.update_yaxes(visible=False, row=4, col=1)
fig.update_xaxes(title_text='Day', row=4, col=1)

fig.update_layout(
    title=f'MLOps Drift Monitoring Dashboard (H={h:.3f}, {len(changepoints)} change points)',
    height=900, width=1000,
    template='plotly_dark',
    showlegend=False,
)
fig.show()
```


<iframe src="/chronos-vector/plots/mlops-drift_fig_6.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

---
## Summary

This tutorial demonstrated **5 independent drift detection signals** from ChronosVector,
all computed natively inside the temporal vector index:

| CVX Function | What It Reveals | Key for MLOps |
|-------------|-----------------|---------------|
| `velocity()` | Rate of change in embedding space | Alert threshold: velocity > baseline |
| `fisher_rao_distance()`, `hellinger_distance()`, `wasserstein_drift()` | How distributions diverge between time steps | Quantify drift magnitude |
| `detect_changepoints()` | When regime transitions occurred (PELT) | Pinpoint drift onset |
| `hurst_exponent()` | Whether drift is persistent (H>0.5) or transient (H<0.5) | Decide: retrain vs. wait |
| `topological_features()` | Whether embedding space is fragmenting | Detect structural collapse |
| `region_trajectory()` | How the distribution over HNSW regions evolves | Foundation for all above |
| `event_features()` | Temporal patterns in event timing | Detect burst vs. steady drift |

### The key insight

Each signal captures a **different dimension of change**:

- **Velocity** detects *speed* of drift but not *direction*.
- **Distributional distances** detect *magnitude* of distributional shift.
- **Change points** detect *timing* of regime transitions.
- **Hurst** detects *persistence* --- will it get worse?
- **Topology** detects *structural* changes --- is the space fragmenting?

A robust MLOps monitoring system should use **all five** in combination.
CVX computes them all from a single temporal vector index, with no external
dependencies beyond the embeddings themselves.
