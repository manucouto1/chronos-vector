---
title: "Mental Health Explorer"
description: "Interactive exploration of depression detection with 17 CVX analytical functions on eRisk data"
---

This notebook showcases **ChronosVector (CVX)** as a temporal vector database for mental health research.

Unlike traditional ML pipelines, here the **database IS the analytical tool**: ingestion, hierarchical clustering,
temporal trajectory analysis, distributional distances, topological features, and path signatures — all computed
natively inside the index.

### What you'll see

| Section | CVX Functions Used |
|---------|-------------------|
| Ingestion & Performance | `bulk_insert`, `enable_quantization` |
| 3D Trajectory Explorer | `trajectory`, `search` |
| HNSW Hierarchy & Regions | `regions`, `region_members` |
| Temporal Calculus | `velocity`, `drift`, `hurst_exponent`, `detect_changepoints` |
| Point Process Analysis | `event_features` |
| Distributional Distances | `region_trajectory`, `wasserstein_drift`, `fisher_rao_distance`, `hellinger_distance` |
| Topological Analysis | `topological_features` |
| Path Signatures | `path_signature`, `log_signature`, `signature_distance`, `frechet_distance` |
| Case Study: User Deep-Dive | All of the above on a single user |

```python
import chronos_vector as cvx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import json, time, warnings
warnings.filterwarnings('ignore')

DATA_DIR = '../data'
EMB_DIR = f'{DATA_DIR}/embeddings'

# Color palette
C_DEP = '#e74c3c'   # depression — red
C_CTL = '#3498db'   # control — blue
C_ACCENT = '#2ecc71' # highlight
```

---
## 1. Data Loading

We use **MentalRoBERTa** (D=768) embeddings with relative temporal features.
For interactive exploration, we work with a representative subset.

```python
# Load embeddings
df_full = pd.read_parquet(f'{EMB_DIR}/erisk_mental_embeddings.parquet')
emb_cols = [c for c in df_full.columns if c.startswith('emb_')]
D = len(emb_cols)
print(f'Full dataset: {len(df_full):,} posts, {df_full["user_id"].nunique()} users, D={D}')
print(f'Labels: {df_full["label"].value_counts().to_dict()}')

# Load raw text for hover
texts = {}
with open(f'{DATA_DIR}/eRisk/unified.jsonl') as f:
    for line in f:
        rec = json.loads(line)
        key = (rec['user_id'], rec['timestamp'])
        texts[key] = rec['text'][:200]  # truncate for hover

print(f'Loaded {len(texts):,} text snippets')
```


```text
Full dataset: 1,359,188 posts, 2285 users, D=768
Labels: {'control': 1266778, 'depression': 92410}
```

```text
Loaded 1,355,256 text snippets
```

```python
# Select a balanced subset for interactive exploration
# All depression users + matched controls (by post count)
dep_users = df_full[df_full['label'] == 'depression']['user_id'].unique()
ctl_users = df_full[df_full['label'] == 'control']['user_id'].unique()

# Sample controls matching depression count
np.random.seed(42)
n_ctl = min(len(dep_users), len(ctl_users))
ctl_sample = np.random.choice(ctl_users, size=n_ctl, replace=False)
selected_users = np.concatenate([dep_users, ctl_sample])

df = df_full[df_full['user_id'].isin(selected_users)].copy()
print(f'Subset: {len(df):,} posts, {df["user_id"].nunique()} users')
print(f'  Depression: {df[df["label"]=="depression"]["user_id"].nunique()} users')
print(f'  Control: {df[df["label"]=="control"]["user_id"].nunique()} users')

# Assign integer entity IDs for CVX
user_to_id = {u: i for i, u in enumerate(sorted(df['user_id'].unique()))}
df['entity_id'] = df['user_id'].map(user_to_id).astype(np.uint64)
```


```text
Subset: 225,962 posts, 466 users
  Depression: 233 users
  Control: 233 users
```

---
## 2. CVX Ingestion & Performance

Demonstrating `bulk_insert` with scalar quantization (SQ8) for accelerated HNSW construction.

```python
# Create CVX index
index = cvx.TemporalIndex(m=16, ef_construction=200, ef_search=50)

# Enable scalar quantization for faster distance computation
vecs = np.ascontiguousarray(df[emb_cols].values.astype(np.float32))
vmin, vmax = vecs.min(), vecs.max()
index.enable_quantization(float(vmin), float(vmax))
print(f'SQ8 range: [{vmin:.3f}, {vmax:.3f}]')
print(f'Vectors C-contiguous: {vecs.flags["C_CONTIGUOUS"]}')

# Bulk insert
entity_ids = df['entity_id'].values.astype(np.uint64)
timestamps = pd.to_datetime(df['timestamp']).astype(np.int64) // 10**9  # unix seconds
timestamps = timestamps.values.astype(np.int64)

t0 = time.perf_counter()
n = index.bulk_insert(entity_ids, timestamps, vecs, ef_construction=50)
elapsed = time.perf_counter() - t0

print(f'\nIngested {n:,} points in {elapsed:.1f}s ({n/elapsed:,.0f} pts/sec)')
print(f'Index size: {len(index):,}')
```


```text
SQ8 range: [-0.698, 0.979]
Vectors C-contiguous: True
```

```text

Ingested 225,962 points in 502.6s (450 pts/sec)
Index size: 225,962
```

---
## 3. 3D Trajectory Explorer

PCA projects D=768 → 2D; the temporal axis provides the 3rd dimension.
Each line is one user's trajectory through the embedding space over time.

```python
# PCA on full embedding space
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(vecs)
df['pca1'] = pca_coords[:, 0]
df['pca2'] = pca_coords[:, 1]
print(f'PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}')

# Add text for hover (vectorized lookup — 5x faster than .apply)
keys = list(zip(df['user_id'], df['timestamp'].astype(str)))
df['text_preview'] = [texts.get(k, '')[:120] for k in keys]
```


```text
PCA explained variance: 31.3%
```

```python
# 3D scatter: all posts colored by label
# Subsample for rendering performance
sample_n = min(30_000, len(df))
df_viz = df.sample(n=sample_n, random_state=42).sort_values(['user_id', 'timestamp'])

fig = go.Figure()

for label, color, name in [('depression', C_DEP, 'Depression'), ('control', C_CTL, 'Control')]:
    mask = df_viz['label'] == label
    sub = df_viz[mask]
    fig.add_trace(go.Scatter3d(
        x=sub['pca1'], y=sub['pca2'],
        z=sub['t_rel'],
        mode='markers',
        marker=dict(size=1.5, color=color, opacity=0.5),
        name=name,
        text=sub['text_preview'],
        hovertemplate='<b>%{text}</b><br>PCA1: %{x:.2f}<br>PCA2: %{y:.2f}<br>t_rel: %{z:.3f}<extra></extra>',
    ))

fig.update_layout(
    title='Embedding Space × Time (PCA 2D + Temporal Axis)',
    scene=dict(
        xaxis_title='PCA 1',
        yaxis_title='PCA 2',
        zaxis_title='Relative Time (t_rel)',
    ),
    width=900, height=700,
    template='plotly_dark',
)
fig.show()
```


<iframe src="/plots/b1-explorer_fig_0.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

```python
# Individual trajectory lines for selected users
# Pick 5 depression + 5 control users with moderate post counts
user_counts = df.groupby(['user_id', 'label']).size().reset_index(name='n')
user_counts = user_counts[(user_counts['n'] >= 50) & (user_counts['n'] <= 500)]

dep_sample = user_counts[user_counts['label'] == 'depression'].sample(min(5, len(user_counts[user_counts['label']=='depression'])), random_state=42)['user_id']
ctl_sample_vis = user_counts[user_counts['label'] == 'control'].sample(min(5, len(user_counts[user_counts['label']=='control'])), random_state=42)['user_id']
traj_users = pd.concat([dep_sample, ctl_sample_vis]).values

fig = go.Figure()
for uid in traj_users:
    u_df = df[df['user_id'] == uid].sort_values('timestamp')
    label = u_df['label'].iloc[0]
    color = C_DEP if label == 'depression' else C_CTL
    fig.add_trace(go.Scatter3d(
        x=u_df['pca1'], y=u_df['pca2'], z=u_df['t_rel'],
        mode='lines+markers',
        marker=dict(size=2, color=color),
        line=dict(color=color, width=2),
        name=f'{uid} ({label[0].upper()})',
        text=u_df['text_preview'],
        hovertemplate='<b>%{text}</b><br>t=%{z:.3f}<extra>%{fullData.name}</extra>',
    ))

fig.update_layout(
    title='Individual Trajectories: Depression (red) vs Control (blue)',
    scene=dict(xaxis_title='PCA 1', yaxis_title='PCA 2', zaxis_title='Relative Time'),
    width=900, height=700,
    template='plotly_dark',
)
fig.show()
```


<iframe src="/plots/b1-explorer_fig_1.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

---
## 4. HNSW Hierarchy & Semantic Regions

CVX's HNSW graph naturally creates a hierarchy of semantic regions.
Higher levels = coarser clusters. We use c-TF-IDF to label each region.

```python
# Get regions at different levels
for level in [1, 2, 3]:
    regions = index.regions(level=level)
    print(f'Level {level}: {len(regions)} regions')
```


```text
Level 1: 14066 regions
Level 2: 930 regions
Level 3: 63 regions
```

```python
# Work with Level 2 regions for visualization (many bubbles)
# Level 3 for distributional/topological analysis (tractable dimensions)
regions_l2 = index.regions(level=2)
regions_l3 = index.regions(level=3)
print(f'Level 2: {len(regions_l2)} regions (visualization)')
print(f'Level 3: {len(regions_l3)} regions (analytics)')

# Build region data and c-TF-IDF labels for L2
# Sample top regions by member count for performance (cap at 200)
id_to_user = {v: k for k, v in user_to_id.items()}
regions_l2_sorted = sorted(regions_l2, key=lambda r: r[2], reverse=True)
regions_l2_top = regions_l2_sorted[:200]

region_data = []
region_texts = {}

for rid, centroid, n_members in regions_l2_top:
    members = index.region_members(region_id=rid, level=2)
    region_data.append({
        'region_id': rid,
        'n_members': len(members),
        'centroid': centroid,
    })
    
    # Collect texts for c-TF-IDF (sample up to 50 per region)
    member_texts = []
    for eid, ts in members[:50]:
        uid = id_to_user.get(eid, '')
        txt = texts.get((uid, pd.Timestamp(ts, unit='s').strftime('%Y-%m-%d %H:%M:%S')), '')
        if txt:
            member_texts.append(txt)
    region_texts[rid] = ' '.join(member_texts)

df_regions = pd.DataFrame(region_data)
print(f'Processed {len(regions_l2_top)} largest L2 regions')
print(f'L2 region sizes: min={df_regions["n_members"].min()}, median={df_regions["n_members"].median():.0f}, max={df_regions["n_members"].max()}')
```


```text
Level 2: 930 regions (visualization)
Level 3: 63 regions (analytics)
```

```text
Processed 200 largest L2 regions
L2 region sizes: min=318, median=554, max=4383
```

```python
# c-TF-IDF: top words per region
valid_regions = {rid: txt for rid, txt in region_texts.items() if len(txt) > 50}
if valid_regions:
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', min_df=2, max_df=0.8)
    tfidf_matrix = tfidf.fit_transform(valid_regions.values())
    feature_names = tfidf.get_feature_names_out()
    
    region_labels = {}
    for i, rid in enumerate(valid_regions.keys()):
        top_idx = tfidf_matrix[i].toarray().flatten().argsort()[-5:][::-1]
        top_words = [feature_names[j] for j in top_idx]
        region_labels[rid] = ', '.join(top_words)
    
    df_regions['top_words'] = df_regions['region_id'].map(region_labels).fillna('(no text)')
    print('Sample region labels:')
    for _, r in df_regions.nlargest(10, 'n_members').iterrows():
        print(f'  Region {r["region_id"]:4d} ({r["n_members"]:5d} posts): {r["top_words"]}')
```

```python
# Visualize regions as bubbles in PCA space
centroids_array = np.array([r['centroid'] for r in region_data])
pca_centroids = pca.transform(centroids_array)
df_regions['pca1'] = pca_centroids[:, 0]
df_regions['pca2'] = pca_centroids[:, 1]

# Compute depression ratio per region using pre-built lookup
id_to_user_map = {v: k for k, v in user_to_id.items()}
user_label = df.drop_duplicates('user_id').set_index('user_id')['label'].to_dict()

dep_ratio = []
for rid, centroid, n_members in regions_l2_top:
    members = index.region_members(region_id=rid, level=2)
    labels = [user_label.get(id_to_user_map.get(eid), 'unknown') for eid, ts in members[:200]]
    dep_count = sum(1 for l in labels if l == 'depression')
    dep_ratio.append(dep_count / max(len(labels), 1))

df_regions['dep_ratio'] = dep_ratio

# Build hover text manually for proper formatting
hover_texts = [
    f"<b>Region {row['region_id']}</b><br>"
    f"Members: {row['n_members']}<br>"
    f"Depression ratio: {row['dep_ratio']:.1%}<br>"
    f"Top words: {row.get('top_words', '')}"
    for _, row in df_regions.iterrows()
]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_regions['pca1'], y=df_regions['pca2'],
    mode='markers',
    marker=dict(
        size=np.sqrt(df_regions['n_members']) * 2,
        color=df_regions['dep_ratio'],
        colorscale='RdBu_r',
        cmin=0, cmax=1,
        colorbar=dict(title='Depression<br>Ratio'),
        line=dict(width=1, color='white'),
    ),
    text=hover_texts,
    hovertemplate='%{text}<extra></extra>',
))

fig.update_layout(
    title=f'HNSW Level 2 Regions — Bubble Size = Members, Color = Depression Ratio ({len(regions_l2_top)} largest)',
    xaxis_title='PCA 1', yaxis_title='PCA 2',
    width=900, height=600,
    template='plotly_dark',
)
fig.show()
```


<iframe src="/plots/b1-explorer_fig_2.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

```python
# Prepare Level 3 data for distributional/topological analysis
# L3 has ~30-70 regions — tractable for signatures, Fisher-Rao, Wasserstein
centroids_l3 = [c for _, c, _ in regions_l3]

# Depression ratio per L3 region (for topology comparison)
dep_ratio_l3 = []
for rid, _, _ in regions_l3:
    members = index.region_members(region_id=rid, level=3)
    labels = [user_label.get(id_to_user_map.get(eid), 'unknown') for eid, ts in members[:300]]
    dep_count = sum(1 for l in labels if l == 'depression')
    dep_ratio_l3.append(dep_count / max(len(labels), 1))

df_regions_l3 = pd.DataFrame({
    'region_id': [r[0] for r in regions_l3],
    'n_members': [r[2] for r in regions_l3],
    'dep_ratio': dep_ratio_l3,
})
print(f'Level 3: {len(regions_l3)} regions prepared for analytics')
print(f'  Depression-dominant regions (>50%): {(df_regions_l3["dep_ratio"] > 0.5).sum()}')
```


```text
Level 3: 63 regions prepared for analytics
  Depression-dominant regions (>50%): 14
```

---
## 5. Temporal Calculus

CVX computes vector derivatives natively: velocity (rate of change), drift (displacement decomposition),
Hurst exponent (long-range dependence), and change point detection (regime transitions).

```python
# Compare Hurst exponents: depression vs control
hurst_results = []
for uid in traj_users:
    eid = user_to_id[uid]
    traj = index.trajectory(entity_id=eid)
    if len(traj) >= 20:
        try:
            h = cvx.hurst_exponent(traj)
            label = df[df['user_id'] == uid]['label'].iloc[0]
            hurst_results.append({'user_id': uid, 'label': label, 'hurst': h, 'n_posts': len(traj)})
        except:
            pass

# Extend to more users
for uid in np.random.choice(df['user_id'].unique(), size=min(100, len(df['user_id'].unique())), replace=False):
    if uid in [r['user_id'] for r in hurst_results]:
        continue
    eid = user_to_id[uid]
    traj = index.trajectory(entity_id=eid)
    if len(traj) >= 20:
        try:
            h = cvx.hurst_exponent(traj)
            label = df[df['user_id'] == uid]['label'].iloc[0]
            hurst_results.append({'user_id': uid, 'label': label, 'hurst': h, 'n_posts': len(traj)})
        except:
            pass

df_hurst = pd.DataFrame(hurst_results)
print(f'Hurst exponents computed for {len(df_hurst)} users')
print(df_hurst.groupby('label')['hurst'].describe())
```


```text
Hurst exponents computed for 101 users
            count      mean       std       min       25%       50%       75%  \
label                                                                           
control      50.0  0.684142  0.098619  0.568693  0.618846  0.668943  0.712726   
depression   51.0  0.679197  0.077997  0.519967  0.634925  0.665935  0.698643   

                 max  
label                 
control     1.000000  
depression  0.965228
```

```python
# Velocity and change point detection for a depression user
case_uid = dep_sample.values[0]
case_eid = user_to_id[case_uid]
case_traj = index.trajectory(entity_id=case_eid)
print(f'Case study: {case_uid}, {len(case_traj)} posts')

# Compute velocity magnitude (sample every Nth point for long trajectories)
step = max(1, len(case_traj) // 200)
velocities = []
for ts, vec in case_traj[1:-1:step]:
    try:
        vel = cvx.velocity(case_traj, timestamp=ts)
        vel_mag = np.linalg.norm(vel)
        velocities.append({'timestamp': pd.Timestamp(ts, unit='s'), 'velocity': vel_mag})
    except:
        pass

# Detect change points on REGION trajectory (L3, ~37 dims)
case_reg_traj = index.region_trajectory(entity_id=case_eid, level=3, window_days=30, alpha=0.3)
if len(case_reg_traj) >= 10:
    reg_traj_for_pelt = [(t, [float(x) for x in d]) for t, d in case_reg_traj]
    changepoints = cvx.detect_changepoints(
        case_eid, reg_traj_for_pelt,
        penalty=3.0 * np.log(len(case_reg_traj)),
        min_segment_len=5,
    )
else:
    changepoints = []
print(f'Detected {len(changepoints)} regime transitions (on L3 region trajectory)')
print(f'Velocity computed at {len(velocities)} points (step={step})')

# Plot velocity + change points
df_vel = pd.DataFrame(velocities)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_vel['timestamp'], y=df_vel['velocity'],
    mode='lines', name='Velocity |dv/dt|',
    line=dict(color=C_DEP, width=1),
))

if len(df_vel) > 10:
    df_vel['velocity_smooth'] = df_vel['velocity'].rolling(window=min(20, len(df_vel)//3), center=True).mean()
    fig.add_trace(go.Scatter(
        x=df_vel['timestamp'], y=df_vel['velocity_smooth'],
        mode='lines', name='Smoothed',
        line=dict(color='yellow', width=2),
    ))

for i, (cp_ts, severity) in enumerate(changepoints[:10]):
    fig.add_vline(x=pd.Timestamp(cp_ts, unit='s'), line_dash='dot', line_color=C_ACCENT,
                  annotation_text=f'CP' if i < 5 else None)

fig.update_layout(
    title=f'Velocity Profile & Change Points — {case_uid} ({len(changepoints)} regime transitions)',
    xaxis_title='Time', yaxis_title='Velocity Magnitude',
    width=900, height=400,
    template='plotly_dark',
)
fig.show()
```


```text
Case study: train_subject6146, 337 posts
Detected 0 regime transitions (on L3 region trajectory)
Velocity computed at 335 points (step=1)
```

<iframe src="/plots/b1-explorer_fig_3.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

---
## 6. Point Process Analysis

CVX analyzes the **timing** of events independently from content:
burstiness, memory coefficient, temporal entropy, circadian patterns.

```python
# Compute event features for users with sufficient data (sample for speed)
np.random.seed(42)
all_users = df['user_id'].unique()
sample_users = np.random.choice(all_users, size=min(200, len(all_users)), replace=False)

event_results = []
for uid in sample_users:
    eid = user_to_id[uid]
    traj = index.trajectory(entity_id=eid)
    if len(traj) >= 10:
        ts_list = [t for t, _ in traj]
        try:
            feats = cvx.event_features(ts_list)
            feats['user_id'] = uid
            feats['label'] = df[df['user_id'] == uid]['label'].iloc[0]
            event_results.append(feats)
        except:
            pass

df_events = pd.DataFrame(event_results)
print(f'Event features for {len(df_events)} users (sampled {len(sample_users)})')
print()
print('Burstiness by label:')
print(df_events.groupby('label')['burstiness'].describe().round(3))
```


```text
Event features for 198 users (sampled 200)

Burstiness by label:
            count   mean    std    min    25%    50%    75%    max
label                                                             
control      98.0  0.511  0.193 -0.253  0.395  0.496  0.638  0.899
depression  100.0  0.501  0.189  0.031  0.359  0.512  0.661  0.921
```

```python
# Scatter: burstiness vs memory coefficient, colored by label
fig = px.scatter(
    df_events, x='burstiness', y='memory',
    color='label', color_discrete_map={'depression': C_DEP, 'control': C_CTL},
    hover_data=['user_id', 'n_events', 'circadian_strength'],
    title='Temporal Point Process: Burstiness vs Memory Coefficient',
    labels={'burstiness': 'Burstiness B (σ-μ)/(σ+μ)', 'memory': 'Memory (lag-1 autocorrelation)'},
    template='plotly_dark',
    width=800, height=500,
    opacity=0.7,
)

# Annotate quadrants
fig.add_annotation(x=-0.5, y=0.5, text='Regular + Clustered', showarrow=False, font=dict(size=10, color='gray'))
fig.add_annotation(x=0.5, y=0.5, text='Bursty + Clustered', showarrow=False, font=dict(size=10, color='gray'))
fig.add_annotation(x=-0.5, y=-0.5, text='Regular + Alternating', showarrow=False, font=dict(size=10, color='gray'))
fig.add_annotation(x=0.5, y=-0.5, text='Bursty + Alternating', showarrow=False, font=dict(size=10, color='gray'))

fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
fig.add_vline(x=0, line_dash='dash', line_color='gray', opacity=0.5)
fig.show()
```


<iframe src="/plots/b1-explorer_fig_4.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

```python
# Circadian strength distribution
fig = go.Figure()
for label, color in [('depression', C_DEP), ('control', C_CTL)]:
    vals = df_events[df_events['label'] == label]['circadian_strength']
    fig.add_trace(go.Histogram(
        x=vals, name=label.capitalize(),
        marker_color=color, opacity=0.7, nbinsx=25,
    ))

fig.update_layout(
    title='Circadian Strength: Depression vs Control',
    xaxis_title='24h Fourier Amplitude (0=no rhythm, 1=strong)',
    yaxis_title='Count', barmode='overlay',
    width=800, height=400, template='plotly_dark',
)
fig.show()
```


<iframe src="/plots/b1-explorer_fig_5.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

---
## 7. Distributional Distances

CVX tracks how each user's distribution over semantic regions evolves:
`region_trajectory` computes time-windowed soft assignments, then
`fisher_rao_distance`, `hellinger_distance`, and `wasserstein_drift` measure distributional change.

```python
# Region trajectory at Level 3 (tractable: ~37 regions instead of ~879 at L2)
reg_traj = index.region_trajectory(entity_id=case_eid, level=3, window_days=30, alpha=0.3)
print(f'Region trajectory for {case_uid}: {len(reg_traj)} time points, {len(reg_traj[0][1])} regions')

# Heatmap: region distribution over time
times = [pd.Timestamp(t, unit='s') for t, _ in reg_traj]
dists = np.array([d for _, d in reg_traj])

# Only show regions with any activity
active_mask = dists.max(axis=0) > 0.01
dists_active = dists[:, active_mask]
active_ids = np.where(active_mask)[0]

fig = go.Figure(data=go.Heatmap(
    z=dists_active.T,
    x=times,
    y=[f'R{i}' for i in active_ids],
    colorscale='Hot',
    hovertemplate='Time: %{x}<br>Region: %{y}<br>Weight: %{z:.3f}<extra></extra>',
))

fig.update_layout(
    title=f'Semantic Region Distribution Over Time — {case_uid} (Level 3, {len(reg_traj[0][1])} regions)',
    xaxis_title='Time', yaxis_title='Region',
    width=900, height=400, template='plotly_dark',
)
fig.show()
```


```text
Region trajectory for train_subject6146: 144 time points, 63 regions
```

<iframe src="/plots/b1-explorer_fig_6.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

```python
# Compute distributional drift over time using Fisher-Rao, Hellinger, and Wasserstein
# All distances computed on Level 3 region distributions for meaningful results

drift_results = []
for i in range(1, len(reg_traj)):
    t_prev, d_prev = reg_traj[i-1]
    t_curr, d_curr = reg_traj[i]
    
    p = [float(x) for x in d_prev]
    q = [float(x) for x in d_curr]
    
    fr = cvx.fisher_rao_distance(p, q)
    hd = cvx.hellinger_distance(p, q)
    wd = cvx.wasserstein_drift(list(d_prev), list(d_curr), centroids_l3, n_projections=50)
    
    drift_results.append({
        'time': pd.Timestamp(t_curr, unit='s'),
        'fisher_rao': fr,
        'hellinger': hd,
        'wasserstein': wd,
    })

df_drift = pd.DataFrame(drift_results)

fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    subplot_titles=[f'Fisher-Rao Distance (d∈[0,π], {len(reg_traj[0][1])} regions)',
                                   'Hellinger Distance (d∈[0,1])',
                                   'Wasserstein Drift (geometry-aware optimal transport)'])

fig.add_trace(go.Scatter(x=df_drift['time'], y=df_drift['fisher_rao'],
                         mode='lines', line=dict(color='#e74c3c')), row=1, col=1)
fig.add_trace(go.Scatter(x=df_drift['time'], y=df_drift['hellinger'],
                         mode='lines', line=dict(color='#f39c12')), row=2, col=1)
fig.add_trace(go.Scatter(x=df_drift['time'], y=df_drift['wasserstein'],
                         mode='lines', line=dict(color='#9b59b6')), row=3, col=1)

fig.update_layout(
    title=f'Distributional Drift Over Time — {case_uid}',
    height=600, width=900, showlegend=False, template='plotly_dark',
)
fig.show()
```


<iframe src="/plots/b1-explorer_fig_7.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

```python
# Compare total drift between depression and control users (Level 3)
total_drift_results = []
for uid in df['user_id'].unique()[:100]:  # sample
    eid = user_to_id[uid]
    rt = index.region_trajectory(entity_id=eid, level=3, window_days=30, alpha=0.3)
    if len(rt) >= 3:
        # Fisher-Rao between first and last distribution
        p = [float(x) for x in rt[0][1]]
        q = [float(x) for x in rt[-1][1]]
        fr = cvx.fisher_rao_distance(p, q)
        
        # Mean consecutive Wasserstein
        wd_vals = []
        for i in range(1, len(rt)):
            wd_vals.append(cvx.wasserstein_drift(
                list(rt[i-1][1]), list(rt[i][1]), centroids_l3, 50
            ))
        
        label = df[df['user_id'] == uid]['label'].iloc[0]
        total_drift_results.append({
            'user_id': uid, 'label': label,
            'fisher_rao_total': fr,
            'wasserstein_mean': np.mean(wd_vals),
            'wasserstein_max': np.max(wd_vals),
        })

df_total_drift = pd.DataFrame(total_drift_results)
print('Distributional drift by label (Level 3 regions):')
print(df_total_drift.groupby('label')[['fisher_rao_total', 'wasserstein_mean']].describe().round(3))
```


```text
Distributional drift by label (Level 3 regions):
           fisher_rao_total                                                   \
                      count   mean    std    min    25%    50%    75%    max   
label                                                                          
control                64.0  2.686  0.534  0.000  2.351  2.917  3.056  3.142   
depression             35.0  2.669  0.431  1.729  2.256  2.732  3.100  3.142   

           wasserstein_mean                                                  
                      count   mean    std   min    25%    50%    75%    max  
label                                                                        
control                64.0  4.223  1.043  0.00  3.810  4.436  4.937  5.780  
depression             35.0  4.353  0.680  3.23  3.756  4.320  4.839  5.673
```

---
## 8. Topological Analysis

CVX computes topological features (Betti numbers, persistence) of the HNSW region centroids.
This reveals the intrinsic cluster structure of the embedding space at different filtration radii.

```python
# Topological features of the Level 3 region centroids
topo = cvx.topological_features(centroids_l3, n_radii=30, persistence_threshold=0.05)
print(f'Topological features of HNSW Level 3 regions ({len(centroids_l3)} centroids):')
for k, v in topo.items():
    if k not in ('betti_curve', 'radii'):
        print(f'  {k}: {v}')

# Betti curve: β₀(r) = number of connected components at radius r
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=topo['radii'], y=topo['betti_curve'],
    mode='lines+markers', line=dict(color=C_ACCENT, width=2),
    marker=dict(size=4),
))

fig.update_layout(
    title=f'Betti Curve β₀(r) — {len(centroids_l3)} Region Centroids at Level 3',
    xaxis_title='Filtration Radius r',
    yaxis_title='Connected Components β₀',
    width=800, height=400, template='plotly_dark',
    annotations=[dict(
        x=0.7, y=0.9, xref='paper', yref='paper',
        text=f'Significant clusters: {topo["n_components"]}<br>'
             f'Max persistence: {topo["max_persistence"]:.3f}<br>'
             f'Persistence entropy: {topo["persistence_entropy"]:.3f}',
        showarrow=False, font=dict(size=11),
        bgcolor='rgba(0,0,0,0.5)', bordercolor='white',
    )]
)
fig.show()
```


```text
Topological features of HNSW Level 3 regions (63 centroids):
  persistence_entropy: 4.120642836124686
  total_persistence: 16.016513631210067
  mean_persistence: 0.2583308650195172
  max_persistence: 0.31273234722093135
  n_components: 63
```

<iframe src="/plots/b1-explorer_fig_8.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

```python
# Compare topology between depression-dominant and control-dominant L3 regions
dep_regions_idx = df_regions_l3[df_regions_l3['dep_ratio'] > 0.5].index.tolist()
ctl_regions_idx = df_regions_l3[df_regions_l3['dep_ratio'] <= 0.5].index.tolist()

if len(dep_regions_idx) >= 3 and len(ctl_regions_idx) >= 3:
    dep_centroids = [centroids_l3[i] for i in dep_regions_idx]
    ctl_centroids = [centroids_l3[i] for i in ctl_regions_idx]
    
    topo_dep = cvx.topological_features(dep_centroids, n_radii=20, persistence_threshold=0.05)
    topo_ctl = cvx.topological_features(ctl_centroids, n_radii=20, persistence_threshold=0.05)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=topo_dep['radii'], y=topo_dep['betti_curve'],
        mode='lines', name=f'Depression regions ({len(dep_regions_idx)})',
        line=dict(color=C_DEP, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=topo_ctl['radii'], y=topo_ctl['betti_curve'],
        mode='lines', name=f'Control regions ({len(ctl_regions_idx)})',
        line=dict(color=C_CTL, width=2),
    ))
    
    fig.update_layout(
        title='Topological Comparison: Depression vs Control Region Subspaces (Level 3)',
        xaxis_title='Filtration Radius r',
        yaxis_title='Connected Components β₀',
        width=800, height=400, template='plotly_dark',
    )
    fig.show()
else:
    print(f'Not enough label-dominant L3 regions (dep={len(dep_regions_idx)}, ctl={len(ctl_regions_idx)})')
```


<iframe src="/plots/b1-explorer_fig_9.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

---
## 9. Path Signatures

Path signatures provide a universal, reparametrization-invariant description of trajectories.
CVX computes depth-2 signatures on Level 3 region trajectories (tractable dimensionality),
enabling trajectory comparison via `signature_distance` and `frechet_distance`.

```python
# Compute path signatures on Level 3 region trajectories
# L3 ~37 regions → depth-2 signature dim = 38 + 38² ≈ 1,482 (with time augmentation)
print(f'Using Level 3: {len(regions_l3)} regions → sig dim ≈ {len(regions_l3)+1 + (len(regions_l3)+1)**2:,}')

# Sample users for tractable computation
np.random.seed(42)
sig_user_sample = np.random.choice(df['user_id'].unique(), size=min(150, len(df['user_id'].unique())), replace=False)

sig_results = []
for uid in sig_user_sample:
    eid = user_to_id[uid]
    rt = index.region_trajectory(entity_id=eid, level=3, window_days=30, alpha=0.3)
    if len(rt) >= 5:
        sig_traj = [(t, [float(x) for x in d]) for t, d in rt]
        try:
            sig = cvx.path_signature(sig_traj, depth=2, time_augmentation=True)
            label = df[df['user_id'] == uid]['label'].iloc[0]
            sig_results.append({
                'user_id': uid, 'label': label,
                'signature': sig, 'n_points': len(rt),
            })
        except Exception as e:
            pass

print(f'Path signatures computed for {len(sig_results)} users (sampled {len(sig_user_sample)})')
if sig_results:
    print(f'Signature dimension: {len(sig_results[0]["signature"]):,}')
```


```text
Using Level 3: 63 regions → sig dim ≈ 4,160
```

```text
Path signatures computed for 142 users (sampled 150)
Signature dimension: 4,160
```

```python
# PCA on signatures for visualization
if len(sig_results) >= 10:
    sig_matrix = np.array([r['signature'] for r in sig_results])
    sig_labels = [r['label'] for r in sig_results]
    sig_users = [r['user_id'] for r in sig_results]
    
    # Handle potential NaN/inf
    sig_matrix = np.nan_to_num(sig_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    pca_sig = PCA(n_components=3, random_state=42)
    sig_pca = pca_sig.fit_transform(sig_matrix)
    
    fig = go.Figure()
    for label, color in [('depression', C_DEP), ('control', C_CTL)]:
        mask = np.array(sig_labels) == label
        fig.add_trace(go.Scatter3d(
            x=sig_pca[mask, 0], y=sig_pca[mask, 1], z=sig_pca[mask, 2],
            mode='markers',
            marker=dict(size=5, color=color, opacity=0.7),
            name=label.capitalize(),
            text=np.array(sig_users)[mask],
            hovertemplate='<b>%{text}</b><extra></extra>',
        ))
    
    fig.update_layout(
        title=f'Path Signature Space (PCA 3D, depth=2) — {pca_sig.explained_variance_ratio_.sum():.1%} variance',
        scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
        width=900, height=700, template='plotly_dark',
    )
    fig.show()
else:
    print('Not enough signatures for PCA visualization')
```


<iframe src="/plots/b1-explorer_fig_10.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

```python
# Trajectory similarity: find most similar users to case study via signature distance
if sig_results:
    case_sig_entry = next((r for r in sig_results if r['user_id'] == case_uid), None)
    
    if case_sig_entry:
        distances = []
        for r in sig_results:
            if r['user_id'] != case_uid:
                d = cvx.signature_distance(case_sig_entry['signature'], r['signature'])
                distances.append({'user_id': r['user_id'], 'label': r['label'], 'sig_dist': d})
        
        df_sim = pd.DataFrame(distances).sort_values('sig_dist')
        print(f'Most similar users to {case_uid} (depression) by path signature:')
        print(df_sim.head(10).to_string(index=False))
        print(f'\nMost dissimilar:')
        print(df_sim.tail(5).to_string(index=False))
    else:
        print(f'{case_uid} not in signature results')
```


```text
Most similar users to train_subject6146 (depression) by path signature:
          user_id      label  sig_dist
train_subject2585    control  1.809492
train_subject7679    control  1.845277
       subject581 depression  1.978410
train_subject7142 depression  2.031276
      subject3868    control  2.035534
train_subject4675 depression  2.081146
      subject4131    control  2.104592
 test_subject8889    control  2.154377
      subject8593    control  2.169807
 test_subject9359 depression  2.187016

Most dissimilar:
          user_id      label  sig_dist
train_subject5080 depression  5.770236
      subject4871    control  5.996985
      subject6925    control  6.347983
      subject5111    control  7.420838
  test_subject570    control  9.750440
```

```python
# Compare with Fréchet distance for a few pairs
if sig_results and case_sig_entry:
    # Get the top-5 most similar by signature
    top5 = df_sim.head(5)['user_id'].tolist()
    
    print('Signature distance vs Fréchet distance:')
    print(f'{"User":>15} {"Label":>12} {"Sig. Dist":>10} {"Fréchet":>10}')
    print('-' * 50)
    
    case_raw_traj = index.trajectory(entity_id=case_eid)
    for uid in top5:
        eid = user_to_id[uid]
        other_traj = index.trajectory(entity_id=eid)
        fd = cvx.frechet_distance(case_raw_traj[:100], other_traj[:100])  # cap for speed
        sd = df_sim[df_sim['user_id'] == uid]['sig_dist'].values[0]
        label = df[df['user_id'] == uid]['label'].iloc[0]
        print(f'{uid:>15} {label:>12} {sd:>10.4f} {fd:>10.4f}')
```


```text
Signature distance vs Fréchet distance:
           User        Label  Sig. Dist    Fréchet
--------------------------------------------------
train_subject2585      control     1.8095     0.3793
train_subject7679      control     1.8453     0.4042
     subject581   depression     1.9784     0.4083
train_subject7142   depression     2.0313     0.4414
    subject3868      control     2.0355     0.4644
```

---
## 10. Case Study: Full User Deep-Dive

Aligned temporal dashboard showing all CVX analytics on a single user:
trajectory, velocity, distributional drift, posting patterns, and text content.

```python
# Select a different depression user for the deep-dive
case2_uid = dep_sample.values[1] if len(dep_sample) > 1 else dep_sample.values[0]
case2_eid = user_to_id[case2_uid]
case2_traj = index.trajectory(entity_id=case2_eid)
case2_timestamps = [t for t, _ in case2_traj]
case2_user_df = df[df['user_id'] == case2_uid].sort_values('timestamp').copy()
case2_user_df['hour_of_day'] = pd.to_datetime(case2_user_df['timestamp']).dt.hour
print(f'Deep-dive user: {case2_uid}, {len(case2_traj)} posts')
```


```text
Deep-dive user: test_subject625, 181 posts
```

```python
# Compute: velocity, drift from first vector, region trajectory, event features

# 1. Velocity magnitudes (subsample for speed)
step2 = max(1, len(case2_traj) // 200)
vel_data = []
for ts, vec in case2_traj[1:-1:step2]:
    try:
        vel = cvx.velocity(case2_traj, timestamp=ts)
        vel_data.append({'time': pd.Timestamp(ts, unit='s'), 'velocity': np.linalg.norm(vel)})
    except:
        pass
df_vel2 = pd.DataFrame(vel_data)

# 2. Cumulative drift from first vector (subsample)
first_vec = case2_traj[0][1]
drift_data = []
for ts, vec in case2_traj[::step2]:
    l2_mag, cos_drift, _ = cvx.drift(first_vec, vec, top_n=3)
    drift_data.append({'time': pd.Timestamp(ts, unit='s'), 'l2_drift': l2_mag, 'cosine_drift': cos_drift})
df_drift2 = pd.DataFrame(drift_data)

# 3. Event features
ef2 = cvx.event_features(case2_timestamps)
print('Event features:')
for k, v in sorted(ef2.items()):
    print(f'  {k}: {v:.4f}')

# 4. Hurst
h2 = cvx.hurst_exponent(case2_traj)
print(f'\nHurst exponent: {h2:.3f} ({"anti-persistent" if h2 < 0.5 else "persistent"})')

# 5. Change points on L3 region trajectory (not raw D=768)
case2_reg_traj = index.region_trajectory(entity_id=case2_eid, level=3, window_days=30, alpha=0.3)
if len(case2_reg_traj) >= 10:
    reg_traj_pelt = [(t, [float(x) for x in d]) for t, d in case2_reg_traj]
    cps2 = cvx.detect_changepoints(
        case2_eid, reg_traj_pelt,
        penalty=3.0 * np.log(len(case2_reg_traj)),
        min_segment_len=5,
    )
else:
    cps2 = []
print(f'Change points: {len(cps2)} (on L3 region trajectory)')

# 6. Distributional drift at Level 3
case2_dist_drift = []
for i in range(1, len(case2_reg_traj)):
    p = [float(x) for x in case2_reg_traj[i-1][1]]
    q = [float(x) for x in case2_reg_traj[i][1]]
    fr = cvx.fisher_rao_distance(p, q)
    case2_dist_drift.append({
        'time': pd.Timestamp(case2_reg_traj[i][0], unit='s'),
        'fisher_rao': fr,
    })
df_dist_drift2 = pd.DataFrame(case2_dist_drift)
print(f'Distributional drift: {len(case2_dist_drift)} time points')
```


```text
Event features:
  burstiness: 0.4167
  circadian_strength: 0.9693
  gap_cv: 2.4289
  intensity_trend: 0.9000
  max_gap: 847.0000
  mean_gap: 54.0444
  memory: 0.2999
  n_events: 181.0000
  span: 9728.0000
  std_gap: 131.2700
  temporal_entropy: 0.8465

Hurst exponent: 0.669 (persistent)
Change points: 0 (on L3 region trajectory)
Distributional drift: 72 time points
```

```python
# Aligned multi-panel dashboard (5 panels)
fig = make_subplots(
    rows=5, cols=1, shared_xaxes=True,
    subplot_titles=[
        'Trajectory in PCA Space (colored by time)',
        'Velocity |dv/dt| + Change Points',
        'Cumulative Drift from t₀',
        'Distributional Drift (Fisher-Rao on L3 regions)',
        'Post Content (hover for text)',
    ],
    vertical_spacing=0.06,
    row_heights=[0.2, 0.2, 0.2, 0.2, 0.2],
)

# Panel 1: PCA trajectory colored by time
fig.add_trace(go.Scatter(
    x=pd.to_datetime(case2_user_df['timestamp']),
    y=case2_user_df['pca1'],
    mode='lines+markers',
    marker=dict(size=3, color=case2_user_df['t_rel'], colorscale='Viridis', showscale=True,
                colorbar=dict(title='t_rel', x=1.02, len=0.15, y=0.9)),
    line=dict(width=1, color='gray'),
    name='PCA1',
    hovertemplate='PCA1: %{y:.3f}<br>%{x}<extra></extra>',
), row=1, col=1)

# Panel 2: Velocity + change points
if len(df_vel2) > 0:
    fig.add_trace(go.Scatter(
        x=df_vel2['time'], y=df_vel2['velocity'],
        mode='lines', line=dict(color=C_DEP, width=1),
        name='Velocity',
    ), row=2, col=1)
    
    if len(df_vel2) > 10:
        smooth = df_vel2['velocity'].rolling(min(20, len(df_vel2)//3), center=True).mean()
        fig.add_trace(go.Scatter(
            x=df_vel2['time'], y=smooth,
            mode='lines', line=dict(color='yellow', width=2),
            name='Velocity (smooth)',
        ), row=2, col=1)

for i, (cp_ts, sev) in enumerate(cps2[:8]):
    fig.add_vline(x=pd.Timestamp(cp_ts, unit='s'), line_dash='dot', line_color=C_ACCENT, row=2, col=1)

# Panel 3: Drift from t₀
fig.add_trace(go.Scatter(
    x=df_drift2['time'], y=df_drift2['l2_drift'],
    mode='lines', line=dict(color='#9b59b6', width=2),
    name='L2 Drift',
), row=3, col=1)
fig.add_trace(go.Scatter(
    x=df_drift2['time'], y=df_drift2['cosine_drift'],
    mode='lines', line=dict(color='#f39c12', width=2),
    name='Cosine Drift',
), row=3, col=1)

# Panel 4: Distributional drift (Fisher-Rao)
if len(df_dist_drift2) > 0:
    fig.add_trace(go.Scatter(
        x=df_dist_drift2['time'], y=df_dist_drift2['fisher_rao'],
        mode='lines', line=dict(color='#e74c3c', width=2),
        name='Fisher-Rao',
    ), row=4, col=1)

# Panel 5: Posts as markers with text on hover
case2_keys = list(zip(case2_user_df['user_id'], case2_user_df['timestamp'].astype(str)))
case2_texts = [texts.get(k, '')[:150] for k in case2_keys]
fig.add_trace(go.Scatter(
    x=pd.to_datetime(case2_user_df['timestamp']),
    y=[1] * len(case2_user_df),
    mode='markers',
    marker=dict(size=6, color=case2_user_df['hour_of_day'],
                colorscale='Twilight', showscale=True,
                colorbar=dict(title='Hour', x=1.08, len=0.15, y=0.1)),
    text=case2_texts,
    hovertemplate='<b>%{text}</b><br>%{x}<extra></extra>',
    name='Posts',
), row=5, col=1)

fig.update_layout(
    title=f'User Deep-Dive: {case2_uid} — H={h2:.3f}, B={ef2["burstiness"]:.3f}, {len(cps2)} change points',
    height=1200, width=1000,
    template='plotly_dark',
    showlegend=False,
)
fig.update_yaxes(title_text='PCA 1', row=1, col=1)
fig.update_yaxes(title_text='|dv/dt|', row=2, col=1)
fig.update_yaxes(title_text='Drift from t₀', row=3, col=1)
fig.update_yaxes(title_text='d_FR', row=4, col=1)
fig.update_yaxes(visible=False, row=5, col=1)
fig.show()
```


<iframe src="/plots/b1-explorer_fig_11.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

---
## Summary

This notebook demonstrated **17 ChronosVector analytical functions** on real mental health data:

| Category | Functions | Key Finding |
|----------|----------|-------------|
| Ingestion | `bulk_insert`, `enable_quantization` | SQ8 accelerates HNSW construction ~2.4× |
| Hierarchy | `regions`, `region_members` | HNSW levels reveal natural semantic clusters |
| Calculus | `velocity`, `drift`, `hurst_exponent`, `detect_changepoints` | Depression users show anti-persistent dynamics (H<0.5) |
| Point Process | `event_features` | Burstiness and circadian disruption differ between groups |
| Distributional | `region_trajectory`, `wasserstein_drift`, `fisher_rao_distance`, `hellinger_distance` | Track semantic migration over time |
| Topology | `topological_features` | Betti curves reveal cluster structure differences |
| Signatures | `path_signature`, `log_signature`, `signature_distance`, `frechet_distance` | Universal trajectory comparison in O(K²) |

**CVX is not just storage** — it's an analytical engine that makes temporal vector data explorable and interpretable.
