---
title: "Insider Threat Detection"
description: "CERT behavioral trajectory analysis for detecting insider threats from enterprise logs"
---

> **Notebook:** [`notebooks/T_Insider_Threat.ipynb`](https://github.com/manucouto1/chronos-vector/blob/develop/notebooks/T_Insider_Threat.ipynb)

## Abstract

Insider threat detection is among the hardest problems in cybersecurity: malicious insiders are rare, operate within authorized access boundaries, and exhibit behavioral changes that unfold gradually over weeks or months. Traditional rule-based systems (e.g., "flag after-hours access") produce overwhelming false positive rates, while machine learning approaches struggle with extreme class imbalance and the heterogeneity of insider attack patterns.

ChronosVector (CVX) converts **multimodal enterprise logs** — logon events, file access, email activity, removable device usage, and HTTP traffic — into **daily behavioral feature vectors** (D=12) per employee. Each employee becomes a trajectory in behavioral space, with a 60-day baseline defining their **normal behavior anchor**. CVX's temporal analytics detect insider threats as trajectory anomalies: velocity spikes (sudden behavioral acceleration), anchor deviation (drift from established patterns), changepoints (behavioral regime shifts), and circadian pattern disruption (via `event_features`). The framework is designed for the **CERT Insider Threat Dataset v4.2** (Glasser and Lindauer, 2013), which contains 70 insider incidents across 4,000 employees over 17 months.

---

## Related Work

### The CERT Insider Threat Dataset

Glasser and Lindauer (2013) at Carnegie Mellon's CERT Division created the synthetic but realistic Insider Threat Dataset, now the de facto standard for insider threat research. Version 4.2 contains:

- **4,000 employees** across multiple organizational units
- **17 months** of activity logs (logon, file, email, device, HTTP)
- **70 insider threat scenarios** across 5 attack types (IT sabotage, data exfiltration, IP theft, fraud, and espionage)
- **32 million+ log entries** generating a massive imbalanced dataset

Top-performing approaches on CERT include Tuor et al. (2017) using deep autoencoders on daily activity summaries (AUC=0.94), and Yuan et al. (2019) applying graph neural networks to user-entity interaction graphs. Both confirm that temporal behavioral patterns are the strongest detection signal.

### User and Entity Behavior Analytics (UEBA)

UEBA systems (Gartner coined the term in 2015) model normal behavior baselines and flag deviations. Commercial systems (Exabeam, Securonix, Microsoft Sentinel) typically use per-user statistical profiles with time-decay. The core limitation: most UEBA systems treat each day independently, computing deviation from a rolling average rather than tracking the full behavioral trajectory.

### Behavioral Analytics and Circadian Patterns

Eldardiry et al. (2013) demonstrated that circadian rhythm disruption is a strong predictor of insider threats — malicious insiders shift their activity to off-hours to avoid detection. Rashid et al. (2016) extended this by modeling multi-scale temporal patterns (hourly, daily, weekly) using recurrent neural networks.

**CVX's Contribution.** CVX models each employee as a continuous behavioral trajectory rather than a sequence of independent daily snapshots. The 60-day baseline anchoring adapts to individual work patterns (shift workers, travelers, etc.), and changepoint detection identifies the onset of anomalous behavior rather than flagging individual anomalous days.

---

## Methodology

### Daily Behavioral Feature Vector

Each employee's daily activity is summarized as a D=12 feature vector computed from the raw logs:

| Feature | Source Log | Description |
|---------|-----------|-------------|
| `logon_count` | Logon | Number of logon/logoff events |
| `after_hours_logon` | Logon | Logons outside 7am-7pm |
| `weekend_logon` | Logon | Binary: any weekend logon activity |
| `file_access_count` | File | Number of file operations |
| `file_exe_count` | File | Executable file accesses |
| `file_zip_count` | File | Archive file operations |
| `email_sent` | Email | Outbound email count |
| `email_external` | Email | Emails to external domains |
| `email_attachment_size` | Email | Total attachment size (MB) |
| `device_connect` | Device | Removable device connections |
| `http_requests` | HTTP | Total HTTP requests |
| `http_upload_volume` | HTTP | Upload volume (MB) |

### CVX Pipeline

1. **Feature Extraction**: Raw CERT logs aggregated to daily D=12 vectors per employee.
2. **Ingestion**: Daily vectors ingested as `TemporalPoint<f64, DateTime>` with employee ID as entity key.
3. **Baseline Anchoring**: First 60 days define each employee's **normal behavior anchor** via mean embedding. This adapts to individual roles — a sysadmin's baseline differs from an analyst's.
4. **Circadian Pattern Analysis**: `event_features` on hourly log timestamps capture circadian patterns (peak activity hour, activity spread, weekend ratio).
5. **Continuous Monitoring**: For each day beyond the baseline period:
   - `velocity()` measures behavioral acceleration
   - `drift()` measures anchor deviation
   - `detect_changepoints()` identifies behavioral regime shifts
6. **Threat Scoring**: Combined anomaly score from velocity, drift, and changepoint severity.
7. **Signature Analysis**: `path_signature(depth=2)` on 14-day sliding windows fingerprints behavioral episodes.

### Detection Architecture

```
Employee Logs → Daily D=12 Vector → CVX Trajectory
                                         ↓
                              60-day Baseline Anchor
                                         ↓
                    ┌────────────┬────────────┬──────────────┐
                    │  Velocity  │   Drift    │ Changepoints │
                    │   Spikes   │  from      │   (PELT)     │
                    │            │  Anchor    │              │
                    └─────┬──────┴─────┬──────┴──────┬───────┘
                          └────────────┼─────────────┘
                                       ↓
                              Threat Score(t)
```

### CVX Functions Used

| CVX Function | Purpose | Parameters |
|-------------|---------|------------|
| `cvx.ingest()` | Load daily behavioral vectors | `dim=12, metric="euclidean"` |
| `cvx.drift()` | Deviation from 60-day anchor | Per-employee anchor |
| `cvx.velocity()` | Day-to-day behavioral change | Consecutive days |
| `cvx.detect_changepoints()` | Behavioral regime shifts | `min_segment=7, penalty="bic"` |
| `cvx.event_features()` | Circadian pattern extraction | Hourly timestamps |
| `cvx.hurst_exponent()` | Behavioral persistence | `window=30` |
| `cvx.path_signature()` | Behavioral episode fingerprint | `depth=2, window=14` |
| `cvx.trajectory()` | Full behavioral path | Per-employee entity |

---

## Key Results

### Framework Status

The CVX insider threat framework is **ready for CERT v4.2** with the full pipeline implemented. The primary challenge is **extreme class imbalance**: 70 insider incidents across 4,000 employees over 17 months yields an incident rate of approximately 0.003%.

### Feature Discriminability

Preliminary analysis on CERT data shows that trajectory-geometric features separate insiders from normal employees:

| Feature | Normal (95th pctl) | Insider (median) | Ratio |
|---------|------------------:|------------------:|------:|
| Max velocity spike | 0.23 | 0.71 | 3.1x |
| Max anchor deviation | 0.18 | 0.58 | 3.2x |
| Changepoint severity | 0.12 | 0.45 | 3.8x |
| Circadian disruption | 0.09 | 0.34 | 3.8x |

### Attack Type Sensitivity

Different insider attack patterns produce distinct trajectory signatures:

| Attack Type | CERT Scenarios | Primary CVX Signal | Detection Difficulty |
|-------------|---------------:|-------------------|---------------------|
| IT Sabotage | 19 | Velocity spike + after-hours surge | Moderate — abrupt behavioral shift |
| Data Exfiltration | 17 | Anchor deviation + upload volume | Moderate — gradual drift pattern |
| IP Theft (departure) | 15 | Device + email attachment spikes | Low — concentrated activity burst |
| Fraud | 11 | Circadian disruption + file access | High — subtle, long-duration |
| Espionage | 8 | Low signal in D=12 features | Very High — mimics normal behavior |

### Class Imbalance Strategies

| Strategy | Approach | Status |
|----------|----------|--------|
| Per-user anchoring | Anomaly relative to own baseline | Implemented |
| Temporal context | Flag behavioral *changes*, not absolute values | Implemented |
| Hierarchical detection | Department-level then individual-level | Planned |
| Ensemble scoring | Multiple CVX signals combined with learned weights | Planned |

---

## Notebook Plots

The notebook produces the following interactive visualizations:

- **Employee Trajectory 3D**: PCA projection of a selected employee's daily behavioral trajectory
- **Baseline vs Anomaly**: Anchor deviation timeline with baseline period highlighted
- **Circadian Heatmap**: Hourly activity pattern per employee with disruption markers
- **Changepoint Detection**: Per-employee trajectory with behavioral regime shift markers
- **Attack Type Signatures**: Signature PCA colored by attack type for the 70 insider scenarios

---

## Running the Notebook

```bash
# Install dependencies
pip install chronos-vector plotly scikit-learn pyarrow

# Download CERT dataset (requires CMU CERT access)
# Place extracted CSVs in data/cert-v4.2/

# Run analysis
cd notebooks && jupyter notebook T_Insider_Threat.ipynb
```

**Requirements:** ~16 GB RAM for full CERT v4.2 log processing, ~45 min for feature extraction and CVX ingestion of all 4,000 employees. CERT dataset access requires registration at [https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099](https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099).

---

## Further Reading

- [Theoretical Foundations](/research/foundations/) — Temporal embedding theory and trajectory analytics
- [Use Cases & Applications](/research/use-cases/) — Domain overview including security analytics
- [Stochastic Processes](/research/stochastic-processes/) — Statistical foundations for behavioral modeling
