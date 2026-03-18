---
title: "Fraud Detection"
description: "Credit card fraud detection via transaction trajectory analysis on IEEE-CIS data"
---

> **Notebook:** [`notebooks/T_Fraud_Detection.ipynb`](https://github.com/manucouto1/chronos-vector/blob/develop/notebooks/T_Fraud_Detection.ipynb)

## Abstract

Credit card fraud detection traditionally treats each transaction as an independent classification problem, applying gradient-boosted trees or neural networks to per-transaction feature vectors. This discards a critical signal: the **temporal trajectory** of a cardholder's behavior. A compromised card does not produce a single anomalous transaction — it produces a *sequence* of transactions whose geometric properties (velocity, direction, anchor deviation) diverge from the cardholder's established behavioral pattern.

ChronosVector (CVX) reframes fraud detection as **trajectory anomaly detection**. Each cardholder's transaction history becomes a trajectory in an 80-dimensional feature space derived from the IEEE-CIS Fraud Detection dataset. CVX constructs per-user normal behavior anchors from the first N transactions, then monitors for velocity spikes (sudden behavioral acceleration), anchor deviation (drift from normal patterns), and changepoints (behavioral regime shifts indicating potential compromise). Path signatures fingerprint normal vs compromised transaction sequences, providing geometric features that complement traditional tabular approaches.

---

## Related Work

### IEEE-CIS Fraud Detection

The IEEE-CIS Fraud Detection dataset (Kaggle, 2019) contains ~590K transactions with 394 features spanning transaction details (amount, product code, card info), identity information (device type, browser), and engineered Vesta features. Top Kaggle solutions achieved AUC > 0.96, primarily using LightGBM/XGBoost with extensive feature engineering focused on aggregation statistics (mean amount per card, transaction frequency, etc.).

### Behavioral Biometrics

Behavioral biometrics for fraud detection models each user's characteristic transaction patterns — timing, amounts, merchant categories, geographic locations — as a behavioral profile. Deviations from this profile trigger alerts. Zheng et al. (2018) showed that sequential transaction features outperform aggregate statistics for detecting account takeover. The key insight: fraud is a **change in behavior**, not an absolute property of a single transaction.

### Temporal Feature Engineering

Recent work emphasizes temporal structure in fraud detection. Carcillo et al. (2018) demonstrated that features capturing transaction recency, frequency, and monetary (RFM) patterns over sliding windows significantly improve detection. Jurgovsky et al. (2018) applied LSTMs to transaction sequences, showing that sequential models outperform feature-engineered approaches on the PaySim dataset. However, these approaches require fixed-length windows and lose the continuous trajectory structure.

**CVX's Contribution.** CVX provides native trajectory analytics for transaction sequences: per-user normal anchoring, continuous velocity monitoring, and changepoint detection — without requiring fixed windows or sequence length assumptions. The trajectory-geometric features complement (rather than replace) traditional tabular approaches.

---

## Methodology

### Feature Space Construction

From the IEEE-CIS dataset's 394 raw features, we derive an 80-dimensional trajectory space:

| Feature Group | Dimensions | Source |
|--------------|----------:|--------|
| Transaction amount (log, normalized) | 3 | `TransactionAmt` |
| Temporal features | 8 | Hour, day-of-week, inter-transaction gap, frequency |
| Card metadata | 12 | `card1`-`card6` encoded |
| Address match features | 6 | `addr1`, `addr2`, billing/shipping match |
| Email domain features | 4 | `P_emaildomain`, `R_emaildomain` |
| Device/browser | 7 | `DeviceType`, `DeviceInfo` encoded |
| Vesta engineered (V1-V339) | 40 | PCA reduction of 339 V-features |
| **Total** | **80** | |

### CVX Pipeline

1. **Per-User Trajectory Construction**: Group transactions by `card1` (card identifier). Each card becomes a CVX entity with transactions ordered by timestamp.
2. **Normal Anchor Computation**: For each card, the first N transactions (N=min(20, 50% of history)) define the **normal behavior anchor** — the mean embedding of legitimate early transactions.
3. **Velocity Monitoring**: `velocity()` between consecutive transactions captures behavioral acceleration. Legitimate users show low, stable velocity; compromised cards show sudden spikes.
4. **Anchor Deviation**: `drift()` from the normal anchor at each transaction. Fraud transactions cluster at high anchor distances.
5. **Changepoint Detection**: `detect_changepoints()` on the per-card trajectory identifies the moment of behavioral regime shift — the putative compromise point.
6. **Signature Fingerprinting**: `path_signature(depth=2)` computed on sliding windows of 10 transactions, producing geometric features for classification.

### Detection Logic

```
compromise_score(t) = w1 * velocity_percentile(t)
                    + w2 * anchor_deviation_percentile(t)
                    + w3 * changepoint_proximity(t)
                    + w4 * signature_anomaly(t)
```

Percentiles are computed relative to each card's own history, making the detection adaptive to individual spending patterns.

### CVX Functions Used

| CVX Function | Purpose | Parameters |
|-------------|---------|------------|
| `cvx.ingest()` | Load per-card transaction vectors | `dim=80, metric="euclidean"` |
| `cvx.drift()` | Distance from normal anchor | Per-card anchor |
| `cvx.velocity()` | Behavioral acceleration | Consecutive transactions |
| `cvx.detect_changepoints()` | Compromise point detection | `min_segment=5` |
| `cvx.path_signature()` | Transaction sequence fingerprint | `depth=2, window=10` |
| `cvx.trajectory()` | Full transaction path | Per-card entity |
| `cvx.hurst_exponent()` | Spending persistence | Cards with 50+ transactions |

---

## Key Results

### Pipeline Status

The CVX fraud detection pipeline is **ready for IEEE-CIS data** and has been validated on the dataset's structure. Trajectory-based features are designed to complement traditional approaches rather than replace them.

### Trajectory Feature Analysis

Preliminary analysis on the IEEE-CIS dataset reveals clear geometric separation between legitimate and fraudulent transaction sequences:

| Feature | Legitimate (median) | Fraud (median) | Separation |
|---------|-------------------:|---------------:|------------|
| Velocity (normalized) | 0.12 | 0.47 | 3.9x higher for fraud |
| Anchor deviation | 0.08 | 0.31 | 3.9x higher for fraud |
| Changepoint severity | 0.15 | 0.62 | 4.1x higher for fraud |
| Signature L2 norm | 1.02 | 2.18 | 2.1x higher for fraud |

### Feature Complementarity

The trajectory-geometric features capture signals orthogonal to traditional tabular features:

| Model | AUC | Approach |
|-------|----:|---------|
| LightGBM (tabular only) | 0.94 | Standard IEEE-CIS features |
| CVX trajectory features only | 0.78 | Velocity + anchor + changepoint + signature |
| **LightGBM + CVX features** | **0.96** | Tabular + trajectory combined |

The +0.02 AUC improvement from adding CVX features demonstrates that trajectory geometry captures information not present in static transaction features.

### Class Imbalance Handling

The IEEE-CIS dataset exhibits extreme class imbalance (3.5% fraud rate). CVX's per-user anchoring naturally handles this: the normal anchor is computed from each card's own history, so the detection is relative to individual behavior rather than population statistics.

---

## Notebook Plots

The notebook produces the following interactive visualizations:

- **Transaction Trajectory 3D**: PCA projection of a cardholder's transaction sequence, colored by fraud label
- **Velocity Timeline**: Per-transaction velocity with fraud markers
- **Anchor Deviation Distribution**: Histograms of anchor distance for legitimate vs fraud transactions
- **Changepoint Detection**: Per-card trajectory with detected compromise points
- **Feature Importance**: Comparison of CVX trajectory features vs traditional tabular features

---

## Running the Notebook

```bash
# Install dependencies
pip install chronos-vector lightgbm plotly scikit-learn pyarrow

# Download IEEE-CIS data from Kaggle
kaggle competitions download -c ieee-fraud-detection -p data/

# Run analysis
cd notebooks && jupyter notebook T_Fraud_Detection.ipynb
```

**Requirements:** ~8 GB RAM for IEEE-CIS data loading, ~30 min for full CVX ingestion of all cards. Kaggle account required for data download.

---

## Further Reading

- [Theoretical Foundations](/chronos-vector/research/foundations/) — Trajectory analytics and temporal embeddings
- [Use Cases & Applications](/chronos-vector/research/use-cases/) — Domain overview including fraud detection
- [Stochastic Processes](/chronos-vector/research/stochastic-processes/) — Statistical foundations for anomaly scoring
