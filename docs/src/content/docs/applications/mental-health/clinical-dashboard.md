---
title: "Clinical Dashboard (B3)"
description: "DSM-5 symptom profiling, HNSW exploration, clinical timelines, and semantic polarization on eRisk data"
---

This notebook applies CVX's temporal analytics to real eRisk depression data (1.36M Reddit posts, 2,285 users) using **centered DSM-5 anchor projections** — the key technique that transforms useless raw cosine distances into clinically discriminative symptom profiles.

**Dataset**: eRisk 2017+2018+2022 (233 depression + 233 control users, 225K posts, MentalRoBERTa D=768)

**CVX Features**: `project_to_anchors`, `anchor_summary`, `compute_centroid`, `regions`, `region_assignments`, `velocity`, `detect_changepoints`, `drift`

---

## 1. DSM-5 Symptom Proximity — Population Average

After centering (subtracting the global mean embedding), we project each user's trajectory onto 9 DSM-5 symptom anchors + 1 healthy baseline. The radar chart shows the **mean proximity** of depression vs control users across all symptoms.

<iframe src="/chronos-vector/plots/b3-clinical_fig_0.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

Depression users show significantly higher proximity to all symptom anchors — particularly `depressed_mood`, `worthlessness`, and `anhedonia`. Control users cluster near the healthy baseline.

---

## 2. Symptom Drift Direction — Who Is Approaching Symptoms?

Beyond static proximity, we measure the **trend** (linear slope over normalized time) of each user's symptom distances. Negative trend = approaching the symptom over time.

<iframe src="/chronos-vector/plots/b3-clinical_fig_1.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

This reveals which symptoms show **active deterioration** vs static elevation — critical for early intervention.

---

## 3. Symptom Evolution Over Normalized Time

Small multiples showing how proximity to each DSM-5 symptom evolves from beginning (0%) to end (100%) of each user's post history. Red = depression, blue = control.

<iframe src="/chronos-vector/plots/b3-clinical_fig_2.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

The separation between groups is visible across all symptoms, with the most dramatic divergence in `depressed_mood` and `worthlessness`.

---

## 4. HNSW Semantic Regions Explorer

The HNSW hierarchy provides unsupervised semantic clustering. Each bubble is a region hub — hover to see the depression ratio, member count, and clinical keywords of posts assigned to that region.

<iframe src="/chronos-vector/plots/b3-clinical_fig_3.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

Regions naturally specialize: some have depression ratios >80% (posts about hopelessness, isolation), others <20% (social activities, hobbies). No labels were used during construction.

---

## 5. Clinical Timeline — Depression User

A 4-panel aligned view for a single depression user showing how symptoms, velocity, change points, and text content evolve together:

<iframe src="/chronos-vector/plots/b3-clinical_fig_4.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

## Clinical Timeline — Control User (comparison)

<iframe src="/chronos-vector/plots/b3-clinical_fig_5.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

The control user shows stable symptom distances and lower velocity throughout — no approaching behavior, no change points.

---

## 6. Semantic Polarization

Polarization measures whether a user's semantic space is **contracting** (becoming obsessively focused) or **expanding** (maintaining diverse topics).

Dispersion ratio < 1.0 = semantic world is shrinking. Computed as `std(second_half) / std(first_half)` of the trajectory vectors.

<iframe src="/chronos-vector/plots/b3-clinical_fig_6.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

<iframe src="/chronos-vector/plots/b3-clinical_fig_7.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

---

## 7. Cohort Divergence in Symptom Space

Heatmap showing `depression_mean - control_mean` for each symptom at each time point. Red = depression users are closer to the symptom than controls.

<iframe src="/chronos-vector/plots/b3-clinical_fig_8.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

The top 4 most discriminative symptoms over time:

<iframe src="/chronos-vector/plots/b3-clinical_fig_9.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

---

## 8. Individual Clinical Profiles

Combined view for a single user: symptom radar + drift + polarization + timeline.

### Depression User Profile

<iframe src="/chronos-vector/plots/b3-clinical_fig_10.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

<iframe src="/chronos-vector/plots/b3-clinical_fig_11.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

<iframe src="/chronos-vector/plots/b3-clinical_fig_12.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

### Control User Profile

<iframe src="/chronos-vector/plots/b3-clinical_fig_13.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

<iframe src="/chronos-vector/plots/b3-clinical_fig_14.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>

<iframe src="/chronos-vector/plots/b3-clinical_fig_15.html" width="100%" height="620" style="border:none; border-radius:8px; margin:1rem 0;"></iframe>
