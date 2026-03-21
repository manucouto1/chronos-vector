---
title: "Political Discourse Analysis"
description: "Tracking rhetorical drift, polarization, and gendered language in parliamentary and social media data"
---

## Overview

CVX applies temporal trajectory analysis to political speech, tracking how rhetorical strategies evolve over time. This has been validated on real parliamentary data with quantitative results.

## ParlaMint-ES Analysis (B8)

### Data

- **ParlaMint-ES v5.0**: 32,541 speeches from the Spanish Parliament (2015-2023)
- **841 MPs** (355 female, 486 male), CC-BY 4.0 license
- TEI XML + plain text + TSV metadata with speaker gender, party, date

### Rhetorical Anchors

8 rhetorical dimensions defined as Spanish-language reference phrases:

| Anchor | Description |
|--------|------------|
| `ataque_personal` | Ad hominem attacks, personal accusations |
| `politica_social` | Social policy, welfare, public services |
| `economia` | Economic policy, budgets, fiscal matters |
| `emocional` | Emotional appeals, dramatic rhetoric |
| `institucional` | Institutional references, rule of law |
| `territorial` | Territorial debates, regional autonomy |
| `genero` | Gender equality, feminist policy |
| `seguridad` | Security, defense, public order |

### Key Results

- **Gender prediction from rhetoric**: F1=0.94, AUC=1.00 — rhetorical profiles strongly differ by gender
- **Party > Gender**: Within-party distance (1.084) < within-gender distance (1.231) — party affiliation drives rhetorical similarity more than gender
- **COVID impact on female MPs**: Counterfactual analysis shows female MPs' rhetorical trajectory diverged significantly from pre-COVID trends, with increased focus on `politica_social` and `emocional` anchors

## CVX Features Used

| Feature | Purpose |
|---------|---------|
| `project_to_anchors()` | Rhetorical proximity profiles per speaker |
| `anchor_summary()` | Mean position + trend per anchor dimension |
| `counterfactual_trajectory()` | What-if analysis (COVID divergence) |
| `cohort_drift()` | Gender/party group-level drift |
| `temporal_join()` | Convergence windows between political groups |
| `granger_causality()` | Cross-group rhetorical influence |
| `discover_motifs()` | Recurring rhetorical patterns |

## Notebooks & Reports

| Notebook | Focus | Status |
|----------|-------|--------|
| B8_parlamint_polarization | Real ParlaMint-ES data, gender/party analysis, COVID counterfactual | Complete |
| B7_political_polarization | Synthetic political polarization framework | Superseded by B8 |

## Interactive Results

- [Political Rhetoric Analysis (B3)](/chronos-vector/tutorials/trump-impact) — Full interactive tutorial with temporal analytics on political speech: rhetorical drift, partisan clustering, counterfactual analysis

## Related

- [RFC-007: Advanced Temporal Primitives](/chronos-vector/rfc/rfc-007) (temporal join, Granger, motifs)
- [Research: Political Rhetoric](/chronos-vector/research/trump-impact) — Methodology and theoretical background
