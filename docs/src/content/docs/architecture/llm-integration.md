---
title: "LLM Integration Guide"
description: "Using ChronosVector as a temporal reasoning tool for LLMs via MCP"
---

ChronosVector can serve as a **temporal reasoning backend** for Large Language Models. This guide covers the MCP server, available tools, and agentic workflow patterns.

## Architecture

```
┌─────────────────┐       stdio (JSON-RPC)       ┌─────────────────┐
│  Claude Code /  │ ◄──────────────────────────► │   cvx-mcp       │
│  Claude Desktop │       tools/resources         │   (Rust binary)  │
│  Any MCP client │                               │                  │
└─────────────────┘                               │  ┌────────────┐ │
                                                  │  │TemporalHnsw│ │
                                                  │  │ (in-memory) │ │
                                                  │  └────────────┘ │
                                                  └─────────────────┘
```

## Available Tools

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `cvx_search` | Temporal RAG — find similar content in time windows | `vector`, `k`, `time_start`, `time_end`, `alpha` |
| `cvx_entity_summary` | Comprehensive temporal overview | `entity_id` |
| `cvx_drift_report` | Quantify semantic change between two time points | `entity_id`, `t1`, `t2` |
| `cvx_detect_anomalies` | Scan entities for change points | `entity_ids`, `lookback_us` |
| `cvx_compare_entities` | Cross-entity temporal analysis | `entity_a`, `entity_b`, `epsilon` |
| `cvx_cohort_analysis` | Group-level drift and convergence | `entity_ids`, `t1`, `t2` |
| `cvx_forecast` | Trajectory prediction | `entity_id`, `target_timestamp` |
| `cvx_ingest` | Add new temporal data | `points` |

## Design Principle: Structured Narratives

All tools return JSON with a `summary` field containing a human-readable interpretation. LLMs receive interpretable metrics, not raw 768-dimensional vectors:

```json
{
  "drift": {
    "l2_magnitude": 0.87,
    "interpretation": "Significant semantic shift"
  },
  "change_points": [
    {"timestamp": 1718409600, "severity": 0.82}
  ],
  "summary": "Entity 42 shows significant drift (L2=0.87) with 1 change point detected."
}
```

## Agentic Workflow Patterns

### Pattern 1: Longitudinal Monitor

**Trigger**: Periodic (daily/weekly)

```
1. cvx_detect_anomalies(entity_ids=[...], lookback_us=7d)
2. For each anomaly with severity > threshold:
   a. cvx_entity_summary(entity_id)
   b. cvx_drift_report(entity_id, t1=anomaly-30d, t2=anomaly)
3. Generate report with findings
```

**Use case**: Clinical monitoring, security surveillance, portfolio risk alerts.

### Pattern 2: Comparative Investigation

**Trigger**: User asks "how do A and B compare?"

```
1. cvx_entity_summary(A)  ║  cvx_entity_summary(B)   [parallel]
2. cvx_compare_entities(A, B)
3. If convergence found:
   a. "During [window], both entities were semantically close"
4. If Granger found:
   a. "A's changes precede B's by N days"
```

**Use case**: Influence analysis, correlation investigation.

### Pattern 3: Cohort Treatment Evaluation

**Trigger**: User asks "did the intervention work?"

```
1. cvx_cohort_analysis(treatment_group, t1=pre, t2=post)
2. If compare_with provided:
   a. cvx_cohort_analysis(control_group, t1=pre, t2=post)
   b. Compare effect sizes
3. For outliers:
   a. cvx_entity_summary(outlier_id)
   b. Classify as responder vs non-responder
4. cvx_forecast(cohort_centroid, horizon=+90d)
```

**Use case**: Treatment evaluation, A/B testing, policy impact.

### Pattern 4: Temporal RAG Chain

**Trigger**: User asks about change over time

```
1. cvx_search(query, time_range=recent) → current matches
2. cvx_search(query, time_range=historical) → past matches
3. For entities that appear in both:
   a. cvx_drift_report(entity_id, t1=historical, t2=recent)
4. Synthesize: "The discourse around [topic] has shifted.
   Entity X drifted toward [direction] (L2=0.6)."
```

**Use case**: Trend analysis, discourse evolution, narrative tracking.

## REST API Endpoints

For applications that don't use MCP, the same capabilities are available via REST:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/entities/{id}/summary` | GET | Entity temporal summary |
| `/v1/anomalies/scan` | POST | Batch anomaly detection |
| `/v1/cohort/drift` | POST | Cohort drift analysis |
| `/v1/temporal-join` | POST | Convergence windows |
| `/v1/granger` | POST | Granger causality |
| `/v1/entities/{id}/motifs` | GET | Recurring patterns |
| `/v1/entities/{id}/discords` | GET | Anomalous subsequences |
| `/v1/entities/{id}/counterfactual` | GET | What-if analysis |

## Inline Embeddings

The `Embedder` trait (`cvx-core`) allows tools to accept text instead of vectors. Current backends:

| Backend | Status | Use case |
|---------|--------|----------|
| `MockEmbedder` | Available | Testing and development |
| `OnnxEmbedder` | Stub | Local inference (planned) |
| `ApiEmbedder` | Stub | Remote API embedding (planned) |
