# RFC-009: CVX as LLM Tool — MCP Server & Agentic Integration

**Status**: Proposed
**Created**: 2026-03-18
**Authors**: Manuel Couto Pintos
**Related**: RFC-005 (Query Capabilities), RFC-007 (Temporal Primitives), RFC-006 (Anchor Projection)

---

## Summary

This RFC proposes transforming CVX from a database with a REST API into a **first-class tool for Large Language Models**. The core insight: LLMs can reason about time, entities, and change — but they cannot *compute* temporal analytics. CVX provides the computational substrate that turns an LLM from a static knowledge system into a **temporal reasoning engine**.

We propose three integration layers:

| Layer | What | Who uses it |
|-------|------|-------------|
| **MCP Server** | Model Context Protocol adapter (JSON-RPC over stdio) | Claude Code, Claude Desktop, any MCP-compatible agent |
| **LLM-Optimized API** | REST endpoints that return structured, interpretable summaries (not raw vectors) | Any LLM via function calling / tool use |
| **Agentic Workflows** | Pre-built multi-step reasoning patterns that combine CVX tools | Autonomous agents monitoring entities over time |

---

## Motivation

### The Temporal Reasoning Gap

LLMs excel at:
- Understanding natural language descriptions of change ("the user seems more negative lately")
- Reasoning about causality ("if X happened before Y, X might have caused Y")
- Generating hypotheses ("the language shift could indicate...")

LLMs cannot:
- Compute whether an entity actually drifted (requires vector arithmetic)
- Detect change points in high-dimensional trajectories (requires PELT/BOCPD)
- Quantify drift magnitude or direction (requires calculus on embeddings)
- Compare trajectories across entities (requires temporal alignment + distance)

**CVX fills exactly this gap.** It computes what the LLM reasons about.

### Why MCP (Model Context Protocol)?

MCP is the emerging standard for LLM tool integration:
- **Anthropic-native**: Claude Code and Claude Desktop support MCP natively
- **Stdio transport**: no HTTP overhead, direct process communication
- **Tool discovery**: LLM sees available tools with schemas at conversation start
- **Streaming**: results can stream back as they're computed

A CVX MCP server means Claude can do temporal analytics directly from the CLI:

```
User: "How has user 42's language changed in the last 3 months?"
Claude: [calls cvx_drift_report(entity_id=42, t1=..., t2=...)]
Claude: "User 42 shows significant semantic drift (L2=0.87, cosine=0.34).
         The top changing dimensions align with the 'social withdrawal'
         anchor, suggesting increased isolation language. Two change points
         detected: one at Feb 3 (severity 0.7) and one at Mar 10 (severity 0.9)."
```

No code, no notebooks, no API calls — just a conversation.

### Why Not Just RAG?

Standard RAG (Retrieval-Augmented Generation) answers "what documents are similar to this query?" CVX enables **Temporal RAG** — a fundamentally richer interaction:

| Standard RAG | Temporal RAG (CVX) |
|--------------|--------------------|
| "Find similar documents" | "Find similar documents *from 3 months ago*" |
| Static snapshot | "How has the discourse *changed* since then?" |
| No entity tracking | "Track this user's semantic evolution" |
| No analytics | "Detect when the conversation shifted" |
| No prediction | "Where is this entity heading?" |

---

## Proposed Changes

### 1. MCP Server (Priority: P0)

#### Architecture

```
┌─────────────────┐       stdio (JSON-RPC)       ┌─────────────────┐
│  Claude Code /  │ ◄──────────────────────────► │   cvx-mcp       │
│  Claude Desktop │       tools/resources         │   (Rust binary)  │
│  Any MCP client │                               │                  │
└─────────────────┘                               │  ┌────────────┐ │
                                                  │  │ TemporalHnsw│ │
                                                  │  │ (in-memory) │ │
                                                  │  └────────────┘ │
                                                  │        or       │
                                                  │  ┌────────────┐ │
                                                  │  │ HTTP client │ │
                                                  │  │ → cvx-api   │ │
                                                  │  └────────────┘ │
                                                  └─────────────────┘
```

Two deployment modes:
1. **Embedded**: cvx-mcp loads the index directly (for local analysis)
2. **Proxy**: cvx-mcp forwards to a running cvx-api server (for shared deployments)

#### Implementation

New crate: `cvx-mcp`

```rust
// cvx-mcp/src/main.rs
use cvx_core::traits::TemporalIndexAccess;
use mcp_server::{McpServer, Tool, ToolResult};

#[tokio::main]
async fn main() {
    let index = load_index_or_connect();

    let server = McpServer::new("cvx", "ChronosVector temporal analytics")
        .tool(search_tool())
        .tool(trajectory_tool())
        .tool(drift_report_tool())
        .tool(change_points_tool())
        .tool(compare_entities_tool())
        .tool(cohort_analysis_tool())
        .tool(forecast_tool())
        .tool(anomaly_scan_tool())
        .resource(entity_list_resource())
        .resource(index_stats_resource());

    server.serve_stdio().await;
}
```

#### MCP Configuration

Users add to their Claude Code config (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "cvx": {
      "command": "cvx-mcp",
      "args": ["--index-path", "/path/to/index.cvx"],
      "env": {
        "CVX_LOG_LEVEL": "info"
      }
    }
  }
}
```

Or for proxy mode:
```json
{
  "mcpServers": {
    "cvx": {
      "command": "cvx-mcp",
      "args": ["--server", "http://localhost:3000"]
    }
  }
}
```

---

### 2. LLM-Optimized Tool Definitions (Priority: P0)

The key design principle: **tools return structured narratives, not raw data**. An LLM receiving a 768-dim vector gains nothing. An LLM receiving "drift magnitude: 0.87, direction: toward 'isolation' anchor, change points: [Feb 3, Mar 10]" can reason.

#### Tool 1: `cvx_search` — Temporal RAG

**Purpose**: Find semantically similar content within a time window.

```json
{
  "name": "cvx_search",
  "description": "Search for semantically similar content in the temporal vector database. Use this when the user asks about finding similar content, related entities, or wants to explore what exists in a time period. Returns the most relevant matches with their timestamps and similarity scores.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Natural language query OR pre-computed vector. If text, will be embedded using the configured model."
      },
      "time_range": {
        "type": "object",
        "properties": {
          "start": {"type": "string", "format": "date-time"},
          "end": {"type": "string", "format": "date-time"}
        },
        "description": "Optional time window to restrict search. Omit for all-time search."
      },
      "k": {
        "type": "integer",
        "default": 10,
        "description": "Number of results to return."
      },
      "alpha": {
        "type": "number",
        "default": 0.8,
        "description": "Balance between semantic (1.0) and temporal (0.0) relevance. Default 0.8 prioritizes semantic match with temporal recency boost."
      },
      "entity_filter": {
        "type": "integer",
        "description": "Optional: restrict search to a specific entity's history."
      }
    },
    "required": ["query"]
  }
}
```

**Response format** (LLM-optimized):
```json
{
  "matches": [
    {
      "entity_id": 42,
      "timestamp": "2025-06-15T14:30:00Z",
      "score": 0.92,
      "semantic_score": 0.95,
      "temporal_score": 0.85,
      "metadata": {"source": "reddit", "subreddit": "depression"}
    }
  ],
  "summary": "Found 10 matches spanning Jun 2025 to Mar 2026. Highest concentration in the 'mental health' semantic region. 7/10 matches are from entity 42."
}
```

#### Tool 2: `cvx_entity_summary` — Entity Overview

**Purpose**: Get a high-level temporal overview of an entity.

```json
{
  "name": "cvx_entity_summary",
  "description": "Get a comprehensive temporal summary of an entity: when it was active, how its semantic representation changed over time, and any notable events (change points, anomalies). Use this as a starting point when investigating an entity.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "entity_id": {"type": "integer"},
      "time_range": {
        "type": "object",
        "properties": {
          "start": {"type": "string", "format": "date-time"},
          "end": {"type": "string", "format": "date-time"}
        }
      }
    },
    "required": ["entity_id"]
  }
}
```

**Response format**:
```json
{
  "entity_id": 42,
  "first_seen": "2025-01-15T00:00:00Z",
  "last_seen": "2026-03-10T00:00:00Z",
  "total_points": 847,
  "active_days": 420,
  "posting_frequency": "2.0 points/day",
  "overall_drift": {
    "l2_magnitude": 0.87,
    "cosine_drift": 0.34,
    "interpretation": "Significant semantic shift over observation period"
  },
  "change_points": [
    {"timestamp": "2025-06-15T00:00:00Z", "severity": 0.82, "description": "Sharp shift detected"},
    {"timestamp": "2025-11-03T00:00:00Z", "severity": 0.65, "description": "Moderate shift detected"}
  ],
  "current_region": {
    "region_id": 7,
    "label": "Region 7 (hub at level 3)",
    "time_in_region": "45 days"
  },
  "trajectory_character": {
    "hurst_exponent": 0.72,
    "interpretation": "Trending (persistent drift, not mean-reverting)",
    "volatility": "moderate"
  }
}
```

#### Tool 3: `cvx_drift_report` — Quantify Change

**Purpose**: Measure how much an entity's semantic representation has changed between two points in time.

```json
{
  "name": "cvx_drift_report",
  "description": "Measure how much an entity's semantic representation has changed between two time points. Returns drift magnitude, direction, and which dimensions changed most. Use this to quantify observed or suspected changes in entity behavior.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "entity_id": {"type": "integer"},
      "t1": {"type": "string", "format": "date-time"},
      "t2": {"type": "string", "format": "date-time"},
      "anchors": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Optional: anchor labels to project drift onto for interpretability (e.g., ['depression', 'anxiety', 'recovery']). If provided, drift is reported relative to these reference points."
      }
    },
    "required": ["entity_id", "t1", "t2"]
  }
}
```

**Response format**:
```json
{
  "entity_id": 42,
  "period": {"from": "2025-06-01", "to": "2025-12-01"},
  "drift": {
    "l2_magnitude": 0.87,
    "cosine_drift": 0.34,
    "percentile": 92,
    "interpretation": "This entity drifted more than 92% of all entities in the index over the same period"
  },
  "anchor_projection": {
    "depression": {"distance_change": -0.15, "direction": "closer"},
    "anxiety": {"distance_change": -0.08, "direction": "closer"},
    "recovery": {"distance_change": +0.22, "direction": "farther"}
  },
  "change_points_in_period": [
    {"timestamp": "2025-08-20T00:00:00Z", "severity": 0.75}
  ],
  "velocity_at_t2": {
    "magnitude": 0.012,
    "interpretation": "Drift is ongoing (non-zero velocity at end of period)"
  }
}
```

#### Tool 4: `cvx_detect_anomalies` — Proactive Monitoring

**Purpose**: Scan entities for unusual changes. Designed for autonomous agent monitoring.

```json
{
  "name": "cvx_detect_anomalies",
  "description": "Scan one or more entities for anomalous semantic changes. Returns entities with detected change points or unusual drift in the specified lookback window. Use this for proactive monitoring — checking if anything noteworthy has happened recently.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "entity_ids": {
        "type": "array",
        "items": {"type": "integer"},
        "description": "Entities to scan. If omitted, scans all entities in the index."
      },
      "lookback_days": {
        "type": "integer",
        "default": 7
      },
      "sensitivity": {
        "type": "number",
        "default": 0.5,
        "description": "0.0 = only extreme anomalies, 1.0 = flag everything unusual"
      }
    }
  }
}
```

**Response format**:
```json
{
  "scan_period": {"from": "2026-03-11", "to": "2026-03-18"},
  "entities_scanned": 1500,
  "anomalies": [
    {
      "entity_id": 42,
      "type": "change_point",
      "timestamp": "2026-03-15T00:00:00Z",
      "severity": 0.88,
      "description": "Sharp semantic shift detected. Entity moved significantly toward 'isolation' region."
    },
    {
      "entity_id": 99,
      "type": "velocity_spike",
      "timestamp": "2026-03-17T00:00:00Z",
      "severity": 0.65,
      "description": "Abnormally high rate of semantic change (3.2σ above entity's baseline)."
    }
  ],
  "summary": "2 of 1500 entities flagged. Entity 42 shows the most concerning pattern (severity 0.88)."
}
```

#### Tool 5: `cvx_compare_entities` — Cross-Entity Analysis

**Purpose**: Compare two or more entities' temporal behavior.

```json
{
  "name": "cvx_compare_entities",
  "description": "Compare the semantic trajectories of two or more entities. Reveals convergence, divergence, correlation, and potential causal relationships. Use when the user asks about relationships between entities over time.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "entity_ids": {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 2
      },
      "time_range": {
        "type": "object",
        "properties": {
          "start": {"type": "string", "format": "date-time"},
          "end": {"type": "string", "format": "date-time"}
        }
      },
      "analyses": {
        "type": "array",
        "items": {
          "type": "string",
          "enum": ["convergence_windows", "correlation", "granger_causality", "trajectory_similarity"]
        },
        "default": ["convergence_windows", "correlation"],
        "description": "Which analyses to run. More analyses = slower but richer results."
      }
    },
    "required": ["entity_ids"]
  }
}
```

**Response format**:
```json
{
  "entities": [42, 99],
  "period": {"from": "2025-01-01", "to": "2026-03-18"},
  "convergence_windows": [
    {"start": "2025-06-01", "end": "2025-07-15", "mean_distance": 0.12, "description": "Entities were semantically close for 6 weeks"},
    {"start": "2025-11-20", "end": "2025-12-10", "mean_distance": 0.18, "description": "Brief convergence period"}
  ],
  "correlation": {
    "trajectory_correlation": 0.62,
    "interpretation": "Moderate positive correlation — entities tend to move in similar directions"
  },
  "granger_causality": {
    "direction": "42 → 99",
    "optimal_lag_days": 3,
    "p_value": 0.02,
    "interpretation": "Entity 42's movements significantly predict entity 99's movements with a 3-day lag"
  },
  "summary": "Entities 42 and 99 show moderate semantic correlation with two convergence periods. Granger analysis suggests entity 42 leads entity 99 by ~3 days (p=0.02)."
}
```

#### Tool 6: `cvx_cohort_analysis` — Group Analysis

**Purpose**: Analyze how a group of entities behaves collectively.

```json
{
  "name": "cvx_cohort_analysis",
  "description": "Analyze the collective behavior of a group of entities. Measures whether the group is converging, diverging, and how the group centroid has moved. Use for population-level analysis: treatment groups, user segments, market sectors.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "entity_ids": {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 2
      },
      "t1": {"type": "string", "format": "date-time"},
      "t2": {"type": "string", "format": "date-time"},
      "compare_with": {
        "type": "array",
        "items": {"type": "integer"},
        "description": "Optional: a second cohort to compare against (e.g., control group)"
      }
    },
    "required": ["entity_ids", "t1", "t2"]
  }
}
```

**Response format**:
```json
{
  "cohort_size": 50,
  "period": {"from": "2025-06-01", "to": "2025-12-01"},
  "centroid_drift": {
    "l2_magnitude": 0.45,
    "direction": "Cohort centroid moved toward 'isolation' semantic region"
  },
  "dispersion_change": {
    "t1": 0.82,
    "t2": 0.65,
    "change": -0.17,
    "interpretation": "Cohort is converging (becoming more semantically homogeneous)"
  },
  "convergence_score": 0.71,
  "interpretation": "71% of entities are moving in a similar direction",
  "outliers": [
    {"entity_id": 17, "drift": 1.95, "z_score": 2.8, "note": "Drifting in opposite direction to cohort"},
    {"entity_id": 42, "drift": 0.02, "z_score": -2.1, "note": "Almost no change while cohort shifted"}
  ],
  "comparison_with_control": {
    "treatment_drift": 0.45,
    "control_drift": 0.12,
    "effect_size": 0.33,
    "interpretation": "Treatment cohort drifted 3.7× more than control group"
  }
}
```

#### Tool 7: `cvx_forecast` — Trajectory Prediction

**Purpose**: Predict where an entity is heading.

```json
{
  "name": "cvx_forecast",
  "description": "Predict the future semantic trajectory of an entity. Uses the entity's historical movement pattern to extrapolate. Returns predicted position and confidence. Use when the user asks 'where is this entity heading?' or 'what will happen next?'",
  "inputSchema": {
    "type": "object",
    "properties": {
      "entity_id": {"type": "integer"},
      "horizon": {
        "type": "string",
        "format": "date-time",
        "description": "Target future timestamp to predict."
      },
      "method": {
        "type": "string",
        "enum": ["linear", "neural_ode"],
        "default": "linear"
      },
      "anchors": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Optional: anchor labels to interpret the predicted position."
      }
    },
    "required": ["entity_id", "horizon"]
  }
}
```

#### Tool 8: `cvx_ingest` — Add Data

**Purpose**: Insert new temporal points into the index.

```json
{
  "name": "cvx_ingest",
  "description": "Insert new temporal data points into the CVX index. Each point has an entity ID, timestamp, and either raw text (will be embedded) or a pre-computed vector. Use this when the user wants to add new data to the index.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "points": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "entity_id": {"type": "integer"},
            "timestamp": {"type": "string", "format": "date-time"},
            "text": {"type": "string"},
            "vector": {"type": "array", "items": {"type": "number"}},
            "metadata": {"type": "object"}
          },
          "required": ["entity_id", "timestamp"]
        }
      }
    },
    "required": ["points"]
  }
}
```

---

### 3. Inline Embedding Layer (Priority: P1)

#### Problem

Current CVX requires pre-computed vectors. For LLM integration, the tool should accept **text** and embed it internally. Otherwise, the LLM would need to:
1. Call an embedding API
2. Get a 768-dim vector
3. Pass it to CVX

This is 3 tool calls instead of 1, and the 768-dim vector wastes context tokens.

#### Solution

Add an optional embedding layer to `cvx-mcp` and `cvx-api`:

```rust
pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError>;
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError>;
    fn dimension(&self) -> usize;
}
```

**Implementations:**

| Backend | Use case | Dependency |
|---------|----------|------------|
| `OnnxEmbedder` | Local inference (e.g., all-MiniLM-L6) | `ort` crate |
| `ApiEmbedder` | Remote API (OpenAI, Cohere, etc.) | HTTP client |
| `TorchScriptEmbedder` | Custom models via tch-rs | `tch` crate |

**Configuration:**
```json
{
  "embedding": {
    "backend": "onnx",
    "model_path": "/path/to/model.onnx",
    "dimension": 384
  }
}
```

When a tool receives `text` instead of `vector`, it passes through the embedder first. This is transparent to the LLM — it just sends text and gets results.

---

### 4. Agentic Workflow Patterns (Priority: P2)

Beyond individual tools, we define **composite workflows** that agents can execute as multi-step reasoning patterns.

#### Pattern 1: Longitudinal Monitor

```
Trigger: Periodic (daily/weekly)
Steps:
  1. cvx_detect_anomalies(lookback_days=7, sensitivity=0.5)
  2. For each anomaly:
     a. cvx_entity_summary(entity_id)
     b. cvx_drift_report(entity_id, t1=anomaly-30d, t2=anomaly)
     c. cvx_search(query=entity_vector_at_anomaly, time_range=pre-anomaly)
  3. Generate report: "Entity X showed unusual change. Before the shift,
     they were similar to [these entities]. The drift is toward [anchor]."
```

#### Pattern 2: Comparative Investigation

```
Trigger: User asks "how do entities A and B compare?"
Steps:
  1. cvx_entity_summary(A), cvx_entity_summary(B)  // parallel
  2. cvx_compare_entities([A, B], analyses=["convergence_windows", "granger_causality"])
  3. If convergence found:
     a. cvx_search(query=A_vector_at_convergence, time_range=convergence_window)
     b. "During convergence, both entities were near [these other entities]"
  4. If granger found:
     a. "A's changes precede B's by N days — possible influence"
```

#### Pattern 3: Cohort Treatment Evaluation

```
Trigger: User asks "did the intervention work?"
Steps:
  1. cvx_cohort_analysis(treatment_group, t1=pre_intervention, t2=post, compare_with=control_group)
  2. For outliers in treatment group:
     a. cvx_entity_summary(outlier_id)
     b. Classify as "responder" vs "non-responder"
  3. cvx_forecast(treatment_centroid, horizon=+90d)
  4. Report: "Treatment cohort drifted X toward [anchor]. Control: Y.
     Effect size: Z. 3 non-responders identified. Projected trajectory
     suggests continued drift if trend holds."
```

#### Pattern 4: Temporal RAG Chain

```
Trigger: User asks a question about change over time
Steps:
  1. Embed user's question
  2. cvx_search(query, time_range=recent) → current state
  3. cvx_search(query, time_range=historical) → past state
  4. Compare: which entities were similar then vs now?
  5. For entities that changed:
     a. cvx_drift_report(entity_id, t1=historical, t2=recent)
  6. Synthesize: "The discourse around [topic] has shifted. In [past],
     entities X,Y,Z were similar. Now, only X remains. Y drifted toward
     [anchor] (drift=0.6), Z is no longer active."
```

These patterns are **not implemented as code** — they are documented as prompt templates that agent frameworks can use. The CVX MCP tools are the building blocks; the LLM composes them.

---

### 5. LLM Response Summarization Layer (Priority: P1)

#### Problem

Raw analytics results are too detailed for LLM context:
- A trajectory of 500 points × 768 dimensions = 384,000 floats
- A drift report with 768 dimensional attributions

The LLM doesn't need raw data — it needs **interpretable summaries** that fit in a few hundred tokens.

#### Solution

Each tool endpoint includes a `summarize` parameter. When true (default for MCP tools), the response replaces raw data with:

1. **Scalar metrics**: magnitude, percentile, p-value
2. **Natural language interpretation**: "significant drift toward isolation"
3. **Anchored descriptions**: drift expressed relative to named reference points
4. **Comparative context**: "more than 92% of entities in the index"

**Implementation:**

```rust
pub struct LlmSummary {
    /// One-sentence summary
    pub headline: String,
    /// Key metrics (name → value with interpretation)
    pub metrics: Vec<NamedMetric>,
    /// Notable findings
    pub findings: Vec<String>,
    /// Suggested follow-up queries
    pub suggested_next: Vec<SuggestedQuery>,
}

pub struct NamedMetric {
    pub name: String,
    pub value: f64,
    pub interpretation: String,  // "high", "normal", "concerning"
    pub percentile: Option<f64>,
}

pub struct SuggestedQuery {
    pub tool: String,
    pub description: String,
    pub parameters: serde_json::Value,
}
```

The `suggested_next` field is powerful: after a drift report, CVX can suggest "you might want to check change points for this entity" — guiding the LLM's reasoning chain.

---

## Implementation Plan

### Phase 1: `feat/mcp-server` (P0)

| Task | Crate | Details |
|------|-------|---------|
| Create `cvx-mcp` crate | cvx-mcp | New crate in workspace |
| MCP protocol handler (JSON-RPC over stdio) | cvx-mcp | Implement `initialize`, `tools/list`, `tools/call` |
| Tool definitions (8 tools) | cvx-mcp | Schema + dispatch to cvx-query |
| Embedded mode (load index from disk) | cvx-mcp | Reuse `TemporalHnsw::load()` |
| Proxy mode (forward to cvx-api) | cvx-mcp | HTTP client to REST endpoints |
| LLM summarization layer | cvx-mcp | `LlmSummary` generation for each tool response |
| Integration test: Claude Code config | cvx-mcp | Verify tool discovery + basic query |

### Phase 2: `feat/llm-api-layer` (P0)

| Task | Crate | Details |
|------|-------|---------|
| `cvx_entity_summary` composite endpoint | cvx-api | Combines trajectory + drift + CPD + Hurst |
| `cvx_detect_anomalies` batch scan | cvx-api | Iterate entities, run CPD, filter by severity |
| `cvx_compare_entities` composite endpoint | cvx-api | temporal_join + correlation + optional Granger |
| `cvx_cohort_analysis` endpoint | cvx-api | Delegates to RFC-007 cohort_drift |
| Summarization middleware | cvx-api | `?summarize=true` query param for all endpoints |
| OpenAPI spec for LLM tool definitions | cvx-api | Generate tool schemas from endpoint specs |

### Phase 3: `feat/inline-embeddings` (P1)

| Task | Crate | Details |
|------|-------|---------|
| `Embedder` trait | cvx-core | Interface for text → vector |
| `OnnxEmbedder` implementation | cvx-mcp | Via `ort` crate, load .onnx model |
| `ApiEmbedder` implementation | cvx-mcp | HTTP client for OpenAI/Cohere embedding API |
| Integration in MCP tools | cvx-mcp | Detect text vs vector in input, embed if needed |
| Integration in REST API | cvx-api | Optional `text` field alongside `vector` |

### Phase 4: `feat/agentic-patterns` (P2)

| Task | Location | Details |
|------|----------|---------|
| Document workflow patterns | docs/ | Longitudinal Monitor, Comparative Investigation, etc. |
| Prompt templates for each pattern | cvx-mcp/prompts/ | Structured prompts for agent frameworks |
| Example: Claude Code hook for monitoring | docs/ | `pre-tool-use` hook that queries CVX |
| Example: Jupyter + LLM integration | notebooks/ | Notebook showing LLM↔CVX interaction |

---

## Impact on Existing Code

| Component | Change |
|-----------|--------|
| **New crate: `cvx-mcp`** | MCP server binary, tool definitions, summarization, embedding layer |
| `cvx-api/handlers.rs` | New composite endpoints (entity_summary, detect_anomalies, compare_entities) |
| `cvx-core/traits` | New `Embedder` trait |
| `cvx-query` | No changes — MCP tools delegate to existing query engine |
| `Cargo.toml` (workspace) | Add `cvx-mcp` member, optional `ort` dependency |

No breaking changes to existing API. All additions are new endpoints/tools.

---

## Verification

### Test Plan

| Component | Test Strategy |
|-----------|--------------|
| MCP protocol | Mock stdio, verify JSON-RPC handshake, tool listing, tool execution |
| Tool responses | Verify summarization produces valid LlmSummary for each tool |
| Embedded mode | Load test index, execute each tool, verify results match direct API |
| Proxy mode | Start cvx-api, verify cvx-mcp forwards correctly |
| Inline embeddings | Embed known text, verify dimension and non-zero output |
| End-to-end | Configure in Claude Code, verify tool discovery and basic query |

### Acceptance Criteria

1. `cvx-mcp --index-path test.cvx` starts and responds to MCP `initialize`
2. Claude Code discovers all 8 tools via MCP
3. `cvx_search` with text query returns relevant results (requires embedder)
4. `cvx_entity_summary` returns a complete profile in < 500 tokens
5. `cvx_detect_anomalies` scans 1000 entities in < 5 seconds
6. All tool responses include `suggested_next` for agentic chaining

---

## Security Considerations

| Risk | Mitigation |
|------|------------|
| Prompt injection via entity metadata | Sanitize all metadata before including in LLM-facing responses |
| Excessive data exposure | Summarization layer limits data volume; raw vectors never exposed |
| Resource exhaustion via anomaly scan | Rate limiting + max_entities parameter |
| Embedding API key exposure | Environment variables, never in MCP config visible to LLM |

---

## References

1. Anthropic (2024). Model Context Protocol specification. https://modelcontextprotocol.io
2. OpenAI (2023). Function calling and tool use. API documentation.
3. Lewis, P. et al. (2020). Retrieval-Augmented Generation for knowledge-intensive NLP tasks. *NeurIPS*.
4. Gao, Y. et al. (2024). Retrieval-Augmented Generation for Large Language Models: A Survey. *arXiv:2312.10997*.
5. Schick, T. et al. (2024). Toolformer: Language models can teach themselves to use tools. *NeurIPS*.
6. Yao, S. et al. (2023). ReAct: Synergizing reasoning and acting in language models. *ICLR*.
