//! MCP tool definitions for ChronosVector.
//!
//! Each tool maps to one or more CVX query operations. Tools return
//! structured JSON summaries optimized for LLM consumption.

use crate::protocol::ToolDefinition;
use serde_json::json;

/// Return all available CVX tools.
pub fn all_tools() -> Vec<ToolDefinition> {
    vec![
        cvx_search(),
        cvx_entity_summary(),
        cvx_drift_report(),
        cvx_detect_anomalies(),
        cvx_compare_entities(),
        cvx_cohort_analysis(),
        cvx_forecast(),
        cvx_ingest(),
    ]
}

fn cvx_search() -> ToolDefinition {
    ToolDefinition {
        name: "cvx_search".to_string(),
        description: "Search for semantically similar content in the temporal vector database. \
            Returns the most relevant matches with their timestamps and similarity scores. \
            Use this when the user asks about finding similar content or wants to explore \
            what exists in a time period."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "vector": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Query vector (embedding)."
                },
                "k": {
                    "type": "integer",
                    "default": 10,
                    "description": "Number of results to return."
                },
                "time_start": {
                    "type": "integer",
                    "description": "Optional: start of time range (microseconds since epoch)."
                },
                "time_end": {
                    "type": "integer",
                    "description": "Optional: end of time range (microseconds since epoch)."
                },
                "alpha": {
                    "type": "number",
                    "default": 0.8,
                    "description": "Balance between semantic (1.0) and temporal (0.0) relevance."
                },
                "entity_filter": {
                    "type": "integer",
                    "description": "Optional: restrict search to a specific entity."
                }
            },
            "required": ["vector"]
        }),
    }
}

fn cvx_entity_summary() -> ToolDefinition {
    ToolDefinition {
        name: "cvx_entity_summary".to_string(),
        description: "Get a comprehensive temporal summary of an entity: activity span, \
            drift magnitude, change points, trajectory character. \
            Use this as a starting point when investigating an entity."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "integer",
                    "description": "Entity identifier."
                }
            },
            "required": ["entity_id"]
        }),
    }
}

fn cvx_drift_report() -> ToolDefinition {
    ToolDefinition {
        name: "cvx_drift_report".to_string(),
        description: "Measure how much an entity's semantic representation has changed \
            between two time points. Returns drift magnitude, direction, and which \
            dimensions changed most. Use this to quantify observed or suspected changes."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "integer",
                    "description": "Entity identifier."
                },
                "t1": {
                    "type": "integer",
                    "description": "Start timestamp (microseconds)."
                },
                "t2": {
                    "type": "integer",
                    "description": "End timestamp (microseconds)."
                },
                "top_n": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of top changed dimensions to report."
                }
            },
            "required": ["entity_id", "t1", "t2"]
        }),
    }
}

fn cvx_detect_anomalies() -> ToolDefinition {
    ToolDefinition {
        name: "cvx_detect_anomalies".to_string(),
        description: "Scan one or more entities for anomalous semantic changes. \
            Returns entities with detected change points or unusual drift. \
            Use this for proactive monitoring."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "entity_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Entities to scan. If omitted, requires at least one."
                },
                "lookback_us": {
                    "type": "integer",
                    "description": "Lookback window in microseconds."
                }
            },
            "required": ["entity_ids"]
        }),
    }
}

fn cvx_compare_entities() -> ToolDefinition {
    ToolDefinition {
        name: "cvx_compare_entities".to_string(),
        description: "Compare the semantic trajectories of two entities. \
            Reveals convergence windows and potential causal relationships. \
            Use when the user asks about relationships between entities over time."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "entity_a": {
                    "type": "integer",
                    "description": "First entity identifier."
                },
                "entity_b": {
                    "type": "integer",
                    "description": "Second entity identifier."
                },
                "epsilon": {
                    "type": "number",
                    "default": 0.5,
                    "description": "Distance threshold for convergence detection."
                },
                "window_us": {
                    "type": "integer",
                    "default": 604_800_000_000i64,
                    "description": "Window size in microseconds (default: 7 days)."
                }
            },
            "required": ["entity_a", "entity_b"]
        }),
    }
}

fn cvx_cohort_analysis() -> ToolDefinition {
    ToolDefinition {
        name: "cvx_cohort_analysis".to_string(),
        description: "Analyze the collective behavior of a group of entities. \
            Measures whether the group is converging, diverging, and how the centroid moved. \
            Use for population-level analysis: treatment groups, user segments, sectors."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "entity_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Entity identifiers in the cohort."
                },
                "t1": {
                    "type": "integer",
                    "description": "Start timestamp."
                },
                "t2": {
                    "type": "integer",
                    "description": "End timestamp."
                },
                "top_n": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of top dimensions to report."
                }
            },
            "required": ["entity_ids", "t1", "t2"]
        }),
    }
}

fn cvx_forecast() -> ToolDefinition {
    ToolDefinition {
        name: "cvx_forecast".to_string(),
        description: "Predict the future semantic trajectory of an entity. \
            Uses the entity's historical movement pattern to extrapolate. \
            Use when the user asks 'where is this entity heading?'"
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "integer",
                    "description": "Entity identifier."
                },
                "target_timestamp": {
                    "type": "integer",
                    "description": "Target future timestamp (microseconds)."
                }
            },
            "required": ["entity_id", "target_timestamp"]
        }),
    }
}

fn cvx_ingest() -> ToolDefinition {
    ToolDefinition {
        name: "cvx_ingest".to_string(),
        description: "Insert new temporal data points into the CVX index. \
            Each point has an entity ID, timestamp, and a vector. \
            Use when the user wants to add new data."
            .to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity_id": {"type": "integer"},
                            "timestamp": {"type": "integer"},
                            "vector": {
                                "type": "array",
                                "items": {"type": "number"}
                            }
                        },
                        "required": ["entity_id", "timestamp", "vector"]
                    },
                    "description": "Points to ingest."
                }
            },
            "required": ["points"]
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_tools_count() {
        assert_eq!(all_tools().len(), 8);
    }

    #[test]
    fn all_tools_have_names_and_descriptions() {
        for tool in all_tools() {
            assert!(!tool.name.is_empty());
            assert!(!tool.description.is_empty());
            assert!(tool.input_schema.is_object());
        }
    }

    #[test]
    fn tool_schemas_have_required_fields() {
        for tool in all_tools() {
            let schema = &tool.input_schema;
            assert_eq!(schema["type"], "object");
            assert!(schema["properties"].is_object());
        }
    }
}
