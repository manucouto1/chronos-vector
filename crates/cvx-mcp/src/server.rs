//! MCP server: handles JSON-RPC messages and dispatches tool calls.

use cvx_core::types::TemporalFilter;
use cvx_core::TemporalIndexAccess;
use serde_json::json;

use crate::protocol::*;
use crate::tools;

/// MCP server state.
pub struct McpServer<I: TemporalIndexAccess> {
    index: I,
}

impl<I: TemporalIndexAccess> McpServer<I> {
    /// Create a new MCP server wrapping a temporal index.
    pub fn new(index: I) -> Self {
        Self { index }
    }

    /// Handle a single JSON-RPC request and return the response.
    pub fn handle_request(&self, request: &JsonRpcRequest) -> JsonRpcResponse {
        match request.method.as_str() {
            "initialize" => self.handle_initialize(&request.id),
            "initialized" => JsonRpcResponse::success(request.id.clone(), json!({})),
            "tools/list" => self.handle_tools_list(&request.id),
            "tools/call" => self.handle_tools_call(&request.id, &request.params),
            "ping" => JsonRpcResponse::success(request.id.clone(), json!({})),
            _ => JsonRpcResponse::error(
                request.id.clone(),
                METHOD_NOT_FOUND,
                format!("unknown method: {}", request.method),
            ),
        }
    }

    /// Handle raw JSON input (parse + dispatch + serialize).
    pub fn handle_message(&self, input: &str) -> String {
        let request: JsonRpcRequest = match serde_json::from_str(input) {
            Ok(r) => r,
            Err(e) => {
                let resp = JsonRpcResponse::error(
                    serde_json::Value::Null,
                    PARSE_ERROR,
                    format!("parse error: {e}"),
                );
                return serde_json::to_string(&resp).unwrap_or_default();
            }
        };

        let response = self.handle_request(&request);
        serde_json::to_string(&response).unwrap_or_default()
    }

    // ─── Method handlers ────────────────────────────────────────

    fn handle_initialize(&self, id: &serde_json::Value) -> JsonRpcResponse {
        let result = InitializeResult {
            protocol_version: "2024-11-05".to_string(),
            capabilities: ServerCapabilities {
                tools: ToolsCapability {
                    list_changed: false,
                },
            },
            server_info: ServerInfo {
                name: "cvx-mcp".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
        };

        JsonRpcResponse::success(id.clone(), serde_json::to_value(result).unwrap())
    }

    fn handle_tools_list(&self, id: &serde_json::Value) -> JsonRpcResponse {
        let tools = tools::all_tools();
        JsonRpcResponse::success(id.clone(), json!({ "tools": tools }))
    }

    fn handle_tools_call(
        &self,
        id: &serde_json::Value,
        params: &serde_json::Value,
    ) -> JsonRpcResponse {
        let tool_name = params
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let arguments = params
            .get("arguments")
            .cloned()
            .unwrap_or(json!({}));

        let result = match tool_name {
            "cvx_search" => self.tool_search(&arguments),
            "cvx_entity_summary" => self.tool_entity_summary(&arguments),
            "cvx_drift_report" => self.tool_drift_report(&arguments),
            "cvx_detect_anomalies" => self.tool_detect_anomalies(&arguments),
            "cvx_compare_entities" => self.tool_compare_entities(&arguments),
            "cvx_cohort_analysis" => self.tool_cohort_analysis(&arguments),
            "cvx_forecast" => self.tool_forecast(&arguments),
            "cvx_causal_search" => self.tool_causal_search(&arguments),
            "cvx_ingest" => ToolCallResult::error("cvx_ingest requires mutable access; use the REST API for ingestion."),
            _ => ToolCallResult::error(format!("unknown tool: {tool_name}")),
        };

        JsonRpcResponse::success(id.clone(), serde_json::to_value(result).unwrap())
    }

    // ─── Tool implementations ───────────────────────────────────

    fn tool_search(&self, args: &serde_json::Value) -> ToolCallResult {
        let vector: Vec<f32> = match args.get("vector").and_then(|v| {
            v.as_array().map(|a| a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect())
        }) {
            Some(v) => v,
            None => return ToolCallResult::error("missing or invalid 'vector' parameter"),
        };

        let k = args.get("k").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
        let alpha = args.get("alpha").and_then(|v| v.as_f64()).unwrap_or(0.8) as f32;

        let filter = match (
            args.get("time_start").and_then(|v| v.as_i64()),
            args.get("time_end").and_then(|v| v.as_i64()),
        ) {
            (Some(start), Some(end)) => TemporalFilter::Range(start, end),
            (Some(start), None) => TemporalFilter::After(start),
            (None, Some(end)) => TemporalFilter::Before(end),
            (None, None) => TemporalFilter::All,
        };

        let query_ts = args.get("time_start").and_then(|v| v.as_i64()).unwrap_or(0);

        let results = self.index.search_raw(&vector, k, filter, alpha, query_ts);

        let matches: Vec<serde_json::Value> = results
            .iter()
            .map(|&(node_id, score)| {
                json!({
                    "entity_id": self.index.entity_id(node_id),
                    "timestamp": self.index.timestamp(node_id),
                    "score": score,
                })
            })
            .collect();

        let summary = format!(
            "Found {} matches{}.",
            matches.len(),
            if matches.is_empty() { "" } else { " ordered by relevance" }
        );

        ToolCallResult::text(serde_json::to_string_pretty(&json!({
            "matches": matches,
            "summary": summary,
        })).unwrap())
    }

    fn tool_entity_summary(&self, args: &serde_json::Value) -> ToolCallResult {
        let entity_id = match args.get("entity_id").and_then(|v| v.as_u64()) {
            Some(id) => id,
            None => return ToolCallResult::error("missing 'entity_id' parameter"),
        };

        let traj = self.index.trajectory(entity_id, TemporalFilter::All);
        if traj.is_empty() {
            return ToolCallResult::error(format!("entity {entity_id} not found"));
        }

        let first_seen = traj.first().unwrap().0;
        let last_seen = traj.last().unwrap().0;
        let total_points = traj.len();

        // Compute drift between first and last
        let first_node = traj.first().unwrap().1;
        let last_node = traj.last().unwrap().1;
        let v_first = self.index.vector(first_node);
        let v_last = self.index.vector(last_node);
        let drift = cvx_analytics::calculus::drift_report(&v_first, &v_last, 5);

        // Velocity at last point (if enough data)
        let velocity_magnitude = if traj.len() >= 2 {
            let vectors: Vec<Vec<f32>> = traj.iter().map(|&(_, nid)| self.index.vector(nid)).collect();
            let traj_refs: Vec<(i64, &[f32])> = traj.iter().zip(vectors.iter())
                .map(|(&(ts, _), v)| (ts, v.as_slice())).collect();
            cvx_analytics::calculus::velocity(&traj_refs, last_seen)
                .ok()
                .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt())
        } else {
            None
        };

        let drift_interpretation = if drift.l2_magnitude < 0.1 {
            "Minimal change"
        } else if drift.l2_magnitude < 0.5 {
            "Moderate drift"
        } else {
            "Significant semantic shift"
        };

        let result = json!({
            "entity_id": entity_id,
            "first_seen": first_seen,
            "last_seen": last_seen,
            "total_points": total_points,
            "overall_drift": {
                "l2_magnitude": drift.l2_magnitude,
                "cosine_drift": drift.cosine_drift,
                "interpretation": drift_interpretation,
            },
            "velocity_magnitude": velocity_magnitude,
            "top_changed_dimensions": drift.top_dimensions.iter()
                .map(|(idx, change)| json!({"dim": idx, "change": change}))
                .collect::<Vec<_>>(),
            "summary": format!(
                "Entity {} has {} data points spanning timestamps {} to {}. {}. Overall L2 drift: {:.4}.",
                entity_id, total_points, first_seen, last_seen,
                drift_interpretation, drift.l2_magnitude
            ),
        });

        ToolCallResult::text(serde_json::to_string_pretty(&result).unwrap())
    }

    fn tool_drift_report(&self, args: &serde_json::Value) -> ToolCallResult {
        let entity_id = match args.get("entity_id").and_then(|v| v.as_u64()) {
            Some(id) => id,
            None => return ToolCallResult::error("missing 'entity_id'"),
        };
        let t1 = match args.get("t1").and_then(|v| v.as_i64()) {
            Some(t) => t,
            None => return ToolCallResult::error("missing 't1'"),
        };
        let t2 = match args.get("t2").and_then(|v| v.as_i64()) {
            Some(t) => t,
            None => return ToolCallResult::error("missing 't2'"),
        };
        let top_n = args.get("top_n").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

        let query = cvx_query::types::TemporalQuery::DriftQuant {
            entity_id,
            t1,
            t2,
            top_n,
        };

        match cvx_query::engine::execute_query(&self.index, query) {
            Ok(cvx_query::types::QueryResult::Drift(drift)) => {
                let interpretation = if drift.l2_magnitude < 0.1 {
                    "Minimal change"
                } else if drift.l2_magnitude < 0.5 {
                    "Moderate drift"
                } else {
                    "Significant shift"
                };

                let result = json!({
                    "entity_id": entity_id,
                    "period": {"t1": t1, "t2": t2},
                    "drift": {
                        "l2_magnitude": drift.l2_magnitude,
                        "cosine_drift": drift.cosine_drift,
                        "interpretation": interpretation,
                    },
                    "top_dimensions": drift.top_dimensions.iter()
                        .map(|(idx, change)| json!({"dim": idx, "change": change}))
                        .collect::<Vec<_>>(),
                    "summary": format!(
                        "Entity {} shows {} (L2={:.4}, cosine={:.4}) between t1 and t2.",
                        entity_id, interpretation, drift.l2_magnitude, drift.cosine_drift
                    ),
                });

                ToolCallResult::text(serde_json::to_string_pretty(&result).unwrap())
            }
            Ok(_) => ToolCallResult::error("unexpected result type"),
            Err(e) => ToolCallResult::error(format!("query error: {e}")),
        }
    }

    fn tool_detect_anomalies(&self, args: &serde_json::Value) -> ToolCallResult {
        let entity_ids: Vec<u64> = match args.get("entity_ids").and_then(|v| {
            v.as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
        }) {
            Some(ids) => ids,
            None => return ToolCallResult::error("missing 'entity_ids'"),
        };

        let lookback = args.get("lookback_us").and_then(|v| v.as_i64())
            .unwrap_or(7 * 86_400_000_000); // default 7 days

        let mut anomalies: Vec<serde_json::Value> = Vec::new();

        for &eid in &entity_ids {
            let traj = self.index.trajectory(eid, TemporalFilter::All);
            if traj.len() < 4 {
                continue;
            }

            let last_ts = traj.last().unwrap().0;
            let start = last_ts - lookback;

            // Run change point detection on the lookback window
            let query = cvx_query::types::TemporalQuery::ChangePointDetect {
                entity_id: eid,
                start,
                end: last_ts,
            };

            if let Ok(cvx_query::types::QueryResult::ChangePoints(cps)) =
                cvx_query::engine::execute_query(&self.index, query)
            {
                for cp in cps {
                    anomalies.push(json!({
                        "entity_id": eid,
                        "type": "change_point",
                        "timestamp": cp.timestamp(),
                        "severity": cp.severity(),
                    }));
                }
            }
        }

        anomalies.sort_by(|a, b| {
            b["severity"].as_f64().unwrap_or(0.0)
                .partial_cmp(&a["severity"].as_f64().unwrap_or(0.0))
                .unwrap()
        });

        let result = json!({
            "entities_scanned": entity_ids.len(),
            "anomalies": anomalies,
            "summary": format!(
                "{} anomalies detected across {} entities.",
                anomalies.len(), entity_ids.len()
            ),
        });

        ToolCallResult::text(serde_json::to_string_pretty(&result).unwrap())
    }

    fn tool_compare_entities(&self, args: &serde_json::Value) -> ToolCallResult {
        let entity_a = match args.get("entity_a").and_then(|v| v.as_u64()) {
            Some(id) => id,
            None => return ToolCallResult::error("missing 'entity_a'"),
        };
        let entity_b = match args.get("entity_b").and_then(|v| v.as_u64()) {
            Some(id) => id,
            None => return ToolCallResult::error("missing 'entity_b'"),
        };
        let epsilon = args.get("epsilon").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
        let window_us = args.get("window_us").and_then(|v| v.as_i64())
            .unwrap_or(7 * 86_400_000_000);

        let query = cvx_query::types::TemporalQuery::TemporalJoin {
            entity_a,
            entity_b,
            epsilon,
            window_us,
        };

        match cvx_query::engine::execute_query(&self.index, query) {
            Ok(cvx_query::types::QueryResult::TemporalJoin(joins)) => {
                let windows: Vec<serde_json::Value> = joins.iter().map(|j| json!({
                    "start": j.start,
                    "end": j.end,
                    "mean_distance": j.mean_distance,
                    "min_distance": j.min_distance,
                })).collect();

                let summary = if windows.is_empty() {
                    format!("Entities {} and {} show no convergence periods (ε={}).", entity_a, entity_b, epsilon)
                } else {
                    format!("Found {} convergence window(s) between entities {} and {}.", windows.len(), entity_a, entity_b)
                };

                ToolCallResult::text(serde_json::to_string_pretty(&json!({
                    "entity_a": entity_a,
                    "entity_b": entity_b,
                    "convergence_windows": windows,
                    "summary": summary,
                })).unwrap())
            }
            Ok(_) => ToolCallResult::error("unexpected result type"),
            Err(e) => ToolCallResult::error(format!("query error: {e}")),
        }
    }

    fn tool_cohort_analysis(&self, args: &serde_json::Value) -> ToolCallResult {
        let entity_ids: Vec<u64> = match args.get("entity_ids").and_then(|v| {
            v.as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
        }) {
            Some(ids) => ids,
            None => return ToolCallResult::error("missing 'entity_ids'"),
        };
        let t1 = match args.get("t1").and_then(|v| v.as_i64()) {
            Some(t) => t,
            None => return ToolCallResult::error("missing 't1'"),
        };
        let t2 = match args.get("t2").and_then(|v| v.as_i64()) {
            Some(t) => t,
            None => return ToolCallResult::error("missing 't2'"),
        };
        let top_n = args.get("top_n").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

        let query = cvx_query::types::TemporalQuery::CohortDrift {
            entity_ids,
            t1,
            t2,
            top_n,
        };

        match cvx_query::engine::execute_query(&self.index, query) {
            Ok(cvx_query::types::QueryResult::CohortDrift(report)) => {
                let disp_interp = if report.dispersion_change < -0.05 {
                    "converging"
                } else if report.dispersion_change > 0.05 {
                    "diverging"
                } else {
                    "stable"
                };

                let result = json!({
                    "n_entities": report.n_entities,
                    "mean_drift_l2": report.mean_drift_l2,
                    "median_drift_l2": report.median_drift_l2,
                    "centroid_drift": report.centroid_l2_magnitude,
                    "dispersion_change": report.dispersion_change,
                    "convergence_score": report.convergence_score,
                    "cohort_behavior": disp_interp,
                    "outliers": report.outliers.iter().map(|o| json!({
                        "entity_id": o.entity_id,
                        "drift": o.drift_magnitude,
                        "z_score": o.z_score,
                    })).collect::<Vec<_>>(),
                    "summary": format!(
                        "Cohort of {} entities: mean drift {:.4}, {} (dispersion Δ={:.4}), convergence score {:.2}.",
                        report.n_entities, report.mean_drift_l2, disp_interp,
                        report.dispersion_change, report.convergence_score
                    ),
                });

                ToolCallResult::text(serde_json::to_string_pretty(&result).unwrap())
            }
            Ok(_) => ToolCallResult::error("unexpected result type"),
            Err(e) => ToolCallResult::error(format!("query error: {e}")),
        }
    }

    fn tool_forecast(&self, args: &serde_json::Value) -> ToolCallResult {
        let entity_id = match args.get("entity_id").and_then(|v| v.as_u64()) {
            Some(id) => id,
            None => return ToolCallResult::error("missing 'entity_id'"),
        };
        let target = match args.get("target_timestamp").and_then(|v| v.as_i64()) {
            Some(t) => t,
            None => return ToolCallResult::error("missing 'target_timestamp'"),
        };

        let query = cvx_query::types::TemporalQuery::Prediction {
            entity_id,
            target_timestamp: target,
        };

        match cvx_query::engine::execute_query(&self.index, query) {
            Ok(cvx_query::types::QueryResult::Prediction(pred)) => {
                let result = json!({
                    "entity_id": entity_id,
                    "target_timestamp": pred.timestamp,
                    "method": format!("{:?}", pred.method),
                    "predicted_vector_dim": pred.vector.len(),
                    "summary": format!(
                        "Predicted position for entity {} at t={} using {:?} extrapolation ({}-dim vector).",
                        entity_id, target, pred.method, pred.vector.len()
                    ),
                });

                ToolCallResult::text(serde_json::to_string_pretty(&result).unwrap())
            }
            Ok(_) => ToolCallResult::error("unexpected result type"),
            Err(e) => ToolCallResult::error(format!("query error: {e}")),
        }
    }

    fn tool_causal_search(&self, args: &serde_json::Value) -> ToolCallResult {
        let vector: Vec<f32> = match args.get("vector").and_then(|v| {
            v.as_array().map(|a| a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect())
        }) {
            Some(v) => v,
            None => return ToolCallResult::error("missing or invalid 'vector' parameter"),
        };

        let k = args.get("k").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let temporal_context = args.get("temporal_context").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let alpha = args.get("alpha").and_then(|v| v.as_f64()).unwrap_or(0.8) as f32;

        let query = cvx_query::types::TemporalQuery::CausalSearch {
            vector,
            k,
            filter: TemporalFilter::All,
            alpha,
            query_timestamp: 0,
            temporal_context,
        };

        match cvx_query::engine::execute_query(&self.index, query) {
            Ok(cvx_query::types::QueryResult::CausalSearch(entries)) => {
                let results: Vec<serde_json::Value> = entries.iter().map(|e| {
                    json!({
                        "entity_id": e.entity_id,
                        "score": e.score,
                        "successors": e.successors.iter()
                            .map(|(nid, ts)| json!({"node_id": nid, "timestamp": ts}))
                            .collect::<Vec<_>>(),
                        "predecessors": e.predecessors.iter()
                            .map(|(nid, ts)| json!({"node_id": nid, "timestamp": ts}))
                            .collect::<Vec<_>>(),
                    })
                }).collect();

                let summary = format!(
                    "Found {} matches with temporal context (±{} steps).",
                    entries.len(), temporal_context
                );

                ToolCallResult::text(serde_json::to_string_pretty(&json!({
                    "results": results,
                    "summary": summary,
                })).unwrap())
            }
            Ok(_) => ToolCallResult::error("unexpected result type"),
            Err(e) => ToolCallResult::error(format!("query error: {e}")),
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use cvx_index::hnsw::temporal::TemporalHnsw;
    use cvx_index::hnsw::HnswConfig;
    use cvx_index::metrics::L2Distance;

    fn setup_server() -> McpServer<TemporalHnsw<L2Distance>> {
        let config = HnswConfig::default();
        let mut index = TemporalHnsw::new(config, L2Distance);

        for e in 0..5u64 {
            for i in 0..20usize {
                let ts = i as i64 * 1_000_000;
                let v: Vec<f32> = vec![
                    e as f32 * 10.0 + i as f32 * 0.1,
                    (i as f32 * 0.1).sin(),
                    e as f32,
                ];
                index.insert(e, ts, &v);
            }
        }

        McpServer::new(index)
    }

    #[test]
    fn handle_initialize() {
        let server = setup_server();
        let resp = server.handle_message(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#);
        let parsed: serde_json::Value = serde_json::from_str(&resp).unwrap();

        assert_eq!(parsed["jsonrpc"], "2.0");
        assert_eq!(parsed["id"], 1);
        assert!(parsed["result"]["protocolVersion"].is_string());
        assert_eq!(parsed["result"]["serverInfo"]["name"], "cvx-mcp");
    }

    #[test]
    fn handle_tools_list() {
        let server = setup_server();
        let resp = server.handle_message(r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#);
        let parsed: serde_json::Value = serde_json::from_str(&resp).unwrap();

        let tools = parsed["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 9);
    }

    #[test]
    fn handle_unknown_method() {
        let server = setup_server();
        let resp = server.handle_message(r#"{"jsonrpc":"2.0","id":3,"method":"nonexistent","params":{}}"#);
        let parsed: serde_json::Value = serde_json::from_str(&resp).unwrap();

        assert!(parsed["error"].is_object());
        assert_eq!(parsed["error"]["code"], METHOD_NOT_FOUND);
    }

    #[test]
    fn handle_parse_error() {
        let server = setup_server();
        let resp = server.handle_message("not json");
        let parsed: serde_json::Value = serde_json::from_str(&resp).unwrap();

        assert_eq!(parsed["error"]["code"], PARSE_ERROR);
    }

    #[test]
    fn tool_call_search() {
        let server = setup_server();
        let req = json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "cvx_search",
                "arguments": {
                    "vector": [5.0, 0.0, 1.0],
                    "k": 3
                }
            }
        });
        let resp = server.handle_message(&req.to_string());
        let parsed: serde_json::Value = serde_json::from_str(&resp).unwrap();

        assert!(parsed["result"]["content"][0]["text"].is_string());
        let text = parsed["result"]["content"][0]["text"].as_str().unwrap();
        let inner: serde_json::Value = serde_json::from_str(text).unwrap();
        assert!(inner["matches"].is_array());
        assert!(!inner["matches"].as_array().unwrap().is_empty());
    }

    #[test]
    fn tool_call_entity_summary() {
        let server = setup_server();
        let req = json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "cvx_entity_summary",
                "arguments": {"entity_id": 0}
            }
        });
        let resp = server.handle_message(&req.to_string());
        let parsed: serde_json::Value = serde_json::from_str(&resp).unwrap();

        let text = parsed["result"]["content"][0]["text"].as_str().unwrap();
        let inner: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(inner["entity_id"], 0);
        assert_eq!(inner["total_points"], 20);
        assert!(inner["summary"].is_string());
    }

    #[test]
    fn tool_call_drift_report() {
        let server = setup_server();
        let req = json!({
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "cvx_drift_report",
                "arguments": {"entity_id": 0, "t1": 0, "t2": 19000000, "top_n": 3}
            }
        });
        let resp = server.handle_message(&req.to_string());
        let parsed: serde_json::Value = serde_json::from_str(&resp).unwrap();

        let text = parsed["result"]["content"][0]["text"].as_str().unwrap();
        let inner: serde_json::Value = serde_json::from_str(text).unwrap();
        assert!(inner["drift"]["l2_magnitude"].is_number());
    }

    #[test]
    fn tool_call_entity_not_found() {
        let server = setup_server();
        let req = json!({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "cvx_entity_summary",
                "arguments": {"entity_id": 999}
            }
        });
        let resp = server.handle_message(&req.to_string());
        let parsed: serde_json::Value = serde_json::from_str(&resp).unwrap();

        let text = parsed["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("not found"));
    }

    #[test]
    fn tool_call_cohort() {
        let server = setup_server();
        let req = json!({
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {
                "name": "cvx_cohort_analysis",
                "arguments": {
                    "entity_ids": [0, 1, 2, 3, 4],
                    "t1": 0,
                    "t2": 19000000,
                    "top_n": 3
                }
            }
        });
        let resp = server.handle_message(&req.to_string());
        let parsed: serde_json::Value = serde_json::from_str(&resp).unwrap();

        let text = parsed["result"]["content"][0]["text"].as_str().unwrap();
        let inner: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(inner["n_entities"], 5);
        assert!(inner["summary"].is_string());
    }

    #[test]
    fn tool_call_forecast() {
        let server = setup_server();
        let req = json!({
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {
                "name": "cvx_forecast",
                "arguments": {"entity_id": 0, "target_timestamp": 25000000}
            }
        });
        let resp = server.handle_message(&req.to_string());
        let parsed: serde_json::Value = serde_json::from_str(&resp).unwrap();

        let text = parsed["result"]["content"][0]["text"].as_str().unwrap();
        let inner: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(inner["entity_id"], 0);
        assert!(inner["method"].is_string());
    }

    #[test]
    fn tool_call_unknown_tool() {
        let server = setup_server();
        let req = json!({
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {"name": "nonexistent", "arguments": {}}
        });
        let resp = server.handle_message(&req.to_string());
        let parsed: serde_json::Value = serde_json::from_str(&resp).unwrap();

        let is_error = parsed["result"]["isError"].as_bool().unwrap_or(false);
        assert!(is_error);
    }
}
