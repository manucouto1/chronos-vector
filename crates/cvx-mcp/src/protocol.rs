//! MCP JSON-RPC protocol types.
//!
//! Implements the Model Context Protocol (MCP) message format:
//! JSON-RPC 2.0 with MCP-specific methods.

use serde::{Deserialize, Serialize};

// ─── JSON-RPC 2.0 ──────────────────────────────────────────────────

/// A JSON-RPC 2.0 request.
#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    /// Protocol version (always "2.0").
    pub jsonrpc: String,
    /// Request ID (number or string).
    pub id: serde_json::Value,
    /// Method name.
    pub method: String,
    /// Parameters (optional).
    #[serde(default)]
    pub params: serde_json::Value,
}

/// A JSON-RPC 2.0 response.
#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    /// Protocol version.
    pub jsonrpc: String,
    /// Request ID (echoed from request).
    pub id: serde_json::Value,
    /// Result (present on success).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    /// Error (present on failure).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// A JSON-RPC 2.0 error.
#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    /// Error code.
    pub code: i64,
    /// Error message.
    pub message: String,
    /// Additional data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl JsonRpcResponse {
    /// Create a success response.
    pub fn success(id: serde_json::Value, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
        }
    }

    /// Create an error response.
    pub fn error(id: serde_json::Value, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }
}

// ─── MCP-specific types ─────────────────────────────────────────────

/// MCP server capabilities returned in `initialize` response.
#[derive(Debug, Serialize)]
pub struct ServerCapabilities {
    /// Available tools.
    pub tools: ToolsCapability,
}

/// Tools capability.
#[derive(Debug, Serialize)]
pub struct ToolsCapability {
    /// Whether tool list can change dynamically.
    #[serde(rename = "listChanged")]
    pub list_changed: bool,
}

/// MCP server info.
#[derive(Debug, Serialize)]
pub struct ServerInfo {
    /// Server name.
    pub name: String,
    /// Server version.
    pub version: String,
}

/// MCP initialize response result.
#[derive(Debug, Serialize)]
pub struct InitializeResult {
    /// Protocol version.
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    /// Server capabilities.
    pub capabilities: ServerCapabilities,
    /// Server info.
    #[serde(rename = "serverInfo")]
    pub server_info: ServerInfo,
}

/// An MCP tool definition.
#[derive(Debug, Clone, Serialize)]
pub struct ToolDefinition {
    /// Tool name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// JSON Schema for the tool's input parameters.
    #[serde(rename = "inputSchema")]
    pub input_schema: serde_json::Value,
}

/// Result of a tool call.
#[derive(Debug, Serialize)]
pub struct ToolCallResult {
    /// Content items returned by the tool.
    pub content: Vec<ContentItem>,
    /// Whether the tool call errored.
    #[serde(rename = "isError", skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

/// A content item in a tool result.
#[derive(Debug, Serialize)]
pub struct ContentItem {
    /// Content type (always "text" for now).
    #[serde(rename = "type")]
    pub content_type: String,
    /// Text content.
    pub text: String,
}

impl ContentItem {
    /// Create a text content item.
    pub fn text(s: impl Into<String>) -> Self {
        Self {
            content_type: "text".to_string(),
            text: s.into(),
        }
    }
}

impl ToolCallResult {
    /// Create a successful result with text content.
    pub fn text(s: impl Into<String>) -> Self {
        Self {
            content: vec![ContentItem::text(s)],
            is_error: None,
        }
    }

    /// Create an error result.
    pub fn error(s: impl Into<String>) -> Self {
        Self {
            content: vec![ContentItem::text(s)],
            is_error: Some(true),
        }
    }
}

// ─── Error codes ────────────────────────────────────────────────────

/// Standard JSON-RPC error codes.
pub const PARSE_ERROR: i64 = -32700;
/// Invalid request.
pub const INVALID_REQUEST: i64 = -32600;
/// Method not found.
pub const METHOD_NOT_FOUND: i64 = -32601;
/// Invalid params.
pub const INVALID_PARAMS: i64 = -32602;
/// Internal error.
pub const INTERNAL_ERROR: i64 = -32603;
