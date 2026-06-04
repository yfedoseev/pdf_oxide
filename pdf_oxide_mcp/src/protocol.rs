use serde_json::{json, Value};

use crate::extract;

const SERVER_NAME: &str = "pdf-oxide-mcp";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");
const PROTOCOL_VERSION: &str = "2024-11-05";

pub fn handle_message(line: &str) -> Option<String> {
    let msg: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => {
            let resp = json!({
                "jsonrpc": "2.0",
                "id": null,
                "error": { "code": -32700, "message": format!("Parse error: {e}") }
            });
            return Some(resp.to_string());
        },
    };

    let id = msg.get("id").cloned();
    let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");

    // Notifications (no id) don't get responses
    if method == "notifications/initialized" || method == "initialized" {
        return None;
    }
    // Any notification (no id field) → no response
    if id.is_none() && method.starts_with("notifications/") {
        return None;
    }

    let result = match method {
        "initialize" => handle_initialize(),
        "ping" => Ok(json!({})),
        "tools/list" => handle_tools_list(),
        "tools/call" => handle_tools_call(&msg),
        _ => Err((-32601, format!("Method not found: {method}"))),
    };

    let resp = match result {
        Ok(res) => json!({ "jsonrpc": "2.0", "id": id, "result": res }),
        Err((code, message)) => {
            json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
        },
    };

    Some(resp.to_string())
}

fn handle_initialize() -> Result<Value, (i32, String)> {
    Ok(json!({
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION
        }
    }))
}

fn handle_tools_list() -> Result<Value, (i32, String)> {
    Ok(json!({
        "tools": [
            {
                "name": "extract",
                "description": "Extract text, markdown, or HTML from a PDF file. Writes output to a file and optionally extracts images.",
                "inputSchema": {
                    "type": "object",
                    "required": ["file_path", "output_path"],
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the PDF file to extract from"
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Path to write extracted content to"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["text", "markdown", "html", "structured"],
                            "default": "text",
                            "description": "Output format: text, markdown, html, or structured (StructuredPage JSON — typed regions with RegionRole kind and column_index, so two-column layouts come out as separate column blocks instead of line-interleaved)"
                        },
                        "pages": {
                            "type": "string",
                            "description": "Page range to extract, e.g. \"1-3,7,10-12\". Defaults to all pages."
                        },
                        "password": {
                            "type": "string",
                            "description": "Password for encrypted PDFs"
                        },
                        "images": {
                            "type": "boolean",
                            "default": false,
                            "description": "Extract images to files alongside the output"
                        },
                        "embed_images": {
                            "type": "boolean",
                            "default": true,
                            "description": "Embed images as base64 data URIs in markdown/html output (true) or save as separate files (false)"
                        }
                    }
                }
            },
            {
                "name": "classify",
                "description": "Cheap per-page text-vs-OCR classification (no OCR, no rasterisation). Returns JSON DocumentClassification with per-page kinds, pages_needing_ocr, and an aggregate summary — so an agent can decide whether/where OCR is needed before extracting.",
                "inputSchema": {
                    "type": "object",
                    "required": ["file_path"],
                    "properties": {
                        "file_path": { "type": "string", "description": "Path to the PDF file" },
                        "password": { "type": "string", "description": "Password for encrypted PDFs" }
                    }
                }
            },
            {
                "name": "auto",
                "description": "Auto-extract text: per-page text-vs-OCR routing with graceful native fallback (never an opaque OCR error). format=text returns assembled text; format=json returns rich per-page PageExtraction with per-region bbox + typed reason codes.",
                "inputSchema": {
                    "type": "object",
                    "required": ["file_path"],
                    "properties": {
                        "file_path": { "type": "string", "description": "Path to the PDF file" },
                        "format": {
                            "type": "string",
                            "enum": ["text", "json"],
                            "default": "text",
                            "description": "text (assembled) or json (rich PageExtraction[] with typed reasons)"
                        },
                        "password": { "type": "string", "description": "Password for encrypted PDFs" }
                    }
                }
            }
        ]
    }))
}

fn handle_tools_call(msg: &Value) -> Result<Value, (i32, String)> {
    let params = msg.get("params").unwrap_or(&Value::Null);
    let tool_name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
    let arguments = params.get("arguments").unwrap_or(&Value::Null);

    match tool_name {
        "extract" => extract::run(arguments),
        "classify" => extract::classify(arguments),
        "auto" => extract::auto(arguments),
        _ => Err((-32602, format!("Unknown tool: {tool_name}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_response(line: &str) -> Value {
        let resp = handle_message(line).expect("expected a response");
        serde_json::from_str(&resp).expect("response should be valid JSON")
    }

    #[test]
    fn test_initialize() {
        let resp = parse_response(
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","clientInfo":{"name":"test","version":"0.1"}}}"#,
        );
        assert_eq!(resp["jsonrpc"], "2.0");
        assert_eq!(resp["id"], 1);
        assert_eq!(resp["result"]["protocolVersion"], PROTOCOL_VERSION);
        assert_eq!(resp["result"]["serverInfo"]["name"], SERVER_NAME);
        assert!(resp.get("error").is_none());
    }

    #[test]
    fn test_ping() {
        let resp = parse_response(r#"{"jsonrpc":"2.0","id":42,"method":"ping"}"#);
        assert_eq!(resp["id"], 42);
        assert_eq!(resp["result"], json!({}));
    }

    #[test]
    fn test_tools_list() {
        let resp = parse_response(r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#);
        let tools = resp["result"]["tools"].as_array().expect("tools array");
        // #517: `extract` (unchanged) + additive `classify` + `auto`.
        assert_eq!(tools.len(), 3);
        assert_eq!(tools[0]["name"], "extract");
        let required = tools[0]["inputSchema"]["required"]
            .as_array()
            .expect("required array");
        assert!(required.contains(&json!("file_path")));
        assert!(required.contains(&json!("output_path")));
        let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
        assert!(names.contains(&"classify"));
        assert!(names.contains(&"auto"));
    }

    #[test]
    fn test_unknown_method() {
        let resp = parse_response(r#"{"jsonrpc":"2.0","id":5,"method":"foo/bar"}"#);
        assert_eq!(resp["error"]["code"], -32601);
        assert!(resp["error"]["message"]
            .as_str()
            .unwrap()
            .contains("foo/bar"));
    }

    #[test]
    fn test_parse_error() {
        let resp = parse_response("not json at all");
        assert_eq!(resp["error"]["code"], -32700);
    }

    #[test]
    fn test_initialized_notification_no_response() {
        let resp = handle_message(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#);
        assert!(resp.is_none());
    }

    #[test]
    fn test_unknown_tool() {
        let resp = parse_response(
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"bogus","arguments":{}}}"#,
        );
        assert_eq!(resp["error"]["code"], -32602);
        assert!(resp["error"]["message"].as_str().unwrap().contains("bogus"));
    }

    #[test]
    fn test_extract_missing_file_path() {
        let resp = parse_response(
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"extract","arguments":{"output_path":"/tmp/out.txt"}}}"#,
        );
        assert_eq!(resp["error"]["code"], -32602);
        assert!(resp["error"]["message"]
            .as_str()
            .unwrap()
            .contains("file_path"));
    }

    #[test]
    fn test_extract_missing_output_path() {
        let resp = parse_response(
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"extract","arguments":{"file_path":"test.pdf"}}}"#,
        );
        assert_eq!(resp["error"]["code"], -32602);
        assert!(resp["error"]["message"]
            .as_str()
            .unwrap()
            .contains("output_path"));
    }

    #[test]
    fn test_extract_nonexistent_pdf() {
        let resp = parse_response(
            r#"{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"extract","arguments":{"file_path":"/nonexistent/file.pdf","output_path":"/tmp/out.txt"}}}"#,
        );
        assert_eq!(resp["error"]["code"], -32603);
        assert!(resp["error"]["message"]
            .as_str()
            .unwrap()
            .contains("Failed to open PDF"));
    }

    #[test]
    fn test_extract_invalid_format() {
        let resp = parse_response(
            r#"{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"extract","arguments":{"file_path":"tests/fixtures/simple.pdf","output_path":"/tmp/out.txt","format":"csv"}}}"#,
        );
        assert_eq!(resp["error"]["code"], -32602);
        assert!(resp["error"]["message"].as_str().unwrap().contains("csv"));
    }

    #[test]
    fn test_response_has_jsonrpc_field() {
        let resp = parse_response(r#"{"jsonrpc":"2.0","id":99,"method":"ping"}"#);
        assert_eq!(resp["jsonrpc"], "2.0");
        assert_eq!(resp["id"], 99);
    }
}
