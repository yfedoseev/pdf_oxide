use std::fs;
use std::path::Path;

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::PdfDocument;
use serde_json::{json, Value};

pub fn run(args: &Value) -> Result<Value, (i32, String)> {
    let file_path = args
        .get("file_path")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "Missing required parameter: file_path".to_string()))?;
    let output_path = args
        .get("output_path")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "Missing required parameter: output_path".to_string()))?;

    let format = args
        .get("format")
        .and_then(|v| v.as_str())
        .unwrap_or("text");
    let pages_str = args.get("pages").and_then(|v| v.as_str());
    let password = args.get("password").and_then(|v| v.as_str());
    let images = args
        .get("images")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let embed_images = args
        .get("embed_images")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    // Validate format
    if !matches!(format, "text" | "markdown" | "html" | "structured") {
        return Err((
            -32602,
            format!("Invalid format: {format}. Must be text, markdown, html, or structured"),
        ));
    }

    // Open document
    let mut doc =
        PdfDocument::open(file_path).map_err(|e| (-32603, format!("Failed to open PDF: {e}")))?;

    // Authenticate if password provided
    if let Some(pw) = password {
        let ok = doc
            .authenticate(pw.as_bytes())
            .map_err(|e| (-32603, format!("Authentication error: {e}")))?;
        if !ok {
            return Err((-32603, "Incorrect password".to_string()));
        }
    }

    // Determine pages
    let page_count = doc
        .page_count()
        .map_err(|e| (-32603, format!("Failed to get page count: {e}")))?;
    let page_indices = match pages_str {
        Some(s) => parse_page_ranges(s)?,
        None => (0..page_count).collect(),
    };

    // Validate page indices
    for &idx in &page_indices {
        if idx >= page_count {
            return Err((
                -32602,
                format!("Page {} out of range (document has {} pages)", idx + 1, page_count),
            ));
        }
    }

    // Build conversion options for markdown/html
    let output_dir = Path::new(output_path)
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();
    let opts = ConversionOptions {
        embed_images,
        image_output_dir: if !embed_images {
            Some(output_dir.to_string_lossy().into_owned())
        } else {
            None
        },
        ..Default::default()
    };

    // Extract content
    let content = extract_pages(&mut doc, &page_indices, format, &opts)?;

    // Write output
    if let Some(parent) = Path::new(output_path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| (-32603, format!("Failed to create output directory: {e}")))?;
        }
    }
    fs::write(output_path, &content)
        .map_err(|e| (-32603, format!("Failed to write output: {e}")))?;

    // Extract images to files if requested
    let mut images_extracted = 0;
    if images {
        let img_dir = output_dir.join(
            Path::new(output_path)
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .as_ref(),
        );
        fs::create_dir_all(&img_dir)
            .map_err(|e| (-32603, format!("Failed to create image directory: {e}")))?;

        for &page_idx in &page_indices {
            match doc.extract_images_to_files(page_idx, &img_dir, None, Some(images_extracted + 1))
            {
                Ok(refs) => images_extracted += refs.len(),
                Err(e) => {
                    eprintln!("Warning: image extraction failed for page {}: {e}", page_idx + 1)
                },
            }
        }
    }

    let file_size = fs::metadata(output_path).map(|m| m.len()).unwrap_or(0);

    let mut message = format!(
        "Extracted {} page(s) as {} to {} ({} bytes)",
        page_indices.len(),
        format,
        output_path,
        file_size
    );
    if images_extracted > 0 {
        message.push_str(&format!(". {} image(s) saved.", images_extracted));
    }

    Ok(json!({
        "content": [{ "type": "text", "text": message }]
    }))
}

fn extract_pages(
    doc: &mut PdfDocument,
    page_indices: &[usize],
    format: &str,
    opts: &ConversionOptions,
) -> Result<String, (i32, String)> {
    // `structured` returns one JSON document for the whole request (an array of
    // per-page StructuredPage), not concatenated text — handle it up front.
    if format == "structured" {
        let mut pages = Vec::with_capacity(page_indices.len());
        for &idx in page_indices {
            let structured = doc.extract_structured(idx).map_err(|e| {
                (-32603, format!("Structured extraction failed on page {}: {e}", idx + 1))
            })?;
            pages.push(json!({
                "page": idx + 1,
                "structured": serde_json::to_value(&structured).unwrap(),
            }));
        }
        return Ok(serde_json::to_string_pretty(&json!({ "pages": pages })).unwrap());
    }

    let mut parts = Vec::with_capacity(page_indices.len());

    for &idx in page_indices {
        let part = match format {
            "text" => doc.extract_text(idx).map_err(|e| {
                (-32603, format!("Text extraction failed on page {}: {e}", idx + 1))
            })?,
            "markdown" => doc.to_markdown(idx, opts).map_err(|e| {
                (-32603, format!("Markdown conversion failed on page {}: {e}", idx + 1))
            })?,
            "html" => doc.to_html(idx, opts).map_err(|e| {
                (-32603, format!("HTML conversion failed on page {}: {e}", idx + 1))
            })?,
            _ => unreachable!(),
        };
        parts.push(part);
    }

    let separator = match format {
        "text" => "\n\n",
        "markdown" => "\n---\n\n",
        "html" => "\n",
        _ => unreachable!(),
    };

    Ok(parts.join(separator))
}

/// Parse page range strings like "1-3,7,10-12" into 0-indexed page numbers.
fn parse_page_ranges(input: &str) -> Result<Vec<usize>, (i32, String)> {
    let mut pages = Vec::new();

    for part in input.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }

        if let Some((start, end)) = part.split_once('-') {
            let start: usize = start
                .trim()
                .parse()
                .map_err(|_| (-32602, format!("Invalid page number: '{}'", start.trim())))?;
            let end: usize = end
                .trim()
                .parse()
                .map_err(|_| (-32602, format!("Invalid page number: '{}'", end.trim())))?;

            if start == 0 || end == 0 {
                return Err((-32602, "Page numbers start at 1".to_string()));
            }
            if start > end {
                return Err((-32602, format!("Invalid range: {start}-{end} (start > end)")));
            }

            for p in start..=end {
                pages.push(p - 1);
            }
        } else {
            let page: usize = part
                .parse()
                .map_err(|_| (-32602, format!("Invalid page number: '{part}'")))?;
            if page == 0 {
                return Err((-32602, "Page numbers start at 1".to_string()));
            }
            pages.push(page - 1);
        }
    }

    if pages.is_empty() {
        return Err((-32602, "No page numbers specified".to_string()));
    }

    Ok(pages)
}

fn open_authed(args: &Value) -> Result<PdfDocument, (i32, String)> {
    let file_path = args
        .get("file_path")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "Missing required parameter: file_path".to_string()))?;
    let password = args.get("password").and_then(|v| v.as_str());
    let doc =
        PdfDocument::open(file_path).map_err(|e| (-32603, format!("Failed to open PDF: {e}")))?;
    if let Some(pw) = password {
        let ok = doc
            .authenticate(pw.as_bytes())
            .map_err(|e| (-32603, format!("Authentication error: {e}")))?;
        if !ok {
            return Err((-32603, "Incorrect password".to_string()));
        }
    }
    Ok(doc)
}

/// MCP `classify` tool (#517): cheap per-page text-vs-OCR
/// classification (no OCR/rasterisation) → JSON `DocumentClassification`
/// as the text content, so an agent can branch on `pages_needing_ocr`.
pub fn classify(args: &Value) -> Result<Value, (i32, String)> {
    let doc = open_authed(args)?;
    let cls = doc
        .classify_document()
        .map_err(|e| (-32603, format!("classify failed: {e}")))?;
    let text = serde_json::to_string(&cls).map_err(|e| (-32603, e.to_string()))?;
    Ok(json!({ "content": [{ "type": "text", "text": text }] }))
}

/// MCP `auto` tool (#517): auto-extract text — per-page text-vs-OCR
/// routing with graceful native fallback (never the opaque OCR error,
/// #513). `format`: `text` (assembled) or `json` (rich
/// `PageExtraction[]` with per-region bbox + typed reasons).
pub fn auto(args: &Value) -> Result<Value, (i32, String)> {
    let format = args
        .get("format")
        .and_then(|v| v.as_str())
        .unwrap_or("text");
    if !matches!(format, "text" | "json") {
        return Err((-32602, format!("Invalid format: {format}. Must be text or json")));
    }
    let doc = open_authed(args)?;
    let n = doc
        .page_count()
        .map_err(|e| (-32603, format!("page_count failed: {e}")))?;
    let ae = pdf_oxide::extractors::AutoExtractor::new();
    let text = if format == "json" {
        let mut v = Vec::with_capacity(n);
        for p in 0..n {
            v.push(
                ae.extract_page(&doc, p)
                    .map_err(|e| (-32603, format!("extract_page {p}: {e}")))?,
            );
        }
        serde_json::to_string(&v).map_err(|e| (-32603, e.to_string()))?
    } else {
        let mut s = String::new();
        for p in 0..n {
            if p > 0 {
                s.push_str("\n\n");
            }
            s.push_str(
                &ae.extract_text(&doc, p)
                    .map_err(|e| (-32603, format!("extract_text {p}: {e}")))?,
            );
        }
        s
    };
    Ok(json!({ "content": [{ "type": "text", "text": text }] }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_path(name: &str) -> PathBuf {
        // Tests run from the pdf_oxide_mcp/ directory, fixtures are in parent
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.pop(); // up from pdf_oxide_mcp/
        p.push("tests/fixtures");
        p.push(name);
        p
    }

    #[test]
    fn test_parse_page_ranges_single() {
        assert_eq!(parse_page_ranges("1").unwrap(), vec![0]);
        assert_eq!(parse_page_ranges("5").unwrap(), vec![4]);
    }

    #[test]
    fn test_parse_page_ranges_range() {
        assert_eq!(parse_page_ranges("1-3").unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn test_parse_page_ranges_mixed() {
        assert_eq!(parse_page_ranges("1-3,7,10-12").unwrap(), vec![0, 1, 2, 6, 9, 10, 11]);
    }

    #[test]
    fn test_parse_page_ranges_zero_rejected() {
        assert!(parse_page_ranges("0").is_err());
    }

    #[test]
    fn test_parse_page_ranges_invalid_range() {
        assert!(parse_page_ranges("5-3").is_err());
    }

    #[test]
    fn test_parse_page_ranges_empty() {
        assert!(parse_page_ranges("").is_err());
    }

    #[test]
    fn test_parse_page_ranges_non_numeric() {
        assert!(parse_page_ranges("abc").is_err());
    }

    #[test]
    fn test_extract_text_to_file() {
        let pdf = fixture_path("simple.pdf");
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.txt");

        let args = json!({
            "file_path": pdf.to_str().unwrap(),
            "output_path": out.to_str().unwrap(),
            "format": "text"
        });

        let result = run(&args).expect("extract should succeed");
        assert!(result["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("1 page(s)"));
        assert!(out.exists());
    }

    // The `structured` format must expose `extract_structured` —
    // StructuredPage JSON with typed regions and per-region column_index.
    #[test]
    fn test_extract_structured_to_file() {
        let pdf = fixture_path("multi_column_table.pdf");
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.json");

        let args = json!({
            "file_path": pdf.to_str().unwrap(),
            "output_path": out.to_str().unwrap(),
            "format": "structured"
        });

        run(&args).expect("structured extract should succeed");
        assert!(out.exists());
        let written = std::fs::read_to_string(&out).unwrap();
        let parsed: Value = serde_json::from_str(&written).expect("output is valid JSON");
        let pages = parsed["pages"].as_array().expect("pages array");
        assert!(!pages.is_empty(), "at least one page");
        // The StructuredPage shape: regions carrying kind + column_index.
        let regions = pages[0]["structured"]["regions"]
            .as_array()
            .expect("regions array");
        assert!(!regions.is_empty(), "page has typed regions");
        assert!(regions[0].get("kind").is_some(), "region has a RegionRole kind");
        assert!(
            regions
                .iter()
                .all(|r| r.as_object().unwrap().contains_key("column_index")),
            "every region carries a column_index field"
        );
    }

    #[test]
    fn test_extract_rejects_unknown_format() {
        let pdf = fixture_path("simple.pdf");
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.txt");
        let args = json!({
            "file_path": pdf.to_str().unwrap(),
            "output_path": out.to_str().unwrap(),
            "format": "bogus"
        });
        assert!(run(&args).is_err(), "unknown format must be rejected");
    }

    #[test]
    fn test_extract_markdown_to_file() {
        let pdf = fixture_path("simple.pdf");
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.md");

        let args = json!({
            "file_path": pdf.to_str().unwrap(),
            "output_path": out.to_str().unwrap(),
            "format": "markdown"
        });

        let result = run(&args).expect("extract should succeed");
        assert!(result["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("markdown"));
        assert!(out.exists());
    }

    #[test]
    fn test_extract_html_to_file() {
        let pdf = fixture_path("simple.pdf");
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.html");

        let args = json!({
            "file_path": pdf.to_str().unwrap(),
            "output_path": out.to_str().unwrap(),
            "format": "html"
        });

        let result = run(&args).expect("extract should succeed");
        assert!(result["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("html"));
        assert!(out.exists());
    }

    #[test]
    fn test_extract_default_format_is_text() {
        let pdf = fixture_path("simple.pdf");
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.txt");

        let args = json!({
            "file_path": pdf.to_str().unwrap(),
            "output_path": out.to_str().unwrap()
        });

        let result = run(&args).expect("extract should succeed");
        assert!(result["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("text"));
    }

    #[test]
    fn test_extract_page_out_of_range() {
        let pdf = fixture_path("simple.pdf");
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.txt");

        let args = json!({
            "file_path": pdf.to_str().unwrap(),
            "output_path": out.to_str().unwrap(),
            "pages": "999"
        });

        let err = run(&args).unwrap_err();
        assert_eq!(err.0, -32602);
        assert!(err.1.contains("out of range"));
    }

    #[test]
    fn test_extract_creates_parent_dirs() {
        let pdf = fixture_path("simple.pdf");
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("nested/deep/out.txt");

        let args = json!({
            "file_path": pdf.to_str().unwrap(),
            "output_path": out.to_str().unwrap()
        });

        run(&args).expect("should create nested dirs");
        assert!(out.exists());
    }

    #[test]
    fn test_extract_nonexistent_pdf() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.txt");

        let args = json!({
            "file_path": "/does/not/exist.pdf",
            "output_path": out.to_str().unwrap()
        });

        let err = run(&args).unwrap_err();
        assert_eq!(err.0, -32603);
        assert!(err.1.contains("Failed to open PDF"));
    }

    #[test]
    fn test_extract_invalid_format() {
        let pdf = fixture_path("simple.pdf");
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.txt");

        let args = json!({
            "file_path": pdf.to_str().unwrap(),
            "output_path": out.to_str().unwrap(),
            "format": "csv"
        });

        let err = run(&args).unwrap_err();
        assert_eq!(err.0, -32602);
        assert!(err.1.contains("csv"));
    }

    #[test]
    fn test_missing_file_path() {
        let args = json!({ "output_path": "/tmp/out.txt" });
        let err = run(&args).unwrap_err();
        assert_eq!(err.0, -32602);
        assert!(err.1.contains("file_path"));
    }

    #[test]
    fn test_missing_output_path() {
        let args = json!({ "file_path": "test.pdf" });
        let err = run(&args).unwrap_err();
        assert_eq!(err.0, -32602);
        assert!(err.1.contains("output_path"));
    }
}
