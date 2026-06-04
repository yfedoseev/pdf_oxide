use pdf_oxide::geometry::Rect;
use pdf_oxide::layout::RectFilterMode;
use std::path::Path;

pub fn run(
    file: &Path,
    format: &str,
    area: Option<&str>,
    pages: Option<&str>,
    output: Option<&Path>,
    password: Option<&str>,
    json: bool,
) -> pdf_oxide::Result<()> {
    let doc = super::open_doc(file, password)?;
    let page_count = doc.page_count()?;
    let page_indices = super::resolve_pages(pages, page_count)?;

    let region = if let Some(area_str) = area {
        Some(parse_area(area_str)?)
    } else {
        None
    };

    // `structured` is inherently structured JSON (typed regions with column
    // assignment), so it is emitted as JSON regardless of the `--json` flag and
    // ignores `--area` (it operates on the whole page).
    if format == "structured" {
        let mut all_pages = Vec::new();
        for &page_idx in &page_indices {
            let structured = doc.extract_structured(page_idx)?;
            all_pages.push(serde_json::json!({
                "page": page_idx + 1,
                "structured": serde_json::to_value(&structured).unwrap(),
            }));
        }
        let json_out = serde_json::json!({
            "file": file.display().to_string(),
            "format": "structured",
            "pages": all_pages,
        });
        super::write_output(&serde_json::to_string_pretty(&json_out).unwrap(), output)?;
        return Ok(());
    }

    if json {
        let mut all_pages = Vec::new();
        for &page_idx in &page_indices {
            let page_data = match format {
                "words" => {
                    let words = if let Some(r) = region {
                        doc.extract_words_in_rect(page_idx, r, RectFilterMode::Intersects)?
                    } else {
                        doc.extract_words(page_idx)?
                    };
                    serde_json::to_value(words).unwrap()
                },
                "lines" => {
                    let lines = if let Some(r) = region {
                        doc.extract_text_lines_in_rect(page_idx, r, RectFilterMode::Intersects)?
                    } else {
                        doc.extract_text_lines(page_idx)?
                    };
                    serde_json::to_value(lines).unwrap()
                },
                _ => {
                    let text = if let Some(r) = region {
                        doc.extract_text_in_rect(page_idx, r, RectFilterMode::Intersects)?
                    } else {
                        doc.extract_text(page_idx)?
                    };
                    serde_json::json!(text)
                },
            };
            all_pages.push(serde_json::json!({
                "page": page_idx + 1,
                "content": page_data,
            }));
        }

        let json_out = serde_json::json!({
            "file": file.display().to_string(),
            "format": format,
            "area": area,
            "pages": all_pages,
        });
        super::write_output(&serde_json::to_string_pretty(&json_out).unwrap(), output)?;
    } else {
        let mut results: Vec<String> = Vec::new();
        for &page_idx in &page_indices {
            let text = match format {
                "words" => {
                    let words = if let Some(r) = region {
                        doc.extract_words_in_rect(page_idx, r, RectFilterMode::Intersects)?
                    } else {
                        doc.extract_words(page_idx)?
                    };
                    words
                        .iter()
                        .map(|w| w.text.as_str())
                        .collect::<Vec<_>>()
                        .join(" ")
                },
                "lines" => {
                    let lines = if let Some(r) = region {
                        doc.extract_text_lines_in_rect(page_idx, r, RectFilterMode::Intersects)?
                    } else {
                        doc.extract_text_lines(page_idx)?
                    };
                    lines
                        .iter()
                        .map(|l| l.text.as_str())
                        .collect::<Vec<_>>()
                        .join("\n")
                },
                _ => {
                    if let Some(r) = region {
                        doc.extract_text_in_rect(page_idx, r, RectFilterMode::Intersects)?
                    } else {
                        doc.extract_text(page_idx)?
                    }
                },
            };
            results.push(text);
        }
        let combined = results.join("\n\n---\n\n");
        super::write_output(&combined, output)?;
    }

    Ok(())
}

fn parse_area(s: &str) -> pdf_oxide::Result<Rect> {
    let parts: Vec<&str> = s.split(',').map(|p| p.trim()).collect();
    if parts.len() != 4 {
        return Err(pdf_oxide::Error::InvalidOperation(
            "Area must be provided as x,y,width,height".to_string(),
        ));
    }

    let x = parts[0].parse::<f32>().map_err(|_| {
        pdf_oxide::Error::InvalidOperation(format!("Invalid x coordinate: {}", parts[0]))
    })?;
    let y = parts[1].parse::<f32>().map_err(|_| {
        pdf_oxide::Error::InvalidOperation(format!("Invalid y coordinate: {}", parts[1]))
    })?;
    let w = parts[2]
        .parse::<f32>()
        .map_err(|_| pdf_oxide::Error::InvalidOperation(format!("Invalid width: {}", parts[2])))?;
    let h = parts[3]
        .parse::<f32>()
        .map_err(|_| pdf_oxide::Error::InvalidOperation(format!("Invalid height: {}", parts[3])))?;

    Ok(Rect::new(x, y, w, h))
}
