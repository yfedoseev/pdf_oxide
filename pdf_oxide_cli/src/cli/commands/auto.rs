//! `pdf-oxide auto <file> [--format text|json]` — auto-extract text:
//! per-page text-vs-OCR routing with graceful native fallback (never
//! the opaque OCR error — #513). `text` = assembled; `json` = rich
//! per-region `PageExtraction` with typed reasons (#517).
//!
//! Strictly additive — the existing `text` command is untouched.

use pdf_oxide::extractors::AutoExtractor;
use std::path::Path;

pub fn run(
    file: &Path,
    format: &str,
    pages: Option<&str>,
    output: Option<&Path>,
    password: Option<&str>,
    _json: bool,
) -> pdf_oxide::Result<()> {
    let doc = super::open_doc(file, password)?;
    let page_count = doc.page_count()?;
    let indices = super::resolve_pages(pages, page_count)?;
    let ae = AutoExtractor::new();

    let body = if format == "json" {
        let mut pages_json = Vec::with_capacity(indices.len());
        for p in &indices {
            pages_json.push(ae.extract_page(&doc, *p)?);
        }
        serde_json::to_string_pretty(&pages_json)
            .map_err(|e| pdf_oxide::Error::InvalidOperation(e.to_string()))?
    } else {
        let mut s = String::new();
        for (i, p) in indices.iter().enumerate() {
            if i > 0 {
                s.push_str("\n\n");
            }
            s.push_str(&ae.extract_text(&doc, *p)?);
        }
        s
    };

    match output {
        Some(path) => {
            std::fs::write(path, body.as_bytes())?;
            eprintln!("Wrote {}", path.display());
        },
        None => println!("{body}"),
    }
    Ok(())
}
