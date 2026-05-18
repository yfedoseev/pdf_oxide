//! `pdf-oxide classify <file>` — cheap per-page text-vs-OCR
//! classification (no OCR, no rasterisation), printed as JSON
//! `DocumentClassification` (#517). The frozen cross-binding envelope.

use std::path::Path;

pub fn run(file: &Path, password: Option<&str>, _json: bool) -> pdf_oxide::Result<()> {
    let doc = super::open_doc(file, password)?;
    let cls = doc.classify_document()?;
    let out = serde_json::to_string_pretty(&cls)
        .map_err(|e| pdf_oxide::Error::InvalidOperation(e.to_string()))?;
    println!("{out}");
    Ok(())
}
