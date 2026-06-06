// Page extraction (Rust-idiomatic; uses DocumentEditor::extract_pages_to_bytes).
//
// Builds a 3-page PDF, then extracts pages 1 and 3 into a new PDF.
//
// Run: cargo run --example showcase_page_extraction

use pdf_oxide::editor::DocumentEditor;
use pdf_oxide::error::Result;
use pdf_oxide::writer::DocumentBuilder;
use pdf_oxide::PdfDocument;
use std::path::PathBuf;

fn main() -> Result<()> {
    let out_dir = PathBuf::from("target/examples_output/page_extraction");
    std::fs::create_dir_all(&out_dir)?;

    // Build a 3-page document.
    let mut builder = DocumentBuilder::new();
    for n in 1..=3 {
        builder
            .letter_page()
            .font("Helvetica", 12.0)
            .at(72.0, 720.0)
            .heading(1, &format!("Page {n}"))
            .at(72.0, 690.0)
            .paragraph(&format!("This is the content of page {n}."))
            .done();
    }
    let bytes = builder.build()?;
    println!(
        "Built {}-page PDF ({} bytes)",
        PdfDocument::from_bytes(bytes.clone())?.page_count()?,
        bytes.len()
    );

    // Extract pages 1 and 3 (0-based indices 0 and 2) into a new PDF.
    let mut editor = DocumentEditor::from_bytes(bytes)?;
    let extracted = editor.extract_pages_to_bytes(&[0, 2])?;
    let doc = PdfDocument::from_bytes(extracted.clone())?;
    println!(
        "Extracted pages [0, 2] → new PDF with {} pages ({} bytes)",
        doc.page_count()?,
        extracted.len()
    );

    let path = out_dir.join("extracted.pdf");
    std::fs::write(&path, &extracted)?;
    println!("Wrote {}", path.display());
    Ok(())
}
