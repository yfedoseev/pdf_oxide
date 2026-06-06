// PDF/A, PDF/X, PDF/UA compliance validation (Rust parity with the
// python/javascript/go/csharp `compliance_validation` showcases).
//
// Run: cargo run --example showcase_compliance_validation

use pdf_oxide::compliance::{
    validate_pdf_a, validate_pdf_ua, validate_pdf_x, PdfALevel, PdfUaLevel, PdfXLevel,
};
use pdf_oxide::error::Result;
use pdf_oxide::writer::DocumentBuilder;
use pdf_oxide::PdfDocument;

fn main() -> Result<()> {
    let mut builder = DocumentBuilder::new();
    builder
        .letter_page()
        .font("Helvetica", 12.0)
        .at(72.0, 720.0)
        .heading(1, "Compliance Validation")
        .at(72.0, 690.0)
        .paragraph("Testing PDF/A, PDF/X, and PDF/UA compliance validators.")
        .done();
    let pdf_bytes = builder.build()?;

    println!("Validating PDF/A-2b compliance...");
    let mut doc = PdfDocument::from_bytes(pdf_bytes.clone())?;
    let a = validate_pdf_a(&mut doc, PdfALevel::A2b)?;
    println!(
        "  is_compliant: {}  errors: {}  warnings: {}",
        a.is_compliant,
        a.errors.len(),
        a.warnings.len()
    );

    println!("Validating PDF/X-4 compliance...");
    let mut doc = PdfDocument::from_bytes(pdf_bytes.clone())?;
    let x = validate_pdf_x(&mut doc, PdfXLevel::X4)?;
    println!("  is_compliant: {}  errors: {}", x.is_compliant, x.errors.len());

    println!("Validating PDF/UA-1 compliance...");
    let mut doc = PdfDocument::from_bytes(pdf_bytes)?;
    let ua = validate_pdf_ua(&mut doc, PdfUaLevel::Ua1)?;
    println!("  is_compliant: {}  errors: {}", ua.is_compliant, ua.errors.len());

    Ok(())
}
