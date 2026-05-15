// Office format conversion: PDF → DOCX / PPTX / XLSX — v0.3.41
//
// Demonstrates bidirectional office format conversion:
//   1. Build a PDF in memory
//   2. Export to DOCX bytes (PDF → DOCX)
//   3. Export to PPTX bytes (PDF → PPTX)
//   4. Export to XLSX bytes (PDF → XLSX)
//   5. Round-trip: import DOCX back to PDF (DOCX → PDF)
//
// Run in CI: cargo run --example showcase_office_conversion
// Any panic fails the CI job.

use pdf_oxide::{
    converters::office::OfficeConverter,
    document::PdfDocument,
    writer::{DocumentBuilder, PageSize},
};

fn build_sample_pdf() -> Vec<u8> {
    let mut builder = DocumentBuilder::new();
    {
        let page = builder.page(PageSize::Letter);
        page.font("Helvetica", 14.0)
            .at(72.0, 720.0)
            .text("Office Conversion Demo")
            .font("Helvetica", 11.0)
            .at(72.0, 690.0)
            .text("This PDF will be exported to DOCX, PPTX, and XLSX formats.")
            .done();
    }
    builder.build().expect("builder failed")
}

fn main() {
    println!("=== Office format conversion showcase ===");

    let pdf_bytes = build_sample_pdf();
    println!("Built sample PDF: {} bytes", pdf_bytes.len());

    let doc = PdfDocument::from_bytes(pdf_bytes.clone()).expect("open PDF");

    // 1. PDF → DOCX
    let docx_bytes = doc.to_docx_bytes().expect("to_docx_bytes");
    assert!(docx_bytes.starts_with(b"PK"), "DOCX output is not a valid ZIP/DOCX");
    println!("PDF → DOCX: {} bytes — PASS", docx_bytes.len());

    // 2. PDF → PPTX
    let pptx_bytes = doc.to_pptx_bytes().expect("to_pptx_bytes");
    assert!(pptx_bytes.starts_with(b"PK"), "PPTX output is not a valid ZIP/PPTX");
    println!("PDF → PPTX: {} bytes — PASS", pptx_bytes.len());

    // 3. PDF → XLSX
    let xlsx_bytes = doc.to_xlsx_bytes().expect("to_xlsx_bytes");
    assert!(xlsx_bytes.starts_with(b"PK"), "XLSX output is not a valid ZIP/XLSX");
    println!("PDF → XLSX: {} bytes — PASS", xlsx_bytes.len());

    // Round-trips: office → PDF → office
    let docx_rt = OfficeConverter::new()
        .convert_docx_bytes(&docx_bytes)
        .expect("DOCX → PDF failed");
    assert!(docx_rt.starts_with(b"%PDF-"), "DOCX → PDF did not produce a valid PDF");
    let doc2 = PdfDocument::from_bytes(docx_rt).expect("re-open");
    let docx2 = doc2.to_docx_bytes().expect("PDF → DOCX (2nd)");
    assert!(docx2.starts_with(b"PK"), "DOCX round-trip output invalid");
    println!("DOCX → PDF → DOCX round-trip: {} bytes — PASS", docx2.len());

    let pptx_rt = OfficeConverter::new()
        .convert_pptx_bytes(&pptx_bytes)
        .expect("PPTX → PDF failed");
    assert!(pptx_rt.starts_with(b"%PDF-"), "PPTX → PDF did not produce a valid PDF");
    let doc3 = PdfDocument::from_bytes(pptx_rt).expect("re-open");
    let pptx2 = doc3.to_pptx_bytes().expect("PDF → PPTX (2nd)");
    assert!(pptx2.starts_with(b"PK"), "PPTX round-trip output invalid");
    println!("PPTX → PDF → PPTX round-trip: {} bytes — PASS", pptx2.len());

    let xlsx_rt = OfficeConverter::new()
        .convert_xlsx_bytes(&xlsx_bytes)
        .expect("XLSX → PDF failed");
    assert!(xlsx_rt.starts_with(b"%PDF-"), "XLSX → PDF did not produce a valid PDF");
    let doc4 = PdfDocument::from_bytes(xlsx_rt).expect("re-open");
    let xlsx2 = doc4.to_xlsx_bytes().expect("PDF → XLSX (2nd)");
    assert!(xlsx2.starts_with(b"PK"), "XLSX round-trip output invalid");
    println!("XLSX → PDF → XLSX round-trip: {} bytes — PASS", xlsx2.len());

    println!("\n=== All office conversion checks passed ===");
}
