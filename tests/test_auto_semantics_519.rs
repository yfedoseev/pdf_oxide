//! Semantic guarantees of the v0.3.51 auto surface (surfaced by the
//! #519 cross-binding smoke — "runs without error" is not enough; the
//! *results* must meet expectations):
//!
//! 1. Auto's native path is byte-faithful to the canonical
//!    `extract_text` (auto must never silently re-segment/lose text).
//! 2. Clean-but-sparse, image-free text is `TextLayer` — NOT `Scanned`
//!    — and is NOT listed in `pages_needing_ocr`.
//! 3. A page whose native text is high-quality reports
//!    `Complete` / `NativeText`, never `PartialSuccess` /
//!    `OcrRequestedButUnavailable`, even if the classifier wanted OCR.

use pdf_oxide::document::PdfDocument;
use pdf_oxide::extractors::auto::{
    AutoExtractor, ExtractSource, ExtractionStatus, PageKind, ReasonCode,
};

const TEXT_PDF: &str = "tests/fixtures/1.pdf"; // ~17k chars, real text

fn tiny_text_pdf() -> Vec<u8> {
    // A short, perfectly clean, image-free text page.
    let stream = b"BT /F1 12 Tf 72 700 Td (Quarterly revenue grew twelve percent.) Tj ET\n";
    let mut p: Vec<u8> = Vec::new();
    macro_rules! push {
        ($s:expr) => {
            p.extend_from_slice($s)
        };
    }
    push!(b"%PDF-1.4\n");
    let o1 = p.len();
    push!(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    let o2 = p.len();
    push!(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let o3 = p.len();
    push!(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n");
    let o4 = p.len();
    push!(format!("4 0 obj\n<< /Length {} >>\nstream\n", stream.len()).as_bytes());
    push!(stream);
    push!(b"\nendstream\nendobj\n");
    let o5 = p.len();
    push!(b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n");
    let xo = p.len();
    push!(format!(
        "xref\n0 6\n0000000000 65535 f \r\n{o1:010} 00000 n \r\n{o2:010} 00000 n \r\n{o3:010} 00000 n \r\n{o4:010} 00000 n \r\n{o5:010} 00000 n \r\n"
    )
    .as_bytes());
    push!(format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xo}\n%%EOF\n").as_bytes());
    p
}

#[test]
fn auto_native_path_is_faithful_to_canonical_extract_text_519() {
    let doc = PdfDocument::open(TEXT_PDF).expect("open 1.pdf");
    let ae = AutoExtractor::new();
    let n = doc.page_count().expect("page_count");
    for p in 0..n {
        let canonical = doc.extract_text(p).expect("extract_text");
        let auto = ae.extract_text(&doc, p).expect("ae.extract_text");
        assert_eq!(
            canonical, auto,
            "auto native path diverged from canonical extract_text on page {p} \
             (auto must never re-segment/lose text)"
        );
    }
}

#[test]
fn clean_sparse_text_is_textlayer_not_scanned_519() {
    let doc = PdfDocument::from_bytes(tiny_text_pdf()).expect("open tiny text pdf");
    let cls = doc.classify_page(0).expect("classify_page");
    assert_eq!(
        cls.kind,
        PageKind::TextLayer,
        "a short clean image-free text page must be TextLayer, not {:?}",
        cls.kind
    );
    let dc = doc.classify_document().expect("classify_document");
    assert!(
        !dc.pages_needing_ocr.contains(&0),
        "a clean text page must NOT be flagged pages_needing_ocr (got {:?})",
        dc.pages_needing_ocr
    );
}

#[test]
fn complete_native_text_reports_complete_not_partial_519() {
    let doc = PdfDocument::from_bytes(tiny_text_pdf()).expect("open tiny text pdf");
    let pe = AutoExtractor::new()
        .extract_page(&doc, 0)
        .expect("extract_page");
    assert!(!pe.text.trim().is_empty(), "expected non-empty native text");
    assert_eq!(
        pe.status,
        ExtractionStatus::Complete,
        "high-quality native text must be Complete, not {:?} (reason {:?})",
        pe.status,
        pe.reason
    );
    assert!(
        matches!(pe.reason, ReasonCode::Ok | ReasonCode::NativeTextHighConfidence),
        "reason must be a success code, got {:?}",
        pe.reason
    );
    let src = &pe.regions[0].source;
    assert!(
        matches!(src, ExtractSource::NativeText),
        "source must be NativeText for a clean native extraction, got {src:?}"
    );
    assert!(!pe.ocr_used, "ocr_used must be false for native text");
}
