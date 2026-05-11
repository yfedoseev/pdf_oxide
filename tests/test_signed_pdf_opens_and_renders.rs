//! Tests that a PDF containing digital signatures opens and renders correctly.
//! on 2026-04-21 (originally as a Reddit comment, same day):
//!
//! > This program in c# blows up with error:
//! > "PdfOxide.Exceptions.SignatureException:
//! >  '[8500] Signature error: Certificate loading, signing, or verification failed'"
//!
//! The bug was entirely in the C# binding — its `ExceptionMapper`
//! was offset-by-N against the Rust FFI error codes, so FFI code 8
//! (Unsupported) landed on `SignatureException` instead of
//! `UnsupportedFeatureException`. See commit 3bb271f1 for the fix.
//!
//! This Rust-level test locks in that **rendering u/gevorgter's
//! fixture succeeds at the core level** — so any future regression
//! is clearly the binding layer's problem, not the engine.

use pdf_oxide::document::PdfDocument;

#[test]
fn signed_pdf_opens_without_error() {
    let path = "tests/fixtures/issue_regressions/issue_395_render_signature_exception.pdf";
    let bytes = std::fs::read(path).expect("fixture missing");
    let doc = PdfDocument::from_bytes(bytes).expect("open");
    let count = doc.page_count().expect("page count");
    assert!(count > 0, "expected at least one page");
}

#[cfg(feature = "rendering")]
#[test]
fn signed_pdf_renders_first_page_without_error() {
    use pdf_oxide::rendering::{self, RenderOptions};

    let path = "tests/fixtures/issue_regressions/issue_395_render_signature_exception.pdf";
    let bytes = std::fs::read(path).expect("fixture missing");
    let doc = PdfDocument::from_bytes(bytes).expect("open");

    let opts = RenderOptions::with_dpi(72);
    let img = rendering::render_page(&doc, 0, &opts)
        .expect("render_page must not return Err for the regression fixture");

    assert!(img.data.len() > 128, "rendered image should be non-trivial");
    assert!(img.width > 0 && img.height > 0);

    // PNG magic — the format must be what we asked for.
    assert!(
        img.data.starts_with(&[0x89, 0x50, 0x4E, 0x47]),
        "expected PNG magic on default render"
    );
}
