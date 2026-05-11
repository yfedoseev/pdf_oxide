//! Rendering a real-world external PDF fixture must complete without error.
//!
//! Guards the render pipeline against regressions on documents that
//! exercise uncommon but valid PDF constructs. The fixture lives in the
//! external `pdf_oxide_tests` corpus; the test skips gracefully when it
//! is not present.

#[cfg(feature = "rendering")]
#[test]
fn issue_395_user_pdf_renders_without_error() {
    use pdf_oxide::document::PdfDocument;
    use pdf_oxide::rendering::{render_page, RenderOptions};

    let Ok(home) = std::env::var("HOME") else {
        return;
    };
    let path = std::path::PathBuf::from(home)
        .join("projects/pdf_oxide_tests/pdfs_issue_regression/issue_395_csharp_render.pdf");
    if !path.exists() {
        eprintln!("Skipping: {} not found", path.display());
        return;
    }

    let doc = PdfDocument::open(&path).expect("open #395 fixture");
    let n_pages = doc.page_count().expect("page count");
    assert!(n_pages > 0, "fixture should have at least one page");

    let opts = RenderOptions::with_dpi(150);
    let img = render_page(&doc, 0, &opts).expect(
        "rendering page 0 of #395 fixture must succeed — was emitting an FFI error code that \
         the C# binding mismapped to SignatureException [8500]",
    );
    assert!(!img.data.is_empty(), "rendered image must have bytes");
    assert!(img.width > 0 && img.height > 0);
}
