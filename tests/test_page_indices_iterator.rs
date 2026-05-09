//! Tests for `PdfDocument::page_indices()`.
//!
//! Verifies idiomatic Rust iteration over page indices — a convenience
//! over `0..doc.page_count()?` that mirrors the C# `doc.Pages` /
//! Go `doc.Pages()` surface.

use pdf_oxide::document::PdfDocument;

#[test]
fn page_indices_yields_zero_to_page_count() {
    let doc = PdfDocument::open("tests/fixtures/simple.pdf").expect("open simple");
    let count = doc.page_count().expect("page count");
    let collected: Vec<usize> = doc.page_indices().collect();
    let expected: Vec<usize> = (0..count).collect();
    assert_eq!(collected, expected, "page_indices() must produce 0..page_count contiguous");
}

#[test]
fn page_indices_is_lazy_iterator_for() {
    // Smoke test the intended user-facing pattern.
    let doc = PdfDocument::open("tests/fixtures/simple.pdf").expect("open simple");
    let mut visited = 0_usize;
    for i in doc.page_indices() {
        let _ = doc.extract_text(i).expect("extract per page");
        visited += 1;
    }
    assert!(visited > 0, "must visit at least one page");
    assert_eq!(visited, doc.page_count().unwrap());
}
