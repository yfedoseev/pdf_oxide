//! `PdfDocument::open(path)` must populate `source_bytes`.
//!
//! Any API that re-reads the raw bytes (e.g. `compliance::convert_to_pdf_a`)
//! depends on `source_bytes` being non-empty after `open(path)`.
//! Guards against regressions where the in-memory copy was left empty.

use pdf_oxide::document::PdfDocument;

#[test]
fn open_path_populates_source_bytes() {
    // Use a small fixture that ships in-tree.
    let doc = PdfDocument::open("tests/fixtures/simple.pdf").expect("simple fixture opens");
    assert!(
        !doc.source_bytes.is_empty(),
        "source_bytes must be populated after `open(path)` so byte-consuming \
         APIs (convert_to_pdf_a, FFI pdf_document_get_source_bytes, etc.) \
         see the document content"
    );
    // Sanity: bytes start with the PDF header.
    assert_eq!(&doc.source_bytes[..5], b"%PDF-", "source_bytes should start with %PDF-");
}

#[test]
fn open_path_and_from_bytes_produce_equal_source_bytes() {
    let path = "tests/fixtures/simple.pdf";
    let bytes = std::fs::read(path).expect("read fixture");
    let from_path = PdfDocument::open(path).expect("open path");
    let from_bytes = PdfDocument::from_bytes(bytes.clone()).expect("from_bytes");
    assert_eq!(
        from_path.source_bytes, from_bytes.source_bytes,
        "open(path) should now hold the same bytes as from_bytes(read(path))"
    );
    // Both should also match the actual file contents.
    assert_eq!(from_path.source_bytes, bytes);
}
