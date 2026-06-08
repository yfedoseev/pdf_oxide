#[cfg(feature = "ocr")]
use pdf_oxide::document::PdfDocument;

#[cfg(feature = "ocr")]
#[test]
fn test_detect_page_type_text_page() {
    use pdf_oxide::ocr::{detect_page_type, PageType};
    let doc = PdfDocument::open("tests/fixtures/simple.pdf").unwrap();
    let page_type = detect_page_type(&doc, 0).unwrap();
    assert_eq!(page_type, PageType::NativeText);
}

#[cfg(feature = "ocr")]
#[test]
fn test_needs_ocr_text_page_false() {
    use pdf_oxide::ocr::needs_ocr;
    let doc = PdfDocument::open("tests/fixtures/simple.pdf").unwrap();
    let needs = needs_ocr(&doc, 0).unwrap();
    assert!(!needs, "Text-based PDF page should not need OCR");
}

/// detect_page_type uses extract_spans (not extract_text) to avoid infinite
/// recursion: extract_text -> needs_ocr -> detect_page_type -> extract_text.
/// If we reach the end without stack overflow, the guard works.
#[cfg(feature = "ocr")]
#[test]
fn test_detect_page_type_no_infinite_recursion() {
    use pdf_oxide::ocr::{detect_page_type, needs_ocr};
    let doc = PdfDocument::open("tests/fixtures/simple.pdf").unwrap();
    let _page_type = detect_page_type(&doc, 0).unwrap();
    let _needs = needs_ocr(&doc, 0).unwrap();
}
