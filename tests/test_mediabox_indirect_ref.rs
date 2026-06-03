//! Regression tests for MediaBox/CropBox stored as indirect references.
//!
//! PDF spec ISO 32000-1 §7.3.10 states that any value may be a direct or an
//! indirect reference; the semantics are equivalent.  Before this fix, the
//! library crashed with "MediaBox not found or not an array" when a page dict
//! contained `/MediaBox 174 0 R` instead of a direct array.

use pdf_oxide::document::PdfDocument;

/// Build a minimal PDF with a given number of cross-reference entries and
/// write the cross-reference table + trailer.
fn write_xref_and_trailer(pdf: &mut Vec<u8>, offsets: &[usize]) {
    let xref_offset = pdf.len();
    let count = offsets.len();
    pdf.extend_from_slice(format!("xref\n0 {}\n", count).as_bytes());
    pdf.extend_from_slice(b"0000000000 65535 f \n");
    for &off in &offsets[1..] {
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    pdf.extend_from_slice(
        format!("trailer\n<</Size {} /Root 1 0 R>>\nstartxref\n{}\n%%EOF\n", count, xref_offset)
            .as_bytes(),
    );
}

/// Build a minimal 1-page PDF where `/MediaBox` in the page dict is an
/// indirect reference to a separate array object.
///
/// Structure:
///   1 0 obj  Catalog  → /Pages 2 0 R
///   2 0 obj  Pages    → /Kids [3 0 R] /Count 1
///   3 0 obj  Page     → /MediaBox 4 0 R  (indirect reference!)
///   4 0 obj  Array    → [0 0 612 792]
fn build_pdf_indirect_mediabox() -> Vec<u8> {
    let mut pdf = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.4\n");

    let off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n");

    let off2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n");

    // Page dict: MediaBox is an indirect reference (4 0 R), NOT a direct array.
    let off3 = pdf.len();
    pdf.extend_from_slice(
        b"3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox 4 0 R /Resources <<>>>>\nendobj\n",
    );

    // The actual MediaBox array stored as its own object.
    let off4 = pdf.len();
    pdf.extend_from_slice(b"4 0 obj\n[0 0 612 792]\nendobj\n");

    write_xref_and_trailer(&mut pdf, &[0, off1, off2, off3, off4]);
    pdf
}

/// Build a minimal 1-page PDF where the *parent Pages node* has `/MediaBox`
/// as an indirect reference.  The page itself has no MediaBox entry, so it
/// must inherit it from the parent — and the inherited value is still an
/// indirect reference object.
///
/// Structure:
///   1 0 obj  Catalog  → /Pages 2 0 R
///   2 0 obj  Pages    → /Kids [3 0 R] /Count 1  /MediaBox 4 0 R  (indirect!)
///   3 0 obj  Page     → (no MediaBox — inherits from parent)
///   4 0 obj  Array    → [0 0 595 842]
fn build_pdf_inherited_indirect_mediabox() -> Vec<u8> {
    let mut pdf = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.4\n");

    let off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n");

    // Pages node carries the MediaBox as an indirect reference.
    let off2 = pdf.len();
    pdf.extend_from_slice(
        b"2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1 /MediaBox 4 0 R>>\nendobj\n",
    );

    // Page dict has NO MediaBox — relies on inheritance from the Pages node.
    let off3 = pdf.len();
    pdf.extend_from_slice(b"3 0 obj\n<</Type /Page /Parent 2 0 R /Resources <<>>>>\nendobj\n");

    // The actual MediaBox array — A4 dimensions.
    let off4 = pdf.len();
    pdf.extend_from_slice(b"4 0 obj\n[0 0 595 842]\nendobj\n");

    write_xref_and_trailer(&mut pdf, &[0, off1, off2, off3, off4]);
    pdf
}

/// Build a minimal 1-page PDF where `/MediaBox` is a direct array (the normal
/// case).  This is the sanity-check that the common path still works.
fn build_pdf_direct_mediabox() -> Vec<u8> {
    let mut pdf = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.4\n");

    let off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n");

    let off2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n");

    // Direct MediaBox array — the conventional form.
    let off3 = pdf.len();
    pdf.extend_from_slice(
        b"3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources <<>>>>\nendobj\n",
    );

    write_xref_and_trailer(&mut pdf, &[0, off1, off2, off3]);
    pdf
}

/// Build a minimal 1-page PDF where each *element* of the `/MediaBox` array is
/// itself an indirect reference (pdf.js issue7872):
///   3 0 obj  Page  → /MediaBox [4 0 R 5 0 R 6 0 R 7 0 R]
///   4..7 0 obj      → 0, 0, 250, 50
/// Before the fix the per-element references read as 0.0, collapsing the page
/// to a zero-area box that clipped away all text.
fn build_pdf_per_element_indirect_mediabox() -> Vec<u8> {
    let mut pdf = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.4\n");

    let off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n");

    let off2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n");

    let off3 = pdf.len();
    pdf.extend_from_slice(
        b"3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [4 0 R 5 0 R 6 0 R 7 0 R] /Resources <<>>>>\nendobj\n",
    );

    let off4 = pdf.len();
    pdf.extend_from_slice(b"4 0 obj\n0\nendobj\n");
    let off5 = pdf.len();
    pdf.extend_from_slice(b"5 0 obj\n0\nendobj\n");
    let off6 = pdf.len();
    pdf.extend_from_slice(b"6 0 obj\n250\nendobj\n");
    let off7 = pdf.len();
    pdf.extend_from_slice(b"7 0 obj\n50\nendobj\n");

    write_xref_and_trailer(&mut pdf, &[0, off1, off2, off3, off4, off5, off6, off7]);
    pdf
}

/// Build a minimal 1-page PDF where `/CropBox` is stored as an indirect reference.
fn build_pdf_indirect_cropbox() -> Vec<u8> {
    let mut pdf = Vec::new();
    pdf.extend_from_slice(b"%PDF-1.4\n");

    let off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n");

    let off2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n");

    // Page dict: direct MediaBox, indirect CropBox.
    let off3 = pdf.len();
    pdf.extend_from_slice(
        b"3 0 obj\n<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /CropBox 4 0 R /Resources <<>>>>\nendobj\n",
    );

    // Crop to the top-left quadrant.
    let off4 = pdf.len();
    pdf.extend_from_slice(b"4 0 obj\n[0 396 306 792]\nendobj\n");

    write_xref_and_trailer(&mut pdf, &[0, off1, off2, off3, off4]);
    pdf
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// A PDF with `/MediaBox 4 0 R` (indirect reference) must not crash.
/// `get_page_media_box` must return the correct dimensions.
#[test]
fn test_indirect_mediabox_direct_on_page() {
    let pdf = build_pdf_indirect_mediabox();
    let doc = PdfDocument::from_bytes(pdf).expect("PDF should parse successfully");

    let (llx, lly, urx, ury) = doc
        .get_page_media_box(0)
        .expect("get_page_media_box must succeed for indirect MediaBox");

    assert_eq!(
        (llx, lly, urx, ury),
        (0.0, 0.0, 612.0, 792.0),
        "Indirect MediaBox [0 0 612 792] should be resolved correctly"
    );
}

/// A PDF where the *parent Pages node* carries an indirect MediaBox reference
/// that is inherited by child pages.
#[test]
fn test_indirect_mediabox_inherited_from_pages_node() {
    let pdf = build_pdf_inherited_indirect_mediabox();
    let doc = PdfDocument::from_bytes(pdf).expect("PDF should parse successfully");

    let (llx, lly, urx, ury) = doc
        .get_page_media_box(0)
        .expect("get_page_media_box must succeed for inherited indirect MediaBox");

    assert_eq!(
        (llx, lly, urx, ury),
        (0.0, 0.0, 595.0, 842.0),
        "Inherited indirect MediaBox [0 0 595 842] should be resolved correctly"
    );
}

/// A PDF where each MediaBox array element is an indirect reference
/// (`/MediaBox [4 0 R 5 0 R 6 0 R 7 0 R]`, pdf.js issue7872) must resolve every
/// element to its true value, not collapse to a zero-area box.
#[test]
fn test_per_element_indirect_mediabox() {
    let pdf = build_pdf_per_element_indirect_mediabox();
    let doc = PdfDocument::from_bytes(pdf).expect("PDF should parse successfully");

    let (llx, lly, urx, ury) = doc
        .get_page_media_box(0)
        .expect("get_page_media_box must succeed for per-element indirect MediaBox");

    assert_eq!(
        (llx, lly, urx, ury),
        (0.0, 0.0, 250.0, 50.0),
        "Per-element indirect MediaBox [4 0 R 5 0 R 6 0 R 7 0 R] -> [0 0 250 50] \
         must resolve each element instead of reading references as 0.0"
    );
}

/// Sanity check: a PDF with a direct MediaBox array still works correctly.
#[test]
fn test_direct_mediabox_still_works() {
    let pdf = build_pdf_direct_mediabox();
    let doc = PdfDocument::from_bytes(pdf).expect("PDF should parse successfully");

    let (llx, lly, urx, ury) = doc
        .get_page_media_box(0)
        .expect("get_page_media_box must succeed for direct MediaBox");

    assert_eq!(
        (llx, lly, urx, ury),
        (0.0, 0.0, 612.0, 792.0),
        "Direct MediaBox [0 0 612 792] should work as before"
    );
}

/// A PDF with `/CropBox 4 0 R` (indirect reference) must be handled gracefully.
/// This test uses `get_page_count` to confirm the document was loaded and does
/// not panic; CropBox resolution is exercised through `get_page_media_box`
/// (which parses the page dict) plus a simple page-count sanity check.
#[test]
fn test_indirect_cropbox_does_not_crash() {
    let pdf = build_pdf_indirect_cropbox();
    let doc = PdfDocument::from_bytes(pdf).expect("PDF should parse successfully");

    // Must not crash when accessing the page.
    let (llx, lly, urx, ury) = doc
        .get_page_media_box(0)
        .expect("get_page_media_box must succeed even when CropBox is indirect");

    assert_eq!(
        (llx, lly, urx, ury),
        (0.0, 0.0, 612.0, 792.0),
        "MediaBox should still be [0 0 612 792] regardless of indirect CropBox"
    );

    assert_eq!(doc.page_count().unwrap(), 1);
}
