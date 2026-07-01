//! Regression tests for running-artifact false positives where a header
//! recurs with a *varying* leading number but the surrounding text is
//! substantive content, not a folio — so it must still be kept on the
//! page where it first appears (same as any other first-occurrence
//! text).
//!
//! Two real cases motivated this:
//! - IRS_Form_1120_2024.pdf p0: "1a Consolidated return  (attach Form
//!   851)" — a form line-item label. Schedule pages renumber the same
//!   line, so its normalised signature ("#a Consolidated return
//!   (attach Form #)") is classified as *varying*, but "1a" is a line
//!   item label, not a folio.
//! - A numbered section heading like "4. Discussion" printed at the top
//!   of each page — the leading number is a section ordinal, not a
//!   page number.

mod common;
use common::build_pdf_with_page_extras;
use pdf_oxide::PdfDocument;

/// Places `header` text at y=750 — inside the top 12% band on a 792pt
/// page (band starts at 792 - 792*0.12 = 697.44).
fn header_line(header: &str) -> String {
    format!("BT /F1 12 Tf 1 0 0 1 72 750 Tm ({header}) Tj ET\n")
}

#[test]
fn varying_line_item_label_kept_on_first_page() {
    // Recurs on every page with a different leading digit each time
    // ("1a", "2a", "3a"), so its normalised signature is classified as
    // varying — but it's a form line-item label, not a folio.
    let labels = [
        "1a Consolidated return  (attach Form 851)",
        "2a Consolidated return  (attach Form 851)",
        "3a Consolidated return  (attach Form 851)",
    ];
    let bytes = build_pdf_with_page_extras(3, |i| header_line(labels[i]));
    let doc = PdfDocument::from_bytes(bytes).unwrap();

    let p0 = doc.extract_text(0).unwrap();
    assert!(p0.contains("Body text placeholder"), "page 0 body missing: {p0:?}");
    assert!(
        p0.contains("1a Consolidated return"),
        "form line-item label '1a Consolidated return...' is substantive \
         content, not a folio, and must survive on the page it first \
         appears on; got {p0:?}"
    );
}

#[test]
fn numbered_section_heading_kept_on_first_page() {
    // Recurs on every page with a different leading number each time
    // ("4.", "5.", "6."), so its normalised signature is classified as
    // varying — but the number is a section ordinal, not a page number.
    let headings = ["4. Discussion", "5. Discussion", "6. Discussion"];
    let bytes = build_pdf_with_page_extras(3, |i| header_line(headings[i]));
    let doc = PdfDocument::from_bytes(bytes).unwrap();

    let p0 = doc.extract_text(0).unwrap();
    assert!(p0.contains("Body text placeholder"), "page 0 body missing: {p0:?}");
    assert!(
        p0.contains("4. Discussion"),
        "numbered section heading '4. Discussion' is substantive content, \
         not a folio, and must survive on the page it first appears on; \
         got {p0:?}"
    );
}
