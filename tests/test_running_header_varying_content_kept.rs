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

use pdf_oxide::PdfDocument;

// ---------------- test helper: build_pdf_with_page_extras -------------------
//
// two fn used only by build_pdf_with_page_extras write one object each,
// recording its offset as they go:
// - `buf`      the buffer we're writing into
// - `off[id]`  start of object definition

// -- write a plain dictionary object --
fn obj(buf: &mut Vec<u8>, off: &mut [usize], id: usize, body: &str) {
    off[id] = buf.len();
    buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
}

// -- write a `stream` object - used here for page content --
fn stream(buf: &mut Vec<u8>, off: &mut [usize], id: usize, data: &[u8]) {
    off[id] = buf.len();
    buf.extend_from_slice(format!("{id} 0 obj\n<< /Length {} >>\nstream\n", data.len()).as_bytes());
    buf.extend_from_slice(data);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
}

/// Minimal single-page-content PDF builder: N pages, each with a body
/// paragraph plus arbitrary extra content-stream text supplied per page.
fn build_pdf_with_page_extras(
    page_count: usize,
    extra_per_page: impl Fn(usize) -> String,
) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 4 + page_count * 2];

    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");

    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");

    let kids: String = (0..page_count)
        .map(|i| format!("{} 0 R", 5 + i * 2))
        .collect::<Vec<_>>()
        .join(" ");
    obj(
        &mut buf,
        &mut off,
        2,
        &format!("<< /Type /Pages /Kids [{kids}] /Count {page_count} >>"),
    );

    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    );

    // --- One content stream + one page object, per page ---
    for i in 0..page_count {
        let content_id = 4 + i * 2; // 4, 6, 8, ...
        let page_id = 5 + i * 2; // 5, 7, 9, ...

        let content = format!(
            "BT /F1 12 Tf 1 0 0 1 72 400 Tm (Body text placeholder) Tj ET\n{}",
            extra_per_page(i)
        );
        stream(&mut buf, &mut off, content_id, content.as_bytes());

        obj(
            &mut buf,
            &mut off,
            page_id,
            &format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
                 /Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            ),
        );
    }

    let xref_off = buf.len();
    let total_objs = off.len();
    buf.extend_from_slice(format!("xref\n0 {}\n", total_objs).as_bytes());
    buf.extend_from_slice(b"0000000000 65535 f \n");
    for offset in &off[1..] {
        buf.extend_from_slice(format!("{:010} 00000 n \n", offset).as_bytes());
    }
    buf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            total_objs, xref_off
        )
        .as_bytes(),
    );
    buf
}

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
