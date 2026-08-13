//! `B`, `B*`, `b`, and `b*` (ISO 32000-1:2008 Table 60) paint a path with a
//! single combined fill-and-stroke operator instead of separate `f`/`S`
//! calls. Graphviz and other diagram/vector-graphics producers rely on this
//! form heavily (e.g. filled+outlined arrowheads and nodes): a path painted
//! with `B`/`B*`/`b*` was silently dropped from `extract_paths` because the
//! operator dispatch only recognized `b` (`CloseFillStroke`), leaving the
//! other three combined-paint operators unhandled — both in the page content
//! stream and inside Form XObjects.

use pdf_oxide::document::PdfDocument;

#[test]
fn fill_stroke_operator_is_extracted_from_page_content() {
    // `1 0 0 1 0 0 cm` no-op transform keeps this a simple rectangle;
    // `B` = close (implicit via re rect) + fill (nonzero) + stroke. `0 0 0
    // rg` sets a fill color first — `has_fill()` checks `fill_color.is_some()`,
    // and a path painted with no fill color set is a color-setting-omitted
    // fixture bug, not evidence the `B` dispatch itself is broken.
    let content: &[u8] = b"0 0 0 rg 1 w 50 50 100 75 re B";
    let doc = PdfDocument::from_bytes(build_minimal_pdf_raw(
        content,
        b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]",
    ))
    .expect("parse");
    let paths = doc.extract_paths(0).expect("paths");
    assert_eq!(paths.len(), 1, "the B-painted rectangle must be extracted");
    assert!(paths[0].has_fill(), "B fills the path");
    assert!(paths[0].has_stroke(), "B strokes the path");
}

#[test]
fn fill_stroke_even_odd_operator_is_extracted_from_page_content() {
    let content: &[u8] = b"0 0 0 rg 1 w 50 50 100 75 re B*";
    let doc = PdfDocument::from_bytes(build_minimal_pdf_raw(
        content,
        b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]",
    ))
    .expect("parse");
    let paths = doc.extract_paths(0).expect("paths");
    assert_eq!(paths.len(), 1, "the B*-painted rectangle must be extracted");
    assert!(paths[0].has_fill());
    assert!(paths[0].has_stroke());
}

#[test]
fn close_fill_stroke_even_odd_operator_is_extracted_from_page_content() {
    // `b*`: close path, fill even-odd, stroke.
    let content: &[u8] = b"0 0 0 rg 1 w 50 50 m 150 50 l 150 125 l b*";
    let doc = PdfDocument::from_bytes(build_minimal_pdf_raw(
        content,
        b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]",
    ))
    .expect("parse");
    let paths = doc.extract_paths(0).expect("paths");
    assert_eq!(paths.len(), 1, "the b*-painted triangle must be extracted");
    assert!(paths[0].has_fill());
    assert!(paths[0].has_stroke());
}

#[test]
fn fill_stroke_operator_is_extracted_from_form_xobject() {
    // Same rectangle, but painted inside a Form XObject invoked via `Do` —
    // exercises the second (nested-XObject) path-extraction dispatch loop.
    let content: &[u8] = b"q 1 0 0 1 0 0 cm /Fm1 Do Q";
    let doc =
        PdfDocument::from_bytes(build_form_xobject_pdf(content, b"0 0 0 rg 50 50 100 75 re B"))
            .expect("parse");
    let paths = doc.extract_paths(0).expect("paths");
    assert_eq!(
        paths.len(),
        1,
        "the B-painted rectangle inside the Form XObject must be extracted"
    );
    assert!(paths[0].has_fill());
    assert!(paths[0].has_stroke());
}

// ---------------------------------------------------------------------------
// Minimal raw PDF builders (same pattern as test_stroke_width_rendered_bbox.rs)
// ---------------------------------------------------------------------------

fn build_minimal_pdf_raw(content: &[u8], page_extra: &[u8]) -> Vec<u8> {
    let mut pdf = b"%PDF-1.4\n".to_vec();

    let off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    let off2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    let off3 = pdf.len();
    pdf.extend_from_slice(b"3 0 obj\n<< ");
    pdf.extend_from_slice(page_extra);
    pdf.extend_from_slice(b" /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n");

    let off4 = pdf.len();
    pdf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    let off5 = pdf.len();
    pdf.extend_from_slice(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>\nendobj\n",
    );

    let xref_pos = pdf.len();
    let offsets = [0usize, off1, off2, off3, off4, off5];
    pdf.extend_from_slice(format!("xref\n0 {}\n", offsets.len()).as_bytes());
    pdf.extend_from_slice(format!("{:010} 65535 f\r\n", 0).as_bytes());
    for &off in &offsets[1..] {
        pdf.extend_from_slice(format!("{:010} 00000 n\r\n", off).as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            offsets.len(),
            xref_pos
        )
        .as_bytes(),
    );
    pdf
}

/// A page whose content stream (`page_content`) invokes a single Form
/// XObject `/Fm1` whose own content stream is `xobject_content`.
fn build_form_xobject_pdf(page_content: &[u8], xobject_content: &[u8]) -> Vec<u8> {
    let mut pdf = b"%PDF-1.4\n".to_vec();

    let off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    let off2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    let off3 = pdf.len();
    pdf.extend_from_slice(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]");
    pdf.extend_from_slice(
        b" /Contents 4 0 R /Resources << /XObject << /Fm1 5 0 R >> >> >>\nendobj\n",
    );

    let off4 = pdf.len();
    pdf.extend_from_slice(
        format!("4 0 obj\n<< /Length {} >>\nstream\n", page_content.len()).as_bytes(),
    );
    pdf.extend_from_slice(page_content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    let off5 = pdf.len();
    pdf.extend_from_slice(
        format!(
            "5 0 obj\n<< /Type /XObject /Subtype /Form /BBox [0 0 612 792] /Length {} >>\nstream\n",
            xobject_content.len()
        )
        .as_bytes(),
    );
    pdf.extend_from_slice(xobject_content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    let xref_pos = pdf.len();
    let offsets = [0usize, off1, off2, off3, off4, off5];
    pdf.extend_from_slice(format!("xref\n0 {}\n", offsets.len()).as_bytes());
    pdf.extend_from_slice(format!("{:010} 65535 f\r\n", 0).as_bytes());
    for &off in &offsets[1..] {
        pdf.extend_from_slice(format!("{:010} 00000 n\r\n", off).as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            offsets.len(),
            xref_pos
        )
        .as_bytes(),
    );
    pdf
}
