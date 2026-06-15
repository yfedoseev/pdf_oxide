//! RW-1e functional regression: honor the Form XObject /BBox clip during text
//! extraction (ISO 32000-1:2008 §8.10.1 — a form's marks are clipped to its
//! /BBox). Some producers (pdfTeX \includegraphics of a figure PDF that retained
//! a full draft-galley page) paint a redundant copy of the article body OUTSIDE
//! the figure form's /BBox. A conformant renderer (pdfium/Acrobat/MuPDF) clips it
//! and shows it 0×; pdf_oxide used to walk the form stream without the clip and
//! emit that out-of-BBox text, duplicating the real page body (~+44% text on the
//! PMC8103263 real-academic case).
//!
//! Guarded behaviours:
//!  - text the form paints OUTSIDE its /BBox is dropped;
//!  - text the form paints INSIDE its /BBox is kept — even when it duplicates a
//!    word that also appears in the page content stream (the genuine figure-label
//!    case, e.g. tracemonkey's "compiled trace" flowchart labels) must NOT be
//!    deleted.
//!
//! Hand-built minimal PDF (no third-party file — project policy).

use pdf_oxide::PdfDocument;

/// Page (612×792) draws body text + invokes a Form XObject (Fm0) translated to
/// (100,400). The form has /BBox [0 0 200 100] → page-space clip [100,400,300,500].
/// Inside the box it draws "InsideBoxKeep duplicateword" (page y≈450 → kept);
/// outside it draws "OutsideBoxDrop" (form y=300 → page y≈700 → clipped).
fn form_bbox_pdf() -> Vec<u8> {
    let page_content = b"BT /F1 10 Tf 1 0 0 1 50 700 Tm (PageBodyText duplicateword here) Tj ET\n\
                         q 1 0 0 1 100 400 cm /Fm0 Do Q\n";
    let form_content = b"BT /F1 10 Tf 1 0 0 1 40 50 Tm (InsideBoxKeep duplicateword) Tj ET\n\
                         BT /F1 10 Tf 1 0 0 1 40 300 Tm (OutsideBoxDrop) Tj ET\n";

    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 7];
    let obj = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, body: &str| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };
    let raw_stream =
        |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, dict: &str, data: &[u8]| {
            off[id] = buf.len();
            buf.extend_from_slice(
                format!("{id} 0 obj\n<< {dict} /Length {} >>\nstream\n", data.len()).as_bytes(),
            );
            buf.extend_from_slice(data);
            buf.extend_from_slice(b"\nendstream\nendobj\n");
        };

    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 5 0 R >> /XObject << /Fm0 6 0 R >> >> /Contents 4 0 R >>",
    );
    raw_stream(&mut buf, &mut off, 4, "", page_content);
    obj(
        &mut buf,
        &mut off,
        5,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    );
    raw_stream(
        &mut buf,
        &mut off,
        6,
        "/Type /XObject /Subtype /Form /BBox [0 0 200 100] /Matrix [1 0 0 1 0 0] \
         /Resources << /Font << /F1 5 0 R >> >>",
        form_content,
    );

    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 7\n0000000000 65535 f \n");
    for id in 1..=6 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

fn count(haystack: &str, needle: &str) -> usize {
    haystack.matches(needle).count()
}

#[test]
fn out_of_bbox_form_text_is_clipped() {
    let doc = PdfDocument::from_bytes(form_bbox_pdf()).expect("parse");
    let text = doc.extract_text(0).expect("extract");
    assert!(text.contains("PageBodyText"), "page body missing:\n{text}");
    assert!(text.contains("InsideBoxKeep"), "in-BBox form text wrongly dropped:\n{text}");
    assert!(
        !text.contains("OutsideBoxDrop"),
        "out-of-BBox form text was NOT clipped (the duplication bug):\n{text}"
    );
}

#[test]
fn in_bbox_form_label_duplicating_page_word_is_kept() {
    // The figure-label regression guard: a form span INSIDE its BBox whose text
    // repeats a page word (here "duplicateword") must be preserved, not deleted.
    let doc = PdfDocument::from_bytes(form_bbox_pdf()).expect("parse");
    let text = doc.extract_text(0).expect("extract");
    assert!(
        count(&text, "duplicateword") >= 2,
        "in-BBox form label duplicating a page word was dropped (only {}× present):\n{text}",
        count(&text, "duplicateword")
    );
}
