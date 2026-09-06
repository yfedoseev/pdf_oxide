//! Text a PDF paints as CMYK black must reach DOCX/PPTX/XLSX with no explicit
//! colour, so the destination theme supplies it.
//!
//! `0 0 0 1 k` does **not** convert to `(0, 0, 0)`. ISO 32000-1:2008 §10.3.5's
//! naive additive complement would give exact black, but this crate converts
//! through the measured process-ink corners, and the 000K corner is
//! approximately `(0.137, 0.122, 0.126)` — a dark grey. That is the right
//! colour to *paint*; it is the wrong thing to hand a word processor as an
//! explicit run colour.
//!
//! Three converters decided "inherit the theme" by testing for exact
//! `(0, 0, 0)`, so every print-origin PDF stamped a hard-coded dark grey into
//! its output instead. No test covered it, and a word-level corpus diff cannot
//! see it: the text is identical and only the colour attribute changed.
//!
//! The converters now share one predicate, `color::is_document_black`, so the
//! assertions here are on the span colour reaching it and on the predicate's
//! own answer. Asserting the emitted DOCX run properties directly would need a
//! zip reader as a dev-dependency, which is not worth adding for one string
//! check; the predicate fully determines the three call sites.

use pdf_oxide::PdfDocument;

/// A one-page PDF whose only text is painted with `paint`.
fn text_pdf(paint: &str) -> Vec<u8> {
    let content = format!("BT /F1 12 Tf {paint} 1 0 0 1 72 700 Tm (Body text) Tj ET\n");

    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 6];
    let obj = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, body: &str| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };

    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
    );
    off[4] = buf.len();
    buf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    buf.extend_from_slice(content.as_bytes());
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    obj(
        &mut buf,
        &mut off,
        5,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    );

    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 6\n0000000000 65535 f \n");
    for id in 1..=5 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

/// The span colour the extractor reports for text painted with `paint`.
fn span_colour(paint: &str) -> (f32, f32, f32) {
    let doc = PdfDocument::from_bytes(text_pdf(paint)).expect("parse");
    let spans = doc.extract_spans(0).expect("spans");
    let span = spans.first().expect("one span");
    (span.color.r, span.color.g, span.color.b)
}

/// The precondition the whole issue rests on: CMYK black is not RGB black
/// after the process-ink conversion. If this ever becomes false the rule below
/// is unnecessary rather than wrong.
#[test]
fn cmyk_black_does_not_convert_to_exact_rgb_black() {
    let (r, g, b) = span_colour("0 0 0 1 k");
    assert!(
        r > 0.0 || g > 0.0 || b > 0.0,
        "precondition: CMYK black converted to exact black ({r},{g},{b}), \
         so the exact-zero test would have been adequate"
    );
}

/// CMYK-black body text must be treated as the document's black.
#[test]
fn cmyk_black_text_is_treated_as_document_black() {
    let (r, g, b) = span_colour("0 0 0 1 k");
    assert!(
        pdf_oxide::color::is_document_black(r, g, b),
        "CMYK black text ({r},{g},{b}) was not recognised as the document's black, \
         so a converter stamps it as an explicit dark grey"
    );
}

/// RGB black is unchanged — the common case must keep working.
#[test]
fn rgb_black_text_is_treated_as_document_black() {
    let (r, g, b) = span_colour("0 g");
    assert!(pdf_oxide::color::is_document_black(r, g, b));
}

/// Coloured text keeps its colour. Without this the rule could pass by
/// declaring everything black.
#[test]
fn coloured_text_is_not_document_black() {
    let (r, g, b) = span_colour("1 0 0 rg");
    assert!(
        !pdf_oxide::color::is_document_black(r, g, b),
        "red text ({r},{g},{b}) must keep its explicit colour"
    );
}

/// A deliberate dark grey is an authored choice and must survive.
#[test]
fn deliberate_dark_grey_text_keeps_its_colour() {
    let (r, g, b) = span_colour("0.25 g");
    assert!(
        !pdf_oxide::color::is_document_black(r, g, b),
        "authored dark grey ({r},{g},{b}) must not be swallowed as black"
    );
}
