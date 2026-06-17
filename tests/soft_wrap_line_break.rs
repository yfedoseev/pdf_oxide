//! Integration coverage for 7338bbee: a soft-wrapped line on the untagged
//! text path must get a synthesized separator, not glue the last word of one
//! line to the first of the next (`tide` + `tables` → `tidetables`).
//!
//! A wrapped line's leading (~1.0 em) is below the same-line tolerance
//! (~1.2 em), so the two lines were treated as one and concatenated with no
//! space. The fix detects a carriage-return — a large backward x gap
//! (`gap < -fs*3`), a real baseline drop (`y_diff > fs*0.5`), and a return to
//! the left margin (`delta_x <= fs*0.5`) — and emits a newline. Hand-built
//! minimal Helvetica PDF (simple single-byte font, no third-party files).

use pdf_oxide::PdfDocument;

/// One-page PDF: two Helvetica runs, each its own `BT…ET`. `r1` ends far to the
/// right; `r2` starts back at the left margin one short line-height below — the
/// soft-wrap (carriage-return) shape.
fn two_line_pdf() -> Vec<u8> {
    // r1 baseline (72,700); r2 baseline (72,691): Δy=9 — inside the same-line
    // tolerance (10*1.2=12) yet above 10*0.5=5, and the x gap is a full
    // carriage return, so the soft-wrap branch must fire.
    let content = "BT /F1 10 Tf 1 0 0 1 72 700 Tm (the spring tide) Tj ET\n\
                   BT /F1 10 Tf 1 0 0 1 72 691 Tm (tables now posted) Tj ET\n";
    let mut buf: Vec<u8> = Vec::new();
    let mut off = [0usize; 6];
    buf.extend_from_slice(b"%PDF-1.7\n");
    let mut obj = |buf: &mut Vec<u8>, id: usize, body: String| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };
    obj(&mut buf, 1, "<< /Type /Catalog /Pages 2 0 R >>".into());
    obj(&mut buf, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".into());
    obj(
        &mut buf,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            .into(),
    );
    obj(
        &mut buf,
        4,
        format!("<< /Length {} >>\nstream\n{content}endstream", content.len()),
    );
    obj(&mut buf, 5, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".into());
    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 6\n0000000000 65535 f \n");
    for id in 1..=5 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

#[test]
fn soft_wrapped_line_is_separated_not_glued() {
    let doc = PdfDocument::from_bytes(two_line_pdf()).expect("fixture parses");
    let text = doc.extract_text(0).expect("extract_text");
    assert!(
        !text.contains("tidetables"),
        "soft-wrap must not glue the wrapped word boundary, got: {text:?}"
    );
    assert!(
        text.contains("tide") && text.contains("tables"),
        "both wrapped words must be present, got: {text:?}"
    );
}
