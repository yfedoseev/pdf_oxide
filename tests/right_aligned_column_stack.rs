//! A right-aligned numeric column whose row pitch sits under the same-line
//! tolerance must not fuse figures from consecutive rows
//! (`22,796` + `3,052` → `22,7963,052`).
//!
//! The joiner's backtracking branch already breaks stacks that share a LEFT
//! edge, but a right-aligned column shares its RIGHT edge: a shorter successor
//! starts further right by the width difference, landing between that branch's
//! `delta_x <= 0.5` and the inflated-width branch's `delta_x > fs*1.5`, where
//! nothing fires. Geometry below is taken from the BEA capital-flows table that
//! reported it (Helvetica 6 pt, 6 pt row pitch, column right edge 582 pt).
//! Hand-built minimal Helvetica PDF (simple single-byte font, no third-party
//! files).

use pdf_oxide::PdfDocument;

/// One-page PDF holding two right-aligned stacks, each a `BT…ET` pair one row
/// pitch apart. Widths are Helvetica's (digit 556/1000, comma 278/1000) at
/// 6 pt, so `x` places each run's RIGHT edge where the column wants it.
fn right_aligned_stacks_pdf() -> Vec<u8> {
    right_aligned_stacks_pdf_at(1.0)
}

/// The same page with every coordinate and the font size multiplied by
/// `scale`. A right-aligned column keeps its shape under scaling, so the
/// joiner's right-edge tolerance must scale with it — a point-constant
/// tolerance passes at 6 pt and fails at 12 pt.
fn right_aligned_stacks_pdf_at(scale: f32) -> Vec<u8> {
    // Stack 1, right edges flush at 118.348: "22,796" (18.348 wide) from 100,
    // "3,052" (15.012 wide) from 103.336 — Δy=6 is inside the same-line
    // tolerance (6*1.2=7.2), Δx=+3.336 is inside the dead band.
    //
    // Stack 2 repeats it with the successor's right edge 0.371 pt to the right:
    // that is the measured offset between the source column's negative rows
    // ("–7,431") and its positive rows ("5,195"), and it is what the branch's
    // right-edge tolerance has to absorb.
    let content = format!(
        "BT /F1 {fs} Tf 1 0 0 1 {x1} {ya1} Tm (22,796) Tj ET\n\
         BT /F1 {fs} Tf 1 0 0 1 {x2} {ya2} Tm (3,052) Tj ET\n\
         BT /F1 {fs} Tf 1 0 0 1 {x1} {yb1} Tm (17,431) Tj ET\n\
         BT /F1 {fs} Tf 1 0 0 1 {x3} {yb2} Tm (5,195) Tj ET\n",
        fs = 6.0 * scale,
        x1 = 100.0 * scale,
        x2 = 103.336 * scale,
        x3 = 103.707 * scale,
        ya1 = 700.0,
        ya2 = 700.0 - 6.0 * scale,
        yb1 = 660.0,
        yb2 = 660.0 - 6.0 * scale,
    );
    let content = content.as_str();
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

fn extract() -> String {
    let doc = PdfDocument::from_bytes(right_aligned_stacks_pdf()).expect("fixture parses");
    doc.extract_text(0).expect("extract_text")
}

/// The identical column at twice the size: the negative-row offset becomes
/// 0.742 pt, which a point-constant tolerance rejects. The tolerance is
/// derived from the run's own glyph advance, so the doubled column must
/// behave exactly like the original.
#[test]
fn right_aligned_stack_tolerance_scales_with_the_font() {
    let doc = PdfDocument::from_bytes(right_aligned_stacks_pdf_at(2.0)).expect("fixture parses");
    let text = doc.extract_text(0).expect("extract_text");
    assert!(
        !text.contains("22,7963,052") && !text.contains("17,4315,195"),
        "the scaled column must not fuse, got: {text:?}"
    );
}

#[test]
fn right_aligned_stack_is_separated_not_glued() {
    let text = extract();
    assert!(
        !text.contains("22,7963,052"),
        "stacked column cells must not fuse, got: {text:?}"
    );
    assert!(
        text.contains("22,796") && text.contains("3,052"),
        "both figures must survive, got: {text:?}"
    );
}

#[test]
fn right_aligned_stack_absorbs_the_negative_row_offset() {
    let text = extract();
    assert!(
        !text.contains("17,4315,195"),
        "a 0.371 pt right-edge offset must stay inside tolerance, got: {text:?}"
    );
}
