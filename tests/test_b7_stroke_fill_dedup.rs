//! Regression test for B7: stroke + fill renders produce doubled words.
//!
//! Maps, posters, and marketing collateral render every label twice:
//! once stroked for the outline effect, once filled for the glyph. Each
//! pass is a separate text object at essentially the same CTM. The old
//! span pipeline emitted both spans; the downstream merge step then
//! concatenated them into `"EverestEverest"`.
//!
//! Fix (src/extractors/text.rs `dedup_stroke_fill_overlap`): before the
//! existing dedup passes, walk spans and drop any that share the same
//! text with an earlier span whose bounding box overlaps by ≥70 % IoU.
//!
//! Test synthesises a single-page PDF with one label ("Everest")
//! rendered twice at identical coordinates. Extracted text must contain
//! "Everest" exactly once.

use pdf_oxide::PdfDocument;

fn stroke_fill_pdf() -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    let mut offsets: Vec<usize> = vec![0];

    out.extend_from_slice(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n");

    let push = |out: &mut Vec<u8>, offsets: &mut Vec<usize>, body: &str| {
        offsets.push(out.len());
        let id = offsets.len() - 1;
        out.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };

    push(&mut out, &mut offsets, "<< /Type /Catalog /Pages 2 0 R >>");
    push(&mut out, &mut offsets, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    push(
        &mut out,
        &mut offsets,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 400] \
           /Resources << /Font << /F0 5 0 R >> >> /Contents 4 0 R >>",
    );

    // Two text objects at identical CTM — stroke rendering mode (1),
    // then fill rendering mode (0). The first pass is the outline, the
    // second is the glyph fill. Content-stream-wise they're separate
    // BT…ET blocks and produce two distinct TextSpans at the same bbox.
    let stream = "BT 1 Tr /F0 24 Tf 100 300 Td (Everest) Tj ET \
                  BT 0 Tr /F0 24 Tf 100 300 Td (Everest) Tj ET\n";
    push(
        &mut out,
        &mut offsets,
        &format!("<< /Length {} >>\nstream\n{stream}\nendstream", stream.len() + 1),
    );
    push(&mut out, &mut offsets, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");

    let xref_offset = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n", offsets.len()).as_bytes());
    out.extend_from_slice(b"0000000000 65535 f \n");
    for &off in &offsets[1..] {
        out.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    out.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            offsets.len(),
            xref_offset
        )
        .as_bytes(),
    );
    out
}

#[test]
fn stroke_fill_overlap_does_not_double_text() {
    let pdf = stroke_fill_pdf();
    let tmp = tempfile::NamedTempFile::new().expect("temp");
    std::fs::write(tmp.path(), &pdf).unwrap();

    let mut doc = PdfDocument::open(tmp.path()).expect("open");
    let text = doc.extract_text(0).expect("extract");

    let count = text.matches("Everest").count();
    assert_eq!(
        count, 1,
        "B7 regression: stroke+fill pass produced {count} occurrences of 'Everest' \
         in extracted text; expected exactly 1. Full output: {text:?}"
    );

    // And must not produce a concatenated doubled form.
    assert!(
        !text.contains("EverestEverest"),
        "output must not contain 'EverestEverest': {text:?}"
    );
}
