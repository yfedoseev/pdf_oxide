//! A cell's retention budget must absorb a flow span that **joins** what the
//! cell builder split, not only one that is contained in a single cell token.
//!
//! The two sides break words at different distances. Word clustering splits at
//! roughly 1.8 pt, while the flow assembler inserts a space only at about
//! 7.2 pt for 10 pt Courier. So a cell built from two show operations yields
//! `{"abc", "def"}` while the flow assembler produces one span `"abcdef"`.
//!
//! The budget could already consume a span token that is a *substring of one*
//! cell token, but had no path for one that is a *concatenation of several*.
//! The span therefore failed to be absorbed, was retained, and its glyphs were
//! emitted a second time beside the table's own rendering — with nothing
//! downstream to deduplicate them.

use pdf_oxide::PdfDocument;

/// A ruled single-cell table whose text is drawn as two show operations
/// separated by a sub-word gap, so the cell splits what the flow joins.
fn split_cell_pdf() -> Vec<u8> {
    // 10 pt Courier: the ~24 pt step between 110 and 134 is wider than the
    // cluster split (~1.8 pt) but narrower than the flow assembler's space
    // threshold (~7.2 pt) is per-character — the two disagree about the seam.
    let content = b"0.5 w\n\
        50 60 m 350 60 l S\n\
        50 110 m 350 110 l S\n\
        50 160 m 350 160 l S\n\
        50 60 m 50 160 l S\n\
        200 60 m 200 160 l S\n\
        350 60 m 350 160 l S\n\
        BT /F1 10 Tf\n\
        1 0 0 1 110 130 Tm (abc) Tj\n\
        1 0 0 1 134 130 Tm (def) Tj\n\
        1 0 0 1 210 130 Tm (Two) Tj\n\
        1 0 0 1 60 80 Tm (Three) Tj\n\
        1 0 0 1 210 80 Tm (Four) Tj\n\
        ET\n";

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
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 200] \
         /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
    );
    off[4] = buf.len();
    buf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    buf.extend_from_slice(content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    obj(
        &mut buf,
        &mut off,
        5,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>",
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

/// Count non-overlapping occurrences of the glyph sequence, ignoring any
/// whitespace the assembler may or may not insert at the seam.
fn glyph_runs(text: &str) -> usize {
    let squashed: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    squashed.matches("abcdef").count()
}

/// The six glyphs must reach the output once, not twice.
#[test]
fn a_joined_span_is_absorbed_by_the_cell_budget() {
    let doc = PdfDocument::from_bytes(split_cell_pdf()).expect("parse");
    let text = doc.extract_text(0).expect("extract");
    let runs = glyph_runs(&text);
    assert!(runs > 0, "the cell's text should appear at all:\n{text}");
    assert_eq!(
        runs, 1,
        "the joined flow span was retained alongside the table's own \
         rendering, so the glyphs appear {runs} times:\n{text}"
    );
}

/// The same on the markdown surface, which renders the table separately.
#[test]
fn the_markdown_surface_shows_the_cell_once() {
    let doc = PdfDocument::from_bytes(split_cell_pdf()).expect("parse");
    let opts = pdf_oxide::converters::ConversionOptions {
        extract_tables: true,
        ..Default::default()
    };
    let md = doc.to_markdown(0, &opts).expect("markdown");
    assert_eq!(glyph_runs(&md), 1, "the cell's glyphs appear more than once in markdown:\n{md}");
}
