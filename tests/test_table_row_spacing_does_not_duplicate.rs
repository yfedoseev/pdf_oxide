//! A recovered orphan must be compared against what the cells actually render.
//!
//! `to_html` recovers spans a table claimed but never rendered, and decides
//! "never rendered" by looking for the span in the table's text. Two things
//! made that lookup ask the wrong question, and both emitted the span a second
//! time beside the table:
//!
//! * it read `cell.text`, but `render_cell_html` walks `cell.spans` instead
//!   whenever the cell has any — inserting a space where `has_horizontal_gap`
//!   finds one, and routing each span through `push_span_text`, which can
//!   itself split a column-spanning decimal. The string tested and the string
//!   rendered were not the same string;
//! * for a multi-word span it compared whitespace-normalised text, but the two
//!   sides disagree about *where the spaces go*, not about the glyphs. A table
//!   of contents renders `Chapter I— Federal Trade Commission ....` from four
//!   cells while the flow span reads `Chapter I—Federal Trade Commission ....`;
//!   one file split `Department` across cells as `D epartm ent`, another joined
//!   `National Park` into `NationalPark`.
//!
//! The comparison is now on glyph sequences, bounded to a single row — cells of
//! one row are adjacent on the page, so matching across them is right, while
//! matching across the whole table is the looseness that duplicated whole
//! paragraphs on an earlier attempt.

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::PdfDocument;

/// A ruled one-row table whose two cells sit either side of the rule at
/// x = 200, with their text drawn close enough across it that the flow
/// assembler produces one span for what the grid renders as two cells.
fn straddling_row_pdf(extra_show_ops: &str) -> Vec<u8> {
    let content = format!(
        "0.5 w\n\
         50 60 m 350 60 l S\n\
         50 110 m 350 110 l S\n\
         50 60 m 50 110 l S\n\
         200 60 m 200 110 l S\n\
         350 60 m 350 110 l S\n\
         BT /F1 10 Tf\n\
         1 0 0 1 160 80 Tm (Chapter) Tj\n\
         1 0 0 1 205 80 Tm (Federal) Tj\n\
         {extra_show_ops}ET\n"
    );
    let content = content.as_bytes();

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

/// Occurrences of a glyph sequence, ignoring whatever whitespace each surface
/// chooses to insert at the seam.
fn glyph_runs(s: &str, needle: &str) -> usize {
    let squashed: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    squashed.matches(needle).count()
}

/// Strip tags so the count is over content, not markup.
fn visible(html: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for c in html.chars() {
        match c {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                out.push(' ');
            },
            _ if !in_tag => out.push(c),
            _ => {},
        }
    }
    out
}

#[test]
fn test_row_straddling_span_is_not_emitted_twice_in_html() {
    let doc = PdfDocument::from_bytes(straddling_row_pdf("")).expect("parse");
    let opts = ConversionOptions {
        extract_tables: true,
        ..Default::default()
    };
    let html = visible(&doc.to_html(0, &opts).expect("html"));

    for word in ["Chapter", "Federal"] {
        let n = glyph_runs(&html, word);
        assert!(n > 0, "{word:?} must reach the HTML surface at all:\n{html}");
        assert_eq!(
            n, 1,
            "{word:?} was recovered beside the table that already renders it, \
             so it appears {n} times:\n{html}"
        );
    }
}

/// Control: a span the table genuinely does not render must still be
/// recovered. Without this the guard could be tightened into dropping content,
/// which is the other half of the same issue.
#[test]
fn test_span_outside_every_row_is_still_recovered() {
    // The same table, plus a line of text below the ruled area that no cell
    // covers.
    let doc = PdfDocument::from_bytes(straddling_row_pdf("1 0 0 1 60 30 Tm (Standalone) Tj\n"))
        .expect("parse");
    let opts = ConversionOptions {
        extract_tables: true,
        ..Default::default()
    };
    let html = visible(&doc.to_html(0, &opts).expect("html"));
    assert_eq!(
        glyph_runs(&html, "Standalone"),
        1,
        "text no cell renders must still reach the HTML surface exactly once:\n{html}"
    );
}
