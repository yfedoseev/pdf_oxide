//! A two-column body reads column-major on every surface, not just plain text.
//!
//! ISO 32000-1:2008 §14.8.2.3.1 leaves an untagged page with no reading order,
//! so the order has to be reconstructed from geometry, and §14.8.3's layout
//! model reads a multi-column body one column at a time. A row-major (y, x)
//! sweep across the full page width instead splices the right column into the
//! middle of the left column's sentences — `... is a less toxic properties
//! against several RNA viruses ...`.
//!
//! The page below is a plain two-column body with a tight gutter: 8pt between
//! two 210pt columns of 9pt type. Nothing crosses that gutter anywhere down the
//! page, so no reader could mistake the layout — but the gutter is under every
//! fixed width the geometric column detectors demand, which is the shape a real
//! dense two-column journal page presents. `extract_text` reads it
//! column-major; `to_markdown` and `to_html` must reach the same conclusion
//! about the same page.

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::PdfDocument;

/// Body line, sans its tag. Every line carries the same text so the two columns
/// have exactly one left edge and one right edge each — the geometry under test
/// is the gutter, and ragged line lengths would blur it.
const BODY: &str = "an ordinary sentence of running body text here";

/// `BODY` with a tag at 9pt Helvetica measures this wide. The tags are
/// anagrams of each other (`LR` / `RL`), so a left line and a right line are
/// glyph-for-glyph the same width and the two columns align exactly.
const COL_W: f32 = 210.11;
/// The gutter. Narrow — a dense two-column page — but completely empty.
const GUTTER: f32 = 8.0;

const LEFT_X: f32 = 72.0;
const RIGHT_X: f32 = LEFT_X + COL_W + GUTTER;
const TOP_Y: f32 = 700.0;
const LEADING: f32 = 14.0;
const ROWS: usize = 10;

fn left_tag(row: usize) -> String {
    format!("LR{:02}", row + 1)
}

fn right_tag(row: usize) -> String {
    format!("RL{:02}", row + 1)
}

/// Wrap a content stream in the smallest PDF that carries it.
fn pdf_with(content: String) -> Vec<u8> {
    let content = content.into_bytes();
    let objects: Vec<Vec<u8>> = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R \
           /Resources << /Font << /F1 5 0 R >> >> >>"
            .to_vec(),
        [
            format!("<< /Length {} >>\nstream\n", content.len()).into_bytes(),
            content,
            b"\nendstream".to_vec(),
        ]
        .concat(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
    ];

    let mut pdf = b"%PDF-1.7\n".to_vec();
    let mut offsets = Vec::new();
    for (i, body) in objects.iter().enumerate() {
        offsets.push(pdf.len());
        pdf.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
        pdf.extend_from_slice(body);
        pdf.extend_from_slice(b"\nendobj\n");
    }
    let xref = pdf.len();
    let n = objects.len() + 1;
    pdf.extend_from_slice(format!("xref\n0 {n}\n0000000000 65535 f \n").as_bytes());
    for off in &offsets {
        pdf.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
    }
    pdf.extend_from_slice(
        format!("trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes(),
    );
    pdf
}

fn show(content: &mut String, x: f32, y: f32, text: &str) {
    content.push_str(&format!("BT /F1 9 Tf {x} {y} Td ({text}) Tj ET\n"));
}

fn two_column_page() -> Vec<u8> {
    let mut content = String::new();
    for r in 0..ROWS {
        let y = TOP_Y - LEADING * r as f32;
        show(&mut content, LEFT_X, y, &format!("{} {BODY}", left_tag(r)));
        show(&mut content, RIGHT_X, y, &format!("{} {BODY}", right_tag(r)));
    }
    pdf_with(content)
}

/// The left column alone, for measuring what a body line actually spans.
fn left_column_only() -> Vec<u8> {
    let mut content = String::new();
    show(&mut content, LEFT_X, TOP_Y, &format!("{} {BODY}", left_tag(0)));
    pdf_with(content)
}

/// Every left-column tag precedes every right-column tag: the definition of
/// reading one column before the other.
fn assert_column_major(surface: &str, out: &str) {
    let at = |tag: &str| {
        out.find(tag)
            .unwrap_or_else(|| panic!("{surface}: `{tag}` missing from output:\n{out}"))
    };
    let last_left = at(&left_tag(ROWS - 1));
    let first_right = at(&right_tag(0));
    assert!(
        last_left < first_right,
        "{surface}: the right column is spliced into the left column — `{}` at byte \
         {first_right} precedes `{}` at byte {last_left}. A two-column body reads \
         column-major (ISO 32000-1 §14.8.3). Got:\n{out}",
        right_tag(0),
        left_tag(ROWS - 1)
    );
}

#[test]
fn plain_text_reads_the_two_column_body_column_major() {
    let doc = PdfDocument::from_bytes(two_column_page()).expect("open");
    assert_column_major("extract_text", &doc.extract_text(0).expect("text"));
}

#[test]
fn markdown_reads_the_two_column_body_column_major() {
    let doc = PdfDocument::from_bytes(two_column_page()).expect("open");
    let out = doc
        .to_markdown(0, &ConversionOptions::default())
        .expect("md");
    assert_column_major("to_markdown", &out);
}

#[test]
fn html_reads_the_two_column_body_column_major() {
    let doc = PdfDocument::from_bytes(two_column_page()).expect("open");
    let out = doc.to_html(0, &ConversionOptions::default()).expect("html");
    assert_column_major("to_html", &out);
}

/// The page only tests the gutter if the gutter is where this file says it is.
/// `COL_W` is a measured Helvetica advance; if the font metrics move, the
/// columns overlap or the gap widens and the detectors under test start seeing
/// a different page.
#[test]
fn test_fixture_puts_an_empty_gutter_between_two_columns() {
    let doc = PdfDocument::from_bytes(left_column_only()).expect("open");
    let lines = doc.extract_text_lines(0).expect("lines");
    assert_eq!(lines.len(), 1, "expected one measured line, got {lines:?}");
    let measured = lines[0].bbox.width;
    assert!(
        (measured - COL_W).abs() < 1.0,
        "a body line measures {measured}pt, not the {COL_W}pt this fixture's \
         geometry is built on"
    );
    assert!(
        LEFT_X + measured < RIGHT_X,
        "the columns overlap: the left column ends at {}",
        LEFT_X + measured
    );
    assert!(
        RIGHT_X - (LEFT_X + measured) < 12.0,
        "the gutter must stay under the clean-corridor sweep's 12pt bar, else the \
         page is a different test"
    );
}
