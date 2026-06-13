//! Integration test for issue #734 (`format=text` reading order on a
//! reference-edition two-column layout).
//!
//! Unlike the clean wide-gutter case in `two_column_reading_order.rs`, a
//! reference Bible page has the three features that defeat geometric column
//! detection at the whole-page level:
//!   * a **full-width book-title line** that spans both columns and the gutter,
//!   * a **narrow gutter** (≈30 pt), and
//!   * **short, ragged verse lines** with leading marginal numerals.
//!
//! The left column (verses 1–6) must be read top-to-bottom before the right
//! column (verses 14–19); a row-by-row scan across the gutter interleaves them
//! (`"1 Au commencement … 14 Et Dieu dit …"`). PDF is hand-built — no
//! third-party fixture (project policy).

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::PdfDocument;

/// One untagged page, 432 pt wide: a full-width title across the top, then two
/// narrow text columns (left x≈40, right x≈235, gutter ≈ [190,235]). Each
/// column line shares a y-band with the opposite column so a naive Y-then-X
/// sort interleaves across the gutter.
fn kjf_two_column_pdf() -> Vec<u8> {
    // Title spans the full content width; verse lines are short and ragged.
    let mut content = String::from("/F1 9 Tf\n");
    content.push_str("BT 1 0 0 1 40 760 Tm (Le Troisieme Livre de Moise Appele GENESE) Tj ET\n");
    // Six shared y-bands; left = verses 1..6, right = verses 14..19.
    let mut y = 730;
    for i in 0..6 {
        let lv = i + 1;
        let rv = i + 14;
        content.push_str(&format!("BT 1 0 0 1 40 {y} Tm ({lv} Au commencement Dieu) Tj ET\n"));
        content.push_str(&format!("BT 1 0 0 1 235 {y} Tm ({rv} Et Dieu dit Quil y ait) Tj ET\n"));
        y -= 24;
    }
    let content = content.into_bytes();

    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 6];
    let obj = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, body: &str| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };
    let stream = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, data: &[u8]| {
        off[id] = buf.len();
        buf.extend_from_slice(
            format!("{id} 0 obj\n<< /Length {} >>\nstream\n", data.len()).as_bytes(),
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
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 432 792] \
         /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
    );
    stream(&mut buf, &mut off, 4, &content);
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

/// Assert the whole left column (verse 6) precedes the first right-column line
/// (verse 14); a row-by-row read interleaves "1 … 14 … 2 … 15 …".
fn assert_column_major(label: &str, out: &str) {
    let last_left = out
        .find("6 Au commencement")
        .unwrap_or_else(|| panic!("{label}: left column missing:\n{out}"));
    let first_right = out
        .find("14 Et Dieu dit")
        .unwrap_or_else(|| panic!("{label}: right column missing:\n{out}"));
    assert!(
        last_left < first_right,
        "{label}: columns interleaved — left not read fully before right:\n{out}"
    );
}

// Issue #734 Fix 2 — TARGET (currently #[ignore]d). The per-flow column-major
// heuristics (a flat sort in `extract_text`, a split-convert in
// `to_markdown`/`to_html`) were REVERTED: the v0.3.64-vs-HEAD corpus sweep
// proved they regress real two-column PDFs (arxiv references scrambled, a road
// data-sheet form glued, a TOC's page numbers detached) because each flow
// rolled its own ordering and the flat sort overrode XY-cut's already-correct
// order. The correct fix is a single recursive XY-cut column-region pass
// (`page_reading_order`) consumed by ALL flows, with the full-width-title band
// separated at the partition level. These assert the goal across text/md/html
// and un-ignore when that universal reading-order lands (corpus-gated).
// #734: pending the G1 column-region model. FOUR attempts to fix this at the
// geometric layer have each traded one win for a new regression on the corpus
// (per-flow flat sort scrambled arxiv refs; content-balance gate regressed a
// TOC; the XY-cut band-peel guard regressed a different arxiv's mid-body section
// heading + a gov form). Robust full-width-band-vs-content separation at every
// nesting level needs layout-model reconstruction, not a geometric heuristic.
// The clean wins (structured column_index, tagged Lbl/Sect, column-mode override)
// ship; this target waits for G1. See column-region-reading-order-design.md.
#[ignore = "#734: needs G1 column-region model — 4 geometric attempts each regressed the corpus"]
#[test]
fn kjf_two_column_reads_column_major_text() {
    let doc = PdfDocument::from_bytes(kjf_two_column_pdf()).unwrap();
    assert_column_major("text", &doc.extract_text(0).unwrap());
}

#[ignore = "#734: pending universal recursive-XY-cut reading order"]
#[test]
fn kjf_two_column_reads_column_major_markdown() {
    let doc = PdfDocument::from_bytes(kjf_two_column_pdf()).unwrap();
    let md = doc.to_markdown(0, &ConversionOptions::default()).unwrap();
    assert_column_major("markdown", &md);
}

#[ignore = "#734: pending universal recursive-XY-cut reading order"]
#[test]
fn kjf_two_column_reads_column_major_html() {
    let doc = PdfDocument::from_bytes(kjf_two_column_pdf()).unwrap();
    let html = doc.to_html(0, &ConversionOptions::default()).unwrap();
    assert_column_major("html", &html);
}

/// #734 item #2: `extract_structured` must group each column into ONE region
/// (left column whole, then right), not interleaved per-line regions — so a
/// consumer reading region text in order gets column-major output, and each
/// region's `text` is a single clean column (not the interleaved "garbage").
#[test]
fn kjf_structured_groups_one_region_per_column() {
    let doc = PdfDocument::from_bytes(kjf_two_column_pdf()).unwrap();
    let page = doc.extract_structured(0).unwrap();

    let col0: Vec<&str> = page
        .regions
        .iter()
        .filter(|r| r.column_index == Some(0))
        .map(|r| r.text.as_str())
        .collect();
    let col1: Vec<&str> = page
        .regions
        .iter()
        .filter(|r| r.column_index == Some(1))
        .map(|r| r.text.as_str())
        .collect();

    // Exactly one region per column.
    assert_eq!(col0.len(), 1, "left column must be one region: {col0:?}");
    assert_eq!(col1.len(), 1, "right column must be one region: {col1:?}");

    // Each region's text is a clean single column read top-to-bottom.
    assert!(col0[0].contains("1 Au") && col0[0].contains("6 Au"), "left text: {}", col0[0]);
    assert!(
        !col0[0].contains("14 Et"),
        "left region must not contain right text: {}",
        col0[0]
    );
    assert!(
        col1[0].contains("14 Et") && col1[0].contains("19 Et"),
        "right text: {}",
        col1[0]
    );

    // Concatenating region text in order is column-major (left before right).
    let joined: String = page
        .regions
        .iter()
        .map(|r| r.text.clone())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        joined.find("6 Au").unwrap() < joined.find("14 Et").unwrap(),
        "structured region order interleaves columns: {joined}"
    );
}
