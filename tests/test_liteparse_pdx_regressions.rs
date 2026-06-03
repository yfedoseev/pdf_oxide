//! Regression guards for the liteparse head-to-head report (PDX-1, PDX-4, PDX-5).
//!
//! These pin behaviour that the report found broken and that was subsequently
//! fixed, using the report's own minimal fixture (`multi_column_table.pdf`:
//! a 4-column × 5-row financial-style table whose cells are each emitted with
//! a separate `Tj` operator and zero `TJ` arrays — the layout that historically
//! triggered word-concatenation and table-detection failures).

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::PdfDocument;

const FIXTURE: &str = "tests/fixtures/multi_column_table.pdf";

/// PDX-1 — adjacent words from separate `Tj` operators must not be glued.
/// Historically `extract_text` returned `Year RevenueCost Net Income` and
/// `2021365,817 212,98194,680`; after the strong-geometric threshold fix the
/// word boundaries are recovered.
#[test]
fn pdx1_words_not_concatenated() {
    let doc = PdfDocument::open(FIXTURE).expect("open multi_column_table fixture");
    let text = doc.extract_text(0).expect("extract_text page 0");

    for needle in ["Year Revenue", "Revenue Cost", "2021 365,817"] {
        assert!(
            text.contains(needle),
            "PDX-1 regression: expected separated words {needle:?} in extracted text, got:\n{text}"
        );
    }
    // The pathological glued tokens from the report must be gone.
    assert!(
        !text.contains("RevenueCost"),
        "PDX-1 regression: words still concatenated (\"RevenueCost\") in:\n{text}"
    );
}

/// PDX-4 — the HTML converter must honour `extract_tables`: when table
/// detection is enabled and a table is detected it emits `<table>/<tr>/<td>`
/// rather than flowing paragraphs. Historically `to_html_all` was byte-identical
/// with the flag on or off.
#[test]
fn pdx4_html_honors_extract_tables() {
    let doc = PdfDocument::open(FIXTURE).expect("open multi_column_table fixture");
    let opts_on = ConversionOptions::default().with_default_table_detection();
    let html = doc.to_html_all(&opts_on).expect("to_html_all with tables");

    assert!(
        html.contains("<table"),
        "PDX-4 regression: HTML did not emit <table> with extract_tables=true:\n{html}"
    );
    assert!(
        html.contains("<td") || html.contains("<th"),
        "PDX-4 regression: HTML table emitted no cells:\n{html}"
    );
    assert!(
        html.contains("391,035"),
        "PDX-4 regression: table cell content missing from HTML:\n{html}"
    );
}

/// PDX-5 — the table detector must find the multi-column financial table, not
/// only single-column TOC/list structures. Historically the strict
/// "every row has row[0]'s column count" predicate rejected this fixture.
#[test]
fn pdx5_multicolumn_table_detected() {
    let doc = PdfDocument::open(FIXTURE).expect("open multi_column_table fixture");
    let opts_on = ConversionOptions::default().with_default_table_detection();
    let md = doc
        .to_markdown_all(&opts_on)
        .expect("to_markdown_all with tables");

    // Collect markdown table rows (lines that are pipe-delimited cells).
    let table_rows: Vec<&str> = md
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with('|') && l.matches('|').count() >= 4)
        .collect();

    assert!(
        !table_rows.is_empty(),
        "PDX-5 regression: no multi-column (>=4 pipe) markdown table rows emitted:\n{md}"
    );
    assert!(
        table_rows.iter().any(|r| r.contains("Revenue")),
        "PDX-5 regression: header row with 'Revenue' not in a table row:\n{md}"
    );
    assert!(
        table_rows.iter().any(|r| r.contains("391,035")),
        "PDX-5 regression: data value '391,035' not inside a table row:\n{md}"
    );
}
