//! Booktabs-style ("three-line") tables must survive table detection.
//!
//! Academic templates (LaTeX `booktabs`, many journal styles) rule a table
//! with only a few full-width HORIZONTAL strokes — a heavy top rule, a thin
//! sub-header rule, a heavy bottom rule — each drawn as a geometrically
//! zero-height straight-line stroke, with NO vertical column dividers at
//! all. Detection therefore rides entirely on the horizontal-rule-bounded
//! fallback (`detect_tables_from_horizontal_rules`), which is gated on the
//! intersection- and cluster-based pipelines finding nothing and on
//! `v_edges` being empty. Any perturbation of those gates silently scatters
//! the table into disconnected paragraphs.

use pdf_oxide::document::PdfDocument;

/// A 6-row logistic-regression-style table in the booktabs idiom:
/// three full-width zero-height horizontal rules (0.8 pt top, 0.3 pt
/// sub-header, 0.8 pt bottom), no vertical rules, 4 text columns.
fn booktabs_fixture_pdf() -> Vec<u8> {
    booktabs_fixture_pdf_with_decorations(false)
}

/// Same table; `with_speck` additionally draws an unrelated decorative
/// stroke far from the table — a ~1 pt segment with a heavy (8 pt) stroke,
/// the shape of a tick mark / list dash / emphasis bar. Real academic pages
/// carry such marks; they must not disable table detection elsewhere on the
/// page.
fn booktabs_fixture_pdf_with_decorations(with_speck: bool) -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(b"0 J 0 j\n");
    if with_speck {
        // Decorative heavy-stroked speck near the page footer, nowhere near
        // the table (table spans y 568..710; this sits at y 120).
        content.extend_from_slice(b"8 w 300 120 m 301 120 l S\n");
    }
    // Top rule (heavy): zero-height full-width stroke at y=710.
    content.extend_from_slice(b"0.8 w 100 710 m 500 710 l S\n");
    // Sub-header rule (thin) at y=688.
    content.extend_from_slice(b"0.3 w 100 688 m 500 688 l S\n");
    // Bottom rule (heavy) at y=568.
    content.extend_from_slice(b"0.8 w 100 568 m 500 568 l S\n");
    // Header + 5 data rows, 4 columns. No vertical rules anywhere.
    content.extend_from_slice(b"BT /F1 10 Tf\n");
    let rows: [[&str; 4]; 6] = [
        ["Variable", "Beta", "SE", "p-Value"],
        ["Age", "0.042", "0.011", "0.001"],
        ["Sex", "0.318", "0.142", "0.025"],
        ["BMI", "0.077", "0.023", "0.004"],
        ["Smoker", "0.512", "0.201", "0.011"],
        ["Diabetes", "0.694", "0.233", "0.003"],
    ];
    let ys = [695, 672, 652, 632, 612, 592];
    let xs = [105, 260, 340, 430];
    for (row, y) in rows.iter().zip(ys) {
        for (cell, x) in row.iter().zip(xs) {
            content.extend_from_slice(format!("1 0 0 1 {x} {y} Tm ({cell}) Tj\n").as_bytes());
        }
    }
    content.extend_from_slice(b"ET");
    build_minimal_pdf_raw(&content, b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]")
}

#[test]
fn booktabs_three_line_table_is_detected() {
    let doc = PdfDocument::from_bytes(booktabs_fixture_pdf()).expect("parse fixture");
    let tables = doc.extract_tables(0).expect("extract tables");
    assert!(
        !tables.is_empty(),
        "a booktabs (three horizontal rules, no vertical rules) table must be detected"
    );
    let cells: Vec<String> = tables[0]
        .rows
        .iter()
        .flat_map(|r| r.cells.iter())
        .map(|c| c.text.trim().to_string())
        .collect();
    assert!(
        cells.iter().any(|c| c.contains("Age")) && cells.iter().any(|c| c.contains("0.042")),
        "row labels must stay aligned with their values, got {:?}",
        cells
    );
    assert!(
        tables[0].rows.len() >= 5,
        "all data rows must be captured, got {} rows",
        tables[0].rows.len()
    );
}

#[test]
fn booktabs_table_survives_unrelated_decorative_speck() {
    // A heavy-stroked ~1 pt decorative segment elsewhere on the page (a
    // tick mark / list dash) must not knock out the horizontal-rule table
    // path for the whole page. Uses the line-pipelines-only configuration
    // (`text_fallback = false`, as `extract_text` / `to_plain_text` do) so
    // the text-alignment fallback cannot mask a loss in the rule-based
    // path: the fallback rides on ideal synthetic alignment that real
    // academic tables don't have.
    let mut config = pdf_oxide::structure::spatial_table_detector::TableDetectionConfig::default();
    config.text_fallback = false;
    let doc = PdfDocument::from_bytes(booktabs_fixture_pdf_with_decorations(true)).expect("parse");
    let tables = doc
        .extract_tables_with_config(0, config)
        .expect("extract tables");
    assert!(
        !tables.is_empty(),
        "booktabs table must still be detected with a decorative speck on the page"
    );
    assert!(
        tables[0].rows.len() >= 5,
        "all data rows must be captured, got {} rows",
        tables[0].rows.len()
    );
}

#[test]
fn booktabs_table_detected_by_line_pipelines_alone() {
    // The clean three-line table must be found by the rule-based pipelines
    // themselves (horizontal-rule-bounded fallback), independent of the
    // text-alignment fallback.
    let mut config = pdf_oxide::structure::spatial_table_detector::TableDetectionConfig::default();
    config.text_fallback = false;
    let doc = PdfDocument::from_bytes(booktabs_fixture_pdf()).expect("parse");
    let tables = doc
        .extract_tables_with_config(0, config)
        .expect("extract tables");
    assert!(!tables.is_empty(), "three-line table must be detected from its rules");
}

#[test]
fn booktabs_three_line_table_renders_as_html_table() {
    let mut doc = PdfDocument::from_bytes(booktabs_fixture_pdf()).expect("parse fixture");
    let html = doc
        .to_html_all(&pdf_oxide::converters::ConversionOptions::default())
        .expect("to_html_all");
    assert!(
        html.contains("<table"),
        "booktabs table must render as an HTML <table>, got: {}",
        &html[..html.len().min(2000)]
    );
    assert!(html.contains("Diabetes"), "table body text present");
}

// ---------------------------------------------------------------------------
// Minimal raw PDF builder (same pattern as test_stroke_width_rendered_bbox.rs)
// ---------------------------------------------------------------------------

fn build_minimal_pdf_raw(content: &[u8], page_extra: &[u8]) -> Vec<u8> {
    let mut pdf = b"%PDF-1.4\n".to_vec();

    let off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    let off2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    let off3 = pdf.len();
    pdf.extend_from_slice(b"3 0 obj\n<< ");
    pdf.extend_from_slice(page_extra);
    pdf.extend_from_slice(b" /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n");

    let off4 = pdf.len();
    pdf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    let off5 = pdf.len();
    pdf.extend_from_slice(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>\nendobj\n",
    );

    let xref_pos = pdf.len();
    let offsets = [0usize, off1, off2, off3, off4, off5];
    pdf.extend_from_slice(format!("xref\n0 {}\n", offsets.len()).as_bytes());
    pdf.extend_from_slice(format!("{:010} 65535 f\r\n", 0).as_bytes());
    for &off in &offsets[1..] {
        pdf.extend_from_slice(format!("{:010} 00000 n\r\n", off).as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            offsets.len(),
            xref_pos
        )
        .as_bytes(),
    );
    pdf
}
