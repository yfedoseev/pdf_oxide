//! Real-document table regression tests on two open-access PMC articles.
//!
//! Opt-in empirical tests in the `v054_empirical_repros` style: the PDFs
//! live in `tests/fixtures/real/` (gitignored — real-world PDFs are not
//! vendored into the repo) and each test skips gracefully when its file
//! is absent. Fetch them with:
//!
//! ```text
//! python3 scripts/fetch_real_fixtures.py
//! ```
//!
//! Both articles are CC BY in the PubMed Central Open Access subset:
//!   - `pmc8103274.pdf` — Tomography 2021;7(2):95-106 (PMC8103274)
//!   - `pmc8025823.pdf` — PMC8025823
//!
//! They exercise the composition that synthetic fixtures kept missing: a
//! booktabs-style three-line table sharing a page with dash-bordered
//! decoration boxes (dash segments, near-square joint specks, and short
//! vertical dash runs). The assertions pin the exact tables that corpus
//! regression sweeps flagged, at the same converter level (`to_html_all`)
//! where the losses were observed.

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::document::PdfDocument;
use std::path::Path;

fn html_for(fixture: &str) -> Option<String> {
    let path = format!("tests/fixtures/real/{fixture}");
    if !Path::new(&path).exists() {
        eprintln!("[pmc] fixture missing, skipping: {path} (run scripts/fetch_real_fixtures.py)");
        return None;
    }
    let bytes = std::fs::read(&path).expect("read fixture");
    let doc = PdfDocument::from_bytes(bytes).expect("parse fixture");
    let opts = ConversionOptions {
        extract_tables: true,
        ..Default::default()
    };
    Some(doc.to_html_all(&opts).expect("to_html_all"))
}

/// Extract each `<table>...</table>` block from the HTML.
fn tables_in(html: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = html;
    while let Some(start) = rest.find("<table") {
        let Some(end) = rest[start..].find("</table>") else {
            break;
        };
        out.push(&rest[start..start + end + "</table>".len()]);
        rest = &rest[start + end + "</table>".len()..];
    }
    out
}

#[test]
fn pmc8103274_logistic_regression_table_renders_as_html_table() {
    // "Table 4": 6-row logistic-regression table (rules at y=663/647/576,
    // 392pt wide) on a page that also carries dash-bordered boxes. The
    // dash boxes' joint specks and vertical dash runs must not disable
    // table detection for the page.
    let Some(html) = html_for("pmc8103274.pdf") else {
        return;
    };
    let tables = tables_in(&html);
    assert!(
        tables
            .iter()
            .any(|t| t.contains("Age") && t.contains("0.05022")),
        "the logistic-regression table (Age row) must render as an HTML <table>; got {} table(s)",
        tables.len()
    );
}

#[test]
fn pmc8103274_tables_are_not_duplicated() {
    // Two rule families can bracket the same rows (the dash box overlaps a
    // table band); the detection must emit each table once.
    let Some(html) = html_for("pmc8103274.pdf") else {
        return;
    };
    let tables = tables_in(&html);
    let copies = tables.iter().filter(|t| t.contains(">Train<")).count();
    assert!(
        copies <= 1,
        "the All/Train/Test table must not be emitted more than once, got {copies}"
    );
}

#[test]
fn pmc8025823_complications_table_renders_as_html_table() {
    // "Table 3": 4-row complications table on a page with a dash-bordered
    // box overlapping a table band and a second booktabs table stacked
    // below.
    let Some(html) = html_for("pmc8025823.pdf") else {
        return;
    };
    let tables = tables_in(&html);
    assert!(
        tables
            .iter()
            .any(|t| t.contains("Prostatitis") && t.contains("Hematuria")),
        "the complications table must render as an HTML <table>; got {} table(s)",
        tables.len()
    );
}
