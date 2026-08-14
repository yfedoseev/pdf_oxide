//! Table extraction must not delete words the grid does not re-emit,
//!
//! Spans inside a detected table were dropped whenever their trimmed text
//! appeared in a `HashSet` of every cell's text. A set records THAT a text is
//! a cell, never HOW MANY cells hold it, so when more spans matched than the
//! grid re-emits, the surplus was deleted outright.
//!
//! The corpus scan that found this reported losses only ever on repeated
//! tokens — `")(" 28->7`, `"+1" 8->3` — which is the shape this pins.

use std::collections::HashMap;

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::document::PdfDocument;

fn token_counts(text: &str) -> HashMap<&str, usize> {
    let mut counts = HashMap::new();
    for t in text.split_whitespace() {
        *counts.entry(t).or_insert(0) += 1;
    }
    counts
}

/// A ruled table whose cells repeat a value, plus the same value again below
/// the grid but still inside the table's outer frame — the case where more
/// spans match a cell text than the grid can re-emit.
fn repeated_value_table_pdf() -> Vec<u8> {
    let mut content = Vec::new();
    // A closed 2x2 grid between y=600 and y=700: three horizontals, three
    // verticals.
    content.extend_from_slice(b"0.5 w\n");
    for (x0, y0, x1, y1) in [
        (100, 600, 100, 700),
        (250, 600, 250, 700),
        (400, 600, 400, 700),
        (100, 700, 400, 700),
        (100, 650, 400, 650),
        (100, 600, 400, 600),
    ] {
        content.extend_from_slice(format!("{x0} {y0} m {x1} {y1} l S\n").as_bytes());
    }
    // One BT/ET block per run: consecutive Tj inside one block merge into a
    // single span and the fixture stops resembling real pages.
    for (x, y, text) in [
        (110, 670, "0.00"),
        (260, 670, "0.00"),
        (110, 620, "0.00"),
        (260, 620, "Total"),
        // Straddles the y=650 rule. The spatial detector adopts it as its own
        // cell, so this does not reproduce the bbox-containment deletion
        // (the un-ruled-column fixture below does); it pins the
        // repeated-value invariant on a grid the detector reshapes.
        (110, 646, "0.00"),
    ] {
        content.extend_from_slice(
            format!("BT /F1 10 Tf 1 0 0 1 {x} {y} Tm ({text}) Tj ET\n").as_bytes(),
        );
    }
    build_minimal_pdf_raw(&content, b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]")
}

/// Turning table extraction on must never emit a token fewer times than
/// leaving it off. Reflow is fine; deletion is not.
#[test]
fn table_extraction_does_not_delete_repeated_tokens() {
    let doc = PdfDocument::from_bytes(repeated_value_table_pdf()).expect("parse fixture");
    let on = ConversionOptions {
        extract_tables: true,
        ..Default::default()
    };
    let off = ConversionOptions {
        extract_tables: false,
        ..Default::default()
    };

    let with_tables = doc.extract_text_with_options(0, &on).expect("tables on");
    let without = doc.extract_text_with_options(0, &off).expect("tables off");

    // The fixture must actually reach the table path, or this pins nothing —
    // three previous attempts at this test failed exactly there.
    let tables = doc.extract_tables(0).unwrap_or_default();
    assert!(
        !tables.is_empty(),
        "fixture produced no detected table; the comparison would be vacuous"
    );

    let (a, b) = (token_counts(&with_tables), token_counts(&without));
    for (token, n_off) in &b {
        let n_on = a.get(token).copied().unwrap_or(0);
        assert!(
            n_on >= *n_off,
            "token {token:?} appears {n_off} times with tables off but only {n_on} times with \
             tables on\n  off: {without:?}\n  on:  {with_tables:?}"
        );
    }
}

/// A ruled table where one cell's text is assembled from two separate spans.
/// The cell re-emits both words, so neither span may stay in the flow — the
/// multiplicity budget must not resurrect fragments of merged cells (a
/// fragment's text is not a cell text, so it has no budget entry).
fn merged_cell_table_pdf() -> Vec<u8> {
    let mut content = Vec::new();
    // A closed 2x2 grid: three horizontals, three verticals.
    content.extend_from_slice(b"0.5 w\n");
    for (x0, y0, x1, y1) in [
        (100, 600, 100, 700),
        (250, 600, 250, 700),
        (400, 600, 400, 700),
        (100, 700, 400, 700),
        (100, 650, 400, 650),
        (100, 600, 400, 600),
    ] {
        content.extend_from_slice(format!("{x0} {y0} m {x1} {y1} l S\n").as_bytes());
    }
    // Top-left cell holds TWO spans; the other cells hold one each. One
    // BT/ET block per run keeps the two fragments as separate spans — inside
    // a single block they merge into one span equal to the cell text and the
    // fragment case is never exercised.
    for (x, y, text) in [
        (110, 670, "Alpha"),
        (160, 670, "Beta"),
        (260, 670, "Gamma"),
        (110, 620, "Delta"),
        (260, 620, "Epsilon"),
    ] {
        content.extend_from_slice(
            format!("BT /F1 10 Tf 1 0 0 1 {x} {y} Tm ({text}) Tj ET\n").as_bytes(),
        );
    }
    build_minimal_pdf_raw(&content, b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]")
}

/// Turning table extraction on must neither delete tokens nor emit them twice:
/// a span whose content the table re-emits as part of a merged cell has to
/// leave the flow even though its own text never appears as a cell text.
#[test]
fn table_extraction_does_not_duplicate_merged_cell_fragments() {
    let doc = PdfDocument::from_bytes(merged_cell_table_pdf()).expect("parse fixture");
    let on = ConversionOptions {
        extract_tables: true,
        ..Default::default()
    };
    let off = ConversionOptions {
        extract_tables: false,
        ..Default::default()
    };

    let with_tables = doc.extract_text_with_options(0, &on).expect("tables on");
    let without = doc.extract_text_with_options(0, &off).expect("tables off");

    let tables = doc.extract_tables(0).unwrap_or_default();
    assert!(
        !tables.is_empty(),
        "fixture produced no detected table; the comparison would be vacuous"
    );
    // The comparison below only pins the fragment case if some cell actually
    // merged two spans into one text.
    assert!(
        tables
            .iter()
            .flat_map(|t| t.rows.iter())
            .flat_map(|r| r.cells.iter())
            .any(|c| c.text.split_whitespace().count() >= 2),
        "no cell merged multiple spans; the fragment case is not exercised"
    );

    let (a, b) = (token_counts(&with_tables), token_counts(&without));
    for (token, n_off) in &b {
        let n_on = a.get(token).copied().unwrap_or(0);
        assert!(
            n_on >= *n_off,
            "token {token:?} appears {n_off} times with tables off but only {n_on} times with \
             tables on\n  off: {without:?}\n  on:  {with_tables:?}"
        );
        assert!(
            n_on <= *n_off,
            "token {token:?} appears {n_on} times with tables on but only {n_off} times with \
             tables off — a merged-cell fragment stayed in the flow\n  off: {without:?}\n  on:  \
             {with_tables:?}"
        );
    }
}

/// A ruled table transcribed from the failing financial-page geometry: the
/// header band is ruled across every column, but below it only the left
/// columns have row-separator rules. The detector merges each right-hand
/// column into ONE tall cell for the whole body. Such a cell's text does not
/// include every value its bbox covers, so dropping spans on bbox containment
/// alone deletes the column's figures while the cell re-emits none of them.
fn unruled_value_column_table_pdf() -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(b"0.5 w\n");
    let lines: [(i32, i32, i32, i32); 8] = [
        // Verticals, full height.
        (100, 600, 100, 700),
        (250, 600, 250, 700),
        (400, 600, 400, 700),
        (460, 600, 460, 700),
        // Header band and outer frame span every column...
        (100, 700, 460, 700),
        (100, 675, 460, 675),
        (100, 600, 460, 600),
        // ...but the interior row rule stops at the value column.
        (100, 640, 400, 640),
    ];
    for (x0, y0, x1, y1) in lines {
        content.extend_from_slice(format!("{x0} {y0} m {x1} {y1} l S\n").as_bytes());
    }
    for (fs, x, y, text) in [
        (10, 110, 682, "Item"),
        (10, 260, 682, "Qty"),
        (10, 410, 682, "Net"),
        (10, 110, 655, "alpha"),
        (10, 260, 655, "10"),
        (10, 110, 615, "beta"),
        (10, 260, 615, "20"),
        // Small figure inside the tall un-ruled value cell.
        (6, 410, 646, "-1,004"),
    ] {
        content.extend_from_slice(
            format!("BT /F1 {fs} Tf 1 0 0 1 {x} {y} Tm ({text}) Tj ET\n").as_bytes(),
        );
    }
    build_minimal_pdf_raw(&content, b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]")
}

/// A value covered only by a cell that does not carry its text must survive:
/// that cell re-emits none of it, so dropping the span deletes real content.
#[test]
fn table_extraction_keeps_values_in_unruled_columns() {
    let doc = PdfDocument::from_bytes(unruled_value_column_table_pdf()).expect("parse fixture");
    let on = ConversionOptions {
        extract_tables: true,
        ..Default::default()
    };
    let off = ConversionOptions {
        extract_tables: false,
        ..Default::default()
    };

    let with_tables = doc.extract_text_with_options(0, &on).expect("tables on");
    let without = doc.extract_text_with_options(0, &off).expect("tables off");

    assert!(
        !doc.extract_tables(0).unwrap_or_default().is_empty(),
        "fixture produced no detected table; the comparison would be vacuous"
    );

    let (a, b) = (token_counts(&with_tables), token_counts(&without));
    for (token, n_off) in &b {
        let n_on = a.get(token).copied().unwrap_or(0);
        assert!(
            n_on >= *n_off,
            "token {token:?} appears {n_off} times with tables off but only {n_on} times with \
             tables on\n  off: {without:?}\n  on:  {with_tables:?}"
        );
    }
}

/// A ruled table where two same-row spans sit closer than the sub-em space
/// threshold (0.15 em): the table side glues them into one compound token
/// ("12,3" + "45" -> "12,345"), so neither span's own text survives in the
/// cell's budget material. Different fonts keep the two runs as separate
/// FLOW spans (a span carries a single font), so each must be absorbed as a
/// fragment of the glued token — verbatim token equality cannot.
fn glued_cell_table_pdf() -> Vec<u8> {
    let mut content = Vec::new();
    // A closed 2x2 grid: three horizontals, three verticals.
    content.extend_from_slice(b"0.5 w\n");
    for (x0, y0, x1, y1) in [
        (100, 600, 100, 700),
        (250, 600, 250, 700),
        (400, 600, 400, 700),
        (100, 700, 400, 700),
        (100, 650, 400, 650),
        (100, 600, 400, 600),
    ] {
        content.extend_from_slice(format!("{x0} {y0} m {x1} {y1} l S\n").as_bytes());
    }
    // "12,3" in Helvetica ends near x=129.5; "45" in Courier starts at
    // x=130 — a sub-point gap, far below the 1.5pt space threshold at
    // 10pt, so the cell builder joins them with no separator.
    content.extend_from_slice(b"BT /F1 10 Tf 1 0 0 1 110 670 Tm (12,3) Tj ET\n");
    content.extend_from_slice(b"BT /F2 10 Tf 1 0 0 1 130 670 Tm (45) Tj ET\n");
    for (x, y, text) in [
        (260, 670, "Gamma"),
        (110, 620, "Delta"),
        (260, 620, "Epsilon"),
    ] {
        content.extend_from_slice(
            format!("BT /F1 10 Tf 1 0 0 1 {x} {y} Tm ({text}) Tj ET\n").as_bytes(),
        );
    }
    build_minimal_pdf_raw(&content, b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]")
}

fn nonspace_char_counts(text: &str) -> HashMap<char, usize> {
    let mut counts = HashMap::new();
    for c in text.chars().filter(|c| !c.is_whitespace()) {
        *counts.entry(c).or_insert(0) += 1;
    }
    counts
}

/// A span whose text the cell builder rewrote away (glued into a compound
/// token) must still leave the flow: the cell re-emits its content, so
/// keeping the span duplicates it. Token boundaries move when spans glue,
/// so compare the non-whitespace character multiset — deletion and
/// duplication both change it, reflow does not.
#[test]
fn table_extraction_absorbs_spans_glued_into_one_cell_token() {
    let doc = PdfDocument::from_bytes(glued_cell_table_pdf()).expect("parse fixture");
    let on = ConversionOptions {
        extract_tables: true,
        ..Default::default()
    };
    let off = ConversionOptions {
        extract_tables: false,
        ..Default::default()
    };

    let with_tables = doc.extract_text_with_options(0, &on).expect("tables on");
    let without = doc.extract_text_with_options(0, &off).expect("tables off");

    let tables = doc.extract_tables(0).unwrap_or_default();
    assert!(
        !tables.is_empty(),
        "fixture produced no detected table; the comparison would be vacuous"
    );
    // The comparison below only pins the rewrite case if the table side
    // actually glued the two runs into one compound token.
    assert!(
        tables
            .iter()
            .flat_map(|t| t.rows.iter())
            .flat_map(|r| r.cells.iter())
            .any(|c| c.text.split_whitespace().any(|t| t == "12,345")),
        "no cell glued the two runs into one token; the rewrite case is not exercised"
    );

    let (a, b) = (nonspace_char_counts(&with_tables), nonspace_char_counts(&without));
    assert_eq!(
        a, b,
        "tables on/off disagree on character content — a glued span was duplicated or \
         deleted\n  off: {without:?}\n  on:  {with_tables:?}"
    );
}

fn build_minimal_pdf_raw(content: &[u8], page_extra: &[u8]) -> Vec<u8> {
    let mut pdf = b"%PDF-1.4\n".to_vec();

    let off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    let off2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    let off3 = pdf.len();
    pdf.extend_from_slice(b"3 0 obj\n<< ");
    pdf.extend_from_slice(page_extra);
    pdf.extend_from_slice(
        b" /Contents 4 0 R /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> >>\nendobj\n",
    );

    let off4 = pdf.len();
    pdf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    let off5 = pdf.len();
    pdf.extend_from_slice(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>\nendobj\n",
    );

    let off6 = pdf.len();
    pdf.extend_from_slice(
        b"6 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>\nendobj\n",
    );

    let xref_pos = pdf.len();
    let offsets = [0usize, off1, off2, off3, off4, off5, off6];
    pdf.extend_from_slice(format!("xref\n0 {}\n", offsets.len()).as_bytes());
    pdf.extend_from_slice(format!("{:010} 65535 f\r\n", 0).as_bytes());
    for &off in &offsets[1..] {
        pdf.extend_from_slice(format!("{off:010} 00000 n\r\n").as_bytes());
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
