//! A rule-bounded row-group of a booktabs table is a table even where its
//! cells are not boxed.
//!
//! A results table drawn the booktabs way: full-width rules between its
//! row-groups, three hairlines between a few of its columns running the whole
//! table's height, and the last row of every group shaded with a rectangle
//! per cell. The intersection grid is built from closed cells, so it reads
//! the groups whose shaded row and hairlines close their cells and nothing
//! for a group's unshaded rows — bounded above and below by rules and
//! crossed by the hairlines, but with no side on their outer cells. The
//! grid's slice of such a group is its one shaded row, which the
//! section-divider split isolates and the validity filter drops, and the
//! group's rows fall to the prose flow while the groups beside it read as
//! tables. On the page this models, two of three groups were tables and the
//! third was emitted as bold fragments.
//!
//! ISO 32000-1:2008 §14.8.4.3.4 makes a row the element that holds its cells
//! (`docs/spec/pdf.md`:37805); a row-group between two rules is a run of such
//! rows whether or not every cell is boxed. So the rule bands are read as
//! well, and a band's table is kept where no grid table covers it.

use pdf_oxide::PdfDocument;

const X0: f32 = 108.0;
const X1: f32 = 502.0;
/// Column left edges: a wide label column, then nine numeric columns.
const COLS: [f32; 10] = [
    112.0, 212.0, 246.0, 275.0, 305.0, 337.0, 367.0, 397.0, 423.0, 468.0,
];
/// Hairlines between columns 2|3, 5|6 and 8|9, as the page draws them.
const HAIRLINES: [f32; 3] = [241.5, 333.0, 418.8];
const ROW: f32 = 9.2;
const ROWS_PER_GROUP: usize = 8;
const SIZE: f32 = 6.8;

fn cell(c: &mut String, x: f32, y: f32, text: &str) {
    c.push_str(&format!("BT /F1 {SIZE} Tf 1 0 0 1 {x:.2} {y:.2} Tm ({text}) Tj ET\n"));
}

/// One row-group: eight rows of a label and nine numbers, the last row
/// shaded cell by cell, between two full-width rules.
fn group(c: &mut String, top: f32, label: &str) {
    let labels = [
        format!("{label} (baseline)"),
        "+ Step skip".to_string(),
        "+ Cache reuse".to_string(),
        "+ Delta blocks".to_string(),
        "+ Token cache".to_string(),
        "+ Region skip".to_string(),
        "+ Sparse attention".to_string(),
        "+ Ours".to_string(),
    ];
    for (r, label) in labels.iter().enumerate() {
        let y_top = top - r as f32 * ROW;
        let baseline = y_top - ROW + 2.4;
        if r == ROWS_PER_GROUP - 1 {
            // The shaded last row: one rectangle per cell. Where a hairline
            // runs between two columns the rectangles stop at its edges, as
            // the page's do — so the shaded row's top edge never crosses the
            // hairline, and the unshaded rows above have no closed cell.
            c.push_str("0.92 g\n");
            let mut edges: Vec<(f32, f32)> = Vec::new();
            let mut left = X0;
            for x in COLS.iter().skip(1) {
                let boundary = x - 4.0;
                match HAIRLINES.iter().find(|h| (*h - boundary).abs() < 3.0) {
                    Some(h) => {
                        edges.push((left, h - 0.15));
                        left = h + 0.15;
                    },
                    None => {
                        edges.push((left, boundary));
                        left = boundary;
                    },
                }
            }
            edges.push((left, X1));
            for (x0, x1) in edges {
                c.push_str(&format!("{x0:.2} {:.2} {:.2} {ROW:.2} re f\n", y_top - ROW, x1 - x0));
            }
            c.push_str("0 g\n");
        }
        cell(c, COLS[0], baseline, label);
        cell(c, COLS[1], baseline, if r == 0 { "" } else { "T" });
        for (k, x) in COLS.iter().enumerate().skip(2) {
            let v = 20.0 + (r * 7 + k * 3) as f32 * 0.137;
            cell(c, *x, baseline, &format!("{v:.3}"));
        }
    }
}

fn page() -> Vec<u8> {
    let mut c = String::new();
    let group_h = ROWS_PER_GROUP as f32 * ROW;
    let header_top = 661.0;
    // Rules: above and below the header, and below every group.
    let mut rule_ys = vec![header_top, header_top - 18.6];
    for g in 0..3 {
        rule_ys.push(header_top - 18.6 - (g as f32 + 1.0) * group_h);
    }
    c.push_str("0.5 w\n");
    for y in &rule_ys {
        c.push_str(&format!("{X0:.2} {y:.2} m {X1:.2} {y:.2} l S\n"));
    }
    // Hairlines: thin filled rectangles running the whole table height.
    let table_bottom = *rule_ys.last().unwrap();
    for x in HAIRLINES {
        c.push_str(&format!(
            "{x:.2} {table_bottom:.2} 0.3 {:.2} re f\n",
            header_top - table_bottom
        ));
    }
    // Header row.
    let hb = header_top - 18.6 + 5.0;
    cell(&mut c, COLS[0], hb, "Model");
    cell(&mut c, COLS[1], hb, "Type");
    for (k, x) in COLS.iter().enumerate().skip(2) {
        cell(
            &mut c,
            *x,
            hb,
            [
                "PSNR", "SSIM", "LPIPS", "G-SC", "G-PQ", "G-O", "Latency", "Speedup",
            ][k - 2],
        );
    }
    // Three groups.
    for (g, label) in ["Alpha-Edit", "Beta Context", "Gamma-Image-Edit"]
        .iter()
        .enumerate()
    {
        group(&mut c, header_top - 18.6 - g as f32 * group_h, label);
    }
    build_page(&c)
}

#[test]
fn every_rule_bounded_group_is_a_table_with_all_its_rows() {
    let doc = PdfDocument::from_bytes(page()).expect("open");
    let tables = doc.extract_tables(0).expect("tables");
    // The grid may read two adjacent groups as one table; what must hold is
    // that every group's rows are table rows — none of the three labels is
    // left to the prose flow.
    let groups: Vec<&pdf_oxide::structure::table_extractor::Table> =
        tables.iter().filter(|t| t.rows.len() >= 7).collect();
    for label in ["Alpha-Edit", "Beta Context", "Gamma-Image-Edit"] {
        let in_a_table = groups.iter().any(|t| {
            t.rows
                .iter()
                .any(|r| r.cells.iter().any(|c| c.text.contains(label)))
        });
        assert!(
            in_a_table,
            "{label:?}'s group is read as a table; got {:?}",
            tables
                .iter()
                .map(|t| (
                    t.rows.len(),
                    t.col_count,
                    t.bbox.map(|b| (b.y.round(), (b.y + b.height).round()))
                ))
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn the_groups_rows_are_table_rows_in_markdown() {
    let doc = PdfDocument::from_bytes(page()).expect("open");
    let md = doc
        .to_markdown(
            0,
            &pdf_oxide::converters::ConversionOptions {
                extract_tables: true,
                ..Default::default()
            },
        )
        .expect("md");
    for label in ["Alpha-Edit", "Beta Context", "Gamma-Image-Edit"] {
        let row = md.lines().find(|l| l.contains(label)).unwrap_or_default();
        assert!(row.starts_with('|'), "{label:?} is emitted as a table row, not prose: {row:?}");
    }
}

/// Minimal single-content-stream page writer.
fn build_page(content: &str) -> Vec<u8> {
    let content = content.as_bytes().to_vec();
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
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
            .to_vec(),
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
