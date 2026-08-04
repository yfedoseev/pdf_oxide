//! A tagged-PDF table cell whose marked-content sequence carries several text
//! runs (a wrapped line, or a gap-separated pair) must keep every run — the
//! cell owns all content sharing its MCID (ISO 32000-1 §14.8.4.3.4), not just
//! the first block encountered.

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::PdfDocument;

/// Tagged 2×2 table: cell(0,0) holds MCID 0 which wraps over two lines
/// ("Hello" / "World"); the other cells hold one run each.
fn tagged_grid_pdf() -> Vec<u8> {
    let content: &[u8] = b"BT /F1 12 Tf
/P <</MCID 0>> BDC
1 0 0 1 72 700 Tm (Hello) Tj
1 0 0 1 72 686 Tm (World) Tj
EMC
/P <</MCID 1>> BDC
1 0 0 1 200 700 Tm (Alpha) Tj
EMC
/P <</MCID 2>> BDC
1 0 0 1 72 660 Tm (Beta) Tj
EMC
/P <</MCID 3>> BDC
1 0 0 1 200 660 Tm (Gamma) Tj
EMC
ET";
    let mut stream = format!("<< /Length {} >>\nstream\n", content.len()).into_bytes();
    stream.extend_from_slice(content);
    stream.extend_from_slice(b"\nendstream");
    let bodies: Vec<(usize, Vec<u8>)> = vec![
        (
            1,
            b"<< /Type /Catalog /Pages 2 0 R /StructTreeRoot 10 0 R /MarkInfo << /Marked true >> >>"
                .to_vec(),
        ),
        (2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec()),
        (
            3,
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R /StructParents 0 >>"
                .to_vec(),
        ),
        (4, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec()),
        (5, stream),
        (10, b"<< /Type /StructTreeRoot /K [11 0 R] >>".to_vec()),
        (11, b"<< /Type /StructElem /S /Table /P 10 0 R /K [12 0 R 13 0 R] >>".to_vec()),
        (12, b"<< /Type /StructElem /S /TR /P 11 0 R /K [14 0 R 15 0 R] >>".to_vec()),
        (13, b"<< /Type /StructElem /S /TR /P 11 0 R /K [16 0 R 17 0 R] >>".to_vec()),
        (14, b"<< /Type /StructElem /S /TD /P 12 0 R /Pg 3 0 R /K 0 >>".to_vec()),
        (15, b"<< /Type /StructElem /S /TD /P 12 0 R /Pg 3 0 R /K 1 >>".to_vec()),
        (16, b"<< /Type /StructElem /S /TD /P 13 0 R /Pg 3 0 R /K 2 >>".to_vec()),
        (17, b"<< /Type /StructElem /S /TD /P 13 0 R /Pg 3 0 R /K 3 >>".to_vec()),
    ];

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    let max_id = 17;
    let mut offsets = vec![0usize; max_id + 1];
    for (id, body) in &bodies {
        offsets[*id] = out.len();
        out.extend_from_slice(format!("{id} 0 obj\n").as_bytes());
        out.extend_from_slice(body);
        out.extend_from_slice(b"\nendobj\n");
    }
    let xref = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", max_id + 1).as_bytes());
    for id in 1..=max_id {
        if offsets[id] != 0 {
            out.extend_from_slice(format!("{:010} 00000 n \n", offsets[id]).as_bytes());
        } else {
            out.extend_from_slice(b"0000000000 65535 f \n");
        }
    }
    out.extend_from_slice(
        format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n", max_id + 1)
            .as_bytes(),
    );
    out
}

#[test]
fn wrapped_cell_keeps_both_lines_in_markdown() {
    let doc = PdfDocument::from_bytes(tagged_grid_pdf()).expect("parse");
    let opts = ConversionOptions {
        extract_tables: true,
        ..Default::default()
    };
    let md = doc.to_markdown(0, &opts).expect("to_markdown");

    let first_row = md
        .lines()
        .find(|l| l.contains("Hello"))
        .unwrap_or_else(|| panic!("no row containing Hello:\n{md}"));
    assert!(
        first_row.contains("World"),
        "\"World\" dropped from cell(0,0); row = {first_row:?}\nfull markdown:\n{md}"
    );
    for word in ["Alpha", "Beta", "Gamma"] {
        assert!(md.contains(word), "{word:?} missing from table:\n{md}");
    }
}

// ---- Same defect through the direct structure API ----

use pdf_oxide::layout::{Color, FontWeight, TextSpan};
use pdf_oxide::structure::{
    extract_table_from_spans, McidScope, StructChild, StructElem, StructType,
};

fn span(text: &str, x: f32, y: f32, mcid: u32, seq: usize) -> TextSpan {
    TextSpan {
        provenance: None,
        artifact_type: None,
        text: text.to_string(),
        bbox: pdf_oxide::geometry::Rect::new(x, y, 30.0, 12.0),
        font_name: "Helvetica".to_string(),
        font_size: 12.0,
        font_weight: FontWeight::Normal,
        is_italic: false,
        is_monospace: false,
        color: Color::black(),
        mcid: Some(mcid),
        mcid_scope: None,
        sequence: seq,
        offset_semantic: false,
        split_boundary_before: false,
        char_spacing: 0.0,
        word_spacing: 0.0,
        horizontal_scaling: 100.0,
        primary_detected: false,
        char_widths: vec![],
        char_x_offsets: Vec::new(),
        heading_level: None,
        rotation_degrees: 0.0,
        wmode: 0,
        text_rise: 0.0,
        rtl_draw_logical: false,
    }
}

fn td(mcid: u32) -> StructElem {
    let mut e = StructElem::new(StructType::TD);
    e.add_child(StructChild::MarkedContentRef {
        mcid,
        page: 0,
        scope: McidScope::Page(0),
    });
    e
}

fn tr(cells: Vec<StructElem>) -> StructElem {
    let mut r = StructElem::new(StructType::TR);
    for c in cells {
        r.add_child(StructChild::StructElem(Box::new(c)));
    }
    r
}

#[test]
fn cell_with_two_wrapped_lines_keeps_both() {
    let spans = vec![
        span("Hello", 72.0, 700.0, 0, 0),
        span("World", 72.0, 686.0, 0, 1),
        span("Alpha", 200.0, 700.0, 1, 2),
    ];
    let mut table = StructElem::new(StructType::Table);
    table.add_child(StructChild::StructElem(Box::new(tr(vec![td(0), td(1)]))));
    let t = extract_table_from_spans(&table, &spans).unwrap();
    assert_eq!(
        t.rows[0].cells[0].text, "Hello World",
        "second span sharing the cell's MCID was dropped"
    );
}

#[test]
fn cell_with_gap_separated_spans_keeps_both() {
    let spans = vec![
        span("Total", 72.0, 700.0, 0, 0),
        span("100", 300.0, 700.0, 0, 1),
        span("x", 400.0, 700.0, 1, 2),
    ];
    let mut table = StructElem::new(StructType::Table);
    table.add_child(StructChild::StructElem(Box::new(tr(vec![td(0), td(1)]))));
    let t = extract_table_from_spans(&table, &spans).unwrap();
    assert_eq!(
        t.rows[0].cells[0].text, "Total 100",
        "second span sharing the cell's MCID was dropped"
    );
}
