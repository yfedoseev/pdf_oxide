//! Integration test: a tagged table (Table/TR/TD) extracted as plain text must
//! place each row on its own line — one newline between rows, not the geometric
//! blank line a ~1.7em row pitch would otherwise produce.
//!
//! ISO 32000-1 §14.8.4.3.4: table rows are stacked block-level rows, not
//! free-leading paragraphs. The PDF is hand-built (no third-party fixture).

use pdf_oxide::PdfDocument;

/// One page with a 2x2 table tagged Table → TR → TD → MCID. Row one is at
/// y=700, row two at y=676 (a 24pt pitch ≈ 1.7x the 12pt leading, which the
/// geometric `num_breaks` rounds to a blank line without the table guard).
fn tagged_table_pdf() -> Vec<u8> {
    // Each TD wraps one MCID-marked Tj. Cells are positioned with absolute Tm.
    let content = b"BT /F1 12 Tf\n\
        /TD <</MCID 0>> BDC 1 0 0 1 72 700 Tm (North) Tj EMC\n\
        /TD <</MCID 1>> BDC 1 0 0 1 200 700 Tm (120) Tj EMC\n\
        /TD <</MCID 2>> BDC 1 0 0 1 72 676 Tm (South) Tj EMC\n\
        /TD <</MCID 3>> BDC 1 0 0 1 200 676 Tm (90) Tj EMC\n\
        ET\n";

    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 16];
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
    obj(
        &mut buf,
        &mut off,
        1,
        "<< /Type /Catalog /Pages 2 0 R /MarkInfo << /Marked true >> /StructTreeRoot 7 0 R >>",
    );
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R /StructParents 0 >>",
    );
    stream(&mut buf, &mut off, 4, content);
    obj(
        &mut buf,
        &mut off,
        5,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    );
    // Structure tree: Table → [TR → [TD, TD]] x2, each TD holding one MCID.
    obj(&mut buf, &mut off, 7, "<< /Type /StructTreeRoot /K [8 0 R] >>");
    obj(
        &mut buf,
        &mut off,
        8,
        "<< /Type /StructElem /S /Table /P 7 0 R /K [9 0 R 10 0 R] >>",
    );
    obj(
        &mut buf,
        &mut off,
        9,
        "<< /Type /StructElem /S /TR /P 8 0 R /K [11 0 R 12 0 R] >>",
    );
    obj(
        &mut buf,
        &mut off,
        10,
        "<< /Type /StructElem /S /TR /P 8 0 R /K [13 0 R 14 0 R] >>",
    );
    obj(&mut buf, &mut off, 11, "<< /Type /StructElem /S /TD /P 9 0 R /Pg 3 0 R /K 0 >>");
    obj(&mut buf, &mut off, 12, "<< /Type /StructElem /S /TD /P 9 0 R /Pg 3 0 R /K 1 >>");
    obj(
        &mut buf,
        &mut off,
        13,
        "<< /Type /StructElem /S /TD /P 10 0 R /Pg 3 0 R /K 2 >>",
    );
    obj(
        &mut buf,
        &mut off,
        14,
        "<< /Type /StructElem /S /TD /P 10 0 R /Pg 3 0 R /K 3 >>",
    );

    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 15\n0000000000 65535 f \n");
    for id in 1..=14 {
        // object 6 is unused; emit a free-style placeholder offset of 0.
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 15 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

#[test]
fn tagged_table_rows_separated_by_single_newline() {
    let doc = PdfDocument::from_bytes(tagged_table_pdf()).unwrap();
    let text = doc.extract_text(0).unwrap();

    assert!(text.contains("North"), "table not decoded: {text:?}");
    // Row one and row two are on adjacent lines — one newline, not a blank line.
    assert!(
        text.contains("North 120\nSouth 90") || text.contains("North 120\nSouth"),
        "table rows not on single-newline-separated lines: {text:?}"
    );
    assert!(
        !text.contains("North 120\n\nSouth"),
        "spurious blank line between table rows: {text:?}"
    );
}
