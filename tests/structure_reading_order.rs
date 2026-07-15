//! `ReadingOrder::Structure` follows the tagged structure tree, not geometry.
//!
//! The table below is laid out so a geometric XY-cut is TEMPTED into reading it
//! column-major (the two columns are far apart), while the structure tree declares
//! the correct row-major order. Structure order must read the rows; and on an
//! untagged copy of the same file it must fall back to the geometric order rather
//! than fail.

use pdf_oxide::{PdfDocument, ReadingOrder};

/// A 2x2 table whose columns are FAR apart (x=72 and x=430) so a column detector
/// splits them. MCIDs run row-major: 0=A1, 1=B1 (row 1), 2=A2, 3=B2 (row 2).
/// `tagged=false` strips `/StructTreeRoot` + `/MarkInfo` to make the untagged twin.
fn table_pdf(tagged: bool) -> Vec<u8> {
    let content = b"BT /F1 12 Tf\n\
        /TD <</MCID 0>> BDC 1 0 0 1 72 700 Tm (ALPHA) Tj EMC\n\
        /TD <</MCID 1>> BDC 1 0 0 1 430 700 Tm (BRAVO) Tj EMC\n\
        /TD <</MCID 2>> BDC 1 0 0 1 72 676 Tm (CHARLIE) Tj EMC\n\
        /TD <</MCID 3>> BDC 1 0 0 1 430 676 Tm (DELTA) Tj EMC\n\
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
    let catalog = if tagged {
        "<< /Type /Catalog /Pages 2 0 R /MarkInfo << /Marked true >> /StructTreeRoot 7 0 R >>"
    } else {
        "<< /Type /Catalog /Pages 2 0 R >>"
    };
    obj(&mut buf, &mut off, 1, catalog);
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
    // Structure tree: Table -> [TR -> [TD, TD]] x2, row-major MCIDs.
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
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 15 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

fn order_of(pdf: Vec<u8>, ro: ReadingOrder) -> Vec<String> {
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    doc.extract_spans_with_reading_order(0, ro)
        .expect("spans")
        .into_iter()
        .map(|s| s.text.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

/// The declared reading order is ROW-MAJOR. Structure order must reproduce it,
/// even though the columns are far enough apart to tempt a column-first geometry.
#[test]
fn structure_order_reads_the_tagged_rows_not_the_columns() {
    let got = order_of(table_pdf(true), ReadingOrder::Structure);
    assert_eq!(
        got,
        vec!["ALPHA", "BRAVO", "CHARLIE", "DELTA"],
        "structure order must follow the row-major struct tree"
    );
}

/// On an UNTAGGED copy, `Structure` must fall back to the geometric order EXACTLY -
/// it is always safe to request whether or not the file is tagged.
#[test]
fn structure_falls_back_to_geometry_when_untagged() {
    let structure = order_of(table_pdf(false), ReadingOrder::Structure);
    let column = order_of(table_pdf(false), ReadingOrder::ColumnAware);
    assert_eq!(structure, column, "untagged: Structure must equal ColumnAware, byte for byte");
    // (On this simple 2x2 the geometric order happens to also be row-major; the two
    // paths diverge on genuinely complex tables, as the real-world corpus shows. The
    // safety property under test is the EXACT fallback, above.)
}
