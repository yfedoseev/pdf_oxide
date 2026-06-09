//! Integration test: an untagged page laid out as two side-by-side columns
//! must be read column-major — the whole left column top-to-bottom, then the
//! whole right column — not interleaved row-by-row across the gutter.
//!
//! Untagged PDFs carry no logical-structure reading-order hint (ISO 32000-1
//! §14.8), so the order has to be recovered from layout (XY-Cut, §9.4). A short
//! article with only a few wrapped lines per column is the case the
//! span-count-based histogram detector is blind to; the clean empty gutter is
//! still unambiguous. PDF is hand-built (no third-party fixture).

use pdf_oxide::PdfDocument;

/// One untagged page: a heading across the top, then two text columns.
/// Left column starts at x=72, right column at x=330, with a wide empty gutter
/// between them. Each column has four lines sharing the same y-bands as the
/// other column, so a naive Y-then-X sort interleaves them across the gutter.
/// Each run is its own `BT … ET` text object (as real producers emit), so the
/// left and right runs on a shared baseline stay distinct spans.
fn two_column_pdf() -> Vec<u8> {
    let content = b"/F1 11 Tf\n\
        BT 1 0 0 1 72 740 Tm (Two Column Heading) Tj ET\n\
        BT 1 0 0 1 72 700 Tm (the left column begins the article here) Tj ET\n\
        BT 1 0 0 1 330 700 Tm (the right column continues after the left) Tj ET\n\
        BT 1 0 0 1 72 680 Tm (and should be read first top to bottom) Tj ET\n\
        BT 1 0 0 1 330 680 Tm (column has fully ended at its last line) Tj ET\n\
        BT 1 0 0 1 72 660 Tm (before the reader moves over the gutter) Tj ET\n\
        BT 1 0 0 1 330 660 Tm (an extractor reading rows across fails) Tj ET\n\
        BT 1 0 0 1 72 640 Tm (to the second column on the right side) Tj ET\n\
        BT 1 0 0 1 330 640 Tm (this two column ordering test entirely) Tj ET\n";

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
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
    );
    stream(&mut buf, &mut off, 4, content);
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

#[test]
fn untagged_two_column_reads_column_major() {
    let doc = PdfDocument::from_bytes(two_column_pdf()).unwrap();
    let text = doc.extract_text(0).unwrap();

    // Every left-column line must appear before every right-column line.
    let last_left = text
        .find("to the second column on the right side")
        .expect("left column missing");
    let first_right = text
        .find("the right column continues after the left")
        .expect("right column missing");
    assert!(
        last_left < first_right,
        "columns interleaved — left column not read fully before the right:\n{text}"
    );

    // The four left lines stay in vertical (top-to-bottom) order.
    let l1 = text
        .find("the left column begins the article here")
        .unwrap();
    let l2 = text.find("and should be read first top to bottom").unwrap();
    let l3 = text
        .find("before the reader moves over the gutter")
        .unwrap();
    let l4 = text.find("to the second column on the right side").unwrap();
    assert!(l1 < l2 && l2 < l3 && l3 < l4, "left column lines out of order:\n{text}");

    // The heading leads the page.
    let title = text.find("Two Column Heading").unwrap();
    assert!(title < l1, "heading not first:\n{text}");
}
