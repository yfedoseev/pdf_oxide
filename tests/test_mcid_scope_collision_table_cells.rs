//! A marked-content id is scoped to the content stream that defines it, so a
//! page and a Form XObject may each number theirs from 0.
//!
//! ISO 32000-1:2008 §14.7.4.2 requires an `/MCID` to be unique only "within
//! its content stream", and §14.7.4.3 gives marked-content reference
//! dictionaries an `/Stm` entry naming that stream precisely so a consumer can
//! tell the two apart. This crate models it as `(McidScope, mcid)` and
//! computes the right key at the boundary.
//!
//! The table-reordering lookup then threw it away: on a miss it retried the
//! *page* namespace, so a form's MCID 0 matched the table's MCID 0 and the
//! form's text was reordered into a table cell. Every other scoped lookup in
//! the same file has no such retry.

use pdf_oxide::PdfDocument;

/// A tagged page carrying a 2x2 table whose cells are page-scoped MCIDs 0..3,
/// plus a Form XObject drawn below it whose own content stream restarts
/// marked-content numbering at 0.
fn colliding_mcid_pdf() -> Vec<u8> {
    // Page stream: four tagged table cells, then the form.
    let content = b"BT /F1 12 Tf\n\
        /TD <</MCID 0>> BDC 1 0 0 1 72 700 Tm (North) Tj EMC\n\
        /TD <</MCID 1>> BDC 1 0 0 1 200 700 Tm (120) Tj EMC\n\
        /TD <</MCID 2>> BDC 1 0 0 1 72 676 Tm (South) Tj EMC\n\
        /TD <</MCID 3>> BDC 1 0 0 1 200 676 Tm (90) Tj EMC\n\
        ET\n\
        q 1 0 0 1 0 0 cm /Fm0 Do Q\n";

    // Form stream: its own MCID 0, tagged as a paragraph, well below the table.
    let form = b"BT /F1 12 Tf\n\
        /P <</MCID 0>> BDC 1 0 0 1 72 600 Tm (Formtext) Tj EMC\n\
        ET\n";

    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 21];
    let obj = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, body: &str| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };
    let stream = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, dict: &str, data: &[u8]| {
        off[id] = buf.len();
        buf.extend_from_slice(
            format!("{id} 0 obj\n<< {dict} /Length {} >>\nstream\n", data.len()).as_bytes(),
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
         /Resources << /Font << /F1 5 0 R >> /XObject << /Fm0 15 0 R >> >> \
         /Contents 4 0 R /StructParents 0 >>",
    );
    stream(&mut buf, &mut off, 4, "", content);
    obj(
        &mut buf,
        &mut off,
        5,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    );
    // Structure tree: Table -> TR -> TD x4 (page-scoped), plus a P whose
    // content lives in the form stream and is named through /Stm.
    obj(&mut buf, &mut off, 7, "<< /Type /StructTreeRoot /K [8 0 R 20 0 R] >>");
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
    // The Form XObject itself.
    stream(
        &mut buf,
        &mut off,
        15,
        "/Type /XObject /Subtype /Form /BBox [0 0 612 792] \
         /Resources << /Font << /F1 5 0 R >> >>",
        form,
    );
    // The paragraph element, pointing at MCID 0 *in the form's stream*.
    obj(
        &mut buf,
        &mut off,
        20,
        "<< /Type /StructElem /S /P /P 7 0 R /K \
         << /Type /MCR /Pg 3 0 R /Stm 15 0 R /MCID 0 >> >>",
    );

    let xref = buf.len();
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", 21).as_bytes());
    for id in 1..=20 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 21 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

/// The form's text must not be pulled into the table's cell ordering.
#[test]
fn form_scoped_mcid_does_not_land_in_a_table_cell() {
    let doc = PdfDocument::from_bytes(colliding_mcid_pdf()).expect("parse");
    let text = doc.extract_text(0).expect("extract");

    assert!(text.contains("Formtext"), "form text missing entirely: {text:?}");
    assert!(text.contains("North"), "table text missing entirely: {text:?}");

    // The table's own cells must stay in structure order, uninterrupted.
    let north = text.find("North").expect("North");
    let one_twenty = text.find("120").expect("120");
    let south = text.find("South").expect("South");
    let ninety = text.find("90").expect("90");
    let formtext = text.find("Formtext").expect("Formtext");

    assert!(
        north < one_twenty && one_twenty < south && south < ninety,
        "table cells reordered: {text:?}"
    );
    assert!(
        formtext > ninety,
        "the form's text was reordered into the table (it is drawn below it): {text:?}"
    );
}

/// A page-scoped MCID must still resolve — the retry was wrong, but the
/// primary lookup it guarded is not.
#[test]
fn page_scoped_table_cells_still_reorder_by_structure() {
    let doc = PdfDocument::from_bytes(colliding_mcid_pdf()).expect("parse");
    let text = doc.extract_text(0).expect("extract");
    // Row-major structure order, which is also this fixture's visual order.
    let compact: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(
        compact.contains("North 120") && compact.contains("South 90"),
        "table rows not assembled in structure order: {compact:?}"
    );
}
