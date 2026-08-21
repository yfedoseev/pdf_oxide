//! A tagged cell must take only the marked-content sequences its own content
//! stream emits.
//!
//! An MCID is numbered within the content stream that draws it (ISO 32000-1
//! §14.7.4.3), not within the page. A page and a Form XObject drawn on it may
//! both number a sequence 0 for unrelated content, so a cell that owns the
//! page's sequence 0 must not also take the form's.
//!
//! The PDF is hand-built (no third-party fixture).

use pdf_oxide::PdfDocument;

/// One page whose content stream tags `CELLZERO`/`CELLONE` as MCID 0/1, plus a
/// Form XObject drawn on it that tags `FORMZERO`/`FORMONE` as MCID 0/1. The
/// structure tree holds one table row of two cells, both page-scoped.
fn page_and_form_share_mcids_pdf() -> Vec<u8> {
    let page_content = b"BT /F1 12 Tf 1 0 0 1 60 700 Tm\n\
        /P <</MCID 0>> BDC (CELLZERO) Tj EMC\n\
        1 0 0 1 200 700 Tm\n\
        /P <</MCID 1>> BDC (CELLONE) Tj EMC\n\
        ET\n\
        q 1 0 0 1 0 0 cm /Fm1 Do Q\n";
    let form_content = b"BT /F1 12 Tf 1 0 0 1 60 400 Tm\n\
        /P <</MCID 0>> BDC (FORMZERO) Tj EMC\n\
        1 0 0 1 200 400 Tm\n\
        /P <</MCID 1>> BDC (FORMONE) Tj EMC\n\
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
    obj(&mut buf, &mut off, 1,
        "<< /Type /Catalog /Pages 2 0 R /StructTreeRoot 10 0 R /MarkInfo << /Marked true >> >>");
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    obj(&mut buf, &mut off, 3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R \
         /Resources << /Font << /F1 5 0 R >> /XObject << /Fm1 6 0 R >> >> /StructParents 0 >>");
    stream(&mut buf, &mut off, 4, "", page_content);
    obj(&mut buf, &mut off, 5, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");
    stream(&mut buf, &mut off, 6,
        "/Type /XObject /Subtype /Form /BBox [0 0 612 792] \
         /Resources << /Font << /F1 5 0 R >> >>", form_content);
    obj(&mut buf, &mut off, 10, "<< /Type /StructTreeRoot /K [11 0 R] >>");
    obj(&mut buf, &mut off, 11,
        "<< /Type /StructElem /S /Table /P 10 0 R /Pg 3 0 R /K [12 0 R] >>");
    obj(&mut buf, &mut off, 12,
        "<< /Type /StructElem /S /TR /P 11 0 R /Pg 3 0 R /K [13 0 R 14 0 R] >>");
    obj(&mut buf, &mut off, 13,
        "<< /Type /StructElem /S /TD /P 12 0 R /Pg 3 0 R \
         /K [<< /Type /MCR /Pg 3 0 R /MCID 0 >>] >>");
    obj(&mut buf, &mut off, 14,
        "<< /Type /StructElem /S /TD /P 12 0 R /Pg 3 0 R \
         /K [<< /Type /MCR /Pg 3 0 R /MCID 1 >>] >>");

    let ids = [1usize, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14];
    let xref = buf.len();
    buf.extend_from_slice(b"xref\n");
    for &id in &ids {
        buf.extend_from_slice(format!("{id} 1\n{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 21 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

#[test]
fn cell_takes_only_its_own_streams_marked_content() {
    let doc = PdfDocument::from_bytes(page_and_form_share_mcids_pdf()).unwrap();
    let spans = doc.extract_spans(0).unwrap();
    let tree = doc.structure_tree().unwrap().expect("structure tree");
    let tables = pdf_oxide::structure::find_table_elements_all_pages(&tree);
    let elems = tables.get(&0).cloned().unwrap_or_default();
    assert_eq!(elems.len(), 1, "one tagged table on the page");
    let table = pdf_oxide::structure::extract_table_from_spans(&elems[0], &spans).unwrap();
    let row = &table.rows[0];

    assert_eq!(
        row.cells[0].text, "CELLZERO",
        "cell absorbed the form's sequence 0: {:?}",
        row.cells[0].text
    );
    assert_eq!(
        row.cells[1].text, "CELLONE",
        "cell absorbed the form's sequence 1: {:?}",
        row.cells[1].text
    );
}
