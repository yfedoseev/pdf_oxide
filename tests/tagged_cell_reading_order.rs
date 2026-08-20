//! A tagged table cell whose marked-content sequence hands over its runs in the
//! reverse of the order the page reads must still assemble the text the reader
//! sees — unless the sequence carries right-to-left text, where the runs keep
//! marked-content order.
//!
//! ISO 32000-2 §14.8.2.5.1 asks a producer to store the content of a
//! marked-content sequence in logical order, and §14.8.4.8.3 NOTE builds the
//! table algorithms on that. Producers exist that do not honour it: on a
//! `/Rotate 270` page a column header is drawn from the right end of its line
//! back to the left, so the runs arrive in descending x and the cell reads
//! backwards.
//!
//! The page rotation is part of the defect. A run's box is mapped into the
//! displayed frame while `rotation_degrees` still reports the frame before the
//! rotation, so an order taken from the reported angle walks the runs against
//! the direction they advance in. Every run here reports -90 degrees on a page
//! displayed unrotated-side-up.
//!
//! The right-to-left case is a guard, not a reproducer: UAX #9 (I2/L2) keeps a
//! numeric run left to right inside a right-to-left paragraph, and the
//! character-level pass already resolves that, so a second permutation of the
//! same text at run level would turn `50%` into `%50`.
//!
//! The PDF is hand-built (no third-party fixture).

use pdf_oxide::structure::TableCell;
use pdf_oxide::PdfDocument;

/// One `/Rotate 270` page holding a tagged two-row table.
///
/// The text matrix is `[0 -1 1 0]`, so every run advances along -y in user
/// space, reads left to right once the page rotation is applied, and reports
/// -90 degrees — the producer shape the defect comes from.
///
/// Row one, cell one is a single marked-content sequence of four abutting runs
/// drawn from the right end of the line back to the left: the tail
/// `0^-3 cgs units) ` first and the leading `K` last. Each run starts where the
/// one that reads before it ends, so the runs carry no gap for a joiner to
/// find and the cell text is decided by their order alone.
///
/// The positions are chosen so the runs land on the frame a real producer of
/// this shape emits: one line at y=482.89, boxes 8.49 wide, and the four runs
/// at x=457.10, 462.64, 470.18 and 475.07 — ascending in reading order, so the
/// sequence hands them over descending.
///
/// Cell two is an ordinary single-run cell. Row two holds one cell whose
/// sequence mixes a Hebrew word at the right end of the line with a number at
/// the left, drawn right to left — which for right-to-left text *is* reading
/// order.
fn rotated_tagged_table_pdf() -> Vec<u8> {
    // Codes A, B, C carry Hebrew through /ToUnicode; the digits map to
    // themselves through WinAnsiEncoding.
    let tounicode = "/CIDInit /ProcSet findresource begin\n\
        12 dict begin begincmap\n\
        1 begincodespacerange <00> <FF> endcodespacerange\n\
        3 beginbfchar\n<41> <05D0>\n<42> <05D1>\n<43> <05D2>\nendbfchar\n\
        endcmap CMapName currentdict /CMap defineresource pop end end";

    // A run drawn at user (x, y) with this matrix lands at displayed
    // (792 - y - w, x), so the user-space y values below place the four runs on
    // the displayed x positions above, and the shared user x places the row on
    // its displayed line. They are drawn in the reverse of reading order.
    let content = b"BT /F1 8.49 Tf\n\
        /TD <</MCID 0>> BDC \
        0 -1 1 0 482.89 308.44 Tm (0^-3 cgs units\\) ) Tj \
        0 -1 1 0 482.89 313.33 Tm (1) Tj \
        0 -1 1 0 482.89 320.87 Tm ( \\() Tj \
        0 -1 1 0 482.89 326.41 Tm (K) Tj EMC\n\
        /TD <</MCID 1>> BDC 0 -1 1 0 482.89 250 Tm (cm) Tj EMC\n\
        /TD <</MCID 2>> BDC 0 -1 1 0 470 400 Tm (ABC) Tj 0 -1 1 0 470 430 Tm (50) Tj EMC\n\
        ET\n";

    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 15];
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
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Rotate 270 \
         /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R /StructParents 0 >>",
    );
    stream(&mut buf, &mut off, 4, content);
    obj(
        &mut buf,
        &mut off,
        5,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica \
         /Encoding /WinAnsiEncoding /ToUnicode 6 0 R >>",
    );
    stream(&mut buf, &mut off, 6, tounicode.as_bytes());
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
    obj(&mut buf, &mut off, 10, "<< /Type /StructElem /S /TR /P 8 0 R /K [13 0 R] >>");
    obj(&mut buf, &mut off, 11, "<< /Type /StructElem /S /TD /P 9 0 R /Pg 3 0 R /K 0 >>");
    obj(&mut buf, &mut off, 12, "<< /Type /StructElem /S /TD /P 9 0 R /Pg 3 0 R /K 1 >>");
    obj(
        &mut buf,
        &mut off,
        13,
        "<< /Type /StructElem /S /TD /P 10 0 R /Pg 3 0 R /K 2 >>",
    );

    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 14\n0000000000 65535 f \n");
    for id in 1..=13 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 14 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

/// The cells of the hand-built table, row by row.
fn table_cells(doc: &PdfDocument) -> Vec<Vec<TableCell>> {
    let spans = doc.extract_spans(0).unwrap();
    let tree = doc.structure_tree().unwrap().expect("structure tree");
    let tables = pdf_oxide::structure::find_table_elements_all_pages(&tree);
    let elems = tables.get(&0).cloned().unwrap_or_default();
    assert_eq!(elems.len(), 1, "one tagged table on the page");
    pdf_oxide::structure::extract_table_from_spans(&elems[0], &spans)
        .unwrap()
        .rows
        .iter()
        .map(|row| row.cells.clone())
        .collect()
}

#[test]
fn rotated_tagged_cell_reads_in_page_order() {
    let doc = PdfDocument::from_bytes(rotated_tagged_table_pdf()).unwrap();
    let rows = table_cells(&doc);

    assert_eq!(
        rows[0][0].text, "K (10^-3 cgs units)",
        "cell assembled in the order the runs were drawn, not the order the page reads"
    );
    assert_eq!(rows[0][1].text, "cm", "single-run cell disturbed");
}

#[test]
fn right_to_left_tagged_cell_keeps_marked_content_order() {
    let doc = PdfDocument::from_bytes(rotated_tagged_table_pdf()).unwrap();
    let rows = table_cells(&doc);
    let cell = &rows[1][0];

    // Reading order here runs right to left, so the leftmost run is the last
    // one read. Ordering the runs left to right would start the cell at the
    // number — the block-level flip that turns "50%" into "%50".
    assert!(
        cell.text.contains('\u{05D0}'),
        "right-to-left cell lost its Hebrew run: {:?}",
        cell.text
    );
    assert!(
        !cell.text.starts_with('5'),
        "right-to-left cell started at the number: {:?}",
        cell.text
    );
}
