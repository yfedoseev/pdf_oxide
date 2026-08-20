//! A tagged table cell whose marked-content sequence hands over its runs in
//! the reverse of the order the page reads must still assemble the text the
//! reader sees. Ordering the runs by their boxes does that, but only where the
//! boxes describe an upright reading frame.
//!
//! ISO 32000-2 §14.8.2.5.1 asks a producer to store the content of a
//! marked-content sequence in logical order, and §14.8.4.8.3 NOTE builds the
//! table algorithms on that. Producers exist that do not honour it, and draw a
//! column header from the right end of its line back to the left.
//!
//! Three shapes must keep marked-content order instead, because ordering them
//! by box reads the cell backwards:
//!
//! * a page carrying a `/Rotate`, where a run's box is mapped into the
//!   displayed frame while `rotation_degrees` still reports the frame before
//!   the turn;
//! * text turned by its own text matrix on a page carrying no `/Rotate`, which
//!   advances along an axis the row-and-column order does not follow;
//! * right-to-left text, where UAX #9 (I2/L2) keeps a numeric run left to right
//!   inside a right-to-left paragraph. The character-level pass already
//!   resolves that, so a second permutation at run level turns `50%` into
//!   `%50`.
//!
//! The PDFs are hand-built (no third-party fixture).

use pdf_oxide::structure::TableCell;
use pdf_oxide::PdfDocument;


/// Upright text on a page carrying no `/Rotate`, with row one's runs drawn from
/// the right end of the line back to the left. Reading order is ascending x,
/// and the sequence hands the runs over descending — the defect this orders out.
const UPRIGHT: &[u8] = b"BT /F1 8.49 Tf\n\
    /TD <</MCID 0>> BDC \
    1 0 0 1 475.07 482.89 Tm (0^-3 cgs units\\) ) Tj \
    1 0 0 1 470.18 482.89 Tm (1) Tj \
    1 0 0 1 462.64 482.89 Tm ( \\() Tj \
    1 0 0 1 457.10 482.89 Tm (K) Tj EMC\n\
    /TD <</MCID 1>> BDC 1 0 0 1 250 482.89 Tm (cm) Tj EMC\n\
    /TD <</MCID 2>> BDC 1 0 0 1 430 470 Tm (ABC) Tj 1 0 0 1 400 470 Tm (50) Tj EMC\n\
    ET\n";

/// The text matrix is `[0 -1 1 0]`, so every run advances along -y in user
/// space, reads left to right once a `/Rotate 270` is applied, and reports -90
/// degrees. Row one's runs are drawn in the reverse of the order the page
/// reads: the tail `0^-3 cgs units) ` first and the leading `K` last.
const TURNED: &[u8] = b"BT /F1 8.49 Tf\n\
    /TD <</MCID 0>> BDC \
    0 -1 1 0 482.89 308.44 Tm (0^-3 cgs units\\) ) Tj \
    0 -1 1 0 482.89 313.33 Tm (1) Tj \
    0 -1 1 0 482.89 320.87 Tm ( \\() Tj \
    0 -1 1 0 482.89 326.41 Tm (K) Tj EMC\n\
    /TD <</MCID 1>> BDC 0 -1 1 0 482.89 250 Tm (cm) Tj EMC\n\
    /TD <</MCID 2>> BDC 0 -1 1 0 470 400 Tm (ABC) Tj 0 -1 1 0 470 430 Tm (50) Tj EMC\n\
    ET\n";

/// The text matrix is `[-1 0 0 -1]`, so the runs report 180 degrees and advance
/// along -x on a page carrying no `/Rotate`. Row one's runs are drawn in the
/// order they read, `K` first at the right end of the line, so ordering them by
/// box — ascending x within the row — reverses the cell.
const HALF_TURNED: &[u8] = b"BT /F1 8.49 Tf\n\
    /TD <</MCID 0>> BDC \
    -1 0 0 -1 475.07 482.89 Tm (K) Tj \
    -1 0 0 -1 470.18 482.89 Tm ( \\() Tj \
    -1 0 0 -1 462.64 482.89 Tm (1) Tj \
    -1 0 0 -1 457.10 482.89 Tm (0^-3 cgs units\\) ) Tj EMC\n\
    /TD <</MCID 1>> BDC -1 0 0 -1 250 482.89 Tm (cm) Tj EMC\n\
    /TD <</MCID 2>> BDC -1 0 0 -1 430 470 Tm (ABC) Tj -1 0 0 -1 400 470 Tm (50) Tj EMC\n\
    ET\n";

/// One page holding a tagged two-row table, drawn by `content`.
///
/// Row one, cell one is a single marked-content sequence of four abutting runs
/// that assemble `K (10^-3 cgs units)`. Each run starts where the one beside it
/// ends, so the runs carry no gap for a joiner to find and the cell text is
/// decided by their order alone. The runs sit on one line at 8.49 pt and 8.49
/// units apart, the frame a real producer of this shape emits.
///
/// Cell two is an ordinary single-run cell. Row two holds one cell whose
/// sequence mixes a Hebrew word at the right end of the line with a number at
/// the left, drawn right to left — which for right-to-left text *is* reading
/// order.
fn tagged_table_pdf(rotate: i32, content: &[u8]) -> Vec<u8> {
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
        &format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Rotate {rotate} \
             /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R /StructParents 0 >>"
        ),
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

/// The cells of the hand-built table, row by row, as the page rotation reports.
fn table_cells(doc: &PdfDocument, page_rotation: i32) -> Vec<Vec<TableCell>> {
    let spans = doc.extract_spans(0).unwrap();
    let tree = doc.structure_tree().unwrap().expect("structure tree");
    let tables = pdf_oxide::structure::find_table_elements_all_pages(&tree);
    let elems = tables.get(&0).cloned().unwrap_or_default();
    assert_eq!(elems.len(), 1, "one tagged table on the page");
    pdf_oxide::structure::extract_table_from_spans(&elems[0], &spans, page_rotation)
        .unwrap()
        .rows
        .iter()
        .map(|row| row.cells.clone())
        .collect()
}

#[test]
fn upright_tagged_cell_reads_in_page_order() {
    let doc = PdfDocument::from_bytes(tagged_table_pdf(0, UPRIGHT)).unwrap();
    let rows = table_cells(&doc, 0);

    assert_eq!(
        rows[0][0].text, "K (10^-3 cgs units)",
        "cell assembled in the order the runs were drawn, not the order the page reads"
    );
    assert_eq!(rows[0][1].text, "cm", "single-run cell disturbed");
}

#[test]
fn rotated_page_tagged_cell_keeps_marked_content_order() {
    let doc = PdfDocument::from_bytes(tagged_table_pdf(270, TURNED)).unwrap();
    let rows = table_cells(&doc, 270);

    // The box is in the displayed frame and the reported angle is not, so an
    // order taken from the box walks the runs against the way they advance.
    assert!(
        rows[0][0].text.starts_with('0'),
        "rotated page ordered by box: {:?}",
        rows[0][0].text
    );
}

#[test]
fn turned_text_on_unrotated_page_keeps_marked_content_order() {
    let doc = PdfDocument::from_bytes(tagged_table_pdf(0, HALF_TURNED)).unwrap();
    let rows = table_cells(&doc, 0);

    // Reading order here is descending x. Ordering by box takes the row
    // ascending and hands back the cell reversed.
    assert!(
        rows[0][0].text.starts_with('K'),
        "turned text ordered by box: {:?}",
        rows[0][0].text
    );
}

#[test]
fn right_to_left_tagged_cell_keeps_marked_content_order() {
    let doc = PdfDocument::from_bytes(tagged_table_pdf(0, UPRIGHT)).unwrap();
    let rows = table_cells(&doc, 0);
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
