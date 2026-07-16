//! `ReadingOrder::Structure` follows the tagged structure tree, not geometry.
//!
//! The fixture is a 4x2 table whose two columns are FAR apart (x=72 and x=430),
//! so the geometric `ColumnAware` order splits it into two columns and reads it
//! COLUMN-MAJOR. The tagged structure tree declares the correct ROW-MAJOR order.
//! The tests assert the two orders genuinely DIFFER and that `Structure`
//! reproduces the row-major tree; on an untagged copy `Structure` must fall back
//! to the geometric order exactly.

use pdf_oxide::{PdfDocument, ReadingOrder};

/// A 4x2 table. Columns are FAR apart (x=72, x=430) so a column detector splits
/// them; four rows (>= 8 spans) give the detector enough evidence to fire. MCIDs
/// run ROW-MAJOR: 0,1 = row 1 (A,B); 2,3 = row 2 (C,D); 4,5 = row 3 (E,F);
/// 6,7 = row 4 (G,H). `tagged=false` strips `/StructTreeRoot` + `/MarkInfo`.
fn table_pdf(tagged: bool) -> Vec<u8> {
    let content = b"BT /F1 12 Tf\n\
        /TD <</MCID 0>> BDC 1 0 0 1 72 650 Tm (ALPHA) Tj EMC\n\
        /TD <</MCID 1>> BDC 1 0 0 1 430 650 Tm (BRAVO) Tj EMC\n\
        /TD <</MCID 2>> BDC 1 0 0 1 72 620 Tm (CHARLIE) Tj EMC\n\
        /TD <</MCID 3>> BDC 1 0 0 1 430 620 Tm (DELTA) Tj EMC\n\
        /TD <</MCID 4>> BDC 1 0 0 1 72 700 Tm (ECHO) Tj EMC\n\
        /TD <</MCID 5>> BDC 1 0 0 1 430 700 Tm (FOXTROT) Tj EMC\n\
        /TD <</MCID 6>> BDC 1 0 0 1 72 680 Tm (GOLF) Tj EMC\n\
        /TD <</MCID 7>> BDC 1 0 0 1 430 680 Tm (HOTEL) Tj EMC\n\
        ET\n";

    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 24];
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
    // Structure tree: Table -> [TR -> [TD, TD]] x4, row-major MCIDs.
    // Rows are objects 9..=12; the eight TD cells are objects 13..=20, MCID = obj-13.
    obj(&mut buf, &mut off, 7, "<< /Type /StructTreeRoot /K [8 0 R] >>");
    obj(
        &mut buf,
        &mut off,
        8,
        "<< /Type /StructElem /S /Table /P 7 0 R /K [9 0 R 10 0 R 11 0 R 12 0 R] >>",
    );
    for (row, obj_id) in (9..=12usize).enumerate() {
        let td0 = 13 + row * 2;
        let td1 = td0 + 1;
        obj(
            &mut buf,
            &mut off,
            obj_id,
            &format!("<< /Type /StructElem /S /TR /P 8 0 R /K [{td0} 0 R {td1} 0 R] >>"),
        );
    }
    for cell in 0..8usize {
        let obj_id = 13 + cell;
        let parent = 9 + cell / 2;
        obj(
            &mut buf,
            &mut off,
            obj_id,
            &format!("<< /Type /StructElem /S /TD /P {parent} 0 R /Pg 3 0 R /K {cell} >>"),
        );
    }

    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 21\n0000000000 65535 f \n");
    for id in 1..=20 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 21 /Root 1 0 R >>\nstartxref\n");
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

// Row-major logical order declared by the structure tree.
const ROW_MAJOR: [&str; 8] = [
    "ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT", "GOLF", "HOTEL",
];
// Geometric order for this fixture: the rows are DRAWN out of logical sequence
// (logical row 3 sits highest on the page), so a position-based reader visits
// them by Y - rows 3, 4, 1, 2 - which is NOT the logical order.
const GEOMETRIC: [&str; 8] = [
    "ECHO", "FOXTROT", "GOLF", "HOTEL", "ALPHA", "BRAVO", "CHARLIE", "DELTA",
];

/// The load-bearing test: `Structure` must actually REORDER, not silently behave
/// like `ColumnAware`. On the tagged bytes the geometric order and the structure
/// order genuinely differ, and only `Structure` recovers the declared row order.
#[test]
fn structure_reorders_table_to_logical_order() {
    let column = order_of(table_pdf(true), ReadingOrder::ColumnAware);
    let structure = order_of(table_pdf(true), ReadingOrder::Structure);

    // Geometry follows the (scrambled) visual layout...
    assert_eq!(column, GEOMETRIC, "ColumnAware should follow visual position");
    // ...while Structure follows the tagged row order...
    assert_eq!(structure, ROW_MAJOR, "Structure should follow the struct tree");
    // ...and the two are genuinely different, so the tagged test cannot pass with
    // a no-op Structure branch.
    assert_ne!(column, structure, "Structure must differ from ColumnAware here");
}

/// On an UNTAGGED copy, `Structure` must fall back to the geometric order EXACTLY -
/// it is always safe to request whether or not the file is tagged.
#[test]
fn structure_falls_back_to_geometry_when_untagged() {
    let structure = order_of(table_pdf(false), ReadingOrder::Structure);
    let column = order_of(table_pdf(false), ReadingOrder::ColumnAware);
    assert_eq!(structure, column, "untagged: Structure must equal ColumnAware, byte for byte");
    assert_eq!(structure, GEOMETRIC, "untagged: falls back to visual/geometric order");
}

/// Assemble a single-page tagged PDF from a content stream and a list of
/// structure-tree object BODIES. `struct_objs[0]` is object 7 (the
/// `/StructTreeRoot`), `struct_objs[1]` is object 8, and so on - so element
/// bodies reference each other by `(7 + index) 0 R`. Objects 1..=5 (catalog,
/// pages, page, content, font) are fixed; object 6 is intentionally unused.
fn tagged_pdf(content: &[u8], struct_objs: &[String]) -> Vec<u8> {
    let last = 6 + struct_objs.len();
    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; last + 2];
    let obj = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, body: &str| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
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
    off[4] = buf.len();
    buf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    buf.extend_from_slice(content);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
    obj(
        &mut buf,
        &mut off,
        5,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    );
    for (i, body) in struct_objs.iter().enumerate() {
        obj(&mut buf, &mut off, 7 + i, body);
    }

    let xref = buf.len();
    buf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", last + 1).as_bytes());
    for id in 1..=last {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n", last + 1).as_bytes(),
    );
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

/// A cell with MULTIPLE spans must keep them contiguous and place the whole cell
/// at its structure rank. Here cell 0 holds "FOO"+"BAR"; cell 1 ("BAZ") is drawn
/// ABOVE it, so geometry reads BAZ first. Structure must read the cells in tree
/// order and keep FOO,BAR together.
#[test]
fn structure_keeps_multi_span_cell_contiguous() {
    // FOO and BAR share cell 0 but sit on two lines (same x, 15 pt apart) so they
    // stay two spans rather than merging into one line.
    let content = b"BT /F1 12 Tf\n\
        /TD <</MCID 0>> BDC 1 0 0 1 72 650 Tm (FOO) Tj 1 0 0 1 72 635 Tm (BAR) Tj EMC\n\
        /TD <</MCID 1>> BDC 1 0 0 1 72 700 Tm (BAZ) Tj EMC\n\
        ET\n";
    let structs = vec![
        "<< /Type /StructTreeRoot /K [8 0 R] >>".to_string(),
        "<< /Type /StructElem /S /Table /P 7 0 R /K [9 0 R 10 0 R] >>".to_string(),
        "<< /Type /StructElem /S /TR /P 8 0 R /K [11 0 R] >>".to_string(),
        "<< /Type /StructElem /S /TR /P 8 0 R /K [12 0 R] >>".to_string(),
        "<< /Type /StructElem /S /TD /P 9 0 R /Pg 3 0 R /K 0 >>".to_string(),
        "<< /Type /StructElem /S /TD /P 10 0 R /Pg 3 0 R /K 1 >>".to_string(),
    ];
    let pdf = tagged_pdf(content, &structs);
    assert_eq!(order_of(pdf.clone(), ReadingOrder::ColumnAware), vec!["BAZ", "FOO", "BAR"]);
    assert_eq!(order_of(pdf, ReadingOrder::Structure), vec!["FOO", "BAR", "BAZ"]);
}

/// Header cells tagged `/TH` participate in structure order exactly like `/TD`.
/// The data row is drawn ABOVE the header row, so only structure order puts the
/// header first.
#[test]
fn structure_orders_th_header_row_first() {
    let content = b"BT /F1 12 Tf\n\
        /TH <</MCID 0>> BDC 1 0 0 1 72 650 Tm (HKEY) Tj EMC\n\
        /TH <</MCID 1>> BDC 1 0 0 1 430 650 Tm (HVAL) Tj EMC\n\
        /TD <</MCID 2>> BDC 1 0 0 1 72 700 Tm (DKEY) Tj EMC\n\
        /TD <</MCID 3>> BDC 1 0 0 1 430 700 Tm (DVAL) Tj EMC\n\
        ET\n";
    let structs = vec![
        "<< /Type /StructTreeRoot /K [8 0 R] >>".to_string(),
        "<< /Type /StructElem /S /Table /P 7 0 R /K [9 0 R 10 0 R] >>".to_string(),
        "<< /Type /StructElem /S /TR /P 8 0 R /K [11 0 R 12 0 R] >>".to_string(),
        "<< /Type /StructElem /S /TR /P 8 0 R /K [13 0 R 14 0 R] >>".to_string(),
        "<< /Type /StructElem /S /TH /P 9 0 R /Pg 3 0 R /K 0 >>".to_string(),
        "<< /Type /StructElem /S /TH /P 9 0 R /Pg 3 0 R /K 1 >>".to_string(),
        "<< /Type /StructElem /S /TD /P 10 0 R /Pg 3 0 R /K 2 >>".to_string(),
        "<< /Type /StructElem /S /TD /P 10 0 R /Pg 3 0 R /K 3 >>".to_string(),
    ];
    let pdf = tagged_pdf(content, &structs);
    assert_eq!(
        order_of(pdf.clone(), ReadingOrder::ColumnAware),
        vec!["DKEY", "DVAL", "HKEY", "HVAL"],
        "geometry reads the visually-higher data row first"
    );
    assert_eq!(
        order_of(pdf, ReadingOrder::Structure),
        vec!["HKEY", "HVAL", "DKEY", "DVAL"],
        "structure puts the tagged header row first"
    );
}

/// A nested table's cells take their place by structure PRE-ORDER: the outer
/// cell, then the nested table's cells. The outer cell is drawn LOWEST so geometry
/// disagrees with the tree.
#[test]
fn structure_orders_nested_table_cells_by_preorder() {
    let content = b"BT /F1 12 Tf\n\
        /TD <</MCID 0>> BDC 1 0 0 1 72 640 Tm (OUTER) Tj EMC\n\
        /TD <</MCID 1>> BDC 1 0 0 1 72 700 Tm (NESTX) Tj EMC\n\
        /TD <</MCID 2>> BDC 1 0 0 1 72 680 Tm (NESTY) Tj EMC\n\
        ET\n";
    // Outer Table(8) -> TR(9) -> TD(11, MCID 0)
    //                -> TR(10) -> TD(12) -> nested Table(13) -> TR(14)
    //                                       -> TD(15, MCID 1), TD(16, MCID 2)
    let structs = vec![
        "<< /Type /StructTreeRoot /K [8 0 R] >>".to_string(),
        "<< /Type /StructElem /S /Table /P 7 0 R /K [9 0 R 10 0 R] >>".to_string(),
        "<< /Type /StructElem /S /TR /P 8 0 R /K [11 0 R] >>".to_string(),
        "<< /Type /StructElem /S /TR /P 8 0 R /K [12 0 R] >>".to_string(),
        "<< /Type /StructElem /S /TD /P 9 0 R /Pg 3 0 R /K 0 >>".to_string(),
        "<< /Type /StructElem /S /TD /P 10 0 R /K [13 0 R] >>".to_string(),
        "<< /Type /StructElem /S /Table /P 12 0 R /K [14 0 R] >>".to_string(),
        "<< /Type /StructElem /S /TR /P 13 0 R /K [15 0 R 16 0 R] >>".to_string(),
        "<< /Type /StructElem /S /TD /P 14 0 R /Pg 3 0 R /K 1 >>".to_string(),
        "<< /Type /StructElem /S /TD /P 14 0 R /Pg 3 0 R /K 2 >>".to_string(),
    ];
    let pdf = tagged_pdf(content, &structs);
    assert_eq!(
        order_of(pdf.clone(), ReadingOrder::ColumnAware),
        vec!["NESTX", "NESTY", "OUTER"],
        "geometry follows visual Y"
    );
    assert_eq!(
        order_of(pdf, ReadingOrder::Structure),
        vec!["OUTER", "NESTX", "NESTY"],
        "structure pre-order: outer cell, then nested cells"
    );
}
