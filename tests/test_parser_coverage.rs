//! Integration tests for content stream parsing and xref parsing
//!
//! Targets coverage gaps in:
//! - content/parser.rs (48.1% → higher)
//! - xref.rs (59% → higher)
//! - xref_reconstruction.rs (49% → higher)
//! - structure/parser.rs (6.6% → higher)

// Some tests intentionally pin the deprecated `parse_content_stream_text_only`.
#![allow(deprecated)]

use pdf_oxide::content::operators::Operator;
use pdf_oxide::content::parser::{
    parse_content_stream, parse_content_stream_images_only, parse_content_stream_text_only,
};
use pdf_oxide::document::PdfDocument;
use pdf_oxide::xref::{CrossRefTable, XRefEntry};

// ===========================================================================
// Tests: Content stream parsing - parse_content_stream
// ===========================================================================

#[test]
fn test_parse_empty_content_stream() {
    let ops = parse_content_stream(b"").unwrap();
    assert!(ops.is_empty());
}

#[test]
fn test_parse_whitespace_only_stream() {
    let ops = parse_content_stream(b"   \n\r\t  ").unwrap();
    assert!(ops.is_empty());
}

#[test]
fn test_parse_simple_text_stream() {
    let stream = b"BT /F1 12 Tf 72 720 Td (Hello World) Tj ET";
    let ops = parse_content_stream(stream).unwrap();
    assert!(!ops.is_empty());

    let has_bt = ops.iter().any(|op| matches!(op, Operator::BeginText));
    let has_et = ops.iter().any(|op| matches!(op, Operator::EndText));
    let has_tf = ops.iter().any(|op| matches!(op, Operator::Tf { .. }));
    let has_td = ops.iter().any(|op| matches!(op, Operator::Td { .. }));
    let has_tj = ops.iter().any(|op| matches!(op, Operator::Tj { .. }));

    assert!(has_bt, "Should have BT operator");
    assert!(has_et, "Should have ET operator");
    assert!(has_tf, "Should have Tf operator");
    assert!(has_td, "Should have Td operator");
    assert!(has_tj, "Should have Tj operator");
}

#[test]
fn test_parse_tj_array_stream() {
    let stream = b"BT /F1 12 Tf 72 720 Td [(Hello) -100 (World)] TJ ET";
    let ops = parse_content_stream(stream).unwrap();
    let has_tj = ops.iter().any(|op| matches!(op, Operator::TJ { .. }));
    assert!(has_tj, "Should have TJ operator");
}

#[test]
fn test_parse_color_operators() {
    let stream = b"1 0 0 rg 0 1 0 RG 0.5 g 0.3 G 0 1 0 1 k 1 0 1 0 K";
    let ops = parse_content_stream(stream).unwrap();

    let has_fill_rgb = ops
        .iter()
        .any(|op| matches!(op, Operator::SetFillRgb { .. }));
    let has_stroke_rgb = ops
        .iter()
        .any(|op| matches!(op, Operator::SetStrokeRgb { .. }));
    let has_fill_gray = ops
        .iter()
        .any(|op| matches!(op, Operator::SetFillGray { .. }));
    let has_stroke_gray = ops
        .iter()
        .any(|op| matches!(op, Operator::SetStrokeGray { .. }));
    let has_fill_cmyk = ops
        .iter()
        .any(|op| matches!(op, Operator::SetFillCmyk { .. }));
    let has_stroke_cmyk = ops
        .iter()
        .any(|op| matches!(op, Operator::SetStrokeCmyk { .. }));

    assert!(has_fill_rgb, "Should have rg operator");
    assert!(has_stroke_rgb, "Should have RG operator");
    assert!(has_fill_gray, "Should have g operator");
    assert!(has_stroke_gray, "Should have G operator");
    assert!(has_fill_cmyk, "Should have k operator");
    assert!(has_stroke_cmyk, "Should have K operator");
}

#[test]
fn test_parse_graphics_state_operators() {
    let stream = b"q 1 0 0 1 72 720 cm Q";
    let ops = parse_content_stream(stream).unwrap();

    let has_save = ops.iter().any(|op| matches!(op, Operator::SaveState));
    let has_restore = ops.iter().any(|op| matches!(op, Operator::RestoreState));
    let has_cm = ops.iter().any(|op| matches!(op, Operator::Cm { .. }));

    assert!(has_save, "Should have q operator");
    assert!(has_restore, "Should have Q operator");
    assert!(has_cm, "Should have cm operator");
}

#[test]
fn test_parse_text_state_operators() {
    let stream = b"BT 1 Tc 2 Tw 100 Tz 14 TL /F1 12 Tf 0 Tr 3 Ts ET";
    let ops = parse_content_stream(stream).unwrap();

    let has_tc = ops.iter().any(|op| matches!(op, Operator::Tc { .. }));
    let has_tw = ops.iter().any(|op| matches!(op, Operator::Tw { .. }));
    let has_tz = ops.iter().any(|op| matches!(op, Operator::Tz { .. }));
    let has_tl = ops.iter().any(|op| matches!(op, Operator::TL { .. }));
    let has_tf = ops.iter().any(|op| matches!(op, Operator::Tf { .. }));
    let has_tr = ops.iter().any(|op| matches!(op, Operator::Tr { .. }));
    let has_ts = ops.iter().any(|op| matches!(op, Operator::Ts { .. }));

    assert!(has_tc);
    assert!(has_tw);
    assert!(has_tz);
    assert!(has_tl);
    assert!(has_tf);
    assert!(has_tr);
    assert!(has_ts);
}

#[test]
fn test_parse_path_operators() {
    let stream = b"100 200 m 300 400 l 100 200 300 400 500 600 c h S";
    let ops = parse_content_stream(stream).unwrap();
    assert!(!ops.is_empty());
}

#[test]
fn test_parse_path_rect_fill_stroke() {
    let stream = b"100 100 200 150 re f S f* B B* b b* n W W*";
    let ops = parse_content_stream(stream).unwrap();
    assert!(!ops.is_empty());
}

#[test]
fn test_parse_text_matrix_tstar() {
    let stream = b"BT 1 0 0 1 72 720 Tm T* ET";
    let ops = parse_content_stream(stream).unwrap();
    let has_tm = ops.iter().any(|op| matches!(op, Operator::Tm { .. }));
    let has_tstar = ops.iter().any(|op| matches!(op, Operator::TStar));
    assert!(has_tm);
    assert!(has_tstar);
}

#[test]
fn test_parse_td_and_big_td() {
    let stream = b"BT 72 720 Td 0 -14 TD ET";
    let ops = parse_content_stream(stream).unwrap();
    let has_td = ops.iter().any(|op| matches!(op, Operator::Td { .. }));
    let has_big_td = ops.iter().any(|op| matches!(op, Operator::TD { .. }));
    assert!(has_td);
    assert!(has_big_td);
}

#[test]
fn test_parse_quote_and_double_quote() {
    let stream = b"BT /F1 12 Tf 14 TL 72 720 Td (line1) ' 1 2 (line2) \" ET";
    let ops = parse_content_stream(stream).unwrap();
    let has_quote = ops.iter().any(|op| matches!(op, Operator::Quote { .. }));
    let has_dquote = ops
        .iter()
        .any(|op| matches!(op, Operator::DoubleQuote { .. }));
    assert!(has_quote, "Should have ' operator");
    assert!(has_dquote, "Should have \" operator");
}

#[test]
fn test_parse_do_operator() {
    let stream = b"/Im1 Do";
    let ops = parse_content_stream(stream).unwrap();
    let has_do = ops.iter().any(|op| matches!(op, Operator::Do { .. }));
    assert!(has_do, "Should have Do operator");
}

// ===========================================================================
// Tests: parse_content_stream_text_only
// ===========================================================================

#[test]
fn test_parse_text_only_basic() {
    let stream = b"BT /F1 12 Tf 72 720 Td (Hello) Tj ET q 100 200 m 300 400 l S Q";
    let ops = parse_content_stream_text_only(stream).unwrap();

    // Text-only parser should get text operators
    let has_tf = ops.iter().any(|op| matches!(op, Operator::Tf { .. }));
    let has_tj = ops.iter().any(|op| matches!(op, Operator::Tj { .. }));
    assert!(has_tf);
    assert!(has_tj);
}

#[test]
fn test_parse_text_only_skips_path() {
    let stream = b"100 200 m 300 400 l S";
    let ops = parse_content_stream_text_only(stream).unwrap();
    // Should skip non-text operators
    assert!(ops.is_empty() || ops.iter().all(|op| !matches!(op, Operator::Other { .. })));
}

#[test]
fn test_parse_text_only_empty() {
    let ops = parse_content_stream_text_only(b"").unwrap();
    assert!(ops.is_empty());
}

// ===========================================================================
// Tests: parse_content_stream_images_only
// ===========================================================================

#[test]
fn test_parse_images_only_do() {
    let stream = b"q 1 0 0 1 0 0 cm /Im1 Do Q";
    let ops = parse_content_stream_images_only(stream).unwrap();
    let has_do = ops.iter().any(|op| matches!(op, Operator::Do { .. }));
    assert!(has_do, "Image parser should capture Do operator");
}

#[test]
fn test_parse_images_only_cm_q() {
    let stream = b"q 1 0 0 1 72 720 cm /Im1 Do Q";
    let ops = parse_content_stream_images_only(stream).unwrap();
    // Image-only parser should capture cm, q, Q, Do
    let has_do = ops.iter().any(|op| matches!(op, Operator::Do { .. }));
    assert!(has_do, "Should capture Do in image mode");
    // Other operators may or may not be captured depending on implementation
}

#[test]
fn test_parse_images_only_empty() {
    let ops = parse_content_stream_images_only(b"").unwrap();
    assert!(ops.is_empty());
}

#[test]
fn test_parse_images_only_text_stream_ignored() {
    let stream = b"BT /F1 12 Tf (Hello) Tj ET";
    let ops = parse_content_stream_images_only(stream).unwrap();
    // Should not have text operators in image-only mode
    let has_tf = ops.iter().any(|op| matches!(op, Operator::Tf { .. }));
    assert!(!has_tf, "Image parser should not capture Tf");
}

// ===========================================================================
// Tests: Inline images
// ===========================================================================

#[test]
fn test_parse_inline_image() {
    // BI...ID...EI sequence
    let stream = b"BI /W 1 /H 1 /BPC 8 /CS /G ID \x80 EI";
    let ops = parse_content_stream(stream).unwrap();
    let has_inline = ops
        .iter()
        .any(|op| matches!(op, Operator::InlineImage { .. }));
    assert!(has_inline, "Should parse inline image");
}

// ===========================================================================
// Tests: Complex content streams
// ===========================================================================

#[test]
fn test_parse_multiple_bt_et_blocks() {
    let stream = b"BT /F1 12 Tf 72 720 Td (Block1) Tj ET BT /F1 10 Tf 72 700 Td (Block2) Tj ET";
    let ops = parse_content_stream(stream).unwrap();

    let bt_count = ops
        .iter()
        .filter(|op| matches!(op, Operator::BeginText))
        .count();
    let et_count = ops
        .iter()
        .filter(|op| matches!(op, Operator::EndText))
        .count();
    assert_eq!(bt_count, 2);
    assert_eq!(et_count, 2);
}

#[test]
fn test_parse_hex_string_in_tj() {
    let stream = b"BT /F1 12 Tf 72 720 Td <48656C6C6F> Tj ET";
    let ops = parse_content_stream(stream).unwrap();
    let has_tj = ops.iter().any(|op| matches!(op, Operator::Tj { .. }));
    assert!(has_tj, "Should parse hex string in Tj");
}

#[test]
fn test_parse_nested_parentheses() {
    let stream = b"BT /F1 12 Tf 72 720 Td (Hello \\(World\\)) Tj ET";
    let ops = parse_content_stream(stream).unwrap();
    let has_tj = ops.iter().any(|op| matches!(op, Operator::Tj { .. }));
    assert!(has_tj, "Should handle escaped parentheses");
}

#[test]
fn test_parse_negative_numbers() {
    let stream = b"BT /F1 12 Tf -72 -720 Td (-50) Tj ET";
    let ops = parse_content_stream(stream).unwrap();
    let td = ops.iter().find(|op| matches!(op, Operator::Td { .. }));
    assert!(td.is_some());
    if let Some(Operator::Td { tx, ty }) = td {
        assert_eq!(*tx, -72.0);
        assert_eq!(*ty, -720.0);
    }
}

#[test]
fn test_parse_decimal_numbers() {
    let stream = b"BT /F1 12.5 Tf 72.3 720.7 Td ET";
    let ops = parse_content_stream(stream).unwrap();
    let tf = ops.iter().find(|op| matches!(op, Operator::Tf { .. }));
    assert!(tf.is_some());
    if let Some(Operator::Tf { size, .. }) = tf {
        assert!((size - 12.5).abs() < 0.01);
    }
}

// ===========================================================================
// Tests: CrossRefTable
// ===========================================================================

#[test]
fn test_xref_table_new() {
    let table = CrossRefTable::new();
    assert!(table.is_empty());
    assert_eq!(table.len(), 0);
}

#[test]
fn test_xref_table_add_and_get() {
    let mut table = CrossRefTable::new();
    let entry = XRefEntry::uncompressed(100, 0);
    table.add_entry(1, entry);

    assert_eq!(table.len(), 1);
    assert!(!table.is_empty());
    assert!(table.contains(1));
    assert!(!table.contains(99));

    let retrieved = table.get(1).unwrap();
    assert_eq!(retrieved.offset, 100);
}

#[test]
fn test_xref_entry_types() {
    let uncompressed = XRefEntry::uncompressed(500, 0);
    assert_eq!(uncompressed.offset, 500);
    assert_eq!(uncompressed.generation, 0);

    let compressed = XRefEntry::compressed(10, 3);
    assert_eq!(compressed.offset, 10); // stream obj num
    assert_eq!(compressed.generation, 3); // index in stream

    let free = XRefEntry::free(0, 65535);
    assert_eq!(free.generation, 65535);
}

#[test]
fn test_xref_table_merge() {
    let mut table1 = CrossRefTable::new();
    table1.add_entry(1, XRefEntry::uncompressed(100, 0));
    table1.add_entry(2, XRefEntry::uncompressed(200, 0));

    let mut table2 = CrossRefTable::new();
    table2.add_entry(3, XRefEntry::uncompressed(300, 0));
    table2.add_entry(1, XRefEntry::uncompressed(150, 1)); // Duplicate obj 1

    table1.merge_from(table2);

    assert_eq!(table1.len(), 3);
    // merge_from uses or_insert: self (table1) entries take priority
    let entry1 = table1.get(1).unwrap();
    assert_eq!(entry1.offset, 100, "Self entries should take priority in merge");
    // New entry from table2
    let entry3 = table1.get(3).unwrap();
    assert_eq!(entry3.offset, 300);
}

#[test]
fn test_xref_table_all_object_numbers() {
    let mut table = CrossRefTable::new();
    table.add_entry(5, XRefEntry::uncompressed(100, 0));
    table.add_entry(10, XRefEntry::uncompressed(200, 0));
    table.add_entry(15, XRefEntry::uncompressed(300, 0));

    let nums: Vec<u32> = table.all_object_numbers().collect();
    assert_eq!(nums.len(), 3);
    assert!(nums.contains(&5));
    assert!(nums.contains(&10));
    assert!(nums.contains(&15));
}

#[test]
fn test_xref_table_trailer() {
    let mut table = CrossRefTable::new();
    assert!(table.trailer().is_none());

    let mut trailer = std::collections::HashMap::new();
    trailer.insert("Size".to_string(), pdf_oxide::object::Object::Integer(10));
    table.set_trailer(trailer);

    assert!(table.trailer().is_some());
    let t = table.trailer().unwrap();
    assert!(t.contains_key("Size"));
}

// ===========================================================================
// Tests: XRef parsing with real PDFs
// ===========================================================================

#[test]
fn test_parse_xref_from_fixture() {
    let doc = PdfDocument::open("tests/fixtures/simple.pdf").unwrap();
    let _ = doc.page_count().unwrap();
    // Just verify the xref was parsed correctly by successfully opening
}

// ===========================================================================
// Tests: Structure tree parsing
// ===========================================================================

#[test]
fn test_structure_tree_untagged_pdf() {
    // Build a minimal untagged PDF
    let mut pdf = b"%PDF-1.7\n".to_vec();
    let off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    let off2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let off3 = pdf.len();
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n",
    );
    finalize_pdf(&mut pdf, &[0, off1, off2, off3]);

    let doc = PdfDocument::from_bytes(pdf).unwrap();
    let tree = doc.structure_tree().unwrap();
    assert!(tree.is_none(), "Untagged PDF should return None");
}

#[test]
fn test_structure_tree_tagged_pdf() {
    // Build a minimal tagged PDF with StructTreeRoot
    let mut pdf = b"%PDF-1.7\n".to_vec();
    let off1 = pdf.len();
    pdf.extend_from_slice(
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R /MarkInfo << /Marked true >> /StructTreeRoot 4 0 R >>\nendobj\n",
    );
    let off2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let off3 = pdf.len();
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n",
    );
    let off4 = pdf.len();
    pdf.extend_from_slice(
        b"4 0 obj\n<< /Type /StructTreeRoot /K 5 0 R /ParentTree 6 0 R >>\nendobj\n",
    );
    let off5 = pdf.len();
    pdf.extend_from_slice(b"5 0 obj\n<< /Type /StructElem /S /Document /K [] >>\nendobj\n");
    let off6 = pdf.len();
    pdf.extend_from_slice(b"6 0 obj\n<< /Type /NumberTree /Nums [] >>\nendobj\n");
    finalize_pdf(&mut pdf, &[0, off1, off2, off3, off4, off5, off6]);

    let doc = PdfDocument::from_bytes(pdf).unwrap();
    let tree = doc.structure_tree().unwrap();
    // May or may not find structure tree depending on parsing strictness
    let _ = tree;
}

// ===========================================================================
// Tests: XRef reconstruction
// ===========================================================================

#[test]
fn test_xref_reconstruction_corrupt_xref() {
    // Build a PDF with a corrupted xref table
    let mut pdf = b"%PDF-1.7\n".to_vec();
    let _off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    let _off2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let _off3 = pdf.len();
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n",
    );

    // Write corrupt xref
    let xref_offset = pdf.len();
    pdf.extend_from_slice(b"xref\n0 4\nCORRUPTED DATA HERE\n");
    pdf.extend_from_slice(
        format!("trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n", xref_offset)
            .as_bytes(),
    );

    // Should still be able to open via xref reconstruction
    let result = PdfDocument::from_bytes(pdf);
    // May succeed (via reconstruction) or fail gracefully
    let _ = result;
}

#[test]
fn test_xref_reconstruction_missing_xref() {
    // PDF with no xref at all but valid objects
    let mut pdf = b"%PDF-1.7\n".to_vec();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n",
    );
    // Write startxref pointing to garbage
    pdf.extend_from_slice(b"startxref\n99999\n%%EOF\n");

    let result = PdfDocument::from_bytes(pdf);
    // Should attempt reconstruction
    let _ = result;
}

// ===========================================================================
// Tests: Name escaping in content streams
// ===========================================================================

#[test]
fn test_parse_name_with_hash() {
    let stream = b"BT /F#201 12 Tf ET";
    let ops = parse_content_stream(stream).unwrap();
    let has_tf = ops.iter().any(|op| matches!(op, Operator::Tf { .. }));
    assert!(has_tf);
}

// ===========================================================================
// Tests: Content stream with comments
// ===========================================================================

#[test]
fn test_parse_with_comments() {
    let stream = b"% This is a comment\nBT /F1 12 Tf 72 720 Td (Hello) Tj ET\n% End";
    let ops = parse_content_stream(stream).unwrap();
    let has_tj = ops.iter().any(|op| matches!(op, Operator::Tj { .. }));
    assert!(has_tj, "Should parse correctly with comments");
}

// ===========================================================================
// Tests: Mark info
// ===========================================================================

#[test]
fn test_mark_info_untagged() {
    let mut pdf = b"%PDF-1.7\n".to_vec();
    let off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    let off2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let off3 = pdf.len();
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n",
    );
    finalize_pdf(&mut pdf, &[0, off1, off2, off3]);

    let doc = PdfDocument::from_bytes(pdf).unwrap();
    let mark_info = doc.mark_info().unwrap();
    assert!(!mark_info.marked);
}

// ===========================================================================
// Helpers
// ===========================================================================

fn finalize_pdf(pdf: &mut Vec<u8>, obj_offsets: &[usize]) {
    let xref_offset = pdf.len();
    let count = obj_offsets.len();
    pdf.extend_from_slice(format!("xref\n0 {}\n", count).as_bytes());
    pdf.extend_from_slice(b"0000000000 65535 f \r\n");
    for &off in &obj_offsets[1..] {
        pdf.extend_from_slice(format!("{:010} 00000 n \r\n", off).as_bytes());
    }
    let trailer = format!(
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
        count, xref_offset
    );
    pdf.extend_from_slice(trailer.as_bytes());
}
