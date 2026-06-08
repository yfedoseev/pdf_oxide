//! Integration test for the CJK↔number spacing fix: a producer that emits a
//! stray space glyph between a CJK ideograph and an embedded ASCII number must
//! not surface that space in extracted text ("公元前 1000 年" → "公元前1000年").
//!
//! The PDF is hand-built (no external fixture). A Type1 font carries a
//! `/ToUnicode` CMap mapping the show-string's byte codes to CJK ideographs and
//! ASCII digits, so the decoded text is genuine CJK with explicit spaces around
//! the number.

use pdf_oxide::PdfDocument;

fn obj(buf: &mut Vec<u8>, offsets: &mut [usize], id: usize, body: &str) {
    offsets[id] = buf.len();
    buf.extend_from_slice(format!("{id} 0 obj\n").as_bytes());
    buf.extend_from_slice(body.as_bytes());
    buf.extend_from_slice(b"\nendobj\n");
}

fn stream_obj(buf: &mut Vec<u8>, offsets: &mut [usize], id: usize, dict: &str, data: &[u8]) {
    offsets[id] = buf.len();
    buf.extend_from_slice(format!("{id} 0 obj\n").as_bytes());
    buf.extend_from_slice(format!("<< {dict} /Length {} >>\nstream\n", data.len()).as_bytes());
    buf.extend_from_slice(data);
    buf.extend_from_slice(b"\nendstream\nendobj\n");
}

/// One page showing "ABC 1000 D" where the font's /ToUnicode maps
/// A→公 B→元 C→前 D→年, so the decoded text is "公元前 1000 年".
fn cjk_number_pdf() -> Vec<u8> {
    let tounicode = b"\
/CIDInit /ProcSet findresource begin
12 dict begin begincmap
1 begincodespacerange <00> <FF> endcodespacerange
8 beginbfchar
<41> <516C>
<42> <5143>
<43> <524D>
<44> <5E74>
<20> <0020>
<30> <0030>
<31> <0031>
<2C> <002C>
endbfchar
endcmap CMapName currentdict /CMap defineresource pop end end";

    let content = b"BT /F1 12 Tf 72 700 Td (ABC 1000 D) Tj ET\n";

    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 8]; // ids 1..=7
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
    stream_obj(&mut buf, &mut off, 4, "", content);
    obj(
        &mut buf,
        &mut off,
        5,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica \
         /Encoding /WinAnsiEncoding /ToUnicode 6 0 R >>",
    );
    stream_obj(&mut buf, &mut off, 6, "", tounicode);

    let xref_off = buf.len();
    buf.extend_from_slice(b"xref\n0 7\n0000000000 65535 f \n");
    for id in 1..=6 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref_off}\n%%EOF\n").as_bytes());
    buf
}

#[test]
fn cjk_embedded_number_has_no_boundary_spaces() {
    let doc = PdfDocument::from_bytes(cjk_number_pdf()).unwrap();
    let text = doc.extract_text(0).unwrap();

    // Sanity: the CJK + number content decoded via /ToUnicode.
    assert!(text.contains("公元前"), "CJK not decoded: {text:?}");
    assert!(text.contains("1000"), "number not decoded: {text:?}");

    // The fix: no space between the ideographs and the embedded number.
    assert!(text.contains("公元前1000年"), "CJK↔number spaces not stripped: {text:?}");
    assert!(!text.contains("公元前 1000"), "stray space before number: {text:?}");
    assert!(!text.contains("1000 年"), "stray space after number: {text:?}");
}
