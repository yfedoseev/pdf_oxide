//! A predefined Adobe-Japan1 CIDFont without /ToUnicode must extract unified
//! ideographs, not CJK Radicals Supplement presentation forms: CID 2664 is 青
//! (U+9752), CID 2666 is 斉 (U+6589). Radical-block codepoints (U+2E80–2FDF)
//! are dictionary glyphs, never running text (ISO 32000-1 §9.10.2 maps CIDs
//! through the character collection).
//!
//! The PDF is hand-built (no external fixture): Type0 / Identity-H with
//! /CIDSystemInfo Adobe-Japan1 and no /ToUnicode, so decoding must go through
//! the predefined CID→Unicode table.

use pdf_oxide::converters::ConversionOptions;
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

fn finish(mut buf: Vec<u8>, offsets: &[usize]) -> Vec<u8> {
    let n = offsets.len();
    let xref = buf.len();
    buf.extend_from_slice(format!("xref\n0 {n}\n0000000000 65535 f \n").as_bytes());
    for &off in &offsets[1..] {
        buf.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
    }
    buf.extend_from_slice(
        format!("trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes(),
    );
    buf
}

/// One page showing CIDs 2664 (青) and 2666 (斉) through a predefined
/// Adobe-Japan1 collection, Identity-H, no ToUnicode.
fn japan1_cidfont_pdf() -> Vec<u8> {
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
    // CIDs 2664 = 0x0A68, 2666 = 0x0A6A.
    stream_obj(&mut buf, &mut off, 4, "", b"BT /F1 24 Tf 72 700 Td <0A680A6A> Tj ET\n");
    obj(
        &mut buf,
        &mut off,
        5,
        "<< /Type /Font /Subtype /Type0 /BaseFont /Ryumin-Light \
         /Encoding /Identity-H /DescendantFonts [6 0 R] >>",
    );
    obj(
        &mut buf,
        &mut off,
        6,
        "<< /Type /Font /Subtype /CIDFontType0 /BaseFont /Ryumin-Light \
         /CIDSystemInfo << /Registry (Adobe) /Ordering (Japan1) /Supplement 7 >> \
         /FontDescriptor 7 0 R /DW 1000 >>",
    );
    obj(
        &mut buf,
        &mut off,
        7,
        "<< /Type /FontDescriptor /FontName /Ryumin-Light /Flags 6 \
         /FontBBox [0 0 1000 1000] /ItalicAngle 0 /Ascent 880 /Descent -120 \
         /CapHeight 880 /StemV 90 >>",
    );
    finish(buf, &off)
}

/// A font whose /ToUnicode CMap itself emits a CJK Radicals Supplement
/// codepoint (U+2ED8 ⻘). The radical-form normalization layer must carry it
/// to the unified ideograph 青 (U+9752), exactly as it already does for the
/// Kangxi block (U+2F00–2FDF).
fn tounicode_radical_pdf() -> Vec<u8> {
    let tounicode = b"\
/CIDInit /ProcSet findresource begin
12 dict begin begincmap
1 begincodespacerange <00> <FF> endcodespacerange
1 beginbfchar
<41> <2ED8>
endbfchar
endcmap CMapName currentdict /CMap defineresource pop end end";

    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 7]; // ids 1..=6
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
    stream_obj(&mut buf, &mut off, 4, "", b"BT /F1 12 Tf 72 700 Td (A) Tj ET\n");
    obj(
        &mut buf,
        &mut off,
        5,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /ToUnicode 6 0 R >>",
    );
    stream_obj(&mut buf, &mut off, 6, "", tounicode);
    finish(buf, &off)
}

#[test]
fn predefined_japan1_cids_extract_unified_ideographs() {
    let doc = PdfDocument::from_bytes(japan1_cidfont_pdf()).expect("parse");
    let text = doc
        .to_plain_text(0, &ConversionOptions::default())
        .expect("extract");

    let radicals: Vec<char> = text
        .chars()
        .filter(|c| (0x2E80..=0x2FDF).contains(&(*c as u32)))
        .collect();
    assert!(
        radicals.is_empty(),
        "radical presentation forms in extracted text: {radicals:?} (text = {text:?})"
    );
    assert!(text.contains('青'), "CID 2664 must extract as 青 (U+9752): {text:?}");
    assert!(text.contains('斉'), "CID 2666 must extract as 斉 (U+6589): {text:?}");
}

#[test]
fn radicals_supplement_block_normalizes_to_ideograph() {
    let doc = PdfDocument::from_bytes(tounicode_radical_pdf()).expect("parse");
    let text = doc
        .to_plain_text(0, &ConversionOptions::default())
        .expect("extract");

    assert!(
        !text.contains('\u{2ED8}'),
        "U+2ED8 ⻘ (CJK Radicals Supplement) must not survive normalization: {text:?}"
    );
    assert!(text.contains('青'), "⻘ must normalize to 青 (U+9752): {text:?}");
}
