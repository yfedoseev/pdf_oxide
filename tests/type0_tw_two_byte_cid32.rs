//! Word spacing (Tw) applies only to the single-byte character code 32, never
//! to a multi-byte code whose value happens to be 32 (ISO 32000-1:2008 §9.3.3:
//! "word spacing shall be applied to every occurrence of the single-byte
//! character code 32 in a string when using a simple font or a composite font
//! that defines code 32 as a single-byte code").
//!
//! A Type0 / Identity-H font here shows the 2-byte code 0x0020 — a real glyph
//! CID, not a space. With `50 Tw` in force the span's right edge must be
//! identical whether the string is shown with `Tj` or `TJ`, and identical to
//! the `Tw 0` control: x0 = 50, two 1000-unit glyphs at 24pt → x1 = 98.

use pdf_oxide::document::PdfDocument;

fn build(content_stream: &[u8]) -> Vec<u8> {
    let tounicode: &[u8] = b"/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def
/CMapName /Adobe-Identity-UCS def
/CMapType 2 def
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
2 beginbfchar
<0020> <0041>
<0021> <0042>
endbfchar
endcmap
CMapName currentdict /CMap defineresource pop
end
end
";
    let mut bodies: Vec<Vec<u8>> = vec![Vec::new(); 9]; // ids 1..=8
    bodies[1] = b"<< /Type /Catalog /Pages 2 0 R >>".to_vec();
    bodies[2] = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec();
    bodies[3] = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        .to_vec();
    let mut content_obj = format!("<< /Length {} >>\nstream\n", content_stream.len()).into_bytes();
    content_obj.extend_from_slice(content_stream);
    content_obj.extend_from_slice(b"\nendstream");
    bodies[4] = content_obj;
    bodies[5] = b"<< /Type /Font /Subtype /Type0 /BaseFont /ABCDEF+MyFont \
/Encoding /Identity-H /DescendantFonts [6 0 R] /ToUnicode 8 0 R >>"
        .to_vec();
    bodies[6] = b"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /ABCDEF+MyFont \
/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> \
/FontDescriptor 7 0 R /DW 1000 /W [32 [1000] 33 [1000]] /CIDToGIDMap /Identity >>"
        .to_vec();
    bodies[7] = b"<< /Type /FontDescriptor /FontName /ABCDEF+MyFont /Flags 4 \
/FontBBox [0 0 1000 1000] /ItalicAngle 0 /Ascent 800 /Descent -200 \
/CapHeight 800 /StemV 80 >>"
        .to_vec();
    let mut tu_obj = format!("<< /Length {} >>\nstream\n", tounicode.len()).into_bytes();
    tu_obj.extend_from_slice(tounicode);
    tu_obj.extend_from_slice(b"\nendstream");
    bodies[8] = tu_obj;

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    let mut offsets = [0usize; 9];
    for id in 1..=8 {
        offsets[id] = out.len();
        out.extend_from_slice(format!("{id} 0 obj\n").as_bytes());
        out.extend_from_slice(&bodies[id]);
        out.extend_from_slice(b"\nendobj\n");
    }
    let xref = out.len();
    out.extend_from_slice(b"xref\n0 9\n0000000000 65535 f \n");
    for id in 1..=8 {
        out.extend_from_slice(format!("{:010} 00000 n \n", offsets[id]).as_bytes());
    }
    out.extend_from_slice(
        format!("trailer\n<< /Size 9 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes(),
    );
    out
}

fn span_right_edge(content: &[u8]) -> f32 {
    let doc = PdfDocument::from_bytes(build(content)).expect("parse pdf");
    let spans = doc.extract_spans(0).expect("spans");
    let s = spans.iter().find(|s| s.text == "AB").unwrap_or_else(|| {
        panic!("no AB span in {:?}", spans.iter().map(|s| &s.text).collect::<Vec<_>>())
    });
    s.bbox.x + s.bbox.width
}

#[test]
fn word_spacing_not_applied_to_two_byte_code_32() {
    // Two 1000/1000-em glyphs at 24pt from x0 = 50 → right edge 50 + 48 = 98.
    let tj_op = span_right_edge(b"BT /F1 24 Tf 50 700 Td 50 Tw <00200021> Tj ET");
    let tj_array = span_right_edge(b"BT /F1 24 Tf 50 700 Td 50 Tw [<00200021>] TJ ET");
    let control = span_right_edge(b"BT /F1 24 Tf 50 700 Td 0 Tw <00200021> Tj ET");

    assert!((control - 98.0).abs() < 0.5, "Tw=0 control right edge: {control}");
    assert!((tj_array - 98.0).abs() < 0.5, "TJ path right edge: {tj_array}");
    assert!(
        (tj_op - 98.0).abs() < 0.5,
        "plain-Tj path applied Tw to a 2-byte CID 32: right edge {tj_op}, expected 98 \
         (TJ path: {tj_array}, control: {control})"
    );
}
