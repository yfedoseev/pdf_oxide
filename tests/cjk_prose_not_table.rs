//! Integration test: horizontal CJK prose whose wrapped lines align into
//! columns must NOT be rendered as a Markdown table — CJK writes without
//! inter-word spaces, so the spatial table detector is prone to mistaking it
//! for a grid. The PDF is hand-built and untagged (no third-party fixture).

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::PdfDocument;

/// Untagged page: two rows, each with a short CJK cell at x=72 and a long CJK
/// run (12 ideographs) at x=120, so a 2x2 grid forms. `/ToUnicode` maps the
/// codes to CJK ideographs.
fn cjk_grid_pdf() -> Vec<u8> {
    let tounicode = b"\
/CIDInit /ProcSet findresource begin
12 dict begin begincmap
1 begincodespacerange <00> <FF> endcodespacerange
3 beginbfchar
<41> <4E00>
<42> <4E8C>
<58> <4E09>
endbfchar
endcmap CMapName currentdict /CMap defineresource pop end end";

    // X (U+4E09) repeated 12x → a long CJK run that marks the cell as prose.
    let content = b"BT /F1 12 Tf\n\
        1 0 0 1 72 700 Tm (AB) Tj 1 0 0 1 120 700 Tm (XXXXXXXXXXXX) Tj\n\
        1 0 0 1 72 680 Tm (AB) Tj 1 0 0 1 120 680 Tm (XXXXXXXXXXXX) Tj\n\
        ET\n";

    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 7];
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
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
    );
    stream(&mut buf, &mut off, 4, "", content);
    obj(
        &mut buf,
        &mut off,
        5,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica \
         /Encoding /WinAnsiEncoding /ToUnicode 6 0 R >>",
    );
    stream(&mut buf, &mut off, 6, "", tounicode);

    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 7\n0000000000 65535 f \n");
    for id in 1..=6 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

#[test]
fn cjk_prose_is_not_rendered_as_table() {
    let doc = PdfDocument::from_bytes(cjk_grid_pdf()).unwrap();
    let md = doc.to_markdown(0, &ConversionOptions::default()).unwrap();
    assert!(md.contains('\u{4E00}'), "CJK not decoded: {md:?}");
    assert!(
        !md.contains("|---|"),
        "CJK prose was wrongly rendered as a Markdown table: {md:?}"
    );
}
