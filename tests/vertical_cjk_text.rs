//! Integration test: a vertical-CJK (tategaki) page extracts column-major —
//! each column top-to-bottom, columns right-to-left — not horizontal row order.
//!
//! ISO 32000-1 §9.7.4.3 (vertical writing mode). The PDF is hand-built: a
//! Type1 font with a `/ToUnicode` CMap mapping the show codes to CJK ideographs,
//! with each glyph absolutely positioned into vertical columns.

use pdf_oxide::PdfDocument;

/// Three columns of three CJK glyphs each. Codes A..I → 一二三四五六七八九.
/// Right column (x=116) holds 一二三 top-to-bottom, middle (x=89) 四五六, left
/// (x=62) 七八九. Read in reading order that is 一二三四五六七八九.
fn vertical_cjk_pdf() -> Vec<u8> {
    let tounicode = b"\
/CIDInit /ProcSet findresource begin
12 dict begin begincmap
1 begincodespacerange <00> <FF> endcodespacerange
9 beginbfchar
<41> <4E00>
<42> <4E8C>
<43> <4E09>
<44> <56DB>
<45> <4E94>
<46> <516D>
<47> <4E03>
<48> <516B>
<49> <4E5D>
endbfchar
endcmap CMapName currentdict /CMap defineresource pop end end";

    // Each glyph absolutely positioned. Columns at x=116/89/62, rows y=719/695/671
    // (24pt pitch with 18pt glyphs → square-ish vertical stacking).
    let content = b"BT /F1 18 Tf\n\
        1 0 0 1 116 719 Tm (A) Tj\n\
        1 0 0 1 116 695 Tm (B) Tj\n\
        1 0 0 1 116 671 Tm (C) Tj\n\
        1 0 0 1 89 719 Tm (D) Tj\n\
        1 0 0 1 89 695 Tm (E) Tj\n\
        1 0 0 1 89 671 Tm (F) Tj\n\
        1 0 0 1 62 719 Tm (G) Tj\n\
        1 0 0 1 62 695 Tm (H) Tj\n\
        1 0 0 1 62 671 Tm (I) Tj\n\
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
fn vertical_cjk_reads_column_major_right_to_left() {
    let doc = PdfDocument::from_bytes(vertical_cjk_pdf()).unwrap();
    let text = doc.extract_text(0).unwrap();
    let compact: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    assert_eq!(
        compact, "\u{4E00}\u{4E8C}\u{4E09}\u{56DB}\u{4E94}\u{516D}\u{4E03}\u{516B}\u{4E5D}",
        "vertical CJK not read column-major right-to-left: {text:?}"
    );
}

/// Cross-format: vertical CJK must be column-major in Markdown and HTML too —
/// not rendered as a table (the columns must not be mistaken for table cells).
#[test]
fn vertical_cjk_markdown_and_html_match_text() {
    use pdf_oxide::converters::ConversionOptions;
    let expected = "\u{4E00}\u{4E8C}\u{4E09}\u{56DB}\u{4E94}\u{516D}\u{4E03}\u{516B}\u{4E5D}";
    let doc = PdfDocument::from_bytes(vertical_cjk_pdf()).unwrap();

    let md = doc.to_markdown(0, &ConversionOptions::default()).unwrap();
    assert!(!md.contains('|'), "vertical CJK rendered as a Markdown table: {md:?}");
    let md_compact: String = md.chars().filter(|c| !c.is_whitespace()).collect();
    assert_eq!(md_compact, expected, "vertical CJK markdown not column-major: {md:?}");

    let html = doc.to_html(0, &ConversionOptions::default()).unwrap();
    assert!(!html.contains("<table"), "vertical CJK rendered as an HTML table: {html:?}");
    let html_text: String = html
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>()
        .replace("<p>", "")
        .replace("</p>", "");
    assert_eq!(html_text, expected, "vertical CJK html not column-major: {html:?}");
}
