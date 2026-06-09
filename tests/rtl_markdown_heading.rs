//! Integration test: a tagged right-to-left heading whose words are emitted as
//! separate fragments must reconstruct into ONE Markdown heading in logical
//! order — not one `#` line per fragment.
//!
//! Reproduces the arabic-structured failure: the converter pipeline re-sorts by
//! `reading_order` and treats an RTL line's decreasing-X as a column wrap, so
//! the heading shattered into a heading-per-word. The PDF is hand-built tagged
//! (no third-party fixture).

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::PdfDocument;

/// One page, one `/H1` element holding three Hebrew word fragments (MCID 0..2)
/// drawn left-to-right in VISUAL order on one line (so the rightmost word is
/// logically first). `/ToUnicode` maps the codes to Hebrew letters.
fn rtl_heading_pdf() -> Vec<u8> {
    let tounicode = b"\
/CIDInit /ProcSet findresource begin
12 dict begin begincmap
1 begincodespacerange <00> <FF> endcodespacerange
6 beginbfchar
<41> <05D0>
<42> <05D1>
<43> <05D2>
<44> <05D3>
<45> <05D4>
<46> <05D5>
endbfchar
endcmap CMapName currentdict /CMap defineresource pop end end";

    // Visual order, left-to-right: word three (x=72), word two (x=130),
    // word one (x=190). Each is a separate MCID-marked Tj at fs=20.
    let content = b"BT /F1 20 Tf\n\
        /H1 <</MCID 0>> BDC 1 0 0 1 72 700 Tm (EF) Tj EMC\n\
        /H1 <</MCID 1>> BDC 1 0 0 1 130 700 Tm (CD) Tj EMC\n\
        /H1 <</MCID 2>> BDC 1 0 0 1 190 700 Tm (AB) Tj EMC\n\
        ET\n";

    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 10];
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
    stream(&mut buf, &mut off, 4, "", content);
    obj(
        &mut buf,
        &mut off,
        5,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica \
         /Encoding /WinAnsiEncoding /ToUnicode 6 0 R >>",
    );
    stream(&mut buf, &mut off, 6, "", tounicode);
    obj(&mut buf, &mut off, 7, "<< /Type /StructTreeRoot /K [8 0 R] >>");
    obj(
        &mut buf,
        &mut off,
        8,
        "<< /Type /StructElem /S /H1 /P 7 0 R /Pg 3 0 R /K [0 1 2] >>",
    );

    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 9\n0000000000 65535 f \n");
    for id in 1..=8 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 9 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

#[test]
fn rtl_heading_reconstructs_as_single_markdown_heading() {
    let doc = PdfDocument::from_bytes(rtl_heading_pdf()).unwrap();
    let md = doc.to_markdown(0, &ConversionOptions::default()).unwrap();

    // Exactly one heading line, not one per word fragment (the core fix).
    let heading_lines = md
        .lines()
        .filter(|l| l.trim_start().starts_with('#'))
        .count();
    assert_eq!(heading_lines, 1, "RTL heading fragmented into multiple headings:\n{md}");
    // All three words land on that single heading line, rightmost-first
    // (logical RTL reading order): word one (x=190) precedes word three (x=72).
    let heading = md
        .lines()
        .find(|l| l.trim_start().starts_with('#'))
        .unwrap();
    for cp in 0x05D0u32..=0x05D5 {
        let c = char::from_u32(cp).unwrap();
        assert!(heading.contains(c), "letter U+{cp:04X} missing from heading: {heading:?}");
    }
    let p_word_one = heading.find('\u{05D0}').unwrap(); // from x=190 (rightmost)
    let p_word_three = heading.find('\u{05D4}').unwrap(); // from x=72 (leftmost)
    assert!(p_word_one < p_word_three, "RTL word order not rightmost-first: {heading:?}");
}
