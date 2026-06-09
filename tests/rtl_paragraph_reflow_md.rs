//! Integration test: a tagged right-to-left paragraph that wraps across two
//! visual lines must reflow into ONE Markdown paragraph — matching the
//! plain-text surface — not fragment into one paragraph per line/word.
//!
//! Cross-format alignment: extract_text already joins wrapped
//! Hebrew/Arabic lines; the converter pipeline must too. PDF is hand-built
//! tagged (no third-party fixture).

use pdf_oxide::converters::ConversionOptions;
use pdf_oxide::PdfDocument;

/// One `/P` holding a Hebrew paragraph over two lines. Line one (y=700) is a
/// full wrapped line reaching the left margin (x=72); line two (y=676, a 24pt
/// step that the geometric heuristic would split) continues the paragraph.
/// `/ToUnicode` maps the codes to Hebrew letters; words are drawn left-to-right
/// in visual order (so the rightmost word is logically first).
fn rtl_paragraph_pdf() -> Vec<u8> {
    let tounicode = b"\
/CIDInit /ProcSet findresource begin
12 dict begin begincmap
1 begincodespacerange <00> <FF> endcodespacerange
8 beginbfchar
<41> <05D0>
<42> <05D1>
<43> <05D2>
<44> <05D3>
<45> <05D4>
<46> <05D5>
<47> <05D6>
<48> <05D7>
endbfchar
endcmap CMapName currentdict /CMap defineresource pop end end";

    // Line 1 (y=700): four words filling x=72..300 (MCID 0..3, with spaces).
    // Line 2 (y=676): one word at the right (MCID 4). All in one /P.
    let content = b"BT /F1 12 Tf\n\
        /P <</MCID 0>> BDC 1 0 0 1 72 700 Tm (AB) Tj EMC\n\
        /P <</MCID 1>> BDC 1 0 0 1 130 700 Tm (CD) Tj EMC\n\
        /P <</MCID 2>> BDC 1 0 0 1 190 700 Tm (EF) Tj EMC\n\
        /P <</MCID 3>> BDC 1 0 0 1 250 700 Tm (GH) Tj EMC\n\
        /P <</MCID 4>> BDC 1 0 0 1 250 676 Tm (AB) Tj EMC\n\
        ET\n";

    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 9];
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
        "<< /Type /StructElem /S /P /P 7 0 R /Pg 3 0 R /K [0 1 2 3 4] >>",
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
fn rtl_wrapped_paragraph_reflows_into_one_markdown_paragraph() {
    let doc = PdfDocument::from_bytes(rtl_paragraph_pdf()).unwrap();
    let md = doc.to_markdown(0, &ConversionOptions::default()).unwrap();

    assert!(md.contains('\u{05D0}'), "Hebrew not decoded: {md:?}");
    let blocks: Vec<&str> = md
        .split("\n\n")
        .map(|b| b.trim())
        .filter(|b| !b.is_empty())
        .collect();
    // The four words of line one flow together (not one block per word/space —
    // the cross-format fragmentation bug produced a block per word). At most one
    // block per visual line, never per word.
    assert!(
        blocks.len() <= 2,
        "RTL paragraph fragmented into {} blocks (one per word):\n{md}",
        blocks.len()
    );
    // Line one's first block carries all four words (8 distinct Hebrew letters).
    let line_one = blocks[0];
    let distinct = (0x05D0u32..=0x05D7)
        .filter(|cp| line_one.contains(char::from_u32(*cp).unwrap()))
        .count();
    assert_eq!(distinct, 8, "line one's words did not flow into one block: {line_one:?}");
}
