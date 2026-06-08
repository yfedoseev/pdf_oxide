//! Integration test for complex-script (Brahmic) word spacing in the tagged
//! extraction path.
//!
//! Brahmic scripts (Devanagari, Bengali, Tamil, …) render dependent vowel
//! signs, conjuncts, and reordered glyphs with their own positional advances,
//! so consecutive glyphs of one word can sit a large fraction of an em apart.
//! The Latin-tuned geometric space heuristic would split such a word into
//! pieces. Word breaks in conforming text are carried by an explicit SPACE
//! glyph (ISO 32000-1 §14.8.2.5), so the heuristic must be suppressed between
//! two glyphs of the same complex script while a real explicit space survives.
//!
//! The PDF is a small hand-built tagged document (no third-party fixture).

use pdf_oxide::PdfDocument;

/// One-page tagged PDF; `content_ops` is the BT…ET body wrapped in a single
/// MCID-0 marked-content sequence, `bf` is the `/ToUnicode` bfchar table.
fn tagged_pdf(bf: &str, ops: &str) -> Vec<u8> {
    let tu = format!(
        "/CIDInit /ProcSet findresource begin\n12 dict begin begincmap\n\
         1 begincodespacerange <00> <FF> endcodespacerange\n\
         {} beginbfchar\n{}endbfchar\nendcmap CMapName currentdict /CMap defineresource pop end end",
        bf.lines().filter(|l| !l.trim().is_empty()).count(),
        bf
    );
    let content = format!("/P <</MCID 0>> BDC\n{ops}\nEMC\n").into_bytes();
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
    stream(&mut buf, &mut off, 4, "", &content);
    obj(
        &mut buf,
        &mut off,
        5,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica \
         /Encoding /WinAnsiEncoding /ToUnicode 6 0 R >>",
    );
    stream(&mut buf, &mut off, 6, "", tu.as_bytes());
    obj(&mut buf, &mut off, 7, "<< /Type /StructTreeRoot /K [8 0 R] >>");
    obj(&mut buf, &mut off, 8, "<< /Type /StructElem /S /P /P 7 0 R /Pg 3 0 R /K [0] >>");
    let xref = buf.len();
    buf.extend_from_slice(b"xref\n0 9\n0000000000 65535 f \n");
    for id in 1..=8 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 9 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

/// A single Devanagari word whose three glyphs are drawn with wide inter-glyph
/// gaps (as a matra-bearing cluster would be) and NO explicit space between
/// them must come out as one unbroken word; a second word after an explicit
/// SPACE glyph keeps its boundary.
#[test]
fn complex_script_word_not_split_explicit_space_kept() {
    // A,B,C → क म ल (Devanagari); D → प; space → space.
    let bf = "\
<41> <0915>
<42> <092E>
<43> <0932>
<44> <092A>
<20> <0020>
";
    // Word one "कमल": three fragments ~9pt apart (a wide intra-word gap), no
    // explicit space. Then an explicit space, then word two "प".
    let ops = "BT /F1 12 Tf\n\
        72 700 Td (A) Tj\n\
        9 0 Td (B) Tj\n\
        9 0 Td (C) Tj\n\
        ( ) Tj\n\
        (D) Tj\nET";
    let doc = PdfDocument::from_bytes(tagged_pdf(bf, ops)).unwrap();
    let text = doc.extract_text(0).unwrap();

    // Word one is contiguous — no spurious space split the cluster.
    assert!(
        text.contains("\u{0915}\u{092E}\u{0932}"),
        "Devanagari cluster split by spurious gap-space: {text:?}"
    );
    // The explicit word break before word two survives.
    assert!(
        text.contains("\u{0932}\u{0020}\u{092A}") || text.contains("\u{0932} \u{092A}"),
        "explicit word-break space was lost: {text:?}"
    );
}
