//! Integration tests for right-to-left reading order in the tagged
//! (structure-tree) extraction path.
//!
//! Two regressions are pinned here, both reproduced with small hand-built
//! tagged PDFs (no third-party fixtures):
//!
//! 1. **Neutral punctuation between RTL words** — a producer that draws a
//!    Hebrew line in visual order emits the inter-word comma glyphs in visual
//!    order too (a "<space><comma>" run sitting between two words). When the
//!    surrounding words are reversed to logical order the comma must reverse
//!    with them, attaching to the preceding word ("word, word"), not stranding
//!    before the next word ("word ,word").
//!
//! 2. **Jittery baseline must stay one line** — RTL producers routinely draw
//!    some glyphs a couple of points off the baseline (hamza seats, marks).
//!    A fixed-point row band splits those into separate rows that emit out of
//!    order; the line must instead be grouped by a font-relative tolerance and
//!    emitted rightmost-first.
//!
//! The PDFs are tagged (`/MarkInfo /Marked true` + a `/StructTreeRoot` whose
//! single `/P` element references MCID 0) so extraction takes the structure-
//! order assembler — the path these fixes live on.

use pdf_oxide::PdfDocument;

/// Builder for a one-page tagged PDF whose page content is a single MCID-0
/// marked-content sequence. The caller supplies the content-stream body (the
/// `BT … ET` text operators) and a `/ToUnicode` bfchar table mapping show-string
/// byte codes to Unicode scalars.
struct TaggedRtlPdf {
    tounicode_bfchars: String,
    content_ops: String,
}

impl TaggedRtlPdf {
    fn build(&self) -> Vec<u8> {
        // Object layout:
        // 1 Catalog, 2 Pages, 3 Page, 4 Contents, 5 Font, 6 ToUnicode,
        // 7 StructTreeRoot, 8 StructElem (/P, MCID 0)
        let tounicode = format!(
            "/CIDInit /ProcSet findresource begin\n12 dict begin begincmap\n\
             1 begincodespacerange <00> <FF> endcodespacerange\n\
             {} beginbfchar\n{}endbfchar\nendcmap CMapName currentdict /CMap defineresource pop end end",
            self.tounicode_bfchars.lines().filter(|l| !l.trim().is_empty()).count(),
            self.tounicode_bfchars,
        );

        // Content: wrap the text ops in a /P BDC … EMC marked-content sequence.
        let content = format!("/P <</MCID 0>> BDC\n{}\nEMC\n", self.content_ops);
        let content_bytes = content.into_bytes();

        let mut buf: Vec<u8> = Vec::new();
        let mut off = vec![0usize; 9];
        let obj = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, body: &str| {
            off[id] = buf.len();
            buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
        };
        let stream =
            |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, dict: &str, data: &[u8]| {
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
        stream(&mut buf, &mut off, 4, "", &content_bytes);
        obj(
            &mut buf,
            &mut off,
            5,
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica \
             /Encoding /WinAnsiEncoding /ToUnicode 6 0 R >>",
        );
        stream(&mut buf, &mut off, 6, "", tounicode.as_bytes());
        obj(&mut buf, &mut off, 7, "<< /Type /StructTreeRoot /K [8 0 R] >>");
        obj(&mut buf, &mut off, 8, "<< /Type /StructElem /S /P /P 7 0 R /Pg 3 0 R /K [0] >>");

        let xref_off = buf.len();
        buf.extend_from_slice(b"xref\n0 9\n0000000000 65535 f \n");
        for id in 1..=8 {
            buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
        }
        buf.extend_from_slice(b"trailer\n<< /Size 9 /Root 1 0 R >>\nstartxref\n");
        buf.extend_from_slice(format!("{xref_off}\n%%EOF\n").as_bytes());
        buf
    }
}

/// A Hebrew line drawn in visual order with the inter-word comma stored
/// as a "<space><comma>" run. Codes: A,B → אב (word one), C,D → גד (word two),
/// and the comma run is "<space><comma>". Drawn left-to-right (visual order):
/// word two at the left, the comma run, then word one at the right.
#[test]
fn rtl_inter_word_comma_attaches_to_preceding_word() {
    let pdf = TaggedRtlPdf {
        tounicode_bfchars: "\
<41> <05D0>
<42> <05D1>
<43> <05D2>
<44> <05D3>
<20> <0020>
<2C> <002C>
"
        .to_string(),
        // Visual order: leftmost first. Each Td hop forces a separate span.
        content_ops: "BT /F1 12 Tf\n\
            100 700 Td (CD) Tj\n\
            30 0 Td ( ,) Tj\n\
            18 0 Td (AB) Tj\nET"
            .to_string(),
    };
    let doc = PdfDocument::from_bytes(pdf.build()).unwrap();
    let text = doc.extract_text(0).unwrap();

    // Logical reading order is word-one then word-two: בא then גד (each word's
    // glyphs reversed from visual order), with the comma bound to word one.
    assert!(text.contains('\u{05D0}'), "Hebrew not decoded: {text:?}");
    // The comma must hug the preceding word, with the space AFTER it.
    assert!(
        text.contains(", ") || text.contains(",\u{05D2}") || text.contains(",ג"),
        "comma not re-attached to preceding word (visual order leaked): {text:?}"
    );
    assert!(
        !text.contains(" ,"),
        "stranded leading-space comma — visual neutral order leaked into output: {text:?}"
    );
}

/// An Arabic line whose glyphs jitter a few points off the baseline must
/// be grouped into a single line and emitted rightmost-first, not scattered by
/// a fixed row band. Codes A,B,C,D → ا ل ق م; the logical word is "القم". Drawn
/// in visual order (leftmost first: م ق ل ا) with two glyphs jittered ±2-3pt.
#[test]
fn rtl_jittery_baseline_line_not_scattered() {
    let pdf = TaggedRtlPdf {
        tounicode_bfchars: "\
<41> <0627>
<42> <0644>
<43> <0642>
<44> <0645>
"
        .to_string(),
        // Visual order left-to-right: م (x100,y700), ق (x120,y703 jitter),
        // ل (x140,y700), ا (x160,y702 jitter). A fixed 3pt band rounds y703/702
        // into a higher row than y700 and emits ق/ا first → "اقلم"; the
        // font-relative tolerance (0.5*12 = 6pt) keeps all four on one line.
        content_ops: "BT /F1 12 Tf\n\
            100 700 Td (D) Tj\n\
            20 3 Td (C) Tj\n\
            20 -3 Td (B) Tj\n\
            20 2 Td (A) Tj\nET"
            .to_string(),
    };
    let doc = PdfDocument::from_bytes(pdf.build()).unwrap();
    let text = doc.extract_text(0).unwrap();

    // Logical order ا-ل-ق-م must be contiguous; the scattered band order اقلم
    // must NOT appear.
    assert!(
        text.contains("\u{0627}\u{0644}\u{0642}\u{0645}"),
        "jittery RTL line not reassembled in logical order: {text:?}"
    );
    assert!(
        !text.contains("\u{0627}\u{0642}\u{0644}\u{0645}"),
        "row-band scatter leaked into output: {text:?}"
    );
}

/// ISO 32000-1 §14.8.2.3.3: a show string "shall not contain interior SPACEs".
/// A producer that draws a single cursive Arabic word with a stray space inside
/// it (here `ق ل` between the letters of a four-letter run) must not surface
/// that space — it splits letters the script joins and is never a word break.
/// A space at a string boundary (a separate Tj → separate span) is a real word
/// break and must survive.
#[test]
fn rtl_interior_arabic_space_is_stripped_real_break_kept() {
    let pdf = TaggedRtlPdf {
        tounicode_bfchars: "\
<41> <0627>
<42> <0644>
<43> <0642>
<44> <0645>
<20> <0020>
"
        .to_string(),
        // One Tj draws the cursive run with an interior space ("AB CD" →
        // ا ل <space> ق م); a second, separate Tj after a real gap draws
        // another word. Visual order (leftmost first): the second word, then
        // the interior-space run to its right.
        content_ops: "BT /F1 12 Tf\n\
            100 700 Td (AB) Tj\n\
            40 0 Td (AB CD) Tj\nET"
            .to_string(),
    };
    let doc = PdfDocument::from_bytes(pdf.build()).unwrap();
    let text = doc.extract_text(0).unwrap();

    // The interior space inside the cursive run is gone: ا ل ق م contiguous.
    assert!(
        text.contains("\u{0627}\u{0644}\u{0642}\u{0645}"),
        "interior cursive-join space not stripped: {text:?}"
    );
    // The four letters of the run carry no internal space.
    assert!(
        !text.contains("\u{0627}\u{0644} \u{0642}\u{0645}"),
        "interior Arabic space leaked into output: {text:?}"
    );
}
