//! Regression coverage for #847 mechanism 2: condensed/tracked lines typeset
//! with NO space glyph (bold headings, running footers) whose inter-word gaps
//! are narrower than the fixed intra-word kerning guard. Per-line bimodal gap
//! clustering rescues the genuine word gap without over-splitting.
//!
//! Both fixtures are hand-built Type0/Identity-H PDFs (no third-party files):
//! each glyph is its own `Tj` positioned by `Tm`, so the extractor sees
//! per-glyph spans exactly like the real condensed-heading producers. `/W` gives
//! every glyph a 500/1000-em advance; intra-word glyphs abut (gap 0) while a
//! word boundary opens a ~0.18-em gap — below the kerning guard, but clearly
//! separated from the zero intra-word gaps, so the bimodal rescue fires.

use pdf_oxide::PdfDocument;

/// One positioned glyph: CID (Identity-H code == GID), its Unicode scalar, and
/// the absolute x of its origin. All glyphs share one baseline.
struct Glyph {
    code: u16,
    ch: char,
    x: f32,
}

/// Minimal single-line Type0/Identity-H PDF at `font_size`, each glyph drawn by
/// its own `Tm`+`Tj` so gaps are exactly the positions given. `/W` = 500 (glyph
/// advance 0.5 em); the ToUnicode CMap maps every CID to its scalar.
fn identity_line_pdf(glyphs: &[Glyph], font_size: f32) -> Vec<u8> {
    let mut content = String::new();
    for g in glyphs {
        content.push_str(&format!(
            "BT /F1 {font_size} Tf 1 0 0 1 {:.2} 700 Tm <{:04X}> Tj ET\n",
            g.x, g.code
        ));
    }
    let mut bf = String::new();
    for g in glyphs {
        bf.push_str(&format!("<{:04X}> <{:04X}>\n", g.code, g.ch as u32));
    }
    let tounicode = format!(
        "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n\
         /CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n\
         1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n\
         {} beginbfchar\n{}endbfchar\nendcmap\nCMapName currentdict /CMap defineresource pop\nend\nend",
        glyphs.len(),
        bf
    );
    let mut w = String::new();
    for g in glyphs {
        w.push_str(&format!("{} [500] ", g.code));
    }

    let mut buf: Vec<u8> = Vec::new();
    let mut off = [0usize; 9];
    buf.extend_from_slice(b"%PDF-1.7\n");
    let mut obj = |buf: &mut Vec<u8>, id: usize, body: String| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };
    obj(&mut buf, 1, "<< /Type /Catalog /Pages 2 0 R >>".into());
    obj(&mut buf, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".into());
    obj(
        &mut buf,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            .into(),
    );
    obj(
        &mut buf,
        4,
        format!("<< /Length {} >>\nstream\n{content}endstream", content.len()),
    );
    obj(
        &mut buf,
        5,
        "<< /Type /Font /Subtype /Type0 /BaseFont /IDFix /Encoding /Identity-H \
         /DescendantFonts [6 0 R] /ToUnicode 7 0 R >>"
            .into(),
    );
    obj(
        &mut buf,
        6,
        format!(
            "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /IDFix \
             /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> \
             /FontDescriptor 8 0 R /DW 500 /W [ {w}] /CIDToGIDMap /Identity >>"
        ),
    );
    obj(
        &mut buf,
        7,
        format!("<< /Length {} >>\nstream\n{tounicode}\nendstream", tounicode.len() + 1),
    );
    obj(
        &mut buf,
        8,
        "<< /Type /FontDescriptor /FontName /IDFix /Flags 4 \
         /FontBBox [0 -200 1000 800] /ItalicAngle 0 /Ascent 800 /Descent -200 \
         /CapHeight 700 /StemV 80 >>"
            .into(),
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

/// Build `"the cat"` at 20 pt: glyph advance = 10 pt; intra-word glyphs abut
/// (gap 0), the `e`→`c` word boundary opens a 3.6 pt (0.18 em) gap — below the
/// kerning guard's ~3.75 pt ceiling, so without the bimodal rescue the two
/// words fuse into `thecat`.
#[test]
fn condensed_heading_word_gap_is_recovered() {
    // codes 1..=6 = t h e c a t; advance 10 pt each.
    let g = |code, ch, x| Glyph { code, ch, x };
    let glyphs = [
        g(1, 't', 100.0),
        g(2, 'h', 110.0),
        g(3, 'e', 120.0),
        // word gap: 133.6 - 130 (e's end) = 3.6 pt ≈ 0.18 em
        g(4, 'c', 133.6),
        g(5, 'a', 143.6),
        g(6, 't', 153.6),
    ];
    let pdf = identity_line_pdf(&glyphs, 20.0);
    let text = PdfDocument::from_bytes(pdf)
        .unwrap()
        .extract_text(0)
        .unwrap();
    let joined: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(
        joined.contains("the cat"),
        "condensed heading word gap must be recovered, got: {text:?}"
    );
    assert!(
        !joined.contains("thecat"),
        "words must not fuse across the 0.18-em word gap, got: {text:?}"
    );
}

/// The rescue must NOT fabricate a boundary inside a single tightly-set word:
/// six abutting glyphs (all gaps 0) are one token and must stay `guards`.
#[test]
fn single_condensed_word_is_not_split() {
    let g = |code, ch, x| Glyph { code, ch, x };
    // g u a r d s — all abutting (advance 10 pt, no gaps).
    let glyphs = [
        g(1, 'g', 100.0),
        g(2, 'u', 110.0),
        g(3, 'a', 120.0),
        g(4, 'r', 130.0),
        g(5, 'd', 140.0),
        g(6, 's', 150.0),
    ];
    let pdf = identity_line_pdf(&glyphs, 20.0);
    let text = PdfDocument::from_bytes(pdf)
        .unwrap()
        .extract_text(0)
        .unwrap();
    assert!(
        text.contains("guards"),
        "a single tightly-set word must not be split, got: {text:?}"
    );
}
