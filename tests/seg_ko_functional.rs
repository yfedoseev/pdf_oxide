//! SEG-KO functional regression coverage (v0.3.65 followup).
//!
//! End-to-end fixtures built as hand-written minimal Type0/Identity-H PDFs (no
//! third-party files). Text is carried by an explicit `/ToUnicode` CMap and glyph
//! advances by `/W`; each run is its own `BT…ET` text object so the extractor
//! keeps them as separate spans (one `BT…ET` = one span), letting
//! `should_insert_space` / the line-wrap logic run between them. This reproduces
//! the two over-segmentation defects and guards the fixes against regression.

use pdf_oxide::PdfDocument;

/// One placed span: text origin, the displayed string, and its 2-byte CIDs.
struct Run {
    x: f32,
    y: f32,
    text: &'static str,
    codes: &'static [u16],
}

/// Build a minimal Type0/Identity-H PDF. Every glyph advances 12 pt (W=1000 at
/// 12 Tf). Each run is emitted as its own `BT…ET`, so the inter-span gap is
/// exactly `next.x - (prev.x + 12*prev.codes.len())`. ToUnicode maps each code to
/// the matching scalar (runs supply parallel text↔codes).
fn type0_pdf(runs: &[Run]) -> Vec<u8> {
    let mut content = String::new();
    for r in runs {
        let hex: String = r.codes.iter().map(|c| format!("{c:04X}")).collect();
        content.push_str(&format!("BT /F1 12 Tf 1 0 0 1 {:.1} {:.1} Tm <{hex}> Tj ET\n", r.x, r.y));
    }

    // ToUnicode: one bfchar per (code, scalar). Runs give text + codes in order.
    let mut pairs: Vec<(u16, char)> = Vec::new();
    for r in runs {
        for (code, ch) in r.codes.iter().zip(r.text.chars()) {
            pairs.push((*code, ch));
        }
    }
    let mut bf = String::new();
    for (code, ch) in &pairs {
        bf.push_str(&format!("<{code:04X}> <{:04X}>\n", *ch as u32));
    }
    let tounicode = format!(
        "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n\
         /CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n\
         1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n\
         {} beginbfchar\n{}endbfchar\nendcmap\nCMapName currentdict /CMap defineresource pop\nend\nend",
        pairs.len(),
        bf
    );

    let mut w = String::new();
    for (code, _) in &pairs {
        w.push_str(&format!("{code} [1000] "));
    }

    let mut buf: Vec<u8> = Vec::new();
    let mut off: Vec<usize> = vec![0; 9];
    buf.extend_from_slice(b"%PDF-1.7\n");
    let obj = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, body: String| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>".into());
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".into());
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
         /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
            .into(),
    );
    obj(
        &mut buf,
        &mut off,
        4,
        format!("<< /Length {} >>\nstream\n{content}endstream", content.len()),
    );
    obj(
        &mut buf,
        &mut off,
        5,
        "<< /Type /Font /Subtype /Type0 /BaseFont /KOFix /Encoding /Identity-H \
         /DescendantFonts [6 0 R] /ToUnicode 7 0 R >>"
            .into(),
    );
    obj(
        &mut buf,
        &mut off,
        6,
        format!(
            "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /KOFix \
             /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> \
             /FontDescriptor 8 0 R /DW 1000 /W [ {w}] /CIDToGIDMap /Identity >>"
        ),
    );
    obj(
        &mut buf,
        &mut off,
        7,
        format!("<< /Length {} >>\nstream\n{tounicode}\nendstream", tounicode.len() + 1),
    );
    obj(
        &mut buf,
        &mut off,
        8,
        "<< /Type /FontDescriptor /FontName /KOFix /Flags 4 \
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

fn extract(runs: &[Run]) -> String {
    let doc = PdfDocument::from_bytes(type0_pdf(runs)).expect("fixture pdf parses");
    doc.extract_text(0).expect("extract_text")
}

/// A Sino-Korean numeral hugs its counter: a separate "1" span tightly followed
/// by a "만년" span must read "1만년", NOT "1 만년".
#[test]
fn korean_numeral_hugs_counter_no_spurious_space() {
    let t = extract(&[
        Run {
            x: 100.0,
            y: 700.0,
            text: "1",
            codes: &[0x0001],
        },
        Run {
            x: 112.0,
            y: 700.0,
            text: "만년",
            codes: &[0x0002, 0x0003],
        },
    ]);
    assert!(t.contains("1만년"), "Hangul↔digit forced space not suppressed — got: {t:?}");
    assert!(!t.contains("1 만년"), "spurious space between numeral and counter — got: {t:?}");
}

// NB: the ideograph↔digit split (issue 484) is guarded by the lib unit test
// `document::tests::test_should_insert_space_ideograph_digit_still_splits` — it
// can't be reproduced here because tightly-adjacent same-line glyphs merge into
// one span before `should_insert_space` runs.

/// A Hangul eojeol that wraps mid-syllable across a line break rejoins with no
/// separator: "집고양" (line end) + "이의" (next line start) → "집고양이의".
#[test]
fn korean_mid_eojeol_line_wrap_rejoins() {
    let t = extract(&[
        Run {
            x: 400.0,
            y: 700.0,
            text: "집고양",
            codes: &[0x0001, 0x0002, 0x0003],
        },
        Run {
            x: 60.0,
            y: 680.0,
            text: "이의",
            codes: &[0x0004, 0x0005],
        },
    ]);
    assert!(t.contains("집고양이의"), "mid-eojeol wrap not rejoined — got: {t:?}");
    assert!(!t.contains("집고양 이의"), "eojeol split by a space — got: {t:?}");
}
