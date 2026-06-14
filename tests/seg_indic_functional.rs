//! SEG-INDIC functional regression coverage (v0.3.65 followup).
//!
//! Hand-written minimal Type0/Identity-H PDFs (no third-party files): a Brahmic
//! word followed by a separate clause-punctuation span. `extract_text` must hug
//! the punctuation to the word (no floating " ।" / " ,"), guarding the
//! complex-script clause-punctuation rule in `should_insert_space`. The lib unit
//! test `test_should_insert_space_indic_clause_punct_hugs` covers the per-pair
//! logic; this exercises it end-to-end through extraction.

use pdf_oxide::PdfDocument;

struct Run {
    x: f32,
    y: f32,
    text: &'static str,
    codes: &'static [u16],
}

fn type0_pdf(runs: &[Run]) -> Vec<u8> {
    let mut content = String::new();
    for r in runs {
        let hex: String = r.codes.iter().map(|c| format!("{c:04X}")).collect();
        content.push_str(&format!("BT /F1 12 Tf 1 0 0 1 {:.1} {:.1} Tm <{hex}> Tj ET\n", r.x, r.y));
    }
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
        "<< /Type /Font /Subtype /Type0 /BaseFont /INFix /Encoding /Identity-H \
         /DescendantFonts [6 0 R] /ToUnicode 7 0 R >>"
            .into(),
    );
    obj(
        &mut buf,
        &mut off,
        6,
        format!(
            "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /INFix \
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
        "<< /Type /FontDescriptor /FontName /INFix /Flags 4 \
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

/// Bengali sentence-final danda hugs the preceding word: "প্রাণী" + "।" with a
/// gap must read "প্রাণী।", not "প্রাণী ।".
#[test]
fn bengali_danda_hugs_word() {
    // প্রাণী = প ্ র া ণ ী  (6 codes), then danda ।
    let t = extract(&[
        Run {
            x: 100.0,
            y: 700.0,
            text: "প্রাণী",
            codes: &[0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006],
        },
        Run {
            x: 175.0,
            y: 700.0,
            text: "।",
            codes: &[0x0007],
        },
    ]);
    assert!(t.contains("প্রাণী।"), "danda not hugged to word — got: {t:?}");
    assert!(!t.contains("প্রাণী ।"), "spurious space before danda — got: {t:?}");
}

/// Hindi: ASCII comma after a Devanagari word hugs it: "रोशनी" + "," → "रोशनी,".
#[test]
fn hindi_comma_hugs_devanagari_word() {
    // रोशनी = र ो श न ी (5 codes), then ASCII comma
    let t = extract(&[
        Run {
            x: 100.0,
            y: 700.0,
            text: "रोशनी",
            codes: &[0x0001, 0x0002, 0x0003, 0x0004, 0x0005],
        },
        Run {
            x: 170.0,
            y: 700.0,
            text: ",",
            codes: &[0x0006],
        },
    ]);
    assert!(t.contains("रोशनी,"), "comma not hugged to Devanagari word — got: {t:?}");
    assert!(!t.contains("रोशनी ,"), "spurious space before comma — got: {t:?}");
}
