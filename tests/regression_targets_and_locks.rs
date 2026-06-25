//! Regression TARGETS (`#[ignore]`d, pass when the deferred fix lands) and
//! behaviour LOCKS (assert current correct behaviour so it can't silently
//! change). All fixtures are hand-written minimal PDFs (no third-party files).
//! Targets document the expected post-fix output for the unfinished items
//! (RW-1 reading order, SEG-AR/SEG-HE RTL); un-ignore them when fixed.

use pdf_oxide::PdfDocument;

// ---------------------------------------------------------------------------
// Type0/Identity-H fixture (carries any script's text via ToUnicode).
// ---------------------------------------------------------------------------
struct Run<'a> {
    x: f32,
    y: f32,
    text: &'a str,
    codes: &'a [u16],
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
    build_pdf(&content, Some((&tounicode, &w)))
}

// ---------------------------------------------------------------------------
// Simple Helvetica (WinAnsi) fixture for Latin reading-order targets.
// ---------------------------------------------------------------------------
struct Text<'a> {
    x: f32,
    y: f32,
    s: &'a str,
}

fn helvetica_pdf(items: &[Text]) -> Vec<u8> {
    let mut content = String::new();
    for it in items {
        content.push_str(&format!(
            "BT /F1 11 Tf 1 0 0 1 {:.1} {:.1} Tm ({}) Tj ET\n",
            it.x, it.y, it.s
        ));
    }
    build_pdf(&content, None)
}

/// Shared object assembler. When `type0` is `Some((tounicode, w))` font obj 5 is
/// a Type0 Identity-H font with those resources; otherwise a base-14 Helvetica.
fn build_pdf(content: &str, type0: Option<(&str, &str)>) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    let n = if type0.is_some() { 9 } else { 6 };
    let mut off: Vec<usize> = vec![0; n];
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
    if let Some((tounicode, w)) = type0 {
        obj(
            &mut buf,
            &mut off,
            5,
            "<< /Type /Font /Subtype /Type0 /BaseFont /F /Encoding /Identity-H \
             /DescendantFonts [6 0 R] /ToUnicode 7 0 R >>"
                .into(),
        );
        obj(
            &mut buf,
            &mut off,
            6,
            format!(
                "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /F \
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
            "<< /Type /FontDescriptor /FontName /F /Flags 4 \
             /FontBBox [0 -200 1000 800] /ItalicAngle 0 /Ascent 800 /Descent -200 \
             /CapHeight 700 /StemV 80 >>"
                .into(),
        );
    } else {
        obj(
            &mut buf,
            &mut off,
            5,
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
                .into(),
        );
    }
    let xref = buf.len();
    buf.extend_from_slice(format!("xref\n0 {n}\n0000000000 65535 f \n").as_bytes());
    for id in 1..n {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(format!("trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n").as_bytes());
    buf.extend_from_slice(format!("{xref}\n%%EOF\n").as_bytes());
    buf
}

fn text_of(bytes: Vec<u8>) -> String {
    PdfDocument::from_bytes(bytes)
        .expect("fixture pdf parses")
        .extract_text(0)
        .expect("extract_text")
}

// ===========================================================================
// RW-1: a full-width title band above a narrow-sidebar + wide-body page must read
// whole. The title spans the page width at the top; a narrow left metadata column
// (Citation / Editor / Received …) sits beside a wide body column below. The plain
// XY-cut misreads this as two columns and slices the title's word-spans at the body
// gutter (title shatter). `sidebar_body_reading_order` peels the band and reads the
// body before the sidebar furniture, so the title stays contiguous.
// ===========================================================================
#[test]
fn rw1_full_width_title_reads_contiguously() {
    // Geometry mirrors a real MDPI first page: x_min≈50, body column starts ~28%
    // from the left (a narrow sidebar, gutter left-of-centre), title spans full
    // width on top. Needs enough lines that the sidebar/body classifier engages.
    // A real journal title line is emitted as one full-width text run on the top
    // baseline (one show operator → one span), starting at the left margin and
    // crossing the body gutter. That single wide band is what the plain XY-cut
    // would otherwise slice along the gutter; `sidebar_body_reading_order` keeps
    // it on top and reads the body before the metadata sidebar.
    let mut owned: Vec<String> = Vec::new();
    let mut items: Vec<Text> = Vec::new();
    items.push(Text {
        x: 55.0,
        y: 752.0,
        s: "A Prospective Study Comparing Laparoscopic Methods",
    });
    // Narrow left metadata sidebar (short lines ending well before the gutter
    // ~185), as a block near the top — distinct baselines from the body below.
    // A real publisher sidebar carries several DISTINCT furniture labels
    // (Citation / Received / Accepted / DOI …); the classifier requires >=2
    // distinct ones so plain narrow columns and label:value forms never engage.
    let furniture = [
        "Citation 0",
        "Received 24",
        "Accepted 24",
        "Published 24",
        "Copyright 24",
        "Licensee X",
        "ISSN 1234",
        "Publisher Y",
    ];
    for line in furniture {
        owned.push(line.to_string());
    }
    // Wide right body column (long lines from x≈195 across to ≈545), below the
    // sidebar block (real first-page layout: metadata block, then the body flow).
    for i in 0..24 {
        owned.push(format!("Body line {i} reads across the wide main column of the page here now"));
    }
    let mut k = 0;
    for i in 0..8 {
        items.push(Text {
            x: 52.0,
            y: 730.0 - i as f32 * 13.0,
            s: &owned[k],
        });
        k += 1;
    }
    for i in 0..24 {
        items.push(Text {
            x: 195.0,
            y: 612.0 - i as f32 * 13.0,
            s: &owned[k],
        });
        k += 1;
    }
    let t = text_of(helvetica_pdf(&items));
    assert!(
        t.contains("A Prospective Study Comparing Laparoscopic Methods"),
        "title shattered across the column gutter — got: {:?}",
        &t[..t.len().min(200)]
    );
    // The body must read before the sidebar furniture (and not be interleaved
    // with the title): the first body line precedes the first Citation line.
    let body0 = t.find("Body line 0").expect("body present");
    let cite0 = t.find("Citation 0").expect("sidebar present");
    assert!(body0 < cite0, "sidebar furniture not read after the body — got: {t:?}");
}

/// Build a `type0_pdf` from a logical-order string as a faithful *visual-order*
/// RTL producer would. A real Arabic/Hebrew PDF positions glyphs left-to-right
/// in on-screen order: the RTL letters run reversed, but an embedded European
/// number is still rendered left-to-right (UAX #9 L2). So the visual order is
/// the reverse of the logical string with each maximal ASCII-digit run kept in
/// place. Computed here independently of the extractor so the test genuinely
/// exercises the extractor's visual→logical recovery, then asserts the original
/// logical string comes back.
fn rtl_visual_pdf(x: f32, y: f32, logical: &str) -> Vec<u8> {
    let chars: Vec<char> = logical.chars().collect();
    let mut visual: Vec<char> = Vec::with_capacity(chars.len());
    let mut i = chars.len();
    while i > 0 {
        i -= 1;
        if chars[i].is_ascii_digit() {
            let end = i + 1;
            while i > 0 && chars[i - 1].is_ascii_digit() {
                i -= 1;
            }
            visual.extend_from_slice(&chars[i..end]);
        } else {
            visual.push(chars[i]);
        }
    }
    let text: String = visual.into_iter().collect();
    let codes: Vec<u16> = (1..=text.chars().count() as u16).collect();
    type0_pdf(&[Run {
        x,
        y,
        text: &text,
        codes: &codes,
    }])
}

// ===========================================================================
// SEG-AR: an Arabic word laid out in visual order must extract intact in
// logical order, not shattered or left visually reversed. الثدييات
// (al-thadyiyaat) is one run; the extractor reverses the visual glyph order
// back to logical. (v0.3.62 SEG-AR.)
// ===========================================================================
#[test]
fn seg_ar_word_not_shattered() {
    // الثدييات = ا ل ث د ي ي ا ت
    let t = text_of(rtl_visual_pdf(400.0, 700.0, "الثدييات"));
    assert!(t.contains("الثدييات"), "Arabic word shattered — got: {t:?}");
}

// ===========================================================================
// SEG-HE: a Hebrew prefix + year + comma read back in logical order with the
// digits NOT reversed. Visual-order "ל-2009," must extract as "ל-2009,", not
// "ל-9002," — a whole-string reversal would mirror the European number, so the
// extractor keeps the digit run left-to-right (UAX #9 L2). (v0.3.62 SEG-HE.)
// ===========================================================================
#[test]
fn seg_he_prefix_year_comma_stays_together() {
    // ל - 2 0 0 9 ,
    let t = text_of(rtl_visual_pdf(300.0, 700.0, "ל-2009,"));
    assert!(t.contains("ל-2009,"), "Hebrew prefix+year+comma mangled — got: {t:?}");
}

// ===========================================================================
// LOCK — sub/superscript: pdf_oxide preserves the producer's Unicode subscript
// (U+2082) rather than normalising to ASCII. This is intentional fidelity
// (v0.3.65_3/_4: gold-normalisation, not a bug); lock it so it can't silently
// flip and so the merge of a subscript run into its base keeps working.
// ===========================================================================
#[test]
fn subscript_unicode_codepoint_is_preserved() {
    // H, ₂ (U+2082), O on one baseline.
    let t = text_of(type0_pdf(&[Run {
        x: 100.0,
        y: 700.0,
        text: "H\u{2082}O",
        codes: &[0x0001, 0x0002, 0x0003],
    }]));
    assert!(t.contains("H\u{2082}O"), "subscript codepoint not preserved — got: {t:?}");
}
