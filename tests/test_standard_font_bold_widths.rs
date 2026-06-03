//! Tests for standard PDF font Bold variant widths.
//!
//! Helvetica-Bold and Times-Bold have different glyph widths from their Regular
//! counterparts (e.g. Helvetica-Bold 'W' = 944 units vs Helvetica 'W' = 722).
//! Before this fix, Bold variants incorrectly used the Regular width table.

use pdf_oxide::PdfDocument;

fn pdf_with_base_font(base_font: &str, text: &str, font_size: f32) -> Vec<u8> {
    let content = format!(
        "BT /{base_font} {font_size} Tf 1 0 0 1 100 500 Tm ({text}) Tj ET\n",
        base_font = "F0",
        font_size = font_size,
        text = text
    );
    let font_dict = format!(
        "<< /Type /Font /Subtype /Type1 /BaseFont /{base_font} >>",
        base_font = base_font
    );

    let mut out: Vec<u8> = Vec::new();
    let mut offsets: Vec<usize> = vec![0];
    out.extend_from_slice(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n");

    let push = |out: &mut Vec<u8>, offsets: &mut Vec<usize>, body: &str| {
        offsets.push(out.len());
        let id = offsets.len() - 1;
        out.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };

    push(&mut out, &mut offsets, "<< /Type /Catalog /Pages 2 0 R >>");
    push(&mut out, &mut offsets, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    push(
        &mut out,
        &mut offsets,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 900] \
         /Resources << /Font << /F0 5 0 R >> >> /Contents 4 0 R >>",
    );
    push(
        &mut out,
        &mut offsets,
        &format!("<< /Length {} >>\nstream\n{content}\nendstream", content.len() + 1),
    );
    push(&mut out, &mut offsets, &font_dict);

    let xref_offset = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n", offsets.len()).as_bytes());
    out.extend_from_slice(b"0000000000 65535 f \n");
    for &off in &offsets[1..] {
        out.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n",
            offsets.len()
        )
        .as_bytes(),
    );
    out
}

/// Helvetica-Bold 'b' (code 98) has width 611 in the Adobe AFM, while
/// Helvetica-Regular 'b' has width 556. The Bold variant must use the Bold table.
#[test]
fn helvetica_bold_b_is_wider_than_regular() {
    let font_size = 1000.0_f32;

    let bold_pdf = pdf_with_base_font("Helvetica-Bold", "b", font_size);
    let regular_pdf = pdf_with_base_font("Helvetica", "b", font_size);

    let bold_tmp = tempfile::NamedTempFile::new().unwrap();
    let regular_tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(bold_tmp.path(), &bold_pdf).unwrap();
    std::fs::write(regular_tmp.path(), &regular_pdf).unwrap();

    let bold_doc = PdfDocument::open(bold_tmp.path()).expect("open bold");
    let regular_doc = PdfDocument::open(regular_tmp.path()).expect("open regular");

    let bold_chars = bold_doc.extract_chars(0).expect("extract bold");
    let regular_chars = regular_doc.extract_chars(0).expect("extract regular");

    let bold_b = bold_chars
        .iter()
        .find(|c| c.char == 'b')
        .expect("'b' in bold");
    let regular_b = regular_chars
        .iter()
        .find(|c| c.char == 'b')
        .expect("'b' in regular");

    // Adobe AFM: Helvetica-Bold 'b' = 611 units; Helvetica 'b' = 556 units.
    // With font_size=1000, advance_width is proportional to the AFM width.
    assert!(
        bold_b.advance_width > regular_b.advance_width,
        "Helvetica-Bold 'b' ({:.1}) must be wider than Helvetica 'b' ({:.1})",
        bold_b.advance_width,
        regular_b.advance_width
    );

    // Ratio should be approximately 611/556 ≈ 1.099
    let ratio = bold_b.advance_width / regular_b.advance_width;
    assert!(
        ratio > 1.05,
        "Bold 'b' ({:.1}) should be wider than Regular 'b' ({:.1}), ratio={:.3}",
        bold_b.advance_width,
        regular_b.advance_width,
        ratio
    );
}

/// Times-Bold 'a' (code 97) is 500 units wide vs Times-Roman's 444.
/// Bold variant must use the Bold-specific table.
#[test]
fn times_bold_a_is_wider_than_roman() {
    let font_size = 1000.0_f32;

    let bold_pdf = pdf_with_base_font("Times-Bold", "a", font_size);
    let roman_pdf = pdf_with_base_font("Times-Roman", "a", font_size);

    let bold_tmp = tempfile::NamedTempFile::new().unwrap();
    let roman_tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(bold_tmp.path(), &bold_pdf).unwrap();
    std::fs::write(roman_tmp.path(), &roman_pdf).unwrap();

    let bold_doc = PdfDocument::open(bold_tmp.path()).expect("open bold");
    let roman_doc = PdfDocument::open(roman_tmp.path()).expect("open roman");

    let bold_chars = bold_doc.extract_chars(0).expect("extract bold");
    let roman_chars = roman_doc.extract_chars(0).expect("extract roman");

    let bold_a = bold_chars
        .iter()
        .find(|c| c.char == 'a')
        .expect("'a' in bold");
    let roman_a = roman_chars
        .iter()
        .find(|c| c.char == 'a')
        .expect("'a' in roman");

    // Adobe AFM: Times-Bold 'a' = 500, Times-Roman 'a' = 444.
    assert!(
        bold_a.advance_width > roman_a.advance_width,
        "Times-Bold 'a' ({:.1}) must be wider than Times-Roman 'a' ({:.1})",
        bold_a.advance_width,
        roman_a.advance_width
    );

    // Ratio should be approximately 500/444 ≈ 1.126
    let ratio = bold_a.advance_width / roman_a.advance_width;
    assert!(
        ratio > 1.05,
        "Times-Bold 'a' should be wider than Times-Roman 'a', ratio={:.3}",
        ratio
    );
}

/// Times-BoldItalic has its own width table, distinct from Times-Bold.
/// Key divergences: W=889 (Bold=1000), !=389 (Bold=333), H=778 (Bold=778).
/// This test pins two chars that clearly distinguish the tables.
#[test]
fn times_bold_italic_differs_from_bold() {
    let font_size = 1000.0_f32;

    // 'W': Adobe AFM Times-BoldItalic=889, Times-Bold=1000 (BoldItalic is narrower).
    let bi_w_pdf = pdf_with_base_font("Times-BoldItalic", "W", font_size);
    let bold_w_pdf = pdf_with_base_font("Times-Bold", "W", font_size);

    let bi_w_tmp = tempfile::NamedTempFile::new().unwrap();
    let bold_w_tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(bi_w_tmp.path(), &bi_w_pdf).unwrap();
    std::fs::write(bold_w_tmp.path(), &bold_w_pdf).unwrap();

    let bi_w_doc = PdfDocument::open(bi_w_tmp.path()).expect("open bold-italic W");
    let bold_w_doc = PdfDocument::open(bold_w_tmp.path()).expect("open bold W");
    let bi_w_chars = bi_w_doc.extract_chars(0).expect("extract bold-italic");
    let bold_w_chars = bold_w_doc.extract_chars(0).expect("extract bold");

    let bi_w = bi_w_chars
        .iter()
        .find(|c| c.char == 'W')
        .expect("'W' in bold-italic");
    let bold_w = bold_w_chars
        .iter()
        .find(|c| c.char == 'W')
        .expect("'W' in bold");

    // Adobe AFM: Times-BoldItalic 'W' = 889, Times-Bold 'W' = 1000.
    assert!(
        bi_w.advance_width < bold_w.advance_width,
        "Times-BoldItalic 'W' ({:.1}) must be narrower than Times-Bold 'W' ({:.1})",
        bi_w.advance_width,
        bold_w.advance_width
    );
}

/// Times-Italic has its own width table, distinct from Times-Roman.
/// Key divergences: W=833 (Roman=944), A=611 (Roman=722), f=278 (Roman=333).
/// This test pins the W glyph (code 87), which is most distinctive.
#[test]
fn times_italic_w_differs_from_roman() {
    let font_size = 1000.0_f32;

    let italic_pdf = pdf_with_base_font("Times-Italic", "W", font_size);
    let roman_pdf = pdf_with_base_font("Times-Roman", "W", font_size);

    let italic_tmp = tempfile::NamedTempFile::new().unwrap();
    let roman_tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(italic_tmp.path(), &italic_pdf).unwrap();
    std::fs::write(roman_tmp.path(), &roman_pdf).unwrap();

    let italic_doc = PdfDocument::open(italic_tmp.path()).expect("open italic");
    let roman_doc = PdfDocument::open(roman_tmp.path()).expect("open roman");

    let italic_chars = italic_doc.extract_chars(0).expect("extract italic");
    let roman_chars = roman_doc.extract_chars(0).expect("extract roman");

    let italic_w = italic_chars
        .iter()
        .find(|c| c.char == 'W')
        .expect("'W' in italic");
    let roman_w = roman_chars
        .iter()
        .find(|c| c.char == 'W')
        .expect("'W' in roman");

    // Adobe AFM: Times-Italic 'W' = 833, Times-Roman 'W' = 944.
    assert!(
        italic_w.advance_width < roman_w.advance_width,
        "Times-Italic 'W' ({:.1}) must be narrower than Times-Roman 'W' ({:.1})",
        italic_w.advance_width,
        roman_w.advance_width
    );
}
