//! Tests for ascent/descent on TextChar and FontInfo.
//!
//! TextChar::ascent/descent are in device space (em fraction × effective font size).
//! FontInfo::ascent/descent are em fractions (the intermediate per-font values).
//!
//! Verifies that ascent/descent are populated from:
//! 1. Standard font metrics lookup (e.g. Helvetica: 0.718em → 8.616 at Tfs=12)
//! 2. Font descriptor /Ascent and /Descent keys (1/1000-em → scaled to device space)
//! 3. Positive /Descent normalization (negated to ensure ≤ 0 before scaling)
//! 4. Fallback defaults (0.95 / -0.35 em → 11.4 / -4.2 at Tfs=12) for unknown fonts

use pdf_oxide::PdfDocument;

fn minimal_pdf_with_font(font_dict: &str, content: &str) -> Vec<u8> {
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
    push(&mut out, &mut offsets, font_dict);

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

/// Helvetica is a standard Type1 font. Without a FontDescriptor, ascent/descent
/// must come from the built-in metrics (0.718em / -0.207em), scaled by Tfs=12.
#[test]
fn ascent_descent_from_standard_font_helvetica() {
    let font_dict = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>";
    let content = "BT /F0 12 Tf 1 0 0 1 100 700 Tm (Hello) Tj ET\n";
    let pdf = minimal_pdf_with_font(font_dict, content);

    let tmp = tempfile::NamedTempFile::new().expect("temp");
    std::fs::write(tmp.path(), &pdf).unwrap();

    let doc = PdfDocument::open(tmp.path()).expect("open");
    let chars = doc.extract_chars(0).expect("extract_chars");

    let letters: Vec<_> = chars.iter().filter(|c| !c.char.is_whitespace()).collect();
    assert!(!letters.is_empty(), "expected letters from 'Hello'");

    for ch in &letters {
        assert!(
            (ch.ascent - 0.718 * 12.0).abs() < 0.1,
            "char {:?}: ascent = {:.4}, expected ~{:.3} (0.718em × Tfs=12)",
            ch.char,
            ch.ascent,
            0.718 * 12.0_f32,
        );
        assert!(
            (ch.descent - (-0.207 * 12.0)).abs() < 0.1,
            "char {:?}: descent = {:.4}, expected ~{:.3} (-0.207em × Tfs=12)",
            ch.char,
            ch.descent,
            -0.207 * 12.0_f32,
        );
    }
}

/// A PDF font with a FontDescriptor specifying /Ascent 800 /Descent -200 (raw
/// 1/1000-em values). At Tfs=12: ascent → 9.6, descent → -2.4.
#[test]
fn ascent_descent_from_font_descriptor() {
    // We need 6+ objects for the font descriptor; add obj 6.
    let content = "BT /F0 12 Tf 1 0 0 1 100 700 Tm (A) Tj ET\n";
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
    // obj 5: Font referencing descriptor at obj 6
    push(
        &mut out,
        &mut offsets,
        "<< /Type /Font /Subtype /Type1 /BaseFont /CustomFont /FontDescriptor 6 0 R >>",
    );
    // obj 6: FontDescriptor with Ascent=800 Descent=-200
    push(
        &mut out,
        &mut offsets,
        "<< /Type /FontDescriptor /FontName /CustomFont /Ascent 800 /Descent -200 /Flags 32 >>",
    );

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

    let tmp = tempfile::NamedTempFile::new().expect("temp");
    std::fs::write(tmp.path(), &out).unwrap();

    let doc = PdfDocument::open(tmp.path()).expect("open");
    let chars = doc.extract_chars(0).expect("extract_chars");

    let letters: Vec<_> = chars.iter().filter(|c| !c.char.is_whitespace()).collect();
    assert!(!letters.is_empty(), "expected 'A'");

    let ch = &letters[0];
    assert!(
        (ch.ascent - 0.8 * 12.0).abs() < 0.1,
        "ascent = {:.4}, expected ~{:.1} (800/1000 em × Tfs=12)",
        ch.ascent,
        0.8 * 12.0_f32,
    );
    assert!(
        (ch.descent - (-0.2 * 12.0)).abs() < 0.1,
        "descent = {:.4}, expected ~{:.1} (-200/1000 em × Tfs=12)",
        ch.descent,
        -0.2 * 12.0_f32,
    );
}

/// A PDF FontDescriptor with a positive /Descent value (some PDFs store descent
/// as a positive magnitude). It must be normalized to negative.
#[test]
fn positive_descent_normalized_to_negative() {
    let content = "BT /F0 12 Tf 1 0 0 1 100 700 Tm (A) Tj ET\n";
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
    push(
        &mut out,
        &mut offsets,
        "<< /Type /Font /Subtype /Type1 /BaseFont /CustomFont2 /FontDescriptor 6 0 R >>",
    );
    // Positive descent (magnitude 150) — should be normalized to -0.15
    push(
        &mut out,
        &mut offsets,
        "<< /Type /FontDescriptor /FontName /CustomFont2 /Ascent 750 /Descent 150 /Flags 32 >>",
    );

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

    let tmp = tempfile::NamedTempFile::new().expect("temp");
    std::fs::write(tmp.path(), &out).unwrap();

    let doc = PdfDocument::open(tmp.path()).expect("open");
    let chars = doc.extract_chars(0).expect("extract_chars");

    let letters: Vec<_> = chars.iter().filter(|c| !c.char.is_whitespace()).collect();
    assert!(!letters.is_empty(), "expected 'A'");

    let ch = &letters[0];
    assert!(ch.descent < 0.0, "descent must be negative; got {:.4}", ch.descent);
    assert!(
        (ch.descent - (-0.15 * 12.0)).abs() < 0.1,
        "descent = {:.4}, expected ~{:.1} (positive 150 normalized to -0.15em × Tfs=12)",
        ch.descent,
        -0.15 * 12.0_f32,
    );
}

/// An unknown font (not one of the 14 standard PDF fonts) without a
/// FontDescriptor should fall back to 0.95em / -0.35em (→ 11.4 / -4.2 at Tfs=12).
#[test]
fn ascent_descent_fallback_defaults_for_unknown_font() {
    let font_dict = "<< /Type /Font /Subtype /Type1 /BaseFont /UnknownFantasyFont >>";
    let content = "BT /F0 12 Tf 1 0 0 1 100 700 Tm (A) Tj ET\n";
    let pdf = minimal_pdf_with_font(font_dict, content);

    let tmp = tempfile::NamedTempFile::new().expect("temp");
    std::fs::write(tmp.path(), &pdf).unwrap();

    let doc = PdfDocument::open(tmp.path()).expect("open");
    let chars = doc.extract_chars(0).expect("extract_chars");

    let letters: Vec<_> = chars.iter().filter(|c| !c.char.is_whitespace()).collect();
    assert!(!letters.is_empty(), "expected 'A'");

    let ch = &letters[0];
    assert!(
        (ch.ascent - 0.95 * 12.0).abs() < 0.1,
        "ascent = {:.4}, expected ~{:.1} (0.95em fallback × Tfs=12)",
        ch.ascent,
        0.95 * 12.0_f32,
    );
    assert!(
        (ch.descent - (-0.35 * 12.0)).abs() < 0.1,
        "descent = {:.4}, expected ~{:.1} (-0.35em fallback × Tfs=12)",
        ch.descent,
        -0.35 * 12.0_f32,
    );
}
