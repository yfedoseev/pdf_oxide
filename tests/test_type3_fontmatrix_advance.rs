//! Tests for Type3 font FontMatrix[0] scaling of advance_width
//!
//! Before the fix, fs_factor was always `font_size / 1000.0`, implicitly
//! assuming glyph widths are in 1/1000-em units (Type1 convention).
//! After the fix, fs_factor = `font_size * font_matrix_a`, where font_matrix_a
//! is FontMatrix[0] from the font dictionary.
//!
//! For a Type3 font with FontMatrix [0.2 0 0 0.2 0 0] (font_matrix_a = 0.2)
//! and a glyph width of 5.0 (glyph-space units), at Tfs = 10.0:
//!   - Fixed:   advance_width ≈ 5.0 × (10.0 × 0.2) = 10.0
//!   - Broken:  advance_width ≈ 5.0 × (10.0 / 1000.0) = 0.05

use pdf_oxide::PdfDocument;

/// Build a minimal Type3 PDF with the given FontMatrix scalar and no /Widths array,
/// so the extractor must fall back to the flags-based `default_width` heuristic.
fn type3_pdf_no_widths_array(font_matrix_a: f64, tfs: f64) -> Vec<u8> {
    let content = format!("BT /F0 {tfs} Tf 1 0 0 1 100 500 Tm (A) Tj ET\n");
    let char_proc_a = "5 0 d0\n".to_string();

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
    // Intentionally omit /FirstChar, /LastChar, /Widths — forces default_width fallback.
    push(
        &mut out,
        &mut offsets,
        &format!(
            "<< /Type /Font /Subtype /Type3 \
             /FontBBox [0 0 1 1] \
             /FontMatrix [{font_matrix_a} 0 0 {font_matrix_a} 0 0] \
             /Encoding << /Type /Encoding /Differences [65 /A] >> \
             /CharProcs << /A 6 0 R >> \
             /Resources << >> \
             >>"
        ),
    );
    push(
        &mut out,
        &mut offsets,
        &format!("<< /Length {} >>\nstream\n{char_proc_a}\nendstream", char_proc_a.len() + 1),
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
    out
}

fn type3_pdf_advance(font_matrix_a: f64, glyph_width: f64, tfs: f64) -> Vec<u8> {
    let content = format!("BT /F0 {tfs} Tf 1 0 0 1 100 500 Tm (A) Tj ET\n");
    let char_proc_a = format!("{glyph_width} 0 d0\n");

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
    let b = 0.0_f64;
    let c = 0.0_f64;
    let e = 0.0_f64;
    let f = 0.0_f64;
    push(
        &mut out,
        &mut offsets,
        &format!(
            "<< /Type /Font /Subtype /Type3 \
             /FontBBox [0 0 1 1] \
             /FontMatrix [{font_matrix_a} {b} {c} {font_matrix_a} {e} {f}] \
             /FirstChar 65 /LastChar 65 \
             /Widths [{glyph_width}] \
             /Encoding << /Type /Encoding /Differences [65 /A] >> \
             /CharProcs << /A 6 0 R >> \
             /Resources << >> \
             >>"
        ),
    );
    push(
        &mut out,
        &mut offsets,
        &format!("<< /Length {} >>\nstream\n{char_proc_a}\nendstream", char_proc_a.len() + 1),
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
    out
}

/// Type3 with FontMatrix [0.2 0 0 0.2 0 0], glyph_width=5.0, Tfs=10.0.
///
/// Fixed:  advance_width = 5.0 × (10.0 × 0.2) = 10.0
/// Broken: advance_width = 5.0 × (10.0 / 1000.0) = 0.05
///
/// The threshold 1.0 cleanly distinguishes the two.
#[test]
fn type3_fontmatrix_advance_width_scaled_correctly() {
    let pdf = type3_pdf_advance(0.2, 5.0, 10.0);

    let tmp = tempfile::NamedTempFile::new().expect("temp");
    std::fs::write(tmp.path(), &pdf).unwrap();

    let doc = PdfDocument::open(tmp.path()).expect("open");
    let chars = doc.extract_chars(0).expect("extract_chars");

    // The CharProc paints `(A)`, so the glyph must extract. Assert rather than
    // silently returning, so a width-scaling regression can't hide behind a
    // no-op test.
    let ch = chars
        .iter()
        .find(|c| c.char == 'A')
        .expect("Type3 glyph 'A' should extract from the constructed CharProc");

    // Fixed:  advance_width ≈ 5.0 × (10.0 × 0.2) = 10.0
    // Broken: advance_width ≈ 5.0 × (10.0 / 1000.0) = 0.05
    assert!(
        ch.advance_width > 1.0,
        "Type3 'A' advance_width ({:.4}) should reflect FontMatrix[0]=0.2 scaling (~10.0); \
         without the fix it would be ~0.05",
        ch.advance_width
    );
}

/// Type3 with identity FontMatrix [1 0 0 1 0 0] and no /Widths array.
///
/// Without /Widths the extractor falls back to the flags-based `default_width`
/// heuristic (~550 in 1/1000-em units).  Before the fix, that value was used
/// as-is with fs_factor = font_size × 1.0, giving an advance ~1000× too wide.
/// After the fix, default_width is rescaled by 0.001/font_matrix_a at parse
/// time, so the advance comes out in the correct ~0.55em range.
#[test]
fn type3_identity_fontmatrix_default_width_not_overscaled() {
    // Identity FontMatrix: font_matrix_a = 1.0, Tfs = 10.0
    // After fix:  advance_width ≈ 550 × 0.001/1.0 × 10.0 × 1.0 ≈ 5.5
    // Before fix: advance_width ≈ 550              × 10.0 × 1.0 ≈ 5500
    let pdf = type3_pdf_no_widths_array(1.0, 10.0);

    let tmp = tempfile::NamedTempFile::new().expect("temp");
    std::fs::write(tmp.path(), &pdf).unwrap();

    let doc = PdfDocument::open(tmp.path()).expect("open");
    let chars = doc.extract_chars(0).expect("extract_chars");

    // Assert the glyph extracts (rather than silently returning) so a
    // default-width over-scaling regression can't pass vacuously.
    let ch = chars
        .iter()
        .find(|c| c.char == 'A')
        .expect("Type3 glyph 'A' should extract from the constructed CharProc");

    assert!(
        ch.advance_width < 100.0,
        "Type3 default advance_width ({:.2}) with identity FontMatrix should be ~5.5 \
         (~0.55em × Tfs=10.0), not ~5500 (1000× overscaled without the default_width fix)",
        ch.advance_width
    );
    assert!(
        ch.advance_width > 0.1,
        "Type3 default advance_width ({:.4}) should be positive",
        ch.advance_width
    );
}

/// A malformed `/FontMatrix [0 0 0 0 0 0]` (degenerate zero horizontal scale,
/// ISO 32000-1 §9.2.4 / §9.6.5) must NOT yield an `inf`/`NaN` advance via the
/// `default_width * 0.001 / font_matrix_a` rescale — `font_matrix_a` falls back
/// to the standard 0.001 (Type 1) scale, so the advance stays finite and sane.
#[test]
fn type3_degenerate_zero_fontmatrix_falls_back_safely() {
    let pdf = type3_pdf_no_widths_array(0.0, 10.0);

    let tmp = tempfile::NamedTempFile::new().expect("temp");
    std::fs::write(tmp.path(), &pdf).unwrap();

    let doc = PdfDocument::open(tmp.path()).expect("open");
    let chars = doc.extract_chars(0).expect("extract_chars");

    let ch = chars
        .iter()
        .find(|c| c.char == 'A')
        .expect("Type3 glyph 'A' should extract even with a degenerate FontMatrix");

    assert!(
        ch.advance_width.is_finite(),
        "degenerate FontMatrix[0]=0 must not produce inf/NaN advance (got {})",
        ch.advance_width
    );
    assert!(
        ch.advance_width > 0.0 && ch.advance_width < 1000.0,
        "degenerate FontMatrix should fall back to a sane advance (~5.5), got {}",
        ch.advance_width
    );
}
