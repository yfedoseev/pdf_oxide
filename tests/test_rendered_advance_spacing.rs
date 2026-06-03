//! Tests for rendered_advance including Tc/Tw character and word spacing.
//!
//! rendered_advance is the per-glyph cursor advance including Tc (+ Tw for
//! U+0020).  TJ array adjustments are NOT folded in — they are emitted as
//! separate synthetic-space TextChars.  This differs from advance_width,
//! which is only the glyph's own width.

use pdf_oxide::PdfDocument;

/// Build a minimal 1-page PDF where the content stream sets character spacing
/// (Tc) to a known value before rendering a short string.
fn pdf_with_char_spacing(tc: f32) -> Vec<u8> {
    let content = format!("BT /F0 12 Tf 1 0 0 1 100 700 Tm {tc} Tc (ABC) Tj ET\n");

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
    push(&mut out, &mut offsets, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");

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

/// Build a minimal 1-page PDF with both character spacing (Tc) and word
/// spacing (Tw) set, and a string that includes a space character.
fn pdf_with_word_spacing(tc: f32, tw: f32) -> Vec<u8> {
    let content = format!("BT /F0 12 Tf 1 0 0 1 100 700 Tm {tc} Tc {tw} Tw (A B) Tj ET\n");

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
    push(&mut out, &mut offsets, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");

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

/// `rendered_advance` for a non-space character must exceed `advance_width` by
/// exactly Tc (character spacing) when Tc > 0.
#[test]
fn rendered_advance_includes_char_spacing() {
    let tc = 2.0_f32;
    let pdf = pdf_with_char_spacing(tc);
    let tmp = tempfile::NamedTempFile::new().expect("temp");
    std::fs::write(tmp.path(), &pdf).unwrap();

    let doc = PdfDocument::open(tmp.path()).expect("open");
    let chars = doc.extract_chars(0).expect("extract_chars");

    // Filter to non-whitespace chars from the "ABC" string.
    let letters: Vec<_> = chars.iter().filter(|c| !c.char.is_whitespace()).collect();
    assert!(!letters.is_empty(), "expected to extract 'A', 'B', 'C'");

    for ch in &letters {
        let delta = ch.rendered_advance - ch.advance_width;
        assert!(
            (delta - tc).abs() < 0.5,
            "char {:?}: rendered_advance - advance_width = {delta:.3}, expected Tc={tc}",
            ch.char
        );
    }
}

/// `rendered_advance` for a space character must exceed `advance_width` by
/// Tc + Tw when both character spacing and word spacing are set.
#[test]
fn rendered_advance_includes_word_spacing_for_space() {
    let tc = 1.0_f32;
    let tw = 3.0_f32;
    let pdf = pdf_with_word_spacing(tc, tw);
    let tmp = tempfile::NamedTempFile::new().expect("temp");
    std::fs::write(tmp.path(), &pdf).unwrap();

    let doc = PdfDocument::open(tmp.path()).expect("open");
    let chars = doc.extract_chars(0).expect("extract_chars");

    let spaces: Vec<_> = chars.iter().filter(|c| c.char == ' ').collect();
    assert!(!spaces.is_empty(), "expected a space character in 'A B'");

    for sp in &spaces {
        let delta = sp.rendered_advance - sp.advance_width;
        let expected = tc + tw;
        assert!(
            (delta - expected).abs() < 0.5,
            "space: rendered_advance - advance_width = {delta:.3}, expected Tc+Tw={expected}"
        );
    }
}

/// With Tc = 0 and Tw = 0, rendered_advance must equal advance_width for every
/// character (no extra spacing is added).
#[test]
fn rendered_advance_equals_advance_width_when_no_spacing() {
    let pdf = pdf_with_char_spacing(0.0);
    let tmp = tempfile::NamedTempFile::new().expect("temp");
    std::fs::write(tmp.path(), &pdf).unwrap();

    let doc = PdfDocument::open(tmp.path()).expect("open");
    let chars = doc.extract_chars(0).expect("extract_chars");

    let letters: Vec<_> = chars.iter().filter(|c| !c.char.is_whitespace()).collect();
    assert!(!letters.is_empty(), "expected to extract 'A', 'B', 'C'");

    for ch in &letters {
        let delta = (ch.rendered_advance - ch.advance_width).abs();
        assert!(
            delta < 0.1,
            "char {:?}: rendered_advance ({:.3}) should equal advance_width ({:.3}) when Tc=0",
            ch.char,
            ch.rendered_advance,
            ch.advance_width
        );
    }
}

/// A glyph inside a TJ array must have rendered_advance ≈ glyph_advance + Tc,
/// NOT including the neighboring TJ kern offset.  The kern is emitted as a
/// separate synthetic-space TextChar between the two letter glyphs.
#[test]
fn tj_kern_offset_is_separate_synthetic_space_not_in_rendered_advance() {
    // TJ array: render 'A', then advance by a large positive kern (-600 units =
    // 7.2 pt at 12 pt), then render 'B'.  The kern exceeds the space-insertion
    // threshold so a synthetic ' ' is inserted between the two glyphs.
    let tc = 3.0_f32;
    let content = format!("BT /F0 12 Tf 1 0 0 1 100 700 Tm {tc} Tc [(A) -600 (B)] TJ ET\n");

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
    push(&mut out, &mut offsets, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");
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

    // Expect: 'A', synthetic ' ' (from the -600 kern), 'B'
    let letter_a = chars.iter().find(|c| c.char == 'A').expect("'A' not found");
    let letter_b = chars.iter().find(|c| c.char == 'B').expect("'B' not found");

    // The kern gap (600/1000 * 12 = 7.2 pt) must NOT be included in 'A's advance.
    let kern_gap = 600.0_f32 / 1000.0 * 12.0; // 7.2 pt
    let max_allowed = letter_a.advance_width + tc + kern_gap * 0.5;
    assert!(
        letter_a.rendered_advance < max_allowed,
        "'A' rendered_advance ({:.3}) should not include the TJ kern gap ({kern_gap:.1} pt); \
         advance_width={:.3} Tc={tc}",
        letter_a.rendered_advance,
        letter_a.advance_width,
    );
    // rendered_advance should be approximately advance_width + Tc
    let expected = letter_a.advance_width + tc;
    assert!(
        (letter_a.rendered_advance - expected).abs() < 0.5,
        "'A' rendered_advance ({:.3}) should ≈ advance_width + Tc = {expected:.3}",
        letter_a.rendered_advance,
    );
    // 'B' should likewise have rendered_advance ≈ advance_width + Tc
    let expected_b = letter_b.advance_width + tc;
    assert!(
        (letter_b.rendered_advance - expected_b).abs() < 0.5,
        "'B' rendered_advance ({:.3}) should ≈ advance_width + Tc = {expected_b:.3}",
        letter_b.rendered_advance,
    );
}

/// Horizontal scaling (Tz) must multiply through both the glyph-width term
/// and the Tc term.  With `Tz = 50` (Th = 0.5) and Tc = 4, the extra
/// spacing over advance_width must be Tc × Th = 2.0, not Tc = 4.0.
#[test]
fn horizontal_scaling_multiplies_both_glyph_and_tc_terms() {
    let tc = 4.0_f32;
    let content = format!("BT /F0 12 Tf 1 0 0 1 100 700 Tm {tc} Tc 50 Tz (A) Tj ET\n");

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
    push(&mut out, &mut offsets, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");
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

    let letter = chars.iter().find(|c| c.char == 'A').expect("'A' not found");

    // With Th=0.5, the Tc contribution to rendered_advance is Tc * Th = 4 * 0.5 = 2.0.
    // If Th were ignored for the Tc term, the delta would be 4.0 instead.
    let th = 0.5_f32;
    let expected_delta = tc * th; // 2.0
    let actual_delta = letter.rendered_advance - letter.advance_width;
    assert!(
        (actual_delta - expected_delta).abs() < 0.3,
        "rendered_advance - advance_width = {actual_delta:.3}; expected Tc×Th = {expected_delta:.3} \
         (Tc={tc}, Th={th}); if Th were not applied to Tc the delta would be {tc:.1}",
    );
}

/// With a flipped CTM (a = -1), rendered_advance must still be positive.
/// The extractor uses `combined_char.a.abs()` to convert tx to device space,
/// so a negative `a` component must not negate the advance.
#[test]
fn flipped_ctm_rendered_advance_is_positive() {
    // Flip the x-axis: CTM = [-1 0 0 1 300 700].  Text advances leftward in
    // page space, but the glyph cursor advance is still a positive distance.
    let content = "BT /F0 12 Tf 0 0 Td (A) Tj ET\n";
    let page_stream = format!("-1 0 0 1 300 700 cm\n{content}");

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
        &format!("<< /Length {} >>\nstream\n{page_stream}\nendstream", page_stream.len() + 1),
    );
    push(&mut out, &mut offsets, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");
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

    let letter = chars.iter().find(|c| c.char == 'A').expect("'A' not found");
    assert!(
        letter.rendered_advance > 0.0,
        "rendered_advance ({:.3}) must be positive even with a flipped CTM (a=-1)",
        letter.rendered_advance,
    );
    // With no Tc/Tw, rendered_advance should equal advance_width
    assert!(
        (letter.rendered_advance - letter.advance_width).abs() < 0.1,
        "rendered_advance ({:.3}) should equal advance_width ({:.3}) when Tc=0 and CTM is flipped",
        letter.rendered_advance,
        letter.advance_width,
    );
}
