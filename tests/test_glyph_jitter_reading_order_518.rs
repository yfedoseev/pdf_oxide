//! Regression: per-glyph `Tm`+`Tj` with sinusoidal baseline jitter
//! must still extract in left-to-right reading order (#518).
//!
//! Microsoft Word emits broken-image placeholder text as one
//! `BT Tm Tj ET` block per glyph with ±2.5–5pt sinusoidal Y-jitter
//! around the baseline. The old `Tm`-run merge tolerated only ±0.5pt
//! of Y delta, so jittered glyphs became separate Y-banded
//! `TextSpan`s and the reading-order sort emitted them top-to-bottom
//! (e.g. `"Hello"` → `"elH l o"`).
//!
//! ISO 32000-1:2008 §9.4 does not define logical reading order — the
//! extractor reconstructs it. A baseline delta far below the glyph
//! height is the same visual line; only a delta on the order of the
//! font size is a real line break. The fix makes the merge tolerance
//! scale-relative (0.5× glyph height). This suite pins both halves of
//! the invariant: jitter must merge, real line breaks must NOT.

use pdf_oxide::document::PdfDocument;

/// Minimal single-page PDF, one `BT … ET` block per glyph, each with
/// an absolute `Tm`. `y_of(i)` supplies the per-glyph baseline.
fn make_per_glyph_pdf(text: &str, x_step: f64, y_of: impl Fn(usize) -> f64) -> Vec<u8> {
    const X_START: f64 = 72.0;

    let mut stream = String::new();
    for (i, ch) in text.chars().enumerate() {
        let x = X_START + i as f64 * x_step;
        let y = y_of(i);
        stream.push_str(&format!("BT /F1 12 Tf 1 0 0 1 {x:.2} {y:.4} Tm ({ch}) Tj ET\n"));
    }

    let mut pdf: Vec<u8> = Vec::new();
    macro_rules! push {
        ($s:expr) => {
            pdf.extend_from_slice($s.as_bytes())
        };
    }

    push!("%PDF-1.4\n");
    let off1 = pdf.len();
    push!("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    let off2 = pdf.len();
    push!("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let off3 = pdf.len();
    push!(
        "3 0 obj\n\
         << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
            /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\n\
         endobj\n"
    );
    let off4 = pdf.len();
    let stream_bytes = stream.as_bytes();
    push!(format!("4 0 obj\n<< /Length {} >>\nstream\n", stream_bytes.len()));
    pdf.extend_from_slice(stream_bytes);
    push!("\nendstream\nendobj\n");
    let off5 = pdf.len();
    push!("5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n");
    let xref_off = pdf.len();
    push!(format!(
        "xref\n0 6\n\
         0000000000 65535 f \r\n\
         {off1:010} 00000 n \r\n\
         {off2:010} 00000 n \r\n\
         {off3:010} 00000 n \r\n\
         {off4:010} 00000 n \r\n\
         {off5:010} 00000 n \r\n"
    ));
    push!(format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_off}\n%%EOF\n"));
    pdf
}

/// The exact reproduction from issue #518: `"Hello"`, 3.5pt amplitude,
/// period 6 sinusoidal Y-jitter. Must extract in reading order.
#[test]
fn glyph_jitter_word_placeholder_reading_order_518() {
    const JITTER_PT: f64 = 3.5;
    const JITTER_PERIOD: usize = 6;
    const Y_BASE: f64 = 700.0;

    let pdf = make_per_glyph_pdf("Hello", 12.0, |i| {
        let angle = std::f64::consts::TAU * i as f64 / JITTER_PERIOD as f64;
        Y_BASE + angle.sin() * JITTER_PT
    });
    let doc = PdfDocument::from_bytes(pdf).unwrap();
    let text = doc.extract_text(0).unwrap();

    assert_eq!(
        text.split_whitespace().collect::<Vec<_>>().join(""),
        "Hello",
        "jittered per-glyph Tm+Tj must extract left-to-right; got: {text:?}"
    );
}

/// Larger amplitude (5pt — the upper end of the Word range) still on a
/// single logical line at 12pt font must also stay in reading order.
#[test]
fn glyph_jitter_max_amplitude_reading_order_518() {
    let pdf = make_per_glyph_pdf("Reading", 11.0, |i| 700.0 + if i % 2 == 0 { 0.0 } else { 5.0 });
    let doc = PdfDocument::from_bytes(pdf).unwrap();
    let text = doc.extract_text(0).unwrap();
    assert_eq!(
        text.split_whitespace().collect::<Vec<_>>().join(""),
        "Reading",
        "5pt jitter at 12pt font is one visual line; got: {text:?}"
    );
}

/// Anti-regression: a genuine two-line layout (12pt font, ~14.4pt
/// leading — well above the scale-relative tolerance) must STILL be
/// split into two lines, not over-merged by the widened tolerance.
///
/// Line 2 (`CD`, y≈685.6) is emitted in the content stream BEFORE
/// line 1 (`AB`, y=700) on purpose: stream order is reverse of
/// reading order, so the assertion genuinely distinguishes the two
/// outcomes. Correct (tolerance splits the 14.4pt gap) → two spans
/// the reading-order sort reorders top-to-bottom → `"ABCD"`. A
/// regressed/too-wide tolerance would merge all four into one
/// stream-order span → `"CDAB"` and the test fails.
#[test]
fn genuine_line_break_still_splits_not_overmerged_518() {
    let glyphs: [(f64, f64, char); 4] = [
        (72.0, 685.6, 'C'), // line 2, emitted first
        (84.0, 685.6, 'D'),
        (72.0, 700.0, 'A'), // line 1, emitted second
        (84.0, 700.0, 'B'),
    ];
    let mut stream = String::new();
    for (x, y, ch) in glyphs {
        stream.push_str(&format!("BT /F1 12 Tf 1 0 0 1 {x:.2} {y:.4} Tm ({ch}) Tj ET\n"));
    }
    let two_line = build_two_line_pdf(&stream);
    let doc = PdfDocument::from_bytes(two_line).unwrap();
    let text = doc.extract_text(0).unwrap();
    let joined = text.split_whitespace().collect::<Vec<_>>().join("");
    assert_eq!(
        joined, "ABCD",
        "14.4pt leading must split into two reading-ordered lines, \
         not over-merge into stream-order 'CDAB' (got: {text:?})"
    );
}

fn build_two_line_pdf(stream: &str) -> Vec<u8> {
    let mut pdf: Vec<u8> = Vec::new();
    macro_rules! push {
        ($s:expr) => {
            pdf.extend_from_slice($s.as_bytes())
        };
    }
    push!("%PDF-1.4\n");
    let off1 = pdf.len();
    push!("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    let off2 = pdf.len();
    push!("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let off3 = pdf.len();
    push!(
        "3 0 obj\n\
         << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
            /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\n\
         endobj\n"
    );
    let off4 = pdf.len();
    let sb = stream.as_bytes();
    push!(format!("4 0 obj\n<< /Length {} >>\nstream\n", sb.len()));
    pdf.extend_from_slice(sb);
    push!("\nendstream\nendobj\n");
    let off5 = pdf.len();
    push!("5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n");
    let xref_off = pdf.len();
    push!(format!(
        "xref\n0 6\n\
         0000000000 65535 f \r\n\
         {off1:010} 00000 n \r\n\
         {off2:010} 00000 n \r\n\
         {off3:010} 00000 n \r\n\
         {off4:010} 00000 n \r\n\
         {off5:010} 00000 n \r\n"
    ));
    push!(format!("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_off}\n%%EOF\n"));
    pdf
}
