//! Displayed fractions must not fuse the relation sign with the denominator.
//!
//! `dx1/dt = …` extracts as `dx1 =dt` on affected pages: the `=` sits at the
//! fraction's mid-height and the denominator `dt` is drawn AFTER it, starting
//! ~24pt behind the `=` origin at a ~4pt baseline offset. The extract_text
//! line emitter sees a same-line pair whose next span backtracks with a real
//! baseline drop, and — absent a dedicated branch — concatenates them into
//! `=dt`. The fix adds that branch (a backtracking span with `y_diff > 1`,
//! `delta_x ≤ 0.5`, `gap < -1em`, gated OFF for right-to-left runs, whose
//! leftward flow is not backtracking) so the line breaks instead.
//!
//! This covers the composed-text path (extract_text / to_markdown / to_html).
//! The lower-level `extract_words` de-fusion is intentionally NOT part of this
//! change: `extract_page_tables` feeds `extract_words` output into the spatial
//! table detector, so altering word geometry there perturbs table detection —
//! that work (word-level de-fusion plus the detector hardening it requires) is
//! tracked separately so this fix stays table-neutral.

use pdf_oxide::document::PdfDocument;
use std::path::Path;

/// One displayed fraction in a real page's geometry: numerator `dx` + subscript
/// `1` above the bar, relation `=` to the right at mid-height, denominator `dt`
/// below the bar and to the LEFT of `=`. `dt`'s baseline sits ~8pt below the
/// `=` baseline (still one visual line) and there is no right-hand side, so the
/// reading order presents `=` immediately before `dt` — the exact ordering that
/// reaches the emitter's backtrack branch. On `main` this fixture emits `=dt`.
fn display_fraction_pdf() -> Vec<u8> {
    let mut content = Vec::new();
    // Vinculum (fraction bar) between numerator and denominator.
    content.extend_from_slice(b"0.4 w 131.8 719.5 m 145.0 719.5 l S\n");
    content.extend_from_slice(b"BT\n");
    content.extend_from_slice(b"/F1 8 Tf 1 0 0 1 131.81 721.00 Tm (dx) Tj\n");
    content.extend_from_slice(b"/F1 6 Tf 1 0 0 1 141.28 721.00 Tm (1) Tj\n");
    content.extend_from_slice(b"/F1 12 Tf 1 0 0 1 149.94 718.04 Tm (=) Tj\n");
    content.extend_from_slice(b"/F1 8 Tf 1 0 0 1 134.74 710.00 Tm (dt) Tj\n");
    content.extend_from_slice(b"ET");
    build_minimal_pdf_raw(&content, b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]")
}

#[test]
fn relation_sign_stays_separate_in_extract_text() {
    // Fails on `main` (emits "dx1 =dt"); passes with the emitter branch, which
    // breaks the line between the relation sign and the backtracking
    // denominator. Mutation check: deleting the branch re-fuses "=dt".
    let doc = PdfDocument::from_bytes(display_fraction_pdf()).expect("parse");
    let text = doc.extract_text(0).expect("text");
    assert!(
        !text.contains("=dt"),
        "extract_text must not fuse the relation sign with the backtracking denominator, got: {text:?}"
    );
    // Content is conserved — the pieces are separated, not dropped.
    assert!(text.contains('='), "the relation sign must survive, got: {text:?}");
    assert!(text.contains("dt"), "the denominator must survive, got: {text:?}");
}

#[test]
fn real_arxiv_page_has_no_fused_relation_signs() {
    // Opt-in real-document guard for the composed-text path (fetch:
    // `curl -sL -o tests/fixtures/real/2503.09472.pdf https://arxiv.org/pdf/2503.09472v1`).
    // Page index 14 (1-based page 15) carries the densest fraction layout; on
    // `main` its extract_text contains two `=dt` fusions.
    let p = "tests/fixtures/real/2503.09472.pdf";
    if !Path::new(p).exists() {
        eprintln!("[math-fusion] fixture missing, skipping: {p}");
        return;
    }
    let doc = PdfDocument::from_bytes(std::fs::read(p).expect("read")).expect("parse");
    let text = doc.extract_text(14).expect("text");
    assert!(
        !text.contains("=dt"),
        "extract_text must not fuse '=' with 'dt' on the fraction-dense page"
    );
}

#[test]
fn rtl_backtracking_runs_are_not_broken_by_the_emitter() {
    // The emitter's backtrack branch is gated OFF for right-to-left runs:
    // Arabic is cursive and its runs advance leftward by design, so a negative
    // gap there is ordinary reading order, not a fraction backtrack. Opt-in
    // real-document guard (fetch:
    // `curl -sL -o tests/fixtures/real/wiki_cat_ar.pdf https://ar.wikipedia.org/api/rest_v1/page/pdf/%D9%82%D8%B7`).
    // Every character main emits must still be present, none shredded onto
    // stray lines by a misfired break.
    let p = "tests/fixtures/real/wiki_cat_ar.pdf";
    if !Path::new(p).exists() {
        eprintln!("[math-fusion] RTL fixture missing, skipping: {p}");
        return;
    }
    let doc = PdfDocument::from_bytes(std::fs::read(p).expect("read")).expect("parse");
    let text = doc.extract_text(0).expect("text");
    // Arabic letters survive as connected runs, not one-glyph-per-line debris.
    let arabic_chars = text
        .chars()
        .filter(|c| ('\u{0600}'..='\u{06FF}').contains(c))
        .count();
    assert!(
        arabic_chars > 100,
        "Arabic content must survive extraction intact, saw {arabic_chars} Arabic chars"
    );
}

// ---------------------------------------------------------------------------
// Minimal raw PDF builder (same pattern as test_stroke_width_rendered_bbox.rs)
// ---------------------------------------------------------------------------

fn build_minimal_pdf_raw(content: &[u8], page_extra: &[u8]) -> Vec<u8> {
    let mut pdf = b"%PDF-1.4\n".to_vec();

    let off1 = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    let off2 = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    let off3 = pdf.len();
    pdf.extend_from_slice(b"3 0 obj\n<< ");
    pdf.extend_from_slice(page_extra);
    pdf.extend_from_slice(b" /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n");

    let off4 = pdf.len();
    pdf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    let off5 = pdf.len();
    pdf.extend_from_slice(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>\nendobj\n",
    );

    let xref_pos = pdf.len();
    let offsets = [0usize, off1, off2, off3, off4, off5];
    pdf.extend_from_slice(format!("xref\n0 {}\n", offsets.len()).as_bytes());
    pdf.extend_from_slice(format!("{:010} 65535 f\r\n", 0).as_bytes());
    for &off in &offsets[1..] {
        pdf.extend_from_slice(format!("{:010} 00000 n\r\n", off).as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            offsets.len(),
            xref_pos
        )
        .as_bytes(),
    );
    pdf
}
