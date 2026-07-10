//! `extract_words` must not fuse a relation sign with a backtracking
//! denominator, nor let the resulting word-topology change fabricate tables.
//!
//! A prior fix taught `extract_text` (via `assemble_text_from_spans`) to
//! break the line between a relation sign and a backtracking fraction
//! denominator (`dx1/dt = …` no longer extracts as `dx1 =dt`). That fix was
//! deliberately scoped to the composed-text path only: `extract_words` has
//! its own, separate post-clustering merge step (in `extract_words_inner`)
//! that joins adjacent words abutting or overlapping on the same line — and
//! it had the identical bug. Its `gap` check had no lower bound, so a word
//! that backtracks far behind the previous word's ORIGIN (not just its end)
//! also satisfies "gap ≤ a small positive number", and because the merge is
//! incremental (the merged word's bbox keeps growing), a chain of such
//! backtracks can collapse an entire displayed equation — and, in the worst
//! observed real-document case, the start of the following sentence — into
//! one word.
//!
//! Fixing that alone is not safe on its own: `extract_tables` (both the
//! internal path `extract_text`/`to_markdown`/`to_html` use, and the public
//! `extract_tables` API) feeds `extract_words` output into the spatial table
//! detector as its word geometry. Changing word topology changes what the
//! detector sees, and its punctuation-based prose-rejection guard
//! (`looks_like_prose_paragraph`) let a fabricated or garbled table through
//! whenever the newly-separated prose had no sentence terminator inside it
//! (a caption, a mid-clause fragment) or held vertically-stacked
//! single-character lines (a misread rotated axis label). Both gaps are
//! closed here with punctuation-independent, shape-based signals; the public
//! `extract_tables` API additionally got the same real-grid/prose filter the
//! internal path already had, since it had none at all.
//!
//! A third, related shape found during corpus validation: the same
//! unbounded-`gap` merge also fires across an ordinary line wrap when the
//! producer emits two consecutive lines at nearly the same y (some PDF
//! generators have sub-1pt baseline drift between lines), since the
//! backtrack guard's `y_diff > 1.0` check doesn't catch it. Guarded
//! separately by rejecting any merge whose `delta_x` backs up more than 5
//! font-sizes regardless of `y_diff` — no genuine same-line construct
//! backtracks that far.
//!
//! Verified empirically against real documents (not committed — this repo's
//! fixture policy keeps third-party PDFs out of the tree; fetch instructions
//! are in each opt-in test below): a displayed-math-heavy arXiv page, a
//! small Apache PDFBox regression PDF, and a PubMed Central article. All
//! three fabricated or garbled a table when the word-layer fix landed alone;
//! none does with the detector hardening also in place.

use pdf_oxide::document::PdfDocument;
use std::path::Path;

/// Same geometry as the composed-text fixture: numerator `dx1` above a
/// vinculum, relation `=` at mid-height, denominator `dt` drawn AFTER it,
/// starting behind the `=` origin at a lower baseline. On `main` this fuses
/// into one `extract_words` token containing `"=dt"`.
fn display_fraction_pdf() -> Vec<u8> {
    let mut content = Vec::new();
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
fn relation_sign_stays_separate_in_extract_words() {
    let doc = PdfDocument::from_bytes(display_fraction_pdf()).expect("parse");
    let words = doc.extract_words(0).expect("words");
    let fused: Vec<&str> = words
        .iter()
        .map(|w| w.text.as_str())
        .filter(|t| t.contains("=dt"))
        .collect();
    assert!(
        fused.is_empty(),
        "extract_words must not fuse the relation sign with the backtracking \
         denominator into one token, got fused tokens: {fused:?} (all words: {:?})",
        words.iter().map(|w| &w.text).collect::<Vec<_>>()
    );
}

/// Genuine same-line, tightly-kerned neighbours (no baseline backtrack) must
/// still merge — this is the ordinary-adjacent-glyph-run feature the merge
/// step exists for (tagged CJK documents split typographically-adjacent
/// glyphs across marked-content runs). The backtrack guard must not
/// over-trigger on this shape: same baseline (`y_diff` ≈ 0), small negative
/// gap from a genuine kerning overlap, not a multi-em backtrack.
#[test]
fn tight_kerning_neighbours_still_merge() {
    let mut content = Vec::new();
    content.extend_from_slice(b"BT\n");
    content.extend_from_slice(b"/F1 12 Tf 1 0 0 1 100.00 700.00 Tm (Q) Tj\n");
    // 0.18pt overlap: ordinary tight kerning, not a math backtrack.
    content.extend_from_slice(b"/F1 12 Tf 1 0 0 1 109.82 700.00 Tm (mark) Tj\n");
    content.extend_from_slice(b"ET");
    let pdf = build_minimal_pdf_raw(&content, b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]");
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let words = doc.extract_words(0).expect("words");
    let joined: String = words
        .iter()
        .map(|w| w.text.as_str())
        .collect::<Vec<_>>()
        .join("|");
    assert!(
        joined.contains("Qmark"),
        "ordinary tightly-kerned same-line neighbours must still merge, got words: {joined:?}"
    );
}

/// A unit price backtracking into a quantity column (`$0.14` then `50,170`
/// drawn starting behind the price's origin) is the same backtrack geometry
/// as the math case, just with digits instead of a relation sign. `main`
/// fuses this into one word/cell (`$0.1450,170`); after the fix the two
/// values are separate. This is an intentional behaviour change — a whole-
/// document diff against `main` will show it, and that is correct, not a
/// regression to revert.
#[test]
fn backtracking_price_and_quantity_split() {
    let mut content = Vec::new();
    content.extend_from_slice(b"BT\n");
    content.extend_from_slice(b"/F1 10 Tf 1 0 0 1 200.00 500.00 Tm ($0.14) Tj\n");
    content.extend_from_slice(b"/F1 10 Tf 1 0 0 1 160.00 494.00 Tm (50,170) Tj\n");
    content.extend_from_slice(b"ET");
    let pdf = build_minimal_pdf_raw(&content, b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]");
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let words = doc.extract_words(0).expect("words");
    let fused = words.iter().any(|w| w.text.contains("0.1450"));
    assert!(
        !fused,
        "backtracking price/quantity pair must split into separate words, got: {:?}",
        words.iter().map(|w| &w.text).collect::<Vec<_>>()
    );
}

/// A line wrap whose two lines happen to sit at nearly the same y (some
/// producers emit sub-1pt baseline drift between consecutive lines — the
/// `y_diff > 1.0` half of the math-backtrack guard doesn't catch this) must
/// still not fuse the wrapped line's tail onto the next line's head. The
/// line's end (far right) and the next line's start (far left, ~35 em back)
/// is an order of magnitude beyond any genuine same-line construct (ordinary
/// kerning is near 0; a fraction backtrack is ~1-2 em) and can only be two
/// different lines. Reproduces a real `main` regression: "of whom" (end of
/// one line) fusing onto "tered with books" (start of the next) into
/// "whomteredwithbooks".
#[test]
fn line_wrap_with_near_zero_y_delta_does_not_fuse() {
    let mut content = Vec::new();
    content.extend_from_slice(b"BT\n");
    content.extend_from_slice(b"/F1 10 Tf 1 0 0 1 361.08 600.76 Tm (of whom) Tj\n");
    content.extend_from_slice(b"/F1 10 Tf 1 0 0 1 36.48 600.08 Tm (tered with books) Tj\n");
    content.extend_from_slice(b"ET");
    let pdf = build_minimal_pdf_raw(&content, b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]");
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let words = doc.extract_words(0).expect("words");
    let fused = words.iter().any(|w| w.text.contains("whomtered"));
    assert!(
        !fused,
        "a wrapped line must not fuse onto the next line's start even when \
         y_diff is under 1pt, got: {:?}",
        words.iter().map(|w| &w.text).collect::<Vec<_>>()
    );
}

/// Opt-in real-document guard. Fetch:
/// `curl -sL -o tests/fixtures/real/2503.09472.pdf https://arxiv.org/pdf/2503.09472v1`.
/// Page index 14 (1-based page 15) is the densest fraction layout in the
/// paper; on `main` its `extract_words` output contains fused tokens
/// including one that swallows the start of the following sentence.
#[test]
fn real_arxiv_page_has_no_fused_math_words_or_fabricated_tables() {
    let p = "tests/fixtures/real/2503.09472.pdf";
    if !Path::new(p).exists() {
        eprintln!("[word-fusion] fixture missing, skipping: {p}");
        return;
    }
    let doc = PdfDocument::from_bytes(std::fs::read(p).expect("read")).expect("parse");
    let words = doc.extract_words(14).expect("words");
    let bad: Vec<&str> = words
        .iter()
        .map(|w| w.text.as_str())
        .filter(|t| t.contains('=') && t.chars().count() > 3)
        .collect();
    assert!(bad.is_empty(), "no word should contain a fused '=' token, got: {bad:?}");
    let longest = words
        .iter()
        .map(|w| w.text.chars().count())
        .max()
        .unwrap_or(0);
    assert!(
        longest < 40,
        "no word should exceed ordinary token length, longest was {longest} chars"
    );

    let text = doc.extract_text(14).expect("text");
    assert_eq!(text.matches("=dt").count(), 0, "extract_text must stay free of '=dt' fusion");

    let tables = doc.extract_tables(14).expect("tables");
    assert!(
        tables.is_empty(),
        "no table should be detected on this prose+math page, got {} tables",
        tables.len()
    );
}

/// Opt-in real-document guard for the table-fabrication risk. Fetch:
/// `curl -sL -o "tests/fixtures/real/PDFBOX-5002.pdf" "https://issues.apache.org/jira/secure/attachment/13014135/small%26Big.pdf"`
/// — a minimal Apache PDFBox regression PDF (single caption sentence, no
/// real table). On `main`, the word-layer fix alone fabricates a 3-row
/// table out of this ordinary wrapped caption; the detector hardening keeps
/// it rejected.
#[test]
fn real_pdfbox_caption_produces_no_fabricated_table() {
    let p = "tests/fixtures/real/PDFBOX-5002.pdf";
    if !Path::new(p).exists() {
        eprintln!("[word-fusion] fixture missing, skipping: {p}");
        return;
    }
    let doc = PdfDocument::from_bytes(std::fs::read(p).expect("read")).expect("parse");
    let tables = doc.extract_tables(0).expect("tables");
    assert!(
        tables.is_empty(),
        "an ordinary wrapped caption must not be detected as a table, got {} tables: {:?}",
        tables.len(),
        tables.iter().map(|t| &t.rows).collect::<Vec<_>>()
    );
}

/// Opt-in real-document guard for the second table-fabrication shape: a
/// rotated axis label whose glyphs are drawn one per line, which reads as a
/// cell of mostly single-character lines. Fetch:
/// `curl -sL -o tests/fixtures/real/pmc8129076.pdf https://europepmc.org/articles/PMC8129076?pdf=render`.
#[test]
fn real_pmc_article_produces_no_rotated_label_table() {
    let p = "tests/fixtures/real/pmc8129076.pdf";
    if !Path::new(p).exists() {
        eprintln!("[word-fusion] fixture missing, skipping: {p}");
        return;
    }
    let doc = PdfDocument::from_bytes(std::fs::read(p).expect("read")).expect("parse");
    let tables = doc.extract_tables(1).expect("tables");
    for t in &tables {
        for row in &t.rows {
            for cell in &row.cells {
                let lines: Vec<&str> = cell
                    .text
                    .split('\n')
                    .map(str::trim)
                    .filter(|l| !l.is_empty())
                    .collect();
                if lines.len() >= 3 {
                    let single = lines.iter().filter(|l| l.chars().count() == 1).count();
                    assert!(
                        (single as f32 / lines.len() as f32) <= 0.5,
                        "a rotated-label-shaped cell must not survive as a table cell, got: {:?}",
                        cell.text
                    );
                }
            }
        }
    }
}

/// The emitter's backtrack branch is gated OFF for right-to-left runs, and
/// the word-merge guard mirrors that gating. Opt-in real-document guard.
/// Fetch: `curl -sL -o tests/fixtures/real/wiki_cat_ar.pdf
/// https://ar.wikipedia.org/api/rest_v1/page/pdf/%D9%82%D8%B7`.
#[test]
fn rtl_words_are_not_broken_by_the_backtrack_guard() {
    let p = "tests/fixtures/real/wiki_cat_ar.pdf";
    if !Path::new(p).exists() {
        eprintln!("[word-fusion] RTL fixture missing, skipping: {p}");
        return;
    }
    let doc = PdfDocument::from_bytes(std::fs::read(p).expect("read")).expect("parse");
    let words = doc.extract_words(0).expect("words");
    let arabic_chars: usize = words
        .iter()
        .flat_map(|w| w.text.chars())
        .filter(|c| ('\u{0600}'..='\u{06FF}').contains(c))
        .count();
    assert!(
        arabic_chars > 100,
        "Arabic content must survive word extraction intact, saw {arabic_chars} Arabic chars"
    );
}

// ---------------------------------------------------------------------------
// Minimal raw PDF builder (same pattern as test_display_math_word_fusion.rs)
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
