//! Word boundaries in and around rotated text runs.
//!
//! Two tests are sensitive to the writing-axis continuation conjunct this
//! file accompanies: `rotated_paragraph_keeps_words_and_lines` fails without
//! it (successive rotated `Tm`s displaced along the perpendicular batched
//! into one line: `brown foxjumps over`), and
//! `rotated_subscript_formula_stays_contiguous` bounds it from the other
//! side (a sub-glyph perpendicular offset must not split a run). The
//! remaining tests pin boundaries that already held and must not regress
//! while runs batch: word gaps inside one rotated `Tj`, TJ-offset-only gaps,
//! grid cells at a true-axis pitch, a minority rotated run on an upright
//! page, and span extent along the writing axis.

use pdf_oxide::document::PdfDocument;

/// Portrait page, no `/Rotate`. One 90°-rotated run containing three
/// space-separated words drawn as a single `Tj`, plus a second run at 270°.
///
/// Turned clockwise the page reads:
///
/// ```text
/// large-scale two-omics integration      (90°  run at x=200)
/// batch correction metrics               (270° run at x=400)
/// ```
fn multiword_rotated_runs_pdf() -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(b"BT /F1 10 Tf\n");
    // 90°: advances along +y. One Tj, three words, real spaces.
    content.extend_from_slice(b"0 1 -1 0 200 200 Tm (large-scale two-omics integration) Tj\n");
    // 270°: advances along -y. One Tj, three words.
    content.extend_from_slice(b"0 -1 1 0 400 600 Tm (batch correction metrics) Tj\n");
    content.extend_from_slice(b"ET");
    build_minimal_pdf_raw(&content, b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]")
}

/// The producer separates words with **TJ kerning offsets only** — no space
/// glyph is ever drawn (ISO 32000-1:2008 §9.4.3: the number is expressed in
/// thousandths of a unit of text space and is subtracted from the current
/// coordinate). This is how the affected real-world documents draw rotated
/// axis labels, and it is the case where a word boundary exists purely as a
/// gap along the writing axis — invisible to a detector measuring on x.
///
/// Turned clockwise the page reads `large-scale two-omics integration`.
fn tj_offset_separated_rotated_run_pdf() -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(b"BT /F1 10 Tf\n");
    content.extend_from_slice(
        b"0 1 -1 0 200 200 Tm [(large-scale)-333(two-omics)-333(integration)] TJ\n",
    );
    content.extend_from_slice(b"ET");
    build_minimal_pdf_raw(&content, b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]")
}

/// A 3×3 grid of numeric cells, every cell drawn with a 90° text matrix, at a
/// pitch along the true writing axis that is wider than a glyph advance — so a
/// correct extractor separates every cell, while flattened geometry makes
/// consecutive cells overlap on the x-axis and fuses them.
///
/// Turned clockwise:
///
/// ```text
/// 0.11 0.12 0.13
/// 0.21 0.22 0.23
/// 0.31 0.32 0.33
/// ```
fn rotated_numeric_grid_pdf() -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(b"BT /F1 9 Tf\n");
    for (row, x) in [(1, 200), (2, 230), (3, 260)] {
        for (col, y) in [(1, 300), (2, 360), (3, 420)] {
            let cell = format!("0.{row}{col}");
            content.extend_from_slice(format!("0 1 -1 0 {x} {y} Tm ({cell}) Tj\n").as_bytes());
        }
    }
    content.extend_from_slice(b"ET");
    build_minimal_pdf_raw(&content, b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]")
}

/// A rotated multi-line paragraph: three 90°-rotated lines, each a single `Tj`
/// holding several words, stacked along the perpendicular axis.
fn rotated_paragraph_pdf() -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(b"BT /F1 10 Tf\n");
    for (x, text) in [
        (200, "the quick brown fox"),
        (214, "jumps over the lazy"),
        (228, "dog and keeps going"),
    ] {
        content.extend_from_slice(format!("0 1 -1 0 {x} 150 Tm ({text}) Tj\n").as_bytes());
    }
    content.extend_from_slice(b"ET");
    build_minimal_pdf_raw(&content, b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]")
}

/// Upright body text plus a rotated table on the same page, with the upright
/// content in the MAJORITY — so no page-level dominant-rotation vote fires and
/// the rotated run must be handled on its own terms.
fn upright_body_with_rotated_table_pdf() -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(b"BT /F1 11 Tf\n");
    for (i, line) in [
        "The committee reviewed the annual report",
        "and approved the budget for the coming",
        "fiscal year without further amendment to",
        "the schedule agreed in the prior session.",
    ]
    .iter()
    .enumerate()
    {
        let y = 700 - (i as i32) * 16;
        content.extend_from_slice(format!("1 0 0 1 72 {y} Tm ({line}) Tj\n").as_bytes());
    }
    // Minority rotated run: a sideways table caption with several words.
    content.extend_from_slice(b"0 1 -1 0 520 200 Tm (Engine oil capacity quarts) Tj\n");
    content.extend_from_slice(b"ET");
    build_minimal_pdf_raw(&content, b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]")
}

/// Every word of a multi-word rotated run survives as its own token — no
/// neighbouring pair is glued into one.
#[test]
fn multiword_rotated_run_keeps_word_boundaries() {
    let doc = PdfDocument::from_bytes(multiword_rotated_runs_pdf()).expect("parse fixture");
    let words: Vec<String> = doc
        .extract_words(0)
        .expect("extract words")
        .into_iter()
        .map(|w| w.text)
        .collect();

    for expected in [
        "large-scale",
        "two-omics",
        "integration",
        "batch",
        "correction",
        "metrics",
    ] {
        assert!(words.iter().any(|w| w == expected), "word {expected:?} missing from {words:?}");
    }
    for glued in [
        "large-scaletwo-omics",
        "two-omicsintegration",
        "large-scaletwo-omicsintegration",
        "batchcorrection",
        "correctionmetrics",
    ] {
        assert!(!words.iter().any(|w| w == glued), "words glued into {glued:?}: {words:?}");
    }
}

/// The same run's words reach assembled text separated, and in forward order.
#[test]
fn multiword_rotated_run_reads_forward_in_text() {
    let doc = PdfDocument::from_bytes(multiword_rotated_runs_pdf()).expect("parse fixture");
    let text = doc.extract_text(0).expect("extract text");
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");

    assert!(
        flat.contains("large-scale two-omics integration"),
        "90° run not separated/forward in text: {flat:?}"
    );
    assert!(
        flat.contains("batch correction metrics"),
        "270° run not separated/forward in text: {flat:?}"
    );
}

/// Words separated only by a TJ kerning offset inside a rotated run must still
/// come apart — the gap is real, it just lies along the writing axis.
#[test]
fn tj_offset_word_gap_in_rotated_run_splits() {
    let doc =
        PdfDocument::from_bytes(tj_offset_separated_rotated_run_pdf()).expect("parse fixture");
    let words: Vec<String> = doc
        .extract_words(0)
        .expect("extract words")
        .into_iter()
        .map(|w| w.text)
        .collect();

    for expected in ["large-scale", "two-omics", "integration"] {
        assert!(words.iter().any(|w| w == expected), "word {expected:?} missing from {words:?}");
    }
    assert!(
        !words
            .iter()
            .any(|w| w.contains("large-scaletwo-omics") || w.contains("two-omicsintegration")),
        "TJ-offset word gaps in a rotated run were not honoured: {words:?}"
    );
}

/// Adjacent cells of a rotated grid never fuse into one token.
#[test]
fn rotated_grid_cells_do_not_fuse() {
    let doc = PdfDocument::from_bytes(rotated_numeric_grid_pdf()).expect("parse fixture");
    let words: Vec<String> = doc
        .extract_words(0)
        .expect("extract words")
        .into_iter()
        .map(|w| w.text)
        .collect();

    for row in 1..=3 {
        for col in 1..=3 {
            let cell = format!("0.{row}{col}");
            assert!(words.iter().any(|w| w == &cell), "cell {cell:?} missing from {words:?}");
        }
    }
    let longest = words.iter().map(|w| w.chars().count()).max().unwrap_or(0);
    assert!(longest <= 4, "cells fused into a longer token (len {longest}): {words:?}");
}

/// Every cell of a rotated grid survives into assembled text — none is dropped
/// by table-region removal that never re-emits it.
#[test]
fn rotated_grid_cells_survive_assembly() {
    let doc = PdfDocument::from_bytes(rotated_numeric_grid_pdf()).expect("parse fixture");
    let text = doc.extract_text(0).expect("extract text");
    for row in 1..=3 {
        for col in 1..=3 {
            let cell = format!("0.{row}{col}");
            assert!(text.contains(&cell), "cell {cell:?} dropped from assembled text: {text:?}");
        }
    }
}

/// A rotated paragraph keeps its words separated and its lines distinct.
#[test]
fn rotated_paragraph_keeps_words_and_lines() {
    let doc = PdfDocument::from_bytes(rotated_paragraph_pdf()).expect("parse fixture");
    let words: Vec<String> = doc
        .extract_words(0)
        .expect("extract words")
        .into_iter()
        .map(|w| w.text)
        .collect();

    for expected in [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "keeps", "going",
    ] {
        assert!(words.iter().any(|w| w == expected), "word {expected:?} missing from {words:?}");
    }
    let text = doc.extract_text(0).expect("extract text");
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(flat.contains("the quick brown fox"), "rotated line 1 not contiguous: {flat:?}");
}

/// On a page whose MAJORITY is upright, a minority rotated run still gets its
/// own word boundaries — and the upright body is untouched.
#[test]
fn minority_rotated_run_keeps_word_boundaries() {
    let doc =
        PdfDocument::from_bytes(upright_body_with_rotated_table_pdf()).expect("parse fixture");
    let words: Vec<String> = doc
        .extract_words(0)
        .expect("extract words")
        .into_iter()
        .map(|w| w.text)
        .collect();

    for expected in ["Engine", "oil", "capacity", "quarts"] {
        assert!(
            words.iter().any(|w| w == expected),
            "rotated word {expected:?} missing from {words:?}"
        );
    }
    for glued in ["Engineoil", "oilcapacity", "capacityquarts"] {
        assert!(
            !words.iter().any(|w| w == glued),
            "rotated words glued into {glued:?}: {words:?}"
        );
    }

    // The upright majority must read normally and contiguously.
    let text = doc.extract_text(0).expect("extract text");
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(
        flat.contains("The committee reviewed the annual report"),
        "upright body disturbed: {flat:?}"
    );
    assert!(
        flat.contains("the schedule agreed in the prior session."),
        "upright body disturbed: {flat:?}"
    );
}

/// A rotated run's span geometry stays in the run's own frame: `width` is the
/// advance along the writing axis and `height` the glyph height, whatever the
/// rotation. This is the convention the reading-order and word passes read, so
/// this test pins it rather than the page-space extent — reporting a page-space
/// (tall, narrow) box for a vertical run is a separate change that has to move
/// those consumers with it.
#[test]
fn rotated_run_span_extent_follows_the_writing_axis() {
    let doc = PdfDocument::from_bytes(multiword_rotated_runs_pdf()).expect("parse fixture");
    let spans = doc.extract_spans(0).expect("extract spans");

    let rotated: Vec<_> = spans.iter().filter(|s| s.rotation_degrees != 0.0).collect();
    assert!(!rotated.is_empty(), "fixture produced no rotated spans: {spans:?}");
    for s in rotated {
        assert!(
            s.bbox.width > s.bbox.height,
            "rotated span {:?} lost its along-axis advance: {}x{}",
            s.text,
            s.bbox.width,
            s.bbox.height
        );
        // Each run holds three words, so the advance must exceed a single
        // glyph height by a wide margin — a run whose width collapsed to the
        // font size would mean the advance was measured on the wrong axis.
        assert!(
            s.bbox.width > s.bbox.height * 3.0,
            "rotated span {:?} advance {} is too small for its text",
            s.text,
            s.bbox.width
        );
    }
}

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

/// A shallowly rotated (5°) label whose middle glyph is a subscript: `N`, a
/// dropped `2`, `O`, one `Tm`+`Tj` each. The drop is 3pt PERPENDICULAR to the
/// writing axis — inside the axis test's tolerance (0.5 × font size) — while
/// the along-axis steps (8pt, then 14pt from the start) stay inside the
/// `|d|`-scaled raw `f` band the axis test is ANDed with. The rotation must
/// stay shallow: at quadrant angles that band (`d == 0` → 0.5pt) vetoes any
/// along-axis advance before the axis test runs, so its tolerance is
/// reachable there only through the veto path pinned by
/// `rotated_paragraph_keeps_words_and_lines`.
fn rotated_formula_with_subscript_pdf() -> Vec<u8> {
    let mut content = Vec::new();
    // cos 5° = 0.99619, sin 5° = 0.08716; each e/f is
    // start + along·(cos, sin) + drop·(sin, -cos).
    content.extend_from_slice(b"BT /F1 10 Tf\n");
    content.extend_from_slice(b"0.99619 0.08716 -0.08716 0.99619 200 300 Tm (N) Tj\n");
    content.extend_from_slice(b"0.99619 0.08716 -0.08716 0.99619 208.231 297.709 Tm (2) Tj\n");
    content.extend_from_slice(b"0.99619 0.08716 -0.08716 0.99619 213.947 301.22 Tm (O) Tj\n");
    content.extend_from_slice(b"ET");
    build_minimal_pdf_raw(&content, b"/Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]")
}

/// The drop is a baseline shift, not a line break: all three glyphs batch into
/// ONE span. Asserted on span membership — assembled text reads "N2O" whether
/// or not the run split, so it cannot observe the difference. This is a pin,
/// not a revert-red test (a veto-only conjunct admits nothing new); it holds
/// the perpendicular tolerance against tightening.
#[test]
fn rotated_subscript_formula_stays_contiguous() {
    let doc = PdfDocument::from_bytes(rotated_formula_with_subscript_pdf()).expect("parse fixture");
    let spans = doc.extract_spans(0).expect("extract spans");
    assert!(
        spans.iter().any(|s| s
            .text
            .split_whitespace()
            .collect::<String>()
            .contains("N2O")),
        "subscript drop split the rotated run: {spans:?}"
    );
}
