//! Reading-order parity tests on external PDF fixtures.
//!
//! Every test is a standing regression guard for the canonical
//! reading-order pipeline (struct tree on tagged PDFs, geometric fallback
//! otherwise). A failure means the pipeline has drifted.
//!
//! Fixtures live in the external pdf_oxide_tests corpus under
//! `pdfs_issue_regression/`. Tests skip gracefully when the corpus is
//! not present.

use pdf_oxide::document::PdfDocument;
use std::path::PathBuf;

fn fixture(name: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let path = PathBuf::from(home)
        .join("projects/pdf_oxide_tests/pdfs_issue_regression")
        .join(name);
    if !path.exists() {
        eprintln!("Skipping: {} not found", path.display());
        return None;
    }
    Some(path)
}

fn open_fixture(name: &str) -> Option<PdfDocument> {
    let path = fixture(name)?;
    let bytes = std::fs::read(&path).expect("fixture readable");
    Some(PdfDocument::from_bytes(bytes).expect("fixture parses"))
}

fn assert_monotonic_line_y(lines: &[pdf_oxide::layout::TextLine]) {
    let mut prev_y = f32::INFINITY;
    for (i, line) in lines.iter().enumerate() {
        let y = line.bbox.y;
        assert!(
            y <= prev_y + 0.5,
            "lines not monotonic at index {}: y={} after prev_y={}, text={:?}",
            i,
            y,
            prev_y,
            line.text
        );
        prev_y = y;
    }
}

// ── PDF #1: pdf_structure.pdf — Lorem-ipsum demo ─────────────────────────────
//
// Original report symptom (0.3.14): "Words and text-lines are extracted starting
// from the bottom of the page, and the word order within lines is incorrect."
// First words / lines are correct on 0.3.42 — the lower table area still
// breaks monotonic ordering because XY-Cut walks each column separately.

#[test]
fn test_211_pdf_structure_first_words_in_order() {
    let Some(doc) = open_fixture("issue_211_pdf_structure.pdf") else {
        return;
    };
    let words = doc.extract_words(0).expect("extract_words succeeds");
    assert!(!words.is_empty(), "must extract at least one word");
    assert_eq!(words[0].text, "Titre", "first word should be 'Titre du document'");
    assert_eq!(words[1].text, "du");
    assert_eq!(words[2].text, "document");
}

#[test]
fn test_211_pdf_structure_lines_monotonic_y() {
    let Some(doc) = open_fixture("issue_211_pdf_structure.pdf") else {
        return;
    };
    let lines = doc
        .extract_text_lines(0)
        .expect("extract_text_lines succeeds");
    assert!(lines.len() >= 20, "should extract ~22 lines");
    assert_monotonic_line_y(&lines);
}

// ── PDF #2: municipal_minutes — centered title above body ────────────────────
//
// Bug: extract_words and extract_text_lines silently move the document's
// "COMITÉ DE DÉMOLITION" and "PROCÈS-VERBAL" headings out of position, even
// though extract_spans/extract_chars/extract_text return them. Root cause:
// XYCutStrategy::partition_region produces blocks in a non-y-monotonic order.

#[test]
fn test_211_municipal_minutes_first_word_is_comite() {
    let Some(doc) = open_fixture("issue_211_municipal_minutes.pdf") else {
        return;
    };
    let words = doc.extract_words(0).expect("extract_words succeeds");
    assert_eq!(
        words[0].text,
        "COMITÉ",
        "first word should be 'COMITÉ' from the document title; got {:?} (full prefix: {:?})",
        words.first().map(|w| &w.text),
        words
            .iter()
            .take(8)
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>(),
    );
}

#[test]
fn test_211_municipal_minutes_first_line_is_title() {
    let Some(doc) = open_fixture("issue_211_municipal_minutes.pdf") else {
        return;
    };
    let lines = doc
        .extract_text_lines(0)
        .expect("extract_text_lines succeeds");
    assert_eq!(
        lines[0].text,
        "COMITÉ DE DÉMOLITION",
        "first line should be the title; got {:?} (full prefix: {:?})",
        lines.first().map(|l| &l.text),
        lines
            .iter()
            .take(5)
            .map(|l| l.text.as_str())
            .collect::<Vec<_>>(),
    );
}

#[test]
fn test_211_municipal_minutes_lines_monotonic_y() {
    let Some(doc) = open_fixture("issue_211_municipal_minutes.pdf") else {
        return;
    };
    let lines = doc
        .extract_text_lines(0)
        .expect("extract_text_lines succeeds");
    assert_monotonic_line_y(&lines);
}

#[test]
fn test_211_municipal_minutes_spans_contain_title() {
    // extract_spans currently DOES include the title in correct order — this
    // is a regression guard for the working path.
    let Some(doc) = open_fixture("issue_211_municipal_minutes.pdf") else {
        return;
    };
    let spans = doc.extract_spans(0).expect("extract_spans succeeds");
    let joined: String = spans
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(joined.contains("COMITÉ DE DÉMOLITION"), "spans must contain the document title");
    assert!(joined.contains("PROCÈS-VERBAL"), "spans must contain the subtitle");
    // Title must come before the body in span order
    let title_pos = joined.find("COMITÉ DE DÉMOLITION").unwrap();
    let body_pos = joined.find("Séance publique").unwrap();
    assert!(title_pos < body_pos, "title must precede body in extract_spans output");
}

// ── PDF #3: government_form — form-style label/value layout ──────────────────
//
// Bug: XY-Cut detects phantom columns at the form's label/value gutter and
// splits prose lines that span both halves. Line[1] gets only the left
// half, and the right half reappears as line[5] AFTER lines at much lower y.

#[test]
fn test_211_government_form_prose_line_not_split() {
    // The sentence "Reports submitted to the Division of Safety and Permanence
    // (DSP) that do not include..." is split across two PDF spans on the SAME y
    // (both at y=735.72). The user-visible bug from #211 is that they end up
    // as two separate, non-adjacent lines. Fix: both pieces must land on the
    // SAME extracted line — the "(DSP" prefix and the "that do not include"
    // continuation. Ignores any small TextLine joiner-whitespace artifact
    // between the pieces ("DSP )" vs "DSP)") which is tracked separately.
    let Some(doc) = open_fixture("issue_211_government_form.pdf") else {
        return;
    };
    let lines = doc
        .extract_text_lines(0)
        .expect("extract_text_lines succeeds");
    let prefix = "Reports submitted to the Division of Safety and Permanence";
    let suffix = "that do not include all of the required information";
    let prose_line = lines
        .iter()
        .find(|l| l.text.contains(prefix))
        .unwrap_or_else(|| {
            panic!(
                "no line contains the prose prefix; lines:\n{}",
                lines
                    .iter()
                    .map(|l| l.text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
        });
    assert!(
        prose_line.text.contains(suffix),
        "prefix and continuation must be on the same line; got:\n{}",
        prose_line.text,
    );
}

#[test]
fn test_211_government_form_lines_monotonic_y() {
    let Some(doc) = open_fixture("issue_211_government_form.pdf") else {
        return;
    };
    let lines = doc
        .extract_text_lines(0)
        .expect("extract_text_lines succeeds");
    assert_monotonic_line_y(&lines);
}

// ── Cross-API content parity ─────────────────────────────────────────────────
//
// extract_words must place tokens in the same reading order as extract_spans.
// Currently fails on PDF #2 because XY-Cut moves the title tokens to the
// middle of the words list (COMITÉ at index 69, PROCÈS-VERBAL at index 221).

#[test]
fn test_211_municipal_minutes_words_match_span_order() {
    let Some(doc) = open_fixture("issue_211_municipal_minutes.pdf") else {
        return;
    };
    let words = doc.extract_words(0).expect("extract_words succeeds");
    let comite_idx = words
        .iter()
        .position(|w| w.text == "COMITÉ")
        .expect("COMITÉ must be in extract_words");
    let seance_idx = words
        .iter()
        .position(|w| w.text == "Séance")
        .expect("Séance must be in extract_words");
    assert!(
        comite_idx < seance_idx,
        "COMITÉ (title, y≈871) must precede Séance (body, y≈827) in words; got COMITÉ@{} Séance@{}",
        comite_idx,
        seance_idx
    );
}
