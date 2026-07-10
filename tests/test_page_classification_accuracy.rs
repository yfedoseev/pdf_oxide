//! `classify_page` must not call a born-digital page `Scanned` when it
//! carries real, usable, extractable text — `pages_needing_ocr` is meant to
//! be safe to act on, and OCR-ing a page that already has good native text
//! replaces it with worse output.
//!
//! Two independent causes, both in the coverage/text-quality decision path:
//!
//! 1. A full-bleed background image (`image_area_ratio` ~1.0) with a real
//!    text headline drawn over it covers only a few percent of the page by
//!    area. The sparse-coverage branch in `classify_from_signals` correctly
//!    keeps the page `Scanned` on coverage grounds (that judgment call is
//!    out of scope here), but it always reported `ReasonCode::
//!    NoTextLayerPresent` even when the text is real, mapped correctly, and
//!    visible (`usable_text == true`) — telling a caller who inspects the
//!    reason that there is nothing to extract, when there is. Fixed to
//!    report `TextLayerBelowThreshold` when text is actually usable.
//!
//! 2. `gather_page_signals` built its fragmentation-detection input by
//!    joining raw content-stream spans with a forced space after every one.
//!    Math typesetting draws each atom (a parenthesis, an operator, a
//!    subscript) as its own span, so a page of ordinary dense LaTeX text
//!    gets treated as a wall of one- and two-character "words", the
//!    fragmented-word ratio spikes, and the text-quality gate — fed via
//!    `text_quality_gate`, which does its own independent word-split on the
//!    same raw text — routes a real text page to OCR. Both places now
//!    build their word list from `extract_words` (the same glyph/span
//!    clustering `extract_text` relies on) instead of raw span punctuation.
//!
//! Switching to `extract_words` clustering (cause 2's fix) surfaced a third
//! issue during corpus validation: CJK/Hangul scripts have no inter-word
//! spaces, so glyph-adjacency clustering naturally produces short (often
//! 1-3 character) "words" — real, correct text, not fragmentation. The
//! `frag`/`avg_word_len` checks (in both `text_quality_gate` and the
//! `fragmented_word_ratio` feeding `classify_from_signals`) are calibrated
//! for space-separated Latin text and misread this as a broken CMap,
//! routing ordinary Japanese/Chinese/Korean pages to `Scanned`. Both now
//! skip that specific check for CJK-dominant text via
//! `is_cjk_dominant_text` (script-agnostic signals — garbled ratio,
//! consecutive-repeat — still apply normally).

use pdf_oxide::document::PdfDocument;
use pdf_oxide::extractors::auto::{PageKind, ReasonCode};

/// A full-bleed background image with a real text headline drawn over it —
/// a slide, a magazine cover, a deck title page. `usable_text` is true (the
/// text is short but clean and visible) and `image_area_ratio` is ~1.0, so
/// coverage-driven `Scanned` is a defensible verdict; the reason must not
/// claim there is no text layer.
#[test]
fn full_bleed_image_with_real_headline_gets_honest_reason() {
    let pdf = full_bleed_slide_pdf();
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let cls = doc.classify_page(0).expect("classify_page");

    // The text is real and must remain fully extractable regardless of
    // how the page gets classified.
    let text = doc.extract_text(0).expect("extract_text");
    assert!(text.contains("Quarterly"), "headline text must still extract, got: {text:?}");

    if matches!(cls.kind, PageKind::Scanned) {
        assert_ne!(
            cls.reason,
            ReasonCode::NoTextLayerPresent,
            "a page with a real, visible, usable text headline must not be reported as having no text layer"
        );
    }
}

/// Ordinary dense text where every "word" is drawn as 2-3 separate,
/// tightly-abutting text-showing operations (no real gap between the
/// pieces) — the same span-per-atom shape displayed math produces, just
/// with plain Latin text so the expected recombined word is unambiguous.
/// Raw span-joining (one forced space per span) fragments each word into
/// its 2-3 pieces; real word-clustering recombines them.
#[test]
fn split_span_words_do_not_trigger_scanned_misclassification() {
    let pdf = fragmented_span_words_pdf();
    let doc = PdfDocument::from_bytes(pdf).expect("parse");
    let cls = doc.classify_page(0).expect("classify_page");

    // `gather_page_signals`/`text_quality_gate` build their word list from
    // `extract_words`, not `extract_text` (a separate emitter) — check the
    // fixture's precondition through the same lens the fix operates on.
    let words = doc.extract_words(0).expect("extract_words");
    assert!(
        words.len() >= 8,
        "fixture must produce enough real words to exercise the gate, got: {:?}",
        words.iter().map(|w| &w.text).collect::<Vec<_>>()
    );

    assert!(
        matches!(cls.kind, PageKind::TextLayer),
        "a page of ordinary text split across tightly-kerned spans (the same shape \
         math typesetting produces) must not be misclassified Scanned, got {:?} \
         (reason {:?}), words: {:?}",
        cls.kind,
        cls.reason,
        words.iter().map(|w| &w.text).collect::<Vec<_>>()
    );
}

/// Opt-in real-document guard for the CJK false-positive: an ordinary
/// Japanese Wikipedia article (about cats), no images, no garbling. Not
/// fetched automatically per this repo's fixture policy — place a copy at
/// the path below to run this locally; it skips cleanly when absent.
#[test]
fn real_japanese_article_is_not_misclassified_scanned() {
    let p = "tests/fixtures/real/wiki_cat_ja.pdf";
    if !std::path::Path::new(p).exists() {
        eprintln!("[classification] CJK fixture missing, skipping: {p}");
        return;
    }
    let doc = PdfDocument::from_bytes(std::fs::read(p).expect("read")).expect("parse");
    let cls = doc.classify_page(0).expect("classify_page");
    assert!(
        matches!(cls.kind, PageKind::TextLayer),
        "ordinary CJK prose (no inter-word spaces, naturally short glyph-clustered \
         tokens) must not be misclassified Scanned, got {:?} (reason {:?})",
        cls.kind,
        cls.reason
    );
}

// ---------------------------------------------------------------------------
// Fixture builders
// ---------------------------------------------------------------------------

fn full_bleed_slide_pdf() -> Vec<u8> {
    let mut pdf = b"%PDF-1.4\n".to_vec();
    let mut off = vec![0usize; 7];

    off[1] = pdf.len();
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    off[2] = pdf.len();
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

    off[3] = pdf.len();
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 720 540] \
         /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> \
         /XObject << /Im0 6 0 R >> >> >>\nendobj\n",
    );

    // Full-bleed image placement, then a short headline drawn on top.
    let content = b"q 720 0 0 540 0 0 cm /Im0 Do Q\n\
                     BT /F1 28 Tf 1 0 0 1 60 400 Tm (Quarterly Results) Tj ET";
    off[4] = pdf.len();
    pdf.extend_from_slice(format!("4 0 obj\n<< /Length {} >>\nstream\n", content.len()).as_bytes());
    pdf.extend_from_slice(content);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    off[5] = pdf.len();
    pdf.extend_from_slice(
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>\nendobj\n",
    );

    // Minimal raw (uncompressed) 1x1 DeviceGray image, scaled to cover the
    // whole page by the `cm` in the content stream above.
    off[6] = pdf.len();
    pdf.extend_from_slice(
        b"6 0 obj\n<< /Type /XObject /Subtype /Image /Width 1 /Height 1 \
         /ColorSpace /DeviceGray /BitsPerComponent 8 /Length 1 >>\nstream\n",
    );
    pdf.extend_from_slice(&[0x80]);
    pdf.extend_from_slice(b"\nendstream\nendobj\n");

    write_xref_trailer(&mut pdf, &off);
    pdf
}

/// 10 words, each split into 2 tightly-abutting Tj pieces (zero gap between
/// the pieces of one word, a real space-sized gap between words).
fn fragmented_span_words_pdf() -> Vec<u8> {
    const PIECES: &[(&str, &str)] = &[
        ("entry", "wise"),
        ("nonneg", "ative"),
        ("matr", "ix"),
        ("writ", "ing"),
        ("sys", "tem"),
        ("equa", "tion"),
        ("vec", "tor"),
        ("solu", "tion"),
        ("prob", "lem"),
        ("theo", "rem"),
    ];
    let mut content = Vec::new();
    content.extend_from_slice(b"BT\n");
    // Fixed, generous offsets rather than character-count-derived widths —
    // real Helvetica glyph widths vary per letter, so a per-char estimate
    // drifts across a run and can't reliably guarantee "abutting" vs.
    // "clearly separated". A deliberate small overlap between a word's two
    // pieces guarantees a merge (well within the ordinary tight-kerning
    // tolerance, nowhere near the multi-em backtrack threshold); a large
    // fixed jump to the next word guarantees a real gap regardless of how
    // wide any piece actually rendered. Alternating the font size (12 /
    // 12.5, visually imperceptible) between every text-showing op forces
    // a genuine span boundary at each one — matching the real-document
    // shape this bug depends on (math typesetting draws each atom with
    // its own state, so it never lands in one span to begin with).
    let mut x = 50.0f32;
    let y = 700.0f32;
    for (a, b) in PIECES {
        content
            .extend_from_slice(format!("/F1 12 Tf 1 0 0 1 {x:.2} {y:.2} Tm ({a}) Tj\n").as_bytes());
        x += 1.0; // deliberate near-zero advance: piece 2 overlaps piece 1's tail
        content.extend_from_slice(
            format!("/F1 12.5 Tf 1 0 0 1 {x:.2} {y:.2} Tm ({b}) Tj\n").as_bytes(),
        );
        x += 150.0; // unambiguous real gap before the next word
    }
    content.extend_from_slice(b"ET");
    // Wide enough for 10 words at a 150pt stride plus margin — content
    // extraction / page-bounds filtering silently drops glyphs placed
    // outside the MediaBox, which a standard 612pt-wide page can't fit.
    build_minimal_pdf_raw(&content, b"/Type /Page /Parent 2 0 R /MediaBox [0 0 1700 792]")
}

fn write_xref_trailer(pdf: &mut Vec<u8>, off: &[usize]) {
    let xref_pos = pdf.len();
    pdf.extend_from_slice(format!("xref\n0 {}\n", off.len()).as_bytes());
    pdf.extend_from_slice(b"0000000000 65535 f\r\n");
    for &o in &off[1..] {
        pdf.extend_from_slice(format!("{o:010} 00000 n\r\n").as_bytes());
    }
    pdf.extend_from_slice(
        format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n", off.len())
            .as_bytes(),
    );
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
