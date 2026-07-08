//! Integration tests for RTL correction on the `TJ`-array
//! (`WordBoundaryMode::Tiebreaker`, the default) text-showing path.
//!
//! Issue #826: a scanned Hebrew OCR PDF (simple TrueType font, one `TJ`
//! array per recognized word, physically left-to-right word placement,
//! invisible `Tr 3` render mode — the standard OCR-text-sandwich shape)
//! extracted with every Hebrew word both letter-reversed *and*
//! word-order-reversed, even though the geometric visual-order detector
//! added for #537/#657 was supposed to prevent exactly this class of bug.
//!
//! Two compounding root causes, both fixed here:
//!
//! 1. `flush_tj_buffer` (the flush function backing the default
//!    `WordBoundaryMode::Tiebreaker` `TJ`-array path) never received the
//!    #537 confidence-gated geometric detector — it still used the
//!    pre-#537 `has_rtl && buffer.accumulated_width > 0.0` heuristic.
//!    Because `accumulated_width` only ever sums positive glyph advance
//!    widths (`TJ` kerning offsets never modify it), the check was true
//!    for nearly every non-empty RTL buffer — so this path didn't
//!    *detect* direction, it unconditionally reversed every RTL `TJ`
//!    buffer it flushed. Fixed by routing all three flush sites
//!    (`flush_tj_buffer`, `flush_tj_span_buffer`, `cluster_to_span`)
//!    through the shared `bidi::apply_rtl_verdict`.
//! 2. Even with the geometric detector wired in everywhere, it still
//!    can't tell "already-logical content placed at plain ascending x"
//!    (the OCR case — no rendering-correctness pressure on invisible
//!    text) from "genuinely visual-order content" (also ascending x) —
//!    both produce an identical geometric signature. Fixed by threading
//!    text render mode (`Tr`) through: for invisible (`3`/`7`) runs, the
//!    geometric/coarse heuristics are skipped entirely and the extracted
//!    content order is trusted as-is (see `bidi::apply_rtl_verdict`'s
//!    doc comment for the full rationale and the accepted trade-off).
//!
//! Both PDFs below are hand-built, untagged, single-page, with a simple
//! (non-Type0) TrueType-subtype font (`/FirstChar 0`, no /Encoding —
//! same minimal shape as the pdfium `hebrew_mirrored.pdf` fixture) plus
//! a `/ToUnicode` CMap, and draw each of two Hebrew words as its own
//! `[...] TJ` array (byte codes with small kerning numbers between them)
//! at increasing x — i.e. physically left-to-right word placement, the
//! shape that scopes each word to its own flush-buffer call.
//!
//! Words: ביצה (bet-yod-tsadi-he, "egg" — codes <01><02><03><04> in
//! logical reading order) and פרק (pe-resh-qof, "chapter" — codes
//! <05><06><07> in logical reading order), matching the real tractate
//! (Beitza) from issue #826. Hebrew reads right-to-left, so the correct
//! logical line is "ביצה פרק" (ביצה first) even though — matching real
//! OCR/scan placement — פרק is drawn first (it sits physically to the
//! left) and ביצה is drawn second (physically to the right).

use pdf_oxide::PdfDocument;

/// Minimal untagged one-page PDF: a simple TrueType font (`/FirstChar 0`,
/// `/Widths`, `/ToUnicode`) and a plain content stream. `content_ops` is
/// the full `BT ... ET` body.
fn build_pdf(
    tounicode_bfchars: &str,
    widths: &str,
    last_char: usize,
    content_ops: &str,
) -> Vec<u8> {
    let tounicode = format!(
        "/CIDInit /ProcSet findresource begin\n12 dict begin begincmap\n\
         1 begincodespacerange <00> <FF> endcodespacerange\n\
         {} beginbfchar\n{}endbfchar\nendcmap CMapName currentdict /CMap defineresource pop end end",
        tounicode_bfchars.lines().filter(|l| !l.trim().is_empty()).count(),
        tounicode_bfchars,
    );

    let content_bytes = content_ops.as_bytes();

    let mut buf: Vec<u8> = Vec::new();
    let mut off = vec![0usize; 7];
    let obj = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, body: &str| {
        off[id] = buf.len();
        buf.extend_from_slice(format!("{id} 0 obj\n{body}\nendobj\n").as_bytes());
    };
    let stream = |buf: &mut Vec<u8>, off: &mut Vec<usize>, id: usize, dict: &str, data: &[u8]| {
        off[id] = buf.len();
        buf.extend_from_slice(
            format!("{id} 0 obj\n<< {dict} /Length {} >>\nstream\n", data.len()).as_bytes(),
        );
        buf.extend_from_slice(data);
        buf.extend_from_slice(b"\nendstream\nendobj\n");
    };

    buf.extend_from_slice(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
    obj(&mut buf, &mut off, 1, "<< /Type /Catalog /Pages 2 0 R >>");
    obj(&mut buf, &mut off, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    obj(
        &mut buf,
        &mut off,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] \
         /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
    );
    stream(&mut buf, &mut off, 4, "", content_bytes);
    obj(
        &mut buf,
        &mut off,
        5,
        &format!(
            "<< /Type /Font /Subtype /TrueType /BaseFont /Synthetic \
             /FirstChar 0 /LastChar {last_char} /Widths [{widths}] /ToUnicode 6 0 R >>"
        ),
    );
    stream(&mut buf, &mut off, 6, "", tounicode.as_bytes());

    let xref_off = buf.len();
    buf.extend_from_slice(b"xref\n0 7\n0000000000 65535 f \n");
    for id in 1..=6 {
        buf.extend_from_slice(format!("{:010} 00000 n \n", off[id]).as_bytes());
    }
    buf.extend_from_slice(b"trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n");
    buf.extend_from_slice(format!("{xref_off}\n%%EOF\n").as_bytes());
    buf
}

const TOUNICODE: &str = "\
<01> <05D1>
<02> <05D9>
<03> <05E6>
<04> <05D4>
<05> <05E4>
<06> <05E8>
<07> <05E7>
<20> <0020>
";
const WIDTHS: &str = "600 600 600 600 600 600 600 600";
const LAST_CHAR: usize = 7;
const WORD_ONE_LOGICAL: &str = "\u{05D1}\u{05D9}\u{05E6}\u{05D4}"; // ביצה
const WORD_ONE_REVERSED: &str = "\u{05D4}\u{05E6}\u{05D9}\u{05D1}"; // הציב
const WORD_TWO_LOGICAL: &str = "\u{05E4}\u{05E8}\u{05E7}"; // פרק
const WORD_TWO_REVERSED: &str = "\u{05E7}\u{05E8}\u{05E4}"; // קרפ

/// #826 shape (the bug): each word's glyph codes are stored in
/// **logical** (already correct) reading order inside its own **invisible
/// (`3 Tr`)** `TJ` array — a Hebrew-aware OCR engine recognizes a whole
/// word correctly, but since the glyphs are never actually shown there's
/// no reason to also mirror their positions; it just places words
/// left-to-right by physical page position. Word one (ביצה) is drawn
/// second at higher x (physically rightmost, since it's read first);
/// word two (פרק) is drawn first at lower x.
#[test]
fn tj_array_invisible_logical_order_per_word_is_not_reversed() {
    let pdf = build_pdf(
        TOUNICODE,
        WIDTHS,
        LAST_CHAR,
        "BT /F1 12 Tf 3 Tr\n\
         50 100 Td [<05>0<06>0<07>] TJ\n\
         80 0 Td [<01>0<02>0<03>0<04>] TJ\nET",
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse synthetic PDF");
    let text = doc.extract_text(0).expect("extract_text");
    eprintln!("[826] invisible logical-per-word extracted: {text:?}");

    // Correct logical reading order: ביצה (word one) then פרק (word two).
    assert!(
        text.contains(WORD_ONE_LOGICAL),
        "word one (ביצה) letters got reversed — invisible-text run wrongly \
         ran through the visual-order heuristic: {text:?}"
    );
    assert!(text.contains(WORD_TWO_LOGICAL), "word two (פרק) letters got reversed: {text:?}");
}

/// Visible-mode counterpart (documents the fix's boundary, not a bug):
/// the identical already-logical, ascending-x content as above, but
/// drawn with the *default visible* render mode (no `Tr`). For visible
/// text, ascending x is a meaningful signal — a producer whose Hebrew
/// renders correctly on screen would have mirrored logical content to
/// descending x, so a real producer emitting ascending-x visible RTL
/// most likely *did* store it in visual order. The invisible-mode fix
/// above must not change this case.
#[test]
fn tj_array_visible_logical_order_per_word_still_reversed() {
    let pdf = build_pdf(
        TOUNICODE,
        WIDTHS,
        LAST_CHAR,
        "BT /F1 12 Tf\n\
         50 100 Td [<05>0<06>0<07>] TJ\n\
         80 0 Td [<01>0<02>0<03>0<04>] TJ\nET",
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse synthetic PDF");
    let text = doc.extract_text(0).expect("extract_text");
    eprintln!("[visible] logical-per-word extracted: {text:?}");

    assert!(
        text.contains(WORD_ONE_REVERSED) && text.contains(WORD_TWO_REVERSED),
        "visible-mode ascending-x RTL behavior changed — the invisible-mode \
         fix must be scoped to Tr 3/7 only: {text:?}"
    );
}

/// #657/hebrew_mirrored.pdf-style shape (regression guard): each word's
/// glyph codes are stored in **visual** (mirrored) order inside its own
/// visible `TJ` array — the case the existing fix is supposed to handle.
/// Must stay correct after the render-mode change.
#[test]
fn tj_array_visual_order_per_word_regression() {
    let pdf = build_pdf(
        TOUNICODE,
        WIDTHS,
        LAST_CHAR,
        "BT /F1 12 Tf\n\
         50 100 Td [<07>0<06>0<05>] TJ\n\
         80 0 Td [<04>0<03>0<02>0<01>] TJ\nET",
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse synthetic PDF");
    let text = doc.extract_text(0).expect("extract_text");
    eprintln!("[657] visual-per-word extracted: {text:?}");

    assert!(
        text.contains(WORD_ONE_LOGICAL),
        "word one (ביצה) not reconstructed from visual order: {text:?}"
    );
    assert!(
        text.contains(WORD_TWO_LOGICAL),
        "word two (פרק) not reconstructed from visual order: {text:?}"
    );
}

/// Known accepted trade-off (documented, not silently left uncovered):
/// genuinely visual-order content drawn *invisible* no longer gets
/// auto-reversed, because invisible-mode runs now trust content order
/// unconditionally (issue #826's fix). This asserts the actual,
/// intentional post-fix behavior for this narrower, now-deprioritized
/// case, so a future change to this trade-off is a deliberate, visible
/// diff here rather than a silent regression.
#[test]
fn tj_array_invisible_visual_order_per_word_is_not_reversed_known_tradeoff() {
    let pdf = build_pdf(
        TOUNICODE,
        WIDTHS,
        LAST_CHAR,
        "BT /F1 12 Tf 3 Tr\n\
         50 100 Td [<07>0<06>0<05>] TJ\n\
         80 0 Td [<04>0<03>0<02>0<01>] TJ\nET",
    );
    let doc = PdfDocument::from_bytes(pdf).expect("parse synthetic PDF");
    let text = doc.extract_text(0).expect("extract_text");
    eprintln!("[tradeoff] invisible visual-per-word extracted: {text:?}");

    assert!(
        text.contains(WORD_ONE_REVERSED) && text.contains(WORD_TWO_REVERSED),
        "expected the documented trade-off (invisible content trusted as-is, \
         so visual-order invisible text stays unreversed): {text:?}"
    );
}
