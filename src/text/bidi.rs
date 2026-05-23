//! Unicode Bidirectional Algorithm (UAX #9) helpers for PDF text
//! extraction.
//!
//! Extracted PDF text can contain Arabic and Hebrew runs in either
//! *visual order* (typical of older Acrobat outputs and a few
//! tagged-PDF flows) or *logical order* (the common case for tools
//! that explicitly post-process to Unicode logical order, including
//! the pdfium `hebrew_mirrored.pdf` test fixture). The PDF
//! specification does not constrain which order a producer chooses;
//! callers must know which case they have before reordering.
//!
//! This module is a thin wrapper around the `unicode-bidi` crate
//! (UAX #9 implementation). It exposes the operations the converters
//! actually need:
//! - `looks_rtl(text)` — quick yes/no check for whether `text` contains
//!   any RTL characters worth running the bidi algorithm against.
//! - `reorder_visual_to_logical(text)` — given a single visual-order
//!   line, returns the logical-order string with embedded LTR runs
//!   (numerals, English words) preserved in their natural reading
//!   direction. **Caller is responsible for knowing the input is in
//!   visual order.** The default markdown converter does NOT call
//!   this for that reason.
//! - `paragraph_is_rtl(text)` — dominant paragraph direction per UAX
//!   #9 §3.3.1 (level of the first strong character).
//!
//! Issue #377 D7 background: the `right_to_left_02` fixture is an
//! Arabic government document where pdf_oxide previously inserted
//! spurious `**bold**` markers around individual letters because
//! contextual glyph forms (initial / medial / final shapes) flipped
//! the font-weight detector. The markdown converter strips those
//! markers (see `pipeline::converters::markdown::strip_inline_emphasis_in_rtl`)
//! while leaving order alone.

#![forbid(unsafe_code)]

use unicode_bidi::BidiInfo;

/// Cheap pre-check: does `text` look like it contains any RTL
/// characters? Used by the converter to skip the bidi pass entirely
/// for pure-LTR pages (the common case).
///
/// Delegates to `crate::text::rtl_detector::is_rtl_text` so the
/// authoritative list of supported RTL Unicode ranges (Hebrew,
/// Arabic main, Arabic Supplement, Arabic Extended-A, Arabic
/// Presentation Forms-A and -B) lives in exactly one place. A
/// previous inline copy of those ranges in this module risked
/// silent drift when one was updated and the other was not.
pub fn looks_rtl(text: &str) -> bool {
    text.chars()
        .any(|c| crate::text::rtl_detector::is_rtl_text(c as u32))
}

/// Reorder a single line of visual-order text into logical order using
/// UAX #9. Returns the original string when no RTL characters are
/// present (fast path).
///
/// Per UAX #9 §3.3.4 (Reordering), embedded LTR runs (digits, Latin
/// words) inside an RTL paragraph are kept in their natural left-to-
/// right direction; only the surrounding RTL runs are reversed to
/// match the paragraph direction.
pub fn reorder_visual_to_logical(text: &str) -> String {
    if !looks_rtl(text) {
        return text.to_string();
    }
    // Default paragraph direction left to UAX #9 to infer from the
    // first strong character; this matches what PDF readers (and
    // pdftotext) do for mixed-direction lines.
    let info = BidiInfo::new(text, None);
    if info.paragraphs.is_empty() {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    for para in &info.paragraphs {
        let line_range = para.range.clone();
        let line = info.reorder_line(para, line_range);
        out.push_str(&line);
    }
    out
}

/// Whether the *dominant* paragraph direction of `text` is RTL,
/// computed per UAX #9 §3.3.1 from the level of the first strong
/// character in the first paragraph. Mixed-direction strings whose
/// first strong char is LTR (e.g. an English label followed by an
/// Arabic value) report as LTR even though they contain RTL chars.
pub fn paragraph_is_rtl(text: &str) -> bool {
    if !looks_rtl(text) {
        return false;
    }
    let info = BidiInfo::new(text, None);
    info.paragraphs
        .first()
        .map(|p| p.level.is_rtl())
        .unwrap_or(false)
}

/// Verdict of the geometric visual-vs-logical detector (#537).
///
/// Returned by [`detect_visual_order_run`] for a contiguous RTL run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RunOrder {
    /// The PDF content stream emitted the run in **visual order** —
    /// glyphs were drawn left-to-right in user space even though the
    /// script reads right-to-left. The caller should apply UAX #9
    /// reordering ([`reorder_visual_to_logical`]) — or the simpler
    /// per-run `.chars().rev()` reversal — to produce logical-order
    /// codepoints for downstream RAG / search / display consumers.
    Visual,
    /// The PDF content stream emitted the run in **logical order**.
    /// Chars are placed right-to-left in user space (because the
    /// producer ran its own bidi pass before drawing), so the
    /// extracted codepoint sequence already matches reading order.
    /// The caller must NOT reorder — doing so would invert the run
    /// and break previously-correct output. The pdfium
    /// `hebrew_mirrored.pdf` test fixture is the canonical example.
    Logical,
    /// Insufficient signal to decide — sparse positions, ties,
    /// mixed direction, or the run is too short. The caller's safe
    /// default is to leave the run alone (the v0.3.53 behaviour).
    Ambiguous,
}

/// Geometric visual-vs-logical detector for a single RTL run (#537).
///
/// Closes the long-standing Hebrew gap captured in
/// `pipeline/converters/markdown.rs:1798-1812`: the bidi machinery
/// is already wired (UAX #9 via `unicode-bidi`, [`reorder_visual_to_logical`])
/// but the markdown converter explicitly does *not* call it because
/// some PDFs store text in visual order and some in logical order,
/// and "without a reliable way to detect which order the source uses
/// we drop the reorder step." This function is that reliable way.
///
/// # Inputs
///
/// `chars_with_x` — a slice of `(codepoint, x_origin_in_user_space)`
/// pairs for the characters that make up the run, in **content-stream
/// order** (i.e. the order the PDF's `Tj`/`TJ` operator emitted them).
/// The `x_origin` is the *user-space* x-coordinate where each glyph
/// was drawn — after `Tm` (text matrix) and `CTM` (current
/// transformation matrix) have been applied. Callers that have only
/// text-space coordinates must transform first; the detector relies
/// on monotonicity in the page's visible coordinate system.
///
/// Whitespace, diacritics, and presentation forms are filtered out
/// before the monotonicity check (they're noise for direction
/// detection).
///
/// # Algorithm
///
/// 1. Require **≥ 4 RTL letters** in the run. Short runs are noise.
/// 2. Bail with [`RunOrder::Ambiguous`] if the run contains any
///    **Arabic Presentation Forms** (U+FB50-U+FDFF, U+FE70-U+FEFF).
///    Those are already handled by the existing Pass 0 of
///    `document::PdfDocument::reverse_rtl_visual_order_runs`, and
///    second-guessing it here would risk double-reversal.
/// 3. Compare adjacent x-coordinates with a `0.5pt` kerning
///    tolerance:
///    - **ascending** (chars placed left-to-right) → visual signal.
///    - **descending** (chars placed right-to-left) → logical signal.
///    - **tie** (within 0.5pt) → no signal for this pair.
/// 4. Require **≥ 90 % monotonicity** (`asc / total > 0.9` or
///    `desc / total > 0.9`) to return [`RunOrder::Visual`] or
///    [`RunOrder::Logical`]. Below threshold → [`RunOrder::Ambiguous`].
///
/// The 90 % floor is deliberately strict: the cost of an unwarranted
/// reversal (logical PDF → visual output) is higher than the cost of
/// a missed reversal (visual PDF → uncorrected output). When in
/// doubt, leave the run alone.
///
/// # Why X-monotonicity is the right signal
///
/// PDF content streams emit glyphs in the order they're drawn, with
/// absolute positions from `Tm` * `CTM` + offset. A visual-order
/// producer (legacy Acrobat, hand-shaped Arabic, the Magic Palace
/// Eilat hotel PDF from issue #537) draws Hebrew left-to-right in
/// user space even though the script reads right-to-left — so the
/// first codepoint in the stream has the smallest x. A logical-order
/// producer (modern Word with bidi pass, the pdfium
/// `hebrew_mirrored.pdf` test fixture) draws Hebrew right-to-left,
/// so the first codepoint has the largest x. The geometric direction
/// is observable and unambiguous — see
/// `docs/releases/plans/v0.3.54/research-bidi-visual-logical-detection.md`
/// for the W3C / PDFuzz / library-by-library survey.
pub(crate) fn detect_visual_order_run(chars_with_x: &[(char, f32)]) -> RunOrder {
    // Arabic Presentation Forms presence → Pass 0 owns this run.
    // Check against the *original* input so PF chars block us even
    // when the letter filter below would strip them.
    if chars_with_x.iter().any(|(c, _)| {
        let cp = *c as u32;
        (0xFB50..=0xFDFF).contains(&cp) || (0xFE70..=0xFEFF).contains(&cp)
    }) {
        return RunOrder::Ambiguous;
    }

    // Filter: keep RTL **letters** only. `is_rtl_text` matches the
    // whole Arabic/Hebrew script range and so would let diacritics and
    // presentation forms count toward the ≥4 threshold and skew the
    // monotonicity numerator — neither is direction signal. Explicit
    // letter checks match the documented algorithm.
    use crate::text::rtl_detector::{is_arabic_letter, is_hebrew_letter};
    let rtl: Vec<(char, f32)> = chars_with_x
        .iter()
        .copied()
        .filter(|(c, _)| {
            let cp = *c as u32;
            is_arabic_letter(cp) || is_hebrew_letter(cp)
        })
        .collect();

    if rtl.len() < 4 {
        return RunOrder::Ambiguous;
    }

    const KERN_TOL: f32 = 0.5; // points
    let mut asc: usize = 0;
    let mut desc: usize = 0;
    for w in rtl.windows(2) {
        let (_, x0) = w[0];
        let (_, x1) = w[1];
        let dx = x1 - x0;
        if dx > KERN_TOL {
            asc += 1;
        } else if dx < -KERN_TOL {
            desc += 1;
        }
        // |dx| <= KERN_TOL → tie, no contribution to either count.
    }
    let total = asc + desc;
    if total == 0 {
        // All ties — degenerate, no signal.
        return RunOrder::Ambiguous;
    }
    // 90 % monotonicity floor — strict-on-purpose so we never reorder
    // a logical-order PDF on a noisy signal.
    // Express as integer math: 10 * asc > 9 * total ↔ asc / total > 0.9.
    if 10 * asc > 9 * total {
        return RunOrder::Visual;
    }
    if 10 * desc > 9 * total {
        return RunOrder::Logical;
    }
    RunOrder::Ambiguous
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn looks_rtl_pure_ascii_is_false() {
        assert!(!looks_rtl("hello world"));
        assert!(!looks_rtl(""));
    }

    #[test]
    fn looks_rtl_arabic_is_true() {
        assert!(looks_rtl("مرحبا"));
        // Mixed line containing any RTL char is true.
        assert!(looks_rtl("year 2024 عام"));
    }

    #[test]
    fn looks_rtl_hebrew_is_true() {
        assert!(looks_rtl("שלום"));
    }

    #[test]
    fn reorder_pure_ltr_is_identity() {
        let s = "Hello, world!";
        assert_eq!(reorder_visual_to_logical(s), s);
    }

    /// D7-fix documentation — `reorder_visual_to_logical` assumes the
    /// input is in *visual* order and converts to logical. PDFs vary:
    /// some store visual order (Arabic news papers, certain Acrobat
    /// outputs) and some store logical order (most modern publishers,
    /// the pdfium hebrew_mirrored.pdf test fixture). Callers MUST
    /// know which case they are in. The default markdown converter
    /// no longer invokes this function for that reason — see
    /// pipeline::converters::markdown.rs RTL emphasis-cleanup block.
    /// This test pins the asymmetric behaviour as a contract.
    #[test]
    fn reorder_is_a_visual_to_logical_converter_not_idempotent() {
        let logical_hebrew = "בנימין";
        let after_first = reorder_visual_to_logical(logical_hebrew);
        // First call REVERSES (treating input as visual).
        assert_ne!(after_first, logical_hebrew);
        // Second call reverses again — back to the original.
        let after_second = reorder_visual_to_logical(&after_first);
        assert_eq!(after_second, logical_hebrew);
    }

    /// D7 RED — A visual-order Arabic line with embedded English
    /// numerals must come back in logical order with the numerals
    /// preserved in their natural reading direction. Reproduces the
    /// `right_to_left_02` fixture pattern.
    #[test]
    fn reorder_arabic_with_numerals_keeps_digits_logical() {
        // Visual order (as PDF emits): "كان 2024 جيدا عام" reversed
        // for the Arabic runs, with "2024" embedded inline.
        // Logical (Unicode code-point) order: "عام 2024 كان جيدا".
        let logical = "عام 2024 كان جيدا";
        // Round-trip: reordering already-logical text should leave it
        // unchanged (the BiDi algorithm is idempotent on logical
        // strings whose paragraph direction matches the dominant
        // strong character).
        let result = reorder_visual_to_logical(logical);
        // Numerals must still be `2024`, not `4202`, regardless of the
        // surrounding RTL runs.
        assert!(result.contains("2024"), "expected `2024` in reordered line, got {:?}", result);
        // Length is preserved (no characters dropped or duplicated).
        assert_eq!(result.chars().count(), logical.chars().count());
    }

    #[test]
    fn paragraph_is_rtl_for_arabic() {
        assert!(paragraph_is_rtl("هذا نص عربي"));
    }

    #[test]
    fn paragraph_is_not_rtl_for_pure_english() {
        assert!(!paragraph_is_rtl("This is English"));
    }

    /// `looks_rtl` and `crate::text::rtl_detector::is_rtl_text` must
    /// agree on every codepoint, since the bidi module delegates to
    /// the detector. Pin the parity to catch any future drift in
    /// either direction.
    #[test]
    fn looks_rtl_delegates_to_rtl_detector() {
        for cp in [
            // Edges of every supported block.
            0x058F, 0x0590, 0x05FF, 0x0600, 0x0633, 0x06FF, 0x0700, 0x074F, 0x0750, 0x077F, 0x0780,
            0x08A0, 0x08FF, 0x0900, 0xFB4F, 0xFB50, 0xFDFF, 0xFE00, 0xFE70, 0xFEFE, 0xFEFF, 0xFF00,
        ] {
            if let Some(c) = char::from_u32(cp) {
                let s = c.to_string();
                let bidi_says = looks_rtl(&s);
                let detector_says = crate::text::rtl_detector::is_rtl_text(cp);
                assert_eq!(
                    bidi_says, detector_says,
                    "U+{:04X}: looks_rtl={} but rtl_detector::is_rtl_text={}",
                    cp, bidi_says, detector_says
                );
            }
        }
    }

    /// `paragraph_is_rtl` must reflect the *dominant* paragraph
    /// direction (per UAX #9 §3.3.1 — the level of the first strong
    /// character). A paragraph led by an LTR token but with RTL
    /// chars further in (e.g. `Foo بار 1`) is logically LTR and
    /// must not report as RTL just because some RTL characters
    /// appear later. Earlier impl returned true on any string
    /// containing RTL chars, conflating with `looks_rtl`.
    #[test]
    fn paragraph_is_rtl_respects_dominant_direction() {
        // Dominant LTR (first strong char is Latin) → false.
        assert!(!paragraph_is_rtl("Foo بار 1"));
        // Dominant RTL (first strong char is Arabic) → true.
        assert!(paragraph_is_rtl("بار Foo 1"));
    }

    /// D7 coverage — the looks_rtl quick-check spans every RTL Unicode
    /// block we declare support for. Used as the converter's gate, so
    /// any block we miss here would entirely bypass the bidi pass for
    /// that script.
    #[test]
    fn looks_rtl_covers_all_supported_blocks() {
        let cases: &[(u32, &str)] = &[
            (0x0590, "Hebrew start"),
            (0x05F4, "Hebrew end-ish"),
            (0x0600, "Arabic start"),
            (0x06FF, "Arabic end"),
            (0x0750, "Arabic Supplement start"),
            (0x077F, "Arabic Supplement end"),
            (0x08A0, "Arabic Extended-A start"),
            (0x08FF, "Arabic Extended-A end"),
            (0xFB50, "Arabic Presentation Forms-A start"),
            (0xFDFF, "Arabic Presentation Forms-A end"),
            (0xFE70, "Arabic Presentation Forms-B start"),
            (0xFEFF, "Arabic Presentation Forms-B end"),
        ];
        for (cp, name) in cases {
            if let Some(c) = char::from_u32(*cp) {
                let s = c.to_string();
                assert!(looks_rtl(&s), "looks_rtl({:?} {}) should be true", s, name);
            }
        }
    }

    /// D7 negative coverage — characters that LOOK like they could be
    /// RTL but are actually neutral or LTR (CJK, math, common
    /// punctuation, the BOM area near U+FEFF).
    #[test]
    fn looks_rtl_rejects_neutral_and_cjk() {
        for s in [
            "中文",   // CJK
            "日本語", // Japanese
            "α β γ",  // Greek (LTR)
            "1234567890",
            "!@#$%^&*()",
            "café",
            "naïve",
        ] {
            assert!(!looks_rtl(s), "looks_rtl({:?}) should be false", s);
        }
    }

    /// D7 coverage — reorder is byte-stable for pure-ASCII strings of
    /// many shapes (no RTL means identity).
    #[test]
    fn reorder_pure_ltr_identity_extras() {
        for s in [
            "",
            "a",
            "Hello, world!",
            "Multi-line\nstays unchanged",
            "Numbers: 1234 5678",
            "Symbols: !@#$%^&*",
            "Whitespace   between   words",
        ] {
            assert_eq!(reorder_visual_to_logical(s), s, "identity broken on {:?}", s);
        }
    }

    /// D7 coverage — reorder preserves character count and never drops
    /// or duplicates content. Property-style spot-check across mixed
    /// inputs.
    #[test]
    fn reorder_preserves_character_count() {
        for s in [
            "عربي",
            "هذا نص عربي للاختبار",
            "year 2024 عام جيد",
            "שלום world",
            "Mixed: عربي + 123 + Latin",
        ] {
            let out = reorder_visual_to_logical(s);
            assert_eq!(
                out.chars().count(),
                s.chars().count(),
                "char count changed: {:?} -> {:?}",
                s,
                out
            );
        }
    }

    /// D7 coverage — embedded LTR runs (English brand names, codes)
    /// inside an Arabic paragraph survive intact in the output. The
    /// English token must still be findable as a contiguous substring,
    /// not reversed.
    #[test]
    fn reorder_keeps_embedded_ltr_token_contiguous() {
        let line = "هذا منتج Microsoft الجديد";
        let result = reorder_visual_to_logical(line);
        assert!(
            result.contains("Microsoft"),
            "embedded LTR token reversed: {:?} -> {:?}",
            line,
            result
        );
    }

    /// D7 coverage — paragraph_is_rtl agrees with looks_rtl on edge
    /// cases (empty string, whitespace, mixed-script).
    #[test]
    fn paragraph_is_rtl_edges() {
        assert!(!paragraph_is_rtl(""));
        assert!(!paragraph_is_rtl("   "));
        assert!(!paragraph_is_rtl("123 456"));
        // Mixed but RTL-dominated.
        assert!(paragraph_is_rtl("نص with English"));
    }

    // ==========================================================================
    // detect_visual_order_run — geometric visual-vs-logical detector (#537)
    // ==========================================================================

    #[test]
    fn detect_visual_run_short_run_is_ambiguous() {
        // < 4 RTL letters → not enough signal.
        let three_chars = [('ק', 0.0), ('ר', 6.0), ('ח', 12.0)];
        assert_eq!(detect_visual_order_run(&three_chars), RunOrder::Ambiguous);
    }

    #[test]
    fn detect_visual_run_hebrew_visual_order() {
        // Hebrew word "מקלדת" (keyboard, 5 letters) emitted in visual
        // order: leftmost glyph first in stream, ascending x.
        let visual = [
            ('מ', 0.0),
            ('ק', 6.0),
            ('ל', 12.0),
            ('ד', 18.0),
            ('ת', 24.0),
        ];
        assert_eq!(detect_visual_order_run(&visual), RunOrder::Visual);
    }

    #[test]
    fn detect_visual_run_hebrew_logical_order() {
        // Same letters, logical order: rightmost glyph first in stream
        // (descending x — the PDF producer ran its own bidi pass before
        // drawing).
        let logical = [
            ('מ', 24.0),
            ('ק', 18.0),
            ('ל', 12.0),
            ('ד', 6.0),
            ('ת', 0.0),
        ];
        assert_eq!(detect_visual_order_run(&logical), RunOrder::Logical);
    }

    #[test]
    fn detect_visual_run_arabic_main_block_visual() {
        // Arabic main block (U+0600-U+06FF), no Presentation Forms.
        // Ascending x → Visual.
        let visual = [('ع', 0.0), ('ر', 7.0), ('ب', 14.0), ('ي', 21.0)];
        assert_eq!(detect_visual_order_run(&visual), RunOrder::Visual);
    }

    #[test]
    fn detect_visual_run_presentation_forms_bails_out() {
        // Arabic Presentation Forms-B in the run — Pass 0 owns this.
        // The geometric detector must bail rather than double-process.
        let with_pfs = [
            ('\u{FE80}', 0.0), // Hamza isolated form
            ('\u{FE91}', 7.0), // Beh initial form
            ('\u{FE9A}', 14.0),
            ('\u{FEAB}', 21.0),
        ];
        assert_eq!(detect_visual_order_run(&with_pfs), RunOrder::Ambiguous);
    }

    #[test]
    fn detect_visual_run_ties_are_ambiguous() {
        // All chars at the same x (degenerate). No monotonicity signal.
        let ties = [('ק', 5.0), ('ר', 5.0), ('ח', 5.0), ('ל', 5.0)];
        assert_eq!(detect_visual_order_run(&ties), RunOrder::Ambiguous);
    }

    #[test]
    fn detect_visual_run_mixed_signal_is_ambiguous() {
        // 4 RTL letters: 1 ascending pair, 2 descending pairs. With
        // only 3 monotonic pairs (asc=1, desc=2, total=3), neither
        // direction reaches the 90 % floor → Ambiguous.
        let mixed = [('ק', 0.0), ('ר', 6.0), ('ח', 3.0), ('ל', 1.0)];
        assert_eq!(detect_visual_order_run(&mixed), RunOrder::Ambiguous);
    }

    #[test]
    fn detect_visual_run_ignores_non_rtl_chars() {
        // Embedded LTR digit ("2024") between Hebrew letters — filtered
        // out before the monotonicity check. Hebrew chars still need
        // to be ≥4 and monotonic.
        let with_digit = [
            ('ק', 0.0),
            ('ר', 6.0),
            ('2', 12.0), // ignored
            ('ח', 18.0),
            ('ל', 24.0),
        ];
        assert_eq!(detect_visual_order_run(&with_digit), RunOrder::Visual);
    }

    #[test]
    fn detect_visual_run_kerning_tolerance() {
        // Tiny x differences within 0.5pt → treated as ties; can't
        // be the dominant signal on their own. Four pairs where dx
        // ≈ 0.3pt → all ties → Ambiguous.
        let kerning_noise = [('ק', 0.0), ('ר', 0.3), ('ח', 0.6), ('ל', 0.9), ('מ', 1.2)];
        assert_eq!(detect_visual_order_run(&kerning_noise), RunOrder::Ambiguous);
    }
}
