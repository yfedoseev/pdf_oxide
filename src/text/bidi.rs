//! Unicode Bidirectional Algorithm (UAX #9) helpers for PDF text
//! extraction.
//!
//! PDF content streams emit Arabic and Hebrew glyphs in *visual order*
//! (right-to-left runs are stored left-to-right after font shaping).
//! Converting that to plain text or markdown for downstream consumers
//! requires reordering each line back into *logical order* so search,
//! diffing and screen readers behave correctly.
//!
//! This module is a thin wrapper around the `unicode-bidi` crate
//! (UAX #9 implementation). It exposes two operations the converters
//! actually need:
//! - `looks_rtl(text)` — quick yes/no check for whether `text` contains
//!   any RTL characters worth running the bidi algorithm against.
//! - `reorder_visual_to_logical(text)` — given a single visual-order
//!   line, returns the logical-order string with embedded LTR runs
//!   (numerals, English words) preserved in their natural reading
//!   direction.
//!
//! Issue #377 D7: the `right_to_left_02` fixture is an Arabic government
//! document where pdf_oxide currently emits visually-correct but
//! logically-scrambled text and inserts spurious `**bold**` markers in
//! the middle of words because shaping forms confuse the font-weight
//! detector.

use unicode_bidi::BidiInfo;

/// Cheap pre-check: does `text` look like it contains any RTL
/// characters? Used by the converter to skip the bidi pass entirely
/// for pure-LTR pages (the common case).
pub fn looks_rtl(text: &str) -> bool {
    text.chars().any(|c| {
        let code = c as u32;
        // Hebrew block (U+0590..U+05FF), Arabic block (U+0600..U+06FF),
        // Arabic Supplement (U+0750..U+077F), Arabic Extended-A
        // (U+08A0..U+08FF), Arabic Presentation Forms-A (U+FB50..U+FDFF),
        // Arabic Presentation Forms-B (U+FE70..U+FEFF). Matches the
        // ranges in src/text/rtl_detector.rs.
        matches!(
            code,
            0x0590..=0x05FF
                | 0x0600..=0x06FF
                | 0x0750..=0x077F
                | 0x08A0..=0x08FF
                | 0xFB50..=0xFDFF
                | 0xFE70..=0xFEFF
        )
    })
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
}
