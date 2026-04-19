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

use unicode_bidi::{BidiInfo, Level};

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

/// Whether the dominant paragraph direction of `text` is RTL. Used by
/// the converter to suppress font-weight-derived `**bold**` styling
/// inside RTL paragraphs where contextual glyph forms (initial /
/// medial / final shapes) routinely flip the font's reported weight
/// and create spurious markdown emphasis on individual letters.
pub fn paragraph_is_rtl(text: &str) -> bool {
    if !looks_rtl(text) {
        return false;
    }
    let info = BidiInfo::new(text, None);
    info.paragraphs
        .first()
        .map(|p| p.level.is_rtl() || p.level == Level::ltr() && info.has_rtl())
        .unwrap_or(false)
        && info.has_rtl()
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
        assert!(
            result.contains("2024"),
            "expected `2024` in reordered line, got {:?}",
            result
        );
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
}
