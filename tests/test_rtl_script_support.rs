#![allow(clippy::useless_vec)]
#![allow(dead_code)]
//! Week 2 Day 10: Right-to-Left Script Support Test Suite
//!
//! Comprehensive tests for Arabic and Hebrew script handling:
//! - Script detection (Arabic, Hebrew, supplements, extensions)
//! - Diacritical mark handling (no boundaries before marks)
//! - Letter detection
//! - Number handling (Western and Eastern Arabic digits)
//! - Boundary rules (space, Tatweel, TJ offsets, transitions)
//! - LAM-ALEF ligature support
//! - Integration scenarios

use pdf_oxide::text::rtl_detector::*;
use pdf_oxide::text::{BoundaryContext, CharacterInfo};

/// Helper to create CharacterInfo for testing
fn make_char(code: u32, x_position: f32, tj_offset: Option<i32>) -> CharacterInfo {
    CharacterInfo {
        code,
        glyph_id: None,
        width: 400.0,
        x_position,
        tj_offset,
        font_size: 12.0,
        is_ligature: false,
        original_ligature: None,
        protected_from_split: false,
    }
}

fn make_context() -> BoundaryContext {
    BoundaryContext::new(12.0)
}

// ============================================================================
// SCRIPT DETECTION TESTS
// ============================================================================

#[cfg(test)]
mod script_detection_tests {
    use super::*;

    #[test]
    fn test_detect_arabic_main_range() {
        // Arabic main range: U+0600-U+06FF
        assert_eq!(detect_rtl_script(0x0600), Some(RTLScript::Arabic));
        assert_eq!(detect_rtl_script(0x0627), Some(RTLScript::Arabic)); // ALEF
        assert_eq!(detect_rtl_script(0x0628), Some(RTLScript::Arabic)); // BEH
        assert_eq!(detect_rtl_script(0x064B), Some(RTLScript::Arabic)); // FATHATAN (diacritic)
        assert_eq!(detect_rtl_script(0x06FF), Some(RTLScript::Arabic));

        assert!(is_rtl_text(0x0627));
        assert!(is_rtl_text(0x064B));
    }

    #[test]
    fn test_detect_arabic_supplement() {
        // Arabic Supplement: U+0750-U+077F
        assert_eq!(detect_rtl_script(0x0750), Some(RTLScript::ArabicSupplement));
        assert_eq!(detect_rtl_script(0x0760), Some(RTLScript::ArabicSupplement));
        assert_eq!(detect_rtl_script(0x077F), Some(RTLScript::ArabicSupplement));

        assert!(is_rtl_text(0x0750));
    }

    #[test]
    fn test_detect_arabic_extended_a() {
        // Arabic Extended-A: U+08A0-U+08FF
        assert_eq!(detect_rtl_script(0x08A0), Some(RTLScript::ArabicExtendedA));
        assert_eq!(detect_rtl_script(0x08B0), Some(RTLScript::ArabicExtendedA));
        assert_eq!(detect_rtl_script(0x08FF), Some(RTLScript::ArabicExtendedA));

        assert!(is_rtl_text(0x08A0));
    }

    #[test]
    fn test_detect_hebrew() {
        // Hebrew: U+0590-U+05FF
        assert_eq!(detect_rtl_script(0x0590), Some(RTLScript::Hebrew));
        assert_eq!(detect_rtl_script(0x05D0), Some(RTLScript::Hebrew)); // ALEF
        assert_eq!(detect_rtl_script(0x05D1), Some(RTLScript::Hebrew)); // BET
        assert_eq!(detect_rtl_script(0x05BC), Some(RTLScript::Hebrew)); // DAGESH (diacritic)
        assert_eq!(detect_rtl_script(0x05FF), Some(RTLScript::Hebrew));

        assert!(is_rtl_text(0x05D0));
        assert!(is_rtl_text(0x05BC));
    }

    #[test]
    fn test_detect_presentation_forms_a() {
        // Arabic Presentation Forms-A: U+FB50-U+FDFF
        assert_eq!(detect_rtl_script(0xFB50), Some(RTLScript::PresentationFormsA));
        assert_eq!(detect_rtl_script(0xFDCF), Some(RTLScript::PresentationFormsA)); // Within range
        assert_eq!(detect_rtl_script(0xFDFF), Some(RTLScript::PresentationFormsA));

        assert!(is_rtl_text(0xFB50));
        assert!(is_rtl_text(0xFDCF));
    }

    #[test]
    fn test_detect_presentation_forms_b() {
        // Arabic Presentation Forms-B: U+FE70-U+FEFF
        assert_eq!(detect_rtl_script(0xFE70), Some(RTLScript::PresentationFormsB));
        assert_eq!(detect_rtl_script(0xFE80), Some(RTLScript::PresentationFormsB));
        assert_eq!(detect_rtl_script(0xFEFF), Some(RTLScript::PresentationFormsB));

        assert!(is_rtl_text(0xFE70));
    }

    #[test]
    fn test_non_rtl_scripts() {
        // Latin
        assert_eq!(detect_rtl_script(0x0041), None); // 'A'
        assert_eq!(detect_rtl_script(0x007A), None); // 'z'

        // CJK
        assert_eq!(detect_rtl_script(0x4E00), None); // CJK ideograph

        // Cyrillic
        assert_eq!(detect_rtl_script(0x0400), None);

        assert!(!is_rtl_text(0x0041));
        assert!(!is_rtl_text(0x4E00));
    }

    #[test]
    fn test_script_boundary_values() {
        // Test boundary values for each range
        assert_eq!(detect_rtl_script(0x05FF), Some(RTLScript::Hebrew));
        assert_eq!(detect_rtl_script(0x0600), Some(RTLScript::Arabic));

        assert_eq!(detect_rtl_script(0x06FF), Some(RTLScript::Arabic));
        assert_eq!(detect_rtl_script(0x0700), None);

        assert_eq!(detect_rtl_script(0x074F), None);
        assert_eq!(detect_rtl_script(0x0750), Some(RTLScript::ArabicSupplement));
    }
}

// ============================================================================
// DIACRITICAL MARK TESTS
// ============================================================================

#[cfg(test)]
mod diacritical_tests {
    use super::*;

    #[test]
    fn test_arabic_diacritics() {
        // Arabic diacritical marks: U+064B-U+0658
        assert!(is_arabic_diacritic(0x064B)); // FATHATAN
        assert!(is_arabic_diacritic(0x064C)); // DAMMATAN
        assert!(is_arabic_diacritic(0x064D)); // KASRATAN
        assert!(is_arabic_diacritic(0x064E)); // FATHA
        assert!(is_arabic_diacritic(0x064F)); // DAMMA
        assert!(is_arabic_diacritic(0x0650)); // KASRA
        assert!(is_arabic_diacritic(0x0651)); // SHADDA
        assert!(is_arabic_diacritic(0x0652)); // SUKUN
        assert!(is_arabic_diacritic(0x0653)); // MADDAH
        assert!(is_arabic_diacritic(0x0654)); // HAMZA ABOVE
        assert!(is_arabic_diacritic(0x0655)); // HAMZA BELOW
        assert!(is_arabic_diacritic(0x0656)); // SUBSCRIPT ALEF
        assert!(is_arabic_diacritic(0x0657)); // INVERTED DAMMA
        assert!(is_arabic_diacritic(0x0658)); // NOON GHUNNA MARK

        // Extended marks
        assert!(is_arabic_diacritic(0x06D6)); // SMALL HIGH LIGATURE SAD WITH LAM WITH ALEF MAKSURA
        assert!(is_arabic_diacritic(0x06DC)); // SMALL HIGH SEEN
        assert!(is_arabic_diacritic(0x06DF)); // SMALL HIGH ROUNDED ZERO
        assert!(is_arabic_diacritic(0x06E0)); // SMALL HIGH UPRIGHT RECTANGULAR ZERO
        assert!(is_arabic_diacritic(0x06E4)); // SMALL HIGH MADDA
        assert!(is_arabic_diacritic(0x06E7)); // SMALL HIGH YEH
        assert!(is_arabic_diacritic(0x06E8)); // SMALL HIGH NOON
        assert!(is_arabic_diacritic(0x06EA)); // EMPTY CENTRE LOW STOP
        assert!(is_arabic_diacritic(0x06EB)); // EMPTY CENTRE HIGH STOP
        assert!(is_arabic_diacritic(0x06EC)); // ROUNDED HIGH STOP WITH FILLED CENTRE
        assert!(is_arabic_diacritic(0x06ED)); // SMALL LOW MEEM

        // Not diacritics
        assert!(!is_arabic_diacritic(0x0627)); // ALEF (letter)
        assert!(!is_arabic_diacritic(0x0628)); // BEH (letter)
    }

    #[test]
    fn test_hebrew_diacritics() {
        // Hebrew diacritical marks
        assert!(is_hebrew_diacritic(0x05BC)); // DAGESH
        assert!(is_hebrew_diacritic(0x05BD)); // METEG
        assert!(is_hebrew_diacritic(0x05BF)); // RAFE
        assert!(is_hebrew_diacritic(0x05C1)); // SHIN DOT
        assert!(is_hebrew_diacritic(0x05C2)); // SIN DOT
        assert!(is_hebrew_diacritic(0x05C4)); // UPPER DOT
        assert!(is_hebrew_diacritic(0x05C5)); // LOWER DOT
        assert!(is_hebrew_diacritic(0x05C7)); // QAMATS QATAN

        // Vowel marks
        assert!(is_hebrew_diacritic(0x05B0)); // SHEVA
        assert!(is_hebrew_diacritic(0x05B1)); // HATAF SEGOL
        assert!(is_hebrew_diacritic(0x05B9)); // HOLAM

        // Not diacritics
        assert!(!is_hebrew_diacritic(0x05D0)); // ALEF (letter)
        assert!(!is_hebrew_diacritic(0x05D1)); // BET (letter)
    }

    #[test]
    fn test_rtl_diacritic_combined() {
        // Test unified diacritic check
        assert!(is_rtl_diacritic(0x064B)); // Arabic FATHATAN
        assert!(is_rtl_diacritic(0x0651)); // Arabic SHADDA
        assert!(is_rtl_diacritic(0x05BC)); // Hebrew DAGESH
        assert!(is_rtl_diacritic(0x05B0)); // Hebrew SHEVA

        assert!(!is_rtl_diacritic(0x0041)); // Latin 'A'
        assert!(!is_rtl_diacritic(0x0627)); // Arabic ALEF (letter)
    }

    #[test]
    fn test_no_boundary_before_arabic_diacritic() {
        let context = make_context();
        let base = make_char(0x0628, 100.0, None); // BEH
        let mark = make_char(0x064E, 100.0, None); // FATHA

        let result = should_split_at_rtl_boundary(&base, &mark, Some(&context));
        assert_eq!(result, Some(false), "Should not split before Arabic diacritic");
    }

    #[test]
    fn test_no_boundary_before_hebrew_diacritic() {
        let context = make_context();
        let base = make_char(0x05D1, 100.0, None); // BET
        let mark = make_char(0x05BC, 100.0, None); // DAGESH

        let result = should_split_at_rtl_boundary(&base, &mark, Some(&context));
        assert_eq!(result, Some(false), "Should not split before Hebrew diacritic");
    }

    #[test]
    fn test_multiple_diacritics_same_base() {
        let context = make_context();

        // First mark after base
        let base = make_char(0x0628, 100.0, None); // BEH
        let mark1 = make_char(0x064E, 100.0, None); // FATHA
        assert_eq!(should_split_at_rtl_boundary(&base, &mark1, Some(&context)), Some(false));

        // Second mark after first mark
        let mark2 = make_char(0x0651, 100.0, None); // SHADDA
        assert_eq!(should_split_at_rtl_boundary(&mark1, &mark2, Some(&context)), Some(false));
    }

    #[test]
    fn test_diacritic_sequences() {
        let context = make_context();

        // Hebrew: letter + vowel + dagesh
        let letter = make_char(0x05D0, 100.0, None); // ALEF
        let vowel = make_char(0x05B0, 100.0, None); // SHEVA
        let dagesh = make_char(0x05BC, 100.0, None); // DAGESH

        assert_eq!(should_split_at_rtl_boundary(&letter, &vowel, Some(&context)), Some(false));
        assert_eq!(should_split_at_rtl_boundary(&vowel, &dagesh, Some(&context)), Some(false));
    }

    #[test]
    fn test_diacritic_on_presentation_form() {
        let context = make_context();

        // Presentation form + diacritic
        let form = make_char(0xFE82, 100.0, None); // ALEF FINAL FORM
        let mark = make_char(0x064E, 100.0, None); // FATHA

        assert_eq!(should_split_at_rtl_boundary(&form, &mark, Some(&context)), Some(false));
    }
}

// ============================================================================
// LETTER DETECTION TESTS
// ============================================================================

#[cfg(test)]
mod letter_detection_tests {
    use super::*;

    #[test]
    fn test_arabic_letters() {
        // Common Arabic letters
        assert!(is_arabic_letter(0x0627)); // ALEF
        assert!(is_arabic_letter(0x0628)); // BEH
        assert!(is_arabic_letter(0x062A)); // TEH
        assert!(is_arabic_letter(0x062B)); // THEH
        assert!(is_arabic_letter(0x062C)); // JEEM
        assert!(is_arabic_letter(0x062D)); // HAH
        assert!(is_arabic_letter(0x062E)); // KHAH
        assert!(is_arabic_letter(0x062F)); // DAL
        assert!(is_arabic_letter(0x0630)); // THAL
        assert!(is_arabic_letter(0x0631)); // REH
        assert!(is_arabic_letter(0x0632)); // ZAIN
        assert!(is_arabic_letter(0x0633)); // SEEN
        assert!(is_arabic_letter(0x0634)); // SHEEN
        assert!(is_arabic_letter(0x0635)); // SAD
        assert!(is_arabic_letter(0x0636)); // DAD
        assert!(is_arabic_letter(0x0637)); // TAH
        assert!(is_arabic_letter(0x0638)); // ZAH
        assert!(is_arabic_letter(0x0639)); // AIN
        assert!(is_arabic_letter(0x063A)); // GHAIN
        assert!(is_arabic_letter(0x0641)); // FEH
        assert!(is_arabic_letter(0x0642)); // QAF
        assert!(is_arabic_letter(0x0643)); // KAF
        assert!(is_arabic_letter(0x0644)); // LAM
        assert!(is_arabic_letter(0x0645)); // MEEM
        assert!(is_arabic_letter(0x0646)); // NOON
        assert!(is_arabic_letter(0x0647)); // HEH
        assert!(is_arabic_letter(0x0648)); // WAW
        assert!(is_arabic_letter(0x064A)); // YEH

        // Not letters
        assert!(!is_arabic_letter(0x064B)); // FATHATAN (diacritic)
        assert!(!is_arabic_letter(0x0660)); // DIGIT ZERO
    }

    #[test]
    fn test_hebrew_letters() {
        // Hebrew alphabet
        assert!(is_hebrew_letter(0x05D0)); // ALEF
        assert!(is_hebrew_letter(0x05D1)); // BET
        assert!(is_hebrew_letter(0x05D2)); // GIMEL
        assert!(is_hebrew_letter(0x05D3)); // DALET
        assert!(is_hebrew_letter(0x05D4)); // HE
        assert!(is_hebrew_letter(0x05D5)); // VAV
        assert!(is_hebrew_letter(0x05D6)); // ZAYIN
        assert!(is_hebrew_letter(0x05D7)); // HET
        assert!(is_hebrew_letter(0x05D8)); // TET
        assert!(is_hebrew_letter(0x05D9)); // YOD
        assert!(is_hebrew_letter(0x05DA)); // FINAL KAF
        assert!(is_hebrew_letter(0x05DB)); // KAF
        assert!(is_hebrew_letter(0x05DC)); // LAMED
        assert!(is_hebrew_letter(0x05DD)); // FINAL MEM
        assert!(is_hebrew_letter(0x05DE)); // MEM
        assert!(is_hebrew_letter(0x05DF)); // FINAL NUN
        assert!(is_hebrew_letter(0x05E0)); // NUN
        assert!(is_hebrew_letter(0x05E1)); // SAMEKH
        assert!(is_hebrew_letter(0x05E2)); // AYIN
        assert!(is_hebrew_letter(0x05E3)); // FINAL PE
        assert!(is_hebrew_letter(0x05E4)); // PE
        assert!(is_hebrew_letter(0x05E5)); // FINAL TSADI
        assert!(is_hebrew_letter(0x05E6)); // TSADI
        assert!(is_hebrew_letter(0x05E7)); // QOF
        assert!(is_hebrew_letter(0x05E8)); // RESH
        assert!(is_hebrew_letter(0x05E9)); // SHIN
        assert!(is_hebrew_letter(0x05EA)); // TAV

        // Not letters
        assert!(!is_hebrew_letter(0x05BC)); // DAGESH (diacritic)
        assert!(!is_hebrew_letter(0x05BE)); // MAQAF (punctuation)
    }

    #[test]
    fn test_letter_detection_ranges() {
        // Test full ranges
        for code in 0x0621..=0x063A {
            if code != 0x0640 {
                // TATWEEL is not a letter
                assert!(is_arabic_letter(code), "Arabic letter 0x{:04X}", code);
            }
        }

        for code in 0x0641..=0x064A {
            assert!(is_arabic_letter(code), "Arabic letter 0x{:04X}", code);
        }

        for code in 0x05D0..=0x05EA {
            assert!(is_hebrew_letter(code), "Hebrew letter 0x{:04X}", code);
        }
    }

    #[test]
    fn test_letter_vs_diacritic_distinction() {
        // Ensure letters and diacritics are mutually exclusive
        assert!(is_arabic_letter(0x0628)); // BEH
        assert!(!is_arabic_diacritic(0x0628));

        assert!(is_arabic_diacritic(0x064E)); // FATHA
        assert!(!is_arabic_letter(0x064E));

        assert!(is_hebrew_letter(0x05D0)); // ALEF
        assert!(!is_hebrew_diacritic(0x05D0));

        assert!(is_hebrew_diacritic(0x05BC)); // DAGESH
        assert!(!is_hebrew_letter(0x05BC));
    }

    #[test]
    fn test_arabic_extended_letters() {
        // Arabic Supplement letters
        assert!(is_arabic_letter(0x0750)); // BEH WITH THREE DOTS HORIZONTALLY BELOW
        assert!(is_arabic_letter(0x0767)); // NOON WITH TWO DOTS BELOW

        // Arabic Extended-A letters
        assert!(is_arabic_letter(0x08A0)); // BEH WITH SMALL V BELOW
        assert!(is_arabic_letter(0x08B4)); // KAF WITH DOT BELOW
    }

    #[test]
    fn test_hebrew_punctuation() {
        assert!(is_hebrew_punctuation(0x05F3)); // GERESH
        assert!(is_hebrew_punctuation(0x05F4)); // GERSHAYIM

        assert!(!is_hebrew_punctuation(0x05D0)); // ALEF (letter)
        assert!(!is_hebrew_punctuation(0x05BC)); // DAGESH (diacritic)
    }
}

// ============================================================================
// NUMBER HANDLING TESTS
// ============================================================================

#[cfg(test)]
mod number_handling_tests {
    use super::*;

    #[test]
    fn test_eastern_arabic_digits() {
        // Eastern Arabic-Indic digits: U+06F0-U+06F9 (٠-٩)
        assert!(is_eastern_arabic_digit(0x06F0)); // ٠
        assert!(is_eastern_arabic_digit(0x06F1)); // ١
        assert!(is_eastern_arabic_digit(0x06F2)); // ٢
        assert!(is_eastern_arabic_digit(0x06F3)); // ٣
        assert!(is_eastern_arabic_digit(0x06F4)); // ٤
        assert!(is_eastern_arabic_digit(0x06F5)); // ٥
        assert!(is_eastern_arabic_digit(0x06F6)); // ٦
        assert!(is_eastern_arabic_digit(0x06F7)); // ٧
        assert!(is_eastern_arabic_digit(0x06F8)); // ٨
        assert!(is_eastern_arabic_digit(0x06F9)); // ٩

        assert!(!is_eastern_arabic_digit(0x0030)); // Western '0'
        assert!(!is_eastern_arabic_digit(0x0627)); // ALEF
    }

    #[test]
    fn test_arabic_number_detection() {
        // Eastern Arabic digits
        assert!(is_arabic_number(0x06F0)); // ٠
        assert!(is_arabic_number(0x06F5)); // ٥

        // Western digits (used in RTL context)
        assert!(is_arabic_number(0x0030)); // '0'
        assert!(is_arabic_number(0x0035)); // '5'
        assert!(is_arabic_number(0x0039)); // '9'

        // Not numbers
        assert!(!is_arabic_number(0x0627)); // ALEF
        assert!(!is_arabic_number(0x0041)); // 'A'
    }

    #[test]
    fn test_no_boundary_within_number_sequence() {
        let context = make_context();

        // Western digits
        let digit1 = make_char(0x0031, 100.0, None); // '1'
        let digit2 = make_char(0x0032, 105.0, None); // '2'
        let digit3 = make_char(0x0033, 110.0, None); // '3'

        assert_eq!(should_split_at_rtl_boundary(&digit1, &digit2, Some(&context)), Some(false));
        assert_eq!(should_split_at_rtl_boundary(&digit2, &digit3, Some(&context)), Some(false));

        // Eastern Arabic digits
        let e_digit1 = make_char(0x06F1, 100.0, None); // ١
        let e_digit2 = make_char(0x06F2, 105.0, None); // ٢

        assert_eq!(should_split_at_rtl_boundary(&e_digit1, &e_digit2, Some(&context)), Some(false));
    }

    #[test]
    fn test_mixed_digit_sequences() {
        let context = make_context();

        // Mixed Western and Eastern Arabic (unusual but valid)
        let western = make_char(0x0031, 100.0, None); // '1'
        let eastern = make_char(0x06F2, 105.0, None); // ٢

        // Should not split within number sequences
        assert_eq!(should_split_at_rtl_boundary(&western, &eastern, Some(&context)), Some(false));
    }
}

// ============================================================================
// BOUNDARY RULE TESTS
// ============================================================================

#[cfg(test)]
mod boundary_rules_tests {
    use super::*;

    #[test]
    fn test_space_creates_boundary() {
        let context = make_context();

        let letter = make_char(0x0628, 100.0, None); // BEH
        let space = make_char(0x0020, 105.0, None); // SPACE
        let next_letter = make_char(0x062A, 110.0, None); // TEH

        // Space always creates boundary
        assert_eq!(should_split_at_rtl_boundary(&letter, &space, Some(&context)), Some(true));
        assert_eq!(should_split_at_rtl_boundary(&space, &next_letter, Some(&context)), Some(true));
    }

    #[test]
    fn test_tatweel_preserves_word() {
        let context = make_context();

        let letter1 = make_char(0x0628, 100.0, None); // BEH
        let tatweel = make_char(0x0640, 105.0, None); // TATWEEL (kashida)
        let letter2 = make_char(0x062A, 110.0, None); // TEH

        // TATWEEL does not create boundary
        assert_eq!(should_split_at_rtl_boundary(&letter1, &tatweel, Some(&context)), Some(false));
        assert_eq!(should_split_at_rtl_boundary(&tatweel, &letter2, Some(&context)), Some(false));
    }

    #[test]
    fn test_tj_offset_large_negative_creates_boundary() {
        let context = make_context();

        let letter1 = make_char(0x0628, 100.0, Some(-60)); // BEH with large negative offset
        let letter2 = make_char(0x062A, 95.0, None); // TEH

        // Large negative TJ offset (< -50) creates boundary in RTL
        assert_eq!(should_split_at_rtl_boundary(&letter1, &letter2, Some(&context)), Some(true));
    }

    #[test]
    fn test_tj_offset_small_negative_no_boundary() {
        let context = make_context();

        let letter1 = make_char(0x0628, 100.0, Some(-30)); // BEH with small negative offset
        let letter2 = make_char(0x062A, 97.0, None); // TEH

        // Small negative TJ offset does not create boundary
        assert_eq!(should_split_at_rtl_boundary(&letter1, &letter2, Some(&context)), Some(false));
    }

    #[test]
    fn test_rtl_to_ltr_transition() {
        let context = make_context();

        let arabic = make_char(0x0628, 100.0, None); // BEH (RTL)
        let latin = make_char(0x0041, 105.0, None); // 'A' (LTR)

        // RTL-to-LTR transition creates boundary
        assert_eq!(should_split_at_rtl_boundary(&arabic, &latin, Some(&context)), Some(true));
    }

    #[test]
    fn test_ltr_to_rtl_transition() {
        let context = make_context();

        let latin = make_char(0x0041, 100.0, None); // 'A' (LTR)
        let arabic = make_char(0x0628, 105.0, None); // BEH (RTL)

        // LTR-to-RTL transition creates boundary
        assert_eq!(should_split_at_rtl_boundary(&latin, &arabic, Some(&context)), Some(true));
    }

    #[test]
    fn test_arabic_punctuation_boundaries() {
        let context = make_context();

        let letter = make_char(0x0628, 100.0, None); // BEH
        let comma = make_char(0x060C, 105.0, None); // ARABIC COMMA
        let semicolon = make_char(0x061B, 110.0, None); // ARABIC SEMICOLON
        let question = make_char(0x061F, 115.0, None); // ARABIC QUESTION MARK

        // Punctuation creates boundaries
        assert_eq!(should_split_at_rtl_boundary(&letter, &comma, Some(&context)), Some(true));
        assert_eq!(should_split_at_rtl_boundary(&letter, &semicolon, Some(&context)), Some(true));
        assert_eq!(should_split_at_rtl_boundary(&letter, &question, Some(&context)), Some(true));
    }

    #[test]
    fn test_hebrew_punctuation_boundaries() {
        let context = make_context();

        let letter = make_char(0x05D0, 100.0, None); // ALEF
        let geresh = make_char(0x05F3, 105.0, None); // GERESH
        let gershayim = make_char(0x05F4, 110.0, None); // GERSHAYIM

        // Hebrew punctuation creates boundaries
        assert_eq!(should_split_at_rtl_boundary(&letter, &geresh, Some(&context)), Some(true));
        assert_eq!(should_split_at_rtl_boundary(&letter, &gershayim, Some(&context)), Some(true));
    }

    #[test]
    fn test_normal_letter_sequence_no_boundary() {
        let context = make_context();

        // Arabic letter sequence
        let beh = make_char(0x0628, 100.0, None); // BEH
        let teh = make_char(0x062A, 105.0, None); // TEH
        let seen = make_char(0x0633, 110.0, None); // SEEN

        assert_eq!(should_split_at_rtl_boundary(&beh, &teh, Some(&context)), Some(false));
        assert_eq!(should_split_at_rtl_boundary(&teh, &seen, Some(&context)), Some(false));

        // Hebrew letter sequence
        let alef = make_char(0x05D0, 100.0, None); // ALEF
        let bet = make_char(0x05D1, 105.0, None); // BET
        let gimel = make_char(0x05D2, 110.0, None); // GIMEL

        assert_eq!(should_split_at_rtl_boundary(&alef, &bet, Some(&context)), Some(false));
        assert_eq!(should_split_at_rtl_boundary(&bet, &gimel, Some(&context)), Some(false));
    }

    #[test]
    fn test_non_rtl_characters_return_none() {
        let context = make_context();

        let latin1 = make_char(0x0041, 100.0, None); // 'A'
        let latin2 = make_char(0x0042, 105.0, None); // 'B'

        // Should return None for non-RTL characters (let other detectors handle)
        assert_eq!(should_split_at_rtl_boundary(&latin1, &latin2, Some(&context)), None);
    }
}

// ============================================================================
// LIGATURE TESTS
// ============================================================================

#[cfg(test)]
mod ligature_tests {
    use super::*;

    #[test]
    fn test_lam_alef_ligature_detection() {
        // LAM-ALEF ligatures: U+FEFB, U+FEFC, U+FEF5-U+FEFA
        assert!(is_lam_alef_ligature(0xFEFB)); // LAM WITH ALEF ISOLATED FORM
        assert!(is_lam_alef_ligature(0xFEFC)); // LAM WITH ALEF FINAL FORM
        assert!(is_lam_alef_ligature(0xFEF5)); // LAM WITH ALEF WITH MADDA ABOVE ISOLATED
        assert!(is_lam_alef_ligature(0xFEF6)); // LAM WITH ALEF WITH MADDA ABOVE FINAL
        assert!(is_lam_alef_ligature(0xFEF7)); // LAM WITH ALEF WITH HAMZA ABOVE ISOLATED
        assert!(is_lam_alef_ligature(0xFEF8)); // LAM WITH ALEF WITH HAMZA ABOVE FINAL
        assert!(is_lam_alef_ligature(0xFEF9)); // LAM WITH ALEF WITH HAMZA BELOW ISOLATED
        assert!(is_lam_alef_ligature(0xFEFA)); // LAM WITH ALEF WITH HAMZA BELOW FINAL

        // Not LAM-ALEF ligatures
        assert!(!is_lam_alef_ligature(0x0644)); // LAM (not ligature)
        assert!(!is_lam_alef_ligature(0x0627)); // ALEF (not ligature)
        assert!(!is_lam_alef_ligature(0xFB50)); // Other presentation form
    }

    #[test]
    fn test_lam_alef_decomposition() {
        // Test all LAM-ALEF ligature decompositions
        assert_eq!(decompose_lam_alef(0xFEFB), Some((0x0644, 0x0627))); // LAM + ALEF
        assert_eq!(decompose_lam_alef(0xFEFC), Some((0x0644, 0x0627))); // LAM + ALEF
        assert_eq!(decompose_lam_alef(0xFEF5), Some((0x0644, 0x0622))); // LAM + ALEF WITH MADDA
        assert_eq!(decompose_lam_alef(0xFEF6), Some((0x0644, 0x0622))); // LAM + ALEF WITH MADDA
        assert_eq!(decompose_lam_alef(0xFEF7), Some((0x0644, 0x0623))); // LAM + ALEF WITH HAMZA ABOVE
        assert_eq!(decompose_lam_alef(0xFEF8), Some((0x0644, 0x0623))); // LAM + ALEF WITH HAMZA ABOVE
        assert_eq!(decompose_lam_alef(0xFEF9), Some((0x0644, 0x0625))); // LAM + ALEF WITH HAMZA BELOW
        assert_eq!(decompose_lam_alef(0xFEFA), Some((0x0644, 0x0625))); // LAM + ALEF WITH HAMZA BELOW

        // Non-ligatures return None
        assert_eq!(decompose_lam_alef(0x0644), None); // LAM
        assert_eq!(decompose_lam_alef(0x0627), None); // ALEF
        assert_eq!(decompose_lam_alef(0x0041), None); // 'A'
    }

    #[test]
    fn test_presentation_form_normalization() {
        // Test contextual form normalization
        // Presentation Forms-A
        assert_eq!(normalize_arabic_contextual_form(0xFB50), 0x0671); // ALEF WASLA
        assert_eq!(normalize_arabic_contextual_form(0xFE82), 0x0627); // ALEF FINAL FORM -> ALEF

        // Presentation Forms-B
        assert_eq!(normalize_arabic_contextual_form(0xFE8F), 0x0628); // BEH ISOLATED -> BEH
        assert_eq!(normalize_arabic_contextual_form(0xFE90), 0x0628); // BEH FINAL -> BEH
        assert_eq!(normalize_arabic_contextual_form(0xFE91), 0x0628); // BEH INITIAL -> BEH
        assert_eq!(normalize_arabic_contextual_form(0xFE92), 0x0628); // BEH MEDIAL -> BEH

        // Non-presentation forms return unchanged
        assert_eq!(normalize_arabic_contextual_form(0x0628), 0x0628); // BEH
        assert_eq!(normalize_arabic_contextual_form(0x0041), 0x0041); // 'A'
    }

    #[test]
    fn test_ligature_boundary_behavior() {
        let context = make_context();

        // LAM-ALEF ligature should not split from following letter
        let ligature = make_char(0xFEFC, 100.0, None); // LAM-ALEF
        let letter = make_char(0x062A, 105.0, None); // TEH

        // Normal letter sequence behavior
        assert_eq!(should_split_at_rtl_boundary(&ligature, &letter, Some(&context)), Some(false));
    }
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_arabic_word_with_diacritics() {
        let context = make_context();

        // كِتَابٌ (kitaabun - "a book")
        // KAF + KASRA + TEH + FATHA + ALEF + BEH + DAMMATAN
        let chars = vec![
            make_char(0x0643, 100.0, None), // KAF
            make_char(0x0650, 100.0, None), // KASRA
            make_char(0x062A, 105.0, None), // TEH
            make_char(0x064E, 105.0, None), // FATHA
            make_char(0x0627, 110.0, None), // ALEF
            make_char(0x0628, 115.0, None), // BEH
            make_char(0x064C, 115.0, None), // DAMMATAN
        ];

        // Verify no boundaries within the word
        for i in 0..chars.len() - 1 {
            let result = should_split_at_rtl_boundary(&chars[i], &chars[i + 1], Some(&context));
            assert_eq!(result, Some(false), "Should not split at position {}", i);
        }
    }

    #[test]
    fn test_arabic_sentence_with_spaces() {
        let context = make_context();

        // مَرْحَبًا بِكَ (marhabaan bika - "hello to you")
        // Word1: MEEM + FATHA + REH + SUKUN + HAH + FATHA + BEH + FATHATAN
        // SPACE
        // Word2: BEH + KASRA + KAF + FATHA

        let word1_last = make_char(0x064C, 130.0, None); // FATHATAN
        let space = make_char(0x0020, 135.0, None); // SPACE
        let word2_first = make_char(0x0628, 140.0, None); // BEH

        // Space should create boundary
        assert_eq!(should_split_at_rtl_boundary(&word1_last, &space, Some(&context)), Some(true));
        assert_eq!(should_split_at_rtl_boundary(&space, &word2_first, Some(&context)), Some(true));
    }

    #[test]
    fn test_hebrew_word_with_vowels() {
        let context = make_context();

        // שָׁלוֹם (shalom - "peace")
        // SHIN + SHIN DOT + QAMATS + LAMED + HOLAM + FINAL MEM
        let chars = vec![
            make_char(0x05E9, 100.0, None), // SHIN
            make_char(0x05C1, 100.0, None), // SHIN DOT
            make_char(0x05B8, 100.0, None), // QAMATS
            make_char(0x05DC, 105.0, None), // LAMED
            make_char(0x05B9, 105.0, None), // HOLAM
            make_char(0x05DD, 110.0, None), // FINAL MEM
        ];

        // Verify no boundaries within the word
        for i in 0..chars.len() - 1 {
            let result = should_split_at_rtl_boundary(&chars[i], &chars[i + 1], Some(&context));
            assert_eq!(result, Some(false), "Should not split at position {}", i);
        }
    }

    #[test]
    fn test_mixed_rtl_and_ltr() {
        let context = make_context();

        // Arabic word + space + English word
        let arabic_letter = make_char(0x0628, 100.0, None); // BEH
        let space = make_char(0x0020, 105.0, None); // SPACE
        let latin_letter = make_char(0x0041, 110.0, None); // 'A'

        // Boundary at space
        assert_eq!(
            should_split_at_rtl_boundary(&arabic_letter, &space, Some(&context)),
            Some(true)
        );

        // Boundary at script transition
        assert_eq!(should_split_at_rtl_boundary(&space, &latin_letter, Some(&context)), Some(true));
    }

    #[test]
    fn test_arabic_with_numbers() {
        let context = make_context();

        // Text: السنة ٢٠٢٥ ("the year 2025" in Eastern Arabic digits)
        let word_end = make_char(0x0629, 100.0, None); // TEH MARBUTA
        let space = make_char(0x0020, 105.0, None); // SPACE
        let digit1 = make_char(0x06F2, 110.0, None); // ٢
        let digit2 = make_char(0x06F0, 115.0, None); // ٠
        let digit3 = make_char(0x06F2, 120.0, None); // ٢
        let digit4 = make_char(0x06F5, 125.0, None); // ٥

        // Boundary at space before number
        assert_eq!(should_split_at_rtl_boundary(&word_end, &space, Some(&context)), Some(true));
        assert_eq!(should_split_at_rtl_boundary(&space, &digit1, Some(&context)), Some(true));

        // No boundaries within number sequence
        assert_eq!(should_split_at_rtl_boundary(&digit1, &digit2, Some(&context)), Some(false));
        assert_eq!(should_split_at_rtl_boundary(&digit2, &digit3, Some(&context)), Some(false));
        assert_eq!(should_split_at_rtl_boundary(&digit3, &digit4, Some(&context)), Some(false));
    }

    #[test]
    fn test_presentation_forms_in_context() {
        let context = make_context();

        // Using presentation forms (as they appear in some PDFs)
        let alef_final = make_char(0xFE82, 100.0, None); // ALEF FINAL FORM
        let beh_initial = make_char(0xFE91, 105.0, None); // BEH INITIAL FORM

        // Should not create boundary
        assert_eq!(
            should_split_at_rtl_boundary(&alef_final, &beh_initial, Some(&context)),
            Some(false)
        );
    }
}
