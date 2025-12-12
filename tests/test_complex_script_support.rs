//! Week 3 Days 11-12: Complex Script Support Test Suite
//!
//! Comprehensive tests for Devanagari, Thai, Khmer, and South Asian script support.

use pdf_oxide::text::complex_script_detector::*;
use pdf_oxide::text::{BoundaryContext, CharacterInfo, WordBoundaryDetector};

/// Helper function to create CharacterInfo for testing
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

// ============================================================================
// SCRIPT DETECTION TESTS
// ============================================================================

#[cfg(test)]
mod script_detection_tests {
    use super::*;

    #[test]
    fn test_detect_devanagari_range() {
        // U+0900-U+097F
        assert_eq!(detect_complex_script(0x0915), Some(ComplexScript::Devanagari)); // क (KA)
        assert_eq!(detect_complex_script(0x0928), Some(ComplexScript::Devanagari)); // न (NA)
        assert_eq!(detect_complex_script(0x0971), Some(ComplexScript::Devanagari));
        assert_eq!(detect_complex_script(0x0900), Some(ComplexScript::Devanagari));
        assert_eq!(detect_complex_script(0x097F), Some(ComplexScript::Devanagari));
    }

    #[test]
    fn test_detect_thai_range() {
        // U+0E00-U+0E7F
        assert_eq!(detect_complex_script(0x0E01), Some(ComplexScript::Thai)); // ก
        assert_eq!(detect_complex_script(0x0E3F), Some(ComplexScript::Thai));
        assert_eq!(detect_complex_script(0x0E00), Some(ComplexScript::Thai));
        assert_eq!(detect_complex_script(0x0E7F), Some(ComplexScript::Thai));
    }

    #[test]
    fn test_detect_khmer_range() {
        // U+1780-U+17FF
        assert_eq!(detect_complex_script(0x1780), Some(ComplexScript::Khmer)); // ក
        assert_eq!(detect_complex_script(0x17FF), Some(ComplexScript::Khmer));
        assert_eq!(detect_complex_script(0x1790), Some(ComplexScript::Khmer));
    }

    #[test]
    fn test_detect_tamil_range() {
        // U+0B80-U+0BFF
        assert_eq!(detect_complex_script(0x0B85), Some(ComplexScript::Tamil)); // அ
        assert_eq!(detect_complex_script(0x0BBF), Some(ComplexScript::Tamil));
        assert_eq!(detect_complex_script(0x0B80), Some(ComplexScript::Tamil));
        assert_eq!(detect_complex_script(0x0BFF), Some(ComplexScript::Tamil));
    }

    #[test]
    fn test_detect_telugu_range() {
        // U+0C00-U+0C7F
        assert_eq!(detect_complex_script(0x0C05), Some(ComplexScript::Telugu));
        assert_eq!(detect_complex_script(0x0C3E), Some(ComplexScript::Telugu));
        assert_eq!(detect_complex_script(0x0C00), Some(ComplexScript::Telugu));
    }

    #[test]
    fn test_detect_kannada_range() {
        // U+0C80-U+0CFF
        assert_eq!(detect_complex_script(0x0C85), Some(ComplexScript::Kannada));
        assert_eq!(detect_complex_script(0x0CBF), Some(ComplexScript::Kannada));
    }

    #[test]
    fn test_detect_malayalam_range() {
        // U+0D00-U+0D7F
        assert_eq!(detect_complex_script(0x0D05), Some(ComplexScript::Malayalam));
        assert_eq!(detect_complex_script(0x0D3E), Some(ComplexScript::Malayalam));
    }

    #[test]
    fn test_detect_non_complex_script() {
        assert_eq!(detect_complex_script(0x0041), None); // Latin 'A'
        assert_eq!(detect_complex_script(0x0020), None); // Space
        assert_eq!(detect_complex_script(0x4E00), None); // CJK Han
    }

    #[test]
    fn test_is_complex_script_helper() {
        assert!(is_complex_script(0x0915)); // Devanagari
        assert!(is_complex_script(0x0E01)); // Thai
        assert!(is_complex_script(0x1780)); // Khmer
        assert!(!is_complex_script(0x0041)); // Latin
    }
}

// ============================================================================
// DEVANAGARI BOUNDARY TESTS
// ============================================================================

#[cfg(test)]
mod devanagari_boundary_tests {
    use super::*;

    #[test]
    fn test_virama_consonant_no_boundary() {
        // Virama + consonant = conjunct (no boundary)
        let chars = vec![
            make_char(0x094D, 0.0, None), // ्
            make_char(0x0937, 0.5, None), // ष (SHA)
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        assert!(
            !boundaries.contains(&1),
            "Virama should not create boundary with following consonant"
        );
    }

    #[test]
    fn test_matra_no_boundary() {
        // Matras (vowel modifiers) don't create boundaries
        let chars = vec![
            make_char(0x0915, 0.0, None), // क (KA)
            make_char(0x0940, 0.5, None), // ी (II matra)
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        assert!(!boundaries.contains(&1), "Matra should not create boundary");
    }

    #[test]
    fn test_nukta_no_boundary() {
        // Nukta modifies preceding character
        let chars = vec![
            make_char(0x0916, 0.0, None), // ख (KHA)
            make_char(0x093C, 0.5, None), // ़ (NUKTA)
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        assert!(!boundaries.contains(&1), "Nukta should not create boundary");
    }

    #[test]
    fn test_anusvara_no_boundary() {
        // Anusvara attaches to syllable
        let chars = vec![
            make_char(0x0928, 0.0, None), // न (NA)
            make_char(0x0902, 0.5, None), // ं (ANUSVARA)
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        assert!(!boundaries.contains(&1), "Anusvara should not create boundary");
    }

    #[test]
    fn test_hindi_word_namaste() {
        // नमस्ते (namaste): न + म + स + ् + त + े
        let chars = vec![
            make_char(0x0928, 0.0, None), // न (NA)
            make_char(0x092E, 1.0, None), // म (MA)
            make_char(0x0938, 2.0, None), // स (SA)
            make_char(0x094D, 3.0, None), // ् (VIRAMA)
            make_char(0x0924, 3.5, None), // त (TA)
            make_char(0x0947, 4.0, None), // े (E matra)
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        // Should not split at virama or matra positions
        assert!(!boundaries.contains(&3), "Should not split at virama (index 3)");
        assert!(!boundaries.contains(&4), "Should not split after virama (index 4)");
        assert!(!boundaries.contains(&5), "Should not split at matra (index 5)");
    }

    #[test]
    fn test_hindi_word_bharat() {
        // भारत (Bharat): भ + ा + र + त
        let chars = vec![
            make_char(0x092D, 0.0, None), // भ (BHA)
            make_char(0x093E, 0.5, None), // ा (AA matra)
            make_char(0x0930, 1.0, None), // र (RA)
            make_char(0x0924, 1.5, None), // त (TA)
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        // Should not split at matra
        assert!(!boundaries.contains(&1), "Should not split at matra");
    }

    #[test]
    fn test_conjunct_consonant_क्ष() {
        // क्ष (ksha): क + ् + ष
        let chars = vec![
            make_char(0x0915, 0.0, None), // क (KA)
            make_char(0x094D, 0.5, None), // ् (VIRAMA)
            make_char(0x0937, 0.7, None), // ष (SHA)
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        // Should not split the conjunct
        assert!(!boundaries.contains(&1), "Should not split at virama");
        assert!(!boundaries.contains(&2), "Should not split after virama");
    }

    #[test]
    fn test_multiple_diacritics_no_boundary() {
        // Base + multiple diacritics
        let chars = vec![
            make_char(0x0915, 0.0, None), // क (KA)
            make_char(0x093C, 0.3, None), // ़ (NUKTA)
            make_char(0x0940, 0.6, None), // ी (II matra)
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        // Should not split between diacritics
        assert!(!boundaries.contains(&1), "Should not split at nukta");
        assert!(!boundaries.contains(&2), "Should not split at matra");
    }

    #[test]
    fn test_devanagari_helper_functions() {
        // Test individual helper functions
        assert!(is_devanagari_virama(0x094D));
        assert!(is_devanagari_matra(0x093E));
        assert!(is_devanagari_matra(0x0940));
        assert!(is_devanagari_nukta(0x093C));
        assert!(is_devanagari_anusvar_visarga(0x0902));
        assert!(is_devanagari_anusvar_visarga(0x0903));
        assert!(is_devanagari_consonant(0x0915));
        assert!(is_devanagari_diacritic(0x094D));
    }
}

// ============================================================================
// THAI BOUNDARY TESTS
// ============================================================================

#[cfg(test)]
mod thai_boundary_tests {
    use super::*;

    #[test]
    fn test_tone_mark_no_boundary() {
        // Tone marks never create boundaries
        let chars = vec![
            make_char(0x0E01, 0.0, None), // ก (KO KAI)
            make_char(0x0E48, 0.5, None), // ่ (MAI EK)
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        assert!(!boundaries.contains(&1), "Tone mark should not create boundary");
    }

    #[test]
    fn test_vowel_modifier_no_boundary() {
        // Vowel modifiers attach to consonants
        let chars = vec![
            make_char(0x0E01, 0.0, None), // ก (KO KAI)
            make_char(0x0E31, 0.5, None), // ั (MAI HAN-AKAT)
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        assert!(!boundaries.contains(&1), "Vowel modifier should not create boundary");
    }

    #[test]
    fn test_thai_digit_sequences() {
        // Thai and Western digits can mix
        let chars = vec![
            make_char(0x0E50, 0.0, None), // ๐
            make_char(0x0031, 0.5, None), // 1
            make_char(0x0E52, 1.0, None), // ๒
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        assert!(!boundaries.contains(&1), "Should not split Thai-Western digit sequence");
        assert!(!boundaries.contains(&2), "Should not split Western-Thai digit sequence");
    }

    #[test]
    fn test_thai_word_with_tone_and_vowel() {
        // Thai syllable: consonant + vowel + tone
        let chars = vec![
            make_char(0x0E01, 0.0, None), // ก (KO KAI)
            make_char(0x0E31, 0.3, None), // ั (vowel)
            make_char(0x0E48, 0.6, None), // ่ (tone mark)
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        // Should not split within syllable
        assert!(!boundaries.contains(&1), "Should not split at vowel");
        assert!(!boundaries.contains(&2), "Should not split at tone mark");
    }

    #[test]
    fn test_thai_multiple_tone_marks() {
        // Base + tone mark + another tone mark (rare but possible)
        let chars = vec![
            make_char(0x0E01, 0.0, None), // ก
            make_char(0x0E48, 0.3, None), // ่
            make_char(0x0E49, 0.6, None), // ้
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        assert!(!boundaries.contains(&1), "Should not split at first tone mark");
        assert!(!boundaries.contains(&2), "Should not split at second tone mark");
    }

    #[test]
    fn test_thai_major_punctuation_creates_boundary() {
        // Major punctuation should create boundary
        let chars = vec![
            make_char(0x0E01, 0.0, None), // ก
            make_char(0x0E2F, 1.0, None), // ฯ (PAIYANNOI)
            make_char(0x0E02, 2.0, None), // ข
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        // Major punctuation creates boundaries
        assert!(
            boundaries.contains(&1) || boundaries.contains(&2),
            "Thai major punctuation should create boundary"
        );
    }

    #[test]
    fn test_thai_helper_functions() {
        // Test individual helper functions
        assert!(is_thai_tone_mark(0x0E48));
        assert!(is_thai_vowel_modifier(0x0E31));
        assert!(is_thai_digit(0x0E50));
        assert!(is_thai_digit(0x0031)); // Western digit
        assert!(is_thai_major_punctuation(0x0E2F));
    }
}

// ============================================================================
// KHMER BOUNDARY TESTS
// ============================================================================

#[cfg(test)]
mod khmer_boundary_tests {
    use super::*;

    #[test]
    fn test_coeng_subscript_consonant() {
        // COENG + consonant = subscript (no boundary)
        let chars = vec![
            make_char(0x1780, 0.0, None), // ក (KA)
            make_char(0x17D2, 0.5, None), // ្ (COENG)
            make_char(0x1799, 1.0, None), // ស (SA - subscript)
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        assert!(
            !boundaries.contains(&2),
            "COENG should not create boundary with following consonant"
        );
    }

    #[test]
    fn test_khmer_vowel_inherent() {
        // Vowel inherents don't create boundaries
        let chars = vec![
            make_char(0x1780, 0.0, None), // ក (KA)
            make_char(0x17BE, 0.5, None), // ើ (vowel)
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        assert!(!boundaries.contains(&1), "Khmer vowel should not create boundary");
    }

    #[test]
    fn test_khmer_tone_mark_no_boundary() {
        // Tone marks attach to syllables
        let chars = vec![
            make_char(0x1780, 0.0, None), // ក
            make_char(0x17C9, 0.5, None), // ៉ (MUUSIKATOAN)
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        assert!(!boundaries.contains(&1), "Khmer tone mark should not create boundary");
    }

    #[test]
    fn test_khmer_syllable_with_coeng_and_vowel() {
        // Complex Khmer syllable: consonant + COENG + consonant + vowel
        let chars = vec![
            make_char(0x1780, 0.0, None), // ក
            make_char(0x17D2, 0.3, None), // ្ (COENG)
            make_char(0x1799, 0.6, None), // ស (subscript)
            make_char(0x17BE, 0.9, None), // ើ (vowel)
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        // Should not split within the complex syllable
        assert!(!boundaries.contains(&2), "Should not split after COENG");
        assert!(!boundaries.contains(&3), "Should not split at vowel");
    }

    #[test]
    fn test_khmer_nikahit() {
        // NIKAHIT (ំ) is a vowel sign
        let chars = vec![
            make_char(0x1780, 0.0, None), // ក
            make_char(0x17C6, 0.5, None), // ំ (NIKAHIT)
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        assert!(!boundaries.contains(&1), "NIKAHIT should not create boundary");
    }

    #[test]
    fn test_khmer_multiple_marks() {
        // Multiple marks on same base
        let chars = vec![
            make_char(0x1780, 0.0, None), // ក
            make_char(0x17BE, 0.3, None), // ើ (vowel)
            make_char(0x17C9, 0.6, None), // ៉ (tone)
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        assert!(!boundaries.contains(&1), "Should not split at vowel");
        assert!(!boundaries.contains(&2), "Should not split at tone mark");
    }

    #[test]
    fn test_khmer_helper_functions() {
        // Test individual helper functions
        assert!(is_khmer_coeng(0x17D2));
        assert!(is_khmer_vowel_inherent(0x17BE));
        assert!(is_khmer_vowel_inherent(0x17C6));
        assert!(is_khmer_tone_mark(0x17C9));
    }
}

// ============================================================================
// INDIC SCRIPTS (TAMIL, TELUGU, KANNADA, MALAYALAM) TESTS
// ============================================================================

#[cfg(test)]
mod indic_scripts_tests {
    use super::*;

    #[test]
    fn test_tamil_matra_no_boundary() {
        // Tamil consonant + matra
        let chars = vec![
            make_char(0x0B95, 0.0, None), // க (KA)
            make_char(0x0BBE, 0.5, None), // ா (AA)
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        assert!(!boundaries.contains(&1), "Tamil matra should not create boundary");
    }

    #[test]
    fn test_tamil_virama_no_boundary() {
        // Tamil virama
        let chars = vec![
            make_char(0x0B95, 0.0, None), // க
            make_char(0x0BCD, 0.5, None), // ் (VIRAMA)
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        assert!(!boundaries.contains(&1), "Tamil virama should not create boundary");
    }

    #[test]
    fn test_telugu_matras_no_boundary() {
        // Telugu consonant + matra
        let chars = vec![
            make_char(0x0C15, 0.0, None), // క (KA)
            make_char(0x0C3E, 0.5, None), // ా (AA)
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        assert!(!boundaries.contains(&1), "Telugu matra should not create boundary");
    }

    #[test]
    fn test_kannada_virama_no_boundary() {
        // Kannada virama
        let chars = vec![
            make_char(0x0C95, 0.0, None), // ಕ (KA)
            make_char(0x0CCD, 0.5, None), // ್ (VIRAMA)
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        assert!(!boundaries.contains(&1), "Kannada virama should not create boundary");
    }

    #[test]
    fn test_malayalam_matra_no_boundary() {
        // Malayalam consonant + matra
        let chars = vec![
            make_char(0x0D15, 0.0, None), // ക (KA)
            make_char(0x0D3E, 0.5, None), // ാ (AA)
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        assert!(!boundaries.contains(&1), "Malayalam matra should not create boundary");
    }

    #[test]
    fn test_bengali_virama_no_boundary() {
        // Bengali virama
        let chars = vec![
            make_char(0x0995, 0.0, None), // ক (KA)
            make_char(0x09CD, 0.5, None), // ্ (VIRAMA)
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        assert!(!boundaries.contains(&1), "Bengali virama should not create boundary");
    }

    #[test]
    fn test_indic_diacritic_detection() {
        // Test is_indic_diacritic helper
        assert!(is_indic_diacritic(0x09CD)); // Bengali virama
        assert!(is_indic_diacritic(0x0BCD)); // Tamil virama
        assert!(is_indic_diacritic(0x0C4D)); // Telugu virama
        assert!(is_indic_diacritic(0x0CCD)); // Kannada virama
        assert!(is_indic_diacritic(0x0D4D)); // Malayalam virama

        // Matras
        assert!(is_indic_diacritic(0x09BE)); // Bengali AA
        assert!(is_indic_diacritic(0x0BBE)); // Tamil AA
        assert!(is_indic_diacritic(0x0C3E)); // Telugu AA
        assert!(is_indic_diacritic(0x0CBE)); // Kannada AA
        assert!(is_indic_diacritic(0x0D3E)); // Malayalam AA
    }
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_hindi_sentence_extraction() {
        // "नमस्ते भारत" (Namaste Bharat) with space
        let chars = vec![
            // नमस्ते
            make_char(0x0928, 0.0, None), // न
            make_char(0x092E, 1.0, None), // म
            make_char(0x0938, 2.0, None), // स
            make_char(0x094D, 3.0, None), // ्
            make_char(0x0924, 3.5, None), // त
            make_char(0x0947, 4.0, None), // े
            // Space
            make_char(0x0020, 5.0, None), // SPACE
            // भारत
            make_char(0x092D, 6.0, None), // भ
            make_char(0x093E, 6.5, None), // ा
            make_char(0x0930, 7.0, None), // र
            make_char(0x0924, 7.5, None), // त
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        // Should have boundary at space (index 7) but not within words
        assert!(boundaries.contains(&7), "Should have boundary at space");
        assert!(!boundaries.contains(&3), "Should not split at virama");
        assert!(!boundaries.contains(&5), "Should not split at matra");
        assert!(!boundaries.contains(&8), "Should not split at matra in second word");
    }

    #[test]
    fn test_thai_text_no_explicit_spaces() {
        // Thai doesn't use spaces between words typically
        // สวัสดี (hello): ส + ว + ั + ส + ด + ี
        let chars = vec![
            make_char(0x0E2A, 0.0, None), // ส (SO RUSI)
            make_char(0x0E27, 1.0, None), // ว (WO WAEN)
            make_char(0x0E31, 1.3, None), // ั (vowel)
            make_char(0x0E2A, 2.0, None), // ส
            make_char(0x0E14, 3.0, None), // ด (DO DEK)
            make_char(0x0E35, 3.3, None), // ี (vowel)
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        // Should not split at vowel positions
        assert!(!boundaries.contains(&2), "Should not split at vowel");
        assert!(!boundaries.contains(&5), "Should not split at vowel");
    }

    #[test]
    fn test_khmer_complex_word() {
        // Complex Khmer word with multiple subscripts
        let chars = vec![
            make_char(0x1780, 0.0, None), // ក
            make_char(0x17D2, 0.3, None), // ្ (COENG)
            make_char(0x1799, 0.6, None), // ស (subscript)
            make_char(0x17BE, 0.9, None), // ើ (vowel)
            make_char(0x1784, 1.5, None), // ង (next consonant)
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        // Should not split the complex syllable
        assert!(!boundaries.contains(&2), "Should not split after COENG");
        assert!(!boundaries.contains(&3), "Should not split at vowel");
    }

    #[test]
    fn test_mixed_devanagari_english() {
        // "Hello भारत" - Mixed English and Devanagari
        let chars = vec![
            // Hello
            make_char(0x0048, 0.0, None), // H
            make_char(0x0065, 1.0, None), // e
            make_char(0x006C, 2.0, None), // l
            make_char(0x006C, 3.0, None), // l
            make_char(0x006F, 4.0, None), // o
            // Space
            make_char(0x0020, 5.0, None), // SPACE
            // भारत
            make_char(0x092D, 6.0, None), // भ
            make_char(0x093E, 6.5, None), // ा
            make_char(0x0930, 7.0, None), // र
            make_char(0x0924, 7.5, None), // त
        ];

        let context = BoundaryContext::new(12.0);
        let detector = WordBoundaryDetector::new();
        let boundaries = detector.detect_word_boundaries(&chars, &context);

        // Should have boundary at space
        assert!(boundaries.contains(&6), "Should have boundary at space");
        // Should not split Devanagari matras
        assert!(!boundaries.contains(&7), "Should not split Devanagari matra");
    }

    #[test]
    fn test_complex_script_mark_helper() {
        // Test is_complex_script_mark convenience function
        assert!(is_complex_script_mark(0x094D)); // Devanagari virama
        assert!(is_complex_script_mark(0x0940)); // Devanagari matra
        assert!(is_complex_script_mark(0x0E48)); // Thai tone mark
        assert!(is_complex_script_mark(0x0E31)); // Thai vowel modifier
        assert!(is_complex_script_mark(0x17D2)); // Khmer COENG
        assert!(is_complex_script_mark(0x17BE)); // Khmer vowel
        assert!(is_complex_script_mark(0x0BCD)); // Tamil virama
        assert!(!is_complex_script_mark(0x0041)); // Latin A
    }
}
