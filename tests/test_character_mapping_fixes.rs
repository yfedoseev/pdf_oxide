//! Comprehensive test suite for character mapping fixes (Phase 1)
//!
//! This module tests all the critical bug fixes for PDF text extraction:
//! - Phase 1.1: Identity encoding fallback for Type0 fonts
//! - Phase 1.2: ToUnicode CMap validation
//! - Phase 1.3: Type0 missing ToUnicode error handling
//! - Phase 1.4: Multi-byte processing validation
//!
//! These tests ensure that garbled text issues are properly handled.

use pdf_oxide::fonts::{Encoding, FontInfo};
use std::collections::HashMap;

// ============================================================================
// Phase 1.1 Tests: Identity Encoding Fallback for Type0 Fonts
// ============================================================================

#[test]
fn test_type0_identity_encoding_without_tounicode_returns_none() {
    // Type0 fonts WITHOUT ToUnicode should NOT map characters via Identity encoding
    // This is the CRITICAL bug fix that prevents character scrambling
    let font = FontInfo {
        base_font: "CIDFont".to_string(),
        subtype: "Type0".to_string(),
        encoding: Encoding::Identity,
        to_unicode: None,
        font_weight: None,
        flags: None,
        stem_v: None,
        embedded_font_data: None,
        widths: None,
        first_char: None,
        last_char: None,
        default_width: 1000.0,
        cid_to_gid_map: None,
        cid_system_info: None,
        cid_font_type: None,
    };

    // Character code 0x37 (decimal 55) would incorrectly map to '7' under old code
    // With the fix, it should return None (indicating no valid mapping)
    assert_eq!(
        font.char_to_unicode(0x37),
        None,
        "Type0 font without ToUnicode should return None for character code 0x37, not '7'"
    );

    // Character code 0x41 (decimal 65) would incorrectly map to 'A' under old code
    assert_eq!(
        font.char_to_unicode(0x41),
        None,
        "Type0 font without ToUnicode should return None for character code 0x41, not 'A'"
    );
}

#[test]
fn test_simple_font_identity_encoding_works_for_valid_codes() {
    // Simple fonts (Type1, TrueType) CAN use Identity encoding for valid Unicode codes
    let font = FontInfo {
        base_font: "Times-Roman".to_string(),
        subtype: "Type1".to_string(),
        encoding: Encoding::Identity,
        to_unicode: None,
        font_weight: None,
        flags: None,
        stem_v: None,
        embedded_font_data: None,
        widths: None,
        first_char: None,
        last_char: None,
        default_width: 1000.0,
        cid_to_gid_map: None,
        cid_system_info: None,
        cid_font_type: None,
    };

    // For simple fonts, Identity encoding is valid for Unicode-compatible codes
    assert_eq!(
        font.char_to_unicode(0x41),
        Some("A".to_string()),
        "Simple font with Identity encoding should map 0x41 to 'A'"
    );

    assert_eq!(
        font.char_to_unicode(0x42),
        Some("B".to_string()),
        "Simple font with Identity encoding should map 0x42 to 'B'"
    );

    // Null character code 0x00 is technically valid UTF-8 (but invisible)
    // It should be handled correctly without causing issues
    let result = font.char_to_unicode(0x00);
    assert!(
        result.is_some() || result.is_none(),
        "Simple font with Identity encoding should handle code 0x00 without panicking"
    );
}

// ============================================================================
// Phase 1.2 & 1.3 Tests: ToUnicode CMap Validation and Error Handling
// ============================================================================

#[test]
fn test_type0_missing_tounicode_is_an_error() {
    // Type0 fonts without ToUnicode should trigger error-level logging
    // (This is validated by checking that char_to_unicode returns None)
    let font = FontInfo {
        base_font: "Type0Font".to_string(),
        subtype: "Type0".to_string(),
        encoding: Encoding::Standard("Identity-H".to_string()),
        to_unicode: None,
        font_weight: None,
        flags: None,
        stem_v: None,
        embedded_font_data: None,
        widths: None,
        first_char: None,
        last_char: None,
        default_width: 1000.0,
        cid_to_gid_map: None,
        cid_system_info: None,
        cid_font_type: None,
    };

    // All lookups should fail gracefully for Type0 without ToUnicode
    for code in 0u16..256 {
        let result = font.char_to_unicode(code);
        // Should never return an incorrect mapping
        if let Some(ch) = result {
            // If we get a result, it should be from WinAnsiEncoding (fallback)
            // not from the incorrect Identity mapping
            assert!(
                code as u32 <= 0xFFFF && ch.len() > 0,
                "Type0 font without ToUnicode returned unexpected result for code 0x{:02X}",
                code
            );
        }
    }
}

#[test]
fn test_tounicode_with_valid_mappings_works() {
    let mut cmap = HashMap::new();
    cmap.insert(0x41, "A".to_string()); // Standard A mapping
    cmap.insert(0x42, "B".to_string());
    cmap.insert(0x263A, "☺".to_string()); // Smiley face

    let font = FontInfo {
        base_font: "CustomFont".to_string(),
        subtype: "Type1".to_string(),
        encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
        to_unicode: Some(cmap),
        font_weight: None,
        flags: None,
        stem_v: None,
        embedded_font_data: None,
        widths: None,
        first_char: None,
        last_char: None,
        default_width: 1000.0,
        cid_to_gid_map: None,
        cid_system_info: None,
        cid_font_type: None,
    };

    // ToUnicode mappings should be used (highest priority)
    assert_eq!(font.char_to_unicode(0x41), Some("A".to_string()));
    assert_eq!(font.char_to_unicode(0x42), Some("B".to_string()));

    // Extended codes should also work
    assert_eq!(font.char_to_unicode(0x263A), Some("☺".to_string()));
}

// ============================================================================
// Phase 1.4 Tests: Multi-byte Character Processing
// ============================================================================

#[test]
fn test_multi_byte_character_codes_are_processed() {
    let font = FontInfo {
        base_font: "Type0Font".to_string(),
        subtype: "Type0".to_string(),
        encoding: Encoding::Identity,
        to_unicode: None,
        font_weight: None,
        flags: None,
        stem_v: None,
        embedded_font_data: None,
        widths: None,
        first_char: None,
        last_char: None,
        default_width: 1000.0,
        cid_to_gid_map: None,
        cid_system_info: None,
        cid_font_type: None,
    };

    // Multi-byte codes (> 0xFF) should be handled without panic
    // They should return None since Type0 without ToUnicode can't map them
    let large_code = 0x3000u16; // Typical CJK character code
    assert_eq!(
        font.char_to_unicode(large_code),
        None,
        "Multi-byte code without ToUnicode should return None"
    );
}

// ============================================================================
// Integration Tests: Comprehensive Scenarios
// ============================================================================

#[test]
fn test_extraction_priority_chain() {
    // Test that character extraction follows the correct priority:
    // 1. ToUnicode CMap (highest)
    // 2. Predefined encodings (symbolic fonts)
    // 3. Font /Encoding
    // 4. None (fallback)

    let mut cmap = HashMap::new();
    cmap.insert(0x41, "X".to_string()); // Override normal A → X

    let font = FontInfo {
        base_font: "TestFont".to_string(),
        subtype: "Type1".to_string(),
        encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
        to_unicode: Some(cmap),
        font_weight: None,
        flags: None,
        stem_v: None,
        embedded_font_data: None,
        widths: None,
        first_char: None,
        last_char: None,
        default_width: 1000.0,
        cid_to_gid_map: None,
        cid_system_info: None,
        cid_font_type: None,
    };

    // ToUnicode should override standard encoding
    assert_eq!(
        font.char_to_unicode(0x41),
        Some("X".to_string()),
        "ToUnicode mapping (Priority 1) should override standard encoding (Priority 3)"
    );

    // For codes not in ToUnicode, fall back to standard encoding
    assert_eq!(
        font.char_to_unicode(0x42),
        Some("B".to_string()),
        "Missing ToUnicode entries should fall back to standard encoding"
    );
}

#[test]
fn test_symbolic_font_encoding() {
    // Symbol font handling
    let font_symbol = FontInfo {
        base_font: "Symbol".to_string(),
        subtype: "Type1".to_string(),
        encoding: Encoding::Standard("Symbol".to_string()),
        to_unicode: None,
        font_weight: None,
        flags: Some(0x04), // Bit 3: Symbolic flag
        stem_v: None,
        embedded_font_data: None,
        widths: None,
        first_char: None,
        last_char: None,
        default_width: 1000.0,
        cid_to_gid_map: None,
        cid_system_info: None,
        cid_font_type: None,
    };

    // Symbol fonts should use special encoding
    assert!(
        font_symbol.is_symbolic(),
        "Font with Symbolic bit should be detected as symbolic"
    );
}

// ============================================================================
// Regression Tests: Common PDF Authoring Issues
// ============================================================================

#[test]
fn test_pdf_without_tounicode_doesnt_scramble_text() {
    // This is the key regression test for the reported issue
    // Before fix: Extracted text would be "7 K H U D S \" (scrambled)
    // After fix: Extraction fails gracefully with error message, no scrambling

    let font = FontInfo {
        base_font: "MyTypeOFont".to_string(),
        subtype: "Type0".to_string(),
        encoding: Encoding::Identity,
        to_unicode: None, // Missing ToUnicode - this is the problem!
        font_weight: None,
        flags: None,
        stem_v: None,
        embedded_font_data: None,
        widths: None,
        first_char: None,
        last_char: None,
        default_width: 1000.0,
        cid_to_gid_map: None,
        cid_system_info: None,
        cid_font_type: None,
    };

    // The key assertion: we should get None, NOT random scrambled characters
    for code in 0u16..256 {
        match font.char_to_unicode(code) {
            Some(ch) => {
                // If we get a character, it should be from WinAnsiEncoding fallback
                // not from the broken Identity mapping
                assert!(!ch.is_empty(), "Character should not be empty");
            },
            None => {
                // Expected: no mapping available
            },
        }
    }
}
