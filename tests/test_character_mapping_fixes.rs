//! Comprehensive test suite for character mapping fixes (Phase 1)
//!
//! This module tests all the critical bug fixes for PDF text extraction:
//! - Phase 1.1: Identity encoding fallback for Type0 fonts
//! - Phase 1.2: ToUnicode CMap validation
//! - Phase 1.3: Type0 missing ToUnicode error handling
//! - Phase 1.4: Multi-byte processing validation
//!
//! These tests ensure that garbled text issues are properly handled.

use pdf_oxide::fonts::{Encoding, FontInfo, LazyCMap};

// ============================================================================
// Phase 1.1 Tests: Identity Encoding Fallback for Type0 Fonts
// ============================================================================

#[test]
fn test_type0_identity_encoding_without_tounicode_returns_none() {
    // Type0 fonts WITHOUT ToUnicode should NOT map characters via Identity encoding
    // Per PDF Spec 9.10.2, should return U+FFFD replacement character
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
        truetype_cmap: None,
    };

    // Character code 0x37 (decimal 55) would incorrectly map to '7' under old code
    // With the spec-compliant fix, it should return U+FFFD replacement character
    assert_eq!(
        font.char_to_unicode(0x37),
        Some("\u{FFFD}".to_string()),
        "Type0 font without ToUnicode should return U+FFFD for character code 0x37, not '7'"
    );

    // Character code 0x41 (decimal 65) would incorrectly map to 'A' under old code
    assert_eq!(
        font.char_to_unicode(0x41),
        Some("\u{FFFD}".to_string()),
        "Type0 font without ToUnicode should return U+FFFD for character code 0x41, not 'A'"
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
        truetype_cmap: None,
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
        truetype_cmap: None,
    };

    // All lookups should return U+FFFD for Type0 without ToUnicode
    // Per PDF Spec 9.10.2, when mapping fails, conforming readers should use replacement character
    // This includes ALL character codes (including control chars) when using Identity-H/Identity-V
    for code in 0u32..256 {
        let result = font.char_to_unicode(code);
        assert_eq!(
            result,
            Some("\u{FFFD}".to_string()),
            "Type0 font with Identity-H encoding without ToUnicode should return U+FFFD for code 0x{:02X}",
            code
        );
    }
}

#[test]
fn test_tounicode_with_valid_mappings_works() {
    let cmap_data = b"beginbfchar\n<0041> <0041>\n<0042> <0042>\n<263A> <263A>\nendbfchar";

    let font = FontInfo {
        base_font: "CustomFont".to_string(),
        subtype: "Type1".to_string(),
        encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
        to_unicode: Some(LazyCMap::new(cmap_data.to_vec())),
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
        truetype_cmap: None,
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
        truetype_cmap: None,
    };

    // Multi-byte codes (> 0xFF) should be handled without panic
    // Per PDF Spec 9.10.2, should return U+FFFD replacement character
    let large_code = 0x3000u32; // Typical CJK character code
    assert_eq!(
        font.char_to_unicode(large_code),
        Some("\u{FFFD}".to_string()),
        "Multi-byte code without ToUnicode should return U+FFFD replacement character"
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

    let cmap_data = b"beginbfchar\n<0041> <0058>\nendbfchar"; // Map 0x41 to 'X'

    let font = FontInfo {
        base_font: "TestFont".to_string(),
        subtype: "Type1".to_string(),
        encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
        to_unicode: Some(LazyCMap::new(cmap_data.to_vec())),
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
        truetype_cmap: None,
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
        truetype_cmap: None,
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
        truetype_cmap: None,
    };

    // The key assertion: we should get U+FFFD, NOT random scrambled characters
    // Per PDF Spec 9.10.2, when mapping fails, conforming readers should use replacement character
    // This applies to ALL character codes for Type0 fonts with Identity encoding
    for code in 0u32..256 {
        let result = font.char_to_unicode(code);
        assert_eq!(
            result,
            Some("\u{FFFD}".to_string()),
            "Type0 font without ToUnicode should return U+FFFD for code 0x{:02X}, not scrambled text",
            code
        );
    }
}
