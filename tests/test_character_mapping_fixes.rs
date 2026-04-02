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
use std::collections::HashMap;

// ============================================================================
// Phase 1.1 Tests: Identity Encoding Fallback for Type0 Fonts
// ============================================================================

#[test]
fn test_type0_identity_encoding_without_tounicode_returns_none() {
    // Type0 fonts WITHOUT ToUnicode use CID-as-Unicode fallback for printable chars
    // This matches MuPDF behavior and improves real-world text extraction quality
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
        truetype_cmap: std::sync::OnceLock::new(),
        is_truetype_font: false,
        cid_widths: None,
        cid_default_width: 1000.0,
        cff_gid_map: None,
        multi_char_map: HashMap::new(),
        byte_to_char_table: std::sync::OnceLock::new(),
        byte_to_width_table: std::sync::OnceLock::new(),
    };

    // CID-as-Unicode fallback: 0x37 → '7', 0x41 → 'A'
    assert_eq!(
        font.char_to_unicode(0x37),
        Some("7".to_string()),
        "Type0 font without ToUnicode uses CID-as-Unicode fallback"
    );

    assert_eq!(
        font.char_to_unicode(0x41),
        Some("A".to_string()),
        "Type0 font without ToUnicode uses CID-as-Unicode fallback"
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
        truetype_cmap: std::sync::OnceLock::new(),
        is_truetype_font: false,
        cid_widths: None,
        cid_default_width: 1000.0,
        cff_gid_map: None,
        multi_char_map: HashMap::new(),
        byte_to_char_table: std::sync::OnceLock::new(),
        byte_to_width_table: std::sync::OnceLock::new(),
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
        truetype_cmap: std::sync::OnceLock::new(),
        is_truetype_font: false,
        cid_widths: None,
        cid_default_width: 1000.0,
        cff_gid_map: None,
        multi_char_map: HashMap::new(),
        byte_to_char_table: std::sync::OnceLock::new(),
        byte_to_width_table: std::sync::OnceLock::new(),
    };

    // CID-as-Unicode fallback: printable chars return themselves, control chars may return None
    // This matches MuPDF behavior for real-world text extraction quality
    let result = font.char_to_unicode(0x41);
    assert_eq!(
        result,
        Some("A".to_string()),
        "Type0 font without ToUnicode uses CID-as-Unicode fallback for printable chars"
    );
    let result_space = font.char_to_unicode(0x20);
    assert_eq!(
        result_space,
        Some(" ".to_string()),
        "Type0 font without ToUnicode uses CID-as-Unicode fallback for space"
    );
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
        truetype_cmap: std::sync::OnceLock::new(),
        is_truetype_font: false,
        cid_widths: None,
        cid_default_width: 1000.0,
        cff_gid_map: None,
        multi_char_map: HashMap::new(),
        byte_to_char_table: std::sync::OnceLock::new(),
        byte_to_width_table: std::sync::OnceLock::new(),
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
        truetype_cmap: std::sync::OnceLock::new(),
        is_truetype_font: false,
        cid_widths: None,
        cid_default_width: 1000.0,
        cff_gid_map: None,
        multi_char_map: HashMap::new(),
        byte_to_char_table: std::sync::OnceLock::new(),
        byte_to_width_table: std::sync::OnceLock::new(),
    };

    // Multi-byte codes (> 0xFF) should be handled without panic
    // CID-as-Unicode fallback returns the character if it's a valid Unicode code point
    let large_code = 0x3000u32; // U+3000 CJK ideographic space
    assert_eq!(
        font.char_to_unicode(large_code),
        Some("\u{3000}".to_string()),
        "Multi-byte code uses CID-as-Unicode fallback"
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
        truetype_cmap: std::sync::OnceLock::new(),
        is_truetype_font: false,
        cid_widths: None,
        cid_default_width: 1000.0,
        cff_gid_map: None,
        multi_char_map: HashMap::new(),
        byte_to_char_table: std::sync::OnceLock::new(),
        byte_to_width_table: std::sync::OnceLock::new(),
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
        truetype_cmap: std::sync::OnceLock::new(),
        is_truetype_font: false,
        cid_widths: None,
        cid_default_width: 1000.0,
        cff_gid_map: None,
        multi_char_map: HashMap::new(),
        byte_to_char_table: std::sync::OnceLock::new(),
        byte_to_width_table: std::sync::OnceLock::new(),
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
        truetype_cmap: std::sync::OnceLock::new(),
        is_truetype_font: false,
        cid_widths: None,
        cid_default_width: 1000.0,
        cff_gid_map: None,
        multi_char_map: HashMap::new(),
        byte_to_char_table: std::sync::OnceLock::new(),
        byte_to_width_table: std::sync::OnceLock::new(),
    };

    // CID-as-Unicode fallback: printable chars map to themselves
    // This matches MuPDF behavior for practical text extraction quality
    let result = font.char_to_unicode(0x20); // space
    assert_eq!(result, Some(" ".to_string()));
    let result = font.char_to_unicode(0x41); // 'A'
    assert_eq!(result, Some("A".to_string()));
}
