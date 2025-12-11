//! PDF Spec 32000-1:2008 Section 9.10.2 Compliance Tests
//!
//! Tests for the fallback behavior when character mapping fails.
//! Spec Section 9.10.2 states:
//! "If these methods fail to produce a Unicode value, there is no way to determine what
//!  the character code represents in which case a conforming reader may choose a character
//!  code of their choosing."
//!
//! Standard practice: Use U+FFFD (REPLACEMENT CHARACTER) as the fallback.

use pdf_oxide::fonts::{CharacterMapper, Encoding, FontInfo};
use std::sync::Arc;

#[test]
fn test_spec_9_10_2_unmapped_character_returns_replacement() {
    //! Test 1: CharacterMapper returns fallback for unmapped character
    //!
    //! When a character code cannot be mapped through any method (ToUnicode, Adobe Glyph List,
    //! Font Encoding), the CharacterMapper should return U+FFFD (REPLACEMENT CHARACTER),
    //! not None.
    //!
    //! This ensures compliance with PDF Spec Section 9.10.2 and prevents silent data loss.

    let mapper = CharacterMapper::new();
    // No ToUnicode, no encoding, no glyph name - completely unmapped
    let result = mapper.map_character(0xFFFF);

    // Spec 9.10.2: "conforming reader may choose a character code of their choosing"
    // Standard practice: U+FFFD REPLACEMENT CHARACTER
    assert!(result.is_some(), "Should return replacement character, not None");
    assert_eq!(result.unwrap(), "\u{FFFD}", "Should return U+FFFD replacement character");
}

#[test]
fn test_type0_identity_encoding_no_tounicode_returns_replacement() {
    //! Test 2: Type0 font without ToUnicode or TrueType cmap
    //!
    //! Type0 fonts using Identity encoding without a ToUnicode CMap and without
    //! an embedded TrueType font have no way to map characters. In this case,
    //! should return U+FFFD, not None (silent omission).

    let font = FontInfo {
        base_font: "Aptos".to_string(),
        subtype: "Type0".to_string(),
        encoding: Encoding::Identity,
        to_unicode: None,
        truetype_cmap: None,      // No TrueType cmap
        embedded_font_data: None, // No embedded data
        cid_to_gid_map: None,
        cid_system_info: None,
        cid_font_type: None,
        font_weight: None,
        flags: None,
        stem_v: None,
        widths: None,
        first_char: None,
        last_char: None,
        default_width: 1000.0,
    };

    let result = font.char_to_unicode(0x0041); // Try to map 'A'

    // Should NOT return None (current bug - silent omission)
    // Should return U+FFFD per spec
    assert!(result.is_some(), "Should return replacement character, not silently omit");
    assert_eq!(
        result.unwrap(),
        "\u{FFFD}",
        "Type0 font without ToUnicode should fall back to replacement character"
    );
}

#[test]
fn test_type0_zero_byte_embedded_font_returns_replacement() {
    //! Test 3: Embedded font is 0 bytes (marked embedded but empty)
    //!
    //! Some PDFs mark a font as embedded but provide 0 bytes of actual font data.
    //! When we can't parse the embedded font, we should fall back to replacement
    //! character, not silently omit the character.

    let font = FontInfo {
        base_font: "Calibri".to_string(),
        subtype: "Type0".to_string(),
        encoding: Encoding::Identity,
        to_unicode: None,
        truetype_cmap: None,
        embedded_font_data: Some(Arc::new(vec![])), // 0 bytes!
        cid_to_gid_map: None,
        cid_system_info: None,
        cid_font_type: None,
        font_weight: None,
        flags: None,
        stem_v: None,
        widths: None,
        first_char: None,
        last_char: None,
        default_width: 1000.0,
    };

    let result = font.char_to_unicode(0x0020); // Try to map space character

    assert!(
        result.is_some(),
        "Font with 0-byte embedded data should return replacement character"
    );
    assert_eq!(result.unwrap(), "\u{FFFD}", "0-byte embedded font should fall back to U+FFFD");
}
