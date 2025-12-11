//! Adobe Glyph List Fallback Tests for Type0 Fonts
//!
//! Tests for Phase 1.2: When Type0 fonts without ToUnicode CMap and without
//! embedded TrueType font data cannot map characters, they should try the
//! Adobe Glyph List as a fallback before returning the replacement character.
//!
//! This addresses the Aptos and LMRoman font errors found in corpus validation:
//! - Aptos (132,374 errors): Office font with 0-byte embedded data
//! - LMRoman (124,960 errors): LaTeX font with custom encoding
//!
//! Spec: PDF 32000-1:2008 Section 9.10.2

use pdf_oxide::fonts::{CIDToGIDMap, Encoding, FontInfo};
use std::sync::Arc;

#[test]
fn test_type0_agl_fallback_for_standard_ascii() {
    //! Test 1: Type0 font with standard glyph should use Adobe Glyph List
    //!
    //! When a Type0 font has 0-byte embedded data and no ToUnicode CMap,
    //! it should try to map GID to a standard glyph name via Adobe Glyph List.
    //! For ASCII range (GID 32-126), standard glyph names work.

    let font = FontInfo {
        base_font: "Aptos".to_string(),
        subtype: "Type0".to_string(),
        encoding: Encoding::Identity,
        to_unicode: None,
        truetype_cmap: None,
        embedded_font_data: Some(Arc::new(vec![])), // 0 bytes
        cid_to_gid_map: Some(CIDToGIDMap::Identity), // CID == GID
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

    // For ASCII range, should try Adobe Glyph List
    // GID 65 (0x41) = 'A' in standard fonts
    let result = font.char_to_unicode(0x41);

    assert!(result.is_some(), "Should find mapping for GID 0x41 via Adobe Glyph List");
    assert_eq!(
        result.unwrap(),
        "A",
        "GID 0x41 should map to 'A' via Adobe Glyph List, not U+FFFD"
    );
}

#[test]
fn test_type0_lmroman_agl_fallback() {
    //! Test 2: LMRoman font (LaTeX) common characters
    //!
    //! LMRoman is a common LaTeX font that shows up in many academic PDFs.
    //! It's a Type0 font with custom encoding but standard glyphs.
    //! Should be able to map common ASCII characters via Adobe Glyph List.

    let font = FontInfo {
        base_font: "LMRoman10-Regular".to_string(),
        subtype: "Type0".to_string(),
        encoding: Encoding::Identity,
        to_unicode: None,
        truetype_cmap: None,
        embedded_font_data: None,
        cid_to_gid_map: Some(CIDToGIDMap::Identity),
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

    // Test common ASCII characters
    assert_eq!(font.char_to_unicode(0x20), Some(" ".to_string()), "GID 0x20 (space)");
    assert_eq!(font.char_to_unicode(0x41), Some("A".to_string()), "GID 0x41 (capital A)");
    assert_eq!(font.char_to_unicode(0x61), Some("a".to_string()), "GID 0x61 (lowercase a)");
    assert_eq!(font.char_to_unicode(0x30), Some("0".to_string()), "GID 0x30 (digit 0)");
}

#[test]
fn test_type0_agl_fallback_then_replacement() {
    //! Test 3: Non-ASCII glyph should use AGL, then fallback to U+FFFD
    //!
    //! For GIDs that don't have a standard glyph name (outside ASCII range),
    //! should eventually fall back to U+FFFD after AGL lookup fails.

    let font = FontInfo {
        base_font: "TestFont".to_string(),
        subtype: "Type0".to_string(),
        encoding: Encoding::Identity,
        to_unicode: None,
        truetype_cmap: None,
        embedded_font_data: None,
        cid_to_gid_map: Some(CIDToGIDMap::Identity),
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

    // GID 0xFFFF won't be in Adobe Glyph List
    let result = font.char_to_unicode(0xFFFF);

    assert!(result.is_some(), "Should return something (not None)");
    assert_eq!(result.unwrap(), "\u{FFFD}", "Non-ASCII glyph should fallback to U+FFFD");
}
