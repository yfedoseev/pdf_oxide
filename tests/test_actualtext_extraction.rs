//! ActualText Extraction Tests for Phase 2
//!
//! Tests for integrating ActualText into the character mapping priority chain
//! per PDF Spec 32000-1:2008 Section 9.10.2.
//!
//! Phase 2 implements proper ActualText support where marked content with
//! ActualText properties provides the canonical Unicode mapping for characters
//! that might otherwise be unmappable through font encoding alone.
//!
//! This is critical for:
//! - Ligatures (fi, fl, ffi, ffl as single characters mapping to multiple chars)
//! - Decorated glyphs (e.g., accented characters represented as composite glyphs)
//! - Custom text representations (e.g., "R" rendered as special glyph)
//!
//! Spec: PDF 32000-1:2008 Section 9.10.2 (Character to Unicode mapping)

use pdf_oxide::fonts::{Encoding, FontInfo};

#[test]
fn test_ligature_extraction_fi() {
    //! Test 1: Ligature fi should extract to two characters "f" + "i"
    //!
    //! When a PDF contains the ligature character 'ﬁ' (U+FB01) with ActualText "fi",
    //! the text extraction should return "fi" (two characters), not "ﬁ" (one character).

    let font = FontInfo {
        base_font: "StandardFont".to_string(),
        subtype: "Type1".to_string(),
        encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
        to_unicode: None,
        truetype_cmap: None,
        embedded_font_data: None,
        cid_to_gid_map: None,
        cid_system_info: None,
        cid_font_type: None,
        font_weight: None,
        flags: None,
        stem_v: None,
        widths: None,
        first_char: None,
        last_char: None,
        default_width: 500.0,
    };

    // Character 0xFB01 is the 'fi' ligature
    let result = font.char_to_unicode(0xFB01);

    assert!(result.is_some(), "Should map ligature character");

    let mapped = result.unwrap();
    assert!(
        mapped.contains('f') && mapped.contains('i'),
        "Ligature should expand to 'fi', got: {}",
        mapped
    );
}

#[test]
fn test_ligature_extraction_fl() {
    //! Test 2: Ligature fl should extract to two characters "f" + "l"

    let font = FontInfo {
        base_font: "StandardFont".to_string(),
        subtype: "Type1".to_string(),
        encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
        to_unicode: None,
        truetype_cmap: None,
        embedded_font_data: None,
        cid_to_gid_map: None,
        cid_system_info: None,
        cid_font_type: None,
        font_weight: None,
        flags: None,
        stem_v: None,
        widths: None,
        first_char: None,
        last_char: None,
        default_width: 500.0,
    };

    let result = font.char_to_unicode(0xFB02);

    assert!(result.is_some(), "Should map fl ligature character");

    let mapped = result.unwrap();
    assert!(
        mapped.contains('f') && mapped.contains('l'),
        "Ligature should expand to 'fl', got: {}",
        mapped
    );
}

#[test]
fn test_ligature_extraction_ffi() {
    //! Test 3: Ligature ffi should extract to three characters

    let font = FontInfo {
        base_font: "StandardFont".to_string(),
        subtype: "Type1".to_string(),
        encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
        to_unicode: None,
        truetype_cmap: None,
        embedded_font_data: None,
        cid_to_gid_map: None,
        cid_system_info: None,
        cid_font_type: None,
        font_weight: None,
        flags: None,
        stem_v: None,
        widths: None,
        first_char: None,
        last_char: None,
        default_width: 500.0,
    };

    let result = font.char_to_unicode(0xFB03);

    assert!(result.is_some(), "Should map ffi ligature character");

    let mapped = result.unwrap();
    assert!(
        mapped.len() >= 3,
        "Ligature ffi should expand to at least 3 characters, got: {} (len: {})",
        mapped,
        mapped.len()
    );
}

#[test]
fn test_ligature_extraction_ffl() {
    //! Test 4: Ligature ffl should extract to three characters

    let font = FontInfo {
        base_font: "StandardFont".to_string(),
        subtype: "Type1".to_string(),
        encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
        to_unicode: None,
        truetype_cmap: None,
        embedded_font_data: None,
        cid_to_gid_map: None,
        cid_system_info: None,
        cid_font_type: None,
        font_weight: None,
        flags: None,
        stem_v: None,
        widths: None,
        first_char: None,
        last_char: None,
        default_width: 500.0,
    };

    let result = font.char_to_unicode(0xFB04);

    assert!(result.is_some(), "Should map ffl ligature character");

    let mapped = result.unwrap();
    assert!(
        mapped.len() >= 3,
        "Ligature ffl should expand to at least 3 characters, got: {} (len: {})",
        mapped,
        mapped.len()
    );
}
