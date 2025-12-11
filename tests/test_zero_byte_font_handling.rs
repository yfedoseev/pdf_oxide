//! Zero-Byte Embedded Font Handling Tests
//!
//! Tests for Phase 1.3: When embedded fonts have 0 bytes of data but are marked as embedded,
//! we should skip attempting to read the embedded TrueType cmap and move to the next fallback
//! (Adobe Glyph List).
//!
//! This addresses performance and correctness issues with fonts like Aptos, Calibri, and
//! common Office fonts that are marked as embedded but have no actual data.
//!
//! Spec: PDF 32000-1:2008 Section 9.4.3 (Font Descriptors)

use pdf_oxide::fonts::{CIDToGIDMap, Encoding, FontInfo};
use std::sync::Arc;

#[test]
fn test_skip_truetype_cmap_when_embedded_font_zero_bytes() {
    //! Test 1: Should skip TrueType cmap for 0-byte font and use AGL fallback
    //!
    //! When a Type0 font has embedded_font_data set to empty vec (0 bytes),
    //! we should NOT attempt to parse TrueType cmap from it.
    //! Instead, skip directly to Adobe Glyph List fallback.

    let font = FontInfo {
        base_font: "Aptos".to_string(),
        subtype: "Type0".to_string(),
        encoding: Encoding::Identity,
        to_unicode: None,
        truetype_cmap: None,                        // No explicit TrueType cmap
        embedded_font_data: Some(Arc::new(vec![])), // 0 bytes - marked embedded but empty
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

    // Should use Adobe Glyph List fallback despite 0-byte embedded data
    let result = font.char_to_unicode(0x41); // Try to map 'A'

    assert!(
        result.is_some(),
        "Should find mapping for GID 0x41 via Adobe Glyph List, not fail"
    );
    assert_eq!(result.unwrap(), "A", "0-byte embedded font should use AGL fallback, not U+FFFD");
}

#[test]
fn test_skip_truetype_cmap_for_common_office_fonts() {
    //! Test 2: Should handle 0-byte Office fonts (Calibri, Helvetica, etc.)
    //!
    //! Many Office document PDFs have fonts marked as embedded with 0 bytes.
    //! These should fall back to Adobe Glyph List properly.

    let fonts = vec![
        ("Calibri", 0x41),       // A
        ("Helvetica", 0x42),     // B
        ("Arial", 0x43),         // C
        ("TimesNewRoman", 0x44), // D
    ];

    for (font_name, gid) in fonts {
        let font = FontInfo {
            base_font: font_name.to_string(),
            subtype: "Type0".to_string(),
            encoding: Encoding::Identity,
            to_unicode: None,
            truetype_cmap: None,
            embedded_font_data: Some(Arc::new(vec![])), // 0 bytes
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

        let result = font.char_to_unicode(gid);
        assert!(
            result.is_some(),
            "Font {} with 0-byte data should map GID 0x{:02X}",
            font_name,
            gid
        );

        // Should get actual character, not replacement
        assert_ne!(
            result.unwrap(),
            "\u{FFFD}",
            "Font {} should use AGL, not U+FFFD for GID 0x{:02X}",
            font_name,
            gid
        );
    }
}

#[test]
fn test_still_use_truetype_cmap_when_embedded_font_has_data() {
    //! Test 3: Should still use TrueType cmap when font data exists
    //!
    //! This test ensures we don't break the normal path: when a font HAS
    //! actual embedded data, we should still try TrueType cmap before AGL.
    //!
    //! (This test might need adjustment based on how we detect valid TrueType data)

    let font_data = vec![0u8; 100]; // Non-empty (simulated valid font)

    let font = FontInfo {
        base_font: "ValidFont".to_string(),
        subtype: "Type0".to_string(),
        encoding: Encoding::Identity,
        to_unicode: None,
        truetype_cmap: None, // Will need to be parsed from embedded_font_data
        embedded_font_data: Some(Arc::new(font_data)), // Has data
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

    // Should attempt mapping - either via embedded data or AGL
    let result = font.char_to_unicode(0x41);

    // With actual embedded data and AGL fallback, should succeed
    assert!(result.is_some(), "Font with embedded data should map successfully");
}
