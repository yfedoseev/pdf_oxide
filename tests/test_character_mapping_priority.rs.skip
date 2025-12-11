//! Tests for ISO 32000-1:2008 Section 9.10.2 Character-to-Unicode Mapping Priorities
//!
//! PDF spec defines a 5-level priority order for character-to-Unicode mapping:
//! 1. ToUnicode CMap (highest priority)
//! 2. Adobe Glyph List (fallback 1)
//! 3. Predefined CMaps (fallback 2)
//! 4. ActualText attribute (fallback 3)
//! 5. Font encoding (lowest priority)

use pdf_oxide::fonts::character_mapper::CharacterMapper;
use std::collections::HashMap;

#[test]
fn test_priority_1_tounicode_cmap_highest() {
    let mut mapper = CharacterMapper::new();

    // Priority 1: ToUnicode CMap should override everything
    let tounicode = {
        let mut cmap = HashMap::new();
        cmap.insert(0x41, "A".to_string());
        cmap
    };
    mapper.set_tounicode_cmap(Some(tounicode));

    // Should return 'A' from ToUnicode
    assert_eq!(mapper.map_character(0x41), Some("A".to_string()));
}

#[test]
fn test_priority_2_adobe_glyph_list_fallback() {
    let mapper = CharacterMapper::new();

    // Without ToUnicode, should fall back to Adobe Glyph List
    // 0x47 maps to "G" in Adobe Glyph List
    assert_eq!(mapper.map_character(0x47), Some("G".to_string()));
}

#[test]
fn test_priority_1_overrides_adobe_glyph_list() {
    let mut mapper = CharacterMapper::new();

    // ToUnicode CMap takes priority over Adobe Glyph List
    let tounicode = {
        let mut cmap = HashMap::new();
        // Map 0x47 to 'X' (instead of 'G' from glyph list)
        cmap.insert(0x47, "X".to_string());
        cmap
    };
    mapper.set_tounicode_cmap(Some(tounicode));

    // Should return 'X' from ToUnicode, not 'G' from glyph list
    assert_eq!(mapper.map_character(0x47), Some("X".to_string()));
}

#[test]
fn test_adobe_glyph_list_symbol_fonts() {
    let mapper = CharacterMapper::new();

    // Symbol fonts like Wingdings use Adobe Glyph List
    // Common symbol mappings
    assert!(mapper.map_character(0x20).is_some()); // Space
    assert!(mapper.map_character(0x41).is_some()); // A
    assert!(mapper.map_character(0x7A).is_some()); // z
}

#[test]
fn test_adobe_glyph_list_special_glyphs() {
    let mapper = CharacterMapper::new();

    // Adobe Glyph List has specific mappings for named glyphs
    // Testing that named glyph lookups work
    assert!(mapper.map_glyph_name("A").is_some());
    assert!(mapper.map_glyph_name("B").is_some());
    assert!(mapper.map_glyph_name("ampersand").is_some());
    assert!(mapper.map_glyph_name("infinity").is_some());
}

#[test]
fn test_adobe_glyph_list_ligatures() {
    let mapper = CharacterMapper::new();

    // Adobe Glyph List includes ligatures
    assert!(mapper.map_glyph_name("fi").is_some()); // fi ligature
    assert!(mapper.map_glyph_name("fl").is_some()); // fl ligature
    assert!(mapper.map_glyph_name("ff").is_some()); // ff ligature
}

#[test]
fn test_priority_chain_tounicode_wins() {
    let mut mapper = CharacterMapper::new();

    // Set up ToUnicode CMap for code 0x50
    let tounicode = {
        let mut cmap = HashMap::new();
        cmap.insert(0x50, "P".to_string());
        cmap
    };
    mapper.set_tounicode_cmap(Some(tounicode));

    // ToUnicode should win over any other priority
    assert_eq!(mapper.map_character(0x50), Some("P".to_string()));
}

#[test]
fn test_incomplete_tounicode_falls_back_to_adobe() {
    let mut mapper = CharacterMapper::new();

    // ToUnicode CMap is incomplete - only has mapping for 0x50
    let tounicode = {
        let mut cmap = HashMap::new();
        cmap.insert(0x50, "P".to_string());
        cmap
    };
    mapper.set_tounicode_cmap(Some(tounicode));

    // 0x50 should use ToUnicode
    assert_eq!(mapper.map_character(0x50), Some("P".to_string()));

    // 0x51 should fall back to Adobe Glyph List
    assert_eq!(mapper.map_character(0x51), Some("Q".to_string()));
}

#[test]
fn test_empty_tounicode_uses_adobe_glyph_list() {
    let mut mapper = CharacterMapper::new();

    // Empty ToUnicode CMap - should fall back to Adobe Glyph List
    mapper.set_tounicode_cmap(Some(HashMap::new()));

    // Should use Adobe Glyph List
    assert_eq!(mapper.map_character(0x41), Some("A".to_string()));
}

#[test]
fn test_character_mapper_state_isolation() {
    let mut mapper1 = CharacterMapper::new();
    let mut mapper2 = CharacterMapper::new();

    // Set different ToUnicode for mapper1
    let tounicode = {
        let mut cmap = HashMap::new();
        cmap.insert(0x41, "X".to_string());
        cmap
    };
    mapper1.set_tounicode_cmap(Some(tounicode));

    // mapper1 should use ToUnicode
    assert_eq!(mapper1.map_character(0x41), Some("X".to_string()));

    // mapper2 should use Adobe Glyph List (default)
    assert_eq!(mapper2.map_character(0x41), Some("A".to_string()));
}

#[test]
fn test_character_mapping_with_font_encoding() {
    let mut mapper = CharacterMapper::new();

    // Set font encoding (lowest priority)
    let mut encoding = HashMap::new();
    encoding.insert(0xAA, 'Z');
    mapper.set_font_encoding(Some(encoding));

    // Without ToUnicode, should use font encoding
    assert_eq!(mapper.map_character(0xAA), Some("Z".to_string()));

    // Now add ToUnicode - it should override font encoding
    let tounicode = {
        let mut cmap = HashMap::new();
        cmap.insert(0xAA, "Y".to_string());
        cmap
    };
    mapper.set_tounicode_cmap(Some(tounicode));

    // ToUnicode should override font encoding
    assert_eq!(mapper.map_character(0xAA), Some("Y".to_string()));
}

#[test]
fn test_no_mapping_returns_none() {
    let mapper = CharacterMapper::new();

    // Character with no mapping anywhere should return None or replacement character
    // Using a high character code unlikely to have a mapping
    let result = mapper.map_character(0xFFFE);
    // Should either return None or a fallback character
    assert!(result.is_none() || result == Some("\u{FFFD}".to_string())); // Replacement char
}

#[test]
fn test_ascii_range_always_mapped() {
    let mapper = CharacterMapper::new();

    // ASCII printable range (0x20-0x7E) should always map
    for code in 0x20..=0x7E {
        assert!(
            mapper.map_character(code as u32).is_some(),
            "ASCII code 0x{:02X} should map",
            code
        );
    }
}

#[test]
fn test_custom_tounicode_overwrites_previous() {
    let mut mapper = CharacterMapper::new();

    // Set initial ToUnicode
    let tounicode1 = {
        let mut cmap = HashMap::new();
        cmap.insert(0x41, "A".to_string());
        cmap
    };
    mapper.set_tounicode_cmap(Some(tounicode1));
    assert_eq!(mapper.map_character(0x41), Some("A".to_string()));

    // Overwrite with new ToUnicode
    let tounicode2 = {
        let mut cmap = HashMap::new();
        cmap.insert(0x41, "X".to_string());
        cmap
    };
    mapper.set_tounicode_cmap(Some(tounicode2));
    assert_eq!(mapper.map_character(0x41), Some("X".to_string()));
}

#[test]
fn test_ligature_expansion() {
    let mut mapper = CharacterMapper::new();

    // ToUnicode can map a single character code to multiple Unicode chars (ligatures)
    let tounicode = {
        let mut cmap = HashMap::new();
        cmap.insert(0x0C, "fi".to_string()); // 0x0C maps to "fi" ligature
        cmap
    };
    mapper.set_tounicode_cmap(Some(tounicode));

    // Should return "fi" as a string (2 characters)
    assert_eq!(mapper.map_character(0x0C), Some("fi".to_string()));
}

#[test]
fn test_utf16_surrogate_pairs() {
    let mut mapper = CharacterMapper::new();

    // ToUnicode CMap handles UTF-16 surrogate pairs for characters > U+FFFF
    let tounicode = {
        let mut cmap = HashMap::new();
        // Example: Mathematical italic small rho U+1D70C
        // In UTF-16: D835 DF0C (high surrogate D835, low surrogate DF0C)
        cmap.insert(0x0001, "𝜌".to_string()); // Mathematical italic rho
        cmap
    };
    mapper.set_tounicode_cmap(Some(tounicode));

    assert_eq!(mapper.map_character(0x0001), Some("𝜌".to_string()));
}

#[test]
fn test_cjk_font_support() {
    let mapper = CharacterMapper::new();

    // CJK fonts use high character codes (16-bit)
    // Adobe Glyph List should handle CJK mappings if available
    // This test verifies that the mapper can handle high code points
    let result = mapper.map_character(0x3042); // Japanese hiragana A
    // May or may not have a mapping - that's okay
    // Just verify it doesn't panic
    let _ = result;
}

#[test]
fn test_batch_character_mapping() {
    let mapper = CharacterMapper::new();

    // Test mapping multiple characters efficiently
    let test_codes = vec![0x41, 0x42, 0x43, 0x44, 0x45];
    let expected = vec!["A", "B", "C", "D", "E"];

    for (code, expected_char) in test_codes.iter().zip(expected.iter()) {
        assert_eq!(mapper.map_character(*code), Some(expected_char.to_string()));
    }
}

#[test]
fn test_spec_compliance_priority_order() {
    // This test verifies the exact priority order from PDF spec
    // ISO 32000-1:2008, Section 9.10.2

    let mut mapper = CharacterMapper::new();

    // Priority 1: ToUnicode CMap - explicit mapping
    let tounicode = {
        let mut cmap = HashMap::new();
        cmap.insert(0x41, "FromToUnicode".to_string());
        cmap
    };
    mapper.set_tounicode_cmap(Some(tounicode));

    // Should get ToUnicode value (priority 1)
    assert_eq!(
        mapper.map_character(0x41),
        Some("FromToUnicode".to_string()),
        "ToUnicode CMap should have highest priority"
    );

    // For character without ToUnicode mapping, test fallback to Adobe Glyph List
    // 0x42 is not in our ToUnicode mapping
    let result_without_tounicode = mapper.map_character(0x42);
    // Should fall back to Adobe Glyph List (should map to "B")
    assert_eq!(result_without_tounicode, Some("B".to_string()));
}
