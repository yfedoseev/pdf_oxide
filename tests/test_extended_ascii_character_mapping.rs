//! TDD Tests for Extended ASCII Character Mapping in CharacterMapper
//!
//! Tests verify that character codes in the extended ASCII range (0x80-0xFF)
//! are properly mapped using WinAnsiEncoding (Windows-1252) fallback.

#[test]
fn test_extended_ascii_en_dash() {
    use pdf_oxide::fonts::character_mapper::CharacterMapper;

    let mapper = CharacterMapper::new();

    // 0x96 in WinAnsiEncoding = en-dash (–, U+2013)
    let result = mapper.code_to_glyph_name_extended(0x96);
    assert_eq!(result, Some("endash".to_string()));
}

#[test]
fn test_extended_ascii_em_dash() {
    use pdf_oxide::fonts::character_mapper::CharacterMapper;

    let mapper = CharacterMapper::new();

    // 0x97 in WinAnsiEncoding = em-dash (—, U+2014)
    let result = mapper.code_to_glyph_name_extended(0x97);
    assert_eq!(result, Some("emdash".to_string()));
}

#[test]
fn test_extended_ascii_left_quote() {
    use pdf_oxide::fonts::character_mapper::CharacterMapper;

    let mapper = CharacterMapper::new();

    // 0x93 in WinAnsiEncoding = left double quotation mark (", U+201C)
    let result = mapper.code_to_glyph_name_extended(0x93);
    assert_eq!(result, Some("quotedblleft".to_string()));
}

#[test]
fn test_extended_ascii_right_quote() {
    use pdf_oxide::fonts::character_mapper::CharacterMapper;

    let mapper = CharacterMapper::new();

    // 0x94 in WinAnsiEncoding = right double quotation mark (", U+201D)
    let result = mapper.code_to_glyph_name_extended(0x94);
    assert_eq!(result, Some("quotedblright".to_string()));
}

#[test]
fn test_extended_ascii_ellipsis() {
    use pdf_oxide::fonts::character_mapper::CharacterMapper;

    let mapper = CharacterMapper::new();

    // 0x85 in WinAnsiEncoding = ellipsis (…, U+2026)
    let result = mapper.code_to_glyph_name_extended(0x85);
    assert_eq!(result, Some("ellipsis".to_string()));
}

#[test]
fn test_extended_ascii_copyright() {
    use pdf_oxide::fonts::character_mapper::CharacterMapper;

    let mapper = CharacterMapper::new();

    // 0xA9 in WinAnsiEncoding = copyright sign (©, U+00A9)
    let result = mapper.code_to_glyph_name_extended(0xA9);
    assert_eq!(result, Some("copyright".to_string()));
}

#[test]
fn test_extended_ascii_registered() {
    use pdf_oxide::fonts::character_mapper::CharacterMapper;

    let mapper = CharacterMapper::new();

    // 0xAE in WinAnsiEncoding = registered trademark sign (®, U+00AE)
    let result = mapper.code_to_glyph_name_extended(0xAE);
    assert_eq!(result, Some("registered".to_string()));
}

#[test]
fn test_extended_ascii_trademark() {
    use pdf_oxide::fonts::character_mapper::CharacterMapper;

    let mapper = CharacterMapper::new();

    // 0x99 in WinAnsiEncoding = trademark sign (™, U+2122)
    let result = mapper.code_to_glyph_name_extended(0x99);
    assert_eq!(result, Some("trademark".to_string()));
}

#[test]
fn test_extended_ascii_degree() {
    use pdf_oxide::fonts::character_mapper::CharacterMapper;

    let mapper = CharacterMapper::new();

    // 0xB0 in WinAnsiEncoding = degree sign (°, U+00B0)
    let result = mapper.code_to_glyph_name_extended(0xB0);
    assert_eq!(result, Some("degree".to_string()));
}

#[test]
fn test_extended_ascii_german_ae() {
    use pdf_oxide::fonts::character_mapper::CharacterMapper;

    let mapper = CharacterMapper::new();

    // 0xE4 in WinAnsiEncoding = latin small letter a with diaeresis (ä, U+00E4)
    let result = mapper.code_to_glyph_name_extended(0xE4);
    assert_eq!(result, Some("adieresis".to_string()));
}

#[test]
fn test_extended_ascii_french_c_cedilla() {
    use pdf_oxide::fonts::character_mapper::CharacterMapper;

    let mapper = CharacterMapper::new();

    // 0xE7 in WinAnsiEncoding = latin small letter c with cedilla (ç, U+00E7)
    let result = mapper.code_to_glyph_name_extended(0xE7);
    assert_eq!(result, Some("ccedilla".to_string()));
}

#[test]
fn test_extended_ascii_euro() {
    use pdf_oxide::fonts::character_mapper::CharacterMapper;

    let mapper = CharacterMapper::new();

    // 0x80 in WinAnsiEncoding = euro sign (€, U+20AC)
    let result = mapper.code_to_glyph_name_extended(0x80);
    assert_eq!(result, Some("Euro".to_string()));
}

#[test]
fn test_map_character_with_extended_ascii() {
    use pdf_oxide::fonts::character_mapper::CharacterMapper;

    let mut mapper = CharacterMapper::new();

    // Set up encoding that doesn't override extended ASCII (None)
    mapper.set_font_encoding(None);

    // 0x96 should map via glyph name to "–" (en-dash)
    let result = mapper.map_character(0x96);
    assert!(result.is_some());
    // Should contain en-dash character or glyph name
    let mapped = result.unwrap();
    assert!(!mapped.is_empty());
}

#[test]
fn test_extended_ascii_fallback_with_custom_encoding() {
    use pdf_oxide::fonts::character_mapper::CharacterMapper;
    use std::collections::HashMap;

    let mut mapper = CharacterMapper::new();

    // Custom encoding that only maps specific characters
    let mut encoding = HashMap::new();
    encoding.insert(0x41, 'A'); // Override A
    mapper.set_font_encoding(Some(encoding));

    // Custom encoding should win for 0x41
    let result = mapper.map_character(0x41);
    assert_eq!(result, Some("A".to_string()));

    // Extended ASCII fallback should still work for unmapped codes like 0x96
    let result = mapper.map_character(0x96);
    assert!(result.is_some());
}

#[test]
fn test_extended_ascii_common_special_chars() {
    use pdf_oxide::fonts::character_mapper::CharacterMapper;

    let mapper = CharacterMapper::new();

    // Test a range of common special characters
    let test_cases = vec![
        (0x80, "Euro"),          // Euro sign
        (0x85, "ellipsis"),      // Ellipsis
        (0x93, "quotedblleft"),  // Left double quote
        (0x94, "quotedblright"), // Right double quote
        (0x96, "endash"),        // En-dash
        (0x97, "emdash"),        // Em-dash
    ];

    for (code, expected_glyph) in test_cases {
        let result = mapper.code_to_glyph_name_extended(code);
        assert_eq!(result, Some(expected_glyph.to_string()), "Failed for code 0x{:02X}", code);
    }
}

#[test]
fn test_extended_ascii_invalid_codes() {
    use pdf_oxide::fonts::character_mapper::CharacterMapper;

    let mapper = CharacterMapper::new();

    // Codes that don't have special mappings should return None (not in extended ASCII range)
    let result = mapper.code_to_glyph_name_extended(0x01);
    assert_eq!(result, None);

    // Valid extended ASCII range but no special mapping
    let result = mapper.code_to_glyph_name_extended(0x81);
    assert_eq!(result, None); // 0x81 doesn't have a special glyph name
}

#[test]
fn test_extended_ascii_currency_symbols() {
    use pdf_oxide::fonts::character_mapper::CharacterMapper;

    let mapper = CharacterMapper::new();

    // 0xA4 = currency sign (¤, U+00A4)
    let result = mapper.code_to_glyph_name_extended(0xA4);
    assert_eq!(result, Some("currency".to_string()));

    // 0xA5 = yen sign (¥, U+00A5)
    let result = mapper.code_to_glyph_name_extended(0xA5);
    assert_eq!(result, Some("yen".to_string()));

    // 0xA3 = pound sign (£, U+00A3)
    let result = mapper.code_to_glyph_name_extended(0xA3);
    assert_eq!(result, Some("sterling".to_string()));
}
