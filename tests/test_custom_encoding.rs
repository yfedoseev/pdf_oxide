//! Test suite for Week 2 Day 7: Custom Encoding Support (2B)
//!
//! This test suite verifies that custom font encodings are properly normalized
//! before word boundary detection, ensuring that word boundaries are detected
//! based on actual Unicode characters rather than raw byte codes.
//!
//! Per PDF Spec ISO 32000-1:2008, Section 9.6.6:
//! - Fonts can have custom encodings with /Differences arrays
//! - /Differences override the base encoding for specific character codes
//! - Character-to-Unicode mapping must respect these custom encodings

use pdf_oxide::fonts::{Encoding, FontInfo};
use std::collections::HashMap;

#[test]
fn test_custom_encoding_basic() {
    // Test that FontInfo can normalize character codes through custom encoding
    // Font with /Differences [0x64 /rho] should map 0x64 to Greek rho (U+03C1)

    let mut mappings = HashMap::new();
    mappings.insert(0x64, 'ρ'); // Greek rho at position 0x64

    let font_info = create_font_with_encoding(Encoding::Custom(mappings));

    // Code 0x64 should normalize to Greek rho (U+03C1)
    let normalized = font_info.get_encoded_char(0x64);
    assert_eq!(normalized, Some('ρ'), "Custom encoding should map 0x64 to Greek rho");
}

#[test]
fn test_custom_encoding_has_custom_method() {
    // Verify that FontInfo.has_custom_encoding() correctly identifies custom encodings

    let mut mappings = HashMap::new();
    mappings.insert(0x64, 'ρ');

    let font_with_custom = create_font_with_encoding(Encoding::Custom(mappings));
    assert!(
        font_with_custom.has_custom_encoding(),
        "Font with Custom encoding should return true"
    );

    let font_with_standard =
        create_font_with_encoding(Encoding::Standard("WinAnsiEncoding".to_string()));
    assert!(
        !font_with_standard.has_custom_encoding(),
        "Font with Standard encoding should return false"
    );

    let font_with_identity = create_font_with_encoding(Encoding::Identity);
    assert!(
        !font_with_identity.has_custom_encoding(),
        "Font with Identity encoding should return false"
    );
}

#[test]
fn test_standard_encoding_no_custom() {
    // Standard encoding fonts should not be identified as having custom encoding

    let font_info = create_font_with_encoding(Encoding::Standard("WinAnsiEncoding".to_string()));

    assert!(
        !font_info.has_custom_encoding(),
        "Standard encoding should not be identified as custom"
    );

    // For ASCII codes, standard encoding should pass through
    let normalized = font_info.get_encoded_char(0x41); // 'A'
    assert_eq!(normalized, Some('A'), "Standard encoding should pass through ASCII");
}

#[test]
fn test_identity_encoding_no_custom() {
    // Identity encoding fonts should not be identified as having custom encoding

    let font_info = create_font_with_encoding(Encoding::Identity);

    assert!(
        !font_info.has_custom_encoding(),
        "Identity encoding should not be identified as custom"
    );
}

#[test]
fn test_encoding_normalization_priority() {
    // Test that /Differences takes priority in custom encoding
    // This simulates a font with /BaseEncoding /WinAnsiEncoding /Differences [0x64 /rho]
    // Code 0x64 should map to rho, not 'd'

    let mut mappings = HashMap::new();
    mappings.insert(0x64, 'ρ'); // Override 0x64 to Greek rho

    let font_info = create_font_with_encoding(Encoding::Custom(mappings));

    let normalized = font_info.get_encoded_char(0x64);
    assert_eq!(normalized, Some('ρ'), "/Differences should override base encoding");
}

#[test]
fn test_custom_encoding_unmapped_code() {
    // Test that unmapped codes in custom encoding return None

    let mut mappings = HashMap::new();
    mappings.insert(0x64, 'ρ'); // Only 0x64 is mapped

    let font_info = create_font_with_encoding(Encoding::Custom(mappings));

    // 0x65 is not mapped
    let normalized = font_info.get_encoded_char(0x65);
    assert_eq!(normalized, None, "Unmapped code in custom encoding should return None");
}

#[test]
fn test_custom_encoding_multiple_mappings() {
    // Test custom encoding with multiple character mappings

    let mut mappings = HashMap::new();
    mappings.insert(0x64, 'ρ'); // Greek rho
    mappings.insert(0x65, 'σ'); // Greek sigma
    mappings.insert(0x66, 'τ'); // Greek tau

    let font_info = create_font_with_encoding(Encoding::Custom(mappings));

    assert_eq!(font_info.get_encoded_char(0x64), Some('ρ'));
    assert_eq!(font_info.get_encoded_char(0x65), Some('σ'));
    assert_eq!(font_info.get_encoded_char(0x66), Some('τ'));
}

#[test]
fn test_custom_encoding_with_special_characters() {
    // Test custom encoding with special characters (symbols, punctuation)

    let mut mappings = HashMap::new();
    mappings.insert(0x80, '€'); // Euro sign
    mappings.insert(0x81, '©'); // Copyright
    mappings.insert(0x82, '®'); // Registered trademark

    let font_info = create_font_with_encoding(Encoding::Custom(mappings));

    assert_eq!(font_info.get_encoded_char(0x80), Some('€'));
    assert_eq!(font_info.get_encoded_char(0x81), Some('©'));
    assert_eq!(font_info.get_encoded_char(0x82), Some('®'));
}

#[test]
fn test_standard_encoding_ascii_range() {
    // Test standard encoding with ASCII character range

    let font_info = create_font_with_encoding(Encoding::Standard("WinAnsiEncoding".to_string()));

    // ASCII lowercase
    assert_eq!(font_info.get_encoded_char(0x61), Some('a'));
    assert_eq!(font_info.get_encoded_char(0x7A), Some('z'));

    // ASCII uppercase
    assert_eq!(font_info.get_encoded_char(0x41), Some('A'));
    assert_eq!(font_info.get_encoded_char(0x5A), Some('Z'));

    // ASCII digits
    assert_eq!(font_info.get_encoded_char(0x30), Some('0'));
    assert_eq!(font_info.get_encoded_char(0x39), Some('9'));
}

#[test]
fn test_identity_encoding_passthrough() {
    // Test that identity encoding passes through codes as-is

    let font_info = create_font_with_encoding(Encoding::Identity);

    // ASCII range should pass through
    assert_eq!(font_info.get_encoded_char(0x41), Some('A'));
    assert_eq!(font_info.get_encoded_char(0x61), Some('a'));
    assert_eq!(font_info.get_encoded_char(0x30), Some('0'));
}

// Helper function to create a FontInfo with specific encoding
fn create_font_with_encoding(encoding: Encoding) -> FontInfo {
    FontInfo {
        base_font: "TestFont".to_string(),
        subtype: "Type1".to_string(),
        encoding,
        to_unicode: None,
        font_weight: Some(400),
        flags: Some(32),
        stem_v: Some(100.0),
        embedded_font_data: None,
        truetype_cmap: None,
        cid_to_gid_map: None,
        cid_system_info: None,
        cid_font_type: None,
        widths: None,
        first_char: None,
        last_char: None,
        default_width: 500.0,
    }
}
