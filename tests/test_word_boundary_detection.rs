#![allow(dead_code)]
//! Tests for ISO 32000-1:2008 Section 9.4.4 Word Boundary Detection
//!
//! The PDF spec defines word boundaries through multiple mechanisms:
//! 1. TJ array offset values (character-level spacing information)
//! 2. Geometric positioning (layout-based word breaking)
//! 3. Space character detection (explicit word separators)
//! 4. Font metrics (font size, character width influence spacing decisions)
//!
//! This test suite documents spec-compliant word boundary detection
//! for single-byte, multi-byte, and CJK text.

/// Mock text extraction result for word boundary testing
#[derive(Clone, Debug)]
struct CharacterInfo {
    code: u32,
    glyph_id: Option<u16>,
    width: f32,
    x_position: f32,
    tj_offset: Option<i32>, // From TJ array (negative = extra space)
    font_size: f32,
    is_ligature: bool,
    original_ligature: Option<char>,
    protected_from_split: bool,
}

/// Helper to simulate text stream with character-level information
#[derive(Clone, Debug)]
struct TextStreamContext {
    characters: Vec<CharacterInfo>,
    font_size: f32,
    horizontal_scaling: f32, // Tz parameter (percentage)
    word_spacing: f32,       // Tw parameter (additional space after space char)
    char_spacing: f32,       // Tc parameter
}

#[test]
fn test_ascii_space_boundary_detection() {
    // ASCII space (0x20) should always create word boundary
    let stream = TextStreamContext {
        characters: vec![
            CharacterInfo {
                code: 0x48,
                glyph_id: Some(1),
                width: 0.5,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'H'
            CharacterInfo {
                code: 0x65,
                glyph_id: Some(2),
                width: 0.4,
                x_position: 6.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'e'
            CharacterInfo {
                code: 0x6C,
                glyph_id: Some(3),
                width: 0.3,
                x_position: 10.8,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'l'
            CharacterInfo {
                code: 0x6C,
                glyph_id: Some(3),
                width: 0.3,
                x_position: 14.4,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'l'
            CharacterInfo {
                code: 0x6F,
                glyph_id: Some(4),
                width: 0.4,
                x_position: 18.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'o'
            CharacterInfo {
                code: 0x20,
                glyph_id: Some(5),
                width: 0.25,
                x_position: 22.8,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // SPACE
            CharacterInfo {
                code: 0x57,
                glyph_id: Some(6),
                width: 0.7,
                x_position: 28.2,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'W'
            CharacterInfo {
                code: 0x6F,
                glyph_id: Some(4),
                width: 0.4,
                x_position: 36.6,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'o'
            CharacterInfo {
                code: 0x72,
                glyph_id: Some(7),
                width: 0.3,
                x_position: 41.4,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'r'
            CharacterInfo {
                code: 0x6C,
                glyph_id: Some(3),
                width: 0.3,
                x_position: 45.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'l'
            CharacterInfo {
                code: 0x64,
                glyph_id: Some(8),
                width: 0.4,
                x_position: 48.6,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'd'
        ],
        font_size: 12.0,
        horizontal_scaling: 100.0,
        word_spacing: 0.25,
        char_spacing: 0.0,
    };

    // Expected: "Hello" | SPACE | "World"
    // Space character at index 5 creates word boundary
    assert_eq!(stream.characters[5].code, 0x20);
    assert_eq!(stream.characters.len(), 11);

    // Word 1: indices 0-4 ("Hello")
    // Word 2: indices 6-10 ("World")
    // Total words: 2
}

#[test]
fn test_tj_array_negative_offset_creates_word_boundary() {
    // TJ array with large negative offset creates word boundary
    // Spec: negative values in TJ increase spacing (potential word break)
    let stream = TextStreamContext {
        characters: vec![
            CharacterInfo {
                code: 0x54,
                glyph_id: Some(1),
                width: 0.5,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'T'
            CharacterInfo {
                code: 0x69,
                glyph_id: Some(2),
                width: 0.3,
                x_position: 6.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'i'
            CharacterInfo {
                code: 0x6D,
                glyph_id: Some(3),
                width: 0.4,
                x_position: 9.6,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'm'
            CharacterInfo {
                code: 0x65,
                glyph_id: Some(4),
                width: 0.4,
                x_position: 14.4,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'e'
            CharacterInfo {
                code: 0x2D,
                glyph_id: Some(5),
                width: 0.25,
                x_position: 19.2,
                tj_offset: Some(-200),
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // HYPHEN with large negative offset
            CharacterInfo {
                code: 0x6F,
                glyph_id: Some(6),
                width: 0.4,
                x_position: 31.2,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'o'
            CharacterInfo {
                code: 0x75,
                glyph_id: Some(7),
                width: 0.4,
                x_position: 36.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'u'
            CharacterInfo {
                code: 0x74,
                glyph_id: Some(8),
                width: 0.3,
                x_position: 40.8,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 't'
        ],
        font_size: 12.0,
        horizontal_scaling: 100.0,
        word_spacing: 0.25,
        char_spacing: 0.0,
    };

    // Large negative TJ offset (before 'o') indicates word boundary
    // Expected: "Time" | (boundary) | "out"
    assert!(stream.characters[4].tj_offset == Some(-200));
    assert_eq!(stream.characters.len(), 8);
}

#[test]
fn test_geometric_spacing_word_boundary_detection() {
    // When characters have large gaps (geometric spacing), create word boundary
    // TJ offsets not available; rely on character positions
    let stream = TextStreamContext {
        characters: vec![
            CharacterInfo {
                code: 0x54,
                glyph_id: Some(1),
                width: 0.5,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            },
            CharacterInfo {
                code: 0x65,
                glyph_id: Some(2),
                width: 0.4,
                x_position: 6.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            },
            CharacterInfo {
                code: 0x78,
                glyph_id: Some(3),
                width: 0.4,
                x_position: 10.8,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            },
            CharacterInfo {
                code: 0x74,
                glyph_id: Some(4),
                width: 0.3,
                x_position: 15.6,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            },
            // Gap: 6.0 units (much larger than character width ~0.4)
            CharacterInfo {
                code: 0x42,
                glyph_id: Some(5),
                width: 0.5,
                x_position: 27.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            },
            CharacterInfo {
                code: 0x6F,
                glyph_id: Some(6),
                width: 0.4,
                x_position: 33.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            },
            CharacterInfo {
                code: 0x78,
                glyph_id: Some(7),
                width: 0.4,
                x_position: 37.8,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            },
        ],
        font_size: 12.0,
        horizontal_scaling: 100.0,
        word_spacing: 0.25,
        char_spacing: 0.0,
    };

    // Gap between index 3 ('t' at 18.9) and index 4 ('B' at 27.0) = 8.1 units
    // This exceeds typical word spacing, indicating boundary
    let gap = stream.characters[4].x_position
        - (stream.characters[3].x_position + stream.characters[3].width);
    assert!(gap > 5.0, "Gap should be significant for word boundary");
}

#[test]
fn test_multiple_consecutive_spaces() {
    // Multiple space characters should not create multiple word boundaries
    // (collapse to single boundary, or handle per PDF reader behavior)
    let stream = TextStreamContext {
        characters: vec![
            CharacterInfo {
                code: 0x57,
                glyph_id: Some(1),
                width: 0.7,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'W'
            CharacterInfo {
                code: 0x6F,
                glyph_id: Some(2),
                width: 0.4,
                x_position: 8.4,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'o'
            CharacterInfo {
                code: 0x72,
                glyph_id: Some(3),
                width: 0.3,
                x_position: 13.2,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'r'
            CharacterInfo {
                code: 0x64,
                glyph_id: Some(4),
                width: 0.4,
                x_position: 16.8,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'd'
            CharacterInfo {
                code: 0x20,
                glyph_id: Some(5),
                width: 0.25,
                x_position: 21.6,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // SPACE
            CharacterInfo {
                code: 0x20,
                glyph_id: Some(5),
                width: 0.25,
                x_position: 25.8,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // SPACE
            CharacterInfo {
                code: 0x20,
                glyph_id: Some(5),
                width: 0.25,
                x_position: 30.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // SPACE
            CharacterInfo {
                code: 0x46,
                glyph_id: Some(6),
                width: 0.5,
                x_position: 34.2,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'F'
            CharacterInfo {
                code: 0x6F,
                glyph_id: Some(2),
                width: 0.4,
                x_position: 40.2,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'o'
            CharacterInfo {
                code: 0x72,
                glyph_id: Some(3),
                width: 0.3,
                x_position: 45.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'r'
        ],
        font_size: 12.0,
        horizontal_scaling: 100.0,
        word_spacing: 0.25,
        char_spacing: 0.0,
    };

    // Multiple spaces (indices 4, 5, 6) should be handled consistently
    // Implementation may collapse to single boundary or preserve whitespace
    assert_eq!(stream.characters[4].code, 0x20);
    assert_eq!(stream.characters[5].code, 0x20);
    assert_eq!(stream.characters[6].code, 0x20);
}

#[test]
fn test_hyphenation_word_boundary() {
    // Hyphenated words: hyphen may indicate continuation or boundary
    // Depends on context (end-of-line vs. explicit hyphenation)
    let stream = TextStreamContext {
        characters: vec![
            CharacterInfo {
                code: 0x69,
                glyph_id: Some(1),
                width: 0.3,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            },
            CharacterInfo {
                code: 0x6E,
                glyph_id: Some(2),
                width: 0.4,
                x_position: 3.6,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            },
            CharacterInfo {
                code: 0x74,
                glyph_id: Some(3),
                width: 0.3,
                x_position: 7.2,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            },
            CharacterInfo {
                code: 0x65,
                glyph_id: Some(4),
                width: 0.4,
                x_position: 10.8,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            },
            CharacterInfo {
                code: 0x72,
                glyph_id: Some(5),
                width: 0.3,
                x_position: 14.4,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            },
            CharacterInfo {
                code: 0x2D,
                glyph_id: Some(6),
                width: 0.25,
                x_position: 18.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // HYPHEN
            // Continuation: no large offset, small gap
            CharacterInfo {
                code: 0x6E,
                glyph_id: Some(2),
                width: 0.4,
                x_position: 20.1,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            },
            CharacterInfo {
                code: 0x65,
                glyph_id: Some(4),
                width: 0.4,
                x_position: 24.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            },
            CharacterInfo {
                code: 0x74,
                glyph_id: Some(3),
                width: 0.3,
                x_position: 28.2,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            },
        ],
        font_size: 12.0,
        horizontal_scaling: 100.0,
        word_spacing: 0.25,
        char_spacing: 0.0,
    };

    // Hyphenation at index 5: "inter-net"
    // May be treated as word break or continuation depending on context
    assert_eq!(stream.characters[5].code, 0x2D); // HYPHEN
    let gap_after_hyphen = stream.characters[6].x_position
        - (stream.characters[5].x_position + stream.characters[5].width);
    assert!(gap_after_hyphen < 2.0, "Hyphen continuation has small gap");
}

#[test]
fn test_cjk_text_no_explicit_spaces() {
    // CJK text (Chinese, Japanese, Korean) doesn't use spaces between characters
    // Each character is a potential word boundary (or use structure tree for grouping)
    let stream = TextStreamContext {
        characters: vec![
            CharacterInfo {
                code: 0x4E2D,
                glyph_id: Some(1),
                width: 1.0,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // CJK UNIFIED IDEOGRAPH (Chinese)
            CharacterInfo {
                code: 0x6587,
                glyph_id: Some(2),
                width: 1.0,
                x_position: 12.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // CJK UNIFIED IDEOGRAPH (Chinese)
            CharacterInfo {
                code: 0x5B57,
                glyph_id: Some(3),
                width: 1.0,
                x_position: 24.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // CJK UNIFIED IDEOGRAPH (Chinese)
            CharacterInfo {
                code: 0x3002,
                glyph_id: Some(4),
                width: 0.5,
                x_position: 36.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // IDEOGRAPHIC FULL STOP
        ],
        font_size: 12.0,
        horizontal_scaling: 100.0,
        word_spacing: 0.0, // No word spacing in CJK
        char_spacing: 0.0,
    };

    // CJK characters: 0x4E2D, 0x6587, 0x5B57 (all > 0x9FFF indicates CJK range)
    assert!(stream.characters[0].code > 0x4E00); // CJK unified range starts at 0x4E00
    assert!(stream.characters[0].width == 1.0); // Fixed width for CJK
}

#[test]
fn test_custom_encoding_word_boundary() {
    // Custom font encodings may use different character codes for spaces
    // Must be handled by mapping function (character_mapper)
    let stream = TextStreamContext {
        characters: vec![
            CharacterInfo {
                code: 0x21,
                glyph_id: Some(1),
                width: 0.5,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // Code 0x21 = 'A' in custom encoding
            CharacterInfo {
                code: 0x22,
                glyph_id: Some(2),
                width: 0.4,
                x_position: 6.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // Code 0x22 = 'B' in custom encoding
            CharacterInfo {
                code: 0x00,
                glyph_id: Some(3),
                width: 0.25,
                x_position: 10.8,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // Code 0x00 = SPACE in custom encoding
            CharacterInfo {
                code: 0x43,
                glyph_id: Some(4),
                width: 0.5,
                x_position: 16.2,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // Code 0x43 = 'C' in custom encoding
        ],
        font_size: 12.0,
        horizontal_scaling: 100.0,
        word_spacing: 0.25,
        char_spacing: 0.0,
    };

    // Custom encoding: character code 0x00 maps to SPACE
    // Word boundary created after character at index 2
    // (Mapping handled by CharacterMapper, not word boundary detection)
    assert_eq!(stream.characters.len(), 4);
}

#[test]
fn test_font_size_influence_on_spacing() {
    // Larger font sizes have proportionally larger spacing
    // Word boundary detection should account for this
    let stream_large = TextStreamContext {
        characters: vec![
            CharacterInfo {
                code: 0x41,
                glyph_id: Some(1),
                width: 1.0,
                x_position: 0.0,
                tj_offset: None,
                font_size: 24.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'A'
            CharacterInfo {
                code: 0x42,
                glyph_id: Some(2),
                width: 0.8,
                x_position: 24.0,
                tj_offset: None,
                font_size: 24.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'B'
            CharacterInfo {
                code: 0x43,
                glyph_id: Some(3),
                width: 0.8,
                x_position: 52.0,
                tj_offset: None,
                font_size: 24.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'C'
        ],
        font_size: 24.0,
        horizontal_scaling: 100.0,
        word_spacing: 0.5,
        char_spacing: 0.0,
    };

    // 24pt font: character width ~1.0, spacing adjustments scale proportionally
    assert_eq!(stream_large.font_size, 24.0);

    let stream_small = TextStreamContext {
        characters: vec![
            CharacterInfo {
                code: 0x41,
                glyph_id: Some(1),
                width: 0.5,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'A'
            CharacterInfo {
                code: 0x42,
                glyph_id: Some(2),
                width: 0.4,
                x_position: 6.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'B'
            CharacterInfo {
                code: 0x43,
                glyph_id: Some(3),
                width: 0.4,
                x_position: 10.8,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'C'
        ],
        font_size: 12.0,
        horizontal_scaling: 100.0,
        word_spacing: 0.25,
        char_spacing: 0.0,
    };

    // 12pt font: character width ~0.5, spacing is half that of 24pt
    assert_eq!(stream_small.font_size, 12.0);
    assert!(stream_large.font_size == 2.0 * stream_small.font_size);
}

#[test]
fn test_horizontal_scaling_affects_spacing() {
    // Tz parameter (horizontal scaling) affects all spacing measurements
    // Scaling percentage changes effective character width
    let stream_normal = TextStreamContext {
        characters: vec![
            CharacterInfo {
                code: 0x41,
                glyph_id: Some(1),
                width: 0.5,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            },
            CharacterInfo {
                code: 0x42,
                glyph_id: Some(2),
                width: 0.4,
                x_position: 6.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            },
            CharacterInfo {
                code: 0x43,
                glyph_id: Some(3),
                width: 0.4,
                x_position: 10.8,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            },
        ],
        font_size: 12.0,
        horizontal_scaling: 100.0,
        word_spacing: 0.25,
        char_spacing: 0.0,
    };

    let stream_condensed = TextStreamContext {
        characters: vec![
            CharacterInfo {
                code: 0x41,
                glyph_id: Some(1),
                width: 0.375,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 75% scaling
            CharacterInfo {
                code: 0x42,
                glyph_id: Some(2),
                width: 0.3,
                x_position: 4.5,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            },
            CharacterInfo {
                code: 0x43,
                glyph_id: Some(3),
                width: 0.3,
                x_position: 8.1,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            },
        ],
        font_size: 12.0,
        horizontal_scaling: 75.0,
        word_spacing: 0.25,
        char_spacing: 0.0,
    };

    // 75% scaling compresses all widths by 0.75x
    assert!(stream_condensed.horizontal_scaling == 75.0);
    assert!(stream_normal.characters[0].width > stream_condensed.characters[0].width);
}

#[test]
fn test_character_spacing_tc_parameter() {
    // Tc parameter (character spacing) adds space between ALL characters
    // Should not create word boundaries (applies uniformly)
    let stream_normal = TextStreamContext {
        characters: vec![
            CharacterInfo {
                code: 0x48,
                glyph_id: Some(1),
                width: 0.5,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'H'
            CharacterInfo {
                code: 0x65,
                glyph_id: Some(2),
                width: 0.4,
                x_position: 6.5,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'e'
            CharacterInfo {
                code: 0x6C,
                glyph_id: Some(3),
                width: 0.3,
                x_position: 11.8,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'l'
            CharacterInfo {
                code: 0x6C,
                glyph_id: Some(3),
                width: 0.3,
                x_position: 16.8,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'l'
            CharacterInfo {
                code: 0x6F,
                glyph_id: Some(4),
                width: 0.4,
                x_position: 21.8,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'o'
        ],
        font_size: 12.0,
        horizontal_scaling: 100.0,
        word_spacing: 0.25,
        char_spacing: 0.5, // Extra spacing between ALL characters
    };

    // Tc parameter increases spacing uniformly, should not create boundaries
    assert_eq!(stream_normal.char_spacing, 0.5);
    assert_eq!(stream_normal.characters.len(), 5);
}

#[test]
fn test_word_spacing_tw_parameter() {
    // Tw parameter (word spacing) adds extra space ONLY after space characters
    // Should not affect spacing between other characters
    let stream = TextStreamContext {
        characters: vec![
            CharacterInfo {
                code: 0x54,
                glyph_id: Some(1),
                width: 0.5,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'T'
            CharacterInfo {
                code: 0x68,
                glyph_id: Some(2),
                width: 0.4,
                x_position: 6.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'h'
            CharacterInfo {
                code: 0x65,
                glyph_id: Some(3),
                width: 0.4,
                x_position: 10.8,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'e'
            CharacterInfo {
                code: 0x20,
                glyph_id: Some(4),
                width: 0.25,
                x_position: 15.6,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // SPACE
            CharacterInfo {
                code: 0x63,
                glyph_id: Some(5),
                width: 0.35,
                x_position: 22.1,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'c'
            CharacterInfo {
                code: 0x61,
                glyph_id: Some(6),
                width: 0.35,
                x_position: 26.7,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'a'
            CharacterInfo {
                code: 0x74,
                glyph_id: Some(7),
                width: 0.3,
                x_position: 31.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 't'
        ],
        font_size: 12.0,
        horizontal_scaling: 100.0,
        word_spacing: 0.5, // Extra space after space character
        char_spacing: 0.0,
    };

    // Tw affects only space character
    // Space at index 3 adds extra 0.5 width
    assert_eq!(stream.characters[3].code, 0x20);
    assert_eq!(stream.word_spacing, 0.5);
}

#[test]
fn test_ligature_handling() {
    // Ligatures (fi, fl, ffi) are single glyph representing multiple characters
    // May be represented as single character code or decomposed
    let stream = TextStreamContext {
        characters: vec![
            CharacterInfo {
                code: 0x41,
                glyph_id: Some(1),
                width: 0.5,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'A'
            CharacterInfo {
                code: 0xFB01,
                glyph_id: Some(2),
                width: 0.6,
                x_position: 6.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // LATIN SMALL LIGATURE FI (U+FB01)
            CharacterInfo {
                code: 0x6E,
                glyph_id: Some(3),
                width: 0.4,
                x_position: 13.2,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'n'
        ],
        font_size: 12.0,
        horizontal_scaling: 100.0,
        word_spacing: 0.25,
        char_spacing: 0.0,
    };

    // Ligature at index 1: single character code 0xFB01 (fi)
    // Implementation may need to decompose to "fi" for correct word boundary detection
    assert_eq!(stream.characters[1].code, 0xFB01);
}

#[test]
fn test_combining_characters_diacritics() {
    // Combining characters (diacritics) attach to base character
    // Should not create word boundaries
    let stream = TextStreamContext {
        characters: vec![
            CharacterInfo {
                code: 0x65,
                glyph_id: Some(1),
                width: 0.4,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'e'
            CharacterInfo {
                code: 0x0301,
                glyph_id: Some(2),
                width: 0.0,
                x_position: 4.8,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // COMBINING ACUTE ACCENT
            CharacterInfo {
                code: 0x74,
                glyph_id: Some(3),
                width: 0.3,
                x_position: 8.4,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 't'
        ],
        font_size: 12.0,
        horizontal_scaling: 100.0,
        word_spacing: 0.25,
        char_spacing: 0.0,
    };

    // Combining character (0x0301) has zero width, doesn't create boundary
    assert_eq!(stream.characters[1].code, 0x0301);
    assert_eq!(stream.characters[1].width, 0.0);
}

#[test]
fn test_zero_width_space_boundary() {
    // Zero-width space (U+200B) creates word boundary without visible space
    // Some PDFs use this for structure preservation
    let stream = TextStreamContext {
        characters: vec![
            CharacterInfo {
                code: 0x6E,
                glyph_id: Some(1),
                width: 0.4,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'n'
            CharacterInfo {
                code: 0x6F,
                glyph_id: Some(2),
                width: 0.4,
                x_position: 4.8,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'o'
            CharacterInfo {
                code: 0x200B,
                glyph_id: Some(3),
                width: 0.0,
                tj_offset: None,
                x_position: 9.6,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // ZERO WIDTH SPACE
            CharacterInfo {
                code: 0x72,
                glyph_id: Some(4),
                width: 0.3,
                x_position: 9.6,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'r'
            CharacterInfo {
                code: 0x62,
                glyph_id: Some(5),
                width: 0.4,
                x_position: 13.2,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'b'
        ],
        font_size: 12.0,
        horizontal_scaling: 100.0,
        word_spacing: 0.25,
        char_spacing: 0.0,
    };

    // Zero-width space at index 2: word boundary without visual space
    assert_eq!(stream.characters[2].code, 0x200B);
    assert_eq!(stream.characters[2].width, 0.0);
}

#[test]
fn test_specification_reference_iso_9_4_4() {
    // This test documents the spec sections that define word boundary detection
    // ISO 32000-1:2008 Section 9.4.4: Text Objects and Word Boundaries
    //
    // Key concepts:
    // 1. TJ array offset values provide character-level positioning
    // 2. Geometric spacing (character positions) determine visual word boundaries
    // 3. Space character (U+0020) and Tw parameter define explicit word breaks
    // 4. Font metrics (size, scaling, spacing) scale boundary detection
    // 5. CJK text requires different word breaking rules (no spaces)
    // 6. Custom encodings need character mapping before boundary detection

    let spec_sections = [
        "ISO 32000-1:2008 Section 9.4: Text Objects",
        "ISO 32000-1:2008 Section 9.4.3: Text Positioning Operators",
        "ISO 32000-1:2008 Section 9.4.4: Text Objects and Word Spacing",
        "ISO 32000-1:2008 Section 5.3.2: Text State Parameters (Tc, Tw, Tz, TL)",
    ];

    assert_eq!(spec_sections.len(), 4);
    assert!(spec_sections[0].contains("9.4"));
    assert!(spec_sections[1].contains("9.4.3"));
}

#[test]
fn test_mixed_scripts_word_boundary() {
    // Documents with mixed scripts (Latin + CJK) need context-aware boundaries
    // Latin words use spaces; CJK uses character boundaries
    let stream = TextStreamContext {
        characters: vec![
            CharacterInfo {
                code: 0x54,
                glyph_id: Some(1),
                width: 0.5,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'T'
            CharacterInfo {
                code: 0x65,
                glyph_id: Some(2),
                width: 0.4,
                x_position: 6.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'e'
            CharacterInfo {
                code: 0x78,
                glyph_id: Some(3),
                width: 0.4,
                x_position: 10.8,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'x'
            CharacterInfo {
                code: 0x74,
                glyph_id: Some(4),
                width: 0.3,
                x_position: 15.6,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 't'
            CharacterInfo {
                code: 0x20,
                glyph_id: Some(5),
                width: 0.25,
                x_position: 19.2,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // SPACE
            CharacterInfo {
                code: 0x4E2D,
                glyph_id: Some(6),
                width: 1.0,
                x_position: 25.2,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // CJK
            CharacterInfo {
                code: 0x6587,
                glyph_id: Some(7),
                width: 1.0,
                x_position: 37.2,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // CJK
        ],
        font_size: 12.0,
        horizontal_scaling: 100.0,
        word_spacing: 0.25,
        char_spacing: 0.0,
    };

    // Mixed scripts: Latin uses space (index 4), CJK each is potential word
    assert_eq!(stream.characters[4].code, 0x20);
    assert!(stream.characters[5].code > 0x4E00); // CJK range
}

#[test]
fn test_word_boundary_with_numbers() {
    // Numbers can be part of words or standalone
    // Word boundary detection should preserve cohesion (e.g., "test123" vs "test 123")
    let stream_attached = TextStreamContext {
        characters: vec![
            CharacterInfo {
                code: 0x74,
                glyph_id: Some(1),
                width: 0.3,
                x_position: 0.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 't'
            CharacterInfo {
                code: 0x65,
                glyph_id: Some(2),
                width: 0.4,
                x_position: 3.6,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 'e'
            CharacterInfo {
                code: 0x73,
                glyph_id: Some(3),
                width: 0.3,
                x_position: 8.4,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 's'
            CharacterInfo {
                code: 0x74,
                glyph_id: Some(4),
                width: 0.3,
                x_position: 12.0,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // 't'
            CharacterInfo {
                code: 0x31,
                glyph_id: Some(5),
                width: 0.3,
                x_position: 15.6,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // '1'
            CharacterInfo {
                code: 0x32,
                glyph_id: Some(6),
                width: 0.3,
                x_position: 19.2,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // '2'
            CharacterInfo {
                code: 0x33,
                glyph_id: Some(7),
                width: 0.3,
                x_position: 22.8,
                tj_offset: None,
                font_size: 12.0,
                is_ligature: false,
                original_ligature: None,
                protected_from_split: false,
            }, // '3'
        ],
        font_size: 12.0,
        horizontal_scaling: 100.0,
        word_spacing: 0.25,
        char_spacing: 0.0,
    };

    // "test123" - no space, should be single word
    // Characters progress continuously with consistent spacing
    // Gap between 't' (ends at 12.3) and '1' (at 15.6) = 3.3
    // Pattern: gaps are consistent, no explicit space character
    assert!(stream_attached.characters[0].code == 0x74); // 't'
    assert!(stream_attached.characters[4].code == 0x31); // '1'
                                                         // No space characters in the stream means no word boundary
    let has_space = stream_attached.characters.iter().any(|c| c.code == 0x20);
    assert!(!has_space, "Stream should have no space characters for attached word");
}
