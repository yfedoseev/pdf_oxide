//! Integration tests for WordBoundaryDetector integration with text extraction.
//!
//! These tests verify that WordBoundaryDetector is properly wired into the
//! text extraction pipeline and resolves word concatenation issues like
//! "VerDateSep" → "VerDate Sep".
//!
//! TDD: Write tests FIRST, then implement the integration.

use pdf_oxide::text::{BoundaryContext, CharacterInfo, WordBoundaryDetector};

/// Test 1: VerDate Sep boundary detection (target issue)
/// This is the primary issue we're fixing - words like "VerDateSep" should
/// be extracted as "VerDate Sep" when TJ offsets indicate word boundaries.
#[test]
fn test_verdate_sep_word_boundary_detection() {
    // Simulates: [("VerDate") (-600) ("Sep") (-1500) ("2014")]
    // The TJ offset -600 between "VerDate" and "Sep" should trigger a word boundary

    let detector = WordBoundaryDetector::new();
    let context = BoundaryContext {
        font_size: 12.0,
        horizontal_scaling: 100.0,
        word_spacing: 0.0,
        char_spacing: 0.0,
    };

    // Build CharacterInfo for the boundary between "VerDate" and "Sep"
    // Last char of "VerDate" is 'e', first char of "Sep" is 'S'
    let characters = vec![
        // 'e' (end of "VerDate") with TJ offset indicating space
        CharacterInfo {
            code: 'e' as u32,
            glyph_id: None,
            width: 6.0,
            x_position: 42.0,
            tj_offset: Some(-600), // Large negative offset = word boundary
            font_size: 12.0,
            is_ligature: false,
            original_ligature: None,
            protected_from_split: false,
        },
        // 'S' (start of "Sep") - gap indicates word boundary
        CharacterInfo {
            code: 'S' as u32,
            glyph_id: None,
            width: 7.0,
            x_position: 55.0, // Gap of 13 units (55 - 42) > threshold
            tj_offset: None,
            font_size: 12.0,
            is_ligature: false,
            original_ligature: None,
            protected_from_split: false,
        },
    ];

    let boundaries = detector.detect_word_boundaries(&characters, &context);

    // Expect boundary at index 1 (between 'e' and 'S')
    assert!(
        boundaries.contains(&1),
        "Should detect word boundary between 'VerDate' and 'Sep', got: {:?}",
        boundaries
    );
}

/// Test 2: No false positives in tight kerning
/// Kerning pairs like "AV" have negative TJ offsets for visual adjustment,
/// but these should NOT create word boundaries.
/// Uses static threshold (not adaptive) to test legacy behavior.
#[test]
fn test_no_false_boundary_in_kerning_pairs() {
    let detector = WordBoundaryDetector::new().with_adaptive_threshold(false); // Use static -100 threshold for kerning test
    let context = BoundaryContext::new(12.0);

    // "AV" with kerning offset -40 (tight kerning, NOT word boundary)
    // Default threshold is -100, so -40 should NOT trigger
    let characters = vec![
        CharacterInfo {
            code: 'A' as u32,
            glyph_id: None,
            width: 8.0,
            x_position: 0.0,
            tj_offset: Some(-40), // Small negative offset = kerning only
            font_size: 12.0,
            is_ligature: false,
            original_ligature: None,
            protected_from_split: false,
        },
        CharacterInfo {
            code: 'V' as u32,
            glyph_id: None,
            width: 8.0,
            x_position: 7.5, // Positioned slightly closer due to kerning
            tj_offset: None,
            font_size: 12.0,
            is_ligature: false,
            original_ligature: None,
            protected_from_split: false,
        },
    ];

    let boundaries = detector.detect_word_boundaries(&characters, &context);

    // Should NOT have a boundary - this is kerning, not a word break
    assert!(
        boundaries.is_empty(),
        "Should NOT detect boundary in kerning pair 'AV', got: {:?}",
        boundaries
    );
}

/// Test 3: Geometric gap detection works
/// Even without TJ offset signal, a large geometric gap should create a boundary.
#[test]
fn test_geometric_gap_creates_boundary() {
    // Use geometric_gap_ratio of 0.5 (50% of font size = 6pt threshold for 12pt font)
    let detector = WordBoundaryDetector::new().with_geometric_gap_ratio(0.5);
    let context = BoundaryContext::new(12.0);

    // Two characters with a gap larger than 50% of font size
    let characters = vec![
        CharacterInfo {
            code: 'A' as u32,
            glyph_id: None,
            width: 8.0,
            x_position: 0.0,
            tj_offset: None, // No TJ offset signal
            font_size: 12.0,
            is_ligature: false,
            original_ligature: None,
            protected_from_split: false,
        },
        CharacterInfo {
            code: 'B' as u32,
            glyph_id: None,
            width: 8.0,
            x_position: 20.0, // Gap of 12 units (20 - 8) > 6pt threshold
            tj_offset: None,
            font_size: 12.0,
            is_ligature: false,
            original_ligature: None,
            protected_from_split: false,
        },
    ];

    let boundaries = detector.detect_word_boundaries(&characters, &context);

    // Should detect boundary due to geometric gap
    assert!(
        boundaries.contains(&1),
        "Should detect geometric gap boundary, got: {:?}",
        boundaries
    );
}

/// Test 4: CJK character boundaries with geometric gaps
/// CJK characters with sufficient geometric spacing should create word boundaries.
/// Week 2 Days 8-9: CJK script detection uses geometric gaps for Chinese text.
#[test]
fn test_cjk_character_word_boundaries() {
    let detector = WordBoundaryDetector::new();
    let context = BoundaryContext::new(12.0);

    // Three CJK characters: 中文字 with geometric gaps
    // For Chinese text, boundaries are determined by geometric spacing, not per-character
    let characters = vec![
        CharacterInfo {
            code: 0x4E2D, // 中
            glyph_id: None,
            width: 1.0, // Small width to create gap
            x_position: 0.0,
            tj_offset: None,
            font_size: 12.0,
            is_ligature: false,
            original_ligature: None,
            protected_from_split: false,
        },
        CharacterInfo {
            code: 0x6587, // 文
            glyph_id: None,
            width: 1.0,
            x_position: 12.0, // Gap of 11.0 units from previous char end (1.0)
            tj_offset: None,
            font_size: 12.0,
            is_ligature: false,
            original_ligature: None,
            protected_from_split: false,
        },
        CharacterInfo {
            code: 0x5B57, // 字
            glyph_id: None,
            width: 1.0,
            x_position: 24.0, // Gap of 11.0 units from previous char end (13.0)
            tj_offset: None,
            font_size: 12.0,
            is_ligature: false,
            original_ligature: None,
            protected_from_split: false,
        },
    ];

    let boundaries = detector.detect_word_boundaries(&characters, &context);

    // With geometric gaps (11.0 > threshold 9.6), boundaries should be created
    assert!(
        boundaries.contains(&1),
        "Should have boundary after first CJK char due to geometric gap, got: {:?}",
        boundaries
    );
    assert!(
        boundaries.contains(&2),
        "Should have boundary after second CJK char due to geometric gap, got: {:?}",
        boundaries
    );
}

/// Test 5: Explicit space character creates boundary
/// ASCII space (U+0020) should always create a word boundary.
#[test]
fn test_explicit_space_creates_boundary() {
    let detector = WordBoundaryDetector::new();
    let context = BoundaryContext::new(12.0);

    // "Hi World" - space character at index 2
    let characters = vec![
        CharacterInfo {
            code: 'H' as u32,
            glyph_id: None,
            width: 8.0,
            x_position: 0.0,
            tj_offset: None,
            font_size: 12.0,
            is_ligature: false,
            original_ligature: None,
            protected_from_split: false,
        },
        CharacterInfo {
            code: 'i' as u32,
            glyph_id: None,
            width: 4.0,
            x_position: 8.0,
            tj_offset: None,
            font_size: 12.0,
            is_ligature: false,
            original_ligature: None,
            protected_from_split: false,
        },
        CharacterInfo {
            code: ' ' as u32, // Space character
            glyph_id: None,
            width: 4.0,
            x_position: 12.0,
            tj_offset: None,
            font_size: 12.0,
            is_ligature: false,
            original_ligature: None,
            protected_from_split: false,
        },
        CharacterInfo {
            code: 'W' as u32,
            glyph_id: None,
            width: 10.0,
            x_position: 16.0,
            tj_offset: None,
            font_size: 12.0,
            is_ligature: false,
            original_ligature: None,
            protected_from_split: false,
        },
    ];

    let boundaries = detector.detect_word_boundaries(&characters, &context);

    // Space at index 2 creates boundary at index 3
    assert!(
        boundaries.contains(&3),
        "Should have boundary after space character, got: {:?}",
        boundaries
    );
}

/// Test 6: Zero-width space creates boundary
/// U+200B (zero-width space) should create a word boundary.
#[test]
fn test_zero_width_space_creates_boundary() {
    let detector = WordBoundaryDetector::new();
    let context = BoundaryContext::new(12.0);

    let characters = vec![
        CharacterInfo {
            code: 'a' as u32,
            glyph_id: None,
            width: 6.0,
            x_position: 0.0,
            tj_offset: None,
            font_size: 12.0,
            is_ligature: false,
            original_ligature: None,
            protected_from_split: false,
        },
        CharacterInfo {
            code: 0x200B, // Zero-width space
            glyph_id: None,
            width: 0.0,
            x_position: 6.0,
            tj_offset: None,
            font_size: 12.0,
            is_ligature: false,
            original_ligature: None,
            protected_from_split: false,
        },
        CharacterInfo {
            code: 'b' as u32,
            glyph_id: None,
            width: 6.0,
            x_position: 6.0, // Same position due to zero width
            tj_offset: None,
            font_size: 12.0,
            is_ligature: false,
            original_ligature: None,
            protected_from_split: false,
        },
    ];

    let boundaries = detector.detect_word_boundaries(&characters, &context);

    // Zero-width space creates boundary
    assert!(
        boundaries.contains(&2),
        "Should have boundary after zero-width space, got: {:?}",
        boundaries
    );
}

/// Test 7: Horizontal scaling affects threshold
/// When horizontal scaling is applied, the gap threshold should adjust accordingly.
#[test]
fn test_horizontal_scaling_affects_threshold() {
    let detector = WordBoundaryDetector::new().with_geometric_gap_ratio(0.8);

    // At 100% scaling: threshold = 12 * 0.8 = 9.6pt
    // At 75% scaling: threshold = 12 * 0.75 * 0.8 = 7.2pt
    // Gap of 8pt should NOT trigger at 100% but SHOULD at 75%

    let characters = vec![
        CharacterInfo {
            code: 'A' as u32,
            glyph_id: None,
            width: 0.5, // Small width to create gap
            x_position: 0.0,
            tj_offset: None,
            font_size: 12.0,
            is_ligature: false,
            original_ligature: None,
            protected_from_split: false,
        },
        CharacterInfo {
            code: 'B' as u32,
            glyph_id: None,
            width: 0.5,
            x_position: 8.0, // Gap of 7.5 units
            tj_offset: None,
            font_size: 12.0,
            is_ligature: false,
            original_ligature: None,
            protected_from_split: false,
        },
    ];

    // At 100% scaling, gap (7.5) < threshold (9.6)
    let context_100 = BoundaryContext {
        font_size: 12.0,
        horizontal_scaling: 100.0,
        word_spacing: 0.0,
        char_spacing: 0.0,
    };
    let boundaries_100 = detector.detect_word_boundaries(&characters, &context_100);

    // At 75% scaling, gap (7.5) > threshold (7.2)
    let context_75 = BoundaryContext {
        font_size: 12.0,
        horizontal_scaling: 75.0,
        word_spacing: 0.0,
        char_spacing: 0.0,
    };
    let boundaries_75 = detector.detect_word_boundaries(&characters, &context_75);

    assert!(
        boundaries_100.is_empty(),
        "At 100% scaling, gap should NOT trigger boundary, got: {:?}",
        boundaries_100
    );
    assert!(
        boundaries_75.contains(&1),
        "At 75% scaling, gap SHOULD trigger boundary, got: {:?}",
        boundaries_75
    );
}

/// Test 8: Multiple signals agreeing
/// When TJ offset AND geometric gap both indicate boundary, detection should be robust.
#[test]
fn test_multiple_signals_agreeing() {
    let detector = WordBoundaryDetector::new().with_geometric_gap_ratio(0.5);
    let context = BoundaryContext::new(12.0);

    let characters = vec![
        CharacterInfo {
            code: 'A' as u32,
            glyph_id: None,
            width: 8.0,
            x_position: 0.0,
            tj_offset: Some(-200), // TJ offset signal
            font_size: 12.0,
            is_ligature: false,
            original_ligature: None,
            protected_from_split: false,
        },
        CharacterInfo {
            code: 'B' as u32,
            glyph_id: None,
            width: 8.0,
            x_position: 20.0, // Geometric gap signal (20 - 8 = 12 > 6)
            tj_offset: None,
            font_size: 12.0,
            is_ligature: false,
            original_ligature: None,
            protected_from_split: false,
        },
    ];

    let boundaries = detector.detect_word_boundaries(&characters, &context);

    // Both signals agree - boundary should definitely be detected
    assert!(
        boundaries.contains(&1),
        "Should detect boundary with both TJ and geometric signals, got: {:?}",
        boundaries
    );
}

/// Test 9: Empty input returns empty boundaries
#[test]
fn test_empty_input_returns_empty() {
    let detector = WordBoundaryDetector::new();
    let context = BoundaryContext::new(12.0);

    let characters: Vec<CharacterInfo> = vec![];
    let boundaries = detector.detect_word_boundaries(&characters, &context);

    assert!(boundaries.is_empty(), "Empty input should return empty boundaries");
}

/// Test 10: Single character returns empty boundaries
#[test]
fn test_single_character_returns_empty() {
    let detector = WordBoundaryDetector::new();
    let context = BoundaryContext::new(12.0);

    let characters = vec![CharacterInfo {
        code: 'A' as u32,
        glyph_id: None,
        width: 8.0,
        x_position: 0.0,
        tj_offset: None,
        font_size: 12.0,
        is_ligature: false,
        original_ligature: None,
        protected_from_split: false,
    }];

    let boundaries = detector.detect_word_boundaries(&characters, &context);

    assert!(boundaries.is_empty(), "Single character should return empty boundaries");
}
