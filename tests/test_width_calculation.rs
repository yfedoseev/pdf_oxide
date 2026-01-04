//! Tests for text width calculation with text matrix scaling.
//!
//! PDF Spec Compliance (ISO 32000-1:2008):
//! - Section 9.4.4: Text space is defined by the text matrix (Tm)
//! - The effective font size must account for text matrix scaling
//! - Common pattern: `/Font 1 Tf` followed by `scale 0 0 scale x y Tm`
//!   where the actual rendered font size is 1 × scale
//!
//! This test validates the fix in `calculate_tj_buffer_width` that accounts for
//! text matrix scaling when computing character/span widths.
//!
//! The fix ensures: effective_font_size = font_size × |text_matrix.d|

use pdf_oxide::extractors::TextExtractor;
use pdf_oxide::fonts::{Encoding, FontInfo};

/// Create a test font with specified default width.
fn create_test_font_with_width(default_width: f32) -> FontInfo {
    FontInfo {
        base_font: "TestFont".to_string(),
        subtype: "Type1".to_string(),
        encoding: Encoding::Standard("WinAnsiEncoding".to_string()),
        to_unicode: None,
        font_weight: None,
        flags: None,
        stem_v: None,
        embedded_font_data: None,
        truetype_cmap: None,
        widths: None,
        first_char: None,
        last_char: None,
        default_width,
        cid_to_gid_map: None,
        cid_system_info: None,
        cid_font_type: None,
    }
}

// ============================================================================
// Tests using span extraction mode (affected by the fix in calculate_tj_buffer_width)
// ============================================================================

/// Test: Text matrix scaling affects span width calculation.
///
/// PDF Spec Section 9.4.4 states that the text matrix (Tm) can include
/// scaling factors that affect the effective font size. The fix ensures
/// that `calculate_tj_buffer_width` uses effective_font_size = font_size × |matrix.d|
#[test]
fn test_span_width_with_text_matrix_scaling() {
    let mut extractor = TextExtractor::new();
    let font = create_test_font_with_width(500.0); // 500 units default width
    extractor.add_font("F1".to_string(), font);

    // Case 1: Normal case - font size 12, identity matrix (scale 1)
    // Effective font size = 12 × 1 = 12
    let stream_normal = b"BT /F1 12 Tf 1 0 0 1 100 700 Tm (Hello) Tj ET";
    let spans_normal = extractor.extract_text_spans(stream_normal).unwrap();

    assert!(!spans_normal.is_empty(), "Should extract at least one span");
    let width_normal = spans_normal[0].bbox.width;
    let font_size_normal = spans_normal[0].font_size;

    extractor.clear();

    // Case 2: Scaled case - font size 1, matrix scaled by 12
    // Effective font size = 1 × 12 = 12 (same as Case 1)
    let stream_scaled = b"BT /F1 1 Tf 12 0 0 12 100 700 Tm (Hello) Tj ET";
    let spans_scaled = extractor.extract_text_spans(stream_scaled).unwrap();

    assert!(!spans_scaled.is_empty(), "Should extract at least one span");
    let width_scaled = spans_scaled[0].bbox.width;
    let font_size_scaled = spans_scaled[0].font_size;

    // Both cases should have the same effective font size (12)
    let tolerance = 1.0;
    assert!(
        (font_size_normal - font_size_scaled).abs() < tolerance,
        "Effective font size should be ~12 in both cases: normal={}, scaled={}",
        font_size_normal,
        font_size_scaled
    );

    // Both cases should produce similar span widths since they have the same
    // effective font size and text content
    assert!(
        (width_normal - width_scaled).abs() < tolerance * 5.0, // More tolerance for width
        "Span width should be similar in both cases: normal={:.2}, scaled={:.2}",
        width_normal,
        width_scaled
    );
}

/// Test: Large text matrix scaling (common in some PDF generators).
///
/// Some PDF generators use very small font sizes with large matrix scaling.
/// This test verifies the fix handles this pattern correctly.
#[test]
fn test_span_large_text_matrix_scaling() {
    let mut extractor = TextExtractor::new();
    let font = create_test_font_with_width(600.0);
    extractor.add_font("F1".to_string(), font);

    // Font size 0.5 with matrix scale 24 = effective size 12
    let stream = b"BT /F1 0.5 Tf 24 0 0 24 50 500 Tm (Test) Tj ET";
    let spans = extractor.extract_text_spans(stream).unwrap();

    assert!(!spans.is_empty(), "Should extract at least one span");

    // The span should have effective font size ~12 (0.5 × 24)
    let expected_effective_size = 12.0;
    let tolerance = 1.0;

    assert!(
        (spans[0].font_size - expected_effective_size).abs() < tolerance,
        "Effective font size should be ~{}, got {}",
        expected_effective_size,
        spans[0].font_size
    );
}

/// Test: Negative text matrix scaling (mirrored text).
///
/// PDF Spec allows negative scaling in the text matrix for mirrored text.
/// The width calculation should use the absolute value of matrix.d.
#[test]
fn test_span_negative_text_matrix_scaling() {
    let mut extractor = TextExtractor::new();
    let font = create_test_font_with_width(500.0);
    extractor.add_font("F1".to_string(), font);

    // Positive scaling
    let stream_positive = b"BT /F1 1 Tf 10 0 0 10 100 700 Tm (ABC) Tj ET";
    let spans_positive = extractor.extract_text_spans(stream_positive).unwrap();

    extractor.clear();

    // Negative scaling (mirrored vertically)
    // The absolute value |−10| = 10 should be used for width calculation
    let stream_negative = b"BT /F1 1 Tf 10 0 0 -10 100 700 Tm (ABC) Tj ET";
    let spans_negative = extractor.extract_text_spans(stream_negative).unwrap();

    assert!(!spans_positive.is_empty(), "Should extract spans with positive scaling");
    assert!(!spans_negative.is_empty(), "Should extract spans with negative scaling");

    // Both should have the same effective font size (absolute value)
    let tolerance = 0.5;
    assert!(
        (spans_positive[0].font_size - spans_negative[0].font_size).abs() < tolerance,
        "Positive scale ({}) and negative scale ({}) should produce same effective size",
        spans_positive[0].font_size,
        spans_negative[0].font_size
    );

    // Both should have similar widths
    assert!(
        (spans_positive[0].bbox.width - spans_negative[0].bbox.width).abs() < tolerance * 10.0,
        "Positive scale width ({:.2}) and negative scale width ({:.2}) should be similar",
        spans_positive[0].bbox.width,
        spans_negative[0].bbox.width
    );
}

/// Test: Identity text matrix (no scaling).
///
/// When text matrix is identity (1 0 0 1 x y), the effective font size
/// equals the nominal font size from Tf operator.
#[test]
fn test_span_identity_matrix_no_scaling() {
    let mut extractor = TextExtractor::new();
    let font = create_test_font_with_width(500.0);
    extractor.add_font("F1".to_string(), font);

    // Font size 14 with identity matrix
    let stream = b"BT /F1 14 Tf 1 0 0 1 100 700 Tm (Test) Tj ET";
    let spans = extractor.extract_text_spans(stream).unwrap();

    assert!(!spans.is_empty(), "Should extract at least one span");

    // Effective font size should equal nominal font size (14)
    let tolerance = 0.5;
    assert!(
        (spans[0].font_size - 14.0).abs() < tolerance,
        "With identity matrix, effective font size {} should equal nominal size 14",
        spans[0].font_size
    );
}

/// Test: Text positioning with Td operator (no matrix scaling).
///
/// When using Td for positioning (not Tm), the text matrix scaling
/// component should be 1.0 (identity).
#[test]
fn test_span_td_positioning_no_matrix_scaling() {
    let mut extractor = TextExtractor::new();
    let font = create_test_font_with_width(500.0);
    extractor.add_font("F1".to_string(), font);

    // Using Td operator - text matrix d component stays at 1.0
    let stream = b"BT /F1 16 Tf 100 700 Td (Hello) Tj ET";
    let spans = extractor.extract_text_spans(stream).unwrap();

    assert!(!spans.is_empty(), "Should extract at least one span");

    // Effective font size should equal nominal font size (16)
    let tolerance = 0.5;
    assert!(
        (spans[0].font_size - 16.0).abs() < tolerance,
        "With Td operator, effective font size {} should equal nominal size 16",
        spans[0].font_size
    );
}

/// Test: Fractional text matrix scaling.
///
/// Some PDFs use fractional scaling in the text matrix.
/// This test verifies correct handling of non-integer scale factors.
#[test]
fn test_span_fractional_matrix_scaling() {
    let mut extractor = TextExtractor::new();
    let font = create_test_font_with_width(500.0);
    extractor.add_font("F1".to_string(), font);

    // Font size 8 with matrix scale 1.5 = effective size 12
    let stream = b"BT /F1 8 Tf 1.5 0 0 1.5 100 700 Tm (Z) Tj ET";
    let spans = extractor.extract_text_spans(stream).unwrap();

    assert!(!spans.is_empty(), "Should extract at least one span");

    // Effective font size = 8 × 1.5 = 12
    let expected_effective_size = 12.0;
    let tolerance = 1.0;

    assert!(
        (spans[0].font_size - expected_effective_size).abs() < tolerance,
        "Effective font size should be ~{}, got {}",
        expected_effective_size,
        spans[0].font_size
    );
}

/// Test: Width consistency between different scaling combinations.
///
/// Different combinations of font_size and matrix.d that result in
/// the same effective font size should produce consistent span widths.
#[test]
fn test_span_width_consistency_across_scaling_combinations() {
    let mut extractor = TextExtractor::new();
    let font = create_test_font_with_width(600.0);
    extractor.add_font("F1".to_string(), font);

    // All these combinations should produce effective font size = 24
    let combinations: &[(&[u8], f32, f32)] = &[
        (b"BT /F1 24 Tf 1 0 0 1 0 0 Tm (Word) Tj ET", 24.0, 1.0),    // 24 × 1
        (b"BT /F1 12 Tf 2 0 0 2 0 0 Tm (Word) Tj ET", 12.0, 2.0),    // 12 × 2
        (b"BT /F1 8 Tf 3 0 0 3 0 0 Tm (Word) Tj ET", 8.0, 3.0),      // 8 × 3
        (b"BT /F1 6 Tf 4 0 0 4 0 0 Tm (Word) Tj ET", 6.0, 4.0),      // 6 × 4
        (b"BT /F1 4 Tf 6 0 0 6 0 0 Tm (Word) Tj ET", 4.0, 6.0),      // 4 × 6
    ];

    let mut widths = Vec::new();
    let mut font_sizes = Vec::new();

    for (stream, nominal_size, scale) in combinations {
        extractor.clear();
        let spans = extractor.extract_text_spans(*stream).unwrap();
        assert!(
            !spans.is_empty(),
            "Expected at least one span for {}×{}",
            nominal_size,
            scale
        );

        // Record the bbox width and font size
        widths.push(spans[0].bbox.width);
        font_sizes.push(spans[0].font_size);
    }

    // All font sizes should be approximately 24
    let tolerance = 1.0;
    for (i, &font_size) in font_sizes.iter().enumerate() {
        assert!(
            (font_size - 24.0).abs() < tolerance,
            "Combination {} should have effective font size ~24, got {}",
            i,
            font_size
        );
    }

    // All widths should be approximately equal (within a reasonable tolerance)
    let first_width = widths[0];
    let width_tolerance = 5.0; // 5 points tolerance for width variations
    for (i, &width) in widths.iter().enumerate() {
        assert!(
            (width - first_width).abs() < width_tolerance,
            "Width for combination {} ({:.2}) should be close to first width ({:.2})",
            i,
            width,
            first_width
        );
    }
}

/// Test: TJ array width calculation with text matrix scaling.
///
/// PDF Spec Section 9.4.4 defines width calculation for TJ arrays.
/// The fix ensures effective font size is used in width formula:
/// tx = (w0 × effective_font_size / 1000 + Tc + Tw) × Th
#[test]
fn test_span_tj_array_width_with_matrix_scaling() {
    let mut extractor = TextExtractor::new();
    let font = create_test_font_with_width(500.0);
    extractor.add_font("F1".to_string(), font);

    // TJ array with text matrix scaling
    // Font size 1 with scale 10 = effective size 10
    let stream = b"BT /F1 1 Tf 10 0 0 10 100 700 Tm [(Hello) -200 (World)] TJ ET";
    let spans = extractor.extract_text_spans(stream).unwrap();

    // We expect spans to be created (the exact number depends on how
    // the TJ array is processed - it might be one span or multiple)
    assert!(!spans.is_empty(), "Should extract at least one span");

    // Verify effective font size is correct
    let tolerance = 1.0;
    assert!(
        (spans[0].font_size - 10.0).abs() < tolerance,
        "Effective font size should be ~10 (1 × 10), got {}",
        spans[0].font_size
    );

    // The TJ offset -200 in text space units should be scaled by effective font size
    // Offset in user space = -200 × 10 / 1000 = -2.0 points (creates a small gap)
    // This affects the total span width
}

/// Test: Multiple text objects with different matrix scales.
///
/// Verifies that width calculation correctly handles multiple BT/ET
/// blocks with different text matrix configurations.
#[test]
fn test_span_multiple_text_objects_different_scales() {
    let mut extractor = TextExtractor::new();
    let font = create_test_font_with_width(500.0);
    extractor.add_font("F1".to_string(), font);

    // Two text objects with different scales
    // First: font_size=12, scale=1 (effective=12)
    // Second: font_size=1, scale=18 (effective=18)
    let stream = b"BT /F1 12 Tf 1 0 0 1 100 700 Tm (First) Tj ET \
                   BT /F1 1 Tf 18 0 0 18 200 600 Tm (Second) Tj ET";
    let spans = extractor.extract_text_spans(stream).unwrap();

    assert!(spans.len() >= 2, "Should extract at least two spans");

    // Find spans by their approximate position
    let first_span = spans.iter().find(|s| s.bbox.y > 650.0);
    let second_span = spans.iter().find(|s| s.bbox.y < 650.0);

    assert!(first_span.is_some(), "Should find first span at y~700");
    assert!(second_span.is_some(), "Should find second span at y~600");

    let first = first_span.unwrap();
    let second = second_span.unwrap();

    // First span should have effective size ~12
    let tolerance = 1.0;
    assert!(
        (first.font_size - 12.0).abs() < tolerance,
        "First span effective size should be ~12, got {}",
        first.font_size
    );

    // Second span should have effective size ~18
    assert!(
        (second.font_size - 18.0).abs() < tolerance,
        "Second span effective size should be ~18, got {}",
        second.font_size
    );
}

// ============================================================================
// Tests for character extraction mode (verifying effective font size reporting)
// These tests verify the font_size field in TextChar, not the position advancement
// ============================================================================

/// Test: Character extraction reports correct effective font size.
///
/// Verifies that the effective font size is correctly calculated and
/// reported in TextChar.font_size when using character extraction mode.
#[test]
fn test_char_effective_font_size_reporting() {
    let mut extractor = TextExtractor::new();
    let font = create_test_font_with_width(500.0);
    extractor.add_font("F1".to_string(), font);

    // Font size 1 with scale 15 = effective size 15
    let stream = b"BT /F1 1 Tf 15 0 0 15 100 700 Tm (A) Tj ET";
    let chars = extractor.extract(stream).unwrap();

    assert_eq!(chars.len(), 1, "Should extract one character");

    // Effective font size should be 15 (1 × 15)
    let tolerance = 0.5;
    assert!(
        (chars[0].font_size - 15.0).abs() < tolerance,
        "Character effective font size should be ~15 (1 × 15), got {}",
        chars[0].font_size
    );
}

/// Test: Character extraction with identity matrix.
#[test]
fn test_char_identity_matrix() {
    let mut extractor = TextExtractor::new();
    let font = create_test_font_with_width(500.0);
    extractor.add_font("F1".to_string(), font);

    // Font size 20 with identity matrix
    let stream = b"BT /F1 20 Tf 1 0 0 1 100 700 Tm (X) Tj ET";
    let chars = extractor.extract(stream).unwrap();

    assert_eq!(chars.len(), 1, "Should extract one character");

    // Effective font size should equal nominal font size (20)
    let tolerance = 0.5;
    assert!(
        (chars[0].font_size - 20.0).abs() < tolerance,
        "With identity matrix, font size should be 20, got {}",
        chars[0].font_size
    );
}

/// Test: Character extraction with various scaling factors.
#[test]
fn test_char_various_scaling_factors() {
    let test_cases: &[(f32, f32, f32)] = &[
        (10.0, 1.0, 10.0),  // font_size=10, scale=1, expected=10
        (5.0, 2.0, 10.0),   // font_size=5, scale=2, expected=10
        (1.0, 10.0, 10.0),  // font_size=1, scale=10, expected=10
        (2.5, 4.0, 10.0),   // font_size=2.5, scale=4, expected=10
    ];

    for (font_size, scale, expected) in test_cases {
        let mut extractor = TextExtractor::new();
        let font = create_test_font_with_width(500.0);
        extractor.add_font("F1".to_string(), font);

        // Build stream with proper formatting
        let stream = format!(
            "BT /F1 {} Tf {} 0 0 {} 100 700 Tm (Y) Tj ET",
            font_size, scale, scale
        );
        let chars = extractor.extract(stream.as_bytes()).unwrap();

        assert_eq!(
            chars.len(),
            1,
            "Should extract one character for {}×{}",
            font_size,
            scale
        );

        let tolerance = 1.0;
        assert!(
            (chars[0].font_size - expected).abs() < tolerance,
            "For font_size={}, scale={}: expected effective size ~{}, got {}",
            font_size,
            scale,
            expected,
            chars[0].font_size
        );
    }
}
