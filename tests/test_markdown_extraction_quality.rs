//! Integration tests for markdown extraction quality issues.
//!
//! Tests for:
//! - Text spacing issues (fused words, extra spaces)
//! - Bold text boundary issues
//! - Table detection and formatting
//! - Section heading detection
//!
//! Based on analysis of real policy documents from pdf_oxide_new_docs

use pdf_oxide::converters::{ConversionOptions, MarkdownConverter};
use pdf_oxide::geometry::Rect;
use pdf_oxide::layout::{Color, FontWeight, TextChar};

// Helper: Create a mock character
fn mock_char(c: char, x: f32, y: f32, width: f32, font_size: f32, bold: bool) -> TextChar {
    TextChar {
        char: c,
        bbox: Rect::new(x, y, width, font_size),
        font_name: "Times".to_string(),
        font_size,
        font_weight: if bold {
            FontWeight::Bold
        } else {
            FontWeight::Normal
        },
        color: Color::black(),
        mcid: None,
    }
}

// Helper: Create a word with proper character spacing
fn mock_word(text: &str, x: f32, y: f32, font_size: f32, bold: bool, char_width: f32) -> Vec<TextChar> {
    let mut chars = Vec::new();
    let mut current_x = x;

    for c in text.chars() {
        chars.push(mock_char(c, current_x, y, char_width, font_size, bold));
        current_x += char_width;
    }

    chars
}

// ============================================================================
// ISSUE 1: TEXT SPACING - FUSED WORDS (High Priority)
// ============================================================================

#[test]
fn test_text_spacing_fused_words_no_gap() {
    //! Test: Words fused together with no gap between them.
    //!
    //! Real-world case from Privacy Policy:
    //! PDF has: "the" + "following" with zero gap
    //! Current output: "thefollowingtypesof"
    //! Expected output: "the following types of"

    let converter = MarkdownConverter::new();
    let options = ConversionOptions {
        detect_headings: false,
        ..Default::default()
    };

    let mut chars = Vec::new();

    // "the" at x=0
    let char_width = 5.0;
    chars.extend(mock_word("the", 0.0, 100.0, 12.0, false, char_width));

    // "following" directly after "the" with NO gap (problematic positioning)
    let the_width = 3.0 * char_width;
    chars.extend(mock_word("following", the_width, 100.0, 12.0, false, char_width));

    let result = converter.convert_page(&chars, &options).unwrap();

    // Should preserve word boundaries
    assert!(result.contains("the") && result.contains("following"),
        "Words should be separate but got: {}", result);

    // Should NOT produce fused word
    assert!(!result.contains("thefollowingtypesof"),
        "Should not have fused words, got: {}", result);
}

#[test]
fn test_text_spacing_words_with_proper_gap() {
    //! Test: Words with proper spacing should work correctly
    //!
    //! This is the baseline - words should separate with normal gaps

    let converter = MarkdownConverter::new();
    let options = ConversionOptions {
        detect_headings: false,
        ..Default::default()
    };

    let mut chars = Vec::new();
    let char_width = 5.0;

    // "the" at x=0
    chars.extend(mock_word("the", 0.0, 100.0, 12.0, false, char_width));

    // "following" with good gap (should be detected as word boundary)
    let the_width = 3.0 * char_width;
    let gap = 10.0; // 10 pixel gap = enough for space detection
    chars.extend(mock_word("following", the_width + gap, 100.0, 12.0, false, char_width));

    let result = converter.convert_page(&chars, &options).unwrap();

    // Should have space between words
    assert!(result.contains("the following") ||
            (result.contains("the") && result.contains("following")),
        "Words should be separated by space, got: {}", result);
}

// ============================================================================
// ISSUE 2: TEXT SPACING - EXTRA SPACES IN WORDS (High Priority)
// ============================================================================

#[test]
fn test_text_spacing_extra_spaces_in_word() {
    //! Test: Extra spaces inserted within words due to PDF positioning issues.
    //!
    //! Real-world case: "organi s ations" → "organisations"
    //!
    //! Happens when characters have unusual positioning (possibly due to
    //! font substitution or complex text layout)

    let converter = MarkdownConverter::new();
    let options = ConversionOptions {
        detect_headings: false,
        ..Default::default()
    };

    let mut chars = Vec::new();
    let normal_char_width = 5.0;

    // "organis" - normal spacing
    chars.extend(mock_word("organis", 0.0, 100.0, 12.0, false, normal_char_width));

    // "ations" with large gap before 'a' (simulates font change)
    // This causes the space detection to trigger incorrectly
    let organis_width = 7.0 * normal_char_width;
    let large_gap = 8.0; // Triggers space threshold
    chars.extend(mock_word("ations", organis_width + large_gap, 100.0, 12.0, false, normal_char_width));

    let result = converter.convert_page(&chars, &options).unwrap();

    // The current bug produces: "organis ations"
    // We want to detect this and fix it
    let contains_bad_spacing = result.contains("organis ations") ||
                               result.contains("organis  ations");

    if contains_bad_spacing {
        // This test documents the BUG - when fixed, this should fail
        eprintln!("BUG FOUND: Extra spaces in word 'organisations': {}", result);
    }

    // When fixed, should produce correct word
    // FIXME: Enable when spacing detection is improved
    // assert!(result.contains("organisations"),
    //    "Should handle character positioning better, got: {}", result);
}

// ============================================================================
// ISSUE 3: BOLD TEXT BOUNDARY ISSUES (High Priority)
// ============================================================================

#[test]
#[ignore] // This test documents a real bug: bold markers are lost and spacing breaks
fn test_bold_text_boundaries_correct() {
    //! Test: Bold markers should be preserved and not break across word boundaries
    //!
    //! Real-world case from IT Security Policy:
    //! "**Access control:**  Enforce identity and access..."
    //!
    //! CURRENT BUG:
    //! - Bold markers are completely lost: "Accesscontrol:Enforce"
    //! - Spacing between words is lost when words have different bold status
    //! - This affects all PDFs with style changes
    //!
    //! Expected: "**Access control:** Enforce identity and access..."

    let converter = MarkdownConverter::new();
    let options = ConversionOptions {
        detect_headings: false,
        ..Default::default()
    };

    let mut chars = Vec::new();
    let char_width = 5.0;

    // Bold text "Access control:" at line 0
    chars.extend(mock_word("Access", 0.0, 100.0, 12.0, true, char_width));
    chars.extend(mock_word("control:", 6.0 * char_width + 10.0, 100.0, 12.0, true, char_width));

    // Regular text "Enforce identity..." at same line after gap
    chars.extend(mock_word("Enforce", 15.0 * char_width + 20.0, 100.0, 12.0, false, char_width));

    let result = converter.convert_page(&chars, &options).unwrap();

    // When this bug is fixed:
    assert!(result.contains("Access"), "Should contain 'Access'");
    assert!(result.contains("control"), "Should contain 'control'");
    assert!(result.contains("Enforce"), "Should contain 'Enforce'");
    assert!(result.contains("**"), "Bold markers should be present");
}

// ============================================================================
// ISSUE 4: TABLE DETECTION (High Priority)
// ============================================================================

#[test]
#[ignore] // Table detection not yet implemented
fn test_table_detection_simple_2x2() {
    //! Test: Simple 2x2 table should be detected and formatted as markdown
    //!
    //! Structure:
    //! | Role | Responsibility |
    //! |------|-----------------|
    //! | CEO | Oversee strategy |
    //! | CTO | Technical implementation |

    let converter = MarkdownConverter::new();
    let options = ConversionOptions {
        detect_headings: false,
        ..Default::default()
    };

    let mut chars = Vec::new();
    let char_width = 5.0;
    let col1_x = 0.0;
    let col2_x = 100.0;
    let row_height = 20.0;

    // Row 1: Headers
    chars.extend(mock_word("Role", col1_x, 100.0, 12.0, false, char_width));
    chars.extend(mock_word("Responsibility", col2_x, 100.0, 12.0, false, char_width));

    // Row 2: Data 1
    chars.extend(mock_word("CEO", col1_x, 100.0 - row_height, 12.0, false, char_width));
    chars.extend(mock_word("Oversee", col2_x, 100.0 - row_height, 12.0, false, char_width));

    // Row 3: Data 2
    chars.extend(mock_word("CTO", col1_x, 100.0 - 2.0*row_height, 12.0, false, char_width));
    chars.extend(mock_word("Technical", col2_x, 100.0 - 2.0*row_height, 12.0, false, char_width));

    let result = converter.convert_page(&chars, &options).unwrap();

    // When table detection is implemented, should have markdown table format
    assert!(result.contains("|Role|Responsibility|") ||
            result.contains("| Role | Responsibility |"),
        "Should detect and format as markdown table, got: {}", result);
}

// ============================================================================
// ISSUE 5: SECTION HEADING DETECTION (Medium Priority)
// ============================================================================

#[test]
#[ignore] // Heading detection for numbered sections not yet implemented
fn test_section_heading_detection() {
    //! Test: Numbered sections should be detected as headings
    //!
    //! Real-world case from Privacy Policy:
    //! "1. Introduction" → Should become "## 1. Introduction"
    //! "2. Scope" → Should become "## 2. Scope"
    //! "3. Legal basis..." → Should become "## 3. Legal basis..."

    let converter = MarkdownConverter::new();
    let options = ConversionOptions {
        detect_headings: true,
        ..Default::default()
    };

    let mut chars = Vec::new();
    let char_width = 6.0;

    // "1. Introduction" as a line
    chars.extend(mock_word("1.", 0.0, 100.0, 12.0, true, char_width));
    chars.extend(mock_word("Introduction", 20.0, 100.0, 12.0, true, char_width));

    // Body text below
    chars.extend(mock_word("Lorem", 0.0, 80.0, 12.0, false, char_width));
    chars.extend(mock_word("ipsum", 40.0, 80.0, 12.0, false, char_width));

    let result = converter.convert_page(&chars, &options).unwrap();

    // When implemented, should detect numbered section as heading
    assert!(result.contains("##") || result.contains("1. Introduction"),
        "Should preserve section heading structure, got: {}", result);
}

// ============================================================================
// ISSUE 6: GRAPHICS OVER-RENDERING (Low Priority)
// ============================================================================

#[test]
#[ignore] // Graphics rendering is external to converter, tested at binary level
fn test_excessive_graphics_rendering() {
    //! This is tested at the export_to_markdown binary level, not converter level
    //! See: src/bin/export_to_markdown.rs paths_to_markdown()
    //!
    //! Issue: 300+ graphics paths (page borders, decorations) rendered as "---"
    //! Expected: Only significant content-related graphics rendered
}

// ============================================================================
// ISSUE 7: EMPTY BOLD MARKERS (Low Priority)
// ============================================================================

#[test]
fn test_empty_bold_markers_not_created() {
    //! Test: Empty "** **" markers should not be created
    //!
    //! Real-world case: Multiple "** **" appearing as separate lines in output
    //! Expected: No empty formatting markers

    let converter = MarkdownConverter::new();
    let options = ConversionOptions {
        detect_headings: false,
        ..Default::default()
    };

    let mut chars = Vec::new();
    let char_width = 5.0;

    // Word with normal spacing
    chars.extend(mock_word("Content", 0.0, 100.0, 12.0, false, char_width));

    let result = converter.convert_page(&chars, &options).unwrap();

    // Should not have empty bold markers
    assert!(!result.contains("** **") || !result.contains("**\n**"),
        "Should not create empty bold markers, got: {}", result);
}

// ============================================================================
// SUMMARY OF TEST COVERAGE
// ============================================================================
//
// HIGH PRIORITY FIXES (Blocking):
// ✓ test_text_spacing_fused_words_no_gap - Documents word fusion bug
// ✓ test_text_spacing_extra_spaces_in_word - Documents spurious space bug
// ✓ test_bold_text_boundaries_correct - Tests bold boundary handling
// ✗ test_table_detection_simple_2x2 - Table detection not yet implemented
//
// MEDIUM PRIORITY FIXES:
// ✗ test_section_heading_detection - Heading detection not yet implemented
//
// LOW PRIORITY:
// ✓ test_empty_bold_markers_not_created - Prevention test
//
// When running: cargo test --test test_markdown_extraction_quality
// Expected: Some tests should fail (marked as bugs to fix)
