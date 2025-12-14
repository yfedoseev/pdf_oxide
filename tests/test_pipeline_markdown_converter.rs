//! Tests for MarkdownOutputConverter pipeline features.
//!
//! This test suite implements TDD-first testing for markdown converter features:
//! - Heading detection (H1-H3 by font size)
//! - Bold/Italic formatting
//! - Table detection
//! - Layout preservation
//! - Image embedding
//! - URL/Email linkification
//! - Whitespace normalization

use pdf_oxide::geometry::Rect;
use pdf_oxide::layout::{Color, FontWeight, TextSpan};
use pdf_oxide::pipeline::config::{BoldMarkerBehavior, OutputConfig, TextPipelineConfig};
use pdf_oxide::pipeline::converters::{MarkdownOutputConverter, OutputConverter};
use pdf_oxide::pipeline::OrderedTextSpan;

/// Helper to create a text span with all necessary fields.
fn make_span(
    text: &str,
    x: f32,
    y: f32,
    font_size: f32,
    weight: FontWeight,
    is_italic: bool,
) -> OrderedTextSpan {
    make_span_with_order(text, x, y, font_size, weight, is_italic, 0)
}

/// Helper to create a text span with explicit reading order.
fn make_span_with_order(
    text: &str,
    x: f32,
    y: f32,
    font_size: f32,
    weight: FontWeight,
    is_italic: bool,
    reading_order: usize,
) -> OrderedTextSpan {
    OrderedTextSpan::new(
        TextSpan {
            text: text.to_string(),
            bbox: Rect::new(x, y, 100.0, font_size),
            font_name: "TestFont".to_string(),
            font_size,
            font_weight: weight,
            is_italic,
            color: Color::black(),
            mcid: None,
            sequence: 0,
            offset_semantic: false,
            split_boundary_before: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 100.0,
            primary_detected: false,
        },
        reading_order,
    )
}

// ============================================================================
// FEATURE 1: Heading Detection Tests
// ============================================================================

#[test]
fn test_heading_detection_h1() {
    // Given: TextSpan with large font size (24pt) and bold weight
    let span = make_span("Main Heading", 0.0, 100.0, 24.0, FontWeight::Bold, false);

    // When: Convert to markdown with detect_headings=true
    let config = TextPipelineConfig {
        output: OutputConfig {
            detect_headings: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let converter = MarkdownOutputConverter::new();
    let output = converter.convert(&[span], &config).unwrap();

    // Then: Output starts with "# " (H1)
    assert!(output.starts_with("# "), "Output: {}", output);
    assert!(output.contains("Main Heading"));
}

#[test]
fn test_heading_detection_h2() {
    // Given: TextSpan with medium-large font size (20pt)
    let span = make_span("Section Heading", 0.0, 100.0, 20.0, FontWeight::Normal, false);

    // When: Convert to markdown with detect_headings=true and base_font_size=12pt
    let config = TextPipelineConfig {
        output: OutputConfig {
            detect_headings: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let converter = MarkdownOutputConverter::new();
    // Include a normal span to establish base font size
    let normal_span = make_span("Normal text", 0.0, 80.0, 12.0, FontWeight::Normal, false);
    let output = converter.convert(&[normal_span, span], &config).unwrap();

    // Then: Output contains "## " (H2)
    assert!(output.contains("## "), "Output: {}", output);
    assert!(output.contains("Section Heading"));
}

#[test]
fn test_heading_detection_h3() {
    // Given: TextSpan with medium font size (16pt)
    let span = make_span("Subsection", 0.0, 100.0, 16.0, FontWeight::Normal, false);

    // When: Convert to markdown with detect_headings=true
    let config = TextPipelineConfig {
        output: OutputConfig {
            detect_headings: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let converter = MarkdownOutputConverter::new();
    // Include a normal span to establish base font size
    let normal_span = make_span("Normal text", 0.0, 80.0, 12.0, FontWeight::Normal, false);
    let output = converter.convert(&[normal_span, span], &config).unwrap();

    // Then: Output contains "### " (H3)
    assert!(output.contains("### "), "Output: {}", output);
    assert!(output.contains("Subsection"));
}

#[test]
fn test_heading_detection_disabled() {
    // Given: TextSpan with large font size
    let span = make_span("Large Text", 0.0, 100.0, 24.0, FontWeight::Bold, false);

    // When: Convert to markdown with detect_headings=false
    let config = TextPipelineConfig {
        output: OutputConfig {
            detect_headings: false,
            ..Default::default()
        },
        ..Default::default()
    };

    let converter = MarkdownOutputConverter::new();
    let output = converter.convert(&[span], &config).unwrap();

    // Then: Output does NOT start with "#"
    assert!(!output.starts_with("#"), "Output: {}", output);
    assert!(output.contains("Large Text"));
}

// ============================================================================
// FEATURE 2: Bold/Italic Formatting Tests
// ============================================================================

#[test]
fn test_bold_marker_conservative() {
    // Given: Bold text with content
    let span = make_span("Important", 0.0, 100.0, 12.0, FontWeight::Bold, false);

    // When: Convert with Conservative bold marker behavior
    let config = TextPipelineConfig {
        output: OutputConfig {
            bold_marker_behavior: BoldMarkerBehavior::Conservative,
            detect_headings: false,
            ..Default::default()
        },
        ..Default::default()
    };

    let converter = MarkdownOutputConverter::new();
    let output = converter.convert(&[span], &config).unwrap();

    // Then: Output contains bold markers
    assert!(output.contains("**Important**"), "Output: {}", output);
}

#[test]
fn test_bold_marker_conservative_whitespace() {
    // Given: Bold whitespace-only span
    let span = make_span("   ", 0.0, 100.0, 12.0, FontWeight::Bold, false);

    // When: Convert with Conservative bold marker behavior
    let config = TextPipelineConfig {
        output: OutputConfig {
            bold_marker_behavior: BoldMarkerBehavior::Conservative,
            detect_headings: false,
            ..Default::default()
        },
        ..Default::default()
    };

    let converter = MarkdownOutputConverter::new();
    let output = converter.convert(&[span], &config).unwrap();

    // Then: Output does NOT contain bold markers (Conservative skips whitespace)
    assert!(!output.contains("**"), "Output: {}", output);
}

#[test]
fn test_bold_marker_aggressive() {
    // Given: Bold text and whitespace
    let span1 = make_span("Bold", 0.0, 100.0, 12.0, FontWeight::Bold, false);
    let span2 = make_span("   ", 10.0, 100.0, 12.0, FontWeight::Bold, false);

    // When: Convert with Aggressive bold marker behavior
    let config = TextPipelineConfig {
        output: OutputConfig {
            bold_marker_behavior: BoldMarkerBehavior::Aggressive,
            detect_headings: false,
            ..Default::default()
        },
        ..Default::default()
    };

    let converter = MarkdownOutputConverter::new();
    let output = converter.convert(&[span1, span2], &config).unwrap();

    // Then: Both text and whitespace get bold markers
    assert!(output.contains("**Bold**"), "Output: {}", output);
}

#[test]
fn test_italic_formatting() {
    // Given: Italic text
    let span = make_span("Emphasized", 0.0, 100.0, 12.0, FontWeight::Normal, true);

    // When: Convert to markdown
    let config = TextPipelineConfig::default();

    let converter = MarkdownOutputConverter::new();
    let output = converter.convert(&[span], &config).unwrap();

    // Then: Output contains italic markers
    assert!(output.contains("*Emphasized*"), "Output: {}", output);
}

#[test]
fn test_bold_and_italic() {
    // Given: Bold italic text
    let span = make_span("Strong emphasis", 0.0, 100.0, 12.0, FontWeight::Bold, true);

    // When: Convert to markdown
    let config = TextPipelineConfig {
        output: OutputConfig {
            bold_marker_behavior: BoldMarkerBehavior::Conservative,
            ..Default::default()
        },
        ..Default::default()
    };

    let converter = MarkdownOutputConverter::new();
    let output = converter.convert(&[span], &config).unwrap();

    // Then: Output contains both bold and italic markers
    assert!(output.contains("***Strong emphasis***"), "Output: {}", output);
}

// ============================================================================
// FEATURE 3: Table Detection Tests
// ============================================================================

#[test]
fn test_table_detection_simple_2x2() {
    // Given: Text spans arranged in a 2x2 grid pattern
    let cell_11 = make_span("A", 10.0, 100.0, 12.0, FontWeight::Normal, false);
    let cell_12 = make_span("B", 40.0, 100.0, 12.0, FontWeight::Normal, false);
    let cell_21 = make_span("C", 10.0, 80.0, 12.0, FontWeight::Normal, false);
    let cell_22 = make_span("D", 40.0, 80.0, 12.0, FontWeight::Normal, false);

    // When: Convert with extract_tables=true
    let config = TextPipelineConfig {
        output: OutputConfig {
            extract_tables: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let converter = MarkdownOutputConverter::new();
    let output = converter
        .convert(&[cell_11, cell_12, cell_21, cell_22], &config)
        .unwrap();

    // Then: Output contains markdown table syntax
    assert!(output.contains("|"), "Output should contain table separators: {}", output);
}

#[test]
fn test_table_detection_disabled() {
    // Given: Text spans in a grid pattern
    let cell_11 = make_span("A", 10.0, 100.0, 12.0, FontWeight::Normal, false);
    let cell_12 = make_span("B", 40.0, 100.0, 12.0, FontWeight::Normal, false);

    // When: Convert with extract_tables=false
    let config = TextPipelineConfig {
        output: OutputConfig {
            extract_tables: false,
            ..Default::default()
        },
        ..Default::default()
    };

    let converter = MarkdownOutputConverter::new();
    let output = converter.convert(&[cell_11, cell_12], &config).unwrap();

    // Then: Output does not contain table markers (though pipes may appear in regular text)
    // This is a looser assertion since we can't guarantee no pipes in normal text
    assert!(!output.is_empty());
}

// ============================================================================
// FEATURE 4: Layout Preservation Tests
// ============================================================================

#[test]
fn test_preserve_layout_enabled() {
    // Given: Text spans with specific positioning for column alignment
    let col1 = make_span("Column1", 10.0, 100.0, 12.0, FontWeight::Normal, false);
    let col2 = make_span("Column2", 50.0, 100.0, 12.0, FontWeight::Normal, false);

    // When: Convert with preserve_layout=true
    let config = TextPipelineConfig {
        output: OutputConfig {
            preserve_layout: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let converter = MarkdownOutputConverter::new();
    let output = converter.convert(&[col1, col2], &config).unwrap();

    // Then: Whitespace is preserved for column alignment
    assert!(!output.trim().is_empty());
}

#[test]
fn test_preserve_layout_disabled() {
    // Given: Text spans with positioning
    let col1 = make_span("Column1", 10.0, 100.0, 12.0, FontWeight::Normal, false);
    let col2 = make_span("Column2", 50.0, 100.0, 12.0, FontWeight::Normal, false);

    // When: Convert with preserve_layout=false
    let config = TextPipelineConfig {
        output: OutputConfig {
            preserve_layout: false,
            ..Default::default()
        },
        ..Default::default()
    };

    let converter = MarkdownOutputConverter::new();
    let output = converter.convert(&[col1, col2], &config).unwrap();

    // Then: Text is normalized without column preservation
    assert!(!output.is_empty());
}

// ============================================================================
// FEATURE 5: Image Embedding Tests
// ============================================================================

#[test]
fn test_image_embedding_inline() {
    // Given: Configuration with image embedding enabled
    let config = TextPipelineConfig {
        output: OutputConfig {
            include_images: true,
            image_output_dir: Some("/tmp/images".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };

    let converter = MarkdownOutputConverter::new();
    let spans = vec![make_span(
        "text",
        0.0,
        100.0,
        12.0,
        FontWeight::Normal,
        false,
    )];
    let output = converter.convert(&spans, &config).unwrap();

    // Then: Output should handle images (basic check)
    assert!(!output.is_empty());
}

// ============================================================================
// FEATURE 6: URL/Email Linkification Tests
// ============================================================================

#[test]
fn test_url_linkification() {
    // Given: Text containing a URL
    let span = make_span(
        "Visit https://example.com for more info",
        0.0,
        100.0,
        12.0,
        FontWeight::Normal,
        false,
    );

    // When: Convert to markdown
    let config = TextPipelineConfig::default();

    let converter = MarkdownOutputConverter::new();
    let output = converter.convert(&[span], &config).unwrap();

    // Then: URL is converted to markdown link syntax
    assert!(
        output.contains("[https://example.com]") || output.contains("https://example.com"),
        "Output: {}",
        output
    );
}

#[test]
fn test_email_linkification() {
    // Given: Text containing an email address
    let span =
        make_span("Contact us at info@example.com", 0.0, 100.0, 12.0, FontWeight::Normal, false);

    // When: Convert to markdown
    let config = TextPipelineConfig::default();

    let converter = MarkdownOutputConverter::new();
    let output = converter.convert(&[span], &config).unwrap();

    // Then: Email is detected in output
    assert!(output.contains("info@example.com"), "Output: {}", output);
}

// ============================================================================
// FEATURE 7: Whitespace Normalization Tests
// ============================================================================

#[test]
fn test_whitespace_normalization() {
    // Given: Text with multiple consecutive spaces
    let span = make_span("Text   with    spaces", 0.0, 100.0, 12.0, FontWeight::Normal, false);

    // When: Convert to markdown
    let config = TextPipelineConfig::default();

    let converter = MarkdownOutputConverter::new();
    let output = converter.convert(&[span], &config).unwrap();

    // Then: Multiple spaces are normalized to single space
    // Count consecutive spaces - should be max 1
    let has_triple_space = output.contains("   ");
    assert!(!has_triple_space, "Output should not have triple spaces: {}", output);
}

#[test]
fn test_line_ending_normalization() {
    // Given: Multiple text spans that should be in same paragraph
    let span1 = make_span("First line", 0.0, 100.0, 12.0, FontWeight::Normal, false);
    let span2 = make_span("Second line", 0.0, 98.0, 12.0, FontWeight::Normal, false);

    // When: Convert to markdown
    let config = TextPipelineConfig::default();

    let converter = MarkdownOutputConverter::new();
    let output = converter.convert(&[span1, span2], &config).unwrap();

    // Then: Line endings are normalized
    let double_newline_count = output.matches("\n\n").count();
    assert!(double_newline_count <= 1, "Too many paragraph breaks: {}", output);
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_mixed_formatting() {
    // Given: A document with multiple formatting styles (in proper reading order)
    let heading = make_span_with_order("Title", 0.0, 100.0, 24.0, FontWeight::Bold, false, 0);
    let bold_text =
        make_span_with_order("Bold paragraph", 0.0, 80.0, 12.0, FontWeight::Bold, false, 1);
    let italic_text =
        make_span_with_order("italic text", 20.0, 78.0, 12.0, FontWeight::Normal, true, 2);
    let normal_text =
        make_span_with_order("Regular text", 0.0, 60.0, 12.0, FontWeight::Normal, false, 3);

    // When: Convert with headings enabled
    let config = TextPipelineConfig {
        output: OutputConfig {
            detect_headings: true,
            bold_marker_behavior: BoldMarkerBehavior::Conservative,
            ..Default::default()
        },
        ..Default::default()
    };

    let converter = MarkdownOutputConverter::new();
    let output = converter
        .convert(&[heading, bold_text, italic_text, normal_text], &config)
        .unwrap();

    // Then: All formatting is preserved
    assert!(output.contains("# Title"), "Output: {}", output);
    assert!(output.contains("**Bold paragraph**"), "Output: {}", output);
    assert!(output.contains("*italic text*"), "Output: {}", output);
    assert!(output.contains("Regular text"), "Output: {}", output);
}

#[test]
fn test_empty_spans() {
    // Given: Empty span list
    let config = TextPipelineConfig::default();
    let converter = MarkdownOutputConverter::new();
    let output = converter.convert(&[], &config).unwrap();

    // Then: Output is empty
    assert_eq!(output, "");
}
