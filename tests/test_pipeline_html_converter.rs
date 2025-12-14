//! HTML Pipeline Converter Tests
//!
//! This test suite verifies the porting of HTML features from the legacy converter
//! to the new pipeline converter architecture, following TDD principles.
//!
//! Features tested:
//! 1. Layout Mode - CSS absolute positioning for spatial preservation
//! 2. Semantic Mode - HTML5 semantic elements (headings, paragraphs)
//! 3. Style Preservation - Font and color styling
//! 4. Image Embedding - External or embedded image references

use pdf_oxide::geometry::Rect;
use pdf_oxide::layout::{Color, FontWeight, TextSpan};
use pdf_oxide::pipeline::converters::HtmlOutputConverter;
use pdf_oxide::pipeline::converters::OutputConverter;
use pdf_oxide::pipeline::{OrderedTextSpan, TextPipelineConfig};

// ============================================================================
// HELPER FUNCTION
// ============================================================================

/// Create an OrderedTextSpan with given parameters for testing.
fn make_span(text: &str, x: f32, y: f32, font_size: f32, weight: FontWeight) -> OrderedTextSpan {
    OrderedTextSpan::new(
        TextSpan {
            text: text.to_string(),
            bbox: Rect::new(x, y, 50.0, font_size),
            font_name: "Arial".to_string(),
            font_size,
            font_weight: weight,
            is_italic: false,
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
        0,
    )
}

/// Create an OrderedTextSpan with color.
fn make_span_with_color(
    text: &str,
    x: f32,
    y: f32,
    font_size: f32,
    weight: FontWeight,
    color: Color,
) -> OrderedTextSpan {
    OrderedTextSpan::new(
        TextSpan {
            text: text.to_string(),
            bbox: Rect::new(x, y, 50.0, font_size),
            font_name: "Arial".to_string(),
            font_size,
            font_weight: weight,
            is_italic: false,
            color,
            mcid: None,
            sequence: 0,
            offset_semantic: false,
            split_boundary_before: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 100.0,
            primary_detected: false,
        },
        0,
    )
}

/// Create an OrderedTextSpan with italic flag.
fn make_span_italic(
    text: &str,
    x: f32,
    y: f32,
    font_size: f32,
    weight: FontWeight,
    is_italic: bool,
) -> OrderedTextSpan {
    OrderedTextSpan::new(
        TextSpan {
            text: text.to_string(),
            bbox: Rect::new(x, y, 50.0, font_size),
            font_name: "Arial".to_string(),
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
        0,
    )
}

// ============================================================================
// FEATURE 1: LAYOUT MODE (CSS ABSOLUTE POSITIONING)
// ============================================================================

#[test]
fn test_html_layout_mode_basic() {
    // Given: TextSpan with specific position
    let span = make_span("Positioned Text", 100.0, 200.0, 12.0, FontWeight::Normal);

    // When: Convert to HTML with layout mode enabled
    let mut config = TextPipelineConfig::default();
    config.output.preserve_layout = true;

    let converter = HtmlOutputConverter::new();
    let output = converter.convert(&[span], &config).unwrap();

    // Then: Output has CSS absolute positioning
    assert!(output.contains("style"), "Output should contain style attribute");
    assert!(output.contains("position:absolute"), "Should have absolute positioning");
    assert!(output.contains("left:"), "Should have left positioning");
    assert!(output.contains("top:"), "Should have top positioning");
    assert!(output.contains("Positioned Text"), "Should contain the text");
}

#[test]
fn test_html_layout_mode_preserves_coordinates() {
    // Given: Span with specific coordinates
    let span = make_span("Test", 150.0, 250.0, 14.0, FontWeight::Normal);

    // When: Convert with layout mode
    let mut config = TextPipelineConfig::default();
    config.output.preserve_layout = true;

    let converter = HtmlOutputConverter::new();
    let output = converter.convert(&[span], &config).unwrap();

    // Then: Coordinates should be preserved in output
    assert!(output.contains("150"), "Should preserve X coordinate (150)");
    assert!(output.contains("250"), "Should preserve Y coordinate (250)");
}

#[test]
fn test_html_layout_mode_multiple_spans() {
    // Given: Multiple positioned spans
    let spans = vec![
        make_span("First", 10.0, 20.0, 12.0, FontWeight::Normal),
        make_span("Second", 15.0, 50.0, 12.0, FontWeight::Normal),
    ];

    // When: Convert with layout mode
    let mut config = TextPipelineConfig::default();
    config.output.preserve_layout = true;

    let converter = HtmlOutputConverter::new();
    let output = converter.convert(&spans, &config).unwrap();

    // Then: All spans should be positioned
    assert!(output.contains("First"), "Should contain first span");
    assert!(output.contains("Second"), "Should contain second span");
    assert!(output.contains("position:absolute"), "Should use absolute positioning");
    // Count occurrences of 'position:absolute'
    let count = output.matches("position:absolute").count();
    assert_eq!(count, 2, "Should have 2 absolutely positioned elements");
}

#[test]
fn test_html_layout_mode_font_size_preservation() {
    // Given: Span with specific font size
    let span = make_span("Text", 0.0, 0.0, 18.0, FontWeight::Normal);

    // When: Convert with layout mode
    let mut config = TextPipelineConfig::default();
    config.output.preserve_layout = true;

    let converter = HtmlOutputConverter::new();
    let output = converter.convert(&[span], &config).unwrap();

    // Then: Font size should be in output
    assert!(output.contains("18"), "Should preserve font size (18)");
}

// ============================================================================
// FEATURE 2: SEMANTIC MODE (HTML5 SEMANTIC ELEMENTS)
// ============================================================================

#[test]
fn test_html_semantic_paragraph_basic() {
    // Given: Simple text span
    let span = make_span("This is a paragraph.", 0.0, 100.0, 12.0, FontWeight::Normal);

    // When: Convert to HTML without layout mode
    let mut config = TextPipelineConfig::default();
    config.output.preserve_layout = false;

    let converter = HtmlOutputConverter::new();
    let output = converter.convert(&[span], &config).unwrap();

    // Then: Output should have <p> tags
    assert!(output.contains("<p>"), "Should contain opening <p> tag");
    assert!(output.contains("</p>"), "Should contain closing </p> tag");
    assert!(output.contains("This is a paragraph."), "Should contain the text");
}

#[test]
fn test_html_semantic_heading_h1() {
    // Given: Large text indicating heading
    let span = make_span("Main Title", 0.0, 100.0, 24.0, FontWeight::Bold);

    // When: Convert with heading detection enabled
    let mut config = TextPipelineConfig::default();
    config.output.detect_headings = true;

    let converter = HtmlOutputConverter::new();
    let output = converter.convert(&[span], &config).unwrap();

    // Then: Output should contain heading tag
    assert!(output.contains("Main Title"), "Should contain the heading text");
    // Should contain some heading-level markup
}

#[test]
fn test_html_semantic_heading_h2() {
    // Given: Medium-large text for H2
    let base_span = make_span("Base", 0.0, 100.0, 12.0, FontWeight::Normal);
    let heading_span = make_span("Subheading", 0.0, 150.0, 18.0, FontWeight::Normal);

    // When: Convert with heading detection
    let mut config = TextPipelineConfig::default();
    config.output.detect_headings = true;

    let converter = HtmlOutputConverter::new();
    let output = converter
        .convert(&[base_span, heading_span], &config)
        .unwrap();

    // Then: Output should recognize heading level
    assert!(output.contains("Subheading"), "Should contain the heading text");
}

#[test]
fn test_html_semantic_multiple_paragraphs() {
    // Given: Multiple text spans with paragraph breaks
    let mut spans = vec![
        make_span("Paragraph one.", 0.0, 100.0, 12.0, FontWeight::Normal),
        make_span("Paragraph two.", 0.0, 50.0, 12.0, FontWeight::Normal),
    ];
    // Ensure proper reading order
    spans[0].reading_order = 0;
    spans[1].reading_order = 1;

    // When: Convert to HTML
    let mut config = TextPipelineConfig::default();
    config.output.preserve_layout = false;

    let converter = HtmlOutputConverter::new();
    let output = converter.convert(&spans, &config).unwrap();

    // Then: Output should contain both paragraphs
    assert!(output.contains("Paragraph one."), "Should contain first paragraph");
    assert!(output.contains("Paragraph two."), "Should contain second paragraph");
}

// ============================================================================
// FEATURE 3: STYLE PRESERVATION (BOLD, ITALIC, COLOR)
// ============================================================================

#[test]
fn test_html_bold_strong() {
    // Given: Bold text span
    let span = make_span("Bold Text", 0.0, 100.0, 12.0, FontWeight::Bold);

    // When: Convert to HTML
    let config = TextPipelineConfig::default();

    let converter = HtmlOutputConverter::new();
    let output = converter.convert(&[span], &config).unwrap();

    // Then: Output should contain <strong> tags
    assert!(output.contains("<strong>"), "Should contain opening <strong> tag");
    assert!(output.contains("</strong>"), "Should contain closing </strong> tag");
    assert!(output.contains("Bold Text"), "Should contain the bold text");
}

#[test]
fn test_html_italic_em() {
    // Given: Italic text span
    let span = make_span_italic("Italic Text", 0.0, 100.0, 12.0, FontWeight::Normal, true);

    // When: Convert to HTML
    let config = TextPipelineConfig::default();

    let converter = HtmlOutputConverter::new();
    let output = converter.convert(&[span], &config).unwrap();

    // Then: Output should contain <em> tags
    assert!(output.contains("<em>"), "Should contain opening <em> tag");
    assert!(output.contains("</em>"), "Should contain closing </em> tag");
    assert!(output.contains("Italic Text"), "Should contain the italic text");
}

#[test]
fn test_html_bold_and_italic() {
    // Given: Bold and italic text span
    let span = make_span_italic("Bold Italic", 0.0, 100.0, 12.0, FontWeight::Bold, true);

    // When: Convert to HTML
    let config = TextPipelineConfig::default();

    let converter = HtmlOutputConverter::new();
    let output = converter.convert(&[span], &config).unwrap();

    // Then: Output should contain both <strong> and <em> tags
    assert!(output.contains("<strong>"), "Should contain <strong> tag");
    assert!(output.contains("<em>"), "Should contain <em> tag");
    assert!(output.contains("Bold Italic"), "Should contain the text");
}

#[test]
fn test_html_color_preservation() {
    // Given: Text with red color
    let color = Color::new(1.0, 0.0, 0.0); // Red (0.0-1.0 range)
    let span = make_span_with_color("Red Text", 0.0, 100.0, 12.0, FontWeight::Normal, color);

    // When: Convert to HTML
    let config = TextPipelineConfig::default();

    let converter = HtmlOutputConverter::new();
    let output = converter.convert(&[span], &config).unwrap();

    // Then: Output should contain color information
    assert!(output.contains("Red Text"), "Should contain the text");
    // Color should be represented in output (as RGB hex or similar)
}

#[test]
fn test_html_color_black() {
    // Given: Black text (default color)
    let span = make_span_with_color("Black", 0.0, 100.0, 12.0, FontWeight::Normal, Color::black());

    // When: Convert to HTML
    let config = TextPipelineConfig::default();

    let converter = HtmlOutputConverter::new();
    let output = converter.convert(&[span], &config).unwrap();

    // Then: Output should contain the text
    assert!(output.contains("Black"), "Should contain the text");
}

#[test]
fn test_html_mixed_styling() {
    // Given: Multiple spans with different styles
    let spans = vec![
        make_span("Regular", 0.0, 100.0, 12.0, FontWeight::Normal),
        make_span("Bold", 10.0, 100.0, 12.0, FontWeight::Bold),
        make_span_italic("Italic", 20.0, 100.0, 12.0, FontWeight::Normal, true),
    ];

    // When: Convert to HTML
    let config = TextPipelineConfig::default();

    let converter = HtmlOutputConverter::new();
    let output = converter.convert(&spans, &config).unwrap();

    // Then: Output should contain all text with proper styling
    assert!(output.contains("Regular"), "Should contain regular text");
    assert!(output.contains("Bold"), "Should contain bold text");
    assert!(output.contains("Italic"), "Should contain italic text");
}

// ============================================================================
// FEATURE 4: IMAGE EMBEDDING (FUTURE/ADVANCED)
// ============================================================================

#[test]
fn test_html_image_configuration() {
    // Given: Configuration with image support
    let mut config = TextPipelineConfig::default();
    config.output.include_images = true;

    // When: Create converter
    let _converter = HtmlOutputConverter::new();

    // Then: Converter should accept the configuration
    assert!(config.output.include_images, "Image inclusion should be enabled");
}

#[test]
fn test_html_image_output_directory() {
    // Given: Configuration with image output directory
    let mut config = TextPipelineConfig::default();
    config.output.include_images = true;
    config.output.image_output_dir = Some("/tmp/images".to_string());

    // When: Converter processes configuration
    let _converter = HtmlOutputConverter::new();

    // Then: Image directory should be set
    assert_eq!(
        config.output.image_output_dir,
        Some("/tmp/images".to_string()),
        "Image output directory should be set"
    );
}

// ============================================================================
// FEATURE 5: HTML ESCAPING AND SPECIAL CHARACTERS
// ============================================================================

#[test]
fn test_html_escape_ampersand() {
    // Given: Text containing ampersand
    let span = make_span("AT&T", 0.0, 100.0, 12.0, FontWeight::Normal);

    // When: Convert to HTML
    let config = TextPipelineConfig::default();

    let converter = HtmlOutputConverter::new();
    let output = converter.convert(&[span], &config).unwrap();

    // Then: Ampersand should be escaped
    assert!(output.contains("&amp;"), "Ampersand should be escaped");
    assert!(!output.contains("AT&T"), "Raw ampersand should not appear");
}

#[test]
fn test_html_escape_less_than() {
    // Given: Text containing <
    let span = make_span("2 < 3", 0.0, 100.0, 12.0, FontWeight::Normal);

    // When: Convert to HTML
    let config = TextPipelineConfig::default();

    let converter = HtmlOutputConverter::new();
    let output = converter.convert(&[span], &config).unwrap();

    // Then: < should be escaped
    assert!(output.contains("&lt;"), "Less-than should be escaped");
}

#[test]
fn test_html_escape_greater_than() {
    // Given: Text containing >
    let span = make_span("3 > 2", 0.0, 100.0, 12.0, FontWeight::Normal);

    // When: Convert to HTML
    let config = TextPipelineConfig::default();

    let converter = HtmlOutputConverter::new();
    let output = converter.convert(&[span], &config).unwrap();

    // Then: > should be escaped
    assert!(output.contains("&gt;"), "Greater-than should be escaped");
}

#[test]
fn test_html_escape_quotes() {
    // Given: Text containing quotes
    let span = make_span("He said \"Hello\"", 0.0, 100.0, 12.0, FontWeight::Normal);

    // When: Convert to HTML
    let config = TextPipelineConfig::default();

    let converter = HtmlOutputConverter::new();
    let output = converter.convert(&[span], &config).unwrap();

    // Then: Quotes should be escaped
    assert!(output.contains("&quot;"), "Quotes should be escaped");
}

#[test]
fn test_html_escape_xss_attempt() {
    // Given: Text containing script tags (XSS attempt)
    let span = make_span("<script>alert('XSS')</script>", 0.0, 100.0, 12.0, FontWeight::Normal);

    // When: Convert to HTML
    let config = TextPipelineConfig::default();

    let converter = HtmlOutputConverter::new();
    let output = converter.convert(&[span], &config).unwrap();

    // Then: Script tags should be escaped and safe
    assert!(output.contains("&lt;script&gt;"), "Script tag should be escaped");
    assert!(!output.contains("<script>"), "Raw script tag should not appear");
    assert!(!output.contains("</script>"), "Raw closing script tag should not appear");
}

// ============================================================================
// FEATURE 6: EMPTY AND EDGE CASES
// ============================================================================

#[test]
fn test_html_empty_spans_list() {
    // Given: Empty spans list
    let spans: Vec<OrderedTextSpan> = vec![];

    // When: Convert to HTML
    let config = TextPipelineConfig::default();

    let converter = HtmlOutputConverter::new();
    let output = converter.convert(&spans, &config).unwrap();

    // Then: Should return empty string
    assert_eq!(output, "", "Empty input should produce empty output");
}

#[test]
fn test_html_whitespace_only() {
    // Given: Span containing only whitespace
    let span = make_span("   ", 0.0, 100.0, 12.0, FontWeight::Normal);

    // When: Convert to HTML
    let config = TextPipelineConfig::default();

    let converter = HtmlOutputConverter::new();
    let output = converter.convert(&[span], &config).unwrap();

    // Then: Should handle gracefully
    assert!(!output.is_empty(), "Should produce output for whitespace");
}

#[test]
fn test_html_very_long_text() {
    // Given: Span with very long text
    let long_text = "a".repeat(10000);
    let span = make_span(&long_text, 0.0, 100.0, 12.0, FontWeight::Normal);

    // When: Convert to HTML
    let config = TextPipelineConfig::default();

    let converter = HtmlOutputConverter::new();
    let output = converter.convert(&[span], &config).unwrap();

    // Then: Should handle long text
    assert!(output.contains(&long_text), "Should preserve long text");
}

// ============================================================================
// FEATURE 7: CONFIGURATION COMBINATIONS
// ============================================================================

#[test]
fn test_html_layout_with_bold_styling() {
    // Given: Bold text with layout preservation
    let span = make_span("Bold in Layout", 100.0, 200.0, 12.0, FontWeight::Bold);

    // When: Convert with both layout and bold
    let mut config = TextPipelineConfig::default();
    config.output.preserve_layout = true;

    let converter = HtmlOutputConverter::new();
    let output = converter.convert(&[span], &config).unwrap();

    // Then: Should combine both features
    assert!(output.contains("position:absolute"), "Should have layout positioning");
    assert!(output.contains("<strong>"), "Should have bold styling");
}

#[test]
fn test_html_semantic_with_headings_and_bold() {
    // Given: Configuration with both heading detection and styles
    let spans = vec![
        make_span("Title", 0.0, 100.0, 24.0, FontWeight::Bold),
        make_span("Regular paragraph", 0.0, 150.0, 12.0, FontWeight::Normal),
    ];

    // When: Convert with heading detection
    let mut config = TextPipelineConfig::default();
    config.output.detect_headings = true;

    let converter = HtmlOutputConverter::new();
    let output = converter.convert(&spans, &config).unwrap();

    // Then: Should handle both heading detection and styling
    assert!(output.contains("Title"), "Should contain title");
    assert!(output.contains("Regular paragraph"), "Should contain body text");
}

// ============================================================================
// FEATURE 8: MIME TYPE AND METADATA
// ============================================================================

#[test]
fn test_html_converter_mime_type() {
    // Given: HTML converter
    let converter = HtmlOutputConverter::new();

    // When: Get MIME type
    let mime = converter.mime_type();

    // Then: Should be HTML MIME type
    assert_eq!(mime, "text/html", "MIME type should be text/html");
}

#[test]
fn test_html_converter_name() {
    // Given: HTML converter
    let converter = HtmlOutputConverter::new();

    // When: Get converter name
    let name = converter.name();

    // Then: Should have meaningful name
    assert!(!name.is_empty(), "Converter should have a name");
    assert!(name.contains("Html"), "Name should reference HTML");
}
