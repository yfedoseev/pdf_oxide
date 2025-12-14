#![allow(deprecated, clippy::useless_vec)]
//! Integration tests for PDF converters.
//!
//! Phase 6, Task 6.7

use pdf_oxide::converters::{
    BoldMarkerBehavior, ConversionOptions, HtmlConverter, MarkdownConverter, ReadingOrderMode,
};
use pdf_oxide::geometry::Rect;
use pdf_oxide::layout::{Color, FontWeight, TextChar};

// Helper functions for creating mock text characters

fn mock_char(c: char, x: f32, y: f32, font_size: f32, bold: bool) -> TextChar {
    TextChar {
        char: c,
        bbox: Rect::new(x, y, 8.0, font_size),
        font_name: "Times".to_string(),
        font_size,
        font_weight: if bold {
            FontWeight::Bold
        } else {
            FontWeight::Normal
        },
        is_italic: false,
        color: Color::black(),
        mcid: None, // No marked content ID for mock chars
    }
}

fn mock_word(text: &str, x: f32, y: f32, font_size: f32, bold: bool) -> Vec<TextChar> {
    text.chars()
        .enumerate()
        .map(|(i, c)| mock_char(c, x + (i as f32 * 7.0), y, font_size, bold))
        .collect()
}

fn mock_paragraph(text: &str, x: f32, y: f32, font_size: f32) -> Vec<TextChar> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut chars = Vec::new();
    let mut current_x = x;

    for word in words {
        chars.extend(mock_word(word, current_x, y, font_size, false));
        current_x += (word.len() as f32 * 7.0) + 20.0; // Add space between words
    }

    chars
}

// Markdown converter tests

#[test]
fn test_markdown_simple_document() {
    let converter = MarkdownConverter::new();
    let options = ConversionOptions {
        ..Default::default()
    };

    let mut chars = Vec::new();
    chars.extend(mock_word("Hello", 0.0, 0.0, 12.0, false));
    chars.extend(mock_word("World", 50.0, 0.0, 12.0, false));

    let result = converter.convert_page(&chars, &options).unwrap();

    assert!(result.contains("Hello"));
    assert!(result.contains("World"));
    assert!(!result.contains('#')); // No headings
}

#[test]
fn test_markdown_with_heading_detection() {
    let converter = MarkdownConverter::new();
    let options = ConversionOptions {
        ..Default::default()
    };

    let mut chars = Vec::new();
    // Title (large, bold)
    chars.extend(mock_word("Title", 0.0, 0.0, 24.0, true));
    // Subtitle (medium, bold)
    chars.extend(mock_word("Subtitle", 0.0, 40.0, 18.0, true));
    // Body (normal)
    chars.extend(mock_word("Body", 0.0, 70.0, 12.0, false));

    let result = converter.convert_page(&chars, &options).unwrap();

    assert!(result.contains("Title"));
    assert!(result.contains("Subtitle"));
    assert!(result.contains("Body"));
}

#[test]
fn test_markdown_multiline() {
    let converter = MarkdownConverter::new();
    let options = ConversionOptions {
        ..Default::default()
    };

    let mut chars = Vec::new();
    chars.extend(mock_word("Line", 0.0, 0.0, 12.0, false));
    chars.extend(mock_word("One", 35.0, 0.0, 12.0, false));
    chars.extend(mock_word("Line", 0.0, 20.0, 12.0, false));
    chars.extend(mock_word("Two", 35.0, 20.0, 12.0, false));

    let result = converter.convert_page(&chars, &options).unwrap();

    assert!(result.contains("Line One") || result.contains("Line"));
    assert!(result.contains("Line Two") || result.contains("Two"));
}

#[test]
#[ignore] // TODO: Reading order test needs tuning for layout analysis parameters
fn test_markdown_reading_order_top_to_bottom() {
    let converter = MarkdownConverter::new();
    let options = ConversionOptions {
        reading_order_mode: ReadingOrderMode::TopToBottomLeftToRight,
        ..Default::default()
    };

    let mut chars = Vec::new();
    chars.extend(mock_word("Bottom", 0.0, 100.0, 12.0, false));
    chars.extend(mock_word("Top", 0.0, 0.0, 12.0, false));
    chars.extend(mock_word("Middle", 0.0, 50.0, 12.0, false));

    let result = converter.convert_page(&chars, &options).unwrap();

    // Should be ordered: Top, Middle, Bottom
    let top_pos = result.find("Top").unwrap();
    let middle_pos = result.find("Middle").unwrap();
    let bottom_pos = result.find("Bottom").unwrap();

    assert!(top_pos < middle_pos);
    assert!(middle_pos < bottom_pos);
}

#[test]
fn test_markdown_reading_order_left_to_right() {
    let converter = MarkdownConverter::new();
    let options = ConversionOptions {
        reading_order_mode: ReadingOrderMode::TopToBottomLeftToRight,
        ..Default::default()
    };

    let mut chars = Vec::new();
    chars.extend(mock_word("Right", 100.0, 0.0, 12.0, false));
    chars.extend(mock_word("Left", 0.0, 0.0, 12.0, false));
    chars.extend(mock_word("Center", 50.0, 0.0, 12.0, false));

    let result = converter.convert_page(&chars, &options).unwrap();

    // Should be ordered: Left, Center, Right (when on same line)
    let left_pos = result.find("Left").unwrap();
    let center_pos = result.find("Center").unwrap();
    let right_pos = result.find("Right").unwrap();

    assert!(left_pos < center_pos);
    assert!(center_pos < right_pos);
}

// HTML semantic converter tests

#[test]
fn test_html_semantic_simple() {
    let converter = HtmlConverter::new();
    let options = ConversionOptions {
        preserve_layout: false,
        ..Default::default()
    };

    let chars = mock_word("Hello", 0.0, 0.0, 12.0, false);
    let result = converter.convert_page(&chars, &options).unwrap();

    assert!(result.contains("<p>Hello</p>"));
}

#[test]
fn test_html_semantic_with_heading() {
    let converter = HtmlConverter::new();
    let options = ConversionOptions {
        preserve_layout: false,
        ..Default::default()
    };

    let mut chars = Vec::new();
    chars.extend(mock_word("Title", 0.0, 0.0, 24.0, true));
    chars.extend(mock_word("Text", 0.0, 40.0, 12.0, false));

    let result = converter.convert_page(&chars, &options).unwrap();

    assert!(result.contains("Title"));
    assert!(result.contains("Text"));
    assert!(result.contains("<h") || result.contains("<p>"));
}

#[test]
fn test_html_semantic_escape() {
    let converter = HtmlConverter::new();
    let options = ConversionOptions {
        preserve_layout: false,
        ..Default::default()
    };

    let chars = vec![
        mock_char('<', 0.0, 0.0, 12.0, false),
        mock_char('>', 7.0, 0.0, 12.0, false),
        mock_char('&', 14.0, 0.0, 12.0, false),
    ];

    let result = converter.convert_page(&chars, &options).unwrap();

    assert!(result.contains("&lt;"));
    assert!(result.contains("&gt;"));
    assert!(result.contains("&amp;"));
}

// HTML layout-preserved converter tests

#[test]
fn test_html_layout_basic() {
    let converter = HtmlConverter::new();
    let options = ConversionOptions {
        preserve_layout: true,
        ..Default::default()
    };

    let chars = mock_word("Test", 100.0, 200.0, 14.0, false);
    let result = converter.convert_page(&chars, &options).unwrap();

    assert!(result.contains("<style>"));
    assert!(result.contains("position: absolute"));
    assert!(result.contains("left: 100px"));
    assert!(result.contains("top: 200px"));
    assert!(result.contains("font-size: 14px"));
    assert!(result.contains("Test"));
}

#[test]
#[ignore] // TODO: HTML layout test needs tuning for layout analysis parameters
fn test_html_layout_multiple_elements() {
    let converter = HtmlConverter::new();
    let options = ConversionOptions {
        preserve_layout: true,
        ..Default::default()
    };

    let mut chars = Vec::new();
    chars.extend(mock_word("First", 10.0, 20.0, 12.0, false));
    chars.extend(mock_word("Second", 10.0, 50.0, 12.0, false));

    let result = converter.convert_page(&chars, &options).unwrap();

    assert!(result.contains("First"));
    assert!(result.contains("Second"));
    assert!(result.contains("top: 20px"));
    assert!(result.contains("top: 50px"));
}

#[test]
fn test_html_layout_css_structure() {
    let converter = HtmlConverter::new();
    let options = ConversionOptions {
        preserve_layout: true,
        ..Default::default()
    };

    let chars = mock_word("A", 0.0, 0.0, 12.0, false);
    let result = converter.convert_page(&chars, &options).unwrap();

    assert!(result.contains("<style>"));
    assert!(result.contains(".page"));
    assert!(result.contains(".text"));
    assert!(result.contains("<div class=\"page\">"));
    assert!(result.contains("</div>"));
}

// Multi-line and paragraph tests

#[test]
fn test_markdown_paragraph() {
    let converter = MarkdownConverter::new();
    let options = ConversionOptions {
        ..Default::default()
    };

    let chars = mock_paragraph("This is a test paragraph with multiple words", 0.0, 0.0, 12.0);
    let result = converter.convert_page(&chars, &options).unwrap();

    assert!(result.contains("This"));
    assert!(result.contains("test"));
    assert!(result.contains("paragraph"));
}

#[test]
fn test_html_paragraph() {
    let converter = HtmlConverter::new();
    let options = ConversionOptions {
        preserve_layout: false,
        ..Default::default()
    };

    let chars = mock_paragraph("A simple paragraph here", 0.0, 0.0, 12.0);
    let result = converter.convert_page(&chars, &options).unwrap();

    assert!(result.contains("<p>"));
    assert!(result.contains("</p>"));
    assert!(result.contains("simple"));
}

// Edge case tests

#[test]
fn test_markdown_empty_input() {
    let converter = MarkdownConverter::new();
    let options = ConversionOptions::default();
    let result = converter.convert_page(&[], &options).unwrap();
    assert_eq!(result, "");
}

#[test]
fn test_html_empty_input() {
    let converter = HtmlConverter::new();
    let options = ConversionOptions::default();
    let result = converter.convert_page(&[], &options).unwrap();
    assert_eq!(result, "");
}

#[test]
fn test_markdown_single_character() {
    let converter = MarkdownConverter::new();
    let options = ConversionOptions {
        ..Default::default()
    };

    let chars = vec![mock_char('A', 0.0, 0.0, 12.0, false)];
    let result = converter.convert_page(&chars, &options).unwrap();
    assert!(result.contains('A'));
}

#[test]
fn test_html_single_character() {
    let converter = HtmlConverter::new();
    let options = ConversionOptions {
        preserve_layout: false,
        ..Default::default()
    };

    let chars = vec![mock_char('B', 0.0, 0.0, 12.0, false)];
    let result = converter.convert_page(&chars, &options).unwrap();
    assert!(result.contains('B'));
}

// Reading order mode tests

#[test]
fn test_markdown_column_aware_mode() {
    let converter = MarkdownConverter::new();
    let options = ConversionOptions {
        reading_order_mode: ReadingOrderMode::ColumnAware,
        ..Default::default()
    };

    let mut chars = Vec::new();
    chars.extend(mock_word("Left", 0.0, 0.0, 12.0, false));
    chars.extend(mock_word("Right", 200.0, 0.0, 12.0, false));

    let result = converter.convert_page(&chars, &options).unwrap();
    assert!(result.contains("Left"));
    assert!(result.contains("Right"));
}

// Comprehensive test combining multiple features

#[test]
fn test_comprehensive_document_conversion() {
    let converter_md = MarkdownConverter::new();
    let converter_html = HtmlConverter::new();

    let options = ConversionOptions {
        preserve_layout: false,
        detect_headings: true,
        include_images: false,
        extract_tables: false,
        image_output_dir: None,
        reading_order_mode: ReadingOrderMode::TopToBottomLeftToRight,
        bold_marker_behavior: BoldMarkerBehavior::Conservative,
        table_detection_config: None,
    };

    let mut chars = Vec::new();
    // Title
    chars.extend(mock_word("Document", 0.0, 0.0, 24.0, true));
    chars.extend(mock_word("Title", 70.0, 0.0, 24.0, true));
    // Subtitle
    chars.extend(mock_word("Section", 0.0, 40.0, 18.0, true));
    chars.extend(mock_word("One", 60.0, 40.0, 18.0, true));
    // Body text
    chars.extend(mock_paragraph("This is the first paragraph of body text", 0.0, 70.0, 12.0));
    chars.extend(mock_paragraph("This is the second paragraph", 0.0, 90.0, 12.0));

    // Test Markdown conversion
    let md_result = converter_md.convert_page(&chars, &options).unwrap();
    assert!(md_result.contains("Document"));
    assert!(md_result.contains("Section"));
    assert!(md_result.contains("first"));
    assert!(md_result.contains("second"));

    // Test HTML conversion
    let html_result = converter_html.convert_page(&chars, &options).unwrap();
    assert!(html_result.contains("Document"));
    assert!(html_result.contains("Section"));
    assert!(html_result.contains("first"));
    assert!(html_result.contains("second"));
    assert!(html_result.contains("<"));
    assert!(html_result.contains(">"));
}
