//! Plain text output converter.
//!
//! Converts ordered text spans to plain text format.

use crate::error::Result;
use crate::pipeline::{OrderedTextSpan, TextPipelineConfig};
use crate::text::HyphenationHandler;

use super::OutputConverter;

/// Plain text output converter.
///
/// Converts ordered text spans to plain text, preserving paragraph structure
/// but removing all formatting.
pub struct PlainTextConverter {
    /// Line spacing threshold ratio for paragraph detection.
    paragraph_gap_ratio: f32,
}

impl PlainTextConverter {
    /// Create a new plain text converter with default settings.
    pub fn new() -> Self {
        Self {
            paragraph_gap_ratio: 1.5,
        }
    }

    /// Detect paragraph breaks between spans based on vertical spacing.
    fn is_paragraph_break(&self, current: &OrderedTextSpan, previous: &OrderedTextSpan) -> bool {
        let line_height = current.span.font_size.max(previous.span.font_size);
        let gap = (previous.span.bbox.y - current.span.bbox.y).abs();
        gap > line_height * self.paragraph_gap_ratio
    }
}

impl Default for PlainTextConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputConverter for PlainTextConverter {
    fn convert(&self, spans: &[OrderedTextSpan], config: &TextPipelineConfig) -> Result<String> {
        if spans.is_empty() {
            return Ok(String::new());
        }

        // Sort by reading order
        let mut sorted: Vec<_> = spans.iter().collect();
        sorted.sort_by_key(|s| s.reading_order);

        let mut result = String::new();
        let mut prev_span: Option<&OrderedTextSpan> = None;

        for span in sorted {
            // Check for paragraph break
            if let Some(prev) = prev_span {
                if self.is_paragraph_break(span, prev) {
                    result.push_str("\n\n");
                } else {
                    // Check if same line
                    let same_line =
                        (span.span.bbox.y - prev.span.bbox.y).abs() < span.span.font_size * 0.5;
                    if !same_line {
                        result.push(' ');
                    }
                }
            }

            result.push_str(&span.span.text);
            prev_span = Some(span);
        }

        // Ensure trailing newline
        if !result.ends_with('\n') {
            result.push('\n');
        }

        // Apply hyphenation reconstruction if enabled
        if config.enable_hyphenation_reconstruction {
            let handler = HyphenationHandler::new();
            result = handler.process_text(&result);
        }

        Ok(result)
    }

    fn name(&self) -> &'static str {
        "PlainTextConverter"
    }

    fn mime_type(&self) -> &'static str {
        "text/plain"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Rect;
    use crate::layout::{Color, FontWeight, TextSpan};

    fn make_span(text: &str, x: f32, y: f32) -> OrderedTextSpan {
        OrderedTextSpan::new(
            TextSpan {
                text: text.to_string(),
                bbox: Rect::new(x, y, 50.0, 12.0),
                font_name: "Test".to_string(),
                font_size: 12.0,
                font_weight: FontWeight::Normal,
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

    #[test]
    fn test_empty_spans() {
        let converter = PlainTextConverter::new();
        let config = TextPipelineConfig::default();
        let result = converter.convert(&[], &config).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_single_line() {
        let converter = PlainTextConverter::new();
        let config = TextPipelineConfig::default();
        let spans = vec![make_span("Hello world", 0.0, 100.0)];
        let result = converter.convert(&spans, &config).unwrap();
        assert_eq!(result, "Hello world\n");
    }

    #[test]
    fn test_paragraph_break() {
        let converter = PlainTextConverter::new();
        let config = TextPipelineConfig::default();
        let mut spans = vec![
            make_span("First paragraph", 0.0, 100.0),
            make_span("Second paragraph", 0.0, 50.0), // Large gap indicates new paragraph
        ];
        spans[1].reading_order = 1;

        let result = converter.convert(&spans, &config).unwrap();
        assert!(result.contains("\n\n"));
    }
}
