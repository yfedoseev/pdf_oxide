//! HTML output converter.
//!
//! Converts ordered text spans to HTML format with support for:
//! - **Layout Mode**: CSS absolute positioning to preserve spatial document layout
//! - **Semantic Mode**: HTML5 semantic elements (h1-h3, p, strong, em)
//! - **Style Preservation**: Font weight, italics, and color attributes
//! - **Proper Escaping**: XSS-safe HTML output

use crate::error::Result;
use crate::layout::FontWeight;
use crate::pipeline::{OrderedTextSpan, TextPipelineConfig};
use crate::text::HyphenationHandler;

use super::OutputConverter;

/// HTML output converter.
///
/// Converts ordered text spans to semantic HTML with proper structure and optional layout preservation.
pub struct HtmlOutputConverter {
    /// Line spacing threshold ratio for paragraph detection.
    paragraph_gap_ratio: f32,
}

impl HtmlOutputConverter {
    /// Create a new HTML converter with default settings.
    pub fn new() -> Self {
        Self {
            paragraph_gap_ratio: 1.5,
        }
    }

    /// Check if a span should be rendered as bold.
    fn is_bold(&self, span: &OrderedTextSpan) -> bool {
        matches!(
            span.span.font_weight,
            FontWeight::Bold | FontWeight::Black | FontWeight::ExtraBold | FontWeight::SemiBold
        )
    }

    /// Check if a span is italic.
    fn is_italic(&self, span: &OrderedTextSpan) -> bool {
        span.span.is_italic
    }

    /// Detect paragraph breaks between spans based on vertical spacing.
    fn is_paragraph_break(&self, current: &OrderedTextSpan, previous: &OrderedTextSpan) -> bool {
        let line_height = current.span.font_size.max(previous.span.font_size);
        let gap = (previous.span.bbox.y - current.span.bbox.y).abs();
        gap > line_height * self.paragraph_gap_ratio
    }

    /// Detect if span should be a heading based on font size.
    fn heading_level(&self, span: &OrderedTextSpan, base_font_size: f32) -> Option<u8> {
        let size_ratio = span.span.font_size / base_font_size;
        if size_ratio >= 2.0 {
            Some(1)
        } else if size_ratio >= 1.5 {
            Some(2)
        } else if size_ratio >= 1.25 {
            Some(3)
        } else {
            None
        }
    }

    /// Escape HTML special characters to prevent XSS.
    fn escape_html(text: &str) -> String {
        text.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
    }

    /// Format a span as styled HTML.
    ///
    /// Applies bold (<strong>) and italic (<em>) tags as needed.
    fn format_span_with_styles(&self, span: &OrderedTextSpan, text: &str) -> String {
        let escaped = Self::escape_html(text);

        let mut result = escaped;

        // Apply italic tag if needed
        if self.is_italic(span) {
            result = format!("<em>{}</em>", result);
        }

        // Apply bold tag if needed
        if self.is_bold(span) {
            result = format!("<strong>{}</strong>", result);
        }

        result
    }

    /// Format a color as CSS hex notation.
    fn format_color(&self, span: &OrderedTextSpan) -> Option<String> {
        let color = &span.span.color;
        // Convert from 0.0-1.0 range to 0-255
        let r = (color.r * 255.0) as u8;
        let g = (color.g * 255.0) as u8;
        let b = (color.b * 255.0) as u8;

        // Only return color if not black (default)
        if r != 0 || g != 0 || b != 0 {
            Some(format!("#{:02x}{:02x}{:02x}", r, g, b))
        } else {
            None
        }
    }
}

impl Default for HtmlOutputConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputConverter for HtmlOutputConverter {
    fn convert(&self, spans: &[OrderedTextSpan], config: &TextPipelineConfig) -> Result<String> {
        if config.output.preserve_layout {
            self.convert_layout_mode(spans, config)
        } else {
            self.convert_semantic_mode(spans, config)
        }
    }

    fn name(&self) -> &'static str {
        "HtmlOutputConverter"
    }

    fn mime_type(&self) -> &'static str {
        "text/html"
    }
}

impl HtmlOutputConverter {
    /// Convert to HTML with layout preservation (CSS absolute positioning).
    ///
    /// Each span is placed in a div with inline CSS positioning to preserve
    /// the exact spatial layout from the PDF.
    fn convert_layout_mode(
        &self,
        spans: &[OrderedTextSpan],
        config: &TextPipelineConfig,
    ) -> Result<String> {
        if spans.is_empty() {
            return Ok(String::new());
        }

        // Sort by reading order
        let mut sorted: Vec<_> = spans.iter().collect();
        sorted.sort_by_key(|s| s.reading_order);

        let mut result = String::new();

        // Generate each span with absolute positioning
        for span in sorted {
            let text = self.format_span_with_styles(span, &span.span.text);
            let x = span.span.bbox.x;
            let y = span.span.bbox.y;
            let font_size = span.span.font_size;

            // Build style attribute
            let mut style =
                format!("position:absolute;left:{}pt;top:{}pt;font-size:{}pt;", x, y, font_size);

            // Add color if present
            if let Some(color) = self.format_color(span) {
                style.push_str(&format!("color:{};", color));
            }

            result.push_str(&format!("<div style=\"{}\">{}</div>\n", style, text));
        }

        // Apply hyphenation reconstruction if enabled
        if config.enable_hyphenation_reconstruction {
            let handler = HyphenationHandler::new();
            result = handler.process_text(&result);
        }

        Ok(result)
    }

    /// Convert to HTML with semantic markup (headings, paragraphs, etc.).
    ///
    /// Detects headings based on font size, creates paragraphs with proper
    /// markup, and applies style tags for bold and italic text.
    fn convert_semantic_mode(
        &self,
        spans: &[OrderedTextSpan],
        config: &TextPipelineConfig,
    ) -> Result<String> {
        if spans.is_empty() {
            return Ok(String::new());
        }

        // Sort by reading order
        let mut sorted: Vec<_> = spans.iter().collect();
        sorted.sort_by_key(|s| s.reading_order);

        // Calculate base font size for heading detection
        let base_font_size = if config.output.detect_headings {
            let sizes: Vec<f32> = sorted.iter().map(|s| s.span.font_size).collect();
            let mut sizes_sorted = sizes.clone();
            sizes_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sizes_sorted
                .get(sizes_sorted.len() / 2)
                .copied()
                .unwrap_or(12.0)
        } else {
            12.0
        };

        let mut result = String::new();
        let mut prev_span: Option<&OrderedTextSpan> = None;
        let mut in_paragraph = false;
        let mut current_content = String::new();

        for span in sorted {
            // Check for paragraph break
            if let Some(prev) = prev_span {
                if self.is_paragraph_break(span, prev) {
                    // End current paragraph
                    if in_paragraph && !current_content.is_empty() {
                        result.push_str(&format!("<p>{}</p>\n", current_content.trim()));
                        current_content.clear();
                        in_paragraph = false;
                    }
                }
            }

            // Check for heading
            if config.output.detect_headings {
                if let Some(level) = self.heading_level(span, base_font_size) {
                    // Close any open paragraph
                    if in_paragraph && !current_content.is_empty() {
                        result.push_str(&format!("<p>{}</p>\n", current_content.trim()));
                        current_content.clear();
                        in_paragraph = false;
                    }

                    // Add heading with style support
                    let text = self.format_span_with_styles(span, span.span.text.trim());
                    result.push_str(&format!("<h{}>{}</h{}>\n", level, text, level));
                    prev_span = Some(span);
                    continue;
                }
            }

            // Start paragraph if not in one
            if !in_paragraph {
                in_paragraph = true;
            }

            // Add text with styles
            let formatted = self.format_span_with_styles(span, &span.span.text);
            current_content.push_str(&formatted);

            prev_span = Some(span);
        }

        // Close any open paragraph
        if in_paragraph && !current_content.is_empty() {
            result.push_str(&format!("<p>{}</p>\n", current_content.trim()));
        }

        // Apply hyphenation reconstruction if enabled
        if config.enable_hyphenation_reconstruction {
            let handler = HyphenationHandler::new();
            result = handler.process_text(&result);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Rect;
    use crate::layout::{Color, TextSpan};

    fn make_span(
        text: &str,
        x: f32,
        y: f32,
        font_size: f32,
        weight: FontWeight,
    ) -> OrderedTextSpan {
        OrderedTextSpan::new(
            TextSpan {
                text: text.to_string(),
                bbox: Rect::new(x, y, 50.0, font_size),
                font_name: "Test".to_string(),
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

    #[test]
    fn test_empty_spans() {
        let converter = HtmlOutputConverter::new();
        let config = TextPipelineConfig::default();
        let result = converter.convert(&[], &config).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_single_paragraph() {
        let converter = HtmlOutputConverter::new();
        let config = TextPipelineConfig::default();
        let spans = vec![make_span(
            "Hello world",
            0.0,
            100.0,
            12.0,
            FontWeight::Normal,
        )];
        let result = converter.convert(&spans, &config).unwrap();
        assert_eq!(result, "<p>Hello world</p>\n");
    }

    #[test]
    fn test_bold_text() {
        let converter = HtmlOutputConverter::new();
        let config = TextPipelineConfig::default();
        let spans = vec![make_span("Bold", 0.0, 100.0, 12.0, FontWeight::Bold)];
        let result = converter.convert(&spans, &config).unwrap();
        assert_eq!(result, "<p><strong>Bold</strong></p>\n");
    }

    #[test]
    fn test_html_escaping() {
        let converter = HtmlOutputConverter::new();
        let config = TextPipelineConfig::default();
        let spans = vec![make_span(
            "<script>alert('XSS')</script>",
            0.0,
            100.0,
            12.0,
            FontWeight::Normal,
        )];
        let result = converter.convert(&spans, &config).unwrap();
        assert!(result.contains("&lt;script&gt;"));
        assert!(!result.contains("<script>"));
    }
}
