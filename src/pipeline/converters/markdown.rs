//! Markdown output converter.
//!
//! Converts ordered text spans to Markdown format.

use crate::error::Result;
use crate::layout::FontWeight;
use crate::pipeline::{OrderedTextSpan, TextPipelineConfig};
use crate::text::HyphenationHandler;
use lazy_static::lazy_static;
use regex::Regex;

use super::OutputConverter;

lazy_static! {
    /// Regex for matching URLs in text
    static ref RE_URL: Regex = Regex::new(r"(https?://[^\s<>\[\]]*[^\s<>\[\].,!?;:])").unwrap();

    /// Regex for matching email addresses
    static ref RE_EMAIL: Regex = Regex::new(r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})").unwrap();
}

/// Markdown output converter.
///
/// Converts ordered text spans to Markdown format with optional formatting:
/// - Bold text using `**text**` markers
/// - Italic text using `*text*` markers
/// - Heading detection based on font size (when enabled)
/// - Paragraph separation based on vertical gaps
/// - Table detection and formatting
/// - Layout preservation with whitespace
/// - URL/Email linkification
/// - Whitespace normalization
pub struct MarkdownOutputConverter {
    /// Line spacing threshold ratio for paragraph detection.
    paragraph_gap_ratio: f32,
}

impl MarkdownOutputConverter {
    /// Create a new Markdown converter with default settings.
    pub fn new() -> Self {
        Self {
            paragraph_gap_ratio: 1.5,
        }
    }

    /// Create a Markdown converter with custom paragraph gap ratio.
    pub fn with_paragraph_gap(ratio: f32) -> Self {
        Self {
            paragraph_gap_ratio: ratio,
        }
    }

    /// Check if a span should be rendered as bold.
    fn is_bold(&self, span: &OrderedTextSpan, config: &TextPipelineConfig) -> bool {
        use crate::pipeline::config::BoldMarkerBehavior;

        match span.span.font_weight {
            FontWeight::Bold | FontWeight::Black | FontWeight::ExtraBold | FontWeight::SemiBold => {
                match config.output.bold_marker_behavior {
                    BoldMarkerBehavior::Aggressive => true,
                    BoldMarkerBehavior::Conservative => {
                        // Only apply bold to content-bearing text
                        span.span.text.chars().any(|c| !c.is_whitespace())
                    },
                }
            },
            _ => false,
        }
    }

    /// Check if a span should be rendered as italic.
    fn is_italic(&self, span: &OrderedTextSpan) -> bool {
        span.span.is_italic && span.span.text.chars().any(|c| !c.is_whitespace())
    }

    /// Format text with bold and/or italic markers.
    fn apply_formatting(&self, text: &str, is_bold: bool, is_italic: bool) -> String {
        if is_bold && is_italic {
            format!("***{}***", text)
        } else if is_bold {
            format!("**{}**", text)
        } else if is_italic {
            format!("*{}*", text)
        } else {
            text.to_string()
        }
    }

    /// Apply linkification to text (URLs and emails).
    fn linkify(&self, text: &str) -> String {
        // First replace URLs
        let mut result = RE_URL
            .replace_all(text, |caps: &regex::Captures| {
                let url = &caps[0];
                format!("[{}]({})", url, url)
            })
            .to_string();

        // Then replace emails
        result = RE_EMAIL
            .replace_all(&result, |caps: &regex::Captures| {
                let email = &caps[0];
                format!("[{}](mailto:{})", email, email)
            })
            .to_string();

        result
    }

    /// Normalize whitespace in text.
    fn normalize_whitespace(&self, text: &str) -> String {
        // Replace multiple spaces with single space
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// Detect paragraph breaks between spans based on vertical spacing.
    fn is_paragraph_break(&self, current: &OrderedTextSpan, previous: &OrderedTextSpan) -> bool {
        let line_height = current.span.font_size.max(previous.span.font_size);
        let gap = (previous.span.bbox.y - current.span.bbox.y).abs();
        gap > line_height * self.paragraph_gap_ratio
    }

    /// Detect if span should be a heading based on font size.
    ///
    /// Uses absolute font sizes (only for clear heading cases):
    /// - H1: 24pt and above
    /// - H2: 18-23pt
    /// - H3: 14-17pt
    ///
    /// Note: Falls back to ratio-based detection for more nuanced cases.
    fn heading_level_absolute(&self, span: &OrderedTextSpan) -> Option<u8> {
        let size = span.span.font_size;
        if size >= 24.0 {
            Some(1)
        } else if size >= 18.0 {
            Some(2)
        } else if size >= 14.0 {
            Some(3)
        } else {
            None
        }
    }

    /// Detect heading level based on font size ratio to base size.
    fn heading_level_ratio(&self, span: &OrderedTextSpan, base_font_size: f32) -> Option<u8> {
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

    /// Detect if text spans form a grid structure (potential table).
    ///
    /// A simple table detection heuristic: multiple rows with aligned columns.
    /// Returns a vector of (row_indices, col_count) tuples for table rows.
    fn detect_table_structure(&self, sorted: &[&OrderedTextSpan]) -> Vec<(Vec<usize>, usize)> {
        if sorted.len() < 4 {
            return Vec::new(); // Minimum 2x2 table
        }

        let mut rows: Vec<Vec<usize>> = Vec::new();
        let mut current_row: Vec<usize> = Vec::new();
        let mut current_y = sorted[0].span.bbox.y;

        for (idx, span) in sorted.iter().enumerate() {
            let y_threshold = span.span.font_size * 0.5;
            if (current_y - span.span.bbox.y).abs() <= y_threshold {
                // Same line
                current_row.push(idx);
            } else {
                // New line
                if !current_row.is_empty() {
                    rows.push(current_row.clone());
                    current_row.clear();
                }
                current_row.push(idx);
                current_y = span.span.bbox.y;
            }
        }

        if !current_row.is_empty() {
            rows.push(current_row);
        }

        // A table should have multiple rows with consistent column count
        if rows.len() >= 2 && rows.iter().all(|r| r.len() == rows[0].len()) {
            let col_count = rows[0].len();
            rows.into_iter().map(|r| (r, col_count)).collect()
        } else {
            Vec::new()
        }
    }
}

impl Default for MarkdownOutputConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputConverter for MarkdownOutputConverter {
    fn convert(&self, spans: &[OrderedTextSpan], config: &TextPipelineConfig) -> Result<String> {
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
            // Use median as base size
            sizes_sorted
                .get(sizes_sorted.len() / 2)
                .copied()
                .unwrap_or(12.0)
        } else {
            12.0
        };

        // Detect table structure if enabled
        let table_rows = if config.output.extract_tables {
            self.detect_table_structure(&sorted)
        } else {
            Vec::new()
        };

        // Build set of all table cell indices
        let table_set: std::collections::HashSet<usize> = table_rows
            .iter()
            .flat_map(|(row, _)| row.iter().copied())
            .collect();

        let mut result = String::new();
        let mut prev_span: Option<&OrderedTextSpan> = None;
        let mut current_line = String::new();

        for (idx, span) in sorted.iter().enumerate() {
            // Skip spans that are part of table (they'll be formatted separately)
            if table_set.contains(&idx) {
                // Format table once when we encounter the first table cell
                if let Some((first_row, _)) = table_rows.first() {
                    if idx == first_row[0] {
                        // Flush current line first
                        if !current_line.is_empty() {
                            result.push_str(current_line.trim());
                            result.push_str("\n\n");
                            current_line.clear();
                        }

                        // Format and add the table
                        let mut table_output = String::new();
                        for (row_idx, (row_indices, _)) in table_rows.iter().enumerate() {
                            table_output.push('|');
                            for &cell_idx in row_indices {
                                let text = sorted[cell_idx].span.text.trim();
                                table_output.push(' ');
                                table_output.push_str(text);
                                table_output.push(' ');
                                table_output.push('|');
                            }
                            table_output.push('\n');

                            // Add header separator after first row
                            if row_idx == 0 {
                                table_output.push('|');
                                for _ in row_indices {
                                    table_output.push_str("---|");
                                }
                                table_output.push('\n');
                            }
                        }

                        result.push_str(&table_output);
                        result.push('\n');
                        prev_span = None;
                        continue;
                    }
                }
                continue; // Skip other table cells
            }

            // Check for paragraph break
            if let Some(prev) = prev_span {
                if self.is_paragraph_break(span, prev) {
                    // End current line and add paragraph break
                    if !current_line.is_empty() {
                        result.push_str(current_line.trim());
                        result.push_str("\n\n");
                        current_line.clear();
                    }
                } else {
                    // Same paragraph - check if new line
                    let same_line =
                        (span.span.bbox.y - prev.span.bbox.y).abs() < span.span.font_size * 0.5;
                    if !same_line {
                        // New line within paragraph
                        if config.output.preserve_layout {
                            // Calculate spacing to preserve column alignment
                            let spacing = (span.span.bbox.x - prev.span.bbox.x).max(0.0) as usize;
                            for _ in 0..spacing.min(20) {
                                current_line.push(' ');
                            }
                        } else {
                            current_line.push(' ');
                        }
                    }
                }
            }

            // Check for heading
            if config.output.detect_headings {
                // Try absolute heading first, then ratio-based
                let level = self
                    .heading_level_absolute(span)
                    .or_else(|| self.heading_level_ratio(span, base_font_size));

                if let Some(level) = level {
                    // Flush current content
                    if !current_line.is_empty() {
                        result.push_str(current_line.trim());
                        result.push_str("\n\n");
                        current_line.clear();
                    }

                    // Add heading
                    let prefix = "#".repeat(level as usize);
                    result.push_str(&format!("{} {}\n\n", prefix, span.span.text.trim()));
                    prev_span = None;
                    continue;
                }
            }

            // Format text with bold/italic and apply linkification
            let mut text = span.span.text.as_str();

            // Normalize whitespace if not preserving layout
            let normalized;
            if !config.output.preserve_layout {
                normalized = self.normalize_whitespace(text);
                text = &normalized;
            }

            // Apply linkification
            let linkified = self.linkify(text);

            // Determine formatting
            let is_bold = self.is_bold(span, config);
            let is_italic = self.is_italic(span);
            let formatted = self.apply_formatting(&linkified, is_bold, is_italic);

            current_line.push_str(&formatted);

            prev_span = Some(span);
        }

        // Flush remaining content
        if !current_line.is_empty() {
            result.push_str(current_line.trim());
            result.push('\n');
        }

        // Final whitespace normalization
        let mut final_result = if config.output.preserve_layout {
            result
        } else {
            // Clean up excessive newlines while preserving paragraph breaks
            let cleaned = result
                .split("\n\n")
                .map(|para| para.trim())
                .filter(|para| !para.is_empty())
                .collect::<Vec<_>>()
                .join("\n\n");

            // Preserve final newline if the original had one
            if result.ends_with('\n') && !cleaned.ends_with('\n') {
                format!("{}\n", cleaned)
            } else {
                cleaned
            }
        };

        // Apply hyphenation reconstruction if enabled
        if config.enable_hyphenation_reconstruction {
            let handler = HyphenationHandler::new();
            final_result = handler.process_text(&final_result);
        }

        Ok(final_result)
    }

    fn name(&self) -> &'static str {
        "MarkdownOutputConverter"
    }

    fn mime_type(&self) -> &'static str {
        "text/markdown"
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
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        let result = converter.convert(&[], &config).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_single_span() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        let spans = vec![make_span(
            "Hello world",
            0.0,
            100.0,
            12.0,
            FontWeight::Normal,
        )];
        let result = converter.convert(&spans, &config).unwrap();
        assert_eq!(result, "Hello world\n");
    }

    #[test]
    fn test_bold_text() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        let spans = vec![make_span("Bold text", 0.0, 100.0, 12.0, FontWeight::Bold)];
        let result = converter.convert(&spans, &config).unwrap();
        assert_eq!(result, "**Bold text**\n");
    }

    #[test]
    fn test_whitespace_bold_conservative() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        // Whitespace-only bold should not have markers in conservative mode
        let spans = vec![make_span("   ", 0.0, 100.0, 12.0, FontWeight::Bold)];
        let result = converter.convert(&spans, &config).unwrap();
        // Should not contain bold markers
        assert!(!result.contains("**"));
    }
}
