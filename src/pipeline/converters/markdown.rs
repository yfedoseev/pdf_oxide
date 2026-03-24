//! Markdown output converter.
//!
//! Converts ordered text spans to Markdown format.

use crate::error::Result;
use crate::layout::FontWeight;
use crate::pipeline::{OrderedTextSpan, TextPipelineConfig};
use crate::structure::table_extractor::ExtractedTable;
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
        // Quick pre-check: skip regex for spans that can't contain URLs or emails.
        // This avoids regex overhead for ~95% of regular text spans.
        let might_have_url = text.contains("://") || text.contains("www.");
        let might_have_email = text.contains('@');

        if !might_have_url && !might_have_email {
            return text.to_string();
        }

        let mut result = if might_have_url {
            RE_URL
                .replace_all(text, |caps: &regex::Captures| {
                    let url = &caps[0];
                    format!("[{}]({})", url, url)
                })
                .to_string()
        } else {
            text.to_string()
        };

        if might_have_email {
            result = RE_EMAIL
                .replace_all(&result, |caps: &regex::Captures| {
                    let email = &caps[0];
                    format!("[{}](mailto:{})", email, email)
                })
                .to_string();
        }

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

    /// Check if a span consists of a single bullet character.
    ///
    /// Common bullet characters used in PDF documents:
    /// ► • ▪ ▸ ‣ ◦ ● ■ ◆ ○ □
    fn is_bullet_span(text: &str) -> bool {
        let t = text.trim();
        matches!(t, "►" | "•" | "▪" | "▸" | "‣" | "◦" | "●" | "■" | "◆" | "○" | "□")
    }

    /// Check if text starts with a bullet character (for inline bullets).
    fn starts_with_bullet(text: &str) -> bool {
        let t = text.trim_start();
        t.starts_with('►')
            || t.starts_with('•')
            || t.starts_with('▪')
            || t.starts_with('▸')
            || t.starts_with('‣')
            || t.starts_with('◦')
            || t.starts_with('●')
            || t.starts_with('■')
            || t.starts_with('◆')
            || t.starts_with('○')
            || t.starts_with('□')
    }

    /// Strip the leading bullet character from text, returning the rest.
    fn strip_bullet(text: &str) -> &str {
        let t = text.trim_start();
        // Bullet characters are single Unicode code points; skip first char
        if Self::starts_with_bullet(t) {
            let mut chars = t.chars();
            chars.next(); // skip bullet
            chars.as_str().trim_start()
        } else {
            text
        }
    }

    /// Detect if span should be a heading based on font size.
    ///
    /// Uses absolute font sizes (only for clear heading cases):
    /// - H1: 24pt and above
    /// - H2: 18-23pt
    /// - H3: 16-17pt
    ///
    /// Note: Falls back to ratio-based detection for more nuanced cases.
    /// Headings must also be short (< 200 chars) to avoid promoting body paragraphs.
    fn heading_level_absolute(&self, span: &OrderedTextSpan) -> Option<u8> {
        let size = span.span.font_size;
        let text_len = span.span.text.trim().len();
        // Headings must be short but non-trivial
        if !(2..=200).contains(&text_len) {
            return None;
        }
        if size >= 24.0 {
            Some(1)
        } else if size >= 18.0 {
            Some(2)
        } else if size >= 16.0 {
            Some(3)
        } else {
            None
        }
    }

    /// Detect heading level based on font size ratio to base size.
    /// Requires a meaningful size difference to avoid promoting slightly-larger text.
    /// Bold text gets a lower threshold since bold+larger is a strong heading signal.
    fn heading_level_ratio(&self, span: &OrderedTextSpan, base_font_size: f32) -> Option<u8> {
        let text_len = span.span.text.trim().len();
        // Headings must be short but non-trivial
        if !(2..=200).contains(&text_len) {
            return None;
        }
        let size_ratio = span.span.font_size / base_font_size;
        let is_bold = matches!(
            span.span.font_weight,
            FontWeight::Bold | FontWeight::Black | FontWeight::ExtraBold | FontWeight::SemiBold
        );
        if size_ratio >= 2.0 {
            Some(1)
        } else if size_ratio >= 1.5 {
            Some(2)
        } else if size_ratio >= 1.3 {
            Some(3)
        } else if is_bold && size_ratio >= 1.15 {
            // Bold text with even slight size increase is a heading signal
            Some(3)
        } else {
            None
        }
    }

    /// Check if a span's bbox overlaps with any table region.
    fn span_in_table(&self, span: &OrderedTextSpan, tables: &[ExtractedTable]) -> Option<usize> {
        let sx = span.span.bbox.x;
        let sy = span.span.bbox.y;

        for (i, table) in tables.iter().enumerate() {
            if let Some(ref bbox) = table.bbox {
                // Use generous tolerance for bbox overlap
                let tolerance = 2.0;
                if sx >= bbox.x - tolerance
                    && sx <= bbox.x + bbox.width + tolerance
                    && sy >= bbox.y - tolerance
                    && sy <= bbox.y + bbox.height + tolerance
                {
                    return Some(i);
                }
            }
        }
        None
    }

    /// Render an ExtractedTable as a markdown table string.
    fn render_table_markdown(table: &ExtractedTable) -> String {
        if table.rows.is_empty() {
            return String::new();
        }

        let mut output = String::new();

        // Determine header row index - use first row if has_header, or first is_header row
        let header_end = if table.has_header {
            table.rows.iter().position(|r| !r.is_header).unwrap_or(1)
        } else {
            // Treat first row as header for markdown (markdown requires a header row)
            1
        };

        for (row_idx, row) in table.rows.iter().enumerate() {
            output.push('|');
            for cell in &row.cells {
                output.push(' ');
                // Escape pipe characters in cell text
                let text = cell.text.replace('|', "\\|");
                let text = text.replace('\n', " ");
                output.push_str(text.trim());
                output.push(' ');
                // Handle colspan by adding extra | separators
                for _ in 1..cell.colspan {
                    output.push_str("| ");
                }
                output.push('|');
            }
            output.push('\n');

            // Add header separator after header rows
            if row_idx + 1 == header_end {
                output.push('|');
                for cell in &row.cells {
                    for _ in 0..cell.colspan {
                        output.push_str("---|");
                    }
                }
                output.push('\n');
            }
        }

        output
    }

    /// Core rendering logic shared between convert() and convert_with_tables().
    fn render_spans(
        &self,
        spans: &[OrderedTextSpan],
        tables: &[ExtractedTable],
        config: &TextPipelineConfig,
    ) -> Result<String> {
        if spans.is_empty() && tables.is_empty() {
            return Ok(String::new());
        }

        // Sort by reading order
        let mut sorted: Vec<_> = spans.iter().collect();
        sorted.sort_by_key(|s| s.reading_order);

        // Calculate base font size for heading detection.
        // Exclude spans < 9pt (bullet characters like ►, subscripts, footnotes)
        // from the median to prevent their small sizes from skewing heading
        // detection — e.g. many 8.8pt ► spans pulling the median down to 8.8pt,
        // causing all 11pt body text to look like headings (ratio 1.25).
        // If all spans are < 9pt (page dominated by small text), falls back to
        // 12pt default. The .max(8.0) is a safety floor for edge cases.
        let base_font_size = if config.output.detect_headings {
            let mut sizes_sorted: Vec<f32> = sorted
                .iter()
                .map(|s| s.span.font_size)
                .filter(|&s| s >= 9.0)
                .collect();
            sizes_sorted.sort_by(|a, b| crate::utils::safe_float_cmp(*a, *b));
            sizes_sorted
                .get(sizes_sorted.len() / 2)
                .copied()
                .unwrap_or(12.0)
                .max(8.0)
        } else {
            12.0
        };

        // Track which tables have been rendered
        let mut tables_rendered = vec![false; tables.len()];

        let mut result = String::new();
        let mut prev_span: Option<&OrderedTextSpan> = None;
        let mut current_line = String::new();

        for span in sorted.iter() {
            // Check if this span belongs to a table region
            if !tables.is_empty() {
                if let Some(table_idx) = self.span_in_table(span, tables) {
                    if !tables_rendered[table_idx] {
                        // Flush current line
                        if !current_line.is_empty() {
                            result.push_str(current_line.trim());
                            result.push_str("\n\n");
                            current_line.clear();
                        }

                        // Render the table
                        let table_md = Self::render_table_markdown(&tables[table_idx]);
                        result.push_str(&table_md);
                        result.push('\n');
                        tables_rendered[table_idx] = true;
                        prev_span = None;
                    }
                    // Skip this span (it's part of a table)
                    continue;
                }
            }

            // Check for paragraph break or line break
            let same_line = prev_span
                .map(|prev| (span.span.bbox.y - prev.span.bbox.y).abs() < span.span.font_size * 0.5)
                .unwrap_or(true);

            if let Some(prev) = prev_span {
                if self.is_paragraph_break(span, prev) {
                    if !current_line.is_empty() {
                        result.push_str(current_line.trim());
                        result.push_str("\n\n");
                        current_line.clear();
                    }
                } else if !same_line {
                    // Different visual line but within paragraph spacing.
                    // Check if a bullet item starts here — if so, start a new line.
                    let is_bullet = Self::is_bullet_span(&span.span.text)
                        || Self::starts_with_bullet(&span.span.text);
                    if is_bullet {
                        // Bullet on new line → flush current line and start list item
                        if !current_line.is_empty() {
                            result.push_str(current_line.trim());
                            result.push('\n');
                            current_line.clear();
                        }
                    } else if config.output.preserve_layout {
                        let spacing = (span.span.bbox.x - prev.span.bbox.x).max(0.0) as usize;
                        for _ in 0..spacing.min(20) {
                            current_line.push(' ');
                        }
                    } else {
                        current_line.push(' ');
                    }
                }
            }

            // Handle bullet character spans: replace with markdown list marker
            if Self::is_bullet_span(&span.span.text) {
                // Standalone bullet char span (e.g., "►" as its own span)
                // Replace with "- " prefix; text follows in next span(s)
                if same_line && !current_line.is_empty() && !current_line.ends_with("- ") {
                    // Bullet on same line as other content — preserve as-is
                    current_line.push_str(&span.span.text);
                } else if !current_line.ends_with("- ") {
                    current_line.push_str("- ");
                }
                prev_span = Some(span);
                continue;
            }

            // Handle inline bullets (text starts with bullet char)
            if Self::starts_with_bullet(&span.span.text) && (!same_line || prev_span.is_none()) {
                let stripped = Self::strip_bullet(&span.span.text);
                if !current_line.ends_with("- ") {
                    current_line.push_str("- ");
                }
                // Process the stripped text through normal formatting below
                // by re-assigning text variable
                let normalized_bullet;
                let mut text = stripped;
                if !config.output.preserve_layout {
                    normalized_bullet = self.normalize_whitespace(text);
                    text = &normalized_bullet;
                }
                let linkified = self.linkify(text);
                let is_bold = self.is_bold(span, config);
                let is_italic = self.is_italic(span);
                let formatted = self.apply_formatting(&linkified, is_bold, is_italic);
                current_line.push_str(&formatted);
                prev_span = Some(span);
                continue;
            }

            // Check for heading (take best level from absolute and ratio methods)
            if config.output.detect_headings {
                let level = match (
                    self.heading_level_absolute(span),
                    self.heading_level_ratio(span, base_font_size),
                ) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (a, b) => a.or(b),
                };

                if let Some(level) = level {
                    if !current_line.is_empty() {
                        result.push_str(current_line.trim());
                        result.push_str("\n\n");
                        current_line.clear();
                    }

                    let prefix = "#".repeat(level as usize);
                    result.push_str(&format!("{} {}\n\n", prefix, span.span.text.trim()));
                    prev_span = None;
                    continue;
                }
            }

            // Format text with bold/italic and apply linkification
            let mut text = span.span.text.as_str();

            let normalized;
            if !config.output.preserve_layout {
                // In PDFs, adjacent spans on the same line often have slightly
                // overlapping bboxes (negative horizontal gap) with the inter-span
                // whitespace encoded as leading/trailing spaces in the span text
                // itself.  normalize_whitespace collapses internal runs of spaces
                // but would also strip these boundary spaces, causing words from
                // neighbouring spans to merge (e.g. "visitwww.example.comto").
                // Preserve a leading space when a same-line predecessor exists and
                // a trailing space unconditionally so the next span can abut
                // correctly.  The plain-text converter avoids this problem by
                // skipping per-span normalization entirely.
                let had_leading_space =
                    same_line && prev_span.is_some() && text.starts_with(char::is_whitespace);
                let had_trailing_space = text.ends_with(char::is_whitespace);
                let mut norm = self.normalize_whitespace(text);
                if had_leading_space && !norm.starts_with(' ') {
                    norm.insert(0, ' ');
                }
                if had_trailing_space && !norm.ends_with(' ') && !norm.is_empty() {
                    norm.push(' ');
                }
                normalized = norm;
                text = &normalized;
            }

            let linkified = self.linkify(text);

            let is_bold = self.is_bold(span, config);
            let is_italic = self.is_italic(span);
            let formatted = self.apply_formatting(&linkified, is_bold, is_italic);

            current_line.push_str(&formatted);

            prev_span = Some(span);
        }

        // Render any tables that weren't matched to spans (e.g., all spans were in tables)
        for (i, table) in tables.iter().enumerate() {
            if !tables_rendered[i] && !table.is_empty() {
                if !current_line.is_empty() {
                    result.push_str(current_line.trim());
                    result.push_str("\n\n");
                    current_line.clear();
                }
                let table_md = Self::render_table_markdown(table);
                result.push_str(&table_md);
                result.push('\n');
            }
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
            let cleaned = result
                .split("\n\n")
                .map(|para| para.trim())
                .filter(|para| !para.is_empty())
                .collect::<Vec<_>>()
                .join("\n\n");

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
}

impl Default for MarkdownOutputConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputConverter for MarkdownOutputConverter {
    fn convert(&self, spans: &[OrderedTextSpan], config: &TextPipelineConfig) -> Result<String> {
        self.render_spans(spans, &[], config)
    }

    fn convert_with_tables(
        &self,
        spans: &[OrderedTextSpan],
        tables: &[ExtractedTable],
        config: &TextPipelineConfig,
    ) -> Result<String> {
        self.render_spans(spans, tables, config)
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
    use crate::structure::table_extractor::{TableCell, TableRow};

    fn make_span(
        text: &str,
        x: f32,
        y: f32,
        font_size: f32,
        weight: FontWeight,
    ) -> OrderedTextSpan {
        OrderedTextSpan::new(
            TextSpan {
                artifact_type: None,
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

    #[test]
    fn test_convert_with_tables_renders_markdown_table() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();

        let mut table = ExtractedTable::new();
        table.bbox = Some(Rect::new(10.0, 50.0, 200.0, 100.0));
        table.col_count = 2;
        table.has_header = true;

        let mut header = TableRow::new(true);
        header.add_cell(TableCell::new("Name".to_string(), true));
        header.add_cell(TableCell::new("Value".to_string(), true));
        table.add_row(header);

        let mut data = TableRow::new(false);
        data.add_cell(TableCell::new("A".to_string(), false));
        data.add_cell(TableCell::new("1".to_string(), false));
        table.add_row(data);

        let result = converter
            .convert_with_tables(&[], &[table], &config)
            .unwrap();

        assert!(result.contains("| Name |"));
        assert!(result.contains("| Value |"));
        assert!(result.contains("---|"));
        assert!(result.contains("| A |"));
        assert!(result.contains("| 1 |"));
    }

    // ============================================================================
    // render_table_markdown() tests
    // ============================================================================

    #[test]
    fn test_render_table_markdown_empty() {
        let table = ExtractedTable::new();
        let result = MarkdownOutputConverter::render_table_markdown(&table);
        assert_eq!(result, "");
    }

    #[test]
    fn test_render_table_markdown_single_row_no_header() {
        let mut table = ExtractedTable::new();
        let mut row = TableRow::new(false);
        row.add_cell(TableCell::new("A".to_string(), false));
        row.add_cell(TableCell::new("B".to_string(), false));
        table.add_row(row);

        let result = MarkdownOutputConverter::render_table_markdown(&table);
        assert!(result.contains("| A |"));
        assert!(result.contains("| B |"));
        // First row treated as header by default in markdown
        assert!(result.contains("---|"));
    }

    #[test]
    fn test_render_table_markdown_with_colspan() {
        let mut table = ExtractedTable::new();
        table.has_header = true;
        let mut header = TableRow::new(true);
        header.add_cell(TableCell::new("Wide".to_string(), true).with_colspan(2));
        table.add_row(header);

        let mut data = TableRow::new(false);
        data.add_cell(TableCell::new("Left".to_string(), false));
        data.add_cell(TableCell::new("Right".to_string(), false));
        table.add_row(data);

        let result = MarkdownOutputConverter::render_table_markdown(&table);
        // Colspan cell should produce extra | separators
        assert!(result.contains("| Wide |"));
        assert!(result.contains("---|---|"));
    }

    #[test]
    fn test_render_table_markdown_escapes_pipes() {
        let mut table = ExtractedTable::new();
        let mut row = TableRow::new(false);
        row.add_cell(TableCell::new("A|B".to_string(), false));
        table.add_row(row);

        let result = MarkdownOutputConverter::render_table_markdown(&table);
        assert!(result.contains("A\\|B"), "Pipes should be escaped: {}", result);
    }

    #[test]
    fn test_render_table_markdown_replaces_newlines() {
        let mut table = ExtractedTable::new();
        let mut row = TableRow::new(false);
        row.add_cell(TableCell::new("Line1\nLine2".to_string(), false));
        table.add_row(row);

        let result = MarkdownOutputConverter::render_table_markdown(&table);
        assert!(!result.contains("Line1\nLine2"), "Newlines in cells should be replaced");
        assert!(result.contains("Line1 Line2"));
    }

    #[test]
    fn test_render_table_markdown_trims_whitespace() {
        let mut table = ExtractedTable::new();
        let mut row = TableRow::new(false);
        row.add_cell(TableCell::new("  padded  ".to_string(), false));
        table.add_row(row);

        let result = MarkdownOutputConverter::render_table_markdown(&table);
        assert!(result.contains("| padded |"));
    }

    #[test]
    fn test_render_table_markdown_multiple_header_rows() {
        let mut table = ExtractedTable::new();
        table.has_header = true;

        let mut h1 = TableRow::new(true);
        h1.add_cell(TableCell::new("H1".to_string(), true));
        table.add_row(h1);

        let mut h2 = TableRow::new(true);
        h2.add_cell(TableCell::new("H2".to_string(), true));
        table.add_row(h2);

        let mut d1 = TableRow::new(false);
        d1.add_cell(TableCell::new("D1".to_string(), false));
        table.add_row(d1);

        let result = MarkdownOutputConverter::render_table_markdown(&table);
        // Separator should appear after last header row (row_idx == 1)
        let lines: Vec<&str> = result.lines().collect();
        assert_eq!(lines.len(), 4); // H1, H2, separator, D1
        assert!(lines[2].contains("---|"));
    }

    // ============================================================================
    // span_in_table() tests
    // ============================================================================

    #[test]
    fn test_span_in_table_match() {
        let converter = MarkdownOutputConverter::new();
        let span = make_span("text", 50.0, 70.0, 12.0, FontWeight::Normal);

        let mut table = ExtractedTable::new();
        table.bbox = Some(Rect::new(10.0, 50.0, 200.0, 100.0));

        assert_eq!(converter.span_in_table(&span, &[table]), Some(0));
    }

    #[test]
    fn test_span_in_table_no_match() {
        let converter = MarkdownOutputConverter::new();
        let span = make_span("text", 500.0, 500.0, 12.0, FontWeight::Normal);

        let mut table = ExtractedTable::new();
        table.bbox = Some(Rect::new(10.0, 50.0, 200.0, 100.0));

        assert_eq!(converter.span_in_table(&span, &[table]), None);
    }

    #[test]
    fn test_span_in_table_none_bbox() {
        let converter = MarkdownOutputConverter::new();
        let span = make_span("text", 50.0, 70.0, 12.0, FontWeight::Normal);

        let table = ExtractedTable::new(); // No bbox
        assert_eq!(converter.span_in_table(&span, &[table]), None);
    }

    #[test]
    fn test_span_in_table_tolerance() {
        let converter = MarkdownOutputConverter::new();
        // Span at bbox edge minus tolerance (2.0)
        let span = make_span("text", 8.5, 48.5, 12.0, FontWeight::Normal);

        let mut table = ExtractedTable::new();
        table.bbox = Some(Rect::new(10.0, 50.0, 200.0, 100.0));

        assert_eq!(
            converter.span_in_table(&span, &[table]),
            Some(0),
            "Should match within tolerance"
        );
    }

    #[test]
    fn test_span_in_table_multiple_tables() {
        let converter = MarkdownOutputConverter::new();
        let span = make_span("text", 350.0, 70.0, 12.0, FontWeight::Normal);

        let mut t1 = ExtractedTable::new();
        t1.bbox = Some(Rect::new(10.0, 50.0, 200.0, 100.0));

        let mut t2 = ExtractedTable::new();
        t2.bbox = Some(Rect::new(300.0, 50.0, 200.0, 100.0));

        assert_eq!(converter.span_in_table(&span, &[t1, t2]), Some(1));
    }

    // ============================================================================
    // convert_with_tables() integration tests
    // ============================================================================

    #[test]
    fn test_convert_with_tables_mixed_content() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();

        // Text before the table
        let mut span_before = make_span("Before table", 10.0, 200.0, 12.0, FontWeight::Normal);
        span_before.reading_order = 0;

        // Text after the table (lower Y = later in reading order)
        let mut span_after = make_span("After table", 10.0, 20.0, 12.0, FontWeight::Normal);
        span_after.reading_order = 2;

        // Text inside table region (should be excluded)
        let mut span_in_table = make_span("In table", 50.0, 70.0, 12.0, FontWeight::Normal);
        span_in_table.reading_order = 1;

        let mut table = ExtractedTable::new();
        table.bbox = Some(Rect::new(10.0, 50.0, 200.0, 100.0));
        table.has_header = true;
        let mut header = TableRow::new(true);
        header.add_cell(TableCell::new("Col".to_string(), true));
        table.add_row(header);
        let mut data = TableRow::new(false);
        data.add_cell(TableCell::new("Val".to_string(), false));
        table.add_row(data);

        let result = converter
            .convert_with_tables(&[span_before, span_in_table, span_after], &[table], &config)
            .unwrap();

        assert!(result.contains("Before table"), "Should contain text before table");
        assert!(result.contains("| Col |"), "Should contain table");
        assert!(result.contains("After table"), "Should contain text after table");
        assert!(!result.contains("In table"), "Should exclude span inside table region");
    }

    #[test]
    fn test_convert_with_tables_no_tables_is_same_as_convert() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        let spans = vec![make_span("Hello", 0.0, 100.0, 12.0, FontWeight::Normal)];

        let result_convert = converter.convert(&spans, &config).unwrap();
        let result_with_tables = converter.convert_with_tables(&spans, &[], &config).unwrap();

        assert_eq!(result_convert, result_with_tables);
    }

    #[test]
    fn test_convert_with_tables_multiple_tables() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();

        let make_table = |x: f32, text: &str| -> ExtractedTable {
            let mut t = ExtractedTable::new();
            t.bbox = Some(Rect::new(x, 50.0, 100.0, 50.0));
            let mut row = TableRow::new(false);
            row.add_cell(TableCell::new(text.to_string(), false));
            t.add_row(row);
            t
        };

        let result = converter
            .convert_with_tables(&[], &[make_table(10.0, "T1"), make_table(200.0, "T2")], &config)
            .unwrap();

        assert!(result.contains("| T1 |"), "Should contain first table");
        assert!(result.contains("| T2 |"), "Should contain second table");
    }

    // ============================================================================
    // Issue #182: Bullet detection tests
    // ============================================================================

    #[test]
    fn test_is_bullet_span() {
        assert!(MarkdownOutputConverter::is_bullet_span("►"));
        assert!(MarkdownOutputConverter::is_bullet_span("•"));
        assert!(MarkdownOutputConverter::is_bullet_span("▪"));
        assert!(MarkdownOutputConverter::is_bullet_span(" ► "));
        assert!(!MarkdownOutputConverter::is_bullet_span("text"));
        assert!(!MarkdownOutputConverter::is_bullet_span("►text"));
        assert!(!MarkdownOutputConverter::is_bullet_span(""));
    }

    #[test]
    fn test_starts_with_bullet() {
        assert!(MarkdownOutputConverter::starts_with_bullet("►text"));
        assert!(MarkdownOutputConverter::starts_with_bullet("• item"));
        assert!(MarkdownOutputConverter::starts_with_bullet("  ► indented"));
        assert!(!MarkdownOutputConverter::starts_with_bullet("text"));
        assert!(!MarkdownOutputConverter::starts_with_bullet(""));
    }

    #[test]
    fn test_strip_bullet() {
        assert_eq!(MarkdownOutputConverter::strip_bullet("► text"), "text");
        assert_eq!(MarkdownOutputConverter::strip_bullet("•item"), "item");
        assert_eq!(MarkdownOutputConverter::strip_bullet("no bullet"), "no bullet");
    }

    #[test]
    fn test_bullet_spans_become_list_items() {
        // Simulates: ► (separate span) + "Analog input" (next span, same Y)
        // on a new line from previous content
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();

        let mut title = make_span("FEATURES", 50.0, 660.0, 11.0, FontWeight::Bold);
        title.reading_order = 0;

        let mut bullet = make_span("►", 50.0, 640.0, 8.8, FontWeight::Normal);
        bullet.reading_order = 1;

        let mut text = make_span("Analog input", 60.0, 640.0, 11.0, FontWeight::Normal);
        text.reading_order = 2;

        let mut bullet2 = make_span("►", 50.0, 626.0, 8.8, FontWeight::Normal);
        bullet2.reading_order = 3;

        let mut text2 = make_span("16-bit ADC", 60.0, 626.0, 11.0, FontWeight::Normal);
        text2.reading_order = 4;

        let spans = vec![title, bullet, text, bullet2, text2];
        let result = converter.convert(&spans, &config).unwrap();

        assert!(
            result.contains("- Analog input"),
            "Should convert bullet to list item: {}",
            result
        );
        assert!(result.contains("- 16-bit ADC"), "Should convert second bullet: {}", result);
        assert!(!result.contains("►"), "Should not contain raw bullet character: {}", result);
    }

    #[test]
    fn test_inline_bullet_becomes_list_item() {
        // Simulates: "► Analog input" as a single span (inline bullet)
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();

        let mut title = make_span("TITLE", 50.0, 660.0, 11.0, FontWeight::Bold);
        title.reading_order = 0;

        let mut bullet_text = make_span("► Analog input", 50.0, 640.0, 11.0, FontWeight::Normal);
        bullet_text.reading_order = 1;

        let spans = vec![title, bullet_text];
        let result = converter.convert(&spans, &config).unwrap();

        assert!(
            result.contains("- Analog input"),
            "Should convert inline bullet to list item: {}",
            result
        );
    }

    #[test]
    fn test_first_span_inline_bullet() {
        // First span on page starts with bullet — no prev_span exists.
        // Should still be converted to a markdown list item.
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();

        let mut bullet_text = make_span("► First item", 50.0, 660.0, 11.0, FontWeight::Normal);
        bullet_text.reading_order = 0;

        let mut bullet_text2 = make_span("► Second item", 50.0, 646.0, 11.0, FontWeight::Normal);
        bullet_text2.reading_order = 1;

        let spans = vec![bullet_text, bullet_text2];
        let result = converter.convert(&spans, &config).unwrap();

        assert!(
            result.contains("- First item"),
            "First-span inline bullet should become list item: {}",
            result
        );
        assert!(
            result.contains("- Second item"),
            "Second inline bullet should become list item: {}",
            result
        );
    }

    // ============================================================================
    // Issue #182: Heading over-detection prevention
    // ============================================================================

    fn config_with_headings() -> TextPipelineConfig {
        let mut config = TextPipelineConfig::default();
        config.output.detect_headings = true;
        config
    }

    #[test]
    fn test_heading_base_font_excludes_small_spans() {
        // When page has many 8.8pt ► spans, the base font size should
        // still be ~11pt (excluding small spans), not 8.8pt
        let converter = MarkdownOutputConverter::new();
        let config = config_with_headings();

        let mut spans = Vec::new();
        let mut order = 0;

        // 10 bullet spans at 8.8pt (should be excluded from median)
        for i in 0..10 {
            let mut s = make_span("►", 50.0, 600.0 - (i as f32) * 14.0, 8.8, FontWeight::Normal);
            s.reading_order = order;
            order += 1;
            spans.push(s);
        }

        // 10 text spans at 11pt (should be the median)
        for i in 0..10 {
            let mut s = make_span(
                "body text content",
                60.0,
                600.0 - (i as f32) * 14.0,
                11.0,
                FontWeight::Bold,
            );
            s.reading_order = order;
            order += 1;
            spans.push(s);
        }

        let result = converter.convert(&spans, &config).unwrap();

        // "body text content" at 11pt should NOT be detected as heading
        // because base_font_size should be ~11pt (ratio 1.0)
        assert!(
            !result.contains("### body text content"),
            "11pt bold text should not be heading when base is 11pt: {}",
            result
        );
    }

    #[test]
    fn test_heading_detection_still_works_for_large_fonts() {
        let converter = MarkdownOutputConverter::new();
        let config = config_with_headings();

        let mut heading = make_span("BIG HEADING", 50.0, 100.0, 24.0, FontWeight::Bold);
        heading.reading_order = 0;

        let mut body = make_span("Body text", 50.0, 70.0, 11.0, FontWeight::Normal);
        body.reading_order = 1;

        let spans = vec![heading, body];
        let result = converter.convert(&spans, &config).unwrap();

        assert!(result.contains("# BIG HEADING"), "24pt text should be H1: {}", result);
    }
}
