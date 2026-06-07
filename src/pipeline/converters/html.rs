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
use crate::structure::table_extractor::Table;
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

    /// Detect if span should be a heading based on font size and content heuristics.
    ///
    /// A span is only promoted to a heading if it meets ALL of these criteria:
    /// - Font size is significantly larger than the base (median) font size
    /// - Text is short enough to be a heading (2-120 characters, ≤12 words)
    /// - Text does not look like non-heading content (addresses, currency, pure numbers, etc.)
    fn heading_level(&self, span: &OrderedTextSpan, base_font_size: f32) -> Option<u8> {
        let text = span.span.text.trim();
        let text_len = text.len();

        // Headings must be short but non-trivial (max ~12 words / 120 chars)
        if !(2..=120).contains(&text_len) {
            return None;
        }
        let word_count = text.split_whitespace().count();
        if word_count > 12 {
            return None;
        }

        // Reject content that looks like non-heading data
        if Self::looks_like_non_heading(text) {
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
        } else if size_ratio >= 1.3 || (is_bold && size_ratio >= 1.15) {
            Some(3)
        } else {
            None
        }
    }

    /// Check if text looks like non-heading content that should not be promoted
    /// to a heading tag regardless of font size.
    fn looks_like_non_heading(text: &str) -> bool {
        let trimmed = text.trim();

        // Currency amounts: $1,234.56 or 1,234.56$ or similar
        if trimmed.contains('$')
            || trimmed.contains('\u{20AC}') // euro
            || trimmed.contains('\u{00A3}')
        // pound
        {
            // If the text is mostly a currency value, reject it
            let non_currency: String = trimmed
                .chars()
                .filter(|c| {
                    !c.is_ascii_digit()
                        && *c != '.'
                        && *c != ','
                        && *c != '$'
                        && *c != ' '
                        && *c != '\u{20AC}'
                        && *c != '\u{00A3}'
                })
                .collect();
            if non_currency.len() <= 2 {
                return true;
            }
        }

        // Pure numbers or numbers with punctuation (e.g., "14", "3.5", "1,234")
        {
            let stripped: String = trimmed
                .chars()
                .filter(|c| !c.is_ascii_digit() && *c != '.' && *c != ',' && *c != ' ' && *c != '-')
                .collect();
            if stripped.is_empty() && !trimmed.is_empty() {
                return true;
            }
        }

        // Short "label + number" pattern common in forms (e.g. "Box 14",
        // "Ligne 23", "Feld 3", "Casilla 5"). Language-agnostic: two tokens
        // where the first is a short alphabetic word and the second is purely
        // numeric.
        {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() == 2
                && parts[0].chars().count() <= 10
                && parts[0].chars().all(|c| c.is_alphabetic())
                && parts[1]
                    .chars()
                    .all(|c| c.is_ascii_digit() || c == '.' || c == '-')
            {
                return true;
            }
        }

        // Street-address pattern: starts with a number followed by multiple
        // alphabetic words (e.g. "123 Main Street", "10 Rue de Rivoli",
        // "45 Calle Mayor"). Language-agnostic — matches Western-style
        // addresses where the street number precedes the street name.
        if let Some(first_char) = trimmed.chars().next() {
            if first_char.is_ascii_digit() {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if (3..=8).contains(&parts.len()) && trimmed.chars().count() < 80 {
                    let first_is_number = parts[0].chars().all(|c| c.is_ascii_digit() || c == '-');
                    let alpha_word_count = parts
                        .iter()
                        .skip(1)
                        .filter(|w| w.chars().any(|c| c.is_alphabetic()))
                        .count();
                    if first_is_number && alpha_word_count >= 2 {
                        return true;
                    }
                }
            }
        }

        false
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
        // Apply the same column-spanning-decimal / char_widths-boundary
        // split that the text extractor's `push_span_text` uses.  For
        // sailing-score PDFs (issue 487 nougat_018) the producer emits two
        // adjacent score cells as a single Tj like "1.10" with cw=[w].
        // Without this step, markdown / HTML keep them glued as `1.10`
        // instead of splitting into the two GT tokens `1` and `10`.
        let mut processed = String::new();
        let synthetic = crate::layout::TextSpan {
            text: text.to_string(),
            ..span.span.clone()
        };
        crate::document::PdfDocument::push_span_text(&mut processed, &synthetic);
        let escaped = Self::escape_html(&processed);

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
            self.convert_semantic_mode(spans, &[], config)
        }
    }

    fn convert_with_tables(
        &self,
        spans: &[OrderedTextSpan],
        tables: &[Table],
        config: &TextPipelineConfig,
    ) -> Result<String> {
        if config.output.preserve_layout {
            self.convert_layout_mode(spans, config)
        } else {
            self.convert_semantic_mode(spans, tables, config)
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
        tables: &[Table],
        config: &TextPipelineConfig,
    ) -> Result<String> {
        if spans.is_empty() && tables.is_empty() {
            return Ok(String::new());
        }

        // Sort by reading order
        let mut sorted: Vec<_> = spans.iter().collect();
        sorted.sort_by_key(|s| s.reading_order);

        // Calculate base font size for heading detection
        let base_font_size = if config.output.detect_headings {
            let sizes: Vec<f32> = sorted.iter().map(|s| s.span.font_size).collect();
            let mut sizes_sorted = sizes.clone();
            sizes_sorted.sort_by(|a, b| crate::utils::safe_float_cmp(*a, *b));
            sizes_sorted
                .get(sizes_sorted.len() / 2)
                .copied()
                .unwrap_or(12.0)
        } else {
            12.0
        };

        // Track which tables have been rendered
        let mut tables_rendered = vec![false; tables.len()];

        let mut result = String::new();
        let mut prev_span: Option<&OrderedTextSpan> = None;
        let mut in_paragraph = false;
        let mut current_content = String::new();

        for span in &sorted {
            // Check if span is in a table region
            if !tables.is_empty() {
                if let Some(table_idx) = super::span_in_table(span, tables) {
                    if !tables_rendered[table_idx] {
                        // Close any open paragraph
                        if in_paragraph && !current_content.is_empty() {
                            result.push_str(&format!("<p>{}</p>\n", current_content.trim()));
                            current_content.clear();
                            in_paragraph = false;
                        }

                        // Render the table
                        result.push_str(&Self::render_table_html(&tables[table_idx]));
                        tables_rendered[table_idx] = true;
                        prev_span = None;
                    }
                    continue;
                }
            }

            // Check for paragraph break
            if let Some(prev) = prev_span {
                if self.is_paragraph_break(span, prev)
                    && in_paragraph
                    && !current_content.is_empty()
                {
                    result.push_str(&format!("<p>{}</p>\n", current_content.trim()));
                    current_content.clear();
                    in_paragraph = false;
                }
            }

            // Check for heading
            if config.output.detect_headings {
                if let Some(level) = self.heading_level(span, base_font_size) {
                    if in_paragraph && !current_content.is_empty() {
                        result.push_str(&format!("<p>{}</p>\n", current_content.trim()));
                        current_content.clear();
                        in_paragraph = false;
                    }

                    let text = self.format_span_with_styles(span, span.span.text.trim());
                    result.push_str(&format!("<h{}>{}</h{}>\n", level, text, level));
                    prev_span = Some(span);
                    continue;
                }
            }

            if !in_paragraph {
                in_paragraph = true;
            }

            // Insert a space when adjacent spans should be separated:
            //   1. Same-line spans with a meaningful horizontal gap
            //      (prevents label+value concatenation like "Subtotal$500.00").
            //   2. Different-line spans within the same paragraph (multi-line
            //      column headers, e.g. "Inpatient" / "Bed" stacked across two
            //      visual lines — issue 487 nougat_026).  Without this, the two
            //      tokens come out as "InpatientBed" because the same_line gate
            //      above skips the space-insertion check whenever y_diff >
            //      0.5 × font_size.
            if let Some(prev) = prev_span {
                let y_diff = (span.span.bbox.y - prev.span.bbox.y).abs();
                let same_line = y_diff < span.span.font_size * 0.5;
                let need_space_between_lines = !same_line
                    && y_diff > 0.0
                    && !current_content.is_empty()
                    && !current_content.ends_with(' ')
                    && !current_content.ends_with('\n')
                    && !span.span.text.starts_with(' ');
                let need_space_same_line = same_line
                    && !current_content.is_empty()
                    && !current_content.ends_with(' ')
                    && !span.span.text.starts_with(' ')
                    && super::has_horizontal_gap(&prev.span, &span.span);
                if need_space_same_line || need_space_between_lines {
                    current_content.push(' ');
                }
            }

            let formatted = self.format_span_with_styles(span, &span.span.text);
            current_content.push_str(&formatted);

            prev_span = Some(span);
        }

        // Render any tables that weren't matched to spans
        for (i, table) in tables.iter().enumerate() {
            if !tables_rendered[i] && !table.is_empty() {
                if in_paragraph && !current_content.is_empty() {
                    result.push_str(&format!("<p>{}</p>\n", current_content.trim()));
                    current_content.clear();
                    in_paragraph = false;
                }
                result.push_str(&Self::render_table_html(table));
            }
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

    /// Render the text content of a single table cell as HTML.
    ///
    /// When the cell has `spans`, this walks them in order — mirroring the
    /// `render_table_markdown` path — so that:
    /// - Adjacent spans with a meaningful horizontal gap get a space between them
    ///   (prevents "Label$500.00"-style concatenation).
    /// - Bold spans are wrapped in `<strong>`, italic spans in `<em>`.
    ///
    /// When `spans` is empty the function falls back to `cell.text`.
    fn render_cell_html(cell: &crate::structure::table_extractor::TableCell) -> String {
        use crate::layout::FontWeight;

        if cell.spans.is_empty() {
            // Fallback: no span metadata available — use the pre-built text field.
            return Self::escape_html(cell.text.trim());
        }

        let mut out = String::new();

        for (i, span) in cell.spans.iter().enumerate() {
            let is_bold = matches!(
                span.font_weight,
                FontWeight::Bold | FontWeight::Black | FontWeight::ExtraBold | FontWeight::SemiBold
            );
            let is_italic = span.is_italic;

            // Insert a space when adjacent same-row spans have a meaningful
            // horizontal gap (mirrors the body-span logic in convert_semantic_mode
            // and the span-gap logic in render_table_markdown).
            if i > 0 {
                let prev = &cell.spans[i - 1];
                let has_gap = super::has_horizontal_gap(prev, span);
                let already_has_space = out.ends_with(' ') || span.text.starts_with(' ');
                if has_gap && !already_has_space {
                    out.push(' ');
                }
            }

            // Apply column-spanning-decimal split (issue 487 nougat_018):
            // sailing-score cells emitted as "1.10" with sparse char_widths
            // split into two tokens "1 10".
            let mut processed = String::new();
            crate::document::PdfDocument::push_span_text(&mut processed, span);
            let escaped = Self::escape_html(&processed);

            let styled = match (is_bold, is_italic) {
                (true, true) => format!("<strong><em>{}</em></strong>", escaped),
                (true, false) => format!("<strong>{}</strong>", escaped),
                (false, true) => format!("<em>{}</em>", escaped),
                (false, false) => escaped,
            };
            out.push_str(&styled);
        }

        out
    }

    /// Render a Table as an HTML table string.
    fn render_table_html(table: &Table) -> String {
        if table.rows.is_empty() {
            return String::new();
        }

        let mut html = String::from("<table>\n");

        // Determine header/body sections
        let has_header = table.has_header || table.rows.first().is_some_and(|r| r.is_header);
        let header_end = if has_header {
            table
                .rows
                .iter()
                .position(|r| !r.is_header)
                .unwrap_or(table.rows.len())
        } else {
            0
        };

        // Render header rows
        if header_end > 0 {
            html.push_str("<thead>\n");
            for row in &table.rows[..header_end] {
                html.push_str("<tr>");
                for cell in &row.cells {
                    let mut attrs = String::new();
                    if cell.colspan > 1 {
                        attrs.push_str(&format!(" colspan=\"{}\"", cell.colspan));
                    }
                    if cell.rowspan > 1 {
                        attrs.push_str(&format!(" rowspan=\"{}\"", cell.rowspan));
                    }
                    let text = Self::render_cell_html(cell);
                    html.push_str(&format!("<th{}>{}</th>", attrs, text));
                }
                html.push_str("</tr>\n");
            }
            html.push_str("</thead>\n");
        }

        // Render body rows
        let body_rows = &table.rows[header_end..];
        if !body_rows.is_empty() {
            html.push_str("<tbody>\n");
            for row in body_rows {
                html.push_str("<tr>");
                for cell in &row.cells {
                    let mut attrs = String::new();
                    if cell.colspan > 1 {
                        attrs.push_str(&format!(" colspan=\"{}\"", cell.colspan));
                    }
                    if cell.rowspan > 1 {
                        attrs.push_str(&format!(" rowspan=\"{}\"", cell.rowspan));
                    }
                    let text = Self::render_cell_html(cell);
                    html.push_str(&format!("<td{}>{}</td>", attrs, text));
                }
                html.push_str("</tr>\n");
            }
            html.push_str("</tbody>\n");
        }

        html.push_str("</table>\n");
        html
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Rect;
    use crate::layout::{Color, TextSpan};
    use crate::pipeline::converters::span_in_table;

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
                is_monospace: false,
                color: Color::black(),
                mcid: None,
                mcid_scope: None,
                sequence: 0,
                offset_semantic: false,
                split_boundary_before: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
                heading_level: None,
                rotation_degrees: 0.0,
                wmode: 0,
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

    // ============================================================================
    // render_table_html() tests
    // ============================================================================

    use crate::structure::table_extractor::{TableCell, TableRow};

    #[test]
    fn test_render_table_html_empty() {
        let table = Table::new();
        let result = HtmlOutputConverter::render_table_html(&table);
        assert_eq!(result, "");
    }

    #[test]
    fn test_render_table_html_basic() {
        let mut table = Table::new();
        table.has_header = true;

        let mut header = TableRow::new(true);
        header.add_cell(TableCell::new("Name".to_string(), true));
        header.add_cell(TableCell::new("Age".to_string(), true));
        table.add_row(header);

        let mut data = TableRow::new(false);
        data.add_cell(TableCell::new("Alice".to_string(), false));
        data.add_cell(TableCell::new("30".to_string(), false));
        table.add_row(data);

        let result = HtmlOutputConverter::render_table_html(&table);
        assert!(result.contains("<table>"));
        assert!(result.contains("</table>"));
        assert!(result.contains("<thead>"));
        assert!(result.contains("</thead>"));
        assert!(result.contains("<tbody>"));
        assert!(result.contains("</tbody>"));
        assert!(result.contains("<th>Name</th>"));
        assert!(result.contains("<th>Age</th>"));
        assert!(result.contains("<td>Alice</td>"));
        assert!(result.contains("<td>30</td>"));
    }

    #[test]
    fn test_render_table_html_no_header() {
        let mut table = Table::new();

        let mut row = TableRow::new(false);
        row.add_cell(TableCell::new("A".to_string(), false));
        table.add_row(row);

        let result = HtmlOutputConverter::render_table_html(&table);
        assert!(result.contains("<table>"));
        assert!(!result.contains("<thead>"), "Should not have thead when no header");
        assert!(result.contains("<tbody>"));
        assert!(result.contains("<td>A</td>"));
    }

    #[test]
    fn test_render_table_html_colspan() {
        let mut table = Table::new();
        let mut row = TableRow::new(false);
        row.add_cell(TableCell::new("Wide".to_string(), false).with_colspan(3));
        table.add_row(row);

        let result = HtmlOutputConverter::render_table_html(&table);
        assert!(result.contains("colspan=\"3\""), "Should have colspan attribute: {}", result);
    }

    #[test]
    fn test_render_table_html_rowspan() {
        let mut table = Table::new();
        let mut row = TableRow::new(false);
        row.add_cell(TableCell::new("Tall".to_string(), false).with_rowspan(2));
        table.add_row(row);

        let result = HtmlOutputConverter::render_table_html(&table);
        assert!(result.contains("rowspan=\"2\""), "Should have rowspan attribute: {}", result);
    }

    #[test]
    fn test_render_table_html_escapes_content() {
        let mut table = Table::new();
        let mut row = TableRow::new(false);
        row.add_cell(TableCell::new("<b>bold</b>".to_string(), false));
        row.add_cell(TableCell::new("A & B".to_string(), false));
        table.add_row(row);

        let result = HtmlOutputConverter::render_table_html(&table);
        assert!(result.contains("&lt;b&gt;bold&lt;/b&gt;"), "HTML should be escaped: {}", result);
        assert!(result.contains("A &amp; B"), "Ampersand should be escaped: {}", result);
        assert!(!result.contains("<b>bold</b>"), "Raw HTML should not appear");
    }

    #[test]
    fn test_render_table_html_all_header_rows() {
        let mut table = Table::new();
        table.has_header = true;

        let mut h1 = TableRow::new(true);
        h1.add_cell(TableCell::new("H1".to_string(), true));
        table.add_row(h1);

        let mut h2 = TableRow::new(true);
        h2.add_cell(TableCell::new("H2".to_string(), true));
        table.add_row(h2);

        let result = HtmlOutputConverter::render_table_html(&table);
        assert!(result.contains("<thead>"));
        assert!(result.contains("<th>H1</th>"));
        assert!(result.contains("<th>H2</th>"));
        // No tbody when all rows are headers
        assert!(!result.contains("<tbody>"));
    }

    // ============================================================================
    // convert_with_tables() tests
    // ============================================================================

    #[test]
    fn test_convert_with_tables_renders_html_table() {
        let converter = HtmlOutputConverter::new();
        let config = TextPipelineConfig::default();

        let mut table = Table::new();
        table.bbox = Some(Rect::new(10.0, 50.0, 200.0, 100.0));
        table.has_header = true;

        let mut header = TableRow::new(true);
        header.add_cell(TableCell::new("X".to_string(), true));
        table.add_row(header);

        let mut data = TableRow::new(false);
        data.add_cell(TableCell::new("Y".to_string(), false));
        table.add_row(data);

        let result = converter
            .convert_with_tables(&[], &[table], &config)
            .unwrap();

        assert!(result.contains("<table>"), "Should contain HTML table: {}", result);
        assert!(result.contains("<th>X</th>"));
        assert!(result.contains("<td>Y</td>"));
    }

    #[test]
    fn test_convert_with_tables_mixed_content() {
        let converter = HtmlOutputConverter::new();
        let config = TextPipelineConfig::default();

        let mut span_before = make_span("Intro", 10.0, 200.0, 12.0, FontWeight::Normal);
        span_before.reading_order = 0;

        let mut span_in_table = make_span("Inside", 50.0, 70.0, 12.0, FontWeight::Normal);
        span_in_table.reading_order = 1;

        let mut table = Table::new();
        table.bbox = Some(Rect::new(10.0, 50.0, 200.0, 100.0));
        let mut row = TableRow::new(false);
        row.add_cell(TableCell::new("Cell".to_string(), false));
        table.add_row(row);

        let result = converter
            .convert_with_tables(&[span_before, span_in_table], &[table], &config)
            .unwrap();

        assert!(result.contains("<p>Intro</p>"), "Should contain paragraph: {}", result);
        assert!(result.contains("<table>"), "Should contain table: {}", result);
        assert!(!result.contains("Inside"), "Should exclude span in table region");
    }

    #[test]
    fn test_convert_with_tables_no_tables_same_as_convert() {
        let converter = HtmlOutputConverter::new();
        let config = TextPipelineConfig::default();
        let spans = vec![make_span("Hello", 0.0, 100.0, 12.0, FontWeight::Normal)];

        let result_convert = converter.convert(&spans, &config).unwrap();
        let result_with_tables = converter.convert_with_tables(&spans, &[], &config).unwrap();

        assert_eq!(result_convert, result_with_tables);
    }

    #[test]
    fn test_heading_not_assigned_to_non_heading_content() {
        // Addresses, box numbers, currency amounts, and long text should NOT be headings
        // even when their font size is large relative to the base size.
        let converter = HtmlOutputConverter::new();
        let mut config = TextPipelineConfig::default();
        config.output.detect_headings = true;

        // Many body text spans at 10pt to establish a clear 10pt median
        let mut body1 = make_span("Gross revenue", 10.0, 200.0, 10.0, FontWeight::Normal);
        body1.reading_order = 4;
        let mut body2 = make_span("Operating expenses", 10.0, 220.0, 10.0, FontWeight::Normal);
        body2.reading_order = 5;
        let mut body3 = make_span("Net income", 10.0, 240.0, 10.0, FontWeight::Normal);
        body3.reading_order = 6;
        let mut body4 = make_span("Interest paid", 10.0, 260.0, 10.0, FontWeight::Normal);
        body4.reading_order = 7;
        let mut body5 = make_span("Depreciation", 10.0, 280.0, 10.0, FontWeight::Normal);
        body5.reading_order = 8;

        // Address at 24pt — large font but NOT a heading (it's an address)
        let mut address = make_span("123 Main Street", 10.0, 20.0, 24.0, FontWeight::Normal);
        address.reading_order = 0;

        // Box/form label at 20pt — NOT a heading (it's a form box number)
        let mut box_label = make_span("Box 14", 10.0, 60.0, 20.0, FontWeight::Normal);
        box_label.reading_order = 1;

        // Currency amount at 24pt — NOT a heading
        let mut amount = make_span("$65,700.00", 10.0, 100.0, 24.0, FontWeight::Normal);
        amount.reading_order = 2;

        // Long text at 24pt — NOT a heading (too long to be a heading)
        let mut long_text = make_span(
            "This is a very long paragraph of text that goes on and on and contains many words and should never be classified as a heading because headings are short descriptive labels",
            10.0, 140.0, 24.0, FontWeight::Normal,
        );
        long_text.reading_order = 3;

        let spans = vec![
            address, box_label, amount, long_text, body1, body2, body3, body4, body5,
        ];
        let result = converter
            .convert_semantic_mode(&spans, &[], &config)
            .unwrap();

        // None of these should be in heading tags
        assert!(!result.contains("<h1>123 Main Street"), "Address should not be h1: {}", result);
        assert!(
            !result.contains("<h2>Box 14") && !result.contains("<h1>Box 14"),
            "Box label should not be a heading: {}",
            result
        );
        assert!(
            !result.contains("<h1>$65,700.00") && !result.contains("<h2>$65,700.00"),
            "Currency amount should not be a heading: {}",
            result
        );
        assert!(
            !result.contains("<h1>This is a very long"),
            "Long text should not be a heading: {}",
            result
        );

        // All content should be in paragraph tags
        assert!(result.contains("<p>"), "Content should be in <p> tags: {}", result);
    }

    #[test]
    fn test_heading_assigned_to_real_headings() {
        // Genuine headings: short, descriptive, larger font, with enough body text
        // to establish a clear base font size.
        let converter = HtmlOutputConverter::new();
        let mut config = TextPipelineConfig::default();
        config.output.detect_headings = true;

        let mut heading = make_span("Introduction", 10.0, 20.0, 24.0, FontWeight::Bold);
        heading.reading_order = 0;

        let mut body1 = make_span(
            "This is the body text of the document.",
            10.0,
            60.0,
            10.0,
            FontWeight::Normal,
        );
        body1.reading_order = 1;
        let mut body2 =
            make_span("More body text follows here.", 10.0, 80.0, 10.0, FontWeight::Normal);
        body2.reading_order = 2;
        let mut body3 = make_span("And even more content.", 10.0, 100.0, 10.0, FontWeight::Normal);
        body3.reading_order = 3;

        let spans = vec![heading, body1, body2, body3];
        let result = converter
            .convert_semantic_mode(&spans, &[], &config)
            .unwrap();

        // "Introduction" should be a heading
        assert!(
            result.contains("<h1>") || result.contains("<h2>") || result.contains("<h3>"),
            "Real heading should be detected: {}",
            result
        );
        assert!(result.contains("Introduction"), "Heading text should appear: {}", result);
    }

    #[test]
    fn test_span_in_table_html() {
        let mut table = Table::new();
        table.bbox = Some(Rect::new(10.0, 50.0, 200.0, 100.0));

        let inside = make_span("inside", 50.0, 70.0, 12.0, FontWeight::Normal);
        let outside = make_span("outside", 500.0, 500.0, 12.0, FontWeight::Normal);

        assert_eq!(span_in_table(&inside, &[table.clone()]), Some(0));
        assert_eq!(span_in_table(&outside, &[table]), None);
    }

    // ============================================================================
    // render_cell_html() tests — span-walking path (#487)
    // ============================================================================

    /// Build a raw TextSpan (not OrderedTextSpan) for use in TableCell.spans.
    fn make_raw_span(
        text: &str,
        x: f32,
        y: f32,
        width: f32,
        font_size: f32,
        weight: FontWeight,
        italic: bool,
    ) -> TextSpan {
        TextSpan {
            artifact_type: None,
            text: text.to_string(),
            bbox: Rect::new(x, y, width, font_size),
            font_name: "Test".to_string(),
            font_size,
            font_weight: weight,
            is_italic: italic,
            is_monospace: false,
            color: Color::black(),
            mcid: None,
            mcid_scope: None,
            sequence: 0,
            offset_semantic: false,
            split_boundary_before: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 100.0,
            primary_detected: false,
            char_widths: vec![],
            heading_level: None,
            rotation_degrees: 0.0,
            wmode: 0,
        }
    }

    #[test]
    fn test_render_cell_html_fallback_to_text_when_no_spans() {
        // When spans is empty the function returns escaped cell.text (trimmed).
        use crate::structure::table_extractor::TableCell;
        let cell = TableCell::new("  hello world  ".to_string(), false);
        let result = HtmlOutputConverter::render_cell_html(&cell);
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_render_cell_html_fallback_escapes_html() {
        use crate::structure::table_extractor::TableCell;
        let cell = TableCell::new("<b>bold</b> & more".to_string(), false);
        let result = HtmlOutputConverter::render_cell_html(&cell);
        assert_eq!(result, "&lt;b&gt;bold&lt;/b&gt; &amp; more");
    }

    #[test]
    fn test_render_cell_html_plain_spans() {
        // Two adjacent normal spans with no gap → concatenated without extra space.
        use crate::structure::table_extractor::TableCell;
        let mut cell = TableCell::new(String::new(), false);
        // Place spans directly adjacent: span1 ends at x=50, span2 starts at x=50.
        cell.spans
            .push(make_raw_span("hello", 0.0, 0.0, 50.0, 12.0, FontWeight::Normal, false));
        cell.spans
            .push(make_raw_span(" world", 50.0, 0.0, 50.0, 12.0, FontWeight::Normal, false));
        let result = HtmlOutputConverter::render_cell_html(&cell);
        // No gap inserted since span2 already starts with a space.
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_render_cell_html_bold_span() {
        use crate::structure::table_extractor::TableCell;
        let mut cell = TableCell::new(String::new(), false);
        cell.spans
            .push(make_raw_span("Total", 0.0, 0.0, 30.0, 12.0, FontWeight::Bold, false));
        let result = HtmlOutputConverter::render_cell_html(&cell);
        assert_eq!(result, "<strong>Total</strong>");
    }

    #[test]
    fn test_render_cell_html_italic_span() {
        use crate::structure::table_extractor::TableCell;
        let mut cell = TableCell::new(String::new(), false);
        cell.spans
            .push(make_raw_span("Note", 0.0, 0.0, 25.0, 12.0, FontWeight::Normal, true));
        let result = HtmlOutputConverter::render_cell_html(&cell);
        assert_eq!(result, "<em>Note</em>");
    }

    #[test]
    fn test_render_cell_html_bold_italic_span() {
        use crate::structure::table_extractor::TableCell;
        let mut cell = TableCell::new(String::new(), false);
        cell.spans
            .push(make_raw_span("Warn", 0.0, 0.0, 25.0, 12.0, FontWeight::Bold, true));
        let result = HtmlOutputConverter::render_cell_html(&cell);
        assert_eq!(result, "<strong><em>Warn</em></strong>");
    }

    #[test]
    fn test_render_cell_html_gap_inserts_space() {
        // Span1 ends at x=30, span2 starts at x=35. Gap=5 > 12*0.15=1.8 → space inserted.
        use crate::structure::table_extractor::TableCell;
        let mut cell = TableCell::new(String::new(), false);
        cell.spans
            .push(make_raw_span("Label", 0.0, 0.0, 30.0, 12.0, FontWeight::Normal, false));
        cell.spans
            .push(make_raw_span("Value", 35.0, 0.0, 30.0, 12.0, FontWeight::Normal, false));
        let result = HtmlOutputConverter::render_cell_html(&cell);
        assert_eq!(result, "Label Value", "Gap should produce a space: {}", result);
    }

    #[test]
    fn test_render_cell_html_no_gap_no_space() {
        // Span1 ends at x=30, span2 starts at x=30. Gap=0 → no space inserted.
        use crate::structure::table_extractor::TableCell;
        let mut cell = TableCell::new(String::new(), false);
        cell.spans
            .push(make_raw_span("foo", 0.0, 0.0, 30.0, 12.0, FontWeight::Normal, false));
        cell.spans
            .push(make_raw_span("bar", 30.0, 0.0, 30.0, 12.0, FontWeight::Normal, false));
        let result = HtmlOutputConverter::render_cell_html(&cell);
        assert_eq!(result, "foobar", "No gap should produce no space: {}", result);
    }

    #[test]
    fn test_render_cell_html_mixed_bold_and_plain_with_gap() {
        // A bold label with a gap before a plain value — matches real table cell pattern.
        use crate::structure::table_extractor::TableCell;
        let mut cell = TableCell::new(String::new(), false);
        cell.spans
            .push(make_raw_span("Subtotal", 0.0, 0.0, 40.0, 12.0, FontWeight::Bold, false));
        cell.spans
            .push(make_raw_span("500.00", 50.0, 0.0, 35.0, 12.0, FontWeight::Normal, false));
        let result = HtmlOutputConverter::render_cell_html(&cell);
        assert_eq!(result, "<strong>Subtotal</strong> 500.00", "Result: {}", result);
    }

    #[test]
    fn test_render_table_html_uses_spans_for_bold() {
        // End-to-end: a table whose cell has a bold span should emit <strong>.
        use crate::structure::table_extractor::{TableCell, TableRow};
        let mut table = Table::new();
        let mut row = TableRow::new(false);
        let mut cell = TableCell::new(String::new(), false);
        cell.spans
            .push(make_raw_span("Total", 0.0, 0.0, 30.0, 12.0, FontWeight::Bold, false));
        row.add_cell(cell);
        table.add_row(row);

        let result = HtmlOutputConverter::render_table_html(&table);
        assert!(
            result.contains("<td><strong>Total</strong></td>"),
            "Should render bold cell via spans: {}",
            result
        );
    }
}
