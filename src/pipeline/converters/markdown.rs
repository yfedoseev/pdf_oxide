//! Markdown output converter.
//!
//! Converts ordered text spans to Markdown format.

use crate::error::Result;
use crate::layout::FontWeight;
use crate::pipeline::{OrderedTextSpan, StructRole, TextPipelineConfig};
use crate::structure::table_extractor::Table;
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
    ///
    /// Two break signals:
    /// 1. Vertical gap larger than `paragraph_gap_ratio × line_height`
    ///    (the classic geometric heuristic).
    /// 2. The current line begins with a list marker (bullet glyph or
    ///    ordered marker) while the previous line did not — list-items
    ///    must always start a fresh paragraph regardless of how tightly
    ///    they sit under the preceding paragraph (issue #377 D4: many
    ///    untagged docs use a sub-1.5× line gap before lists, which
    ///    glues the first item to the intro sentence).
    fn is_paragraph_break(&self, current: &OrderedTextSpan, previous: &OrderedTextSpan) -> bool {
        let line_height = current.span.font_size.max(previous.span.font_size);
        let gap = (previous.span.bbox.y - current.span.bbox.y).abs();
        if gap > line_height * self.paragraph_gap_ratio {
            return true;
        }
        // List-prefix transition guard. Bullet glyph or `1.` / `a)` /
        // `i.` ordered marker at the start of the current line, with
        // the previous line on a different baseline and not itself a
        // list item. The ordered-marker detection is conservative
        // (single digit/letter at line start) so figure captions
        // ("1.1 Foo") and years ("1986") are not promoted to lists.
        let line_changed =
            (previous.span.bbox.y - current.span.bbox.y).abs() > current.span.font_size * 0.5;
        if line_changed {
            let cur_text = current.span.text.trim_start();
            let cur_starts_list = Self::is_bullet_span(cur_text)
                || Self::starts_with_bullet(cur_text)
                || Self::is_ordered_list_marker(cur_text).is_some();
            let prev_text = previous.span.text.trim_start();
            let prev_starts_list = Self::is_bullet_span(prev_text)
                || Self::starts_with_bullet(prev_text)
                || Self::is_ordered_list_marker(prev_text).is_some();
            if cur_starts_list && !prev_starts_list {
                return true;
            }
        }
        false
    }

    /// Detect a markdown ordered-list marker at the start of `text`.
    /// Recognises `1.`, `12.`, `a.`, `iv.`, `1)`, `a)` followed by a
    /// space. Returns the (1-based) position number when known
    /// (Roman numerals coerced to position 1 for now), or `None`.
    ///
    /// Conservative on purpose — only single digit/letter tokens at
    /// the very start of the trimmed text qualify, so figure captions
    /// like "1.1 Foo" and years like "1986" are not falsely promoted
    /// to numbered lists. See issue #377 D3.
    fn is_ordered_list_marker(text: &str) -> Option<u32> {
        let t = text.trim_start();
        let bytes = t.as_bytes();
        if bytes.is_empty() {
            return None;
        }
        // Find the marker token (digits, single ASCII letter, or short
        // roman numeral) and the trailing punctuation `.` or `)`.
        let mut idx = 0;
        // Numeric form: `\d{1,3}`.
        while idx < bytes.len() && bytes[idx].is_ascii_digit() && idx < 3 {
            idx += 1;
        }
        let numeric_n = if idx > 0 {
            std::str::from_utf8(&bytes[..idx])
                .ok()
                .and_then(|s| s.parse::<u32>().ok())
        } else {
            None
        };
        // Single ASCII letter form (a) / b. / I.).
        if idx == 0 && bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() {
            // Roman numerals up to 4 chars (i, ii, iii, iv).
            let mut roman_end = 0;
            while roman_end < bytes.len().min(4)
                && matches!(bytes[roman_end], b'i' | b'v' | b'x' | b'I' | b'V' | b'X')
            {
                roman_end += 1;
            }
            if roman_end >= 1 && bytes.len() > roman_end {
                let punct = bytes[roman_end];
                if matches!(punct, b'.' | b')') && bytes.get(roman_end + 1).copied() == Some(b' ') {
                    return Some(1); // unknown roman position
                }
            }
            // Single letter: a) Foo, A. Bar.
            if bytes.len() >= 3
                && matches!(bytes[1], b'.' | b')')
                && bytes[2] == b' '
                && bytes[0].is_ascii_alphabetic()
            {
                return Some(1);
            }
            return None;
        }
        // For the numeric branch, check trailing `.` / `)` and a space.
        if idx > 0 && bytes.len() > idx {
            let punct = bytes[idx];
            if matches!(punct, b'.' | b')') && bytes.get(idx + 1).copied() == Some(b' ') {
                return numeric_n;
            }
        }
        None
    }

    /// Check if a span consists of a single bullet character.
    ///
    /// Common bullet characters used in PDF documents:
    /// ► • ▪ ▸ ‣ ◦ ● ■ ◆ ○ □ ❍ ❖ ✓ ✔ ➢ ➤ 
    fn is_bullet_span(text: &str) -> bool {
        let t = text.trim();
        matches!(
            t,
            "►" | "•"
                | "▪"
                | "▸"
                | "‣"
                | "◦"
                | "●"
                | "■"
                | "◆"
                | "○"
                | "□"
                | "❍"
                | "❖"
                | "✓"
                | "✔"
                | "➢"
                | "➤"
                | "\x7f"
        )
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
            || t.starts_with('❍')
            || t.starts_with('❖')
            || t.starts_with('✓')
            || t.starts_with('✔')
            || t.starts_with('➢')
            || t.starts_with('➤')
            || t.starts_with('\x7f')
    }

    /// Validate that a string looks like a heading (not a paragraph or noise).
    ///
    /// Content-based guards only — no language/locale-specific keyword lists.
    fn is_valid_heading_text(text: &str) -> bool {
        let trimmed = text.trim();
        let text_len = trimmed.chars().count();
        // Headings must be non-trivial but also not full paragraphs.
        // 200 chars is ~35 words, which safely accommodates long wrapped titles
        // while excluding paragraph-length runs that share a larger font.
        if !(2..=200).contains(&text_len) {
            return false;
        }
        // Sentence-length guards: a heading rarely exceeds 20 words and
        // almost never contains a full stop followed by more text (that's
        // a paragraph, even if it happens to be set in a larger font).
        let word_count = trimmed.split_whitespace().count();
        if word_count > 20 {
            return false;
        }
        // Exclude runs with mid-sentence punctuation ("foo. Bar baz") —
        // real headings don't contain sentence boundaries.
        let bytes = trimmed.as_bytes();
        for i in 0..bytes.len().saturating_sub(2) {
            if bytes[i] == b'.' && bytes[i + 1] == b' ' {
                let next = bytes[i + 2];
                if next.is_ascii_alphabetic() {
                    return false;
                }
            }
        }

        // Reject if dominated by digits/punctuation (KPI numbers, page numbers,
        // "$100", "23.5K"). Require a minimum alphabetic ratio that scales:
        // very short strings need at least 2 letters; longer strings need
        // >=30% alphabetic characters.
        let alpha_count = trimmed.chars().filter(|c| c.is_alphabetic()).count();
        if text_len <= 8 {
            if alpha_count < 2 {
                return false;
            }
        } else if alpha_count * 10 < text_len * 3 {
            return false;
        }

        // Reject KPI-style values ("4.2 days", "+15% QoQ", "$1.2M Total"):
        // strings that LEAD with a number/sign/currency symbol are almost
        // always data values, not headings, even in a larger font. A real
        // heading leads with a word.
        let first = trimmed.chars().next().unwrap_or(' ');
        if first.is_ascii_digit() || matches!(first, '+' | '-' | '$' | '€' | '£' | '¥' | '%') {
            return false;
        }

        true
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

    /// Detect heading level from the span's font size relative to the
    /// document's body size (caller-provided, typically the mode of
    /// observed sizes). Ratios: H1 >=1.8x, H2 >=1.4x, H3 >=1.2x, or
    /// H4 for bold at >=1.05x.
    ///
    /// The bold-threshold tier exists for documents whose section
    /// headings are set in the same family as body text but bumped by
    /// only a few percent of point size — common in corporate manuals
    /// (issue #377 D2: amt_handbook_sample, nougat_032, technical
    /// docs). Without the bold gate this would over-promote
    /// emphasised inline phrases.
    fn heading_level_ratio(&self, span: &OrderedTextSpan, base_font_size: f32) -> Option<u8> {
        if !Self::is_valid_heading_text(span.span.text.trim()) {
            return None;
        }
        if base_font_size <= 0.0 {
            return None;
        }
        let size_ratio = span.span.font_size / base_font_size;
        let is_bold = matches!(
            span.span.font_weight,
            FontWeight::Bold | FontWeight::Black | FontWeight::ExtraBold | FontWeight::SemiBold
        );
        if size_ratio >= 1.8 {
            Some(1)
        } else if size_ratio >= 1.4 {
            Some(2)
        } else if size_ratio >= 1.2 {
            Some(3)
        } else if is_bold && size_ratio >= 1.05 {
            // Bold text with even slight size increase is a heading signal.
            // H4 (was H3) since the weaker signal warrants a lower level.
            Some(4)
        } else {
            None
        }
    }

    /// Render a Table as a markdown table string.
    ///
    /// Normalizes column counts so every row has the same number of pipe-delimited
    /// cells. Without this, markdown parsers silently drop trailing cells from
    /// short rows, which causes data loss (e.g. "CERTIFICATE NO.: 403852" missing
    /// from converted output).
    fn render_table_markdown(&self, table: &Table, config: &TextPipelineConfig) -> String {
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

        // Find the maximum effective column count across all rows.
        // Each cell contributes `colspan` columns (default 1).
        let max_cols = table
            .rows
            .iter()
            .map(|row| {
                row.cells
                    .iter()
                    .map(|c| c.colspan.max(1) as usize)
                    .sum::<usize>()
            })
            .max()
            .unwrap_or(0);

        for (row_idx, row) in table.rows.iter().enumerate() {
            output.push('|');
            let mut cols_written: usize = 0;
            for cell in &row.cells {
                output.push(' ');

                // Render bold/italic from span metadata when available;
                // fall back to plain text for cells without span info.
                let cell_text = if !cell.spans.is_empty() {
                    let mut cell_md = String::new();
                    let mut active_bold = false;
                    let mut active_italic = false;

                    // Order per-span emit: close-old-markers → inter-span
                    // space → open-new-markers → text. This keeps whitespace
                    // OUTSIDE emphasis delimiters, which CommonMark requires
                    // (`** text**` and `**text **` are both rejected as
                    // literal asterisks by strict renderers).
                    for (i, span) in cell.spans.iter().enumerate() {
                        let is_bold = self.is_bold_raw(span, config);
                        let is_italic = span.is_italic;
                        let formatting_changed =
                            is_bold != active_bold || is_italic != active_italic;

                        if formatting_changed {
                            if active_italic {
                                cell_md.push('*');
                            }
                            if active_bold {
                                cell_md.push_str("**");
                            }
                        }

                        if i > 0 {
                            let prev = &cell.spans[i - 1];
                            let has_gap = super::has_horizontal_gap(prev, span);
                            let already_has_space =
                                cell_md.ends_with(' ') || span.text.starts_with(' ');
                            if has_gap && !already_has_space {
                                cell_md.push(' ');
                            }
                        }

                        if formatting_changed {
                            if is_bold {
                                cell_md.push_str("**");
                            }
                            if is_italic {
                                cell_md.push('*');
                            }
                            active_bold = is_bold;
                            active_italic = is_italic;
                        }

                        let mut text = span.text.replace('|', "\\|").replace('\n', " ");
                        let just_opened = is_bold || is_italic;
                        if just_opened && (cell_md.ends_with("**") || cell_md.ends_with('*')) {
                            while text.starts_with(' ') {
                                text.remove(0);
                            }
                        }
                        cell_md.push_str(&text);
                    }

                    // Final close: CommonMark forbids whitespace adjacent
                    // to closing markers; strip it before the markers and
                    // re-append after.
                    if active_italic || active_bold {
                        let content_end = cell_md.trim_end().len();
                        let trailing = cell_md[content_end..].to_string();
                        cell_md.truncate(content_end);
                        if active_italic {
                            cell_md.push('*');
                        }
                        if active_bold {
                            cell_md.push_str("**");
                        }
                        cell_md.push_str(&trailing);
                    }

                    cell_md
                } else {
                    cell.text.replace('|', "\\|").replace('\n', " ")
                };

                output.push_str(cell_text.trim());
                output.push(' ');
                // Handle colspan by adding extra | separators
                let span = cell.colspan.max(1) as usize;
                for _ in 1..span {
                    output.push_str("| ");
                }
                output.push('|');
                cols_written += span;
            }
            // Pad short rows with empty cells so every row has `max_cols` columns.
            for _ in cols_written..max_cols {
                output.push_str(" |");
            }
            output.push('\n');

            // Add header separator after header rows
            if row_idx + 1 == header_end {
                output.push('|');
                // Separator must also match max_cols
                let header_cols: usize = row.cells.iter().map(|c| c.colspan.max(1) as usize).sum();
                for _ in 0..max_cols.max(header_cols) {
                    output.push_str("---|");
                }
                output.push('\n');
            }
        }

        output
    }

    /// Resolve bold emphasis for a raw TextSpan honoring config.
    fn is_bold_raw(&self, span: &crate::layout::TextSpan, config: &TextPipelineConfig) -> bool {
        use crate::pipeline::config::BoldMarkerBehavior;
        match span.font_weight {
            FontWeight::Bold | FontWeight::Black | FontWeight::ExtraBold | FontWeight::SemiBold => {
                match config.output.bold_marker_behavior {
                    BoldMarkerBehavior::Aggressive => true,
                    BoldMarkerBehavior::Conservative => {
                        span.text.chars().any(|c| !c.is_whitespace())
                    },
                }
            },
            _ => false,
        }
    }

    /// Core rendering logic shared between convert() and convert_with_tables().
    fn render_spans(
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

        // Body-font size for the heading-ratio reference. Span-count
        // mode bucketed to 0.5pt, with smaller-bucket tiebreak so body
        // text wins over headings when counts are close. Capped at 12pt
        // so that heading-only documents still produce sensible ratios.
        let base_font_size = if config.output.detect_headings {
            // Exclude sub-9pt spans (bullet glyphs, subscripts, footnotes)
            // that would skew the mode downward.
            let mut size_counts: std::collections::HashMap<u32, usize> =
                std::collections::HashMap::new();
            for s in sorted.iter() {
                let sz = s.span.font_size;
                if sz < 9.0 {
                    continue;
                }
                *size_counts.entry((sz * 2.0).round() as u32).or_insert(0) += 1;
            }
            let mode = size_counts
                .into_iter()
                .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
                .map(|(bucket, _)| bucket as f32 / 2.0)
                .unwrap_or(12.0);
            mode.min(12.0)
        } else {
            12.0
        };

        // Track which tables have been rendered
        let mut tables_rendered = vec![false; tables.len()];
        // Pre-render table markdown so we can check for orphaned spans.
        let table_mds: Vec<String> = tables
            .iter()
            .map(|t| self.render_table_markdown(t, config))
            .collect();
        // Collect spans skipped because they fall inside a table region.
        let mut table_skipped_spans: Vec<Vec<&OrderedTextSpan>> = vec![Vec::new(); tables.len()];

        let mut result = String::new();
        let mut prev_span: Option<&OrderedTextSpan> = None;
        let mut current_line = String::new();
        // Track open inline formatting to consolidate adjacent bold/italic spans.
        // When consecutive same-line spans share the same bold or italic style,
        // we keep the markers open and only close them when the style changes or
        // the line is flushed, producing e.g. **ACME GLOBAL LTD.** instead
        // of **ACME** **GLOBAL** **LTD.**.
        let mut active_bold = false;
        let mut active_italic = false;
        let mut current_heading_level: Option<u8> = None;

        /// Close any open bold/italic markers on `line`.
        ///
        /// CommonMark forbids whitespace adjacent to closing emphasis markers
        /// (e.g. `**bold **` is rendered as literal asterisks). Strip trailing
        /// whitespace before closing, then restore it after the markers.
        fn close_formatting(line: &mut String, bold: &mut bool, italic: &mut bool) {
            if !*bold && !*italic {
                return;
            }
            let content_end = line.trim_end().len();
            let trailing_ws = line[content_end..].to_string();
            line.truncate(content_end);
            // Close in reverse order of opening: italic first, then bold.
            if *italic {
                line.push('*');
                *italic = false;
            }
            if *bold {
                line.push_str("**");
                *bold = false;
            }
            line.push_str(&trailing_ws);
        }

        // Strip markdown emphasis markers (**bold**, *italic*) from a line.
        // Used when emitting heading lines, where the `#` prefix already
        // provides emphasis and nested markers (e.g. `# **Title**`) are
        // redundant and can confuse strict CommonMark renderers.
        fn strip_emphasis(s: &str) -> String {
            let mut out = String::with_capacity(s.len());
            let chars: Vec<char> = s.chars().collect();
            let mut i = 0;
            while i < chars.len() {
                if chars[i] == '*' {
                    // Skip one or two asterisks
                    i += 1;
                    if i < chars.len() && chars[i] == '*' {
                        i += 1;
                    }
                    continue;
                }
                out.push(chars[i]);
                i += 1;
            }
            out
        }

        for span in sorted.iter() {
            // Skip artifacts (pagination, headers, footers)
            if span.span.artifact_type.is_some() {
                continue;
            }

            // Skip "noise" spans: isolated single-character fragments that
            // are purely punctuation/symbol (e.g. a bare "|" or "—" on its
            // own baseline from a decorative PDF separator). These add no
            // semantic value but pollute output as lone-line paragraphs.
            // Bullet characters are excluded from this filter since they
            // are meaningful list markers handled downstream.
            {
                let t = span.span.text.trim();
                let char_count = t.chars().count();
                if char_count > 0
                    && char_count <= 2
                    && !t.chars().any(|c| c.is_alphanumeric())
                    && !Self::is_bullet_span(t)
                    && !Self::starts_with_bullet(t)
                {
                    continue;
                }
            }

            // Check if this span belongs to a table region
            if !tables.is_empty() {
                if let Some(table_idx) = super::span_in_table(span, tables) {
                    if !tables_rendered[table_idx] {
                        // Flush current line
                        close_formatting(&mut current_line, &mut active_bold, &mut active_italic);
                        if !current_line.is_empty() {
                            result.push_str(current_line.trim());
                            result.push_str("\n\n");
                            current_line.clear();
                        }

                        // Render the table
                        result.push_str(&table_mds[table_idx]);
                        result.push('\n');
                        tables_rendered[table_idx] = true;
                        prev_span = None;
                    }
                    // Track span for orphan recovery
                    table_skipped_spans[table_idx].push(span);
                    // Skip this span (it's part of a table)
                    continue;
                }
            }

            // Heading level: structure-tree role takes precedence over
            // font-size heuristics when the source PDF is tagged. This
            // is the issue #377 D1 unlock — Word/Acrobat tagged PDFs
            // that set body and heading text in the same point size
            // would otherwise lose all heading hierarchy.
            let span_heading_level = match span.struct_role {
                Some(StructRole::Heading(level)) => Some(level.clamp(1, 6)),
                _ if config.output.detect_headings => {
                    self.heading_level_ratio(span, base_font_size)
                },
                _ => None,
            };

            // List-item role from the structure tree. When set, we emit
            // a markdown `- ` bullet at the start of the line for this
            // span (mirroring `is_bullet_span`/`starts_with_bullet`
            // detection used for untagged docs).
            let is_list_item_role = matches!(
                span.struct_role,
                Some(StructRole::ListItemBody)
                    | Some(StructRole::ListItem)
                    | Some(StructRole::ListItemLabel)
            );

            // Check for paragraph break or line break
            let same_line = prev_span
                .map(|prev| (span.span.bbox.y - prev.span.bbox.y).abs() < span.span.font_size * 0.5)
                .unwrap_or(true);

            if let Some(prev) = prev_span {
                // Group boundary: when group_id changes, insert a paragraph break
                // to keep spatially partitioned regions (e.g. columns) contiguous.
                let group_changed = match (span.group_id, prev.group_id) {
                    (Some(a), Some(b)) => a != b,
                    _ => false,
                };

                let heading_changed = current_heading_level != span_heading_level;

                // A reading-order group change only forces a paragraph break
                // when the visual line also changes — this keeps horizontally
                // split elements (e.g. multi-span footer lines) together.
                let group_flush = group_changed && !same_line;

                let prev_was_list_item = matches!(
                    prev.struct_role,
                    Some(StructRole::ListItemBody)
                        | Some(StructRole::ListItem)
                        | Some(StructRole::ListItemLabel)
                );
                let list_item_changed = is_list_item_role != prev_was_list_item;

                // Tagged-PDF block boundary (issue #377 D5): adjacent
                // spans whose nearest paragraph-level structure ancestor
                // differs are explicitly separate paragraphs even when
                // the geometric gap is small (pdfa_049 has body-tight
                // inter-paragraph gaps that the gap heuristic never
                // catches).
                //
                // D5b refinement: gate this on `!same_line` so a tagged
                // form whose horizontal heading band is split into
                // multiple /P sub-elements on one line (irs_f1040 has
                // `Form` + `1040` + `U.S. Individual Income Tax Return`
                // as three sibling /P blocks at the same y) does not
                // become three separate `# Form` / `# 1040` / ... lines.
                //
                // D5c refinement: when same_line is true but a
                // multi-column gutter separates the spans (large
                // horizontal gap), restore the break. Newspapers
                // (IA_0047) and other multi-column tagged docs
                // otherwise produce concatenated tokens like
                // `andmight` from adjacent-column content sharing a
                // baseline.
                let column_gap = is_column_gap(prev, span);
                let line_truly_continuous = same_line && !column_gap;
                let block_changed = match (span.block_id, prev.block_id) {
                    (Some(a), Some(b)) => a != b,
                    _ => false,
                } && !line_truly_continuous;

                // For heading transitions: same logic — visual line
                // continuity wins over structure-tree fragmentation.
                // For list-item transitions: ALWAYS break because a
                // bullet `- ` needs its own markdown line regardless
                // of whether the source PDF rendered the marker
                // inline with a leading caption.
                let heading_changed_break = heading_changed && !line_truly_continuous;

                if group_flush
                    || self.is_paragraph_break(span, prev)
                    || heading_changed_break
                    || list_item_changed
                    || block_changed
                    || column_gap
                {
                    close_formatting(&mut current_line, &mut active_bold, &mut active_italic);
                    if !current_line.is_empty() {
                        if let Some(level) = current_heading_level {
                            let prefix = "#".repeat(level as usize);
                            result.push_str(&format!(
                                "{} {}\n\n",
                                prefix,
                                strip_emphasis(current_line.trim())
                            ));
                        } else {
                            result.push_str(current_line.trim());
                            result.push_str("\n\n");
                        }
                        current_line.clear();
                    }
                    current_heading_level = span_heading_level;
                    if is_list_item_role {
                        current_line.push_str("- ");
                    }
                } else if !same_line {
                    // Different visual line but within paragraph spacing.
                    // Check if a bullet or ordered-marker item starts here
                    // — if so, start a new line. Issue #377 D3 guards
                    // numbered lists (`1. Foo` / `2. Bar` / `3. Baz`) at
                    // the same X across consecutive baselines: the items
                    // must not concatenate into one line of running text.
                    //
                    // Only fire on a list-item *transition* (the body of
                    // a wrapped LI keeps the same role across visual
                    // lines and must NOT emit a fresh bullet on each
                    // wrapped line).
                    let is_bullet = Self::is_bullet_span(&span.span.text)
                        || Self::starts_with_bullet(&span.span.text);
                    let is_ordered =
                        Self::is_ordered_list_marker(span.span.text.trim_start()).is_some();
                    // Tagged docs: each /LI gets its own `block_id`,
                    // so wrapped multi-line items share the same id
                    // and we should only fire on a TRANSITION
                    // (different block_id or list_item_changed).
                    // Untagged docs (block_id None on both): can't
                    // tell which body lines are wrapped vs which are
                    // new items, so fall back to "any list-role on a
                    // new baseline starts a new item".
                    let starts_new_list_item = if span.block_id.is_some() && prev.block_id.is_some()
                    {
                        is_list_item_role && (list_item_changed || block_changed)
                    } else {
                        is_list_item_role
                    };
                    if is_bullet || is_ordered || starts_new_list_item {
                        // Bullet on new line → flush current line and start list item
                        close_formatting(&mut current_line, &mut active_bold, &mut active_italic);
                        if !current_line.is_empty() {
                            if let Some(level) = current_heading_level {
                                let prefix = "#".repeat(level as usize);
                                result.push_str(&format!(
                                    "{} {}\n\n",
                                    prefix,
                                    strip_emphasis(current_line.trim())
                                ));
                            } else {
                                result.push_str(current_line.trim());
                                result.push('\n');
                            }
                            current_line.clear();
                        }
                        current_heading_level = span_heading_level;
                        if starts_new_list_item {
                            current_line.push_str("- ");
                        }
                    } else {
                        // Different visual line within the same paragraph — close
                        // open formatting before the line-join space so that
                        // formatting is re-evaluated for the new line's spans.
                        close_formatting(&mut current_line, &mut active_bold, &mut active_italic);
                        if config.output.preserve_layout {
                            let spacing = (span.span.bbox.x - prev.span.bbox.x).max(0.0) as usize;
                            for _ in 0..spacing.min(20) {
                                current_line.push(' ');
                            }
                        } else {
                            current_line.push(' ');
                        }
                    }
                }
            } else {
                current_heading_level = span_heading_level;
                if is_list_item_role {
                    current_line.push_str("- ");
                }
            }

            // Standalone bullet-glyph span → markdown list marker.
            if Self::is_bullet_span(&span.span.text) {
                if !current_line.ends_with("- ") {
                    if !current_line.is_empty() && !current_line.ends_with(' ') {
                        current_line.push(' ');
                    }
                    current_line.push_str("- ");
                }
                prev_span = Some(span);
                continue;
            }

            let mut text_str = span.span.text.clone();

            // Normalize known mis-extracted bullet glyphs (DEL from Zapf
            // Dingbats mappings, ❍ from ligature remaps) to U+2022 so the
            // bullet-span logic above can recognize them uniformly.
            if text_str.contains('\x7f') || text_str.contains('❍') {
                text_str = text_str.replace(['\x7f', '❍'], "•");
            }

            // Pipe characters are only markdown-syntactic inside table
            // cells; in paragraph flow they are just text. Pipe escaping
            // for tables is handled in render_table_markdown. Leaving `|`
            // alone in flow avoids showing `&#124;` in user-visible prose.

            let mut text = text_str.as_str();

            // Handle inline bullets (text starts with bullet char)
            if Self::starts_with_bullet(text) {
                let stripped = Self::strip_bullet(text);
                if !current_line.ends_with("- ") {
                    if !current_line.is_empty() && !current_line.ends_with(' ') {
                        current_line.push(' ');
                    }
                    current_line.push_str("- ");
                }
                text = stripped;
            }

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

            // Issue #260: Detect horizontal gaps between same-line spans and
            // insert a space.  PDFs generated by PDFKit.NET (and similar) place
            // each word in its own BT/ET block with absolute positioning.  The
            // spans carry no leading/trailing whitespace so the PR #273
            // whitespace-preservation logic above cannot help.  We replicate the
            // same gap heuristic used by extract_text()'s should_insert_space():
            // gap > 15% of font size → space, but not if > 5× font size (column
            // boundary).
            if same_line && !current_line.is_empty() {
                if let Some(prev) = prev_span {
                    let needs_gap_space = !current_line.ends_with(' ')
                        && !linkified.starts_with(' ')
                        && super::has_horizontal_gap(&prev.span, &span.span);
                    if needs_gap_space {
                        current_line.push(' ');
                    }
                }
            }

            // Consolidate adjacent spans with the same formatting style into
            // a single bold/italic block instead of wrapping each span
            // individually (e.g. **ACME GLOBAL LTD.** not
            // **ACME** **GLOBAL** **LTD.**).
            //
            // When the formatting changes we close the old markers and open
            // new ones.  When it stays the same we just append the text.
            if is_bold != active_bold || is_italic != active_italic {
                // Close previous formatting markers (if any)
                close_formatting(&mut current_line, &mut active_bold, &mut active_italic);
                // Open new markers
                if is_bold {
                    current_line.push_str("**");
                    active_bold = true;
                }
                if is_italic {
                    current_line.push('*');
                    active_italic = true;
                }
            }

            current_line.push_str(&linkified);

            prev_span = Some(span);
        }

        // Close any open formatting before final flushes
        close_formatting(&mut current_line, &mut active_bold, &mut active_italic);

        // Recover orphaned spans: spans inside a table region whose text does
        // not appear in the rendered table output.
        for (table_idx, skipped) in table_skipped_spans.iter().enumerate() {
            if !tables_rendered[table_idx] || skipped.is_empty() {
                continue;
            }
            let rendered = &table_mds[table_idx];
            let mut orphans: Vec<&&OrderedTextSpan> = skipped
                .iter()
                .filter(|s| {
                    let trimmed = s.span.text.trim();
                    !trimmed.is_empty() && !rendered.contains(trimmed)
                })
                .collect();
            if !orphans.is_empty() {
                orphans.sort_by_key(|s| s.reading_order);
                for orphan in orphans {
                    if !result.ends_with(' ') && !result.ends_with('\n') {
                        result.push(' ');
                    }
                    result.push_str(&orphan.span.text);
                }
            }
        }

        // Render any tables that weren't matched to spans (e.g., all spans were in tables)
        for (i, table) in tables.iter().enumerate() {
            if !tables_rendered[i] && !table.is_empty() {
                if !current_line.is_empty() {
                    if let Some(level) = current_heading_level {
                        let prefix = "#".repeat(level as usize);
                        result.push_str(&format!(
                            "{} {}\n\n",
                            prefix,
                            strip_emphasis(current_line.trim())
                        ));
                    } else {
                        result.push_str(current_line.trim());
                        result.push_str("\n\n");
                    }
                    current_line.clear();
                }
                result.push_str(&table_mds[i]);
                result.push('\n');
            }
        }

        // Flush remaining content
        if !current_line.is_empty() {
            if let Some(level) = current_heading_level {
                let prefix = "#".repeat(level as usize);
                result.push_str(&format!("{} {}\n", prefix, strip_emphasis(current_line.trim())));
            } else {
                result.push_str(current_line.trim());
                result.push('\n');
            }
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

        // Merge key-value pairs that were split across lines due to column-based
        // reading order (e.g. "Grand Total\n$750.00" → "Grand Total $750.00").
        final_result = super::merge_key_value_pairs(&final_result);

        // Apply hyphenation reconstruction if enabled
        if config.enable_hyphenation_reconstruction {
            let handler = HyphenationHandler::new();
            final_result = handler.process_text(&final_result);
        }

        // RTL emphasis cleanup (#377 D7-fix). The original D7 also
        // unconditionally re-ordered each RTL line via
        // `reorder_visual_to_logical`, on the assumption that PDF
        // content streams always emit RTL runs in *visual* order. In
        // practice some PDFs (notably the pdfium hebrew_mirrored.pdf
        // test fixture and Arabic CID-TrueType samples) already store
        // text in *logical* order and our blanket reorder reversed
        // them again, breaking previously-correct output (`בנימין` →
        // `ןימינב`, `# heading` → `heading #`). Without a reliable
        // way to detect which order the source uses we drop the
        // reorder step. The other half of D7 — stripping spurious
        // `**bold**` / `*italic*` markers that the font-weight
        // detector emits around Arabic contextual glyph forms — is
        // safe and stays.
        if crate::text::bidi::looks_rtl(&final_result) {
            final_result = final_result
                .lines()
                .map(|line| {
                    if crate::text::bidi::looks_rtl(line) {
                        strip_inline_emphasis_in_rtl(line)
                    } else {
                        line.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
        }

        Ok(final_result)
    }
}

/// Remove markdown `**` and `*` emphasis pairs that surround RTL
/// (Arabic / Hebrew) tokens. Inserted by the bold/italic detector
/// when the source PDF reports a font-weight change between
/// contextual glyph forms (initial / medial / final shapes); they
/// fragment the line into spurious emphasis spans and break bidi
/// reordering. Keeps emphasis around purely LTR runs intact.
///
/// Implementation note: the byte-position search via
/// `find_matching` is safe even on multi-byte UTF-8 because we only
/// look for ASCII `*` (0x2A) which never appears as a continuation
/// byte; matched indices always fall on a UTF-8 boundary. We then
/// build the output by appending UTF-8 string slices between the
/// matched positions, never reinterpreting individual bytes as
/// chars. (Copilot review #3108056051: the previous implementation
/// emitted `bytes[i] as char` for non-marker bytes and corrupted
/// non-ASCII content like `בנימין * world` → `×<ctrl>×<ctrl>... * world`.)
fn strip_inline_emphasis_in_rtl(line: &str) -> String {
    // Cheap path: if there are no asterisks, nothing to do.
    if !line.contains('*') {
        return line.to_string();
    }
    let bytes = line.as_bytes();
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    let mut last_copy = 0;
    while i < bytes.len() {
        // Try to match `**` first.
        if i + 1 < bytes.len() && bytes[i] == b'*' && bytes[i + 1] == b'*' {
            if let Some(close) = find_matching(bytes, i + 2, b"**") {
                // Copy any text that came before this `**` token verbatim.
                if i > last_copy {
                    out.push_str(&line[last_copy..i]);
                }
                let inner = &line[i + 2..close];
                if crate::text::bidi::looks_rtl(inner) {
                    out.push_str(inner);
                } else {
                    out.push_str("**");
                    out.push_str(inner);
                    out.push_str("**");
                }
                i = close + 2;
                last_copy = i;
                continue;
            }
        }
        // Then `*` (italic).
        if bytes[i] == b'*' {
            if let Some(close) = find_matching(bytes, i + 1, b"*") {
                if i > last_copy {
                    out.push_str(&line[last_copy..i]);
                }
                let inner = &line[i + 1..close];
                if crate::text::bidi::looks_rtl(inner) {
                    out.push_str(inner);
                } else {
                    out.push('*');
                    out.push_str(inner);
                    out.push('*');
                }
                i = close + 1;
                last_copy = i;
                continue;
            }
        }
        i += 1;
    }
    if last_copy < bytes.len() {
        out.push_str(&line[last_copy..]);
    }
    out
}

fn find_matching(bytes: &[u8], from: usize, needle: &[u8]) -> Option<usize> {
    let mut i = from;
    while i + needle.len() <= bytes.len() {
        if &bytes[i..i + needle.len()] == needle {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Detect a multi-column gutter between two spans on the same baseline.
///
/// Used by the markdown converter to refine its `same_line` gate: two
/// spans at the same y but separated by a large horizontal gap are
/// almost certainly in different columns (newspaper / two-column
/// academic paper). They must NOT be merged into one paragraph even
/// if their `block_id`s suggest a structural transition would be
/// suppressed by D5b.
///
/// Returns true in two distinct shapes (issue #377 D5d):
///
/// 1. **Forward column gap.** The horizontal gap from the right edge
///    of the previous span to the left edge of the current span
///    exceeds `max(3 × font_size, 30 pt)`. 3× font size catches
///    typical body-text columns (12pt body → 36pt gutter); the 30pt
///    floor catches small-font cases where a literal 36pt gap would
///    be too lenient.
///
/// 2. **Backward column wrap (x went backwards on the same baseline).**
///    LTR text on a single visual line always advances x forward; if
///    the current span starts to the left of the previous span by
///    more than `2 × font_size`, that is a column-major reading order
///    wrapping from the end of one column back to the top of the
///    next. The IA_0047 newspaper struct tree emits content this way:
///    `constitution` at x=976 ends a column, `Assailing` at x=192
///    starts the next, both at the same baseline. Without the
///    backward-wrap detection the converter joins them into the
///    nonsense token `constitutionAssailing`.
fn is_column_gap(prev: &OrderedTextSpan, current: &OrderedTextSpan) -> bool {
    let prev_right = prev.span.bbox.x + prev.span.bbox.width;
    let cur_left = current.span.bbox.x;
    let font_size = current.span.font_size.max(prev.span.font_size).max(1.0);

    // Backward wrap: x went meaningfully backwards on the same y
    // baseline. Strongest possible signal of a column-major reading
    // order transition.
    if cur_left + font_size * 2.0 < prev.span.bbox.x {
        return true;
    }

    // Forward gutter: gap exceeds typical inter-word spacing.
    let gap = cur_left - prev_right;
    if gap <= 0.0 {
        return false;
    }
    let threshold = (font_size * 3.0).max(30.0);
    gap > threshold
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
        tables: &[Table],
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
    use crate::pipeline::converters::span_in_table;
    use crate::pipeline::StructRole;
    use crate::structure::table_extractor::{TableCell, TableRow};

    /// D1 RED — when the structure tree carries an explicit heading role
    /// for a span (Word/Acrobat style: H1 → Span → MCR resolved by D8b),
    /// the markdown converter must emit `# title` regardless of font-size
    /// heuristics. Without this, every tagged Word document loses its
    /// heading hierarchy because body and heading text are often the
    /// same point size.
    #[test]
    fn test_struct_role_heading_emits_markdown_heading() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        let mut title = make_span("Document Title", 0.0, 100.0, 12.0, FontWeight::Normal);
        title.struct_role = Some(StructRole::Heading(1));
        let body = make_span("Body paragraph one.", 0.0, 80.0, 12.0, FontWeight::Normal);
        let result = converter.convert(&[title, body], &config).unwrap();
        assert!(
            result.contains("# Document Title"),
            "expected '# Document Title' in output, got:\n{}",
            result
        );
        assert!(result.contains("Body paragraph one."));
    }

    /// D1 RED — heading role precedence: even on the same font size as
    /// body, Heading(2) must produce `## ...`. Mirrors the `nougat_011`
    /// failure pattern where per-section headers are body-sized.
    #[test]
    fn test_struct_role_h2_overrides_font_size_heuristic() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        let mut h2 = make_span("Section Header", 0.0, 100.0, 11.0, FontWeight::Normal);
        h2.struct_role = Some(StructRole::Heading(2));
        let result = converter.convert(&[h2], &config).unwrap();
        assert!(result.starts_with("## "), "expected `## ` heading prefix, got:\n{}", result);
    }

    /// D3 unit — `is_ordered_list_marker` recognises common forms and
    /// rejects look-alikes that are NOT lists (figure captions, years).
    #[test]
    fn test_is_ordered_list_marker_recognition() {
        // Recognised forms.
        assert_eq!(MarkdownOutputConverter::is_ordered_list_marker("1. Foo"), Some(1));
        assert_eq!(MarkdownOutputConverter::is_ordered_list_marker("12. Foo"), Some(12));
        assert_eq!(MarkdownOutputConverter::is_ordered_list_marker("a) Foo"), Some(1));
        assert_eq!(MarkdownOutputConverter::is_ordered_list_marker("A. Foo"), Some(1));
        assert_eq!(MarkdownOutputConverter::is_ordered_list_marker("iv. Foo"), Some(1));
        // Conservative rejections so figure captions and years are not promoted.
        assert!(MarkdownOutputConverter::is_ordered_list_marker("1.1 Foo").is_none());
        assert!(MarkdownOutputConverter::is_ordered_list_marker("1986 was").is_none());
        assert!(MarkdownOutputConverter::is_ordered_list_marker("Item one").is_none());
    }

    /// D3 RED — three numbered items on consecutive lines must each
    /// land on their own markdown line. Reproduces the nougat_037
    /// "1. Treasurer ... 2. Safeguarding ... 3. Volunteering"
    /// collapse pattern (those three were on different baselines but
    /// joined by tight gap).
    #[test]
    fn test_numbered_list_consecutive_lines_separate() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        let s1 = make_span("1. Treasurer", 0.0, 100.0, 12.0, FontWeight::Normal);
        let s2 = make_span("2. Safeguarding", 0.0, 88.0, 12.0, FontWeight::Normal);
        let s3 = make_span("3. Volunteering", 0.0, 76.0, 12.0, FontWeight::Normal);
        let result = converter.convert(&[s1, s2, s3], &config).unwrap();
        for marker in ["1. Treasurer", "2. Safeguarding", "3. Volunteering"] {
            assert!(
                result.lines().any(|l| l.trim_start().starts_with(marker)),
                "expected line starting with `{}`, got:\n{}",
                marker,
                result
            );
        }
    }

    /// D4 RED — when an untagged paragraph is followed by a bullet list
    /// with a small geometric gap, the list must still start on a new
    /// line preceded by a blank line. Reproduces the `Intro sentence.•
    /// First` glue pattern.
    #[test]
    fn test_bullet_after_paragraph_forces_break() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        // Tight gap: body 12pt, gap 4pt (well below 1.5×).
        let intro = make_span("Intro sentence.", 0.0, 100.0, 12.0, FontWeight::Normal);
        let b1 = make_span("• First item", 0.0, 88.0, 12.0, FontWeight::Normal);
        let b2 = make_span("• Second item", 0.0, 76.0, 12.0, FontWeight::Normal);
        let result = converter.convert(&[intro, b1, b2], &config).unwrap();
        assert!(
            result.contains("Intro sentence.\n\n- First item"),
            "expected blank line + bullet after intro, got:\n{}",
            result
        );
    }

    /// D1 coverage — every heading level H1..H6 from the structure tree
    /// emits the matching markdown prefix. Lock-in for #377 word /
    /// adobe-tagged docs whose body and heading text share a size.
    #[test]
    fn test_struct_role_emits_each_heading_level() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        for level in 1u8..=6 {
            let mut s =
                make_span(&format!("Title L{}", level), 0.0, 100.0, 12.0, FontWeight::Normal);
            s.struct_role = Some(StructRole::Heading(level));
            let body = make_span("body", 0.0, 80.0, 12.0, FontWeight::Normal);
            let result = converter.convert(&[s, body], &config).unwrap();
            let prefix = "#".repeat(level as usize);
            let expected = format!("{} Title L{}", prefix, level);
            assert!(result.contains(&expected), "expected `{}`, got:\n{}", expected, result);
        }
    }

    /// D1 coverage — out-of-range Heading level values are clamped to
    /// the H1..H6 range. Defensive: a malformed structure tree
    /// reporting Heading(0) or Heading(99) should not produce 0 or
    /// 99 `#` characters.
    #[test]
    fn test_struct_role_heading_level_is_clamped() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        for raw_level in [0u8, 7, 99, 250] {
            let mut s = make_span("Edgy", 0.0, 100.0, 12.0, FontWeight::Normal);
            s.struct_role = Some(StructRole::Heading(raw_level));
            let result = converter.convert(&[s], &config).unwrap();
            // Find the prefix in the first line: count `#`s.
            let first_line = result.lines().next().unwrap_or("");
            let hash_count = first_line.chars().take_while(|c| *c == '#').count();
            assert!(
                (1..=6).contains(&hash_count),
                "raw_level {} produced {} `#`s in `{}`",
                raw_level,
                hash_count,
                first_line
            );
        }
    }

    /// D1 coverage — every list-role variant (LI / Lbl / LBody) on a
    /// span emits a `- ` bullet prefix. Lock-in against treating the
    /// three roles inconsistently, which was the original
    /// word365_structure regression.
    #[test]
    fn test_struct_role_all_list_variants_emit_bullets() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        for role in [
            StructRole::ListItem,
            StructRole::ListItemLabel,
            StructRole::ListItemBody,
        ] {
            let mut s = make_span("Item", 0.0, 100.0, 12.0, FontWeight::Normal);
            s.struct_role = Some(role);
            let result = converter.convert(&[s], &config).unwrap();
            assert!(
                result.lines().any(|l| l.starts_with("- ")),
                "role {:?} did not emit a bullet, got:\n{}",
                role,
                result
            );
        }
    }

    /// D1 coverage — heading immediately followed by a list-item must
    /// transition cleanly: heading flushes, list emits bullet on a
    /// fresh line. Cross-defect interaction guard.
    #[test]
    fn test_heading_then_list_item_transition() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        let mut h = make_span("Section", 0.0, 100.0, 12.0, FontWeight::Normal);
        h.struct_role = Some(StructRole::Heading(2));
        let mut li = make_span("First", 0.0, 80.0, 12.0, FontWeight::Normal);
        li.struct_role = Some(StructRole::ListItemBody);
        let result = converter.convert(&[h, li], &config).unwrap();
        assert!(result.contains("## Section"));
        assert!(result.contains("- First"));
        // The heading line must not also carry the bullet.
        assert!(
            !result.contains("## - "),
            "heading prefix and bullet must not co-occur, got:\n{}",
            result
        );
    }

    /// D5 coverage — three sequential block_id transitions produce
    /// three paragraphs. Lock against off-by-one in the transition
    /// detector that would group two of three.
    #[test]
    fn test_block_id_three_paragraphs_three_breaks() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        let mut spans = Vec::new();
        for (i, t) in ["alpha", "beta", "gamma"].iter().enumerate() {
            let mut s = make_span(t, 0.0, 100.0 - (i as f32 * 14.0), 12.0, FontWeight::Normal);
            s.block_id = Some((i + 1) as u32);
            spans.push(s);
        }
        let result = converter.convert(&spans, &config).unwrap();
        let paras: Vec<&str> = result
            .split("\n\n")
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .collect();
        assert_eq!(
            paras,
            vec!["alpha", "beta", "gamma"],
            "expected 3 separate paragraphs, got {:?}",
            paras
        );
    }

    /// D5 coverage — when only one of two adjacent spans has a
    /// block_id (mixed tagged + untagged), no spurious break is
    /// emitted. Defends against the `(Some, None)` case being misread
    /// as a transition.
    #[test]
    fn test_partial_block_id_does_not_force_break() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        let s1 = make_span("first", 0.0, 100.0, 12.0, FontWeight::Normal);
        let mut s2 = make_span("second", 0.0, 88.0, 12.0, FontWeight::Normal);
        s2.block_id = Some(1);
        let result = converter.convert(&[s1, s2], &config).unwrap();
        // Without explicit block transition, fall through to geometry —
        // 12pt gap below 1.5× threshold, so no double newline.
        assert!(
            !result.contains("\n\n"),
            "partial block_id must not introduce paragraph break, got:\n{}",
            result
        );
    }

    /// D3 coverage — extra whitelist + reject cases for ordered marker
    /// detection. Locks the conservative behaviour that distinguishes
    /// real lists from prose / numbers / captions.
    #[test]
    fn test_is_ordered_list_marker_extras() {
        // Recognised: trailing space required, both `.` and `)` close.
        assert_eq!(MarkdownOutputConverter::is_ordered_list_marker("99. Foo"), Some(99));
        assert_eq!(MarkdownOutputConverter::is_ordered_list_marker("z) Last"), Some(1));
        // Without the trailing space, not a list marker.
        assert!(MarkdownOutputConverter::is_ordered_list_marker("1.Foo").is_none());
        // Without a digit/letter prefix, not a list marker.
        assert!(MarkdownOutputConverter::is_ordered_list_marker(". Foo").is_none());
        assert!(MarkdownOutputConverter::is_ordered_list_marker(") Foo").is_none());
        // Empty / whitespace-only.
        assert!(MarkdownOutputConverter::is_ordered_list_marker("").is_none());
        assert!(MarkdownOutputConverter::is_ordered_list_marker("   ").is_none());
        // Looks like a list but is currency / unit / decimal.
        assert!(MarkdownOutputConverter::is_ordered_list_marker("$1. Total").is_none());
        assert!(MarkdownOutputConverter::is_ordered_list_marker("3.14 pi").is_none());
        // Long numeric (>3 digits) is not a marker (years, IDs).
        assert!(MarkdownOutputConverter::is_ordered_list_marker("2024. Year").is_none());
    }

    /// D6 coverage — superscript inline merging across multiple
    /// markers in the same line ("On the 1st, 2nd, and 3rd days").
    /// Each "st"/"nd"/"rd" must inline-merge with its preceding
    /// number.
    #[test]
    fn test_multiple_superscripts_one_line() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        // Body baseline at y=100 with 11pt; three superscripts raised
        // by 2.5pt with 7pt font.
        let parts: Vec<OrderedTextSpan> = vec![
            make_span("On the 1", 0.0, 100.0, 11.0, FontWeight::Normal),
            make_span("st", 25.0, 102.5, 7.0, FontWeight::Normal),
            make_span(", 2", 30.0, 100.0, 11.0, FontWeight::Normal),
            make_span("nd", 40.0, 102.5, 7.0, FontWeight::Normal),
            make_span(", and 3", 47.0, 100.0, 11.0, FontWeight::Normal),
            make_span("rd", 70.0, 102.5, 7.0, FontWeight::Normal),
            make_span(" days", 75.0, 100.0, 11.0, FontWeight::Normal),
        ];
        let result = converter.convert(&parts, &config).unwrap();
        // No bare superscript line.
        for sup in ["st", "nd", "rd"] {
            assert!(
                !result.lines().any(|l| l.trim() == sup),
                "bare `{}` line found in:\n{}",
                sup,
                result
            );
        }
        // Each composed token appears.
        for token in ["1st", "2nd", "3rd"] {
            assert!(result.contains(token), "expected `{}` in output, got:\n{}", token, result);
        }
    }

    /// D2 RED — bold text only slightly larger than body must still be
    /// detected as a heading. Many tagged-but-untyped corporate docs
    /// (amt_handbook_sample, manuals) use bold + 1.05–1.1× body for
    /// section headings without /H tags. Previous threshold was bold +
    /// 1.10×.
    #[test]
    fn test_bold_slight_size_bump_is_heading() {
        let converter = MarkdownOutputConverter::new();
        let mut config = TextPipelineConfig::default();
        config.output.detect_headings = true;
        // Body at 11pt, "section header" bold at 11.55pt (1.05× body).
        let body_a = make_span("First body sentence.", 0.0, 100.0, 11.0, FontWeight::Normal);
        let body_b = make_span("Second body sentence.", 0.0, 88.0, 11.0, FontWeight::Normal);
        let head = make_span("Section Header", 0.0, 76.0, 11.55, FontWeight::Bold);
        let body_c = make_span("After-heading body.", 0.0, 64.0, 11.0, FontWeight::Normal);
        let result = converter
            .convert(&[body_a, body_b, head, body_c], &config)
            .unwrap();
        assert!(
            result.contains("### Section Header") || result.contains("#### Section Header"),
            "expected heading prefix on bold +5% line, got:\n{}",
            result
        );
    }

    /// D5b RED — same-baseline spans with different `block_id`s
    /// from the structure tree (form-style PDFs that split a single
    /// horizontal heading into multiple /P sub-elements, e.g.
    /// `Form` + `1040` + `U.S. Individual Income Tax Return` rendered
    /// on one line) must NOT trigger a structure-tree paragraph break.
    /// Otherwise one heading becomes three `#` lines (irs_f1040
    /// regression observed in v0.3.36).
    #[test]
    fn test_same_baseline_blocks_do_not_split_heading() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        // Three pieces of one visual H1, same y=100, same font, but
        // each with its own structure-tree block_id (mimicking the
        // tagged form's three /P elements under one /H1 visually
        // joined in a horizontal heading band).
        let mk = |t: &str, x: f32, bid: u32| {
            let mut s = make_span(t, x, 100.0, 18.0, FontWeight::Bold);
            s.struct_role = Some(StructRole::Heading(1));
            s.block_id = Some(bid);
            s
        };
        let spans = vec![
            mk("Form", 0.0, 1),
            mk("1040", 50.0, 2),
            mk("U.S. Individual Income Tax Return", 100.0, 3),
        ];
        let result = converter.convert(&spans, &config).unwrap();
        let heading_lines: Vec<&str> = result
            .lines()
            .filter(|l| l.trim_start().starts_with("# "))
            .collect();
        assert_eq!(
            heading_lines.len(),
            1,
            "expected one combined heading line, got {} in:\n{}",
            heading_lines.len(),
            result
        );
        assert!(
            heading_lines[0].contains("Form")
                && heading_lines[0].contains("1040")
                && heading_lines[0].contains("U.S. Individual Income Tax Return"),
            "all three pieces must be in the single heading line, got: {}",
            heading_lines[0]
        );
    }

    /// D5b coverage — same-baseline list-item segments don't fragment.
    /// Some forms wrap each item label in its own /LI struct elem but
    /// render the whole list horizontally on one line; the converter
    /// must keep them together when y matches.
    #[test]
    fn test_same_baseline_blocks_do_not_split_list_items() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        let mk = |t: &str, x: f32, bid: u32| {
            let mut s = make_span(t, x, 100.0, 12.0, FontWeight::Normal);
            s.struct_role = Some(StructRole::ListItemBody);
            s.block_id = Some(bid);
            s
        };
        let spans = vec![
            mk("Apple", 0.0, 1),
            mk("Banana", 60.0, 2),
            mk("Cherry", 120.0, 3),
        ];
        let result = converter.convert(&spans, &config).unwrap();
        let bullet_lines: Vec<&str> = result.lines().filter(|l| l.starts_with("- ")).collect();
        assert_eq!(
            bullet_lines.len(),
            1,
            "horizontal list on one line must stay one bullet, got {} in:\n{}",
            bullet_lines.len(),
            result
        );
    }

    /// D5b coverage — different baselines must STILL fragment as
    /// before. Negative regression check on the D5 win: nougat_011
    /// went from 64 to 266 lines because each /P became its own
    /// paragraph; our same_line gate must not undo that for spans on
    /// different baselines.
    #[test]
    fn test_different_baseline_blocks_still_split() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        let mut p1 = make_span("First.", 0.0, 100.0, 12.0, FontWeight::Normal);
        p1.block_id = Some(1);
        let mut p2 = make_span("Second.", 0.0, 70.0, 12.0, FontWeight::Normal);
        p2.block_id = Some(2);
        let mut p3 = make_span("Third.", 0.0, 40.0, 12.0, FontWeight::Normal);
        p3.block_id = Some(3);
        let result = converter.convert(&[p1, p2, p3], &config).unwrap();
        let paras: Vec<&str> = result
            .split("\n\n")
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .collect();
        assert_eq!(
            paras,
            vec!["First.", "Second.", "Third."],
            "different baselines must still produce three paragraphs"
        );
    }

    /// `strip_inline_emphasis_in_rtl` must preserve non-ASCII (Arabic
    /// / Hebrew) characters in the non-emphasis portion of an RTL
    /// line. Earlier the function iterated the UTF-8 byte array and
    /// pushed each byte as a Latin-1 char, corrupting `בנימין * world`
    /// into `×<ctrl>×<ctrl>... * world`. The no-`*` short-circuit hid
    /// the bug from earlier RTL tests.
    #[test]
    fn test_strip_inline_emphasis_preserves_rtl_chars_around_lone_asterisk() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        let span = make_span("בנימין * world", 0.0, 100.0, 12.0, FontWeight::Normal);
        let result = converter.convert(&[span], &config).unwrap();
        assert!(
            result.contains("בנימין"),
            "Hebrew letters lost — UTF-8 corruption: {:?}",
            result
        );
        assert!(
            !result
                .chars()
                .any(|c| (c as u32) == 0x91 || (c as u32) == 0xA0),
            "byte-as-char ghost characters present in: {:?}",
            result
        );
    }

    /// Arabic regression coverage — confirms `strip_inline_emphasis_in_rtl`
    /// preserves Arabic across the no-`*`, single-`*`, paired-`*`,
    /// and paired-`**` cases. Locks the Copilot-found UTF-8
    /// corruption out for good across realistic shapes.
    #[test]
    fn test_arabic_strip_inline_emphasis_matrix() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        // Each tuple: (input span text, list of expected substrings).
        let cases: &[(&str, &[&str])] = &[
            // No `*` — short-circuit; must round-trip.
            ("اللغة العربية بسيطة", &["اللغة", "العربية", "بسيطة"]),
            // Hebrew with stray `*` (lone asterisk, no pair).
            ("בנימין * world", &["בנימין", "* world"]),
            // Arabic paragraph with `*emphasis*` around RTL token.
            ("مرحبا *عالم* اليوم", &["مرحبا", "عالم", "اليوم"]),
            // Arabic paragraph with `**bold**` around RTL token.
            ("مرحبا **عالم** اليوم", &["مرحبا", "عالم", "اليوم"]),
            // Mixed: emphasis around LTR (must keep markers) plus Arabic.
            ("مرحبا *Hello* اليوم", &["مرحبا", "*Hello*", "اليوم"]),
        ];
        for (input, expected_subs) in cases {
            let span = make_span(input, 0.0, 100.0, 12.0, FontWeight::Normal);
            let result = converter.convert(&[span], &config).unwrap();
            for needle in *expected_subs {
                assert!(
                    result.contains(needle),
                    "input {:?} → expected {:?} in output:\n{}",
                    input,
                    needle,
                    result
                );
            }
            // Ghost-byte check: no Latin-1 control chars from
            // mis-cast UTF-8 should appear.
            assert!(
                !result.chars().any(|c| {
                    let n = c as u32;
                    (0x80..=0x9F).contains(&n) || n == 0xA0
                }),
                "input {:?} produced Latin-1 ghost chars in: {:?}",
                input,
                result
            );
        }
    }

    /// Wrapped list-item body that spans multiple visual lines (same
    /// /LI struct elem, same block_id, same struct_role=ListItemBody)
    /// must NOT emit a fresh `- ` bullet on the second visual line.
    /// The break should fire on a list-item *transition*, not on the
    /// mere presence of a list role.
    #[test]
    fn test_wrapped_list_item_body_does_not_emit_extra_bullet() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        let mut a = make_span("First half of an item", 0.0, 100.0, 12.0, FontWeight::Normal);
        a.struct_role = Some(StructRole::ListItemBody);
        a.block_id = Some(7);
        let mut b = make_span("that wraps to next line.", 0.0, 86.0, 12.0, FontWeight::Normal);
        b.struct_role = Some(StructRole::ListItemBody);
        b.block_id = Some(7);
        let result = converter.convert(&[a, b], &config).unwrap();
        let bullet_lines: Vec<&str> = result.lines().filter(|l| l.starts_with("- ")).collect();
        assert_eq!(
            bullet_lines.len(),
            1,
            "wrapped list item body must stay one bullet, got {} lines:\n{}",
            bullet_lines.len(),
            result
        );
    }

    /// D7-fix RED — Hebrew text already in logical Unicode order
    /// (pdfium hebrew_mirrored.pdf shape) must NOT be reversed by
    /// the markdown converter. Reproduces the v0.3.36 regression
    /// where `בנימין` (logical) became `ןימינב` (reversed) after
    /// the unconditional bidi reorder pass.
    #[test]
    fn test_logical_hebrew_passes_through_unchanged() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        let span = make_span("בנימין", 0.0, 100.0, 12.0, FontWeight::Normal);
        let result = converter.convert(&[span], &config).unwrap();
        assert!(result.contains("בנימין"), "Hebrew must survive intact; got: {:?}", result);
        assert!(
            !result.contains("ןימינב"),
            "must NOT contain reversed Hebrew; got: {:?}",
            result
        );
    }

    /// D7-fix RED — Arabic heading line must keep `#` at the start
    /// after the converter runs. Reproduces the
    /// pdfs_pdfjs/ArabicCIDTrueType.pdf regression where `# ﺔﻴﺑﺮﻌﻟا`
    /// became `ﺔﻴﺑﺮﻌﻟا #` (hash moved to the end).
    #[test]
    fn test_arabic_heading_keeps_hash_at_start() {
        let converter = MarkdownOutputConverter::new();
        let mut config = TextPipelineConfig::default();
        config.output.detect_headings = true;
        let mut h = make_span("ﺔﻴﺑﺮﻌﻟا", 0.0, 100.0, 24.0, FontWeight::Bold);
        h.struct_role = Some(StructRole::Heading(1));
        let result = converter.convert(&[h], &config).unwrap();
        for line in result.lines() {
            if line.contains("ﺔﻴﺑﺮﻌﻟا") {
                assert!(
                    line.trim_start().starts_with('#'),
                    "heading line must start with `#`, got: {:?}",
                    line
                );
            }
        }
    }

    /// D5d RED — IA_0047 reproducer. The struct tree emits the last
    /// span of one column ("constitution" at x=976.7) immediately
    /// followed by the first span of the next column ("Assailing" at
    /// x=192.6) at the SAME baseline (y diff ≈ 1.5pt). A naive
    /// converter joins these into "constitutionAssailing".
    #[test]
    fn test_backward_x_wrap_at_same_baseline_splits_paragraph() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        let mk = |t: &str, x: f32, y: f32| make_span(t, x, y, 12.0, FontWeight::Normal);
        // Mirrors IA_0047 spans 1677 → 1678 (column wrap on same line).
        let prev = mk("constitution", 976.7, 1013.2);
        let cur = mk("Assailing", 192.6, 1011.7);
        let result = converter.convert(&[prev, cur], &config).unwrap();
        assert!(
            !result.contains("constitutionAssailing"),
            "column wrap created concatenation, got:\n{}",
            result
        );
        // Both words must be present, on different paragraphs.
        let paras: Vec<&str> = result
            .split("\n\n")
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .collect();
        assert!(
            paras.len() >= 2,
            "expected ≥2 paragraphs from column wrap, got {} in:\n{}",
            paras.len(),
            result
        );
        assert!(result.contains("constitution"));
        assert!(result.contains("Assailing"));
    }

    /// D5d coverage — minor x backwards (≤ 2× font_size) is NOT a
    /// column wrap. Could happen with tight kerning, italic
    /// overhang, or the existing dedup code emitting near-duplicate
    /// glyphs. Must NOT be promoted to a paragraph break.
    #[test]
    fn test_minor_x_backwards_within_tolerance_does_not_split() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        // 12pt font, x backs up by only 8pt (< 2 × 12 = 24pt).
        let prev = make_span("hello", 100.0, 100.0, 12.0, FontWeight::Normal);
        let cur = make_span("world", 92.0, 100.0, 12.0, FontWeight::Normal);
        let result = converter.convert(&[prev, cur], &config).unwrap();
        let paras: Vec<&str> = result
            .split("\n\n")
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .collect();
        assert_eq!(paras.len(), 1, "minor backstep must stay on one paragraph: {:?}", result);
    }

    /// D5d coverage — the same backward-wrap detector fires when
    /// block_ids are present (IA_0047 tagged paths) AND when no
    /// block_ids are present (untagged multi-column docs).
    #[test]
    fn test_backward_x_wrap_works_with_or_without_block_id() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        for assign_block in [false, true] {
            let mut a = make_span("end of col1", 800.0, 100.0, 12.0, FontWeight::Normal);
            let mut b = make_span("Start of col2", 100.0, 100.0, 12.0, FontWeight::Normal);
            if assign_block {
                a.block_id = Some(1);
                b.block_id = Some(2);
            }
            let result = converter.convert(&[a, b], &config).unwrap();
            assert!(
                !result.contains("col1Start"),
                "block_id={}: column wrap concat in:\n{}",
                assign_block,
                result
            );
        }
    }

    /// D5d coverage — backward wrap on different baselines should
    /// also produce a paragraph break (defensive: even if same_line
    /// is false, a backwards x indicates layout boundary).
    #[test]
    fn test_backward_x_wrap_on_different_baseline() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        // Mimics column wrap with different baselines (column 1
        // bottom at y=200, column 2 top at y=600). is_paragraph_break
        // catches this via the gap heuristic, but we ensure the
        // backward-x detector does too as a safety net.
        let prev = make_span("col1 last", 800.0, 200.0, 12.0, FontWeight::Normal);
        let cur = make_span("Col2 first", 100.0, 600.0, 12.0, FontWeight::Normal);
        let result = converter.convert(&[prev, cur], &config).unwrap();
        assert!(!result.contains("lastCol2"));
    }

    /// D5d coverage — the exact pattern of all 5 regressions found in
    /// IA_0047_20200204: lowercase end + uppercase start, same y,
    /// negative x delta. Each must split into separate paragraphs.
    #[test]
    fn test_all_five_ia_0047_patterns_split() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        // Each tuple is (col1-end-word, col2-start-word, y, font_size).
        let patterns: &[(&str, &str, f32, f32)] = &[
            ("constitution", "Assailing", 1013.0, 12.0),
            ("harvesting", "Senator", 1162.0, 12.0),
            ("humoro", "Spartacus", 950.0, 11.0),
            ("posscssec", "France", 800.0, 12.0),
            ("should", "Satisfy", 600.0, 12.0),
        ];
        for (a, b, y, sz) in patterns {
            let prev = make_span(a, 800.0, *y, *sz, FontWeight::Normal);
            let cur = make_span(b, 150.0, *y - 1.0, *sz, FontWeight::Normal);
            let result = converter.convert(&[prev, cur], &config).unwrap();
            let joined = format!("{}{}", a, b);
            assert!(
                !result.contains(&joined),
                "pattern {:?}+{:?} created `{}` in:\n{}",
                a,
                b,
                joined,
                result
            );
        }
    }

    /// D5d coverage — column-wrap detector composes with D5b form
    /// fix. A form heading split into pieces on the same baseline
    /// (small forward gaps) still joins; only when the gap is
    /// genuinely a column boundary (large forward OR backward) does
    /// it split.
    #[test]
    fn test_column_wrap_does_not_break_form_heading_join() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        let mk = |t: &str, x: f32, bid: u32| {
            let mut s = make_span(t, x, 100.0, 18.0, FontWeight::Bold);
            s.struct_role = Some(StructRole::Heading(1));
            s.block_id = Some(bid);
            s
        };
        // All forward, small gaps.
        let spans = vec![
            mk("Form", 0.0, 1),
            mk("1040", 35.0, 2),
            mk("Title", 80.0, 3),
        ];
        let result = converter.convert(&spans, &config).unwrap();
        let heading_lines: Vec<&str> = result.lines().filter(|l| l.starts_with("# ")).collect();
        assert_eq!(heading_lines.len(), 1, "form heading still joins: {}", result);
    }

    /// D5d unit — the helper itself. Property-style: matrix of
    /// gap/baseline/font shapes covering positive, zero, small
    /// negative, large negative, large positive.
    #[test]
    fn test_is_column_gap_matrix() {
        // (prev_x, prev_w, cur_x, font, expected)
        let cases: &[(f32, f32, f32, f32, bool)] = &[
            // Word gap inside a normal sentence: prev=Hello (50w) → cur="world".
            (100.0, 50.0, 154.0, 12.0, false),
            // Right at the 3× threshold: 36pt forward gap.
            (100.0, 50.0, 186.5, 12.0, true),
            // Far below threshold.
            (100.0, 50.0, 160.0, 12.0, false),
            // Backward 30pt at 12pt font (>2x = 24pt threshold).
            (200.0, 50.0, 100.0, 12.0, true),
            // Backward 8pt at 12pt font (under 24pt threshold).
            (100.0, 50.0, 92.0, 12.0, false),
            // Newspaper case: x=976→x=192 with 12pt font.
            (976.7, 37.8, 192.6, 12.0, true),
        ];
        for (px, pw, cx, font, expected) in cases {
            let prev = make_span("p", *px, 100.0, *font, FontWeight::Normal);
            let mut prev = prev;
            prev.span.bbox.width = *pw;
            let cur = make_span("c", *cx, 100.0, *font, FontWeight::Normal);
            let actual = is_column_gap(&prev, &cur);
            assert_eq!(
                actual, *expected,
                "(px={}, pw={}, cx={}, font={}) expected {} got {}",
                px, pw, cx, font, expected, actual
            );
        }
    }

    /// D5c RED — multi-column newspaper case. Two text spans on the
    /// same baseline but in different columns (large horizontal gap
    /// between the right edge of the previous span and the left edge
    /// of the current one), with different structure-tree block_ids.
    /// D5b would join them on one line and produce concatenated
    /// gibberish like `andmight`. The column-gap detector must split
    /// them into two paragraphs.
    #[test]
    fn test_column_gap_with_block_change_splits() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        // Column 1: "and" at x=0, width 30, baseline 100.
        // Column 2: "might" at x=180 (column gutter ≈ 150pt), baseline 100.
        // Body font 12pt, so the gap is well over 3× font_size.
        let mut col1 = make_span("and", 0.0, 100.0, 12.0, FontWeight::Normal);
        col1.block_id = Some(1);
        let mut col2 = make_span("might", 180.0, 100.0, 12.0, FontWeight::Normal);
        col2.block_id = Some(2);
        let result = converter.convert(&[col1, col2], &config).unwrap();
        // The two tokens must NOT be joined into `andmight`.
        assert!(
            !result.contains("andmight"),
            "column-gap join produced concatenated token, got:\n{}",
            result
        );
        // They must appear as separate words on separate lines or with
        // a paragraph break between them.
        assert!(result.contains("and"));
        assert!(result.contains("might"));
        // No `and might` glued onto one heading or paragraph either —
        // we want the two columns rendered as separate paragraphs.
        let paras: Vec<&str> = result
            .split("\n\n")
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .collect();
        assert!(
            paras.len() >= 2,
            "expected ≥2 paragraphs separated by column gap, got {} in:\n{}",
            paras.len(),
            result
        );
    }

    /// D5c coverage — same-baseline pieces of a tagged form heading
    /// (small inline gap, different block_ids) must still JOIN even
    /// after the column-gap detector. Regression guard for D5b.
    #[test]
    fn test_form_heading_inline_gap_still_joins() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        // `Form` ends at x≈30, `1040` starts at x≈40 — small inline
        // gap (≈10pt, well under 3× font_size = 54pt for 18pt heading).
        let mk = |t: &str, x: f32, bid: u32| {
            let mut s = make_span(t, x, 100.0, 18.0, FontWeight::Bold);
            s.struct_role = Some(StructRole::Heading(1));
            s.block_id = Some(bid);
            s
        };
        let spans = vec![
            mk("Form", 0.0, 1),
            mk("1040", 40.0, 2),
            mk("U.S.", 100.0, 3),
        ];
        let result = converter.convert(&spans, &config).unwrap();
        let heading_lines: Vec<&str> = result.lines().filter(|l| l.starts_with("# ")).collect();
        assert_eq!(
            heading_lines.len(),
            1,
            "small-gap form pieces must stay on one heading line, got:\n{}",
            result
        );
    }

    /// D5c coverage — boundary case: a moderate gap (e.g. 2× font
    /// size, like a wide indent or cell separator) should NOT trigger
    /// column split. Only truly large gaps (multi-column gutter)
    /// trigger the break.
    #[test]
    fn test_moderate_gap_does_not_force_column_break() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        // Body 12pt, gap of 24pt (2× font_size) — wide indent but not
        // a column gutter.
        let mut a = make_span("First field", 0.0, 100.0, 12.0, FontWeight::Normal);
        a.block_id = Some(1);
        let mut b = make_span("Second field", 80.0, 100.0, 12.0, FontWeight::Normal);
        b.block_id = Some(2);
        // The gap from x=0+50 (text "First field" width=50 in make_span) to x=80 = 30pt = 2.5× font_size.
        // Just below the column-gap threshold (3× = 36pt).
        let result = converter.convert(&[a, b], &config).unwrap();
        let paras: Vec<&str> = result
            .split("\n\n")
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .collect();
        assert_eq!(
            paras.len(),
            1,
            "moderate gap (≈2.5× font) must keep content on one paragraph, got:\n{}",
            result
        );
    }

    /// D5c coverage — three columns at the same baseline with large
    /// gaps must split into three paragraphs.
    #[test]
    fn test_three_column_layout_splits_into_three_paragraphs() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        let mk = |t: &str, x: f32, bid: u32| {
            let mut s = make_span(t, x, 100.0, 12.0, FontWeight::Normal);
            s.block_id = Some(bid);
            s
        };
        // Three 12pt-body columns at x=0, 200, 400 (gaps of ~150pt).
        let spans = vec![
            mk("col one", 0.0, 1),
            mk("col two", 200.0, 2),
            mk("col three", 400.0, 3),
        ];
        let result = converter.convert(&spans, &config).unwrap();
        let paras: Vec<&str> = result
            .split("\n\n")
            .map(|p| p.trim())
            .filter(|p| !p.is_empty())
            .collect();
        assert_eq!(paras.len(), 3, "three columns must produce three paragraphs, got:\n{}", result);
    }

    /// D5c coverage — column-gap detector applies even when no
    /// block_id is set (untagged document with multi-column layout).
    /// Without this, untagged newspapers would also produce
    /// `andmight`-style joins.
    #[test]
    fn test_column_gap_without_block_id_still_splits() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        // No block_id assigned (untagged).
        let a = make_span("left column.", 0.0, 100.0, 12.0, FontWeight::Normal);
        let b = make_span("right column.", 200.0, 100.0, 12.0, FontWeight::Normal);
        let result = converter.convert(&[a, b], &config).unwrap();
        // Pre-existing geometric heuristics should split too via the
        // group_id / has_horizontal_gap logic — verify the combined
        // result keeps the two columns as separate words at minimum.
        assert!(
            result.contains("left column") && result.contains("right column"),
            "both columns must surface, got:\n{}",
            result
        );
        // No concatenation across the gap.
        assert!(
            !result.contains("column.right"),
            "must not concatenate across column gap, got:\n{}",
            result
        );
    }

    /// D5b coverage — three-piece headings with a TINY (<1pt) y
    /// jitter still considered same-line. Forms often have minute
    /// baseline jitter due to font metric variation; the gate must be
    /// tolerant.
    #[test]
    fn test_minor_baseline_jitter_still_joins() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        let mk = |t: &str, x: f32, y: f32, bid: u32| {
            let mut s = make_span(t, x, y, 18.0, FontWeight::Bold);
            s.struct_role = Some(StructRole::Heading(1));
            s.block_id = Some(bid);
            s
        };
        // y values jitter within 0.5pt — well within the same_line
        // threshold (font_size * 0.5 = 9pt for an 18pt heading).
        let spans = vec![
            mk("A", 0.0, 100.0, 1),
            mk("B", 30.0, 100.3, 2),
            mk("C", 60.0, 99.7, 3),
        ];
        let result = converter.convert(&spans, &config).unwrap();
        let heading_lines: Vec<&str> = result.lines().filter(|l| l.starts_with("# ")).collect();
        assert_eq!(heading_lines.len(), 1, "tiny jitter must not split heading, got:\n{}", result);
    }

    /// D5b coverage — large baseline drop (well past same_line) DOES
    /// split, even with same heading_level. Proves the gate isn't
    /// over-suppressing.
    #[test]
    fn test_large_baseline_drop_still_splits_heading() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        let mk = |t: &str, y: f32, bid: u32| {
            let mut s = make_span(t, 0.0, y, 18.0, FontWeight::Bold);
            s.struct_role = Some(StructRole::Heading(1));
            s.block_id = Some(bid);
            s
        };
        // 30pt drop between baselines — far beyond `font_size * 0.5`.
        let spans = vec![mk("First Heading", 100.0, 1), mk("Second Heading", 70.0, 2)];
        let result = converter.convert(&spans, &config).unwrap();
        let heading_lines: Vec<&str> = result.lines().filter(|l| l.starts_with("# ")).collect();
        assert_eq!(
            heading_lines.len(),
            2,
            "two visually-separated headings must both surface, got:\n{}",
            result
        );
    }

    /// D5 RED — when adjacent spans carry different `block_id` from
    /// the source PDF's structure tree, force a paragraph break even
    /// when the geometric gap is too small for the
    /// `paragraph_gap_ratio` heuristic. Reproduces the pdfa_049
    /// pattern where two body-sized paragraphs sit ~14pt apart on a
    /// 12pt body and our 1.5× heuristic (16.5pt threshold) merges
    /// them. Tagged structure tree gives us authoritative paragraph
    /// boundaries via `OrderedContent.block_id`.
    #[test]
    fn test_block_id_change_forces_paragraph_break() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        // Two paragraphs separated by 12pt (less than 1.5× line_height).
        let mut p1 = make_span("Paragraph one body text.", 0.0, 100.0, 12.0, FontWeight::Normal);
        p1.block_id = Some(1);
        let mut p2 = make_span("Paragraph two starts here.", 0.0, 88.0, 12.0, FontWeight::Normal);
        p2.block_id = Some(2);
        let result = converter.convert(&[p1, p2], &config).unwrap();
        assert!(
            result.contains("Paragraph one body text.\n\nParagraph two starts here."),
            "expected double newline between block_ids 1→2, got:\n{:?}",
            result
        );
    }

    /// D5 RED (negative) — same `block_id` keeps spans on the same
    /// logical paragraph, even on different baselines (line wrap
    /// inside one /P struct elem).
    #[test]
    fn test_same_block_id_keeps_paragraph_continuous() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        let mut l1 = make_span("first line", 0.0, 100.0, 12.0, FontWeight::Normal);
        l1.block_id = Some(7);
        let mut l2 = make_span("second line", 0.0, 88.0, 12.0, FontWeight::Normal);
        l2.block_id = Some(7);
        let result = converter.convert(&[l1, l2], &config).unwrap();
        // No blank line between them.
        assert!(
            !result.contains("\n\n"),
            "same block_id must not introduce paragraph break, got:\n{:?}",
            result
        );
    }

    /// D6 RED — a small superscript span (≤4 chars, fontSize < 0.7× the
    /// preceding span) on a slightly raised baseline (PDF Ts/text-rise,
    /// spec §9.4.3) must merge into the same logical line as the body
    /// text instead of becoming its own paragraph. Reproduces the
    /// `21st → "21" + bare "st"` corruption visible in nougat_002 and
    /// the `23rd Street → "23" + "rd Street"` split visible in
    /// nougat_011 line 43.
    #[test]
    fn test_superscript_text_rise_does_not_split_line() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        // Body baseline at y=100 with 11pt body font.
        let pre = make_span("On June 21", 0.0, 100.0, 11.0, FontWeight::Normal);
        // Superscript "st" raised ~2.5pt with 7pt font (smaller than body).
        let sup = make_span("st", 35.0, 102.5, 7.0, FontWeight::Normal);
        let post = make_span(" they met.", 42.0, 100.0, 11.0, FontWeight::Normal);
        let result = converter.convert(&[pre, sup, post], &config).unwrap();
        assert!(
            result.contains("21st they met"),
            "expected '21st they met' inline, got:\n{}",
            result
        );
        assert!(
            !result.lines().any(|l| l.trim() == "st"),
            "no bare 'st' line allowed, got:\n{}",
            result
        );
    }

    /// D1 RED — list item body MCRs must emit a bullet on a new line.
    /// Reproduces the word365_structure / nougat_037 pattern where
    /// consecutive items collapse into a single line because the
    /// converter sees them as plain spans.
    #[test]
    fn test_struct_role_list_items_emit_bullets() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();
        let mut items = Vec::new();
        for (i, t) in ["Apple", "Banana", "Cherry"].iter().enumerate() {
            let mut s = make_span(t, 0.0, 100.0 - (i as f32 * 14.0), 12.0, FontWeight::Normal);
            s.struct_role = Some(StructRole::ListItemBody);
            items.push(s);
        }
        let result = converter.convert(&items, &config).unwrap();
        for t in ["- Apple", "- Banana", "- Cherry"] {
            assert!(result.contains(t), "expected `{}` line in output, got:\n{}", t, result);
        }
    }

    fn make_span_w(
        text: &str,
        x: f32,
        y: f32,
        width: f32,
        font_size: f32,
        weight: FontWeight,
    ) -> OrderedTextSpan {
        OrderedTextSpan::new(
            TextSpan {
                artifact_type: None,
                text: text.to_string(),
                bbox: Rect::new(x, y, width, font_size),
                font_name: "Test".to_string(),
                font_size,
                font_weight: weight,
                is_italic: false,
                is_monospace: false,
                color: Color::black(),
                mcid: None,
                sequence: 0,
                offset_semantic: false,
                split_boundary_before: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
            0,
        )
    }

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
                sequence: 0,
                offset_semantic: false,
                split_boundary_before: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
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

        let mut table = Table::new();
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
        let table = Table::new();
        let result = MarkdownOutputConverter::new()
            .render_table_markdown(&table, &crate::pipeline::TextPipelineConfig::default());
        assert_eq!(result, "");
    }

    #[test]
    fn test_render_table_markdown_single_row_no_header() {
        let mut table = Table::new();
        let mut row = TableRow::new(false);
        row.add_cell(TableCell::new("A".to_string(), false));
        row.add_cell(TableCell::new("B".to_string(), false));
        table.add_row(row);

        let result = MarkdownOutputConverter::new()
            .render_table_markdown(&table, &crate::pipeline::TextPipelineConfig::default());
        assert!(result.contains("| A |"));
        assert!(result.contains("| B |"));
        // First row treated as header by default in markdown
        assert!(result.contains("---|"));
    }

    #[test]
    fn test_render_table_markdown_with_colspan() {
        let mut table = Table::new();
        table.has_header = true;
        let mut header = TableRow::new(true);
        header.add_cell(TableCell::new("Wide".to_string(), true).with_colspan(2));
        table.add_row(header);

        let mut data = TableRow::new(false);
        data.add_cell(TableCell::new("Left".to_string(), false));
        data.add_cell(TableCell::new("Right".to_string(), false));
        table.add_row(data);

        let result = MarkdownOutputConverter::new()
            .render_table_markdown(&table, &crate::pipeline::TextPipelineConfig::default());
        // Colspan cell should produce extra | separators
        assert!(result.contains("| Wide |"));
        assert!(result.contains("---|---|"));
    }

    #[test]
    fn test_render_table_markdown_escapes_pipes() {
        let mut table = Table::new();
        let mut row = TableRow::new(false);
        row.add_cell(TableCell::new("A|B".to_string(), false));
        table.add_row(row);

        let result = MarkdownOutputConverter::new()
            .render_table_markdown(&table, &crate::pipeline::TextPipelineConfig::default());
        assert!(result.contains("A\\|B"), "Pipes should be backslash-escaped: {}", result);
    }

    #[test]
    fn test_render_table_markdown_replaces_newlines() {
        let mut table = Table::new();
        let mut row = TableRow::new(false);
        row.add_cell(TableCell::new("Line1\nLine2".to_string(), false));
        table.add_row(row);

        let result = MarkdownOutputConverter::new()
            .render_table_markdown(&table, &crate::pipeline::TextPipelineConfig::default());
        assert!(!result.contains("Line1\nLine2"), "Newlines in cells should be replaced");
        assert!(result.contains("Line1 Line2"));
    }

    #[test]
    fn test_render_table_markdown_trims_whitespace() {
        let mut table = Table::new();
        let mut row = TableRow::new(false);
        row.add_cell(TableCell::new("  padded  ".to_string(), false));
        table.add_row(row);

        let result = MarkdownOutputConverter::new()
            .render_table_markdown(&table, &crate::pipeline::TextPipelineConfig::default());
        assert!(result.contains("| padded |"));
    }

    #[test]
    fn test_render_table_markdown_multiple_header_rows() {
        let mut table = Table::new();
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

        let result = MarkdownOutputConverter::new()
            .render_table_markdown(&table, &crate::pipeline::TextPipelineConfig::default());
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
        let span = make_span("text", 50.0, 70.0, 12.0, FontWeight::Normal);

        let mut table = Table::new();
        table.bbox = Some(Rect::new(10.0, 50.0, 200.0, 100.0));

        assert_eq!(span_in_table(&span, &[table]), Some(0));
    }

    #[test]
    fn test_span_in_table_no_match() {
        let span = make_span("text", 500.0, 500.0, 12.0, FontWeight::Normal);

        let mut table = Table::new();
        table.bbox = Some(Rect::new(10.0, 50.0, 200.0, 100.0));

        assert_eq!(span_in_table(&span, &[table]), None);
    }

    #[test]
    fn test_span_in_table_none_bbox() {
        let span = make_span("text", 50.0, 70.0, 12.0, FontWeight::Normal);

        let table = Table::new(); // No bbox
        assert_eq!(span_in_table(&span, &[table]), None);
    }

    #[test]
    fn test_span_in_table_tolerance() {
        // Span at bbox edge minus tolerance (2.0)
        let span = make_span("text", 8.5, 48.5, 12.0, FontWeight::Normal);

        let mut table = Table::new();
        table.bbox = Some(Rect::new(10.0, 50.0, 200.0, 100.0));

        assert_eq!(span_in_table(&span, &[table]), Some(0), "Should match within tolerance");
    }

    #[test]
    fn test_span_in_table_multiple_tables() {
        let span = make_span("text", 350.0, 70.0, 12.0, FontWeight::Normal);

        let mut t1 = Table::new();
        t1.bbox = Some(Rect::new(10.0, 50.0, 200.0, 100.0));

        let mut t2 = Table::new();
        t2.bbox = Some(Rect::new(300.0, 50.0, 200.0, 100.0));

        assert_eq!(span_in_table(&span, &[t1, t2]), Some(1));
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

        // Text inside table region whose text matches table cell content
        // (not an orphan — absorbed by the table rendering).
        let mut span_in_table = make_span("Val", 50.0, 70.0, 12.0, FontWeight::Normal);
        span_in_table.reading_order = 1;

        let mut table = Table::new();
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

        let make_table = |x: f32, text: &str| -> Table {
            let mut t = Table::new();
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

    // ============================================================================
    // Issue #260: Single-word BT/ET blocks should have spaces between words
    // ============================================================================

    /// Helper to create a span with a specific width (for gap-detection tests).
    fn make_span_with_width(
        text: &str,
        x: f32,
        y: f32,
        width: f32,
        font_size: f32,
        weight: FontWeight,
        order: usize,
    ) -> OrderedTextSpan {
        let mut s = OrderedTextSpan::new(
            TextSpan {
                artifact_type: None,
                text: text.to_string(),
                bbox: Rect::new(x, y, width, font_size),
                font_name: "Test".to_string(),
                font_size,
                font_weight: weight,
                is_italic: false,
                is_monospace: false,
                color: Color::black(),
                mcid: None,
                sequence: 0,
                offset_semantic: false,
                split_boundary_before: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
                char_widths: vec![],
            },
            order,
        );
        s.reading_order = order;
        s
    }

    #[test]
    fn test_issue_260_single_word_bt_et_blocks_get_spaces() {
        // PDFKit.NET places each word in its own BT/ET block with absolute positioning.
        // The markdown converter must detect the horizontal gap and insert a space.
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();

        // Simulate: "The" at x=72 w=20, "quick" at x=96 w=30, "brown" at x=130 w=33
        // All same Y=500, font_size=12. Gaps: 96-92=4pt, 130-126=4pt.
        // 4pt gap > 0.15*12=1.8pt threshold → should insert space.
        let spans = vec![
            make_span_with_width("The", 72.0, 500.0, 20.0, 12.0, FontWeight::Normal, 0),
            make_span_with_width("quick", 96.0, 500.0, 30.0, 12.0, FontWeight::Normal, 1),
            make_span_with_width("brown", 130.0, 500.0, 33.0, 12.0, FontWeight::Normal, 2),
        ];

        let result = converter.convert(&spans, &config).unwrap();
        assert!(
            result.contains("The quick brown"),
            "Single-word BT/ET spans with gaps should have spaces inserted: got {:?}",
            result
        );
    }

    #[test]
    fn test_issue_260_no_space_for_tight_spans() {
        // When spans are tightly packed (no significant gap), no extra space should be added.
        // This covers ligature fragments or split characters.
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();

        // "Hel" at x=72 w=18, "lo" at x=90 w=12 — gap = 90-90 = 0pt, no space needed
        let spans = vec![
            make_span_with_width("Hel", 72.0, 500.0, 18.0, 12.0, FontWeight::Normal, 0),
            make_span_with_width("lo", 90.0, 500.0, 12.0, 12.0, FontWeight::Normal, 1),
        ];

        let result = converter.convert(&spans, &config).unwrap();
        assert!(
            result.contains("Hello"),
            "Tight spans should be merged without space: got {:?}",
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

    // ============================================================================
    // Bold consolidation tests
    // ============================================================================

    #[test]
    fn test_bold_consolidation_adjacent_bold_spans() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();

        // Three adjacent bold spans on the same line — each word is a separate span.
        // Use realistic bbox widths so that horizontal gap detection inserts spaces.
        let mut s1 = make_span_w("ACME", 72.0, 700.0, 55.0, 12.0, FontWeight::Bold);
        s1.reading_order = 0;

        let mut s2 = make_span_w("GLOBAL", 130.0, 700.0, 42.0, 12.0, FontWeight::Bold);
        s2.reading_order = 1;

        let mut s3 = make_span_w("LTD.", 175.0, 700.0, 24.0, 12.0, FontWeight::Bold);
        s3.reading_order = 2;

        let spans = vec![s1, s2, s3];
        let result = converter.convert(&spans, &config).unwrap();

        // Should consolidate into a single bold block
        assert!(
            result.contains("**ACME GLOBAL LTD.**"),
            "Adjacent bold spans should be consolidated into one bold block, got: {}",
            result
        );
        // Should NOT have per-word bold markers
        assert!(
            !result.contains("**ACME** **GLOBAL**"),
            "Should not wrap each word individually in bold markers, got: {}",
            result
        );
    }

    // ============================================================================
    // Issue: table cell dropping during markdown conversion
    // ============================================================================

    #[test]
    fn test_render_table_markdown_all_cells_present() {
        // Simulates a financial statement table:
        //   Row 1 (header): "Account No." | "Reference" | "Tax ID" | "Confirmation"
        //   Row 2 (data):   "20003035"    | "403852"    | "123 456 789" | "4351966"
        let mut table = Table::new();
        table.has_header = true;
        table.col_count = 4;

        let mut header = TableRow::new(true);
        header.add_cell(TableCell::new("Account No.".to_string(), true));
        header.add_cell(TableCell::new("Reference".to_string(), true));
        header.add_cell(TableCell::new("Tax ID".to_string(), true));
        header.add_cell(TableCell::new("Confirmation".to_string(), true));
        table.add_row(header);

        let mut data = TableRow::new(false);
        data.add_cell(TableCell::new("20003035".to_string(), false));
        data.add_cell(TableCell::new("403852".to_string(), false));
        data.add_cell(TableCell::new("123 456 789".to_string(), false));
        data.add_cell(TableCell::new("4351966".to_string(), false));
        table.add_row(data);

        let result = MarkdownOutputConverter::new()
            .render_table_markdown(&table, &crate::pipeline::TextPipelineConfig::default());

        // All cells must be present
        assert!(
            result.contains("403852"),
            "Reference value '403852' must be present in markdown table: {}",
            result
        );
        assert!(result.contains("20003035"), "Account No. value must be present: {}", result);
        assert!(result.contains("123 456 789"), "Tax ID value must be present: {}", result);
        assert!(result.contains("4351966"), "Confirmation value must be present: {}", result);
        assert!(result.contains("Reference"), "Header must be present: {}", result);

        // Must have pipe separators (markdown table format)
        assert!(result.contains("|"), "Must be markdown table format with pipe separators");
    }

    #[test]
    fn test_render_table_markdown_short_row_padded() {
        // When a data row has fewer cells than the header, the markdown table
        // must pad with empty cells so every row has the same column count.
        // Otherwise markdown parsers silently drop trailing columns.
        let mut table = Table::new();
        table.has_header = true;
        table.col_count = 4;

        let mut header = TableRow::new(true);
        header.add_cell(TableCell::new("A".to_string(), true));
        header.add_cell(TableCell::new("B".to_string(), true));
        header.add_cell(TableCell::new("C".to_string(), true));
        header.add_cell(TableCell::new("D".to_string(), true));
        table.add_row(header);

        // Data row with only 2 cells (e.g., merge detection removed 2 cells)
        let mut data = TableRow::new(false);
        data.add_cell(TableCell::new("1".to_string(), false));
        data.add_cell(TableCell::new("2".to_string(), false));
        table.add_row(data);

        let result = MarkdownOutputConverter::new()
            .render_table_markdown(&table, &crate::pipeline::TextPipelineConfig::default());

        // Count pipes in header vs data row — they must match
        let lines: Vec<&str> = result.lines().collect();
        assert!(lines.len() >= 3, "Must have header, separator, and data row: {}", result);

        let header_pipes = lines[0].matches('|').count();
        let data_pipes = lines[2].matches('|').count();
        assert_eq!(
            header_pipes, data_pipes,
            "Header and data rows must have same number of pipe separators.\nHeader ({}): {}\nData   ({}): {}",
            header_pipes, lines[0], data_pipes, lines[2]
        );
    }

    #[test]
    fn test_render_table_markdown_short_header_padded() {
        // When the header has fewer cells than the widest data row, the header
        // must also be padded.
        let mut table = Table::new();
        table.has_header = true;
        table.col_count = 3;

        let mut header = TableRow::new(true);
        header.add_cell(TableCell::new("X".to_string(), true));
        header.add_cell(TableCell::new("Y".to_string(), true));
        table.add_row(header);

        let mut data = TableRow::new(false);
        data.add_cell(TableCell::new("1".to_string(), false));
        data.add_cell(TableCell::new("2".to_string(), false));
        data.add_cell(TableCell::new("3".to_string(), false));
        table.add_row(data);

        let result = MarkdownOutputConverter::new()
            .render_table_markdown(&table, &crate::pipeline::TextPipelineConfig::default());

        let lines: Vec<&str> = result.lines().collect();
        assert!(lines.len() >= 3, "Must have header, separator, and data row: {}", result);

        let header_pipes = lines[0].matches('|').count();
        let data_pipes = lines[2].matches('|').count();
        assert_eq!(
            header_pipes, data_pipes,
            "Header and data rows must have same number of pipe separators.\nHeader ({}): {}\nData   ({}): {}",
            header_pipes, lines[0], data_pipes, lines[2]
        );

        // All data values must be present
        assert!(result.contains("| 3 |"), "Third cell in data row must be present: {}", result);
    }

    #[test]
    fn test_key_value_pair_merging_in_markdown() {
        let converter = MarkdownOutputConverter::new();
        let config = TextPipelineConfig::default();

        // Simulate a single label on one line followed by its value on the next.
        // This happens when spans from different groups produce separate lines.
        let mut s0 = make_span("Grand Total", 50.0, 200.0, 12.0, FontWeight::Normal);
        s0.reading_order = 0;
        s0.group_id = Some(0);

        // Value on a different line (different Y), next in reading order, different group
        let mut s1 = make_span("$750.00", 300.0, 185.0, 12.0, FontWeight::Normal);
        s1.reading_order = 1;
        s1.group_id = Some(1);

        let spans = vec![s0, s1];
        let result = converter.convert(&spans, &config).unwrap();

        assert!(
            result.contains("Grand Total $750.00"),
            "Should merge label with value on same line: {:?}",
            result,
        );
    }
}
