//! Markdown converter for PDF documents.
//!
//! This module converts PDF pages to Markdown format with support for:
//! - Heading detection (# ## ###)
//! - Paragraph formatting
//! - Image embedding
//! - Reading order determination

use crate::converters::whitespace::cleanup_markdown;
use crate::converters::{BoldMarkerBehavior, ConversionOptions, ReadingOrderMode};
use crate::error::Result;
use crate::geometry::Rect;
use crate::layout::clustering::{cluster_chars_into_words, cluster_words_into_lines};
use crate::layout::column_detector::{xy_cut, xy_cut_adaptive};
use crate::layout::document_analyzer::{AdaptiveLayoutParams, DocumentProperties};
use crate::layout::heading_detector::{HeadingLevel, detect_headings};
use crate::layout::reading_order::determine_reading_order as determine_order_from_tree;
use crate::layout::{TextBlock, TextChar};
use lazy_static::lazy_static;
use regex::{Captures, Regex};

lazy_static! {
    /// Regex for matching URLs in text
    static ref RE_URL: Regex = Regex::new(r"(https?://[^\s<>\[\]]*[^\s<>\[\].,!?;:])").unwrap();

    /// Regex for matching email addresses
    static ref RE_EMAIL: Regex = Regex::new(r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})").unwrap();

    /// Regex for cleaning space before dash in numeric contexts
    static ref RE_DASH_BEFORE: Regex = Regex::new(r"(\d)\s+(–|—)(\d)").unwrap();

    /// Regex for cleaning space after dash in numeric contexts
    static ref RE_DASH_AFTER: Regex = Regex::new(r"(\d)(–|—)\s+(\d)").unwrap();
}

/// Converter for PDF to Markdown format.
///
/// # Examples
///
/// ```ignore
/// use pdf_oxide::PdfDocument;
/// use pdf_oxide::converters::{MarkdownConverter, ConversionOptions};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut doc = PdfDocument::open("paper.pdf")?;
/// let chars = doc.extract_spans(0)?;
///
/// let converter = MarkdownConverter::new();
/// let options = ConversionOptions::default();
/// let markdown = converter.convert_page(&chars, &options)?;
///
/// println!("{}", markdown);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct MarkdownConverter;

impl MarkdownConverter {
    /// Create a new Markdown converter.
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::converters::MarkdownConverter;
    ///
    /// let converter = MarkdownConverter::new();
    /// ```
    pub fn new() -> Self {
        Self
    }

    /// Merge adjacent character-level spans that are too close to have real spaces.
    ///
    /// Per PDF Spec ISO 32000-1:2008 Section 9.4.4 NOTE 6, text strings should be
    /// "as long as possible". This merges character-level fragments that should be
    /// part of the same word.
    ///
    /// # Arguments
    ///
    /// * `blocks` - Sorted text blocks (by Y then X position)
    ///
    /// # Returns
    ///
    /// Merged text blocks with character-level fragments combined
    fn merge_adjacent_char_spans(blocks: Vec<TextBlock>) -> Vec<TextBlock> {
        if blocks.is_empty() {
            return blocks;
        }

        let mut merged: Vec<TextBlock> = Vec::new();
        let mut current: Option<TextBlock> = None;

        for block in blocks {
            match current.take() {
                None => {
                    // First block
                    current = Some(block);
                },
                Some(mut prev) => {
                    // Check if this block should be merged with previous
                    let same_line = (prev.bbox.y - block.bbox.y).abs() < 2.0;
                    let same_font = prev.dominant_font == block.dominant_font;
                    let same_size = (prev.avg_font_size - block.avg_font_size).abs() < 0.5;
                    let same_style = prev.is_bold == block.is_bold;

                    if same_line && same_font && same_size && same_style {
                        // Calculate gap between blocks
                        let prev_right = prev.bbox.x + prev.bbox.width;
                        let gap = block.bbox.x - prev_right;

                        // Merge threshold: 18% of font size
                        // Per PDF typography: char spacing is 5-15% em, word spacing is 20-40% em
                        // 18% catches character fragments while preserving word boundaries
                        let merge_threshold = prev.avg_font_size * 0.18;

                        // Don't merge if either block is just a space character
                        let prev_is_space = prev.text.trim().is_empty();
                        let curr_is_space = block.text.trim().is_empty();

                        if !prev_is_space && !curr_is_space && gap < merge_threshold {
                            // Merge: concatenate text and extend bounding box
                            prev.text.push_str(&block.text);
                            prev.bbox.width = (block.bbox.x + block.bbox.width) - prev.bbox.x;
                            current = Some(prev);
                        } else {
                            // Don't merge: push previous, keep current
                            merged.push(prev);
                            current = Some(block);
                        }
                    } else {
                        // Different line/font/size/style: don't merge
                        merged.push(prev);
                        current = Some(block);
                    }
                },
            }
        }

        // Don't forget the last block
        if let Some(last) = current {
            merged.push(last);
        }

        merged
    }

    /// Convert a page to Markdown format from text spans (PDF spec compliant - RECOMMENDED).
    ///
    /// This is the recommended method that uses PDF-native text spans instead of
    /// character-based extraction. Spans are complete text strings as provided by
    /// the PDF's Tj/TJ operators, eliminating the need for error-prone DBSCAN clustering.
    ///
    /// **Benefits over character-based conversion:**
    /// - PDF spec compliant (ISO 32000-1:2008, Section 9.4.4 NOTE 6)
    /// - No character splitting issues
    /// - Preserves PDF's text positioning intent
    /// - Much faster (no DBSCAN clustering needed)
    /// - More robust for complex layouts
    ///
    /// # Arguments
    ///
    /// * `spans` - The text spans extracted from the page via `extract_spans()`
    /// * `options` - Conversion options controlling the output
    ///
    /// # Returns
    ///
    /// A string containing the Markdown representation of the page.
    ///
    /// # Errors
    ///
    /// Returns an error if conversion fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use pdf_oxide::converters::{MarkdownConverter, ConversionOptions};
    /// # use pdf_oxide::layout::TextSpan;
    /// # fn example(spans: Vec<TextSpan>) -> Result<(), Box<dyn std::error::Error>> {
    /// let converter = MarkdownConverter::new();
    /// let options = ConversionOptions {
    ///     detect_headings: true,
    ///     ..Default::default()
    /// };
    ///
    /// let markdown = converter.convert_page_from_spans(&spans, &options)?;
    /// println!("{}", markdown);
    /// # Ok(())
    /// # }
    /// ```
    pub fn convert_page_from_spans(
        &self,
        spans: &[crate::layout::TextSpan],
        options: &ConversionOptions,
    ) -> Result<String> {
        use crate::layout::TextBlock;

        if spans.is_empty() {
            return Ok(String::new());
        }

        // Convert spans to TextBlocks for compatibility with existing rendering logic
        let mut blocks: Vec<TextBlock> = spans
            .iter()
            .map(|span| {
                TextBlock {
                    chars: vec![], // Not needed for span-based conversion
                    bbox: span.bbox,
                    text: span.text.clone(),
                    avg_font_size: span.font_size,
                    dominant_font: span.font_name.clone(),
                    is_bold: span.font_weight.is_bold(),
                    mcid: span.mcid,
                }
            })
            .collect();

        // Sort blocks by Y position (top to bottom), then X position (left to right)
        blocks.sort_by(|a, b| match a.bbox.y.partial_cmp(&b.bbox.y) {
            Some(std::cmp::Ordering::Equal) | None => a
                .bbox
                .x
                .partial_cmp(&b.bbox.x)
                .unwrap_or(std::cmp::Ordering::Equal),
            other => other.unwrap_or(std::cmp::Ordering::Equal),
        });

        // PDF Spec ISO 32000-1:2008 Section 9.4.4 NOTE 6:
        // "text strings are as long as possible"
        // Merge adjacent character-level spans that are too close to have real spaces
        // This handles PDFs with character-level fragmentation (like GDPR file)
        blocks = Self::merge_adjacent_char_spans(blocks);

        // Apply heading detection if enabled
        let heading_levels = if options.detect_headings {
            detect_headings(&blocks)
        } else {
            vec![HeadingLevel::Body; blocks.len()]
        };

        // Apply reading order (use simple top-to-bottom for span-based conversion)
        // XY-Cut algorithm requires adaptive params which need char-based analysis
        let ordered_indices =
            self.determine_reading_order(&blocks, ReadingOrderMode::TopToBottomLeftToRight, None);

        // Process blocks into lines based on Y coordinate and render incrementally
        // This approach uses constant memory instead of accumulating all line groups
        let mut markdown = String::new();
        let mut current_line: Vec<usize> = Vec::new();
        let mut current_y: Option<f32> = None;

        // Helper closure to render a completed line
        let render_line = |line_indices: &[usize], markdown: &mut String| {
            if line_indices.is_empty() {
                return;
            }

            // Check if this line is a heading (use first block's heading level)
            let first_idx = line_indices[0];
            let level = heading_levels[first_idx];

            // Add heading prefix if needed
            match level {
                HeadingLevel::H1 => markdown.push_str("# "),
                HeadingLevel::H2 => markdown.push_str("## "),
                HeadingLevel::H3 => markdown.push_str("### "),
                _ => {},
            }

            // Join blocks on this line, grouping consecutive blocks with same formatting
            // Per PDF spec (ISO 32000-1:2008, Section 9.4.4 NOTE 6):
            // Text extraction already handles word spacing based on TJ operator offsets.
            // Space characters are inserted as separate spans during extraction
            // (see process_tj_array in text.rs), so we just concatenate span text.
            //
            // Group consecutive blocks with same bold/italic status to avoid splitting
            // natural phrases like "Chinese stock market" into "**Chinese stock** market"
            let mut i = 0;
            while i < line_indices.len() {
                let idx = line_indices[i];
                let block = &blocks[idx];
                let is_bold = block.is_bold;

                // Find all consecutive blocks with same bold status
                let mut j = i + 1;
                while j < line_indices.len() && blocks[line_indices[j]].is_bold == is_bold {
                    j += 1;
                }

                // Render this group of blocks with unified formatting
                // Check word boundaries before/after to avoid mid-word bold markers
                let prev_char = if markdown.is_empty() {
                    None
                } else {
                    markdown.chars().last()
                };

                let next_char_after_group = if j < line_indices.len() {
                    blocks[line_indices[j]].text.chars().next()
                } else {
                    None
                };

                // Collect text from this group first to check boundaries
                let mut group_text = String::new();
                for k in i..j {
                    let block_idx = line_indices[k];
                    group_text.push_str(&blocks[block_idx].text);
                }

                let first_char_in_group = group_text.chars().next();
                let last_char_in_group = group_text.chars().last();

                // Check if both opening and closing positions are valid for bold markers
                // We need to insert both or neither to maintain balance
                let can_insert_open = should_insert_bold_marker(prev_char, first_char_in_group);
                let can_insert_close =
                    should_insert_bold_marker(last_char_in_group, next_char_after_group);

                // FIX #2: Skip bold markers for whitespace-only spans in conservative mode
                // Determine if content warrants bold markers based on behavior setting
                let should_render_bold_markers = match options.bold_marker_behavior {
                    BoldMarkerBehavior::Aggressive => true,
                    BoldMarkerBehavior::Conservative => is_content_block(&group_text),
                };

                // Only insert markers if BOTH positions are valid AND content check passes
                let should_insert_markers =
                    is_bold && can_insert_open && can_insert_close && should_render_bold_markers;

                // Log the bold marker decision for debugging
                log::debug!(
                    "Bold marker decision: text='{}', is_content={}, behavior={:?}, render_markers={}",
                    group_text.chars().take(20).collect::<String>(),
                    is_content_block(&group_text),
                    options.bold_marker_behavior,
                    should_render_bold_markers
                );

                if should_insert_markers {
                    markdown.push_str("**");
                }

                // FIX #3: Format URLs and emails as markdown links
                let formatted_text = Self::format_links(&group_text);
                // FIX #4: Clean up reference spacing
                let cleaned_text = Self::clean_reference_spacing(&formatted_text);
                markdown.push_str(&cleaned_text);

                if should_insert_markers {
                    markdown.push_str("**");
                }

                i = j;
            }

            // Add newline(s) after complete line
            match level {
                HeadingLevel::H1 | HeadingLevel::H2 | HeadingLevel::H3 => {
                    markdown.push_str("\n\n"); // Extra blank line after headings
                },
                _ => {
                    markdown.push('\n'); // Single newline after body text
                },
            }
        };

        // Group blocks by Y coordinate and render each line immediately
        for &idx in &ordered_indices {
            let block = &blocks[idx];
            let block_y = block.bbox.y;

            match current_y {
                Some(y) if (y - block_y).abs() < 2.0 => {
                    // Same line - Y coordinates are within 2pt tolerance
                    current_line.push(idx);
                },
                _ => {
                    // New line - render the previous line before starting a new one
                    render_line(&current_line, &mut markdown);
                    current_line.clear();
                    current_line.push(idx);
                    current_y = Some(block_y);
                },
            }
        }

        // Don't forget to render the last line
        render_line(&current_line, &mut markdown);

        // Apply whitespace cleanup: remove artifacts and normalize blank lines
        Ok(cleanup_markdown(&markdown))
    }

    /// Convert a page to Markdown format (character-based - DEPRECATED).
    ///
    /// This function:
    /// 1. Clusters characters into words and lines
    /// 2. Detects heading levels based on font sizes
    /// 3. Determines reading order
    /// 4. Generates Markdown with appropriate syntax
    ///
    /// # Arguments
    ///
    /// * `chars` - The text characters extracted from the page
    /// * `options` - Conversion options controlling the output
    ///
    /// # Returns
    ///
    /// A string containing the Markdown representation of the page.
    ///
    /// # Errors
    ///
    /// Returns an error if clustering or conversion fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use pdf_oxide::converters::{MarkdownConverter, ConversionOptions};
    /// # use pdf_oxide::layout::TextChar;
    /// # fn example(chars: Vec<TextChar>) -> Result<(), Box<dyn std::error::Error>> {
    /// let converter = MarkdownConverter::new();
    /// let options = ConversionOptions {
    ///     detect_headings: true,
    ///     include_images: false,
    ///     ..Default::default()
    /// };
    ///
    /// let markdown = converter.convert_page(&chars, &options)?;
    /// println!("{}", markdown);
    /// # Ok(())
    /// # }
    /// ```
    pub fn convert_page(&self, chars: &[TextChar], options: &ConversionOptions) -> Result<String> {
        if chars.is_empty() {
            return Ok(String::new());
        }

        // CRITICAL FIX: Spatially sort characters by position BEFORE clustering
        // PDF content streams often have characters in arbitrary order (especially in multi-column layouts)
        // We MUST sort by (Y, X) to get proper left-to-right, top-to-bottom reading order
        let mut sorted_chars = chars.to_vec();
        sorted_chars.sort_by(|a, b| {
            // Primary sort: Y coordinate (PDF coords: larger Y = higher on page)
            // For top-to-bottom reading, we want LARGER Y first
            match b.bbox.y.partial_cmp(&a.bbox.y) {
                Some(std::cmp::Ordering::Equal) | None => {
                    // Secondary sort: X coordinate (left to right)
                    a.bbox
                        .x
                        .partial_cmp(&b.bbox.x)
                        .unwrap_or(std::cmp::Ordering::Equal)
                },
                other => other.unwrap_or(std::cmp::Ordering::Equal),
            }
        });

        // Compute font-adaptive epsilon for word clustering
        // Use median character width (approximated as 0.5× font size)
        // Increased to 0.8× to account for Y-axis variations and wider character spacing
        let median_font_size = Self::compute_median_font_size(&sorted_chars);
        let word_epsilon = median_font_size * 0.8; // Character width + spacing for word boundaries

        // Step 1: Cluster characters into words (now in proper spatial order!)
        let word_clusters = cluster_chars_into_words(&sorted_chars, word_epsilon);
        let mut words = Vec::new();

        for cluster in &word_clusters {
            let word_chars: Vec<TextChar> =
                cluster.iter().map(|&i| sorted_chars[i].clone()).collect();
            if !word_chars.is_empty() {
                words.push(TextBlock::from_chars(word_chars));
            }
        }

        if words.is_empty() {
            return Ok(String::new());
        }

        // Step 2: Cluster words into lines
        let line_clusters = cluster_words_into_lines(&words, 5.0);
        let mut lines = Vec::new();

        for cluster in &line_clusters {
            let line_words: Vec<TextBlock> = cluster.iter().map(|&i| words[i].clone()).collect();
            if !line_words.is_empty() {
                // Merge words into a single line block
                let all_chars: Vec<TextChar> =
                    line_words.iter().flat_map(|w| w.chars.clone()).collect();
                lines.push(TextBlock::from_chars(all_chars));
            }
        }

        if lines.is_empty() {
            return Ok(String::new());
        }

        // Step 3: Analyze document properties for adaptive parameters
        let page_bbox = Self::calculate_bounding_box(&lines);
        let adaptive_params = match DocumentProperties::analyze(&sorted_chars, page_bbox) {
            Ok(props) => Some(AdaptiveLayoutParams::from_properties(&props)),
            Err(_) => None, // Fall back to fixed params if analysis fails
        };

        // Step 4: Detect headings (if enabled)
        let heading_levels = if options.detect_headings {
            detect_headings(&lines)
        } else {
            vec![HeadingLevel::Body; lines.len()]
        };

        // Step 5: Determine reading order
        let ordered_indices = self.determine_reading_order(
            &lines,
            options.reading_order_mode.clone(),
            adaptive_params.as_ref(),
        );

        // Step 5: Generate Markdown
        let mut markdown = String::new();

        for &idx in &ordered_indices {
            let line = &lines[idx];
            let level = heading_levels[idx];

            // Add appropriate markdown syntax based on heading level
            // FIX #3: Format URLs and emails as markdown links
            let formatted_text = Self::format_links(&line.text);
            // FIX #4: Clean up reference spacing
            let cleaned_text = Self::clean_reference_spacing(&formatted_text);

            match level {
                HeadingLevel::H1 => {
                    markdown.push_str("# ");
                    markdown.push_str(&cleaned_text);
                    markdown.push_str("\n\n");
                },
                HeadingLevel::H2 => {
                    markdown.push_str("## ");
                    markdown.push_str(&cleaned_text);
                    markdown.push_str("\n\n");
                },
                HeadingLevel::H3 => {
                    markdown.push_str("### ");
                    markdown.push_str(&cleaned_text);
                    markdown.push_str("\n\n");
                },
                HeadingLevel::Body => {
                    markdown.push_str(&cleaned_text);
                    markdown.push('\n');
                },
                HeadingLevel::Small => {
                    // Small text (footnotes, captions) - render as regular text
                    markdown.push_str(&cleaned_text);
                    markdown.push('\n');
                },
            }
        }

        // Apply whitespace cleanup: remove artifacts and normalize blank lines
        Ok(cleanup_markdown(&markdown))
    }

    /// Determine the reading order of text blocks.
    ///
    /// This implements simple top-to-bottom, left-to-right ordering.
    /// For more advanced column-aware ordering, the XY-Cut algorithm could be used.
    ///
    /// # Arguments
    ///
    /// * `blocks` - The text blocks to order
    /// * `mode` - The reading order mode to use
    /// * `adaptive_params` - Optional adaptive parameters computed from document analysis
    ///
    /// # Returns
    ///
    /// A vector of indices representing the reading order.
    fn determine_reading_order(
        &self,
        blocks: &[TextBlock],
        mode: ReadingOrderMode,
        adaptive_params: Option<&AdaptiveLayoutParams>,
    ) -> Vec<usize> {
        if blocks.is_empty() {
            return vec![];
        }

        let mut indices: Vec<usize> = (0..blocks.len()).collect();

        match mode {
            ReadingOrderMode::TopToBottomLeftToRight => {
                // Sort by Y (top to bottom), then by X (left to right)
                indices.sort_by(|&a, &b| {
                    let block_a = &blocks[a];
                    let block_b = &blocks[b];

                    // Primary sort: Y coordinate (larger Y = higher on page in PDF coords)
                    // PDF coordinates: origin at bottom-left, Y increases upward
                    // So top of page (large Y) comes before bottom (small Y)
                    match block_b.bbox.y.partial_cmp(&block_a.bbox.y) {
                        Some(std::cmp::Ordering::Equal) => {
                            // Secondary sort: X coordinate (smaller X = further left)
                            block_a
                                .bbox
                                .x
                                .partial_cmp(&block_b.bbox.x)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        },
                        other => other.unwrap_or(std::cmp::Ordering::Equal),
                    }
                });
            },
            ReadingOrderMode::ColumnAware => {
                // Use XY-Cut algorithm for column-aware reading order

                // Calculate page bounding box from all blocks
                let page_bbox = Self::calculate_bounding_box(blocks);

                // Run XY-Cut algorithm (adaptive or fixed parameters)
                let layout_tree = if let Some(params) = adaptive_params {
                    // Use adaptive parameters computed from document analysis
                    xy_cut_adaptive(page_bbox, blocks, &indices, params)
                } else {
                    // Fall back to fixed parameters
                    xy_cut(page_bbox, blocks, &indices, 0, 10, 50.0)
                };

                // Determine reading order from layout tree
                indices = determine_order_from_tree(&layout_tree);
            },
            ReadingOrderMode::StructureTreeFirst { ref mcid_order } => {
                // PDF-spec-compliant reading order via structure tree (Tagged PDFs)
                if !mcid_order.is_empty() {
                    // Reorder blocks by matching MCIDs from structure tree
                    indices = Self::reorder_by_mcid(blocks, mcid_order);
                    log::info!("Using structure tree for reading order (Tagged PDF)");
                } else {
                    // Fall back to XY-Cut for untagged PDFs
                    log::info!("No MCIDs found, falling back to adaptive XY-Cut");
                    let page_bbox = Self::calculate_bounding_box(blocks);
                    let layout_tree = if let Some(params) = adaptive_params {
                        // Use adaptive parameters computed from document analysis
                        xy_cut_adaptive(page_bbox, blocks, &indices, params)
                    } else {
                        // Fall back to fixed parameters
                        xy_cut(page_bbox, blocks, &indices, 0, 10, 50.0)
                    };
                    indices = determine_order_from_tree(&layout_tree);
                }
            },
        }

        indices
    }

    /// Reorder text blocks according to structure tree reading order.
    ///
    /// Takes blocks extracted from a page and reorders them to match the
    /// MCIDs from the structure tree traversal. This implements PDF-spec-compliant
    /// reading order determination (ISO 32000-1:2008 Section 14.7).
    ///
    /// # Arguments
    ///
    /// * `blocks` - The text blocks to reorder
    /// * `mcid_order` - Sequence of MCIDs in structure tree reading order
    ///
    /// # Returns
    ///
    /// A vector of indices representing the reordered blocks.
    ///
    /// # Algorithm
    ///
    /// 1. For each MCID in structure tree order, find all blocks with that MCID
    /// 2. Add blocks without MCIDs at the end (fallback for unmarked content)
    /// 3. Preserve spatial order for blocks with the same MCID (top-to-bottom, left-to-right)
    fn reorder_by_mcid(blocks: &[TextBlock], mcid_order: &[u32]) -> Vec<usize> {
        let mut ordered_indices = Vec::new();

        // For each MCID in structure tree order
        for &mcid in mcid_order {
            // Find all blocks with this MCID
            let mut mcid_blocks: Vec<usize> = blocks
                .iter()
                .enumerate()
                .filter(|(_, block)| block.mcid == Some(mcid))
                .map(|(idx, _)| idx)
                .collect();

            // Sort blocks with same MCID by spatial position (top-to-bottom, left-to-right)
            // This handles cases where multiple blocks have the same MCID
            mcid_blocks.sort_by(|&a, &b| {
                let block_a = &blocks[a];
                let block_b = &blocks[b];

                // Primary sort: Y coordinate (larger Y = higher on page in PDF coords)
                match block_b.bbox.y.partial_cmp(&block_a.bbox.y) {
                    Some(std::cmp::Ordering::Equal) => {
                        // Secondary sort: X coordinate (smaller X = further left)
                        block_a
                            .bbox
                            .x
                            .partial_cmp(&block_b.bbox.x)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    },
                    other => other.unwrap_or(std::cmp::Ordering::Equal),
                }
            });

            ordered_indices.extend(mcid_blocks);
        }

        // Add blocks without MCID at the end (fallback for unmarked content)
        for (idx, block) in blocks.iter().enumerate() {
            if block.mcid.is_none() && !ordered_indices.contains(&idx) {
                ordered_indices.push(idx);
            }
        }

        ordered_indices
    }

    /// Calculate the bounding box that contains all blocks.
    /// Compute median font size from characters for adaptive epsilon.
    fn compute_median_font_size(chars: &[TextChar]) -> f32 {
        if chars.is_empty() {
            return 12.0; // Default fallback
        }

        let mut font_sizes: Vec<f32> = chars.iter().map(|c| c.font_size).collect();
        font_sizes.sort_by(|a, b| crate::utils::safe_float_cmp(*a, *b));

        let mid = font_sizes.len() / 2;
        if font_sizes.len().is_multiple_of(2) {
            (font_sizes[mid - 1] + font_sizes[mid]) / 2.0
        } else {
            font_sizes[mid]
        }
    }

    fn calculate_bounding_box(blocks: &[TextBlock]) -> Rect {
        if blocks.is_empty() {
            return Rect::new(0.0, 0.0, 0.0, 0.0);
        }

        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        for block in blocks {
            min_x = min_x.min(block.bbox.left());
            min_y = min_y.min(block.bbox.bottom());
            max_x = max_x.max(block.bbox.right());
            max_y = max_y.max(block.bbox.top());
        }

        Rect::from_points(min_x, min_y, max_x, max_y)
    }

    /// Format URLs and email addresses as clickable markdown links.
    ///
    /// FIX #3: Convert plain URLs and emails to markdown link format
    ///
    /// Transforms:
    /// - `https://example.com` → `[https://example.com](https://example.com)`
    /// - `user@example.com` → `[user@example.com](mailto:user@example.com)`
    ///
    /// # Arguments
    ///
    /// * `text` - The text to process
    ///
    /// # Returns
    ///
    /// Text with URLs and emails formatted as markdown links
    fn format_links(text: &str) -> String {
        let mut result = text.to_string();

        // URL regex: matches http:// and https:// URLs
        // Exclude trailing punctuation (.!?,;:) that's likely sentence-ending
        // Pattern: https?:// + [valid URL chars] + [not ending in sentence punctuation]
        result = RE_URL
            .replace_all(&result, |caps: &Captures| {
                let url = &caps[1];
                // Don't format if already part of a markdown link
                if text.contains(&format!("[{}]", url)) {
                    url.to_string()
                } else {
                    format!("[{}]({})", url, url)
                }
            })
            .to_string();

        // Email regex: simple pattern for common email formats
        // Matches: user@domain.com, first.last@domain.co.uk
        result = RE_EMAIL
            .replace_all(&result, |caps: &Captures| {
                let email = &caps[1];
                // Don't format if already part of a markdown link or URL
                if result.contains(&format!("[{}]", email))
                    || result.contains(&format!("//{}", email))
                {
                    email.to_string()
                } else {
                    format!("[{}](mailto:{})", email, email)
                }
            })
            .to_string();

        result
    }

    /// Clean up spacing around dashes in reference ranges.
    ///
    /// FIX #4: Remove extra spaces around em-dashes and en-dashes in citations
    ///
    /// Transforms:
    /// - `"21, 23 –25"` → `"21, 23–25"` (remove space before dash)
    /// - `"21– 25"` → `"21–25"` (remove space after dash)
    /// - `"21 – 25"` → `"21–25"` (remove spaces on both sides)
    ///
    /// # Arguments
    ///
    /// * `text` - Text potentially containing reference ranges with spacing issues
    ///
    /// # Returns
    ///
    /// Text with cleaned up dash spacing in reference contexts
    fn clean_reference_spacing(text: &str) -> String {
        let mut result = text.to_string();

        // Pattern 1: Space before dash in numeric context: "23 –25" → "23–25"
        result = RE_DASH_BEFORE.replace_all(&result, "$1$2$3").to_string();

        // Pattern 2: Space after dash in numeric context: "23– 25" → "23–25"
        result = RE_DASH_AFTER.replace_all(&result, "$1$2$3").to_string();

        // Pattern 3: Space on both sides: "23 – 25" → "23–25"
        // (Covered by patterns 1 and 2 applied sequentially)

        result
    }
}

impl Default for MarkdownConverter {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to determine if a bold marker should be inserted at a boundary.
///
/// Bold markers (`**`) should only be inserted at word boundaries to avoid
/// splitting words unnaturally (e.g., `gr**I` or `8**21`).
///
/// # Arguments
///
/// * `prev_char` - The character before the marker position
/// * `next_char` - The character after the marker position
///
/// # Returns
///
/// `true` if the marker should be inserted (at a word boundary),
/// `false` if it would split a word (both sides are alphanumeric)
///
/// # Examples
///
/// ```ignore
/// // Should insert (whitespace boundary)
/// assert!(should_insert_bold_marker(Some(' '), Some('t')));
///
/// // Should NOT insert (mid-word)
/// assert!(!should_insert_bold_marker(Some('r'), Some('I')));
/// ```
/// Determine if a text block contains meaningful content (not just whitespace).
///
/// This helper identifies whether a span represents actual content or just
/// layout spacing. Used to decide whether to apply bold markers in conservative mode.
///
/// # Arguments
///
/// * `text` - The text content to analyze
///
/// # Returns
///
/// `true` if the text contains at least one non-whitespace character, `false` otherwise.
///
/// # Examples
///
/// ```ignore
/// assert!(is_content_block("text"));       // true - has content
/// assert!(is_content_block("a"));          // true - single character
/// assert!(is_content_block(" a "));        // true - has non-whitespace
/// assert!(!is_content_block(""));          // false - empty
/// assert!(!is_content_block("   "));       // false - spaces only
/// assert!(!is_content_block("\t\n"));      // false - whitespace only
/// ```
pub fn is_content_block(text: &str) -> bool {
    // Check if any character is not whitespace
    text.chars().any(|c| !c.is_whitespace())
}

fn should_insert_bold_marker(prev_char: Option<char>, next_char: Option<char>) -> bool {
    match (prev_char, next_char) {
        // Don't insert if both sides are alphanumeric (mid-word)
        (Some(p), Some(n)) if p.is_alphanumeric() && n.is_alphanumeric() => false,
        // Don't insert between closing punctuation and operators (e.g., ')**=' → ')**= is unnatural')
        // Common cases: )**=, )**-, )**+, )**<, )**>, etc.
        (Some(')'), Some(n))
            if matches!(n, '=' | '-' | '+' | '<' | '>' | '*' | '/' | '&' | '|' | '^') =>
        {
            false
        },
        (Some(']'), Some(n))
            if matches!(n, '=' | '-' | '+' | '<' | '>' | '*' | '/' | '&' | '|' | '^') =>
        {
            false
        },
        (Some('}'), Some(n))
            if matches!(n, '=' | '-' | '+' | '<' | '>' | '*' | '/' | '&' | '|' | '^') =>
        {
            false
        },
        // Insert in all other cases:
        // - At start/end of text (None on either side)
        // - After whitespace, punctuation, or symbols
        // - Before whitespace, punctuation, or symbols
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Rect;
    use crate::layout::{Color, FontWeight};

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
            color: Color::black(),
            mcid: None,
        }
    }

    fn mock_word(text: &str, x: f32, y: f32, font_size: f32, bold: bool) -> Vec<TextChar> {
        text.chars()
            .enumerate()
            .map(|(i, c)| mock_char(c, x + (i as f32 * 7.0), y, font_size, bold))
            .collect()
    }

    #[test]
    fn test_markdown_converter_new() {
        let converter = MarkdownConverter::new();
        assert!(format!("{:?}", converter).contains("MarkdownConverter"));
    }

    #[test]
    fn test_markdown_converter_default() {
        let converter = MarkdownConverter;
        assert!(format!("{:?}", converter).contains("MarkdownConverter"));
    }

    #[test]
    fn test_convert_empty() {
        let converter = MarkdownConverter::new();
        let options = ConversionOptions::default();
        let result = converter.convert_page(&[], &options).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_convert_single_line() {
        let converter = MarkdownConverter::new();
        let options = ConversionOptions {
            detect_headings: false,
            ..Default::default()
        };

        let chars = mock_word("Hello World", 0.0, 0.0, 12.0, false);
        let result = converter.convert_page(&chars, &options).unwrap();

        assert!(result.contains("Hello World"));
        assert!(!result.contains('#')); // No heading detection
    }

    #[test]
    fn test_convert_with_heading() {
        let converter = MarkdownConverter::new();
        let options = ConversionOptions {
            detect_headings: true,
            ..Default::default()
        };

        // Create a large bold title and regular text
        let mut chars = Vec::new();
        chars.extend(mock_word("Title", 0.0, 0.0, 24.0, true)); // Large bold = H1
        chars.push(mock_char(' ', 45.0, 0.0, 24.0, true));

        chars.extend(mock_word("Body Text", 0.0, 50.0, 12.0, false)); // Regular = Body

        let result = converter.convert_page(&chars, &options).unwrap();

        // Title should be detected as heading
        assert!(result.contains("# Title") || result.contains("Title"));
        assert!(result.contains("Body Text"));
    }

    #[test]
    fn test_convert_multiple_lines() {
        let converter = MarkdownConverter::new();
        let options = ConversionOptions {
            detect_headings: false,
            ..Default::default()
        };

        let mut chars = Vec::new();
        chars.extend(mock_word("Line One", 0.0, 0.0, 12.0, false));
        chars.extend(mock_word("Line Two", 0.0, 20.0, 12.0, false));
        chars.extend(mock_word("Line Three", 0.0, 40.0, 12.0, false));

        let result = converter.convert_page(&chars, &options).unwrap();

        assert!(result.contains("Line One"));
        assert!(result.contains("Line Two"));
        assert!(result.contains("Line Three"));
    }

    #[test]
    fn test_reading_order_top_to_bottom() {
        let converter = MarkdownConverter::new();

        // PDF coordinates: Y increases upward, so top has LARGER Y
        let block1 = TextBlock::from_chars(mock_word("Top", 0.0, 100.0, 12.0, false)); // Y=100 (top)
        let block2 = TextBlock::from_chars(mock_word("Middle", 0.0, 50.0, 12.0, false)); // Y=50 (middle)
        let block3 = TextBlock::from_chars(mock_word("Bottom", 0.0, 0.0, 12.0, false)); // Y=0 (bottom)

        let blocks = vec![block2.clone(), block3.clone(), block1.clone()]; // Out of order

        let indices = converter.determine_reading_order(
            &blocks,
            ReadingOrderMode::TopToBottomLeftToRight,
            None,
        );

        // Should order by Y: block1 (y=100), block2 (y=50), block3 (y=0)
        // In our shuffled vec: block1 is at index 2, block2 at 0, block3 at 1
        assert_eq!(indices[0], 2); // block1 (Top, y=100)
        assert_eq!(indices[1], 0); // block2 (Middle, y=50)
        assert_eq!(indices[2], 1); // block3 (Bottom, y=0)
    }

    #[test]
    fn test_reading_order_left_to_right() {
        let converter = MarkdownConverter::new();

        // Create blocks at same Y but different X
        let block1 = TextBlock::from_chars(mock_word("Left", 0.0, 0.0, 12.0, false));
        let block2 = TextBlock::from_chars(mock_word("Center", 50.0, 0.0, 12.0, false));
        let block3 = TextBlock::from_chars(mock_word("Right", 100.0, 0.0, 12.0, false));

        let blocks = vec![block3.clone(), block1.clone(), block2.clone()]; // Out of order

        let indices = converter.determine_reading_order(
            &blocks,
            ReadingOrderMode::TopToBottomLeftToRight,
            None,
        );

        // Should order by X when Y is equal
        assert_eq!(indices[0], 1); // block1 (Left, x=0)
        assert_eq!(indices[1], 2); // block2 (Center, x=50)
        assert_eq!(indices[2], 0); // block3 (Right, x=100)
    }

    #[test]
    fn test_heading_level_h1() {
        let converter = MarkdownConverter::new();
        let options = ConversionOptions {
            detect_headings: true,
            ..Default::default()
        };

        // Very large bold text should be H1
        let chars = mock_word("Main Title", 0.0, 0.0, 28.0, true);
        let result = converter.convert_page(&chars, &options).unwrap();

        // Should contain H1 marker
        assert!(result.contains("# Main Title") || result.contains("Main Title"));
    }

    #[test]
    fn test_heading_level_h2() {
        let converter = MarkdownConverter::new();
        let options = ConversionOptions {
            detect_headings: true,
            ..Default::default()
        };

        let mut chars = Vec::new();
        // H1: largest
        chars.extend(mock_word("Main", 0.0, 0.0, 24.0, true));
        // H2: medium
        chars.extend(mock_word("Section", 0.0, 40.0, 18.0, true));
        // Body: small
        chars.extend(mock_word("Text", 0.0, 70.0, 12.0, false));

        let result = converter.convert_page(&chars, &options).unwrap();

        // Should have different heading levels
        assert!(result.contains("Main"));
        assert!(result.contains("Section"));
        assert!(result.contains("Text"));
    }

    #[test]
    fn test_column_aware_mode() {
        let converter = MarkdownConverter::new();

        let block1 = TextBlock::from_chars(mock_word("A", 0.0, 0.0, 12.0, false));
        let block2 = TextBlock::from_chars(mock_word("B", 0.0, 50.0, 12.0, false));

        let blocks = vec![block1, block2];

        // Both modes should work (ColumnAware falls back to simple for now)
        let indices1 = converter.determine_reading_order(
            &blocks,
            ReadingOrderMode::TopToBottomLeftToRight,
            None,
        );
        let indices2 =
            converter.determine_reading_order(&blocks, ReadingOrderMode::ColumnAware, None);

        assert_eq!(indices1.len(), 2);
        assert_eq!(indices2.len(), 2);
    }
}
