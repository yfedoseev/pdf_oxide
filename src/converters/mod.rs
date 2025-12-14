//! Format converters for PDF documents.
//!
//! This module provides functionality to convert PDF pages to different formats:
//! - **Markdown**: Semantic text with headings, paragraphs, and images
//! - **HTML**: Both semantic and layout-preserved modes
//! - **Plain text**: Simple text extraction
//!
//! # Examples
//!
//! ```no_run
//! use pdf_oxide::PdfDocument;
//! use pdf_oxide::converters::ConversionOptions;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut doc = PdfDocument::open("paper.pdf")?;
//!
//! // Convert to Markdown with heading detection
//! let options = ConversionOptions {
//!     detect_headings: true,
//!     ..Default::default()
//! };
//! let markdown = doc.to_markdown(0, &options)?;
//!
//! // Convert to semantic HTML
//! let html = doc.to_html(0, &options)?;
//!
//! // Convert to layout-preserved HTML
//! let layout_options = ConversionOptions {
//!     preserve_layout: true,
//!     ..Default::default()
//! };
//! let layout_html = doc.to_html(0, &layout_options)?;
//! # Ok(())
//! # }
//! ```

pub mod html;
pub mod markdown;
pub mod table_formatter;
pub mod text_post_processor;
pub mod whitespace;

// Re-export main types
#[allow(deprecated)]
pub use html::HtmlConverter;
#[allow(deprecated)]
pub use markdown::MarkdownConverter;
pub use table_formatter::MarkdownTableFormatter;
pub use text_post_processor::TextPostProcessor;
pub use whitespace::{cleanup_markdown, normalize_whitespace, remove_page_artifacts};

// Re-export BoldMarkerBehavior from pipeline config (single source of truth)
pub use crate::pipeline::config::BoldMarkerBehavior;

/// Configuration for table formatting in markdown.
///
/// All formatting parameters are configurable with no magic numbers.
#[derive(Debug, Clone)]
pub struct TableFormatConfig {
    /// Include markdown table header separator (default: true)
    pub include_header_separator: bool,
    /// Spaces around cell content (default: 1)
    pub cell_padding: usize,
    /// Minimum column width in characters (default: 3)
    pub min_column_width: usize,
    /// Merge adjacent empty cells (default: true)
    pub merge_adjacent_empty_cells: bool,
    /// Preserve bold/italic formatting in cells (default: true)
    pub preserve_cell_formatting: bool,
    /// Text to use for empty cells (default: "-")
    pub empty_cell_text: String,
}

impl TableFormatConfig {
    /// Create a standard markdown table configuration.
    pub fn default() -> Self {
        Self {
            include_header_separator: true,
            cell_padding: 1,
            min_column_width: 3,
            merge_adjacent_empty_cells: true,
            preserve_cell_formatting: true,
            empty_cell_text: "-".to_string(),
        }
    }

    /// Create a compact markdown table configuration.
    pub fn compact() -> Self {
        Self {
            include_header_separator: true,
            cell_padding: 0,
            min_column_width: 1,
            merge_adjacent_empty_cells: true,
            preserve_cell_formatting: false,
            empty_cell_text: String::new(),
        }
    }

    /// Create a detailed markdown table configuration.
    pub fn detailed() -> Self {
        Self {
            include_header_separator: true,
            cell_padding: 2,
            min_column_width: 5,
            merge_adjacent_empty_cells: false,
            preserve_cell_formatting: true,
            empty_cell_text: "—".to_string(),
        }
    }

    /// Create a custom markdown table configuration.
    pub fn custom() -> Self {
        Self::default()
    }

    /// Set cell padding (builder pattern).
    pub fn with_cell_padding(mut self, padding: usize) -> Self {
        self.cell_padding = padding;
        self
    }

    /// Set minimum column width (builder pattern).
    pub fn with_min_column_width(mut self, width: usize) -> Self {
        self.min_column_width = width;
        self
    }

    /// Set empty cell text (builder pattern).
    pub fn with_empty_cell_text(mut self, text: &str) -> Self {
        self.empty_cell_text = text.to_string();
        self
    }
}

impl Default for TableFormatConfig {
    fn default() -> Self {
        TableFormatConfig::default()
    }
}

/// Options for converting PDF pages to different formats.
///
/// These options control how the conversion is performed, including
/// layout preservation, heading detection, image handling, etc.
///
/// # Examples
///
/// ```
/// use pdf_oxide::converters::{BoldMarkerBehavior, ConversionOptions, ReadingOrderMode};
///
/// // Default options
/// let opts = ConversionOptions::default();
///
/// // Custom options
/// let opts = ConversionOptions {
///     preserve_layout: true,
///     detect_headings: false,
///     extract_tables: false,
///     include_images: true,
///     image_output_dir: Some("images/".to_string()),
///     reading_order_mode: ReadingOrderMode::ColumnAware,
///     bold_marker_behavior: BoldMarkerBehavior::Conservative,
///     table_detection_config: None,
/// };
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ConversionOptions {
    /// Preserve exact layout with CSS positioning (HTML only).
    ///
    /// When true, generates HTML with absolute positioning to match the PDF layout.
    /// When false, generates semantic HTML with natural flow.
    pub preserve_layout: bool,

    /// Automatically detect headings based on font size and weight.
    ///
    /// When true, uses font clustering to identify heading levels (H1, H2, H3).
    /// When false, treats all text as paragraphs.
    pub detect_headings: bool,

    /// Extract tables from the document.
    ///
    /// Note: Table extraction is currently not fully implemented.
    pub extract_tables: bool,

    /// Include images in the output.
    ///
    /// When true, images are included as Markdown image syntax or HTML img tags.
    /// When false, images are omitted from the output.
    pub include_images: bool,

    /// Directory path for saving extracted images.
    ///
    /// If None, images are referenced but not saved.
    /// If Some(path), images are saved to the specified directory.
    pub image_output_dir: Option<String>,

    /// Reading order determination mode.
    ///
    /// Controls how text blocks are ordered in the output.
    pub reading_order_mode: ReadingOrderMode,

    /// Control how bold markers are applied in markdown conversion.
    ///
    /// Determines whether bold formatting markers are applied to whitespace-only
    /// content (Aggressive) or only to content-bearing text (Conservative).
    /// See BoldMarkerBehavior for details.
    pub bold_marker_behavior: BoldMarkerBehavior,

    /// Configuration for spatial table detection.
    ///
    /// If None, uses default configuration.
    /// Only applies when extract_tables = true.
    pub table_detection_config: Option<crate::structure::TableDetectionConfig>,
}

impl Default for ConversionOptions {
    /// Create default conversion options.
    ///
    /// Defaults:
    /// - preserve_layout: false (semantic mode)
    /// - detect_headings: true (enabled for proper markdown output)
    /// - extract_tables: false
    /// - include_images: true
    /// - image_output_dir: None
    /// - reading_order_mode: StructureTreeFirst (PDF-spec-compliant for Tagged PDFs, falls back to XY-Cut for untagged)
    /// - bold_marker_behavior: Conservative (no bold markers for whitespace-only content)
    /// - table_detection_config: None (uses defaults when table detection is enabled)
    fn default() -> Self {
        Self {
            preserve_layout: false,
            detect_headings: true,
            extract_tables: false,
            include_images: true,
            image_output_dir: None,
            reading_order_mode: ReadingOrderMode::StructureTreeFirst { mcid_order: vec![] },
            bold_marker_behavior: BoldMarkerBehavior::Conservative,
            table_detection_config: None,
        }
    }
}

impl ConversionOptions {
    /// Enable table detection with custom configuration.
    ///
    /// Sets extract_tables = true and uses the provided configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::converters::ConversionOptions;
    /// use pdf_oxide::structure::TableDetectionConfig;
    ///
    /// let config = TableDetectionConfig::strict();
    /// let opts = ConversionOptions::default().with_table_detection(config);
    ///
    /// assert!(opts.extract_tables);
    /// assert!(opts.table_detection_config.is_some());
    /// ```
    pub fn with_table_detection(mut self, config: crate::structure::TableDetectionConfig) -> Self {
        self.extract_tables = true;
        self.table_detection_config = Some(config);
        self
    }

    /// Enable table detection with default configuration.
    ///
    /// Sets extract_tables = true and table_detection_config = None,
    /// which will use the default TableDetectionConfig when detection runs.
    ///
    /// # Examples
    ///
    /// ```
    /// use pdf_oxide::converters::ConversionOptions;
    ///
    /// let opts = ConversionOptions::default().with_default_table_detection();
    ///
    /// assert!(opts.extract_tables);
    /// assert!(opts.table_detection_config.is_none());
    /// ```
    pub fn with_default_table_detection(mut self) -> Self {
        self.extract_tables = true;
        self.table_detection_config = None;
        self
    }
}

/// Reading order determination mode for text blocks.
///
/// Determines how text blocks are ordered when converting to output formats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadingOrderMode {
    /// Simple top-to-bottom, left-to-right ordering.
    ///
    /// Sorts all blocks by Y coordinate (top to bottom), then by X coordinate (left to right).
    /// This works well for single-column documents.
    TopToBottomLeftToRight,

    /// Column-aware reading order.
    ///
    /// Uses the XY-Cut algorithm to detect columns and determines proper reading order
    /// across multiple columns. This works better for multi-column documents.
    ColumnAware,

    /// Structure tree first, with fallback to column-aware.
    ///
    /// For Tagged PDFs: Uses the PDF logical structure tree (ISO 32000-1:2008 Section 14.7)
    /// to determine reading order via Marked Content IDs (MCIDs). This is the PDF-spec-compliant
    /// approach and provides perfect reading order for Tagged PDFs.
    ///
    /// For Untagged PDFs: Falls back to ColumnAware (XY-Cut algorithm).
    ///
    /// This mode requires passing MCID reading order through ConversionOptions.mcid_order.
    StructureTreeFirst {
        /// Reading order as a sequence of MCIDs from structure tree traversal.
        /// If empty, falls back to ColumnAware mode.
        mcid_order: Vec<u32>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversion_options_default() {
        let opts = ConversionOptions::default();
        assert!(!opts.preserve_layout);
        assert!(opts.detect_headings);
        assert!(!opts.extract_tables);
        assert!(opts.include_images);
        assert_eq!(opts.image_output_dir, None);
        assert_eq!(
            opts.reading_order_mode,
            ReadingOrderMode::StructureTreeFirst { mcid_order: vec![] }
        );
    }

    #[test]
    fn test_conversion_options_custom() {
        let opts = ConversionOptions {
            preserve_layout: true,
            detect_headings: false,
            extract_tables: false,
            include_images: false,
            image_output_dir: Some("output/".to_string()),
            reading_order_mode: ReadingOrderMode::ColumnAware,
            bold_marker_behavior: BoldMarkerBehavior::Aggressive,
            table_detection_config: None,
        };

        assert!(opts.preserve_layout);
        assert!(!opts.detect_headings);
        assert!(!opts.include_images);
        assert_eq!(opts.image_output_dir, Some("output/".to_string()));
        assert_eq!(opts.reading_order_mode, ReadingOrderMode::ColumnAware);
        assert_eq!(opts.bold_marker_behavior, BoldMarkerBehavior::Aggressive);
        assert!(opts.table_detection_config.is_none());
    }

    #[test]
    fn test_reading_order_mode_equality() {
        assert_eq!(
            ReadingOrderMode::TopToBottomLeftToRight,
            ReadingOrderMode::TopToBottomLeftToRight
        );
        assert_ne!(ReadingOrderMode::TopToBottomLeftToRight, ReadingOrderMode::ColumnAware);
    }

    #[test]
    fn test_conversion_options_clone() {
        let opts1 = ConversionOptions::default();
        let opts2 = opts1.clone();
        assert_eq!(opts1, opts2);
    }

    #[test]
    fn test_conversion_options_debug() {
        let opts = ConversionOptions::default();
        let debug_str = format!("{:?}", opts);
        assert!(debug_str.contains("ConversionOptions"));
    }

    #[test]
    fn test_bold_marker_behavior_default() {
        assert_eq!(BoldMarkerBehavior::default(), BoldMarkerBehavior::Conservative);
    }

    #[test]
    fn test_bold_marker_behavior_equality() {
        assert_eq!(BoldMarkerBehavior::Conservative, BoldMarkerBehavior::Conservative);
        assert_eq!(BoldMarkerBehavior::Aggressive, BoldMarkerBehavior::Aggressive);
        assert_ne!(BoldMarkerBehavior::Conservative, BoldMarkerBehavior::Aggressive);
    }

    #[test]
    fn test_bold_marker_behavior_copy_clone() {
        let behavior = BoldMarkerBehavior::Aggressive;
        let copied = behavior;
        assert_eq!(behavior, copied);
    }

    #[test]
    fn test_with_default_table_detection() {
        let opts = ConversionOptions::default().with_default_table_detection();
        assert!(opts.extract_tables);
        assert!(opts.table_detection_config.is_none());
    }

    #[test]
    fn test_with_table_detection() {
        let config = crate::structure::TableDetectionConfig::strict();
        let opts = ConversionOptions::default().with_table_detection(config);
        assert!(opts.extract_tables);
        assert!(opts.table_detection_config.is_some());
        let cfg = opts.table_detection_config.unwrap();
        assert_eq!(cfg.min_table_columns, 3);
        assert_eq!(cfg.column_tolerance, 2.0);
    }

    #[test]
    fn test_conversion_options_default_table_config() {
        let opts = ConversionOptions::default();
        assert!(!opts.extract_tables);
        assert!(opts.table_detection_config.is_none());
    }
}
