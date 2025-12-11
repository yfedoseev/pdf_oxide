//! Unified configuration for the text extraction pipeline.
//!
//! This module consolidates configuration that was previously scattered across:
//! - TextExtractionConfig
//! - SpanMergingConfig
//! - SpacingConfig
//! - ConversionOptions

/// Unified configuration for the text extraction pipeline.
///
/// This replaces the scattered configuration across multiple modules
/// and provides a single configuration point for the entire pipeline.
#[derive(Debug, Clone)]
pub struct TextPipelineConfig {
    /// Spacing configuration for span merging
    pub spacing: SpacingConfig,

    /// TJ offset threshold configuration
    pub tj_threshold: TjThresholdConfig,

    /// Reading order strategy to use
    pub reading_order: ReadingOrderConfig,

    /// Output formatting options
    pub output: OutputConfig,
}

impl Default for TextPipelineConfig {
    fn default() -> Self {
        Self {
            spacing: SpacingConfig::default(),
            tj_threshold: TjThresholdConfig::default(),
            reading_order: ReadingOrderConfig::default(),
            output: OutputConfig::default(),
        }
    }
}

impl TextPipelineConfig {
    /// Create config with pdfplumber-compatible defaults.
    ///
    /// Uses word_margin = 0.1 and simple reading order.
    pub fn pdfplumber_compatible() -> Self {
        Self {
            spacing: SpacingConfig { word_margin: 0.1 },
            tj_threshold: TjThresholdConfig {
                space_insertion_threshold: -120.0,
                use_adaptive: false,
            },
            reading_order: ReadingOrderConfig {
                strategy: ReadingOrderStrategyType::Simple,
            },
            output: OutputConfig::default(),
        }
    }

    /// Create from legacy ConversionOptions for backwards compatibility.
    ///
    /// Maps old `ConversionOptions` to new `TextPipelineConfig` to maintain
    /// backwards compatibility while migrating to the new pipeline architecture.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use pdf_oxide::converters::ConversionOptions;
    /// use pdf_oxide::pipeline::config::TextPipelineConfig;
    ///
    /// let options = ConversionOptions::default();
    /// let config = TextPipelineConfig::from_conversion_options(&options);
    /// ```
    pub fn from_conversion_options(opts: &crate::converters::ConversionOptions) -> Self {
        use crate::converters::ReadingOrderMode;
        use crate::converters::BoldMarkerBehavior as OldBMB;

        // Map reading order mode to strategy type
        let strategy = match &opts.reading_order_mode {
            ReadingOrderMode::TopToBottomLeftToRight => ReadingOrderStrategyType::Simple,
            ReadingOrderMode::ColumnAware => ReadingOrderStrategyType::XYCut,
            ReadingOrderMode::StructureTreeFirst { .. } => ReadingOrderStrategyType::StructureTreeFirst,
        };

        // Map BoldMarkerBehavior enum (both have same variants)
        let bold_marker_behavior = match opts.bold_marker_behavior {
            OldBMB::Aggressive => BoldMarkerBehavior::Aggressive,
            OldBMB::Conservative => BoldMarkerBehavior::Conservative,
        };

        Self {
            spacing: SpacingConfig::default(),
            tj_threshold: TjThresholdConfig::default(),
            reading_order: ReadingOrderConfig { strategy },
            output: OutputConfig {
                detect_headings: opts.detect_headings,
                include_images: opts.include_images,
                bold_marker_behavior,
                preserve_layout: opts.preserve_layout,
                extract_tables: opts.extract_tables,
                image_output_dir: opts.image_output_dir.clone(),
            },
        }
    }
}

/// Configuration for geometric spacing decisions.
///
/// Controls how gaps between text spans are interpreted.
#[derive(Debug, Clone, Copy)]
pub struct SpacingConfig {
    /// Word margin as ratio of character size.
    ///
    /// If the gap between spans is less than this ratio times the average
    /// character width, they are considered part of the same word.
    ///
    /// Default: 0.1 (pdfplumber default)
    pub word_margin: f32,
}

impl Default for SpacingConfig {
    fn default() -> Self {
        Self { word_margin: 0.1 }
    }
}

/// Configuration for TJ offset threshold.
///
/// Controls how TJ array offsets are interpreted for word boundary detection.
#[derive(Debug, Clone)]
pub struct TjThresholdConfig {
    /// Static threshold for TJ offsets (in text space units).
    ///
    /// Offsets more negative than this value indicate a word boundary.
    /// Default: -120.0 (pdfplumber default)
    pub space_insertion_threshold: f32,

    /// Whether to use adaptive threshold based on font metrics.
    ///
    /// When true, the threshold is computed dynamically based on
    /// the statistical distribution of gaps in the document.
    pub use_adaptive: bool,
}

impl Default for TjThresholdConfig {
    fn default() -> Self {
        Self {
            space_insertion_threshold: -120.0,
            use_adaptive: false,
        }
    }
}

/// Configuration for reading order strategy.
#[derive(Debug, Clone)]
pub struct ReadingOrderConfig {
    /// The reading order strategy to use.
    pub strategy: ReadingOrderStrategyType,
}

impl Default for ReadingOrderConfig {
    fn default() -> Self {
        Self {
            strategy: ReadingOrderStrategyType::StructureTreeFirst,
        }
    }
}

/// Available reading order strategy types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadingOrderStrategyType {
    /// Use structure tree MCIDs for reading order, fallback to geometric.
    ///
    /// This is the PDF-spec-compliant approach for Tagged PDFs
    /// (ISO 32000-1:2008 Section 14.7).
    StructureTreeFirst,

    /// Column-aware geometric analysis.
    ///
    /// Uses horizontal gap detection to identify columns and
    /// processes each column top-to-bottom.
    Geometric,

    /// Recursive XY-Cut spatial partitioning.
    ///
    /// Uses projection profiles to detect columns in complex layouts
    /// like newspapers and academic papers. More sophisticated than
    /// simple gap detection (ISO 32000-1:2008 Section 9.4).
    XYCut,

    /// Simple top-to-bottom, left-to-right ordering.
    ///
    /// Sorts spans by Y coordinate (descending) then X coordinate (ascending).
    Simple,
}

/// Output formatting configuration.
#[derive(Debug, Clone)]
pub struct OutputConfig {
    /// Whether to detect and format headings.
    ///
    /// When true, larger text is formatted as markdown headings.
    /// Default: false (disabled for PDF spec compliance)
    pub detect_headings: bool,

    /// Whether to include images in output.
    pub include_images: bool,

    /// Bold marker behavior for whitespace-only spans.
    pub bold_marker_behavior: BoldMarkerBehavior,

    /// Preserve document layout using whitespace (Markdown) or CSS positioning (HTML).
    ///
    /// When enabled:
    /// - Markdown converter preserves column alignment via whitespace
    /// - HTML converter uses CSS `position:absolute` for spatial preservation
    /// - Plain text converter preserves spacing
    ///
    /// Default: false
    pub preserve_layout: bool,

    /// Extract table structures and format as markdown tables or HTML tables.
    ///
    /// When enabled:
    /// - Grid-aligned text is detected and formatted as tables
    /// - Markdown converter outputs markdown table syntax
    /// - HTML converter outputs HTML table elements
    ///
    /// Default: false
    pub extract_tables: bool,

    /// Directory to save extracted images.
    ///
    /// When Some(path):
    /// - Images are extracted to the specified directory
    /// - Image references use the provided path
    ///
    /// When None:
    /// - Images are referenced but not extracted to disk
    ///
    /// Default: None
    pub image_output_dir: Option<String>,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            detect_headings: false, // Disabled for spec compliance
            include_images: true,
            bold_marker_behavior: BoldMarkerBehavior::Conservative,
            preserve_layout: false,
            extract_tables: false,
            image_output_dir: None,
        }
    }
}

/// Behavior for bold markers around whitespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoldMarkerBehavior {
    /// Don't apply bold markers to whitespace-only spans.
    ///
    /// This prevents patterns like "** **" in output.
    Conservative,

    /// Apply bold markers to any span with bold font weight.
    Aggressive,
}

impl Default for BoldMarkerBehavior {
    /// Default behavior is Conservative mode.
    fn default() -> Self {
        Self::Conservative
    }
}
