//! PDF text extraction pipeline with clean abstraction layers.
//!
//! This module provides a PDF-spec-compliant pipeline for text extraction:
//!
//! ```text
//! PDF File
//!     ↓
//! [TextExtractor] (content stream → TextSpan[])
//!     ↓
//! TextSpan[] (single intermediate representation)
//!     ↓
//! [ReadingOrderStrategy] (pluggable ordering)
//!     ↓
//! OrderedTextSpan[]
//!     ↓
//! [OutputConverter] (Markdown/HTML/Text)
//!     ↓
//! Output String
//! ```
//!
//! # Key Design Principles
//!
//! 1. **Single Intermediate Representation**: TextSpan is the only representation
//!    between PDF parsing and output conversion.
//!
//! 2. **PDF Spec Compliance**: Per ISO 32000-1:2008, text strings from Tj/TJ
//!    operators are preserved as-is. No linguistic heuristics for word segmentation.
//!
//! 3. **Pluggable Strategies**: Reading order and output conversion are trait-based
//!    for extensibility.
//!
//! 4. **Unified Configuration**: All settings in TextPipelineConfig.

pub mod config;
pub mod converters;
pub mod logging;
pub mod metrics;
// pub mod input_parsers;  // Keep disabled - for PDF creation feature later
pub mod ordered_span;
pub mod reading_order;
pub mod text_processing;

// Re-export main types
pub use config::{
    BoldMarkerBehavior, LogLevel, OutputConfig, ReadingOrderConfig, ReadingOrderStrategyType,
    SpacingConfig, TextPipelineConfig, TjThresholdConfig, WordBoundaryMode,
};
pub use converters::{
    HtmlOutputConverter, MarkdownOutputConverter, OutputConverter, PlainTextConverter,
};
pub use logging::{
    extract_log_debug, extract_log_error, extract_log_info, extract_log_trace, extract_log_warn,
};
pub use metrics::{BatchMetrics, ExtractionMetrics};
pub use ordered_span::{OrderedSpans, OrderedTextSpan};
pub use reading_order::{ReadingOrderContext, ReadingOrderStrategy, XYCutStrategy};
pub use text_processing::WhitespaceNormalizer;

use crate::error::Result;
use crate::layout::TextSpan;
use reading_order::create_strategy;

/// The text extraction pipeline - orchestrates the full flow.
///
/// This is the main entry point for the new pipeline architecture.
/// It processes TextSpans through reading order determination and
/// prepares them for output conversion.
pub struct TextPipeline {
    config: TextPipelineConfig,
    reading_order_strategy: Box<dyn ReadingOrderStrategy>,
}

impl TextPipeline {
    /// Create a new pipeline with default configuration.
    pub fn new() -> Self {
        Self::with_config(TextPipelineConfig::default())
    }

    /// Create a pipeline with custom configuration.
    pub fn with_config(config: TextPipelineConfig) -> Self {
        let strategy = create_strategy(&config.reading_order);
        Self {
            config,
            reading_order_strategy: strategy,
        }
    }

    /// Process spans through the pipeline.
    ///
    /// 1. Apply reading order strategy
    /// 2. Return ordered spans ready for conversion
    pub fn process(
        &self,
        spans: Vec<TextSpan>,
        context: ReadingOrderContext,
    ) -> Result<Vec<OrderedTextSpan>> {
        self.reading_order_strategy.apply(spans, &context)
    }

    /// Get the current configuration.
    pub fn config(&self) -> &TextPipelineConfig {
        &self.config
    }
}

impl Default for TextPipeline {
    fn default() -> Self {
        Self::new()
    }
}
