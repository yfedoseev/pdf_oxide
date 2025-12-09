//! Output converters for the text extraction pipeline.
//!
//! This module provides the OutputConverter trait and implementations for
//! converting ordered text spans to various output formats.
//!
//! # Available Converters
//!
//! - [`MarkdownOutputConverter`]: Convert to Markdown format
//! - [`HtmlOutputConverter`]: Convert to HTML format
//! - [`PlainTextConverter`]: Convert to plain text
//!
//! # Example
//!
//! ```ignore
//! use pdf_oxide::pipeline::converters::{OutputConverter, MarkdownOutputConverter};
//! use pdf_oxide::pipeline::TextPipelineConfig;
//!
//! let converter = MarkdownOutputConverter::new();
//! let config = TextPipelineConfig::default();
//! let output = converter.convert(&ordered_spans, &config)?;
//! ```

mod html;
mod markdown;
mod plain_text;
pub mod toc_detector;

pub use html::HtmlOutputConverter;
pub use markdown::MarkdownOutputConverter;
pub use plain_text::PlainTextConverter;
pub use toc_detector::{TocDetector, TocEntry};

use crate::error::Result;
use crate::pipeline::{OrderedTextSpan, TextPipelineConfig};

/// Trait for converting ordered text spans to output formats.
///
/// Implementations transform a sequence of ordered text spans into a specific
/// output format (Markdown, HTML, plain text, etc.).
///
/// This trait provides a clean abstraction layer between the PDF extraction
/// pipeline and the output generation, following the PDF spec compliance goal
/// of separating PDF representation from output formatting.
pub trait OutputConverter: Send + Sync {
    /// Convert ordered spans to the target format.
    ///
    /// # Arguments
    ///
    /// * `spans` - Ordered text spans from the reading order strategy
    /// * `config` - Pipeline configuration affecting output formatting
    ///
    /// # Returns
    ///
    /// The formatted output string.
    fn convert(&self, spans: &[OrderedTextSpan], config: &TextPipelineConfig) -> Result<String>;

    /// Return the name of this converter for debugging.
    fn name(&self) -> &'static str;

    /// Return the MIME type for the output format.
    fn mime_type(&self) -> &'static str;
}

/// Create a converter based on the output format name.
pub fn create_converter(format: &str) -> Option<Box<dyn OutputConverter>> {
    match format.to_lowercase().as_str() {
        "markdown" | "md" => Some(Box::new(MarkdownOutputConverter::new())),
        "html" => Some(Box::new(HtmlOutputConverter::new())),
        "text" | "plain" | "txt" => Some(Box::new(PlainTextConverter::new())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_converter_markdown() {
        let converter = create_converter("markdown").unwrap();
        assert_eq!(converter.name(), "MarkdownOutputConverter");
        assert_eq!(converter.mime_type(), "text/markdown");
    }

    #[test]
    fn test_create_converter_html() {
        let converter = create_converter("html").unwrap();
        assert_eq!(converter.name(), "HtmlOutputConverter");
        assert_eq!(converter.mime_type(), "text/html");
    }

    #[test]
    fn test_create_converter_text() {
        let converter = create_converter("text").unwrap();
        assert_eq!(converter.name(), "PlainTextConverter");
        assert_eq!(converter.mime_type(), "text/plain");
    }

    #[test]
    fn test_create_converter_unknown() {
        assert!(create_converter("unknown").is_none());
    }
}
