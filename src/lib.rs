// Allow some clippy lints that are too pedantic for this project
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::enum_variant_names)]
#![allow(clippy::wrong_self_convention)]
#![allow(clippy::explicit_counter_loop)]
#![allow(clippy::doc_overindented_list_items)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::redundant_guards)]
#![allow(clippy::regex_creation_in_loops)]
#![allow(clippy::manual_find)]
#![allow(clippy::match_like_matches_macro)]
// Allow unused for tests
#![cfg_attr(test, allow(dead_code))]
#![cfg_attr(test, allow(unused_variables))]

//! # PDF Oxide
//!
//! Production-grade PDF parsing in Rust: 47.9× faster than PyMuPDF4LLM with PDF spec compliance.
//!
//! ## Core Features (v0.2.0)
//!
//! - **PDF Spec Compliance**: ISO 32000-1:2008 sections 9, 14.7-14.8
//! - **Text Extraction**: 5-level character-to-Unicode priority (§9.10.2)
//! - **Reading Order**: 4 pluggable strategies (XY-Cut, Structure Tree, Geometric, Simple)
//! - **Font Support**: 70-80% character recovery with CID-to-GID mapping
//! - **OCR Support**: DBNet++ detection + SVTR recognition with smart auto-detection
//! - **Complex Scripts**: RTL (Arabic/Hebrew), CJK (Japanese/Korean/Chinese), Devanagari, Thai
//! - **Format Conversion**: Markdown, HTML, PlainText, TOC
//! - **Pluggable Architecture**: Trait-based design for extensibility
//! - **Python Bindings**: Full API via PyO3
//!
//! ## Planned for v0.3.0+
//!
//! - **Bidirectional**: PDF generation from Markdown/HTML (v0.3.0)
//! - **Tables**: Extract and generate tables (v0.4.0)
//! - **Forms**: Interactive fillable form support (v0.4.0)
//! - **Advanced**: Figures, citations, annotations, accessibility (v0.5.0+)
//!
//! ## Quick Start - Rust
//!
//! ```ignore
//! use pdf_oxide::PdfDocument;
//! use pdf_oxide::pipeline::{TextPipeline, TextPipelineConfig};
//! use pdf_oxide::pipeline::converters::MarkdownOutputConverter;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Open a PDF
//! let mut doc = PdfDocument::open("paper.pdf")?;
//!
//! // Extract text with reading order (multi-column support)
//! let spans = doc.extract_spans(0)?;
//! let config = TextPipelineConfig::default();
//! let pipeline = TextPipeline::with_config(config.clone());
//! let ordered_spans = pipeline.process(spans, Default::default())?;
//!
//! // Convert to Markdown
//! let converter = MarkdownOutputConverter::new();
//! let markdown = converter.convert(&ordered_spans, &config)?;
//! println!("{}", markdown);
//! # Ok(())
//! # }
//! ```ignore
//!
//! ## Quick Start - Python
//!
//! ```python
//! from pdf_oxide import PdfDocument
//!
//! # Open and extract with automatic reading order
//! doc = PdfDocument("paper.pdf")
//! markdown = doc.to_markdown(0)
//! print(markdown)
//! ```ignore
//!
//! ## License
//!
//! Licensed under either of:
//!
//! * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
//! * MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)
//!
//! at your option.

#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// Error handling
pub mod error;

// Core PDF parsing
pub mod document;
pub mod lexer;
pub mod object;
pub mod objstm;
pub mod parser;
/// Parser configuration options
pub mod parser_config;
pub mod xref;
pub mod xref_reconstruction;

// Stream decoders
pub mod decoders;

// Encryption support
pub mod encryption;

// Layout analysis
pub mod geometry;
pub mod layout;

// Text extraction
pub mod content;
pub mod extractors;
pub mod fonts;
pub mod text;

// Image extraction
pub mod images;

// Document structure
pub mod annotations;
pub mod outline;
/// PDF logical structure (Tagged PDFs)
pub mod structure;

// Format converters
pub mod converters;

// Pipeline architecture for text extraction
pub mod pipeline;

// Re-export specific types from pipeline for use by converters
pub use pipeline::XYCutStrategy;

// Configuration
pub mod config;

// Hybrid classical + ML orchestration
pub mod hybrid;

// OCR - PaddleOCR via ONNX Runtime (optional)
#[cfg(feature = "ocr")]
#[cfg_attr(docsrs, doc(cfg(feature = "ocr")))]
pub mod ocr;

// Python bindings (optional)
#[cfg(feature = "python")]
mod python;

// WASM bindings (optional)
#[cfg(target_arch = "wasm32")]
#[cfg(feature = "wasm")]
pub mod wasm;

// Re-exports
pub use annotations::{Annotation, LinkAction, LinkDestination};
pub use config::{DocumentType, ExtractionProfile};
pub use document::{ExtractedImageRef, ImageFormat, PdfDocument};
pub use error::{Error, Result};
pub use outline::{Destination, OutlineItem};

// Internal utilities
pub(crate) mod utils {
    //! Internal utility functions for the library.

    use std::cmp::Ordering;

    /// Safely compare two floating point numbers, handling NaN cases.
    ///
    /// NaN values are treated as equal to each other and greater than all other values.
    /// This ensures that sorting operations never panic due to NaN comparisons.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use std::cmp::Ordering;
    /// # use pdf_oxide::utils::safe_float_cmp;
    /// assert_eq!(safe_float_cmp(1.0, 2.0), Ordering::Less);
    /// assert_eq!(safe_float_cmp(2.0, 1.0), Ordering::Greater);
    /// assert_eq!(safe_float_cmp(1.0, 1.0), Ordering::Equal);
    ///
    /// // NaN handling
    /// assert_eq!(safe_float_cmp(f32::NAN, f32::NAN), Ordering::Equal);
    /// assert_eq!(safe_float_cmp(f32::NAN, 1.0), Ordering::Greater);
    /// assert_eq!(safe_float_cmp(1.0, f32::NAN), Ordering::Less);
    /// ```
    #[inline]
    pub fn safe_float_cmp(a: f32, b: f32) -> Ordering {
        match (a.is_nan(), b.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater, // NaN > all numbers
            (false, true) => Ordering::Less,    // all numbers < NaN
            (false, false) => {
                // Both are normal numbers, safe to unwrap
                a.partial_cmp(&b).unwrap()
            },
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_safe_float_cmp_normal() {
            assert_eq!(safe_float_cmp(1.0, 2.0), Ordering::Less);
            assert_eq!(safe_float_cmp(2.0, 1.0), Ordering::Greater);
            assert_eq!(safe_float_cmp(1.5, 1.5), Ordering::Equal);
        }

        #[test]
        fn test_safe_float_cmp_nan() {
            assert_eq!(safe_float_cmp(f32::NAN, f32::NAN), Ordering::Equal);
            assert_eq!(safe_float_cmp(f32::NAN, 0.0), Ordering::Greater);
            assert_eq!(safe_float_cmp(0.0, f32::NAN), Ordering::Less);
        }

        #[test]
        fn test_safe_float_cmp_infinity() {
            assert_eq!(safe_float_cmp(f32::INFINITY, f32::INFINITY), Ordering::Equal);
            assert_eq!(safe_float_cmp(f32::INFINITY, 1.0), Ordering::Greater);
            assert_eq!(safe_float_cmp(f32::NEG_INFINITY, f32::INFINITY), Ordering::Less);
        }
    }
}

// Version info
/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Library name
pub const NAME: &str = env!("CARGO_PKG_NAME");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        // VERSION is populated from CARGO_PKG_VERSION at compile time
        assert!(VERSION.starts_with("0."));
    }

    #[test]
    fn test_name() {
        assert_eq!(NAME, "pdf_oxide");
    }
}
