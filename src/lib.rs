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

//! # PDFoxide
//!
//! High-performance PDF parsing and conversion library built in Rust with Python bindings.
//!
//! ## Features (v0.1.0)
//!
//! - **PDF Parsing**: Parse PDF 1.0-1.7 documents with full encryption support
//! - **Text Extraction**: Extract text with accurate Unicode mapping and ToUnicode CMap support
//! - **Layout Analysis**: Multi-column detection with XY-Cut and DBSCAN clustering
//! - **Format Conversion**: Convert to Markdown, HTML, and plain text
//! - **Image Extraction**: Extract embedded images (JPEG, PNG) with metadata
//! - **Structure Tree**: Parse PDF logical structure (tagged PDFs)
//! - **Annotations**: Extract PDF annotations, comments, and highlights
//! - **Bookmarks**: Extract document outline/bookmarks with hierarchy
//! - **Python Bindings**: Easy-to-use Python API via PyO3
//!
//! ## Planned for v1.0
//!
//! - **ML Integration**: Advanced layout analysis with ONNX models
//! - **Table Detection**: Production-ready ML-based table extraction
//! - **OCR**: Text extraction from scanned PDFs via PaddleOCR (ONNX)
//! - **WASM Target**: Run in browsers via WebAssembly
//! - **Digital Signatures**: Signature verification and creation
//!
//! ## Quick Start
//!
//! ```ignore
//! use pdf_oxide::PdfDocument;
//! use pdf_oxide::converters::ConversionOptions;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Open a PDF
//! let mut doc = PdfDocument::open("paper.pdf")?;
//!
//! // Extract text from first page
//! let text = doc.extract_text(0)?;
//! println!("{}", text);
//!
//! // Convert to Markdown
//! let options = ConversionOptions::default();
//! let markdown = doc.to_markdown(0, &options)?;
//!
//! // Extract images
//! let images = doc.extract_images(0)?;
//! # Ok(())
//! # }
//! ```ignore
//!
//! ## Python Usage
//!
//! ```python
//! from pdf_oxide import PdfDocument
//!
//! doc = PdfDocument("paper.pdf")
//! text = doc.extract_text(0)
//! markdown = doc.to_markdown(0)
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

// Core PDF parsing (Phase 1)
pub mod document;
pub mod lexer;
pub mod object;
pub mod objstm;
pub mod parser;
/// Parser configuration options
pub mod parser_config;
pub mod xref;
pub mod xref_reconstruction;

// Stream decoders (Phase 2)
pub mod decoders;

// Encryption support (Phase 8)
pub mod encryption;

// Layout analysis (Phase 3)
pub mod geometry;
pub mod layout;

// Text extraction (Phase 4)
pub mod content;
pub mod extractors;
pub mod fonts;

// Image extraction (Phase 5)
pub mod images;

// Document structure (Phase 9)
pub mod annotations;
pub mod outline;
/// PDF logical structure (Tagged PDFs)
pub mod structure;

// Converters (Phase 6)
pub mod converters;

// Configuration
pub mod config;

// ML integration (Phase 8 - optional)
#[cfg(feature = "ml")]
#[cfg_attr(docsrs, doc(cfg(feature = "ml")))]
pub mod ml;

// OCR integration (optional) - PaddleOCR via ONNX Runtime
#[cfg(feature = "ocr")]
#[cfg_attr(docsrs, doc(cfg(feature = "ocr")))]
pub mod ocr;

// Hybrid classical + ML orchestration (Phase 8)
pub mod hybrid;

// Python bindings (Phase 7 - optional)
#[cfg(feature = "python")]
mod python;

// WASM bindings (Phase 9E - optional)
#[cfg(target_arch = "wasm32")]
#[cfg(feature = "wasm")]
pub mod wasm;

// Re-exports
pub use annotations::{Annotation, LinkAction, LinkDestination};
pub use config::PdfConfig;
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
