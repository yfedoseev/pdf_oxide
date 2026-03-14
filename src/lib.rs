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
//! The fastest PDF library for Python and Rust. 0.8ms mean text extraction — 5× faster than
//! PyMuPDF, 15× faster than pypdf, 29× faster than pdfplumber. 100% pass rate on 3,830
//! real-world PDFs. MIT licensed. A drop-in PyMuPDF alternative with no AGPL restrictions.
//!
//! ## Performance (v0.3.10)
//!
//! Benchmarked against 14 text extraction libraries on 3,830 PDFs from 3 public test suites
//! (veraPDF, Mozilla pdf.js, DARPA SafeDocs). Single-thread, 60s timeout, no warm-up.
//!
//! ### Python PDF Libraries
//!
//! | Library | Mean | Pass Rate | License |
//! |---------|------|-----------|---------|
//! | **pdf_oxide** | **0.8ms** | **100%** | **MIT** |
//! | PyMuPDF | 4.6ms | 99.3% | AGPL-3.0 |
//! | pypdfium2 | 4.1ms | 99.2% | Apache-2.0 |
//! | pymupdf4llm | 55.5ms | 99.1% | AGPL-3.0 |
//! | pdftext | 7.3ms | 99.0% | GPL-3.0 |
//! | pdfminer | 16.8ms | 98.8% | MIT |
//! | pdfplumber | 23.2ms | 98.8% | MIT |
//! | markitdown | 108.8ms | 98.6% | MIT |
//! | pypdf | 12.1ms | 98.4% | BSD-3 |
//!
//! ### Rust PDF Libraries
//!
//! | Library | Mean | Pass Rate | Text Extraction |
//! |---------|------|-----------|-----------------|
//! | **pdf_oxide** | **0.8ms** | **100%** | **Built-in** |
//! | oxidize_pdf | 13.5ms | 99.1% | Basic |
//! | unpdf | 2.8ms | 95.1% | Basic |
//! | pdf_extract | 4.08ms | 91.5% | Basic |
//! | lopdf | 0.3ms | 80.2% | No built-in extraction |
//!
//! 99.5% text quality parity vs PyMuPDF and pypdfium2 across the full corpus.
//! Full benchmark details: <https://pdf.oxide.fyi/docs/performance>
//!
//! ## Core Features
//!
//! ### Reading & Extraction
//! - **Text Extraction**: Character, span, and page-level with font metadata and bounding boxes
//! - **Reading Order**: 4 pluggable strategies (XY-Cut, Structure Tree, Geometric, Simple)
//! - **Complex Scripts**: RTL (Arabic/Hebrew), CJK (Japanese/Korean/Chinese), Devanagari, Thai
//! - **Format Conversion**: PDF → Markdown, HTML, PlainText
//! - **Image Extraction**: Content streams, Form XObjects, inline images
//! - **Forms & Annotations**: Read/write form fields, all annotation types, bookmarks
//! - **Text Search**: Regex and case-insensitive search with page-level results
//!
//! ### Writing & Creation
//! - **PDF Generation**: Fluent DocumentBuilder API for programmatic PDF creation
//! - **Format Conversion**: Markdown → PDF, HTML → PDF, Plain Text → PDF, Image → PDF
//! - **Advanced Graphics**: Path operations, image embedding, table generation
//! - **Font Embedding**: Automatic font subsetting for compact output
//! - **Interactive Forms**: Fillable forms with text fields, checkboxes, radio buttons, dropdowns
//! - **QR Codes & Barcodes**: Code128, EAN-13, UPC-A (feature flag: `barcodes`)
//!
//! ### Editing
//! - **DOM-like API**: Query and modify PDF content with strongly-typed wrappers
//! - **Element Modification**: Find and replace text, modify images, paths, tables
//! - **Page Operations**: Add, remove, reorder, merge, rotate, crop pages
//! - **Encryption**: AES-256, password protection
//! - **Incremental Saves**: Efficient appending without full rewrite
//!
//! ### Compliance
//! - **PDF/A**: Validation and conversion
//! - **PDF/UA**: Accessibility checks
//! - **PDF/X**: Print production validation
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
//! ```
//!
//! ## Quick Start - Python
//!
//! ```text
//! from pdf_oxide import PdfDocument
//!
//! # Open and extract with automatic reading order
//! doc = PdfDocument("paper.pdf")
//! markdown = doc.to_markdown(0)
//! print(markdown)
//! ```
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

// Document structure
/// Core annotation types and enums per PDF spec
pub mod annotation_types;
pub mod annotations;
/// Content elements for PDF generation
pub mod elements;
pub mod outline;
/// PDF logical structure (Tagged PDFs)
pub mod structure;

// Format converters
pub mod converters;

// Pipeline architecture for text extraction
pub mod pipeline;

// PDF writing/creation (v0.3.0)
pub mod writer;

// FDF/XFDF form data export (v0.3.3)
pub mod fdf;

// XFA forms support (v0.3.2)
pub mod xfa;

// PDF editing (v0.3.0)
pub mod editor;

// Text search (v0.3.0)
pub mod search;

// Page rendering to images (optional, v0.3.0)
#[cfg(feature = "rendering")]
#[cfg_attr(docsrs, doc(cfg(feature = "rendering")))]
pub mod rendering;

// Debug visualization for PDF analysis (optional, v0.3.0)
#[cfg(feature = "rendering")]
#[cfg_attr(docsrs, doc(cfg(feature = "rendering")))]
pub mod debug;

// Digital signatures (optional, v0.3.0)
#[cfg(feature = "signatures")]
#[cfg_attr(docsrs, doc(cfg(feature = "signatures")))]
pub mod signatures;

// Parallel page extraction (optional, v0.3.10)
#[cfg(feature = "parallel")]
#[cfg_attr(docsrs, doc(cfg(feature = "parallel")))]
pub mod parallel;

// Batch processing API (v0.3.10)
#[cfg(not(target_arch = "wasm32"))]
pub mod batch;

// PDF/A compliance validation (v0.3.0)
pub mod compliance;

// High-level API (v0.3.0)
pub mod api;

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
#[cfg(any(target_arch = "wasm32", test))]
#[cfg(feature = "wasm")]
pub mod wasm;

// Re-exports
pub use annotation_types::{
    AnnotationBorderStyle, AnnotationColor, AnnotationFlags, AnnotationSubtype, BorderEffectStyle,
    BorderStyleType, CaretSymbol, FileAttachmentIcon, FreeTextIntent, HighlightMode,
    LineEndingStyle, QuadPoint, ReplyType, StampType, TextAlignment, TextAnnotationIcon,
    TextMarkupType, WidgetFieldType,
};
pub use annotations::{Annotation, LinkAction, LinkDestination};
pub use config::{DocumentType, ExtractionProfile};
pub use document::{ExtractedImageRef, ImageFormat, PdfDocument};
pub use error::{Error, Result};
pub use outline::{Destination, OutlineItem};

// Global font cache for batch processing
pub use fonts::global_cache::{
    clear_global_font_cache, global_font_cache_stats, set_global_font_cache_capacity,
};

#[cfg(feature = "parallel")]
pub use parallel::{extract_all_markdown_parallel, extract_all_text_parallel, ParallelExtractor};

// Internal utilities
pub(crate) mod utils {
    //! Internal utility functions for the library.

    use std::cmp::Ordering;

    /// Safely truncate a string to at most `max_bytes` from the start
    /// without splitting a multi-byte UTF-8 character.
    ///
    /// Returns the full string if it is shorter than `max_bytes`.
    /// When truncation lands inside a multi-byte character, the boundary
    /// is rounded **down** to the nearest char boundary (floor).
    #[inline]
    pub fn safe_prefix(s: &str, max_bytes: usize) -> &str {
        if s.len() <= max_bytes {
            return s;
        }
        let mut end = max_bytes;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }

    /// Safely take the last `max_bytes` of a string without splitting
    /// a multi-byte UTF-8 character.
    ///
    /// Returns the full string if it is shorter than `max_bytes`.
    /// When the computed start offset lands inside a multi-byte character,
    /// the boundary is rounded **up** to the nearest char boundary (ceil).
    #[inline]
    pub fn safe_suffix(s: &str, max_bytes: usize) -> &str {
        if s.len() <= max_bytes {
            return s;
        }
        let start = s.len() - max_bytes;
        let mut safe_start = start;
        while safe_start < s.len() && !s.is_char_boundary(safe_start) {
            safe_start += 1;
        }
        &s[safe_start..]
    }

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

        /// Verify that sort_by using safe_float_cmp never panics with NaN values.
        /// This is a regression test for the "total order" panic that affected 42
        /// PDFs across 5 test datasets (issue found in v0.3.11-pre).
        #[test]
        fn test_sort_with_nan_does_not_panic() {
            let mut values = [3.0_f32, f32::NAN, 1.0, f32::NAN, 2.0, f32::NAN, 0.5];
            values.sort_by(|a, b| safe_float_cmp(*a, *b));
            // NaN values should sort to the end (NaN > all numbers)
            assert!(values[0..4].iter().all(|v| !v.is_nan()));
            assert!(values[4..].iter().all(|v| v.is_nan()));
        }

        /// Verify transitivity: if a < b and b < c then a < c.
        /// The previous `partial_cmp().unwrap_or(Equal)` pattern violated this
        /// when NaN was involved, causing Rust's sort to panic.
        #[test]
        fn test_safe_float_cmp_transitivity() {
            let a = 1.0_f32;
            let b = 2.0_f32;
            let nan = f32::NAN;

            // a < b
            assert_eq!(safe_float_cmp(a, b), Ordering::Less);
            // b < NaN
            assert_eq!(safe_float_cmp(b, nan), Ordering::Less);
            // Therefore a < NaN (transitivity)
            assert_eq!(safe_float_cmp(a, nan), Ordering::Less);
        }

        /// Sort a large array with mixed NaN/normal values to stress-test.
        #[test]
        fn test_sort_stress_with_nan() {
            let mut values: Vec<f32> = (0..100).map(|i| i as f32).collect();
            // Insert NaN at various positions
            for i in (0..100).step_by(7) {
                values[i] = f32::NAN;
            }
            // Must not panic
            values.sort_by(|a, b| safe_float_cmp(*a, *b));
        }

        #[test]
        fn test_safe_prefix_ascii() {
            assert_eq!(safe_prefix("hello", 3), "hel");
            assert_eq!(safe_prefix("hello", 10), "hello");
            assert_eq!(safe_prefix("", 5), "");
            assert_eq!(safe_prefix("hi", 0), "");
        }

        #[test]
        fn test_safe_prefix_multibyte() {
            let text = "✚✳★✵"; // 4 × 3-byte chars = 12 bytes
            assert_eq!(safe_prefix(text, 10), "✚✳★"); // rounds down from 10 to 9
            assert_eq!(safe_prefix(text, 9), "✚✳★"); // exact boundary
            assert_eq!(safe_prefix(text, 12), "✚✳★✵"); // full string
        }

        #[test]
        fn test_safe_suffix_ascii() {
            assert_eq!(safe_suffix("hello", 3), "llo");
            assert_eq!(safe_suffix("hello", 10), "hello");
            assert_eq!(safe_suffix("", 5), "");
            assert_eq!(safe_suffix("hi", 0), "");
        }

        #[test]
        fn test_safe_suffix_multibyte() {
            let text = "AB✚✳★✵"; // 14 bytes: A(0) B(1) ✚(2..5) ✳(5..8) ★(8..11) ✵(11..14)
                                 // 14 - 10 = 4, byte 4 is inside ✚ → rounds up to 5
            assert_eq!(safe_suffix(text, 10), "✳★✵");
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
