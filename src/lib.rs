// SPDX-License-Identifier: MIT OR Apache-2.0
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
#![allow(clippy::collapsible_match)]
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
//! use pdf_oxide::pipeline::converters::OutputConverter;
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

// Glibc 2.34 compatibility (#416): LLVM may emit calls to __memcmpeq@GLIBC_2.35,
// which does not exist in glibc 2.34 (Amazon Linux 2023, some Ubuntu 22.04 builds).
// `fips` and `legacy-crypto` are mutually exclusive: FIPS 140-3 forbids MD5
// and RC4, which `legacy-crypto` pulls in. Build FIPS without legacy crypto:
//   cargo build --no-default-features --features fips,icc
#[cfg(all(feature = "fips", feature = "legacy-crypto"))]
compile_error!(
    "Features `fips` and `legacy-crypto` are mutually exclusive. \
     FIPS 140-3 forbids MD5 (pulled in by `legacy-crypto`). \
     Build with: --no-default-features --features fips,icc"
);

// A weak stub redirecting to plain memcmp satisfies the reference on older glibc;
// glibc 2.35's own definition wins when available. global_asm! works with both
// GNU ld and lld, unlike --defsym which lld rejects for PLT-resolved symbols.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
core::arch::global_asm!(
    ".weak __memcmpeq",
    ".type __memcmpeq, @function",
    "__memcmpeq:",
    "jmp memcmp@PLT",
);

// Error handling
pub mod error;

// General-purpose caching utilities
pub(crate) mod cache;

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

// PDF function evaluators (Type 4 PostScript calculator)
pub mod functions;

// Colour management (ICC profile handling)
pub mod color;

// Pluggable cryptographic backend (FIPS / sovereign-jurisdiction
// providers). Issue #236.
pub mod crypto;

// Encryption support
pub mod encryption;

// Layout analysis
pub mod geometry;
pub mod layout;

// Text extraction
pub mod content;
pub mod extractors;
pub mod fonts;
pub mod optional_content;
pub mod text;

// Document structure
/// Core annotation types and enums per PDF spec
pub mod annotation_types;
pub mod annotations;
/// Content elements for PDF generation
pub mod elements;
/// Cross-platform-safe filename slug helpers (shared, pure).
pub mod filename;
pub mod outline;
/// True/destructive redaction + document sanitization (#231).
pub mod redaction;
/// Split a PDF into multiple PDFs at outline (bookmark) boundaries (#482).
pub mod split_bookmarks;
/// PDF logical structure (Tagged PDFs)
pub mod structure;

/// Structured per-page extraction (`extract_structured`, #536)
pub mod structured;

// Format converters
pub mod converters;

// Pipeline architecture for text extraction
pub mod pipeline;

// PDF writing/creation (v0.3.0)
pub mod writer;

// HTML + CSS → PDF pipeline (v0.3.35, issue #248). Hand-rolled tokenizer,
// parser, selector matcher, cascade, layout glue, paginator, and paint
// emitter. MIT/Apache-only deps (no MPL); see deny.toml + the v0.3.35
// pre-flight audit doc for the rationale.
pub mod html_css;

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

// OCR - PaddleOCR via a pluggable inference backend (optional).
// Native ONNX Runtime when `ocr` is on; otherwise the pure-Rust
// `tract` backend (`ocr-tract`, which `ml` implies and the
// browser/Deno/edge `wasm-ocr` build uses — issue #524). Exposing OCR
// wherever the tract backend is available costs only the small OCR
// module itself and keeps it host-testable without a native dylib.
#[cfg(any(feature = "ocr", feature = "ocr-tract"))]
#[cfg_attr(docsrs, doc(cfg(any(feature = "ocr", feature = "ocr-tract"))))]
pub mod ocr;

// C FFI for Go, Node.js, C# bindings (not available on wasm32)
#[cfg(not(target_arch = "wasm32"))]
pub mod ffi;

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
pub use document::{ExtractedImageRef, ImageFormat, PdfDocument, ReadingOrder};
pub use error::{Error, Result};
pub use extractors::images::{PdfFilter, PdfImageHandle};
pub use layout::PageText;
pub use outline::{Destination, OutlineItem};
pub use redaction::{
    redact_content_stream, Classification, FontInfoMetrics, OcgPolicy, RedactionOptions,
    RedactionRegion, RedactionReport, RegionSet,
};
pub use structured::{RegionRole, StructuredPage, StructuredRegion};

// Global font cache for batch processing
pub use fonts::global_cache::{
    clear_global_font_cache, global_font_cache_stats, set_global_font_cache_capacity,
};

// Global CMap cache management
pub use fonts::cmap::{clear_cmap_cache, cmap_cache_size};

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

    /// Y-band tolerance used by `row_aware_span_cmp`.
    ///
    /// Two spans whose top-Y differs by less than this amount are treated
    /// as lying on the same row. Chosen to absorb typographic baseline
    /// jitter for 10-12pt body text and glyph-cluster offsets in CJK
    /// fonts without merging adjacent 14pt-leading lines.
    pub const ROW_BAND_TOLERANCE_PT: f32 = 3.0;

    /// Row-aware reading-order comparator for spans.
    ///
    /// Sorts primarily by "row band" (top-Y quantized to
    /// `ROW_BAND_TOLERANCE_PT`, larger Y first per PDF Spec ISO 32000-1:2008
    /// §8.3.2.3) and secondarily by X (left-to-right within a row). This
    /// keeps tabular layouts where cells in the same logical row have
    /// slightly different Y values (font-metric jitter, superscripts, CJK
    /// glyph centering) from being interleaved by a strict Y sort.
    ///
    /// Uses `i32` band keys so the ordering is a valid total order —
    /// comparing raw Y values with tolerance is non-transitive and would
    /// break `sort_by`.
    #[inline]
    pub fn row_aware_span_cmp(a_y: f32, a_x: f32, b_y: f32, b_x: f32) -> Ordering {
        // Non-finite Y (NaN/±Inf) cannot be quantized into an i32 band —
        // `as i32` saturates, collapsing distinct non-finite values into
        // the same band and reordering them unpredictably against finite
        // spans. Fall back to `safe_float_cmp` so non-finite values follow
        // the same NaN-last / total-order policy used everywhere else.
        if !a_y.is_finite() || !b_y.is_finite() {
            return safe_float_cmp(b_y, a_y).then_with(|| safe_float_cmp(a_x, b_x));
        }
        let band_a = (a_y / ROW_BAND_TOLERANCE_PT).round() as i32;
        let band_b = (b_y / ROW_BAND_TOLERANCE_PT).round() as i32;
        // Larger Y = higher on page → descending band order.
        match band_b.cmp(&band_a) {
            Ordering::Equal => safe_float_cmp(a_x, b_x),
            other => other,
        }
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

    /// Sort `items` into row-band reading order, computing each element's band
    /// key once instead of re-quantizing on every `row_aware_span_cmp`
    /// comparison.
    ///
    /// When all `y`/`x` are finite this is a cached-key stable sort with the
    /// same order as `sort_by(row_aware_span_cmp)` (band descending, then `x`
    /// ascending — `f32::total_cmp` equals `safe_float_cmp` for finite values,
    /// and both are stable on ties). Otherwise it falls back to the comparator
    /// so the NaN/±∞ policy is unchanged.
    pub fn sort_by_row_band<T>(
        items: &mut [T],
        get_y: impl Fn(&T) -> f32,
        get_x: impl Fn(&T) -> f32,
    ) {
        let all_finite = items
            .iter()
            .all(|it| get_y(it).is_finite() && get_x(it).is_finite());
        if !all_finite {
            items.sort_by(|a, b| row_aware_span_cmp(get_y(a), get_x(a), get_y(b), get_x(b)));
            return;
        }
        // Cached-key stable sort. `total_cmp` matches `safe_float_cmp` for the
        // finite values we gated on above.
        items.sort_by_cached_key(|it| {
            let band = (get_y(it) / ROW_BAND_TOLERANCE_PT).round() as i32;
            // Reverse band → larger Y (higher on page) first, matching the
            // comparator's `band_b.cmp(&band_a)`.
            (std::cmp::Reverse(band), F32Ord(get_x(it)))
        });
    }

    /// Total-order wrapper over `f32` for use as a sort key. For finite values
    /// `total_cmp` is identical to `safe_float_cmp` / `partial_cmp`.
    #[derive(Clone, Copy, PartialEq)]
    struct F32Ord(f32);
    impl Eq for F32Ord {}
    impl PartialOrd for F32Ord {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for F32Ord {
        fn cmp(&self, other: &Self) -> Ordering {
            self.0.total_cmp(&other.0)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// The cached-key sort must produce the identical permutation to
        /// `sort_by(row_aware_span_cmp)` on finite inputs.
        #[test]
        fn test_sort_by_row_band_matches_comparator() {
            // Deterministic pseudo-random spans (no rng in tests).
            let raw: Vec<(f32, f32)> = (0..500)
                .map(|i| {
                    let y = ((i * 37 % 113) as f32) * 1.3;
                    let x = ((i * 71 % 97) as f32) * 2.1;
                    (y, x)
                })
                .collect();
            let mut a = raw.clone();
            let mut b = raw.clone();
            sort_by_row_band(&mut a, |t| t.0, |t| t.1);
            b.sort_by(|p, q| row_aware_span_cmp(p.0, p.1, q.0, q.1));
            assert_eq!(a, b, "cached-key sort must match the comparator permutation");
        }

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

        /// Cells in the same tabular row with slightly-different Y values
        /// must stay together and be ordered by X, not interleaved with
        /// cells from other rows.
        #[test]
        fn test_row_aware_span_cmp_tolerates_y_jitter() {
            // Row 1 at y ≈ 100 with small per-cell jitter.
            // Row 2 at y ≈ 86 (14pt leading below).
            // A strict Y sort would interleave them because some row-1
            // cells have lower Y than some row-2 cells.
            #[derive(Debug, Clone, Copy)]
            struct Cell {
                y: f32,
                x: f32,
                id: &'static str,
            }
            let mut cells = [
                Cell {
                    y: 100.5,
                    x: 50.0,
                    id: "r1-c1",
                },
                Cell {
                    y: 99.7,
                    x: 150.0,
                    id: "r1-c2",
                },
                Cell {
                    y: 100.2,
                    x: 250.0,
                    id: "r1-c3",
                },
                Cell {
                    y: 86.4,
                    x: 50.0,
                    id: "r2-c1",
                },
                Cell {
                    y: 85.8,
                    x: 150.0,
                    id: "r2-c2",
                },
                Cell {
                    y: 86.1,
                    x: 250.0,
                    id: "r2-c3",
                },
            ];
            cells.sort_by(|a, b| row_aware_span_cmp(a.y, a.x, b.y, b.x));
            let order: Vec<&str> = cells.iter().map(|c| c.id).collect();
            assert_eq!(
                order,
                vec!["r1-c1", "r1-c2", "r1-c3", "r2-c1", "r2-c2", "r2-c3"],
                "cells from the same row must stay contiguous and X-sorted"
            );
        }

        /// Row-aware comparator must still put distinct-leading rows in
        /// top-to-bottom reading order.
        #[test]
        fn test_row_aware_span_cmp_distinct_rows_descending() {
            let mut rows = [
                (100.0f32, 0.0f32, "top"),
                (50.0, 0.0, "middle"),
                (10.0, 0.0, "bottom"),
            ];
            rows.sort_by(|a, b| row_aware_span_cmp(a.0, a.1, b.0, b.1));
            assert_eq!(rows[0].2, "top");
            assert_eq!(rows[1].2, "middle");
            assert_eq!(rows[2].2, "bottom");
        }

        /// The comparator is used by sort_by, which requires a valid total
        /// order. Run a randomized stress test to confirm no transitivity
        /// panics.
        #[test]
        fn test_row_aware_span_cmp_is_total_order() {
            let mut v: Vec<(f32, f32)> = (0..200)
                .map(|i| ((i as f32) * 0.73, ((i * 17) % 500) as f32))
                .collect();
            v.sort_by(|a, b| row_aware_span_cmp(a.0, a.1, b.0, b.1));
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
