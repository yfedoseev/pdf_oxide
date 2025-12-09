//! OCR (Optical Character Recognition) module for scanned PDF text extraction.
//!
//! This module provides PaddleOCR-based text extraction for scanned PDFs using
//! ONNX Runtime for CPU-only inference. It integrates seamlessly with the existing
//! text extraction pipeline.
//!
//! # Features
//!
//! - **Auto-detect scanned pages**: Automatically identify pages that need OCR
//! - **Unified output**: OCR results match the format of native text extraction
//! - **Style detection**: Infer font sizes and heading styles from OCR geometry
//! - **Fast CPU inference**: Target < 1 second per A4 page on modern CPU
//!
//! # Architecture
//!
//! The OCR pipeline consists of:
//! 1. **Preprocessing**: Image resizing, normalization, tensor conversion
//! 2. **Detection**: DBNet++ model finds text regions (bounding boxes)
//! 3. **Recognition**: SVTR model reads text from cropped regions
//! 4. **Postprocessing**: Convert OCR results to TextSpan format
//!
//! # Example
//!
//! ```ignore
//! use pdf_oxide::{PdfDocument, ocr::OcrEngine};
//!
//! let mut doc = PdfDocument::open("scanned.pdf")?;
//! let engine = OcrEngine::new()?;
//!
//! // Check if page needs OCR
//! if ocr::needs_ocr(&doc, 0)? {
//!     let result = engine.ocr_page(&mut doc, 0)?;
//!     for span in result.spans {
//!         println!("{} at {:?}", span.text, span.bbox);
//!     }
//! }
//! ```

// Sub-modules
mod config;
mod detector;
mod engine;
mod error;
mod postprocessor;
mod preprocessor;
mod recognizer;

// Re-exports
pub use config::{OcrConfig, OcrConfigBuilder};
pub use detector::TextDetector;
pub use engine::{OcrEngine, OcrOutput, OcrSpan};
pub use error::OcrError;
pub use postprocessor::DetectedBox;
pub use preprocessor::{crop_text_region, preprocess_for_detection, preprocess_for_recognition};
pub use recognizer::{RecognitionResult, TextRecognizer};

// High-level OCR functions are exported at module level (needs_ocr, ocr_page, etc.)

use crate::{PdfDocument, Result};

/// Check if a PDF page needs OCR (is a scanned page).
///
/// A page is considered "scanned" if:
/// 1. It has no native text (or very little)
/// 2. It contains images (typically a full-page scan)
///
/// # Arguments
///
/// * `doc` - The PDF document
/// * `page` - Page number (0-indexed)
///
/// # Returns
///
/// `true` if the page likely needs OCR, `false` otherwise.
///
/// # Example
///
/// ```ignore
/// use pdf_oxide::{PdfDocument, ocr};
///
/// let mut doc = PdfDocument::open("document.pdf")?;
/// if ocr::needs_ocr(&doc, 0)? {
///     println!("Page 0 is scanned, needs OCR");
/// }
/// ```
pub fn needs_ocr(doc: &mut PdfDocument, page: usize) -> Result<bool> {
    // Check for native text first
    let native_text = doc.extract_text(page).unwrap_or_default();
    let has_substantial_text = native_text.trim().len() > 50;

    if has_substantial_text {
        return Ok(false);
    }

    // Check if page has images (scanned pages typically have a full-page image)
    let images = doc.extract_images(page)?;
    if images.is_empty() {
        return Ok(false);
    }

    // If there's no substantial text but there are images, likely a scanned page
    // For more accurate detection, we could check if there's a single large image
    // that covers most of the page, but this simple heuristic works for most cases.
    Ok(true)
}

/// OCR text extraction options.
#[derive(Debug, Clone)]
pub struct OcrExtractOptions {
    /// OCR configuration
    pub config: OcrConfig,
    /// Scale factor for coordinate conversion (image DPI / 72.0)
    /// Default: 300.0 / 72.0 ≈ 4.17 (assumes 300 DPI scan)
    pub scale: f32,
    /// Whether to fall back to native text if OCR fails
    pub fallback_to_native: bool,
}

impl Default for OcrExtractOptions {
    fn default() -> Self {
        Self {
            config: OcrConfig::default(),
            scale: 300.0 / 72.0, // Assume 300 DPI scanned document
            fallback_to_native: true,
        }
    }
}

impl OcrExtractOptions {
    /// Create options with a custom DPI.
    pub fn with_dpi(dpi: f32) -> Self {
        Self {
            scale: dpi / 72.0,
            ..Default::default()
        }
    }
}

/// OCR a single page of a PDF document.
///
/// This function:
/// 1. Extracts the largest image from the page (assumed to be the scan)
/// 2. Converts it to a DynamicImage
/// 3. Runs OCR on the image
/// 4. Returns the recognized text
///
/// # Arguments
///
/// * `doc` - The PDF document
/// * `page` - Page number (0-indexed)
/// * `engine` - The OCR engine to use
/// * `options` - OCR extraction options
///
/// # Returns
///
/// The recognized text from the page.
///
/// # Example
///
/// ```ignore
/// use pdf_oxide::{PdfDocument, ocr::{self, OcrEngine, OcrConfig}};
///
/// let mut doc = PdfDocument::open("scanned.pdf")?;
/// let engine = OcrEngine::new("det.onnx", "rec.onnx", "dict.txt", OcrConfig::default())?;
///
/// let text = ocr::ocr_page(&mut doc, 0, &engine, OcrExtractOptions::default())?;
/// println!("OCR text: {}", text);
/// ```
pub fn ocr_page(
    doc: &mut PdfDocument,
    page: usize,
    engine: &OcrEngine,
    options: &OcrExtractOptions,
) -> Result<String> {
    // Extract images from the page
    let images = doc.extract_images(page)?;

    if images.is_empty() {
        if options.fallback_to_native {
            return doc.extract_text(page);
        }
        return Ok(String::new());
    }

    // Find the largest image (assumed to be the page scan)
    let largest_image = images
        .iter()
        .max_by_key(|img| (img.width() as u64) * (img.height() as u64))
        .unwrap();

    // Convert to DynamicImage
    let dynamic_image = largest_image.to_dynamic_image()?;

    // Run OCR
    let ocr_result = engine
        .ocr_image(&dynamic_image)
        .map_err(|e| crate::error::Error::Image(format!("OCR failed: {}", e)))?;

    // Return the text in reading order
    Ok(ocr_result.text_in_reading_order())
}

/// OCR a page and return TextSpans for layout integration.
///
/// This function is similar to `ocr_page` but returns structured TextSpans
/// that can be used with the existing layout analysis pipeline.
///
/// # Arguments
///
/// * `doc` - The PDF document
/// * `page` - Page number (0-indexed)
/// * `engine` - The OCR engine to use
/// * `options` - OCR extraction options
///
/// # Returns
///
/// Vector of TextSpans from the OCR result.
pub fn ocr_page_spans(
    doc: &mut PdfDocument,
    page: usize,
    engine: &OcrEngine,
    options: &OcrExtractOptions,
) -> Result<Vec<crate::layout::text_block::TextSpan>> {
    // Extract images from the page
    let images = doc.extract_images(page)?;

    if images.is_empty() {
        return Ok(Vec::new());
    }

    // Find the largest image (assumed to be the page scan)
    let largest_image = images
        .iter()
        .max_by_key(|img| (img.width() as u64) * (img.height() as u64))
        .unwrap();

    // Convert to DynamicImage
    let dynamic_image = largest_image.to_dynamic_image()?;

    // Run OCR
    let ocr_result = engine
        .ocr_image(&dynamic_image)
        .map_err(|e| crate::error::Error::Image(format!("OCR failed: {}", e)))?;

    // Convert to TextSpans
    Ok(ocr_result.to_text_spans(options.scale))
}

/// Extract text from a page, automatically using OCR if needed.
///
/// This is the main entry point for text extraction that handles both
/// native PDF text and scanned pages transparently.
///
/// # Arguments
///
/// * `doc` - The PDF document
/// * `page` - Page number (0-indexed)
/// * `engine` - The OCR engine to use (optional, only needed for scanned pages)
/// * `options` - OCR extraction options
///
/// # Returns
///
/// The extracted text, either from native PDF text or OCR.
///
/// # Example
///
/// ```ignore
/// use pdf_oxide::{PdfDocument, ocr::{self, OcrEngine, OcrConfig, OcrExtractOptions}};
///
/// let mut doc = PdfDocument::open("mixed.pdf")?;
/// let engine = OcrEngine::new("det.onnx", "rec.onnx", "dict.txt", OcrConfig::default())?;
///
/// // Automatically uses native text or OCR as needed
/// let text = ocr::extract_text_with_ocr(&mut doc, 0, Some(&engine), OcrExtractOptions::default())?;
/// ```
pub fn extract_text_with_ocr(
    doc: &mut PdfDocument,
    page: usize,
    engine: Option<&OcrEngine>,
    options: OcrExtractOptions,
) -> Result<String> {
    // First, check if native text extraction works
    let native_text = doc.extract_text(page).unwrap_or_default();

    // If we got substantial text, return it
    if native_text.trim().len() > 50 {
        return Ok(native_text);
    }

    // Check if we have images (potential scanned page)
    let images = doc.extract_images(page)?;
    if images.is_empty() {
        // No images, return whatever native text we got
        return Ok(native_text);
    }

    // We have images but no/little text - try OCR if engine is available
    if let Some(ocr_engine) = engine {
        match ocr_page(doc, page, ocr_engine, &options) {
            Ok(ocr_text) => return Ok(ocr_text),
            Err(e) => {
                log::warn!("OCR failed for page {}: {}", page, e);
                if options.fallback_to_native {
                    return Ok(native_text);
                }
                return Err(e);
            },
        }
    }

    // No OCR engine, return native text
    Ok(native_text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ocr_module_compiles() {
        // Basic compile test - verify config can be created
        let _ = OcrConfig::default();
    }

    #[test]
    fn test_ocr_extract_options_default() {
        let options = OcrExtractOptions::default();
        assert!((options.scale - 300.0 / 72.0).abs() < 0.01);
        assert!(options.fallback_to_native);
    }

    #[test]
    fn test_ocr_extract_options_with_dpi() {
        let options = OcrExtractOptions::with_dpi(200.0);
        assert!((options.scale - 200.0 / 72.0).abs() < 0.01);
    }
}
