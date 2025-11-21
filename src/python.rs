//! Python bindings via PyO3.
//!
//! This module provides Python bindings for the PDF library, exposing the core functionality
//! through a Python-friendly API with proper error handling and type hints.
//!
//! # Architecture
//!
//! - `PyPdfDocument`: Python wrapper around Rust `PdfDocument`
//! - `PyOcrEngine`: Python wrapper for OCR functionality (requires `ocr` feature)
//! - `PyOcrConfig`: Python wrapper for OCR configuration
//! - Error mapping: Rust errors → Python exceptions
//! - Default arguments using `#[pyo3(signature = ...)]`
//!
//! # Example
//!
//! ```python
//! from pdf_oxide import PdfDocument
//!
//! doc = PdfDocument("document.pdf")
//! text = doc.extract_text(0)
//! markdown = doc.to_markdown(0, detect_headings=True)
//! ```
//!
//! # OCR Example (requires `ocr` feature)
//!
//! ```python
//! from pdf_oxide import PdfDocument, OcrEngine, OcrConfig
//!
//! # Create OCR engine
//! config = OcrConfig()
//! engine = OcrEngine("det.onnx", "rec.onnx", "en_dict.txt", config)
//!
//! # Extract text with automatic OCR fallback
//! doc = PdfDocument("scanned.pdf")
//! text = doc.extract_text_with_ocr(0, engine)
//! ```

use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;

use crate::converters::ConversionOptions as RustConversionOptions;
use crate::document::PdfDocument as RustPdfDocument;

#[cfg(feature = "ocr")]
use crate::ocr::{OcrConfig as RustOcrConfig, OcrEngine as RustOcrEngine, OcrExtractOptions};

/// Python wrapper for PdfDocument.
///
/// Provides PDF parsing, text extraction, and format conversion capabilities.
///
/// # Methods
///
/// - `__init__(path)`: Open a PDF file
/// - `version()`: Get PDF version tuple
/// - `page_count()`: Get number of pages
/// - `extract_text(page)`: Extract text from a page
/// - `to_markdown(page, ...)`: Convert page to Markdown
/// - `to_html(page, ...)`: Convert page to HTML
/// - `to_markdown_all(...)`: Convert all pages to Markdown
/// - `to_html_all(...)`: Convert all pages to HTML
#[pyclass(name = "PdfDocument", unsendable)]
pub struct PyPdfDocument {
    /// Inner Rust document
    inner: RustPdfDocument,
}

#[pymethods]
impl PyPdfDocument {
    /// Open a PDF file.
    ///
    /// Args:
    ///     path (str): Path to the PDF file
    ///
    /// Returns:
    ///     PdfDocument: Opened PDF document
    ///
    /// Raises:
    ///     IOError: If the file cannot be opened or is not a valid PDF
    ///
    /// Example:
    ///     >>> doc = PdfDocument("sample.pdf")
    ///     >>> print(doc.version())
    ///     (1, 7)
    #[new]
    fn new(path: String) -> PyResult<Self> {
        let doc = RustPdfDocument::open(&path)
            .map_err(|e| PyIOError::new_err(format!("Failed to open PDF: {}", e)))?;

        Ok(PyPdfDocument { inner: doc })
    }

    /// Get PDF version.
    ///
    /// Returns:
    ///     tuple[int, int]: PDF version as (major, minor), e.g. (1, 7) for PDF 1.7
    ///
    /// Example:
    ///     >>> doc = PdfDocument("sample.pdf")
    ///     >>> version = doc.version()
    ///     >>> print(f"PDF {version[0]}.{version[1]}")
    ///     PDF 1.7
    fn version(&self) -> (u8, u8) {
        self.inner.version()
    }

    /// Get number of pages in the document.
    ///
    /// Returns:
    ///     int: Number of pages
    ///
    /// Raises:
    ///     RuntimeError: If page count cannot be determined
    ///
    /// Example:
    ///     >>> doc = PdfDocument("sample.pdf")
    ///     >>> print(f"Pages: {doc.page_count()}")
    ///     Pages: 42
    fn page_count(&mut self) -> PyResult<usize> {
        self.inner
            .page_count()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get page count: {}", e)))
    }

    /// Check if document has a structure tree (Tagged PDF).
    ///
    /// Tagged PDFs contain explicit document structure that defines reading order,
    /// semantic meaning, and accessibility information. This is the PDF-spec-compliant
    /// way to determine reading order.
    ///
    /// Returns:
    ///     bool: True if document has logical structure (Tagged PDF), False otherwise
    ///
    /// Example:
    ///     >>> doc = PdfDocument("sample.pdf")
    ///     >>> if doc.has_structure_tree():
    ///     ...     print("Tagged PDF with logical structure")
    ///     ... else:
    ///     ...     print("Untagged PDF - uses page content order")
    fn has_structure_tree(&mut self) -> bool {
        match self.inner.structure_tree() {
            Ok(Some(_)) => true,
            _ => false,
        }
    }

    /// Extract text from a page.
    ///
    /// Args:
    ///     page (int): Page index (0-based)
    ///
    /// Returns:
    ///     str: Extracted text
    ///
    /// Raises:
    ///     RuntimeError: If text extraction fails or page index is invalid
    ///
    /// Example:
    ///     >>> doc = PdfDocument("sample.pdf")
    ///     >>> text = doc.extract_text(0)
    ///     >>> print(text[:100])
    ///     This is the text from the first page...
    fn extract_text(&mut self, page: usize) -> PyResult<String> {
        self.inner
            .extract_text(page)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to extract text: {}", e)))
    }

    /// Convert page to plain text.
    ///
    /// Args:
    ///     page (int): Page index (0-based)
    ///     preserve_layout (bool): Preserve visual layout (default: False) [currently unused]
    ///     detect_headings (bool): Detect headings (default: True) [currently unused]
    ///     include_images (bool): Include images (default: True) [currently unused]
    ///     image_output_dir (str | None): Directory for images (default: None) [currently unused]
    ///
    /// Returns:
    ///     str: Plain text from the page
    ///
    /// Raises:
    ///     RuntimeError: If conversion fails
    ///
    /// Example:
    ///     >>> doc = PdfDocument("paper.pdf")
    ///     >>> text = doc.to_plain_text(0)
    ///     >>> print(text[:100])
    ///
    /// Note:
    ///     Options parameters are accepted for API consistency but currently unused for plain text.
    #[pyo3(signature = (page, preserve_layout=false, detect_headings=true, include_images=true, image_output_dir=None))]
    fn to_plain_text(
        &mut self,
        page: usize,
        preserve_layout: bool,
        detect_headings: bool,
        include_images: bool,
        image_output_dir: Option<String>,
    ) -> PyResult<String> {
        let options = RustConversionOptions {
            preserve_layout,
            detect_headings,
            extract_tables: false,
            include_images,
            image_output_dir,
            ..Default::default()
        };

        self.inner
            .to_plain_text(page, &options)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to convert to plain text: {}", e)))
    }

    /// Convert all pages to plain text.
    ///
    /// Args:
    ///     preserve_layout (bool): Preserve visual layout (default: False) [currently unused]
    ///     detect_headings (bool): Detect headings (default: True) [currently unused]
    ///     include_images (bool): Include images (default: True) [currently unused]
    ///     image_output_dir (str | None): Directory for images (default: None) [currently unused]
    ///
    /// Returns:
    ///     str: Plain text from all pages separated by horizontal rules
    ///
    /// Raises:
    ///     RuntimeError: If conversion fails
    ///
    /// Example:
    ///     >>> doc = PdfDocument("book.pdf")
    ///     >>> text = doc.to_plain_text_all()
    ///     >>> with open("book.txt", "w") as f:
    ///     ...     f.write(text)
    ///
    /// Note:
    ///     Options parameters are accepted for API consistency but currently unused for plain text.
    #[pyo3(signature = (preserve_layout=false, detect_headings=true, include_images=true, image_output_dir=None))]
    fn to_plain_text_all(
        &mut self,
        preserve_layout: bool,
        detect_headings: bool,
        include_images: bool,
        image_output_dir: Option<String>,
    ) -> PyResult<String> {
        let options = RustConversionOptions {
            preserve_layout,
            detect_headings,
            extract_tables: false,
            include_images,
            image_output_dir,
            ..Default::default()
        };

        self.inner.to_plain_text_all(&options).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to convert all pages to plain text: {}", e))
        })
    }

    /// Convert page to Markdown.
    ///
    /// Args:
    ///     page (int): Page index (0-based)
    ///     preserve_layout (bool): Preserve visual layout (default: False)
    ///     detect_headings (bool): Detect headings based on font size (default: True)
    ///     include_images (bool): Include images in output (default: True)
    ///     image_output_dir (str | None): Directory to save images (default: None)
    ///
    /// Returns:
    ///     str: Markdown text
    ///
    /// Raises:
    ///     RuntimeError: If conversion fails
    ///
    /// Example:
    ///     >>> doc = PdfDocument("paper.pdf")
    ///     >>> markdown = doc.to_markdown(0, detect_headings=True)
    ///     >>> with open("output.md", "w") as f:
    ///     ...     f.write(markdown)
    #[pyo3(signature = (page, preserve_layout=false, detect_headings=true, include_images=true, image_output_dir=None))]
    fn to_markdown(
        &mut self,
        page: usize,
        preserve_layout: bool,
        detect_headings: bool,
        include_images: bool,
        image_output_dir: Option<String>,
    ) -> PyResult<String> {
        let options = RustConversionOptions {
            preserve_layout,
            detect_headings,
            extract_tables: false,
            include_images,
            image_output_dir,
            ..Default::default()
        };

        self.inner
            .to_markdown(page, &options)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to convert to Markdown: {}", e)))
    }

    /// Convert page to HTML.
    ///
    /// Args:
    ///     page (int): Page index (0-based)
    ///     preserve_layout (bool): Preserve visual layout with CSS positioning (default: False)
    ///     detect_headings (bool): Detect headings based on font size (default: True)
    ///     include_images (bool): Include images in output (default: True)
    ///     image_output_dir (str | None): Directory to save images (default: None)
    ///
    /// Returns:
    ///     str: HTML text
    ///
    /// Raises:
    ///     RuntimeError: If conversion fails
    ///
    /// Example:
    ///     >>> doc = PdfDocument("paper.pdf")
    ///     >>> html = doc.to_html(0, preserve_layout=False)
    ///     >>> with open("output.html", "w") as f:
    ///     ...     f.write(html)
    #[pyo3(signature = (page, preserve_layout=false, detect_headings=true, include_images=true, image_output_dir=None))]
    fn to_html(
        &mut self,
        page: usize,
        preserve_layout: bool,
        detect_headings: bool,
        include_images: bool,
        image_output_dir: Option<String>,
    ) -> PyResult<String> {
        let options = RustConversionOptions {
            preserve_layout,
            detect_headings,
            extract_tables: false,
            include_images,
            image_output_dir,
            ..Default::default()
        };

        self.inner
            .to_html(page, &options)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to convert to HTML: {}", e)))
    }

    /// Convert all pages to Markdown.
    ///
    /// Args:
    ///     preserve_layout (bool): Preserve visual layout (default: False)
    ///     detect_headings (bool): Detect headings based on font size (default: True)
    ///     include_images (bool): Include images in output (default: True)
    ///     image_output_dir (str | None): Directory to save images (default: None)
    ///
    /// Returns:
    ///     str: Markdown text with all pages separated by horizontal rules
    ///
    /// Raises:
    ///     RuntimeError: If conversion fails
    ///
    /// Example:
    ///     >>> doc = PdfDocument("book.pdf")
    ///     >>> markdown = doc.to_markdown_all(detect_headings=True)
    ///     >>> with open("book.md", "w") as f:
    ///     ...     f.write(markdown)
    #[pyo3(signature = (preserve_layout=false, detect_headings=true, include_images=true, image_output_dir=None))]
    fn to_markdown_all(
        &mut self,
        preserve_layout: bool,
        detect_headings: bool,
        include_images: bool,
        image_output_dir: Option<String>,
    ) -> PyResult<String> {
        let options = RustConversionOptions {
            preserve_layout,
            detect_headings,
            extract_tables: false,
            include_images,
            image_output_dir,
            ..Default::default()
        };

        self.inner.to_markdown_all(&options).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to convert all pages to Markdown: {}", e))
        })
    }

    /// Convert all pages to HTML.
    ///
    /// Args:
    ///     preserve_layout (bool): Preserve visual layout with CSS positioning (default: False)
    ///     detect_headings (bool): Detect headings based on font size (default: True)
    ///     include_images (bool): Include images in output (default: True)
    ///     image_output_dir (str | None): Directory to save images (default: None)
    ///
    /// Returns:
    ///     str: HTML text with all pages wrapped in div.page elements
    ///
    /// Raises:
    ///     RuntimeError: If conversion fails
    ///
    /// Example:
    ///     >>> doc = PdfDocument("book.pdf")
    ///     >>> html = doc.to_html_all(preserve_layout=True)
    ///     >>> with open("book.html", "w") as f:
    ///     ...     f.write(html)
    #[pyo3(signature = (preserve_layout=false, detect_headings=true, include_images=true, image_output_dir=None))]
    fn to_html_all(
        &mut self,
        preserve_layout: bool,
        detect_headings: bool,
        include_images: bool,
        image_output_dir: Option<String>,
    ) -> PyResult<String> {
        let options = RustConversionOptions {
            preserve_layout,
            detect_headings,
            extract_tables: false,
            include_images,
            image_output_dir,
            ..Default::default()
        };

        self.inner.to_html_all(&options).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to convert all pages to HTML: {}", e))
        })
    }

    /// String representation of the document.
    ///
    /// Returns:
    ///     str: Representation showing PDF version
    fn __repr__(&self) -> String {
        format!("PdfDocument(version={}.{})", self.inner.version().0, self.inner.version().1)
    }

    /// Check if a page needs OCR (is a scanned page).
    ///
    /// A page is considered "scanned" if it has no native text (or very little)
    /// but contains images (typically a full-page scan).
    ///
    /// Args:
    ///     page (int): Page index (0-based)
    ///
    /// Returns:
    ///     bool: True if the page likely needs OCR, False otherwise
    ///
    /// Raises:
    ///     RuntimeError: If page analysis fails
    ///
    /// Example:
    ///     >>> doc = PdfDocument("scanned.pdf")
    ///     >>> if doc.needs_ocr(0):
    ///     ...     print("Page 0 is scanned, needs OCR")
    #[cfg(feature = "ocr")]
    fn needs_ocr(&mut self, page: usize) -> PyResult<bool> {
        crate::ocr::needs_ocr(&mut self.inner, page)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to check if OCR needed: {}", e)))
    }

    /// Extract text from a page, automatically using OCR if needed.
    ///
    /// This function first attempts native text extraction. If the page has little
    /// or no text but contains images (indicating a scanned page), it falls back
    /// to OCR if an engine is provided.
    ///
    /// Args:
    ///     page (int): Page index (0-based)
    ///     engine (OcrEngine | None): OCR engine to use for scanned pages (default: None)
    ///     dpi (float): DPI for coordinate conversion (default: 300.0)
    ///     fallback_to_native (bool): Fall back to native text if OCR fails (default: True)
    ///
    /// Returns:
    ///     str: Extracted text (either native or OCR)
    ///
    /// Raises:
    ///     RuntimeError: If text extraction fails
    ///
    /// Example:
    ///     >>> doc = PdfDocument("mixed.pdf")
    ///     >>> engine = OcrEngine("det.onnx", "rec.onnx", "dict.txt")
    ///     >>> text = doc.extract_text_with_ocr(0, engine)
    #[cfg(feature = "ocr")]
    #[pyo3(signature = (page, engine=None, dpi=300.0, fallback_to_native=true))]
    fn extract_text_with_ocr(
        &mut self,
        page: usize,
        engine: Option<&PyOcrEngine>,
        dpi: f32,
        fallback_to_native: bool,
    ) -> PyResult<String> {
        let options = OcrExtractOptions {
            scale: dpi / 72.0,
            fallback_to_native,
            ..Default::default()
        };

        let rust_engine = engine.map(|e| &e.inner);

        crate::ocr::extract_text_with_ocr(&mut self.inner, page, rust_engine, options)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to extract text with OCR: {}", e)))
    }

    /// OCR a page and return the extracted text.
    ///
    /// This function always performs OCR on the page, regardless of whether
    /// the page has native text. Use this when you specifically need OCR results.
    ///
    /// Args:
    ///     page (int): Page index (0-based)
    ///     engine (OcrEngine): OCR engine to use
    ///     dpi (float): DPI for coordinate conversion (default: 300.0)
    ///
    /// Returns:
    ///     str: OCR extracted text in reading order
    ///
    /// Raises:
    ///     RuntimeError: If OCR fails
    ///
    /// Example:
    ///     >>> doc = PdfDocument("scanned.pdf")
    ///     >>> engine = OcrEngine("det.onnx", "rec.onnx", "dict.txt")
    ///     >>> text = doc.ocr_page(0, engine)
    #[cfg(feature = "ocr")]
    #[pyo3(signature = (page, engine, dpi=300.0))]
    fn ocr_page(&mut self, page: usize, engine: &PyOcrEngine, dpi: f32) -> PyResult<String> {
        let options = OcrExtractOptions {
            scale: dpi / 72.0,
            ..Default::default()
        };

        crate::ocr::ocr_page(&mut self.inner, page, &engine.inner, &options)
            .map_err(|e| PyRuntimeError::new_err(format!("OCR failed: {}", e)))
    }
}

// =============================================================================
// OCR Python Bindings (feature-gated)
// =============================================================================

/// OCR configuration for text detection and recognition.
///
/// Controls various thresholds and parameters for the OCR pipeline.
///
/// # Attributes
///
/// - `det_threshold`: Detection confidence threshold (0.0-1.0, default: 0.3)
/// - `box_threshold`: Text box confidence threshold (0.0-1.0, default: 0.5)
/// - `unclip_ratio`: Expansion ratio for detected boxes (default: 1.5)
/// - `rec_threshold`: Recognition confidence threshold (0.0-1.0, default: 0.5)
/// - `det_max_side`: Maximum side length for detection input (default: 960)
/// - `rec_target_height`: Target height for recognition input (default: 48)
/// - `num_threads`: Number of threads for ONNX inference (default: 4)
///
/// # Example
///
/// ```python
/// from pdf_oxide import OcrConfig
///
/// # Default configuration
/// config = OcrConfig()
///
/// # Custom configuration
/// config = OcrConfig(
///     det_threshold=0.4,
///     box_threshold=0.6,
///     num_threads=8
/// )
/// ```
#[cfg(feature = "ocr")]
#[pyclass(name = "OcrConfig")]
#[derive(Clone)]
pub struct PyOcrConfig {
    inner: RustOcrConfig,
}

#[cfg(feature = "ocr")]
#[pymethods]
impl PyOcrConfig {
    /// Create a new OCR configuration.
    ///
    /// Args:
    ///     det_threshold (float): Detection confidence threshold (default: 0.3)
    ///     box_threshold (float): Text box confidence threshold (default: 0.5)
    ///     unclip_ratio (float): Expansion ratio for detected boxes (default: 1.5)
    ///     rec_threshold (float): Recognition confidence threshold (default: 0.5)
    ///     det_max_side (int): Maximum side length for detection input (default: 960)
    ///     rec_target_height (int): Target height for recognition input (default: 48)
    ///     num_threads (int): Number of threads for ONNX inference (default: 4)
    #[new]
    #[pyo3(signature = (det_threshold=0.3, box_threshold=0.5, unclip_ratio=1.5, rec_threshold=0.5, det_max_side=960, rec_target_height=48, num_threads=4))]
    fn new(
        det_threshold: f32,
        box_threshold: f32,
        unclip_ratio: f32,
        rec_threshold: f32,
        det_max_side: u32,
        rec_target_height: u32,
        num_threads: usize,
    ) -> Self {
        use crate::ocr::OcrConfigBuilder;

        let config = OcrConfigBuilder::new()
            .det_threshold(det_threshold)
            .box_threshold(box_threshold)
            .unclip_ratio(unclip_ratio)
            .rec_threshold(rec_threshold)
            .det_max_side(det_max_side)
            .rec_target_height(rec_target_height)
            .num_threads(num_threads)
            .build();

        PyOcrConfig { inner: config }
    }

    /// Get detection threshold.
    #[getter]
    fn det_threshold(&self) -> f32 {
        self.inner.det_threshold
    }

    /// Get box threshold.
    #[getter]
    fn box_threshold(&self) -> f32 {
        self.inner.box_threshold
    }

    /// Get unclip ratio.
    #[getter]
    fn unclip_ratio(&self) -> f32 {
        self.inner.unclip_ratio
    }

    /// Get recognition threshold.
    #[getter]
    fn rec_threshold(&self) -> f32 {
        self.inner.rec_threshold
    }

    /// Get max detection side.
    #[getter]
    fn det_max_side(&self) -> u32 {
        self.inner.det_max_side
    }

    /// Get recognition target height.
    #[getter]
    fn rec_target_height(&self) -> u32 {
        self.inner.rec_target_height
    }

    /// Get number of threads.
    #[getter]
    fn num_threads(&self) -> usize {
        self.inner.num_threads
    }

    fn __repr__(&self) -> String {
        format!(
            "OcrConfig(det_threshold={}, box_threshold={}, rec_threshold={})",
            self.inner.det_threshold, self.inner.box_threshold, self.inner.rec_threshold
        )
    }
}

/// OCR engine for text extraction from images.
///
/// Combines PaddleOCR detection (DBNet++) and recognition (SVTR) models
/// for end-to-end OCR on scanned documents.
///
/// # Example
///
/// ```python
/// from pdf_oxide import OcrEngine, OcrConfig
///
/// # Create engine with default config
/// engine = OcrEngine(
///     det_model_path="models/en_PP-OCRv5_det_infer.onnx",
///     rec_model_path="models/en_PP-OCRv5_rec_infer.onnx",
///     dict_path="models/en_dict.txt"
/// )
///
/// # Create engine with custom config
/// config = OcrConfig(det_threshold=0.4, num_threads=8)
/// engine = OcrEngine(
///     det_model_path="models/det.onnx",
///     rec_model_path="models/rec.onnx",
///     dict_path="models/dict.txt",
///     config=config
/// )
/// ```
#[cfg(feature = "ocr")]
#[pyclass(name = "OcrEngine", unsendable)]
pub struct PyOcrEngine {
    inner: RustOcrEngine,
}

#[cfg(feature = "ocr")]
#[pymethods]
impl PyOcrEngine {
    /// Create a new OCR engine.
    ///
    /// Args:
    ///     det_model_path (str): Path to DBNet++ detection model (ONNX format)
    ///     rec_model_path (str): Path to SVTR recognition model (ONNX format)
    ///     dict_path (str): Path to character dictionary file
    ///     config (OcrConfig | None): OCR configuration (default: None uses defaults)
    ///
    /// Returns:
    ///     OcrEngine: Initialized OCR engine
    ///
    /// Raises:
    ///     IOError: If model files cannot be loaded
    ///     ValueError: If models are invalid
    ///
    /// Example:
    ///     >>> engine = OcrEngine("det.onnx", "rec.onnx", "dict.txt")
    #[new]
    #[pyo3(signature = (det_model_path, rec_model_path, dict_path, config=None))]
    fn new(
        det_model_path: String,
        rec_model_path: String,
        dict_path: String,
        config: Option<PyOcrConfig>,
    ) -> PyResult<Self> {
        let rust_config = config.map(|c| c.inner).unwrap_or_default();

        let engine = RustOcrEngine::new(&det_model_path, &rec_model_path, &dict_path, rust_config)
            .map_err(|e| PyIOError::new_err(format!("Failed to create OCR engine: {}", e)))?;

        Ok(PyOcrEngine { inner: engine })
    }

    /// OCR an image file directly.
    ///
    /// Args:
    ///     image_path (str): Path to the image file
    ///
    /// Returns:
    ///     dict: OCR result containing 'text', 'confidence', and 'spans'
    ///
    /// Raises:
    ///     IOError: If image cannot be loaded
    ///     RuntimeError: If OCR fails
    ///
    /// Example:
    ///     >>> engine = OcrEngine("det.onnx", "rec.onnx", "dict.txt")
    ///     >>> result = engine.ocr_image("document.png")
    ///     >>> print(result['text'])
    fn ocr_image(&self, py: Python<'_>, image_path: String) -> PyResult<Py<pyo3::types::PyDict>> {
        let image = image::open(&image_path)
            .map_err(|e| PyIOError::new_err(format!("Failed to open image: {}", e)))?;

        let result = self
            .inner
            .ocr_image(&image)
            .map_err(|e| PyRuntimeError::new_err(format!("OCR failed: {}", e)))?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("text", result.text_in_reading_order())?;
        dict.set_item("confidence", result.total_confidence)?;

        // Convert spans to list of dicts
        let spans_list = pyo3::types::PyList::empty(py);
        for span in &result.spans {
            let span_dict = pyo3::types::PyDict::new(py);
            span_dict.set_item("text", &span.text)?;
            span_dict.set_item("confidence", span.confidence)?;

            // Convert polygon to list
            let polygon: Vec<Vec<f32>> = span.polygon.iter().map(|p| vec![p[0], p[1]]).collect();
            span_dict.set_item("polygon", polygon)?;

            spans_list.append(span_dict)?;
        }
        dict.set_item("spans", spans_list)?;

        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        "OcrEngine(loaded)".to_string()
    }
}

/// Check if OCR feature is available.
///
/// Returns:
///     bool: True if OCR feature is compiled in, False otherwise
///
/// Example:
///     >>> from pdf_oxide import has_ocr
///     >>> if has_ocr():
///     ...     from pdf_oxide import OcrEngine
#[pyfunction]
fn has_ocr() -> bool {
    cfg!(feature = "ocr")
}

/// Python module for PDF library.
///
/// This is the internal module (pdf_oxide) that gets imported by the Python package.
#[pymodule]
fn pdf_oxide(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPdfDocument>()?;
    m.add_function(wrap_pyfunction!(has_ocr, m)?)?;
    m.add("VERSION", env!("CARGO_PKG_VERSION"))?;

    // Add OCR classes if feature is enabled
    #[cfg(feature = "ocr")]
    {
        m.add_class::<PyOcrConfig>()?;
        m.add_class::<PyOcrEngine>()?;
    }

    Ok(())
}
