//! WebAssembly bindings for PDF Oxide.
//!
//! Provides a JavaScript/TypeScript API for PDF operations in browser
//! environments. Requires the `wasm` feature flag.
//!
//! # Example (JavaScript)
//!
//! ```javascript
//! import init, { WasmPdfDocument, WasmPdf } from 'pdf_oxide';
//!
//! await init();
//!
//! // Read an existing PDF
//! const response = await fetch('document.pdf');
//! const bytes = new Uint8Array(await response.arrayBuffer());
//! const doc = new WasmPdfDocument(bytes);
//! console.log(`Pages: ${doc.pageCount()}`);
//! console.log(doc.extractText(0));
//! console.log(doc.toMarkdown(0));
//!
//! // Create a new PDF from Markdown
//! const pdf = WasmPdf.fromMarkdown("# Hello\n\nWorld");
//! const pdfBytes = pdf.toBytes(); // Uint8Array
//!
//! // Edit a PDF
//! doc.setTitle("My Document");
//! doc.setPageRotation(0, 90);
//! const edited = doc.saveToBytes(); // Uint8Array
//! doc.free();
//! ```

use wasm_bindgen::prelude::*;

use crate::api::PdfBuilder;
use crate::converters::ConversionOptions;
use crate::document::PdfDocument;
use crate::editor::{
    DocumentEditor, EncryptionAlgorithm, EncryptionConfig, Permissions, SaveOptions,
};
use crate::search::{SearchOptions, TextSearcher};

// ============================================================================
// WasmPdfDocument — read, convert, search, extract, and edit PDFs
// ============================================================================

use std::sync::{Arc, Mutex};

/// A PDF document loaded from bytes for use in WebAssembly.
///
/// Create an instance by passing PDF file bytes to the constructor.
/// Call `.free()` when done to release memory.
#[wasm_bindgen]
#[derive(Clone)]
pub struct WasmPdfDocument {
    inner: Arc<Mutex<PdfDocument>>,
    /// Raw bytes for editor initialization (kept for lazy editor creation)
    raw_bytes: Arc<Vec<u8>>,
    /// Lazy-initialized editor for mutation operations
    editor: Option<Arc<Mutex<DocumentEditor>>>,
}

#[wasm_bindgen]
impl WasmPdfDocument {
    /// Ensure the editor is initialized, creating it from the raw bytes if needed.
    fn ensure_editor(&mut self) -> Result<Arc<Mutex<DocumentEditor>>, JsValue> {
        if self.editor.is_none() {
            let editor = DocumentEditor::from_bytes(self.raw_bytes.to_vec())
                .map_err(|e| JsValue::from_str(&format!("Failed to open editor: {}", e)))?;
            self.editor = Some(Arc::new(Mutex::new(editor)));
        }
        Ok(self
            .editor
            .as_ref()
            .expect("editor just initialized")
            .clone())
    }
}

#[wasm_bindgen]
impl WasmPdfDocument {
    // ========================================================================
    // Constructor
    // ========================================================================

    /// Load a PDF document from raw bytes.
    ///
    /// @param data - PDF file contents as Uint8Array
    /// @param password - Optional password for encrypted PDFs
    /// @throws Error if the PDF is invalid or cannot be parsed
    #[wasm_bindgen(constructor)]
    pub fn new(data: &[u8], password: Option<String>) -> Result<WasmPdfDocument, JsValue> {
        #[cfg(feature = "wasm")]
        console_error_panic_hook::set_once();

        let bytes = data.to_vec();
        let mut inner = PdfDocument::from_bytes(bytes.clone())
            .map_err(|e| JsValue::from_str(&format!("Failed to open PDF: {}", e)))?;

        if let Some(pw) = password {
            inner
                .authenticate(pw.as_bytes())
                .map_err(|e| JsValue::from_str(&format!("Authentication failed: {}", e)))?;
        }

        Ok(WasmPdfDocument {
            inner: Arc::new(Mutex::new(inner)),
            raw_bytes: Arc::new(bytes),
            editor: None,
        })
    }

    // ========================================================================
    // Group 1: Core Read-Only
    // ========================================================================

    /// Get the number of pages in the document.
    #[wasm_bindgen(js_name = "pageCount")]
    pub fn page_count(&mut self) -> Result<usize, JsValue> {
        self.inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?
            .page_count()
            .map_err(|e| JsValue::from_str(&format!("Failed to get page count: {}", e)))
    }

    /// Get the PDF version as [major, minor].
    #[wasm_bindgen(js_name = "version")]
    pub fn version(&self) -> Result<Vec<u8>, JsValue> {
        let (major, minor) = self
            .inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?
            .version();
        Ok(vec![major, minor])
    }

    /// Authenticate with a password to decrypt an encrypted PDF.
    ///
    /// @param password - The password string
    /// @returns true if authentication succeeded
    #[wasm_bindgen(js_name = "authenticate")]
    pub fn authenticate(&mut self, password: &str) -> Result<bool, JsValue> {
        self.inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?
            .authenticate(password.as_bytes())
            .map_err(|e| JsValue::from_str(&format!("Authentication failed: {}", e)))
    }

    /// Check if the document has a structure tree (Tagged PDF).
    #[wasm_bindgen(js_name = "hasStructureTree")]
    pub fn has_structure_tree(&mut self) -> Result<bool, JsValue> {
        Ok(matches!(
            self.inner
                .lock()
                .map_err(|_| JsValue::from_str("Mutex lock failed"))?
                .structure_tree(),
            Ok(Some(_))
        ))
    }

    // ========================================================================
    // Group 2: Text Extraction
    // ========================================================================

    /// Extract plain text from a single page.
    ///
    /// @param page_index - Zero-based page number
    /// @param region - Optional [x, y, width, height] to filter by
    #[wasm_bindgen(js_name = "extractText")]
    pub fn extract_text(
        &mut self,
        page_index: usize,
        region: JsValue, // Use JsValue to allow optional/undefined from JS
    ) -> Result<String, JsValue> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;

        if !region.is_undefined() && !region.is_null() {
            let r: Vec<f32> = serde_wasm_bindgen::from_value(region)
                .map_err(|_| JsValue::from_str("Invalid region format. Expected [x, y, w, h]"))?;

            if r.len() != 4 {
                return Err(JsValue::from_str("Region must have exactly 4 elements [x, y, w, h]"));
            }
            inner
                .extract_text_in_rect(
                    page_index,
                    crate::geometry::Rect::new(r[0], r[1], r[2], r[3]),
                    crate::layout::RectFilterMode::Intersects,
                )
                .map_err(|e| JsValue::from_str(&format!("Failed to extract text: {}", e)))
        } else {
            inner
                .extract_text(page_index)
                .map_err(|e| JsValue::from_str(&format!("Failed to extract text: {}", e)))
        }
    }

    /// Extract plain text from all pages, separated by form feed characters.
    #[wasm_bindgen(js_name = "extractAllText")]
    pub fn extract_all_text(&mut self) -> Result<String, JsValue> {
        self.inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?
            .extract_all_text()
            .map_err(|e| JsValue::from_str(&format!("Failed to extract all text: {}", e)))
    }

    /// Identify and remove headers.
    ///
    /// Uses spec-compliant /Artifact tags when available (100% accuracy), or
    /// falls back to heuristic analysis of the top 15% of pages.
    ///
    /// @param threshold - Fraction of pages (0.0-1.0) where text must repeat (heuristic mode)
    #[wasm_bindgen(js_name = "removeHeaders")]
    pub fn remove_headers(&mut self, threshold: f32) -> Result<usize, JsValue> {
        self.inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?
            .remove_headers(threshold)
            .map_err(|e| JsValue::from_str(&format!("Header removal failed: {}", e)))
    }

    /// Identify and remove footers.
    ///
    /// Uses spec-compliant /Artifact tags when available (100% accuracy), or
    /// falls back to heuristic analysis of the bottom 15% of pages.
    ///
    /// @param threshold - Fraction of pages (0.0-1.0) where text must repeat (heuristic mode)
    #[wasm_bindgen(js_name = "removeFooters")]
    pub fn remove_footers(&mut self, threshold: f32) -> Result<usize, JsValue> {
        self.inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?
            .remove_footers(threshold)
            .map_err(|e| JsValue::from_str(&format!("Footer removal failed: {}", e)))
    }

    /// Identify and remove both headers and footers.
    ///
    /// Prioritizes ISO 32000 spec-compliant /Artifact tags, with a heuristic
    /// fallback for untagged PDFs.
    ///
    /// @param threshold - Fraction of pages (0.0-1.0) where text must repeat (heuristic mode)
    #[wasm_bindgen(js_name = "removeArtifacts")]
    pub fn remove_artifacts(&mut self, threshold: f32) -> Result<usize, JsValue> {
        self.inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?
            .remove_artifacts(threshold)
            .map_err(|e| JsValue::from_str(&format!("Artifact removal failed: {}", e)))
    }

    /// Erase existing header content.
    ///
    /// Identifies existing text in the header area (top 15%) and marks it for erasure.
    ///
    /// @param page_index - Zero-based page number
    #[wasm_bindgen(js_name = "eraseHeader")]
    pub fn erase_header(&mut self, page_index: usize) -> Result<(), JsValue> {
        self.inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?
            .erase_header(page_index)
            .map_err(|e| JsValue::from_str(&format!("Failed to erase header: {}", e)))
    }

    /// Deprecated: Use eraseHeader instead.
    #[wasm_bindgen(js_name = "editHeader")]
    pub fn edit_header(&mut self, page_index: usize) -> Result<(), JsValue> {
        self.erase_header(page_index)
    }

    /// Erase existing footer content.
    ///
    /// Identifies existing text in the footer area (bottom 15%) and marks it for erasure.
    ///
    /// @param page_index - Zero-based page number
    #[wasm_bindgen(js_name = "eraseFooter")]
    pub fn erase_footer(&mut self, page_index: usize) -> Result<(), JsValue> {
        self.inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?
            .erase_footer(page_index)
            .map_err(|e| JsValue::from_str(&format!("Failed to erase footer: {}", e)))
    }

    /// Deprecated: Use eraseFooter instead.
    #[wasm_bindgen(js_name = "editFooter")]
    pub fn edit_footer(&mut self, page_index: usize) -> Result<(), JsValue> {
        self.erase_footer(page_index)
    }

    /// Erase both header and footer content.
    ///
    /// @param page_index - Zero-based page number
    #[wasm_bindgen(js_name = "eraseArtifacts")]
    pub fn erase_artifacts(&mut self, page_index: usize) -> Result<(), JsValue> {
        self.inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?
            .erase_artifacts(page_index)
            .map_err(|e| JsValue::from_str(&format!("Failed to erase artifacts: {}", e)))
    }

    /// Focus extraction on a specific rectangular region of a page (v0.3.14).
    ///
    /// @param page_index - Zero-based page number
    /// @param region - [x, y, width, height] in points
    #[wasm_bindgen(js_name = "within")]
    pub fn within(
        &self,
        page_index: usize,
        region: Vec<f32>,
    ) -> Result<WasmPdfPageRegion, JsValue> {
        if region.len() != 4 {
            return Err(JsValue::from_str("Region must have exactly 4 elements [x, y, w, h]"));
        }
        Ok(WasmPdfPageRegion {
            doc: self.clone(),
            page_index,
            region: crate::geometry::Rect::new(region[0], region[1], region[2], region[3]),
        })
    }

    /// Render a page to an image (PNG).
    ///
    /// Requires the `rendering` feature.
    ///
    /// @param page_index - Zero-based page number
    /// @param dpi - Dots per inch (default: 150)
    /// @returns Uint8Array containing the PNG image data
    #[cfg(feature = "rendering")]
    #[wasm_bindgen(js_name = "renderPage")]
    pub fn render_page(&mut self, page_index: usize, dpi: Option<u32>) -> Result<Vec<u8>, JsValue> {
        let opts = crate::rendering::RenderOptions::with_dpi(dpi.unwrap_or(150));
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        let img = crate::rendering::render_page(&mut inner, page_index, &opts)
            .map_err(|e| JsValue::from_str(&format!("Failed to render page: {}", e)))?;
        Ok(img.as_bytes().to_vec())
    }

    // ========================================================================
    // Group 3: Format Conversion
    // ========================================================================

    /// Convert a single page to Markdown.
    ///
    /// @param page_index - Zero-based page number
    /// @param detect_headings - Whether to detect headings (default: true)
    /// @param include_images - Whether to include images (default: true)
    #[wasm_bindgen(js_name = "toMarkdown")]
    pub fn to_markdown(
        &mut self,
        page_index: usize,
        detect_headings: Option<bool>,
        include_images: Option<bool>,
        include_form_fields: Option<bool>,
    ) -> Result<String, JsValue> {
        let mut opts = ConversionOptions::default();
        if let Some(dh) = detect_headings {
            opts.detect_headings = dh;
        }
        if let Some(ii) = include_images {
            opts.include_images = ii;
        }
        if let Some(iff) = include_form_fields {
            opts.include_form_fields = iff;
        }
        self.inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?
            .to_markdown(page_index, &opts)
            .map_err(|e| JsValue::from_str(&format!("Failed to convert to markdown: {}", e)))
    }

    /// Convert all pages to Markdown.
    #[wasm_bindgen(js_name = "toMarkdownAll")]
    pub fn to_markdown_all(
        &mut self,
        detect_headings: Option<bool>,
        include_images: Option<bool>,
        include_form_fields: Option<bool>,
    ) -> Result<String, JsValue> {
        let mut opts = ConversionOptions::default();
        if let Some(dh) = detect_headings {
            opts.detect_headings = dh;
        }
        if let Some(ii) = include_images {
            opts.include_images = ii;
        }
        if let Some(iff) = include_form_fields {
            opts.include_form_fields = iff;
        }
        self.inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?
            .to_markdown_all(&opts)
            .map_err(|e| JsValue::from_str(&format!("Failed to convert to markdown: {}", e)))
    }

    /// Convert a single page to HTML.
    ///
    /// @param page_index - Zero-based page number
    /// @param preserve_layout - Use CSS positioning to preserve layout (default: false)
    /// @param detect_headings - Whether to detect headings (default: true)
    #[wasm_bindgen(js_name = "toHtml")]
    pub fn to_html(
        &mut self,
        page_index: usize,
        preserve_layout: Option<bool>,
        detect_headings: Option<bool>,
        include_form_fields: Option<bool>,
    ) -> Result<String, JsValue> {
        let mut opts = ConversionOptions::default();
        if let Some(pl) = preserve_layout {
            opts.preserve_layout = pl;
        }
        if let Some(dh) = detect_headings {
            opts.detect_headings = dh;
        }
        if let Some(iff) = include_form_fields {
            opts.include_form_fields = iff;
        }
        self.inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?
            .to_html(page_index, &opts)
            .map_err(|e| JsValue::from_str(&format!("Failed to convert to HTML: {}", e)))
    }

    /// Convert all pages to HTML.
    #[wasm_bindgen(js_name = "toHtmlAll")]
    pub fn to_html_all(
        &mut self,
        preserve_layout: Option<bool>,
        detect_headings: Option<bool>,
        include_form_fields: Option<bool>,
    ) -> Result<String, JsValue> {
        let mut opts = ConversionOptions::default();
        if let Some(pl) = preserve_layout {
            opts.preserve_layout = pl;
        }
        if let Some(dh) = detect_headings {
            opts.detect_headings = dh;
        }
        if let Some(iff) = include_form_fields {
            opts.include_form_fields = iff;
        }
        self.inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?
            .to_html_all(&opts)
            .map_err(|e| JsValue::from_str(&format!("Failed to convert to HTML: {}", e)))
    }

    /// Convert a single page to plain text (with layout preservation options).
    #[wasm_bindgen(js_name = "toPlainText")]
    pub fn to_plain_text(&mut self, page_index: usize) -> Result<String, JsValue> {
        let opts = ConversionOptions::default();
        self.inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?
            .to_plain_text(page_index, &opts)
            .map_err(|e| JsValue::from_str(&format!("Failed to convert to plain text: {}", e)))
    }

    /// Convert all pages to plain text.
    #[wasm_bindgen(js_name = "toPlainTextAll")]
    pub fn to_plain_text_all(&mut self) -> Result<String, JsValue> {
        let opts = ConversionOptions::default();
        self.inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?
            .to_plain_text_all(&opts)
            .map_err(|e| JsValue::from_str(&format!("Failed to convert to plain text: {}", e)))
    }

    // ========================================================================
    // Group 4: Structured Extraction (returns JS objects via serde-wasm-bindgen)
    // ========================================================================

    /// Extract character-level data from a page.
    ///
    /// Returns an array of objects with: char, bbox {x, y, width, height},
    /// font_name, font_size, font_weight, is_italic, color {r, g, b}, etc.
    ///
    /// @param page_index - Zero-based page number
    /// @param region - Optional [x, y, width, height] to filter by
    #[wasm_bindgen(js_name = "extractChars")]
    pub fn extract_chars(
        &mut self,
        page_index: usize,
        region: Option<Vec<f32>>,
    ) -> Result<JsValue, JsValue> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;

        let chars_result = if let Some(r) = region {
            if r.len() != 4 {
                return Err(JsValue::from_str("Region must have exactly 4 elements [x, y, w, h]"));
            }
            inner.extract_chars_in_rect(
                page_index,
                crate::geometry::Rect::new(r[0], r[1], r[2], r[3]),
                crate::layout::RectFilterMode::Intersects,
            )
        } else {
            inner.extract_chars(page_index)
        };

        let chars = chars_result
            .map_err(|e| JsValue::from_str(&format!("Failed to extract chars: {}", e)))?;

        serde_wasm_bindgen::to_value(&chars)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    /// Extract span-level data from a page.
    ///
    /// Returns an array of objects with: text, bbox, font_name, font_size,
    /// font_weight, is_italic, color, etc.
    #[wasm_bindgen(js_name = "extractSpans")]
    pub fn extract_spans(
        &mut self,
        page_index: usize,
        region: Option<Vec<f32>>,
    ) -> Result<JsValue, JsValue> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        let spans_result = if let Some(r) = region {
            if r.len() != 4 {
                return Err(JsValue::from_str("Region must have exactly 4 elements [x, y, w, h]"));
            }
            inner.extract_spans_in_rect(
                page_index,
                crate::geometry::Rect::new(r[0], r[1], r[2], r[3]),
                crate::layout::RectFilterMode::Intersects,
            )
        } else {
            inner.extract_spans(page_index)
        };

        let spans = spans_result
            .map_err(|e| JsValue::from_str(&format!("Failed to extract spans: {}", e)))?;
        serde_wasm_bindgen::to_value(&spans)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    /// Extract word-level data from a page.
    ///
    /// Returns an array of objects with: text, bbox, font_name, font_size,
    /// font_weight, is_italic, is_bold.
    #[wasm_bindgen(js_name = "extractWords")]
    pub fn extract_words(
        &mut self,
        page_index: usize,
        region: Option<Vec<f32>>,
    ) -> Result<JsValue, JsValue> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        let words_result = if let Some(r) = region {
            if r.len() != 4 {
                return Err(JsValue::from_str("Region must have exactly 4 elements [x, y, w, h]"));
            }
            inner.extract_words_in_rect(
                page_index,
                crate::geometry::Rect::new(r[0], r[1], r[2], r[3]),
                crate::layout::RectFilterMode::Intersects,
            )
        } else {
            inner.extract_words(page_index)
        };

        let words = words_result
            .map_err(|e| JsValue::from_str(&format!("Failed to extract words: {}", e)))?;
        serde_wasm_bindgen::to_value(&words)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    /// Extract text lines from a page.
    ///
    /// Returns an array of objects with: text, bbox, words (array of Word objects).
    #[wasm_bindgen(js_name = "extractTextLines")]
    pub fn extract_text_lines(
        &mut self,
        page_index: usize,
        region: Option<Vec<f32>>,
    ) -> Result<JsValue, JsValue> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        let lines_result = if let Some(r) = region {
            if r.len() != 4 {
                return Err(JsValue::from_str("Region must have exactly 4 elements [x, y, w, h]"));
            }
            inner.extract_text_lines_in_rect(
                page_index,
                crate::geometry::Rect::new(r[0], r[1], r[2], r[3]),
                crate::layout::RectFilterMode::Intersects,
            )
        } else {
            inner.extract_text_lines(page_index)
        };

        let lines = lines_result
            .map_err(|e| JsValue::from_str(&format!("Failed to extract lines: {}", e)))?;
        serde_wasm_bindgen::to_value(&lines)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    /// Extract tables from a page (v0.3.14).
    ///
    /// @param page_index - Zero-based page number
    /// @param region - Optional [x, y, width, height] to filter by
    #[wasm_bindgen(js_name = "extractTables")]
    pub fn extract_tables(
        &mut self,
        page_index: usize,
        region: Option<Vec<f32>>,
    ) -> Result<JsValue, JsValue> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        let tables_result = if let Some(r) = region {
            if r.len() != 4 {
                return Err(JsValue::from_str("Region must have exactly 4 elements [x, y, w, h]"));
            }
            inner.extract_tables_in_rect(
                page_index,
                crate::geometry::Rect::new(r[0], r[1], r[2], r[3]),
            )
        } else {
            inner.extract_tables(page_index)
        };

        let tables = tables_result
            .map_err(|e| JsValue::from_str(&format!("Failed to extract tables: {}", e)))?;

        // Convert tables to a simplified JSON-friendly format
        let json_tables: Vec<serde_json::Value> = tables
            .iter()
            .map(|t| {
                serde_json::json!({
                    "col_count": t.col_count,
                    "row_count": t.rows.len(),
                    "bbox": t.bbox.map(|b| serde_json::json!({"x": b.x, "y": b.y, "width": b.width, "height": b.height})),
                    "has_header": t.has_header,
                    "rows": t.rows.iter().map(|r| {
                        serde_json::json!({
                            "is_header": r.is_header,
                            "cells": r.cells.iter().map(|c| {
                                serde_json::json!({
                                    "text": c.text,
                                    "bbox": c.bbox.map(|b| serde_json::json!({"x": b.x, "y": b.y, "width": b.width, "height": b.height}))
                                })
                            }).collect::<Vec<_>>()
                        })
                    }).collect::<Vec<_>>()
                })
            })
            .collect();

        serde_wasm_bindgen::to_value(&json_tables)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    // ========================================================================
    // Group 5: Search
    // ========================================================================

    /// Search for text across all pages.
    ///
    /// @param pattern - Regex pattern or literal text to search for
    /// @param case_insensitive - Case insensitive search (default: false)
    /// @param literal - Treat pattern as literal text, not regex (default: false)
    /// @param whole_word - Match whole words only (default: false)
    /// @param max_results - Maximum results to return, 0 = unlimited (default: 0)
    ///
    /// Returns an array of {page, text, bbox, start_index, end_index, span_boxes}.
    #[wasm_bindgen(js_name = "search")]
    pub fn search(
        &mut self,
        pattern: &str,
        case_insensitive: Option<bool>,
        literal: Option<bool>,
        whole_word: Option<bool>,
        max_results: Option<usize>,
    ) -> Result<JsValue, JsValue> {
        let options = SearchOptions {
            case_insensitive: case_insensitive.unwrap_or(false),
            literal: literal.unwrap_or(false),
            whole_word: whole_word.unwrap_or(false),
            max_results: max_results.unwrap_or(0),
            page_range: None,
        };
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        let results = TextSearcher::search(&mut inner, pattern, &options)
            .map_err(|e| JsValue::from_str(&format!("Search failed: {}", e)))?;
        serde_wasm_bindgen::to_value(&results)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    /// Search for text on a specific page.
    #[wasm_bindgen(js_name = "searchPage")]
    pub fn search_page(
        &mut self,
        page_index: usize,
        pattern: &str,
        case_insensitive: Option<bool>,
        literal: Option<bool>,
        whole_word: Option<bool>,
        max_results: Option<usize>,
    ) -> Result<JsValue, JsValue> {
        let options = SearchOptions {
            case_insensitive: case_insensitive.unwrap_or(false),
            literal: literal.unwrap_or(false),
            whole_word: whole_word.unwrap_or(false),
            max_results: max_results.unwrap_or(0),
            page_range: Some((page_index, page_index)),
        };
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        let results = TextSearcher::search(&mut inner, pattern, &options)
            .map_err(|e| JsValue::from_str(&format!("Search failed: {}", e)))?;
        serde_wasm_bindgen::to_value(&results)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    // ========================================================================
    // Group 6: Image Info (read-only metadata)
    // ========================================================================

    /// Extract image metadata from a page.
    ///
    /// Returns an array of objects with: width, height, color_space,
    /// bits_per_component, bbox (if available). Does NOT return raw image bytes.
    ///
    /// @param page_index - Zero-based page number
    /// @param region - Optional [x, y, width, height] to filter by
    #[wasm_bindgen(js_name = "extractImages")]
    pub fn extract_images(
        &mut self,
        page_index: usize,
        region: Option<Vec<f32>>,
    ) -> Result<JsValue, JsValue> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        let images_result = if let Some(r) = region {
            if r.len() != 4 {
                return Err(JsValue::from_str("Region must have exactly 4 elements [x, y, w, h]"));
            }
            inner.extract_images_in_rect(
                page_index,
                crate::geometry::Rect::new(r[0], r[1], r[2], r[3]),
            )
        } else {
            inner.extract_images(page_index)
        };

        let images = images_result
            .map_err(|e| JsValue::from_str(&format!("Failed to extract images: {}", e)))?;

        // Serialize image metadata (not raw bytes)
        let metadata: Vec<serde_json::Value> = images
            .iter()
            .map(|img| {
                let mut obj = serde_json::Map::new();
                obj.insert("width".into(), serde_json::Value::from(img.width()));
                obj.insert("height".into(), serde_json::Value::from(img.height()));
                obj.insert(
                    "color_space".into(),
                    serde_json::Value::from(format!("{:?}", img.color_space())),
                );
                obj.insert(
                    "bits_per_component".into(),
                    serde_json::Value::from(img.bits_per_component()),
                );
                if let Some(bbox) = img.bbox() {
                    let bbox_obj = serde_json::json!({
                        "x": bbox.x,
                        "y": bbox.y,
                        "width": bbox.width,
                        "height": bbox.height
                    });
                    obj.insert("bbox".into(), bbox_obj);
                }
                obj.insert("rotation".into(), serde_json::Value::from(img.rotation_degrees()));
                obj.insert("matrix".into(), serde_json::json!(img.matrix()));
                serde_json::Value::Object(obj)
            })
            .collect();

        serde_wasm_bindgen::to_value(&metadata)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    // ========================================================================
    // Group 6b: Document Structure (Outline, Annotations, Paths)
    // ========================================================================

    /// Get the document outline (bookmarks / table of contents).
    ///
    /// @returns Array of outline items or null if no outline exists.
    /// Each item has: { title, page (number|null), dest_name (string, optional), children (array) }
    #[wasm_bindgen(js_name = "getOutline")]
    pub fn get_outline(&mut self) -> Result<JsValue, JsValue> {
        let outline = self
            .inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?
            .get_outline()
            .map_err(|e| JsValue::from_str(&format!("Failed to get outline: {}", e)))?;

        match outline {
            None => Ok(JsValue::NULL),
            Some(items) => {
                let json = outline_to_json(&items);
                serde_wasm_bindgen::to_value(&json)
                    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
            },
        }
    }

    /// Get annotations from a page.
    ///
    /// @param page_index - Zero-based page number
    /// @returns Array of annotation objects with fields like subtype, rect, contents, etc.
    #[wasm_bindgen(js_name = "getAnnotations")]
    pub fn get_annotations(&mut self, page_index: usize) -> Result<JsValue, JsValue> {
        let annotations = self
            .inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?
            .get_annotations(page_index)
            .map_err(|e| JsValue::from_str(&format!("Failed to get annotations: {}", e)))?;

        let result: Vec<serde_json::Value> = annotations
            .iter()
            .map(|ann| {
                let mut obj = serde_json::Map::new();

                if let Some(ref subtype) = ann.subtype {
                    obj.insert("subtype".into(), serde_json::Value::from(subtype.as_str()));
                }
                if let Some(ref contents) = ann.contents {
                    obj.insert("contents".into(), serde_json::Value::from(contents.as_str()));
                }
                if let Some(rect) = ann.rect {
                    obj.insert(
                        "rect".into(),
                        serde_json::json!([rect[0], rect[1], rect[2], rect[3]]),
                    );
                }
                if let Some(ref author) = ann.author {
                    obj.insert("author".into(), serde_json::Value::from(author.as_str()));
                }
                if let Some(ref date) = ann.creation_date {
                    obj.insert("creation_date".into(), serde_json::Value::from(date.as_str()));
                }
                if let Some(ref date) = ann.modification_date {
                    obj.insert("modification_date".into(), serde_json::Value::from(date.as_str()));
                }
                if let Some(ref subject) = ann.subject {
                    obj.insert("subject".into(), serde_json::Value::from(subject.as_str()));
                }
                if let Some(ref color) = ann.color {
                    if color.len() >= 3 {
                        obj.insert(
                            "color".into(),
                            serde_json::json!([color[0], color[1], color[2]]),
                        );
                    }
                }
                if let Some(opacity) = ann.opacity {
                    obj.insert("opacity".into(), serde_json::Value::from(opacity));
                }
                if let Some(ref ft) = ann.field_type {
                    obj.insert("field_type".into(), serde_json::Value::from(format!("{:?}", ft)));
                }
                if let Some(ref name) = ann.field_name {
                    obj.insert("field_name".into(), serde_json::Value::from(name.as_str()));
                }
                if let Some(ref val) = ann.field_value {
                    obj.insert("field_value".into(), serde_json::Value::from(val.as_str()));
                }
                if let Some(crate::annotations::LinkAction::Uri(ref uri)) = ann.action {
                    obj.insert("action_uri".into(), serde_json::Value::from(uri.as_str()));
                }

                serde_json::Value::Object(obj)
            })
            .collect();

        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    /// Extract vector paths (lines, curves, shapes) from a page.
    ///
    /// @param page_index - Zero-based page number
    /// @param region - Optional [x, y, width, height] to filter by
    /// @returns Array of path objects with bbox, stroke_color, fill_color, etc.
    #[wasm_bindgen(js_name = "extractPaths")]
    pub fn extract_paths(
        &mut self,
        page_index: usize,
        region: Option<Vec<f32>>,
    ) -> Result<JsValue, JsValue> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;

        let paths_result = if let Some(r) = region {
            if r.len() != 4 {
                return Err(JsValue::from_str("Region must have exactly 4 elements [x, y, w, h]"));
            }
            inner.extract_paths_in_rect(
                page_index,
                crate::geometry::Rect::new(r[0], r[1], r[2], r[3]),
            )
        } else {
            inner.extract_paths(page_index)
        };

        let paths = paths_result
            .map_err(|e| JsValue::from_str(&format!("Failed to extract paths: {}", e)))?;

        let result: Vec<serde_json::Value> = paths
            .iter()
            .map(|path| {
                let mut obj = serde_json::Map::new();

                obj.insert(
                    "bbox".into(),
                    serde_json::json!({
                        "x": path.bbox.x,
                        "y": path.bbox.y,
                        "width": path.bbox.width,
                        "height": path.bbox.height
                    }),
                );
                obj.insert("stroke_width".into(), serde_json::Value::from(path.stroke_width));

                if let Some(ref color) = path.stroke_color {
                    obj.insert(
                        "stroke_color".into(),
                        serde_json::json!({"r": color.r, "g": color.g, "b": color.b}),
                    );
                }
                if let Some(ref color) = path.fill_color {
                    obj.insert(
                        "fill_color".into(),
                        serde_json::json!({"r": color.r, "g": color.g, "b": color.b}),
                    );
                }

                let cap_str = match path.line_cap {
                    crate::elements::LineCap::Butt => "butt",
                    crate::elements::LineCap::Round => "round",
                    crate::elements::LineCap::Square => "square",
                };
                obj.insert("line_cap".into(), serde_json::Value::from(cap_str));

                let join_str = match path.line_join {
                    crate::elements::LineJoin::Miter => "miter",
                    crate::elements::LineJoin::Round => "round",
                    crate::elements::LineJoin::Bevel => "bevel",
                };
                obj.insert("line_join".into(), serde_json::Value::from(join_str));

                obj.insert(
                    "operations_count".into(),
                    serde_json::Value::from(path.operations.len()),
                );

                serde_json::Value::Object(obj)
            })
            .collect();

        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    /// Extract only rectangles from a page (v0.3.14).
    ///
    /// Identifies paths that form axis-aligned rectangles.
    ///
    /// @param page_index - Zero-based page number
    /// @param region - Optional [x, y, width, height] to filter by
    /// @returns Array of path objects
    #[wasm_bindgen(js_name = "extractRects")]
    pub fn extract_rects(
        &mut self,
        page_index: usize,
        region: Option<Vec<f32>>,
    ) -> Result<JsValue, JsValue> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        let rects_result = if let Some(r) = region {
            if r.len() != 4 {
                return Err(JsValue::from_str("Region must have exactly 4 elements [x, y, w, h]"));
            }
            inner.extract_rects(page_index).map(|list| {
                use crate::layout::SpatialCollectionFiltering;
                list.filter_by_rect(
                    &crate::geometry::Rect::new(r[0], r[1], r[2], r[3]),
                    crate::layout::RectFilterMode::Intersects,
                )
            })
        } else {
            inner.extract_rects(page_index)
        };

        let rects = rects_result
            .map_err(|e| JsValue::from_str(&format!("Failed to extract rects: {}", e)))?;
        serde_wasm_bindgen::to_value(&rects)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    /// Extract only straight lines from a page (v0.3.14).
    ///
    /// Identifies paths that form a single straight line segment.
    ///
    /// @param page_index - Zero-based page number
    /// @param region - Optional [x, y, width, height] to filter by
    /// @returns Array of path objects
    #[wasm_bindgen(js_name = "extractLines")]
    pub fn extract_lines(
        &mut self,
        page_index: usize,
        region: Option<Vec<f32>>,
    ) -> Result<JsValue, JsValue> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        let lines_result = if let Some(r) = region {
            if r.len() != 4 {
                return Err(JsValue::from_str("Region must have exactly 4 elements [x, y, w, h]"));
            }
            inner.extract_lines(page_index).map(|list| {
                use crate::layout::SpatialCollectionFiltering;
                list.filter_by_rect(
                    &crate::geometry::Rect::new(r[0], r[1], r[2], r[3]),
                    crate::layout::RectFilterMode::Intersects,
                )
            })
        } else {
            inner.extract_lines(page_index)
        };

        let lines = lines_result
            .map_err(|e| JsValue::from_str(&format!("Failed to extract lines: {}", e)))?;
        serde_wasm_bindgen::to_value(&lines)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }
}

/// A focused view of a PDF page region for scoped extraction (v0.3.14).
#[wasm_bindgen]
#[derive(Clone)]
pub struct WasmPdfPageRegion {
    doc: WasmPdfDocument,
    page_index: usize,
    region: crate::geometry::Rect,
}

#[wasm_bindgen]
impl WasmPdfPageRegion {
    /// Extract text from this region.
    #[wasm_bindgen(js_name = "extractText")]
    pub fn extract_text(&mut self) -> Result<String, JsValue> {
        self.doc
            .inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?
            .extract_text_in_rect(
                self.page_index,
                self.region,
                crate::layout::RectFilterMode::Intersects,
            )
            .map_err(|e| JsValue::from_str(&format!("Failed to extract text: {}", e)))
    }

    /// Extract character-level data from this region.
    #[wasm_bindgen(js_name = "extractChars")]
    pub fn extract_chars(&mut self) -> Result<JsValue, JsValue> {
        self.doc.extract_chars(
            self.page_index,
            Some(vec![
                self.region.x,
                self.region.y,
                self.region.width,
                self.region.height,
            ]),
        )
    }

    /// Extract words from this region.
    #[wasm_bindgen(js_name = "extractWords")]
    pub fn extract_words(&mut self) -> Result<JsValue, JsValue> {
        self.doc.extract_words(
            self.page_index,
            Some(vec![
                self.region.x,
                self.region.y,
                self.region.width,
                self.region.height,
            ]),
        )
    }

    /// Extract text lines from this region.
    #[wasm_bindgen(js_name = "extractTextLines")]
    pub fn extract_text_lines(&mut self) -> Result<JsValue, JsValue> {
        self.doc.extract_text_lines(
            self.page_index,
            Some(vec![
                self.region.x,
                self.region.y,
                self.region.width,
                self.region.height,
            ]),
        )
    }

    /// Extract tables from this region.
    #[wasm_bindgen(js_name = "extractTables")]
    pub fn extract_tables(&mut self) -> Result<JsValue, JsValue> {
        self.doc.extract_tables(
            self.page_index,
            Some(vec![
                self.region.x,
                self.region.y,
                self.region.width,
                self.region.height,
            ]),
        )
    }

    /// Extract images from this region.
    #[wasm_bindgen(js_name = "extractImages")]
    pub fn extract_images(&mut self) -> Result<JsValue, JsValue> {
        self.doc.extract_images(
            self.page_index,
            Some(vec![
                self.region.x,
                self.region.y,
                self.region.width,
                self.region.height,
            ]),
        )
    }

    /// Extract vector paths from this region.
    #[wasm_bindgen(js_name = "extractPaths")]
    pub fn extract_paths(&mut self) -> Result<JsValue, JsValue> {
        self.doc.extract_paths(
            self.page_index,
            Some(vec![
                self.region.x,
                self.region.y,
                self.region.width,
                self.region.height,
            ]),
        )
    }

    /// Extract rectangles from this region.
    #[wasm_bindgen(js_name = "extractRects")]
    pub fn extract_rects(&mut self) -> Result<JsValue, JsValue> {
        self.doc.extract_rects(
            self.page_index,
            Some(vec![
                self.region.x,
                self.region.y,
                self.region.width,
                self.region.height,
            ]),
        )
    }

    /// Extract straight lines from this region.
    #[wasm_bindgen(js_name = "extractLines")]
    pub fn extract_lines(&mut self) -> Result<JsValue, JsValue> {
        self.doc.extract_lines(
            self.page_index,
            Some(vec![
                self.region.x,
                self.region.y,
                self.region.width,
                self.region.height,
            ]),
        )
    }

    /// Extract text using OCR from this region.
    #[wasm_bindgen(js_name = "extractTextOcr")]
    pub fn extract_text_ocr(&mut self, _engine: Option<WasmOcrEngine>) -> Result<String, JsValue> {
        Err(JsValue::from_str(
            "OCR is not yet supported in WebAssembly. Please use the Python or Rust APIs for OCR.",
        ))
    }
}

/// OCR configuration for WebAssembly.
#[wasm_bindgen]
#[derive(Clone, Default)]
pub struct WasmOcrConfig {}

#[wasm_bindgen]
impl WasmOcrConfig {
    /// Create a new OCR configuration.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }
}

/// OCR engine for WebAssembly.
#[wasm_bindgen]
pub struct WasmOcrEngine {}

#[wasm_bindgen]
impl WasmOcrEngine {
    /// Create a new OCR engine.
    #[wasm_bindgen(constructor)]
    pub fn new(
        _det_model_path: &str,
        _rec_model_path: &str,
        _dict_path: &str,
        _config: Option<WasmOcrConfig>,
    ) -> Result<WasmOcrEngine, JsValue> {
        Err(JsValue::from_str(
            "OCR is not yet supported in WebAssembly. Please use the Python or Rust APIs for OCR.",
        ))
    }
}

#[wasm_bindgen]
impl WasmPdfDocument {
    // =================================Group 6b: OCR========================================

    /// Extract text using OCR (optical character recognition).
    ///
    /// NOTE: OCR is not yet supported in the WebAssembly build due to missing
    /// ONNX Runtime support for the web backend in the current implementation.
    #[wasm_bindgen(js_name = "extractTextOcr")]
    pub fn extract_text_ocr(
        &mut self,
        _page_index: usize,
        _engine: Option<WasmOcrEngine>,
    ) -> Result<String, JsValue> {
        Err(JsValue::from_str(
            "OCR is not yet supported in WebAssembly. Please use the Python or Rust APIs for OCR.",
        ))
    }

    // ========================================================================
    // Group 6c: Form Fields
    // ========================================================================

    /// Get all form fields from the document.
    ///
    /// Returns an array of form field objects, each with:
    /// - name: Full qualified field name
    /// - field_type: "text", "button", "choice", "signature", or "unknown"
    /// - value: string, boolean, array of strings, or null
    /// - tooltip: string or null
    /// - bounds: [x1, y1, x2, y2] or null
    /// - flags: number or null
    /// - max_length: number or null
    /// - is_readonly: boolean
    /// - is_required: boolean
    #[wasm_bindgen(js_name = "getFormFields")]
    pub fn get_form_fields(&mut self) -> Result<JsValue, JsValue> {
        use crate::extractors::forms::{field_flags, FieldType, FieldValue, FormExtractor};

        let mut inner = self
            .inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        let fields = FormExtractor::extract_fields(&mut inner)
            .map_err(|e| JsValue::from_str(&format!("Failed to extract form fields: {}", e)))?;

        let result: Vec<serde_json::Value> = fields
            .iter()
            .map(|field| {
                let mut obj = serde_json::Map::new();

                obj.insert("name".into(), serde_json::Value::from(field.full_name.as_str()));

                let ft_str = match &field.field_type {
                    FieldType::Text => "text",
                    FieldType::Button => "button",
                    FieldType::Choice => "choice",
                    FieldType::Signature => "signature",
                    FieldType::Unknown(_) => "unknown",
                };
                obj.insert("field_type".into(), serde_json::Value::from(ft_str));

                let value = match &field.value {
                    FieldValue::Text(s) => serde_json::Value::from(s.as_str()),
                    FieldValue::Name(s) => serde_json::Value::from(s.as_str()),
                    FieldValue::Boolean(b) => serde_json::Value::from(*b),
                    FieldValue::Array(v) => serde_json::Value::Array(
                        v.iter()
                            .map(|s| serde_json::Value::from(s.as_str()))
                            .collect(),
                    ),
                    FieldValue::None => serde_json::Value::Null,
                };
                obj.insert("value".into(), value);

                match &field.tooltip {
                    Some(t) => obj.insert("tooltip".into(), serde_json::Value::from(t.as_str())),
                    None => obj.insert("tooltip".into(), serde_json::Value::Null),
                };

                match &field.bounds {
                    Some(b) => {
                        obj.insert("bounds".into(), serde_json::json!([b[0], b[1], b[2], b[3]]))
                    },
                    None => obj.insert("bounds".into(), serde_json::Value::Null),
                };

                match field.flags {
                    Some(f) => {
                        obj.insert("flags".into(), serde_json::Value::from(f));
                        obj.insert(
                            "is_readonly".into(),
                            serde_json::Value::from(f & field_flags::READ_ONLY != 0),
                        );
                        obj.insert(
                            "is_required".into(),
                            serde_json::Value::from(f & field_flags::REQUIRED != 0),
                        );
                    },
                    None => {
                        obj.insert("flags".into(), serde_json::Value::Null);
                        obj.insert("is_readonly".into(), serde_json::Value::from(false));
                        obj.insert("is_required".into(), serde_json::Value::from(false));
                    },
                };

                match field.max_length {
                    Some(ml) => obj.insert("max_length".into(), serde_json::Value::from(ml)),
                    None => obj.insert("max_length".into(), serde_json::Value::Null),
                };

                serde_json::Value::Object(obj)
            })
            .collect();

        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    /// Check if the document contains XFA form data.
    ///
    /// @returns true if the document has XFA form data
    #[wasm_bindgen(js_name = "hasXfa")]
    pub fn has_xfa(&mut self) -> Result<bool, JsValue> {
        use crate::xfa::XfaExtractor;

        let mut inner = self
            .inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        XfaExtractor::has_xfa(&mut inner)
            .map_err(|e| JsValue::from_str(&format!("Failed to check XFA: {}", e)))
    }

    /// Export form field data as FDF or XFDF bytes.
    ///
    /// @param format - "fdf" or "xfdf" (default: "fdf")
    /// @returns Uint8Array containing the exported form data
    #[wasm_bindgen(js_name = "exportFormData")]
    pub fn export_form_data(&mut self, format: Option<String>) -> Result<Vec<u8>, JsValue> {
        let fmt = format.as_deref().unwrap_or("fdf");

        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;

        // Write to a temporary in-memory buffer via a temp file path
        let tmp_path = "/tmp/pdf_oxide_form_export_wasm.tmp";
        match fmt {
            "fdf" => editor
                .export_form_data_fdf(tmp_path)
                .map_err(|e| JsValue::from_str(&format!("Failed to export FDF: {}", e)))?,
            "xfdf" => editor
                .export_form_data_xfdf(tmp_path)
                .map_err(|e| JsValue::from_str(&format!("Failed to export XFDF: {}", e)))?,
            _ => {
                return Err(JsValue::from_str(&format!(
                    "Unknown format '{}'. Use 'fdf' or 'xfdf'.",
                    fmt
                )))
            },
        }

        let bytes = std::fs::read(tmp_path)
            .map_err(|e| JsValue::from_str(&format!("Failed to read exported data: {}", e)))?;
        let _ = std::fs::remove_file(tmp_path);
        Ok(bytes)
    }

    // ========================================================================
    // Group 6d: Form Field Get/Set Values
    // ========================================================================

    /// Get the value of a specific form field by name.
    ///
    /// @param name - Full qualified field name (e.g., "name" or "topmostSubform[0].Page1[0].f1_01[0]")
    /// @returns The field value: string for text, boolean for checkbox, null if not found
    #[wasm_bindgen(js_name = "getFormFieldValue")]
    pub fn get_form_field_value(&mut self, name: &str) -> Result<JsValue, JsValue> {
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        let value = editor
            .get_form_field_value(name)
            .map_err(|e| JsValue::from_str(&format!("Failed to get field value: {}", e)))?;

        match value {
            Some(v) => wasm_form_field_value_to_js(&v),
            None => Ok(JsValue::NULL),
        }
    }

    /// Set the value of a form field.
    ///
    /// @param name - Full qualified field name
    /// @param value - New value: string for text fields, boolean for checkboxes
    #[wasm_bindgen(js_name = "setFormFieldValue")]
    pub fn set_form_field_value(&mut self, name: &str, value: JsValue) -> Result<(), JsValue> {
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        let field_value = js_to_form_field_value(&value)?;
        editor
            .set_form_field_value(name, field_value)
            .map_err(|e| JsValue::from_str(&format!("Failed to set field value: {}", e)))
    }

    // ========================================================================
    // Group 6e: Image Bytes Extraction
    // ========================================================================

    /// Extract image bytes from a page as PNG data.
    ///
    /// Returns an array of objects with: width, height, data (Uint8Array of PNG bytes), format ("png").
    #[wasm_bindgen(js_name = "extractImageBytes")]
    pub fn extract_image_bytes(&mut self, page_index: usize) -> Result<JsValue, JsValue> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        let images = inner
            .extract_images(page_index)
            .map_err(|e| JsValue::from_str(&format!("Failed to extract images: {}", e)))?;

        let arr = js_sys::Array::new();
        for img in &images {
            let png_data = img.to_png_bytes().map_err(|e| {
                JsValue::from_str(&format!("Failed to convert image to PNG: {}", e))
            })?;

            let obj = js_sys::Object::new();
            js_sys::Reflect::set(&obj, &JsValue::from_str("width"), &JsValue::from(img.width()))?;
            js_sys::Reflect::set(&obj, &JsValue::from_str("height"), &JsValue::from(img.height()))?;
            js_sys::Reflect::set(&obj, &JsValue::from_str("format"), &JsValue::from_str("png"))?;
            let uint8_array = js_sys::Uint8Array::from(png_data.as_slice());
            js_sys::Reflect::set(&obj, &JsValue::from_str("data"), &uint8_array)?;
            arr.push(&obj);
        }
        Ok(arr.into())
    }

    // ========================================================================
    // Group 6f: Form Flattening
    // ========================================================================

    /// Flatten all form fields into page content.
    ///
    /// After flattening, form field values become static text and are no longer editable.
    #[wasm_bindgen(js_name = "flattenForms")]
    pub fn flatten_forms(&mut self) -> Result<(), JsValue> {
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        editor
            .flatten_forms()
            .map_err(|e| JsValue::from_str(&format!("Failed to flatten forms: {}", e)))
    }

    /// Flatten form fields on a specific page.
    ///
    /// @param page_index - Zero-based page number
    #[wasm_bindgen(js_name = "flattenFormsOnPage")]
    pub fn flatten_forms_on_page(&mut self, page_index: usize) -> Result<(), JsValue> {
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        editor
            .flatten_forms_on_page(page_index)
            .map_err(|e| JsValue::from_str(&format!("Failed to flatten forms on page: {}", e)))
    }

    // ========================================================================
    // Group 6g: PDF Merging
    // ========================================================================

    /// Merge another PDF (provided as bytes) into this document.
    ///
    /// @param data - The PDF file contents to merge as a Uint8Array
    /// @returns Number of pages merged
    #[wasm_bindgen(js_name = "mergeFrom")]
    pub fn merge_from(&mut self, data: &[u8]) -> Result<usize, JsValue> {
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        editor
            .merge_from_bytes(data)
            .map_err(|e| JsValue::from_str(&format!("Failed to merge PDF: {}", e)))
    }

    // ========================================================================
    // Group 6h: File Embedding
    // ========================================================================

    /// Embed a file into the PDF document.
    ///
    /// @param name - Display name for the embedded file
    /// @param data - File contents as a Uint8Array
    #[wasm_bindgen(js_name = "embedFile")]
    pub fn embed_file(&mut self, name: &str, data: &[u8]) -> Result<(), JsValue> {
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        editor
            .embed_file(name, data.to_vec())
            .map_err(|e| JsValue::from_str(&format!("Failed to embed file: {}", e)))
    }

    // ========================================================================
    // Group 6i: Page Labels
    // ========================================================================

    /// Get page label ranges from the document.
    ///
    /// @returns Array of {start_page, style, prefix, start_value} objects, or empty array
    #[wasm_bindgen(js_name = "pageLabels")]
    pub fn page_labels(&mut self) -> Result<JsValue, JsValue> {
        use crate::extractors::page_labels::PageLabelExtractor;

        let mut inner = self
            .inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        let labels = PageLabelExtractor::extract(&mut inner)
            .map_err(|e| JsValue::from_str(&format!("Failed to get page labels: {}", e)))?;

        let result: Vec<serde_json::Value> = labels
            .iter()
            .map(|label| {
                let mut obj = serde_json::Map::new();
                obj.insert("start_page".into(), serde_json::Value::from(label.start_page));
                obj.insert("style".into(), serde_json::Value::from(format!("{:?}", label.style)));
                match &label.prefix {
                    Some(p) => obj.insert("prefix".into(), serde_json::Value::from(p.as_str())),
                    None => obj.insert("prefix".into(), serde_json::Value::Null),
                };
                obj.insert("start_value".into(), serde_json::Value::from(label.start_value));
                serde_json::Value::Object(obj)
            })
            .collect();

        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    // ========================================================================
    // Group 6j: XMP Metadata
    // ========================================================================

    /// Get XMP metadata from the document.
    ///
    /// @returns Object with XMP fields (dc_title, dc_creator, etc.) or null if no XMP
    #[wasm_bindgen(js_name = "xmpMetadata")]
    pub fn xmp_metadata(&mut self) -> Result<JsValue, JsValue> {
        use crate::extractors::xmp::XmpExtractor;

        let mut inner = self
            .inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        let metadata = XmpExtractor::extract(&mut inner)
            .map_err(|e| JsValue::from_str(&format!("Failed to get XMP metadata: {}", e)))?;

        match metadata {
            None => Ok(JsValue::NULL),
            Some(xmp) => {
                let mut obj = serde_json::Map::new();
                if let Some(ref title) = xmp.dc_title {
                    obj.insert("dc_title".into(), serde_json::Value::from(title.as_str()));
                }
                if !xmp.dc_creator.is_empty() {
                    obj.insert(
                        "dc_creator".into(),
                        serde_json::Value::Array(
                            xmp.dc_creator
                                .iter()
                                .map(|s| serde_json::Value::from(s.as_str()))
                                .collect(),
                        ),
                    );
                }
                if let Some(ref desc) = xmp.dc_description {
                    obj.insert("dc_description".into(), serde_json::Value::from(desc.as_str()));
                }
                if !xmp.dc_subject.is_empty() {
                    obj.insert(
                        "dc_subject".into(),
                        serde_json::Value::Array(
                            xmp.dc_subject
                                .iter()
                                .map(|s| serde_json::Value::from(s.as_str()))
                                .collect(),
                        ),
                    );
                }
                if let Some(ref lang) = xmp.dc_language {
                    obj.insert("dc_language".into(), serde_json::Value::from(lang.as_str()));
                }
                if let Some(ref tool) = xmp.xmp_creator_tool {
                    obj.insert("xmp_creator_tool".into(), serde_json::Value::from(tool.as_str()));
                }
                if let Some(ref date) = xmp.xmp_create_date {
                    obj.insert("xmp_create_date".into(), serde_json::Value::from(date.as_str()));
                }
                if let Some(ref date) = xmp.xmp_modify_date {
                    obj.insert("xmp_modify_date".into(), serde_json::Value::from(date.as_str()));
                }
                if let Some(ref producer) = xmp.pdf_producer {
                    obj.insert("pdf_producer".into(), serde_json::Value::from(producer.as_str()));
                }
                if let Some(ref keywords) = xmp.pdf_keywords {
                    obj.insert("pdf_keywords".into(), serde_json::Value::from(keywords.as_str()));
                }

                serde_wasm_bindgen::to_value(&serde_json::Value::Object(obj))
                    .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
            },
        }
    }

    // ========================================================================
    // Group 7: Editing — Metadata
    // ========================================================================

    /// Set the document title.
    #[wasm_bindgen(js_name = "setTitle")]
    pub fn set_title(&mut self, title: &str) -> Result<(), JsValue> {
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        editor.set_title(title);
        Ok(())
    }

    /// Set the document author.
    #[wasm_bindgen(js_name = "setAuthor")]
    pub fn set_author(&mut self, author: &str) -> Result<(), JsValue> {
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        editor.set_author(author);
        Ok(())
    }

    /// Set the document subject.
    #[wasm_bindgen(js_name = "setSubject")]
    pub fn set_subject(&mut self, subject: &str) -> Result<(), JsValue> {
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        editor.set_subject(subject);
        Ok(())
    }

    /// Set the document keywords.
    #[wasm_bindgen(js_name = "setKeywords")]
    pub fn set_keywords(&mut self, keywords: &str) -> Result<(), JsValue> {
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        editor.set_keywords(keywords);
        Ok(())
    }

    // ========================================================================
    // Group 7: Editing — Page Properties
    // ========================================================================

    /// Get the rotation of a page in degrees (0, 90, 180, 270).
    #[wasm_bindgen(js_name = "pageRotation")]
    pub fn page_rotation(&mut self, page_index: usize) -> Result<i32, JsValue> {
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        editor
            .get_page_rotation(page_index)
            .map_err(|e| JsValue::from_str(&format!("Failed to get rotation: {}", e)))
    }

    /// Set the rotation of a page (0, 90, 180, or 270 degrees).
    #[wasm_bindgen(js_name = "setPageRotation")]
    pub fn set_page_rotation(&mut self, page_index: usize, degrees: i32) -> Result<(), JsValue> {
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        editor
            .set_page_rotation(page_index, degrees)
            .map_err(|e| JsValue::from_str(&format!("Failed to set rotation: {}", e)))
    }

    /// Rotate a page by the given degrees (adds to current rotation).
    #[wasm_bindgen(js_name = "rotatePage")]
    pub fn rotate_page(&mut self, page_index: usize, degrees: i32) -> Result<(), JsValue> {
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        editor
            .rotate_page_by(page_index, degrees)
            .map_err(|e| JsValue::from_str(&format!("Failed to rotate page: {}", e)))
    }

    /// Rotate all pages by the given degrees.
    #[wasm_bindgen(js_name = "rotateAllPages")]
    pub fn rotate_all_pages(&mut self, degrees: i32) -> Result<(), JsValue> {
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        editor
            .rotate_all_pages(degrees)
            .map_err(|e| JsValue::from_str(&format!("Failed to rotate all pages: {}", e)))
    }

    /// Get the MediaBox of a page as [llx, lly, urx, ury].
    #[wasm_bindgen(js_name = "pageMediaBox")]
    pub fn page_media_box(&mut self, page_index: usize) -> Result<Vec<f32>, JsValue> {
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        let mbox = editor
            .get_page_media_box(page_index)
            .map_err(|e| JsValue::from_str(&format!("Failed to get media box: {}", e)))?;
        Ok(mbox.to_vec())
    }

    /// Set the MediaBox of a page.
    #[wasm_bindgen(js_name = "setPageMediaBox")]
    pub fn set_page_media_box(
        &mut self,
        page_index: usize,
        llx: f32,
        lly: f32,
        urx: f32,
        ury: f32,
    ) -> Result<(), JsValue> {
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        editor
            .set_page_media_box(page_index, [llx, lly, urx, ury])
            .map_err(|e| JsValue::from_str(&format!("Failed to set media box: {}", e)))
    }

    /// Get the CropBox of a page as [llx, lly, urx, ury], or null if not set.
    #[wasm_bindgen(js_name = "pageCropBox")]
    pub fn page_crop_box(&mut self, page_index: usize) -> Result<JsValue, JsValue> {
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        let cbox = editor
            .get_page_crop_box(page_index)
            .map_err(|e| JsValue::from_str(&format!("Failed to get crop box: {}", e)))?;
        match cbox {
            Some(b) => serde_wasm_bindgen::to_value(&b.to_vec())
                .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e))),
            None => Ok(JsValue::NULL),
        }
    }

    /// Set the CropBox of a page.
    #[wasm_bindgen(js_name = "setPageCropBox")]
    pub fn set_page_crop_box(
        &mut self,
        page_index: usize,
        llx: f32,
        lly: f32,
        urx: f32,
        ury: f32,
    ) -> Result<(), JsValue> {
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        editor
            .set_page_crop_box(page_index, [llx, lly, urx, ury])
            .map_err(|e| JsValue::from_str(&format!("Failed to set crop box: {}", e)))
    }

    /// Crop margins from all pages.
    #[wasm_bindgen(js_name = "cropMargins")]
    pub fn crop_margins(
        &mut self,
        left: f32,
        right: f32,
        top: f32,
        bottom: f32,
    ) -> Result<(), JsValue> {
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        editor
            .crop_margins(left, right, top, bottom)
            .map_err(|e| JsValue::from_str(&format!("Failed to crop margins: {}", e)))
    }

    // ========================================================================
    // Group 7: Editing — Erase / Whiteout
    // ========================================================================

    /// Erase (whiteout) a rectangular region on a page.
    #[wasm_bindgen(js_name = "eraseRegion")]
    pub fn erase_region(
        &mut self,
        page_index: usize,
        llx: f32,
        lly: f32,
        urx: f32,
        ury: f32,
    ) -> Result<(), JsValue> {
        // Mark in inner document for extraction filtering
        self.inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?
            .erase_region(page_index, crate::geometry::Rect::new(llx, lly, urx - llx, ury - lly))
            .map_err(|e| JsValue::from_str(&format!("Failed to mark region: {}", e)))?;

        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        editor
            .erase_region(page_index, [llx, lly, urx, ury])
            .map_err(|e| JsValue::from_str(&format!("Failed to erase region: {}", e)))
    }

    /// Erase multiple rectangular regions on a page.
    ///
    /// @param page_index - Zero-based page number
    /// @param rects - Flat array of coordinates [llx1,lly1,urx1,ury1, llx2,lly2,urx2,ury2, ...]
    #[wasm_bindgen(js_name = "eraseRegions")]
    pub fn erase_regions(&mut self, page_index: usize, rects: &[f32]) -> Result<(), JsValue> {
        if !rects.len().is_multiple_of(4) {
            return Err(JsValue::from_str("rects must have a length that is a multiple of 4"));
        }

        // Mark all regions in inner document
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        for chunk in rects.chunks_exact(4) {
            let (llx, lly, urx, ury) = (chunk[0], chunk[1], chunk[2], chunk[3]);
            inner
                .erase_region(
                    page_index,
                    crate::geometry::Rect::new(llx, lly, urx - llx, ury - lly),
                )
                .map_err(|e| JsValue::from_str(&format!("Failed to mark region: {}", e)))?;
        }
        drop(inner);

        let rect_arrays: Vec<[f32; 4]> = rects
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect();
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        editor
            .erase_regions(page_index, &rect_arrays)
            .map_err(|e| JsValue::from_str(&format!("Failed to erase regions: {}", e)))
    }

    /// Clear all pending erase operations for a page.
    #[wasm_bindgen(js_name = "clearEraseRegions")]
    pub fn clear_erase_regions(&mut self, page_index: usize) -> Result<(), JsValue> {
        // Clear inner document regions
        self.inner
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?
            .clear_erase_regions(page_index)
            .map_err(|e| JsValue::from_str(&format!("Failed to clear regions: {}", e)))?;

        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        editor.clear_erase_regions(page_index);
        Ok(())
    }

    // ========================================================================
    // Group 7: Editing — Annotations
    // ========================================================================

    /// Flatten annotations on a page into the page content.
    #[wasm_bindgen(js_name = "flattenPageAnnotations")]
    pub fn flatten_page_annotations(&mut self, page_index: usize) -> Result<(), JsValue> {
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        editor
            .flatten_page_annotations(page_index)
            .map_err(|e| JsValue::from_str(&format!("Failed to flatten annotations: {}", e)))
    }

    /// Flatten all annotations in the document into page content.
    #[wasm_bindgen(js_name = "flattenAllAnnotations")]
    pub fn flatten_all_annotations(&mut self) -> Result<(), JsValue> {
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        editor
            .flatten_all_annotations()
            .map_err(|e| JsValue::from_str(&format!("Failed to flatten annotations: {}", e)))
    }

    // ========================================================================
    // Group 7: Editing — Redaction
    // ========================================================================

    /// Apply redactions on a page (removes redacted content permanently).
    #[wasm_bindgen(js_name = "applyPageRedactions")]
    pub fn apply_page_redactions(&mut self, page_index: usize) -> Result<(), JsValue> {
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        editor
            .apply_page_redactions(page_index)
            .map_err(|e| JsValue::from_str(&format!("Failed to apply redactions: {}", e)))
    }

    /// Apply all redactions in the document.
    #[wasm_bindgen(js_name = "applyAllRedactions")]
    pub fn apply_all_redactions(&mut self) -> Result<(), JsValue> {
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        editor
            .apply_all_redactions()
            .map_err(|e| JsValue::from_str(&format!("Failed to apply redactions: {}", e)))
    }
}

/// Style configuration for header/footer text.
#[wasm_bindgen(js_name = "ArtifactStyle")]
#[derive(Clone)]
pub struct WasmArtifactStyle {
    inner: crate::writer::ArtifactStyle,
}

impl Default for WasmArtifactStyle {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen(js_name = "ArtifactStyle")]
impl WasmArtifactStyle {
    /// Create a new artifact style.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: crate::writer::ArtifactStyle::new(),
        }
    }

    /// Set font for the artifact.
    pub fn font(mut self, name: &str, size: f32) -> Self {
        self.inner = self.inner.font(name, size);
        self
    }

    /// Set bold font for the artifact.
    pub fn bold(mut self) -> Self {
        self.inner = self.inner.bold();
        self
    }

    /// Set color for the artifact.
    pub fn color(mut self, r: f32, g: f32, b: f32) -> Self {
        self.inner = self.inner.color(r, g, b);
        self
    }
}

/// A header or footer artifact definition.
#[wasm_bindgen]
#[derive(Clone)]
pub struct WasmArtifact {
    inner: crate::writer::Artifact,
}

impl Default for WasmArtifact {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl WasmArtifact {
    /// Create a new artifact.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: crate::writer::Artifact::new(),
        }
    }

    /// Create a left-aligned artifact.
    #[wasm_bindgen(js_name = "left", static_method_of = WasmArtifact)]
    pub fn left(text: &str) -> WasmArtifact {
        WasmArtifact {
            inner: crate::writer::Artifact::left(text),
        }
    }

    /// Create a center-aligned artifact.
    #[wasm_bindgen(js_name = "center", static_method_of = WasmArtifact)]
    pub fn center(text: &str) -> WasmArtifact {
        WasmArtifact {
            inner: crate::writer::Artifact::center(text),
        }
    }

    /// Create a right-aligned artifact.
    #[wasm_bindgen(js_name = "right", static_method_of = WasmArtifact)]
    pub fn right(text: &str) -> WasmArtifact {
        WasmArtifact {
            inner: crate::writer::Artifact::right(text),
        }
    }

    /// Set style for the artifact.
    #[wasm_bindgen(js_name = "withStyle")]
    pub fn with_style(mut self, style: &WasmArtifactStyle) -> Self {
        self.inner = self.inner.with_style(style.inner.clone());
        self
    }

    /// Set vertical offset for the artifact.
    #[wasm_bindgen(js_name = "withOffset")]
    pub fn with_offset(mut self, offset: f32) -> Self {
        self.inner = self.inner.with_offset(offset);
        self
    }
}

/// A header definition.
#[wasm_bindgen]
#[derive(Clone)]
pub struct WasmHeader {
    inner: WasmArtifact,
}

impl Default for WasmHeader {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl WasmHeader {
    /// Create a new empty header.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: WasmArtifact::new(),
        }
    }

    /// Create a left-aligned header.
    #[wasm_bindgen(js_name = "left", static_method_of = WasmHeader)]
    pub fn left(text: &str) -> WasmHeader {
        WasmHeader {
            inner: WasmArtifact::left(text),
        }
    }

    /// Create a center-aligned header.
    #[wasm_bindgen(js_name = "center", static_method_of = WasmHeader)]
    pub fn center(text: &str) -> WasmHeader {
        WasmHeader {
            inner: WasmArtifact::center(text),
        }
    }

    /// Create a right-aligned header.
    #[wasm_bindgen(js_name = "right", static_method_of = WasmHeader)]
    pub fn right(text: &str) -> WasmHeader {
        WasmHeader {
            inner: WasmArtifact::right(text),
        }
    }
}

/// A footer definition.
#[wasm_bindgen]
#[derive(Clone)]
pub struct WasmFooter {
    inner: WasmArtifact,
}

impl Default for WasmFooter {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl WasmFooter {
    /// Create a new empty footer.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: WasmArtifact::new(),
        }
    }

    /// Create a left-aligned footer.
    #[wasm_bindgen(js_name = "left", static_method_of = WasmFooter)]
    pub fn left(text: &str) -> WasmFooter {
        WasmFooter {
            inner: WasmArtifact::left(text),
        }
    }

    /// Create a center-aligned footer.
    #[wasm_bindgen(js_name = "center", static_method_of = WasmFooter)]
    pub fn center(text: &str) -> WasmFooter {
        WasmFooter {
            inner: WasmArtifact::center(text),
        }
    }

    /// Create a right-aligned footer.
    #[wasm_bindgen(js_name = "right", static_method_of = WasmFooter)]
    pub fn right(text: &str) -> WasmFooter {
        WasmFooter {
            inner: WasmArtifact::right(text),
        }
    }
}

/// A complete page template with header and footer.
#[wasm_bindgen]
#[derive(Clone)]
pub struct WasmPageTemplate {
    inner: crate::writer::PageTemplate,
}

impl Default for WasmPageTemplate {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl WasmPageTemplate {
    /// Create a new page template.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: crate::writer::PageTemplate::new(),
        }
    }

    /// Set header artifact.
    pub fn header(mut self, header: &WasmArtifact) -> Self {
        self.inner = self.inner.header(header.inner.clone());
        self
    }

    /// Set footer artifact.
    pub fn footer(mut self, footer: &WasmArtifact) -> Self {
        self.inner = self.inner.footer(footer.inner.clone());
        self
    }

    /// Skip rendering template on the first page.
    #[wasm_bindgen(js_name = "skipFirstPage")]
    pub fn skip_first_page(mut self) -> Self {
        self.inner = self.inner.skip_first_page();
        self
    }
}

#[wasm_bindgen]
impl WasmPdfDocument {
    // ========================================================================
    // Group 7: Editing — Image Manipulation
    // ========================================================================

    /// Get information about images on a page.
    ///
    /// Returns an array of {name, bounds: [x, y, width, height], matrix: [a, b, c, d, e, f]}.
    #[wasm_bindgen(js_name = "pageImages")]
    pub fn page_images(&mut self, page_index: usize) -> Result<JsValue, JsValue> {
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        let images = editor
            .get_page_images(page_index)
            .map_err(|e| JsValue::from_str(&format!("Failed to get page images: {}", e)))?;
        serde_wasm_bindgen::to_value(&images)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    /// Reposition an image on a page.
    #[wasm_bindgen(js_name = "repositionImage")]
    pub fn reposition_image(
        &mut self,
        page_index: usize,
        name: &str,
        x: f32,
        y: f32,
    ) -> Result<(), JsValue> {
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        editor
            .reposition_image(page_index, name, x, y)
            .map_err(|e| JsValue::from_str(&format!("Failed to reposition image: {}", e)))
    }

    /// Resize an image on a page.
    #[wasm_bindgen(js_name = "resizeImage")]
    pub fn resize_image(
        &mut self,
        page_index: usize,
        name: &str,
        width: f32,
        height: f32,
    ) -> Result<(), JsValue> {
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        editor
            .resize_image(page_index, name, width, height)
            .map_err(|e| JsValue::from_str(&format!("Failed to resize image: {}", e)))
    }

    /// Set the complete bounds of an image on a page.
    #[wasm_bindgen(js_name = "setImageBounds")]
    pub fn set_image_bounds(
        &mut self,
        page_index: usize,
        name: &str,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    ) -> Result<(), JsValue> {
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        editor
            .set_image_bounds(page_index, name, x, y, width, height)
            .map_err(|e| JsValue::from_str(&format!("Failed to set image bounds: {}", e)))
    }

    // ========================================================================
    // Group 7: Editing — Save
    // ========================================================================

    /// Save all edits and return the resulting PDF as bytes.
    ///
    /// @returns Uint8Array containing the modified PDF
    #[wasm_bindgen(js_name = "save")]
    pub fn save(&mut self) -> Result<Vec<u8>, JsValue> {
        self.save_to_bytes()
    }

    /// Save the modified PDF and return as bytes.
    /// `saveToBytes()` is the original method; `save()` is a convenience alias.
    ///
    /// @returns Uint8Array containing the modified PDF
    #[wasm_bindgen(js_name = "saveToBytes")]
    pub fn save_to_bytes(&mut self) -> Result<Vec<u8>, JsValue> {
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        editor
            .save_to_bytes()
            .map_err(|e| JsValue::from_str(&format!("Failed to save PDF: {}", e)))
    }

    /// Save with encryption and return the resulting PDF as bytes.
    #[wasm_bindgen(js_name = "saveEncryptedToBytes")]
    pub fn save_encrypted_to_bytes(
        &mut self,
        user_password: &str,
        owner_password: Option<String>,
        allow_print: Option<bool>,
        allow_copy: Option<bool>,
        allow_modify: Option<bool>,
        allow_annotate: Option<bool>,
    ) -> Result<Vec<u8>, JsValue> {
        let owner_pwd = owner_password.as_deref().unwrap_or(user_password);

        let permissions = Permissions {
            print: allow_print.unwrap_or(true),
            print_high_quality: allow_print.unwrap_or(true),
            modify: allow_modify.unwrap_or(true),
            copy: allow_copy.unwrap_or(true),
            annotate: allow_annotate.unwrap_or(true),
            fill_forms: allow_annotate.unwrap_or(true),
            accessibility: true,
            assemble: allow_modify.unwrap_or(true),
        };

        let config = EncryptionConfig::new(user_password, owner_pwd)
            .with_algorithm(EncryptionAlgorithm::Aes256)
            .with_permissions(permissions);

        let options = SaveOptions::with_encryption(config);
        let editor_arc = self.ensure_editor()?;
        let mut editor = editor_arc
            .lock()
            .map_err(|_| JsValue::from_str("Mutex lock failed"))?;
        editor
            .save_to_bytes_with_options(options)
            .map_err(|e| JsValue::from_str(&format!("Failed to save encrypted PDF: {}", e)))
    }

    // ========================================================================
    // Group 9: Validation — PDF/A, PDF/UA, PDF/X
    // ========================================================================

    /// Validate PDF/A compliance. Level: "1b", "2b", etc.
    #[wasm_bindgen(js_name = "validatePdfA")]
    pub fn validate_pdf_a(&mut self, level: &str) -> Result<JsValue, JsValue> {
        use crate::compliance::pdf_a::validate_pdf_a;
        use crate::compliance::types::PdfALevel;
        let pdf_level = match level {
            "1a" => PdfALevel::A1a,
            "1b" => PdfALevel::A1b,
            "2a" => PdfALevel::A2a,
            "2b" => PdfALevel::A2b,
            "2u" => PdfALevel::A2u,
            "3a" => PdfALevel::A3a,
            "3b" => PdfALevel::A3b,
            "3u" => PdfALevel::A3u,
            _ => return Err(JsValue::from_str(&format!("Unknown PDF/A level: {}", level))),
        };
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| JsValue::from_str("Lock failed"))?;
        let result =
            validate_pdf_a(&mut inner, pdf_level).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let errors: Vec<String> = result.errors.iter().map(|e| e.to_string()).collect();
        let warnings: Vec<String> = result.warnings.iter().map(|w| w.to_string()).collect();
        serde_wasm_bindgen::to_value(&serde_json::json!({
            "valid": errors.is_empty(),
            "level": level,
            "errors": errors,
            "warnings": warnings,
        }))
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    // ========================================================================
    // Group 10: Annotations
    // Note: add_link, add_highlight, add_note require editor API rework
    // to properly support PdfPage annotations. Tracked for future release.

    // ========================================================================
    // Group 11: Page Operations
    // ========================================================================

    /// Delete a page by index (0-based).
    #[wasm_bindgen(js_name = "deletePage")]
    pub fn delete_page(&mut self, index: usize) -> Result<(), JsValue> {
        use crate::editor::EditableDocument;
        let bytes = self.raw_bytes.to_vec();
        let mut editor = crate::editor::DocumentEditor::from_bytes(bytes)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        editor
            .remove_page(index)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let new_bytes = editor
            .save_to_bytes()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let new_doc = crate::document::PdfDocument::from_bytes(new_bytes.clone())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| JsValue::from_str("Lock failed"))?;
        *inner = new_doc;
        self.raw_bytes = Arc::new(new_bytes);
        Ok(())
    }

    /// Extract specific pages to a new PDF (returns bytes).
    #[wasm_bindgen(js_name = "extractPages")]
    pub fn extract_pages(&mut self, pages: Vec<usize>) -> Result<Vec<u8>, JsValue> {
        use crate::editor::EditableDocument;
        let bytes = self.raw_bytes.to_vec();
        let mut editor = crate::editor::DocumentEditor::from_bytes(bytes)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        // Keep only the requested pages by removing others in reverse order
        let page_count = editor
            .page_count()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        for i in (0..page_count).rev() {
            if !pages.contains(&i) {
                let _ = editor.remove_page(i);
            }
        }
        editor
            .save_to_bytes()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Create a flattened PDF where each page is rendered as an image.
    /// Burns in all annotations, form fields, and overlays.
    /// Returns the flattened PDF as bytes.
    #[cfg(feature = "rendering")]
    #[wasm_bindgen(js_name = "flattenToImages")]
    pub fn flatten_to_images(&mut self, dpi: Option<u32>) -> Result<Vec<u8>, JsValue> {
        let dpi = dpi.unwrap_or(150);
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| JsValue::from_str("Lock failed"))?;
        crate::rendering::flatten_to_images(&mut inner, dpi)
            .map_err(|e| JsValue::from_str(&format!("Failed to flatten: {}", e)))
    }
}

// ============================================================================
// WasmPdf — PDF creation from content
// ============================================================================

/// Create new PDF documents from Markdown, HTML, or plain text.
///
/// ```javascript
/// const pdf = WasmPdf.fromMarkdown("# Hello\n\nWorld");
/// const bytes = pdf.toBytes(); // Uint8Array
/// console.log(`PDF size: ${pdf.size} bytes`);
/// ```
#[wasm_bindgen]
pub struct WasmPdf {
    bytes: Vec<u8>,
}

#[wasm_bindgen]
impl WasmPdf {
    /// Merge multiple PDF byte arrays into a single PDF.
    ///
    /// @param pdfs - Array of Uint8Array, each containing a PDF
    /// @returns WasmPdf containing all pages
    #[wasm_bindgen(js_name = "merge")]
    pub fn merge(pdfs: Vec<js_sys::Uint8Array>) -> Result<WasmPdf, JsValue> {
        if pdfs.is_empty() {
            return Err(JsValue::from_str("No PDFs provided"));
        }
        let first_bytes = pdfs[0].to_vec();
        let first = crate::document::PdfDocument::from_bytes(first_bytes)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let mut editor = crate::editor::DocumentEditor::from_document(first)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        for pdf_data in &pdfs[1..] {
            editor
                .merge_from_bytes(&pdf_data.to_vec())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        let bytes = editor
            .save_to_bytes()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(WasmPdf { bytes })
    }

    /// Create a PDF from Markdown content.
    ///
    /// @param content - Markdown string
    /// @param title - Optional document title
    /// @param author - Optional document author
    #[wasm_bindgen(js_name = "fromMarkdown")]
    pub fn from_markdown(
        content: &str,
        title: Option<String>,
        author: Option<String>,
    ) -> Result<WasmPdf, JsValue> {
        let mut builder = PdfBuilder::new();
        if let Some(t) = title {
            builder = builder.title(t);
        }
        if let Some(a) = author {
            builder = builder.author(a);
        }
        let pdf = builder
            .from_markdown(content)
            .map_err(|e| JsValue::from_str(&format!("Failed to create PDF: {}", e)))?;
        Ok(WasmPdf {
            bytes: pdf.into_bytes(),
        })
    }

    /// Create a PDF from HTML content.
    ///
    /// @param content - HTML string
    /// @param title - Optional document title
    /// @param author - Optional document author
    #[wasm_bindgen(js_name = "fromHtml")]
    pub fn from_html(
        content: &str,
        title: Option<String>,
        author: Option<String>,
    ) -> Result<WasmPdf, JsValue> {
        let mut builder = PdfBuilder::new();
        if let Some(t) = title {
            builder = builder.title(t);
        }
        if let Some(a) = author {
            builder = builder.author(a);
        }
        let pdf = builder
            .from_html(content)
            .map_err(|e| JsValue::from_str(&format!("Failed to create PDF: {}", e)))?;
        Ok(WasmPdf {
            bytes: pdf.into_bytes(),
        })
    }

    /// Create a PDF from plain text.
    ///
    /// @param content - Plain text string
    /// @param title - Optional document title
    /// @param author - Optional document author
    #[wasm_bindgen(js_name = "fromText")]
    pub fn from_text(
        content: &str,
        title: Option<String>,
        author: Option<String>,
    ) -> Result<WasmPdf, JsValue> {
        let mut builder = PdfBuilder::new();
        if let Some(t) = title {
            builder = builder.title(t);
        }
        if let Some(a) = author {
            builder = builder.author(a);
        }
        let pdf = builder
            .from_text(content)
            .map_err(|e| JsValue::from_str(&format!("Failed to create PDF: {}", e)))?;
        Ok(WasmPdf {
            bytes: pdf.into_bytes(),
        })
    }

    /// Create a PDF from image bytes (PNG, JPEG, etc.).
    ///
    /// @param data - Image file contents as a Uint8Array
    #[wasm_bindgen(js_name = "fromImageBytes")]
    pub fn from_image_bytes(data: &[u8]) -> Result<WasmPdf, JsValue> {
        use crate::api::Pdf;
        let pdf = Pdf::from_image_bytes(data)
            .map_err(|e| JsValue::from_str(&format!("Failed to create PDF from image: {}", e)))?;
        Ok(WasmPdf {
            bytes: pdf.into_bytes(),
        })
    }

    /// Create a PDF from multiple image byte arrays.
    ///
    /// Each image becomes a separate page. Pass an array of Uint8Arrays.
    ///
    /// @param images_array - Array of Uint8Arrays, each containing image file bytes (PNG/JPEG)
    #[wasm_bindgen(js_name = "fromMultipleImageBytes")]
    pub fn from_multiple_image_bytes(images_array: JsValue) -> Result<WasmPdf, JsValue> {
        use crate::writer::ImageData;

        let arr = js_sys::Array::from(&images_array);
        if arr.length() == 0 {
            return Err(JsValue::from_str("Empty image array"));
        }

        let mut images = Vec::new();
        for i in 0..arr.length() {
            let item = arr.get(i);
            let uint8 = js_sys::Uint8Array::new(&item);
            let bytes = uint8.to_vec();
            let image = ImageData::from_bytes(&bytes)
                .map_err(|e| JsValue::from_str(&format!("Failed to load image {}: {}", i, e)))?;
            images.push(image);
        }

        let pdf = PdfBuilder::new()
            .from_image_data_multiple(images)
            .map_err(|e| JsValue::from_str(&format!("Failed to create PDF from images: {}", e)))?;

        Ok(WasmPdf {
            bytes: pdf.into_bytes(),
        })
    }

    /// Get the PDF as a Uint8Array.
    #[wasm_bindgen(js_name = "toBytes")]
    pub fn to_bytes(&self) -> Vec<u8> {
        self.bytes.clone()
    }

    /// Get the size of the PDF in bytes.
    #[wasm_bindgen(getter)]
    pub fn size(&self) -> usize {
        self.bytes.len()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert an editor FormFieldValue to a JsValue.
fn wasm_form_field_value_to_js(
    value: &crate::editor::form_fields::FormFieldValue,
) -> Result<JsValue, JsValue> {
    use crate::editor::form_fields::FormFieldValue;
    match value {
        FormFieldValue::Text(s) => Ok(JsValue::from_str(s)),
        FormFieldValue::Choice(s) => Ok(JsValue::from_str(s)),
        FormFieldValue::Boolean(b) => Ok(JsValue::from(*b)),
        FormFieldValue::MultiChoice(v) => {
            let arr = js_sys::Array::new();
            for s in v {
                arr.push(&JsValue::from_str(s));
            }
            Ok(arr.into())
        },
        FormFieldValue::None => Ok(JsValue::NULL),
    }
}

/// Convert a JsValue to an editor FormFieldValue.
fn js_to_form_field_value(
    value: &JsValue,
) -> Result<crate::editor::form_fields::FormFieldValue, JsValue> {
    use crate::editor::form_fields::FormFieldValue;

    if value.is_null() || value.is_undefined() {
        Ok(FormFieldValue::None)
    } else if let Some(b) = value.as_bool() {
        Ok(FormFieldValue::Boolean(b))
    } else if let Some(s) = value.as_string() {
        Ok(FormFieldValue::Text(s))
    } else if js_sys::Array::is_array(value) {
        let arr = js_sys::Array::from(value);
        let mut strings = Vec::new();
        for i in 0..arr.length() {
            let item = arr.get(i);
            strings.push(
                item.as_string()
                    .ok_or_else(|| JsValue::from_str("Array elements must be strings"))?,
            );
        }
        Ok(FormFieldValue::MultiChoice(strings))
    } else {
        Err(JsValue::from_str(
            "Value must be string, boolean, array of strings, null, or undefined",
        ))
    }
}

/// Convert OutlineItem tree to JSON for WASM serialization.
fn outline_to_json(items: &[crate::outline::OutlineItem]) -> Vec<serde_json::Value> {
    items
        .iter()
        .map(|item| {
            let mut obj = serde_json::Map::new();
            obj.insert("title".into(), serde_json::Value::from(item.title.as_str()));

            match &item.dest {
                Some(crate::outline::Destination::PageIndex(idx)) => {
                    obj.insert("page".into(), serde_json::Value::from(*idx));
                },
                Some(crate::outline::Destination::Named(name)) => {
                    obj.insert("page".into(), serde_json::Value::Null);
                    obj.insert("dest_name".into(), serde_json::Value::from(name.as_str()));
                },
                None => {
                    obj.insert("page".into(), serde_json::Value::Null);
                },
            }

            let children = outline_to_json(&item.children);
            obj.insert("children".into(), serde_json::Value::from(children));

            serde_json::Value::Object(obj)
        })
        .collect()
}

// ============================================================================
// Unit Tests
// ============================================================================
//
// JsValue is not functional on non-wasm32 targets (wasm-bindgen stubs abort).
// Tests are split into two groups:
//   1. Native-safe: methods returning Rust types on the happy path (no JsValue at runtime)
//   2. Wasm-only: methods that return JsValue or whose error paths create JsValue
//
// Run native tests:  cargo test --lib --features wasm -- wasm::tests
// Run wasm tests:    wasm-pack test --headless --node --features wasm

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Test Helpers
    // ========================================================================

    fn make_text_pdf(text: &str) -> Vec<u8> {
        crate::api::Pdf::from_text(text).unwrap().into_bytes()
    }

    fn doc_from_text(text: &str) -> WasmPdfDocument {
        WasmPdfDocument::new(&make_text_pdf(text), None).unwrap()
    }

    fn make_markdown_pdf(md: &str) -> Vec<u8> {
        crate::api::PdfBuilder::new()
            .from_markdown(md)
            .unwrap()
            .into_bytes()
    }

    // ========================================================================
    // Group: Constructor
    // ========================================================================

    #[test]
    fn test_new_valid_pdf() {
        let bytes = make_text_pdf("Hello world");
        let result = WasmPdfDocument::new(&bytes, None);
        assert!(result.is_ok());
    }

    // Error-path tests require JsValue::from_str() which only works on wasm32
    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_new_invalid_bytes() {
        let result = WasmPdfDocument::new(b"not a pdf at all", None);
        assert!(result.is_err());
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_new_empty_bytes() {
        let result = WasmPdfDocument::new(b"", None);
        assert!(result.is_err());
    }

    // ========================================================================
    // Group: Core Read-Only
    // ========================================================================

    #[test]
    fn test_page_count() {
        let mut doc = doc_from_text("Hello");
        let count = doc.page_count().unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_version() {
        let doc = doc_from_text("Hello");
        let ver = doc.version().unwrap();
        assert_eq!(ver.len(), 2);
        assert!(ver[0] >= 1, "major version should be at least 1");
    }

    #[test]
    fn test_authenticate_unencrypted() {
        let mut doc = doc_from_text("Hello");
        let result = doc.authenticate("password");
        assert!(result.is_ok());
    }

    #[test]
    fn test_has_structure_tree_false() {
        let mut doc = doc_from_text("Hello");
        assert!(!doc.has_structure_tree().unwrap_or(false));
    }

    #[test]
    fn test_page_count_from_markdown() {
        let bytes = make_markdown_pdf("# Title\n\nSome content");
        let mut doc = WasmPdfDocument::new(&bytes, None).unwrap();
        assert!(doc.page_count().unwrap() >= 1);
    }

    // ========================================================================
    // Group: Text Extraction
    // ========================================================================

    #[test]
    fn test_extract_text() {
        let mut doc = doc_from_text("Hello world");
        let text = doc.extract_text(0, JsValue::UNDEFINED).unwrap();
        assert!(
            text.contains("Hello") || text.contains("world"),
            "extracted text should contain source content, got: {}",
            text
        );
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_extract_text_invalid_page() {
        let mut doc = doc_from_text("Hello");
        let result = doc.extract_text(999, JsValue::UNDEFINED);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_all_text() {
        let mut doc = doc_from_text("Hello world");
        let text = doc.extract_all_text().unwrap();
        assert!(!text.is_empty(), "extract_all_text should return non-empty");
    }

    #[test]
    fn test_extract_text_preserves_content() {
        let mut doc = doc_from_text("Test content 12345");
        let text = doc.extract_text(0, JsValue::UNDEFINED).unwrap();
        assert!(text.contains("12345"), "should preserve numeric content, got: {}", text);
    }

    // ========================================================================
    // Group: Format Conversion
    // ========================================================================

    #[test]
    fn test_to_markdown() {
        let mut doc = doc_from_text("Hello markdown");
        let md = doc.to_markdown(0, None, None, None).unwrap();
        assert!(!md.is_empty());
    }

    #[test]
    fn test_to_markdown_all() {
        let mut doc = doc_from_text("Hello markdown");
        let md = doc.to_markdown_all(None, None, None).unwrap();
        assert!(!md.is_empty());
    }

    #[test]
    fn test_to_html() {
        let mut doc = doc_from_text("Hello html");
        let html = doc.to_html(0, None, None, None).unwrap();
        assert!(!html.is_empty());
    }

    #[test]
    fn test_to_html_all() {
        let mut doc = doc_from_text("Hello html");
        let html = doc.to_html_all(None, None, None).unwrap();
        assert!(!html.is_empty());
    }

    #[test]
    fn test_to_plain_text() {
        let mut doc = doc_from_text("Hello plain");
        let text = doc.to_plain_text(0).unwrap();
        assert!(!text.is_empty());
    }

    #[test]
    fn test_to_plain_text_all() {
        let mut doc = doc_from_text("Hello plain");
        let text = doc.to_plain_text_all().unwrap();
        assert!(!text.is_empty());
    }

    #[test]
    fn test_to_markdown_with_options() {
        let mut doc = doc_from_text("Hello options");
        let md = doc.to_markdown(0, Some(false), Some(false), None).unwrap();
        assert!(!md.is_empty());
    }

    #[test]
    fn test_to_html_with_options() {
        let mut doc = doc_from_text("Hello options");
        let html = doc.to_html(0, Some(true), Some(false), None).unwrap();
        assert!(!html.is_empty());
    }

    // ========================================================================
    // Group: Structured Extraction (serde_wasm_bindgen — wasm32 only)
    // ========================================================================

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_extract_chars_ok() {
        let mut doc = doc_from_text("ABC");
        let result = doc.extract_chars(0);
        assert!(result.is_ok());
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_extract_spans_ok() {
        let mut doc = doc_from_text("Hello spans");
        let result = doc.extract_spans(0);
        assert!(result.is_ok());
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_extract_chars_invalid_page() {
        let mut doc = doc_from_text("ABC");
        let result = doc.extract_chars(999);
        assert!(result.is_err());
    }

    // ========================================================================
    // Group: Search (serde_wasm_bindgen — wasm32 only)
    // ========================================================================

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_search_found() {
        let mut doc = doc_from_text("Hello world test search");
        let result = doc.search("Hello", None, Some(true), None, None);
        assert!(result.is_ok());
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_search_not_found() {
        let mut doc = doc_from_text("Hello world");
        let result = doc.search("ZZZZZ_NONEXISTENT", None, Some(true), None, None);
        assert!(result.is_ok());
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_search_page_found() {
        let mut doc = doc_from_text("Hello searchable content");
        let result = doc.search_page(0, "Hello", None, Some(true), None, None);
        assert!(result.is_ok());
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_search_page_invalid() {
        let mut doc = doc_from_text("Hello");
        let result = doc.search_page(999, "Hello", None, Some(true), None, None);
        let _ = result;
    }

    // ========================================================================
    // Group: Image Info (serde_wasm_bindgen — wasm32 only)
    // ========================================================================

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_extract_images_ok() {
        let mut doc = doc_from_text("No images here");
        let result = doc.extract_images(0);
        assert!(result.is_ok());
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_extract_images_invalid_page() {
        let mut doc = doc_from_text("Hello");
        let result = doc.extract_images(999);
        assert!(result.is_err());
    }

    // ========================================================================
    // Group: Document Structure (serde_wasm_bindgen — wasm32 only)
    // ========================================================================

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_get_outline_ok() {
        let mut doc = doc_from_text("No outline here");
        let result = doc.get_outline();
        assert!(result.is_ok());
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_get_annotations_ok() {
        let mut doc = doc_from_text("No annotations here");
        let result = doc.get_annotations(0);
        assert!(result.is_ok());
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_get_annotations_invalid_page() {
        let mut doc = doc_from_text("Hello");
        let result = doc.get_annotations(999);
        assert!(result.is_err());
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_extract_paths_ok() {
        let mut doc = doc_from_text("No paths here");
        let result = doc.extract_paths(0);
        assert!(result.is_ok());
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_extract_paths_invalid_page() {
        let mut doc = doc_from_text("Hello");
        let result = doc.extract_paths(999);
        assert!(result.is_err());
    }

    // ========================================================================
    // Group: Metadata Editing
    // ========================================================================

    #[test]
    fn test_set_title() {
        let mut doc = doc_from_text("Hello");
        assert!(doc.set_title("My Title").is_ok());
    }

    #[test]
    fn test_set_author() {
        let mut doc = doc_from_text("Hello");
        assert!(doc.set_author("Author Name").is_ok());
    }

    #[test]
    fn test_set_subject() {
        let mut doc = doc_from_text("Hello");
        assert!(doc.set_subject("Subject Line").is_ok());
    }

    #[test]
    fn test_set_keywords() {
        let mut doc = doc_from_text("Hello");
        assert!(doc.set_keywords("pdf, test, rust").is_ok());
    }

    // ========================================================================
    // Group: Page Properties
    // ========================================================================

    #[test]
    fn test_page_rotation() {
        let mut doc = doc_from_text("Hello");
        let rotation = doc.page_rotation(0).unwrap();
        assert_eq!(rotation, 0);
    }

    #[test]
    fn test_set_page_rotation() {
        let mut doc = doc_from_text("Hello");
        assert!(doc.set_page_rotation(0, 90).is_ok());
        let rotation = doc.page_rotation(0).unwrap();
        assert_eq!(rotation, 90);
    }

    #[test]
    fn test_rotate_page() {
        let mut doc = doc_from_text("Hello");
        assert!(doc.rotate_page(0, 90).is_ok());
    }

    #[test]
    fn test_rotate_all_pages() {
        let mut doc = doc_from_text("Hello");
        assert!(doc.rotate_all_pages(180).is_ok());
    }

    #[test]
    fn test_page_media_box() {
        let mut doc = doc_from_text("Hello");
        let mbox = doc.page_media_box(0).unwrap();
        assert_eq!(mbox.len(), 4, "media box should have 4 coordinates");
        assert!(mbox[2] > mbox[0], "urx should be greater than llx");
        assert!(mbox[3] > mbox[1], "ury should be greater than lly");
    }

    #[test]
    fn test_set_page_media_box() {
        let mut doc = doc_from_text("Hello");
        assert!(doc.set_page_media_box(0, 0.0, 0.0, 612.0, 792.0).is_ok());
    }

    // page_crop_box returns JsValue via serde_wasm_bindgen — wasm32 only
    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_page_crop_box_unset() {
        let mut doc = doc_from_text("Hello");
        let result = doc.page_crop_box(0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_set_page_crop_box() {
        let mut doc = doc_from_text("Hello");
        assert!(doc.set_page_crop_box(0, 10.0, 10.0, 600.0, 780.0).is_ok());
    }

    #[test]
    fn test_crop_margins() {
        let mut doc = doc_from_text("Hello");
        assert!(doc.crop_margins(10.0, 10.0, 10.0, 10.0).is_ok());
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_page_rotation_invalid_page() {
        let mut doc = doc_from_text("Hello");
        let result = doc.page_rotation(999);
        assert!(result.is_err());
    }

    // ========================================================================
    // Group: Erase / Whiteout
    // ========================================================================

    #[test]
    fn test_erase_region() {
        let mut doc = doc_from_text("Hello");
        assert!(doc.erase_region(0, 0.0, 0.0, 100.0, 100.0).is_ok());
    }

    #[test]
    fn test_erase_regions_valid() {
        let mut doc = doc_from_text("Hello");
        let rects = [0.0, 0.0, 100.0, 100.0, 200.0, 200.0, 300.0, 300.0];
        assert!(doc.erase_regions(0, &rects).is_ok());
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_erase_regions_invalid_length() {
        let mut doc = doc_from_text("Hello");
        let rects = [0.0, 0.0, 100.0]; // Not a multiple of 4
        let result = doc.erase_regions(0, &rects);
        assert!(result.is_err());
    }

    #[test]
    fn test_clear_erase_regions() {
        let mut doc = doc_from_text("Hello");
        doc.erase_region(0, 0.0, 0.0, 100.0, 100.0).unwrap();
        assert!(doc.clear_erase_regions(0).is_ok());
    }

    // ========================================================================
    // Group: Annotations
    // ========================================================================

    #[test]
    fn test_flatten_page_annotations() {
        let mut doc = doc_from_text("Hello");
        assert!(doc.flatten_page_annotations(0).is_ok());
    }

    #[test]
    fn test_flatten_all_annotations() {
        let mut doc = doc_from_text("Hello");
        assert!(doc.flatten_all_annotations().is_ok());
    }

    // ========================================================================
    // Group: Redaction
    // ========================================================================

    #[test]
    fn test_apply_page_redactions() {
        let mut doc = doc_from_text("Hello");
        assert!(doc.apply_page_redactions(0).is_ok());
    }

    #[test]
    fn test_apply_all_redactions() {
        let mut doc = doc_from_text("Hello");
        assert!(doc.apply_all_redactions().is_ok());
    }

    // ========================================================================
    // Group: Form Fields
    // ========================================================================

    fn make_form_pdf() -> Vec<u8> {
        use crate::geometry::Rect;
        use crate::writer::{CheckboxWidget, ComboBoxWidget, PdfWriter, TextFieldWidget};

        let mut writer = PdfWriter::new();
        {
            let mut page = writer.add_page(612.0, 792.0);
            page.add_text_field(
                TextFieldWidget::new("name", Rect::new(72.0, 700.0, 200.0, 20.0))
                    .with_value("Alice"),
            );
            page.add_checkbox(
                CheckboxWidget::new("agree", Rect::new(72.0, 650.0, 15.0, 15.0)).checked(),
            );
            page.add_combo_box(
                ComboBoxWidget::new("color", Rect::new(72.0, 600.0, 150.0, 20.0))
                    .with_options(vec!["Red", "Blue", "Green"])
                    .with_value("Blue"),
            );
        }
        writer.finish().expect("Failed to create form PDF")
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_get_form_fields_returns_array() {
        let bytes = make_form_pdf();
        let mut doc = WasmPdfDocument::new(&bytes, None).unwrap();
        let result = doc.get_form_fields().unwrap();
        assert!(js_sys::Array::is_array(&result));
        let arr = js_sys::Array::from(&result);
        assert!(arr.length() >= 3, "Should have at least 3 fields, got {}", arr.length());
    }

    #[test]
    fn test_has_xfa_on_plain_pdf() {
        let mut doc = doc_from_text("No XFA");
        assert!(!doc.has_xfa().unwrap(), "Plain text PDF should not have XFA");
    }

    #[test]
    fn test_has_xfa_on_form_pdf() {
        let bytes = make_form_pdf();
        let mut doc = WasmPdfDocument::new(&bytes, None).unwrap();
        assert!(!doc.has_xfa().unwrap(), "PdfWriter form should not have XFA");
    }

    // ========================================================================
    // Group: Image Manipulation (serde_wasm_bindgen — wasm32 only)
    // ========================================================================

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_page_images() {
        let mut doc = doc_from_text("Hello");
        let result = doc.page_images(0);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Group: Save
    // ========================================================================

    #[test]
    fn test_save_to_bytes() {
        let mut doc = doc_from_text("Hello save");
        let bytes = doc.save_to_bytes().unwrap();
        assert!(!bytes.is_empty(), "saved bytes should not be empty");
    }

    #[test]
    fn test_save_to_bytes_pdf_header() {
        let mut doc = doc_from_text("Hello header");
        let bytes = doc.save_to_bytes().unwrap();
        assert!(bytes.starts_with(b"%PDF"), "saved bytes should start with PDF header");
    }

    #[test]
    fn test_save_encrypted_to_bytes() {
        let mut doc = doc_from_text("Hello encrypted");
        let bytes = doc
            .save_encrypted_to_bytes("pass", None, None, None, None, None)
            .unwrap();
        assert!(!bytes.is_empty());
        assert!(bytes.starts_with(b"%PDF"));
    }

    #[test]
    fn test_save_roundtrip() {
        let mut doc = doc_from_text("Roundtrip test");
        doc.set_title("Roundtrip Title").unwrap();
        let bytes = doc.save_to_bytes().unwrap();

        let mut doc2 = WasmPdfDocument::new(&bytes, None).unwrap();
        let text = doc2.extract_text(0, JsValue::UNDEFINED).unwrap();
        assert!(text.contains("Roundtrip"), "roundtrip should preserve text, got: {}", text);
    }

    // ========================================================================
    // Group: WasmPdf Creation
    // ========================================================================

    #[test]
    fn test_wasm_pdf_from_markdown() {
        let result = WasmPdf::from_markdown("# Hello\n\nWorld", None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_wasm_pdf_from_html() {
        let result = WasmPdf::from_html("<h1>Hello</h1><p>World</p>", None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_wasm_pdf_from_text() {
        let result = WasmPdf::from_text("Hello world", None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_wasm_pdf_to_bytes() {
        let pdf = WasmPdf::from_text("Hello bytes", None, None).unwrap();
        let bytes = pdf.to_bytes();
        assert!(!bytes.is_empty());
        assert!(bytes.starts_with(b"%PDF"));
    }

    #[test]
    fn test_wasm_pdf_size() {
        let pdf = WasmPdf::from_text("Hello size", None, None).unwrap();
        assert!(pdf.size() > 0, "PDF size should be positive");
    }

    #[test]
    fn test_wasm_pdf_with_metadata() {
        let pdf = WasmPdf::from_markdown(
            "# Test",
            Some("Test Title".to_string()),
            Some("Test Author".to_string()),
        )
        .unwrap();
        assert!(pdf.size() > 0);
        let mut doc = WasmPdfDocument::new(&pdf.to_bytes(), None).unwrap();
        assert_eq!(doc.page_count().unwrap(), 1);
    }

    // ========================================================================
    // Group: Lazy Editor Init
    // ========================================================================

    #[test]
    fn test_editor_lazy_init() {
        let doc = doc_from_text("Hello");
        assert!(doc.editor.is_none());
    }

    #[test]
    fn test_editor_initialized_on_edit() {
        let mut doc = doc_from_text("Hello");
        assert!(doc.editor.is_none());
        doc.set_title("Title").unwrap();
        assert!(doc.editor.is_some());
    }

    // ========================================================================
    // Group: Form Field Get/Set Values
    // ========================================================================

    #[test]
    fn test_get_form_field_value_text() {
        let bytes = make_form_pdf();
        let mut doc = WasmPdfDocument::new(&bytes, None).unwrap();
        // get_form_field_value returns JsValue which aborts on non-wasm32,
        // so test the underlying Rust API directly here.
        let editor_mutex = doc.ensure_editor().unwrap();
        let mut editor = editor_mutex.lock().unwrap();
        let value = editor.get_form_field_value("name");
        assert!(value.is_ok(), "field 'name' should have a value");
    }

    #[test]
    fn test_set_form_field_value_text() {
        let bytes = make_form_pdf();
        let mut doc = WasmPdfDocument::new(&bytes, None).unwrap();
        // set_form_field_value with a string JsValue
        // On native, JsValue operations are stubbed, so we test via the Rust API
        // instead — just verify the method exists and the type signatures match
        let editor_mutex = doc.ensure_editor().unwrap();
        let mut editor = editor_mutex.lock().unwrap();
        let result = editor.set_form_field_value(
            "name",
            crate::editor::form_fields::FormFieldValue::Text("Bob".to_string()),
        );
        assert!(result.is_ok(), "set_form_field_value should succeed");
    }

    // ========================================================================
    // Group: Image Bytes Extraction (native-safe: no JsValue in happy path)
    // ========================================================================

    #[test]
    fn test_extract_image_bytes_empty_on_text_pdf() {
        // Text-only PDF has no images — should not error
        let doc = doc_from_text("No images here");
        // extract_image_bytes returns JsValue, tested in wasm_bindgen_tests
        // For native, test that the underlying API works
        let images = doc.inner.lock().unwrap().extract_images(0).unwrap();
        assert_eq!(images.len(), 0);
    }

    // ========================================================================
    // Group: Form Flattening
    // ========================================================================

    #[test]
    fn test_flatten_forms() {
        let bytes = make_form_pdf();
        let mut doc = WasmPdfDocument::new(&bytes, None).unwrap();
        let editor_mutex = doc.ensure_editor().unwrap();
        let mut editor = editor_mutex.lock().unwrap();
        let result = editor.flatten_forms();
        assert!(result.is_ok(), "flatten_forms should succeed");
    }

    #[test]
    fn test_flatten_forms_on_page() {
        let bytes = make_form_pdf();
        let mut doc = WasmPdfDocument::new(&bytes, None).unwrap();
        let editor_mutex = doc.ensure_editor().unwrap();
        let mut editor = editor_mutex.lock().unwrap();
        let result = editor.flatten_forms_on_page(0);
        assert!(result.is_ok(), "flatten_forms_on_page should succeed");
    }

    // ========================================================================
    // Group: PDF Merging
    // ========================================================================

    #[test]
    fn test_merge_from_bytes() {
        let bytes1 = make_text_pdf("Page 1");
        let bytes2 = make_text_pdf("Page 2");
        let mut doc = WasmPdfDocument::new(&bytes1, None).unwrap();
        let editor_mutex = doc.ensure_editor().unwrap();
        let mut editor = editor_mutex.lock().unwrap();
        let count = editor.merge_from_bytes(&bytes2).unwrap();
        assert_eq!(count, 1, "should merge 1 page");
    }

    // ========================================================================
    // Group: File Embedding
    // ========================================================================

    #[test]
    fn test_embed_file() {
        let bytes = make_text_pdf("Hello");
        let mut doc = WasmPdfDocument::new(&bytes, None).unwrap();
        let editor_mutex = doc.ensure_editor().unwrap();
        let mut editor = editor_mutex.lock().unwrap();
        let result = editor.embed_file("readme.txt", b"Hello World".to_vec());
        assert!(result.is_ok(), "embed_file should succeed");
    }

    // ========================================================================
    // Group: Page Labels
    // ========================================================================

    #[test]
    fn test_page_labels_empty() {
        let doc = doc_from_text("Hello");
        let labels = crate::extractors::page_labels::PageLabelExtractor::extract(
            &mut doc.inner.lock().unwrap(),
        );
        // Simple generated PDFs typically have no page labels
        assert!(labels.is_ok());
    }

    // ========================================================================
    // Group: XMP Metadata
    // ========================================================================

    #[test]
    fn test_xmp_metadata_none_for_simple_pdf() {
        let doc = doc_from_text("Hello");
        let metadata =
            crate::extractors::xmp::XmpExtractor::extract(&mut doc.inner.lock().unwrap());
        assert!(metadata.is_ok());
        // Simple generated PDFs may or may not have XMP
    }

    // ========================================================================
    // Group: PDF from Images
    // ========================================================================

    #[test]
    fn test_from_image_bytes() {
        // WasmPdf::from_image_bytes uses JsValue in error path, so test the
        // underlying Rust API directly on non-wasm32 targets.
        use crate::api::Pdf;
        let jpeg_data = create_minimal_jpeg();
        let result = Pdf::from_image_bytes(&jpeg_data);
        assert!(result.is_ok(), "Pdf::from_image_bytes should succeed: {:?}", result.err());
        let pdf = result.unwrap();
        assert!(!pdf.into_bytes().is_empty());
    }

    /// Create a minimal valid 1x1 white JPEG image (known-good bytes).
    fn create_minimal_jpeg() -> Vec<u8> {
        vec![
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00,
            0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43, 0x00, 0x08, 0x06, 0x06,
            0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09, 0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D,
            0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12, 0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D,
            0x1A, 0x1C, 0x1C, 0x20, 0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28,
            0x37, 0x29, 0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
            0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01,
            0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01,
            0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02,
            0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10,
            0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00,
            0x01, 0x7D, 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
            0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42,
            0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16,
            0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37,
            0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55,
            0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73,
            0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
            0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
            0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA,
            0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6,
            0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA,
            0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08,
            0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD5, 0xDB, 0x20, 0xA8, 0xF9, 0xFF, 0xD9,
        ]
    }

    // ========================================================================
    // Tests for new binding methods (v0.3.18)
    // ========================================================================

    #[test]
    fn test_validate_pdf_a() {
        let mut doc = doc_from_text("Hello World");
        let result = doc.validate_pdf_a("1b");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_pdf_a_invalid_level() {
        let mut doc = doc_from_text("Hello");
        let result = doc.validate_pdf_a("invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_delete_page() {
        // Create a 2-page PDF
        let bytes = make_markdown_pdf("# Page 1\n\n---\n\n# Page 2");
        let mut doc = WasmPdfDocument::new(&bytes, None).unwrap();
        let initial_count = doc.page_count().unwrap();
        if initial_count >= 2 {
            assert!(doc.delete_page(0).is_ok());
        }
    }

    #[test]
    fn test_extract_pages() {
        let mut doc = doc_from_text("Extract me");
        let result = doc.extract_pages(vec![0]);
        assert!(result.is_ok());
        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
        // Verify the extracted PDF is valid
        let extracted = WasmPdfDocument::new(&bytes, None);
        assert!(extracted.is_ok());
    }
}
