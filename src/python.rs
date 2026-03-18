//! Python bindings via PyO3.
//!
//! This module provides Python bindings for the PDF library, exposing the core functionality
//! through a Python-friendly API with proper error handling and type hints.

use std::path::PathBuf;

use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyBytes;
#[cfg(feature = "python")]
#[cfg(any(not(feature = "office"), not(feature = "ocr")))]
use pyo3::types::{PyDict, PyTuple};

// Register module-level variable for .pyi (pyo3-stub-gen); matches m.add("VERSION", ...) below.
use crate::api::PdfBuilder as RustPdfBuilder;
#[cfg(feature = "python")]
use crate::converters::ConversionOptions as RustConversionOptions;
use crate::document::PdfDocument as RustPdfDocument;
use crate::extractors::forms::{
    field_flags, FieldType as RustFieldType, FieldValue as RustFieldValue,
    FormField as RustFormField,
};
use crate::layout::{Color as RustColor, TextChar as RustTextChar};
use crate::writer::{BlendMode as RustBlendMode, LineCap as RustLineCap, LineJoin as RustLineJoin};

/// Python wrapper for PdfDocument.
///
/// Provides PDF parsing, text extraction, and format conversion capabilities.
#[pyclass(module = "pdf_oxide.pdf_oxide", name = "PdfDocument", unsendable)]
pub struct PyPdfDocument {
    pub(crate) inner: RustPdfDocument,
    pub(crate) path: Option<String>,
    pub(crate) raw_bytes: Option<Vec<u8>>,
    pub(crate) editor: Option<crate::editor::DocumentEditor>,
}

impl PyPdfDocument {
    /// Ensure the editor is initialized for DOM access.
    fn ensure_editor(&mut self) -> PyResult<()> {
        if self.editor.is_none() {
            let editor = if let Some(ref path) = self.path {
                crate::editor::DocumentEditor::open(path)
            } else if let Some(ref bytes) = self.raw_bytes {
                crate::editor::DocumentEditor::from_bytes(bytes.clone())
            } else {
                return Err(PyRuntimeError::new_err("No document source available"));
            };
            self.editor =
                Some(editor.map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to open editor: {}", e))
                })?);
        }
        Ok(())
    }
}

#[pymethods]
impl PyPdfDocument {
    /// Open a PDF file.
    ///
    /// Args:
    ///     path (str | pathlib.Path): Path to the PDF file
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        let doc = RustPdfDocument::open(&path)
            .map_err(|e| PyIOError::new_err(format!("Failed to open PDF: {}", e)))?;

        let path_str = path.to_string_lossy().into_owned();
        Ok(PyPdfDocument {
            inner: doc,
            path: Some(path_str),
            raw_bytes: None,
            editor: None,
        })
    }

    /// Open a PDF from bytes.
    #[staticmethod]
    fn from_bytes(data: &Bound<'_, PyBytes>) -> PyResult<Self> {
        let bytes = data.as_bytes().to_vec();
        let doc = RustPdfDocument::from_bytes(bytes.clone())
            .map_err(|e| PyIOError::new_err(format!("Failed to open PDF from bytes: {}", e)))?;

        Ok(PyPdfDocument {
            inner: doc,
            path: None,
            raw_bytes: Some(bytes),
            editor: None,
        })
    }

    /// Context manager support.
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Context manager support.
    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        Ok(false)
    }

    /// Get PDF version.
    fn version(&self) -> (u8, u8) {
        self.inner.version()
    }

    /// Authenticate with a password.
    fn authenticate(&mut self, password: &str) -> PyResult<bool> {
        self.inner
            .authenticate(password.as_bytes())
            .map_err(|e| PyRuntimeError::new_err(format!("Authentication failed: {}", e)))
    }

    /// Get number of pages.
    fn page_count(&mut self) -> PyResult<usize> {
        self.inner
            .page_count()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get page count: {}", e)))
    }

    /// Extract text from a page.
    #[pyo3(signature = (page, region=None))]
    fn extract_text(
        &mut self,
        page: usize,
        region: Option<(f32, f32, f32, f32)>,
    ) -> PyResult<String> {
        if let Some((x, y, w, h)) = region {
            self.inner
                .extract_text_in_rect(
                    page,
                    crate::geometry::Rect::new(x, y, w, h),
                    crate::layout::RectFilterMode::Intersects,
                )
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to extract text in region: {}", e))
                })
        } else {
            self.inner
                .extract_text(page)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to extract text: {}", e)))
        }
    }

    /// Identify and remove headers.
    #[pyo3(signature = (threshold=0.8))]
    fn remove_headers(&mut self, threshold: f32) -> PyResult<usize> {
        let count = self
            .inner
            .remove_headers(threshold)
            .map_err(|e| PyRuntimeError::new_err(format!("Header removal failed: {}", e)))?;

        self.sync_editor_erasures()?;
        Ok(count)
    }

    /// Identify and remove footers.
    #[pyo3(signature = (threshold=0.8))]
    fn remove_footers(&mut self, threshold: f32) -> PyResult<usize> {
        let count = self
            .inner
            .remove_footers(threshold)
            .map_err(|e| PyRuntimeError::new_err(format!("Footer removal failed: {}", e)))?;

        self.sync_editor_erasures()?;
        Ok(count)
    }

    /// Identify and remove both headers and footers.
    #[pyo3(signature = (threshold=0.8))]
    fn remove_artifacts(&mut self, threshold: f32) -> PyResult<usize> {
        let count = self
            .inner
            .remove_artifacts(threshold)
            .map_err(|e| PyRuntimeError::new_err(format!("Artifact removal failed: {}", e)))?;

        self.sync_editor_erasures()?;
        Ok(count)
    }

    /// Synchronize erasures to editor.
    fn sync_editor_erasures(&mut self) -> PyResult<()> {
        if let Some(ref mut editor) = self.editor {
            for (page, regions) in self.inner.erase_regions.iter() {
                editor.clear_erase_regions(*page);
                for rect in regions {
                    let _ = editor.erase_region(
                        *page,
                        [rect.x, rect.y, rect.x + rect.width, rect.y + rect.height],
                    );
                }
            }
        }
        Ok(())
    }

    /// Erase header area.
    fn erase_header(&mut self, page: usize) -> PyResult<()> {
        self.ensure_editor()?;
        self.inner
            .erase_header(page)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to erase header: {}", e)))?;
        self.sync_editor_erasures()?;
        Ok(())
    }

    /// Deprecated: Use erase_header instead.
    fn edit_header(&mut self, page: usize) -> PyResult<()> {
        self.erase_header(page)
    }

    /// Erase footer area.
    fn erase_footer(&mut self, page: usize) -> PyResult<()> {
        self.ensure_editor()?;
        self.inner
            .erase_footer(page)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to erase footer: {}", e)))?;
        self.sync_editor_erasures()?;
        Ok(())
    }

    /// Deprecated: Use erase_footer instead.
    fn edit_footer(&mut self, page: usize) -> PyResult<()> {
        self.erase_footer(page)
    }

    /// Erase both header and footer content.
    fn erase_artifacts(&mut self, page: usize) -> PyResult<()> {
        self.inner
            .erase_artifacts(page)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to erase artifacts: {}", e)))
    }

    /// Focus extraction on a region.
    fn within(slf: Py<Self>, page: usize, bbox: (f32, f32, f32, f32)) -> PyResult<PyPdfPageRegion> {
        Ok(PyPdfPageRegion {
            doc: slf,
            page_index: page,
            region: crate::geometry::Rect::new(bbox.0, bbox.1, bbox.2, bbox.3),
        })
    }

    /// Render a page to image bytes.
    #[pyo3(signature = (page, dpi=None, format=None))]
    fn render_page(
        &mut self,
        page: usize,
        dpi: Option<u32>,
        format: Option<&str>,
    ) -> PyResult<Vec<u8>> {
        #[cfg(feature = "rendering")]
        {
            let mut options = crate::rendering::RenderOptions::with_dpi(dpi.unwrap_or(72));
            if let Some(fmt) = format {
                match fmt.to_lowercase().as_str() {
                    "jpeg" | "jpg" => {
                        options = options.as_jpeg(85);
                    },
                    _ => {},
                }
            }

            crate::rendering::render_page(&mut self.inner, page, &options)
                .map(|img| img.data)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to render page: {}", e)))
        }
        #[cfg(not(feature = "rendering"))]
        {
            let _ = (page, dpi, format);
            Err(PyRuntimeError::new_err("Rendering feature not enabled."))
        }
    }

    /// Extract low-level characters.
    #[pyo3(signature = (page, region=None))]
    fn extract_chars(
        &mut self,
        page: usize,
        region: Option<(f32, f32, f32, f32)>,
    ) -> PyResult<Vec<PyTextChar>> {
        let chars_result = if let Some((x, y, w, h)) = region {
            self.inner.extract_chars_in_rect(
                page,
                crate::geometry::Rect::new(x, y, w, h),
                crate::layout::RectFilterMode::Intersects,
            )
        } else {
            self.inner.extract_chars(page)
        };

        chars_result
            .map(|chars| {
                chars
                    .into_iter()
                    .map(|ch| PyTextChar { inner: ch })
                    .collect()
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to extract characters: {}", e)))
    }

    /// Extract words.
    #[pyo3(signature = (page, region=None))]
    fn extract_words(
        &mut self,
        page: usize,
        region: Option<(f32, f32, f32, f32)>,
    ) -> PyResult<Vec<PyWord>> {
        let words_result = if let Some((x, y, w, h)) = region {
            self.inner.extract_words_in_rect(
                page,
                crate::geometry::Rect::new(x, y, w, h),
                crate::layout::RectFilterMode::Intersects,
            )
        } else {
            self.inner.extract_words(page)
        };

        words_result
            .map(|words| words.into_iter().map(|w| PyWord { inner: w }).collect())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to extract words: {}", e)))
    }

    /// Extract text lines.
    #[pyo3(signature = (page, region=None))]
    fn extract_text_lines(
        &mut self,
        page: usize,
        region: Option<(f32, f32, f32, f32)>,
    ) -> PyResult<Vec<PyTextLine>> {
        let lines_result = if let Some((x, y, w, h)) = region {
            self.inner.extract_text_lines_in_rect(
                page,
                crate::geometry::Rect::new(x, y, w, h),
                crate::layout::RectFilterMode::Intersects,
            )
        } else {
            self.inner.extract_text_lines(page)
        };

        lines_result
            .map(|lines| lines.into_iter().map(|l| PyTextLine { inner: l }).collect())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to extract lines: {}", e)))
    }

    /// Check if Tagged PDF.
    fn has_structure_tree(&mut self) -> bool {
        self.inner.structure_tree().ok().flatten().is_some()
    }

    /// Convert page to plain text.
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
    #[pyo3(signature = (page, preserve_layout=false, detect_headings=true, include_images=true, image_output_dir=None, embed_images=true, include_form_fields=true))]
    fn to_markdown(
        &mut self,
        page: usize,
        preserve_layout: bool,
        detect_headings: bool,
        include_images: bool,
        image_output_dir: Option<String>,
        embed_images: bool,
        include_form_fields: bool,
    ) -> PyResult<String> {
        let options = RustConversionOptions {
            preserve_layout,
            detect_headings,
            extract_tables: true,
            include_images,
            image_output_dir,
            embed_images,
            include_form_fields,
            ..Default::default()
        };

        self.inner
            .to_markdown(page, &options)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to convert to Markdown: {}", e)))
    }

    /// Convert page to HTML.
    #[pyo3(signature = (page, preserve_layout=false, detect_headings=true, include_images=true, image_output_dir=None, embed_images=true, include_form_fields=true))]
    fn to_html(
        &mut self,
        page: usize,
        preserve_layout: bool,
        detect_headings: bool,
        include_images: bool,
        image_output_dir: Option<String>,
        embed_images: bool,
        include_form_fields: bool,
    ) -> PyResult<String> {
        let options = RustConversionOptions {
            preserve_layout,
            detect_headings,
            extract_tables: true,
            include_images,
            image_output_dir,
            embed_images,
            include_form_fields,
            ..Default::default()
        };

        self.inner
            .to_html(page, &options)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to convert to HTML: {}", e)))
    }

    /// Convert all pages to Markdown.
    #[pyo3(signature = (preserve_layout=false, detect_headings=true, include_images=true, image_output_dir=None, embed_images=true, include_form_fields=true))]
    fn to_markdown_all(
        &mut self,
        preserve_layout: bool,
        detect_headings: bool,
        include_images: bool,
        image_output_dir: Option<String>,
        embed_images: bool,
        include_form_fields: bool,
    ) -> PyResult<String> {
        let options = RustConversionOptions {
            preserve_layout,
            detect_headings,
            extract_tables: true,
            include_images,
            image_output_dir,
            embed_images,
            include_form_fields,
            ..Default::default()
        };

        self.inner.to_markdown_all(&options).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to convert all pages to Markdown: {}", e))
        })
    }

    /// Convert all pages to HTML.
    #[pyo3(signature = (preserve_layout=false, detect_headings=true, include_images=true, image_output_dir=None, embed_images=true, include_form_fields=true))]
    fn to_html_all(
        &mut self,
        preserve_layout: bool,
        detect_headings: bool,
        include_images: bool,
        image_output_dir: Option<String>,
        embed_images: bool,
        include_form_fields: bool,
    ) -> PyResult<String> {
        let options = RustConversionOptions {
            preserve_layout,
            detect_headings,
            extract_tables: true,
            include_images,
            image_output_dir,
            embed_images,
            include_form_fields,
            ..Default::default()
        };

        self.inner.to_html_all(&options).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to convert all pages to HTML: {}", e))
        })
    }

    /// Get page object for DOM access.
    fn page(&mut self, index: usize) -> PyResult<PyPdfPage> {
        self.ensure_editor()?;
        let editor = self.editor.as_mut().unwrap();
        let page = editor
            .get_page(index)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get page: {}", e)))?;
        Ok(PyPdfPage { inner: page })
    }

    /// Save modification to page.
    fn save_page(&mut self, page: &PyPdfPage) -> PyResult<()> {
        self.ensure_editor()?;
        let editor = self.editor.as_mut().unwrap();
        editor
            .save_page(page.inner.clone())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to save page: {}", e)))
    }

    /// Save document to path.
    fn save(&mut self, path: &str) -> PyResult<()> {
        use crate::editor::EditableDocument;
        self.ensure_editor()?;
        if let Some(ref mut editor) = self.editor {
            editor
                .save(path)
                .map_err(|e| PyIOError::new_err(format!("Failed to save PDF: {}", e)))
        } else {
            Err(PyRuntimeError::new_err("No editor initialized."))
        }
    }

    /// Save encrypted PDF.
    #[pyo3(signature = (path, user_password, owner_password=None, allow_print=true, allow_copy=true, allow_modify=true, allow_annotate=true))]
    fn save_encrypted(
        &mut self,
        path: &str,
        user_password: &str,
        owner_password: Option<&str>,
        allow_print: bool,
        allow_copy: bool,
        allow_modify: bool,
        allow_annotate: bool,
    ) -> PyResult<()> {
        use crate::editor::{
            EditableDocument, EncryptionAlgorithm, EncryptionConfig, Permissions, SaveOptions,
        };
        self.ensure_editor()?;
        if let Some(ref mut editor) = self.editor {
            let owner_pwd = owner_password.unwrap_or(user_password);
            let permissions = Permissions {
                print: allow_print,
                print_high_quality: allow_print,
                modify: allow_modify,
                copy: allow_copy,
                annotate: allow_annotate,
                fill_forms: allow_annotate,
                accessibility: true,
                assemble: allow_modify,
            };
            let config = EncryptionConfig::new(user_password, owner_pwd)
                .with_algorithm(EncryptionAlgorithm::Aes256)
                .with_permissions(permissions);
            let options = SaveOptions::with_encryption(config);
            editor
                .save_with_options(path, options)
                .map_err(|e| PyIOError::new_err(format!("Failed to save encrypted PDF: {}", e)))
        } else {
            Err(PyRuntimeError::new_err("No editor initialized."))
        }
    }

    /// Set document metadata title.
    fn set_title(&mut self, title: &str) -> PyResult<()> {
        self.ensure_editor()?;
        if let Some(ref mut editor) = self.editor {
            editor.set_title(title);
        }
        Ok(())
    }

    /// Set document metadata author.
    fn set_author(&mut self, author: &str) -> PyResult<()> {
        self.ensure_editor()?;
        if let Some(ref mut editor) = self.editor {
            editor.set_author(author);
        }
        Ok(())
    }

    /// Set document metadata subject.
    fn set_subject(&mut self, subject: &str) -> PyResult<()> {
        self.ensure_editor()?;
        if let Some(ref mut editor) = self.editor {
            editor.set_subject(subject);
        }
        Ok(())
    }

    /// Set document metadata keywords.
    fn set_keywords(&mut self, keywords: &str) -> PyResult<()> {
        self.ensure_editor()?;
        if let Some(ref mut editor) = self.editor {
            editor.set_keywords(keywords);
        }
        Ok(())
    }

    /// Get page rotation.
    fn page_rotation(&mut self, page: usize) -> PyResult<i32> {
        self.ensure_editor()?;
        if let Some(ref mut editor) = self.editor {
            editor
                .get_page_rotation(page)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        } else {
            Err(PyRuntimeError::new_err("No editor initialized."))
        }
    }

    /// Set page rotation.
    fn set_page_rotation(&mut self, page: usize, degrees: i32) -> PyResult<()> {
        self.ensure_editor()?;
        if let Some(ref mut editor) = self.editor {
            editor
                .set_page_rotation(page, degrees)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        } else {
            Ok(())
        }
    }

    /// Rotate page by degrees.
    fn rotate_page(&mut self, page: usize, degrees: i32) -> PyResult<()> {
        self.ensure_editor()?;
        if let Some(ref mut editor) = self.editor {
            editor
                .rotate_page_by(page, degrees)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        } else {
            Ok(())
        }
    }

    /// Rotate all pages.
    fn rotate_all_pages(&mut self, degrees: i32) -> PyResult<()> {
        self.ensure_editor()?;
        if let Some(ref mut editor) = self.editor {
            editor
                .rotate_all_pages(degrees)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        } else {
            Ok(())
        }
    }

    /// Get page mediabox.
    fn page_media_box(&mut self, page: usize) -> PyResult<(f32, f32, f32, f32)> {
        self.ensure_editor()?;
        if let Some(ref mut editor) = self.editor {
            let b = editor
                .get_page_media_box(page)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok((b[0], b[1], b[2], b[3]))
        } else {
            Err(PyRuntimeError::new_err("No editor initialized."))
        }
    }

    /// Set page mediabox.
    fn set_page_media_box(
        &mut self,
        page: usize,
        llx: f32,
        lly: f32,
        urx: f32,
        ury: f32,
    ) -> PyResult<()> {
        self.ensure_editor()?;
        if let Some(ref mut editor) = self.editor {
            editor
                .set_page_media_box(page, [llx, lly, urx, ury])
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        } else {
            Ok(())
        }
    }

    /// Get page cropbox.
    fn page_crop_box(&mut self, page: usize) -> PyResult<Option<(f32, f32, f32, f32)>> {
        self.ensure_editor()?;
        if let Some(ref mut editor) = self.editor {
            let b = editor
                .get_page_crop_box(page)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(b.map(|v| (v[0], v[1], v[2], v[3])))
        } else {
            Ok(None)
        }
    }

    /// Set page cropbox.
    fn set_page_crop_box(
        &mut self,
        page: usize,
        llx: f32,
        lly: f32,
        urx: f32,
        ury: f32,
    ) -> PyResult<()> {
        self.ensure_editor()?;
        if let Some(ref mut editor) = self.editor {
            editor
                .set_page_crop_box(page, [llx, lly, urx, ury])
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        } else {
            Ok(())
        }
    }

    /// Crop all pages margins.
    fn crop_margins(&mut self, left: f32, right: f32, top: f32, bottom: f32) -> PyResult<()> {
        self.ensure_editor()?;
        if let Some(ref mut editor) = self.editor {
            editor
                .crop_margins(left, right, top, bottom)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        } else {
            Ok(())
        }
    }

    /// Erase rectangular region.
    fn erase_region(
        &mut self,
        page: usize,
        llx: f32,
        lly: f32,
        urx: f32,
        ury: f32,
    ) -> PyResult<()> {
        let rect = crate::geometry::Rect::new(llx, lly, urx - llx, ury - lly);
        self.inner
            .erase_region(page, rect)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        self.ensure_editor()?;
        if let Some(ref mut editor) = self.editor {
            editor
                .erase_region(page, [llx, lly, urx, ury])
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }
        Ok(())
    }

    /// Erase multiple regions.
    fn erase_regions(&mut self, page: usize, rects: Vec<(f32, f32, f32, f32)>) -> PyResult<()> {
        for (llx, lly, urx, ury) in &rects {
            let rect = crate::geometry::Rect::new(*llx, *lly, *urx - *llx, *ury - *lly);
            self.inner
                .erase_region(page, rect)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }
        self.ensure_editor()?;
        if let Some(ref mut editor) = self.editor {
            let arrays: Vec<[f32; 4]> = rects.iter().map(|r| [r.0, r.1, r.2, r.3]).collect();
            editor
                .erase_regions(page, &arrays)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }
        Ok(())
    }

    /// Clear erase regions.
    fn clear_erase_regions(&mut self, page: usize) -> PyResult<()> {
        self.inner
            .clear_erase_regions(page)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        if let Some(ref mut editor) = self.editor {
            editor.clear_erase_regions(page);
        }
        Ok(())
    }

    /// Flatten page annotations.
    fn flatten_page_annotations(&mut self, page: usize) -> PyResult<()> {
        self.ensure_editor()?;
        if let Some(ref mut editor) = self.editor {
            editor
                .flatten_page_annotations(page)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }
        Ok(())
    }

    /// Flatten all annotations.
    fn flatten_all_annotations(&mut self) -> PyResult<()> {
        self.ensure_editor()?;
        if let Some(ref mut editor) = self.editor {
            editor
                .flatten_all_annotations()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }
        Ok(())
    }

    /// Check if page marked for flatten.
    fn is_page_marked_for_flatten(&self, page: usize) -> bool {
        self.editor
            .as_ref()
            .is_some_and(|e| e.is_page_marked_for_flatten(page))
    }

    /// Unmark page for flatten.
    fn unmark_page_for_flatten(&mut self, page: usize) {
        if let Some(ref mut editor) = self.editor {
            editor.unmark_page_for_flatten(page);
        }
    }

    /// Apply page redactions.
    fn apply_page_redactions(&mut self, page: usize) -> PyResult<()> {
        self.ensure_editor()?;
        if let Some(ref mut editor) = self.editor {
            editor
                .apply_page_redactions(page)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }
        Ok(())
    }

    /// Apply all redactions.
    fn apply_all_redactions(&mut self) -> PyResult<()> {
        self.ensure_editor()?;
        if let Some(ref mut editor) = self.editor {
            editor
                .apply_all_redactions()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }
        Ok(())
    }

    /// Check if page marked for redaction.
    fn is_page_marked_for_redaction(&self, page: usize) -> bool {
        self.editor
            .as_ref()
            .is_some_and(|e| e.is_page_marked_for_redaction(page))
    }

    /// Unmark page for redaction.
    fn unmark_page_for_redaction(&mut self, page: usize) {
        if let Some(ref mut editor) = self.editor {
            editor.unmark_page_for_redaction(page);
        }
    }

    /// Get page images info.
    fn page_images(&mut self, page: usize, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.ensure_editor()?;
        if let Some(ref mut editor) = self.editor {
            let images = editor
                .get_page_images(page)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let list = pyo3::types::PyList::empty(py);
            for img in images {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("name", &img.name)?;
                dict.set_item("x", img.bounds[0])?;
                dict.set_item("y", img.bounds[1])?;
                dict.set_item("width", img.bounds[2])?;
                dict.set_item("height", img.bounds[3])?;
                dict.set_item(
                    "matrix",
                    (
                        img.matrix[0],
                        img.matrix[1],
                        img.matrix[2],
                        img.matrix[3],
                        img.matrix[4],
                        img.matrix[5],
                    ),
                )?;
                list.append(dict)?;
            }
            Ok(list.into())
        } else {
            Err(PyRuntimeError::new_err("No editor initialized."))
        }
    }

    /// Reposition image.
    fn reposition_image(&mut self, page: usize, image_name: &str, x: f32, y: f32) -> PyResult<()> {
        self.ensure_editor()?;
        if let Some(ref mut editor) = self.editor {
            editor
                .reposition_image(page, image_name, x, y)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }
        Ok(())
    }

    /// Resize image.
    fn resize_image(
        &mut self,
        page: usize,
        image_name: &str,
        width: f32,
        height: f32,
    ) -> PyResult<()> {
        self.ensure_editor()?;
        if let Some(ref mut editor) = self.editor {
            editor
                .resize_image(page, image_name, width, height)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }
        Ok(())
    }

    /// Set image bounds.
    fn set_image_bounds(
        &mut self,
        page: usize,
        image_name: &str,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    ) -> PyResult<()> {
        self.ensure_editor()?;
        if let Some(ref mut editor) = self.editor {
            editor
                .set_image_bounds(page, image_name, x, y, width, height)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }
        Ok(())
    }

    /// Clear image modifications.
    fn clear_image_modifications(&mut self, page: usize) {
        if let Some(ref mut editor) = self.editor {
            editor.clear_image_modifications(page);
        }
    }

    /// Has image modifications.
    fn has_image_modifications(&self, page: usize) -> bool {
        self.editor
            .as_ref()
            .is_some_and(|e| e.has_image_modifications(page))
    }

    /// Search text.
    #[pyo3(signature = (pattern, case_insensitive=false, literal=false, whole_word=false, max_results=0))]
    fn search(
        &mut self,
        py: Python<'_>,
        pattern: &str,
        case_insensitive: bool,
        literal: bool,
        whole_word: bool,
        max_results: usize,
    ) -> PyResult<Py<PyAny>> {
        use crate::search::{SearchOptions, TextSearcher};
        let opts = SearchOptions::new()
            .with_case_insensitive(case_insensitive)
            .with_literal(literal)
            .with_whole_word(whole_word)
            .with_max_results(max_results);
        let results = TextSearcher::search(&mut self.inner, pattern, &opts)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let list = pyo3::types::PyList::empty(py);
        for r in results {
            let d = pyo3::types::PyDict::new(py);
            d.set_item("page", r.page)?;
            d.set_item("text", &r.text)?;
            d.set_item("x", r.bbox.x)?;
            d.set_item("y", r.bbox.y)?;
            d.set_item("width", r.bbox.width)?;
            d.set_item("height", r.bbox.height)?;
            list.append(d)?;
        }
        Ok(list.into())
    }

    /// Search page text.
    #[pyo3(signature = (page, pattern, case_insensitive=false, literal=false, whole_word=false, max_results=0))]
    fn search_page(
        &mut self,
        py: Python<'_>,
        page: usize,
        pattern: &str,
        case_insensitive: bool,
        literal: bool,
        whole_word: bool,
        max_results: usize,
    ) -> PyResult<Py<PyAny>> {
        use crate::search::{SearchOptions, TextSearcher};
        let opts = SearchOptions::new()
            .with_case_insensitive(case_insensitive)
            .with_literal(literal)
            .with_whole_word(whole_word)
            .with_max_results(max_results)
            .with_page_range(page, page);
        let results = TextSearcher::search(&mut self.inner, pattern, &opts)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let list = pyo3::types::PyList::empty(py);
        for r in results {
            let d = pyo3::types::PyDict::new(py);
            d.set_item("page", r.page)?;
            d.set_item("text", &r.text)?;
            d.set_item("x", r.bbox.x)?;
            d.set_item("y", r.bbox.y)?;
            d.set_item("width", r.bbox.width)?;
            d.set_item("height", r.bbox.height)?;
            list.append(d)?;
        }
        Ok(list.into())
    }

    /// Extract images metadata.
    #[pyo3(signature = (page, region=None))]
    fn extract_images(
        &mut self,
        py: Python<'_>,
        page: usize,
        region: Option<(f32, f32, f32, f32)>,
    ) -> PyResult<Py<PyAny>> {
        let res = if let Some(r) = region {
            self.inner
                .extract_images_in_rect(page, crate::geometry::Rect::new(r.0, r.1, r.2, r.3))
        } else {
            self.inner.extract_images(page)
        };
        let images = res.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let list = pyo3::types::PyList::empty(py);
        for img in &images {
            let d = pyo3::types::PyDict::new(py);
            d.set_item("width", img.width())?;
            d.set_item("height", img.height())?;
            d.set_item("color_space", format!("{:?}", img.color_space()))?;
            d.set_item("bits_per_component", img.bits_per_component())?;
            if let Some(b) = img.bbox() {
                d.set_item("bbox", (b.x, b.y, b.width, b.height))?;
            } else {
                d.set_item("bbox", py.None())?;
            }
            d.set_item("rotation", img.rotation_degrees())?;
            d.set_item("matrix", img.matrix())?;
            list.append(d)?;
        }
        Ok(list.into())
    }

    /// Extract tables.
    #[pyo3(signature = (page, region=None, table_settings=None))]
    fn extract_tables(
        &mut self,
        py: Python<'_>,
        page: usize,
        region: Option<(f32, f32, f32, f32)>,
        table_settings: Option<Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let config = table_settings_to_config(table_settings)?;
        let res = if let Some(r) = region {
            self.inner.extract_tables_in_rect_with_config(
                page,
                crate::geometry::Rect::new(r.0, r.1, r.2, r.3),
                config,
            )
        } else {
            self.inner.extract_tables_with_config(page, config)
        };
        let tables = res.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let list = pyo3::types::PyList::empty(py);
        for t in &tables {
            let d = pyo3::types::PyDict::new(py);
            d.set_item("col_count", t.col_count)?;
            d.set_item("row_count", t.rows.len())?;
            if let Some(b) = t.bbox {
                d.set_item("bbox", (b.x, b.y, b.width, b.height))?;
            } else {
                d.set_item("bbox", py.None())?;
            }
            d.set_item("has_header", t.has_header)?;
            let rows = pyo3::types::PyList::empty(py);
            for r in &t.rows {
                let rd = pyo3::types::PyDict::new(py);
                rd.set_item("is_header", r.is_header)?;
                let cells = pyo3::types::PyList::empty(py);
                for c in &r.cells {
                    let cd = pyo3::types::PyDict::new(py);
                    cd.set_item("text", &c.text)?;
                    if let Some(b) = c.bbox {
                        cd.set_item("bbox", (b.x, b.y, b.width, b.height))?;
                    }
                    cells.append(cd)?;
                }
                rd.set_item("cells", cells)?;
                rows.append(rd)?;
            }
            d.set_item("rows", rows)?;
            list.append(d)?;
        }
        Ok(list.into())
    }

    /// Extract text spans.
    #[pyo3(signature = (page, region=None))]
    fn extract_spans(
        &mut self,
        page: usize,
        region: Option<(f32, f32, f32, f32)>,
    ) -> PyResult<Vec<PyTextSpan>> {
        let res = if let Some(r) = region {
            self.inner.extract_spans_in_rect(
                page,
                crate::geometry::Rect::new(r.0, r.1, r.2, r.3),
                crate::layout::RectFilterMode::Intersects,
            )
        } else {
            self.inner.extract_spans(page)
        };
        res.map(|spans| spans.into_iter().map(|s| PyTextSpan { inner: s }).collect())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Get document outline.
    fn get_outline(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let outline = self
            .inner
            .get_outline()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        match outline {
            Some(items) => Ok(Some(outline_items_to_py(py, &items)?)),
            None => Ok(None),
        }
    }

    /// Get page annotations info.
    fn get_annotations(&mut self, py: Python<'_>, page: usize) -> PyResult<Py<PyAny>> {
        let annos = self
            .inner
            .get_annotations(page)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let list = pyo3::types::PyList::empty(py);
        for a in &annos {
            let d = pyo3::types::PyDict::new(py);
            if let Some(ref s) = a.subtype {
                d.set_item("subtype", s)?;
            }
            if let Some(ref c) = a.contents {
                d.set_item("contents", c)?;
            }
            if let Some(r) = a.rect {
                d.set_item("rect", (r[0], r[1], r[2], r[3]))?;
            }
            if let Some(ref au) = a.author {
                d.set_item("author", au)?;
            }
            if let Some(ref d1) = a.creation_date {
                d.set_item("creation_date", d1)?;
            }
            if let Some(ref d2) = a.modification_date {
                d.set_item("modification_date", d2)?;
            }
            if let Some(ref s) = a.subject {
                d.set_item("subject", s)?;
            }
            if let Some(ref c) = a.color {
                if c.len() >= 3 {
                    d.set_item("color", (c[0], c[1], c[2]))?;
                }
            }
            if let Some(o) = a.opacity {
                d.set_item("opacity", o)?;
            }
            if let Some(ref f) = a.field_type {
                d.set_item("field_type", format!("{:?}", f))?;
            }
            if let Some(ref n) = a.field_name {
                d.set_item("field_name", n)?;
            }
            if let Some(ref v) = a.field_value {
                d.set_item("field_value", v)?;
            }
            if let Some(crate::annotations::LinkAction::Uri(ref u)) = a.action {
                d.set_item("action_uri", u)?;
            }
            list.append(d)?;
        }
        Ok(list.into())
    }

    /// Extract vector paths.
    #[pyo3(signature = (page, region=None))]
    fn extract_paths(
        &mut self,
        py: Python<'_>,
        page: usize,
        region: Option<(f32, f32, f32, f32)>,
    ) -> PyResult<Py<PyAny>> {
        let res = if let Some(r) = region {
            self.inner
                .extract_paths_in_rect(page, crate::geometry::Rect::new(r.0, r.1, r.2, r.3))
        } else {
            self.inner.extract_paths(page)
        };
        let paths = res.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let list = pyo3::types::PyList::empty(py);
        for p in &paths {
            list.append(path_to_py_dict(py, p)?)?;
        }
        Ok(list.into())
    }

    /// Extract rectangles.
    #[pyo3(signature = (page, region=None))]
    fn extract_rects(
        &mut self,
        py: Python<'_>,
        page: usize,
        region: Option<(f32, f32, f32, f32)>,
    ) -> PyResult<Py<PyAny>> {
        let res = if let Some(r) = region {
            self.inner
                .extract_rects_in_rect(page, crate::geometry::Rect::new(r.0, r.1, r.2, r.3))
        } else {
            self.inner.extract_rects(page)
        };
        let paths = res.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let list = pyo3::types::PyList::empty(py);
        for p in &paths {
            list.append(path_to_py_dict(py, p)?)?;
        }
        Ok(list.into())
    }

    /// Extract lines.
    #[pyo3(signature = (page, region=None))]
    fn extract_lines(
        &mut self,
        py: Python<'_>,
        page: usize,
        region: Option<(f32, f32, f32, f32)>,
    ) -> PyResult<Py<PyAny>> {
        let res = if let Some(r) = region {
            self.inner
                .extract_lines_in_rect(page, crate::geometry::Rect::new(r.0, r.1, r.2, r.3))
        } else {
            self.inner.extract_lines(page)
        };
        let paths = res.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let list = pyo3::types::PyList::empty(py);
        for p in &paths {
            list.append(path_to_py_dict(py, p)?)?;
        }
        Ok(list.into())
    }

    /// Extract text using OCR.
    #[pyo3(signature = (page, engine=None))]
    fn extract_text_ocr(
        &mut self,
        _py: Python<'_>,
        page: usize,
        engine: Option<Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        #[cfg(feature = "ocr")]
        {
            let ocr_engine = if let Some(eng) = engine {
                Some(eng.extract::<PyRef<PyOcrEngine>>()?)
            } else {
                None
            };
            let engine_inner = ocr_engine.as_ref().map(|e| &e.inner);
            let options = crate::ocr::OcrExtractOptions::default();
            self.inner
                .extract_text_with_ocr(page, engine_inner, options)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        }
        #[cfg(not(feature = "ocr"))]
        {
            let _ = (engine, page);
            Err(PyRuntimeError::new_err("OCR feature not enabled."))
        }
    }

    /// Get form fields.
    fn get_form_fields(&mut self) -> PyResult<Vec<PyFormField>> {
        use crate::extractors::forms::FormExtractor;
        let fields = FormExtractor::extract_fields(&mut self.inner)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(fields
            .into_iter()
            .map(|f| PyFormField { inner: f })
            .collect())
    }

    /// Get specific form field value.
    fn get_form_field_value(&mut self, name: &str, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.ensure_editor()?;
        let editor = self.editor.as_mut().unwrap();
        let value = editor
            .get_form_field_value(name)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        match value {
            Some(v) => form_field_value_to_python(&v, py),
            None => Ok(py.None()),
        }
    }

    /// Set form field value.
    fn set_form_field_value(&mut self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.ensure_editor()?;
        let editor = self.editor.as_mut().unwrap();
        let field_value = python_to_form_field_value(value)?;
        editor
            .set_form_field_value(name, field_value)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Has XFA.
    fn has_xfa(&mut self) -> PyResult<bool> {
        use crate::xfa::XfaExtractor;
        XfaExtractor::has_xfa(&mut self.inner).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Export form data.
    #[pyo3(signature = (path, format="fdf"))]
    fn export_form_data(&mut self, path: &str, format: &str) -> PyResult<()> {
        self.ensure_editor()?;
        let editor = self.editor.as_mut().unwrap();
        match format {
            "fdf" => editor
                .export_form_data_fdf(path)
                .map_err(|e| PyRuntimeError::new_err(e.to_string())),
            "xfdf" => editor
                .export_form_data_xfdf(path)
                .map_err(|e| PyRuntimeError::new_err(e.to_string())),
            _ => Err(PyRuntimeError::new_err("Unknown format.")),
        }
    }

    /// Extract image bytes as PNG.
    fn extract_image_bytes(&mut self, py: Python<'_>, page: usize) -> PyResult<Py<PyAny>> {
        let images = self
            .inner
            .extract_images(page)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let list = pyo3::types::PyList::empty(py);
        for img in &images {
            let png_data = img
                .to_png_bytes()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let d = pyo3::types::PyDict::new(py);
            d.set_item("width", img.width())?;
            d.set_item("height", img.height())?;
            d.set_item("format", "png")?;
            d.set_item("data", pyo3::types::PyBytes::new(py, &png_data))?;
            list.append(d)?;
        }
        Ok(list.into())
    }

    /// Flatten forms.
    fn flatten_forms(&mut self) -> PyResult<()> {
        self.ensure_editor()?;
        if let Some(ref mut editor) = self.editor {
            editor
                .flatten_forms()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }
        Ok(())
    }

    /// Flatten page forms.
    fn flatten_forms_on_page(&mut self, page: usize) -> PyResult<()> {
        self.ensure_editor()?;
        if let Some(ref mut editor) = self.editor {
            editor
                .flatten_forms_on_page(page)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }
        Ok(())
    }

    /// Merge from source.
    fn merge_from(&mut self, source: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.ensure_editor()?;
        let editor = self.editor.as_mut().unwrap();
        if let Ok(path) = source.extract::<String>() {
            editor
                .merge_from(&path)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        } else if let Ok(data) = source.extract::<Vec<u8>>() {
            editor
                .merge_from_bytes(&data)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        } else {
            Err(PyRuntimeError::new_err("Invalid source."))
        }
    }

    /// Embed file.
    fn embed_file(&mut self, name: &str, data: &Bound<'_, PyBytes>) -> PyResult<()> {
        self.ensure_editor()?;
        if let Some(ref mut editor) = self.editor {
            editor
                .embed_file(name, data.as_bytes().to_vec())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }
        Ok(())
    }

    /// Get page labels.
    fn page_labels(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use crate::extractors::page_labels::PageLabelExtractor;
        let labels = PageLabelExtractor::extract(&mut self.inner)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let list = pyo3::types::PyList::empty(py);
        for l in &labels {
            let d = pyo3::types::PyDict::new(py);
            d.set_item("start_page", l.start_page)?;
            d.set_item("style", format!("{:?}", l.style))?;
            if let Some(ref p) = l.prefix {
                d.set_item("prefix", p)?;
            } else {
                d.set_item("prefix", py.None())?;
            }
            d.set_item("start_value", l.start_value)?;
            list.append(d)?;
        }
        Ok(list.into())
    }

    /// Get XMP metadata.
    fn xmp_metadata(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use crate::extractors::xmp::XmpExtractor;
        let meta = XmpExtractor::extract(&mut self.inner)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        match meta {
            Some(xmp) => {
                let d = pyo3::types::PyDict::new(py);
                if let Some(ref t) = xmp.dc_title {
                    d.set_item("dc_title", t)?;
                }
                if !xmp.dc_creator.is_empty() {
                    d.set_item("dc_creator", &xmp.dc_creator)?;
                }
                if let Some(ref desc) = xmp.dc_description {
                    d.set_item("dc_description", desc)?;
                }
                if !xmp.dc_subject.is_empty() {
                    d.set_item("dc_subject", &xmp.dc_subject)?;
                }
                if let Some(ref l) = xmp.dc_language {
                    d.set_item("dc_language", l)?;
                }
                if let Some(ref t) = xmp.xmp_creator_tool {
                    d.set_item("xmp_creator_tool", t)?;
                }
                if let Some(ref d1) = xmp.xmp_create_date {
                    d.set_item("xmp_create_date", d1)?;
                }
                if let Some(ref d2) = xmp.xmp_modify_date {
                    d.set_item("xmp_modify_date", d2)?;
                }
                if let Some(ref p) = xmp.pdf_producer {
                    d.set_item("pdf_producer", p)?;
                }
                if let Some(ref k) = xmp.pdf_keywords {
                    d.set_item("pdf_keywords", k)?;
                }
                Ok(d.into())
            },
            None => Ok(py.None()),
        }
    }

    fn __repr__(&self) -> String {
        format!("PdfDocument(version={}.{})", self.inner.version().0, self.inner.version().1)
    }
}

/// A form field extracted from a PDF AcroForm.
#[pyclass(module = "pdf_oxide.pdf_oxide", name = "FormField", unsendable)]
pub struct PyFormField {
    inner: RustFormField,
}

#[pymethods]
impl PyFormField {
    #[allow(clippy::misnamed_getters)]
    #[getter]
    fn name(&self) -> &str {
        &self.inner.full_name
    }
    #[getter]
    fn field_type(&self) -> &str {
        match &self.inner.field_type {
            RustFieldType::Text => "text",
            RustFieldType::Button => "button",
            RustFieldType::Choice => "choice",
            RustFieldType::Signature => "signature",
            RustFieldType::Unknown(_) => "unknown",
        }
    }
    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        field_value_to_python(&self.inner.value, py)
    }
    #[getter]
    fn tooltip(&self) -> Option<&str> {
        self.inner.tooltip.as_deref()
    }
    #[getter]
    fn bounds(&self) -> Option<(f64, f64, f64, f64)> {
        self.inner.bounds.map(|b| (b[0], b[1], b[2], b[3]))
    }
    #[getter]
    fn flags(&self) -> Option<u32> {
        self.inner.flags
    }
    #[getter]
    fn max_length(&self) -> Option<u32> {
        self.inner.max_length
    }
    #[getter]
    fn is_readonly(&self) -> bool {
        self.inner
            .flags
            .is_some_and(|f| f & field_flags::READ_ONLY != 0)
    }
    #[getter]
    fn is_required(&self) -> bool {
        self.inner
            .flags
            .is_some_and(|f| f & field_flags::REQUIRED != 0)
    }
    fn __repr__(&self) -> String {
        format!("FormField(name=\"{}\", type=\"{}\")", self.inner.full_name, self.field_type())
    }
}

fn field_value_to_python(value: &RustFieldValue, py: Python<'_>) -> PyResult<Py<PyAny>> {
    match value {
        RustFieldValue::Text(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        RustFieldValue::Name(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        RustFieldValue::Boolean(b) => Ok(b.into_pyobject(py)?.to_owned().into_any().unbind()),
        RustFieldValue::Array(v) => Ok(v.into_pyobject(py)?.into_any().unbind()),
        RustFieldValue::None => Ok(py.None()),
    }
}

fn form_field_value_to_python(
    value: &crate::editor::form_fields::FormFieldValue,
    py: Python<'_>,
) -> PyResult<Py<PyAny>> {
    use crate::editor::form_fields::FormFieldValue;
    match value {
        FormFieldValue::Text(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        FormFieldValue::Choice(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        FormFieldValue::Boolean(b) => Ok(b.into_pyobject(py)?.to_owned().into_any().unbind()),
        FormFieldValue::MultiChoice(v) => Ok(v.into_pyobject(py)?.into_any().unbind()),
        FormFieldValue::None => Ok(py.None()),
    }
}

fn python_to_form_field_value(
    value: &Bound<'_, PyAny>,
) -> PyResult<crate::editor::form_fields::FormFieldValue> {
    use crate::editor::form_fields::FormFieldValue;
    if let Ok(b) = value.extract::<bool>() {
        Ok(FormFieldValue::Boolean(b))
    } else if let Ok(s) = value.extract::<String>() {
        Ok(FormFieldValue::Text(s))
    } else if let Ok(v) = value.extract::<Vec<String>>() {
        Ok(FormFieldValue::MultiChoice(v))
    } else if value.is_none() {
        Ok(FormFieldValue::None)
    } else {
        Err(PyRuntimeError::new_err("Invalid value."))
    }
}

/// Python wrapper for PDF creation.
#[pyclass(module = "pdf_oxide.pdf_oxide", name = "Pdf")]
pub struct PyPdf {
    bytes: Vec<u8>,
}

#[pymethods]
impl PyPdf {
    #[staticmethod]
    #[pyo3(signature = (content, title=None, author=None))]
    fn from_markdown(content: &str, title: Option<&str>, author: Option<&str>) -> PyResult<Self> {
        let mut b = RustPdfBuilder::new();
        if let Some(t) = title {
            b = b.title(t);
        }
        if let Some(a) = author {
            b = b.author(a);
        }
        let pdf = b
            .from_markdown(content)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyPdf {
            bytes: pdf.into_bytes(),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (content, title=None, author=None))]
    fn from_html(content: &str, title: Option<&str>, author: Option<&str>) -> PyResult<Self> {
        let mut b = RustPdfBuilder::new();
        if let Some(t) = title {
            b = b.title(t);
        }
        if let Some(a) = author {
            b = b.author(a);
        }
        let pdf = b
            .from_html(content)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyPdf {
            bytes: pdf.into_bytes(),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (content, title=None, author=None))]
    fn from_text(content: &str, title: Option<&str>, author: Option<&str>) -> PyResult<Self> {
        let mut b = RustPdfBuilder::new();
        if let Some(t) = title {
            b = b.title(t);
        }
        if let Some(a) = author {
            b = b.author(a);
        }
        let pdf = b
            .from_text(content)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyPdf {
            bytes: pdf.into_bytes(),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (content, template, title=None, author=None))]
    fn from_markdown_with_template(
        content: &str,
        template: &PyPageTemplate,
        title: Option<&str>,
        author: Option<&str>,
    ) -> PyResult<Self> {
        let mut b = RustPdfBuilder::new().template(template.inner.clone());
        if let Some(t) = title {
            b = b.title(t);
        }
        if let Some(a) = author {
            b = b.author(a);
        }
        let pdf = b
            .from_markdown(content)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyPdf {
            bytes: pdf.into_bytes(),
        })
    }

    fn save(&self, path: &str) -> PyResult<()> {
        std::fs::write(path, &self.bytes).map_err(|e| PyIOError::new_err(e.to_string()))
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> Py<PyBytes> {
        PyBytes::new(py, &self.bytes).unbind()
    }

    #[staticmethod]
    fn from_image(path: &str) -> PyResult<Self> {
        use crate::api::Pdf;
        let pdf = Pdf::from_image(path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyPdf {
            bytes: pdf.into_bytes(),
        })
    }

    #[staticmethod]
    fn from_images(paths: Vec<String>) -> PyResult<Self> {
        use crate::api::Pdf;
        let pdf = Pdf::from_images(&paths).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyPdf {
            bytes: pdf.into_bytes(),
        })
    }

    #[staticmethod]
    fn from_image_bytes(data: &Bound<'_, PyBytes>) -> PyResult<Self> {
        use crate::api::Pdf;
        let pdf = Pdf::from_image_bytes(data.as_bytes())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyPdf {
            bytes: pdf.into_bytes(),
        })
    }

    fn __len__(&self) -> usize {
        self.bytes.len()
    }
    fn __repr__(&self) -> String {
        format!("Pdf({} bytes)", self.bytes.len())
    }
}

#[cfg(feature = "office")]
use crate::converters::office::OfficeConverter as RustOfficeConverter;

#[cfg(feature = "office")]
#[pyclass(module = "pdf_oxide.pdf_oxide", name = "OfficeConverter")]
pub struct PyOfficeConverter;

#[cfg(not(feature = "office"))]
#[pyclass(module = "pdf_oxide.pdf_oxide", name = "OfficeConverter")]
pub struct PyOfficeConverter;

#[cfg(not(feature = "office"))]
#[pymethods]
impl PyOfficeConverter {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyRuntimeError::new_err("Office feature not enabled."))
    }
    #[staticmethod]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn convert(
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        Err(PyRuntimeError::new_err("Office feature not enabled."))
    }
    #[staticmethod]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn from_docx(
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        Err(PyRuntimeError::new_err("Office feature not enabled."))
    }
    #[staticmethod]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn from_xlsx(
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        Err(PyRuntimeError::new_err("Office feature not enabled."))
    }
    #[staticmethod]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn from_pptx(
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        Err(PyRuntimeError::new_err("Office feature not enabled."))
    }
}

#[cfg(feature = "office")]
#[pymethods]
impl PyOfficeConverter {
    #[staticmethod]
    fn from_docx(path: &str) -> PyResult<PyPdf> {
        let res = RustOfficeConverter::new()
            .convert_docx(path)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyPdf { bytes: res })
    }
    #[staticmethod]
    fn from_docx_bytes(data: &Bound<'_, PyBytes>) -> PyResult<PyPdf> {
        let res = RustOfficeConverter::new()
            .convert_docx_bytes(data.as_bytes())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyPdf { bytes: res })
    }
    #[staticmethod]
    fn from_xlsx(path: &str) -> PyResult<PyPdf> {
        let res = RustOfficeConverter::new()
            .convert_xlsx(path)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyPdf { bytes: res })
    }
    #[staticmethod]
    fn from_xlsx_bytes(data: &Bound<'_, PyBytes>) -> PyResult<PyPdf> {
        let res = RustOfficeConverter::new()
            .convert_xlsx_bytes(data.as_bytes())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyPdf { bytes: res })
    }
    #[staticmethod]
    fn from_pptx(path: &str) -> PyResult<PyPdf> {
        let res = RustOfficeConverter::new()
            .convert_pptx(path)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyPdf { bytes: res })
    }
    #[staticmethod]
    fn from_pptx_bytes(data: &Bound<'_, PyBytes>) -> PyResult<PyPdf> {
        let res = RustOfficeConverter::new()
            .convert_pptx_bytes(data.as_bytes())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyPdf { bytes: res })
    }
    #[staticmethod]
    fn convert(path: &str) -> PyResult<PyPdf> {
        let res = RustOfficeConverter::new()
            .convert(path)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyPdf { bytes: res })
    }
}

use crate::editor::{ElementId, PdfElement, PdfPage as RustPdfPage, PdfText as RustPdfText};

#[pyclass(module = "pdf_oxide.pdf_oxide", name = "PdfPageRegion")]
pub struct PyPdfPageRegion {
    pub doc: Py<PyPdfDocument>,
    pub page_index: usize,
    pub region: crate::geometry::Rect,
}

#[pymethods]
impl PyPdfPageRegion {
    #[getter]
    fn bbox(&self) -> (f32, f32, f32, f32) {
        (self.region.x, self.region.y, self.region.width, self.region.height)
    }
    fn extract_text(&self, py: Python<'_>) -> PyResult<String> {
        let mut d = self.doc.bind(py).borrow_mut();
        d.extract_text(self.page_index, Some(self.bbox()))
    }
    fn extract_words(&self, py: Python<'_>) -> PyResult<Vec<PyWord>> {
        let mut d = self.doc.bind(py).borrow_mut();
        d.extract_words(self.page_index, Some(self.bbox()))
    }
    fn extract_text_lines(&self, py: Python<'_>) -> PyResult<Vec<PyTextLine>> {
        let mut d = self.doc.bind(py).borrow_mut();
        d.extract_text_lines(self.page_index, Some(self.bbox()))
    }
    #[pyo3(signature = (table_settings=None))]
    fn extract_tables(
        &self,
        py: Python<'_>,
        table_settings: Option<Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let mut d = self.doc.bind(py).borrow_mut();
        d.extract_tables(py, self.page_index, Some(self.bbox()), table_settings)
    }
    fn extract_images(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut d = self.doc.bind(py).borrow_mut();
        d.extract_images(py, self.page_index, Some(self.bbox()))
    }
    fn extract_paths(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut d = self.doc.bind(py).borrow_mut();
        let res = d
            .inner
            .extract_paths_in_rect(self.page_index, self.region)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let list = pyo3::types::PyList::empty(py);
        for p in &res {
            list.append(path_to_py_dict(py, p)?)?;
        }
        Ok(list.into())
    }
    fn __repr__(&self) -> String {
        format!("PdfPageRegion(page={}, bbox={:?})", self.page_index, self.region)
    }
}

#[pyclass(module = "pdf_oxide.pdf_oxide", name = "PdfPage", unsendable)]
pub struct PyPdfPage {
    inner: RustPdfPage,
}

#[pymethods]
impl PyPdfPage {
    #[getter]
    fn index(&self) -> usize {
        self.inner.page_index
    }
    #[getter]
    fn width(&self) -> f32 {
        self.inner.width
    }
    #[getter]
    fn height(&self) -> f32 {
        self.inner.height
    }
    fn children(&self) -> Vec<PyPdfElement> {
        self.inner
            .children()
            .into_iter()
            .map(|e| PyPdfElement { inner: e })
            .collect()
    }
    fn find_text_containing(&self, needle: &str) -> Vec<PyPdfText> {
        self.inner
            .find_text_containing(needle)
            .into_iter()
            .map(|t| PyPdfText { inner: t })
            .collect()
    }
    fn find_images(&self) -> Vec<PyPdfImage> {
        self.inner
            .find_images()
            .into_iter()
            .map(|i| PyPdfImage { inner: i })
            .collect()
    }
    fn get_element(&self, _id: &str) -> Option<PyPdfElement> {
        None
    }
    fn set_text(&mut self, text_id: &PyPdfTextId, new_text: &str) -> PyResult<()> {
        self.inner
            .set_text(text_id.inner, new_text)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }
    fn annotations(&self) -> Vec<PyAnnotationWrapper> {
        self.inner
            .annotations()
            .iter()
            .map(|a| PyAnnotationWrapper { inner: a.clone() })
            .collect()
    }
    fn add_link(&mut self, x: f32, y: f32, width: f32, height: f32, url: &str) -> String {
        use crate::writer::LinkAnnotation;
        let l = LinkAnnotation::uri(crate::geometry::Rect::new(x, y, width, height), url);
        format!("{:?}", self.inner.add_annotation(l))
    }
    fn add_highlight(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        color: (f32, f32, f32),
    ) -> String {
        use crate::writer::TextMarkupAnnotation;
        use crate::TextMarkupType;
        let l = TextMarkupAnnotation::from_rect(
            TextMarkupType::Highlight,
            crate::geometry::Rect::new(x, y, width, height),
        )
        .with_color(color.0, color.1, color.2);
        format!("{:?}", self.inner.add_annotation(l))
    }
    fn add_note(&mut self, x: f32, y: f32, text: &str) -> String {
        use crate::writer::TextAnnotation;
        let l = TextAnnotation::new(crate::geometry::Rect::new(x, y, 24.0, 24.0), text);
        format!("{:?}", self.inner.add_annotation(l))
    }
    fn remove_annotation(&mut self, index: usize) -> bool {
        self.inner.remove_annotation(index).is_some()
    }
    #[pyo3(signature = (text, x, y, font_size=12.0))]
    fn add_text(&mut self, text: &str, x: f32, y: f32, font_size: f32) -> PyPdfTextId {
        use crate::elements::{FontSpec, TextContent, TextStyle};
        let c = TextContent {
            text: text.to_string(),
            bbox: crate::geometry::Rect::new(x, y, text.len() as f32 * font_size * 0.6, font_size),
            font: FontSpec {
                name: "Helvetica".to_string(),
                size: font_size,
            },
            style: TextStyle::default(),
            reading_order: None,
            artifact_type: None,
            origin: None,
            rotation_degrees: None,
            matrix: None,
        };
        PyPdfTextId {
            inner: self.inner.add_text(c),
        }
    }
    fn remove_element(&mut self, id: &PyPdfTextId) -> bool {
        self.inner.remove_element(id.inner)
    }
    fn __repr__(&self) -> String {
        format!(
            "PdfPage(index={}, width={:.1}, height={:.1})",
            self.inner.page_index, self.inner.width, self.inner.height
        )
    }
}

#[pyclass(module = "pdf_oxide.pdf_oxide", name = "PdfTextId")]
#[derive(Clone)]
pub struct PyPdfTextId {
    inner: ElementId,
}
#[pymethods]
impl PyPdfTextId {
    fn __repr__(&self) -> String {
        format!("PdfTextId({:?})", self.inner)
    }
}

#[pyclass(module = "pdf_oxide.pdf_oxide", name = "PdfText")]
#[derive(Clone)]
pub struct PyPdfText {
    inner: RustPdfText,
}
#[pymethods]
impl PyPdfText {
    #[getter]
    fn id(&self) -> PyPdfTextId {
        PyPdfTextId {
            inner: self.inner.id(),
        }
    }
    #[getter]
    fn value(&self) -> String {
        self.inner.text().to_string()
    }
    #[getter]
    fn text(&self) -> String {
        self.value()
    }
    #[getter]
    fn bbox(&self) -> (f32, f32, f32, f32) {
        let r = self.inner.bbox();
        (r.x, r.y, r.width, r.height)
    }
    #[getter]
    fn font_name(&self) -> String {
        self.inner.font_name().to_string()
    }
    #[getter]
    fn font_size(&self) -> f32 {
        self.inner.font_size()
    }
    #[getter]
    fn is_bold(&self) -> bool {
        self.inner.is_bold()
    }
    #[getter]
    fn is_italic(&self) -> bool {
        self.inner.is_italic()
    }
    fn contains(&self, n: &str) -> bool {
        self.inner.contains(n)
    }
    fn starts_with(&self, p: &str) -> bool {
        self.inner.starts_with(p)
    }
    fn ends_with(&self, s: &str) -> bool {
        self.inner.ends_with(s)
    }
    fn __repr__(&self) -> String {
        format!("PdfText({:?})", self.inner.text())
    }
}

#[pyclass(module = "pdf_oxide.pdf_oxide", name = "PdfImage")]
#[derive(Clone)]
pub struct PyPdfImage {
    inner: crate::editor::PdfImage,
}
#[pymethods]
impl PyPdfImage {
    #[getter]
    fn bbox(&self) -> (f32, f32, f32, f32) {
        let r = self.inner.bbox();
        (r.x, r.y, r.width, r.height)
    }
    #[getter]
    fn width(&self) -> u32 {
        self.inner.dimensions().0
    }
    #[getter]
    fn height(&self) -> u32 {
        self.inner.dimensions().1
    }
    #[getter]
    fn aspect_ratio(&self) -> f32 {
        self.inner.aspect_ratio()
    }
    fn __repr__(&self) -> String {
        let (w, h) = self.inner.dimensions();
        format!("PdfImage({}x{})", w, h)
    }
}

#[pyclass(module = "pdf_oxide.pdf_oxide", name = "PdfAnnotation")]
#[derive(Clone)]
pub struct PyAnnotationWrapper {
    inner: crate::editor::AnnotationWrapper,
}
#[pymethods]
impl PyAnnotationWrapper {
    #[getter]
    fn subtype(&self) -> String {
        format!("{:?}", self.inner.subtype())
    }
    #[getter]
    fn rect(&self) -> (f32, f32, f32, f32) {
        let r = self.inner.rect();
        (r.x, r.y, r.width, r.height)
    }
    #[getter]
    fn contents(&self) -> Option<String> {
        self.inner.contents().map(|s| s.to_string())
    }
    #[getter]
    fn color(&self) -> Option<(f32, f32, f32)> {
        self.inner.color()
    }
    #[getter]
    fn is_modified(&self) -> bool {
        self.inner.is_modified()
    }
    #[getter]
    fn is_new(&self) -> bool {
        self.inner.is_new()
    }
    fn __repr__(&self) -> String {
        format!("PdfAnnotation(subtype={:?})", self.inner.subtype())
    }
}

#[pyclass(module = "pdf_oxide.pdf_oxide", name = "PdfElement")]
#[derive(Clone)]
pub struct PyPdfElement {
    inner: PdfElement,
}
#[pymethods]
impl PyPdfElement {
    fn is_text(&self) -> bool {
        self.inner.is_text()
    }
    fn is_image(&self) -> bool {
        self.inner.is_image()
    }
    fn is_path(&self) -> bool {
        self.inner.is_path()
    }
    fn is_table(&self) -> bool {
        self.inner.is_table()
    }
    fn is_structure(&self) -> bool {
        self.inner.is_structure()
    }
    fn as_text(&self) -> Option<PyPdfText> {
        if let PdfElement::Text(t) = &self.inner {
            Some(PyPdfText { inner: t.clone() })
        } else {
            None
        }
    }
    fn as_image(&self) -> Option<PyPdfImage> {
        if let PdfElement::Image(i) = &self.inner {
            Some(PyPdfImage { inner: i.clone() })
        } else {
            None
        }
    }
    #[getter]
    fn bbox(&self) -> (f32, f32, f32, f32) {
        let r = self.inner.bbox();
        (r.x, r.y, r.width, r.height)
    }
    fn __repr__(&self) -> String {
        "PdfElement".to_string()
    }
}

#[pyclass(module = "pdf_oxide.pdf_oxide", name = "TextChar")]
#[derive(Clone)]
pub struct PyTextChar {
    inner: RustTextChar,
}
#[pymethods]
impl PyTextChar {
    #[getter]
    fn char(&self) -> char {
        self.inner.char
    }
    #[getter]
    fn bbox(&self) -> (f32, f32, f32, f32) {
        (
            self.inner.bbox.x,
            self.inner.bbox.y,
            self.inner.bbox.width,
            self.inner.bbox.height,
        )
    }
    #[getter]
    fn font_name(&self) -> String {
        self.inner.font_name.clone()
    }
    #[getter]
    fn font_size(&self) -> f32 {
        self.inner.font_size
    }
    #[getter]
    fn font_weight(&self) -> String {
        format!("{:?}", self.inner.font_weight)
    }
    #[getter]
    fn is_italic(&self) -> bool {
        self.inner.is_italic
    }
    #[getter]
    fn color(&self) -> (f32, f32, f32) {
        (self.inner.color.r, self.inner.color.g, self.inner.color.b)
    }
    #[getter]
    fn rotation_degrees(&self) -> f32 {
        self.inner.rotation_degrees
    }
    #[getter]
    fn origin_x(&self) -> f32 {
        self.inner.origin_x
    }
    #[getter]
    fn origin_y(&self) -> f32 {
        self.inner.origin_y
    }
    #[getter]
    fn advance_width(&self) -> f32 {
        self.inner.advance_width
    }
    #[getter]
    fn mcid(&self) -> Option<u32> {
        self.inner.mcid
    }
}

#[pyclass(module = "pdf_oxide.pdf_oxide", name = "TextSpan")]
#[derive(Clone)]
pub struct PyTextSpan {
    inner: crate::layout::TextSpan,
}
#[pymethods]
impl PyTextSpan {
    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }
    #[getter]
    fn bbox(&self) -> (f32, f32, f32, f32) {
        (
            self.inner.bbox.x,
            self.inner.bbox.y,
            self.inner.bbox.width,
            self.inner.bbox.height,
        )
    }
    #[getter]
    fn font_name(&self) -> &str {
        &self.inner.font_name
    }
    #[getter]
    fn font_size(&self) -> f32 {
        self.inner.font_size
    }
    #[getter]
    fn is_bold(&self) -> bool {
        self.inner.font_weight as u16 >= 700
    }
    #[getter]
    fn is_italic(&self) -> bool {
        self.inner.is_italic
    }
    #[getter]
    fn color(&self) -> (f32, f32, f32) {
        (self.inner.color.r, self.inner.color.g, self.inner.color.b)
    }
}

#[pyclass(module = "pdf_oxide.pdf_oxide", name = "TextWord")]
#[derive(Clone)]
pub struct PyWord {
    inner: crate::layout::Word,
}
#[pymethods]
impl PyWord {
    #[getter]
    fn text(&self) -> String {
        self.inner.text.clone()
    }
    #[getter]
    fn bbox(&self) -> (f32, f32, f32, f32) {
        (
            self.inner.bbox.x,
            self.inner.bbox.y,
            self.inner.bbox.width,
            self.inner.bbox.height,
        )
    }
    #[getter]
    fn font_name(&self) -> String {
        self.inner.dominant_font.clone()
    }
    #[getter]
    fn font_size(&self) -> f32 {
        self.inner.avg_font_size
    }
    #[getter]
    fn is_bold(&self) -> bool {
        self.inner.is_bold
    }
    #[getter]
    fn is_italic(&self) -> bool {
        self.inner.is_italic
    }
    #[getter]
    fn chars(&self) -> Vec<PyTextChar> {
        self.inner
            .chars
            .iter()
            .map(|c| PyTextChar { inner: c.clone() })
            .collect()
    }
}

#[pyclass(module = "pdf_oxide.pdf_oxide", name = "TextLine")]
#[derive(Clone)]
pub struct PyTextLine {
    inner: crate::layout::TextLine,
}
#[pymethods]
impl PyTextLine {
    #[getter]
    fn text(&self) -> String {
        self.inner.text.clone()
    }
    #[getter]
    fn bbox(&self) -> (f32, f32, f32, f32) {
        (
            self.inner.bbox.x,
            self.inner.bbox.y,
            self.inner.bbox.width,
            self.inner.bbox.height,
        )
    }
    #[getter]
    fn words(&self) -> Vec<PyWord> {
        self.inner
            .words
            .iter()
            .map(|w| PyWord { inner: w.clone() })
            .collect()
    }
    #[getter]
    fn chars(&self) -> Vec<PyTextChar> {
        self.inner
            .words
            .iter()
            .flat_map(|w| w.chars.iter().map(|c| PyTextChar { inner: c.clone() }))
            .collect()
    }
}

fn path_to_py_dict(py: Python<'_>, path: &crate::elements::PathContent) -> PyResult<Py<PyAny>> {
    let d = pyo3::types::PyDict::new(py);
    d.set_item("bbox", (path.bbox.x, path.bbox.y, path.bbox.width, path.bbox.height))?;
    d.set_item("stroke_width", path.stroke_width)?;
    if let Some(ref c) = path.stroke_color {
        d.set_item("stroke_color", (c.r, c.g, c.b))?;
    } else {
        d.set_item("stroke_color", py.None())?;
    }
    if let Some(ref c) = path.fill_color {
        d.set_item("fill_color", (c.r, c.g, c.b))?;
    } else {
        d.set_item("fill_color", py.None())?;
    }
    d.set_item("operations_count", path.operations.len())?;

    // Expose path operations as list of dicts for vector extraction use cases
    let ops_list = pyo3::types::PyList::empty(py);
    for op in &path.operations {
        let op_dict = pyo3::types::PyDict::new(py);
        match op {
            crate::elements::PathOperation::MoveTo(x, y) => {
                op_dict.set_item("op", "move_to")?;
                op_dict.set_item("x", *x)?;
                op_dict.set_item("y", *y)?;
            },
            crate::elements::PathOperation::LineTo(x, y) => {
                op_dict.set_item("op", "line_to")?;
                op_dict.set_item("x", *x)?;
                op_dict.set_item("y", *y)?;
            },
            crate::elements::PathOperation::CurveTo(cx1, cy1, cx2, cy2, x, y) => {
                op_dict.set_item("op", "curve_to")?;
                op_dict.set_item("cx1", *cx1)?;
                op_dict.set_item("cy1", *cy1)?;
                op_dict.set_item("cx2", *cx2)?;
                op_dict.set_item("cy2", *cy2)?;
                op_dict.set_item("x", *x)?;
                op_dict.set_item("y", *y)?;
            },
            crate::elements::PathOperation::Rectangle(x, y, w, h) => {
                op_dict.set_item("op", "rectangle")?;
                op_dict.set_item("x", *x)?;
                op_dict.set_item("y", *y)?;
                op_dict.set_item("width", *w)?;
                op_dict.set_item("height", *h)?;
            },
            crate::elements::PathOperation::ClosePath => {
                op_dict.set_item("op", "close_path")?;
            },
        }
        ops_list.append(op_dict)?;
    }
    d.set_item("operations", ops_list)?;

    Ok(d.into())
}

fn table_settings_to_config(
    settings: Option<Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<crate::structure::spatial_table_detector::TableDetectionConfig> {
    use crate::structure::spatial_table_detector::{TableDetectionConfig, TableStrategy};
    let mut c = TableDetectionConfig::relaxed();
    if let Some(d) = settings {
        if let Some(v) = d.get_item("horizontal_strategy")? {
            let s: String = v.extract()?;
            c.horizontal_strategy = match s.as_str() {
                "lines" => TableStrategy::Lines,
                "text" => TableStrategy::Text,
                "both" => TableStrategy::Both,
                _ => return Err(PyRuntimeError::new_err("Invalid strategy")),
            };
        }
        if let Some(v) = d.get_item("column_tolerance")? {
            c.column_tolerance = v.extract()?;
        }
        if let Some(v) = d.get_item("row_tolerance")? {
            c.row_tolerance = v.extract()?;
        }
        if let Some(v) = d.get_item("min_table_cells")? {
            c.min_table_cells = v.extract()?;
        }
    }
    Ok(c)
}

fn outline_items_to_py(
    py: Python<'_>,
    items: &[crate::outline::OutlineItem],
) -> PyResult<Py<PyAny>> {
    let list = pyo3::types::PyList::empty(py);
    for i in items {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("title", &i.title)?;
        match &i.dest {
            Some(crate::outline::Destination::PageIndex(idx)) => {
                d.set_item("page", *idx)?;
            },
            _ => {
                d.set_item("page", py.None())?;
            },
        }
        d.set_item("children", outline_items_to_py(py, &i.children)?)?;
        list.append(d)?;
    }
    Ok(list.into())
}

#[cfg(feature = "ocr")]
#[pyclass(module = "pdf_oxide.pdf_oxide", name = "OcrEngine", unsendable)]
pub struct PyOcrEngine {
    inner: crate::ocr::OcrEngine,
}
#[cfg(not(feature = "ocr"))]
#[pyclass(module = "pdf_oxide.pdf_oxide", name = "OcrEngine", unsendable)]
pub struct PyOcrEngine {}
#[cfg(not(feature = "ocr"))]
#[pymethods]
impl PyOcrEngine {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<Bound<'_, PyDict>>) -> PyResult<Self> {
        Err(PyRuntimeError::new_err("OCR not enabled."))
    }
}
#[cfg(feature = "ocr")]
#[pymethods]
impl PyOcrEngine {
    #[new]
    #[pyo3(signature = (det_model_path, rec_model_path, dict_path, config=None))]
    fn new(
        det_model_path: &str,
        rec_model_path: &str,
        dict_path: &str,
        config: Option<&PyOcrConfig>,
    ) -> PyResult<Self> {
        let c = config.map(|c| c.inner.clone()).unwrap_or_default();
        let e = crate::ocr::OcrEngine::new(det_model_path, rec_model_path, dict_path, c)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyOcrEngine { inner: e })
    }
}

#[cfg(feature = "ocr")]
#[pyclass(module = "pdf_oxide.pdf_oxide", name = "OcrConfig")]
#[derive(Clone)]
pub struct PyOcrConfig {
    inner: crate::ocr::OcrConfig,
}
#[cfg(not(feature = "ocr"))]
#[pyclass(module = "pdf_oxide.pdf_oxide", name = "OcrConfig")]
#[derive(Clone)]
pub struct PyOcrConfig {}
#[cfg(not(feature = "ocr"))]
#[pymethods]
impl PyOcrConfig {
    #[new]
    #[pyo3(signature = (**_kwargs))]
    fn new(_kwargs: Option<Bound<'_, PyDict>>) -> PyResult<Self> {
        Err(PyRuntimeError::new_err("OCR not enabled."))
    }
}
#[cfg(feature = "ocr")]
#[pymethods]
impl PyOcrConfig {
    #[new]
    #[pyo3(signature = (det_threshold=None, rec_threshold=None, num_threads=None))]
    fn new(
        det_threshold: Option<f32>,
        rec_threshold: Option<f32>,
        num_threads: Option<usize>,
    ) -> Self {
        let mut c = crate::ocr::OcrConfig::default();
        if let Some(v) = det_threshold {
            c.det_threshold = v;
        }
        if let Some(v) = rec_threshold {
            c.rec_threshold = v;
        }
        if let Some(v) = num_threads {
            c.num_threads = v;
        }
        PyOcrConfig { inner: c }
    }
}

#[pyclass(module = "pdf_oxide.pdf_oxide", name = "Color")]
#[derive(Clone)]
pub struct PyColor {
    inner: RustColor,
}
#[pymethods]
impl PyColor {
    #[new]
    fn new(r: f32, g: f32, b: f32) -> Self {
        PyColor {
            inner: RustColor::new(r, g, b),
        }
    }
    #[staticmethod]
    fn black() -> Self {
        PyColor {
            inner: RustColor::black(),
        }
    }
    #[staticmethod]
    fn white() -> Self {
        PyColor {
            inner: RustColor::white(),
        }
    }
    #[staticmethod]
    fn red() -> Self {
        PyColor {
            inner: RustColor::new(1.0, 0.0, 0.0),
        }
    }
    #[staticmethod]
    fn green() -> Self {
        PyColor {
            inner: RustColor::new(0.0, 1.0, 0.0),
        }
    }
    #[staticmethod]
    fn blue() -> Self {
        PyColor {
            inner: RustColor::new(0.0, 0.0, 1.0),
        }
    }
    #[staticmethod]
    fn from_hex(hex: &str) -> PyResult<Self> {
        let hex = hex.trim_start_matches('#');
        if hex.len() != 6 {
            return Err(PyRuntimeError::new_err("Invalid hex color length"));
        }
        let r = u8::from_str_radix(&hex[0..2], 16)
            .map_err(|_| PyRuntimeError::new_err("Invalid hex color"))? as f32
            / 255.0;
        let g = u8::from_str_radix(&hex[2..4], 16)
            .map_err(|_| PyRuntimeError::new_err("Invalid hex color"))? as f32
            / 255.0;
        let b = u8::from_str_radix(&hex[4..6], 16)
            .map_err(|_| PyRuntimeError::new_err("Invalid hex color"))? as f32
            / 255.0;
        Ok(PyColor {
            inner: RustColor::new(r, g, b),
        })
    }
    #[getter]
    fn r(&self) -> f32 {
        self.inner.r
    }
    #[getter]
    fn g(&self) -> f32 {
        self.inner.g
    }
    #[getter]
    fn b(&self) -> f32 {
        self.inner.b
    }
}

#[allow(dead_code)]
#[pyclass(module = "pdf_oxide.pdf_oxide", name = "BlendMode")]
#[derive(Clone)]
pub struct PyBlendMode {
    inner: RustBlendMode,
}
#[pymethods]
impl PyBlendMode {
    #[staticmethod]
    #[allow(non_snake_case)]
    fn NORMAL() -> Self {
        PyBlendMode {
            inner: RustBlendMode::Normal,
        }
    }
    #[staticmethod]
    #[allow(non_snake_case)]
    fn MULTIPLY() -> Self {
        PyBlendMode {
            inner: RustBlendMode::Multiply,
        }
    }
    #[staticmethod]
    #[allow(non_snake_case)]
    fn SCREEN() -> Self {
        PyBlendMode {
            inner: RustBlendMode::Screen,
        }
    }
    #[staticmethod]
    #[allow(non_snake_case)]
    fn OVERLAY() -> Self {
        PyBlendMode {
            inner: RustBlendMode::Overlay,
        }
    }
    #[staticmethod]
    #[allow(non_snake_case)]
    fn DARKEN() -> Self {
        PyBlendMode {
            inner: RustBlendMode::Darken,
        }
    }
    #[staticmethod]
    #[allow(non_snake_case)]
    fn LIGHTEN() -> Self {
        PyBlendMode {
            inner: RustBlendMode::Lighten,
        }
    }
    #[staticmethod]
    #[allow(non_snake_case)]
    fn COLOR_DODGE() -> Self {
        PyBlendMode {
            inner: RustBlendMode::ColorDodge,
        }
    }
    #[staticmethod]
    #[allow(non_snake_case)]
    fn COLOR_BURN() -> Self {
        PyBlendMode {
            inner: RustBlendMode::ColorBurn,
        }
    }
    #[staticmethod]
    #[allow(non_snake_case)]
    fn HARD_LIGHT() -> Self {
        PyBlendMode {
            inner: RustBlendMode::HardLight,
        }
    }
    #[staticmethod]
    #[allow(non_snake_case)]
    fn SOFT_LIGHT() -> Self {
        PyBlendMode {
            inner: RustBlendMode::SoftLight,
        }
    }
    #[staticmethod]
    #[allow(non_snake_case)]
    fn DIFFERENCE() -> Self {
        PyBlendMode {
            inner: RustBlendMode::Difference,
        }
    }
    #[staticmethod]
    #[allow(non_snake_case)]
    fn EXCLUSION() -> Self {
        PyBlendMode {
            inner: RustBlendMode::Exclusion,
        }
    }
}

#[allow(dead_code)]
#[pyclass(module = "pdf_oxide.pdf_oxide", name = "ExtGState")]
#[derive(Clone)]
pub struct PyExtGState {
    fill_alpha: Option<f32>,
    stroke_alpha: Option<f32>,
    blend_mode: Option<RustBlendMode>,
}
#[pymethods]
impl PyExtGState {
    #[new]
    fn new() -> Self {
        PyExtGState {
            fill_alpha: None,
            stroke_alpha: None,
            blend_mode: None,
        }
    }
    fn alpha(&self, a: f32) -> Self {
        let v = Some(a.clamp(0.0, 1.0));
        PyExtGState {
            fill_alpha: v,
            stroke_alpha: v,
            blend_mode: self.blend_mode,
        }
    }
    fn fill_alpha(&self, a: f32) -> Self {
        PyExtGState {
            fill_alpha: Some(a.clamp(0.0, 1.0)),
            stroke_alpha: self.stroke_alpha,
            blend_mode: self.blend_mode,
        }
    }
    fn stroke_alpha(&self, a: f32) -> Self {
        PyExtGState {
            fill_alpha: self.fill_alpha,
            stroke_alpha: Some(a.clamp(0.0, 1.0)),
            blend_mode: self.blend_mode,
        }
    }
    fn blend_mode(&self, mode: &PyBlendMode) -> Self {
        PyExtGState {
            fill_alpha: self.fill_alpha,
            stroke_alpha: self.stroke_alpha,
            blend_mode: Some(mode.inner),
        }
    }
    #[staticmethod]
    fn semi_transparent() -> Self {
        PyExtGState {
            fill_alpha: Some(0.5),
            stroke_alpha: Some(0.5),
            blend_mode: None,
        }
    }
}

#[pyclass(module = "pdf_oxide.pdf_oxide", name = "LinearGradient")]
#[derive(Clone)]
pub struct PyLinearGradient {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    stops: Vec<(f32, RustColor)>,
}
#[pymethods]
impl PyLinearGradient {
    #[new]
    fn new() -> Self {
        PyLinearGradient {
            x1: 0.0,
            y1: 0.0,
            x2: 100.0,
            y2: 100.0,
            stops: Vec::new(),
        }
    }
    fn start(&self, x: f32, y: f32) -> Self {
        let mut slf = self.clone();
        slf.x1 = x;
        slf.y1 = y;
        slf
    }
    fn end(&self, x: f32, y: f32) -> Self {
        let mut slf = self.clone();
        slf.x2 = x;
        slf.y2 = y;
        slf
    }
    fn add_stop(&self, offset: f32, color: &PyColor) -> Self {
        let mut slf = self.clone();
        slf.stops.push((offset, color.inner));
        slf
    }
    #[staticmethod]
    fn horizontal(width: f32, start: &PyColor, end: &PyColor) -> Self {
        PyLinearGradient {
            x1: 0.0,
            y1: 0.0,
            x2: width,
            y2: 0.0,
            stops: vec![(0.0, start.inner), (1.0, end.inner)],
        }
    }
    #[staticmethod]
    fn vertical(height: f32, start: &PyColor, end: &PyColor) -> Self {
        PyLinearGradient {
            x1: 0.0,
            y1: 0.0,
            x2: 0.0,
            y2: height,
            stops: vec![(0.0, start.inner), (1.0, end.inner)],
        }
    }
}

#[pyclass(module = "pdf_oxide.pdf_oxide", name = "RadialGradient")]
#[derive(Clone)]
pub struct PyRadialGradient {
    x1: f32,
    y1: f32,
    r1: f32,
    x2: f32,
    y2: f32,
    r2: f32,
    stops: Vec<(f32, RustColor)>,
}
#[pymethods]
impl PyRadialGradient {
    #[new]
    fn new() -> Self {
        PyRadialGradient {
            x1: 50.0,
            y1: 50.0,
            r1: 0.0,
            x2: 50.0,
            y2: 50.0,
            r2: 50.0,
            stops: Vec::new(),
        }
    }
    fn inner_circle(&self, x: f32, y: f32, r: f32) -> Self {
        let mut slf = self.clone();
        slf.x1 = x;
        slf.y1 = y;
        slf.r1 = r;
        slf
    }
    fn outer_circle(&self, x: f32, y: f32, r: f32) -> Self {
        let mut slf = self.clone();
        slf.x2 = x;
        slf.y2 = y;
        slf.r2 = r;
        slf
    }
    fn add_stop(&self, offset: f32, color: &PyColor) -> Self {
        let mut slf = self.clone();
        slf.stops.push((offset, color.inner));
        slf
    }
    #[staticmethod]
    fn centered(x: f32, y: f32, radius: f32) -> Self {
        PyRadialGradient {
            x1: x,
            y1: y,
            r1: 0.0,
            x2: x,
            y2: y,
            r2: radius,
            stops: Vec::new(),
        }
    }
}

#[pyclass(module = "pdf_oxide.pdf_oxide", name = "LineCap")]
#[derive(Clone)]
pub struct PyLineCap {
    pub inner: RustLineCap,
}
#[pymethods]
impl PyLineCap {
    #[staticmethod]
    fn butt() -> Self {
        Self {
            inner: RustLineCap::Butt,
        }
    }
    #[staticmethod]
    fn round() -> Self {
        Self {
            inner: RustLineCap::Round,
        }
    }
    #[staticmethod]
    fn square() -> Self {
        Self {
            inner: RustLineCap::Square,
        }
    }
    #[staticmethod]
    #[allow(non_snake_case)]
    fn BUTT() -> Self {
        Self::butt()
    }
    #[staticmethod]
    #[allow(non_snake_case)]
    fn ROUND() -> Self {
        Self::round()
    }
    #[staticmethod]
    #[allow(non_snake_case)]
    fn SQUARE() -> Self {
        Self::square()
    }
}

#[pyclass(module = "pdf_oxide.pdf_oxide", name = "LineJoin")]
#[derive(Clone)]
pub struct PyLineJoin {
    pub inner: RustLineJoin,
}
#[pymethods]
impl PyLineJoin {
    #[staticmethod]
    fn miter() -> Self {
        Self {
            inner: RustLineJoin::Miter,
        }
    }
    #[staticmethod]
    fn round() -> Self {
        Self {
            inner: RustLineJoin::Round,
        }
    }
    #[staticmethod]
    fn bevel() -> Self {
        Self {
            inner: RustLineJoin::Bevel,
        }
    }
    #[staticmethod]
    #[allow(non_snake_case)]
    fn MITER() -> Self {
        Self::miter()
    }
    #[staticmethod]
    #[allow(non_snake_case)]
    fn ROUND() -> Self {
        Self::round()
    }
    #[staticmethod]
    #[allow(non_snake_case)]
    fn BEVEL() -> Self {
        Self::bevel()
    }
}

#[pyclass(module = "pdf_oxide.pdf_oxide", name = "PatternPresets")]
#[derive(Clone)]
pub struct PyPatternPresets;
#[pymethods]
impl PyPatternPresets {
    #[staticmethod]
    fn horizontal_stripes(width: f32, height: f32, stripe_height: f32, color: &PyColor) -> Vec<u8> {
        crate::writer::PatternPresets::horizontal_stripes(width, height, stripe_height, color.inner)
    }
    #[staticmethod]
    fn vertical_stripes(width: f32, height: f32, stripe_width: f32, color: &PyColor) -> Vec<u8> {
        crate::writer::PatternPresets::vertical_stripes(width, height, stripe_width, color.inner)
    }
    #[staticmethod]
    fn checkerboard(size: f32, color1: &PyColor, color2: &PyColor) -> Vec<u8> {
        crate::writer::PatternPresets::checkerboard(size, color1.inner, color2.inner)
    }
    #[staticmethod]
    fn dots(spacing: f32, radius: f32, color: &PyColor) -> Vec<u8> {
        crate::writer::PatternPresets::dots(spacing, radius, color.inner)
    }
    #[staticmethod]
    fn diagonal_lines(size: f32, line_width: f32, color: &PyColor) -> Vec<u8> {
        crate::writer::PatternPresets::diagonal_lines(size, line_width, color.inner)
    }
    #[staticmethod]
    fn crosshatch(size: f32, line_width: f32, color: &PyColor) -> Vec<u8> {
        crate::writer::PatternPresets::crosshatch(size, line_width, color.inner)
    }
}

#[pyclass(module = "pdf_oxide.pdf_oxide", name = "ArtifactStyle")]
#[derive(Clone)]
pub struct PyArtifactStyle {
    pub inner: crate::writer::ArtifactStyle,
}
#[pymethods]
impl PyArtifactStyle {
    #[new]
    fn new() -> Self {
        Self {
            inner: crate::writer::ArtifactStyle::default(),
        }
    }
    fn font<'a>(
        mut slf: PyRefMut<'a, Self>,
        name: &str,
        size: f32,
    ) -> PyResult<PyRefMut<'a, Self>> {
        slf.inner = slf.inner.clone().font(name, size);
        Ok(slf)
    }
    fn bold<'a>(mut slf: PyRefMut<'a, Self>) -> PyResult<PyRefMut<'a, Self>> {
        slf.inner = slf.inner.clone().bold();
        Ok(slf)
    }
}

#[pyclass(module = "pdf_oxide.pdf_oxide", name = "Artifact")]
#[derive(Clone)]
pub struct PyArtifact {
    pub inner: crate::writer::Artifact,
}
#[pymethods]
impl PyArtifact {
    #[new]
    fn new() -> Self {
        Self {
            inner: crate::writer::Artifact::new(),
        }
    }
    #[staticmethod]
    fn center(t: &str) -> Self {
        Self {
            inner: crate::writer::Artifact::center(t),
        }
    }
    fn with_left<'a>(mut slf: PyRefMut<'a, Self>, t: &str) -> PyResult<PyRefMut<'a, Self>> {
        slf.inner = slf.inner.clone().with_left(t);
        Ok(slf)
    }
}

#[pyclass(module = "pdf_oxide.pdf_oxide", name = "Header")]
#[derive(Clone)]
pub struct PyHeader {
    pub inner: PyArtifact,
}
#[pymethods]
impl PyHeader {
    #[new]
    fn new() -> Self {
        Self {
            inner: PyArtifact {
                inner: crate::writer::Artifact::new(),
            },
        }
    }
    #[staticmethod]
    fn center(t: &str) -> Self {
        Self {
            inner: PyArtifact {
                inner: crate::writer::Artifact::center(t),
            },
        }
    }
}

#[pyclass(module = "pdf_oxide.pdf_oxide", name = "Footer")]
#[derive(Clone)]
pub struct PyFooter {
    pub inner: PyArtifact,
}
#[pymethods]
impl PyFooter {
    #[new]
    fn new() -> Self {
        Self {
            inner: PyArtifact {
                inner: crate::writer::Artifact::new(),
            },
        }
    }
    #[staticmethod]
    fn center(t: &str) -> Self {
        Self {
            inner: PyArtifact {
                inner: crate::writer::Artifact::center(t),
            },
        }
    }
}

#[pyclass(module = "pdf_oxide.pdf_oxide", name = "PageTemplate")]
#[derive(Clone)]
pub struct PyPageTemplate {
    pub inner: crate::writer::PageTemplate,
}
#[pymethods]
impl PyPageTemplate {
    #[new]
    fn new() -> Self {
        Self {
            inner: crate::writer::PageTemplate::new(),
        }
    }
    fn header<'a>(
        mut slf: PyRefMut<'a, Self>,
        h: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        let a = if let Ok(ph) = h.extract::<PyHeader>() {
            ph.inner.inner.clone()
        } else {
            h.extract::<PyArtifact>()?.inner.clone()
        };
        slf.inner = slf.inner.clone().header(a);
        Ok(slf)
    }
    fn footer<'a>(
        mut slf: PyRefMut<'a, Self>,
        f: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        let a = if let Ok(pf) = f.extract::<PyFooter>() {
            pf.inner.inner.clone()
        } else {
            f.extract::<PyArtifact>()?.inner.clone()
        };
        slf.inner = slf.inner.clone().footer(a);
        Ok(slf)
    }
}

#[pymodule]
fn pdf_oxide(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPdfDocument>()?;
    m.add_class::<PyPdf>()?;
    m.add_class::<PyPdfPage>()?;
    m.add_class::<PyPdfText>()?;
    m.add_class::<PyPdfTextId>()?;
    m.add_class::<PyPdfImage>()?;
    m.add_class::<PyPdfElement>()?;
    m.add_class::<PyAnnotationWrapper>()?;
    m.add_class::<PyTextChar>()?;
    m.add_class::<PyTextSpan>()?;
    m.add_class::<PyWord>()?;
    m.add_class::<PyTextLine>()?;
    m.add_class::<PyPdfPageRegion>()?;
    m.add_class::<PyFormField>()?;
    m.add_class::<PyOcrEngine>()?;
    m.add_class::<PyOcrConfig>()?;
    m.add_class::<PyColor>()?;
    m.add_class::<PyBlendMode>()?;
    m.add_class::<PyExtGState>()?;
    m.add_class::<PyPageTemplate>()?;
    m.add_class::<PyArtifact>()?;
    m.add_class::<PyHeader>()?;
    m.add_class::<PyFooter>()?;
    m.add_class::<PyArtifactStyle>()?;
    m.add_class::<PyLinearGradient>()?;
    m.add_class::<PyRadialGradient>()?;
    m.add_class::<PyLineCap>()?;
    m.add_class::<PyLineJoin>()?;
    m.add_class::<PyPatternPresets>()?;
    m.add_class::<PyOfficeConverter>()?;
    m.add("VERSION", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
