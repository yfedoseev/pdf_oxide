//! High-level document builder with fluent API.
//!
//! Provides a convenient interface for building PDF documents
//! using method chaining, wrapping the lower-level PdfWriter.

use super::font_manager::TextLayout;
use super::pdf_writer::{PdfWriter, PdfWriterConfig};
use crate::elements::{ContentElement, TextContent};
use crate::error::Result;
use crate::geometry::Rect;
use std::path::Path;

/// Metadata for a PDF document.
#[derive(Debug, Clone, Default)]
pub struct DocumentMetadata {
    /// Document title
    pub title: Option<String>,
    /// Document author
    pub author: Option<String>,
    /// Document subject
    pub subject: Option<String>,
    /// Document keywords
    pub keywords: Option<String>,
    /// Creator application
    pub creator: Option<String>,
    /// PDF version (default: "1.7")
    pub version: Option<String>,
}

impl DocumentMetadata {
    /// Create new empty metadata.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set document title.
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set document author.
    pub fn author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Set document subject.
    pub fn subject(mut self, subject: impl Into<String>) -> Self {
        self.subject = Some(subject.into());
        self
    }

    /// Set document keywords.
    pub fn keywords(mut self, keywords: impl Into<String>) -> Self {
        self.keywords = Some(keywords.into());
        self
    }

    /// Set creator application.
    pub fn creator(mut self, creator: impl Into<String>) -> Self {
        self.creator = Some(creator.into());
        self
    }
}

/// Standard page sizes.
#[derive(Debug, Clone, Copy)]
pub enum PageSize {
    /// US Letter (8.5" x 11")
    Letter,
    /// A4 (210mm x 297mm)
    A4,
    /// Legal (8.5" x 14")
    Legal,
    /// A3 (297mm x 420mm)
    A3,
    /// Custom dimensions in points
    Custom(f32, f32),
}

impl PageSize {
    /// Get dimensions in points (1 inch = 72 points).
    pub fn dimensions(&self) -> (f32, f32) {
        match self {
            PageSize::Letter => (612.0, 792.0),
            PageSize::A4 => (595.0, 842.0),
            PageSize::Legal => (612.0, 1008.0),
            PageSize::A3 => (842.0, 1190.0),
            PageSize::Custom(w, h) => (*w, *h),
        }
    }
}

/// Text alignment options.
#[derive(Debug, Clone, Copy, Default)]
pub enum TextAlign {
    /// Left-aligned text (default)
    #[default]
    Left,
    /// Center-aligned text
    Center,
    /// Right-aligned text
    Right,
}

/// Configuration for text rendering.
#[derive(Debug, Clone)]
pub struct TextConfig {
    /// Font name (default: Helvetica)
    pub font: String,
    /// Font size in points (default: 12)
    pub size: f32,
    /// Text alignment
    pub align: TextAlign,
    /// Line height multiplier (default: 1.2)
    pub line_height: f32,
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            font: "Helvetica".to_string(),
            size: 12.0,
            align: TextAlign::Left,
            line_height: 1.2,
        }
    }
}

/// Page builder for adding content to a page with fluent API.
pub struct FluentPageBuilder<'a> {
    builder: &'a mut DocumentBuilder,
    page_index: usize,
    cursor_x: f32,
    cursor_y: f32,
    text_config: TextConfig,
    text_layout: TextLayout,
}

impl<'a> FluentPageBuilder<'a> {
    /// Set the text configuration for subsequent text operations.
    pub fn text_config(mut self, config: TextConfig) -> Self {
        self.text_config = config;
        self
    }

    /// Set font for subsequent text operations.
    pub fn font(mut self, name: &str, size: f32) -> Self {
        self.text_config.font = name.to_string();
        self.text_config.size = size;
        self
    }

    /// Set cursor position for text placement.
    pub fn at(mut self, x: f32, y: f32) -> Self {
        self.cursor_x = x;
        self.cursor_y = y;
        self
    }

    /// Add text at the current cursor position.
    pub fn text(mut self, text: &str) -> Self {
        let text_width = self.text_layout.font_manager().text_width(
            text,
            &self.text_config.font,
            self.text_config.size,
        );

        let page = &mut self.builder.pages[self.page_index];
        page.elements.push(ContentElement::Text(TextContent {
            text: text.to_string(),
            bbox: Rect::new(self.cursor_x, self.cursor_y, text_width, self.text_config.size),
            font: crate::elements::FontSpec {
                name: self.text_config.font.clone(),
                size: self.text_config.size,
            },
            style: Default::default(),
            reading_order: Some(page.elements.len()),
        }));
        // Move cursor down for next line
        self.cursor_y -= self.text_config.size * self.text_config.line_height;
        self
    }

    /// Add a heading (larger, bold text).
    pub fn heading(self, level: u8, text: &str) -> Self {
        let size = match level {
            1 => 24.0,
            2 => 20.0,
            3 => 16.0,
            _ => 14.0,
        };
        let font = match level {
            1 | 2 => "Helvetica-Bold",
            _ => "Helvetica",
        };
        self.font(font, size).text(text)
    }

    /// Add a paragraph of text with automatic word wrapping.
    pub fn paragraph(mut self, text: &str) -> Self {
        // Use FontManager-based word wrapping for accurate metrics
        let page = &mut self.builder.pages[self.page_index];
        let max_width = page.width - self.cursor_x - 72.0; // 72pt right margin

        let lines = self.text_layout.wrap_text(
            text,
            &self.text_config.font,
            self.text_config.size,
            max_width,
        );

        for (line_text, line_width) in lines {
            let page = &mut self.builder.pages[self.page_index];
            page.elements.push(ContentElement::Text(TextContent {
                text: line_text,
                bbox: Rect::new(self.cursor_x, self.cursor_y, line_width, self.text_config.size),
                font: crate::elements::FontSpec {
                    name: self.text_config.font.clone(),
                    size: self.text_config.size,
                },
                style: Default::default(),
                reading_order: Some(page.elements.len()),
            }));
            self.cursor_y -= self.text_config.size * self.text_config.line_height;
        }
        // Add extra space after paragraph
        self.cursor_y -= self.text_config.size * 0.5;
        self
    }

    /// Add vertical space.
    pub fn space(mut self, points: f32) -> Self {
        self.cursor_y -= points;
        self
    }

    /// Add a horizontal line.
    pub fn horizontal_rule(mut self) -> Self {
        let page = &mut self.builder.pages[self.page_index];
        let line_y = self.cursor_y + self.text_config.size * 0.5;
        page.elements
            .push(ContentElement::Path(crate::elements::PathContent {
                operations: vec![
                    crate::elements::PathOperation::MoveTo(self.cursor_x, line_y),
                    crate::elements::PathOperation::LineTo(page.width - 72.0, line_y),
                ],
                bbox: Rect::new(self.cursor_x, line_y, page.width - 72.0 - self.cursor_x, 1.0),
                stroke_color: Some(crate::layout::Color {
                    r: 0.5,
                    g: 0.5,
                    b: 0.5,
                }),
                fill_color: None,
                stroke_width: 0.5,
                line_cap: crate::elements::LineCap::Butt,
                line_join: crate::elements::LineJoin::Miter,
                reading_order: None,
            }));
        self.cursor_y -= self.text_config.size;
        self
    }

    /// Add a content element directly.
    pub fn element(self, element: ContentElement) -> Self {
        let page = &mut self.builder.pages[self.page_index];
        page.elements.push(element);
        self
    }

    /// Add multiple content elements.
    pub fn elements(self, elements: Vec<ContentElement>) -> Self {
        let page = &mut self.builder.pages[self.page_index];
        page.elements.extend(elements);
        self
    }

    /// Finish building this page and return to the document builder.
    pub fn done(self) -> &'a mut DocumentBuilder {
        self.builder
    }
}

/// Internal page data for DocumentBuilder.
struct PageData {
    width: f32,
    height: f32,
    elements: Vec<ContentElement>,
}

/// High-level document builder with fluent API.
///
/// Provides a convenient way to build PDF documents using method chaining.
///
/// # Example
///
/// ```ignore
/// use pdf_oxide::writer::{DocumentBuilder, PageSize, DocumentMetadata};
///
/// let pdf_bytes = DocumentBuilder::new()
///     .metadata(DocumentMetadata::new().title("My Document"))
///     .page(PageSize::Letter)
///         .at(72.0, 720.0)
///         .heading(1, "Hello, World!")
///         .paragraph("This is a simple PDF document.")
///         .done()
///     .build()?;
/// ```
pub struct DocumentBuilder {
    metadata: DocumentMetadata,
    pages: Vec<PageData>,
}

impl DocumentBuilder {
    /// Create a new document builder.
    pub fn new() -> Self {
        Self {
            metadata: DocumentMetadata::default(),
            pages: Vec::new(),
        }
    }

    /// Set document metadata.
    pub fn metadata(mut self, metadata: DocumentMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Add a page with the specified size and return a page builder.
    pub fn page(&mut self, size: PageSize) -> FluentPageBuilder {
        let (width, height) = size.dimensions();
        let page_index = self.pages.len();
        self.pages.push(PageData {
            width,
            height,
            elements: Vec::new(),
        });
        FluentPageBuilder {
            builder: self,
            page_index,
            cursor_x: 72.0,          // 1 inch margin
            cursor_y: height - 72.0, // Start from top with 1 inch margin
            text_config: TextConfig::default(),
            text_layout: TextLayout::new(),
        }
    }

    /// Add a Letter-sized page.
    pub fn letter_page(&mut self) -> FluentPageBuilder {
        self.page(PageSize::Letter)
    }

    /// Add an A4-sized page.
    pub fn a4_page(&mut self) -> FluentPageBuilder {
        self.page(PageSize::A4)
    }

    /// Build the PDF document and return the bytes.
    pub fn build(self) -> Result<Vec<u8>> {
        let mut config = PdfWriterConfig::default();
        if let Some(version) = self.metadata.version {
            config.version = version;
        }
        config.title = self.metadata.title;
        config.author = self.metadata.author;
        config.subject = self.metadata.subject;
        config.keywords = self.metadata.keywords;
        if self.metadata.creator.is_some() {
            config.creator = self.metadata.creator;
        }

        let mut writer = PdfWriter::with_config(config);

        for page_data in &self.pages {
            let mut page = writer.add_page(page_data.width, page_data.height);
            page.add_elements(&page_data.elements);
            page.finish();
        }

        writer.finish()
    }

    /// Build and save the PDF to a file.
    pub fn save(self, path: impl AsRef<Path>) -> Result<()> {
        let bytes = self.build()?;
        std::fs::write(path, bytes)?;
        Ok(())
    }
}

impl Default for DocumentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple word wrapping utility.
fn wrap_text(text: &str, max_chars: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current_line = String::new();

    for word in text.split_whitespace() {
        if current_line.is_empty() {
            current_line = word.to_string();
        } else if current_line.len() + 1 + word.len() <= max_chars {
            current_line.push(' ');
            current_line.push_str(word);
        } else {
            lines.push(current_line);
            current_line = word.to_string();
        }
    }

    if !current_line.is_empty() {
        lines.push(current_line);
    }

    if lines.is_empty() {
        lines.push(String::new());
    }

    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_size_dimensions() {
        assert_eq!(PageSize::Letter.dimensions(), (612.0, 792.0));
        assert_eq!(PageSize::A4.dimensions(), (595.0, 842.0));
        assert_eq!(PageSize::Legal.dimensions(), (612.0, 1008.0));
        assert_eq!(PageSize::Custom(100.0, 200.0).dimensions(), (100.0, 200.0));
    }

    #[test]
    fn test_document_metadata() {
        let meta = DocumentMetadata::new()
            .title("Test Title")
            .author("Test Author")
            .subject("Test Subject");

        assert_eq!(meta.title, Some("Test Title".to_string()));
        assert_eq!(meta.author, Some("Test Author".to_string()));
        assert_eq!(meta.subject, Some("Test Subject".to_string()));
    }

    #[test]
    fn test_wrap_text() {
        let text = "This is a test of the word wrapping function";
        let wrapped = wrap_text(text, 20);
        assert!(wrapped.len() > 1);
        for line in &wrapped {
            assert!(line.len() <= 20 || line.split_whitespace().count() == 1);
        }
    }

    #[test]
    fn test_wrap_text_empty() {
        let wrapped = wrap_text("", 20);
        assert_eq!(wrapped.len(), 1);
        assert_eq!(wrapped[0], "");
    }

    #[test]
    fn test_document_builder_basic() {
        let mut builder = DocumentBuilder::new();
        builder
            .letter_page()
            .at(72.0, 720.0)
            .text("Hello, World!")
            .done();

        let bytes = builder.build().unwrap();
        let content = String::from_utf8_lossy(&bytes);

        assert!(content.starts_with("%PDF-1.7"));
        assert!(content.contains("%%EOF"));
    }

    #[test]
    fn test_document_builder_with_metadata() {
        let mut builder = DocumentBuilder::new().metadata(
            DocumentMetadata::new()
                .title("Test Document")
                .author("Test Author"),
        );

        builder.letter_page().text("Content").done();

        let bytes = builder.build().unwrap();
        let content = String::from_utf8_lossy(&bytes);

        assert!(content.contains("/Title (Test Document)"));
        assert!(content.contains("/Author (Test Author)"));
    }

    #[test]
    fn test_document_builder_multiple_pages() {
        let mut builder = DocumentBuilder::new();
        builder.letter_page().text("Page 1").done();
        builder.a4_page().text("Page 2").done();

        let bytes = builder.build().unwrap();
        let content = String::from_utf8_lossy(&bytes);

        assert!(content.contains("/Count 2"));
    }

    #[test]
    fn test_fluent_page_builder() {
        let mut builder = DocumentBuilder::new();
        builder
            .letter_page()
            .at(72.0, 720.0)
            .font("Helvetica-Bold", 18.0)
            .text("Title")
            .font("Helvetica", 12.0)
            .text("Body text")
            .space(12.0)
            .text("More text")
            .done();

        let bytes = builder.build().unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_text_config() {
        let config = TextConfig {
            font: "Times-Roman".to_string(),
            size: 14.0,
            align: TextAlign::Center,
            line_height: 1.5,
        };

        assert_eq!(config.font, "Times-Roman");
        assert_eq!(config.size, 14.0);
    }
}
