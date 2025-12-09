//! PDF document writer.
//!
//! Assembles complete PDF documents with proper structure:
//! header, body, xref table, and trailer.

use super::content_stream::ContentStreamBuilder;
use super::object_serializer::ObjectSerializer;
use crate::elements::ContentElement;
use crate::error::Result;
use crate::object::{Object, ObjectRef};
use std::collections::HashMap;
use std::io::Write;

/// Configuration for PDF generation.
#[derive(Debug, Clone)]
pub struct PdfWriterConfig {
    /// PDF version (e.g., "1.7")
    pub version: String,
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
    /// Whether to compress streams
    pub compress: bool,
}

impl Default for PdfWriterConfig {
    fn default() -> Self {
        Self {
            version: "1.7".to_string(),
            title: None,
            author: None,
            subject: None,
            keywords: None,
            creator: Some("pdf_oxide".to_string()),
            compress: false, // Disable compression for now (requires flate2)
        }
    }
}

impl PdfWriterConfig {
    /// Set document title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set document author.
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Set document subject.
    pub fn with_subject(mut self, subject: impl Into<String>) -> Self {
        self.subject = Some(subject.into());
        self
    }
}

/// A page being built.
pub struct PageBuilder<'a> {
    writer: &'a mut PdfWriter,
    page_index: usize,
}

impl<'a> PageBuilder<'a> {
    /// Add text to the page.
    pub fn add_text(
        &mut self,
        text: &str,
        x: f32,
        y: f32,
        font_name: &str,
        font_size: f32,
    ) -> &mut Self {
        let page = &mut self.writer.pages[self.page_index];
        page.content_builder
            .begin_text()
            .set_font(font_name, font_size)
            .text(text, x, y);
        self
    }

    /// Add a content element to the page.
    pub fn add_element(&mut self, element: &ContentElement) -> &mut Self {
        let page = &mut self.writer.pages[self.page_index];
        page.content_builder.add_element(element);
        self
    }

    /// Add multiple content elements.
    pub fn add_elements(&mut self, elements: &[ContentElement]) -> &mut Self {
        let page = &mut self.writer.pages[self.page_index];
        page.content_builder.add_elements(elements);
        self
    }

    /// Draw a rectangle on the page.
    pub fn draw_rect(&mut self, x: f32, y: f32, width: f32, height: f32) -> &mut Self {
        let page = &mut self.writer.pages[self.page_index];
        page.content_builder.end_text();
        page.content_builder.rect(x, y, width, height).stroke();
        self
    }

    /// Finish building this page and return to the writer.
    pub fn finish(self) -> &'a mut PdfWriter {
        let page = &mut self.writer.pages[self.page_index];
        page.content_builder.end_text();
        self.writer
    }
}

/// Internal page data.
struct PageData {
    width: f32,
    height: f32,
    content_builder: ContentStreamBuilder,
}

/// PDF document writer.
///
/// Builds a complete PDF document with pages, fonts, and content.
pub struct PdfWriter {
    config: PdfWriterConfig,
    pages: Vec<PageData>,
    /// Object ID counter
    next_obj_id: u32,
    /// Allocated objects (id -> object)
    objects: HashMap<u32, Object>,
    /// Font resources used (name -> object ref)
    fonts: HashMap<String, ObjectRef>,
}

impl PdfWriter {
    /// Create a new PDF writer with default config.
    pub fn new() -> Self {
        Self::with_config(PdfWriterConfig::default())
    }

    /// Create a PDF writer with custom config.
    pub fn with_config(config: PdfWriterConfig) -> Self {
        Self {
            config,
            pages: Vec::new(),
            next_obj_id: 1,
            objects: HashMap::new(),
            fonts: HashMap::new(),
        }
    }

    /// Allocate a new object ID.
    fn alloc_obj_id(&mut self) -> u32 {
        let id = self.next_obj_id;
        self.next_obj_id += 1;
        id
    }

    /// Add a page with the given dimensions.
    pub fn add_page(&mut self, width: f32, height: f32) -> PageBuilder {
        let page_index = self.pages.len();
        self.pages.push(PageData {
            width,
            height,
            content_builder: ContentStreamBuilder::new(),
        });
        PageBuilder {
            writer: self,
            page_index,
        }
    }

    /// Add a US Letter sized page (8.5" x 11").
    pub fn add_letter_page(&mut self) -> PageBuilder {
        self.add_page(612.0, 792.0)
    }

    /// Add an A4 sized page (210mm x 297mm).
    pub fn add_a4_page(&mut self) -> PageBuilder {
        self.add_page(595.0, 842.0)
    }

    /// Get a font reference, creating the font object if needed.
    fn get_font_ref(&mut self, font_name: &str) -> ObjectRef {
        if let Some(font_ref) = self.fonts.get(font_name) {
            return *font_ref;
        }

        let font_id = self.alloc_obj_id();
        let font_obj = ObjectSerializer::dict(vec![
            ("Type", ObjectSerializer::name("Font")),
            ("Subtype", ObjectSerializer::name("Type1")),
            ("BaseFont", ObjectSerializer::name(font_name)),
            ("Encoding", ObjectSerializer::name("WinAnsiEncoding")),
        ]);

        self.objects.insert(font_id, font_obj);
        let font_ref = ObjectRef::new(font_id, 0);
        self.fonts.insert(font_name.to_string(), font_ref);
        font_ref
    }

    /// Build the complete PDF document.
    pub fn finish(mut self) -> Result<Vec<u8>> {
        let serializer = ObjectSerializer::compact();
        let mut output = Vec::new();
        let mut xref_offsets: Vec<(u32, usize)> = Vec::new();

        // PDF Header
        writeln!(output, "%PDF-{}", self.config.version)?;
        // Binary marker (recommended for binary content)
        output.extend_from_slice(b"%\xE2\xE3\xCF\xD3\n");

        // Collect all fonts used across pages
        let font_names: Vec<String> = vec![
            "Helvetica".to_string(),
            "Helvetica-Bold".to_string(),
            "Times-Roman".to_string(),
            "Times-Bold".to_string(),
            "Courier".to_string(),
            "Courier-Bold".to_string(),
        ];

        for font_name in &font_names {
            self.get_font_ref(font_name);
        }

        // Build font resources dictionary
        let font_resources: HashMap<String, Object> = self
            .fonts
            .iter()
            .map(|(name, obj_ref)| {
                // Use simple names like F1, F2, etc. for the resource dict
                // but map from the actual font name
                let simple_name = name.replace('-', "");
                (simple_name, Object::Reference(*obj_ref))
            })
            .collect();

        // Catalog object (object 1)
        let catalog_id = self.alloc_obj_id();
        let pages_id = self.alloc_obj_id();

        // Pre-allocate object IDs for all pages
        let page_count = self.pages.len();
        let mut page_ids: Vec<(u32, u32)> = Vec::with_capacity(page_count);
        for _ in 0..page_count {
            let page_id = self.alloc_obj_id();
            let content_id = self.alloc_obj_id();
            page_ids.push((page_id, content_id));
        }

        // Create page objects
        let mut page_refs: Vec<Object> = Vec::new();
        let mut page_objects: Vec<(u32, Object, Vec<u8>)> = Vec::new();

        for (i, page_data) in self.pages.iter().enumerate() {
            let (page_id, content_id) = page_ids[i];

            // Build content stream
            let content_bytes = page_data.content_builder.build()?;

            // Create content stream object
            let mut content_dict = HashMap::new();
            content_dict.insert("Length".to_string(), Object::Integer(content_bytes.len() as i64));

            // Page object
            let page_obj = ObjectSerializer::dict(vec![
                ("Type", ObjectSerializer::name("Page")),
                ("Parent", ObjectSerializer::reference(pages_id, 0)),
                (
                    "MediaBox",
                    ObjectSerializer::rect(
                        0.0,
                        0.0,
                        page_data.width as f64,
                        page_data.height as f64,
                    ),
                ),
                ("Contents", ObjectSerializer::reference(content_id, 0)),
                (
                    "Resources",
                    ObjectSerializer::dict(vec![(
                        "Font",
                        Object::Dictionary(font_resources.clone()),
                    )]),
                ),
            ]);

            page_refs.push(Object::Reference(ObjectRef::new(page_id, 0)));
            page_objects.push((page_id, page_obj, Vec::new()));
            page_objects.push((
                content_id,
                Object::Stream {
                    dict: content_dict,
                    data: bytes::Bytes::from(content_bytes),
                },
                Vec::new(),
            ));
        }

        // Pages object
        let pages_obj = ObjectSerializer::dict(vec![
            ("Type", ObjectSerializer::name("Pages")),
            ("Kids", Object::Array(page_refs)),
            ("Count", ObjectSerializer::integer(self.pages.len() as i64)),
        ]);

        // Catalog object
        let catalog_obj = ObjectSerializer::dict(vec![
            ("Type", ObjectSerializer::name("Catalog")),
            ("Pages", ObjectSerializer::reference(pages_id, 0)),
        ]);

        // Info object (optional metadata)
        let info_id = self.alloc_obj_id();
        let mut info_entries = Vec::new();
        if let Some(title) = &self.config.title {
            info_entries.push(("Title", ObjectSerializer::string(title)));
        }
        if let Some(author) = &self.config.author {
            info_entries.push(("Author", ObjectSerializer::string(author)));
        }
        if let Some(subject) = &self.config.subject {
            info_entries.push(("Subject", ObjectSerializer::string(subject)));
        }
        if let Some(creator) = &self.config.creator {
            info_entries.push(("Creator", ObjectSerializer::string(creator)));
        }
        let info_obj = ObjectSerializer::dict(info_entries);

        // Write all objects
        // Catalog
        xref_offsets.push((catalog_id, output.len()));
        output.extend_from_slice(&serializer.serialize_indirect(catalog_id, 0, &catalog_obj));

        // Pages
        xref_offsets.push((pages_id, output.len()));
        output.extend_from_slice(&serializer.serialize_indirect(pages_id, 0, &pages_obj));

        // Font objects
        for (_font_name, font_ref) in &self.fonts {
            if let Some(font_obj) = self.objects.get(&font_ref.id) {
                xref_offsets.push((font_ref.id, output.len()));
                output.extend_from_slice(&serializer.serialize_indirect(font_ref.id, 0, font_obj));
            }
        }

        // Page and content objects
        for (obj_id, obj, _) in &page_objects {
            xref_offsets.push((*obj_id, output.len()));
            output.extend_from_slice(&serializer.serialize_indirect(*obj_id, 0, obj));
        }

        // Info object
        xref_offsets.push((info_id, output.len()));
        output.extend_from_slice(&serializer.serialize_indirect(info_id, 0, &info_obj));

        // Write xref table
        let xref_start = output.len();
        writeln!(output, "xref")?;
        writeln!(output, "0 {}", self.next_obj_id)?;

        // Object 0 is always free
        writeln!(output, "0000000000 65535 f ")?;

        // Sort xref entries by object ID
        xref_offsets.sort_by_key(|(id, _)| *id);

        for (_, offset) in &xref_offsets {
            writeln!(output, "{:010} 00000 n ", offset)?;
        }

        // Write trailer
        let trailer = ObjectSerializer::dict(vec![
            ("Size", ObjectSerializer::integer(self.next_obj_id as i64)),
            ("Root", ObjectSerializer::reference(catalog_id, 0)),
            ("Info", ObjectSerializer::reference(info_id, 0)),
        ]);

        writeln!(output, "trailer")?;
        output.extend_from_slice(&serializer.serialize(&trailer));
        writeln!(output)?;
        writeln!(output, "startxref")?;
        writeln!(output, "{}", xref_start)?;
        write!(output, "%%EOF")?;

        Ok(output)
    }

    /// Save the PDF to a file.
    pub fn save(self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let bytes = self.finish()?;
        std::fs::write(path, bytes)?;
        Ok(())
    }
}

impl Default for PdfWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_empty_pdf() {
        let writer = PdfWriter::new();
        let mut writer = writer;
        writer.add_letter_page().finish();
        let bytes = writer.finish().unwrap();

        let content = String::from_utf8_lossy(&bytes);
        assert!(content.starts_with("%PDF-1.7"));
        assert!(content.contains("/Type /Catalog"));
        assert!(content.contains("/Type /Pages"));
        assert!(content.contains("/Type /Page"));
        assert!(content.contains("%%EOF"));
    }

    #[test]
    fn test_pdf_with_text() {
        let mut writer = PdfWriter::new();
        {
            let mut page = writer.add_letter_page();
            page.add_text("Hello, World!", 72.0, 720.0, "Helvetica", 12.0);
            page.finish();
        }

        let bytes = writer.finish().unwrap();
        let content = String::from_utf8_lossy(&bytes);

        assert!(content.contains("/Type /Font"));
        assert!(content.contains("/BaseFont /Helvetica"));
        assert!(content.contains("BT"));
        assert!(content.contains("(Hello, World!) Tj"));
        assert!(content.contains("ET"));
    }

    #[test]
    fn test_pdf_with_metadata() {
        let config = PdfWriterConfig::default()
            .with_title("Test Document")
            .with_author("Test Author");

        let mut writer = PdfWriter::with_config(config);
        writer.add_letter_page().finish();

        let bytes = writer.finish().unwrap();
        let content = String::from_utf8_lossy(&bytes);

        assert!(content.contains("/Title (Test Document)"));
        assert!(content.contains("/Author (Test Author)"));
    }

    #[test]
    fn test_multiple_pages() {
        let mut writer = PdfWriter::new();
        writer.add_letter_page().finish();
        writer.add_a4_page().finish();

        let bytes = writer.finish().unwrap();
        let content = String::from_utf8_lossy(&bytes);

        assert!(content.contains("/Count 2"));
        // Two MediaBox entries for different page sizes
        assert!(content.contains("[0 0 612 792]")); // Letter
        assert!(content.contains("[0 0 595 842]")); // A4
    }

    #[test]
    fn test_page_builder() {
        let mut writer = PdfWriter::new();
        {
            let mut page = writer.add_letter_page();
            page.add_text("Line 1", 72.0, 720.0, "Helvetica", 12.0);
            page.add_text("Line 2", 72.0, 700.0, "Helvetica", 12.0);
            page.finish();
        }

        let bytes = writer.finish().unwrap();
        assert!(!bytes.is_empty());
    }
}
