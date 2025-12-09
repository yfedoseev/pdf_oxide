//! PDF writing module for generating PDF files.
//!
//! This module provides components for creating PDF files from ContentElements.
//!
//! ## Architecture
//!
//! ```text
//! ContentElement[]
//!     ↓
//! [DocumentBuilder] (high-level fluent API)
//!     ↓
//! [ContentStreamBuilder] (elements → content stream bytes)
//!     ↓
//! [PdfWriter] (assembles complete PDF structure)
//!     ↓
//! [ObjectSerializer] (serializes PDF objects)
//!     ↓
//! PDF bytes
//! ```
//!
//! ## High-Level API (DocumentBuilder)
//!
//! ```ignore
//! use pdf_oxide::writer::{DocumentBuilder, PageSize, DocumentMetadata};
//!
//! let bytes = DocumentBuilder::new()
//!     .metadata(DocumentMetadata::new().title("My Document"))
//!     .page(PageSize::Letter)
//!         .at(72.0, 720.0)
//!         .heading(1, "Hello, World!")
//!         .paragraph("This is a PDF document.")
//!         .done()
//!     .build()?;
//! ```
//!
//! ## Low-Level API (PdfWriter)
//!
//! ```ignore
//! use pdf_oxide::writer::{PdfWriter, ContentStreamBuilder};
//!
//! let mut writer = PdfWriter::new();
//! let page = writer.add_page(612.0, 792.0);
//! page.add_text("Hello, World!", 72.0, 720.0);
//! let bytes = writer.finish()?;
//! ```

mod content_stream;
mod document_builder;
mod font_manager;
mod object_serializer;
mod pdf_writer;

pub use content_stream::{ContentStreamBuilder, ContentStreamOp};
pub use document_builder::{
    DocumentBuilder, DocumentMetadata, FluentPageBuilder, PageSize, TextAlign, TextConfig,
};
pub use font_manager::{FontFamily, FontInfo, FontManager, FontWeight, TextLayout};
pub use object_serializer::ObjectSerializer;
pub use pdf_writer::{PageBuilder, PdfWriter, PdfWriterConfig};

use crate::elements::ContentElement;
use crate::error::Result;

/// Trait for building content streams from elements.
///
/// Content streams contain the PDF operators that render content on a page.
pub trait ContentBuilder: Send + Sync {
    /// Build a content stream from elements.
    fn build(&self, elements: &[ContentElement]) -> Result<Vec<u8>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify key types are exported
        let _serializer = ObjectSerializer::new();
        let _builder = ContentStreamBuilder::new();
    }
}
