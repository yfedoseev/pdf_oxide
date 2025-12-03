//! Text and content extraction from PDF documents.
//!
//! Phase 4 (text) and Phase 5 (images)

pub mod forms;
pub mod images;
pub mod structured;
pub mod table_detector;
pub mod text;

pub use forms::{FieldType, FieldValue, FormExtractor, FormField};
pub use images::{ColorSpace, ImageData, PdfImage, PixelFormat, extract_image_from_xobject};
pub use structured::{
    BoundingBox, DocumentElement, DocumentMetadata, ExtractorConfig, ListItem, StructuredDocument,
    StructuredExtractor, TextAlignment, TextStyle,
};
pub use table_detector::{DetectedTable, TableDetector, TableDetectorConfig};
pub use text::{SpanMergingConfig, TextExtractionConfig, TextExtractor};
