//! Text and content extraction from PDF documents.
//!
//! Phase 4 (text), Phase 5 (images), and Phase 5.1 (gap analysis)

pub mod forms;
pub mod gap_statistics;
pub mod geometric_spacing;
pub mod images;
pub mod structured;
pub mod table_detector;
pub mod text;

#[cfg(feature = "debug-span-merging")]
pub mod debug_span_merging;

pub use forms::{FieldType, FieldValue, FormExtractor, FormField};
pub use gap_statistics::{
    AdaptiveThresholdConfig, AdaptiveThresholdResult, DocumentProfile, GapStatistics,
    analyze_document_gaps, calculate_statistics, determine_adaptive_threshold, extract_gaps,
};
pub use geometric_spacing::{SpaceInsertion, SpacingConfig, should_insert_space};
pub use images::{ColorSpace, ImageData, PdfImage, PixelFormat, extract_image_from_xobject};
pub use structured::{
    BoundingBox, DocumentElement, DocumentMetadata, ExtractorConfig, ListItem, StructuredDocument,
    StructuredExtractor, TextAlignment, TextStyle,
};
pub use table_detector::{DetectedTable, TableDetector, TableDetectorConfig};
pub use text::{SpanMergingConfig, TextExtractionConfig, TextExtractor};
