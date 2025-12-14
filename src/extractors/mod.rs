//! Text and content extraction from PDF documents.
//!
//! Provides high-performance extraction of text, images, and layout analysis.

pub mod ccitt_bilevel;
pub mod forms;
pub mod gap_statistics;
pub mod geometric_spacing;
pub mod images;
pub mod pattern_detector;
pub mod structured;
pub mod text;

#[cfg(feature = "debug-span-merging")]
pub mod debug_span_merging;

pub use forms::{FieldType, FieldValue, FormExtractor, FormField};
pub use gap_statistics::{
    analyze_document_gaps, calculate_statistics, determine_adaptive_threshold, extract_gaps,
    AdaptiveThresholdConfig, AdaptiveThresholdResult, GapStatistics,
};
pub use geometric_spacing::{should_insert_space, SpaceInsertion, SpacingConfig};
pub use images::{extract_image_from_xobject, ColorSpace, ImageData, PdfImage, PixelFormat};
pub use pattern_detector::{PatternDetector, PatternPreservationConfig};
pub use structured::{
    BoundingBox, DocumentElement, DocumentMetadata, ExtractorConfig, ListItem, StructuredDocument,
    StructuredExtractor, TextAlignment, TextStyle,
};
pub use text::{SpanMergingConfig, TextExtractionConfig, TextExtractor};
