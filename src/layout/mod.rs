//! Layout analysis algorithms for PDF documents.
//!
//! This module provides algorithms for analyzing document layout:
//! - DBSCAN clustering (characters → words → lines)
//! - Reading order determination
//! - Font clustering and normalization

pub mod clustering;
pub mod document_analyzer;
pub mod reading_order;
pub mod text_block;

// Phase 2: Core architectural components
pub mod bold_validation;
pub mod font_normalization;

// Re-export main types
pub use document_analyzer::{AdaptiveLayoutParams, DocumentProperties};
pub use reading_order::graph_based_reading_order;
pub use text_block::{Color, FontWeight, TextBlock, TextChar, TextSpan};

// Re-export Phase 2 components
pub use bold_validation::{BoldGroup, BoldMarkerDecision, BoldMarkerValidator};
pub use font_normalization::{FontWeightNormalizer, NormalizedSpan, SpanType};
