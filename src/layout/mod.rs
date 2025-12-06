//! Layout analysis algorithms for PDF documents.
//!
//! This module provides sophisticated algorithms for analyzing document layout:
//! - DBSCAN clustering (characters → words → lines)
//! - XY-Cut algorithm for column detection
//! - Reading order determination
//! - Font clustering and heading detection
//! - Basic table detection

pub mod clustering;
pub mod column_detector;
pub mod document_analyzer;
pub mod heading_detector;
pub mod reading_order;
pub mod table_detector;
pub mod text_block;

// Phase 2: Core architectural components
pub mod bold_validation;
pub mod font_normalization;

// Re-export main types
pub use column_detector::{CutDirection, LayoutTree, xy_cut, xy_cut_adaptive};
pub use document_analyzer::{AdaptiveLayoutParams, DocumentProperties};
pub use heading_detector::{HeadingLevel, detect_headings};
pub use reading_order::{determine_reading_order, graph_based_reading_order};
pub use table_detector::{Table, detect_tables, detect_tables_aggressive};
pub use text_block::{Color, FontWeight, TextBlock, TextChar, TextSpan};

// Re-export Phase 2 components
pub use bold_validation::{BoldGroup, BoldMarkerDecision, BoldMarkerValidator};
pub use font_normalization::{FontWeightNormalizer, NormalizedSpan, SpanType};
