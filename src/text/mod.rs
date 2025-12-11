//! Text processing and analysis module.
//!
//! This module provides tools for working with extracted text from PDF documents,
//! including word boundary detection per ISO 32000-1:2008 specification.

pub mod document_classifier;
pub mod hyphenation;
pub mod justification;
pub mod word_boundary;

pub use document_classifier::{DocumentClassifier, DocumentStats};
pub use hyphenation::HyphenationHandler;
pub use justification::{JustificationDetector, JustificationMode};
pub use word_boundary::{
    BoundaryContext, CharacterInfo, WordBoundaryDetector, detect_word_boundaries,
};
