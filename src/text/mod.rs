//! Text processing and analysis module.
//!
//! This module provides tools for working with extracted text from PDF documents,
//! including word boundary detection per ISO 32000-1:2008 specification.

pub mod justification;
pub mod word_boundary;

pub use justification::{JustificationDetector, JustificationMode};
pub use word_boundary::{
    BoundaryContext, CharacterInfo, WordBoundaryDetector, detect_word_boundaries,
};
