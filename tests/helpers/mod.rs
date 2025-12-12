//! Test helpers module for corpus loading and golden file management.
//!
//! This module provides utilities for:
//! - Loading PDFs from test corpus directories
//! - Managing golden files for regression testing
//! - Metadata extraction and comparison
//! - Quality scoring and validation

pub mod corpus_loader;
pub mod golden_file_manager;

pub use corpus_loader::{
    CorpusLoader, PdfMetadata, get_pdf_metadata, list_corpus_pdfs, load_test_pdf,
};
pub use golden_file_manager::{ComparisonResult, ComparisonStatus, GoldenFile, GoldenFileManager};
