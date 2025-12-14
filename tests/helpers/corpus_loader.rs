#![allow(dead_code)]
//! Corpus loader for loading PDFs from test directories.
//!
//! Provides utilities to:
//! - List available PDFs by category
//! - Load PDFs from the corpus
//! - Extract metadata (file size, page count, etc.)

use pdf_oxide::document::PdfDocument;
use pdf_oxide::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Default test corpus directory
pub const DEFAULT_CORPUS_DIR: &str = "/home/yfedoseev/projects/pdf_oxide_tests/pdfs";

/// PDF metadata for corpus tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdfMetadata {
    /// PDF file path
    pub path: PathBuf,
    /// Category (academic, multilingual, etc.)
    pub category: String,
    /// File size in bytes
    pub file_size: u64,
    /// Number of pages (if successfully parsed)
    pub page_count: Option<usize>,
    /// Character count (if extracted)
    pub char_count: Option<usize>,
    /// Word count (if extracted)
    pub word_count: Option<usize>,
    /// Script distribution (if analyzed)
    pub script_distribution: Option<HashMap<String, f32>>,
    /// SHA-256 hash of file
    pub file_hash: Option<String>,
}

/// Corpus loader for accessing test PDFs
pub struct CorpusLoader {
    corpus_dir: PathBuf,
}

impl CorpusLoader {
    /// Create a new corpus loader with the specified directory
    pub fn new(corpus_dir: impl AsRef<Path>) -> Self {
        CorpusLoader {
            corpus_dir: corpus_dir.as_ref().to_path_buf(),
        }
    }

    /// Create a corpus loader with the default test directory
    pub fn default() -> Self {
        CorpusLoader::new(DEFAULT_CORPUS_DIR)
    }

    /// List all available categories
    pub fn list_categories(&self) -> Result<Vec<String>> {
        let mut categories = Vec::new();

        if !self.corpus_dir.exists() {
            return Ok(categories);
        }

        for entry in fs::read_dir(&self.corpus_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                if let Some(name) = path.file_name() {
                    if let Some(name_str) = name.to_str() {
                        categories.push(name_str.to_string());
                    }
                }
            }
        }

        categories.sort();
        Ok(categories)
    }

    /// List all PDFs in a specific category
    pub fn list_pdfs(&self, category: &str) -> Result<Vec<PathBuf>> {
        let category_path = self.corpus_dir.join(category);
        let mut pdfs = Vec::new();

        if !category_path.exists() {
            return Ok(pdfs);
        }

        for entry in fs::read_dir(&category_path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == "pdf" || ext == "PDF" {
                        pdfs.push(path);
                    }
                }
            }
        }

        pdfs.sort();
        Ok(pdfs)
    }

    /// Load a PDF document from the corpus
    pub fn load_pdf(&self, category: &str, filename: &str) -> Result<PdfDocument> {
        let path = self.corpus_dir.join(category).join(filename);
        PdfDocument::open(&path)
    }

    /// Get metadata for a PDF file
    pub fn get_metadata(&self, path: &Path) -> Result<PdfMetadata> {
        let metadata = fs::metadata(path)?;
        let file_size = metadata.len();

        // Extract category from path
        let category = path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Try to open and get page count
        let page_count = if let Ok(mut doc) = PdfDocument::open(path) {
            doc.page_count().ok()
        } else {
            None
        };

        Ok(PdfMetadata {
            path: path.to_path_buf(),
            category,
            file_size,
            page_count,
            char_count: None,
            word_count: None,
            script_distribution: None,
            file_hash: None,
        })
    }

    /// Count total PDFs across all categories
    pub fn total_pdf_count(&self) -> Result<usize> {
        let categories = self.list_categories()?;
        let mut total = 0;

        for category in categories {
            total += self.list_pdfs(&category)?.len();
        }

        Ok(total)
    }
}

/// Load a test PDF by category and filename (convenience function)
pub fn load_test_pdf(category: &str, filename: &str) -> Result<PdfDocument> {
    let loader = CorpusLoader::default();
    loader.load_pdf(category, filename)
}

/// List all PDFs in a category (convenience function)
pub fn list_corpus_pdfs(category: &str) -> Result<Vec<PathBuf>> {
    let loader = CorpusLoader::default();
    loader.list_pdfs(category)
}

/// Get metadata for a PDF (convenience function)
pub fn get_pdf_metadata(path: &Path) -> Result<PdfMetadata> {
    let loader = CorpusLoader::default();
    loader.get_metadata(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_corpus_loader_initialization() {
        let loader = CorpusLoader::default();
        assert_eq!(loader.corpus_dir.to_str().unwrap(), DEFAULT_CORPUS_DIR);
    }

    #[test]
    fn test_list_categories() {
        let loader = CorpusLoader::default();
        if let Ok(categories) = loader.list_categories() {
            // If the corpus directory exists, we should have categories
            if !categories.is_empty() {
                assert!(
                    categories.contains(&"academic".to_string())
                        || categories.contains(&"diverse".to_string())
                );
            }
        }
    }

    #[test]
    fn test_list_academic_pdfs() {
        let loader = CorpusLoader::default();
        if let Ok(pdfs) = loader.list_pdfs("academic") {
            // If academic directory exists and has PDFs, they should be sorted
            if pdfs.len() > 1 {
                let first = pdfs[0].file_name().unwrap().to_str().unwrap();
                let second = pdfs[1].file_name().unwrap().to_str().unwrap();
                assert!(first <= second, "PDFs should be sorted");
            }
        }
    }

    #[test]
    fn test_total_count() {
        let loader = CorpusLoader::default();
        if let Ok(count) = loader.total_pdf_count() {
            // We know from the directory listing we have 356 PDFs
            // But this test should work even if the directory doesn't exist
            println!("Total PDFs found: {}", count);
        }
    }
}
