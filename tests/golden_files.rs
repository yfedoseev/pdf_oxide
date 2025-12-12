//! Golden file regression tests
//!
//! This test suite validates that text extraction produces consistent results
//! across different PDF categories by comparing against golden files.
//!
//! Test Categories:
//! - Academic papers (arxiv)
//! - Diverse documents
//! - Forms
//! - Government documents
//! - Mixed layouts
//! - Newspapers
//! - Technical documents
//! - Theses
//! - Text-heavy documents
//! - Tables
//!
//! Each test loads PDFs from the corpus, extracts text, and compares against
//! saved golden files. Regressions are detected via hash comparison and
//! character/word count tolerances.

mod helpers;

use helpers::corpus_loader::CorpusLoader;
use helpers::golden_file_manager::GoldenFileManager;
use pdf_oxide::document::PdfDocument;

/// Extract text from a PDF using the standard text extraction pipeline
fn extract_text_from_pdf(
    doc: &mut PdfDocument,
    page: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let text = doc.extract_text(page)?;
    Ok(text)
}

/// Extract text from all pages of a PDF
fn extract_all_pages(doc: &mut PdfDocument) -> Result<String, Box<dyn std::error::Error>> {
    let page_count = doc.page_count()?;
    let mut all_text = String::new();

    for page in 0..page_count {
        let text = extract_text_from_pdf(doc, page)?;
        all_text.push_str(&text);
        if page < page_count - 1 {
            all_text.push('\n');
        }
    }

    Ok(all_text)
}

/// Test helper: Run golden file test for a category
fn test_golden_files_for_category(category: &str, max_pdfs: Option<usize>) {
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let pdfs = loader.list_pdfs(category).expect("Failed to list PDFs");
    if pdfs.is_empty() {
        println!("No PDFs found in category: {}", category);
        return;
    }

    let pdfs_to_test = if let Some(max) = max_pdfs {
        &pdfs[..pdfs.len().min(max)]
    } else {
        &pdfs
    };

    let mut passed = 0;
    let mut failed = 0;
    let mut skipped = 0;

    for pdf_path in pdfs_to_test {
        let filename = pdf_path.file_name().unwrap().to_str().unwrap();

        // Check if golden file exists
        if !manager.has_golden_file(pdf_path) {
            println!("  [SKIP] {}: No golden file", filename);
            skipped += 1;
            continue;
        }

        // Load PDF and extract text
        let mut doc = match PdfDocument::open(pdf_path) {
            Ok(d) => d,
            Err(e) => {
                println!("  [FAIL] {}: Cannot open PDF: {}", filename, e);
                failed += 1;
                continue;
            },
        };

        let extracted = match extract_all_pages(&mut doc) {
            Ok(t) => t,
            Err(e) => {
                println!("  [FAIL] {}: Extraction failed: {}", filename, e);
                failed += 1;
                continue;
            },
        };

        // Load golden file
        let golden = match manager.load_golden_file(pdf_path) {
            Ok(g) => g,
            Err(e) => {
                println!("  [FAIL] {}: Cannot load golden file: {}", filename, e);
                failed += 1;
                continue;
            },
        };

        // Compare
        let result = manager.compare_extraction(&extracted, &golden);

        if result.passes() {
            println!("  [PASS] {}", filename);
            passed += 1;
        } else {
            println!("  [FAIL] {}: {}", filename, result.details());
            failed += 1;
        }
    }

    println!(
        "\n{} Summary: {} passed, {} failed, {} skipped (total: {})",
        category,
        passed,
        failed,
        skipped,
        pdfs_to_test.len()
    );

    // Fail the test if any PDFs failed (but allow skipped)
    if failed > 0 {
        panic!(
            "Golden file regression detected in category '{}': {} failures",
            category, failed
        );
    }
}

#[test]
fn test_golden_file_academic_papers() {
    test_golden_files_for_category("academic", Some(10));
}

#[test]
fn test_golden_file_diverse_docs() {
    test_golden_files_for_category("diverse", Some(10));
}

#[test]
fn test_golden_file_forms() {
    test_golden_files_for_category("forms", Some(10));
}

#[test]
fn test_golden_file_government_docs() {
    test_golden_files_for_category("government", Some(10));
}

#[test]
fn test_golden_file_mixed_layouts() {
    test_golden_files_for_category("mixed", Some(10));
}

#[test]
fn test_golden_file_newspapers() {
    test_golden_files_for_category("newspapers", Some(10));
}

#[test]
fn test_golden_file_technical_docs() {
    test_golden_files_for_category("technical", Some(10));
}

#[test]
fn test_golden_file_theses() {
    test_golden_files_for_category("theses", Some(10));
}

#[test]
fn test_golden_file_text_heavy() {
    test_golden_files_for_category("text_heavy", Some(10));
}

#[test]
fn test_golden_file_tables() {
    test_golden_files_for_category("tables", Some(10));
}

/// Integration test: Create golden files for a sample of PDFs
/// This test is marked with #[ignore] so it only runs when explicitly requested
#[test]
#[ignore]
fn create_golden_files_sample() {
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let categories = vec!["academic", "diverse", "forms", "government", "mixed"];

    for category in categories {
        println!("\nCreating golden files for category: {}", category);

        let pdfs = loader.list_pdfs(category).expect("Failed to list PDFs");
        let sample_size = pdfs.len().min(5); // First 5 PDFs per category

        for pdf_path in &pdfs[..sample_size] {
            let filename = pdf_path.file_name().unwrap().to_str().unwrap();

            // Load PDF and extract text
            let mut doc = match PdfDocument::open(pdf_path) {
                Ok(d) => d,
                Err(e) => {
                    println!("  [SKIP] {}: Cannot open PDF: {}", filename, e);
                    continue;
                },
            };

            let extracted = match extract_all_pages(&mut doc) {
                Ok(t) => t,
                Err(e) => {
                    println!("  [SKIP] {}: Extraction failed: {}", filename, e);
                    continue;
                },
            };

            // Save golden file
            match manager.save_golden_file(pdf_path, category, &extracted) {
                Ok(_) => println!("  [SAVED] {}", filename),
                Err(e) => println!("  [ERROR] {}: {}", filename, e),
            }
        }
    }

    println!("\nGolden file creation complete!");
}

/// Utility test: List available PDFs by category
#[test]
#[ignore]
fn list_corpus_summary() {
    let loader = CorpusLoader::default();

    println!("\nTest Corpus Summary:");
    println!("{:-<50}", "");

    let categories = loader.list_categories().expect("Failed to list categories");

    let mut total = 0;
    for category in categories {
        let pdfs = loader.list_pdfs(&category).expect("Failed to list PDFs");
        let count = pdfs.len();
        total += count;
        println!("{:20} : {:5} PDFs", category, count);
    }

    println!("{:-<50}", "");
    println!("{:20} : {:5} PDFs", "TOTAL", total);
}
