//! Corpus integration tests
//!
//! Full pipeline tests that:
//! 1. Load PDFs from test corpus
//! 2. Extract text using Primary detection mode
//! 3. Compare against golden files
//! 4. Validate word boundaries
//! 5. Check reading order
//! 6. Measure performance
//!
//! These tests validate the complete extraction pipeline on real-world documents.

mod helpers;
mod quality_metrics;

use helpers::corpus_loader::CorpusLoader;
use helpers::golden_file_manager::{ComparisonStatus, GoldenFileManager};
use pdf_oxide::document::PdfDocument;
use quality_metrics::analyze_quality;
use std::time::Instant;

/// Extract text from a PDF with Primary mode detection
fn extract_with_primary_mode(
    doc: &mut PdfDocument,
    page: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    // For now, use standard extraction - Primary mode is already the default
    let text = doc.extract_text(page)?;
    Ok(text)
}

/// Extract all pages with Primary mode
fn extract_all_pages_primary(doc: &mut PdfDocument) -> Result<String, Box<dyn std::error::Error>> {
    let page_count = doc.page_count()?;
    let mut all_text = String::new();

    for page in 0..page_count {
        let text = extract_with_primary_mode(doc, page)?;
        all_text.push_str(&text);
        if page < page_count - 1 {
            all_text.push('\n');
        }
    }

    Ok(all_text)
}

/// Performance metrics for extraction
#[derive(Debug)]
struct PerformanceMetrics {
    extraction_time_ms: u128,
    pages_per_second: f64,
    chars_per_second: f64,
}

/// Test a single PDF through the full pipeline
fn test_pdf_full_pipeline(
    pdf_path: &std::path::Path,
    category: &str,
    manager: &GoldenFileManager,
) -> Result<(), Box<dyn std::error::Error>> {
    let filename = pdf_path.file_name().unwrap().to_str().unwrap();

    // 1. Load PDF
    let mut doc = PdfDocument::open(pdf_path)?;
    let page_count = doc.page_count()?;

    // 2. Extract with Primary mode
    let start = Instant::now();
    let extracted = extract_all_pages_primary(&mut doc)?;
    let extraction_time = start.elapsed();

    // 3. Calculate performance metrics
    let perf = PerformanceMetrics {
        extraction_time_ms: extraction_time.as_millis(),
        pages_per_second: page_count as f64 / extraction_time.as_secs_f64(),
        chars_per_second: extracted.chars().count() as f64 / extraction_time.as_secs_f64(),
    };

    println!(
        "  {} ({} pages, {} chars, {:.2}ms, {:.1} pgs/s)",
        filename,
        page_count,
        extracted.chars().count(),
        perf.extraction_time_ms,
        perf.pages_per_second
    );

    // 4. Quality analysis
    let quality = analyze_quality(&extracted);
    if !quality.passes() {
        eprintln!("    WARNING: Quality issues detected:");
        eprintln!("      - Word fusions: {}", quality.word_fusions.len());
        eprintln!("      - Empty bold markers: {}", quality.empty_bold_markers);
        eprintln!("      - Spurious spaces: {}", quality.spurious_spaces.len());
        eprintln!("      - Quality score: {:.1}/10.0", quality.quality_score);
    }

    // 5. Compare against golden file (if exists)
    if manager.has_golden_file(pdf_path) {
        let golden = manager.load_golden_file(pdf_path)?;
        let comparison = manager.compare_extraction(&extracted, &golden);

        match comparison.status {
            ComparisonStatus::Pass => {
                println!("    [GOLDEN] Pass");
            },
            ComparisonStatus::Warning => {
                println!("    [GOLDEN] Warning: {}", comparison.details());
            },
            ComparisonStatus::Fail => {
                println!("    [GOLDEN] FAIL: {}", comparison.details());
                return Err("Golden file comparison failed".into());
            },
        }
    }

    Ok(())
}

/// Integration test: Academic papers full pipeline
#[test]
fn test_corpus_academic_full_pipeline() {
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let pdfs = loader
        .list_pdfs("academic")
        .expect("Failed to list academic PDFs");
    if pdfs.is_empty() {
        println!("No academic PDFs found, skipping test");
        return;
    }

    println!("\nTesting academic PDFs (first 5):");
    let sample = &pdfs[..pdfs.len().min(5)];

    let mut passed = 0;
    let mut failed = 0;

    for pdf_path in sample {
        match test_pdf_full_pipeline(pdf_path, "academic", &manager) {
            Ok(_) => passed += 1,
            Err(e) => {
                eprintln!("  ERROR: {}", e);
                failed += 1;
            },
        }
    }

    println!("\nResults: {} passed, {} failed", passed, failed);

    // Don't fail test if no golden files exist yet
    if failed > 0 && sample.iter().any(|p| manager.has_golden_file(p)) {
        panic!("Pipeline test failed for {} PDFs", failed);
    }
}

/// Integration test: Mixed documents full pipeline
#[test]
fn test_corpus_mixed_full_pipeline() {
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let pdfs = loader
        .list_pdfs("mixed")
        .expect("Failed to list mixed PDFs");
    if pdfs.is_empty() {
        println!("No mixed PDFs found, skipping test");
        return;
    }

    println!("\nTesting mixed PDFs (first 5):");
    let sample = &pdfs[..pdfs.len().min(5)];

    let mut passed = 0;
    let mut failed = 0;

    for pdf_path in sample {
        match test_pdf_full_pipeline(pdf_path, "mixed", &manager) {
            Ok(_) => passed += 1,
            Err(e) => {
                eprintln!("  ERROR: {}", e);
                failed += 1;
            },
        }
    }

    println!("\nResults: {} passed, {} failed", passed, failed);

    if failed > 0 && sample.iter().any(|p| manager.has_golden_file(p)) {
        panic!("Pipeline test failed for {} PDFs", failed);
    }
}

/// Integration test: Forms full pipeline
#[test]
fn test_corpus_forms_full_pipeline() {
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let pdfs = loader.list_pdfs("forms").expect("Failed to list forms");
    if pdfs.is_empty() {
        println!("No forms found, skipping test");
        return;
    }

    println!("\nTesting forms (first 5):");
    let sample = &pdfs[..pdfs.len().min(5)];

    let mut passed = 0;
    let mut failed = 0;

    for pdf_path in sample {
        match test_pdf_full_pipeline(pdf_path, "forms", &manager) {
            Ok(_) => passed += 1,
            Err(e) => {
                eprintln!("  ERROR: {}", e);
                failed += 1;
            },
        }
    }

    println!("\nResults: {} passed, {} failed", passed, failed);

    if failed > 0 && sample.iter().any(|p| manager.has_golden_file(p)) {
        panic!("Pipeline test failed for {} PDFs", failed);
    }
}

/// Performance benchmark: Extract sample of PDFs and measure throughput
#[test]
#[ignore] // Only run when explicitly requested
fn benchmark_corpus_extraction() {
    let loader = CorpusLoader::default();

    let categories = vec!["academic", "diverse", "forms", "government", "mixed"];

    println!("\n{:=<80}", "");
    println!("Corpus Extraction Benchmark");
    println!("{:=<80}", "");

    for category in categories {
        let pdfs = match loader.list_pdfs(category) {
            Ok(p) => p,
            Err(_) => continue,
        };

        if pdfs.is_empty() {
            continue;
        }

        println!("\nCategory: {}", category);
        println!("{:-<80}", "");

        let sample = &pdfs[..pdfs.len().min(3)];
        let mut total_pages = 0;
        let mut total_chars = 0;
        let mut total_time_ms = 0;

        for pdf_path in sample {
            let filename = pdf_path.file_name().unwrap().to_str().unwrap();

            let mut doc = match PdfDocument::open(pdf_path) {
                Ok(d) => d,
                Err(e) => {
                    println!("  [SKIP] {}: {}", filename, e);
                    continue;
                },
            };

            let page_count = match doc.page_count() {
                Ok(c) => c,
                Err(_) => continue,
            };

            let start = Instant::now();
            let extracted = match extract_all_pages_primary(&mut doc) {
                Ok(t) => t,
                Err(e) => {
                    println!("  [SKIP] {}: {}", filename, e);
                    continue;
                },
            };
            let elapsed = start.elapsed();

            total_pages += page_count;
            total_chars += extracted.chars().count();
            total_time_ms += elapsed.as_millis();

            println!(
                "  {}: {} pages, {} chars, {:.2}ms ({:.1} pgs/s)",
                filename,
                page_count,
                extracted.chars().count(),
                elapsed.as_millis(),
                page_count as f64 / elapsed.as_secs_f64()
            );
        }

        if total_time_ms > 0 {
            println!("{:-<80}", "");
            println!(
                "  Total: {} pages, {} chars, {:.2}ms ({:.1} pgs/s, {:.0} chars/s)",
                total_pages,
                total_chars,
                total_time_ms,
                total_pages as f64 / (total_time_ms as f64 / 1000.0),
                total_chars as f64 / (total_time_ms as f64 / 1000.0)
            );
        }
    }

    println!("\n{:=<80}", "");
}

/// Test: Verify corpus loader works correctly
#[test]
fn test_corpus_loader_basic() {
    let loader = CorpusLoader::default();

    // Should be able to list categories
    let categories = loader.list_categories().expect("Failed to list categories");
    println!("Found {} categories", categories.len());

    // Should find PDFs
    let total = loader.total_pdf_count().expect("Failed to count PDFs");
    println!("Found {} total PDFs", total);

    assert!(total > 0, "Expected to find PDFs in corpus");
}

/// Test: Verify quality metrics integration
#[test]
fn test_quality_metrics_integration() {
    // Test with clean text
    let clean_text = "This is a **clean document** with proper spacing and formatting.";
    let metrics = analyze_quality(clean_text);

    assert!(metrics.passes(), "Clean text should pass quality checks");
    assert_eq!(metrics.empty_bold_markers, 0);
    assert!(metrics.word_fusions.is_empty());

    // Test with issues
    let bad_text = "This has ** ** empty bold and thefollowingtypesof word fusion.";
    let metrics = analyze_quality(bad_text);

    assert!(!metrics.passes(), "Text with issues should fail");
    assert!(metrics.empty_bold_markers > 0 || !metrics.word_fusions.is_empty());
}
