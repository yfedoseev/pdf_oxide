//! Golden File Baseline Generation Test
//!
//! This test suite generates baseline golden files for all PDFs in the test corpus.
//! It is designed to be run explicitly when establishing or updating baselines.
//!
//! ## Purpose
//! - Create initial baseline golden files for regression testing
//! - Update baselines after intentional quality improvements
//! - Provide detailed progress and error reporting
//!
//! ## Usage
//! ```bash
//! # Generate all golden files (full corpus)
//! cargo test --test test_golden_file_generation -- --ignored --nocapture
//!
//! # Generate for specific category
//! cargo test --test test_golden_file_generation test_generate_academic_golden_files -- --ignored --nocapture
//! ```
//!
//! ## Output Structure
//! ```
//! tests/golden_files/extracted_text/
//!   academic/
//!     paper1.json
//!     paper2.json
//!   diverse/
//!     doc1.json
//!   ...
//! ```
//!
//! ## JSON Format
//! Each golden file contains:
//! - `pdf_path`: Original PDF path
//! - `category`: Document category
//! - `extracted_text`: Full extracted text
//! - `text_hash`: SHA-256 hash for quick comparison
//! - `char_count`: Character count for tolerance checks
//! - `word_count`: Word count for tolerance checks
//! - `script_distribution`: Script analysis (Latin, CJK, Arabic, etc.)
//! - `extraction_timestamp`: When the baseline was created

mod helpers;
mod quality_metrics;

use helpers::corpus_loader::CorpusLoader;
use helpers::golden_file_manager::GoldenFileManager;
use pdf_oxide::document::PdfDocument;
use quality_metrics::analyze_quality;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

// ============================================================================
// Configuration Constants
// ============================================================================

/// Maximum time allowed for extracting a single PDF (prevents hanging on corrupted files)
const MAX_EXTRACTION_TIME_SECS: u64 = 120;

/// Maximum file size to process (skip very large PDFs that may cause memory issues)
const MAX_FILE_SIZE_MB: u64 = 100;

/// Categories to process (all available in corpus)
const ALL_CATEGORIES: &[&str] = &[
    "academic",
    "diverse",
    "forms",
    "government",
    "mixed",
    "newspapers",
    "technical",
    "theses",
    "text_heavy",
    "tables",
    "multilingual",
    "scanned",
    "images",
    "test_datasets",
];

// ============================================================================
// Generation Statistics
// ============================================================================

/// Statistics for a single category generation run
#[derive(Debug, Default, Clone)]
struct CategoryStats {
    total_pdfs: usize,
    generated: usize,
    skipped: usize,
    failed: usize,
    total_chars: usize,
    total_words: usize,
    total_pages: usize,
    total_time_ms: u128,
    errors: Vec<String>,
    skipped_reasons: Vec<String>,
}

impl CategoryStats {
    fn success_rate(&self) -> f64 {
        if self.total_pdfs == 0 {
            return 0.0;
        }
        (self.generated as f64 / self.total_pdfs as f64) * 100.0
    }

    fn avg_time_per_pdf_ms(&self) -> f64 {
        if self.generated == 0 {
            return 0.0;
        }
        self.total_time_ms as f64 / self.generated as f64
    }
}

/// Comprehensive generation summary across all categories
#[derive(Debug, Default)]
struct GenerationSummary {
    categories: HashMap<String, CategoryStats>,
    overall_start: Option<Instant>,
    overall_end: Option<Instant>,
}

impl GenerationSummary {
    fn new() -> Self {
        GenerationSummary {
            categories: HashMap::new(),
            overall_start: Some(Instant::now()),
            overall_end: None,
        }
    }

    fn finalize(&mut self) {
        self.overall_end = Some(Instant::now());
    }

    fn total_generated(&self) -> usize {
        self.categories.values().map(|s| s.generated).sum()
    }

    fn total_failed(&self) -> usize {
        self.categories.values().map(|s| s.failed).sum()
    }

    fn total_skipped(&self) -> usize {
        self.categories.values().map(|s| s.skipped).sum()
    }

    fn total_chars(&self) -> usize {
        self.categories.values().map(|s| s.total_chars).sum()
    }

    fn total_words(&self) -> usize {
        self.categories.values().map(|s| s.total_words).sum()
    }

    fn total_pages(&self) -> usize {
        self.categories.values().map(|s| s.total_pages).sum()
    }

    fn overall_duration(&self) -> Duration {
        match (self.overall_start, self.overall_end) {
            (Some(start), Some(end)) => end.duration_since(start),
            _ => Duration::ZERO,
        }
    }

    fn print_summary(&self) {
        println!("\n");
        println!("{}", "=".repeat(80));
        println!("GOLDEN FILE GENERATION SUMMARY");
        println!("{}", "=".repeat(80));

        // Per-category breakdown
        println!("\nCategory Breakdown:");
        println!(
            "{:<15} {:>8} {:>8} {:>8} {:>10} {:>12}",
            "Category", "Total", "Success", "Failed", "Skip", "Success%"
        );
        println!("{}", "-".repeat(70));

        let mut sorted_categories: Vec<_> = self.categories.iter().collect();
        sorted_categories.sort_by_key(|(name, _)| *name);

        for (category, stats) in sorted_categories {
            println!(
                "{:<15} {:>8} {:>8} {:>8} {:>10} {:>11.1}%",
                category,
                stats.total_pdfs,
                stats.generated,
                stats.failed,
                stats.skipped,
                stats.success_rate()
            );
        }

        // Overall totals
        println!("{}", "-".repeat(70));
        let total_pdfs: usize = self.categories.values().map(|s| s.total_pdfs).sum();
        println!(
            "{:<15} {:>8} {:>8} {:>8} {:>10} {:>11.1}%",
            "TOTAL",
            total_pdfs,
            self.total_generated(),
            self.total_failed(),
            self.total_skipped(),
            if total_pdfs > 0 {
                (self.total_generated() as f64 / total_pdfs as f64) * 100.0
            } else {
                0.0
            }
        );

        // Content statistics
        println!("\nContent Statistics:");
        println!("  Total Characters: {:>15}", format_number(self.total_chars()));
        println!("  Total Words:      {:>15}", format_number(self.total_words()));
        println!("  Total Pages:      {:>15}", format_number(self.total_pages()));

        // Timing
        let duration = self.overall_duration();
        println!("\nTiming:");
        println!("  Total Duration:   {:>15}", format_duration(duration));
        if self.total_generated() > 0 {
            let avg_ms = duration.as_millis() as f64 / self.total_generated() as f64;
            println!("  Avg per PDF:      {:>15.1}ms", avg_ms);
        }

        // Error summary
        let all_errors: Vec<_> = self
            .categories
            .values()
            .flat_map(|s| s.errors.iter())
            .collect();
        if !all_errors.is_empty() {
            println!("\nErrors ({}):", all_errors.len());
            for (i, error) in all_errors.iter().take(10).enumerate() {
                println!("  {}. {}", i + 1, error);
            }
            if all_errors.len() > 10 {
                println!("  ... and {} more errors", all_errors.len() - 10);
            }
        }

        println!("\n{}", "=".repeat(80));
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Format large numbers with comma separators
fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.insert(0, ',');
        }
        result.insert(0, c);
    }
    result
}

/// Format duration as human-readable string
fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    if secs < 60 {
        format!("{:.1}s", d.as_secs_f64())
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    }
}

/// Extract text from all pages of a PDF
fn extract_all_pages(doc: &mut PdfDocument) -> Result<String, Box<dyn std::error::Error>> {
    let page_count = doc.page_count()?;
    let mut all_text = String::new();

    for page in 0..page_count {
        let text = doc.extract_text(page)?;
        all_text.push_str(&text);
        if page < page_count - 1 {
            all_text.push('\n');
        }
    }

    Ok(all_text)
}

/// Check if PDF should be skipped based on file size
fn should_skip_file(path: &PathBuf) -> Option<String> {
    match std::fs::metadata(path) {
        Ok(meta) => {
            let size_mb = meta.len() / (1024 * 1024);
            if size_mb > MAX_FILE_SIZE_MB {
                Some(format!("File too large: {}MB (max {}MB)", size_mb, MAX_FILE_SIZE_MB))
            } else {
                None
            }
        },
        Err(e) => Some(format!("Cannot read metadata: {}", e)),
    }
}

/// Generate golden file for a single PDF
fn generate_golden_for_pdf(
    pdf_path: &PathBuf,
    category: &str,
    manager: &GoldenFileManager,
) -> Result<(usize, usize, usize), Box<dyn std::error::Error>> {
    let mut doc = PdfDocument::open(pdf_path)?;
    let page_count = doc.page_count()?;

    let extracted = extract_all_pages(&mut doc)?;
    let char_count = extracted.chars().count();
    let word_count = extracted.split_whitespace().count();

    manager.save_golden_file(pdf_path, category, &extracted)?;

    Ok((char_count, word_count, page_count))
}

/// Generate golden files for an entire category
fn generate_category_golden_files(
    category: &str,
    loader: &CorpusLoader,
    manager: &GoldenFileManager,
    verbose: bool,
) -> CategoryStats {
    let mut stats = CategoryStats::default();

    let pdfs = match loader.list_pdfs(category) {
        Ok(p) => p,
        Err(e) => {
            stats.errors.push(format!("Cannot list PDFs: {}", e));
            return stats;
        },
    };

    stats.total_pdfs = pdfs.len();

    if verbose {
        println!("\n[{}] Processing {} PDFs...", category.to_uppercase(), pdfs.len());
    }

    for (i, pdf_path) in pdfs.iter().enumerate() {
        let filename = pdf_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        // Check if should skip
        if let Some(reason) = should_skip_file(pdf_path) {
            stats.skipped += 1;
            stats
                .skipped_reasons
                .push(format!("{}: {}", filename, reason));
            if verbose {
                println!("  [{}/{}] SKIP {} - {}", i + 1, pdfs.len(), filename, reason);
            }
            continue;
        }

        let start = Instant::now();

        match generate_golden_for_pdf(pdf_path, category, manager) {
            Ok((chars, words, pages)) => {
                let elapsed = start.elapsed();
                stats.generated += 1;
                stats.total_chars += chars;
                stats.total_words += words;
                stats.total_pages += pages;
                stats.total_time_ms += elapsed.as_millis();

                if verbose {
                    println!(
                        "  [{}/{}] OK {} ({} pages, {} chars, {:.0}ms)",
                        i + 1,
                        pdfs.len(),
                        filename,
                        pages,
                        chars,
                        elapsed.as_millis()
                    );
                }
            },
            Err(e) => {
                stats.failed += 1;
                let error_msg = format!("{}: {}", filename, e);
                stats.errors.push(error_msg.clone());

                if verbose {
                    println!("  [{}/{}] FAIL {} - {}", i + 1, pdfs.len(), filename, e);
                }
            },
        }
    }

    if verbose {
        println!(
            "[{}] Complete: {} generated, {} failed, {} skipped",
            category.to_uppercase(),
            stats.generated,
            stats.failed,
            stats.skipped
        );
    }

    stats
}

// ============================================================================
// Main Generation Tests
// ============================================================================

/// Generate golden files for ALL PDFs in ALL categories.
///
/// This is the main baseline generation test. Run with:
/// ```bash
/// cargo test --test test_golden_file_generation test_generate_all_golden_files -- --ignored --nocapture
/// ```
///
/// ## Behavior
/// - Iterates through all 14 categories
/// - Extracts text from each PDF
/// - Saves golden files to `tests/golden_files/extracted_text/{category}/`
/// - Provides detailed progress and summary reporting
///
/// ## When to Run
/// 1. Initial baseline creation (first-time setup)
/// 2. After intentional quality improvements
/// 3. When adding new PDFs to the corpus
#[test]
#[ignore]
fn test_generate_all_golden_files() {
    println!("\n");
    println!("{}", "=".repeat(80));
    println!("GOLDEN FILE BASELINE GENERATION - FULL CORPUS");
    println!("{}", "=".repeat(80));
    println!("Output: tests/golden_files/extracted_text/{{category}}/*.json");
    println!("Categories: {}", ALL_CATEGORIES.len());
    println!();

    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();
    let mut summary = GenerationSummary::new();

    // Process each category
    for category in ALL_CATEGORIES {
        let stats = generate_category_golden_files(category, &loader, &manager, true);
        summary.categories.insert(category.to_string(), stats);
    }

    summary.finalize();
    summary.print_summary();

    // Verify at least some files were generated
    let total = summary.total_generated();
    assert!(
        total > 0,
        "No golden files were generated! Check corpus path and PDF availability."
    );

    println!("\nGolden file generation complete!");
    println!("Next steps:");
    println!("  1. Review generated files in tests/golden_files/extracted_text/");
    println!("  2. Spot-check 5-10 PDFs per category for quality");
    println!("  3. Run regression tests: cargo test --test test_extraction_regression");
}

/// Generate golden files for academic papers category only.
#[test]
#[ignore]
fn test_generate_academic_golden_files() {
    println!("\n[GENERATING GOLDEN FILES: ACADEMIC]");
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let stats = generate_category_golden_files("academic", &loader, &manager, true);
    println!(
        "\nSummary: {} generated, {} failed, {} skipped",
        stats.generated, stats.failed, stats.skipped
    );

    assert!(stats.generated > 0, "No academic golden files generated");
}

/// Generate golden files for diverse documents category only.
#[test]
#[ignore]
fn test_generate_diverse_golden_files() {
    println!("\n[GENERATING GOLDEN FILES: DIVERSE]");
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let stats = generate_category_golden_files("diverse", &loader, &manager, true);
    println!(
        "\nSummary: {} generated, {} failed, {} skipped",
        stats.generated, stats.failed, stats.skipped
    );

    assert!(
        stats.generated > 0 || stats.total_pdfs == 0,
        "No diverse golden files generated"
    );
}

/// Generate golden files for forms category only.
#[test]
#[ignore]
fn test_generate_forms_golden_files() {
    println!("\n[GENERATING GOLDEN FILES: FORMS]");
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let stats = generate_category_golden_files("forms", &loader, &manager, true);
    println!(
        "\nSummary: {} generated, {} failed, {} skipped",
        stats.generated, stats.failed, stats.skipped
    );

    assert!(stats.generated > 0 || stats.total_pdfs == 0, "No forms golden files generated");
}

/// Generate golden files for government documents category only.
#[test]
#[ignore]
fn test_generate_government_golden_files() {
    println!("\n[GENERATING GOLDEN FILES: GOVERNMENT]");
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let stats = generate_category_golden_files("government", &loader, &manager, true);
    println!(
        "\nSummary: {} generated, {} failed, {} skipped",
        stats.generated, stats.failed, stats.skipped
    );

    assert!(
        stats.generated > 0 || stats.total_pdfs == 0,
        "No government golden files generated"
    );
}

/// Generate golden files for mixed layout documents category only.
#[test]
#[ignore]
fn test_generate_mixed_golden_files() {
    println!("\n[GENERATING GOLDEN FILES: MIXED]");
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let stats = generate_category_golden_files("mixed", &loader, &manager, true);
    println!(
        "\nSummary: {} generated, {} failed, {} skipped",
        stats.generated, stats.failed, stats.skipped
    );

    assert!(stats.generated > 0 || stats.total_pdfs == 0, "No mixed golden files generated");
}

/// Generate golden files for newspapers category only.
#[test]
#[ignore]
fn test_generate_newspapers_golden_files() {
    println!("\n[GENERATING GOLDEN FILES: NEWSPAPERS]");
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let stats = generate_category_golden_files("newspapers", &loader, &manager, true);
    println!(
        "\nSummary: {} generated, {} failed, {} skipped",
        stats.generated, stats.failed, stats.skipped
    );

    assert!(
        stats.generated > 0 || stats.total_pdfs == 0,
        "No newspapers golden files generated"
    );
}

/// Generate golden files for technical documents category only.
#[test]
#[ignore]
fn test_generate_technical_golden_files() {
    println!("\n[GENERATING GOLDEN FILES: TECHNICAL]");
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let stats = generate_category_golden_files("technical", &loader, &manager, true);
    println!(
        "\nSummary: {} generated, {} failed, {} skipped",
        stats.generated, stats.failed, stats.skipped
    );

    assert!(
        stats.generated > 0 || stats.total_pdfs == 0,
        "No technical golden files generated"
    );
}

/// Generate golden files for theses category only.
#[test]
#[ignore]
fn test_generate_theses_golden_files() {
    println!("\n[GENERATING GOLDEN FILES: THESES]");
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let stats = generate_category_golden_files("theses", &loader, &manager, true);
    println!(
        "\nSummary: {} generated, {} failed, {} skipped",
        stats.generated, stats.failed, stats.skipped
    );

    assert!(stats.generated > 0 || stats.total_pdfs == 0, "No theses golden files generated");
}

/// Generate golden files for text-heavy documents category only.
#[test]
#[ignore]
fn test_generate_text_heavy_golden_files() {
    println!("\n[GENERATING GOLDEN FILES: TEXT_HEAVY]");
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let stats = generate_category_golden_files("text_heavy", &loader, &manager, true);
    println!(
        "\nSummary: {} generated, {} failed, {} skipped",
        stats.generated, stats.failed, stats.skipped
    );

    assert!(
        stats.generated > 0 || stats.total_pdfs == 0,
        "No text_heavy golden files generated"
    );
}

/// Generate golden files for tables category only.
#[test]
#[ignore]
fn test_generate_tables_golden_files() {
    println!("\n[GENERATING GOLDEN FILES: TABLES]");
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let stats = generate_category_golden_files("tables", &loader, &manager, true);
    println!(
        "\nSummary: {} generated, {} failed, {} skipped",
        stats.generated, stats.failed, stats.skipped
    );

    assert!(stats.generated > 0 || stats.total_pdfs == 0, "No tables golden files generated");
}

/// Generate golden files for multilingual documents category only.
#[test]
#[ignore]
fn test_generate_multilingual_golden_files() {
    println!("\n[GENERATING GOLDEN FILES: MULTILINGUAL]");
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let stats = generate_category_golden_files("multilingual", &loader, &manager, true);
    println!(
        "\nSummary: {} generated, {} failed, {} skipped",
        stats.generated, stats.failed, stats.skipped
    );

    assert!(
        stats.generated > 0 || stats.total_pdfs == 0,
        "No multilingual golden files generated"
    );
}

// ============================================================================
// Quality Validation During Generation
// ============================================================================

/// Generate golden files with quality validation.
///
/// This test not only generates golden files but also runs quality metrics
/// on each extraction to identify potential issues before committing baselines.
#[test]
#[ignore]
fn test_generate_with_quality_validation() {
    println!("\n");
    println!("{}", "=".repeat(80));
    println!("GOLDEN FILE GENERATION WITH QUALITY VALIDATION");
    println!("{}", "=".repeat(80));

    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let categories_to_validate = vec!["academic", "diverse", "forms", "government", "mixed"];
    let mut quality_issues: Vec<(String, String, f32)> = Vec::new();

    for category in categories_to_validate {
        let pdfs = match loader.list_pdfs(category) {
            Ok(p) => p,
            Err(_) => continue,
        };

        println!("\n[{}] Validating {} PDFs...", category.to_uppercase(), pdfs.len());

        for pdf_path in pdfs.iter().take(10) {
            // Sample first 10 per category
            let filename = pdf_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");

            match PdfDocument::open(pdf_path) {
                Ok(mut doc) => {
                    if let Ok(text) = extract_all_pages(&mut doc) {
                        let metrics = analyze_quality(&text);

                        // Save golden file
                        let _ = manager.save_golden_file(pdf_path, category, &text);

                        if !metrics.passes() {
                            quality_issues.push((
                                category.to_string(),
                                filename.to_string(),
                                metrics.quality_score,
                            ));
                            println!(
                                "  WARNING: {} - Quality score {:.1}/10 (fusions: {}, empty bold: {})",
                                filename,
                                metrics.quality_score,
                                metrics.word_fusions.len(),
                                metrics.empty_bold_markers
                            );
                        } else {
                            println!(
                                "  OK: {} - Quality score {:.1}/10",
                                filename, metrics.quality_score
                            );
                        }
                    }
                },
                Err(e) => {
                    println!("  SKIP: {} - {}", filename, e);
                },
            }
        }
    }

    // Summary of quality issues
    if !quality_issues.is_empty() {
        println!("\n{}", "-".repeat(80));
        println!("QUALITY ISSUES FOUND ({} files):", quality_issues.len());
        for (cat, file, score) in &quality_issues {
            println!("  [{}/{}] Score: {:.1}", cat, file, score);
        }
        println!("\nNote: These files have quality issues but baselines were still created.");
        println!("Review and address the underlying extraction issues if needed.");
    } else {
        println!("\nAll validated files passed quality checks!");
    }
}

// ============================================================================
// Corpus Information Tests
// ============================================================================

/// List all PDFs in the corpus with statistics.
/// Useful for understanding corpus composition before generation.
#[test]
#[ignore]
fn test_list_corpus_detailed() {
    println!("\n");
    println!("{}", "=".repeat(80));
    println!("TEST CORPUS DETAILED INVENTORY");
    println!("{}", "=".repeat(80));

    let loader = CorpusLoader::default();
    let categories = loader.list_categories().unwrap_or_default();

    let mut total_pdfs = 0;
    let mut total_size_bytes: u64 = 0;

    println!("\n{:<20} {:>10} {:>15}", "Category", "PDF Count", "Total Size");
    println!("{}", "-".repeat(50));

    for category in &categories {
        let pdfs = loader.list_pdfs(category).unwrap_or_default();
        let count = pdfs.len();
        total_pdfs += count;

        let size: u64 = pdfs
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .sum();
        total_size_bytes += size;

        let size_mb = size as f64 / (1024.0 * 1024.0);
        println!("{:<20} {:>10} {:>14.1}MB", category, count, size_mb);
    }

    println!("{}", "-".repeat(50));
    let total_mb = total_size_bytes as f64 / (1024.0 * 1024.0);
    println!("{:<20} {:>10} {:>14.1}MB", "TOTAL", total_pdfs, total_mb);

    println!("\n{}", "=".repeat(80));
}

/// Verify golden file generation infrastructure is working.
/// This is a quick sanity check that doesn't require --ignored flag.
#[test]
fn test_generation_infrastructure_available() {
    // Verify corpus loader works
    let loader = CorpusLoader::default();
    let categories = loader.list_categories();
    assert!(categories.is_ok(), "CorpusLoader should be able to list categories");

    // Verify golden file manager works
    let manager = GoldenFileManager::default();
    // Just verify it can be instantiated (actual file operations tested in ignored tests)
    let _ = manager;

    println!("Golden file generation infrastructure is available and ready.");
}
