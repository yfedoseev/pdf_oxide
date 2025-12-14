//! Extraction Regression Test Suite
//!
//! This test suite validates text extraction against golden file baselines to detect regressions.
//! Each test compares current extraction output with saved baselines, using configurable tolerances.
//!
//! ## Purpose
//! - Detect regressions in text extraction quality
//! - Validate extraction consistency across releases
//! - Provide detailed diff reports for investigation
//!
//! ## Usage
//! ```bash
//! # Run all regression tests
//! cargo test --test test_extraction_regression --release
//!
//! # Run specific category
//! cargo test --test test_extraction_regression test_regression_academic -- --nocapture
//!
//! # Run with detailed output
//! cargo test --test test_extraction_regression -- --nocapture
//! ```
//!
//! ## Tolerance Thresholds
//! - Character count: +/- 0.5% (allows minor encoding differences)
//! - Word count: +/- 1.0% (allows minor spacing differences)
//! - Line count: +/- 2.0% (allows layout differences)
//!
//! ## Prerequisites
//! Golden files must be generated first:
//! ```bash
//! cargo test --test test_golden_file_generation -- --ignored --nocapture
//! ```

mod helpers;

use helpers::corpus_loader::CorpusLoader;
use helpers::golden_file_manager::{ComparisonResult, ComparisonStatus, GoldenFileManager};
use pdf_oxide::document::PdfDocument;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

// ============================================================================
// Configuration
// ============================================================================

/// Maximum number of PDFs to test per category (None = test all)
const DEFAULT_MAX_PDFS_PER_CATEGORY: Option<usize> = None;

/// Whether to show detailed diff output for failures
const SHOW_DETAILED_DIFFS: bool = true;

/// Whether to include quality metrics in regression output
const INCLUDE_QUALITY_METRICS: bool = true;

/// Character count tolerance (0.5% = 0.005)
const CHAR_COUNT_TOLERANCE: f64 = 0.005;

/// Word count tolerance (1.0% = 0.01)
const WORD_COUNT_TOLERANCE: f64 = 0.01;

/// Line count tolerance (2.0% = 0.02)
const LINE_COUNT_TOLERANCE: f64 = 0.02;

// ============================================================================
// Regression Statistics
// ============================================================================

/// Individual PDF regression result
#[derive(Debug, Clone)]
struct PdfRegressionResult {
    filename: String,
    status: RegressionStatus,
    char_diff: i64,
    char_diff_pct: f64,
    word_diff: i64,
    word_diff_pct: f64,
    extraction_time_ms: u128,
    quality_score: Option<f32>,
    error_message: Option<String>,
    diff_context: Option<String>,
}

/// Status of a single PDF regression check
#[derive(Debug, Clone, PartialEq)]
enum RegressionStatus {
    /// Extraction matches baseline perfectly
    Pass,
    /// Minor differences within tolerance
    Warning,
    /// Significant regression detected
    Regression,
    /// Golden file not found (skipped)
    NoBaseline,
    /// Error during extraction
    Error,
}

impl RegressionStatus {
    fn symbol(&self) -> &'static str {
        match self {
            RegressionStatus::Pass => "PASS",
            RegressionStatus::Warning => "WARN",
            RegressionStatus::Regression => "FAIL",
            RegressionStatus::NoBaseline => "SKIP",
            RegressionStatus::Error => "ERR ",
        }
    }

    fn is_failure(&self) -> bool {
        matches!(self, RegressionStatus::Regression | RegressionStatus::Error)
    }
}

/// Category-level regression statistics
#[derive(Debug, Default)]
struct CategoryRegressionStats {
    total_tested: usize,
    passed: usize,
    warnings: usize,
    regressions: usize,
    no_baseline: usize,
    errors: usize,
    total_extraction_time_ms: u128,
    results: Vec<PdfRegressionResult>,
}

impl CategoryRegressionStats {
    fn add_result(&mut self, result: PdfRegressionResult) {
        self.total_tested += 1;
        self.total_extraction_time_ms += result.extraction_time_ms;

        match result.status {
            RegressionStatus::Pass => self.passed += 1,
            RegressionStatus::Warning => self.warnings += 1,
            RegressionStatus::Regression => self.regressions += 1,
            RegressionStatus::NoBaseline => self.no_baseline += 1,
            RegressionStatus::Error => self.errors += 1,
        }

        self.results.push(result);
    }

    fn has_failures(&self) -> bool {
        self.regressions > 0 || self.errors > 0
    }

    fn pass_rate(&self) -> f64 {
        let testable = self.total_tested - self.no_baseline;
        if testable == 0 {
            return 100.0;
        }
        ((self.passed + self.warnings) as f64 / testable as f64) * 100.0
    }
}

/// Overall regression summary
#[derive(Debug, Default)]
struct RegressionSummary {
    categories: HashMap<String, CategoryRegressionStats>,
    start_time: Option<Instant>,
}

impl RegressionSummary {
    fn new() -> Self {
        RegressionSummary {
            categories: HashMap::new(),
            start_time: Some(Instant::now()),
        }
    }

    fn total_tested(&self) -> usize {
        self.categories.values().map(|s| s.total_tested).sum()
    }

    fn total_passed(&self) -> usize {
        self.categories.values().map(|s| s.passed).sum()
    }

    fn total_warnings(&self) -> usize {
        self.categories.values().map(|s| s.warnings).sum()
    }

    fn total_regressions(&self) -> usize {
        self.categories.values().map(|s| s.regressions).sum()
    }

    fn total_errors(&self) -> usize {
        self.categories.values().map(|s| s.errors).sum()
    }

    fn total_no_baseline(&self) -> usize {
        self.categories.values().map(|s| s.no_baseline).sum()
    }

    fn has_failures(&self) -> bool {
        self.categories.values().any(|s| s.has_failures())
    }

    fn print_summary(&self) {
        println!("\n");
        println!("{}", "=".repeat(80));
        println!("REGRESSION TEST SUMMARY");
        println!("{}", "=".repeat(80));

        // Per-category breakdown
        println!(
            "\n{:<15} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}",
            "Category", "Tested", "Pass", "Warn", "Regress", "Error", "Pass%"
        );
        println!("{}", "-".repeat(75));

        let mut sorted: Vec<_> = self.categories.iter().collect();
        sorted.sort_by_key(|(name, _)| *name);

        for (category, stats) in &sorted {
            println!(
                "{:<15} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9.1}%",
                category,
                stats.total_tested,
                stats.passed,
                stats.warnings,
                stats.regressions,
                stats.errors,
                stats.pass_rate()
            );
        }

        // Totals
        println!("{}", "-".repeat(75));
        let total = self.total_tested();
        let testable = total - self.total_no_baseline();
        let pass_rate = if testable > 0 {
            ((self.total_passed() + self.total_warnings()) as f64 / testable as f64) * 100.0
        } else {
            100.0
        };
        println!(
            "{:<15} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9.1}%",
            "TOTAL",
            total,
            self.total_passed(),
            self.total_warnings(),
            self.total_regressions(),
            self.total_errors(),
            pass_rate
        );

        // Timing
        if let Some(start) = self.start_time {
            let elapsed = start.elapsed();
            println!("\nTotal time: {:.1}s", elapsed.as_secs_f64());
        }

        // List failures if any
        if self.has_failures() {
            println!("\n{}", "-".repeat(80));
            println!("FAILURES:");

            for (category, stats) in &sorted {
                for result in &stats.results {
                    if result.status.is_failure() {
                        println!(
                            "\n  [{}/{}] {}",
                            category,
                            result.filename,
                            result.status.symbol()
                        );
                        if let Some(msg) = &result.error_message {
                            println!("    Error: {}", msg);
                        }
                        if let Some(ctx) = &result.diff_context {
                            println!("    Diff: {}", ctx);
                        }
                        println!(
                            "    Char diff: {} ({:+.2}%)",
                            result.char_diff, result.char_diff_pct
                        );
                        println!(
                            "    Word diff: {} ({:+.2}%)",
                            result.word_diff, result.word_diff_pct
                        );
                    }
                }
            }
        }

        // Overall status
        println!("\n{}", "=".repeat(80));
        if self.has_failures() {
            println!(
                "RESULT: FAILED - {} regressions, {} errors detected",
                self.total_regressions(),
                self.total_errors()
            );
        } else if self.total_no_baseline() == total {
            println!("RESULT: SKIPPED - No golden file baselines found");
            println!("Run: cargo test --test test_golden_file_generation -- --ignored --nocapture");
        } else {
            println!("RESULT: PASSED - All tests within tolerance");
        }
        println!("{}", "=".repeat(80));
    }
}

// ============================================================================
// Core Regression Testing Logic
// ============================================================================

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

/// Run regression test for a single PDF
fn test_pdf_regression(pdf_path: &PathBuf, manager: &GoldenFileManager) -> PdfRegressionResult {
    let filename = pdf_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    // Check if golden file exists
    if !manager.has_golden_file(pdf_path) {
        return PdfRegressionResult {
            filename,
            status: RegressionStatus::NoBaseline,
            char_diff: 0,
            char_diff_pct: 0.0,
            word_diff: 0,
            word_diff_pct: 0.0,
            extraction_time_ms: 0,
            quality_score: None,
            error_message: None,
            diff_context: None,
        };
    }

    // Load golden file
    let golden = match manager.load_golden_file(pdf_path) {
        Ok(g) => g,
        Err(e) => {
            return PdfRegressionResult {
                filename,
                status: RegressionStatus::Error,
                char_diff: 0,
                char_diff_pct: 0.0,
                word_diff: 0,
                word_diff_pct: 0.0,
                extraction_time_ms: 0,
                quality_score: None,
                error_message: Some(format!("Cannot load golden file: {}", e)),
                diff_context: None,
            };
        },
    };

    // Extract text
    let start = Instant::now();
    let extracted = match PdfDocument::open(pdf_path) {
        Ok(mut doc) => match extract_all_pages(&mut doc) {
            Ok(text) => text,
            Err(e) => {
                return PdfRegressionResult {
                    filename,
                    status: RegressionStatus::Error,
                    char_diff: 0,
                    char_diff_pct: 0.0,
                    word_diff: 0,
                    word_diff_pct: 0.0,
                    extraction_time_ms: start.elapsed().as_millis(),
                    quality_score: None,
                    error_message: Some(format!("Extraction failed: {}", e)),
                    diff_context: None,
                };
            },
        },
        Err(e) => {
            return PdfRegressionResult {
                filename,
                status: RegressionStatus::Error,
                char_diff: 0,
                char_diff_pct: 0.0,
                word_diff: 0,
                word_diff_pct: 0.0,
                extraction_time_ms: start.elapsed().as_millis(),
                quality_score: None,
                error_message: Some(format!("Cannot open PDF: {}", e)),
                diff_context: None,
            };
        },
    };
    let extraction_time_ms = start.elapsed().as_millis();

    // Compare against golden file
    let comparison = manager.compare_extraction(&extracted, &golden);

    // Calculate quality score if enabled
    let quality_score = if INCLUDE_QUALITY_METRICS {
        Some(analyze_quality(&extracted).quality_score)
    } else {
        None
    };

    // Calculate diffs
    let extracted_chars = extracted.chars().count() as i64;
    let extracted_words = extracted.split_whitespace().count() as i64;
    let char_diff = extracted_chars - golden.char_count as i64;
    let word_diff = extracted_words - golden.word_count as i64;

    let char_diff_pct = if golden.char_count > 0 {
        (char_diff as f64 / golden.char_count as f64) * 100.0
    } else {
        0.0
    };
    let word_diff_pct = if golden.word_count > 0 {
        (word_diff as f64 / golden.word_count as f64) * 100.0
    } else {
        0.0
    };

    // Determine status
    let status = match comparison.status {
        ComparisonStatus::Pass => RegressionStatus::Pass,
        ComparisonStatus::Warning => RegressionStatus::Warning,
        ComparisonStatus::Fail => RegressionStatus::Regression,
    };

    PdfRegressionResult {
        filename,
        status,
        char_diff,
        char_diff_pct,
        word_diff,
        word_diff_pct,
        extraction_time_ms,
        quality_score,
        error_message: None,
        diff_context: comparison.diff_context,
    }
}

/// Run regression tests for an entire category
fn test_category_regression(
    category: &str,
    loader: &CorpusLoader,
    manager: &GoldenFileManager,
    max_pdfs: Option<usize>,
    verbose: bool,
) -> CategoryRegressionStats {
    let mut stats = CategoryRegressionStats::default();

    let pdfs = match loader.list_pdfs(category) {
        Ok(p) => p,
        Err(_) => return stats,
    };

    let pdfs_to_test = if let Some(max) = max_pdfs {
        &pdfs[..pdfs.len().min(max)]
    } else {
        &pdfs[..]
    };

    if verbose {
        println!(
            "\n[{}] Testing {} PDFs against baselines...",
            category.to_uppercase(),
            pdfs_to_test.len()
        );
    }

    for (i, pdf_path) in pdfs_to_test.iter().enumerate() {
        let result = test_pdf_regression(pdf_path, manager);

        if verbose {
            let quality_str = result
                .quality_score
                .map(|q| format!(" Q:{:.1}", q))
                .unwrap_or_default();

            match result.status {
                RegressionStatus::Pass => {
                    println!(
                        "  [{}/{}] [{}] {}{}",
                        i + 1,
                        pdfs_to_test.len(),
                        result.status.symbol(),
                        result.filename,
                        quality_str
                    );
                },
                RegressionStatus::Warning => {
                    println!(
                        "  [{}/{}] [{}] {} (chars: {:+.2}%, words: {:+.2}%){}",
                        i + 1,
                        pdfs_to_test.len(),
                        result.status.symbol(),
                        result.filename,
                        result.char_diff_pct,
                        result.word_diff_pct,
                        quality_str
                    );
                },
                RegressionStatus::Regression => {
                    println!(
                        "  [{}/{}] [{}] {} - REGRESSION DETECTED",
                        i + 1,
                        pdfs_to_test.len(),
                        result.status.symbol(),
                        result.filename
                    );
                    println!(
                        "         Char diff: {} ({:+.2}%)",
                        result.char_diff, result.char_diff_pct
                    );
                    println!(
                        "         Word diff: {} ({:+.2}%)",
                        result.word_diff, result.word_diff_pct
                    );
                    if SHOW_DETAILED_DIFFS {
                        if let Some(ctx) = &result.diff_context {
                            println!("         Context: {}", ctx);
                        }
                    }
                },
                RegressionStatus::NoBaseline => {
                    println!(
                        "  [{}/{}] [{}] {} - No baseline",
                        i + 1,
                        pdfs_to_test.len(),
                        result.status.symbol(),
                        result.filename
                    );
                },
                RegressionStatus::Error => {
                    println!(
                        "  [{}/{}] [{}] {} - {}",
                        i + 1,
                        pdfs_to_test.len(),
                        result.status.symbol(),
                        result.filename,
                        result.error_message.as_deref().unwrap_or("Unknown error")
                    );
                },
            }
        }

        stats.add_result(result);
    }

    if verbose {
        println!(
            "[{}] Complete: {} pass, {} warn, {} regress, {} skip, {} error",
            category.to_uppercase(),
            stats.passed,
            stats.warnings,
            stats.regressions,
            stats.no_baseline,
            stats.errors
        );
    }

    stats
}

// ============================================================================
// Per-Category Regression Tests
// ============================================================================

/// Regression test for academic papers category
#[test]
fn test_regression_academic() {
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let stats = test_category_regression("academic", &loader, &manager, Some(20), true);

    // Only fail if we had baselines and found regressions
    if stats.total_tested > stats.no_baseline && stats.has_failures() {
        panic!(
            "Academic regression test failed: {} regressions, {} errors",
            stats.regressions, stats.errors
        );
    }
}

/// Regression test for diverse documents category
#[test]
fn test_regression_diverse() {
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let stats = test_category_regression("diverse", &loader, &manager, Some(20), true);

    if stats.total_tested > stats.no_baseline && stats.has_failures() {
        panic!(
            "Diverse regression test failed: {} regressions, {} errors",
            stats.regressions, stats.errors
        );
    }
}

/// Regression test for forms category
#[test]
fn test_regression_forms() {
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let stats = test_category_regression("forms", &loader, &manager, Some(20), true);

    if stats.total_tested > stats.no_baseline && stats.has_failures() {
        panic!(
            "Forms regression test failed: {} regressions, {} errors",
            stats.regressions, stats.errors
        );
    }
}

/// Regression test for government documents category
#[test]
fn test_regression_government() {
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let stats = test_category_regression("government", &loader, &manager, Some(20), true);

    if stats.total_tested > stats.no_baseline && stats.has_failures() {
        panic!(
            "Government regression test failed: {} regressions, {} errors",
            stats.regressions, stats.errors
        );
    }
}

/// Regression test for mixed layout documents category
#[test]
fn test_regression_mixed() {
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let stats = test_category_regression("mixed", &loader, &manager, Some(20), true);

    if stats.total_tested > stats.no_baseline && stats.has_failures() {
        panic!(
            "Mixed regression test failed: {} regressions, {} errors",
            stats.regressions, stats.errors
        );
    }
}

/// Regression test for newspapers category
#[test]
fn test_regression_newspapers() {
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let stats = test_category_regression("newspapers", &loader, &manager, Some(20), true);

    if stats.total_tested > stats.no_baseline && stats.has_failures() {
        panic!(
            "Newspapers regression test failed: {} regressions, {} errors",
            stats.regressions, stats.errors
        );
    }
}

/// Regression test for technical documents category
#[test]
fn test_regression_technical() {
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let stats = test_category_regression("technical", &loader, &manager, Some(20), true);

    if stats.total_tested > stats.no_baseline && stats.has_failures() {
        panic!(
            "Technical regression test failed: {} regressions, {} errors",
            stats.regressions, stats.errors
        );
    }
}

/// Regression test for theses category
#[test]
fn test_regression_theses() {
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let stats = test_category_regression("theses", &loader, &manager, Some(20), true);

    if stats.total_tested > stats.no_baseline && stats.has_failures() {
        panic!(
            "Theses regression test failed: {} regressions, {} errors",
            stats.regressions, stats.errors
        );
    }
}

/// Regression test for text-heavy documents category
#[test]
fn test_regression_text_heavy() {
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let stats = test_category_regression("text_heavy", &loader, &manager, Some(20), true);

    if stats.total_tested > stats.no_baseline && stats.has_failures() {
        panic!(
            "Text-heavy regression test failed: {} regressions, {} errors",
            stats.regressions, stats.errors
        );
    }
}

/// Regression test for tables category
#[test]
fn test_regression_tables() {
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let stats = test_category_regression("tables", &loader, &manager, Some(20), true);

    if stats.total_tested > stats.no_baseline && stats.has_failures() {
        panic!(
            "Tables regression test failed: {} regressions, {} errors",
            stats.regressions, stats.errors
        );
    }
}

/// Regression test for multilingual documents category
#[test]
fn test_regression_multilingual() {
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let stats = test_category_regression("multilingual", &loader, &manager, Some(20), true);

    if stats.total_tested > stats.no_baseline && stats.has_failures() {
        panic!(
            "Multilingual regression test failed: {} regressions, {} errors",
            stats.regressions, stats.errors
        );
    }
}

// ============================================================================
// Comprehensive Regression Tests
// ============================================================================

/// Run full regression test across all categories.
/// This is the main regression test for CI/CD pipelines.
#[test]
#[ignore]
fn test_regression_full_corpus() {
    println!("\n");
    println!("{}", "=".repeat(80));
    println!("FULL CORPUS REGRESSION TEST");
    println!("{}", "=".repeat(80));

    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();
    let mut summary = RegressionSummary::new();

    let categories = vec![
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
    ];

    for category in categories {
        let stats = test_category_regression(
            category,
            &loader,
            &manager,
            DEFAULT_MAX_PDFS_PER_CATEGORY,
            true,
        );
        summary.categories.insert(category.to_string(), stats);
    }

    summary.print_summary();

    if summary.has_failures() {
        panic!(
            "Regression test failed: {} regressions, {} errors",
            summary.total_regressions(),
            summary.total_errors()
        );
    }
}

/// Quick regression test with limited samples per category.
/// Useful for fast CI checks.
#[test]
fn test_regression_quick() {
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();
    let mut summary = RegressionSummary::new();

    // Test 5 PDFs per category for quick validation
    let categories = vec!["academic", "diverse", "forms", "government", "mixed"];

    for category in categories {
        let stats = test_category_regression(category, &loader, &manager, Some(5), false);
        summary.categories.insert(category.to_string(), stats);
    }

    // Don't fail on no baselines, only on actual regressions
    let has_baselines = summary.total_tested() > summary.total_no_baseline();
    if has_baselines && summary.has_failures() {
        summary.print_summary();
        panic!(
            "Quick regression test failed: {} regressions, {} errors",
            summary.total_regressions(),
            summary.total_errors()
        );
    }
}

// ============================================================================
// Baseline Update Tests (for intentional improvements)
// ============================================================================

/// Update baselines for PDFs that have improved.
///
/// Run this after intentional quality improvements to update baselines.
/// Always review changes before committing!
#[test]
#[ignore]
fn test_update_baselines_for_improved() {
    println!("\n");
    println!("{}", "=".repeat(80));
    println!("UPDATE BASELINES FOR IMPROVED EXTRACTIONS");
    println!("{}", "=".repeat(80));
    println!("\nWARNING: This will overwrite existing baselines!");
    println!("Only run this after intentional quality improvements.\n");

    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let categories = vec!["academic", "diverse", "forms", "government", "mixed"];
    let mut updated_count = 0;

    for category in categories {
        let pdfs = match loader.list_pdfs(category) {
            Ok(p) => p,
            Err(_) => continue,
        };

        println!("\n[{}] Checking {} PDFs...", category.to_uppercase(), pdfs.len());

        for pdf_path in pdfs.iter().take(10) {
            let filename = pdf_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown");

            // Only update if golden file exists
            if !manager.has_golden_file(pdf_path) {
                continue;
            }

            // Extract current text
            let extracted = match PdfDocument::open(pdf_path) {
                Ok(mut doc) => match extract_all_pages(&mut doc) {
                    Ok(text) => text,
                    Err(_) => continue,
                },
                Err(_) => continue,
            };

            // Load existing golden file
            let golden = match manager.load_golden_file(pdf_path) {
                Ok(g) => g,
                Err(_) => continue,
            };

            // Compare quality scores
            let current_quality = analyze_quality(&extracted).quality_score;
            let golden_quality = analyze_quality(&golden.extracted_text).quality_score;

            // If current extraction is better, update baseline
            if current_quality > golden_quality {
                println!(
                    "  UPDATING {}: quality improved from {:.1} to {:.1}",
                    filename, golden_quality, current_quality
                );

                if let Ok(_) = manager.save_golden_file(pdf_path, category, &extracted) {
                    updated_count += 1;
                }
            }
        }
    }

    println!("\n{}", "=".repeat(80));
    println!("Updated {} baseline files", updated_count);
    println!("Review changes with: git diff tests/golden_files/");
    println!("{}", "=".repeat(80));
}

// ============================================================================
// Diagnostic Tests
// ============================================================================

/// Show detailed comparison for a specific category.
/// Useful for investigating failures.
#[test]
#[ignore]
fn test_regression_diagnostic_academic() {
    println!("\n");
    println!("{}", "=".repeat(80));
    println!("DIAGNOSTIC: Academic Category Detailed Analysis");
    println!("{}", "=".repeat(80));

    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    let pdfs = loader.list_pdfs("academic").unwrap_or_default();

    for pdf_path in pdfs.iter().take(5) {
        let filename = pdf_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        println!("\n--- {} ---", filename);

        if !manager.has_golden_file(pdf_path) {
            println!("  No baseline available");
            continue;
        }

        let golden = match manager.load_golden_file(pdf_path) {
            Ok(g) => g,
            Err(e) => {
                println!("  Error loading baseline: {}", e);
                continue;
            },
        };

        let extracted = match PdfDocument::open(pdf_path) {
            Ok(mut doc) => match extract_all_pages(&mut doc) {
                Ok(text) => text,
                Err(e) => {
                    println!("  Error extracting: {}", e);
                    continue;
                },
            },
            Err(e) => {
                println!("  Error opening PDF: {}", e);
                continue;
            },
        };

        let comparison = manager.compare_extraction(&extracted, &golden);
        let quality = analyze_quality(&extracted);

        println!("  Baseline: {} chars, {} words", golden.char_count, golden.word_count);
        println!(
            "  Current:  {} chars, {} words",
            extracted.chars().count(),
            extracted.split_whitespace().count()
        );
        println!("  Hash match: {}", comparison.hash_match);
        println!("  Status: {:?}", comparison.status);
        println!("  Quality score: {:.1}/10", quality.quality_score);

        if let Some(ctx) = &comparison.diff_context {
            println!("  First diff: {}", ctx);
        }
    }
}

/// Verify regression test infrastructure is available.
#[test]
fn test_regression_infrastructure_available() {
    let loader = CorpusLoader::default();
    let manager = GoldenFileManager::default();

    // Verify we can list categories
    let categories = loader.list_categories();
    assert!(categories.is_ok(), "Should be able to list categories");

    // Verify manager exists
    let _ = manager;

    println!("Regression test infrastructure is available.");
}
