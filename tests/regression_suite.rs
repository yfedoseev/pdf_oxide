//! Real PDF regression testing suite using curated test fixtures.
//!
//! This suite tests extraction quality on real PDFs to prevent regressions in:
//! - Fix #1: Word fusion (36+ → 0 instances)
//! - Fix #2: Empty bold markers (0 instances)
//! - Fix #3: Negative gap handling (0 text corruption)
//! - Phase 3: Table detection
//! - Phase 5: Adaptive threshold algorithm
//! - Phase 6: Production validation
//!
//! Two test tiers:
//! - Quick: 5 PDFs, ~2-3 minutes (every PR)
//! - Comprehensive: 15 PDFs, ~5-6 minutes (PR merge only)

#[path = "quality_metrics.rs"]
mod quality_metrics;

use pdf_oxide::converters::{ConversionOptions, MarkdownConverter};
use pdf_oxide::document::PdfDocument;
use pdf_oxide::extractors::SpanMergingConfig;
use quality_metrics::*;
use std::path::PathBuf;

const FIXTURES_DIR: &str = "tests/fixtures/regression";

/// Test mode for regression suite
#[derive(Debug, Clone, Copy)]
enum TestMode {
    Quick,
    Comprehensive,
}

impl TestMode {
    fn name(&self) -> &str {
        match self {
            TestMode::Quick => "Quick",
            TestMode::Comprehensive => "Comprehensive",
        }
    }
}

/// Extract markdown from a PDF file
fn extract_markdown(
    pdf_path: &str,
    config: SpanMergingConfig,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut doc = PdfDocument::open(pdf_path)?;
    let converter = MarkdownConverter::new();
    let options = ConversionOptions::default();

    let mut full_markdown = String::new();
    let page_count = doc.page_count()?;

    // Extract text from first 5 pages (sufficient for regression detection)
    for page_num in 0..std::cmp::min(5, page_count) {
        let spans = doc.extract_spans_with_config(page_num, config.clone())?;
        let page_md = converter.convert_page_from_spans(&spans, &options)?;
        full_markdown.push_str(&page_md);
        full_markdown.push_str("\n\n");
    }

    Ok(full_markdown)
}

/// Run regression tests on a set of PDFs
fn run_regression_tests(pdfs: &[&str], mode: TestMode) {
    let mut all_passed = true;
    let mut total_quality_score = 0.0;
    let mut failed_pdfs = Vec::new();

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  PDF Extraction Quality Regression Suite                    ║");
    println!("║  Mode: {}                                            ║", mode.name());
    println!(
        "║  PDFs to test: {}                                               ║",
        pdfs.len()
    );
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    for (i, pdf_name) in pdfs.iter().enumerate() {
        let pdf_path = PathBuf::from(FIXTURES_DIR).join(pdf_name);
        println!("[{}/{}] Testing: {}", i + 1, pdfs.len(), pdf_name);

        match extract_and_analyze(&pdf_path, pdf_name) {
            Ok(metrics) => {
                total_quality_score += metrics.quality_score;
                print_metrics(&metrics);

                // Critical assertions
                let mut pdf_failed = false;

                // Check for true regressions (High/Medium confidence fusions)
                // PDF structure defects (PdfStructure confidence) are allowed
                let true_regressions: Vec<_> = metrics
                    .word_fusions
                    .iter()
                    .filter(|f| {
                        matches!(f.confidence, FusionConfidence::High | FusionConfidence::Medium)
                    })
                    .collect();

                let pdf_defects: Vec<_> = metrics
                    .word_fusions
                    .iter()
                    .filter(|f| matches!(f.confidence, FusionConfidence::PdfStructure))
                    .collect();

                if !true_regressions.is_empty() {
                    eprintln!(
                        "  ❌ FAIL: {} word fusion regressions detected (Fix #1 regression)",
                        true_regressions.len()
                    );
                    for fusion in true_regressions.iter().take(3) {
                        eprintln!("      Line {}: \"{}\"", fusion.line_number, fusion.text);
                    }
                    pdf_failed = true;
                }

                if !pdf_defects.is_empty() {
                    println!(
                        "  ℹ️  INFO: {} PDF structure defects (expected, not regressions)",
                        pdf_defects.len()
                    );
                    for defect in pdf_defects.iter().take(2) {
                        println!(
                            "      Line {}: \"{}\" (single-string TJ encoding)",
                            defect.line_number, defect.text
                        );
                    }
                }

                if metrics.empty_bold_markers > 0 {
                    eprintln!(
                        "  ❌ FAIL: {} empty bold markers detected (Fix #2 regression)",
                        metrics.empty_bold_markers
                    );
                    pdf_failed = true;
                }

                if metrics.quality_score < 8.0 {
                    eprintln!(
                        "  ❌ FAIL: Quality score {:.1} < 8.0 (below threshold)",
                        metrics.quality_score
                    );
                    pdf_failed = true;
                }

                if !pdf_failed {
                    println!("  ✅ PASS");
                } else {
                    all_passed = false;
                    failed_pdfs.push(*pdf_name);
                }
            },
            Err(e) => {
                eprintln!("  ❌ ERROR: {}", e);
                all_passed = false;
                failed_pdfs.push(*pdf_name);
            },
        }
        println!();
    }

    // Print summary
    println!("════════════════════════════════════════════════════════════════");
    println!("Average Quality Score: {:.2}/10.0", total_quality_score / pdfs.len() as f32);
    if !failed_pdfs.is_empty() {
        println!("Failed PDFs: {}", failed_pdfs.join(", "));
    }
    println!("════════════════════════════════════════════════════════════════\n");

    if !all_passed {
        panic!("Regression suite failed on {} PDFs. See details above.", failed_pdfs.len());
    }
}

/// Extract and analyze a single PDF
fn extract_and_analyze(
    pdf_path: &PathBuf,
    _pdf_name: &str,
) -> Result<QualityMetrics, Box<dyn std::error::Error>> {
    let markdown =
        extract_markdown(pdf_path.to_str().ok_or("Invalid path")?, SpanMergingConfig::adaptive())?;

    Ok(analyze_quality(&markdown))
}

/// Print metrics for a PDF
fn print_metrics(metrics: &QualityMetrics) {
    println!("  Quality Score: {:.1}/10.0", metrics.quality_score);
    println!("  Word Fusions: {}", metrics.word_fusions.len());
    println!("  Empty Bold Markers: {}", metrics.empty_bold_markers);
    println!("  Spurious Spaces: {}", metrics.spurious_spaces.len());
    println!("  Tables Detected: {}", metrics.tables_detected);
}

// ============================================================================
// CORE REGRESSION SUITE (Quick - 5 PDFs, ~2-3 minutes)
// ============================================================================

/// Core regression suite - quick validation on 5 representative PDFs
/// This runs on every PR and should complete in < 3 minutes
#[test]
fn test_core_regression_suite() {
    let pdfs = vec![
        "policy/Anti-bribery and Corruption Policy Template (UK).pdf", // Fix #1 primary
        "policy/Diligent Security Policy.pdf",                         // Fix #1, Phase 3
        "policy/Code of Conduct Policy Template (EU).pdf",             // Fix #2, Fix #3
        "academic/arxiv_2510.21165v1.pdf",                             // Phase 5, Phase 6
        "mixed/7A3MBRLFC6OU5KGMFIDEQPUOQTROBYUS.pdf",                  // Quick test
    ];

    run_regression_tests(&pdfs, TestMode::Quick);
}

// ============================================================================
// COMPREHENSIVE REGRESSION SUITE (Full - 15 PDFs, ~5-6 minutes)
// ============================================================================

/// Comprehensive regression suite - full validation on all 15 curated PDFs
/// This runs on PR merge and validates all phases of fixes comprehensively
#[test]
#[ignore] // Run with --include-ignored for full validation
fn test_comprehensive_regression_suite() {
    let pdfs = vec![
        // Policy documents (Fix #1, #2, #3)
        "policy/Anti-bribery and Corruption Policy Template (UK).pdf",
        "policy/Code of Conduct Policy Template (EU).pdf",
        "policy/Conflict of Interest Policy Template.pdf",
        "policy/Diligent Security Policy.pdf",
        "policy/Template - AI Guiding Policy.pdf",
        "policy/diligent_ai_acceptable_use_policy_1.0.pdf",
        // Academic documents (Phase 5, Phase 6)
        "academic/arxiv_2510.21165v1.pdf",
        "academic/arxiv_2510.21912v1.pdf",
        "academic/arxiv_2510.22293v1.pdf",
        // Mixed documents (Fix #3, Phase 3, Phase 5)
        "mixed/5JWNPTKTIAPTHTEGVKW7WVNBDBKQMRJO.pdf",
        "mixed/5PFVA6CO2FP66IJYJJ4YMWOLK5EHRCCD.pdf",
        "mixed/7A3MBRLFC6OU5KGMFIDEQPUOQTROBYUS.pdf",
        "mixed/7GB7EXTYK2SHE3R3CBCOYKLOQT4CMEAF.pdf",
        "mixed/7N6KRBZIEFV4F5QLLW3GBF6LKNNWSWVB.pdf",
        // Government documents (Phase 3, complex tables)
        "government/cfr_excerpt.pdf",
    ];

    run_regression_tests(&pdfs, TestMode::Comprehensive);
}

// ============================================================================
// FOCUSED ISSUE TESTS
// ============================================================================

/// Test Fix #1: Word Fusion Regression
///
/// Validates that Fix #1 (adaptive threshold) prevents word fusion in policy documents.
/// This was the critical regression issue discovered in Phase 4.
#[test]
fn test_word_fusion_regression_policy() {
    let policy_pdfs = vec![
        "policy/Anti-bribery and Corruption Policy Template (UK).pdf",
        "policy/Code of Conduct Policy Template (EU).pdf",
        "policy/Conflict of Interest Policy Template.pdf",
    ];

    for pdf_name in policy_pdfs {
        let pdf_path = PathBuf::from(FIXTURES_DIR).join(pdf_name);
        let markdown = extract_markdown(pdf_path.to_str().unwrap(), SpanMergingConfig::adaptive())
            .expect("Failed to extract markdown");

        let metrics = analyze_quality(&markdown);

        assert_eq!(
            metrics.word_fusions.len(),
            0,
            "Found {} word fusions in {}. Fix #1 regression!",
            metrics.word_fusions.len(),
            pdf_name
        );
    }
}

/// Test Fix #2: Empty Bold Markers Regression
///
/// Validates that Fix #2 (conservative bold markers) prevents empty bold markers.
#[test]
fn test_empty_bold_markers_regression() {
    let styled_pdfs = vec![
        "policy/Code of Conduct Policy Template (EU).pdf",
        "policy/Diligent Security Policy.pdf",
    ];

    for pdf_name in styled_pdfs {
        let pdf_path = PathBuf::from(FIXTURES_DIR).join(pdf_name);
        let markdown = extract_markdown(pdf_path.to_str().unwrap(), SpanMergingConfig::adaptive())
            .expect("Failed to extract markdown");

        let metrics = analyze_quality(&markdown);

        assert_eq!(
            metrics.empty_bold_markers, 0,
            "Found {} empty bold markers in {}. Fix #2 regression!",
            metrics.empty_bold_markers, pdf_name
        );
    }
}

/// Test Phase 5: Adaptive Threshold Effectiveness
///
/// Validates that adaptive threshold produces quality extraction for different document types.
#[test]
fn test_adaptive_threshold_effectiveness() {
    let test_cases = vec![
        ("policy/Anti-bribery and Corruption Policy Template (UK).pdf", 9.0, "policy"),
        ("academic/arxiv_2510.21165v1.pdf", 8.5, "academic"),
        ("mixed/5JWNPTKTIAPTHTEGVKW7WVNBDBKQMRJO.pdf", 8.0, "mixed"),
    ];

    for (pdf_name, min_score, doc_type) in test_cases {
        let pdf_path = PathBuf::from(FIXTURES_DIR).join(pdf_name);
        let markdown = extract_markdown(pdf_path.to_str().unwrap(), SpanMergingConfig::adaptive())
            .expect("Failed to extract markdown");

        let metrics = analyze_quality(&markdown);

        assert!(
            metrics.quality_score >= min_score,
            "{} document '{}' quality score {:.1} < minimum {:.1}",
            doc_type,
            pdf_name,
            metrics.quality_score,
            min_score
        );

        assert_eq!(
            metrics.word_fusions.len(),
            0,
            "{} document should have 0 word fusions",
            doc_type
        );
    }
}

/// Test Backward Compatibility
///
/// Validates that default configuration still works (adaptive is opt-in).
#[test]
fn test_backward_compatibility_default_config() {
    let pdf_path = PathBuf::from(FIXTURES_DIR)
        .join("policy/Anti-bribery and Corruption Policy Template (UK).pdf");

    // Default configuration should not use adaptive threshold
    let default_config = SpanMergingConfig::default();
    assert!(
        !default_config.use_adaptive_threshold,
        "Adaptive threshold should be opt-in by default"
    );

    // Both default and adaptive should extract some text
    let default_md = extract_markdown(pdf_path.to_str().unwrap(), SpanMergingConfig::default())
        .expect("Failed with default config");

    let adaptive_md = extract_markdown(pdf_path.to_str().unwrap(), SpanMergingConfig::adaptive())
        .expect("Failed with adaptive config");

    assert!(!default_md.is_empty(), "Default config should extract text");
    assert!(!adaptive_md.is_empty(), "Adaptive config should extract text");
}

/// Test Configuration Factory Methods
///
/// Validates that configuration factory methods produce correct settings.
#[test]
fn test_configuration_factories() {
    let default = SpanMergingConfig::default();
    assert!(!default.use_adaptive_threshold);

    let adaptive = SpanMergingConfig::adaptive();
    assert!(adaptive.use_adaptive_threshold);

    // Verify adaptive config is set
    assert!(adaptive.adaptive_config.is_some());
}
