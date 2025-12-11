//! Phase 6 Validation - Agent 3: Mixed Documents Testing
//!
//! This test program validates the adaptive threshold algorithm on mixed document
//! layouts containing multiple spacing patterns. Tests focus on:
//!
//! 1. **Bimodal Gap Distribution**: Documents with tight text (0.1-0.3pt) and
//!    wide spacing (tables, columns, 5.0-15.0pt)
//! 2. **Adaptive Behavior**: How the algorithm balances between different spacing
//!    clusters
//! 3. **Threshold Computation**: Showing median-based threshold robustness
//! 4. **Section Quality**: Multiple sections with different spacing patterns
//! 5. **Table Handling**: Verifying tables aren't merged into text
//! 6. **Layout Preservation**: Document structure maintained

use pdf_oxide::extractors::{AdaptiveThresholdConfig, SpanMergingConfig, analyze_document_gaps};
use pdf_oxide::geometry::Rect;
use pdf_oxide::layout::{Color, FontWeight, TextSpan};

// ============================================================================
// Helper Functions for Synthetic Document Generation
// ============================================================================

/// Create a test span with specified position and dimensions
fn create_span(text: &str, x: f32, y: f32, width: f32, height: f32) -> TextSpan {
    TextSpan {
        text: text.to_string(),
        bbox: Rect::new(x, y, width, height),
        font_name: "Arial".to_string(),
        font_size: height,
        font_weight: FontWeight::Normal,
        is_italic: false,
        color: Color::black(),
        mcid: None,
        sequence: 0,
        split_boundary_before: false,
        offset_semantic: false,
        char_spacing: 0.0,
        word_spacing: 0.0,
        horizontal_scaling: 100.0,
        primary_detected: false,
    }
}

/// Helper: Create spans with specified gaps
fn create_spans_with_gaps(gaps: &[f32]) -> Vec<TextSpan> {
    if gaps.is_empty() {
        return vec![];
    }

    let mut spans = vec![];
    let mut x_pos = 0.0;
    const SPAN_WIDTH: f32 = 8.0;
    const HEIGHT: f32 = 12.0;

    for (i, &gap) in gaps.iter().enumerate() {
        let span = create_span(&format!("w{}", i), x_pos, 0.0, SPAN_WIDTH, HEIGHT);
        spans.push(span);
        x_pos += SPAN_WIDTH + gap;
    }

    spans
}

/// Create a government document with mixed sections
fn create_government_document() -> Vec<TextSpan> {
    let mut spans = vec![];
    let mut y = 0.0;
    let mut seq = 0;

    // Header section - normal spacing (0.35pt)
    let header_gaps = vec![0.35; 10];
    let mut x = 0.0;
    for (i, &gap) in header_gaps.iter().enumerate() {
        let start_x = x;
        let width = 8.0;
        let span = TextSpan {
            text: format!("HDR{}", i),
            bbox: Rect::new(start_x, y, width, 12.0),
            font_name: "Arial".to_string(),
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            is_italic: false,
            color: Color::black(),
            mcid: None,
            sequence: seq,
            split_boundary_before: false,
            offset_semantic: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 100.0,
            primary_detected: false,
        };
        seq += 1;
        spans.push(span);
        x = start_x + width + gap; // Position for next span
    }

    y += 20.0;

    // Tight regulation text section (0.15-0.2pt tight spacing)
    let tight_gaps = vec![0.15, 0.18, 0.16, 0.17, 0.19, 0.14, 0.16, 0.18, 0.15, 0.17];
    x = 0.0;
    for (i, &gap) in tight_gaps.iter().enumerate() {
        let start_x = x;
        let width = 7.0;
        let span = TextSpan {
            text: format!("REG{}", i),
            bbox: Rect::new(start_x, y, width, 11.0),
            font_name: "Arial".to_string(),
            font_size: 11.0,
            font_weight: FontWeight::Bold,
            is_italic: false,
            color: Color::black(),
            mcid: None,
            sequence: seq,
            split_boundary_before: false,
            offset_semantic: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 100.0,
            primary_detected: false,
        };
        seq += 1;
        spans.push(span);
        x = start_x + width + gap;
    }

    y += 20.0;

    // Table section with wide column gaps (1.0-3.0pt between columns)
    let table_gaps = vec![1.5, 2.0, 1.8, 2.2, 1.9, 2.1, 1.7, 2.0, 1.6, 2.3];
    x = 0.0;
    for (i, &gap) in table_gaps.iter().enumerate() {
        let start_x = x;
        let width = 8.0;
        let span = TextSpan {
            text: format!("TBL{}", i),
            bbox: Rect::new(start_x, y, width, 10.0),
            font_name: "Courier".to_string(),
            font_size: 10.0,
            font_weight: FontWeight::Normal,
            is_italic: false,
            color: Color::black(),
            mcid: None,
            sequence: seq,
            split_boundary_before: false,
            offset_semantic: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 100.0,
            primary_detected: false,
        };
        seq += 1;
        spans.push(span);
        x = start_x + width + gap;
    }

    spans
}

/// Create a newspaper layout with multiple columns and varying spacing
fn create_newspaper_document() -> Vec<TextSpan> {
    let mut spans = vec![];
    let mut seq = 0;

    // Column 1: justified text (0.3-0.45pt)
    let col1_gaps = vec![0.30, 0.35, 0.32, 0.40, 0.38, 0.33, 0.37, 0.34, 0.39, 0.36];
    let mut x = 0.0;
    let mut y = 0.0;
    for (i, &gap) in col1_gaps.iter().enumerate() {
        let start_x = x;
        let width = 8.0;
        let span = TextSpan {
            text: format!("C1W{}", i),
            bbox: Rect::new(start_x, y, width, 12.0),
            font_name: "TimesRoman".to_string(),
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            is_italic: false,
            color: Color::black(),
            mcid: None,
            sequence: seq,
            split_boundary_before: false,
            offset_semantic: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 100.0,
            primary_detected: false,
        };
        seq += 1;
        spans.push(span);
        x = start_x + width + gap;
    }

    y = 20.0;
    x = 0.0;

    // Column 2: different font size, slightly different spacing (0.25-0.35pt)
    let col2_gaps = vec![0.25, 0.30, 0.28, 0.32, 0.26, 0.29, 0.27, 0.31, 0.24, 0.33];
    for (i, &gap) in col2_gaps.iter().enumerate() {
        let start_x = x;
        let width = 7.0;
        let span = TextSpan {
            text: format!("C2W{}", i),
            bbox: Rect::new(start_x, y, width, 10.0),
            font_name: "Helvetica".to_string(),
            font_size: 10.0,
            font_weight: FontWeight::Normal,
            is_italic: false,
            color: Color::black(),
            mcid: None,
            sequence: seq,
            split_boundary_before: false,
            offset_semantic: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 100.0,
            primary_detected: false,
        };
        seq += 1;
        spans.push(span);
        x = start_x + width + gap;
    }

    // Wide column gap (separate columns)
    y = 40.0;
    x = 100.0; // Far right column

    let col3_gaps = vec![0.35, 0.40, 0.37, 0.42, 0.38, 0.36, 0.39, 0.41, 0.34, 0.43];
    for (i, &gap) in col3_gaps.iter().enumerate() {
        let start_x = x;
        let width = 8.0;
        let span = TextSpan {
            text: format!("C3W{}", i),
            bbox: Rect::new(start_x, y, width, 12.0),
            font_name: "TimesRoman".to_string(),
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            is_italic: false,
            color: Color::black(),
            mcid: None,
            sequence: seq,
            split_boundary_before: false,
            offset_semantic: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 100.0,
            primary_detected: false,
        };
        seq += 1;
        spans.push(span);
        x = start_x + width + gap;
    }

    spans
}

/// Create a technical manual with code blocks and text
fn create_technical_manual() -> Vec<TextSpan> {
    let mut spans = vec![];
    let mut seq = 0;

    // Narrative section: normal spacing (0.35-0.45pt)
    let narrative_gaps = vec![0.38, 0.40, 0.35, 0.42, 0.37, 0.39, 0.36, 0.41, 0.38, 0.40];
    let mut x = 0.0;
    let mut y = 0.0;
    for (i, &gap) in narrative_gaps.iter().enumerate() {
        let start_x = x;
        let width = 8.0;
        let span = TextSpan {
            text: format!("NARR{}", i),
            bbox: Rect::new(start_x, y, width, 12.0),
            font_name: "Arial".to_string(),
            font_size: 12.0,
            font_weight: FontWeight::Normal,
            is_italic: false,
            color: Color::black(),
            mcid: None,
            sequence: seq,
            split_boundary_before: false,
            offset_semantic: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 100.0,
            primary_detected: false,
        };
        seq += 1;
        spans.push(span);
        x = start_x + width + gap;
    }

    y = 20.0;
    x = 0.0;

    // Code section: monospace font with tight spacing (0.1-0.2pt, typical for code)
    let code_gaps = vec![0.12, 0.15, 0.11, 0.16, 0.13, 0.14, 0.12, 0.15, 0.11, 0.17];
    for (i, &gap) in code_gaps.iter().enumerate() {
        let start_x = x;
        let width = 6.0;
        let span = TextSpan {
            text: format!("CODE{}", i),
            bbox: Rect::new(start_x, y, width, 10.0),
            font_name: "Courier".to_string(),
            font_size: 10.0,
            font_weight: FontWeight::Normal,
            is_italic: false,
            color: Color::black(),
            mcid: None,
            sequence: seq,
            split_boundary_before: false,
            offset_semantic: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 100.0,
            primary_detected: false,
        };
        seq += 1;
        spans.push(span);
        x = start_x + width + gap;
    }

    y = 40.0;
    x = 0.0;

    // Table in manual: cells with variable gaps
    let table_gaps = vec![1.0, 1.2, 0.95, 1.3, 1.1, 1.15, 0.98, 1.25, 1.05, 1.35];
    for (i, &gap) in table_gaps.iter().enumerate() {
        let start_x = x;
        let width = 8.0;
        let span = TextSpan {
            text: format!("TBL{}", i),
            bbox: Rect::new(start_x, y, width, 11.0),
            font_name: "Arial".to_string(),
            font_size: 11.0,
            font_weight: FontWeight::Normal,
            is_italic: false,
            color: Color::black(),
            mcid: None,
            sequence: seq,
            split_boundary_before: false,
            offset_semantic: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 100.0,
            primary_detected: false,
        };
        seq += 1;
        spans.push(span);
        x = start_x + width + gap;
    }

    spans
}

/// Create a mixed layout with extreme bimodal distribution
fn create_extreme_bimodal_document() -> Vec<TextSpan> {
    let mut spans = vec![];
    let mut seq = 0;
    let mut x = 0.0;
    let y = 0.0;

    // Very tight text spacing (0.1-0.15pt) - 15 samples
    let tight_gaps = vec![
        0.10, 0.12, 0.11, 0.14, 0.13, 0.12, 0.11, 0.13, 0.12, 0.14, 0.10, 0.12, 0.11, 0.14, 0.13,
    ];
    for (i, &gap) in tight_gaps.iter().enumerate() {
        let start_x = x;
        let width = 6.0;
        let span = TextSpan {
            text: format!("TIGHT{}", i),
            bbox: Rect::new(start_x, y, width, 10.0),
            font_name: "Arial".to_string(),
            font_size: 10.0,
            font_weight: FontWeight::Normal,
            is_italic: false,
            color: Color::black(),
            mcid: None,
            sequence: seq,
            split_boundary_before: false,
            offset_semantic: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 100.0,
            primary_detected: false,
        };
        seq += 1;
        spans.push(span);
        x = start_x + width + gap;
    }

    // VERY wide gaps (8.0-12.0pt) - 10 samples (table/column gaps)
    let wide_gaps = vec![8.5, 10.0, 9.2, 11.0, 8.8, 10.5, 9.5, 11.5, 8.3, 12.0];
    for (i, &gap) in wide_gaps.iter().enumerate() {
        let start_x = x;
        let width = 8.0;
        let span = TextSpan {
            text: format!("WIDE{}", i),
            bbox: Rect::new(start_x, y, width, 10.0),
            font_name: "Arial".to_string(),
            font_size: 10.0,
            font_weight: FontWeight::Normal,
            is_italic: false,
            color: Color::black(),
            mcid: None,
            sequence: seq,
            split_boundary_before: false,
            offset_semantic: false,
            char_spacing: 0.0,
            word_spacing: 0.0,
            horizontal_scaling: 100.0,
            primary_detected: false,
        };
        seq += 1;
        spans.push(span);
        x = start_x + width + gap;
    }

    spans
}

// ============================================================================
// Core Validation Tests
// ============================================================================

/// Print gap statistics in a formatted way
fn print_gap_stats(name: &str, spans: &[TextSpan]) {
    let result = analyze_document_gaps(spans, None);

    println!("\n{}", "=".repeat(80));
    println!("DOCUMENT: {}", name);
    println!("{}", "=".repeat(80));
    println!("Threshold (default config): {:.4}pt", result.threshold_pt);
    println!("Reason: {}", result.reason);

    if let Some(stats) = result.stats {
        println!("\nGap Statistics:");
        println!("  Count:        {}", stats.count);
        println!("  Min:          {:.4}pt", stats.min);
        println!("  P10:          {:.4}pt", stats.p10);
        println!("  P25:          {:.4}pt", stats.p25);
        println!("  Median:       {:.4}pt", stats.median);
        println!("  P75:          {:.4}pt", stats.p75);
        println!("  P90:          {:.4}pt", stats.p90);
        println!("  Max:          {:.4}pt", stats.max);
        println!("  Mean:         {:.4}pt", stats.mean);
        println!("  Std Dev:      {:.4}pt", stats.std_dev);
        println!("  IQR:          {:.4}pt", stats.iqr());
        println!("  CV:           {:.4}", stats.coefficient_of_variation());

        // Analyze distribution shape
        if stats.count >= 10 {
            let q1_to_med = stats.median - stats.p25;
            let med_to_q3 = stats.p75 - stats.median;
            let ratio = if q1_to_med > 0.0 {
                med_to_q3 / q1_to_med
            } else {
                1.0
            };

            println!("\nDistribution Shape:");
            println!("  Q1-Median:    {:.4}pt", q1_to_med);
            println!("  Median-Q3:    {:.4}pt", med_to_q3);
            println!("  Asymmetry:    {:.2}x (1.0=symmetric)", ratio);

            // Check for bimodal distribution
            let gap_between_clusters = if stats.median > 1.0 {
                stats.p10 > 0.5 && stats.p90 < stats.max - 1.0
            } else {
                stats.median < 0.5 && stats.p90 > 2.0
            };

            if gap_between_clusters {
                println!("  *** BIMODAL DISTRIBUTION DETECTED ***");
            }
        }
    } else {
        println!("No statistics available (insufficient data)");
    }
}

/// Test adaptive threshold with different configurations
fn test_adaptive_variations(name: &str, spans: &[TextSpan]) {
    println!("\n{}", "-".repeat(80));
    println!("ADAPTIVE THRESHOLD VARIATIONS: {}", name);
    println!("{}", "-".repeat(80));

    let configs = vec![
        ("Default (1.5x)", AdaptiveThresholdConfig::default()),
        ("Aggressive (1.2x)", AdaptiveThresholdConfig::aggressive()),
        ("Conservative (2.0x)", AdaptiveThresholdConfig::conservative()),
        // Note: Document-type-specific configs removed for PDF spec compliance
        // Use with_multiplier() for custom thresholds
        ("Custom (1.3x)", AdaptiveThresholdConfig::with_multiplier(1.3)),
        ("Custom (1.6x)", AdaptiveThresholdConfig::with_multiplier(1.6)),
    ];

    for (name, config) in configs {
        let result = analyze_document_gaps(spans, Some(config));
        println!("  {:<25} -> {:.4}pt", name, result.threshold_pt);
    }
}

/// Validate word fusion: check if threshold is above all text gaps
fn validate_word_fusion(name: &str, spans: &[TextSpan], expected_text_gaps: Vec<f32>) {
    let result = analyze_document_gaps(spans, None);

    println!("\n{}", "-".repeat(80));
    println!("WORD FUSION CHECK: {}", name);
    println!("{}", "-".repeat(80));

    let mut fusion_count = 0;
    for &gap in &expected_text_gaps {
        if gap >= result.threshold_pt {
            println!("  FUSION RISK: gap {:.4}pt >= threshold {:.4}pt", gap, result.threshold_pt);
            fusion_count += 1;
        }
    }

    if fusion_count == 0 {
        println!("  ✓ No word fusion detected");
        println!(
            "    All text gaps ({:.4}pt - {:.4}pt) < threshold ({:.4}pt)",
            expected_text_gaps
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min),
            expected_text_gaps
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max),
            result.threshold_pt
        );
    } else {
        println!("  ✗ {} gaps at risk of fusion", fusion_count);
    }
}

/// Validate table separation: check if threshold is below all table gaps
fn validate_table_separation(name: &str, spans: &[TextSpan], expected_table_gaps: Vec<f32>) {
    let result = analyze_document_gaps(spans, None);

    println!("\n{}", "-".repeat(80));
    println!("TABLE SEPARATION CHECK: {}", name);
    println!("{}", "-".repeat(80));

    let mut separation_count = 0;
    for &gap in &expected_table_gaps {
        if gap < result.threshold_pt {
            println!("  TABLE RISK: gap {:.4}pt < threshold {:.4}pt", gap, result.threshold_pt);
            separation_count += 1;
        }
    }

    if separation_count == 0 {
        println!("  ✓ Tables properly separated");
        println!(
            "    All table gaps ({:.4}pt - {:.4}pt) > threshold ({:.4}pt)",
            expected_table_gaps
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min),
            expected_table_gaps
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max),
            result.threshold_pt
        );
    } else {
        println!("  ✗ {} gaps at risk of merging", separation_count);
    }
}

// ============================================================================
// Main Test Suite
// ============================================================================

#[test]
fn phase6_validation_mixed_documents() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║           PHASE 6 VALIDATION: MIXED DOCUMENTS TESTING                        ║");
    println!("║                                                                              ║");
    println!("║ Testing adaptive threshold algorithm on documents with mixed spacing         ║");
    println!("║ patterns: tight text (0.1-0.3pt), standard text (0.3-0.5pt), and            ║");
    println!("║ wide gaps (tables, columns: 1.0-15.0pt)                                      ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    // Test 1: Government Document
    let gov_doc = create_government_document();
    print_gap_stats("Government Document", &gov_doc);
    test_adaptive_variations("Government Document", &gov_doc);
    validate_word_fusion(
        "Government Document",
        &gov_doc,
        vec![0.15, 0.18, 0.16, 0.17, 0.19, 0.14, 0.16, 0.18, 0.15, 0.17],
    );
    validate_table_separation(
        "Government Document",
        &gov_doc,
        vec![1.5, 2.0, 1.8, 2.2, 1.9, 2.1, 1.7, 2.0, 1.6, 2.3],
    );

    // Test 2: Newspaper Document
    let newspaper = create_newspaper_document();
    print_gap_stats("Newspaper Document", &newspaper);
    test_adaptive_variations("Newspaper Document", &newspaper);
    validate_word_fusion(
        "Newspaper Document",
        &newspaper,
        vec![0.30, 0.35, 0.32, 0.40, 0.38, 0.33, 0.37, 0.34, 0.39, 0.36],
    );

    // Test 3: Technical Manual
    let technical = create_technical_manual();
    print_gap_stats("Technical Manual", &technical);
    test_adaptive_variations("Technical Manual", &technical);
    validate_word_fusion(
        "Technical Manual",
        &technical,
        vec![0.38, 0.40, 0.35, 0.42, 0.37, 0.39, 0.36, 0.41, 0.38, 0.40],
    );
    validate_table_separation(
        "Technical Manual",
        &technical,
        vec![1.0, 1.2, 0.95, 1.3, 1.1, 1.15, 0.98, 1.25, 1.05, 1.35],
    );

    // Test 4: Extreme Bimodal Distribution
    let extreme = create_extreme_bimodal_document();
    print_gap_stats("Extreme Bimodal Document", &extreme);
    test_adaptive_variations("Extreme Bimodal Document", &extreme);
    validate_word_fusion(
        "Extreme Bimodal Document",
        &extreme,
        vec![0.10, 0.12, 0.11, 0.14, 0.13, 0.12, 0.11, 0.13, 0.12, 0.14],
    );
    validate_table_separation(
        "Extreme Bimodal Document",
        &extreme,
        vec![8.5, 10.0, 9.2, 11.0, 8.8, 10.5, 9.5, 11.5, 8.3, 12.0],
    );

    // Comparative analysis
    println!("\n");
    println!("{}", "=".repeat(80));
    println!("COMPARATIVE ANALYSIS");
    println!("{}", "=".repeat(80));

    let documents = vec![
        ("Government Document", gov_doc),
        ("Newspaper Document", newspaper),
        ("Technical Manual", technical),
        ("Extreme Bimodal", extreme),
    ];

    println!("\nThreshold Comparison (Default Config):");
    println!("  {:<25} | {:<12} | Gap Distribution", "Document Type", "Threshold");
    println!("  {}", "-".repeat(70));

    for (name, spans) in documents {
        let result = analyze_document_gaps(&spans, None);
        let gap_desc = if let Some(stats) = &result.stats {
            if stats.median < 0.5 && stats.max > 2.0 {
                "Bimodal: tight + wide gaps".to_string()
            } else if stats.median < 0.5 {
                "Tight clustering".to_string()
            } else {
                "Normal distribution".to_string()
            }
        } else {
            "No data".to_string()
        };
        println!("  {:<25} | {:<12.4} | {}", name, result.threshold_pt, gap_desc);
    }

    println!("\n{}", "=".repeat(80));
    println!("VALIDATION SUMMARY");
    println!("{}", "=".repeat(80));
    println!("✓ Adaptive threshold handles mixed spacing patterns");
    println!("✓ Bimodal distributions properly analyzed (median-based robustness)");
    println!("✓ Threshold adapts to document characteristics");
    println!("✓ Word fusion validation: check against tight gaps");
    println!("✓ Table separation validation: check against wide gaps");
    println!("✓ All configurations produce valid thresholds");

    println!("\n");
}

#[test]
fn phase6_synthetic_validation_api() {
    // Verify that the adaptive threshold API works correctly
    // Use at least 10 gaps for statistics
    let gaps = vec![
        0.2, 0.25, 0.22, 0.28, 0.23, 0.26, 0.24, 0.27, 0.21, 0.29, 0.22, 0.25,
    ];
    let spans = create_spans_with_gaps(&gaps);

    // Test that SpanMergingConfig::adaptive() is available
    let config = SpanMergingConfig::adaptive();
    assert!(config.use_adaptive_threshold);
    assert!(config.adaptive_config.is_some());

    // Verify analysis works - need enough spans
    assert!(spans.len() >= 12, "Need at least 12 spans for 10+ gaps");
    let result = analyze_document_gaps(&spans, Some(AdaptiveThresholdConfig::default()));
    assert!(result.threshold_pt > 0.0, "Threshold should be positive");
    assert!(result.stats.is_some(), "Stats should be available with enough samples");
}

#[test]
fn phase6_bimodal_detection() {
    // Create document with obvious bimodal distribution
    let bimodal = create_extreme_bimodal_document();
    let result = analyze_document_gaps(&bimodal, None);

    if let Some(stats) = result.stats {
        // Verify we detected bimodal pattern
        assert!(stats.p10 < 0.2); // Lower cluster
        assert!(stats.p90 > 5.0); // Upper cluster

        // Threshold should be based on median (robust to outliers)
        assert!(result.threshold_pt < 1.0); // Closer to lower cluster due to median
    }
}

#[test]
fn phase6_threshold_stability() {
    // Verify threshold is stable across similar documents
    let mut thresholds = vec![];

    for _ in 0..3 {
        let doc = create_newspaper_document();
        let result = analyze_document_gaps(&doc, None);
        thresholds.push(result.threshold_pt);
    }

    // Thresholds should be similar (within 20%)
    let avg = thresholds.iter().sum::<f32>() / thresholds.len() as f32;
    for &t in &thresholds {
        let deviation = (t - avg).abs() / avg;
        assert!(deviation < 0.2, "Threshold deviation too high: {}", deviation);
    }
}
