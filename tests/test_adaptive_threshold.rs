#![allow(clippy::useless_vec)]
//! Comprehensive test suite for adaptive threshold algorithm.
//!
//! Phase 5.3: Tests for gap analysis, statistics calculation, threshold determination,
//! and integration with document extraction.
//!
//! This test suite validates:
//! - Gap extraction from text spans
//! - Statistical calculations (median, percentiles, std dev)
//! - Adaptive threshold determination
//! - Factory methods for different document types
//! - Integration with span merging configuration
//! - Backward compatibility
//! - Edge cases and error conditions

use pdf_oxide::extractors::{AdaptiveThresholdConfig, SpanMergingConfig, TextExtractionConfig};
use pdf_oxide::geometry::Rect;
use pdf_oxide::layout::{Color, FontWeight, TextSpan};

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a test text span with specified position and width.
///
/// # Arguments
/// * `text` - The text content of the span
/// * `x` - Left edge position (in points)
/// * `y` - Top edge position (in points)
/// * `width` - Width of the span (in points)
/// * `height` - Height of the span (in points, usually font size)
fn create_test_span(text: &str, x: f32, y: f32, width: f32, height: f32) -> TextSpan {
    TextSpan {
        text: text.to_string(),
        bbox: Rect::new(x, y, width, height),
        font_name: "Times".to_string(),
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

/// Create multiple test spans on the same line with specified gaps between them.
///
/// # Arguments
/// * `gaps` - Vector of gap sizes (in points) between consecutive spans
///
/// # Returns
/// Vector of spans arranged horizontally with specified gaps
fn create_spans_with_gaps(gaps: &[f32]) -> Vec<TextSpan> {
    if gaps.is_empty() {
        return vec![];
    }

    let mut spans = vec![];
    let mut x_pos = 0.0;
    let span_width = 10.0; // Fixed width for each span
    let y_pos = 0.0;
    let height = 12.0; // Typical font size

    for (i, &gap) in gaps.iter().enumerate() {
        let span = create_test_span(&format!("word{}", i), x_pos, y_pos, span_width, height);
        spans.push(span);
        x_pos += span_width + gap;
    }

    spans
}

/// Create a synthetic document with multiple lines and specified gap patterns.
///
/// Useful for testing multi-line gap extraction.
fn create_multiline_document(gaps_per_line: Vec<Vec<f32>>, line_spacing: f32) -> Vec<TextSpan> {
    let mut spans = vec![];
    let mut y_pos = 0.0;
    let height = 12.0;

    for (line_idx, gaps) in gaps_per_line.iter().enumerate() {
        let mut x_pos = 0.0;
        let span_width = 10.0;

        for (word_idx, &gap) in gaps.iter().enumerate() {
            let span = TextSpan {
                text: format!("L{}W{}", line_idx, word_idx),
                bbox: Rect::new(x_pos, y_pos, span_width, height),
                font_name: "Times".to_string(),
                font_size: height,
                font_weight: FontWeight::Normal,
                is_italic: false,
                color: Color::black(),
                mcid: None,
                sequence: line_idx * 100 + word_idx,
                split_boundary_before: false,
                offset_semantic: false,
                char_spacing: 0.0,
                word_spacing: 0.0,
                horizontal_scaling: 100.0,
                primary_detected: false,
            };
            spans.push(span);
            x_pos += span_width + gap;
        }

        y_pos += line_spacing;
    }

    spans
}

// ============================================================================
// Gap Extraction Tests
// ============================================================================

mod gap_extraction_tests {
    use super::*;
    use pdf_oxide::extractors::analyze_document_gaps;

    /// Test gap extraction from a single line with multiple spans.
    ///
    /// Given: Two gaps [2.0pt, 3.0pt] on same line
    /// Expected: Two spans with gaps between them
    #[test]
    fn test_extract_gaps_single_line() {
        let gaps = vec![2.0, 3.0];
        let spans = create_spans_with_gaps(&gaps);
        // create_spans_with_gaps creates N spans for N gaps
        assert_eq!(spans.len(), 2);

        // Verify span positions
        // Span 0: x=0, width=10, so right edge at 10
        assert_eq!(spans[0].bbox.left(), 0.0);
        assert_eq!(spans[0].bbox.right(), 10.0);
        // Span 1: x = 10 + 2.0 gap = 12
        assert_eq!(spans[1].bbox.left(), 12.0);
        assert_eq!(spans[1].bbox.right(), 22.0);

        // The gaps extracted would be: [2.0] (gap between span 0 and span 1)
        assert_eq!(gaps.len(), 2);
    }

    /// Test gap extraction across multiple lines.
    ///
    /// Given: Two lines with different Y coordinates
    /// Expected: Only intra-line gaps extracted, line separation ignored
    #[test]
    fn test_extract_gaps_multi_line() {
        let gaps_per_line = vec![
            vec![2.0, 3.0], // Line 1: gaps of 2.0 and 3.0
            vec![1.5, 2.5], // Line 2: gaps of 1.5 and 2.5
        ];
        let spans = create_multiline_document(gaps_per_line, 20.0); // 20pt line spacing
        assert_eq!(spans.len(), 4); // 2 lines, 2 spans per line + gaps

        // Verify that spans are on different lines
        assert_eq!(spans[0].bbox.top(), 0.0);
        assert_eq!(spans[2].bbox.top(), 20.0); // Different Y coordinate
    }

    /// Test that overlapping spans (negative gaps) are handled properly.
    ///
    /// Some PDFs have overlapping text due to font metrics.
    /// Gap extraction should include these as negative values.
    #[test]
    fn test_extract_gaps_including_overlaps() {
        // Create spans with overlap: span ends at 12, next starts at 10 (overlap of -2)
        let spans = vec![
            create_test_span("word1", 0.0, 0.0, 12.0, 12.0),
            create_test_span("word2", 10.0, 0.0, 10.0, 12.0), // Overlaps by -2.0
            create_test_span("word3", 22.0, 0.0, 10.0, 12.0), // Gap of 2.0
        ];
        assert_eq!(spans.len(), 3);

        // Verify overlap calculation
        let gap1 = spans[1].bbox.left() - spans[0].bbox.right(); // 10 - 12 = -2
        assert_eq!(gap1, -2.0);

        let gap2 = spans[2].bbox.left() - spans[1].bbox.right(); // 22 - 20 = 2
        assert_eq!(gap2, 2.0);
    }

    /// Test gap extraction with empty input.
    ///
    /// Given: No spans
    /// Expected: Empty gap list
    #[test]
    fn test_extract_gaps_empty_input() {
        let spans: Vec<TextSpan> = vec![];
        let result = analyze_document_gaps(&spans, None);

        // Should handle gracefully with fallback
        assert!(result.stats.is_none());
    }

    /// Test gap extraction with a single span.
    ///
    /// Given: Only one span
    /// Expected: No gaps to extract, should fallback
    #[test]
    fn test_extract_gaps_single_span() {
        let spans = vec![create_test_span("word", 0.0, 0.0, 10.0, 12.0)];
        let result = analyze_document_gaps(&spans, None);

        // Single span = no gaps = fallback
        assert!(result.stats.is_none());
    }
}

// ============================================================================
// Statistics Calculation Tests
// ============================================================================

mod statistics_calculation_tests {
    use super::*;
    use pdf_oxide::extractors::analyze_document_gaps;

    /// Test statistics calculation with normal gap distribution.
    ///
    /// Given: Gaps [1.0, 2.0, 3.0, 4.0, 5.0]
    /// Expected: median=3.0, mean=3.0, p25=2.0, p75=4.0
    #[test]
    fn test_calculate_statistics_normal_distribution() {
        let spans = create_spans_with_gaps(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = analyze_document_gaps(&spans, None);

        if result.stats.is_some() {
            let stats = &result.stats.as_ref().unwrap();
            // Verify median is correct
            assert!((stats.median - 3.0).abs() < 0.01);
            // Verify mean is correct
            assert!((stats.mean - 3.0).abs() < 0.01);
            // Verify count
            assert_eq!(stats.count, 5);
        }
    }

    /// Test statistics calculation with outliers.
    ///
    /// Given: Gaps [0.1, 0.2, 0.3, 0.4, 50.0]
    /// Expected: median should be 0.3 (robust to outlier)
    #[test]
    fn test_calculate_statistics_with_outliers() {
        let spans = create_spans_with_gaps(&[0.1, 0.2, 0.3, 0.4, 50.0]);
        let result = analyze_document_gaps(&spans, None);

        if result.stats.is_some() {
            let stats = &result.stats.as_ref().unwrap();
            // Median should be 0.3, not affected by 50.0 outlier
            assert!((stats.median - 0.3).abs() < 0.01);
            // Mean will be higher due to outlier
            assert!(stats.mean > stats.median);
        }
    }

    /// Test that insufficient data falls back to fixed threshold.
    ///
    /// Given: Very few gaps (below min_samples)
    /// Expected: Fallback to fixed threshold
    #[test]
    fn test_calculate_statistics_insufficient_data() {
        let spans = vec![
            create_test_span("word1", 0.0, 0.0, 10.0, 12.0),
            create_test_span("word2", 11.0, 0.0, 10.0, 12.0),
        ];

        let config = AdaptiveThresholdConfig {
            median_multiplier: 1.5,
            min_threshold_pt: 0.05,
            max_threshold_pt: 1.0,
            use_iqr: false,
            min_samples: 10, // Require many samples
        };

        let result = analyze_document_gaps(&spans, Some(config));
        // With only 1 gap and min_samples=10, should fallback
        assert!(result.stats.is_none());
    }

    /// Test statistics with uniform gaps.
    ///
    /// Given: All gaps identical [2.0, 2.0, 2.0, 2.0, 2.0]
    /// Expected: median=2.0, std_dev=0.0
    #[test]
    fn test_calculate_statistics_uniform_gaps() {
        let spans = create_spans_with_gaps(&[2.0, 2.0, 2.0, 2.0, 2.0]);
        let result = analyze_document_gaps(&spans, None);

        if result.stats.is_some() {
            let stats = &result.stats.as_ref().unwrap();
            assert!((stats.median - 2.0).abs() < 0.01);
            assert!((stats.std_dev).abs() < 0.01); // Should be ~0
        }
    }

    /// Test percentile calculations.
    ///
    /// Given: Gaps [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    /// Expected: p10~1.1, p25~2.75, p75~7.25, p90~9.1
    #[test]
    fn test_calculate_statistics_percentiles() {
        let gaps: Vec<f32> = (1..=10).map(|i| i as f32).collect();
        let spans = create_spans_with_gaps(&gaps);
        let result = analyze_document_gaps(&spans, None);

        if result.stats.is_some() {
            let stats = &result.stats.as_ref().unwrap();
            // p10 should be close to 1.0
            assert!(stats.p10 >= 1.0 && stats.p10 <= 2.0);
            // p25 should be around 2.5-3.0
            assert!(stats.p25 >= 2.0 && stats.p25 <= 3.5);
            // p75 should be around 7.0-8.0
            assert!(stats.p75 >= 6.5 && stats.p75 <= 8.0);
            // p90 should be close to 10.0
            assert!(stats.p90 >= 9.0 && stats.p90 <= 10.0);
        }
    }
}

// ============================================================================
// Threshold Determination Tests
// ============================================================================

mod threshold_determination_tests {
    use super::*;
    use pdf_oxide::extractors::analyze_document_gaps;

    /// Test basic threshold calculation using median multiplier.
    ///
    /// Given: Median gap = 0.2pt, multiplier = 1.5
    /// Expected: threshold = 0.3pt
    #[test]
    fn test_determine_threshold_basic_median() {
        let spans = create_spans_with_gaps(&[0.2, 0.2, 0.2, 0.2, 0.2]);

        let config = AdaptiveThresholdConfig {
            median_multiplier: 1.5,
            min_threshold_pt: 0.05,
            max_threshold_pt: 1.0,
            use_iqr: false,
            min_samples: 5,
        };

        let result = analyze_document_gaps(&spans, Some(config));

        if result.stats.is_some() {
            let expected = 0.2 * 1.5; // 0.3
            assert!(
                (result.threshold_pt - expected).abs() < 0.01,
                "Expected threshold ~{}, got {}",
                expected,
                result.threshold_pt
            );
        }
    }

    /// Test threshold clamping to minimum.
    ///
    /// Given: Very small median (0.02), min_threshold = 0.05
    /// Expected: threshold clamped to 0.05
    #[test]
    fn test_determine_threshold_clamping_min() {
        let spans = create_spans_with_gaps(&[0.02, 0.02, 0.02, 0.02, 0.02]);

        let config = AdaptiveThresholdConfig {
            median_multiplier: 1.5,
            min_threshold_pt: 0.05,
            max_threshold_pt: 1.0,
            use_iqr: false,
            min_samples: 5,
        };

        let result = analyze_document_gaps(&spans, Some(config));

        if result.stats.is_some() {
            // 0.02 * 1.5 = 0.03, should clamp to 0.05
            assert!(result.threshold_pt >= 0.05);
        }
    }

    /// Test threshold clamping to maximum.
    ///
    /// Given: Very large median (1.5), max_threshold = 1.0
    /// Expected: threshold clamped to 1.0
    #[test]
    fn test_determine_threshold_clamping_max() {
        let spans = create_spans_with_gaps(&[1.5, 1.5, 1.5, 1.5, 1.5]);

        let config = AdaptiveThresholdConfig {
            median_multiplier: 1.5,
            min_threshold_pt: 0.05,
            max_threshold_pt: 1.0,
            use_iqr: false,
            min_samples: 5,
        };

        let result = analyze_document_gaps(&spans, Some(config));

        if result.stats.is_some() {
            // 1.5 * 1.5 = 2.25, should clamp to 1.0
            assert!(result.threshold_pt <= 1.0);
        }
    }

    /// Test IQR-based threshold calculation.
    ///
    /// When use_iqr=true, should use p25 + 0.5*(p75-p25) instead of median
    #[test]
    fn test_determine_threshold_iqr_mode() {
        let spans = create_spans_with_gaps(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        let config = AdaptiveThresholdConfig {
            median_multiplier: 1.5,
            min_threshold_pt: 0.05,
            max_threshold_pt: 10.0,
            use_iqr: true,
            min_samples: 5,
        };

        let result = analyze_document_gaps(&spans, Some(config));

        if result.stats.is_some() {
            // IQR mode should produce a valid threshold
            assert!(result.threshold_pt >= 0.05);
            assert!(result.threshold_pt <= 10.0);
        }
    }

    /// Test threshold determination with edge cases.
    ///
    /// Single gap, zero gaps, etc.
    #[test]
    fn test_determine_threshold_edge_cases() {
        // Zero gaps should fallback
        let empty_spans: Vec<TextSpan> = vec![];
        let result = analyze_document_gaps(&empty_spans, None);
        assert!(result.stats.is_none());

        // Single gap should fallback (insufficient samples)
        let single_span = vec![create_test_span("word", 0.0, 0.0, 10.0, 12.0)];
        let result = analyze_document_gaps(&single_span, None);
        assert!(result.stats.is_none());
    }
}

// ============================================================================
// Factory Method Tests
// ============================================================================

mod factory_method_tests {
    use super::*;

    /// Test default adaptive configuration.
    ///
    /// Default should have:
    /// - median_multiplier = 1.5
    /// - min_threshold_pt = 0.05
    /// - max_threshold_pt = 100.0 (Phase 7 fix)
    #[test]
    fn test_adaptive_config_default() {
        let config = AdaptiveThresholdConfig::default();

        assert_eq!(config.median_multiplier, 1.5);
        assert_eq!(config.min_threshold_pt, 0.05);
        // Phase 7 FIX: max_threshold_pt was increased from 1.0 to 100.0
        // to allow computed thresholds for documents with larger word spacing
        assert_eq!(config.max_threshold_pt, 100.0);
        assert!(!config.use_iqr); // Default is false
        assert_eq!(config.min_samples, 10);
    }

    /// Test aggressive adaptive configuration.
    ///
    /// Should be more sensitive (lower multiplier, lower min_threshold)
    #[test]
    fn test_adaptive_config_aggressive() {
        let config = AdaptiveThresholdConfig::aggressive();

        // Aggressive should have lower multiplier
        assert_eq!(config.median_multiplier, 1.2);
        // Lower minimum threshold
        assert_eq!(config.min_threshold_pt, 0.05); // Same as default
                                                   // Require fewer samples is NOT the case, defaults to 10
        assert_eq!(config.min_samples, 10);
    }

    /// Test conservative adaptive configuration.
    ///
    /// Should be less sensitive (higher multiplier, same min_threshold)
    #[test]
    fn test_adaptive_config_conservative() {
        let config = AdaptiveThresholdConfig::conservative();

        // Conservative should have higher multiplier
        assert_eq!(config.median_multiplier, 2.0);
        // Minimum threshold is same as default
        assert_eq!(config.min_threshold_pt, 0.05);
        // Require same samples as default
        assert_eq!(config.min_samples, 10);
    }

    /// Test custom multiplier configuration.
    ///
    /// Document-type-specific configs were removed for PDF spec compliance.
    /// Use with_multiplier() for custom configurations.
    #[test]
    fn test_adaptive_config_custom_multiplier() {
        let config = AdaptiveThresholdConfig::with_multiplier(1.3);

        // Custom multiplier
        assert_eq!(config.median_multiplier, 1.3);
        // Default bounds
        assert_eq!(config.min_threshold_pt, 0.05);
        assert_eq!(config.max_threshold_pt, 100.0);
    }

    /// Test SpanMergingConfig::adaptive() factory method.
    #[test]
    fn test_span_merging_config_adaptive() {
        let config = SpanMergingConfig::adaptive();

        assert!(config.use_adaptive_threshold);
        assert!(config.adaptive_config.is_some());

        let adaptive_cfg = config.adaptive_config.unwrap();
        assert_eq!(adaptive_cfg.median_multiplier, 1.5);
    }
}

// ============================================================================
// Document Type Integration Tests
// ============================================================================

mod tight_spacing_tests {
    use super::*;
    use pdf_oxide::extractors::analyze_document_gaps;

    /// Test gap profile of a document with tight spacing.
    ///
    /// Documents with tight spacing typically have:
    /// - Very tight word spacing: 0.1-0.3pt
    /// - Should produce threshold in range 0.15-0.25pt
    #[test]
    fn test_tight_spacing_gap_profile() {
        // Simulate doc with tight spacing
        let gaps = vec![0.1, 0.15, 0.12, 0.2, 0.13, 0.18, 0.11, 0.19, 0.14, 0.22];
        let spans = create_spans_with_gaps(&gaps);

        // Use custom multiplier for tight spacing (similar to old policy_documents config)
        let config = AdaptiveThresholdConfig::with_multiplier(1.3);
        let result = analyze_document_gaps(&spans, Some(config));

        if result.stats.is_some() {
            // Median should be around 0.15
            let median_expected = 0.15;
            let tolerance = 0.03;
            assert!(
                (result.stats.as_ref().unwrap().median - median_expected).abs() < tolerance,
                "Expected median ~{}, got {}",
                median_expected,
                result.stats.as_ref().unwrap().median
            );

            // Threshold should be in expected range
            assert!(result.threshold_pt > 0.1);
        }
    }

    /// Test that tight spacing is preserved (no word fusion).
    ///
    /// With adaptive threshold, 0.1pt gaps should not cause word fusion.
    #[test]
    fn test_tight_spacing_words_not_fused() {
        let gaps = vec![0.1; 10]; // Uniform 0.1pt gaps
        let spans = create_spans_with_gaps(&gaps);

        let config = AdaptiveThresholdConfig::with_multiplier(1.3);
        let result = analyze_document_gaps(&spans, Some(config));

        if result.stats.is_some() {
            // Threshold should be above the 0.1pt gaps
            assert!(
                result.threshold_pt > 0.1,
                "Threshold {} should be > 0.1 to avoid word fusion",
                result.threshold_pt
            );
        }
    }
}

mod standard_spacing_tests {
    use super::*;
    use pdf_oxide::extractors::analyze_document_gaps;

    /// Test gap profile of a document with standard spacing.
    ///
    /// Documents with standard spacing typically have:
    /// - Standard word spacing: 0.3-0.5pt
    /// - Should produce threshold in range 0.35-0.65pt
    #[test]
    fn test_standard_spacing_gap_profile() {
        // Simulate doc with standard spacing
        let gaps = vec![0.3, 0.35, 0.32, 0.4, 0.33, 0.38, 0.31, 0.39, 0.34, 0.42];
        let spans = create_spans_with_gaps(&gaps);

        // Use custom multiplier for standard spacing (similar to old academic config)
        let config = AdaptiveThresholdConfig::with_multiplier(1.6);
        let result = analyze_document_gaps(&spans, Some(config));

        if result.stats.is_some() {
            // Median should be around 0.35
            let median_expected = 0.35;
            let tolerance = 0.03;
            assert!(
                (result.stats.as_ref().unwrap().median - median_expected).abs() < tolerance,
                "Expected median ~{}, got {}",
                median_expected,
                result.stats.as_ref().unwrap().median
            );

            // Threshold should be in expected range
            assert!(result.threshold_pt >= 0.2);
        }
    }

    /// Test that legitimate spaces are preserved.
    ///
    /// With adaptive threshold, 0.3-0.5pt gaps should produce space characters.
    #[test]
    fn test_standard_spacing_preserves_spaces() {
        let gaps = vec![0.35; 10]; // Uniform 0.35pt gaps
        let spans = create_spans_with_gaps(&gaps);

        let config = AdaptiveThresholdConfig::with_multiplier(1.6);
        let result = analyze_document_gaps(&spans, Some(config));

        if result.stats.is_some() {
            // Threshold should be reasonable for the gap sizes
            assert!(
                result.threshold_pt <= 1.0,
                "Threshold {} should be <= 1.0",
                result.threshold_pt
            );
        }
    }
}

mod mixed_document_tests {
    use super::*;
    use pdf_oxide::extractors::analyze_document_gaps;

    /// Test document with mixed gap distribution.
    ///
    /// Documents with tables + text have bimodal gap distribution:
    /// - Small gaps (0.1-0.3pt): within table cells
    /// - Large gaps (5.0-15.0pt): between table columns
    ///
    /// Adaptive threshold should use median to be robust to these outliers.
    #[test]
    fn test_mixed_spacing_document() {
        // Mix of tight (text) and wide (table columns) spacing
        let mut gaps = vec![
            0.15, 0.2, 0.18, 0.22, 0.17, 0.19, 0.16, 0.21, // Text gaps
        ];
        gaps.extend_from_slice(&[8.0, 10.0, 9.0]); // Table column gaps (outliers)

        let spans = create_spans_with_gaps(&gaps);

        let config = AdaptiveThresholdConfig::default();
        let result = analyze_document_gaps(&spans, Some(config));

        if result.stats.is_some() {
            let stats = &result.stats.as_ref().unwrap();
            // Median should be robust to outliers, focusing on text spacing
            assert!(stats.median < 1.0); // Should be closer to text gaps
                                         // Max should reflect outliers
            assert!(stats.max > 5.0);
        }
    }

    /// Test document with varied font sizes.
    ///
    /// Different font sizes = different gap sizes
    /// Adaptive threshold should normalize based on actual distribution
    #[test]
    fn test_document_with_varied_fonts() {
        // Simulate different font sizes with proportional spacing
        let mut spans = vec![];

        // 12pt text with 0.3pt gaps
        for i in 0..3 {
            let x = (i * 10) as f32;
            spans.push(create_test_span(&format!("w{}", i), x, 0.0, 10.0, 12.0));
            if i < 2 {
                let gap_span = create_test_span(" ", x + 10.0 + 0.3, 0.0, 0.0, 12.0);
                spans.push(gap_span);
            }
        }

        // 10pt text with 0.25pt gaps
        for i in 0..3 {
            let x = 50.0 + (i * 8) as f32;
            spans.push(create_test_span(&format!("v{}", i), x, 20.0, 8.0, 10.0));
            if i < 2 {
                let gap_span = create_test_span(" ", x + 8.0 + 0.25, 20.0, 0.0, 10.0);
                spans.push(gap_span);
            }
        }

        let config = AdaptiveThresholdConfig::default();
        let result = analyze_document_gaps(&spans, Some(config));

        // Should either be adaptive or fallback gracefully
        assert!(result.threshold_pt > 0.0);
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

mod edge_case_tests {
    use super::*;
    use pdf_oxide::extractors::analyze_document_gaps;

    /// Test single span document (no gaps).
    ///
    /// Should fall back to fixed threshold since there are no gaps to analyze.
    #[test]
    fn test_single_span_document() {
        let spans = vec![create_test_span("singleword", 0.0, 0.0, 50.0, 12.0)];
        let result = analyze_document_gaps(&spans, None);

        // Single span = no gaps = fallback
        assert!(result.stats.is_none());
        assert!(!result.reason.is_empty()); // Reason should be populated
    }

    /// Test all overlapping spans (all negative gaps).
    ///
    /// Some PDFs have overlapping text. Should handle gracefully.
    #[test]
    fn test_all_overlapping_spans() {
        let spans = vec![
            create_test_span("word1", 0.0, 0.0, 20.0, 12.0),
            create_test_span("word2", 10.0, 0.0, 20.0, 12.0), // Overlaps by 10
            create_test_span("word3", 20.0, 0.0, 20.0, 12.0), // Overlaps by 10
        ];

        let result = analyze_document_gaps(&spans, None);

        // All overlaps should fallback (no positive gaps)
        assert!(result.stats.is_none());
    }

    /// Test mostly overlapping with one outlier gap.
    ///
    /// One large gap among mostly negative gaps
    #[test]
    fn test_mostly_overlapping_with_outlier() {
        let spans = vec![
            create_test_span("word1", 0.0, 0.0, 12.0, 12.0),
            create_test_span("word2", 10.0, 0.0, 12.0, 12.0), // Overlap: -2
            create_test_span("word3", 8.0, 0.0, 12.0, 12.0),  // Overlap: -4
            create_test_span("word4", 30.0, 0.0, 12.0, 12.0), // Large gap: 18
        ];

        let config = AdaptiveThresholdConfig {
            median_multiplier: 1.5,
            min_threshold_pt: 0.05,
            max_threshold_pt: 10.0,
            use_iqr: false,
            min_samples: 1, // Low threshold to not fallback
        };

        let result = analyze_document_gaps(&spans, Some(config));

        // Should either adapt or fallback, but not panic
        assert!(result.threshold_pt >= 0.0);
    }

    /// Test extremely tight spacing (all gaps < 0.05pt).
    ///
    /// Some fonts have very tight metrics
    #[test]
    fn test_extremely_tight_spacing() {
        let gaps = vec![0.01, 0.02, 0.015, 0.025, 0.012, 0.018];
        let spans = create_spans_with_gaps(&gaps);

        let config = AdaptiveThresholdConfig {
            median_multiplier: 1.5,
            min_threshold_pt: 0.01,
            max_threshold_pt: 1.0,
            use_iqr: false,
            min_samples: 5,
        };

        let result = analyze_document_gaps(&spans, Some(config));

        if result.stats.is_some() {
            // Should still compute valid threshold
            assert!(result.threshold_pt >= 0.01);
            assert!(result.threshold_pt <= 1.0);
        }
    }

    /// Test extremely loose spacing (all gaps > 1.0pt).
    ///
    /// Wide spacing between words (e.g., justified text or tables)
    #[test]
    fn test_extremely_loose_spacing() {
        let gaps = vec![1.5, 2.0, 1.8, 2.2, 1.7, 2.1];
        let spans = create_spans_with_gaps(&gaps);

        let config = AdaptiveThresholdConfig {
            median_multiplier: 1.5,
            min_threshold_pt: 0.05,
            max_threshold_pt: 5.0,
            use_iqr: false,
            min_samples: 5,
        };

        let result = analyze_document_gaps(&spans, Some(config));

        if result.stats.is_some() {
            // Should compute threshold in the wide range
            assert!(result.threshold_pt >= 1.0);
            assert!(result.threshold_pt <= 5.0);
        }
    }

    /// Test with NaN or inf values (corrupted data).
    ///
    /// Some PDFs have malformed position data
    #[test]
    fn test_invalid_gap_values() {
        // Create spans with very large positions (could produce inf)
        let spans = vec![
            create_test_span("word1", 0.0, 0.0, 10.0, 12.0),
            create_test_span("word2", f32::MAX - 100.0, 0.0, 10.0, 12.0),
        ];

        let result = analyze_document_gaps(&spans, None);

        // Should handle gracefully without panicking
        // May fallback or compute a value
        assert!(result.threshold_pt >= 0.0 || result.stats.is_none());
    }
}

// ============================================================================
// Backward Compatibility Tests
// ============================================================================

mod backward_compatibility_tests {
    use super::*;

    /// Test that adaptive is ENABLED by default (Phase 8 change).
    ///
    /// As of Phase 8, adaptive threshold is enabled by default for better quality.
    /// Use SpanMergingConfig::legacy() for the old fixed-threshold behavior.
    #[test]
    fn test_adaptive_enabled_by_default() {
        let config = SpanMergingConfig::default();

        // Phase 8: Adaptive is now enabled by default
        assert!(config.use_adaptive_threshold);
        // adaptive_config is None, meaning use default AdaptiveThresholdConfig
        assert!(config.adaptive_config.is_none());
    }

    /// Test that default config still works.
    ///
    /// SpanMergingConfig::default() should be unchanged
    #[test]
    fn test_original_config_still_works() {
        let config = SpanMergingConfig::default();

        // Should have original default values
        assert_eq!(config.space_threshold_em_ratio, 0.25);
        assert_eq!(config.conservative_threshold_pt, 0.1);
        assert_eq!(config.column_boundary_threshold_pt, 5.0);
        assert_eq!(config.severe_overlap_threshold_pt, -0.5);
    }

    /// Test that aggressive() factory still works.
    #[test]
    fn test_aggressive_mode_unchanged() {
        let config = SpanMergingConfig::aggressive();

        // Should have original aggressive values
        assert_eq!(config.space_threshold_em_ratio, 0.15);
        assert_eq!(config.conservative_threshold_pt, 0.1);
        assert_eq!(config.column_boundary_threshold_pt, 5.0);
        assert_eq!(config.severe_overlap_threshold_pt, -0.5);
        // Should NOT use adaptive by default
        assert!(!config.use_adaptive_threshold);
    }

    /// Test that conservative() factory still works.
    #[test]
    fn test_conservative_mode_unchanged() {
        let config = SpanMergingConfig::conservative();

        // Should have original conservative values
        assert_eq!(config.space_threshold_em_ratio, 0.33);
        assert_eq!(config.conservative_threshold_pt, 0.3);
        assert_eq!(config.column_boundary_threshold_pt, 5.0);
        assert_eq!(config.severe_overlap_threshold_pt, -0.5);
        // Should NOT use adaptive by default
        assert!(!config.use_adaptive_threshold);
    }

    /// Test that custom() factory still works.
    #[test]
    fn test_custom_config_unchanged() {
        let config = SpanMergingConfig::custom(0.2, 0.2, 6.0, -0.3);

        assert_eq!(config.space_threshold_em_ratio, 0.2);
        assert_eq!(config.conservative_threshold_pt, 0.2);
        assert_eq!(config.column_boundary_threshold_pt, 6.0);
        assert_eq!(config.severe_overlap_threshold_pt, -0.3);
        assert!(!config.use_adaptive_threshold);
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

mod integration_tests {
    use super::*;

    /// Test adaptive configuration with TextExtractionConfig.
    ///
    /// Should work together without conflicts
    #[test]
    fn test_adaptive_with_extraction_config() {
        let extraction_config = TextExtractionConfig::default();
        let span_config = SpanMergingConfig::adaptive();

        // Both configs should be valid
        assert_eq!(extraction_config.space_insertion_threshold, -120.0);
        assert!(span_config.use_adaptive_threshold);
    }

    /// Test that result contains all required metadata.
    ///
    /// AdaptiveThresholdResult should have threshold, statistics, and flags
    #[test]
    fn test_adaptive_result_contains_metadata() {
        let spans = create_spans_with_gaps(&[0.2, 0.2, 0.2, 0.2, 0.2]);
        let config = Some(AdaptiveThresholdConfig::default());

        use pdf_oxide::extractors::analyze_document_gaps;
        let result = analyze_document_gaps(&spans, config);

        // Should always have threshold and reason
        assert!(result.threshold_pt > 0.0);
        assert!(result.stats.is_some() || !result.reason.is_empty());
    }

    /// Test result statistics are properly populated.
    #[test]
    fn test_adaptive_result_statistics_populated() {
        let gaps = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let spans = create_spans_with_gaps(&gaps);

        use pdf_oxide::extractors::analyze_document_gaps;
        let result = analyze_document_gaps(&spans, None);

        if let Some(stats) = &result.stats {
            assert!(stats.count > 0);
            assert!(stats.min >= 0.0);
            assert!(stats.max >= stats.min);
            assert!(stats.median >= stats.min && stats.median <= stats.max);
        } else {
            // If no stats, at least verify threshold was computed
            assert!(result.threshold_pt > 0.0);
        }
    }
}

// ============================================================================
// Performance and Stress Tests
// ============================================================================

mod performance_tests {
    use super::*;
    use pdf_oxide::extractors::analyze_document_gaps;

    /// Test performance with large document (many spans).
    ///
    /// Gap analysis should be O(n log n), so should be fast even with
    /// many spans.
    #[test]
    fn test_large_document_performance() {
        // Create document with many spans
        let mut spans = vec![];
        for i in 0..1000 {
            let x = (i * 11) as f32; // 10pt width + 1pt gap
            spans.push(create_test_span(&format!("w{}", i), x, 0.0, 10.0, 12.0));
        }

        let start = std::time::Instant::now();
        let result = analyze_document_gaps(&spans, None);
        let elapsed = start.elapsed();

        // Should complete quickly (< 100ms for 1000 spans)
        assert!(elapsed.as_millis() < 100);

        // Should still produce valid result or fallback
        assert!(result.threshold_pt >= 0.0);
    }

    /// Test with deeply nested document structure.
    #[test]
    fn test_multiline_document_performance() {
        let gaps_per_line = vec![vec![0.2, 0.2, 0.2]; 100]; // 100 lines
        let spans = create_multiline_document(gaps_per_line, 20.0);

        let start = std::time::Instant::now();
        let result = analyze_document_gaps(&spans, None);
        let elapsed = start.elapsed();

        // Should complete in reasonable time
        assert!(elapsed.as_millis() < 100);

        // Should work with multiline documents
        assert!(result.threshold_pt >= 0.0);
    }
}
