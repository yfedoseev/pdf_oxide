#![allow(warnings)]
//! Integration tests for OCR module API
//!
//! Tests the high-level OCR API including:
//! - OCR configuration workflows
//! - Options building
//! - Module API compatibility
//! - Error handling

#[cfg(feature = "ocr")]
mod ocr_integration_tests {
    use pdf_oxide::ocr::{OcrConfig, OcrConfigBuilder, OcrExtractOptions};

    // ========================================================================
    // WORKFLOW TESTS
    // ========================================================================

    #[test]
    fn test_ocr_typical_workflow() {
        // Simulate typical OCR workflow
        let config = OcrConfig::builder()
            .det_threshold(0.35)
            .box_threshold(0.55)
            .num_threads(4)
            .detect_styles(true)
            .build();

        let options = OcrExtractOptions {
            config: config.clone(),
            scale: 300.0 / 72.0,
            fallback_to_native: true,
        };

        // Verify workflow created valid objects
        assert!((options.config.det_threshold - 0.35).abs() < f32::EPSILON);
        assert_eq!(options.config.num_threads, 4);
        assert!(options.fallback_to_native);
    }

    #[test]
    fn test_ocr_high_precision_workflow() {
        // Workflow for high-precision OCR
        let config = OcrConfig::builder()
            .det_threshold(0.5) // Higher threshold = fewer false positives
            .box_threshold(0.7) // Higher threshold = only confident detections
            .rec_threshold(0.6) // Higher threshold = only confident recognitions
            .num_threads(8) // More threads for better performance
            .build();

        assert!((config.det_threshold - 0.5).abs() < f32::EPSILON);
        assert!((config.box_threshold - 0.7).abs() < f32::EPSILON);
        assert!((config.rec_threshold - 0.6).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ocr_fast_workflow() {
        // Workflow for fast OCR (lower accuracy, higher speed)
        let config = OcrConfig::builder()
            .det_threshold(0.2) // Lower threshold = more detections
            .box_threshold(0.3) // Lower threshold = more boxes
            .rec_threshold(0.3) // Lower threshold = more recognitions
            .num_threads(2) // Fewer threads = less overhead
            .det_max_side(512) // Smaller input = faster detection
            .rec_target_height(32) // Smaller input = faster recognition
            .build();

        assert!((config.det_threshold - 0.2).abs() < f32::EPSILON);
        assert_eq!(config.det_max_side, 512);
        assert_eq!(config.rec_target_height, 32);
    }

    #[test]
    fn test_ocr_multilingual_workflow() {
        // Workflow that might support multilingual OCR
        let config = OcrConfig::builder()
            .det_threshold(0.3) // Standard detection
            .num_threads(4)
            .max_candidates(2000) // More candidates for complex scripts
            .build();

        assert_eq!(config.max_candidates, 2000);
    }

    // ========================================================================
    // OPTIONS BUILDING PATTERNS
    // ========================================================================

    #[test]
    fn test_ocr_options_for_200dpi_scan() {
        let options = OcrExtractOptions::with_dpi(200.0);
        let expected_scale = 200.0 / 72.0;

        assert!(
            (options.scale - expected_scale).abs() < 0.01,
            "200 DPI should scale to {}",
            expected_scale
        );
    }

    #[test]
    fn test_ocr_options_for_300dpi_scan() {
        let options = OcrExtractOptions::with_dpi(300.0);
        let expected_scale = 300.0 / 72.0;

        assert!(
            (options.scale - expected_scale).abs() < 0.01,
            "300 DPI should scale to {}",
            expected_scale
        );
    }

    #[test]
    fn test_ocr_options_for_600dpi_scan() {
        let options = OcrExtractOptions::with_dpi(600.0);
        let expected_scale = 600.0 / 72.0;

        assert!(
            (options.scale - expected_scale).abs() < 0.01,
            "600 DPI should scale to {}",
            expected_scale
        );
    }

    #[test]
    fn test_ocr_options_manual_scale() {
        let config = OcrConfig::default();
        let options = OcrExtractOptions {
            config,
            scale: 2.5, // Custom scale
            fallback_to_native: false,
        };

        assert!((options.scale - 2.5).abs() < f32::EPSILON);
        assert!(!options.fallback_to_native);
    }

    // ========================================================================
    // BUILDER PATTERN ADVANCED TESTS
    // ========================================================================

    #[test]
    fn test_ocr_config_builder_multiple_modifications() {
        let config = OcrConfig::builder()
            .det_threshold(0.3)
            .det_threshold(0.4) // Overwrite previous value
            .det_threshold(0.35) // Overwrite again
            .build();

        // Last value should win
        assert!((config.det_threshold - 0.35).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ocr_config_builder_partial_setup() {
        // Test that builder works with partial configuration
        let config1 = OcrConfig::builder().det_threshold(0.4).build();

        let config2 = OcrConfig::builder()
            .det_threshold(0.4)
            .num_threads(8)
            .build();

        // Defaults should be applied to missing values
        assert_eq!(config1.num_threads, OcrConfig::default().num_threads);
        assert_eq!(config2.box_threshold, OcrConfig::default().box_threshold);
    }

    // ========================================================================
    // CONFIGURATION CONSISTENCY TESTS
    // ========================================================================

    #[test]
    fn test_ocr_config_threshold_validity() {
        let config = OcrConfig::builder()
            .det_threshold(0.3)
            .box_threshold(0.5)
            .rec_threshold(0.5)
            .build();

        // All thresholds should be valid (0.0 - 1.0)
        assert!(config.det_threshold >= 0.0 && config.det_threshold <= 1.0);
        assert!(config.box_threshold >= 0.0 && config.box_threshold <= 1.0);
        assert!(config.rec_threshold >= 0.0 && config.rec_threshold <= 1.0);
    }

    #[test]
    fn test_ocr_config_dimension_validity() {
        let config = OcrConfig::builder()
            .det_max_side(960)
            .rec_target_height(48)
            .build();

        // All dimensions should be reasonable
        assert!(config.det_max_side > 0);
        assert!(config.rec_target_height > 0);
        assert!(config.det_max_side >= config.rec_target_height);
    }

    #[test]
    fn test_ocr_config_thread_validity() {
        let config = OcrConfig::builder().num_threads(4).build();

        // Thread count should be positive
        assert!(config.num_threads > 0);
    }

    // ========================================================================
    // ERROR RECOVERY TESTS
    // ========================================================================

    #[test]
    fn test_ocr_options_with_invalid_dpi_recovers() {
        // Very low DPI
        let options_low = OcrExtractOptions::with_dpi(1.0);
        assert!(options_low.scale > 0.0);

        // Very high DPI
        let options_high = OcrExtractOptions::with_dpi(10000.0);
        assert!(options_high.scale > 0.0);
    }

    #[test]
    fn test_ocr_config_with_invalid_values_clamps() {
        let config = OcrConfig::builder()
            .det_threshold(-1.0) // Out of range
            .box_threshold(2.0) // Out of range
            .num_threads(0) // Out of range
            .build();

        // Values should be clamped to valid ranges
        assert!(config.det_threshold >= 0.0 && config.det_threshold <= 1.0);
        assert!(config.box_threshold >= 0.0 && config.box_threshold <= 1.0);
        assert!(config.num_threads >= 1);
    }

    // ========================================================================
    // COMPOSITION TESTS
    // ========================================================================

    #[test]
    fn test_ocr_config_and_options_composition() {
        // Build a config
        let config = OcrConfig::builder()
            .det_threshold(0.4)
            .num_threads(8)
            .detect_styles(false)
            .build();

        // Create options with that config
        let options = OcrExtractOptions {
            config: config.clone(),
            scale: 4.17,
            fallback_to_native: true,
        };

        // Verify composition works correctly
        assert_eq!(options.config.det_threshold, config.det_threshold);
        assert_eq!(options.config.num_threads, config.num_threads);
        assert_eq!(options.config.detect_styles, config.detect_styles);
        assert!((options.scale - 4.17).abs() < f32::EPSILON);
    }

    #[test]
    fn test_multiple_independent_configs() {
        // Create multiple independent configurations
        let config1 = OcrConfig::builder().det_threshold(0.3).build();

        let config2 = OcrConfig::builder().det_threshold(0.5).build();

        let config3 = OcrConfig::builder().det_threshold(0.7).build();

        // They should all be independent
        assert!((config1.det_threshold - 0.3).abs() < f32::EPSILON);
        assert!((config2.det_threshold - 0.5).abs() < f32::EPSILON);
        assert!((config3.det_threshold - 0.7).abs() < f32::EPSILON);
    }

    // ========================================================================
    // BOUNDARY TESTS
    // ========================================================================

    #[test]
    fn test_ocr_extract_options_scale_boundaries() {
        // Test scale at boundaries
        let options_min_scale = OcrExtractOptions {
            config: OcrConfig::default(),
            scale: 0.01, // Very small
            fallback_to_native: true,
        };

        let options_max_scale = OcrExtractOptions {
            config: OcrConfig::default(),
            scale: 100.0, // Very large
            fallback_to_native: true,
        };

        assert!(options_min_scale.scale > 0.0);
        assert!(options_max_scale.scale > 0.0);
    }

    #[test]
    fn test_ocr_config_max_candidates_boundary() {
        // Test max_candidates at boundaries
        let config_min = OcrConfig::builder().max_candidates(1).build();

        let config_max = OcrConfig::builder().max_candidates(100000).build();

        assert_eq!(config_min.max_candidates, 1);
        assert_eq!(config_max.max_candidates, 100000);
    }

    // ========================================================================
    // REALISTIC SCENARIO TESTS
    // ========================================================================

    #[test]
    fn test_ocr_scenario_legal_document() {
        // Configuration for legal documents (high accuracy needed)
        let config = OcrConfig::builder()
            .det_threshold(0.45) // Conservative detection
            .box_threshold(0.65) // Conservative box filtering
            .rec_threshold(0.60) // Conservative recognition
            .num_threads(4)
            .detect_styles(true) // Preserve formatting
            .build();

        let options = OcrExtractOptions::with_dpi(300.0); // Standard legal scan DPI

        assert!((config.det_threshold - 0.45).abs() < f32::EPSILON);
        assert!((options.scale - 300.0 / 72.0).abs() < 0.01);
    }

    #[test]
    fn test_ocr_scenario_ebook() {
        // Configuration for ebook extraction (balance speed and accuracy)
        let config = OcrConfig::builder()
            .det_threshold(0.35) // Balanced detection
            .box_threshold(0.55) // Balanced box filtering
            .rec_threshold(0.50) // Balanced recognition
            .num_threads(2)
            .build();

        let options = OcrExtractOptions::with_dpi(200.0); // Common ebook scan DPI

        assert!((config.det_threshold - 0.35).abs() < f32::EPSILON);
        assert!((options.scale - 200.0 / 72.0).abs() < 0.01);
    }

    #[test]
    fn test_ocr_scenario_batch_processing() {
        // Configuration for fast batch processing
        let config = OcrConfig::builder()
            .det_threshold(0.25) // Aggressive detection
            .box_threshold(0.40) // Aggressive box filtering
            .num_threads(8) // Utilize multiple cores
            .det_max_side(512) // Smaller input for speed
            .build();

        assert!((config.det_threshold - 0.25).abs() < f32::EPSILON);
        assert_eq!(config.num_threads, 8);
        assert_eq!(config.det_max_side, 512);
    }
}

// ============================================================================
// TESTS FOR WHEN OCR FEATURE IS NOT ENABLED
// ============================================================================

#[cfg(not(feature = "ocr"))]
mod ocr_integration_not_enabled_tests {
    #[test]
    fn test_ocr_integration_feature_disabled() {
        // This test confirms feature gating is working
        println!("OCR feature is not enabled for integration tests");
    }
}
