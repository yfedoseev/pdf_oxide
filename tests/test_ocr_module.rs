//! Comprehensive tests for the OCR module
//!
//! Tests the OCR functionality including:
//! - Configuration and builder pattern
//! - Extraction options
//! - Module compilation with feature flags
//! - Basic API compatibility

#[cfg(feature = "ocr")]
mod ocr_tests {
    use pdf_oxide::ocr::{OcrConfig, OcrConfigBuilder, OcrExtractOptions};

    // ========================================================================
    // CONFIGURATION TESTS
    // ========================================================================

    #[test]
    fn test_ocr_config_default_values() {
        let config = OcrConfig::default();

        assert!(
            (config.det_threshold - 0.3).abs() < f32::EPSILON,
            "Default detection threshold should be 0.3"
        );
        assert!(
            (config.box_threshold - 0.5).abs() < f32::EPSILON,
            "Default box threshold should be 0.5"
        );
        assert!(
            (config.rec_threshold - 0.5).abs() < f32::EPSILON,
            "Default recognition threshold should be 0.5"
        );
        assert_eq!(config.det_max_side, 960, "Default detection max side should be 960");
        assert_eq!(config.rec_target_height, 48, "Default recognition target height should be 48");
        assert_eq!(config.num_threads, 4, "Default number of threads should be 4");
        assert!(
            (config.unclip_ratio - 1.5).abs() < f32::EPSILON,
            "Default unclip ratio should be 1.5"
        );
        assert_eq!(config.max_candidates, 1000, "Default max candidates should be 1000");
        assert!(config.detect_styles, "Style detection should be enabled by default");
        assert!(config.det_model_path.is_none());
        assert!(config.rec_model_path.is_none());
        assert!(config.dict_path.is_none());
    }

    #[test]
    fn test_ocr_config_new() {
        let config = OcrConfig::new();
        let default_config = OcrConfig::default();

        assert_eq!(
            config.det_threshold, default_config.det_threshold,
            "new() should match default()"
        );
        assert_eq!(config.num_threads, default_config.num_threads, "new() should match default()");
    }

    // ========================================================================
    // BUILDER PATTERN TESTS
    // ========================================================================

    #[test]
    fn test_ocr_config_builder_basic() {
        let config = OcrConfig::builder()
            .det_threshold(0.4)
            .box_threshold(0.6)
            .rec_threshold(0.7)
            .build();

        assert!((config.det_threshold - 0.4).abs() < f32::EPSILON, "det_threshold should be 0.4");
        assert!((config.box_threshold - 0.6).abs() < f32::EPSILON, "box_threshold should be 0.6");
        assert!((config.rec_threshold - 0.7).abs() < f32::EPSILON, "rec_threshold should be 0.7");
    }

    #[test]
    fn test_ocr_config_builder_all_options() {
        let config = OcrConfig::builder()
            .det_threshold(0.35)
            .box_threshold(0.55)
            .rec_threshold(0.65)
            .det_max_side(1024)
            .rec_target_height(64)
            .num_threads(8)
            .unclip_ratio(1.8)
            .max_candidates(2000)
            .detect_styles(false)
            .build();

        assert!((config.det_threshold - 0.35).abs() < f32::EPSILON);
        assert!((config.box_threshold - 0.55).abs() < f32::EPSILON);
        assert!((config.rec_threshold - 0.65).abs() < f32::EPSILON);
        assert_eq!(config.det_max_side, 1024);
        assert_eq!(config.rec_target_height, 64);
        assert_eq!(config.num_threads, 8);
        assert!((config.unclip_ratio - 1.8).abs() < f32::EPSILON);
        assert_eq!(config.max_candidates, 2000);
        assert!(!config.detect_styles);
    }

    #[test]
    fn test_ocr_config_builder_clamping_thresholds() {
        // Test threshold clamping to [0.0, 1.0]
        let config = OcrConfig::builder()
            .det_threshold(-0.5) // Should clamp to 0.0
            .box_threshold(1.5) // Should clamp to 1.0
            .rec_threshold(2.0) // Should clamp to 1.0
            .build();

        assert!(config.det_threshold >= 0.0 && config.det_threshold <= 1.0);
        assert!(config.box_threshold >= 0.0 && config.box_threshold <= 1.0);
        assert!(config.rec_threshold >= 0.0 && config.rec_threshold <= 1.0);
    }

    #[test]
    fn test_ocr_config_builder_clamping_dimensions() {
        // Test dimension clamping to minimum values
        let config = OcrConfig::builder()
            .det_max_side(10) // Should clamp to 32
            .rec_target_height(8) // Should clamp to 16
            .num_threads(0) // Should clamp to 1
            .max_candidates(0) // Should clamp to 1
            .unclip_ratio(0.5) // Should clamp to 1.0
            .build();

        assert!(config.det_max_side >= 32, "det_max_side should be at least 32");
        assert!(config.rec_target_height >= 16, "rec_target_height should be at least 16");
        assert!(config.num_threads >= 1, "num_threads should be at least 1");
        assert!(config.max_candidates >= 1, "max_candidates should be at least 1");
        assert!(config.unclip_ratio >= 1.0, "unclip_ratio should be at least 1.0");
    }

    #[test]
    fn test_ocr_config_builder_chainable() {
        // Test that builder methods are chainable
        let config = OcrConfigBuilder::new()
            .det_threshold(0.25)
            .num_threads(2)
            .detect_styles(false)
            .build();

        assert!((config.det_threshold - 0.25).abs() < f32::EPSILON);
        assert_eq!(config.num_threads, 2);
        assert!(!config.detect_styles);
    }

    #[test]
    fn test_ocr_config_builder_with_model_paths() {
        use std::path::PathBuf;

        let det_path = PathBuf::from("/models/det.onnx");
        let rec_path = PathBuf::from("/models/rec.onnx");
        let dict_path = PathBuf::from("/models/dict.txt");

        let config = OcrConfig::builder()
            .det_model_path(det_path.clone())
            .rec_model_path(rec_path.clone())
            .dict_path(dict_path.clone())
            .build();

        assert_eq!(config.det_model_path, Some(det_path), "Detection model path should be set");
        assert_eq!(config.rec_model_path, Some(rec_path), "Recognition model path should be set");
        assert_eq!(config.dict_path, Some(dict_path), "Dictionary path should be set");
    }

    // ========================================================================
    // EXTRACTION OPTIONS TESTS
    // ========================================================================

    #[test]
    fn test_ocr_extract_options_default() {
        let options = OcrExtractOptions::default();

        // Default scale should be 300 DPI / 72 points
        let expected_scale = 300.0 / 72.0;
        assert!(
            (options.scale - expected_scale).abs() < 0.01,
            "Default scale should be 300 DPI / 72 points = {}, got {}",
            expected_scale,
            options.scale
        );

        assert!(
            options.fallback_to_native,
            "Fallback to native text should be enabled by default"
        );
    }

    #[test]
    fn test_ocr_extract_options_with_dpi() {
        // Test 200 DPI
        let options_200 = OcrExtractOptions::with_dpi(200.0);
        let expected_200 = 200.0 / 72.0;
        assert!(
            (options_200.scale - expected_200).abs() < 0.01,
            "Scale for 200 DPI should be {}",
            expected_200
        );

        // Test 300 DPI
        let options_300 = OcrExtractOptions::with_dpi(300.0);
        let expected_300 = 300.0 / 72.0;
        assert!(
            (options_300.scale - expected_300).abs() < 0.01,
            "Scale for 300 DPI should be {}",
            expected_300
        );

        // Test 600 DPI
        let options_600 = OcrExtractOptions::with_dpi(600.0);
        let expected_600 = 600.0 / 72.0;
        assert!(
            (options_600.scale - expected_600).abs() < 0.01,
            "Scale for 600 DPI should be {}",
            expected_600
        );
    }

    #[test]
    fn test_ocr_extract_options_custom_config() {
        let custom_config = OcrConfig::builder()
            .det_threshold(0.25)
            .num_threads(2)
            .build();

        let options = OcrExtractOptions {
            config: custom_config.clone(),
            scale: 4.17,
            fallback_to_native: true,
        };

        assert_eq!(options.config.det_threshold, custom_config.det_threshold);
        assert_eq!(options.config.num_threads, custom_config.num_threads);
        assert!((options.scale - 4.17).abs() < 0.01);
    }

    #[test]
    fn test_ocr_extract_options_fallback_disabled() {
        let options = OcrExtractOptions {
            config: OcrConfig::default(),
            scale: 4.17,
            fallback_to_native: false,
        };

        assert!(!options.fallback_to_native);
    }

    // ========================================================================
    // CONFIGURATION MUTATION TESTS
    // ========================================================================

    #[test]
    fn test_ocr_config_is_cloneable() {
        let config1 = OcrConfig::builder()
            .det_threshold(0.4)
            .num_threads(8)
            .build();

        let config2 = config1.clone();

        assert_eq!(config1.det_threshold, config2.det_threshold);
        assert_eq!(config1.num_threads, config2.num_threads);
    }

    #[test]
    fn test_ocr_config_is_debuggable() {
        let config = OcrConfig::default();
        let debug_string = format!("{:?}", config);

        assert!(!debug_string.is_empty());
        assert!(debug_string.contains("OcrConfig"));
    }

    // ========================================================================
    // EXTRACTION OPTIONS MUTATION TESTS
    // ========================================================================

    #[test]
    fn test_ocr_extract_options_is_cloneable() {
        let options1 = OcrExtractOptions::default();
        let options2 = options1.clone();

        assert!((options1.scale - options2.scale).abs() < f32::EPSILON);
        assert_eq!(options1.fallback_to_native, options2.fallback_to_native);
    }

    #[test]
    fn test_ocr_extract_options_dpi_range() {
        // Test various realistic DPI values
        let dpi_values = vec![72.0, 100.0, 150.0, 200.0, 300.0, 600.0];

        for dpi in dpi_values {
            let options = OcrExtractOptions::with_dpi(dpi);
            let expected_scale = dpi / 72.0;

            assert!(
                (options.scale - expected_scale).abs() < 0.01,
                "DPI {} should produce scale {}",
                dpi,
                expected_scale
            );
        }
    }

    // ========================================================================
    // INTEGRATION TESTS (Compile-time compatibility)
    // ========================================================================

    #[test]
    fn test_ocr_module_exports() {
        // Verify all public exports are accessible
        let _config = OcrConfig::default();
        let _builder = OcrConfig::builder();
        let _options = OcrExtractOptions::default();

        // These would only compile if the exports are correct
        // If any export is missing, this test would fail at compile time
    }

    #[test]
    fn test_ocr_builder_default_from_trait() {
        let builder = OcrConfigBuilder::default();
        let config = builder.build();

        assert_eq!(config.det_threshold, OcrConfig::default().det_threshold);
    }

    // ========================================================================
    // EDGE CASE TESTS
    // ========================================================================

    #[test]
    fn test_ocr_config_extreme_values() {
        let config = OcrConfig::builder()
            .det_threshold(0.0) // Minimum threshold
            .box_threshold(1.0) // Maximum threshold
            .num_threads(1024) // Very high thread count
            .det_max_side(4096) // Very large image size
            .build();

        assert!(config.det_threshold >= 0.0);
        assert!(config.box_threshold <= 1.0);
        assert_eq!(config.num_threads, 1024);
        assert_eq!(config.det_max_side, 4096);
    }

    #[test]
    fn test_ocr_extract_options_extreme_dpi() {
        let low_dpi = OcrExtractOptions::with_dpi(36.0);
        let high_dpi = OcrExtractOptions::with_dpi(2400.0);

        assert!((low_dpi.scale - 0.5).abs() < 0.01);
        assert!((high_dpi.scale - 33.33).abs() < 0.01);
    }
}

// ============================================================================
// TESTS FOR WHEN OCR FEATURE IS NOT ENABLED
// ============================================================================

#[cfg(not(feature = "ocr"))]
mod ocr_not_enabled_tests {
    #[test]
    fn test_ocr_feature_disabled() {
        // This test confirms that the module is properly feature-gated
        // If this compiles, it means the feature gate is working correctly
        println!("OCR feature is not enabled");
    }
}
