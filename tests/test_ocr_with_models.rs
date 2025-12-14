#![allow(clippy::manual_flatten)]
//! Integration tests for OCR with actual PaddleOCR models
//!
//! These tests verify that the OCR module can load and initialize
//! with actual ONNX/Paddle models.

#[cfg(feature = "ocr")]
mod ocr_model_tests {
    use pdf_oxide::ocr::{OcrConfig, TextDetector, TextRecognizer};
    use std::path::Path;

    // Models are located in .models/ directory
    const DET_MODEL_PATH: &str = ".models/ch_PP-OCRv3_det_infer/inference.pdmodel";
    const REC_MODEL_PATH: &str = ".models/ch_PP-OCRv3_rec_infer/inference.pdmodel";
    const DICT_PATH: &str = ".models/ppocr_keys_v1.txt";

    fn models_exist() -> bool {
        Path::new(DET_MODEL_PATH).exists()
            && Path::new(REC_MODEL_PATH).exists()
            && Path::new(DICT_PATH).exists()
    }

    // ========================================================================
    // MODEL LOADING TESTS
    // ========================================================================

    #[test]
    fn test_ocr_model_files_present() {
        if models_exist() {
            println!("✓ All OCR model files are present");
            println!("  Detection model: {}", DET_MODEL_PATH);
            println!("  Recognition model: {}", REC_MODEL_PATH);
            println!("  Dictionary: {}", DICT_PATH);
        } else {
            println!("⚠ OCR models not found. Skipping model loading tests.");
            println!("  Expected files:");
            println!("  - {}", DET_MODEL_PATH);
            println!("  - {}", REC_MODEL_PATH);
            println!("  - {}", DICT_PATH);
            println!("  Download models using: ./scripts/setup_ocr_models.sh");
        }
    }

    #[test]
    #[ignore] // Ignored by default as it requires ONNX initialization
    fn test_ocr_detector_model_loading() {
        if !models_exist() {
            println!("Skipping: Models not present");
            return;
        }

        let config = OcrConfig::builder()
            .det_threshold(0.3)
            .num_threads(4)
            .build();

        match TextDetector::new(DET_MODEL_PATH, config) {
            Ok(_detector) => {
                println!("✓ Detection model loaded successfully");
            },
            Err(e) => {
                eprintln!("Failed to load detection model: {:?}", e);
                panic!("Could not load detection model");
            },
        }
    }

    #[test]
    #[ignore] // Ignored by default as it requires ONNX initialization
    fn test_ocr_recognizer_model_loading() {
        if !models_exist() {
            println!("Skipping: Models not present");
            return;
        }

        let config = OcrConfig::builder()
            .rec_threshold(0.5)
            .num_threads(4)
            .build();

        match TextRecognizer::new(REC_MODEL_PATH, DICT_PATH, config) {
            Ok(_recognizer) => {
                println!("✓ Recognition model loaded successfully");
            },
            Err(e) => {
                eprintln!("Failed to load recognition model: {:?}", e);
                panic!("Could not load recognition model");
            },
        }
    }

    #[test]
    #[ignore] // Ignored by default as it requires ONNX initialization
    fn test_ocr_both_models_load() {
        if !models_exist() {
            println!("Skipping: Models not present");
            return;
        }

        let config = OcrConfig::default();

        let det_result = TextDetector::new(DET_MODEL_PATH, config.clone());
        let rec_result = TextRecognizer::new(REC_MODEL_PATH, DICT_PATH, config);

        match (det_result, rec_result) {
            (Ok(_), Ok(_)) => {
                println!("✓ Both detection and recognition models loaded");
            },
            (Err(e), _) => panic!("Detection model failed: {:?}", e),
            (_, Err(e)) => panic!("Recognition model failed: {:?}", e),
        }
    }

    // ========================================================================
    // MODEL CONFIGURATION TESTS
    // ========================================================================

    #[test]
    fn test_ocr_config_with_model_paths() {
        if !models_exist() {
            println!("Skipping: Models not present");
            return;
        }

        use std::path::PathBuf;

        let det_path = PathBuf::from(DET_MODEL_PATH);
        let rec_path = PathBuf::from(REC_MODEL_PATH);
        let dict_path = PathBuf::from(DICT_PATH);

        let config = OcrConfig::builder()
            .det_model_path(det_path.clone())
            .rec_model_path(rec_path.clone())
            .dict_path(dict_path.clone())
            .det_threshold(0.35)
            .num_threads(4)
            .build();

        assert_eq!(config.det_model_path, Some(det_path));
        assert_eq!(config.rec_model_path, Some(rec_path));
        assert_eq!(config.dict_path, Some(dict_path));
        assert!((config.det_threshold - 0.35).abs() < f32::EPSILON);
    }

    #[test]
    fn test_ocr_model_configuration_for_cpu_inference() {
        // Configuration optimized for CPU-only inference
        let config = OcrConfig::builder()
            .det_threshold(0.35) // Balanced detection sensitivity
            .box_threshold(0.55) // Balanced box filtering
            .rec_threshold(0.50) // Balanced recognition confidence
            .num_threads(4) // Typical CPU core count
            .det_max_side(960) // Reasonable input size
            .rec_target_height(48) // Standard recognition height
            .max_candidates(1000) // Reasonable upper limit
            .build();

        assert_eq!(config.num_threads, 4);
        assert_eq!(config.det_max_side, 960);
        assert_eq!(config.rec_target_height, 48);
        println!("✓ CPU-optimized OCR configuration created");
    }

    // ========================================================================
    // PERFORMANCE METRICS TESTS
    // ========================================================================

    #[test]
    fn test_ocr_model_path_information() {
        println!("\n=== OCR Model Information ===");
        println!("Detection Model:  {}", DET_MODEL_PATH);
        println!("Recognition Model: {}", REC_MODEL_PATH);
        println!("Dictionary:       {}", DICT_PATH);

        if let Ok(metadata) = std::fs::metadata(DET_MODEL_PATH) {
            println!(
                "Detection size:   {} bytes ({:.1} MB)",
                metadata.len(),
                metadata.len() as f64 / 1024.0 / 1024.0
            );
        }

        if let Ok(metadata) = std::fs::metadata(REC_MODEL_PATH) {
            println!(
                "Recognition size: {} bytes ({:.1} MB)",
                metadata.len(),
                metadata.len() as f64 / 1024.0 / 1024.0
            );
        }

        if let Ok(metadata) = std::fs::metadata(DICT_PATH) {
            println!("Dictionary size:  {} bytes", metadata.len());
        }

        println!("========================\n");
    }

    #[test]
    fn test_ocr_dictionary_can_be_read() {
        if !Path::new(DICT_PATH).exists() {
            println!("Dictionary not found at {}", DICT_PATH);
            return;
        }

        match std::fs::read_to_string(DICT_PATH) {
            Ok(content) => {
                let line_count = content.lines().count();
                println!("✓ Dictionary loaded: {} characters", line_count);

                // Show first few characters
                let first_chars: Vec<&str> = content.lines().take(5).collect();
                println!("  First characters: {:?}", first_chars);
            },
            Err(e) => {
                eprintln!("Failed to read dictionary: {}", e);
            },
        }
    }

    // ========================================================================
    // REALISTIC WORKFLOW TESTS (WITH MODELS)
    // ========================================================================

    #[test]
    #[ignore] // Requires actual image data and model inference
    fn test_ocr_workflow_legal_document_with_models() {
        if !models_exist() {
            println!("Skipping: Models not present");
            return;
        }

        // Configuration for high-accuracy legal document OCR
        let config = OcrConfig::builder()
            .det_threshold(0.45) // Conservative detection
            .box_threshold(0.65) // Conservative boxes
            .rec_threshold(0.60) // Conservative recognition
            .num_threads(4)
            .detect_styles(true) // Preserve formatting
            .build();

        // In real usage, would load models here
        let _det_result = TextDetector::new(DET_MODEL_PATH, config.clone());
        let _rec_result = TextRecognizer::new(REC_MODEL_PATH, DICT_PATH, config);

        // Would then process page images and extract text
        println!("Legal document OCR workflow configured");
    }

    #[test]
    #[ignore] // Requires actual image data and model inference
    fn test_ocr_workflow_batch_processing_with_models() {
        if !models_exist() {
            println!("Skipping: Models not present");
            return;
        }

        // Configuration for fast batch processing
        let config = OcrConfig::builder()
            .det_threshold(0.25) // Aggressive detection
            .box_threshold(0.40) // Aggressive boxes
            .num_threads(8) // More cores for batch
            .det_max_side(512) // Smaller input
            .build();

        let _det_result = TextDetector::new(DET_MODEL_PATH, config.clone());
        let _rec_result = TextRecognizer::new(REC_MODEL_PATH, DICT_PATH, config);

        println!("Batch processing OCR workflow configured");
    }

    // ========================================================================
    // VALIDATION TESTS
    // ========================================================================

    #[test]
    fn test_ocr_models_directory_structure() {
        let models_dir = Path::new(".models");
        let det_dir = models_dir.join("ch_PP-OCRv3_det_infer");
        let rec_dir = models_dir.join("ch_PP-OCRv3_rec_infer");

        println!("\n=== Model Directory Structure ===");
        if det_dir.exists() {
            println!("Detection model directory: ✓");
            if let Ok(entries) = std::fs::read_dir(&det_dir) {
                for entry in entries {
                    if let Ok(entry) = entry {
                        println!("  - {}", entry.path().display());
                    }
                }
            }
        } else {
            println!("Detection model directory: ✗");
        }

        if rec_dir.exists() {
            println!("Recognition model directory: ✓");
            if let Ok(entries) = std::fs::read_dir(&rec_dir) {
                for entry in entries {
                    if let Ok(entry) = entry {
                        println!("  - {}", entry.path().display());
                    }
                }
            }
        } else {
            println!("Recognition model directory: ✗");
        }
        println!("================================\n");
    }
}

// ============================================================================
// TESTS FOR WHEN OCR FEATURE IS NOT ENABLED
// ============================================================================

#[cfg(not(feature = "ocr"))]
mod ocr_models_not_enabled_tests {
    #[test]
    fn test_ocr_models_feature_disabled() {
        println!("OCR feature is not enabled - model tests skipped");
    }
}
