#![allow(clippy::manual_is_multiple_of)]
#![allow(dead_code)]
//! OCR integration tests.
//!
//! These tests verify the OCR pipeline works correctly.
//! Note: Tests requiring actual ONNX models are marked with #[ignore]
//! and should be run with `cargo test --features ocr -- --ignored`
//! after placing model files in the appropriate location.

#![cfg(feature = "ocr")]

use image::{DynamicImage, GenericImageView, RgbImage};
use pdf_oxide::ocr::{
    crop_text_region, preprocess_for_detection, preprocess_for_recognition, OcrConfig,
    OcrConfigBuilder, OcrExtractOptions, OcrOutput, OcrSpan,
};

/// Create a simple test image with solid color.
fn create_test_image(width: u32, height: u32) -> DynamicImage {
    let img = RgbImage::from_fn(width, height, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, 128u8])
    });
    DynamicImage::ImageRgb8(img)
}

// =============================================================================
// Preprocessing Tests
// =============================================================================

#[test]
fn test_preprocess_for_detection_basic() {
    let img = create_test_image(640, 480);
    let (tensor, scale) = preprocess_for_detection(&img, 960).unwrap();

    // Check tensor shape [1, 3, H, W]
    assert_eq!(tensor.shape()[0], 1); // Batch size
    assert_eq!(tensor.shape()[1], 3); // RGB channels

    // Height and width should be padded to multiple of 32
    assert!(tensor.shape()[2] % 32 == 0);
    assert!(tensor.shape()[3] % 32 == 0);

    // Scale should be 1.0 since image fits within max_side
    assert!((scale - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_preprocess_for_detection_large_image() {
    let img = create_test_image(2000, 1500);
    let (tensor, scale) = preprocess_for_detection(&img, 960).unwrap();

    // Scale should be < 1.0 since image is larger than max_side
    assert!(scale < 1.0);

    // Dimensions should be reduced
    assert!(tensor.shape()[2] <= 960);
    assert!(tensor.shape()[3] <= 960);
}

#[test]
fn test_preprocess_for_recognition_basic() {
    let img = create_test_image(200, 50);
    let tensor = preprocess_for_recognition(&img, 48).unwrap();

    // Check tensor shape [1, 3, 48, W]
    assert_eq!(tensor.shape()[0], 1);
    assert_eq!(tensor.shape()[1], 3);
    assert_eq!(tensor.shape()[2], 48); // Target height

    // Width should be padded to multiple of 4
    assert!(tensor.shape()[3] % 4 == 0);
}

#[test]
fn test_preprocess_for_recognition_normalization() {
    let img = create_test_image(100, 50);
    let tensor = preprocess_for_recognition(&img, 48).unwrap();

    // Values should be in [-1, 1] range (symmetric normalization)
    for val in tensor.iter() {
        assert!(*val >= -1.0 && *val <= 1.0, "Value {} out of range", val);
    }
}

#[test]
fn test_crop_text_region_basic() {
    let img = create_test_image(100, 100);
    let polygon = [[10.0, 10.0], [50.0, 10.0], [50.0, 30.0], [10.0, 30.0]];

    let crop = crop_text_region(&img, &polygon).unwrap();
    let (w, h) = crop.dimensions();

    assert_eq!(w, 40); // 50 - 10
    assert_eq!(h, 20); // 30 - 10
}

#[test]
fn test_crop_text_region_clamps_to_bounds() {
    let img = create_test_image(100, 100);
    // Polygon extends beyond image bounds
    let polygon = [
        [-10.0, -10.0],
        [150.0, -10.0],
        [150.0, 150.0],
        [-10.0, 150.0],
    ];

    let crop = crop_text_region(&img, &polygon).unwrap();
    let (w, h) = crop.dimensions();

    // Should be clamped to image size
    assert!(w <= 100);
    assert!(h <= 100);
}

// =============================================================================
// Configuration Tests
// =============================================================================

#[test]
fn test_ocr_config_default() {
    let config = OcrConfig::default();

    assert!((config.det_threshold - 0.3).abs() < 0.01);
    assert!((config.box_threshold - 0.5).abs() < 0.01);
    assert!((config.unclip_ratio - 1.5).abs() < 0.01);
    assert_eq!(config.det_max_side, 960);
    assert_eq!(config.rec_target_height, 48);
}

#[test]
fn test_ocr_config_builder() {
    let config = OcrConfigBuilder::new()
        .det_threshold(0.4)
        .box_threshold(0.6)
        .unclip_ratio(2.0)
        .det_max_side(1280)
        .rec_target_height(32)
        .num_threads(4)
        .build();

    assert!((config.det_threshold - 0.4).abs() < 0.01);
    assert!((config.box_threshold - 0.6).abs() < 0.01);
    assert!((config.unclip_ratio - 2.0).abs() < 0.01);
    assert_eq!(config.det_max_side, 1280);
    assert_eq!(config.rec_target_height, 32);
    assert_eq!(config.num_threads, 4);
}

#[test]
fn test_ocr_config_clamping() {
    let config = OcrConfigBuilder::new()
        .det_threshold(2.0) // Should be clamped to 1.0
        .box_threshold(-0.5) // Should be clamped to 0.0
        .build();

    assert!((config.det_threshold - 1.0).abs() < 0.01);
    assert!((config.box_threshold - 0.0).abs() < 0.01);
}

// =============================================================================
// OcrExtractOptions Tests
// =============================================================================

#[test]
fn test_ocr_extract_options_default() {
    let options = OcrExtractOptions::default();

    // Default assumes 300 DPI
    let expected_scale = 300.0 / 72.0;
    assert!((options.scale - expected_scale).abs() < 0.01);
    assert!(options.fallback_to_native);
}

#[test]
fn test_ocr_extract_options_with_dpi() {
    let options = OcrExtractOptions::with_dpi(150.0);

    let expected_scale = 150.0 / 72.0;
    assert!((options.scale - expected_scale).abs() < 0.01);
}

// =============================================================================
// OcrSpan and OcrOutput Tests
// =============================================================================

#[test]
fn test_ocr_span_to_text_span() {
    let span = OcrSpan {
        text: "Hello".to_string(),
        polygon: [[0.0, 0.0], [100.0, 0.0], [100.0, 40.0], [0.0, 40.0]],
        confidence: 0.95,
        char_confidences: vec![0.9, 0.95, 0.92, 0.97, 0.96],
    };

    // Scale of 4.0 (simulating 288 DPI)
    let text_span = span.to_text_span(0, 4.0);

    assert_eq!(text_span.text, "Hello");
    assert_eq!(text_span.font_name, "OCR");
    assert_eq!(text_span.sequence, 0);

    // Bounding box should be scaled down by factor of 4
    assert!((text_span.bbox.x - 0.0).abs() < 0.01);
    assert!((text_span.bbox.right() - 25.0).abs() < 0.01); // 100 / 4
    assert!((text_span.bbox.bottom() - 10.0).abs() < 0.01); // 40 / 4

    // Font size estimated from height (40 pixels / 4 scale * 0.75)
    let expected_font_size = (40.0 / 4.0) * 0.75;
    assert!((text_span.font_size - expected_font_size).abs() < 0.5);
}

#[test]
fn test_ocr_span_bounding_rect() {
    let span = OcrSpan {
        text: "Test".to_string(),
        polygon: [[10.0, 20.0], [110.0, 20.0], [110.0, 60.0], [10.0, 60.0]],
        confidence: 0.9,
        char_confidences: vec![],
    };

    let rect = span.bounding_rect();

    assert!((rect.x - 10.0).abs() < 0.01);
    assert!((rect.y - 20.0).abs() < 0.01);
    assert!((rect.right() - 110.0).abs() < 0.01);
    assert!((rect.bottom() - 60.0).abs() < 0.01);
}

#[test]
fn test_ocr_output_text() {
    let output = OcrOutput {
        spans: vec![
            OcrSpan {
                text: "Hello".to_string(),
                polygon: [[0.0, 0.0], [50.0, 0.0], [50.0, 20.0], [0.0, 20.0]],
                confidence: 0.95,
                char_confidences: vec![],
            },
            OcrSpan {
                text: "World".to_string(),
                polygon: [[60.0, 0.0], [110.0, 0.0], [110.0, 20.0], [60.0, 20.0]],
                confidence: 0.92,
                char_confidences: vec![],
            },
        ],
        total_confidence: 0.935,
    };

    assert_eq!(output.text(), "Hello World");
}

#[test]
fn test_ocr_output_reading_order() {
    let output = OcrOutput {
        spans: vec![
            // Second line (higher Y)
            OcrSpan {
                text: "Line2".to_string(),
                polygon: [[0.0, 50.0], [50.0, 50.0], [50.0, 70.0], [0.0, 70.0]],
                confidence: 0.9,
                char_confidences: vec![],
            },
            // First line (lower Y)
            OcrSpan {
                text: "Line1".to_string(),
                polygon: [[0.0, 0.0], [50.0, 0.0], [50.0, 20.0], [0.0, 20.0]],
                confidence: 0.9,
                char_confidences: vec![],
            },
        ],
        total_confidence: 0.9,
    };

    // Should sort by Y position (top to bottom)
    assert_eq!(output.text_in_reading_order(), "Line1 Line2");
}

#[test]
fn test_ocr_output_to_text_spans() {
    let output = OcrOutput {
        spans: vec![
            OcrSpan {
                text: "First".to_string(),
                polygon: [[0.0, 0.0], [50.0, 0.0], [50.0, 20.0], [0.0, 20.0]],
                confidence: 0.95,
                char_confidences: vec![],
            },
            OcrSpan {
                text: "Second".to_string(),
                polygon: [[60.0, 0.0], [120.0, 0.0], [120.0, 20.0], [60.0, 20.0]],
                confidence: 0.92,
                char_confidences: vec![],
            },
        ],
        total_confidence: 0.935,
    };

    let text_spans = output.to_text_spans(1.0);

    assert_eq!(text_spans.len(), 2);
    assert_eq!(text_spans[0].text, "First");
    assert_eq!(text_spans[0].sequence, 0);
    assert_eq!(text_spans[1].text, "Second");
    assert_eq!(text_spans[1].sequence, 1);
}

// =============================================================================
// Integration Tests (require models - marked as ignored)
// =============================================================================

/// Test end-to-end OCR on a simple test image.
///
/// To run: Place models in tests/fixtures/ocr/models/ and run:
/// `cargo test --features ocr -- --ignored test_ocr_simple_image`
#[test]
#[ignore = "Requires ONNX model files"]
fn test_ocr_simple_image() {
    use pdf_oxide::ocr::{OcrConfig, OcrEngine};

    let det_model = "tests/fixtures/ocr/models/en_PP-OCRv5_det_infer.onnx";
    let rec_model = "tests/fixtures/ocr/models/en_PP-OCRv5_rec_infer.onnx";
    let dict_path = "tests/fixtures/ocr/models/en_dict.txt";

    let engine = OcrEngine::new(det_model, rec_model, dict_path, OcrConfig::default())
        .expect("Failed to create OCR engine");

    // Create a simple test image with text
    let img = image::open("tests/fixtures/ocr/images/hello_world.png")
        .expect("Failed to load test image");

    let result = engine.ocr_image(&img).expect("OCR failed");

    assert!(!result.spans.is_empty(), "No text detected");
    assert!(result.total_confidence > 0.5, "Low confidence");

    let text = result.text_in_reading_order().to_lowercase();
    assert!(
        text.contains("hello") || text.contains("world"),
        "Expected 'hello' or 'world' in output, got: {}",
        text
    );
}

/// Test OCR on a scanned PDF page.
#[test]
#[ignore = "Requires ONNX model files and scanned PDF"]
fn test_ocr_scanned_pdf() {
    use pdf_oxide::{
        ocr::{self, OcrConfig, OcrEngine, OcrExtractOptions},
        PdfDocument,
    };

    let det_model = "tests/fixtures/ocr/models/en_PP-OCRv5_det_infer.onnx";
    let rec_model = "tests/fixtures/ocr/models/en_PP-OCRv5_rec_infer.onnx";
    let dict_path = "tests/fixtures/ocr/models/en_dict.txt";

    let engine = OcrEngine::new(det_model, rec_model, dict_path, OcrConfig::default())
        .expect("Failed to create OCR engine");

    let mut doc = PdfDocument::open("tests/fixtures/ocr/pdfs/scanned_sample.pdf")
        .expect("Failed to open PDF");

    // Check if page needs OCR
    let needs_ocr = ocr::needs_ocr(&mut doc, 0).expect("Failed to check if OCR needed");
    assert!(needs_ocr, "Expected scanned PDF to need OCR");

    // Run OCR
    let text =
        ocr::ocr_page(&mut doc, 0, &engine, &OcrExtractOptions::default()).expect("OCR failed");

    assert!(!text.is_empty(), "No text extracted from scanned PDF");
}

/// Test automatic OCR fallback.
#[test]
#[ignore = "Requires ONNX model files"]
fn test_extract_text_with_ocr_auto() {
    use pdf_oxide::{
        ocr::{self, OcrConfig, OcrEngine, OcrExtractOptions},
        PdfDocument,
    };

    let det_model = "tests/fixtures/ocr/models/en_PP-OCRv5_det_infer.onnx";
    let rec_model = "tests/fixtures/ocr/models/en_PP-OCRv5_rec_infer.onnx";
    let dict_path = "tests/fixtures/ocr/models/en_dict.txt";

    let engine = OcrEngine::new(det_model, rec_model, dict_path, OcrConfig::default())
        .expect("Failed to create OCR engine");

    // Test with native PDF (should use native extraction)
    let mut native_doc =
        PdfDocument::open("tests/fixtures/simple.pdf").expect("Failed to open native PDF");
    let native_text =
        ocr::extract_text_with_ocr(&mut native_doc, 0, Some(&engine), OcrExtractOptions::default())
            .expect("Failed to extract text");
    // Native PDF should have text without needing OCR
    assert!(!native_text.is_empty());

    // Test with scanned PDF (should use OCR)
    let mut scanned_doc = PdfDocument::open("tests/fixtures/ocr/pdfs/scanned_sample.pdf")
        .expect("Failed to open scanned PDF");
    let ocr_text = ocr::extract_text_with_ocr(
        &mut scanned_doc,
        0,
        Some(&engine),
        OcrExtractOptions::default(),
    )
    .expect("Failed to extract text with OCR");
    assert!(!ocr_text.is_empty());
}
