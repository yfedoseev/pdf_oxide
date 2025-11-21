# OCR Feature Implementation Plan

## Executive Summary

This document provides a comprehensive, TDD-driven implementation plan for adding PaddleOCR-based text extraction to pdf_oxide. The implementation uses ONNX Runtime for CPU-only inference, integrates seamlessly with existing text extraction pipelines, and maintains the library's performance standards.

**Target Completion**: 3 implementation phases
**Dependencies**: `ort` crate (ONNX Runtime), `image` crate
**Models**: PaddleOCR PP-OCRv4 (detection + recognition)

---

## Table of Contents

1. [Research Summary](#1-research-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Dependencies Analysis](#3-dependencies-analysis)
4. [Implementation Phases](#4-implementation-phases)
5. [TDD Strategy](#5-tdd-strategy)
6. [Benchmarking Plan](#6-benchmarking-plan)
7. [Task Breakdown](#7-task-breakdown)
8. [Risk Mitigation](#8-risk-mitigation)

---

## 1. Research Summary

### 1.1 Existing Rust PaddleOCR Implementations

| Crate | Status | License | Notes |
|-------|--------|---------|-------|
| `paddle-ocr-rs` | Active | MIT | PP-OCRv5 support, uses `ort` |
| `paddleocr_rs` | Active | AGPL-3.0 | PP-OCRv4, fewer downloads |
| `PaddleOCR.rs` | Maintained | Apache-2.0 | V3 inference framework |

**Decision**: Build custom implementation rather than wrap existing crates because:
- Need tight integration with existing `TextSpan` pipeline
- Control over preprocessing/postprocessing
- Avoid AGPL license contamination
- Can optimize specifically for our use case

### 1.2 ONNX Runtime (`ort` crate)

**Selected Version**: `ort = "2.0"` (latest stable)

Key features:
- Safe Rust wrapper for ONNX Runtime 1.22
- CPU optimization with multi-threading
- Supports dynamic input shapes (required for OCR)
- Used by production systems (Hugging Face Text Embeddings, Google Magika)

**API Pattern**:
```rust
use ort::{Session, GraphOptimizationLevel};

let session = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .with_intra_threads(4)?
    .commit_from_file("model.onnx")?;

let outputs = session.run(ort::inputs![input_tensor]?)?;
```

### 1.3 Image Source for OCR

**Key Insight**: Scanned PDFs are typically just PDF wrappers around embedded images. We don't need to *render* the PDF - we just need to **extract the embedded images**.

**Decision**: Use existing `extract_images()` from pdf_oxide
- Already implemented in `src/extractors/images.rs`
- Pure Rust, no external dependencies
- Works for 99% of scanned PDF use cases
- `PdfImage` struct already has width, height, pixel data

**How it works**:
```rust
// Scanned PDF = PDF wrapper around a full-page image
let images = doc.extract_images(page)?;
let page_image = images.first(); // This IS the scan
let ocr_result = engine.ocr_image(page_image)?;
```

**No pdfium needed** - keeps pdf_oxide pure Rust with no C++ dependencies.

### 1.4 Integration Points in pdf_oxide

**Existing structures to leverage**:

```rust
// src/layout/text_block.rs - Target output format
pub struct TextSpan {
    pub text: String,
    pub bbox: Rect,
    pub font_name: String,
    pub font_size: f32,
    pub font_weight: FontWeight,
    pub color: Color,
    pub mcid: Option<u32>,
    pub sequence: usize,
}

// src/extractors/images.rs - Image handling exists
pub struct PdfImage {
    width: u32,
    height: u32,
    color_space: ColorSpace,
    // ... can be used as OCR input
}

// src/geometry/mod.rs - Geometry primitives
pub struct Rect { x1, y1, x2, y2 }
pub struct Point { x, y }
```

### 1.5 OCR Accuracy Metrics

**Primary Metrics**:
- **CER (Character Error Rate)**: `(Insertions + Deletions + Substitutions) / Total Characters`
- **WER (Word Error Rate)**: Same formula at word level
- **Target**: CER < 5%, WER < 10% on clean scans

**Secondary Metrics**:
- Inference latency per page
- Memory usage
- Bounding box accuracy (IoU with ground truth)

---

## 2. Architecture Overview

### 2.1 Module Structure

```
src/
├── ocr/
│   ├── mod.rs              # Public API, feature gate
│   ├── engine.rs           # OcrEngine struct, model management
│   ├── detector.rs         # DBNet++ text detection
│   ├── recognizer.rs       # SVTR text recognition
│   ├── preprocessor.rs     # Image preprocessing
│   ├── postprocessor.rs    # Box extraction, NMS
│   ├── page_analyzer.rs    # Scanned page detection
│   └── models/
│       ├── mod.rs          # Model loading utilities
│       └── embedded.rs     # Bundled model bytes (optional)
├── lib.rs                  # Add `pub mod ocr` with feature gate
└── python.rs               # Add OCR bindings
```

### 2.2 Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         OCR Pipeline                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  PDF Page ─────┬──► Native Text Check ──► Has Text? ──► Return Text │
│                │                              │                      │
│                │                              ▼ No                   │
│                │                                                     │
│                └──► Render to Image ──► Preprocess ──► Detection    │
│                          │                                │          │
│                          ▼                                ▼          │
│                     RGB Image              Text Boxes (polygons)     │
│                     (300 DPI)                     │                  │
│                          │                        │                  │
│                          └────────────────────────┘                  │
│                                      │                               │
│                                      ▼                               │
│                              Crop Text Regions                       │
│                                      │                               │
│                                      ▼                               │
│                              Recognition ──► OcrSpan[]               │
│                                      │                               │
│                                      ▼                               │
│                              Style Detection                         │
│                                      │                               │
│                                      ▼                               │
│                              Convert to TextSpan[]                   │
│                                      │                               │
│                                      ▼                               │
│                              Layout Analysis (existing)              │
│                                      │                               │
│                                      ▼                               │
│                              Markdown/HTML Export                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Key Data Structures

```rust
/// OCR engine managing models and inference
pub struct OcrEngine {
    detector: TextDetector,
    recognizer: TextRecognizer,
    config: OcrConfig,
}

/// Configuration for OCR processing
#[derive(Debug, Clone)]
pub struct OcrConfig {
    /// DPI for page rendering (default: 300)
    pub render_dpi: u32,
    /// Detection confidence threshold (default: 0.5)
    pub det_threshold: f32,
    /// Recognition confidence threshold (default: 0.5)
    pub rec_threshold: f32,
    /// Number of inference threads (default: 4)
    pub num_threads: usize,
    /// Enable style detection from OCR (default: true)
    pub detect_styles: bool,
}

/// Raw OCR result before conversion to TextSpan
#[derive(Debug, Clone)]
pub struct OcrSpan {
    pub text: String,
    pub polygon: [Point; 4],  // Quadrilateral bounding box
    pub confidence: f32,
    pub char_boxes: Option<Vec<Rect>>,  // Per-character boxes if available
}

/// Style information inferred from OCR geometry
#[derive(Debug, Clone)]
pub struct InferredStyle {
    pub estimated_font_size: f32,
    pub is_bold: bool,
    pub is_heading: bool,
}
```

---

## 3. Dependencies Analysis

### 3.1 Required Dependencies

```toml
[dependencies]
# OCR feature dependencies
ort = { version = "2.0", optional = true, default-features = false, features = ["ndarray"] }
image = { version = "0.25", optional = true }
imageproc = { version = "0.24", optional = true }
ndarray = { version = "0.15", optional = true }

[features]
default = []
ocr = ["dep:ort", "dep:image", "dep:imageproc", "dep:ndarray"]
```

**Note**: No external C++ dependencies needed! We use existing `extract_images()` to get page images from scanned PDFs.

### 3.2 Model Files

**Detection Model (DBNet++ PP-OCRv4)**:
- File: `ch_PP-OCRv4_det_infer.onnx`
- Size: ~4.5 MB
- Input: `[1, 3, H, W]` float32, normalized
- Output: `[1, 1, H, W]` probability map

**Recognition Model (SVTR PP-OCRv4)**:
- File: `ch_PP-OCRv4_rec_infer.onnx`
- Size: ~10 MB
- Input: `[B, 3, 48, W]` float32, normalized
- Output: `[B, W/4, num_chars]` logits

**Character Dictionary**:
- File: `ppocr_keys_v1.txt`
- Size: ~100 KB
- Contains 6623 characters for Chinese/English/symbols

### 3.3 Model Distribution Strategy

**Option A: Bundled Models (Recommended for v1)**
```rust
// Compile models into binary
const DET_MODEL: &[u8] = include_bytes!("models/det.onnx");
const REC_MODEL: &[u8] = include_bytes!("models/rec.onnx");
```
- Pros: Zero-config, works offline
- Cons: +15 MB binary size

**Option B: Download on First Use**
```rust
impl OcrEngine {
    pub fn new() -> Result<Self> {
        let model_dir = dirs::cache_dir()?.join("pdf_oxide/models");
        ensure_models_downloaded(&model_dir)?;
        // ...
    }
}
```
- Pros: Smaller initial binary
- Cons: Network dependency, first-run delay

**Decision**: Start with Option A for simplicity, add Option B later

---

## 4. Implementation Phases

### Phase 1: Core OCR Engine (MVP)

**Goal**: Basic OCR working end-to-end on image input

**Deliverables**:
- [ ] `OcrEngine` struct with model loading
- [ ] `TextDetector` with DBNet++ inference
- [ ] `TextRecognizer` with SVTR inference
- [ ] Basic preprocessing (resize, normalize)
- [ ] `ocr_image()` function for direct image OCR

**Duration**: ~1 week
**Tests**: Unit tests for each component

### Phase 2: PDF Integration

**Goal**: OCR integrated into PDF extraction pipeline

**Deliverables**:
- [ ] `PageAnalyzer` for scanned page detection
- [ ] Extract page images using existing `extract_images()`
- [ ] `OcrSpan` to `TextSpan` conversion
- [ ] Style inference (font size, bold detection)
- [ ] `doc.extract_text_with_ocr()` API
- [ ] `fallback_to_ocr` option in extraction

**Duration**: ~1 week
**Tests**: Integration tests with sample PDFs

### Phase 3: Polish & Python

**Goal**: Production-ready with full API

**Deliverables**:
- [ ] Python bindings for OCR
- [ ] Performance optimization (batching, threading)
- [ ] Model quantization (INT8 option)
- [ ] Documentation and examples
- [ ] Benchmarks

**Duration**: ~1 week
**Tests**: End-to-end tests, benchmark suite

---

## 5. TDD Strategy

### 5.1 Test Categories

```
tests/
├── ocr/
│   ├── test_detector.rs      # Unit tests for detection
│   ├── test_recognizer.rs    # Unit tests for recognition
│   ├── test_preprocessor.rs  # Preprocessing tests
│   ├── test_integration.rs   # End-to-end OCR tests
│   └── test_accuracy.rs      # CER/WER measurements
├── fixtures/
│   └── ocr/
│       ├── images/           # Test images
│       │   ├── simple_text.png
│       │   ├── multi_line.png
│       │   ├── rotated.png
│       │   └── low_quality.png
│       ├── pdfs/             # Scanned PDF samples
│       │   ├── scanned_single.pdf
│       │   ├── scanned_multi.pdf
│       │   └── mixed_native_scan.pdf
│       └── ground_truth/     # Expected outputs
│           ├── simple_text.txt
│           └── multi_line.txt
```

### 5.2 Test-First Development Order

#### Phase 1 Tests (Write First)

```rust
// tests/ocr/test_preprocessor.rs
#[test]
fn test_resize_maintains_aspect_ratio() {
    let img = create_test_image(800, 600);
    let resized = preprocess_for_detection(&img, 640);
    assert!(resized.width() <= 640);
    assert!(resized.height() <= 640);
    // Aspect ratio preserved within 1%
    let orig_ratio = 800.0 / 600.0;
    let new_ratio = resized.width() as f32 / resized.height() as f32;
    assert!((orig_ratio - new_ratio).abs() < 0.01);
}

#[test]
fn test_normalize_values() {
    let img = create_solid_color_image(100, 100, [128, 128, 128]);
    let tensor = normalize_image(&img);
    // Values should be in [-1, 1] range after normalization
    for val in tensor.iter() {
        assert!(*val >= -1.0 && *val <= 1.0);
    }
}

// tests/ocr/test_detector.rs
#[test]
fn test_detector_loads_model() {
    let detector = TextDetector::new(DET_MODEL_PATH);
    assert!(detector.is_ok());
}

#[test]
fn test_detector_finds_text_boxes() {
    let detector = TextDetector::new(DET_MODEL_PATH).unwrap();
    let img = load_test_image("fixtures/ocr/images/simple_text.png");
    let boxes = detector.detect(&img).unwrap();

    assert!(!boxes.is_empty(), "Should detect at least one text region");
    for b in &boxes {
        assert!(b.confidence > 0.5);
        assert!(b.polygon.iter().all(|p| p.x >= 0.0 && p.y >= 0.0));
    }
}

// tests/ocr/test_recognizer.rs
#[test]
fn test_recognizer_decodes_text() {
    let recognizer = TextRecognizer::new(REC_MODEL_PATH, DICT_PATH).unwrap();
    let img = load_test_image("fixtures/ocr/images/simple_text.png");
    // Assume simple_text.png contains "Hello World"
    let result = recognizer.recognize(&img).unwrap();

    assert_eq!(result.text.trim(), "Hello World");
    assert!(result.confidence > 0.8);
}

// tests/ocr/test_accuracy.rs
#[test]
fn test_cer_calculation() {
    let ground_truth = "Hello World";
    let ocr_output = "Helo Worid";
    let cer = calculate_cer(ground_truth, ocr_output);
    // 2 errors (missing 'l', 'l' -> 'i') out of 11 chars = ~18%
    assert!((cer - 0.18).abs() < 0.05);
}

#[test]
fn test_english_accuracy_threshold() {
    let engine = OcrEngine::new().unwrap();
    let test_cases = load_test_cases("fixtures/ocr/ground_truth/");

    let mut total_cer = 0.0;
    for case in &test_cases {
        let result = engine.ocr_image(&case.image).unwrap();
        total_cer += calculate_cer(&case.ground_truth, &result.text);
    }
    let avg_cer = total_cer / test_cases.len() as f32;

    assert!(avg_cer < 0.05, "Average CER {} exceeds 5% threshold", avg_cer);
}
```

#### Phase 2 Tests (Write First)

```rust
// tests/ocr/test_page_analyzer.rs
#[test]
fn test_detects_scanned_page() {
    let doc = PdfDocument::open("fixtures/ocr/pdfs/scanned_single.pdf").unwrap();
    let is_scanned = needs_ocr(&doc, 0).unwrap();
    assert!(is_scanned, "Should detect scanned PDF page");
}

#[test]
fn test_native_text_page_not_scanned() {
    let doc = PdfDocument::open("fixtures/simple.pdf").unwrap();
    let is_scanned = needs_ocr(&doc, 0).unwrap();
    assert!(!is_scanned, "Should not flag native text page as scanned");
}

#[test]
fn test_mixed_pdf_detection() {
    let doc = PdfDocument::open("fixtures/ocr/pdfs/mixed_native_scan.pdf").unwrap();
    // Page 0: native text, Page 1: scanned
    assert!(!needs_ocr(&doc, 0).unwrap());
    assert!(needs_ocr(&doc, 1).unwrap());
}

// tests/ocr/test_integration.rs
#[test]
fn test_ocr_span_to_text_span_conversion() {
    let ocr_span = OcrSpan {
        text: "Test".to_string(),
        polygon: [Point::new(0.0, 0.0), Point::new(100.0, 0.0),
                  Point::new(100.0, 20.0), Point::new(0.0, 20.0)],
        confidence: 0.95,
        char_boxes: None,
    };

    let text_span: TextSpan = ocr_span.into();

    assert_eq!(text_span.text, "Test");
    assert!(text_span.font_size > 0.0);
    assert_eq!(text_span.font_name, "OCR-Detected");
}

#[test]
fn test_extract_text_with_ocr_fallback() {
    let mut doc = PdfDocument::open("fixtures/ocr/pdfs/scanned_single.pdf").unwrap();

    // Without OCR: should return empty or minimal text
    let native = doc.extract_text(0).unwrap();
    assert!(native.trim().is_empty() || native.len() < 50);

    // With OCR fallback: should return actual content
    let with_ocr = doc.extract_text_with_options(0, ExtractOptions {
        fallback_to_ocr: true,
        ..Default::default()
    }).unwrap();

    assert!(with_ocr.len() > 100, "OCR should extract substantial text");
}

#[test]
fn test_markdown_export_with_ocr() {
    let mut doc = PdfDocument::open("fixtures/ocr/pdfs/scanned_single.pdf").unwrap();

    let md = doc.to_markdown_with_options(0, MarkdownOptions {
        fallback_to_ocr: true,
        detect_headings: true,
        ..Default::default()
    }).unwrap();

    // Should have markdown structure
    assert!(md.contains('#') || md.contains("**"), "Should detect headings or bold");
}
```

### 5.3 Property-Based Tests

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_preprocessing_preserves_content(
        width in 100u32..2000,
        height in 100u32..2000,
    ) {
        let img = create_random_image(width, height);
        let processed = preprocess_for_detection(&img, 640);

        // Should not panic
        // Output should have valid dimensions
        prop_assert!(processed.width() > 0);
        prop_assert!(processed.height() > 0);
        prop_assert!(processed.width() <= 640);
        prop_assert!(processed.height() <= 640);
    }

    #[test]
    fn test_cer_is_bounded(
        s1 in "[a-z]{1,100}",
        s2 in "[a-z]{1,100}",
    ) {
        let cer = calculate_cer(&s1, &s2);
        // CER can exceed 1.0 with many insertions, but should be non-negative
        prop_assert!(cer >= 0.0);
    }
}
```

### 5.4 Benchmark Tests

```rust
// benches/ocr_benchmark.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_detection(c: &mut Criterion) {
    let detector = TextDetector::new(DET_MODEL_PATH).unwrap();
    let img = load_test_image("fixtures/ocr/images/document_page.png");

    c.bench_function("text_detection_a4_300dpi", |b| {
        b.iter(|| detector.detect(black_box(&img)))
    });
}

fn bench_recognition(c: &mut Criterion) {
    let recognizer = TextRecognizer::new(REC_MODEL_PATH, DICT_PATH).unwrap();
    let crops: Vec<_> = (0..20).map(|_| create_text_crop(200, 48)).collect();

    c.bench_function("recognition_20_words", |b| {
        b.iter(|| {
            for crop in &crops {
                black_box(recognizer.recognize(crop).unwrap());
            }
        })
    });
}

fn bench_full_pipeline(c: &mut Criterion) {
    let engine = OcrEngine::new().unwrap();
    let img = load_test_image("fixtures/ocr/images/document_page.png");

    c.bench_function("full_ocr_pipeline_a4", |b| {
        b.iter(|| engine.ocr_image(black_box(&img)))
    });
}

criterion_group!(benches, bench_detection, bench_recognition, bench_full_pipeline);
criterion_main!(benches);
```

---

## 6. Benchmarking Plan

### 6.1 Performance Targets

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Detection latency | < 50ms/page | A4 @ 300 DPI |
| Recognition latency | < 500ms/page | ~100 words |
| Total pipeline | < 1s/page | End-to-end |
| Memory usage | < 200 MB | Peak during inference |
| Model load time | < 500ms | Cold start |

### 6.2 Benchmark Dataset

Create standardized benchmark set:

```
benches/fixtures/
├── synthetic/
│   ├── lorem_ipsum_1page.png    # Clean synthetic text
│   ├── lorem_ipsum_10pages/     # Multi-page consistency
│   └── dense_text.png           # Maximum text density
├── real_world/
│   ├── scanned_book_page.png    # Typical book scan
│   ├── receipt.png              # Low quality scan
│   ├── newspaper.png            # Multi-column layout
│   └── handwritten.png          # Edge case (expected low accuracy)
└── stress/
    ├── 4k_resolution.png        # Large image handling
    ├── tiny_text.png            # Minimum readable size
    └── rotated_45deg.png        # Rotation handling
```

### 6.3 Accuracy Benchmark

```rust
// benches/accuracy_benchmark.rs
struct AccuracyReport {
    dataset: String,
    num_samples: usize,
    avg_cer: f32,
    avg_wer: f32,
    p95_cer: f32,  // 95th percentile
    avg_latency_ms: f32,
}

fn run_accuracy_benchmark() -> Vec<AccuracyReport> {
    let datasets = vec![
        ("synthetic", "benches/fixtures/synthetic/"),
        ("real_world", "benches/fixtures/real_world/"),
    ];

    datasets.iter().map(|(name, path)| {
        let samples = load_benchmark_samples(path);
        let results: Vec<_> = samples.iter().map(|s| {
            let start = Instant::now();
            let ocr_text = engine.ocr_image(&s.image).unwrap();
            let latency = start.elapsed();

            BenchmarkResult {
                cer: calculate_cer(&s.ground_truth, &ocr_text.text),
                wer: calculate_wer(&s.ground_truth, &ocr_text.text),
                latency_ms: latency.as_millis() as f32,
            }
        }).collect();

        AccuracyReport {
            dataset: name.to_string(),
            num_samples: samples.len(),
            avg_cer: results.iter().map(|r| r.cer).sum::<f32>() / results.len() as f32,
            avg_wer: results.iter().map(|r| r.wer).sum::<f32>() / results.len() as f32,
            p95_cer: percentile(&results.iter().map(|r| r.cer).collect::<Vec<_>>(), 0.95),
            avg_latency_ms: results.iter().map(|r| r.latency_ms).sum::<f32>() / results.len() as f32,
        }
    }).collect()
}
```

---

## 7. Task Breakdown

### 7.1 Phase 1: Core OCR Engine

#### Task 1.1: Project Setup
- **Dependencies**: None
- **Effort**: 2 hours
- **Deliverables**:
  - [ ] Add `ocr` feature flag to `Cargo.toml`
  - [ ] Create `src/ocr/mod.rs` with feature gate
  - [ ] Download and verify PaddleOCR ONNX models
  - [ ] Add models to `models/ocr/` directory
  - [ ] Create test fixtures directory structure

```toml
# Cargo.toml additions
[features]
ocr = ["dep:ort", "dep:image", "dep:ndarray"]

[dependencies]
ort = { version = "2.0", optional = true }
image = { version = "0.25", optional = true }
ndarray = { version = "0.15", optional = true }
```

#### Task 1.2: Image Preprocessing (TDD)
- **Dependencies**: Task 1.1
- **Effort**: 4 hours
- **Tests First**:
  - [ ] `test_resize_maintains_aspect_ratio`
  - [ ] `test_normalize_values`
  - [ ] `test_pad_to_divisible`
  - [ ] `test_hwc_to_chw_conversion`
- **Implementation**:
  - [ ] `preprocessor.rs`: `resize_image()`, `normalize()`, `to_tensor()`

```rust
// Expected API
pub fn preprocess_for_detection(img: &DynamicImage, max_side: u32) -> Array4<f32>;
pub fn preprocess_for_recognition(crop: &DynamicImage, target_height: u32) -> Array4<f32>;
```

#### Task 1.3: Text Detection (TDD)
- **Dependencies**: Task 1.2
- **Effort**: 8 hours
- **Tests First**:
  - [ ] `test_detector_loads_model`
  - [ ] `test_detector_output_shape`
  - [ ] `test_detector_finds_text_boxes`
  - [ ] `test_detection_threshold_filtering`
- **Implementation**:
  - [ ] `detector.rs`: `TextDetector` struct
  - [ ] Model loading with `ort::Session`
  - [ ] Probability map → polygon extraction
  - [ ] Non-maximum suppression (NMS)

```rust
// Expected API
pub struct TextDetector { session: Session }

impl TextDetector {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self>;
    pub fn detect(&self, image: &DynamicImage) -> Result<Vec<DetectedBox>>;
}

pub struct DetectedBox {
    pub polygon: [Point; 4],
    pub confidence: f32,
}
```

#### Task 1.4: Text Recognition (TDD)
- **Dependencies**: Task 1.2
- **Effort**: 8 hours
- **Tests First**:
  - [ ] `test_recognizer_loads_model`
  - [ ] `test_dictionary_loading`
  - [ ] `test_ctc_decoding`
  - [ ] `test_recognizer_decodes_text`
  - [ ] `test_confidence_calculation`
- **Implementation**:
  - [ ] `recognizer.rs`: `TextRecognizer` struct
  - [ ] Dictionary loading
  - [ ] CTC greedy decoding
  - [ ] Confidence score calculation

```rust
// Expected API
pub struct TextRecognizer {
    session: Session,
    dictionary: Vec<char>,
}

impl TextRecognizer {
    pub fn new(model_path: impl AsRef<Path>, dict_path: impl AsRef<Path>) -> Result<Self>;
    pub fn recognize(&self, crop: &DynamicImage) -> Result<RecognitionResult>;
    pub fn recognize_batch(&self, crops: &[DynamicImage]) -> Result<Vec<RecognitionResult>>;
}

pub struct RecognitionResult {
    pub text: String,
    pub confidence: f32,
    pub char_confidences: Vec<f32>,
}
```

#### Task 1.5: OCR Engine Assembly (TDD)
- **Dependencies**: Tasks 1.3, 1.4
- **Effort**: 4 hours
- **Tests First**:
  - [ ] `test_engine_initialization`
  - [ ] `test_full_pipeline_simple_image`
  - [ ] `test_multi_line_detection`
  - [ ] `test_empty_image_handling`
- **Implementation**:
  - [ ] `engine.rs`: `OcrEngine` combining detector + recognizer
  - [ ] Crop extraction from detection boxes
  - [ ] Perspective correction for rotated text
  - [ ] Result aggregation

```rust
// Expected API
pub struct OcrEngine {
    detector: TextDetector,
    recognizer: TextRecognizer,
    config: OcrConfig,
}

impl OcrEngine {
    pub fn new() -> Result<Self>;  // Uses bundled models
    pub fn with_models(det: &Path, rec: &Path, dict: &Path) -> Result<Self>;
    pub fn ocr_image(&self, image: &DynamicImage) -> Result<OcrResult>;
}

pub struct OcrResult {
    pub spans: Vec<OcrSpan>,
    pub total_confidence: f32,
}
```

#### Task 1.6: Phase 1 Integration & Testing
- **Dependencies**: Task 1.5
- **Effort**: 4 hours
- **Deliverables**:
  - [ ] Run full test suite
  - [ ] Verify accuracy targets on test images
  - [ ] Document API in code comments
  - [ ] Create `examples/ocr_basic.rs`

---

### 7.2 Phase 2: PDF Integration

#### Task 2.1: Scanned Page Detection (TDD)
- **Dependencies**: Phase 1 complete
- **Effort**: 4 hours
- **Tests First**:
  - [ ] `test_detects_scanned_page`
  - [ ] `test_native_text_not_scanned`
  - [ ] `test_mixed_pdf_detection`
  - [ ] `test_image_coverage_calculation`
- **Implementation**:
  - [ ] `page_analyzer.rs`: `needs_ocr()` function
  - [ ] Native text check
  - [ ] Image coverage analysis
  - [ ] Hybrid page detection

```rust
// Expected API
pub fn needs_ocr(doc: &PdfDocument, page: usize) -> Result<bool>;

pub struct PageAnalysis {
    pub has_native_text: bool,
    pub image_coverage: f32,  // 0.0 - 1.0
    pub recommended_ocr: bool,
}

pub fn analyze_page(doc: &PdfDocument, page: usize) -> Result<PageAnalysis>;
```

#### Task 2.2: Extract Page Images for OCR
- **Dependencies**: Task 2.1
- **Effort**: 4 hours
- **Tests First**:
  - [ ] `test_extract_page_image_from_scanned_pdf`
  - [ ] `test_extract_largest_image`
  - [ ] `test_pdfimage_to_dynamic_image_conversion`
- **Implementation**:
  - [ ] Use existing `extract_images()` to get embedded images
  - [ ] Select largest image (covers full page for scans)
  - [ ] Convert `PdfImage` → `image::DynamicImage` for OCR input

```rust
// Expected API - leverages existing pdf_oxide functionality
pub fn extract_page_image(doc: &mut PdfDocument, page: usize) -> Result<Option<DynamicImage>> {
    let images = doc.extract_images(page)?;
    // Return largest image (typically the full-page scan)
    images.into_iter()
        .max_by_key(|img| img.width() * img.height())
        .map(|img| img.to_dynamic_image())
        .transpose()
}
```

**Note**: No external rendering library needed - scanned PDFs contain embedded images.

#### Task 2.3: OCR to TextSpan Conversion (TDD)
- **Dependencies**: Task 2.2
- **Effort**: 6 hours
- **Tests First**:
  - [ ] `test_ocr_span_to_text_span`
  - [ ] `test_polygon_to_rect_conversion`
  - [ ] `test_style_inference_large_text`
  - [ ] `test_style_inference_bold_detection`
  - [ ] `test_coordinate_scaling`
- **Implementation**:
  - [ ] `OcrSpan` → `TextSpan` conversion
  - [ ] Style inference from geometry
  - [ ] Coordinate transformation (image → PDF space)
  - [ ] Font size estimation

```rust
// Expected API
impl From<OcrSpan> for TextSpan {
    fn from(ocr: OcrSpan) -> Self;
}

pub struct StyleInference {
    pub page_width: f32,
    pub page_height: f32,
    pub avg_char_height: f32,
}

impl StyleInference {
    pub fn infer_style(&self, span: &OcrSpan) -> InferredStyle;
}
```

#### Task 2.4: Document API Integration (TDD)
- **Dependencies**: Task 2.3
- **Effort**: 6 hours
- **Tests First**:
  - [ ] `test_extract_text_with_ocr_fallback`
  - [ ] `test_ocr_only_mode`
  - [ ] `test_markdown_with_ocr`
  - [ ] `test_html_with_ocr`
- **Implementation**:
  - [ ] Add OCR options to `ExtractOptions`
  - [ ] Modify `extract_text()` for OCR fallback
  - [ ] Integrate with Markdown/HTML export

```rust
// Expected API additions to PdfDocument
impl PdfDocument {
    #[cfg(feature = "ocr")]
    pub fn extract_text_with_options(
        &mut self,
        page: usize,
        options: ExtractOptions
    ) -> Result<String>;

    #[cfg(feature = "ocr")]
    pub fn ocr_page(&mut self, page: usize) -> Result<Vec<TextSpan>>;
}

pub struct ExtractOptions {
    pub fallback_to_ocr: bool,
    pub ocr_config: Option<OcrConfig>,
    // ... existing options
}
```

#### Task 2.5: Phase 2 Integration & Testing
- **Dependencies**: Task 2.4
- **Effort**: 4 hours
- **Deliverables**:
  - [ ] Integration tests with real scanned PDFs
  - [ ] Verify Markdown output quality
  - [ ] Performance benchmarks
  - [ ] Update documentation

---

### 7.3 Phase 3: Polish & Python

#### Task 3.1: Python Bindings (TDD)
- **Dependencies**: Phase 2 complete
- **Effort**: 6 hours
- **Tests First**:
  - [ ] `test_python_ocr_image` (pytest)
  - [ ] `test_python_extract_with_ocr` (pytest)
  - [ ] `test_python_ocr_options` (pytest)
- **Implementation**:
  - [ ] Add OCR methods to PyO3 bindings
  - [ ] Python type stubs update
  - [ ] Python examples

```python
# Expected Python API
from pdf_oxide import PdfDocument

doc = PdfDocument("scanned.pdf")

# Method 1: Automatic fallback
text = doc.extract_text(0, fallback_to_ocr=True)

# Method 2: Explicit OCR
text = doc.ocr_page(0)

# Method 3: Detailed results
result = doc.ocr_page_detailed(0)
for span in result.spans:
    print(f"{span.text} @ {span.bbox} (conf: {span.confidence})")
```

#### Task 3.2: Performance Optimization
- **Dependencies**: Task 3.1
- **Effort**: 8 hours
- **Deliverables**:
  - [ ] Batch recognition optimization
  - [ ] Multi-threaded detection
  - [ ] Model quantization (INT8)
  - [ ] Memory usage optimization

```rust
// Optimization targets
impl OcrEngine {
    /// Process multiple crops in single inference call
    pub fn recognize_batch(&self, crops: &[DynamicImage]) -> Result<Vec<RecognitionResult>>;
}

// INT8 quantized model support
impl OcrConfig {
    pub fn with_quantized_models(mut self, quantized: bool) -> Self;
}
```

#### Task 3.3: Documentation & Examples
- **Dependencies**: Tasks 3.1, 3.2
- **Effort**: 4 hours
- **Deliverables**:
  - [ ] API documentation (rustdoc)
  - [ ] Python docstrings
  - [ ] `examples/ocr_scanned_pdf.rs`
  - [ ] `examples/ocr_python_example.py`
  - [ ] Update README.md with OCR section
  - [ ] Update CHANGELOG.md

#### Task 3.4: Final Testing & Benchmarks
- **Dependencies**: Task 3.3
- **Effort**: 4 hours
- **Deliverables**:
  - [ ] Full test suite passing
  - [ ] Accuracy benchmarks documented
  - [ ] Performance benchmarks documented
  - [ ] CI/CD integration for OCR tests

---

## 8. Risk Mitigation

### 8.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Model size too large | Medium | Medium | Quantization, model download option |
| Accuracy below target | Low | High | Use PP-OCRv4 (proven accuracy), fallback to v5 |
| ONNX Runtime compatibility | Low | High | Pin ort version, test on multiple platforms |
| Memory usage spikes | Medium | Medium | Streaming inference, batch size limits |
| Image extraction fails | Low | Medium | Validate image exists before OCR, clear error messages |

### 8.2 Mitigation Strategies

**Model Size**:
```rust
// Option 1: Lazy loading
static OCR_ENGINE: OnceLock<OcrEngine> = OnceLock::new();

// Option 2: Model path configuration
let engine = OcrEngine::with_model_dir("/custom/path")?;
```

**Cross-Platform Testing**:
```yaml
# CI matrix
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
# All platforms use the same pure-Rust OCR feature
```

**Accuracy Fallback**:
```rust
impl OcrEngine {
    /// Try multiple recognition attempts with different preprocessing
    pub fn recognize_with_retry(&self, crop: &DynamicImage) -> Result<RecognitionResult> {
        // Attempt 1: Standard
        // Attempt 2: Enhanced contrast
        // Attempt 3: Binarization
        // Return best confidence result
    }
}
```

---

## 9. Success Criteria

### 9.1 Phase 1 Complete When:
- [ ] `cargo test --features ocr` passes
- [ ] CER < 10% on synthetic test images
- [ ] Detection latency < 100ms on A4 image
- [ ] `examples/ocr_basic.rs` runs successfully

### 9.2 Phase 2 Complete When:
- [ ] Scanned PDF detection works reliably
- [ ] `extract_text(fallback_to_ocr=True)` works
- [ ] Markdown export includes OCR content
- [ ] Integration tests pass

### 9.3 Phase 3 Complete When:
- [ ] Python bindings work
- [ ] CER < 5% on benchmark dataset
- [ ] Full pipeline < 1s per page
- [ ] Documentation complete
- [ ] All CI tests pass

---

## 10. Appendix

### A. Model Preprocessing Details

**Detection Preprocessing (DBNet++)**:
```python
# Reference from PaddleOCR
def preprocess_det(img, max_side=960):
    h, w = img.shape[:2]
    ratio = max_side / max(h, w)
    new_h, new_w = int(h * ratio), int(w * ratio)

    # Resize
    resized = cv2.resize(img, (new_w, new_h))

    # Pad to multiple of 32
    pad_h = (32 - new_h % 32) % 32
    pad_w = (32 - new_w % 32) % 32
    padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT)

    # Normalize: (img - mean) / std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalized = (padded / 255.0 - mean) / std

    # HWC -> CHW, add batch
    tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]
    return tensor.astype(np.float32)
```

**Recognition Preprocessing (SVTR)**:
```python
def preprocess_rec(crop, target_height=48):
    h, w = crop.shape[:2]
    ratio = target_height / h
    new_w = int(w * ratio)

    # Resize maintaining aspect ratio
    resized = cv2.resize(crop, (new_w, target_height))

    # Pad width to multiple of 4
    pad_w = (4 - new_w % 4) % 4
    padded = cv2.copyMakeBorder(resized, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT)

    # Normalize
    normalized = (padded / 255.0 - 0.5) / 0.5

    # HWC -> CHW, add batch
    tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]
    return tensor.astype(np.float32)
```

### B. CTC Decoding Algorithm

```rust
/// Greedy CTC decoding
pub fn ctc_greedy_decode(logits: &Array2<f32>, dictionary: &[char]) -> (String, f32) {
    let blank_idx = dictionary.len();
    let mut result = String::new();
    let mut prev_idx = blank_idx;
    let mut confidence_sum = 0.0;
    let mut char_count = 0;

    for t in 0..logits.shape()[0] {
        let probs = softmax(logits.slice(s![t, ..]));
        let (max_idx, max_prob) = probs.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        if max_idx != blank_idx && max_idx != prev_idx {
            result.push(dictionary[max_idx]);
            confidence_sum += max_prob;
            char_count += 1;
        }
        prev_idx = max_idx;
    }

    let avg_confidence = if char_count > 0 {
        confidence_sum / char_count as f32
    } else {
        0.0
    };

    (result, avg_confidence)
}
```

### C. Levenshtein Distance for CER

```rust
/// Calculate Character Error Rate using Levenshtein distance
pub fn calculate_cer(ground_truth: &str, prediction: &str) -> f32 {
    let gt_chars: Vec<char> = ground_truth.chars().collect();
    let pred_chars: Vec<char> = prediction.chars().collect();

    if gt_chars.is_empty() {
        return if pred_chars.is_empty() { 0.0 } else { 1.0 };
    }

    let m = gt_chars.len();
    let n = pred_chars.len();

    // DP table
    let mut dp = vec![vec![0usize; n + 1]; m + 1];

    for i in 0..=m { dp[i][0] = i; }
    for j in 0..=n { dp[0][j] = j; }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if gt_chars[i-1] == pred_chars[j-1] { 0 } else { 1 };
            dp[i][j] = (dp[i-1][j] + 1)           // deletion
                .min(dp[i][j-1] + 1)              // insertion
                .min(dp[i-1][j-1] + cost);        // substitution
        }
    }

    dp[m][n] as f32 / m as f32
}
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-XX-XX | AI Assistant | Initial comprehensive plan |

---

*This plan is based on research conducted using web searches, existing Rust crate analysis, and pdf_oxide codebase examination. All technical decisions are subject to validation during implementation.*
