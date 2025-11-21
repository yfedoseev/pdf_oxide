# OCR Feature Requirements for pdf_oxide

## Overview

Add PaddleOCR-based text extraction for scanned PDFs, integrated seamlessly with existing text extraction pipeline. Uses ONNX Runtime for CPU-only inference - no GPU required.

## Goals

1. **Auto-detect scanned pages** - Automatically identify pages that need OCR (image-only pages)
2. **Unified output** - OCR results should match the quality/format of native text extraction
3. **Preserve positions & styles** - Extract bounding boxes, detect fonts/styles for Markdown export
4. **Feature flag** - Available via `--features ocr` in Rust and optional dependency in Python
5. **Fast CPU inference** - Target < 1 second per A4 page on modern CPU

---

## PaddleOCR Model Selection

### Minimal Model Set (2 models)

| Model | Purpose | Size | CPU Speed |
|-------|---------|------|-----------|
| **DBNet++ (PP-OCRv4 det)** | Text detection | ~4 MB | 20-50ms/page |
| **SVTR-LCNet (PP-OCRv4 rec)** | Text recognition | ~10 MB | 1-3ms/word |

**Total: ~14 MB** (can be compressed further with quantization)

### Why These Models?

- **DBNet++**: Best detection accuracy/speed ratio, handles rotated text
- **SVTR-LCNet**: Lightweight transformer-CNN hybrid, excellent accuracy
- **Skip angle classifier**: DBNet++ handles rotation, not needed for most documents

### Model Sources

Pre-converted ONNX models available:
- https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/models_list_en.md
- Or convert using `paddle2onnx`

---

## Technical Architecture

### Rust Integration Pipeline

```
PDF Page → Render to Image → Detection → Crop Text Regions → Recognition → Structured Output
                ↓                ↓              ↓                 ↓              ↓
           image crate      DBNet++        Box coords         SVTR-LCNet    TextSpan[]
```

### Dependencies

```toml
[dependencies]
# OCR feature
ort = { version = "2.0", optional = true }           # ONNX Runtime
image = { version = "0.25", optional = true }        # Image processing
imageproc = { version = "0.24", optional = true }    # Image operations

[features]
ocr = ["ort", "image", "imageproc"]
```

### Core Components

```rust
// src/ocr/mod.rs
pub mod detector;      // DBNet++ text detection
pub mod recognizer;    // SVTR text recognition
pub mod pipeline;      // Full OCR pipeline
pub mod page_analyzer; // Scanned page detection

// Main structs
pub struct OcrEngine {
    detector: TextDetector,
    recognizer: TextRecognizer,
}

pub struct OcrResult {
    pub spans: Vec<OcrSpan>,
    pub confidence: f32,
}

pub struct OcrSpan {
    pub text: String,
    pub bbox: BoundingBox,        // Position on page
    pub confidence: f32,
    pub style: TextStyle,         // Detected style (bold, size estimate)
}
```

---

## Scanned Page Detection

### Auto-Detection Algorithm

A page is considered "scanned" (needs OCR) if:

1. **No native text** - `extract_text()` returns empty or near-empty
2. **Contains large images** - Image covers >80% of page area
3. **Image-only content stream** - Only `Do` operators (XObject references), no text operators

```rust
pub fn needs_ocr(page: &Page) -> bool {
    let native_text = page.extract_text().unwrap_or_default();
    let has_text = native_text.trim().len() > 50; // threshold

    if has_text {
        return false; // Native text exists, no OCR needed
    }

    // Check if page has images
    let images = page.get_images();
    let page_area = page.width() * page.height();
    let image_coverage: f32 = images.iter()
        .map(|img| img.width * img.height)
        .sum::<f32>() / page_area;

    image_coverage > 0.5 // >50% image coverage = likely scanned
}
```

---

## API Design

### Rust API

```rust
use pdf_oxide::{PdfDocument, OcrOptions};

// Option 1: Explicit OCR
let doc = PdfDocument::open("scanned.pdf")?;
let text = doc.ocr_page(0)?;

// Option 2: Auto-detect (recommended)
let text = doc.extract_text_with_options(0, ExtractOptions {
    fallback_to_ocr: true,
    ..Default::default()
})?;

// Option 3: Full control
let ocr_result = doc.ocr_page_detailed(0, OcrOptions {
    languages: vec!["en", "de"],
    detect_styles: true,
    dpi: 300,
})?;

for span in ocr_result.spans {
    println!("{} at {:?} (conf: {})", span.text, span.bbox, span.confidence);
}

// Markdown export with OCR
let markdown = doc.to_markdown_with_options(0, MarkdownOptions {
    fallback_to_ocr: true,
    ..Default::default()
})?;
```

### Python API

```python
from pdf_oxide import PdfDocument

doc = PdfDocument("scanned.pdf")

# Auto-detect and OCR if needed
text = doc.extract_text(0, fallback_to_ocr=True)

# Explicit OCR
text = doc.ocr_page(0)

# Detailed results
result = doc.ocr_page_detailed(0, languages=["en", "de"])
for span in result.spans:
    print(f"{span.text} at {span.bbox} (conf: {span.confidence})")

# Markdown with OCR fallback
markdown = doc.to_markdown(0, fallback_to_ocr=True)
```

---

## Style Detection from OCR

### Detecting Bold/Headers from OCR

Since OCR doesn't give us font metadata, we infer styles from:

1. **Text size** - Measure character height from bounding boxes
2. **Position** - Top of page + larger size = likely heading
3. **Spacing** - Extra space before = paragraph break
4. **All caps** - Often indicates headers

```rust
pub fn detect_style(span: &OcrSpan, page_context: &PageContext) -> TextStyle {
    let char_height = span.bbox.height() / span.text.len() as f32;
    let avg_height = page_context.average_char_height;

    let is_large = char_height > avg_height * 1.3;
    let is_all_caps = span.text == span.text.to_uppercase()
                      && span.text.chars().any(|c| c.is_alphabetic());
    let near_top = span.bbox.y < page_context.height * 0.15;

    TextStyle {
        is_bold: is_large || is_all_caps,
        is_heading: is_large && near_top,
        font_size_estimate: char_height,
    }
}
```

---

## Integration with Existing Pipeline

### Unified TextSpan Output

OCR results must produce the same `TextSpan` structure as native extraction:

```rust
// From native extraction
pub struct TextSpan {
    pub text: String,
    pub bbox: BoundingBox,
    pub font_name: Option<String>,
    pub font_size: f32,
    pub is_bold: bool,
    pub is_italic: bool,
}

// OCR produces compatible spans
impl From<OcrSpan> for TextSpan {
    fn from(ocr: OcrSpan) -> Self {
        TextSpan {
            text: ocr.text,
            bbox: ocr.bbox,
            font_name: None, // Unknown from OCR
            font_size: ocr.style.font_size_estimate,
            is_bold: ocr.style.is_bold,
            is_italic: false, // Hard to detect from OCR
        }
    }
}
```

### Markdown Generation

Once OCR spans are converted to `TextSpan`, existing Markdown pipeline works unchanged:

```
OCR → OcrSpan[] → TextSpan[] → Layout Analysis → Markdown
                      ↑
              Native extraction also produces TextSpan[]
```

---

## Model Management

### Options for Model Distribution

1. **Bundled in crate** (recommended for small models)
   - Include ONNX files in `models/` directory
   - Use `include_bytes!` macro
   - Total ~14 MB added to crate

2. **Download on first use**
   - Store in `~/.cache/pdf_oxide/models/`
   - Auto-download from GitHub releases
   - Better for large models

3. **User-provided paths**
   - `OcrOptions { model_path: "/path/to/models" }`
   - Maximum flexibility

### Recommended Approach

```rust
// Default: bundled models
let engine = OcrEngine::new()?;

// Custom models
let engine = OcrEngine::with_models(
    "/path/to/detection.onnx",
    "/path/to/recognition.onnx",
)?;
```

---

## Language Support

### Phase 1: English + Latin Scripts
- English, German, French, Spanish, Italian, Portuguese
- Single recognition model handles all Latin scripts

### Phase 2: Multi-Script
- Chinese (Simplified/Traditional)
- Japanese
- Korean
- Arabic
- Requires additional recognition models (~10 MB each)

### API

```rust
let result = doc.ocr_page_detailed(0, OcrOptions {
    languages: vec!["en"],  // Phase 1: ignored, uses Latin model
    // languages: vec!["ch", "en"],  // Phase 2: loads Chinese + Latin
    ..Default::default()
})?;
```

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Detection | < 50ms/page | DBNet++ lite |
| Recognition | < 500ms/page | ~100-200 words typical |
| Total per page | < 1 second | A4 @ 300 DPI |
| Memory | < 200 MB | Models + inference buffers |
| Model size | < 20 MB | Detection + Recognition |

### Optimization Strategies

1. **Quantized models** - INT8 inference (2-4x speedup)
2. **Multi-threading** - Parallel word recognition
3. **Batch processing** - Multiple pages concurrently
4. **Resolution scaling** - 150 DPI often sufficient

---

## Implementation Plan

### Phase 1: Core OCR (MVP)
- [ ] Add `ort` dependency with feature flag
- [ ] Implement `TextDetector` (DBNet++)
- [ ] Implement `TextRecognizer` (SVTR)
- [ ] Basic `OcrEngine` pipeline
- [ ] Scanned page detection
- [ ] `doc.ocr_page()` API
- [ ] Tests with sample scanned PDFs

### Phase 2: Integration
- [ ] Convert `OcrSpan` to `TextSpan`
- [ ] Style detection from OCR
- [ ] `fallback_to_ocr` option in `extract_text()`
- [ ] Markdown export with OCR
- [ ] Python bindings

### Phase 3: Polish
- [ ] Model bundling / download system
- [ ] Multi-language support
- [ ] Performance optimization (INT8, batching)
- [ ] Documentation and examples

---

## Testing Strategy

### Test Cases

1. **Pure scanned PDF** - All pages are images
2. **Mixed PDF** - Some pages native, some scanned
3. **Hybrid page** - Native text + embedded scanned image
4. **Rotated text** - 90°, 180° rotations
5. **Multi-column** - Newspaper/academic layouts
6. **Low quality scans** - Faded, skewed, noisy

### Test Data

- Create `tests/fixtures/scanned/` directory
- Include diverse scanned samples
- Ground truth text files for accuracy comparison

---

## Open Questions

1. **Model licensing** - PaddleOCR is Apache 2.0, compatible with our MIT/Apache-2.0
2. **ONNX Runtime version** - Use `ort` v2.0 (newest) or v1.x (more stable)?
3. **Image rendering** - Use `pdfium` or pure-Rust solution for PDF→image?
4. **Crate size limit** - Is 14 MB models acceptable for bundling?

---

## References

- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleOCR Model List](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/models_list_en.md)
- [ONNX Runtime Rust](https://github.com/pykeio/ort)
- [PP-OCRv4 Technical Report](https://arxiv.org/abs/2206.03001)
