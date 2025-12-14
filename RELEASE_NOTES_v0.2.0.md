# PDF Oxide v0.2.0 Release Notes

**Release Date:** December 13, 2025
**Version:** 0.2.0 (Part of Forever 0.x Philosophy)
**Status:** ✅ Production Ready | 906 Tests Passing | 47.9× Faster than PyMuPDF4LLM

---

## 🎯 What's New in v0.2.0

### 1. **Intelligent PDF Understanding** 🧠

pdf_oxide now automatically understands whether a PDF is **native text** (generated digitally) or **scanned** (from a photocopier or camera). It intelligently adapts its processing per text block - no global configuration needed.

**What This Means:**
- 📄 Native PDFs extracted perfectly with original formatting preserved
- 📸 Scanned PDFs automatically cleaned up (removes OCR artifacts, expands ligatures, fixes punctuation)
- 🔄 Mixed documents (some pages native, some scanned) handled seamlessly
- ⚙️ Zero configuration - works automatically on any PDF

**Example:**
```python
from pdf_oxide import PdfDocument

doc = PdfDocument("mixed_document.pdf")  # Has both native and scanned pages
markdown = doc.to_markdown(0, detect_headings=True)
# Automatically applies intelligent processing per block!
# Scanned text is cleaned up, native text stays perfect
```

### 2. **Professional Reading Order** 📖

PDFs now extract in the correct **reading order** - no more scrambled columns or out-of-sequence text. Supports multiple layout strategies for different document types.

**What This Means:**
- ✅ Multi-column documents read left-to-right, top-to-bottom properly
- ✅ Handles complex layouts (2-3+ columns, sidebars, footers)
- ✅ PDF spec compliant (ISO 32000-1:2008 §14.7-14.8)
- ✅ Accessible PDFs (tagged) use structure tree for perfect ordering
- ✅ Untagged PDFs use smart geometric analysis

**Strategies Supported:**
1. **XY-Cut** - Best for multi-column technical papers and magazines
2. **Structure Tree** - Best for accessible/tagged PDFs
3. **Geometric** - Smart positioning analysis (default)
4. **Simple** - Linear top-to-bottom (backward compatible)

### 3. **Modern Architecture with Better APIs** 🏗️

The library's internal architecture has been completely modernized. This gives you:

**Better Code Organization:**
```rust
// New recommended way (modern)
use pdf_oxide::pipeline::{TextPipeline, TextPipelineConfig};
use pdf_oxide::pipeline::converters::MarkdownOutputConverter;

let pipeline = TextPipeline::with_config(config);
let ordered_spans = pipeline.process(spans, context)?;
let converter = MarkdownOutputConverter::new();
let markdown = converter.convert(&ordered_spans, &config)?;
```

**Benefits:**
- Clean separation between text extraction, reading order, and formatting
- Easy to extend (add custom converters, reading orders, processors)
- Better testability
- Faster to add new features

**Old APIs Still Work:**
```rust
// Old way (deprecated but functional until v0.5.0)
let converter = MarkdownConverter::new();
let md = converter.convert(&spans, &options)?;
```

### 4. **Better Scanned PDF Support** 🖼️

Scanned PDFs (from scanners, cameras, fax machines) are now handled much better:

**Improvements:**
- ✅ CCITT Group 3/4 image decompression (scanned document standard)
- ✅ Automatic 1-bit to 8-bit image conversion
- ✅ Better image extraction for OCR preprocessing
- ✅ TIFF format support (common for scanned documents)

**Example:**
```rust
let images = doc.extract_images(0)?;
// Now properly decompresses CCITT-compressed images from scanners
for image in images {
    image.save(format!("page_{}.png", image.index))?;
}
```

### 5. **Experimental OCR Support** 🤖 (Optional)

For scanned PDFs without embedded text, optical character recognition can extract text:

**Enable with feature flag:**
```bash
cargo build --features ocr
```

**Use it:**
```rust
#[cfg(feature = "ocr")]
{
    let ocr_text = doc.extract_text_with_ocr(0)?;
    // Recognizes text from scanned images
}
```

**Details:**
- Uses PaddleOCR v3 (industry-standard, multilingual)
- ONNX Runtime for fast CPU inference (< 1 second per page)
- ~200MB model download (one-time)
- Optional - no forced dependencies

---

## 📊 Quality Metrics

| Metric | Value | vs v0.1.x |
|--------|-------|-----------|
| **Tests Passing** | 906 ✅ | +63 tests |
| **Speed** | 47.9× faster | Same |
| **Warnings** | Minimal | 72% reduction |
| **Code Quality** | Clean | No dead code |
| **PDF Spec** | §9, 14.7-14.8 ✅ | Extended |

---

## 🔄 Migration Guide for v0.1.x Users

### ✅ No Breaking Changes - Your Code Still Works

But we **strongly recommend** updating to use the new APIs:

### Option 1: Continue Using Old API (Not Recommended)
```rust
// This still works in v0.2.0, v0.3.0, v0.4.0
// Will be removed in v0.5.0 (likely late 2025)
use pdf_oxide::converters::MarkdownConverter;

let converter = MarkdownConverter::new();
let md = converter.convert(&spans, &options)?;
```

### Option 2: Upgrade to New API (Recommended)

**Step 1: Update imports**
```rust
// Remove old imports
// use pdf_oxide::converters::MarkdownConverter;

// Add new imports
use pdf_oxide::pipeline::{TextPipeline, TextPipelineConfig};
use pdf_oxide::pipeline::converters::MarkdownOutputConverter;
```

**Step 2: Update code**
```rust
// Create configuration
let config = TextPipelineConfig::from_conversion_options(&ConversionOptions {
    detect_headings: true,
    include_images: true,
    preserve_layout: false,
    image_output_dir: Some("./images".to_string()),
});

// Create pipeline
let pipeline = TextPipeline::with_config(config.clone());

// Process spans through pipeline (applies reading order)
let ordered_spans = pipeline.process(spans, Default::default())?;

// Convert to desired format
let converter = MarkdownOutputConverter::new();
let markdown = converter.convert(&ordered_spans, &config)?;
```

### Why Upgrade?
✅ Get automatic reading order (multi-column support)
✅ Benefit from intelligent text processing
✅ Prepare for v0.3.0+ features (write PDFs, etc.)
✅ Future-proof your code

### Deprecation Timeline
- **v0.2.0-v0.4.0**: Old APIs work with deprecation warnings
- **v0.5.0+**: Old APIs removed
- **Timeline**: ~6+ months to migrate

---

## 🎓 Examples

### Basic Usage (Unchanged)
```python
from pdf_oxide import PdfDocument

doc = PdfDocument("paper.pdf")
print(f"Pages: {doc.page_count()}")

text = doc.extract_text(0)
print(text)
```

### Using Intelligent Processing (New!)
```rust
use pdf_oxide::PdfDocument;

let mut doc = PdfDocument::open("mixed.pdf")?;

// Automatically detects OCR and applies intelligent cleanup
let processed = doc.apply_intelligent_text_processing(
    doc.extract_spans(0)?
)?;

// Now use for conversion
let markdown = doc.to_markdown(0, Default::default())?;
```

### Using Reading Order (New!)
```rust
use pdf_oxide::pipeline::{TextPipeline, TextPipelineConfig};

let config = TextPipelineConfig::default();
let pipeline = TextPipeline::with_config(config);

// Automatically applies best reading order strategy
let ordered_spans = pipeline.process(spans, Default::default())?;
```

### HTML Conversion (New!)
```rust
use pdf_oxide::pipeline::converters::HtmlOutputConverter;

let converter = HtmlOutputConverter::new();
let html = converter.convert(&ordered_spans, &config)?;
```

### Form Field Extraction (Unchanged)
```rust
let fields = doc.extract_form_fields(0)?;
for field in fields {
    println!("Field: {}", field.name);
    println!("Value: {:?}", field.value);
    println!("Type: {:?}", field.field_type);
}
```

---

## 🚀 What's Coming Next

### v0.3.0 (Early 2025) - PDF Creation
- 📝 Generate PDFs from Markdown
- 🌐 Generate PDFs from HTML
- 🎨 Template system for consistent layouts
- 📄 Fluent PdfBuilder API

### v0.4.0 (Mid 2025) - Structured Data
- 📊 Extract and generate tables
- 📋 Create interactive fillable forms
- 🗂️ Generate document outlines/bookmarks

### v0.5.0+ - Advanced Features
- 🖼️ Figures & captions
- 📚 Citation management
- 💬 Annotations
- ♿ Accessible PDF creation
- 🔐 Encryption & signatures

---

## 💡 Upgrading Checklist

- [ ] Update pdf_oxide dependency: `pdf_oxide = "0.2"`
- [ ] Test your code (old APIs still work)
- [ ] Run new tests: `cargo test`
- [ ] If using converters, update to new pipeline API
- [ ] Read README examples for new patterns
- [ ] Enable `ocr` feature if needed: `cargo build --features ocr`

---

## ⚡ Performance

**Same speed as v0.1.x:**
- 47.9× faster than PyMuPDF4LLM
- Process 100 PDFs in 5.3 seconds
- Average 53ms per PDF

**New architecture enables:**
- Parallel reading order strategies (coming in v0.3.0)
- Custom pipeline optimizations
- Per-document metrics and profiling

---

## 🐛 Known Issues & Limitations

### Experimental (Feature-Gated)
- **OCR**: Requires `ocr` feature, may not handle all edge cases
- **Complex Layouts**: Some 3+ column documents may need tuning

### Not Yet Supported
- Form field editing (read-only extraction works)
- Vector graphics (planned v0.6.0+)
- Mathematical formulas (planned v0.7.0+)
- GPU-accelerated OCR (CPU only)

### Minor Limitations
- RTL (Arabic, Hebrew) text: basic support only
- CJK (Chinese/Japanese/Korean): requires feature flag

---

## 🙏 Thank You

This release represents months of work on PDF specification compliance, intelligent text processing, and modern architecture redesign. Special thanks to:

- **PDF Community** - For detailed spec feedback
- **Test Users** - For real-world PDF examples and bug reports
- **Contributors** - For reviews and improvements

---

## 📖 Documentation

- **[Full README](README.md)** - Complete features, examples, and API reference
- **[CHANGELOG](CHANGELOG.md)** - Detailed technical changes
- **[API Docs](https://docs.rs/pdf_oxide)** - Comprehensive API documentation

---

## 🔗 Links

- **[GitHub Repository](https://github.com/yfedoseev/pdf_oxide)**
- **[crates.io Page](https://crates.io/crates/pdf_oxide)**
- **[Issue Tracker](https://github.com/yfedoseev/pdf_oxide/issues)**
- **[Discussions](https://github.com/yfedoseev/pdf_oxide/discussions)**

---

## 📝 License

pdf_oxide is dual-licensed under **MIT OR Apache-2.0**, giving you flexibility to use it in any project.

---

## 🎉 Enjoy v0.2.0!

This release brings production-grade PDF handling to Rust. Whether you're building LLM pipelines, document processing systems, or research tools, pdf_oxide has you covered.

**Get Started:**
```bash
cargo add pdf_oxide@0.2
```

```python
pip install pdf_oxide  # Python support
```

Happy PDF processing! 🚀
