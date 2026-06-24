# Getting Started with PDFOxide (Rust)

PDFOxide is the complete PDF toolkit for Rust. One library for extracting, creating, and editing PDFs with a unified API.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
pdf_oxide = "0.3"
```

### Feature Flags

Select only the features you need:

```toml
[dependencies]
# Default - text extraction, creation, editing
pdf_oxide = "0.3"

# With barcode generation
pdf_oxide = { version = "0.3", features = ["barcodes"] }

# With Office document conversion (DOCX, XLSX, PPTX)
pdf_oxide = { version = "0.3", features = ["office"] }

# With digital signatures
pdf_oxide = { version = "0.3", features = ["signatures"] }

# With OCR for scanned PDFs (PaddleOCR via ONNX Runtime)
pdf_oxide = { version = "0.3", features = ["ocr"] }

# With page rendering to images
pdf_oxide = { version = "0.3", features = ["rendering"] }

# All features
pdf_oxide = { version = "0.3", features = ["full"] }
```

## Quick Start - The Unified `Pdf` API

The `Pdf` class is your main entry point for all PDF operations:

```rust
use pdf_oxide::api::Pdf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create from Markdown
    let mut pdf = Pdf::from_markdown("# Hello World\n\nThis is a PDF.")?;
    pdf.save("output.pdf")?;

    Ok(())
}
```

## Creating PDFs

### From Markdown

```rust
use pdf_oxide::api::Pdf;

let mut pdf = Pdf::from_markdown(r#"
# Report Title

## Introduction

This is **bold** and *italic* text.

- Item 1
- Item 2
- Item 3

## Code Example

```python
print("Hello, World!")
```
"#)?;
pdf.save("report.pdf")?;
```

### From HTML

```rust
use pdf_oxide::api::Pdf;

let mut pdf = Pdf::from_html(r#"
<h1>Invoice</h1>
<p>Thank you for your purchase.</p>
<table>
    <tr><th>Item</th><th>Price</th></tr>
    <tr><td>Widget</td><td>$10.00</td></tr>
</table>
"#)?;
pdf.save("invoice.pdf")?;
```

### From Plain Text

```rust
use pdf_oxide::api::Pdf;

let mut pdf = Pdf::from_text("Simple plain text document.\n\nWith paragraphs.")?;
pdf.save("notes.pdf")?;
```

### From Images

```rust
use pdf_oxide::api::Pdf;

// Single image
let mut pdf = Pdf::from_image("photo.jpg")?;
pdf.save("photo.pdf")?;

// Multiple images (one per page)
let mut album = Pdf::from_images(&["page1.jpg", "page2.png", "page3.jpg"])?;
album.save("album.pdf")?;
```

## Opening and Reading PDFs

```rust
use pdf_oxide::api::Pdf;

// Open existing PDF
let mut pdf = Pdf::open("document.pdf")?;

// Extract text from page 0
let text = pdf.extract_text(0)?;
println!("Text: {}", text);

// Convert to Markdown
let markdown = pdf.to_markdown(0)?;
println!("Markdown:\n{}", markdown);

// Get page count
println!("Pages: {}", pdf.page_count());
```

## Editing PDFs

### DOM-like Navigation

```rust
use pdf_oxide::api::{Pdf, PdfElement};

let mut pdf = Pdf::open("document.pdf")?;

// Get a page for DOM-like access
let page = pdf.page(0)?;

// Find text elements
for text in page.find_text_containing("Hello") {
    println!("Found '{}' at {:?}", text.text(), text.bbox());
}

// Iterate through all elements
for element in page.children() {
    match element {
        PdfElement::Text(t) => println!("Text: {}", t.text()),
        PdfElement::Image(i) => println!("Image: {}x{}", i.width(), i.height()),
        PdfElement::Path(p) => println!("Path at {:?}", p.bbox()),
        _ => {}
    }
}
```

### Modifying Content

```rust
use pdf_oxide::api::Pdf;

let mut pdf = Pdf::open("document.pdf")?;

// Get mutable page
let mut page = pdf.page(0)?;

// Find and replace text
let texts = page.find_text_containing("old");
for t in &texts {
    page.set_text(t.id(), "new")?;
}

// Save changes back
pdf.save_page(page)?;
pdf.save("modified.pdf")?;
```

### Adding Annotations

```rust
use pdf_oxide::api::Pdf;

let mut pdf = Pdf::open("document.pdf")?;

// Add highlight
pdf.add_highlight(0, [100.0, 700.0, 300.0, 720.0], None)?;

// Add sticky note
pdf.add_sticky_note(0, 500.0, 750.0, "Review this section")?;

// Add link
pdf.add_link(0, [100.0, 600.0, 200.0, 620.0], "https://example.com")?;

pdf.save("annotated.pdf")?;
```

### Working with Form Fields

Extract, read, fill, and save PDF form fields (AcroForm):

```rust
use pdf_oxide::PdfDocument;
use pdf_oxide::extractors::forms::FormExtractor;

let mut doc = PdfDocument::open("tax-form.pdf")?;

// List all form fields
let fields = FormExtractor::extract_fields(&mut doc)?;
for f in &fields {
    println!("{} ({:?}) = {:?}", f.full_name, f.field_type, f.value);
}
```

#### Fill Form Fields and Save

```rust
use pdf_oxide::editor::{DocumentEditor, EditableDocument, SaveOptions};
use pdf_oxide::editor::form_fields::FormFieldValue;

let mut editor = DocumentEditor::open("w2.pdf")?;

// Set text values
editor.set_form_field_value("employee_name", FormFieldValue::Text("Jane Doe".into()))?;
editor.set_form_field_value("wages", FormFieldValue::Text("85000.00".into()))?;

// Set checkbox
editor.set_form_field_value("retirement_plan", FormFieldValue::Boolean(true))?;

// Save with incremental update (preserves original, appends changes)
editor.save_with_options("filled_w2.pdf", SaveOptions::incremental())?;
```

#### Extract Text with Filled Values

Filled values appear inline in `extract_text` and `to_markdown`:

```rust
use pdf_oxide::PdfDocument;
use pdf_oxide::converters::ConversionOptions;

let mut doc = PdfDocument::open("filled_w2.pdf")?;

// Form values appear where the fields are positioned
let text = doc.extract_text(0)?;
println!("{}", text); // "Jane Doe" appears inline

// Include form fields in Markdown (default)
let opts = ConversionOptions { include_form_fields: true, ..Default::default() };
let md = doc.to_markdown(0, &opts)?;

// Exclude form fields
let opts_off = ConversionOptions { include_form_fields: false, ..Default::default() };
let md_clean = doc.to_markdown(0, &opts_off)?;
```

#### Adding New Form Fields

```rust
use pdf_oxide::api::Pdf;

let mut pdf = Pdf::open("form-template.pdf")?;

// Add text field
pdf.add_text_field("name", [100.0, 700.0, 300.0, 720.0])?;

// Add checkbox
pdf.add_checkbox("agree", [100.0, 650.0, 120.0, 670.0], false)?;

pdf.save("form.pdf")?;
```

## Builder Pattern for Advanced Creation

For full control over PDF creation, use `PdfBuilder`:

```rust
use pdf_oxide::api::PdfBuilder;
use pdf_oxide::writer::PageSize;

let mut pdf = PdfBuilder::new()
    .title("Annual Report 2025")
    .author("Company Inc.")
    .subject("Financial Summary")
    .page_size(PageSize::A4)
    .margins(72.0, 72.0, 72.0, 72.0)  // 1 inch margins
    .font_size(11.0)
    .from_markdown("# Annual Report\n\n...")?;

pdf.save("annual-report.pdf")?;
```

## Encryption and Security

### Password Protection

```rust
use pdf_oxide::api::Pdf;

let mut pdf = Pdf::from_markdown("# Confidential Document")?;

// Simple password protection (AES-256)
pdf.save_encrypted("secure.pdf", "user-password", Some("owner-password"))?;
```

### Advanced Encryption Options

```rust
use pdf_oxide::api::Pdf;
use pdf_oxide::editor::{EncryptionConfig, EncryptionAlgorithm, Permissions};

let mut pdf = Pdf::from_markdown("# Protected")?;

let config = EncryptionConfig::new("user", Some("owner"))
    .algorithm(EncryptionAlgorithm::Aes256)
    .permissions(Permissions::PRINT | Permissions::COPY);

pdf.save_with_encryption("protected.pdf", config)?;
```

## PDF Compliance

### PDF/A Validation and Conversion

```rust
use pdf_oxide::compliance::{PdfAValidator, PdfALevel, PdfAConverter};

// Validate
let validator = PdfAValidator::new();
let result = validator.validate_file("document.pdf", PdfALevel::PdfA2b)?;
if result.is_compliant {
    println!("PDF/A-2b compliant!");
} else {
    for error in result.errors {
        println!("Error: {:?}", error);
    }
}

// Convert to PDF/A
let converter = PdfAConverter::new(PdfALevel::PdfA2b);
converter.convert("input.pdf", "archive.pdf")?;
```

## OCR - Extracting Text from Scanned PDFs

> For a comprehensive guide covering model selection, configuration reference, resize strategies, and troubleshooting, see the [OCR Guide](OCR_GUIDE.md).

PDFOxide can extract text from scanned PDFs using PaddleOCR models via ONNX Runtime. Enable the `ocr` feature:

```toml
[dependencies]
pdf_oxide = { version = "0.3", features = ["ocr"] }
```

### Model Setup

PDFOxide supports PaddleOCR v3, v4, and v5 models. You can mix detection and recognition models from different versions.

**Quick start** — download the recommended models:

```bash
./scripts/setup_ocr_models.sh
```

#### Model Selection Guide

| Combination | Detection | Recognition | English Accuracy | Total Size |
|---|---|---|---|---|
| **V4 det + V5 rec (recommended)** | ch_PP-OCRv4_det | en_PP-OCRv5_mobile_rec | Best | ~12.5 MB |
| V4 det + V4 rec | ch_PP-OCRv4_det | en_PP-OCRv4_rec | Good | ~12.4 MB |
| V5 det + V5 rec | PP-OCRv5_server_det | en_PP-OCRv5_mobile_rec | Good (different errors) | ~96 MB |
| V3 det + V3 rec | en_PP-OCRv3_det | en_PP-OCRv3_rec | Fair | ~11 MB |

The **V4 detection + V5 recognition** combination gives the best results for English documents: V4 detection reliably segments text lines, while V5 recognition has the highest character-level accuracy.

**Manual download:**

```bash
# Recommended: V4 detection + V5 recognition
# Detection (4.7 MB):
curl -L https://huggingface.co/deepghs/paddleocr/resolve/main/det/ch_PP-OCRv4_det/model.onnx -o .models/det.onnx

# Recognition (7.8 MB):
curl -L https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/english/rec.onnx -o .models/rec.onnx

# Dictionary (must include space as last entry):
curl -L https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/english/dict.txt -o .models/en_dict.txt
echo " " >> .models/en_dict.txt
```

### Basic OCR Usage

```rust
use pdf_oxide::PdfDocument;
use pdf_oxide::ocr::{OcrEngine, OcrConfig, OcrExtractOptions, needs_ocr};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create OCR engine (reuse across pages)
    let engine = OcrEngine::new(
        ".models/det.onnx",
        ".models/rec.onnx",
        ".models/en_dict.txt",
        OcrConfig::default(),
    )?;

    let mut doc = PdfDocument::open("scanned.pdf")?;
    let options = OcrExtractOptions::with_dpi(300.0);

    for page in 0..doc.page_count()? {
        if needs_ocr(&mut doc, page)? {
            let text = pdf_oxide::ocr::ocr_page(&mut doc, page, &engine, &options)?;
            println!("Page {} (OCR): {}", page + 1, text);
        } else {
            let text = doc.extract_text(page)?;
            println!("Page {} (native): {}", page + 1, text);
        }
    }

    Ok(())
}
```

### Using PP-OCRv5 Detection

If you use the full PP-OCRv5 stack (v5 detection + v5 recognition), use `OcrConfig::v5()` which preserves the original image resolution instead of downscaling to 960px:

```rust
// For PP-OCRv5 server detection model (88 MB)
let config = OcrConfig::v5();
let engine = OcrEngine::new("v5_det.onnx", "v5_rec.onnx", "v5_dict.txt", config)?;
```

> **Note:** ONNX Runtime (`libonnxruntime` v1.23+) is loaded dynamically at runtime. Install it (system package, `brew install onnxruntime`, or a manual download) and either set `ORT_DYLIB_PATH` to the shared-library **file** (`libonnxruntime.so` / `.dylib` / `onnxruntime.dll`), or add its directory to `LD_LIBRARY_PATH` (`DYLD_LIBRARY_PATH` on macOS). `ORT_LIB_LOCATION` is a build-time variable and has no effect with the dynamic backend this crate uses.

## Lower-Level APIs

For specialized use cases, PDFOxide provides lower-level APIs:

| API | Use Case |
|-----|----------|
| `PdfDocument` | Direct PDF parsing and text extraction |
| `DocumentBuilder` | Low-level PDF generation with full control |
| `DocumentEditor` | Direct editing without the `Pdf` wrapper |

### Using PdfDocument Directly

```rust
use pdf_oxide::PdfDocument;

let mut doc = PdfDocument::open("paper.pdf")?;

// Low-level text extraction with spans
let spans = doc.extract_spans(0)?;
for span in spans {
    println!("{} at ({}, {})", span.text, span.x, span.y);
}

// Access raw PDF objects
let page = doc.get_page(0)?;
let media_box = page.get("MediaBox");
```

### Using DocumentBuilder Directly

```rust
use pdf_oxide::writer::DocumentBuilder;

let mut builder = DocumentBuilder::new();
builder.add_page(612.0, 792.0)  // Letter size in points
    .text("Custom positioned text", 72.0, 720.0, 12.0)
    .rect(100.0, 600.0, 200.0, 50.0)
    .image_at("logo.png", 400.0, 700.0, 100.0, 50.0)?;

builder.save("custom.pdf")?;
```

## Examples

See the [examples/](../examples/) directory for complete working examples:

- `create_pdf_from_markdown.rs` - Creating PDFs from Markdown
- `edit_existing_pdf.rs` - Opening and modifying PDFs
- `edit_text_content.rs` - In-place text editing
- `add_form_fields.rs` - Interactive form creation
- `encrypt_pdf.rs` - Password protection

Run an example:

```bash
cargo run --example create_pdf_from_markdown
```

## Next Steps

- [PDF Creation Guide](PDF_CREATION_GUIDE.md) - Advanced creation options
- [Architecture](ARCHITECTURE.md) - Understanding the library structure
- [API Documentation](https://docs.rs/pdf_oxide) - Full API reference
