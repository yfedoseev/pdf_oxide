# Getting Started with PDFOxide (Python)

PDFOxide is the complete PDF toolkit. One library for extracting, creating, and editing PDFs with a unified API. Built on a Rust core for maximum performance.

## Installation

```bash
pip install pdf_oxide
```

## Quick Start - The Unified `Pdf` API

The `Pdf` class is your main entry point for all PDF operations:

```python
from pdf_oxide import Pdf

# Create from Markdown
pdf = Pdf.from_markdown("# Hello World\n\nThis is a PDF.")
pdf.save("output.pdf")
```

## Creating PDFs

### From Markdown

```python
from pdf_oxide import Pdf

pdf = Pdf.from_markdown("""
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
""")
pdf.save("report.pdf")
```

### From HTML

```python
from pdf_oxide import Pdf

pdf = Pdf.from_html("""
<h1>Invoice</h1>
<p>Thank you for your purchase.</p>
<table>
    <tr><th>Item</th><th>Price</th></tr>
    <tr><td>Widget</td><td>$10.00</td></tr>
</table>
""")
pdf.save("invoice.pdf")
```

### From Plain Text

```python
from pdf_oxide import Pdf

pdf = Pdf.from_text("Simple plain text document.\n\nWith paragraphs.")
pdf.save("notes.pdf")
```

### From Images

```python
from pdf_oxide import Pdf

# Single image
pdf = Pdf.from_image("photo.jpg")
pdf.save("photo.pdf")

# Multiple images (one per page)
album = Pdf.from_images(["page1.jpg", "page2.png", "page3.jpg"])
album.save("album.pdf")
```

## Opening and Reading PDFs

```python
from pdf_oxide import PdfDocument

# Open existing PDF (path can be str or pathlib.Path)
doc = PdfDocument("document.pdf")

# Or use as a context manager
with PdfDocument("document.pdf") as doc:
    text = doc.extract_text(0)
    print(f"Text: {text}")
    markdown = doc.to_markdown(0)
    print(f"Pages: {doc.page_count()}")

# Extract text from page 0
text = doc.extract_text(0)
print(f"Text: {text}")

# Convert to Markdown
markdown = doc.to_markdown(0)
print(f"Markdown:\n{markdown}")

# Get page count
print(f"Pages: {doc.page_count()}")
```

## Builder Pattern for Advanced Creation

For full control over PDF creation, use `PdfBuilder`:

```python
from pdf_oxide import PdfBuilder, PageSize

pdf = (PdfBuilder()
    .title("Annual Report 2025")
    .author("Company Inc.")
    .subject("Financial Summary")
    .page_size(PageSize.A4)
    .margins(72.0, 72.0, 72.0, 72.0)  # 1 inch margins
    .font_size(11.0)
    .from_markdown("# Annual Report\n\n..."))

pdf.save("annual-report.pdf")
```

## Encryption and Security

### Password Protection

```python
from pdf_oxide import Pdf

pdf = Pdf.from_markdown("# Confidential Document")

# Simple password protection (AES-256)
pdf.save_encrypted("secure.pdf", "user-password", "owner-password")
```

## Text Extraction Options

### Basic Extraction

```python
from pdf_oxide import PdfDocument

doc = PdfDocument("paper.pdf")
text = doc.extract_text(0)
```

### With Options

```python
from pdf_oxide import PdfDocument, ConversionOptions

doc = PdfDocument("paper.pdf")
options = ConversionOptions(
    detect_headings=True,
    detect_lists=True,
    embed_images=True
)
markdown = doc.to_markdown(0, options)
```

### Extract All Pages

```python
from pdf_oxide import PdfDocument

doc = PdfDocument("book.pdf")

# Extract text from all pages
all_text = doc.extract_text_all()

# Convert entire document to Markdown
all_markdown = doc.to_markdown_all()
```

## Office Document Conversion

Convert DOCX, XLSX, and PPTX files to PDF:

```python
from pdf_oxide import OfficeConverter

# Auto-detect format
converter = OfficeConverter()

# Convert Word document
converter.convert("report.docx", "report.pdf")

# Convert Excel spreadsheet
converter.convert("data.xlsx", "data.pdf")

# Convert PowerPoint presentation
converter.convert("slides.pptx", "slides.pdf")
```

## Working with Images

### Extract Images from PDF

```python
from pdf_oxide import PdfDocument

doc = PdfDocument("document.pdf")
images = doc.extract_images(0)

for i, img in enumerate(images):
    img.save(f"image_{i}.png")
```

### Embed Images in Output

```python
from pdf_oxide import PdfDocument, ConversionOptions

doc = PdfDocument("paper.pdf")
options = ConversionOptions(embed_images=True)

# Images embedded as base64 data URIs
html = doc.to_html(0, options)
```

## OCR - Extracting Text from Scanned PDFs

> For a comprehensive guide covering model selection, configuration reference, resize strategies, and troubleshooting, see the [OCR Guide](OCR_GUIDE.md).

### Setup

The OCR feature requires heavy machine learning dependencies (ONNX Runtime) and is optional.

```bash
# Recommended: Install with OCR support
pip install pdf_oxide[ocr]

# Or build from source with OCR
maturin develop --features python,ocr
```

> **Troubleshooting:** If you see `RuntimeError: OCR feature not enabled`, it means the library was installed without OCR support. Re-install using the `[ocr]` extra above.

**Quick start** — download the recommended models:

```bash
./scripts/setup_ocr_models.sh
```

#### Model Selection Guide

PDFOxide supports PaddleOCR v3, v4, and v5 models. You can mix detection and recognition models from different versions.

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

```python
from pdf_oxide import PdfDocument, OcrEngine, OcrConfig

# Create OCR engine (default config works with recommended V4 det + V5 rec models)
engine = OcrEngine(
    det_model_path=".models/det.onnx",
    rec_model_path=".models/rec.onnx",
    dict_path=".models/en_dict.txt",
)

# Extract text using OCR
doc = PdfDocument("scanned.pdf")
text = doc.extract_text_ocr(page=0, engine=engine)
print(text)
```

### Processing Multiple Pages

```python
doc = PdfDocument("scanned.pdf")
for page in range(doc.page_count()):
    text = doc.extract_text_ocr(page=page, engine=engine)
    if text.strip():
        print(f"--- Page {page + 1} ---")
        print(text)
```

### OCR on Pages with a Text Layer

`extract_text_ocr` returns the native text of a page that has a usable text layer, and runs OCR only on scanned pages. A searchable PDF from a scanner pipeline has a text layer, so `extract_text_ocr` returns that layer, even when it is poor. To OCR the page in any case, use `extract_text_ocr_only`:

```python
doc = PdfDocument("searchable.pdf")
text = doc.extract_text_ocr_only(page=0, engine=engine)
```

`extract_text_ocr_only` runs OCR on the largest embedded image of the page. It does not render the page, so it suits PDFs with one scanned image per page. A page with no embedded image gives an empty string.

### Using PP-OCRv5 Detection

If you use the full PP-OCRv5 stack (v5 detection + v5 recognition), pass `use_v5=True` to `OcrConfig`. This preserves the original image resolution instead of downscaling to 960px, which the larger v5 detection model needs:

```python
config = OcrConfig(use_v5=True)
engine = OcrEngine(
    det_model_path="v5_det.onnx",
    rec_model_path="v5_rec.onnx",
    dict_path="v5_dict.txt",
    config=config,
)
```

> **Note:** The `OcrEngine` is reusable — create it once and pass it to multiple `extract_text_ocr` calls. ONNX Runtime requires `libonnxruntime.so` (v1.23+) to be available at runtime (via `LD_LIBRARY_PATH` or system install).

## Structured Extraction

Beyond plain text, PDFOxide can extract structured content from pages:

```python
from pdf_oxide import PdfDocument

doc = PdfDocument("document.pdf")

# 1. Scoped extraction from a specific area (v0.3.14)
# Area: (x, y, width, height) in points
header = doc.within(0, (0, 700, 612, 92)).extract_text()

# 2. Text spans with font info, position, and style
spans = doc.extract_spans(0)
for span in spans:
    print(f"{span.text} — {span.font_name} {span.font_size}pt")

# 3. Word-level extraction (v0.3.14)
words = doc.extract_words(0)
for w in words:
    print(f"Word: {w.text} at {w.bbox}")
    # Access character metadata for the word
    # print(w.chars[0].font_name)

# Optional: override the adaptive word gap threshold (in PDF points).
# Smaller values split more aggressively; useful for dense forms.
words = doc.extract_words(0, word_gap_threshold=2.5)

# 4. Line-level extraction (v0.3.14)
lines = doc.extract_text_lines(0)
for line in lines:
    print(f"Line: {line.text}")

# Optional: override word and/or line gap thresholds (in PDF points).
lines = doc.extract_text_lines(0, word_gap_threshold=2.5, line_gap_threshold=4.0)

# 5. Inspect computed layout params before overriding
params = doc.page_layout_params(0)
print(f"Adaptive word gap: {params.word_gap_threshold:.1f}pt")
print(f"Adaptive line gap: {params.line_gap_threshold:.1f}pt")

# 6. Pre-tuned extraction profiles for different document types
from pdf_oxide import ExtractionProfile
profile = ExtractionProfile.form()
print(f"Profile: {profile.name}, word_margin_ratio={profile.word_margin_ratio}")

# Pass a profile to extraction methods to control how raw text is parsed
words = doc.extract_words(0, profile=ExtractionProfile.form())
lines = doc.extract_text_lines(0, profile=ExtractionProfile.academic())

# Combine profile with threshold overrides (profile controls span parsing,
# thresholds control word/line clustering)
words = doc.extract_words(0, word_gap_threshold=1.5, profile=ExtractionProfile.aggressive())

# 7. Image metadata
images = doc.extract_images(0)
for img in images:
    print(f"{img['width']}x{img['height']} {img['color_space']}")

# 8. Bookmarks / table of contents
outline = doc.get_outline()  # None if no outline
if outline:
    for item in outline:
        print(f"{item['title']} -> page {item.get('page')}")

# 9. Vector paths (lines, curves, shapes)
paths = doc.extract_paths(0)
for path in paths:
    print(f"bbox={path['bbox']}, stroke={path.get('stroke_color')}")
```

## Working with Form Fields

PDFOxide can extract, read, fill, and export PDF form field data (AcroForm fields).

### List All Form Fields

```python
from pdf_oxide import PdfDocument

doc = PdfDocument("tax-form.pdf")
fields = doc.get_form_fields()

for f in fields:
    print(f"{f.name} ({f.field_type}) = {f.value}")
```

Each `FormField` has:
- `name` — fully qualified field name (e.g. `"topmostSubform[0].Page1[0].f1_01[0]"`)
- `field_type` — `"text"`, `"button"`, `"choice"`, `"signature"`
- `value` — current value (`str`, `bool`, or `None`)
- `flags` — field flags (read-only, required, etc.)

### Read and Set Field Values

```python
doc = PdfDocument("w2.pdf")

# Read a field value
ssn = doc.get_form_field_value("topmostSubform[0].CopyA[0].f1_01[0]")
print(f"SSN: {ssn}")

# Fill fields
doc.set_form_field_value("employee_name", "Jane Doe")
doc.set_form_field_value("wages", "85000.00")
doc.set_form_field_value("retirement_plan", True)  # checkbox

# Save (values are persisted via incremental save)
doc.save("filled_w2.pdf")
```

### Extract Text with Form Field Values

Filled form field values appear inline in `extract_text` and `to_markdown`:

```python
doc = PdfDocument("filled_w2.pdf")

# Form values appear inline in extracted text
text = doc.extract_text(0)
print(text)  # "Jane Doe" appears where the name field is

# to_markdown includes form fields by default
md = doc.to_markdown(0, include_form_fields=True)

# Exclude form field values
md_clean = doc.to_markdown(0, include_form_fields=False)
```

### Export Form Data

```python
doc = PdfDocument("filled-form.pdf")

# Export as FDF
doc.export_form_data("form_data.fdf")

# Export as XFDF
doc.export_form_data("form_data.xfdf", format="xfdf")
```

## Performance Tips

1. **Reuse document objects** - Opening a PDF has overhead, reuse the object for multiple operations
2. **Use specific page extraction** - `extract_text(page_num)` is faster than `extract_text_all()` if you only need some pages
3. **Disable features you don't need** - Use `ConversionOptions` to skip heading detection, image extraction, etc.

```python
from pdf_oxide import PdfDocument, ConversionOptions

doc = PdfDocument("large.pdf")

# Fast extraction - minimal processing
options = ConversionOptions(
    detect_headings=False,
    detect_lists=False,
    embed_images=False
)
text = doc.to_markdown(0, options)
```

## Error Handling

```python
from pdf_oxide import PdfDocument, PdfError

try:
    doc = PdfDocument("document.pdf")
    text = doc.extract_text(0)
except PdfError as e:
    print(f"PDF error: {e}")
except FileNotFoundError:
    print("File not found")
```

## Examples

See the [examples/](../examples/) directory for complete working examples.

### Quick Script Examples

**Extract text from all PDFs in a folder:**

```python
from pdf_oxide import PdfDocument
from pathlib import Path

for pdf_path in Path("documents").glob("*.pdf"):
    # PdfDocument accepts pathlib.Path directly
    with PdfDocument(pdf_path) as doc:
        text = doc.extract_text_all()

    output_path = pdf_path.with_suffix(".txt")
    output_path.write_text(text)
    print(f"Extracted: {pdf_path.name}")
```

**Batch convert Markdown to PDF:**

```python
from pdf_oxide import Pdf
from pathlib import Path

for md_path in Path("notes").glob("*.md"):
    content = md_path.read_text()
    pdf = Pdf.from_markdown(content)

    output_path = md_path.with_suffix(".pdf")
    pdf.save(str(output_path))
    print(f"Created: {output_path.name}")
```

## Next Steps

- [API Reference](https://docs.rs/pdf_oxide) - Full API documentation
- [PDF Creation Guide](PDF_CREATION_GUIDE.md) - Advanced creation options
- [GitHub Issues](https://github.com/yfedoseev/pdf_oxide/issues) - Report bugs or request features
