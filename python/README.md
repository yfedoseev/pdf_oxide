# PDF Oxide for Python — The Fastest PDF Toolkit for Python

The fastest Python PDF library for text extraction, image extraction, and markdown conversion. Powered by a pure-Rust core, exposed to Python through PyO3. 0.8ms mean per document, 5× faster than PyMuPDF, 15× faster than pypdf. 100% pass rate on 3,830 real-world PDFs. MIT / Apache-2.0 licensed.

[![PyPI](https://img.shields.io/pypi/v/pdf_oxide.svg)](https://pypi.org/project/pdf_oxide/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/pdf-oxide)](https://pypi.org/project/pdf-oxide/)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](https://opensource.org/licenses)

> **Part of the [PDF Oxide](https://github.com/yfedoseev/pdf_oxide) toolkit.** Same Rust core, same speed, same 100% pass rate as the [Rust](https://docs.rs/pdf_oxide), [Go](../go/README.md), [JavaScript / TypeScript](../js/README.md), [C# / .NET](../csharp/README.md), and [WASM](../wasm-pkg/README.md) bindings.

## Quick Start

```bash
pip install pdf_oxide
```

```python
from pdf_oxide import PdfDocument

doc = PdfDocument("paper.pdf")
text = doc.extract_text(0)
markdown = doc.to_markdown(0, detect_headings=True)
```

## Why pdf_oxide?

- **Fast** — 0.8ms mean per document, 5× faster than PyMuPDF, 15× faster than pypdf, 29× faster than pdfplumber
- **Reliable** — 100% pass rate on 3,830 test PDFs, zero panics, zero timeouts, no segfaults
- **Complete** — Text extraction, image extraction, search, form fields, PDF creation, and editing in one package
- **Permissive license** — MIT / Apache-2.0, unlike PyMuPDF (AGPL-3.0) — use freely in commercial and closed-source projects
- **Pure Rust core** — Memory-safe, panic-free, no C dependencies
- **Native wheels** — No build step, no system dependencies, no Rust toolchain required

## Performance

Benchmarked on 3,830 PDFs from three independent public test suites (veraPDF, Mozilla pdf.js, DARPA SafeDocs). Text extraction libraries only. Single-thread, 60s timeout, no warm-up.

| Library | Mean | p99 | Pass Rate | License |
|---------|------|-----|-----------|---------|
| **PDF Oxide** | **0.8ms** | **9ms** | **100%** | **MIT / Apache-2.0** |
| PyMuPDF | 4.6ms | 28ms | 99.3% | AGPL-3.0 |
| pypdfium2 | 4.1ms | 42ms | 99.2% | Apache-2.0 |
| pymupdf4llm | 55.5ms | 280ms | 99.1% | AGPL-3.0 |
| pdftext | 7.3ms | 82ms | 99.0% | GPL-3.0 |
| pdfminer | 16.8ms | 124ms | 98.8% | MIT |
| pdfplumber | 23.2ms | 189ms | 98.8% | MIT |
| markitdown | 108.8ms | 378ms | 98.6% | MIT |
| pypdf | 12.1ms | 97ms | 98.4% | BSD-3 |

99.5% text parity vs PyMuPDF and pypdfium2 across the full corpus. PDF Oxide extracts text from 7–10× more "hard" files than it misses vs any competitor.

## Installation

```bash
pip install pdf_oxide
```

Pre-built wheels for Linux (x86_64, aarch64, musl), macOS (x86_64, arm64), and Windows (x86_64). Python 3.8 through 3.14. No system dependencies, no Rust toolchain required.

## API Tour

### Open a document

```python
from pdf_oxide import PdfDocument

# Path can be str or pathlib.Path
doc = PdfDocument("report.pdf")
print(f"Pages: {doc.page_count()}")
print(f"PDF version: {doc.version()}")

# Or use as a context manager — closes automatically
with PdfDocument("report.pdf") as doc:
    text = doc.extract_text(0)

# From bytes or with a password
doc = PdfDocument.from_bytes(pdf_bytes)
doc = PdfDocument("encrypted.pdf", password="secret")
```

### Text extraction

```python
text = doc.extract_text(0)            # single page
all_text = doc.extract_text_all()     # all pages joined

# Character-level
chars = doc.extract_chars(0)
for ch in chars:
    print(f"{ch.char} at ({ch.x:.1f}, {ch.y:.1f}) size={ch.font_size:.1f}")

# Word-level
words = doc.extract_words(0)
for w in words:
    print(f"{w.text} at {w.bbox}")

# Line-level
lines = doc.extract_text_lines(0)
for line in lines:
    print(f"Line: {line.text}")

# Override the adaptive word/line gap thresholds (in PDF points)
words = doc.extract_words(0, word_gap_threshold=2.5)
lines = doc.extract_text_lines(0, word_gap_threshold=2.5, line_gap_threshold=4.0)
```

### Format conversion

```python
# Markdown with optional heading detection and form-field inclusion
md = doc.to_markdown(0, detect_headings=True)
md_all = doc.to_markdown_all()

# HTML with optional CSS layout preservation
html = doc.to_html(0, preserve_layout=False)
html_all = doc.to_html_all()

# Plain text with automatic reading order
text = doc.to_plain_text(0)
text_all = doc.to_plain_text_all()
```

### Scoped extraction

Extract content from a region of a page using `within()`. The region is `(x, y, width, height)` in PDF points.

```python
header = doc.within(0, (0, 700, 612, 92)).extract_text()

region = doc.within(0, (50, 400, 500, 200))
region_words = region.extract_words()
region_images = region.extract_images()
```

### Tables

```python
tables = doc.extract_tables(0)
for table in tables:
    print(f"Table with {table.row_count} rows")
```

### Search

```python
results = doc.search("quarterly revenue", case_insensitive=True)
for r in results:
    print(f"Page {r['page']}: '{r['text']}' at {r['bbox']}")

# Single-page literal search
results = doc.search_page(0, "total", case_insensitive=True, literal=True)
```

### Extraction profiles

Pre-tuned profiles adjust how raw text is parsed into words and lines for different document types.

```python
from pdf_oxide import ExtractionProfile

words = doc.extract_words(0, profile=ExtractionProfile.form())
lines = doc.extract_text_lines(0, profile=ExtractionProfile.academic())

# Combine a profile with manual overrides
words = doc.extract_words(0, word_gap_threshold=1.5, profile=ExtractionProfile.aggressive())
```

### Form fields

```python
# Read all form fields
fields = doc.get_form_fields()
for f in fields:
    print(f"{f.name} ({f.field_type}) = {f.value}")

# Fill and save
doc.set_form_field_value("employee_name", "Jane Doe")
doc.set_form_field_value("wages", "85000.00")
doc.set_form_field_value("retirement_plan", True)
doc.save("filled.pdf")

# Export form data as FDF or XFDF
doc.export_form_data("data.fdf")
doc.export_form_data("data.xfdf", format="xfdf")
```

### Images

```python
images = doc.extract_images(0)
for i, img in enumerate(images):
    print(f"{img['width']}x{img['height']} {img['color_space']}")
    img.save(f"image_{i}.png")
```

### PDF creation

```python
from pdf_oxide import Pdf, PdfBuilder, PageSize

# From Markdown, HTML, plain text, or images
Pdf.from_markdown("# Report\n\nHello **world**.").save("report.pdf")
Pdf.from_html("<h1>Invoice</h1><p>Total: $42</p>").save("invoice.pdf")
Pdf.from_text("Simple document content.").save("notes.pdf")
Pdf.from_image("photo.jpg").save("photo.pdf")
Pdf.from_images(["page1.jpg", "page2.png"]).save("album.pdf")

# Builder pattern for advanced control
pdf = (PdfBuilder()
    .title("Annual Report 2025")
    .author("Company Inc.")
    .page_size(PageSize.A4)
    .margins(72.0, 72.0, 72.0, 72.0)
    .from_markdown("# Annual Report\n\n..."))
pdf.save("annual-report.pdf")

# Encryption
pdf = Pdf.from_markdown("# Confidential")
pdf.save_encrypted("secure.pdf", "user-password", "owner-password")
```

### Async support

`AsyncPdfDocument` and `AsyncPdf` run all operations in a background thread, keeping your event loop free. Every method from the sync classes is available as an `async` counterpart.

```python
import asyncio
from pdf_oxide import AsyncPdfDocument, AsyncPdf

async def main():
    doc = await AsyncPdfDocument.open("report.pdf")
    text = await doc.extract_text(0)
    md = await doc.to_markdown(0, detect_headings=True)

    pdf = await AsyncPdf.from_markdown("# Hello")
    await pdf.save("hello.pdf")

asyncio.run(main())
```

## OCR & Auto Mode

The published Python wheel ships with `ocr` built in. Install ONNX
Runtime, drop the models in `PDF_OXIDE_MODEL_DIR`, then let
`pdf_oxide` route per page (native text where present, OCR where the
page is image-only, graceful fallback when OCR is unavailable):

```python
from pdf_oxide import PdfDocument

doc = PdfDocument("scanned-or-mixed.pdf")
text = doc.extract_text_auto(0)         # recommended
```

For manual `OcrEngine(det, rec, dict)` usage,
`doc.extract_text_ocr(page, engine)`, page-type classification, model
selection, and ONNX Runtime install recipes:
**[OCR Guide](https://github.com/yfedoseev/pdf_oxide/blob/main/docs/OCR_GUIDE.md)**.

## Other languages

PDF Oxide ships the same Rust core through six bindings:

- **Rust** — `cargo add pdf_oxide` — see [docs.rs/pdf_oxide](https://docs.rs/pdf_oxide)
- **Go** — `go get github.com/yfedoseev/pdf_oxide/go` — see [go/README.md](../go/README.md)
- **JavaScript / TypeScript (Node.js)** — `npm install pdf-oxide` — see [js/README.md](../js/README.md)
- **C# / .NET** — `dotnet add package PdfOxide` — see [csharp/README.md](../csharp/README.md)
- **WASM (browsers, Deno, Bun, edge runtimes)** — `npm install pdf-oxide-wasm` — see [wasm-pkg/README.md](../wasm-pkg/README.md)

A bug fix in the Rust core lands in every binding on the next release.

## Documentation

- **[Full Documentation](https://pdf.oxide.fyi)** — Complete documentation site
- **[Python Getting Started](https://pdf.oxide.fyi/docs/getting-started/python)** — Step-by-step Python guide
- **[Main Repository](https://github.com/yfedoseev/pdf_oxide)** — Rust core, CLI, MCP server, all bindings
- **[Performance Benchmarks](https://pdf.oxide.fyi/docs/performance)** — Full benchmark methodology and results
- **[GitHub Issues](https://github.com/yfedoseev/pdf_oxide/issues)** — Bug reports and feature requests

## Use Cases

- **RAG / LLM pipelines** — Convert PDFs to clean Markdown for retrieval-augmented generation with LangChain, LlamaIndex, or any framework
- **Document processing at scale** — Extract text, images, and metadata from thousands of PDFs in seconds
- **Data extraction** — Pull structured data from forms, tables, and layouts
- **Academic research** — Parse papers, extract citations, and process large corpora
- **PDF generation** — Create invoices, reports, certificates, and templated documents programmatically
- **PyMuPDF alternative** — MIT licensed, 5× faster, no AGPL restrictions

## Why I built this

I needed PyMuPDF's speed without its AGPL license, and I needed it in more than one language. Nothing existed that ticked all three boxes — fast, MIT, multi-language — so I wrote it. The Rust core is what does the real work; the bindings for Python, Go, JS/TS, C#, and WASM are thin shells around the same code, so a bug fix in one lands in all of them. It now passes 100% of the veraPDF + Mozilla pdf.js + DARPA SafeDocs test corpora (3,830 PDFs) on every platform I've tested.

If it's useful to you, a star on GitHub genuinely helps. If something's broken or missing, [open an issue](https://github.com/yfedoseev/pdf_oxide/issues) — I read all of them.

— Yury

## License

Dual-licensed under [MIT](https://github.com/yfedoseev/pdf_oxide/blob/main/LICENSE-MIT) or [Apache-2.0](https://github.com/yfedoseev/pdf_oxide/blob/main/LICENSE-APACHE) at your option. Unlike AGPL-licensed alternatives, pdf_oxide can be used freely in any project — commercial or open-source — with no copyleft restrictions.

## Citation

```bibtex
@software{pdf_oxide,
  title = {PDF Oxide: Fast PDF Toolkit for Rust, Python, Go, JavaScript, and C#},
  author = {Yury Fedoseev},
  year = {2025},
  url = {https://github.com/yfedoseev/pdf_oxide}
}
```

---

**Python** + **Rust core** | MIT / Apache-2.0 | 100% pass rate on 3,830 PDFs | 0.8ms mean | 5× faster than the industry leaders
