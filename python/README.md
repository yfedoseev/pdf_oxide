# PDF Oxide - Python Bindings

High-performance PDF parsing for Python with PDF specification compliance.

## Features

- **PDF Spec Compliance**: ISO 32000-1:2008 sections 9, 14.7-14.8
- **Intelligent Text Extraction**: Automatic reading order detection
- **Multi-Column Support**: 4 pluggable layout strategies
- **Font Recovery**: 70-80% character recovery with advanced font support
- **Complex Scripts**: RTL (Arabic/Hebrew), CJK (Japanese/Korean/Chinese), Devanagari, Thai
- **OCR Support**: Optional DBNet++/SVTR for scanned PDFs
- **Format Conversion**: Markdown, HTML, PlainText
- **Performance**: 47.9× faster than PyMuPDF4LLM

## Quick Start

```python
from pdf_oxide import PdfDocument

# Open a PDF
doc = PdfDocument("document.pdf")

# Extract as plain text (with automatic reading order)
text = doc.to_plain_text(0)
print(text)

# Convert to Markdown
markdown = doc.to_markdown(0, detect_headings=True)
with open("output.md", "w") as f:
    f.write(markdown)

# Convert to HTML
html = doc.to_html(0, preserve_layout=False)
with open("output.html", "w") as f:
    f.write(html)
```

## Installation

```bash
pip install pdf_oxide
```

## Development

### Building

```bash
maturin develop
```

### Testing

```bash
pytest
```

### Type Checking

The package includes type stubs (`__init__.pyi`) for full IDE support:

```bash
mypy script.py
```

## API Documentation

See the main README for full API documentation and examples.
