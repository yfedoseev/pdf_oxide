"""
PDF Oxide - Production-Grade PDF Parsing in Python

A high-performance PDF parsing library built with Rust.
47.9× faster than PyMuPDF4LLM with PDF specification compliance.

# Core Features (v0.2.0)

- **PDF Spec Compliance**: ISO 32000-1:2008 sections 9, 14.7-14.8
- **Text Extraction**: Intelligent character-to-Unicode mapping with 5-level priority
- **Reading Order**: Automatic multi-column detection (4 strategies)
- **Font Support**: 70-80% character recovery including CID-keyed fonts
- **OCR Support**: DBNet++ detection + SVTR recognition for scanned PDFs
- **Complex Scripts**: RTL (Arabic/Hebrew), CJK (Japanese/Korean/Chinese), Devanagari, Thai
- **Format Conversion**: Markdown, HTML, PlainText
- **High Performance**: Built with Rust for safety and speed

# Quick Start

```python
from pdf_oxide import PdfDocument

# Open a PDF
doc = PdfDocument("document.pdf")

# Get metadata
print(f"PDF version: {doc.version()}")
print(f"Pages: {doc.page_count()}")

# Extract as plain text (with automatic reading order)
text = doc.to_plain_text(0)
print(text)

# Convert to Markdown (with automatic reading order & layout handling)
markdown = doc.to_markdown(0, detect_headings=True)
with open("output.md", "w") as f:
    f.write(markdown)

# Convert to HTML (with multi-column support & semantic structure)
html = doc.to_html(0, preserve_layout=False)
with open("output.html", "w") as f:
    f.write(html)

# Process all pages
markdown_all = doc.to_markdown_all(detect_headings=True)
```

# Performance

- **47.9× faster** than PyMuPDF4LLM
- Average 53ms per PDF
- Processes 100 PDFs in 5.3 seconds

# License

Dual-licensed under MIT OR Apache-2.0.
"""

from .pdf_oxide import PdfDocument, VERSION

__all__ = ["PdfDocument", "VERSION"]
__version__ = "0.2.0"
