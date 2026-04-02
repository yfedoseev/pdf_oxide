# PDF Oxide - The Fastest PDF Toolkit for Python, Rust, WASM, CLI & AI

The fastest PDF library for text extraction, image extraction, and markdown conversion. Rust core with Python bindings, WASM support, CLI tool, and MCP server for AI assistants. 0.8ms mean per document, 5× faster than PyMuPDF, 15× faster than pypdf. 100% pass rate on 3,830 real-world PDFs. MIT licensed.

[![Crates.io](https://img.shields.io/crates/v/pdf_oxide.svg)](https://crates.io/crates/pdf_oxide)
[![PyPI](https://img.shields.io/pypi/v/pdf_oxide.svg)](https://pypi.org/project/pdf_oxide/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/pdf-oxide)](https://pypi.org/project/pdf-oxide/)
[![npm](https://img.shields.io/npm/v/pdf-oxide-wasm)](https://www.npmjs.com/package/pdf-oxide-wasm)
[![Documentation](https://docs.rs/pdf_oxide/badge.svg)](https://docs.rs/pdf_oxide)
[![Build Status](https://github.com/yfedoseev/pdf_oxide/workflows/CI/badge.svg)](https://github.com/yfedoseev/pdf_oxide/actions)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](https://opensource.org/licenses)

## Quick Start

### Python
```python
from pdf_oxide import PdfDocument

# path can be str or pathlib.Path; use with for scoped access
doc = PdfDocument("paper.pdf")
# or: with PdfDocument("paper.pdf") as doc: ...
text = doc.extract_text(0)
chars = doc.extract_chars(0)
markdown = doc.to_markdown(0, detect_headings=True)
```

```bash
pip install pdf_oxide
```

### Rust
```rust
use pdf_oxide::PdfDocument;

let mut doc = PdfDocument::open("paper.pdf")?;
let text = doc.extract_text(0)?;
let images = doc.extract_images(0)?;
let markdown = doc.to_markdown(0, Default::default())?;
```

```toml
[dependencies]
pdf_oxide = "0.3"
```

### CLI
```bash
pdf-oxide text document.pdf
pdf-oxide markdown document.pdf -o output.md
pdf-oxide search document.pdf "pattern"
pdf-oxide merge a.pdf b.pdf -o combined.pdf
```

```bash
brew install yfedoseev/tap/pdf-oxide
```

### MCP Server (for AI assistants)
```bash
# Install
brew install yfedoseev/tap/pdf-oxide   # includes pdf-oxide-mcp

# Configure in Claude Desktop / Claude Code / Cursor
{
  "mcpServers": {
    "pdf-oxide": { "command": "crgx", "args": ["pdf_oxide_mcp@latest"] }
  }
}
```

## Why pdf_oxide?

- **Fast** — 0.8ms mean per document, 5× faster than PyMuPDF, 15× faster than pypdf, 29× faster than pdfplumber
- **Reliable** — 100% pass rate on 3,830 test PDFs, zero panics, zero timeouts
- **Complete** — Text extraction, image extraction, PDF creation, and editing in one library
- **Multi-platform** — Rust, Python, JavaScript/WASM, CLI, and MCP server for AI assistants
- **Permissive license** — MIT / Apache-2.0 — use freely in commercial and open-source projects

## Performance

Benchmarked on 3,830 PDFs from three independent public test suites (veraPDF, Mozilla pdf.js, DARPA SafeDocs). Text extraction libraries only (no OCR). Single-thread, 60s timeout, no warm-up.

### Python Libraries

| Library | Mean | p99 | Pass Rate | License |
|---------|------|-----|-----------|---------|
| **PDF Oxide** | **0.8ms** | **9ms** | **100%** | **MIT** |
| PyMuPDF | 4.6ms | 28ms | 99.3% | AGPL-3.0 |
| pypdfium2 | 4.1ms | 42ms | 99.2% | Apache-2.0 |
| pymupdf4llm | 55.5ms | 280ms | 99.1% | AGPL-3.0 |
| pdftext | 7.3ms | 82ms | 99.0% | GPL-3.0 |
| pdfminer | 16.8ms | 124ms | 98.8% | MIT |
| pdfplumber | 23.2ms | 189ms | 98.8% | MIT |
| markitdown | 108.8ms | 378ms | 98.6% | MIT |
| pypdf | 12.1ms | 97ms | 98.4% | BSD-3 |

### Rust Libraries

| Library | Mean | p99 | Pass Rate | Text Extraction |
|---------|------|-----|-----------|-----------------|
| **PDF Oxide** | **0.8ms** | **9ms** | **100%** | **Built-in** |
| oxidize_pdf | 13.5ms | 11ms | 99.1% | Basic |
| unpdf | 2.8ms | 10ms | 95.1% | Basic |
| pdf_extract | 4.08ms | 37ms | 91.5% | Basic |
| lopdf | 0.3ms | 2ms | 80.2% | No built-in extraction |

### Text Quality

99.5% text parity vs PyMuPDF and pypdfium2 across the full corpus. PDF Oxide extracts text from 7–10× more "hard" files than it misses vs any competitor.

### Corpus

| Suite | PDFs | Pass Rate |
|-------|-----:|----------:|
| [veraPDF](https://github.com/veraPDF/veraPDF-corpus) (PDF/A compliance) | 2,907 | 100% |
| [Mozilla pdf.js](https://github.com/mozilla/pdf.js/tree/master/test/pdfs) | 897 | 99.2% |
| [SafeDocs](https://github.com/pdf-association/safedocs) (targeted edge cases) | 26 | 100% |
| **Total** | **3,830** | **100%** |

100% pass rate on all valid PDFs — the 7 non-passing files across the corpus are intentionally broken test fixtures (missing PDF header, fuzz-corrupted catalogs, invalid xref streams).

## Features

| Extract | Create | Edit |
|---------|--------|------|
| Text & Layout | Documents | Annotations |
| Images | Tables | Form Fields |
| Forms | Graphics | Bookmarks |
| Annotations | Templates | Links |
| Bookmarks | Images | Content |

## Python API

```python
from pdf_oxide import PdfDocument

# Path can be str or pathlib.Path; use "with PdfDocument(...) as doc" for context manager
doc = PdfDocument("report.pdf")
print(f"Pages: {doc.page_count()}")
print(f"Version: {doc.version()}")

# 1. Scoped extraction (v0.3.14)
# Extract only from a specific area: (x, y, width, height)
header = doc.within(0, (0, 700, 612, 92)).extract_text()

# 2. Word-level extraction (v0.3.14)
words = doc.extract_words(0)
for w in words:
    print(f"{w.text} at {w.bbox}")
    # Access individual characters in the word
    # print(w.chars[0].font_name)

# 3. Line-level extraction (v0.3.14)
lines = doc.extract_text_lines(0)
for line in lines:
    print(f"Line: {line.text}")

# 4. Table extraction (v0.3.14)
tables = doc.extract_tables(0)
for table in tables:
    print(f"Table with {table.row_count} rows")

# 5. Traditional extraction
text = doc.extract_text(0)
chars = doc.extract_chars(0)
```

### Form Fields

```python
# Extract form fields
fields = doc.get_form_fields()
for f in fields:
    print(f"{f.name} ({f.field_type}) = {f.value}")

# Fill and save
doc.set_form_field_value("employee_name", "Jane Doe")
doc.set_form_field_value("wages", "85000.00")
doc.save("filled.pdf")
```

## Rust API

```rust
use pdf_oxide::PdfDocument;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut doc = PdfDocument::open("paper.pdf")?;

    // Extract text
    let text = doc.extract_text(0)?;

    // Character-level extraction
    let chars = doc.extract_chars(0)?;

    // Extract images
    let images = doc.extract_images(0)?;

    // Vector graphics
    let paths = doc.extract_paths(0)?;

    Ok(())
}
```

### Form Fields (Rust)

```rust
use pdf_oxide::editor::{DocumentEditor, EditableDocument, SaveOptions};
use pdf_oxide::editor::form_fields::FormFieldValue;

let mut editor = DocumentEditor::open("w2.pdf")?;
editor.set_form_field_value("employee_name", FormFieldValue::Text("Jane Doe".into()))?;
editor.save_with_options("filled.pdf", SaveOptions::incremental())?;
```

## Installation

### Python

```bash
pip install pdf_oxide
```

Wheels available for Linux, macOS, and Windows. Python 3.8–3.14.

### Rust

```toml
[dependencies]
pdf_oxide = "0.3"
```

### JavaScript/WASM

```bash
npm install pdf-oxide-wasm
```

```javascript
const { WasmPdfDocument } = require("pdf-oxide-wasm");
```

### CLI

```bash
brew install yfedoseev/tap/pdf-oxide    # Homebrew (macOS/Linux)
cargo install pdf_oxide_cli             # Cargo
cargo binstall pdf_oxide_cli            # Pre-built binary via cargo-binstall
```

### MCP Server

```bash
brew install yfedoseev/tap/pdf-oxide    # Included with CLI in Homebrew
cargo install pdf_oxide_mcp             # Cargo
```

## CLI

22 commands for PDF processing directly from your terminal:

```bash
pdf-oxide text report.pdf                      # Extract text
pdf-oxide markdown report.pdf -o report.md     # Convert to Markdown
pdf-oxide html report.pdf -o report.html       # Convert to HTML
pdf-oxide info report.pdf                      # Show metadata
pdf-oxide search report.pdf "neural.?network"  # Search (regex)
pdf-oxide images report.pdf -o ./images/       # Extract images
pdf-oxide merge a.pdf b.pdf -o combined.pdf    # Merge PDFs
pdf-oxide split report.pdf -o ./pages/         # Split into pages
pdf-oxide watermark doc.pdf "DRAFT"            # Add watermark
pdf-oxide forms w2.pdf --fill "name=Jane"      # Fill form fields
```

Run `pdf-oxide` with no arguments for interactive REPL mode. Use `--pages 1-5` to process specific pages, `--json` for machine-readable output.

## MCP Server

`pdf-oxide-mcp` lets AI assistants (Claude, Cursor, etc.) extract content from PDFs locally via the [Model Context Protocol](https://modelcontextprotocol.io/).

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "pdf-oxide": { "command": "crgx", "args": ["pdf_oxide_mcp@latest"] }
  }
}
```

The server exposes an `extract` tool that supports text, markdown, and HTML output formats with optional page ranges and image extraction. All processing runs locally — no files leave your machine.

## Building from Source

```bash
# Clone and build
git clone https://github.com/yfedoseev/pdf_oxide
cd pdf_oxide
cargo build --release

# Run tests
cargo test

# Build Python bindings
maturin develop
```

## Documentation

- **[Full Documentation](https://pdf.oxide.fyi)** - Complete documentation site
- **[Getting Started (Rust)](https://pdf.oxide.fyi/docs/getting-started/rust)** - Rust guide
- **[Getting Started (Python)](https://pdf.oxide.fyi/docs/getting-started/python)** - Python guide
- **[Getting Started (WASM)](https://pdf.oxide.fyi/docs/getting-started/javascript)** - Browser and Node.js guide
- **[Getting Started (CLI)](https://pdf.oxide.fyi/docs/getting-started/cli)** - CLI guide
- **[Getting Started (MCP)](https://pdf.oxide.fyi/docs/getting-started/mcp)** - MCP server for AI assistants
- **[API Docs](https://docs.rs/pdf_oxide)** - Full Rust API reference
- **[Performance Benchmarks](https://pdf.oxide.fyi/docs/performance)** - Full benchmark methodology and results

## Use Cases

- **RAG / LLM pipelines** — Convert PDFs to clean Markdown for retrieval-augmented generation with LangChain, LlamaIndex, or any framework
- **AI assistants** — Give Claude, Cursor, or any MCP-compatible tool direct PDF access via the MCP server
- **Document processing at scale** — Extract text, images, and metadata from thousands of PDFs in seconds
- **Data extraction** — Pull structured data from forms, tables, and layouts
- **Academic research** — Parse papers, extract citations, and process large corpora
- **PDF generation** — Create invoices, reports, certificates, and templated documents programmatically
- **PyMuPDF alternative** — MIT licensed, 5× faster, no AGPL restrictions

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option. Unlike AGPL-licensed alternatives, pdf_oxide can be used freely in any project — commercial or open-source — with no copyleft restrictions.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
cargo build && cargo test && cargo fmt && cargo clippy -- -D warnings
```

## Citation

```bibtex
@software{pdf_oxide,
  title = {PDF Oxide: Fast PDF Toolkit for Rust and Python},
  author = {Yury Fedoseev},
  year = {2025},
  url = {https://github.com/yfedoseev/pdf_oxide}
}
```

---

**Rust** + **Python** + **WASM** + **CLI** + **MCP** | MIT/Apache-2.0 | 100% pass rate on 3,830 PDFs | 0.8ms mean | 5× faster than the industry leaders
