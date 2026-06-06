# PDF Oxide for Node.js — The Fastest PDF Toolkit for JavaScript & TypeScript

The fastest Node.js PDF library for text extraction, image extraction, and markdown conversion. Powered by a pure-Rust core, exposed to Node.js through a native N-API addon. 0.8ms mean per document, 5× faster than PyMuPDF, 15× faster than pypdf. 100% pass rate on 3,830 real-world PDFs. MIT / Apache-2.0 licensed.

[![npm](https://img.shields.io/npm/v/pdf-oxide)](https://www.npmjs.com/package/pdf-oxide)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](https://opensource.org/licenses)

> **Part of the [PDF Oxide](https://github.com/yfedoseev/pdf_oxide) toolkit.** Same Rust core, same speed, same 100% pass rate as the [Rust](https://docs.rs/pdf_oxide), [Python](../python/README.md), [Go](../go/README.md), [C# / .NET](../csharp/README.md), and [WASM](../wasm-pkg/README.md) bindings.
>
> Need to run in browsers, Deno, Bun, or Cloudflare Workers? Use the [WASM build](../wasm-pkg/README.md) instead — same API, no native binaries.

## Quick Start

```bash
npm install pdf-oxide
```

```javascript
import { PdfDocument } from "pdf-oxide";

const doc = PdfDocument.open("paper.pdf");
const text = doc.extractText(0);
const markdown = doc.toMarkdown(0);
doc.close();
```

TypeScript:

```typescript
import { PdfDocument } from "pdf-oxide";

const doc = PdfDocument.open("paper.pdf");
const text: string = doc.extractText(0);
const markdown: string = doc.toMarkdown(0);
doc.close();
```

## Why pdf_oxide?

- **Fast** — 0.8ms mean per document, 5× faster than PyMuPDF, 15× faster than pypdf, 29× faster than pdfplumber
- **Reliable** — 100% pass rate on 3,830 test PDFs, zero panics, zero timeouts, no segfaults
- **Complete** — Text extraction, image extraction, search, form fields, PDF creation, and editing in one package
- **Permissive license** — MIT / Apache-2.0 — use freely in commercial and closed-source projects
- **Pure Rust core** — Memory-safe, panic-free, no C dependencies beyond the N-API glue
- **Native binaries** — Pre-built `.node` addons for Linux, macOS, and Windows (x64 + ARM64)
- **Full TypeScript support** — Type definitions ship in the package

## Performance

Benchmarked on 3,830 PDFs from three independent public test suites (veraPDF, Mozilla pdf.js, DARPA SafeDocs). Text extraction libraries only. Single-thread, 60s timeout, no warm-up.

| Library | Mean | p99 | Pass Rate | License |
|---------|------|-----|-----------|---------|
| **PDF Oxide** | **0.8ms** | **9ms** | **100%** | **MIT / Apache-2.0** |
| PyMuPDF | 4.6ms | 28ms | 99.3% | AGPL-3.0 |
| pypdfium2 | 4.1ms | 42ms | 99.2% | Apache-2.0 |
| pdftext | 7.3ms | 82ms | 99.0% | GPL-3.0 |
| pdfminer | 16.8ms | 124ms | 98.8% | MIT |
| pypdf | 12.1ms | 97ms | 98.4% | BSD-3 |

99.5% text parity vs PyMuPDF and pypdfium2 across the full corpus. The Node.js binding adds negligible overhead — extraction stays within ~25% of direct Rust calls on real-world fixtures.

## Installation

```bash
npm install pdf-oxide
```

Pre-built native addons for:

| Platform | x64 | ARM64 |
|---|---|---|
| Linux (glibc) | Yes | Yes |
| Linux (musl)  | Yes | Yes |
| macOS         | Yes | Yes (Apple Silicon) |
| Windows       | Yes | Yes |

Requires Node.js 18 or newer. No system dependencies. No Rust toolchain required.

## API Tour

### Open a document

```javascript
import { PdfDocument } from "pdf-oxide";

const doc = PdfDocument.open("report.pdf");
console.log(`Pages: ${doc.getPageCount()}`);

const { major, minor } = doc.getVersion();
console.log(`PDF version: ${major}.${minor}`);

doc.close();
```

Use `using` for automatic cleanup (Node.js 22+):

```javascript
{
  using doc = PdfDocument.open("report.pdf");
  const text = doc.extractText(0);
} // doc.close() called automatically
```

### Text extraction

```javascript
const text = doc.extractText(0);            // single page
const markdown = doc.toMarkdown(0);         // single page → Markdown
const html = doc.toHtml(0);                 // single page → HTML
const plain = doc.toPlainText(0);           // single page → plain text

const allMarkdown = doc.toMarkdownAll();    // entire document
const allHtml = doc.toHtmlAll();
```

### Iterate all pages

```javascript
const doc = PdfDocument.open("document.pdf");
const pageCount = doc.getPageCount();

const pages = [];
for (let i = 0; i < pageCount; i++) {
  pages.push(doc.extractText(i));
}

doc.close();
```

### Async wrapper

```javascript
async function extractAll(filePath) {
  const doc = PdfDocument.open(filePath);
  try {
    const pageCount = doc.getPageCount();
    const pages = [];
    for (let i = 0; i < pageCount; i++) {
      pages.push(doc.extractText(i));
    }
    return pages;
  } finally {
    doc.close();
  }
}

const pages = await extractAll("document.pdf");
```

### Error handling

All methods throw on failure. Catch with try/catch:

```javascript
try {
  const text = doc.extractText(0);
} catch (err) {
  console.error("Extraction failed:", err.message);
} finally {
  doc.close();
}
```

## OCR & Auto Mode

OCR ships in the prebuilt `pdf-oxide` native addon as of v0.3.52 — no
`--build-from-source`. Install ONNX Runtime via npm, point at it once,
then let `pdf_oxide` route per page (native text where present, OCR
where the page is image-only, graceful fallback when OCR is
unavailable):

```js
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
process.env.ORT_DYLIB_PATH = require.resolve(
  'onnxruntime-node/bin/napi-v6/linux/x64/libonnxruntime.so.1');

const px = await import('pdf-oxide');
px.prefetchModels(['english']);                    // one-off provisioning

const doc = px.PdfDocument.open('scanned-or-mixed.pdf');
console.log(doc.extractTextAuto(0));               // recommended
```

For manual OCR engine setup, `doc.classifyPage(0)` routing,
custom configs, the WebAssembly (`wasm-ocr`) build, and full
per-binding recipes:
**[OCR Guide](https://github.com/yfedoseev/pdf_oxide/blob/main/docs/OCR_GUIDE.md)**.

## Other languages

PDF Oxide ships the same Rust core through six bindings:

- **Rust** — `cargo add pdf_oxide` — see [docs.rs/pdf_oxide](https://docs.rs/pdf_oxide)
- **Python** — `pip install pdf_oxide` — see [python/README.md](../python/README.md)
- **Go** — `go get github.com/yfedoseev/pdf_oxide/go` — see [go/README.md](../go/README.md)
- **C# / .NET** — `dotnet add package PdfOxide` — see [csharp/README.md](../csharp/README.md)
- **WASM (browsers, Deno, Bun, edge runtimes)** — `npm install pdf-oxide-wasm` — see [wasm-pkg/README.md](../wasm-pkg/README.md)

A bug fix in the Rust core lands in every binding on the next release.

## Documentation

- **[Full Documentation](https://pdf.oxide.fyi)** — Complete documentation site
- **[JavaScript Getting Started](https://pdf.oxide.fyi/docs/getting-started/javascript)** — Step-by-step Node.js guide
- **[Main Repository](https://github.com/yfedoseev/pdf_oxide)** — Rust core, CLI, MCP server, all bindings
- **[Performance Benchmarks](https://pdf.oxide.fyi/docs/performance)** — Full benchmark methodology and results
- **[GitHub Issues](https://github.com/yfedoseev/pdf_oxide/issues)** — Bug reports and feature requests

## Use Cases

- **RAG / LLM pipelines** — Convert PDFs to clean Markdown for retrieval-augmented generation with LangChain.js, LlamaIndex.js, or any framework
- **Document processing at scale** — Extract text, images, and metadata from thousands of PDFs in seconds
- **Server-side PDF rendering** — Extract structured content for search indexing, archival, or transformation pipelines
- **PDF generation** — Create invoices, reports, certificates, and templated documents programmatically
- **PyMuPDF alternative** — MIT licensed, 5× faster, no AGPL restrictions, no Python required

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

**JavaScript** + **TypeScript** + **Rust core** | MIT / Apache-2.0 | 100% pass rate on 3,830 PDFs | 0.8ms mean | 5× faster than the industry leaders
