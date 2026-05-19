# PDF Oxide for .NET — The Fastest PDF Toolkit for C# & .NET

The fastest .NET PDF library for text extraction, image extraction, and markdown conversion. Powered by a pure-Rust core, exposed to .NET through P/Invoke. 0.8ms mean per document, 5× faster than PyMuPDF, 15× faster than pypdf. 100% pass rate on 3,830 real-world PDFs. MIT / Apache-2.0 licensed.

[![NuGet](https://img.shields.io/nuget/v/PdfOxide)](https://www.nuget.org/packages/PdfOxide)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](https://opensource.org/licenses)

> **Part of the [PDF Oxide](https://github.com/yfedoseev/pdf_oxide) toolkit.** Same Rust core, same speed, same 100% pass rate as the [Rust](https://docs.rs/pdf_oxide), [Python](../python/README.md), [Go](../go/README.md), [JavaScript / TypeScript](../js/README.md), and [WASM](../wasm-pkg/README.md) bindings.

## Quick Start

```bash
dotnet add package PdfOxide
```

```csharp
using PdfOxide.Core;

using var doc = PdfDocument.Open("paper.pdf");
string text = doc.ExtractText(0);
string markdown = doc.ToMarkdown(0);
```

## Why pdf_oxide?

- **Fast** — 0.8ms mean per document, 5× faster than PyMuPDF, 15× faster than pypdf, 29× faster than pdfplumber
- **Reliable** — 100% pass rate on 3,830 test PDFs, zero panics, zero timeouts, no segfaults
- **Complete** — Text extraction, image extraction, search, form fields, PDF creation, and editing in one package
- **Permissive license** — MIT / Apache-2.0 — use freely in commercial and closed-source projects
- **Pure Rust core** — Memory-safe, panic-free, no C dependencies beyond the P/Invoke layer
- **Native binaries included** — Pre-built libraries for Windows, macOS, and Linux (x64 + ARM64) ship in the NuGet package
- **Idiomatic .NET** — `using` statements, async counterparts, LINQ-friendly collections, nullable reference types

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

99.5% text parity vs PyMuPDF and pypdfium2 across the full corpus. The .NET binding is sometimes faster than direct Rust calls on small documents because the P/Invoke path bypasses the Rust-side mutex used by other bindings.

## Installation

```bash
dotnet add package PdfOxide
```

Pre-built native libraries for:

| Platform | x64 | ARM64 |
|---|---|---|
| Windows | Yes | Yes |
| macOS   | Yes | Yes (Apple Silicon) |
| Linux   | Yes | Yes |

Compatible with .NET Standard 2.1, .NET 5, .NET 6, .NET 8, .NET Framework 4.8+, .NET Core, Xamarin, MAUI, and Blazor Server. No system dependencies, no Rust toolchain required.

## API Tour

### Open a document

```csharp
using PdfOxide.Core;

using var doc = PdfDocument.Open("report.pdf");
Console.WriteLine($"Pages: {doc.PageCount}");
Console.WriteLine($"PDF version: {doc.Version.Major}.{doc.Version.Minor}");

// From a stream
using var stream = File.OpenRead("report.pdf");
using var docFromStream = PdfDocument.Open(stream);

// Encrypted PDFs
using var encrypted = PdfDocument.OpenWithPassword("secure.pdf", "user-password");
```

### Text extraction

```csharp
using var doc = PdfDocument.Open("document.pdf");

string text = doc.ExtractText(0);          // single page
string allText = doc.ExtractAllText();     // entire document

string markdown = doc.ToMarkdown(0);
string allMarkdown = doc.ToMarkdownAll();

string html = doc.ToHtml(0);
string allHtml = doc.ToHtmlAll();
```

### Structured text

```csharp
var words = doc.ExtractWords(0);
foreach (var (text, x, y, w, h) in words)
{
    Console.WriteLine($"\"{text}\" at ({x:F1}, {y:F1})");
}

// Text inside a rectangle
string regionText = doc.ExtractTextInRect(0, x: 50, y: 700, width: 200, height: 50);

// Tables
var tables = doc.ExtractTables(0);
foreach (var (rows, cols) in tables)
{
    Console.WriteLine($"{rows}x{cols} table");
}
```

### Search

```csharp
var results = doc.SearchAll("quarterly revenue");
foreach (var (page, text, x, y, w, h) in results)
{
    Console.WriteLine($"Page {page}: \"{text}\" at ({x}, {y})");
}

// Single-page case-sensitive search
var pageResults = doc.SearchPage(0, "exact phrase", caseSensitive: true);
```

### Image extraction

```csharp
using PdfOxide.Core;

using var doc = PdfDocument.Open("document.pdf");
var images = doc.ExtractImages(0);
foreach (var img in images)
{
    Console.WriteLine($"{img.Width}x{img.Height} {img.Format} ({img.Colorspace}, {img.BitsPerComponent} bpc, {img.Data.Length} bytes)");
}
```

### Form fields

```csharp
using PdfOxide.Core;

// Read form fields from an existing PDF
using var doc = PdfDocument.Open("form.pdf");
foreach (var f in doc.GetFormFields())
{
    Console.WriteLine($"{f.Name} ({f.FieldType}) = \"{f.Value}\"");
}

// Fill and flatten form fields via DocumentEditor
using var editor = DocumentEditor.Open("form.pdf");
editor.SetFormFieldValue("employee.name", "Jane Doe");
editor.SetFormFieldValue("employee.email", "jane@example.com");
editor.FlattenForms();
editor.Save("filled-form.pdf");
```

### Document editing — metadata

```csharp
using PdfOxide.Core;

using var editor = DocumentEditor.Open("document.pdf");

// Read metadata
Console.WriteLine($"Title: {editor.Title}");
Console.WriteLine($"Author: {editor.Author}");
Console.WriteLine($"Pages: {editor.PageCount}");

// Update metadata (properties are get/set)
editor.Title = "Quarterly Report";
editor.Author = "Example Author";
editor.Subject = "Q1 2026 Results";

// Save (or save async)
editor.Save("edited.pdf");
// await editor.SaveAsync("edited.pdf");
```

> **Note:** the .NET binding currently exposes document open/read/convert/create, image extraction, form field read/fill/flatten, and metadata editing. Page operations, annotations, rendering, and signatures are available through the Rust core and other language bindings; equivalent .NET surface will be added in a future release — track progress at [issues](https://github.com/yfedoseev/pdf_oxide/issues).

### Creating PDFs

```csharp
using PdfOxide.Core;

// From Markdown, HTML, or plain text
using (var pdf = Pdf.FromMarkdown("# Invoice\n\nTotal: **$42.00**"))
{
    pdf.Save("invoice.pdf");
}

using (var pdf = Pdf.FromHtml("<h1>Report</h1><p>Generated 2026-04-09</p>"))
{
    byte[] bytes = pdf.SaveToBytes();
    File.WriteAllBytes("report.pdf", bytes);
}

// Save to a stream
using (var pdf = Pdf.FromMarkdown("# Stream Example"))
using (var file = File.Create("output.pdf"))
{
    pdf.SaveToStream(file);
}
```

### Page rendering

```csharp
using var doc = PdfDocument.Open("document.pdf");

// Render to PNG
byte[] png = doc.RenderPage(0);
File.WriteAllBytes("page0.png", png);

// Render with zoom
byte[] zoomed = doc.RenderPageZoom(0, zoom: 2.0f);

// Render as JPEG
byte[] jpeg = doc.RenderPage(0, format: 1);

// Thumbnail
byte[] thumb = doc.RenderThumbnail(0);
```

### Async support

```csharp
using var doc = PdfDocument.Open("document.pdf");
string text = await doc.ExtractTextAsync(0);

using var pdf = Pdf.FromMarkdown("# Async");
await pdf.SaveAsync("output.pdf");
```

## OCR & Auto Mode

OCR ships in the prebuilt `PdfOxide` NuGet package as of v0.3.52 — no
`--build-from-source`. Supply an ONNX Runtime shared library (point
`ORT_DYLIB_PATH` at it) and the models, then let `pdf_oxide` route per
page (native text where present, OCR where the page is image-only,
graceful fallback when OCR is unavailable):

```csharp
using PdfOxide.Core;

OcrEngine.PrefetchModels("english");                   // one-off provisioning

using var doc = PdfDocument.Open("scanned-or-mixed.pdf");
string text = doc.ExtractTextAuto(0);                  // recommended
```

For manual `OcrEngine` usage (`Load(...)` + `ExtractText(doc, page)`),
page-type classification (`doc.ClassifyPage(0)`), config knobs, model
selection, and ONNX Runtime install recipes:
**[OCR Guide](https://github.com/yfedoseev/pdf_oxide/blob/main/docs/OCR_GUIDE.md)**.

## Other languages

PDF Oxide ships the same Rust core through six bindings:

- **Rust** — `cargo add pdf_oxide` — see [docs.rs/pdf_oxide](https://docs.rs/pdf_oxide)
- **Python** — `pip install pdf_oxide` — see [python/README.md](../python/README.md)
- **Go** — `go get github.com/yfedoseev/pdf_oxide/go` — see [go/README.md](../go/README.md)
- **JavaScript / TypeScript (Node.js)** — `npm install pdf-oxide` — see [js/README.md](../js/README.md)
- **WASM (browsers, Deno, Bun, edge runtimes)** — `npm install pdf-oxide-wasm` — see [wasm-pkg/README.md](../wasm-pkg/README.md)

A bug fix in the Rust core lands in every binding on the next release.

## Documentation

- **[Full Documentation](https://pdf.oxide.fyi)** — Complete documentation site
- **[Main Repository](https://github.com/yfedoseev/pdf_oxide)** — Rust core, CLI, MCP server, all bindings
- **[Performance Benchmarks](https://pdf.oxide.fyi/docs/performance)** — Full benchmark methodology and results
- **[GitHub Issues](https://github.com/yfedoseev/pdf_oxide/issues)** — Bug reports and feature requests

## Use Cases

- **RAG / LLM pipelines** — Convert PDFs to clean Markdown for retrieval-augmented generation
- **Enterprise document processing** — Extract text, images, and metadata from thousands of PDFs in seconds
- **Form processing** — Read and fill AcroForm fields, flatten forms into static content
- **PDF generation** — Create invoices, reports, certificates, and templated documents programmatically
- **Metadata editing** — Update title, author, subject on existing PDFs without rewriting content
- **PyMuPDF alternative** — MIT licensed, 5× faster, no AGPL restrictions, native .NET API

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

**C#** + **.NET** + **Rust core** | MIT / Apache-2.0 | 100% pass rate on 3,830 PDFs | 0.8ms mean | 5× faster than the industry leaders
