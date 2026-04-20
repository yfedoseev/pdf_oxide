# PDF Oxide for Go — The Fastest PDF Toolkit for Go

The fastest Go PDF library for text extraction, image extraction, and markdown conversion. Powered by a pure-Rust core, exposed to Go through CGo. 0.8ms mean per document, 5× faster than PyMuPDF, 15× faster than pypdf. 100% pass rate on 3,830 real-world PDFs. MIT / Apache-2.0 licensed.

[![Go Reference](https://pkg.go.dev/badge/github.com/yfedoseev/pdf_oxide/go.svg)](https://pkg.go.dev/github.com/yfedoseev/pdf_oxide/go)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](https://opensource.org/licenses)

> **Part of the [PDF Oxide](https://github.com/yfedoseev/pdf_oxide) toolkit.** Same Rust core, same speed, same 100% pass rate as the [Rust](https://docs.rs/pdf_oxide), [Python](../python/README.md), [JavaScript / TypeScript](../js/README.md), [C# / .NET](../csharp/README.md), and [WASM](../wasm-pkg/README.md) bindings.

## Quick Start

```bash
go get github.com/yfedoseev/pdf_oxide/go

# One-time per machine: download the native FFI library
# Use @latest to always get the newest release:
go run github.com/yfedoseev/pdf_oxide/go/cmd/install@latest
# Or pin to a specific version:
go run github.com/yfedoseev/pdf_oxide/go/cmd/install@v0.3.36

# Installer prints the CGO_* env vars to export — e.g.:
export CGO_CFLAGS="-I$HOME/.pdf_oxide/v0.3.36/include"
export CGO_LDFLAGS="$HOME/.pdf_oxide/v0.3.36/lib/linux_amd64/libpdf_oxide.a \
  -lm -lpthread -ldl -lrt -lgcc_s -lutil -lc"
```

> **Why `go run ...install`?** Since v0.3.31 the native Rust staticlib is
> fetched from GitHub Release assets on demand instead of being committed to
> git (previously ~310 MB per release). See `go/lib/README.md` for details
> including the `//go:generate` and `-tags pdf_oxide_dev` flows.

```go
package main

import (
    "fmt"
    "log"

    pdfoxide "github.com/yfedoseev/pdf_oxide/go"
)

func main() {
    doc, err := pdfoxide.Open("paper.pdf")
    if err != nil {
        log.Fatal(err)
    }
    defer doc.Close()

    text, _ := doc.ExtractText(0)
    fmt.Println(text)
}
```

## Why pdf_oxide?

- **Fast** — 0.8ms mean per document, 5× faster than PyMuPDF, 15× faster than pypdf, 29× faster than pdfplumber
- **Reliable** — 100% pass rate on 3,830 test PDFs, zero panics, zero timeouts, no segfaults
- **Complete** — Text extraction, image extraction, search, form fields, PDF creation, and editing in one module
- **Permissive license** — MIT / Apache-2.0 — use freely in commercial and closed-source projects
- **Pure Rust core** — Memory-safe, panic-free, no C dependencies beyond CGo glue
- **Idiomatic Go** — Sentinel errors with `errors.Is`, `defer doc.Close()`, slices instead of opaque iterators, thread-safe concurrent reads
- **No Rust toolchain required** — Pre-built native libraries for Linux, macOS, and Windows (x64, plus Linux/macOS ARM64) are fetched from GitHub Releases by a one-line installer. CGo is required (`CGO_ENABLED=1`, the default).

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

99.5% text parity vs PyMuPDF and pypdfium2 across the full corpus. The Go binding adds negligible overhead — extraction stays within ~15% of direct Rust calls on real-world fixtures.

## Installation

```bash
go get github.com/yfedoseev/pdf_oxide/go
```

Pre-built native libraries for Linux (x64, ARM64), macOS (x64, Apple Silicon), and Windows (x64, ARM64) ship with the module under `lib/<os>_<arch>/`. CGo is required (`CGO_ENABLED=1`, the default). No Rust toolchain is needed to consume the library.

## API Tour

### Open a document

```go
doc, err := pdfoxide.Open("report.pdf")
if err != nil {
    log.Fatal(err)
}
defer doc.Close()

count, _ := doc.PageCount()
major, minor, _ := doc.Version()
fmt.Printf("%d pages, PDF %d.%d\n", count, major, minor)

// Or open from bytes / with a password
doc, _ = pdfoxide.OpenFromBytes(pdfBytes)
doc, _ = pdfoxide.OpenWithPassword("encrypted.pdf", "secret")
```

### Text extraction

```go
text, _    := doc.ExtractText(0)
markdown, _ := doc.ToMarkdown(0)
html, _    := doc.ToHtml(0)
plain, _   := doc.ToPlainText(0)

allText, _     := doc.ExtractAllText()
allMarkdown, _ := doc.ToMarkdownAll()
```

### Structured text

```go
words, _ := doc.ExtractWords(0)
for _, w := range words {
    fmt.Printf("%q @ (%.1f, %.1f)\n", w.Text, w.X, w.Y)
}

lines, _ := doc.ExtractTextLines(0)
chars, _ := doc.ExtractChars(0)

tables, _ := doc.ExtractTables(0)
for _, t := range tables {
    fmt.Printf("%dx%d table\n", t.RowCount, t.ColCount)
}

// Text inside a rectangle
region, _ := doc.ExtractTextInRect(0, 100, 100, 400, 200)
```

### Search

```go
hits, _ := doc.SearchAll("invoice", false)
for _, h := range hits {
    fmt.Printf("page %d: %s\n", h.Page, h.Text)
}

// Single-page search
pageHits, _ := doc.SearchPage(0, "total", false)
```

### Fonts, images, annotations, elements

```go
fonts, _ := doc.Fonts(0)
for _, f := range fonts {
    fmt.Println(f.Name, f.IsEmbedded)
}

images, _ := doc.Images(0)
for _, img := range images {
    fmt.Printf("%dx%d %s (%d bytes)\n", img.Width, img.Height, img.Format, len(img.Data))
}

anns, _ := doc.Annotations(0)
for _, a := range anns {
    fmt.Printf("%s by %s: %s\n", a.Subtype, a.Author, a.Content)
}

elements, _ := doc.PageElements(0)
```

### Form fields

```go
fields, _ := doc.FormFields()
for _, f := range fields {
    fmt.Printf("%s = %s\n", f.Name, f.Value)
}

// Editing form fields requires a DocumentEditor
editor, _ := pdfoxide.OpenEditor("form.pdf")
defer editor.Close()

editor.SetFormFieldValue("employee_name", "Jane Doe")
editor.Save("filled.pdf")
```

### Document editing

```go
editor, err := pdfoxide.OpenEditor("input.pdf")
if err != nil {
    log.Fatal(err)
}
defer editor.Close()

// Apply several metadata fields in one call
_ = editor.ApplyMetadata(pdfoxide.Metadata{
    Title:  "Quarterly Report",
    Author: "Finance Team",
})

// Page operations
_ = editor.SetPageRotation(0, 90)
_ = editor.MovePage(2, 0)
_ = editor.DeletePage(5)
_ = editor.CropMargins(36, 36, 36, 36)
_ = editor.FlattenAllAnnotations()

// Save (or save encrypted)
_ = editor.Save("output.pdf")
_ = editor.SaveEncrypted("secure.pdf", "user-pw", "owner-pw")
```

### Creating PDFs

```go
// From Markdown, HTML, plain text, or images
md, _ := pdfoxide.FromMarkdown("# Title\n\nHello, world.")
defer md.Close()
md.Save("from-md.pdf")

html, _ := pdfoxide.FromHtml("<h1>Title</h1>")
html.Save("from-html.pdf")

txt, _ := pdfoxide.FromText("Hello, world.")
txt.Save("from-txt.pdf")

img, _ := pdfoxide.FromImage("photo.jpg")
img.Save("from-img.pdf")

// Merge several PDFs
merged, _ := pdfoxide.Merge([]string{"a.pdf", "b.pdf", "c.pdf"})
```

### Page rendering

```go
// Render page 0 to PNG (format: 0 = PNG, 1 = JPEG)
img, _ := doc.RenderPage(0, 0)
defer img.Close()
img.SaveToFile("page0.png")

zoomed, _ := doc.RenderPageZoom(0, 2.0, 0)
thumb, _  := doc.RenderThumbnail(0, 200, 0)
```

### Concurrency

Read operations on a `PdfDocument` are protected by an internal `sync.RWMutex`, so multiple goroutines can read concurrently. Writes on a `DocumentEditor` are serialized.

```go
var wg sync.WaitGroup
pageCount, _ := doc.PageCount()

for i := 0; i < pageCount; i++ {
    wg.Add(1)
    go func(page int) {
        defer wg.Done()
        text, _ := doc.ExtractText(page)
        _ = text
    }(i)
}
wg.Wait()
```

### Error handling

All operations return errors explicitly. Sentinel errors are exposed as package-level variables — use `errors.Is` to check for specific conditions:

```go
text, err := doc.ExtractText(0)
if err != nil {
    switch {
    case errors.Is(err, pdfoxide.ErrDocumentClosed):
        log.Print("document is closed")
    case errors.Is(err, pdfoxide.ErrInvalidPageIndex):
        log.Print("invalid page index")
    case errors.Is(err, pdfoxide.ErrExtractionFailed):
        log.Print("extraction failed")
    default:
        log.Printf("unexpected error: %v", err)
    }
}
```

Available sentinels: `ErrInvalidPath`, `ErrDocumentNotFound`, `ErrInvalidFormat`, `ErrExtractionFailed`, `ErrParseError`, `ErrInvalidPageIndex`, `ErrSearchFailed`, `ErrInternal`, `ErrDocumentClosed`, `ErrEditorClosed`, `ErrCreatorClosed`, `ErrIndexOutOfBounds`, `ErrEmptyContent`.

## Other languages

PDF Oxide ships the same Rust core through six bindings:

- **Rust** — `cargo add pdf_oxide` — see [docs.rs/pdf_oxide](https://docs.rs/pdf_oxide)
- **Python** — `pip install pdf_oxide` — see [python/README.md](../python/README.md)
- **JavaScript / TypeScript (Node.js)** — `npm install pdf-oxide` — see [js/README.md](../js/README.md)
- **C# / .NET** — `dotnet add package PdfOxide` — see [csharp/README.md](../csharp/README.md)
- **WASM (browsers, Deno, Bun, edge runtimes)** — `npm install pdf-oxide-wasm` — see [wasm-pkg/README.md](../wasm-pkg/README.md)

A bug fix in the Rust core lands in every binding on the next release.

## Documentation

- **[Full Documentation](https://pdf.oxide.fyi)** — Complete documentation site
- **[Go API Reference](https://pkg.go.dev/github.com/yfedoseev/pdf_oxide/go)** — Full Go API on pkg.go.dev
- **[Main Repository](https://github.com/yfedoseev/pdf_oxide)** — Rust core, CLI, MCP server, all bindings
- **[Performance Benchmarks](https://pdf.oxide.fyi/docs/performance)** — Full benchmark methodology and results
- **[GitHub Issues](https://github.com/yfedoseev/pdf_oxide/issues)** — Bug reports and feature requests

## Use Cases

- **RAG / LLM pipelines** — Convert PDFs to clean Markdown for retrieval-augmented generation
- **Document processing at scale** — Extract text, images, and metadata from thousands of PDFs in seconds
- **Data extraction** — Pull structured data from forms, tables, and layouts
- **PDF generation** — Create invoices, reports, certificates, and templated documents programmatically
- **PyMuPDF alternative** — MIT licensed, 5× faster, no AGPL restrictions, no CPython dependency

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

**Go** + **Rust core** | MIT / Apache-2.0 | 100% pass rate on 3,830 PDFs | 0.8ms mean | 5× faster than the industry leaders
