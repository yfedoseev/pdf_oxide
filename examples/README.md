# pdf_oxide Examples

Runnable examples demonstrating pdf_oxide across every supported language. The
**core scenarios (01–07)** are mirrored in each language and exercised in CI on
Linux, macOS, and Windows; the **09-new-features** showcases run on Linux.

## Layout

```
examples/
  rust/        01..08 + 09-new-features/   (also registered as cargo --example targets)
  python/      01..08 + 09-new-features/
  javascript/  01..08 + 09-new-features/
  go/          01..08 + 09-new-features/
  csharp/      01..08 + 09-new-features/
  java/        01..07                      (core scenarios)
  ruby/        01..07                      (core scenarios)
  php/         01..07                      (core scenarios)
  wasm_node/   extract_text.mjs            (WASM on a Node host)
```

## Core scenarios (01–07)

| # | Scenario | What it shows |
|---|----------|---------------|
| 01 | extract-text | open a PDF, page count, per-page text |
| 02 | convert-formats | page → Markdown / HTML / plain text |
| 03 | create-pdf | build a PDF from Markdown/HTML/text |
| 04 | search-text | full-text search across pages |
| 05 | extract-structured | words + bounding boxes, lines, tables |
| 06 | edit-document | metadata edit, page delete, merge |
| 07 | forms-annotations | extract form fields + annotations |

08 (batch-processing) and the `09-new-features` showcases (barcodes, signing,
PDF/A & PDF/UA, encryption, image embedding, …) are demonstrated per language
where supported.

## Running

```bash
# Rust
cargo run --example tutorial_extract_text -- tests/fixtures/simple.pdf

# Python
cd examples/python/01-extract-text && python main.py ../../../tests/fixtures/simple.pdf

# Node.js
node examples/javascript/01-extract-text/index.js tests/fixtures/simple.pdf

# Go
cd examples/go/01-extract-text && go run main.go ../../../tests/fixtures/simple.pdf

# C#
dotnet run --project examples/csharp/01-extract-text/ExtractText.csproj -- tests/fixtures/simple.pdf
```

## Documentation

- API docs: https://docs.rs/pdf_oxide
- Main README: [../README.md](../README.md)
- Contributing guide: [../CONTRIBUTING.md](../CONTRIBUTING.md)

## License

All examples are licensed under MIT OR Apache-2.0, same as the main library.
