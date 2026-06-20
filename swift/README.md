# pdf_oxide — Swift bindings

Idiomatic Swift bindings over the pdf_oxide C ABI. A `CPdfOxide` system-library
module exposes the cbindgen header via a module map; `PdfOxide` is the Swift
wrapper. Handles are owned by classes (freed in `deinit`); returned C
strings/buffers are copied into Swift `String`/`[UInt8]` and freed via
`free_string`; non-success C-ABI error codes are thrown as `PdfOxideError`.

## Build & test (macOS / Linux with Swift)

The binding links the **default-feature cdylib** (not the Python wheel):

```bash
# 1. build the native library (shipped binding feature set)
cargo build --release --lib --features ocr,rendering,signatures,barcodes,tsa-client,system-fonts

# 2. test + run the example (Package.swift reads PDF_OXIDE_INCLUDE_DIR / _LIB_DIR)
cd swift
export PDF_OXIDE_INCLUDE_DIR="$PWD/../include"
export PDF_OXIDE_LIB_DIR="$PWD/../target/release"
DYLD_LIBRARY_PATH="$PDF_OXIDE_LIB_DIR" swift test
DYLD_LIBRARY_PATH="$PDF_OXIDE_LIB_DIR" swift run basic_extraction
```

## Use

```swift
import PdfOxide

let pdf = try Pdf.fromMarkdown("# Hello\n\nbody\n")
let doc = try Document.open(bytes: try pdf.toBytes())

let pages = try doc.pageCount()
let text  = try doc.extractText(0)
let md    = try doc.toMarkdownAll()
```

## Layout

```
swift/
  Package.swift
  Sources/CPdfOxide/         system-library module (module.modulemap + shim.h)
  Sources/PdfOxide/          idiomatic Swift wrapper (Document, Pdf, PdfOxideError)
  Sources/Example/main.swift runnable example (asserted in CI)
  Tests/PdfOxideTests/       XCTest api-coverage (one test per method)
```

## Verification (CI — same set as every binding)

`.github/workflows/swift.yml` on macOS: build cdylib → `swift test`
(api-coverage) → `swift run basic_extraction` with an output assertion.
