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

## Installation (Swift Package Manager)

Swift's native distribution channel is SwiftPM via a git tag — no package
registry upload is required. Depend on the package by URL + version:

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/yfedoseev/pdf_oxide", from: "0.3.69"),
],
targets: [
    .target(name: "YourTarget", dependencies: [
        .product(name: "PdfOxide", package: "pdf_oxide"),
    ]),
]
```

The Swift package builds against the native `libpdf_oxide` and the cbindgen
header, which are NOT vendored in the package. You must build the native library
(see "Build & test" above) and point the build at it via two environment
variables that `Package.swift` reads:

```bash
export PDF_OXIDE_LIB_DIR="/path/to/target/release"      # dir containing libpdf_oxide
export PDF_OXIDE_INCLUDE_DIR="/path/to/include"         # dir containing the cbindgen header
```

At run time the dynamic library must be locatable (e.g. `DYLD_LIBRARY_PATH` on
macOS, or an `-rpath` baked at link time).

**Future work — zero-config binaryTarget.** A self-contained distribution would
add a `binaryTarget` referencing a prebuilt `PdfOxide.xcframework` (the same
artifact the CocoaPods release assembles, see `../objc/PUBLISHING.md`), removing
the `PDF_OXIDE_LIB_DIR` / `PDF_OXIDE_INCLUDE_DIR` requirement. That is not wired
up yet, so the env-var path above is current.

**Why no Swift CocoaPods podspec.** SwiftPM-by-tag is Swift's idiomatic channel
and needs no registry. A second CocoaPods spec for Swift would duplicate the
ObjC pod (`../objc/PdfOxide.podspec`) for the identical native library with no
added reach, so it is intentionally omitted.

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
