# pdf_oxide — Zig bindings

Idiomatic Zig bindings over the pdf_oxide C ABI via `@cImport` — first-class C
interop, no shim. Handles are structs with `deinit`; returned C strings/buffers
are copied into a caller-provided allocator and the C buffer freed via
`free_string`; non-success C-ABI error codes map to `error.PdfOxide`.

> Pinned to **Zig 0.15.1** (pre-1.0 — the build/C-import API drifts between
> releases). CI uses the same version.

## Install

Zig has no central registry — depend on a release tarball by URL + content hash.
`zig fetch --save` downloads it, computes the hash, and writes the dependency
into your `build.zig.zon`:

```bash
zig fetch --save https://github.com/yfedoseev/pdf_oxide/archive/refs/tags/v0.3.69.tar.gz
```

Then in `build.zig`:

```zig
const pdf_oxide = b.dependency("pdf_oxide", .{});
exe.root_module.addImport("pdf_oxide", pdf_oxide.module("pdf_oxide"));
```

The wrapper links the native `libpdf_oxide` cdylib — see "Build & test" for
building it and pointing the build at the header + library.

## Build & test

The binding links the **default-feature cdylib** (not the Python wheel):

```bash
# 1. build the native library (shipped binding feature set)
cargo build --release --lib --features ocr,rendering,signatures,barcodes,tsa-client,system-fonts

# 2. test + run the example (point at the header + cdylib)
cd zig
LD_LIBRARY_PATH="$PWD/../target/release" \
  zig build test    -DPDF_OXIDE_INCLUDE_DIR="$PWD/../include" -DPDF_OXIDE_LIB_DIR="$PWD/../target/release"
LD_LIBRARY_PATH="$PWD/../target/release" \
  zig build example -DPDF_OXIDE_INCLUDE_DIR="$PWD/../include" -DPDF_OXIDE_LIB_DIR="$PWD/../target/release"
```

## Use

```zig
const pdf_oxide = @import("pdf_oxide");

var pdf = try pdf_oxide.Pdf.fromMarkdown("# Hello\n\nbody\n");
defer pdf.deinit();
const bytes = try pdf.toBytes(allocator);
defer allocator.free(bytes);

var doc = try pdf_oxide.Document.openFromBytes(bytes);
defer doc.deinit();
const text = try doc.extractText(allocator, 0);
defer allocator.free(text);
```

## Layout

```
zig/
  lib/pdf_oxide.zig           @cImport wrapper (Document, Pdf) + api-coverage tests
  examples/basic_extraction.zig  runnable example (asserted in CI)
  build.zig / build.zig.zon
```

## Verification (CI — same set as every binding)

`.github/workflows/zig.yml` on Linux + macOS: build cdylib → pinned Zig 0.15.1 →
`zig build test` (api-coverage) → `zig build example` with an output assertion.
