# PdfOxide — Julia bindings

Idiomatic Julia bindings over the pdf_oxide C ABI via `ccall` (direct, no shim).
The native library (`libpdf_oxide`) is loaded at runtime; handles are wrapped in
mutable structs with finalizers; C strings/buffers are copied into Julia and
freed via `free_string`; non-success C-ABI error codes throw `PdfOxideError`.
Page indices are 0-based.

## Install

Once registered in Julia's General registry, the package name is `PdfOxide`:

```julia
using Pkg
Pkg.add("PdfOxide")
```

The native `libpdf_oxide` cdylib is loaded at runtime and is **not** bundled —
see "Setup" for building it and pointing the loader at it.

## Setup

The binding links the **default-feature cdylib** (not the Python wheel):

```bash
# 1. build the native library (shipped binding feature set)
cargo build --release --lib --features ocr,rendering,signatures,barcodes,tsa-client,system-fonts

# 2. test + run the example (point the loader at the cdylib)
cd julia
LD_LIBRARY_PATH="$PWD/../target/release" julia --project=. -e 'using Pkg; Pkg.test()'
LD_LIBRARY_PATH="$PWD/../target/release" julia --project=. examples/basic_extraction.jl
```

Native library resolution: `PDF_OXIDE_LIB_PATH` (full path) → `PDF_OXIDE_LIB_DIR`
→ `../target/release` → `target/release` → system loader.

## Use

```julia
using PdfOxide

pdf = from_markdown("# Hello\n\nbody\n")
doc = open_from_bytes(to_bytes(pdf))

page_count(doc)
extract_text(doc, 0)     # 0-based page index
to_markdown_all(doc)
```

## Layout

```
julia/
  src/PdfOxide.jl            ccall wrapper (PdfDocument, Pdf, PdfOxideError)
  examples/basic_extraction.jl  runnable example (asserted in CI)
  test/runtests.jl           one test per public function
  Project.toml
```

## Verification (CI — same set as every binding)

`.github/workflows/julia.yml` on Linux + macOS: build cdylib → `Pkg.test()`
(api-coverage) → run example with an output assertion.
