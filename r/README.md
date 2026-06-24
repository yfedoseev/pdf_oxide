# pdfoxide — R bindings

Idiomatic R bindings for pdf_oxide: fast PDF text / Markdown / HTML extraction,
plus building PDFs from Markdown / HTML / text. A small C shim bridges R's
`.Call` interface to the pdf_oxide C ABI; handles are R external pointers freed
by the GC; C-ABI error codes are raised as R errors. Page indices are 0-based.

## Install

Once accepted on CRAN, the package name is `pdfoxide`:

```r
install.packages("pdfoxide")
```

To build from this checkout instead, the package links the **default-feature
cdylib** (not the Python wheel):

```bash
# 1. build the native library (shipped binding feature set)
cargo build --release --lib --features ocr,rendering,signatures,barcodes,tsa-client,system-fonts

# 2. install the package (point it at the header + cdylib)
PDF_OXIDE_INCLUDE_DIR="$PWD/include" PDF_OXIDE_LIB_DIR="$PWD/target/release" \
  R CMD INSTALL r/

# 3. run the tests
LD_LIBRARY_PATH="$PWD/target/release" \
  Rscript -e 'library(tinytest); print(test_package("pdfoxide"))'
```

## Use

```r
library(pdfoxide)

pdf <- pdf_from_markdown("# Hello\n\nbody\n")
doc <- pdf_open_bytes(pdf_to_bytes(pdf))

pdf_page_count(doc)
pdf_extract_text(doc, 0)     # 0-based page index
pdf_to_markdown_all(doc)
```

## Layout

```
r/
  R/pdf_oxide.R                 exported R functions
  src/pdf_oxide.c               C shim (.Call -> C ABI, external pointers)
  src/Makevars                  include + link the cdylib
  inst/tinytest/                api-coverage tests (one per function)
  inst/examples/                runnable example (asserted in CI)
  DESCRIPTION / NAMESPACE
```

## Verification (CI — same set as every binding)

`.github/workflows/r.yml` on Linux + macOS: build cdylib → `R CMD INSTALL` →
`tinytest` (api-coverage) → run example with an output assertion.
