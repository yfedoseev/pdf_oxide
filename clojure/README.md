# pdf_oxide — Clojure bindings

Idiomatic Clojure bindings over the pdf_oxide C ABI via **JNA** (same FFI
mechanism as the Kotlin/Scala bindings). JNA loads
`libpdf_oxide.{so,dylib,dll}` at runtime; document/pdf handles are `Closeable`
(use with `with-open`); returned C strings/buffers are copied into Clojure and
freed via `free_string`; non-success C-ABI error codes throw `ex-info` carrying
`{:code …}`.

## Build & test

The binding links the **default-feature cdylib** (not the Python wheel):

```bash
# 1. build the native library (shipped binding feature set)
cargo build --release --lib --features ocr,rendering,signatures,barcodes,tsa-client,system-fonts

# 2. test (JNA finds the cdylib via jna.library.path)
cd clojure
clojure -J-Djna.library.path="$PWD/../target/release" -M:test
clojure -J-Djna.library.path="$PWD/../target/release" -M:example
```

## Use

```clojure
(require '[pdf-oxide.core :as pdf])

(with-open [p (pdf/from-markdown "# Hello\n\nbody\n")
            d (pdf/open-bytes (pdf/to-bytes p))]
  (pdf/page-count d)
  (pdf/extract-text d 0)
  (pdf/to-markdown-all d))
```

## Layout

```
clojure/
  src/pdf_oxide/core.clj     JNA wrapper (Document, Pdf, fns)
  src/pdf_oxide/example.clj  runnable example (asserted in CI)
  test/pdf_oxide/core_test.clj  one test per public fn
  deps.edn
```

## Verification (CI — same set as every binding)

`.github/workflows/clojure.yml` on Linux + macOS: build cdylib → JDK 17 +
Clojure CLI → `clojure -M:test` (api-coverage) → run example with an output
assertion.
