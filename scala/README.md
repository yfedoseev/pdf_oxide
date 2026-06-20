# pdf_oxide — Scala bindings

Idiomatic Scala 3 bindings over the pdf_oxide C ABI via **JNA** (the same FFI
mechanism as the Kotlin binding) — no native compile step. JNA loads
`libpdf_oxide.{so,dylib,dll}` at runtime; handles are `AutoCloseable` (use with
`scala.util.Using`); returned C strings/buffers are copied into Scala and freed
via `free_string`; C-ABI error codes throw `PdfOxideException`.

## Build & test

The binding links the **default-feature cdylib** (not the Python wheel):

```bash
# 1. build the native library (shipped binding feature set)
cargo build --release --lib --features ocr,rendering,signatures,barcodes,tsa-client,system-fonts

# 2. test (JNA finds the cdylib via jna.library.path)
cd scala
sbt -DPDF_OXIDE_LIB_DIR="$PWD/../target/release" test
sbt -Djna.library.path="$PWD/../target/release" 'runMain examples.basicExtraction'
```

## Use

```scala
import fyi.oxide.pdf.{Pdf, PdfDocument}
import scala.util.Using

Using.resource(Pdf.fromMarkdown("# Hello\n\nbody\n")): pdf =>
  Using.resource(PdfDocument.openFromBytes(pdf.toBytes())): doc =>
    println(doc.pageCount())
    println(doc.extractText(0))
    println(doc.toMarkdownAll())
```

## Layout

```
scala/
  src/main/scala/fyi/oxide/pdf/PdfOxide.scala   JNA wrapper (PdfDocument, Pdf)
  src/main/scala/examples/BasicExtraction.scala runnable example (asserted in CI)
  src/test/scala/fyi/oxide/pdf/ApiCoverageSpec.scala  one test per method
  build.sbt
```

## Verification (CI — same set as every binding)

`.github/workflows/scala.yml` on Linux + macOS: build cdylib → JDK 17 + sbt →
`sbt test` (api-coverage) → run example with an output assertion.
