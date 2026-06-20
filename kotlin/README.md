# pdf_oxide — Kotlin bindings

Idiomatic Kotlin/JVM bindings (Android-ready) over the pdf_oxide C ABI via
**JNA** — pure Kotlin, no native compile step. JNA loads
`libpdf_oxide.{so,dylib,dll}` at runtime; handles are `AutoCloseable`; returned
C strings/buffers are copied into Kotlin and freed via `free_string`; C-ABI
error codes throw `PdfOxideException`. Suspending coroutine helpers run the
CPU-bound native work off the caller's thread.

## Build & test

The binding links the **default-feature cdylib** (not the Python wheel):

```bash
# 1. build the native library (shipped binding feature set)
cargo build --release --lib --features ocr,rendering,signatures,barcodes,tsa-client,system-fonts

# 2. test (JNA finds the cdylib via jna.library.path)
cd kotlin
gradle test -DPDF_OXIDE_LIB_DIR="$PWD/../target/release"
gradle runExample -DPDF_OXIDE_LIB_DIR="$PWD/../target/release"
```

`jna.library.path` is set from `-DPDF_OXIDE_LIB_DIR`, the `PDF_OXIDE_LIB_DIR`
env var, or `../target/release` by default. On Android, ship the `.so` in
`jniLibs/<abi>/` and JNA resolves it automatically.

## Use

```kotlin
import fyi.oxide.pdf.Pdf
import fyi.oxide.pdf.PdfDocument

Pdf.fromMarkdown("# Hello\n\nbody\n").use { pdf ->
    PdfDocument.openFromBytes(pdf.toBytes()).use { doc ->
        println(doc.pageCount())
        println(doc.extractText(0))
        println(doc.toMarkdownAll())
    }
}

// Coroutine-friendly:
val md = doc.toMarkdownAllAsync()   // suspends on Dispatchers.Default
```

## Layout

```
kotlin/
  src/main/kotlin/fyi/oxide/pdf/PdfOxide.kt     JNA wrapper (PdfDocument, Pdf)
  src/main/kotlin/fyi/oxide/pdf/Coroutines.kt   suspending extensions
  src/main/kotlin/examples/BasicExtraction.kt   runnable example (asserted in CI)
  src/test/kotlin/fyi/oxide/pdf/ApiCoverageTest.kt  one test per public method
  build.gradle.kts / settings.gradle.kts
```

## Verification (CI — same set as every binding)

`.github/workflows/kotlin.yml` on Linux + macOS: build cdylib → JDK 17 + Gradle
→ `gradle test` (api-coverage incl. coroutine helpers) → run example with an
output assertion.
