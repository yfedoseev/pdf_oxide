# pdf_oxide — Kotlin bindings

Idiomatic Kotlin/JVM bindings (Android-ready) — a **thin facade** over the
mature [`fyi.oxide:pdf-oxide`](../java) Java binding, which owns the single JNI
native bridge (the `pdf_oxide_jni` crate). This module adds **zero native code**:
it re-exports the Java types (`PdfDocument`, `Pdf`, `PdfPage`, `DocumentEditor`,
`PdfSigner`, `PdfValidator`, `AutoExtractor`, the `geometry`/`text`/`table`/
`search` value types, …) and layers Kotlin sugar — `Optional<T>` → `T?`
(`producerOrNull()`, `valueOrNull()`, …) and `use { }` on the `AutoCloseable`
handles.

> Why a facade? Java, Kotlin, and Scala all run on the JVM, so the native bridge
> is written and tested **once** (in the Java binding). The Kotlin module is a
> pure-JVM library that depends on it.

## Install

Published to Maven Central as `fyi.oxide:pdf-oxide-kotlin`:

```kotlin
// build.gradle.kts
dependencies {
    implementation("fyi.oxide:pdf-oxide-kotlin:0.3.69")
}
```

The JNI native library (`libpdf_oxide_jni`) is not bundled — load it via
`System.loadLibrary("pdf_oxide_jni")` (ship the `.so`/`.dylib` on your
`java.library.path`, or in `jniLibs/<abi>/` on Android), or point the Java
`NativeLoader` at it with `-Dfyi.oxide.pdf.lib.path=<path>`.

## Build & test

The Java binding owns the JNI library (`pdf_oxide_jni`); build it, install the
Java artifact to your local Maven repo, then build the facade:

```bash
# 1. build the JNI native library (full feature set)
cargo build --release -p pdf_oxide_jni --features full

# 2. install the Java binding to ~/.m2 (skip the dev profile's Rust rebuild)
( cd java && mvn -P'!dev' -DskipTests install )

# 3. test the Kotlin facade (JNI lib located via fyi.oxide.pdf.lib.path)
cd kotlin
gradle test      -DPDF_OXIDE_JNI_LIB="$PWD/../target/release/libpdf_oxide_jni.so"
gradle runExample -DPDF_OXIDE_JNI_LIB="$PWD/../target/release/libpdf_oxide_jni.so"
```

`PDF_OXIDE_JNI_LIB` points the Java `NativeLoader` at the JNI library (via the
`-Dfyi.oxide.pdf.lib.path` system property); it defaults to
`../target/release/libpdf_oxide_jni.so`. On Android, ship the `.so` in
`jniLibs/<abi>/` and load it via `System.loadLibrary("pdf_oxide_jni")`.

## Use

```kotlin
import fyi.oxide.pdf.Pdf
import fyi.oxide.pdf.PdfDocument
import fyi.oxide.pdf.producerOrNull

Pdf.fromMarkdown("# Hello\n\nbody\n").use { pdf ->
    PdfDocument.open(pdf.save()).use { doc ->
        println(doc.pageCount())
        println(doc.extractText(0))
        println(doc.toMarkdown())
        println(doc.page(0).words().map { it.text() })
        println(doc.producerOrNull() ?: "(no producer)")   // Optional -> nullable
    }
}
```

## Layout

```
kotlin/
  src/main/kotlin/fyi/oxide/pdf/PdfOxide.kt     Kotlin idioms (Optional -> nullable extensions)
  src/main/kotlin/examples/BasicExtraction.kt   runnable example (asserted in CI)
  src/test/kotlin/fyi/oxide/pdf/ApiCoverageTest.kt  coverage over the Java-backed API
  build.gradle.kts / settings.gradle.kts
```

## Verification (CI)

`.github/workflows/kotlin.yml` on Linux + macOS: build the JNI cdylib → install
the Java binding to local Maven → JDK 17 + Gradle → ktlint + detekt → `gradle
test` (api-coverage) → run example with an output assertion.
