# pdf_oxide — Scala bindings

Idiomatic Scala 3 bindings — a **thin facade** over the mature
[`fyi.oxide:pdf-oxide`](../java) Java binding, which owns the single JNI native
bridge (the `pdf_oxide_jni` crate). This module adds **zero native code**: it
uses the Java types directly and layers Scala 3 `extension` methods —
`java.util.Optional[T]` → `Option[T]` (`producerOption`, `valueOption`, …) and
`java.util.List[T]` → `Seq[T]` (`wordsSeq`, `searchSeq`, …). `scala.util.Using`
works on the `AutoCloseable` handles.

> Why a facade? Java, Kotlin, and Scala all run on the JVM, so the native bridge
> is written and tested **once** (in the Java binding). The Scala module is a
> pure-JVM library that depends on it.

## Install

Published to Maven Central. The artifact is cross-versioned for Scala 3, so use
sbt's `%%` (which resolves the `_3` suffix → `pdf-oxide-scala_3`):

```scala
// build.sbt
libraryDependencies += "fyi.oxide" %% "pdf-oxide-scala" % "0.3.69"
```

The JNI native library (`libpdf_oxide_jni`) is not bundled — make it loadable via
`System.loadLibrary("pdf_oxide_jni")` on your `java.library.path`, or point the
Java `NativeLoader` at it with `-Dfyi.oxide.pdf.lib.path=<path>`.

## Build & test

```bash
# 1. build the JNI native library (full feature set)
cargo build --release -p pdf_oxide_jni --features full

# 2. install the Java binding to ~/.m2 (skip the dev profile's Rust rebuild)
( cd java && mvn -P'!dev' -DskipTests install )

# 3. test the Scala facade (JNI lib located via fyi.oxide.pdf.lib.path)
cd scala
sbt -DPDF_OXIDE_JNI_LIB="$PWD/../target/release/libpdf_oxide_jni.so" test
sbt -Dfyi.oxide.pdf.lib.path="$PWD/../target/release/libpdf_oxide_jni.so" 'runMain examples.basicExtraction'
```

`PDF_OXIDE_JNI_LIB` points the Java `NativeLoader` at the JNI library; it
defaults to `../target/release/libpdf_oxide_jni.so`.

## Use

```scala
import fyi.oxide.pdf.{Pdf, PdfDocument, producerOption, wordsSeq}
import scala.util.Using

Using.resource(Pdf.fromMarkdown("# Hello\n\nbody\n")): pdf =>
  Using.resource(PdfDocument.open(pdf.save())): doc =>
    println(doc.pageCount())
    println(doc.extractText(0))
    println(doc.toMarkdown())
    println(doc.page(0).wordsSeq.map(_.text))       // List -> Seq
    println(doc.producerOption.getOrElse("(none)")) // Optional -> Option
```

## Layout

```
scala/
  src/main/scala/fyi/oxide/pdf/PdfOxide.scala   Scala idioms (Optional -> Option, List -> Seq extensions)
  src/main/scala/examples/BasicExtraction.scala runnable example (asserted in CI)
  src/test/scala/fyi/oxide/pdf/ApiCoverageSpec.scala  coverage over the Java-backed API
  build.sbt
```

## Verification (CI)

`.github/workflows/scala.yml` on Linux + macOS: build the JNI cdylib → install
the Java binding to local Maven → JDK 17 + sbt → scalafmt check → `sbt test`
(api-coverage) → run example with an output assertion.
