# pdf_oxide — Java binding (`fyi.oxide:pdf-oxide`)

Native Java binding to [pdf_oxide](https://github.com/yfedoseev/pdf_oxide) via JNI (jni-rs 0.22). Same Rust core as the Python / Go / JS / C# / WASM bindings, sub-millisecond text extraction, 100% pass rate on 3,830 real-world PDFs. **JDK 11 LTS floor**, **free Kotlin interop** via the same JAR.

## Install

### Maven

```xml
<dependency>
  <groupId>fyi.oxide</groupId>
  <artifactId>pdf-oxide</artifactId>
  <version>0.3.53</version>
</dependency>
```

### Gradle

```kotlin
// Kotlin DSL
implementation("fyi.oxide:pdf-oxide:0.3.53")
```

```groovy
// Groovy
implementation 'fyi.oxide:pdf-oxide:0.3.53'
```

The JAR embeds native libraries for **linux x86_64**, **linux aarch64**, **macOS x86_64**, **macOS aarch64**, and **windows x86_64**. The right one is extracted to a UUID-suffixed temp file on first call via `NativeLoader` (snappy-java pattern — multi-classloader safe).

## Quick start

```java
import fyi.oxide.pdf.PdfDocument;
import fyi.oxide.pdf.AutoExtractor;
import fyi.oxide.pdf.Pdf;
import fyi.oxide.pdf.MarkdownConverter;

// Open + extract text
try (PdfDocument doc = PdfDocument.open(Path.of("report.pdf"))) {
    System.out.println("pages: " + doc.pageCount());
    System.out.println(doc.extractText(0));
}

// Convert to Markdown
try (PdfDocument doc = PdfDocument.open(Path.of("report.pdf"))) {
    String md = MarkdownConverter.toMarkdown(doc);
    Files.writeString(Path.of("report.md"), md);
}

// Smart text routing — picks text-layer or OCR per page automatically
try (PdfDocument doc = PdfDocument.open(Path.of("mixed.pdf"))) {
    AutoExtractor extractor = AutoExtractor.balanced(doc);
    String text = extractor.extractText();
}

// Markdown → PDF
try (Pdf pdf = Pdf.fromMarkdown("# Hello\n\nWorld")) {
    pdf.saveTo(Path.of("out.pdf"));
}
```

## Surface

All v0.3.52 features available in Java:

- **`PdfDocument`** — open, authenticate, extractText (page or auto), render PNG, formFields, search, producer/creator, toMarkdown/toHtml convenience
- **`PdfPage`** — words, lines, chars, images, tables, annotations, text(BBox region)
- **`DocumentEditor`** — setFormField, addRedaction, applyRedactionsDestructive (v0.3.50 #231), scrubMetadata, save
- **`Pdf`** — fromMarkdown, fromHtml, fromImages, split-by-bookmarks (v0.3.50 #482)
- **`MarkdownConverter`** — toMarkdown/toHtml × {whole-doc, per-page}
- **`AutoExtractor`** (v0.3.51 #517) — classifyPageKind, classifyDocumentKinds, extractText, extractAutoPage with simplified `AutoResult`, plus `extractPageJson` / `extractDocumentJson` escape hatch for the full v0.3.51 rich shape (typed reasons + per-region bboxes + confidence)
- **`PdfSigner`** (v0.3.50 #235) — fromPkcs12, sign with PAdES B-B / B-T / B-LT (TSA over RFC 3161 HTTP), verify, classifyLevel
- **`PdfValidator`** — PDF/A and PDF/UA verdict
- **`PdfPolicy`** (v0.3.50 #230) — crypto-governance set-once policy

## Exception model

`PdfException extends RuntimeException` (unchecked, per Effective Java Item 71) + 8 typed subclasses (`PdfParseException`, `PdfEncryptedException`, `PdfPermissionException`, `PdfIoException`, `PdfOcrUnavailableException`, `PdfSignatureException`, `PdfInvalidStateException`, `PdfUnsupportedException`) + a `PdfErrorKind` enum for switch-on-enum dispatch.

```java
try (PdfDocument doc = PdfDocument.open(Path.of("encrypted.pdf"))) {
    // ...
} catch (PdfEncryptedException e) {
    // Use PdfDocument.openWithPassword(path, password) instead
} catch (PdfException e) {
    switch (e.kind()) {
        case PARSE -> log.warn("malformed PDF");
        case IO    -> log.warn("io error");
        default    -> log.error("pdf error", e);
    }
}
```

## Lifecycle

`PdfDocument`, `Pdf`, and `DocumentEditor` are `AutoCloseable` with **idempotent close**:

- Calling `close()` twice is safe (no double-free).
- `AtomicLong`-shared state coordinates concurrent close so callers can call `close()` safely from any thread.
- {@link PdfDocument} additionally registers a `Cleaner` backstop that frees the native handle if you forget `close()`. **`Pdf` and `DocumentEditor` do not** — always wrap them in try-with-resources or call `close()` explicitly, or the native handle leaks for the lifetime of the JVM.

```java
try (PdfDocument doc = PdfDocument.open(file)) {
    // ... handle freed at end of try-with-resources
}
```

## System properties (advanced)

| Property | Default | Purpose |
|---|---|---|
| `fyi.oxide.pdf.lib.path` | unset | Path to a pre-extracted native library (skip JAR extraction) |
| `fyi.oxide.pdf.use.systemlib` | `false` | Use `System.loadLibrary("pdf_oxide_jni")` from `java.library.path` |
| `fyi.oxide.pdf.tempdir` | `java.io.tmpdir` | Override the temp directory for native extraction (useful for read-only `/tmp` deployments) |

## Kotlin

The JAR works directly from Kotlin — no extra adapter artifact needed. All value types use record-shaped accessors (`bbox.x()`, `bbox.y()`) which become Kotlin properties (`bbox.x`, `bbox.y`).

```kotlin
import fyi.oxide.pdf.PdfDocument

PdfDocument.open(Path.of("report.pdf")).use { doc ->
    println("pages: ${doc.pageCount}")
    println(doc.extractText(0))
}
```

A future companion artifact will add Kotlin extension functions for idiomatic flow / coroutine APIs.

## FIPS 140-3

For FIPS-validated deployments, build `pdf_oxide_jni` with `--no-default-features --features fips,signatures` (excludes MD5/RC4 legacy-crypto). See [FIPS guide](../docs/FIPS_GUIDE.md).

## License

MIT OR Apache-2.0 — same as the rest of pdf_oxide. Free for commercial use, no attribution required (though appreciated).
