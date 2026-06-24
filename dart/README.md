# pdf_oxide — Dart / Flutter bindings

Idiomatic Dart bindings over the pdf_oxide C ABI via `dart:ffi`. The native
library (`libpdf_oxide.{so,dylib,dll}`) is loaded at runtime; handles are freed
by a `NativeFinalizer` (and explicit `close()`); C strings/buffers are copied
into Dart and freed for you; C-ABI error codes are thrown as `PdfOxideError`.

## Install

Published to [pub.dev](https://pub.dev/) as `pdf_oxide`:

```bash
dart pub add pdf_oxide
```

```yaml
# or in pubspec.yaml
dependencies:
  pdf_oxide: ^0.3.69
```

The native `libpdf_oxide.{so,dylib,dll}` is loaded at runtime and is **not**
bundled — see "Setup" for building it and pointing the loader at it.

## Setup

The binding links the **default-feature cdylib** (not the Python wheel):

```bash
# 1. build the native library (shipped binding feature set)
cargo build --release --lib --features ocr,rendering,signatures,barcodes,tsa-client,system-fonts

# 2. install deps + run tests (point the loader at the cdylib)
cd dart
dart pub get
PDF_OXIDE_LIB_DIR="$PWD/../target/release" dart test
PDF_OXIDE_LIB_DIR="$PWD/../target/release" dart run example/basic_extraction.dart
```

Native library resolution order: `PDF_OXIDE_LIB_PATH` (full path) →
`PDF_OXIDE_LIB_DIR` → `../target/release` → `target/release` → system loader.
For Flutter, ship the platform library and set `PDF_OXIDE_LIB_PATH`.

## Use

```dart
import 'package:pdf_oxide/pdf_oxide.dart';

void main() {
  final pdf = Pdf.fromMarkdown('# Hello\n\nbody\n');
  final doc = PdfDocument.openFromBytes(pdf.toBytes());
  try {
    print('pages: ${doc.pageCount}');
    print(doc.extractText(0));
    print(doc.toMarkdownAll());
  } finally {
    doc.close();
    pdf.close();
  }
}
```

## Layout

```
dart/
  lib/pdf_oxide.dart            dart:ffi wrapper (PdfDocument, Pdf, PdfOxideError)
  example/basic_extraction.dart runnable example (asserted in CI)
  test/api_coverage_test.dart   one test per public method
  pubspec.yaml
```

## Verification (CI — same set as every binding)

`.github/workflows/dart.yml` on Linux + macOS: build cdylib → `dart pub get` →
`dart analyze` → `dart format` check → `dart test` (api-coverage) → run example
with an output assertion.
