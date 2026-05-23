# pdf_oxide (PHP)

PHP binding for [pdf_oxide](https://github.com/fyi-oxide/pdf_oxide) — a
Rust-backed PDF processing toolkit. This package is pure PHP code on top
of PHP's built-in FFI extension; the heavy lifting happens in the same
`libpdf_oxide` cdylib used by the Python, Node, Go, C#, Ruby, and Java
bindings.

## Installation

```bash
composer require oxide/pdf-oxide
```

Composer's post-install hook (`scripts/download-native-lib.php`) downloads
the matching prebuilt `libpdf_oxide` from the GitHub Release tagged
`v0.3.55` into `vendor/oxide/pdf-oxide/lib/`. Five platforms ship:

- `linux-x86_64`
- `linux-aarch64`
- `darwin-x86_64`
- `darwin-arm64`
- `windows-x64`

Set `PDF_OXIDE_SKIP_DOWNLOAD=1` to skip the post-install download (CI /
offline / corp-proxy use case). Set `PDF_OXIDE_NATIVE_VERSION=vX.Y.Z` to
pin a specific release. The runtime library search order is:

1. The path in `PDF_OXIDE_CDYLIB_PATH` (env var override).
2. `vendor/oxide/pdf-oxide/lib/<platform>/libpdf_oxide.{so,dylib,dll}`
3. `/usr/local/lib/libpdf_oxide.{so,dylib}` (Linux/macOS fallback).

## Prerequisites

- **PHP 8.1+** (8.2 / 8.3 / 8.4 also fully supported).
- **`ext-ffi` enabled.** Confirm with `php -m | grep -i ffi`. Some
  managed PHP hosts (shared cPanel, Plesk) disable `ext-ffi` at the
  `php.ini` level for security reasons; consult your host or use a
  Docker image such as `php:8.3-cli` if unsure.
- **`ext-mbstring`** (almost always already enabled).
- A platform with one of the five published native binaries above. If
  you're on a different platform you can build `libpdf_oxide` yourself
  from source (`cargo build --release --lib` against the root crate)
  and point `PDF_OXIDE_CDYLIB_PATH` at it.

## Quickstart

### 1. Open a PDF and read pages

```php
use PdfOxide\PdfDocument;

$doc = new PdfDocument('report.pdf');
echo $doc->getPageCount(), " pages\n";

// Extract plain text from page 0:
echo $doc->extractText(0);

// Or Markdown for the whole document:
echo $doc->toMarkdownAll();
```

### 2. Auto-extraction with typed reasons

```php
use PdfOxide\PdfDocument;
use PdfOxide\Enums\ExtractReason;

$doc    = new PdfDocument('mixed.pdf');
$result = $doc->auto()->extractText($doc, 0);

echo $result->text;
if ($result->reason !== ExtractReason::Ok) {
    error_log("[pdf_oxide] degraded extraction: " . $result->reason->value);
}
```

### 3. Destructive redaction (security operation — fails closed)

```php
use PdfOxide\Managers\RedactionManager;
use PdfOxide\Types\Rect;

$redact = RedactionManager::openFile('in.pdf');
$redact->mark(0, new Rect(100, 700, 200, 20));   // page 0, x,y,w,h in PDF points
$redact->apply(scrubMetadata: true);
```

### 4. PAdES B-T signature

```php
use PdfOxide\PdfDocument;
use PdfOxide\Enums\PadesLevel;

$doc  = new PdfDocument('contract.pdf');
$sigs = $doc->signatures();
$signed = $sigs->signPades(
    pdfData:           file_get_contents('contract.pdf'),
    certificateHandle: $certHandle,                        // load via signatures()->loadPkcs12()
    level:             PadesLevel::BT,
    tsaUrl:            'https://freetsa.org/tsr',
    reason:            'Final contract',
);
file_put_contents('signed.pdf', $signed);
```

### 5. Office to PDF + PDF to DOCX

```php
use PdfOxide\PdfDocument;
use PdfOxide\Managers\OfficeConverter;

// DOCX bytes -> PdfDocument
$pdfDoc = PdfDocument::fromDocxBytes(file_get_contents('memo.docx'));

// PdfDocument -> DOCX bytes
$docx = (new OfficeConverter($pdfDoc->getHandle()))->toDocx();
file_put_contents('memo-pages.docx', $docx);
```

## Capability surface

PHP parity with Python / Java for the v0.3.54 FFI surface:

- Reading: text / Markdown / HTML / plain extraction, search,
  metadata, fonts, images, annotations, outlines.
- Auto-extraction (`AutoExtractor`) with the typed `ExtractReason`
  enum (frozen wire format).
- Region extraction (text / words / lines / tables / images in a
  rect).
- Forms: AcroForm + XFA, FDF/XFDF bytes import/export.
- Outline / bookmarks (incl. split-by-bookmarks planning).
- Page editing: rotate / delete / move / merge, header/footer
  erasure, artifact removal.
- Destructive redaction (`RedactionManager`) + metadata scrub —
  security-op semantics (fail-closed).
- PAdES B-B / B-T / B-LT / B-LTA signatures via the v0.3.51 5-arg
  options shim.
- Office: DOCX / PPTX / XLSX <-> PDF round-trip.
- Watermarks, stamps, freetext annotations.
- OCR (manual + by-default fallback).
- Barcodes (generate + detect).
- PDF/A, PDF/UA, PDF/X compliance (validate + convert).
- Models subsystem (`prefetch_models`, `model_manifest`,
  `prefetch_available`).

## Testing

```bash
composer test                  # full suite
composer test:unit             # unit suite only (no cdylib required)
composer test:integration      # integration suite (cdylib required)
composer lint                  # php -l on every PHP file
```

The integration suite reads `PDF_OXIDE_CDYLIB_PATH` if set; otherwise it
falls back to `target/release/libpdf_oxide.{so,dylib,dll}` relative to
the repo root. Tests self-skip when the cdylib isn't reachable so the
unit suite still runs on any box.

## Links

- Root project: https://github.com/fyi-oxide/pdf_oxide
- Rust source: https://github.com/fyi-oxide/pdf_oxide/tree/main/src
- Packagist: https://packagist.org/packages/oxide/pdf-oxide
- Other bindings: Python (`pip install pdf_oxide`), Node
  (`npm i pdf_oxide`), Ruby (`gem install pdf_oxide`), Go (`go get
  github.com/fyi-oxide/pdf-oxide`), C# (`dotnet add package
  PdfOxide`), Java (`fyi.oxide:pdf-oxide` on Maven Central).

## License

Dual-licensed under MIT or Apache-2.0 at your option, matching the root
project.
