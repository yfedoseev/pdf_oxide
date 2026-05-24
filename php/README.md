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

- **PHP 8.2+** — supported versions: 8.2, 8.3, 8.4, 8.5 (CI matrix).
- **`ext-ffi` enabled.** Confirm with `php -m | grep -i ffi`. Some
  managed PHP hosts (shared cPanel, Plesk) disable `ext-ffi` at the
  `php.ini` level for security reasons; use a Docker image such as
  `php:8.3-cli` if unsure.
- **`ext-mbstring`** (almost always already enabled).
- A platform with one of the five published native binaries above. If
  you're on a different platform, build `libpdf_oxide` from source
  (`cargo build --release --lib` against the root crate) and point
  `PDF_OXIDE_CDYLIB_PATH` at it.

## API shape

The PHP binding mirrors the Java binding's surface — **9 classes**, all
under the `PdfOxide\` namespace:

| Class                       | Purpose                                                      |
| --------------------------- | ------------------------------------------------------------ |
| `PdfOxide\PdfDocument`      | open/read/extract/page-iterate                               |
| `PdfOxide\Pdf`              | create PDFs (markdown→PDF, html→PDF), `version()`, prefetch  |
| `PdfOxide\PdfPage`          | per-page lightweight view                                    |
| `PdfOxide\MarkdownConverter`| static markdown/HTML conversion                              |
| `PdfOxide\AutoExtractor`    | v0.3.51 typed-reason extraction + classification             |
| `PdfOxide\AutoExtractResult`| readonly result value-object for `AutoExtractor`             |
| `PdfOxide\DocumentEditor`   | edit / form-fill / destructive redaction / save              |
| `PdfOxide\PdfSigner`        | PAdES B-B / B-T / B-LT / B-LTA signing                       |
| `PdfOxide\PdfValidator`     | PDF/A and PDF/UA compliance checks                           |
| `PdfOxide\PdfPolicy`        | set-once process-global crypto-governance policy             |

The FFI / exception infrastructure lives under `PdfOxide\FFI\…` and
`PdfOxide\Exceptions\…`.

## Quickstart

### 1. Open a PDF and read pages

```php
use PdfOxide\PdfDocument;

$doc = PdfDocument::open('report.pdf');
echo $doc->pageCount(), " pages\n";

// Extract plain text from page 0:
echo $doc->extractText(0);

// Whole-document Markdown:
echo $doc->toMarkdownAll();

$doc->close();  // or rely on __destruct()
```

### 2. Auto-extraction with typed reasons (v0.3.51 #517 / #519)

```php
use PdfOxide\AutoExtractor;
use PdfOxide\AutoExtractResult;
use PdfOxide\PdfDocument;

$doc = PdfDocument::open('mixed.pdf');
$ex  = AutoExtractor::of($doc);

$result = $ex->extractAutoPage(0);
echo $result->text;
if (!$result->isOk()) {
    error_log("[pdf_oxide] degraded extraction: {$result->reason}");
}

$doc->close();
```

### 3. Create a PDF from Markdown

```php
use PdfOxide\Pdf;

$pdf = Pdf::fromMarkdown("# Invoice\n\n**Total:** $42.00\n");
file_put_contents('invoice.pdf', $pdf->save());
$pdf->close();
```

### 4. Destructive redaction (security operation — fails closed)

```php
use PdfOxide\DocumentEditor;

$editor = DocumentEditor::open('in.pdf');
$editor->addRedaction(0, 100.0, 700.0, 300.0, 720.0);  // x1,y1,x2,y2 in points
$editor->applyRedactionsDestructive();
$editor->saveTo('redacted.pdf');
$editor->close();
```

### 5. PAdES B-T signature

```php
use PdfOxide\PdfSigner;

$signer = PdfSigner::fromPkcs12('certs/sign.p12', 'p12-password');
$signed = $signer->sign(
    pdfBytes: file_get_contents('contract.pdf'),
    level:    PdfSigner::LEVEL_B_T,
    tsaUrl:   'https://freetsa.org/tsr',
    reason:   'Final contract',
);
file_put_contents('signed.pdf', $signed);
$signer->close();
```

### 6. Crypto-governance policy (set-once)

```php
use PdfOxide\PdfPolicy;

// MUST run before opening any PDF.
PdfPolicy::set(PdfPolicy::STRICT);
echo PdfPolicy::current();   // → "strict"
```

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
