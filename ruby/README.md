# PDF Oxide — Ruby Bindings

Idiomatic Ruby bindings for [PDF Oxide](https://github.com/fyi-oxide/pdf_oxide),
the same `libpdf_oxide` cdylib that powers the Python, Java, Node, Go, C#,
and PHP bindings.

Status: **v0.3.55** — production gem.  The full v0.3.50–v0.3.54 feature
surface (auto-extraction with typed reasons, PAdES B/T/LT/LTA signing,
destructive redaction with metadata scrub, PDF/A · PDF/UA validation,
markdown / html → PDF) is reachable through 9 idiomatic Ruby classes
mirroring the Java binding shape at `fyi.oxide.pdf.*`.

## Installation

```bash
gem install pdf_oxide
```

Or in a Gemfile:

```ruby
gem 'pdf_oxide', '~> 0.3.55'
```

RubyGems picks the most-specific platform-tagged gem for your runtime; the
prebuilt `libpdf_oxide.{so,dylib,dll}` ships inside the gem at
`ext/pdf_oxide/`.  No system-wide install of the native library is needed.

**Supported platforms** (each Ruby 3.1, 3.2, 3.3, 3.4):

- `x86_64-linux`
- `aarch64-linux`
- `x86_64-darwin` (Intel Mac)
- `arm64-darwin` (Apple Silicon)
- `x64-mingw-ucrt` (Windows, Ruby 3.1+)

## Public API

| Class                            | Role                                                       |
| -------------------------------- | ---------------------------------------------------------- |
| `PdfOxide::PdfDocument`          | open / extract / search / render / metadata                |
| `PdfOxide::PdfPage`              | lightweight per-page view                                  |
| `PdfOxide::Pdf`                  | create + transform (markdown / html / text → PDF)          |
| `PdfOxide::DocumentEditor`       | form-fill, destructive redaction, save                     |
| `PdfOxide::AutoExtractor`        | typed-reason auto-extraction (v0.3.51 #519)                |
| `PdfOxide::MarkdownConverter`    | PDF → Markdown / HTML                                      |
| `PdfOxide::PdfValidator`         | PDF/A · PDF/UA compliance                                  |
| `PdfOxide::PdfSigner`            | PAdES B-B / B-T / B-LT / B-LTA signing                     |
| `PdfOxide::PdfPolicy`            | process-global crypto-governance                           |

## Quickstart

### 1. Open + extract text

```ruby
require 'pdf_oxide'

PdfOxide::PdfDocument.open('invoice.pdf') do |doc|
  puts doc.page_count
  puts doc.extract_text(0)
end
# Block form auto-closes; explicit `doc.close` also works (idempotent).
```

### 2. Markdown → PDF

```ruby
PdfOxide::Pdf.from_markdown("# Hello\n\nworld.") do |pdf|
  pdf.save('hello.pdf')
end
```

### 3. Auto-extraction with typed reasons

```ruby
PdfOxide::PdfDocument.open('scan.pdf') do |doc|
  result = doc.auto_extractor.extract_page(0)
  puts result[:text]
  warn "degraded: #{result[:reason]}" unless result[:reason] == :ok
  # When OCR is needed but the build lacks the `ocr` feature, the reason
  # is :ocr_requested_but_unavailable — extraction NEVER raises an
  # "OCR unavailable" error (graceful-fallback contract).
end
```

### 4. Destructive redaction (v0.3.50 #231)

```ruby
PdfOxide::DocumentEditor.open('source.pdf') do |ed|
  ed.add_redaction(page: 0, rect: [100, 200, 300, 250])
  ed.apply_redactions!(scrub_metadata: true)
  ed.save_to('redacted.pdf')
end
# Security op: any non-zero return from the cdylib fails closed —
# no silent under-redaction.
```

### 5. PAdES signing

```ruby
# certificate_handle comes from your credentials API (PKCS#12 / PEM)
signer = PdfOxide::PdfSigner.new(certificate_handle)
signed_bytes = signer.sign(
  File.binread('source.pdf'),
  level:    :t,                              # :b, :t, :lt, :lta
  tsa_url:  'http://timestamp.example.com',  # required for >= :t
  reason:   'Approved',
  location: 'Berlin, DE'
)
File.binwrite('signed.pdf', signed_bytes)
```

## Cross-binding parity

The Ruby surface mirrors the Java binding's 9-class shape one-for-one;
the underlying `libpdf_oxide` cdylib is the same C ABI exercised by the
Python, PHP, Node, Go, C#, Java, and WASM bindings.  Bug fixes and new
features in the upstream cdylib reach Ruby on the next gem release.

## License

Dual-licensed Apache-2.0 OR MIT.  See `LICENSE`.
