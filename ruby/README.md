# PDF Oxide — Ruby Bindings

Idiomatic Ruby bindings for [PDF Oxide](https://github.com/fyi-oxide/pdf_oxide),
the same `libpdf_oxide` cdylib that powers the Python, Java, Node, Go, C#,
and PHP bindings.

Status: **v0.3.55** — production gem.  The full v0.3.50–v0.3.54 feature
surface (auto-extraction with reason taxonomy, Office→PDF conversion, PAdES
B-T / B-LT / B-LTA signing, destructive redaction, bookmark-split, OCR-by-
default fallback, models prefetch) is reachable through Ruby.

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
- `x64-mingw32` (Windows)

**Source-gem fallback**: if no platform variant matches, `gem install`
selects the plain `pdf_oxide-0.3.55.gem` which expects a host `cargo` /
`rustc` to build `libpdf_oxide` from the project's `Cargo.toml` at install
time.  Requires Rust 1.85+ on PATH.

## Prerequisites

- Ruby >= 3.1
- `ffi` ~> 1.16 (resolved automatically by Bundler/RubyGems)
- Source-gem path only: Rust 1.85+ (`rustup default stable`)

## Quickstart

### 1. Open + extract text

```ruby
require 'pdf_oxide'

PdfOxide::Document.open('input.pdf') do |doc|
  puts "Pages: #{doc.page_count}"
  puts "Page 0 text:"
  puts doc.extraction.extract_text(0)
end
```

### 2. Render a thumbnail

```ruby
require 'pdf_oxide'

PdfOxide::Document.open('input.pdf') do |doc|
  thumb = doc.rendering.render_thumbnail(0, max_size: 256)
  File.binwrite('thumb.png', thumb)
end
```

### 3. Sign with PAdES B-T

```ruby
require 'pdf_oxide'

credentials = PdfOxide::Types::SigningCredentials.from_pkcs12(
  'signer.p12',
  ENV.fetch('PFX_PASSWORD')
)

signer = PdfOxide::PadesSigner.new('input.pdf')
signed_bytes = signer.sign_b_t(
  credentials: credentials,
  tsa_url:     'https://freetsa.org/tsr',
  reason:      'Approval',
  location:    'Remote'
)
File.binwrite('signed.pdf', signed_bytes)
```

### 4. Destructive redaction

```ruby
require 'pdf_oxide'

redactor = PdfOxide::RedactionManager.new('input.pdf')
redactor.add_redaction(page: 0, rect: [50.0, 700.0, 250.0, 720.0])
redactor.add_redaction(page: 1, rect: [60.0, 600.0, 300.0, 620.0])
redactor.apply!                              # in-place, destructive
redactor.scrub_metadata!                     # remove producer/author trails
redactor.save('redacted.pdf')
```

### 5. Auto-extract with OCR fallback

```ruby
require 'pdf_oxide'

result = PdfOxide::AutoExtractor.extract('scanned-or-digital.pdf')

case result.reason
when :digital_text
  puts "Took digital-text path (page text layer present)."
when :ocr_fallback
  puts "Page had no text; OCR ran and produced #{result.text.bytesize} bytes."
when :ocr_skipped_no_engine
  warn  'OCR engine not bundled in this build; returned page metadata only.'
end

puts result.text
```

The `result.reason` taxonomy ([`PdfOxide::ExtractReason`]) is the same one
Python, Java, Node, Go, and C# expose — v0.3.51 cross-binding parity.

## Surface map

- `PdfOxide::Document` — open, read, dispatch to managers
- `PdfOxide::Creator` — build PDFs from Markdown / HTML / plain text
- `PdfOxide::AutoExtractor` — graceful-fallback extraction (v0.3.51)
- `PdfOxide::OfficeConverter` — Word/Excel/PowerPoint → PDF (v0.3.52)
- `PdfOxide::RedactionManager` — content + metadata redaction (v0.3.50)
- `PdfOxide::PadesSigner` — PAdES B-T / B-LT / B-LTA (v0.3.50)
- `PdfOxide::Document#extraction|rendering|metadata|outline|search|...`
  — 22 manager classes covering the rest of the public C ABI.

## Project links

- Project root: https://github.com/fyi-oxide/pdf_oxide
- Rust source: https://github.com/fyi-oxide/pdf_oxide/tree/main/src
- Issue tracker: https://github.com/fyi-oxide/pdf_oxide/issues
- CHANGELOG: https://github.com/fyi-oxide/pdf_oxide/blob/main/CHANGELOG.md

## License

Apache-2.0 — see `LICENSE`.
