# PDF Oxide — Ruby Bindings

Idiomatic Ruby bindings for [PDF Oxide](https://github.com/fyi-oxide/pdf_oxide),
the same `libpdf_oxide` cdylib used by the Python, Java, Node, Go, and C#
bindings.

Status: **v0.3.55** — Phase 2 (repair).  The gem loads cleanly against the
v0.3.55 cdylib and exposes the read-side PDF API (document, pages, text +
markdown + HTML extraction, search, metadata, outline, layer, rendering,
annotations, forms, OCR, compliance, signatures, barcodes, analysis) plus a
real `PdfOxide::Creator` backed by the `pdf_from_markdown` / `pdf_from_html` /
`pdf_from_text` cdylib factories.

Phase 3 (extend) will land auto-extraction, office converter, destructive
redaction, PAdES B-T/B-LT/B-LTA, watermarks, bookmark split, compression,
JSON-shaped analysis, models prefetch, OCR-by-default fallback, owner-password
management, and the region-based detection APIs.

## Installation

Add to your Gemfile:

```ruby
gem 'pdf_oxide', '~> 0.3.55'
```

Then run:

```bash
bundle install
```

Or install directly:

```bash
gem install pdf_oxide
```

The native `libpdf_oxide.{so,dylib,dll}` must be discoverable on the loader
path.  Phase 4 will ship platform-specific gems that bundle a prebuilt binary;
until then, build the cdylib locally with `cargo build --release` from the
project root and either install it system-wide or point
`LD_LIBRARY_PATH` (Linux) / `DYLD_LIBRARY_PATH` (macOS) / `PATH` (Windows) at
`target/release/`.

## Quick Start

### Opening a PDF

```ruby
require 'pdf_oxide'

# With automatic resource cleanup
PdfOxide::Document.open('document.pdf') do |doc|
  puts "Pages: #{doc.page_count}"
  puts "Version: #{doc.version}"
end
```

### Text Search

```ruby
PdfOxide::Document.open('document.pdf') do |doc|
  # Search on specific page
  results = doc.search.search_page('Ruby', 0)
  results.each do |result|
    puts "Found: #{result.text}"
    puts "Location: page #{result.page_number}, bbox: #{result.bbox}"
  end

  # Search all pages
  all_results = doc.search.search_all('Ruby', case_sensitive: false)
  puts "Total matches: #{all_results.count}"
end
```

### Text Extraction

```ruby
PdfOxide::Document.open('document.pdf') do |doc|
  # Extract plain text
  text = doc.extraction.extract_text(0)
  puts text

  # Extract as Markdown
  markdown = doc.extraction.extract_to_markdown(0)
  File.write('output.md', markdown)

  # Extract as HTML
  html = doc.extraction.extract_to_html(0)
  File.write('output.html', html)
end
```

### Page Rendering

```ruby
PdfOxide::Document.open('document.pdf') do |doc|
  # Render with default settings
  doc.rendering.render_page_to_file(0, 'page.png')

  # Render with custom options
  options = PdfOxide::Types::RenderOptions.new(dpi: 300, quality: 95)
  doc.rendering.render_page_to_file(0, 'high-quality.png', options)

  # Render all pages
  doc.rendering.render_all('output_dir')

  # Get thumbnail
  thumb_bytes = doc.rendering.render_thumbnail(0, max_size: 200)
  File.write('thumb.png', thumb_bytes)
end
```

### Annotations

```ruby
PdfOxide::Document.open('document.pdf') do |doc|
  # List annotations
  annotations = doc.annotations.list_annotations(0)
  annotations.each do |annotation|
    puts "Type: #{annotation.type_name}"
    puts "Text: #{annotation.text}"
  end

  # Add annotations
  doc.annotations.add_highlight(0, 10, 20, 100, 30, color: 0xFFFF00)
  doc.annotations.add_comment(0, 50, 50, 'Great point!')
end
```

### Metadata

```ruby
PdfOxide::Document.open('document.pdf') do |doc|
  # Get metadata
  puts "Title: #{doc.metadata.title}"
  puts "Author: #{doc.metadata.author}"
  puts "Created: #{doc.metadata.creation_date}"

  # Set metadata
  doc.metadata.set_title('New Title')
  doc.metadata.set_author('John Doe')
end
```

### OCR (Optical Character Recognition)

```ruby
PdfOxide::Document.open('scanned.pdf') do |doc|
  # Check if page needs OCR
  needs_ocr = doc.ocr.page_is_scanned?(0)

  if needs_ocr
    # Apply OCR to page
    text = doc.ocr.ocr_page(0)
    puts text
  end

  # Apply OCR to entire document
  doc.ocr.apply_ocr_to_document
end
```

### Digital Signatures

```ruby
PdfOxide::Document.open('document.pdf') do |doc|
  # Check signatures
  sig_count = doc.signatures.signature_count
  if sig_count > 0
    is_valid = doc.signatures.verify_signature(0)
    puts "Signature valid: #{is_valid}"
  end

  # Sign document
  doc.signatures.sign_document('cert.pfx', 'password')
end
```

### PDF Compliance

```ruby
PdfOxide::Document.open('document.pdf') do |doc|
  # Validate PDF/A compliance
  is_pdf_a = doc.compliance.validate_pdf_a('level_1b')

  # Convert to PDF/A
  doc.compliance.convert_to_pdf_a('level_3b', 'output.pdf')

  # Validate PDF/UA accessibility
  is_accessible = doc.compliance.validate_pdf_ua('level_1')
end
```

### Creating PDFs

```ruby
require 'pdf_oxide'

# Create from Markdown
creator = PdfOxide::Creator.from_markdown(<<~MD
  # Hello PDF Oxide

  This is a PDF created from Ruby!
MD
)
creator.save('output.pdf')

# Create with builder pattern
creator = PdfOxide::Creator.new
  .title('My Document')
  .author('John Doe')
  .add_blank_page(612, 792)
  .merge('another.pdf')
  .save('combined.pdf')
```

## API Reference

### Core Classes

#### `PdfOxide::Document`

Main interface for reading and analyzing PDFs.

```ruby
# Open document
doc = PdfOxide::Document.open('file.pdf')
doc = PdfOxide::Document.open('file.pdf') { |d| ... }

# Properties
doc.page_count       # => Integer
doc.version          # => String
doc.encrypted?       # => Boolean
doc.closed?          # => Boolean
doc.path             # => String

# Managers (lazy-initialized)
doc.search           # => Managers::Search
doc.rendering        # => Managers::Rendering
doc.annotations      # => Managers::Annotation
doc.forms            # => Managers::Form
doc.pages            # => Managers::Page
doc.metadata         # => Managers::Metadata
doc.outline          # => Managers::Outline
doc.layers           # => Managers::Layer
doc.cache            # => Managers::Cache
doc.extraction       # => Managers::Extraction
doc.ocr              # => Managers::Ocr
doc.compliance       # => Managers::Compliance
doc.signatures       # => Managers::Signature
doc.barcodes         # => Managers::Barcode
doc.analysis         # => Managers::Analysis

# Resource management
doc.close
```

#### `PdfOxide::Creator`

Interface for creating and modifying PDFs.

```ruby
# Create from templates
creator = PdfOxide::Creator.from_markdown(markdown_text)
creator = PdfOxide::Creator.from_html(html_text)
creator = PdfOxide::Creator.from_text(plain_text)

# Builder methods
creator
  .add_blank_page(width, height)
  .add_page_from_template('template.pdf', page_index)
  .add_page_from_document('other.pdf', page_index)
  .merge('another.pdf')
  .title('Document Title')
  .author('Author Name')
  .subject('Subject')
  .keywords('keyword1, keyword2')
  .creator('Application')
  .save('output.pdf')

# Get bytes
bytes = creator.to_bytes
```

### Data Types

All data types are immutable and support conversion to/from hashes:

```ruby
# Bounding boxes
bbox = PdfOxide::Types::BoundingBox.new(x: 10, y: 20, width: 100, height: 50)
bbox.right                # => 110
bbox.bottom               # => 70
bbox.area                 # => 5000
bbox.contains_point?(50, 45)
bbox.overlaps_with?(other_bbox)
bbox.to_h

# Page dimensions
dims = PdfOxide::Types::PageDimensions.new(width: 612, height: 792, unit: 'pt')
dims.to_inches
dims.to_millimeters
dims.landscape?           # => Boolean
dims.portrait?            # => Boolean
dims.aspect_ratio         # => Float

# Render options
opts = PdfOxide::Types::RenderOptions.new(dpi: 300, quality: 95)
opts = PdfOxide::Types::RenderOptions.draft
opts = PdfOxide::Types::RenderOptions.high

# Search results
result.page               # => Integer (0-indexed)
result.page_number        # => Integer (1-indexed)
result.text               # => String
result.bbox               # => BoundingBox
result.context            # => String
```

## Error Handling

Comprehensive exception hierarchy for robust error handling:

```ruby
begin
  doc = PdfOxide::Document.open('file.pdf')
rescue PdfOxide::FileNotFoundError => e
  puts "File not found: #{e.message}"
rescue PdfOxide::ParseError => e
  puts "Invalid PDF: #{e.message}"
  puts "Details: #{e.details}"
rescue PdfOxide::PermissionError => e
  puts "Access denied: #{e.message}"
rescue PdfOxide::Error => e
  puts "General error: #{e.message}"
  puts "Code: #{e.code}"
end
```

## Performance Tips

- Use block syntax for automatic resource cleanup
- Lazy managers reduce initialization overhead
- Cache search/extraction results when possible
- Use appropriate render quality for your needs
- Enable GPU acceleration for OCR when available

## Testing

Run the test suite:

```bash
bundle exec rspec
```

Run with coverage:

```bash
COVERAGE=true bundle exec rspec
```

Run style checks:

```bash
bundle exec rubocop
```

## Building

```bash
rake build
```

## Requirements

- Ruby 2.7+
- FFI gem 1.16+
- libpdf_oxide native library

## License

Apache License 2.0

## Contributing

Contributions welcome! Please ensure tests pass and code follows style guidelines.

## Support

For issues and questions:
- GitHub Issues: https://github.com/fyi-oxide/pdf_oxide/issues
- Documentation: https://rubydoc.info/gems/pdf_oxide
