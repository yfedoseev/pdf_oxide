# Changelog

All notable changes to PDFOxide are documented here.

## [0.2.5] - 2026-01-09

### Added
- **Image embedding**: Both HTML and Markdown now support embedded base64 images when `embed_images=true` (default)
  - HTML: `<img src="data:image/png;base64,...">`
  - Markdown: `![alt](data:image/png;base64,...)` (works in Obsidian, Typora, VS Code, Jupyter)
- **Image file export**: Set `embed_images=false` + `image_output_dir` to save images as files with relative path references
- New `embed_images` option in `ConversionOptions` to control embedding behavior
- `PdfImage::to_base64_data_uri()` method for converting images to data URIs
- `PdfImage::to_png_bytes()` method for in-memory PNG encoding
- Python bindings: new `embed_images` parameter for `to_html`, `to_markdown`, and `*_all` methods

## [0.2.4] - 2026-01-09

### Fixed
- CTM (Current Transformation Matrix) now correctly applied to text positions per PDF Spec ISO 32000-1:2008 Section 9.4.4 (#11)

### Added
- Structure tree: `/Alt` (alternate description) parsing for accessibility text on formulas and figures
- Structure tree: `/Pg` (page reference) resolution - correctly maps structure elements to page numbers
- `FormulaRenderer` module for extracting formula regions as base64 images from rendered pages
- `ConversionOptions`: new fields `render_formulas`, `page_images`, `page_dimensions` for formula image embedding
- Regression tests for CTM transformation

## [0.2.3] - 2026-01-07

### Fixed
- BT/ET matrix reset per PDF spec Section 9.4.1 (PR #10 by @drahnr)
- Geometric spacing detection in markdown converter (#5)
- Verbose extractor logs changed from info to trace (#7)
- docs.rs build failure (excluded tesseract-rs)

### Added
- `apply_intelligent_text_processing()` method for ligature expansion, hyphenation reconstruction, and OCR cleanup (#6)

### Changed
- Removed unused tesseract-rs dependency

## [0.2.2] - 2025-12-15

### Changed
- Optimized crate keywords for better discoverability

## [0.2.1] - 2025-12-15

### Fixed
- Encrypted stream decoding improvements (#3)
- CI/CD pipeline fixes

## [0.1.4] - 2025-12-12

### Fixed
- Encrypted stream decoding (#2)
- Documentation and doctest fixes

## [0.1.3] - 2025-12-12

### Fixed
- Encrypted stream decoding refinements

## [0.1.2] - 2025-11-27

### Added
- Python 3.13 support
- GitHub sponsor configuration

## [0.1.1] - 2025-11-26

### Added
- Cross-platform binary builds (Linux, macOS, Windows)

## [0.1.0] - 2025-11-06

### Added
- Initial release
- PDF text extraction with spec-compliant Unicode mapping
- Intelligent reading order detection
- Python bindings via PyO3
- Support for encrypted PDFs
- Form field extraction
- Image extraction
