# Changelog

All notable changes to PDFOxide are documented here.

## [0.3.18] - 2026-04-01
> Rendering Engine Overhaul, Visual Parity, and Expanded API

### Rendering Engine — Visual Parity

Major rendering improvements achieving near-perfect visual fidelity across academic papers, government documents, CJK content, presentations, forms, and complex multi-layer PDFs.

#### Font Rendering
- **Correct Character Spacing** — Fixed proportional width resolution for CID, CFF, and TrueType subset fonts. Documents that previously rendered with monospace-like spacing now display with correct kerning and proportional widths.
- **Embedded Font Support** — Render directly from embedded CFF and TrueType font programs, producing accurate glyph shapes that match the original document's typography.
- **Standard Font Metrics** — Built-in width tables for the PDF standard 14 fonts (Times, Helvetica, Courier). Fixes uniform character spacing when explicit widths are absent.
- **Improved Font Matching** — Better system font fallback for URW, LaTeX, and other common font families. Automatic serif/sans-serif detection for appropriate substitution.

#### Operators & Path Rendering
- **Fill-and-Stroke Support** — Full implementation of combined fill-and-stroke operators (`B`, `B*`, `b`, `b*`), fixing missing border strokes on rectangles and paths.
- **Clip Path Support** — Proper handling of clip-without-paint patterns, resolving issues where body text was hidden behind unclipped background fills.
- **Gradient Shading** — Axial (linear) and radial gradient rendering with support for exponential interpolation and stitching functions.
- **Negative Rectangle Handling** — Correct normalization of rectangles with negative dimensions per the PDF specification.

#### Transparency & Compositing
- **Alpha Transparency** — Fixed fill and stroke alpha application per PDF specification. Semi-transparent rectangles, images, and paths now blend correctly.
- **Graphics State Resolution** — Proper indirect reference resolution for extended graphics state parameters, ensuring alpha and blend mode values are applied.
- **Isolated Transparency Groups** — Support for rendering transparency groups to separate compositing surfaces.

#### Image Rendering
- **Stencil Image Masks** — Support for 1-bit stencil masks with CCITT Group 4 decompression. Fixes decorative borders, corner ornaments, and masked image elements.

#### Page Handling
- **Page Rotation** — Full support for the `/Rotate` attribute (90°, 180°, 270°), correctly rendering landscape slides and rotated documents.

#### Color Space
- **Separation Color Spaces** — Proper tint transform evaluation for Separation and DeviceN colors against their alternate color spaces.

### Bug Fixes

- **Fixed process abort on degenerate CTM coordinates** — A malformed CTM could place text spans at extreme coordinates, causing allocation abort. Projection functions now safely skip the split instead of crashing.
- **FlateDecode flate-bomb protection** — All zlib/deflate decompression paths are now capped, preventing a crafted PDF stream from exhausting virtual memory. The cap defaults to 256 MB and can be adjusted via the `PDF_OXIDE_MAX_DECOMPRESS_MB` environment variable or programmatically with `FlateDecoder::with_limit(n)`.
- **Fixed Clipping Stack Synchronization** — Resolved a critical issue where the clipping stack could get out of sync with the graphics state, leading to incorrect content being hidden.
- **Standardized Image Extraction** — Refactored the image extraction logic to support document-wide color space resolution.
- **Fixed Python Rendering Accessibility** (#240) — Resolved an issue where the `render_page` method was unreachable in standard Python builds.

### Changed

- **Python type stubs** — Switched from mypy stubgen to [Rylai](https://github.com/monchin/Rylai) for generating `.pyi` from PyO3 Rust source statically (no compilation). CI and release workflows updated.

### API — Python

New methods on `PdfDocument`:
- `validate_pdf_a(level)` — PDF/A compliance validation (1a/1b/2a/2b/2u/3a/3b/3u)
- `validate_pdf_ua()` — PDF/UA accessibility validation
- `validate_pdf_x(level)` — PDF/X print compliance
- `extract_pages(pages, output)` — Extract page subset to a new PDF file
- `delete_page(index)` — Remove a page by index
- `move_page(from, to)` — Reorder pages
- `flatten_to_images(dpi)` — Create flattened PDF from rendered pages
- `PdfDocument(path, password=)` — Open encrypted PDFs in one step (#247)
- `PdfDocument.from_bytes(data, password=)` — Same for in-memory PDFs
- `Pdf.merge(paths)` — Merge multiple PDF files into one

### API — WASM / JavaScript

New methods on `WasmPdfDocument`:
- `validatePdfA(level)` — PDF/A compliance validation
- `deletePage(index)` — Remove a page
- `extractPages(pages)` — Extract pages to new PDF bytes
- `save()` — Save modified PDF (alias for `saveToBytes()`)
- `new WasmPdfDocument(data, password?)` — Open encrypted PDFs (#247)
- `WasmPdf.merge(pdfs)` — Merge multiple PDFs from byte arrays

### Core Rust API

- `rendering::flatten_to_images(doc, dpi)` — Shared implementation for all bindings
- `api::merge_pdfs(paths)` — Merge multiple PDFs (shared across all bindings)

### Features

- **Rendering Engine Overhaul** — Major improvements to the rendering pipeline, achieving high visual parity with industry standards.
- **Batteries-Included Python Bindings** — The Python distribution now automatically enables page rendering, parallel extraction, digital signatures, and office document conversion by default. (#240)

### 🏆 Community Contributors

🥇 **@tiennh-h2** — Thank you for reporting the rendering accessibility issue (#240). Your feedback helped us identify that our Python distribution was too minimal, leading to an improved "batteries-included" experience for all Python users! 🚀

🥈 **@Suleman-Elahi** — Thank you for the suggestion to add flattened PDF creation (#240). This led to the new `flatten_to_images()` API available across Rust, Python, and WASM! 🚀

🥉 **@hoesler** — Thank you for the XY-cut projection fix (#274) that prevents allocation abort on degenerate CTM coordinates, and the FlateDecoder configurability improvement (#275)! 🚀

🏅 **@Leon-Degel-Koehn** — Thank you for fixing the Quick Start Rust documentation (#277)! 🚀

🏅 **@XO9A8** — Thank you for improving the `PdfDocument::from_bytes` documentation (#276)! 🚀

🏅 **@monchin** — Thank you for replacing manual stub generation with Rylai (#250) and for helping diagnose the password API issue (#247) with a clear workaround and API improvement suggestion! 🚀

🏅 **@marph91** — Thank you for reporting the password constructor issue (#247), improving the developer experience for encrypted PDF workflows! 🚀

## [0.3.17] - 2026-03-08
> Stable Recursion and Refined Table Heuristics

### Features

- **Refined Table Detection** — The spatial table detector now requires at least **2 columns** to identify a region as a table. This significantly reduces false positives where single-column lists or bullet points were incorrectly wrapped in ASCII boxes.
- **Optimized Text Extraction** — Refactored the internal extraction pipeline to eliminate redundant work when processing Tagged PDFs. The structure tree and page spans are now extracted once and shared across the detection and rendering phases.

### Bug Fixes

- **Resolved `RefCell` already borrowed panic** (#237) — Fixed a critical reentrancy issue where recursive Form XObject processing (e.g., extracting images from nested forms) could trigger a runtime panic. Replaced long-lived borrows with scoped, tiered cache access using Rust best practices. (Reported by **@marph91**)

### 🏆 Community Contributors

🥇 **@marph91** — Thank you for identifying the complex `RefCell` borrow conflict in nested image extraction (#237). This report led to a comprehensive safety audit of our interior mutability patterns and a more robust, recursion-safe caching architecture! 🚀

## [0.3.16] - 2026-03-08
> Advanced Visual Table Detection and Automated Python Stubs

### Features

- **Smart Hybrid Table Extraction** (#206) — Introduced a robust, zero-config visual detection engine that handles both bordered and borderless tables.
    - **Localized Grid Detection:** Uses Union-Find clustering to group vector paths into discrete table regions, enabling multiple tables per page.
    - **Visual Line Analysis:** Detects cell boundaries from actual drawing primitives (lines and rectangles), significantly improving accuracy for untagged PDFs.
    - **Visual Spans:** Identifies colspans and rowspans by analyzing the absence of internal grid lines and text-overflow signals.
    - **Visual Headers:** Heuristically identifies hierarchical (multi-row) header rows.
- **Professional ASCII Tables:** Added high-quality ASCII table formatting for plain text output, featuring automatic multiline text wrapping and balanced column alignment.
- **Auto-generated Python type stubs** (#220) — Integrated automated `.pyi` stub generation using **mypy's stubgen** in the CI pipeline, ensuring Python IDEs always have up-to-date type information for the Rust bindings.
- **Python `PdfDocument` path-like and context manager** (#223) — `PdfDocument` now accepts `pathlib.Path` (or any path-like object) and supports the context manager protocol (`with PdfDocument(path) as doc:`), ensuring scoped usage and automatic resource cleanup.
- **Enabled by Default:** Table extraction is now active by default in all Markdown, HTML, and Plain Text conversions.
- **Robust Geometry:** Updated `Rect` primitive to handle negative dimensions and coordinate normalization natively.

### Bug Fixes

- **Fixed segfault in nested Form XObject text extraction** (#228) — Resolved aliased `&mut` references during recursive XObject processing using interior mutability (`RefCell`/`Cell`).
- **Fixed Python Coordinate Scaling:** Corrected `erase_region` coordinate mapping in Python bindings to use the standard `[x1, y1, x2, y2]` format.
- **Improved ASCII Table Wrapping:** Reworked text wrapping to be UTF-8 safe, preventing panics on multi-byte characters.
- **Refined Rendering API:** Restored backward compatibility for the `render_page` method.

### 🏆 Community Contributors

🥇 **@hoesler** — Huge thanks for PR #228! Your fix for the nested XObject aliasing UB is a critical stability improvement that eliminates segfaults in complex PDFs. By correctly employing interior mutability, you've made the core extraction engine significantly more robust and spec-compliant. Outstanding work! 🚀

🥈 **@monchin** — Thank you for the fantastic initiative on automated stub generation (#220) and the ergonomic improvements for Python (#223)! We've integrated these into the v0.3.16 release, providing consistent, IDE-friendly type hints and modern path-like/context manager support. Outstanding contributions! 🚀


## [0.3.15] - 2026-03-06
> Header & Footer Management, Multi-Column Stability, and Font Fixes

### Features

- **PDF Header/Footer Management API** (#207) — Added a dedicated API for managing page artifacts across Rust, Python, and WASM.
    - **Add:** Ability to insert custom headers and footers with styling and placeholders via `PageTemplate`.
    - **Remove:** Heuristic detection engine to automatically identify and strip repeating artifacts. Includes modular methods: `remove_headers()`, `remove_footers()`, and `remove_artifacts()`. Prioritizes ISO 32000 spec-compliant `/Artifact` tags when available.
    - **Edit:** Ability to mask or erase existing content on a per-page basis via `erase_header()`, `erase_footer()`, and `erase_artifacts()`.
- **Page Templates** — Introduced `PageTemplate`, `Artifact`, and `ArtifactStyle` classes for reusable page design. Supports dynamic placeholders like `{page}`, `{pages}`, `{title}`, and `{author}`.
- **Scoped Extraction Filtering** — Updated all extraction methods to respect `erase_regions`, enabling clean text extraction by excluding identified headers and footers.
- **Python `PdfDocument.from_bytes()`** — Open PDFs directly from in-memory bytes without requiring a file path. (Contributed by **@hoesler** in #216)
- **Future-Proofed Rust API** — Implemented `Default` trait for key extraction structs (`TextSpan`, `TextChar`, `TextContent`) to protect users from future field additions.

### Bug Fixes

- **Fixed Multi-Column Reading Order** (#211) — Refactored `extract_words()` and `extract_text_lines()` to use XY-Cut partitioning. This prevents text from adjacent columns from being interleaved and standardizes top-to-bottom extraction. (Reported by **@ankursri494**)
- **Resolved Font Identity Collisions** (#213) — Improved font identity hashing to include `ToUnicode` and `DescendantFonts` references. Fixes garbled text extraction in documents where multiple fonts share the same name but use different character mappings. (Reported by **@productdevbook**)
- **Fixed `Lines` table strategy false positives** (#215) — `extract_tables()` with `horizontal_strategy="lines"` now builds the grid purely from vector path geometry and returns empty when no lines are found, preventing spurious tables on plain-text pages. (Contributed by **@hoesler**)
- **Optimized CMap Parsing** — Standardized 2-byte consumption for Identity-H fonts and improved robust decoding for Turkish and other extended character sets.

### 🏆 Community Contributors

🥇 **@hoesler** — Huge thanks for PR #216 and #215! Your contribution of `from_bytes()` for Python unlocks new serverless and in-memory workflows for the entire community. Additionally, your fix for the `Lines` table strategy significantly improves the precision of our table extraction engine. Outstanding work! 🚀

🥈 **@ankursri494** (Ankur Srivastava) — Thank you for identifying the multi-column reading order issue (#211). Your detailed report and sample document were the catalyst for our new XY-Cut partitioning engine, which makes PDFOxide's reading order detection among the best in the ecosystem! 🎯

🥉 **@productdevbook** — Thanks for reporting the complex font identity collision issue (#213). This report led to a deep dive into PDF font internals and a significantly more robust font hashing system that fixes garbled text for thousands of professional documents! 🔍✨

## [0.3.14] - 2026-03-03
> Parity in API & Bug Fixing (Issue #185, #193, #202)

### Features

- **High-Level Rendering API** (#185, #190) — added `Pdf::render_page()` to Rust, Python, and WASM. Supports rendering any page to `Image` (Png/Jpeg). Restored backward compatibility for Rust by maintaining the 1-argument `render_page` and adding `render_page_with_options`.
- **Word and Line Extraction** (#185, #189) — added `extract_words()` and `extract_text_lines()` to all bindings. Provides semantic grouping of characters with bounding boxes, font info, and styling (parity with `pdfplumber`).
- **Geometric Primitive Extraction** (#185, #191) — added `extract_rects()` and `extract_lines()` to identify vector graphics.
- **Hybrid Table Detection** (#185, #192) — updated `SpatialTableDetector` to use vector lines as hints, significantly improving detection of "bordered" tables.
- **API Harmonization** — implemented the fluent `.within(page, rect)` pattern across Rust, Python, and WASM for scoped extraction.
- **Area Filtering** — added optional `region` support to all extraction methods (`extract_text`, `extract_chars`, etc.) in Python and WASM, using backward-compatible signatures.
- **Deep Data Access** — added `.chars` property to `TextWord` and `TextLine` objects in Python, enabling granular access to individual character metadata.
- **CLI Enhancements** — added `pdf-oxide render` for image generation and `pdf-oxide paths` for geometric JSON extraction. Integrated `--area` filtering across all extraction commands.

### Bug Fixes — Text Extraction (#193, #202, #204)

Reported by **@MarcRene71** — `AttributeError: 'builtins.PdfDocument' object has no attribute 'extract_text_ocr'` when using the library without the OCR feature enabled.

- **Improved Feature Gating Discovery** (#204) — ensured that all optional features (OCR, Office, Rendering) are always visible in the Python API. If a feature is disabled at build time, calling its methods now returns a helpful `RuntimeError` explaining how to enable it (e.g., `pip install pdf_oxide[ocr]`), instead of throwing an `AttributeError`.
- **Always-on Type Stubs** (#204) — updated `.pyi` files to include all methods regardless of build features, providing full IDE autocompletion support for all capabilities.

Reported by **@cole-dda** — repeated calls to `extract_texts()` and `extract_spans()` return inconsistent results (empty lists on second/third calls).

- **Fixed XObject span cache poisoning** (#193) — resolved an issue where `extract_chars()` (low-level API) would incorrectly populate the high-level `xobject_spans_cache` with empty results. Because `extract_chars()` does not collect spans, it was "poisoning" the cache for subsequent `extract_spans()` calls, causing them to return empty data for any content inside Form XObjects.
- **Improved extraction mode isolation** (#193) — ensured that the text extractor explicitly separates character and span extraction paths. The span result cache is now only accessed and updated when in span extraction mode, and internal span buffers are cleared when entering character mode.

Reported by **@vincenzopalazzo** — `extract_text()` returns empty string for encrypted PDFs with CID TrueType Identity-H fonts.

- **Support for V=4 Crypt Filters** (#202) — fixed a bug in `EncryptDict` where version 4 encryption was hardcoded to AES-128. It now correctly parses the `/CF` dictionary and `/CFM` entry to select between RC4-128 (`/V2`) and AES-128 (`/AESV2`), enabling support for PDFs produced by OpenPDF.
- **Encrypted CIDToGIDMap decryption** (#202) — fixed a missing decryption step when loading `CIDToGIDMap` streams. Previously, the stream was decompressed but remained encrypted, causing invalid glyph mapping and failed text extraction.
- **Enhanced font diagnostic logging** (#202) — replaced silent failures with descriptive warnings when ToUnicode CMaps or FontFile2 streams fail to load or decrypt, making it easier to diagnose complex extraction issues.

### Refactoring

- **Consolidated text decoding and positioning logic** (#187) — unified the high-level `extract_text_spans()` and low-level `extract_chars()` paths into a single shared engine to prevent logic drift and ensure consistent character handling.
- **Fixed render_page for in-memory PDFs** — ensured that PDFs created from bytes or strings can be rendered by automatically initializing a temporary editor if needed.
- **Improved Clustering Accuracy** — updated character clustering to use gap-based distance instead of center-to-center distance, ensuring accurate word grouping regardless of font size.

### Community Contributors

Thank you to **@MarcRene71** for identifying the critical API discoverability issue with OCR (#204). Your report led to a more robust "Pythonic" approach to feature gating, ensuring that users always see the full API and receive helpful guidance when features are disabled!

Thank you to **@vincenzopalazzo** for identifying and fixing the critical issues with encrypted CID fonts and V=4 crypt filters (#202). Your contribution of both the fix and the reproduction fixture was essential for ensuring PDFOxide handles professional PDFs from diverse producers!

Thank you to **@ankursri494** (Ankur Srivastava) for the excellent proposal to bridge the gap between `PdfPlumber`'s flexibility and PDFOxide's performance (#185). Your detailed breakdown of word-level and table extraction requirements was the roadmap for this release!

Thank you to **@cole-dda** for identifying the critical caching bug (#193). The detailed reproduction case was essential for pinpointing the interaction between the low-level character API and the document-level XObject caches.

## [0.3.13] - 2026-03-02
> Character Extraction Quality, Multi-byte Encoding (Issue #186)

### Bug Fixes — Character Extraction (#186)

Reported by **@cole-dda** — garbled output when using `extract_chars()` on PDFs with multi-byte encodings (CJK text, Type0 fonts).

- **Multi-byte decoding in show_text** — fixed `extract_chars()` to correctly handle 2-byte and variable-width encodings (Identity-H/V, Shift-JIS, etc.). Previously, characters were processed byte-by-byte, causing multi-byte characters to be split and garbled. Now uses the same robust decoding logic as `extract_spans()`.
- **Improved character positioning accuracy** — replaced the 0.5em fixed-width estimate in `show_text` with actual glyph widths from the font dictionary. This ensures that character bounding boxes (`bbox`) and origins are precisely positioned, matching the actual PDF rendering.
- **Accurate character advancement** — character spacing (`Tc`) and word spacing (`Tw`) are now correctly scaled by horizontal scaling (`Th`) during character-level extraction, ensuring correct text matrix updates.

### Community Contributors

Thank you to **@cole-dda** for identifying and reporting the character extraction quality issue with an excellent reproduction case (#186). Your report directly led to identifying the divergence between our high-level and low-level extraction paths, making `extract_chars()` significantly more robust for CJK and other multi-byte documents. We really appreciate your contribution to making PDF Oxide better!

## [0.3.12] - 2026-03-01
> Text Extraction Quality, Determinism, Performance, Markdown Conversion

### Bug Fixes — Text Extraction (#181)

Reported by **@Goldziher** — systematic evaluation across 10 PDFs covering word merging, encoding failures, and RTL text.

- **CID font width calculation** — fixed text-to-user space conversion for CID fonts. Glyph widths were not correctly scaled, causing word boundary detection to merge adjacent words (`destinationmachine` → `destination machine`, `helporganizeas` → `help organize as`).

- **Font-change word boundary detection** — when PDF font changes mid-line (e.g., regular→italic for product names in LaTeX), we now detect this as a word boundary even if the visual gap is small. Previously, these were merged into single words with mixed formatting.

- **Non-Standard CID mapping fallback** — implemented a fallback mechanism for CID fonts with broken `/ToUnicode` maps. If mapping fails, we now attempt to use the font's internal `cmap` table directly. Fixed encoding failures in 3 PDFs from the corpus.

- **RTL text directionality foundation** — added basic support for identifying RTL (Right-to-Left) script spans (Arabic, Hebrew) based on Unicode range. Provides correctly ordered spans for simple RTL layouts.

### Features — Markdown Conversion

- **Optimized Markdown engine** — significantly improved the performance of `to_markdown()` by implementing recursive spatial partitioning (XY-Cut). This ensures that multi-column layouts and complex document structures are converted into accurate, readable Markdown.
- **Heading Detection** — automated identification of headers (H1-H6) based on font size variance and document-wide frequency analysis.
- **List Reconstruction** — detects bulleted and numbered lists by analyzing leading character patterns and indentation consistency.

### Performance

- **Zero-copy page tree traversal** — refactored internal page navigation to avoid redundant dictionary cloning during deep page tree traversal for multi-page extraction.
- **Structure tree caching** — Structure tree result cached after first access, avoiding redundant parsing on every `extract_text()` call (major impact on tagged PDFs like PDF32000_2008.pdf).
- **BT operator early-out** — `extract_spans()`, `extract_spans_with_config()`, and `extract_chars()` skip the full text extraction pipeline for image-only pages that contain no `BT` (Begin Text) operators.
- **Larger I/O buffer for big files** — `BufReader` capacity increased from 8 KB to 256 KB for files >100 MB, reducing syscall overhead on 1.5 GB newspaper archives.
- **Xref reconstruction threshold removed** — Eliminated the `xref.len() < 5` heuristic that triggered full-file reconstruction on valid portfolio PDFs with few objects (5-13s → <100ms).

### Community Contributors

Thank you to **@Goldziher** for the exhaustive evaluation of PDF extraction quality (#181). Your systematic approach to testing across 10 diverse documents directly resulted in critical fixes for font scaling and encoding fallbacks. The feedback from power users like you is what drives PDF Oxide's quality forward!

## [0.3.5] - 2026-02-20
> Stability, Image Extraction & Error Recovery (Issue #41, #44, #45, #46)

### Verified — 3,830-PDF Corpus

- **100% pass rate** on 3,830 PDFs across three independent test suites: veraPDF (2,907), Mozilla pdf.js (897), SafeDocs (26).
- **Zero timeouts, zero panics** — every PDF completes within 120 seconds.
- **p50 = 0.6ms, p90 = 3.0ms, p99 = 33ms** — 97.6% of PDFs complete in under 10ms.
- Added `verify_corpus` example binary for reproducible batch verification with CSV output, timeout handling, and per-corpus breakdown.

### Added - Encryption

- **Owner password authentication** (Algorithm 7 for R≤4, Algorithm 12 for R≥5).
  - R≤4: Derives RC4 key from owner password via MD5 hash chain, decrypts `/O` value to recover user password, then validates via user password authentication.
  - R≥5: SHA-256 verification with SASLprep normalization and owner validation/key salts per PDF spec §7.6.3.4.
  - Both algorithms now fully wired into `EncryptionHandler::authenticate()`.
- **R≥5 user password verification with SASLprep** — Full AES-256 password verification using SHA-256 with validation and key salts per PDF spec §7.6.4.3.3.
- **Public password authentication API** — `Pdf::authenticate(password)` and `PdfDocument::authenticate(password)` exposed for user-facing password entry.

### Added - PDF/A Compliance Validation

- **XMP metadata validation** — Parses XMP metadata stream and checks for `pdfaid:part` and `pdfaid:conformance` identification entries (clause 6.7.11).
- **Color space validation** — Scans page content streams for device-dependent color operators (`rg`, `RG`, `k`, `K`, `g`, `G`) without output intent (clause 6.2).
- **AFRelationship validation** — For PDF/A-3 documents with embedded files, validates each file specification dictionary contains the required `AFRelationship` key (clause 6.8).

### Added - PDF/X Compliance Validation

- **XMP PDF/X identification** — Parses XMP metadata for `pdfxid:GTS_PDFXVersion`, validates against declared level (clause 6.7.2).
- **Page box relationship validation** — Validates TrimBox ⊆ BleedBox ⊆ MediaBox and ArtBox ⊆ MediaBox with 0.01pt tolerance (clause 6.1.1).
- **ExtGState transparency detection** — Checks `SMask` (not `/None`), `CA`/`ca` < 1.0, and `BM` not `Normal`/`Compatible` in extended graphics state dictionaries (clause 6.3).
- **Device-dependent color detection** — Flags DeviceRGB/CMYK/Gray color spaces used without output intent (clause 6.2.3).
- **ICC profile validation** — Validates ICCBased color space profile streams contain required `/N` entry (clause 6.2.3).

### Added - Rendering

- **Spec-correct clipping** (PDF §8.5.4) — Clip state scoped to `q`/`Q` save/restore via clip stack; new clips intersect with existing clip region; `W`/`W*` no longer consume the current path (deferred to next paint operator); clip mask applied to all painting operations including text and images.
- **Glyph advance width calculation** — Text position advances per PDF spec §9.4.4: `tx = (w0/1000 × Tfs + Tc + Tw) × Th` with 600-unit default glyph width.
- **Form XObject rendering** — Parses `/Matrix` transform, uses form's `/Resources` (or inherits from parent), and recursively executes form content stream operators.

### Fixed - Error Recovery (28+ real-world PDFs)

- **Missing objects resolve to Null** — Per PDF spec §7.3.10, unresolvable indirect references now return `Null` instead of errors, fixing 16 files across veraPDF/pdf.js corpora.
- **Lenient header version parsing** — Fixed fast-path bug where valid headers with unusual version strings were rejected.
- **Non-standard encryption algorithm matching** — V=1,R=3 combinations now handled leniently instead of rejected.
- **Non-dictionary Resources** — Pages with invalid `/Resources` entries (e.g., Null, Integer) treated as empty resources instead of erroring.
- **Null nodes in page tree** — Null or non-dictionary child nodes in page tree gracefully skipped during traversal.
- **Corrupt content streams** — Malformed content streams return empty content instead of propagating parse errors.
- **Enhanced page tree scanning** — `/Resources`+`/Parent` heuristic and `/Kids` direct resolution added as fallback passes for damaged page trees.

### Fixed - DoS Protection

- **Bogus /Count bounds checking** — Page count validated against PDF spec Annex C.2 limit (8,388,607) and total object count; unreasonable values fall back to tree scanning.

### Fixed - Image Extraction
- **Content stream image extraction** — `extract_images()` now processes page content streams to find `Do` operator calls, extracting images referenced via XObjects that were previously missed.
- **Nested Form XObject images** — Recursive extraction with cycle detection handles images inside Form XObjects.
- **Inline images** — `BI`...`ID`...`EI` sequences parsed with abbreviation expansion per PDF spec.
- **CTM transformations** — Image bounding boxes correctly transformed using full 4-corner affine transform (handles rotation, shear, and negative scaling).
- **ColorSpace indirect references** — Resolved indirect references (e.g., `7 0 R`) in image color space entries before extraction.

### Fixed - Parser Robustness

- **Multi-line object headers** — Parser now handles `1 0\nobj` format used by Google-generated PDFs instead of requiring `1 0 obj` on a single line.
- **Extended header search** — Header search window extended from 1024 to 8192 bytes to handle PDFs with large binary prefixes.
- **Lenient version parsing** — Malformed version strings like `%PDF-1.a` or truncated headers no longer cause parse failures in lenient mode.

### Fixed - Page Access Robustness

- **Missing Contents entry** — Pages without a `/Contents` key now return empty content data instead of erroring.
- **Cyclic page tree detection** — Page tree traversal tracks visited nodes to prevent stack overflow on malformed circular references.
- **Null stream references** — Null or invalid stream references handled gracefully instead of panicking.
- **Wider page scanning fallback** — Page scanning fallback triggers on more error conditions, improving compatibility with damaged PDFs.
- **Pages without /Type entry** — Page scanning now finds pages missing the `/Type /Page` entry by checking for `/MediaBox` or `/Contents` keys.

### Fixed - Encryption Robustness

- **Short encryption key panic** — AES decryption with undersized keys now returns an error instead of panicking.
- **Xref stream parsing hardened** — Malformed xref streams with invalid entry sizes or out-of-bounds data no longer cause panics.
- **Indirect /Encrypt references** — `/Encrypt` dictionary values that are indirect references are now resolved before parsing.

### Fixed - Content Stream Processing

- **Dictionary-as-Stream fallback** — When a stream object is a bare dictionary (no stream data), it is now treated as an empty stream instead of causing a decode error.
- **Filter abbreviations** — Abbreviated filter names (`AHx`, `A85`, `LZW`, `Fl`, `RL`, `CCF`, `DCT`) and case-insensitive matching now supported.
- **Operator limit** — Content stream parsing enforces a configurable operator limit (default 1,000,000) to prevent pathological slowdowns on malformed streams.

### Fixed - Code Quality

- **Structure tree indirect object references** — `ObjectRef` variants in structure tree `/K` entries are now resolved at parse time instead of being silently skipped, ensuring complete structure tree traversal.
- **Lexer `R` token disambiguation** — `tag(b"R")` no longer matches the `R` prefix of `RG`/`ri`/`re` operators; `1 0 RG` is now correctly parsed as a color operator instead of indirect reference `1 0 R` + orphan `G`.
- **Stream whitespace trimming** — `trim_leading_stream_whitespace` now only strips CR/LF (0x0D/0x0A), no longer strips NUL bytes (0x00) or spaces from binary stream data (fixes grayscale image extraction and object stream parsing).

### Tests

- **8 previously ignored tests un-ignored and fixed**:
  - `test_extract_raw_grayscale_image_from_xobject` — Fixed stream trimming stripping binary pixel data.
  - `test_parse_object_stream_with_whitespace` — Fixed stream trimming affecting object stream offsets.
  - `test_parse_object_stream_graceful_failure` — Relaxed assertion for improved parser recovery.
  - `test_markdown_reading_order_top_to_bottom` — Fixed test coordinates to use PDF convention (Y increases upward).
  - `test_html_layout_multiple_elements` — Fixed assertions for per-character positioning.
  - `test_reading_order_graph_based_simple` — Fixed test coordinates to PDF convention.
  - `test_reading_order_two_columns` — Fixed test coordinates to PDF convention.
  - `test_parse_color_operators` — Fixed lexer R/RG token disambiguation.

### Removed

- Deleted empty `PdfImage` stub (`src/images.rs`) and its module export — image extraction uses `ImageInfo` from `src/extractors/images.rs`.
- Deleted commented-out `DocumentType::detect()` test block in `src/extractors/gap_statistics.rs`.
- Removed stale TODO comments in `scripts/setup-hooks.sh`, `src/bin/analyze_pdf_features.rs`, `src/document.rs`.

### 🏆 Community Contributors

🥇 **@SeanPedersen** — Huge thanks for reporting multiple issues (#41, #44, #45, #46) that drove the entire stability focus of this release. His real-world testing uncovered a parser bug with Google-generated PDFs, image extraction failures on content stream references, and performance problems — each report triggering deep investigation and significant fixes. The parser robustness, image extraction, and testing infrastructure improvements in v0.3.5 all trace back to Sean's thorough bug reports. 🙏🔍

## [0.3.4] - 2026-02-12
> Parsing Robustness, Character Extraction & XObject Paths

### ⚠️ Breaking Changes
- **`parse_header()` function signature** - Now includes offset tracking.
  - **Before**: `parse_header(reader) -> Result<(u8, u8)>`
  - **After**: `parse_header(reader, lenient) -> Result<(u8, u8, u64)>`
  - **Migration**: Replace `let (major, minor) = parse_header(&mut reader)?;` with `let (major, minor, _offset) = parse_header(&mut reader, true)?;`
  - Note: This is a public API function; consider using `doc.version()` for typical use cases instead.

### Fixed - PDF Parsing Robustness (Issue #41)
- **Header offset support** - PDFs with binary prefixes or BOM headers now open successfully.
  - Parse header function now searches first 1024 bytes for `%PDF-` marker (PDF spec compliant).
  - Supports UTF-8 BOM, email headers, and other leading binary data.
  - `parse_header()` returns byte offset where header was found.
  - Lenient mode (default) handles real-world malformed PDFs; strict mode for compliance testing.
  - Fixes parsing errors like "expected '%PDF-', found '1b965'".

### Added - Character-Level Text Extraction (Issue #39)
- **`extract_chars()` API** - Low-level character-level extraction for layout analysis.
  - Returns `Vec<TextChar>` with per-character positioning, font, and styling data.
  - Includes transformation matrix, rotation angle, advance width.
  - Sorted in reading order (top-to-bottom, left-to-right).
  - Overlapping characters (rendered multiple times) deduplicated.
  - 30-50% faster than span extraction for character-only use cases.
  - Exposed in both Rust and Python APIs.
  - **Python binding**: `doc.extract_chars(page_index)` returns list of `TextChar` objects.

### Added - XObject Path Extraction (Issue #40)
- **Form XObject support in path extraction** - Now extracts vectors from embedded XObjects.
  - `extract_paths()` recursively processes Form XObjects via `Do` operator.
  - Image XObjects properly skipped (only Form XObjects extracted).
  - Coordinate transformations via `/Matrix` properly applied.
  - Graphics state properly isolated (save/restore).
  - Duplicate XObject detection prevents infinite loops.
  - Nested XObjects (XObject containing XObject) supported.

### Changed
- **Dependencies**: Upgraded nom parser library from 7.1 to 8.0.
  - Updated all parser combinators to use `.parse()` method.
  - No user-facing API changes.
  - All parser functionality maintained.
  - Performance stable (no regressions detected).
- `parse_header()` signature updated: now returns `(major, minor, offset)` tuple.
- All parse_header test cases updated to use new signature.

## [0.3.1] - 2026-01-14
> Form Fields, Multimedia & Python 3.8-3.14

### Added - Form Field Coverage (95% across Read/Create/Modify)

#### Hierarchical Field Creation
- **Parent/Child Field Structures** - Create complex form hierarchies like `address.street`, `address.city`.
  - `add_parent_field()` - Create container fields without widgets.
  - `add_child_field()` - Add child fields to existing parents.
  - `add_form_field_hierarchical()` - Auto-create parent hierarchy from dotted names.
  - `ParentFieldConfig` for configuring container fields.
  - Property inheritance between parent and child fields (FT, V, DV, Ff, DA, Q).

#### Field Property Modification
- **Edit All Field Properties** - Beyond just values.
  - `set_form_field_readonly()` / `set_form_field_required()` - Flag manipulation.
  - `set_form_field_rect()` - Reposition/resize fields.
  - `set_form_field_tooltip()` - Set hover text (TU).
  - `set_form_field_max_length()` - Text field length limits.
  - `set_form_field_alignment()` - Text alignment (left/center/right).
  - `set_form_field_default_value()` - Default values (DV).
  - `BorderStyle` and `AppearanceCharacteristics` support.
- **Critical Bug Fix** - Modified existing fields now persist on save (was only saving new fields).

#### FDF/XFDF Export
- **Forms Data Format Export** - ISO 32000-1:2008 Section 12.7.7.
  - `FdfWriter` - Binary FDF export for form data exchange.
  - `XfdfWriter` - XML XFDF export for web integration.
  - `export_form_data_fdf()` / `export_form_data_xfdf()` on FormExtractor, DocumentEditor, Pdf.
  - Hierarchical field representation in exports.

### Added - Text Extraction Enhancements
- **TextChar Transformation** - Per-character positioning metadata (#27).
  - `origin` - Font baseline coordinates (x, y).
  - `rotation_degrees` - Character rotation angle.
  - `matrix` - Full transformation matrix.
  - Essential for pdfium-render migration.

### Added - Image Metadata
- **DPI Calculation** - Resolution metadata for images.
  - `horizontal_dpi` / `vertical_dpi` fields on `ImageContent`.
  - `resolution()` - Get (h_dpi, v_dpi) tuple.
  - `is_high_resolution()` / `is_low_resolution()` / `is_medium_resolution()` helpers.
  - `calculate_dpi()` - Compute from pixel dimensions and bbox.

### Added - Bounded Text Extraction
- **Spatial Filtering** - Extract text from rectangular regions.
  - `RectFilterMode::Intersects` - Any overlap (default).
  - `RectFilterMode::FullyContained` - Completely within bounds.
  - `RectFilterMode::MinOverlap(f32)` - Minimum overlap fraction.
  - `TextSpanSpatial` trait - `intersects_rect()`, `contained_in_rect()`, `overlap_with_rect()`.
  - `TextSpanFiltering` trait - `filter_by_rect()`, `extract_text_in_rect()`.

### Added - Multimedia Annotations
- **MovieAnnotation** - Embedded video content.
- **SoundAnnotation** - Audio content with playback controls.
- **ScreenAnnotation** - Media renditions (video/audio players).
- **RichMediaAnnotation** - Flash/video rich media content.

### Added - 3D Annotations
- **ThreeDAnnotation** - 3D model embedding.
  - U3D and PRC format support.
  - `ThreeDView` - Camera angles and lighting.
  - `ThreeDAnimation` - Playback controls.

### Added - Path Extraction
- **PathExtractor** - Vector graphics extraction.
  - Lines, curves, rectangles, complex paths.
  - Path transformation and bounding box calculation.

### Added - XFA Form Support
- **XfaExtractor** - Extract XFA form data.
- **XfaParser** - Parse XFA XML templates.
- **XfaConverter** - Convert XFA forms to AcroForm.

### Changed - Python Bindings
- **True Python 3.8-3.14 Support** - Fixed via `abi3-py38` (was only working on 3.11).
- **Modern Tooling** - uv, pdm, ruff integration.
- **Code Quality** - All Python code formatted with ruff.

### 🏆 Community Contributors

🥇 **@monchin** - Massive thanks for revolutionizing our Python ecosystem! Your PR #29 fixed a critical compatibility issue where PDFOxide only worked on Python 3.11 despite claiming 3.8+ support. By switching to `abi3-py38`, you enabled true cross-version compatibility (Python 3.8-3.14). The introduction of modern tooling (uv, pdm, ruff) brings PDFOxide's Python development to 2026 standards. This work directly enables thousands more Python developers to use PDFOxide. 💪🐍

🥈 **@bikallem** - Thanks for the thoughtful feature request (#27) comparing PDFOxide to pdfium-render. Your detailed analysis of missing origin coordinates and rotation angles led directly to our TextChar transformation feature. This makes PDFOxide a viable migration path for pdfium-render users. 🎯

## [0.3.0] - 2026-01-10
> Unified API, PDF Creation & Editing

### Added - Unified `Pdf` API
- **One API for Extract, Create, and Edit** - The new `Pdf` class unifies all PDF operations.
  - `Pdf::open("input.pdf")` - Open existing PDF for reading and editing.
  - `Pdf::from_markdown(content)` - Create new PDF from Markdown.
  - `Pdf::from_html(content)` - Create new PDF from HTML.
  - `Pdf::from_text(content)` - Create new PDF from plain text.
  - `Pdf::from_image(path)` - Create PDF from image file.
  - DOM-like page navigation with `pdf.page(0)` for querying and modifying content.
  - Seamless save with `pdf.save("output.pdf")` or `pdf.save_encrypted()`.
- **Fluent Builder Pattern** - `PdfBuilder` for advanced configuration.
  ```rust
  PdfBuilder::new()
      .title("My Document")
      .author("Author Name")
      .page_size(PageSize::A4)
      .from_markdown("# Content")?
  ```

### Added - PDF Creation
- **PDF Creation API** - Fluent `DocumentBuilder` for programmatic PDF generation.
  - `Pdf::create()` / `DocumentBuilder::new()` entry points.
  - Page sizing (Letter, A4, custom dimensions).
  - Text rendering with Base14 fonts and styling.
  - Image embedding (JPEG/PNG) with positioning.
- **Table Rendering** - `TableRenderer` for styled tables.
  - Headers, borders, cell spans, alternating row colors.
  - Column width control (fixed, percentage, auto).
  - Cell alignment and padding.
- **Graphics API** - Advanced visual effects.
  - Colors (RGB, CMYK, grayscale).
  - Linear and radial gradients.
  - Tiling patterns with presets.
  - Blend modes and transparency (ExtGState).
- **Page Templates** - Reusable page elements.
  - Headers and footers with placeholders.
  - Page numbering formats.
  - Watermarks (text-based).
- **Barcode Generation** (requires `barcodes` feature)
  - QR codes with configurable size and error correction.
  - Code128, EAN-13, UPC-A, Code39, ITF barcodes.
  - Customizable colors and dimensions.

### Added - PDF Editing
- **Editor API** - DOM-like editing with round-trip preservation.
  - `DocumentEditor` for modifying existing PDFs.
  - Content addition without breaking existing structure.
  - Resource management for fonts and images.
- **Annotation Support** - Full read/write for all types.
  - Text markup: highlights, underlines, strikeouts, squiggly.
  - Notes: sticky notes, comments, popups.
  - Shapes: rectangles, circles, lines, polygons, polylines.
  - Drawing: ink/freehand annotations.
  - Stamps: standard and custom stamps.
  - Special: file attachments, redactions, carets.
- **Form Fields** - Interactive form creation.
  - Text fields (single/multiline, password, comb).
  - Checkboxes with custom appearance.
  - Radio button groups.
  - Dropdown and list boxes.
  - Push buttons with actions.
  - Form flattening (convert fields to static content).
- **Link Annotations** - Navigation support.
  - External URLs.
  - Internal page navigation.
  - Styled link appearance.
- **Outline Builder** - Bookmark/TOC creation.
  - Hierarchical structure.
  - Page destinations.
  - Styling (bold, italic, colors).
- **PDF Layers** - Optional Content Groups (OCG).
  - Create and manage content layers.
  - Layer visibility controls.

### Added - PDF Compliance & Validation
- **PDF/A Validation** - ISO 19005 compliance checking.
  - PDF/A-1a, PDF/A-1b (ISO 19005-1).
  - PDF/A-2a, PDF/A-2b, PDF/A-2u (ISO 19005-2).
  - PDF/A-3a, PDF/A-3b (ISO 19005-3).
- **PDF/A Conversion** - Convert documents to archival format.
  - Automatic font embedding.
  - XMP metadata injection.
  - ICC color profile conversion.
- **PDF/X Validation** - ISO 15930 print production compliance.
  - PDF/X-1a:2001, PDF/X-1a:2003.
  - PDF/X-3:2002, PDF/X-3:2003.
  - PDF/X-4, PDF/X-4p.
  - PDF/X-5g, PDF/X-5n, PDF/X-5pg.
  - PDF/X-6, PDF/X-6n, PDF/X-6p.
  - 40+ specific error codes for violations.
- **PDF/UA Validation** - ISO 14289 accessibility compliance.
  - Tagged PDF structure validation.
  - Language specification checks.
  - Alt text requirements.
  - Heading hierarchy validation.
  - Table header validation.
  - Form field accessibility.
  - Reading order verification.

### Added - Security & Encryption
- **Encryption on Write** - Password-protect PDFs when saving.
  - AES-256 (V=5, R=6) - Modern 256-bit encryption (default).
  - AES-128 (V=4, R=4) - Modern 128-bit encryption.
  - RC4-128 (V=2, R=3) - Legacy 128-bit encryption.
  - RC4-40 (V=1, R=2) - Legacy 40-bit encryption.
  - `Pdf::save_encrypted()` for simple password protection.
  - `Pdf::save_with_encryption()` for full configuration.
- **Permission Controls** - Granular access restrictions.
  - Print, copy, modify, annotate permissions.
  - Form fill and accessibility extraction controls.
- **Digital Signatures** (foundation, requires `signatures` feature)
  - ByteRange calculation for signature placeholders.
  - PKCS#7/CMS signature structure support.
  - X.509 certificate parsing.
  - Signature verification framework.

### Added - Document Features
- **Page Labels** - Custom page numbering.
  - Roman numerals, letters, decimal formats.
  - Prefix support (e.g., "A-1", "B-2").
  - `PageLabelsBuilder` for creation.
  - Extract existing labels from documents.
- **XMP Metadata** - Extensible metadata support.
  - Dublin Core properties (title, creator, description).
  - PDF properties (producer, keywords) .
  - Custom namespace support.
  - Full read/write capability.
- **Embedded Files** - File attachments.
  - Attach files to PDF documents.
  - MIME type and description support.
  - Relationship specification (Source, Data, etc.).
- **Linearization** - Web-optimized PDFs.
  - Fast web view support.
  - Streaming delivery optimization.

### Added - Search & Analysis
- **Text Search** - Pattern-based document search.
  - Regex pattern support.
  - Case-sensitive/insensitive options.
  - Position tracking with page/coordinates.
  - Whole word matching.
- **Page Rendering** (requires `rendering` feature)
  - Render pages to PNG/JPEG images.
  - Configurable DPI and scale.
  - Pure Rust via tiny-skia (no external dependencies).
- **Debug Visualization** (requires `rendering` feature)
  - Visualize text bounding boxes.
  - Element highlighting for debugging.
  - Export annotated page images.

### Added - Document Conversion
- **Office to PDF** (requires `office` feature)
  - **DOCX**: Word documents with paragraphs, headings, lists, formatting.
  - **XLSX**: Excel spreadsheets via calamine (sheets, cells, tables).
  - **PPTX**: PowerPoint presentations (slides, titles, text boxes).
  - `OfficeConverter` with auto-detection.
  - `OfficeConfig` for page size, margins, fonts.
  - Python bindings: `OfficeConverter.from_docx()`, `from_xlsx()`, `from_pptx()`.

### Added - Python Bindings
- `Pdf` class for PDF creation.
- `Color`, `BlendMode`, `ExtGState` for graphics.
- `LinearGradient`, `RadialGradient` for gradients.
- `LineCap`, `LineJoin`, `PatternPresets` for styling.
- `save_encrypted()` method with permission flags.
- `OfficeConverter` class for Office document conversion.

### Changed
- Description updated to "The Complete PDF Toolkit: extract, create, and edit PDFs".
- Python module docstring updated for v0.3.0 features.
- Branding updated with Extract/Create/Edit pillars.

### Fixed
- **Outline action handling** - correctly dereference actions indirectly referenced by outline items.

### 🏆 Community Contributors

🥇 **@jvantuyl** - Thanks for the thorough PR #16 fixing outline action dereferencing! Your investigation uncovered that some PDFs embed actions directly while others use indirect references - a subtle PDF spec detail that was breaking bookmark navigation. Your fix included comprehensive tests ensuring this won't regress. 🔍✨

🙏 **@mert-kurttutan** - Thanks for the honest feedback in issue #15 about README clutter. Your perspective as a new user helped us realize we were overwhelming people with information. The resulting documentation cleanup makes PDFOxide more approachable. 📚

## [0.2.6] - 2026-01-09
> CJK Support & Structure Tree Enhancements

### Added
- **TagSuspect/MarkInfo support** (ISO 32000-1 Section 14.7.1).
  - Parse MarkInfo dictionary from document catalog (`marked`, `suspects`, `user_properties`).
  - `PdfDocument::mark_info()` method to retrieve MarkInfo.
  - Automatic fallback to geometric ordering when structure tree is marked as suspect.
- **Word Break /WB structure element** (Section 14.8.4.4).
  - Support for explicit word boundaries in CJK text.
  - `StructType::WB` variant and `is_word_break()` helper.
  - Word break markers emitted during structure tree traversal.
- **Predefined CMap support for CJK fonts** (Section 9.7.5.2).
  - Adobe-GB1 (Simplified Chinese) - ~500 common character mappings.
  - Adobe-Japan1 (Japanese) - Hiragana, Katakana, Kanji mappings.
  - Adobe-CNS1 (Traditional Chinese) - Bopomofo and CJK mappings.
  - Adobe-Korea1 (Korean) - Hangul and Hanja mappings.
  - Fallback identity mapping for common Unicode ranges.
- **Abbreviation expansion /E support** (Section 14.9.5).
  - Parse `/E` entry from marked content properties.
  - `expansion` field on `StructElem` for structure-level abbreviations.
- **Object reference resolution utility**.
  - `PdfDocument::resolve_references()` for recursive reference handling in complex PDF structures.
- **Type 0 /W array parsing** for CIDFont glyph widths.
  - Proper spacing for CJK text using CIDFont width specifications.
- **ActualText verification tests** - comprehensive test coverage for PDF Spec Section 14.9.4.

### Fixed
- **Soft hyphen handling** (U+00AD) - now correctly treated as valid continuation hyphen for word reconstruction.

### Changed
- **Enhanced artifact filtering** with subtype support.
  - `ArtifactType::Pagination` with subtypes: Header, Footer, Watermark, PageNumber.
  - `ArtifactType::Layout` and `ArtifactType::Background` classification.
- `OrderedContent.mcid` changed to `Option<u32>` to support word break markers.

## [0.2.5] - 2026-01-09
> Image Embedding & Export

### Added
- **Image embedding**: Both HTML and Markdown now support embedded base64 images when `embed_images=true` (default).
  - HTML: `<img src="data:image/png;base64,...">`
  - Markdown: `![alt](data:image/png;base64,...)` (works in Obsidian, Typora, VS Code, Jupyter).
- **Image file export**: Set `embed_images=false` + `image_output_dir` to save images as files with relative path references.
- New `embed_images` option in `ConversionOptions` to control embedding behavior.
- `PdfImage::to_base64_data_uri()` method for converting images to data URIs.
- `PdfImage::to_png_bytes()` method for in-memory PNG encoding.
- Python bindings: new `embed_images` parameter for `to_html`, `to_markdown`, and `*_all` methods.

## [0.2.4] - 2026-01-09
> CTM Fix & Formula Rendering

### Fixed
- CTM (Current Transformation Matrix) now correctly applied to text positions per PDF Spec ISO 32000-1:2008 Section 9.4.4 (#11).

### Added
- Structure tree: `/Alt` (alternate description) parsing for accessibility text on formulas and figures.
- Structure tree: `/Pg` (page reference) resolution - correctly maps structure elements to page numbers.
- `FormulaRenderer` module for extracting formula regions as base64 images from rendered pages.
- `ConversionOptions`: new fields `render_formulas`, `page_images`, `page_dimensions` for formula image embedding.
- Regression tests for CTM transformation.

### 🏆 Community Contributors

🐛➡️✅ **@mert-kurttutan** - Thanks for the detailed bug report (#11) with reproducible sample PDF! Your report exposed a fundamental CTM transformation bug affecting text positioning across the entire library. This fix was critical for production use. 🎉

## [0.2.3] - 2026-01-07
> BT/ET Matrix Reset & Text Processing

### Fixed
- BT/ET matrix reset per PDF spec Section 9.4.1 (PR #10 by @drahnr).
- Geometric spacing detection in markdown converter (#5).
- Verbose extractor logs changed from info to trace (#7).
- docs.rs build failure (excluded tesseract-rs).

### Added
- `apply_intelligent_text_processing()` method for ligature expansion, hyphenation reconstruction, and OCR cleanup (#6).

### Changed
- Removed unused tesseract-rs dependency.

### 🏆 Community Contributors

🥇 **@drahnr** - Huge thanks for PR #10 fixing the BT/ET matrix reset issue! This was a subtle PDF spec compliance bug (Section 9.4.1) where text matrices weren't being reset between text blocks, causing positions to accumulate and become unusable. Your fix restored correct text positioning for all PDFs. 💪📐

🔬 **@JanIvarMoldekleiv** - Thanks for the detailed bug report (#5) about missing spaces and lost table structure! Your analysis even identified the root cause in the code - the markdown converter wasn't using geometric spacing analysis. This level of investigation made the fix straightforward. 🕵️‍♂️

🎯 **@Borderliner** - Thanks for two important catches! Issue #6 revealed that `apply_intelligent_text_processing()` was documented but not actually available (oops! 😅), and #7 caught our overly verbose INFO-level logging flooding terminals. Both fixed immediately! 🔧

## [0.2.2] - 2025-12-15
> Discoverability Improvements

### Changed
- Optimized crate keywords for better discoverability.

## [0.2.1] - 2025-12-15
> Encrypted PDF Fixes

### Fixed
- Encrypted stream decoding improvements (#3).
- CI/CD pipeline fixes.

### 🏆 Community Contributors

🥇 **@threebeanbags** - Huge thanks for PRs #2 and #3 fixing encrypted PDF support! 🔐 Your first PR identified that decryption needed to happen before decompression - a critical ordering issue. Your follow-up PR #3 went deeper, fixing encryption handler initialization timing and adding Form XObject encryption support. These fixes made PDFOxide actually work with password-protected PDFs in production. 💪🎉

## [0.1.4] - 2025-12-12

### Fixed
- Encrypted stream decoding (#2).
- Documentation and doctest fixes.

## [0.1.3] - 2025-12-12

### Fixed
- Encrypted stream decoding refinements.

## [0.1.2] - 2025-11-27

### Added
- Python 3.13 support.
- GitHub sponsor configuration.

## [0.1.1] - 2025-11-26

### Added
- Cross-platform binary builds (Linux, macOS, Windows).

## [0.1.0] - 2025-11-06

### Added
- Initial release.
- PDF text extraction with spec-compliant Unicode mapping.
- Intelligent reading order detection.
- Python bindings via PyO3.
- Support for encrypted PDFs.
- Form field extraction.
- Image extraction.

### 🌟 Early Adopters

💖 **@magnus-trent** - Thanks for issue #1, our first community feedback! Your message that PDFOxide "unlocked an entire pipeline" you'd been working on for a month validated that we were solving real problems. Early encouragement like this keeps open source projects going. 🚀
