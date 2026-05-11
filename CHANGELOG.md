# Changelog

All notable changes to PDFOxide are documented here.

## [0.3.46] - 2026-05-10

### Added

- **Raw RGBA pixel buffer, SIMD downscaling, and thread-safe rendering
  ([#446](https://github.com/yfedoseev/pdf_oxide/issues/446),
  [#481](https://github.com/yfedoseev/pdf_oxide/issues/481))** —
  `page.render_pixmap()` (Python), `renderToPixmap()` (Node.js / Go),
  and `Page.RenderToRgba()` (C#) expose the premultiplied RGBA8888
  buffer directly from `tiny_skia::Pixmap::data()`, eliminating the
  encode→decode roundtrip for callers that need raw pixels (PIL,
  sharp, `System.Drawing.Bitmap`, `image.RGBA`). Downscaling is now SIMD-accelerated via
  `fast_image_resize` (ARM NEON, x86 AVX2), replacing the previous
  bilinear path. Concurrent `render_*` calls on the same
  `PdfDocument` are now safe: all rendering functions take `&PdfDocument`
  (shared reference) and all interior-mutable state is already guarded by
  per-field `Mutex`, so the FFI layer no longer produces aliased `&mut`
  references and concurrent renders run without a global serialisation
  bottleneck.
  Requested by @mara004 and @potatochipcoconut.

- **`ConversionOptions::exclude_regions` / `include_region`
  ([#484](https://github.com/yfedoseev/pdf_oxide/issues/484))** — New
  spatial filtering fields allow callers to exclude rectangular regions
  from extraction output or restrict extraction to a single bounding
  rectangle. Backed by `SpatialCollectionFiltering` trait methods
  `filter_by_rect` / `exclude_rects`.

- **`PageFontStats`
  ([#484](https://github.com/yfedoseev/pdf_oxide/issues/484))** — New
  `layout::PageFontStats` struct computed in O(n) over spans; exposes
  `dominant_em`, `dominant_line_height`, `dominant_char_width`, and
  `body_font_name`. All layout heuristics now derive absolute thresholds
  from these measurements instead of hardcoded constants, improving
  correctness across a wider range of font sizes.

### Fixed

- **JBIG2-compressed scanner PDFs render as blank pages
  ([#332](https://github.com/yfedoseev/pdf_oxide/issues/332))** —
  The pass-through `Jbig2Decoder` returned compressed bytes unchanged,
  causing a dimension mismatch and a silent image drop. Integrates
  `hayro-jbig2` v0.3 (pure-Rust, Apache-2.0 OR MIT); embedded JBIG2
  bitstreams are decoded via `hayro_jbig2::Image::new_embedded`, with
  JBIG2Globals loaded from `/DecodeParms` when present.
  `BitsPerComponent` is overridden to 8 post-decode so
  `to_dynamic_image()` does not attempt CCITT bilevel decompression of
  already-decoded pixels. Reported by @frederikhors, who also confirmed
  the original vertical-flip / glyph-substitution symptom is resolved
  in v0.3.45.

- **`add_text` on existing PDF produces blank or discarded content
  ([#483](https://github.com/yfedoseev/pdf_oxide/issues/483))** —
  `DocumentEditor::add_text` on a page of an existing PDF either blanked
  the page or (when combined with `select_pages`) silently returned the
  unmodified original. Root causes: the storage-side page-index mapping
  after `select_pages` was off by one, and `add_text` failed to preserve
  the existing content stream when writing the new text layer. Both are
  fixed; an end-to-end regression suite is added. Reported by
  @stephenjudkins.

- **Text extraction corpus quality improvements across 166 PDFs
  ([#484](https://github.com/yfedoseev/pdf_oxide/issues/484))** —
  Systematic audit driven by @Goldziher's calibrated 166-document corpus
  (the [kreuzberg](https://github.com/kreuzberg-dev/kreuzberg) test suite),
  which provides per-document ground-truth `.txt` files and a word-F1
  harness. Multiple extraction failures identified and fixed:

  - **Newline/CR-only spans treated as line breaks** — Spans consisting
    entirely of `\n` or `\r` bytes are now emitted as a single newline
    rather than verbatim byte sequences, eliminating spurious blank lines
    from some PDF generators.
  - **Annotation text double-emitted** — `append_non_widget_annotation_text`
    was called after the main span assembly pass even though
    `annotation_content_spans()` already inlined annotation `/Contents`
    into the span list. The redundant call is removed.
  - **Markup annotation `/Contents` correctly filtered** — Per ISO
    32000-1 §12.5.6.2, `/Contents` on Highlight, Underline, StrikeOut,
    Squiggly, Caret, Ink, FileAttachment, and Redact annotations is
    popup/tooltip text, not page content. These subtypes are now excluded
    from `annotation_content_spans` and `append_non_widget_annotation_text`.
  - **No space inserted between adjacent CJK characters** —
    `should_insert_space` now returns `false` when both the trailing and
    leading characters are CJK (Hiragana, Katakana, CJK Unified
    Ideographs, Hangul, CJK Extension B).
  - **Unicode ligatures preserved; adjacent CJK spans merged** — Latin
    ligatures (U+FB00–U+FB06) are now preserved in the span stream
    rather than dropped. Adjacent CJK spans from the same run are merged
    into a single span, eliminating inter-character noise.
  - **Lower→upper CID range boundary split restored** — The CID range
    boundary split now consistently applies the lower→upper ordering
    correction that was accidentally dropped; the fix propagates to
    Markdown and HTML output paths.
  - **Non-adjacent subscript/superscript spans merged** —
    `merge_sub_superscript_spans` handles spans separated by intervening
    content, using em-relative thresholds `[-0.1×em, +0.25×em]` instead
    of hardcoded absolute values so detection scales with body font size.
  - **Column-spanning decimals split at table cell boundaries** —
    Decimal numbers that span two adjacent table cells are split at the
    cell boundary rather than merged into a single token.
  - **Position-aware space insertion between adjacent MCID spans** —
    Spaces between MCID-tagged spans are inserted based on actual
    rendered x-positions rather than always or never.
  - **Boundary split on letter→digit transition only** —
    `char_widths_boundary_split` now splits only at a letter-to-digit
    boundary (e.g. `Theorem1`), removing false splits on UpperCamelCase
    terms that previously broke word-shape heuristics.
  - **Same-line threshold formula fixed** — `same_line_threshold` now
    uses `(min_fs × 1.2).max(max_fs × 0.3)`, handling mixed-size lines
    (heading + caption on the same line) without cliff effects.
  - **Bare-word identifiers and corrupt `StructTreeRoot` handled** —
    Parser now tolerates bare-word tokens as dictionary values; a
    corrupt or absent `StructTreeRoot` no longer aborts extraction.
  - **Standard-14 font matching strips `SUBSET+` prefix; accepts
    canonical PostScript aliases** — Per ISO 32000-1 §9.6.2.2 Annex D,
    standard font names are matched after stripping any `ABCDEF+` prefix.
    `HelveticaOblique` (no hyphen) is now accepted alongside
    `Helvetica-Oblique`.
  - **Explicit `/DW` tracked in `FontInfo`** — `has_explicit_dw: bool`
    added; `has_explicit_widths()` returns `true` when `/DW` is
    explicitly present, enabling correct width lookup for CIDFonts that
    declare only `/DW` (no `/W` array).
  - **CIDFont width fallback corrected** — When `/DW` is absent and a
    CID is not in the `/W` array, `get_glyph_width` now falls through to
    `default_width` rather than `cid_default_width`, matching real-world
    PDF behaviour.
  - **Word extractor honours `split_boundary_before`** — Words that
    straddle a table-cell or column boundary are no longer merged.
  - **Ligature expansion option** — `ConversionOptions` gains
    `expand_ligatures: bool` (default `false`). When enabled, Latin
    ligatures (U+FB00–U+FB06: ff, fi, fl, ffi, ffl, ſt, st) are
    expanded to component letters.
  - **Extraction warnings API** — `PdfDocument::warnings()` (clones) and
    `take_warnings()` (drains) expose non-fatal extraction warnings
    (missing MCIDs, encrypted-PDF fallback) accumulated during a run.

- **Same-line span reorder: x-gap validation guard
  ([#413](https://github.com/yfedoseev/pdf_oxide/pull/413))** —
  After the row-aware sort, mixed-baseline glyphs (superscripts,
  subscripts) could appear before their base glyphs. The
  `reorder_same_line_runs` helper now validates that a candidate run is
  horizontally contiguous before X-sorting it; runs with a large X gap
  are left in row-aware order, preventing disjoint footer/header content
  from being collapsed into a fake same-line sequence. Fixes `"8th"`
  ordering (was `"th8"`). Contributed by
  [@RolandWArnold](https://github.com/RolandWArnold) in
  [PR #413](https://github.com/yfedoseev/pdf_oxide/pull/413).

- **Layout word-merge O(n²) → O(n)** — The word-merge pass previously
  re-scanned the entire accumulator for every candidate span; it is now
  O(n) via an index map.

- **Wide spatial false-positive tables rejected via dense-row-ratio** —
  Table detection now computes the fraction of rows with dense (≥50%)
  column coverage and rejects candidates below the threshold, eliminating
  false positives on wide but sparsely populated layouts.

- **Bare-identifier lexer leniency confined to dict-value position** —
  The lexer's tolerance for bare (unquoted) name-like tokens is now
  restricted to dictionary value positions, preventing mis-tokenisation
  of content streams where the same byte sequences are valid operators.

- **Typographic Unicode spaces normalised in extracted spans** —
  Non-breaking, thin, en, em, and other Unicode space variants in span
  text are normalised to ASCII space before the word-spacing heuristics
  run, eliminating invisible gaps in the extracted output.

### Performance

- **Rendering: per-segment font re-parsing eliminated** — The text
  rasterizer no longer re-parses font data on every span segment; `Arc`
  clones across the hot render loop and redundant CJK subsetter
  invocations are also eliminated, reducing CPU time for text-heavy
  pages by 30–60%.

### Dependencies

- **`fast_image_resize` added
  ([#454](https://github.com/yfedoseev/pdf_oxide/issues/454))** —
  New dependency enabling SIMD-accelerated (ARM NEON, x86 AVX2) image
  downscaling for the raw-RGBA render path.

### CI

- **FIPS release workflow now validates on pull requests** —
  `release-fips.yml` now triggers on PRs to `main` that touch source,
  language-binding, or workflow files. The full build across all five
  platforms and all four language bindings runs without publishing,
  so the tag push is a pure deployment step after a confirmed-green PR.
- **macOS x86\_64 FIPS builds moved to free runners** — All four
  `macos-13-xlarge` (paid Intel Larger Runner, causing indefinite queue
  waits on plans without access) replaced with `macos-latest`
  (free ARM runner cross-compiling to `x86_64-apple-darwin`).
- **Cargo registry caching added to all 20 FIPS build jobs** —
  Per-target cache keys (`$runner_os-$target-fips-cargo-$lock_hash`)
  are restored before each build, substantially reducing re-run time
  on warm caches.

### Community contributors

- **[@RolandWArnold](https://github.com/RolandWArnold)** — contributed
  the same-line x-gap validation fix in
  [PR #413](https://github.com/yfedoseev/pdf_oxide/pull/413). Roland
  diagnosed that `reorder_same_line_runs` was collapsing disjoint
  footer/header spans into a fake same-line sequence and designed the
  horizontal-contiguity guard that prevents it. The fix also correctly
  handles superscript/subscript ordering (`"8th"` instead of `"th8"`).
- **[@Goldziher](https://github.com/Goldziher)** (Na'aman Hirschfeld) —
  filed [#484](https://github.com/yfedoseev/pdf_oxide/issues/484) with a
  calibrated 166-document corpus, per-document ground-truth `.txt` files,
  and a word-F1 harness, providing the systematic test bed that drove the
  bulk of the extraction improvements in this release.
- **[@stephenjudkins](https://github.com/stephenjudkins)** (Stephen
  Judkins) — filed [#483](https://github.com/yfedoseev/pdf_oxide/issues/483)
  with a minimal, precisely-scoped reproduction of the `add_text`
  regression that made the root-cause analysis straightforward.
- **[@mara004](https://github.com/mara004)** and
  **[@potatochipcoconut](https://github.com/potatochipcoconut)** —
  requested the raw RGBA pixel buffer API in comments on
  [#325](https://github.com/yfedoseev/pdf_oxide/issues/325) with clear
  use cases across PIL, sharp, `System.Drawing.Bitmap`, and
  Go's `image.RGBA`, and engaged on the pixel-format details (premultiplied
  vs straight alpha, tiny-skia format constraints) that shaped the final
  API design.
- **[@frederikhors](https://github.com/frederikhors)** — reported the
  JBIG2 blank-page symptom in a comment on
  [#332](https://github.com/yfedoseev/pdf_oxide/issues/332) and
  confirmed that both the JBIG2 fix and the earlier vertical-flip
  regression are resolved.

## [0.3.45] - 2026-05-07

### Fixed

- **CJK OTF (CFF) font subsetter corrupts glyph order
  ([#449](https://github.com/yfedoseev/pdf_oxide/issues/449))** —
  OTF fonts with CFF outlines (SFNT magic `OTTO`) were embedded as
  `FontFile2 / CIDFontType2` (the TrueType path), causing PDF readers
  to misparse the CFF data and render wrong glyphs. Writer now detects
  CFF magic post-subsetting and emits the correct PDF object graph:
  `FontFile3` (with `/Subtype /CIDFontType0C`) + `CIDFontType0` (no
  `CIDToGIDMap`).
- **`AwsLcProvider::verify_rsa_pkcs1v15` now fully implemented
  ([#475](https://github.com/yfedoseev/pdf_oxide/issues/475))** —
  Changed `SignatureVerifier::verify_rsa_pkcs1v15` to accept the raw
  message bytes (consistent with `verify_rsa_pss` / `verify_ecdsa`).
  Under the default `RustCryptoProvider` the hash is now computed
  inside the trait implementation. Under `AwsLcProvider` (FIPS) the
  new call path uses aws-lc-rs's `RSA_PKCS1_2048_8192_SHA{256,384,512}`
  verifiers — RSA-PKCS#1 v1.5 signature verification now works under
  FIPS instead of returning `SignerVerify::Unknown`.
- **`render_page_fit` produces images smaller than the requested box
  ([#480](https://github.com/yfedoseev/pdf_oxide/issues/480))** —
  Integer-DPI conversion via `floor()` lost up to 3 pixels from the
  constrained dimension (e.g. a 1040 px fit yielded 1037 px on Letter).
  The renderer now computes a float scale directly (`fit_px / page_pt`)
  and stores it in the crate-private `RenderOptions::scale_override`
  field, bypassing the DPI round-trip entirely. The constrained
  dimension is now exact for all integer pixel inputs. Reported by
  @gevorgter.

### Added

- **`legacy-crypto` compile-time feature flag (default-on)
  ([#230](https://github.com/yfedoseev/pdf_oxide/issues/230))** —
  New default-on Cargo feature that gates MD5 key-derivation and RC4
  cipher support for PDF Standard Security R≤4 documents. Downstream
  crates that must not load legacy cryptography can opt out with
  `default-features = false`; they will receive a clear
  `Error::InvalidPdf` instead of silently accepting RC4/MD5-encrypted
  PDFs. The `md-5` crate is now an optional dependency gated behind
  this feature. RC4 (pure Rust, no crate) is also disabled: both
  `RustCryptoProvider::rc4()` and `rc4_crypt_impl` are compiled out,
  and the provider returns `AlgorithmNotPermitted` at runtime when the
  feature is absent. Phase A of Issue #230.

### Changed

- **Stub parity gate for Python wheels
  ([#464](https://github.com/yfedoseev/pdf_oxide/issues/464))** —
  `rylai.toml` now uses `--features python` only (matching the released
  wheel) so generated `.pyi` stubs no longer include symbols from
  `office` or other optional features. A new CI step
  (`Verify stub symbol parity`) checks that every stub symbol exists in
  the installed wheel.
- **TypeScript 6 + @types/node 25 upgrade for JS bindings
  ([#438](https://github.com/yfedoseev/pdf_oxide/issues/438),
  [#440](https://github.com/yfedoseev/pdf_oxide/issues/440))** —
  JS dev dependencies bumped to TypeScript `^6.0.3` and `@types/node`
  `^25.6.0`. `tsconfig.json` gains `"types": ["node"]` (required by
  @types/node 25's ambient-global model) and `"ignoreDeprecations": "6.0"`
  (to acknowledge the TS6-deprecated `moduleResolution: node` — full
  migration to `node16` deferred until the import-path audit is done).

## [0.3.44] - 2026-05-05

> Pluggable cryptographic provider — FIPS 140-3 compliance for
> government / regulated deployments.

### Highlights

- **`pdf_oxide::crypto::CryptoProvider` trait** — new abstraction
  that decouples PDF encryption and signature paths from any one
  cryptography crate. Two providers ship out of the box:
  - **`RustCryptoProvider`** (default): pure-Rust stack as before
    (`sha2`, `aes`, `rsa`, `p256`, `p384`, `getrandom`, `md-5`,
    `sha1`). Permits every algorithm PDF specs reference, including
    the legacy MD5+RC4 path required by ISO 32000-1 R≤4 documents.
  - **`AwsLcProvider`** (opt-in via `--features fips`):
    backed by `aws-lc-rs`, FIPS 140-3 validated since 2024. Refuses
    MD5 / SHA-1-for-signing / RC4 with `Error::AlgorithmNotPermitted`
    and a clear remediation message.
- **Single source of randomness.** `src/encryption/algorithms.rs`'s
  former `SHA-256(uuid_v4 || timestamp_ns || …)` cascade is replaced
  with `crypto::active().random_bytes()` — under the default
  provider this is `getrandom::fill()` (OS entropy pool); under FIPS
  it's `aws_lc_rs::rand::SystemRandom`. Cryptographically suitable
  for AES-256 file keys and salts; auditable.
- **Closes [#236](https://github.com/yfedoseev/pdf_oxide/issues/236).**

### Architecture

Three sub-traits compose into `CryptoProvider`:

- `Hasher` — incremental hashing (`update` / `finalize`).
- `SymmetricCipher` — AES-128/256-CBC (PKCS#7 + no-padding) and RC4.
- `SignatureVerifier` — RSA-PKCS#1-v1.5, RSA-PSS, ECDSA P-256/P-384.

Plus an opaque `Signer` handle so HSM / PKCS#11 / Cloud KMS
backends can plug in via `SigningKeyMaterial` (which is
`#[non_exhaustive]` — future variants for HSM slots etc. are not
breaking changes).

The `is_legacy_allowed()` policy bit lets each provider declare
whether MD5 / SHA-1-sign / RC4 are permitted. PDF Standard Security
R≤4 documents are gated at `EncryptionHandler::new`: under a FIPS
provider they fail with a remediation message ("re-encrypt at R=6
or build pdf_oxide without the 'fips' feature so the default
'rust-crypto' provider stays active") rather than panic
deep inside the cipher path.

### Usage

```rust
use std::sync::Arc;
use pdf_oxide::crypto::{set_provider, AwsLcProvider};

set_provider(Arc::new(AwsLcProvider::new()))?;
let doc = pdf_oxide::PdfDocument::open("encrypted-r6.pdf")?;
```

See `docs/CRYPTO_PROVIDERS.md` for the algorithm coverage matrix,
custom-provider walkthrough (sovereign-jurisdiction algorithms,
HSMs), and the legacy-PDF policy table.

### CI

- New `fips` job in `.github/workflows/ci.yml` builds with
  `--features fips`, runs the 11-test AwsLcProvider suite
  including a `cross_provider_aes_compat` check that asserts the
  FIPS and rust-crypto AES paths produce byte-identical output, and
  enforces clippy `-D warnings` under the FIPS feature.

### Release

- New `.github/workflows/release-fips.yml` workflow (manually
  triggered) builds and publishes parallel FIPS distributions on
  every package index, all from the same Rust source compiled with
  `--features fips` so each binary contains only AWS-LC's
  FIPS-validated module:

  | Ecosystem | Package | Install |
  |---|---|---|
  | PyPI | `pdf_oxide_fips` | `pip install pdf_oxide_fips==0.3.44` |
  | npm | `pdf-oxide-fips` | `npm install pdf-oxide-fips@0.3.44` |
  | NuGet | `PdfOxide.Fips` | `dotnet add package PdfOxide.Fips --version 0.3.44` |
  | Go | `github.com/yfedoseev/pdf_oxide/go-fips` | `go get github.com/yfedoseev/pdf_oxide/go-fips@v0.3.44` |

  Platform matrix in v0.3.44 (every binding × every platform):

  | Platform | Python | npm | NuGet | Go |
  |---|:---:|:---:|:---:|:---:|
  | Linux x86_64 | ✅ | ✅ | ✅ | ✅ |
  | Linux aarch64 | ✅ | ✅ | ✅ | ✅ |
  | macOS x86_64 | ✅ | ✅ | ✅ | ✅ |
  | macOS arm64 | ✅ | ✅ | ✅ | ✅ |
  | Windows x86_64 | ✅ | ✅ | ✅ | ✅ |

  All distributions move in lockstep with the regular release —
  FIPS and default variants of the same release tag are byte-equal
  in their non-crypto code paths. Per-platform smoke tests in the
  workflow confirm the FIPS provider is reachable AND
  `crypto_use_fips()` (or equivalent) flips the active provider
  as expected — catches API mismatches before publishing.

  Why `pdf_oxide_fips` (underscore) for Python: PyPI normalizes
  hyphens / underscores to the same canonical form per PEP 503
  (`pip install pdf_oxide_fips` and `pip install pdf-oxide-fips`
  resolve to the same package). Using underscore in `pyproject.toml`
  makes the wheel filename and the `import pdf_oxide` path
  identical to the default distribution — only the package name
  differs.

  Why parallel distributions instead of `pip install pdf_oxide[fips]`:
  Python extras (PEP 508) can add Python dependencies but cannot
  swap the compiled `.so` baked inside a wheel. The industry
  pattern (cryptography, pyOpenSSL) ships separate FIPS
  distributions; we follow suit.

  Why a `go-fips` submodule path: Go modules are import-path-bound,
  so users pick at `go get` time:
  ```
  go get github.com/yfedoseev/pdf_oxide/go            # default
  go get github.com/yfedoseev/pdf_oxide/go-fips       # FIPS
  ```
  Both submodules re-export the same Go API; only the linked native
  static lib differs.

### Fixes

- **Restore `manylinux_2_28` glibc floor for Python wheels.** 0.3.42 and
  0.3.43 published only `manylinux_2_35` Linux glibc wheels because the
  release workflow ran `maturin build` directly on `ubuntu-latest`
  (Ubuntu 24.04, glibc 2.39), letting the runner's glibc set the wheel
  tag. That excluded Amazon Linux 2023 / AWS Lambda Python (glibc 2.34),
  RHEL 8, Ubuntu 20.04 and Debian 11 — pip rejected the wheel and fell
  back to a source build that OOM-killed `rustup-init` inside the Lambda
  build container. Reported by @potatochipcoconut on
  [PR #463](https://github.com/yfedoseev/pdf_oxide/pull/463#issuecomment-4376490292).
  Both `release.yml` (default wheels) and `release-fips.yml`
  (`pdf_oxide_fips` wheels) now build the Linux glibc wheels via
  `PyO3/maturin-action` inside the `manylinux_2_28` container, and a CI
  guard step fails the job if a `manylinux_2_28` wheel is not produced
  for either Linux target — preventing this regression from recurring.
  The 0.3.21 baseline (originally added in #284) is restored.

### Performance — `extract_pages_to_bytes` 12–54× faster

Extraction of page ranges from large PDFs is now bound by
serialisation work instead of redundant document rebuilds and tree
walks. Closes [#474](https://github.com/yfedoseev/pdf_oxide/issues/474),
reported by community contributor
[@potatochipcoconut](https://github.com/potatochipcoconut),
whose careful root-cause writeup (chunk-by-chunk timings, comparison
against PyMuPDF's `doc.select()`, and a profiling-grade reproduction
case from an AWS Lambda IDP pipeline) made this fix possible.

Measured on the public 1112-page / 38 MB *Artificial Intelligence — A
Modern Approach* corpus (`pdfs_slow2/`) on an idle laptop:

| Workload | 0.3.43 | 0.3.44 | Speedup |
|---|---|---|---|
| `extract_pages_to_bytes(0..300)` | 7301 ms / 36 MB out | **382 ms / 12 MB out** | **19×** + 3× smaller |
| `extract_pages_to_bytes(0..50)`  | 7983 ms / 36 MB out | **155 ms / 4 MB out**  | **51×** + 9× smaller |
| Sequential 23 × 50-page chunks   | ~3 min               | **1542 ms total**       | ~120× |

Extrapolating to the reporter's 12k-page / 50 MB document chunked
into five 3000-page slices: an AWS Lambda invocation that previously
timed out at 900 s after two chunks now finishes the entire
five-chunk batch in roughly 30 s.

#### Root causes

All in `src/editor/document_editor.rs` + `src/document.rs`:

1. **Triple full-document rewrite.** `extract_pages_to_bytes`
   serialised the whole doc, re-parsed the bytes, removed pages
   one at a time, and serialised again — three full passes when
   one would do. Replaced with a non-mutating in-place trimmed
   `page_order`, restored after the save (even on `Err`).
2. **Garbage collector walked the original page tree.** The
   trimmed `/Pages` dict was rebuilt locally inside
   `write_full_to_writer`, but `collect_reachable_ids()` started
   its BFS from the *unmodified* catalog and pulled in every
   dropped page's resources — so the output never shrank no
   matter how few pages were kept. Fixed by staging the trimmed
   `/Pages` dict in `modified_objects` before the save; the
   GC walker already prefers staged dicts over source.
3. **`get_page_ref(i)` in a 0..n loop is O(n²).** Each call walks
   the page tree from the root and stops at the i-th leaf, so
   collecting all n leaf refs walks 1 + 2 + … + n nodes. New
   helper `PdfDocument::all_page_refs()` does it in one DFS.
   The flat-tree common case (root `/Pages` whose `/Count`
   matches `Kids.len()`) reads the ref array straight out of
   `/Kids` without touching individual leaves at all.

The same n² loop pattern was lurking in four other call sites
on the reporter's hot path (their pipeline does PDF/A validate +
convert before the chunked extract). All five collapsed to a
single `all_page_refs()` call:

- `src/outline.rs` — `find_page_index` (O(n²) per outline
  entry → O(n³) on documents with bookmarks).
- `src/editor/document_editor.rs` line ~4275 — page-ref → index
  map for partial form-flatten.
- `src/editor/document_editor.rs` line ~4505 — same map for
  `get_form_fields()`.
- `src/compliance/validators.rs` — `validate_fonts`
  (`doc.validate_pdf_a('2b')`).
- `src/compliance/converter.rs` — per-page `/AA` strip
  (`doc.convert_to_pdfa('2b')`).

#### New API

Two additions, both directly requested by @potatochipcoconut in
#474; both available in Rust and Python (the other bindings can
be added on demand):

```python
# Batch extraction — same single-call efficiency, ergonomic for
# the chunked-for-OCR / chunked-for-S3 pattern.
chunks = doc.extract_page_ranges_to_bytes(
    [(0, 3000), (3000, 6000), (6000, 9000), (9000, 12000)]
)

# In-place selection — equivalent to PyMuPDF's doc.select(...).
# After this call, the document holds only the listed pages,
# in the order given. doc.save() / doc.save_to_bytes() then
# emit only those pages with garbage-collected resources.
doc.select_pages([1, 4, 7, 99])
```

#### Known limitation

PDFs whose `/Pages` root publishes shared `/Resources` used by
*all* leaf pages (typical of high-resolution book scans, atypical
of office documents with subset fonts) still produce full-size
chunk output: GC correctly preserves resources reachable from
kept pages, and a single shared resource pool stays reachable as
long as any kept page references it. The principled fix is
per-page resource sub-setting — parsing each kept page's content
stream to determine which fonts / XObjects are actually used and
emitting a minimal `/Resources` for that page. That is a feature,
not a bug fix, and is deferred from this release. The wall-clock
speedup (12–54×) holds regardless.

### Tests

- 5050 lib tests pass under `--features python,fips`
  (5039 default + 11 FIPS-only).
- 119 encryption tests still pass byte-equal post-rewire to the
  trait.
- 69 signatures tests still pass byte-equal post-rewire.
- Hash vectors validated against NIST FIPS 180-4 for SHA-256/384/512
  and RFC 1321 / 3174 for MD5 / SHA-1.
- New regression tests cover the issue #474 workflow:
  `test_extract_pages_chunked_sequential` (4 sequential chunks
  on the same `DocumentEditor`, source observably unchanged
  between calls), `test_extract_pages_non_sequential`
  (out-of-order indices `[3, 0, 4]`),
  `test_extract_page_ranges_to_bytes_batch`,
  `test_select_pages_in_place`, and
  `test_select_pages_out_of_range`.

### Known follow-ups (v0.3.45)

- **`AwsLcProvider` signing wiring** — signing calls are currently
  routed to `RustCryptoProvider`. Full AWS-LC signing integration
  lands in v0.3.45.
- **musllinux Python wheels for the FIPS variant** — FIPS musllinux
  wheels (Alpine / musl libc) require a musl-targeted
  `aws-lc-fips-sys` build; work in progress.

## [0.3.43] - 2026-05-03

> Cross-binding parity, WASI build target, and a basket of issue fixes.

### Highlights

- **`render_page_fit()` now ships in all five bindings** (Rust core +
  Python, Node.js / TypeScript, C#, Go). Picks the largest DPI such
  that both rendered dimensions fit inside a target pixel box,
  preserving aspect ratio. No more "what DPI hits 1024×768?" math
  on the caller's side. Fixes [#441](https://github.com/yfedoseev/pdf_oxide/issues/441),
  closes [#448](https://github.com/yfedoseev/pdf_oxide/issues/448).
- **Idiomatic page iteration parity across bindings.** Rust gets
  `page_indices()`, Python gets `.pages`, Node.js gets
  `[Symbol.asyncIterator]` (the sync `[Symbol.iterator]` was already
  there). C# `Pages` and Go `Pages()` were already shipped. Closes
  [#447](https://github.com/yfedoseev/pdf_oxide/issues/447).
- **WASI build target** — `cargo build --target wasm32-wasip1` now
  builds the lib cleanly on stable Rust. Unblocks @RALaBarge's
  external `pdf-oxide-wasi` stdin→stdout wrapper and any other
  consumer wanting to embed pdf_oxide in a sandboxed WASI runtime.
  CI now gates that the WASI build stays green. Closes
  [#214](https://github.com/yfedoseev/pdf_oxide/issues/214).
- **Spurious-table fix on dense word grids** — Roland's #405 lands
  via cherry-pick. A new `has_split_modal_column_groups` validator
  inspects the column co-occurrence graph across modal rows and
  rejects candidates whose populated columns split into two or more
  disconnected components — the signature of two adjacent text
  flows mis-clustered as one table. Composes cleanly with v0.3.42's
  `Table::is_real_grid` filter. Validated against the 86-PDF
  cross-build corpus: 888 / 888 byte-equal — zero observable change
  on common documents, the gate's value is in the safety net for
  adversarial cases.

### Fixes

- **#456** — `PdfDocument::open(path)` now populates `source_bytes`,
  unblocking `convert_to_pdf_a()`, the C FFI `pdf_document_get_source_bytes`,
  and any other API that re-reads the in-memory copy. Path-loaded
  documents previously got an empty `Vec<u8>` and hit
  `"Invalid PDF header: File is empty (0 bytes read)"` from the
  PDF/A converter. Reported by @potatochipcoconut on PR #445.
- **#451** — Standard14 PostScript fonts with no open-source
  equivalent (`Symbol`, `ZapfDingbats`) are now downgraded from
  hard `FontNotEmbedded` errors to a new `KnownUnembeddableFont`
  warning during PDF/A conversion. A document that's otherwise
  compliant no longer fails solely because of one symbolic font.
- **#395** — closed; verified the off-by-one C# `ExceptionMapper`
  fix in v0.3.38 actually resolves the reported `RenderPage` →
  `SignatureException [8500]`. Added a Rust regression test that
  opens @gevorgter's exact reproducer PDF and asserts `render_page`
  succeeds. The fixture is pinned in `pdf_oxide_tests`.
- **#462** — dropped the `scripts/modernize_stubs.py` post-processor
  and the `python_version = "3.8"` setting from `rylai.toml`. Rylai's
  default already emits PEP-585 / PEP-604 syntax with
  `from __future__ import annotations` at the top, so post-processing
  was duplicate work in opposite directions. Runtime support for
  Python 3.8/3.9 is unaffected — `.pyi` stubs are type-checker
  artifacts, never imported at runtime. Reported by @monchin with a
  clean diagnosis of the root cause.

### Behavior changes

- `PdfDocument::open(path)` now reads the file once into memory
  rather than streaming via `BufReader<File>`. The doc comment
  already promised "Reads the entire file into memory"; this makes
  it true. Memory usage on `open()` is now equivalent to
  `from_bytes(std::fs::read(path)?)`. Required by #456; the
  streaming reader was a partial optimisation no caller could rely
  on (every code path that touched `source_bytes` already required
  the in-memory copy).
- `PdfReader` enum collapsed to a single in-memory variant —
  removed unused `File` variant. `std::io::{Read, Seek, BufRead, …}`
  imports are no longer cfg-gated, which is what unblocked the
  wasm32-wasip1 build target.

### Dependencies

- Batch-applied 9 dependabot bumps onto `release/v0.3.43`:
  CI workflows (`golangci-lint-action` v7→v9, `setup-go` 5.5→6.4,
  `setup-node` 4.4→6.4, `github-script` SHA refresh,
  `scorecard-action` 2.4.0→2.4.3), Go (`testify` 1.8→1.11 — was
  declared but unimported, dropped entirely), JS (`rimraf` 5→6 —
  `@types/node` deferred to a follow-up after a TypeScript-strict
  shake-out), Python (`onnx` ≥1.14→≥1.19.1).
- The RustCrypto 0.8 stack (`pkcs8 0.11`, `spki 0.8`, `der 0.8`,
  `digest 0.11`, `crypto-common 0.2`, `block-buffer 0.12`) stays
  pinned — `rsa 0.10` and `p256/p384 0.14` are still RC upstream.
  See the existing pin note at `Cargo.toml:185-187`.

### Internal

- New `wasm32-wasip1` build smoke check in `.github/workflows/ci.yml`
  alongside the existing `wasm32-unknown-unknown` job.
- Regenerated SBOMs (`pdf_oxide_cli/sbom.cdx.json`,
  `pdf_oxide_mcp/sbom.cdx.json`) for 0.3.43.
- New regression tests:
  - `tests/test_issue_456_path_open_source_bytes.rs`
  - `tests/test_issue_447_page_indices.rs`
  - `tests/test_issue_395_render_page.rs`
- New unit tests on `compliance::converter::downgrade_known_unembeddable_fonts`.

### Validation

86-PDF stratified corpus comparison (academic, mixed, forms,
government, newspapers, theses, plus the three #211 fixtures), 888
sampled `(pdf, page, method)` triples across `extract_text`,
`to_plain_text`, `to_markdown`, `to_html`:

- v0.3.43 vs v0.3.42 — **888 / 888 byte-equal, zero deltas**
- v0.3.43 vs PyPI v0.3.41 — 860 equal, 28 reorder/de-dup, 0 real
  content losses (same profile as v0.3.42's regression report)

### Community contributors

This release exists because of the community. Special thanks to:

- **[@RolandWArnold](https://github.com/RolandWArnold)** — landed
  the spurious-table fix in [#405](https://github.com/yfedoseev/pdf_oxide/pull/405).
  After iterating away from an earlier density-gate framing, the
  shipped form is `has_split_modal_column_groups`: a connected-
  component check on the column co-occurrence graph across modal
  rows that flags two-flow grids the regular-row-ratio gate
  accepts. Roland's doc-comment explicitly flags it as a heuristic,
  making it easy to revisit later. The fix composes with v0.3.42's
  struct-tree-aware reading-order rewire without any merge conflict.
- **[@RALaBarge](https://github.com/RALaBarge)** — built an
  external WASI binary wrapper for pdf_oxide
  ([pdf-oxide-wasi](https://github.com/RALaBarge/pdf-oxide-wasi))
  and reported in [#214](https://github.com/yfedoseev/pdf_oxide/issues/214)
  that it required nightly Rust because of an internal
  `ceil_char_boundary` call. That call was already removed; this
  release fixes the second hidden blocker (cfg-gated `std::io`
  imports) and adds CI gating so the WASI target stays green.
- **[@gevorgter](https://github.com/gevorgter)** — flagged two
  rendering-area gaps: the C# binding's misleading
  `SignatureException` on `RenderPage` ([#395](https://github.com/yfedoseev/pdf_oxide/issues/395),
  fixed in v0.3.38, regression-guarded here) and the lack of a
  pixel-dimension render API ([#441](https://github.com/yfedoseev/pdf_oxide/issues/441),
  closed by `render_page_fit` shipping in all five bindings).
- **[@potatochipcoconut](https://github.com/potatochipcoconut)** —
  surfaced the `convert_to_pdf_a` failure on path-loaded documents
  while testing PR #445; the investigation traced it to the empty
  `source_bytes` field and produced the one-line fix in this
  release ([#456](https://github.com/yfedoseev/pdf_oxide/issues/456)).
- **[@monchin](https://github.com/monchin)** — pointed out
  ([#462](https://github.com/yfedoseev/pdf_oxide/issues/462)) that
  `scripts/modernize_stubs.py` was redundant work because rylai itself
  controls the typing flavour via its `python_version` setting, and
  noted that `office`/`barcodes`/`ocr` feature alignment between
  `rylai.toml` and the released wheel is worth a follow-up. The
  cleaner stub pipeline ships in this release.

## [0.3.42] - 2026-05-02

> Text-extraction reading-order rewire — fixes [#211](https://github.com/yfedoseev/pdf_oxide/issues/211)
> and closes the [#457](https://github.com/yfedoseev/pdf_oxide/issues/457) refactor.

### Highlights

- `extract_words` and `extract_text_lines` now honor the structure tree
  on tagged PDFs (per ISO 32000-1:2008 §14.7 / §14.8.2.3) instead of
  applying XY-Cut block partitioning. On the three #211 fixtures from
  pdfplumber's public test corpus this restores correct reading order
  for centered titles above body text (Quebec municipal minutes case)
  and stops splitting prose lines across phantom column gutters in
  form-style layouts (US child-welfare report case).
- Spurious markdown / HTML tables on form-style layouts (label-colon-
  value pairs) are gone — spatial table detection is now gated on a
  real-grid validator (≥2 rows × ≥2 cols, ≥50% of rows with at least
  two non-empty cells).
- New `include_artifacts` kwarg on `extract_words` /
  `extract_text_lines` (Python) gates the spec-correct behavior of
  excluding `/Artifact`-tagged content (running headers, footers,
  page numbers, watermarks; ISO 32000-1:2008 §14.8.2.2.1).
  **Default is `True`** — preserves pre-0.3.42 behavior so existing
  scripts don't lose content. Pass `include_artifacts=False` to
  opt into the spec-correct exclude. The default may flip in a
  future major release once the artifact-detection heuristic is
  hardened against false positives on docs whose body text recurs
  across pages.
- The default API surface is now knob-free: `region`,
  `word_gap_threshold`, `line_gap_threshold`, `profile` are deprecated
  on `extract_words` / `extract_text_lines` (Python). They still work
  but emit `DeprecationWarning`; they will move to a separate
  `extract_*_advanced` surface in a future release.
- ~6× faster on `extract_words` / `extract_text_lines` because the
  XY-Cut partition is no longer in the hot path.

### Fixes

- **#211 — `extract_words` / `extract_text_lines` produce wrong reading
  order on tagged PDFs.** Headings and prose lines that XY-Cut had
  moved out of position now appear where the document author marked
  them via the `/StructTreeRoot` MCID order. Reported by @ankursri494
  against pdfplumber's `pdf_structure.pdf`, `2023-06-20-PV.pdf`, and
  `150109DSP-Milw-505-90D.pdf` test fixtures.

### Behavior changes

- `extract_words(page)` / `extract_text_lines(page)` gain an
  `include_artifacts` kwarg (default `True` — backward-compatible).
  Pass `include_artifacts=False` to drop spans tagged as artifacts
  per ISO 32000-1:2008 §14.8.2.2.1. Word counts on documents with
  running headers / footers will decrease in that mode.
- Multi-column reading-order detection on untagged PDFs is now
  conservative: column-aware mode opts in only when the page
  presents ≥3 distinct vertical gutters, each ≥`median_char_width × 4`
  wide, with text on both sides. 1- and 2-column synthetic layouts
  default to row-aware top-to-bottom ordering — matches pdfplumber.
  Tagged multi-column PDFs are unaffected: they reach the column-aware
  path via the structure tree.
- `to_markdown(page)` / `to_html(page)` no longer emit `<table>` for
  layout-only structures detected by the spatial heuristic. Real
  tables (`<Table>` in the struct tree, or grids ≥2×2 with ≥50% of
  rows populating ≥2 cells) still render as tables.

### Refactor #457 — internal

- New `pdf_oxide::pipeline::page_reading_order(doc, page)` helper:
  single source of truth for canonical reading-order span sequence.
  Tagged + struct tree (no `/Suspects`) → walks the tree; otherwise
  → geometric top-to-bottom + y-tolerance. Companion variant
  `page_reading_order_no_artifacts` strips spans tagged as
  `/Artifact` for the spec-correct exclude case.
- `extract_words_with_thresholds` and
  `extract_text_lines_with_thresholds` delegate through the helper
  for the default code path (artifacts retained). New
  `extract_words_with_thresholds_no_artifacts` and
  `extract_text_lines_with_thresholds_no_artifacts` surfaces are
  available for the spec-correct artifact-excluded behavior. The
  `profile=Some(...)` path retains its previous XY-Cut behavior
  pending the planned removal of the `profile` kwarg.
- `GeometricStrategy` now defaults to row-aware top-to-bottom ordering;
  column-aware mode gated by the strict multi-column criterion above.
- `Table::is_real_grid()` introduced as the real-table validator;
  `extract_page_tables` filters the spatial heuristic's output through it.

### Validation

75-PDF stratified-sample corpus (academic, mixed, forms, government,
newspapers, theses, plus the three #211 fixtures) compared between
0.3.41 and 0.3.42 across all eight extraction methods on the first
3 pages of each PDF — 1592 comparisons total. **Zero content
regressions**: every word the baseline extracted is also extracted
by 0.3.42; only ordering / line-grouping / table-rendering changed.

### Dependencies

- **#453** — drop the unused `lzw` direct dependency. `LzwDecoder`
  already routed through `weezl` plus a custom fallback; the `lzw`
  crate was declared in `Cargo.toml` but never imported. Silences
  RUSTSEC-2020-0144 (unmaintained advisory) for downstream cargo-deny
  consumers as a side-effect.
- **#454 (partial)** — `cargo update` lockfile refresh: `fax 0.2.6 → 0.2.7`,
  `imageproc 0.26.1 → 0.26.2`, `js-sys` / `web-sys` `0.3.95 → 0.3.97`,
  `pdfium-render 0.9.0 → 0.9.1`, `rustls 0.23.39 → 0.23.40`,
  `wasm-bindgen` family `0.2.118 → 0.2.120`, plus 12 other transitive
  patch / minor bumps. The remaining major-version items in #454
  (RustCrypto 0.8 stack — `pkcs8 0.11`, `spki 0.8`, `der 0.8`,
  `digest 0.11`, `crypto-common 0.2`, `block-buffer 0.12`) stay
  pinned: `rsa 0.10` and `p256 0.14` / `p384 0.14` are still RC
  upstream as of 2026-04 (see the existing pin note in
  `Cargo.toml:185-187`).

### Community contributors

This release exists because of the community. Special thanks to:

- **[@ankursri494](https://github.com/ankursri494)** — reported
  [#211](https://github.com/yfedoseev/pdf_oxide/issues/211) with three
  carefully chosen pdfplumber-corpus fixtures (`pdf_structure.pdf`,
  `2023-06-20-PV.pdf`, `150109DSP-Milw-505-90D.pdf`) that isolate three
  distinct failure modes — wrong reading order on tagged PDFs, dropped
  document headings, and prose-line splits at form gutters. They also
  kept the issue alive through two rounds of "is this still broken on
  the latest version?", which forced the deeper investigation that
  ultimately exposed the architectural gap behind #457. Without that
  persistence and that specific repro set, this rewire would not have
  shipped.
- **[@lingcoder](https://github.com/lingcoder)** — flagged the
  unmaintained `lzw` advisory in
  [#453](https://github.com/yfedoseev/pdf_oxide/issues/453) with a
  precise pointer to RUSTSEC-2020-0144 and the `weezl` migration
  path; the investigation surfaced that the dep was unreferenced
  entirely, turning it into a one-line cleanup.

## [0.3.41] - 2026-04-29

> Real PDF/A conversion, LaTeX symbolic-font glyph rendering fix, and
> image deduplication — all 7 bindings.

### Community contributors

This release exists because of the community. Special thanks to:

- **[@FireMasterK](https://github.com/FireMasterK)** — reported
  [#307](https://github.com/yfedoseev/pdf_oxide/issues/307) with a
  precise reproduction case: a LaTeX-generated PDF where accented characters
  and ligatures (ú, á, fi) rendered as blank gaps across all pages. The report
  identified the exact document class (DC/EC TrueType fonts with Mac Roman
  cmap, no `/Encoding` dict), which made the root cause in
  `render_cid_direct()` straightforward to isolate and fix.

- **[@sparkyandrew](https://github.com/sparkyandrew)** — followed up on
  [#425](https://github.com/yfedoseev/pdf_oxide/issues/425) with
  [#443](https://github.com/yfedoseev/pdf_oxide/issues/443), noticing that
  the output PDF was 2.32 MB when the two source images summed to under
  1.6 MB — even after the #425 image-pipeline fix. That single observation
  pinpointed the missing XObject deduplication: the same image data encoded
  twice produced two independent compressed streams. Fixed.

- **[@potatochipcoconut](https://github.com/potatochipcoconut)** —
  [#418](https://github.com/yfedoseev/pdf_oxide/issues/418), the original
  PDF/A binding-completeness report that drove the full implementation in
  [#442](https://github.com/yfedoseev/pdf_oxide/issues/442).
  `convert_to_pdf_a()` existed in Rust but was a no-op: it recorded actions
  and returned success while leaving the document bytes untouched. The report
  surfaced this silently-broken state across all seven bindings.

- **[@nickpetrovic](https://github.com/nickpetrovic)** — filed
  [#444](https://github.com/yfedoseev/pdf_oxide/issues/444) with a
  precise four-row reproduction table showing ligature glyphs in subset
  Calibri fonts decoded to wrong Unicode codepoints (`ti`→`O`, `tf`→`[`,
  `ft`→`e`). The report included the exact PDF and the per-font-subset
  mapping failures, which led directly to the ICCBased color-space warn
  spam fix and the rowspan-label reading-order scramble fix.

- **[@RubberDuckShobe](https://github.com/RubberDuckShobe)** — reported
  [#450](https://github.com/yfedoseev/pdf_oxide/issues/450): any PDF
  containing a PNG with an alpha channel showed a diagonal stripe through
  the image. A minimal reproduction confirmed the bug was reproducible
  across Acrobat, Preview, and browser PDF viewers. The report made the
  scope unambiguous — every image with transparency was affected — and
  led directly to the missing `DecodeParms` fix in `build_soft_mask_dict()`.

- **[@truffle-dev](https://github.com/truffle-dev)** — first code
  contribution to the project: completed the CLI output-path fix for
  [#412](https://github.com/yfedoseev/pdf_oxide/issues/412) in
  [#452](https://github.com/yfedoseev/pdf_oxide/pull/452). The original
  audit in #412 covered all 11 CLI commands with exact line references and
  two proposed design options; the PR was clean on first submission. Picks
  up the four commands (`crop`, `decrypt`, `delete`, `reorder`) missed by
  the earlier partial fix, and also enforces `-o/--output` for `merge`
  instead of silently defaulting to the first input's directory.

### Scope at-a-glance

- **Real PDF/A conversion** — XMP metadata stream, `pdfaid:part`/`conformance`
  identification, OutputIntents (sRGB), language tag, JavaScript removal;
  all 7 bindings (#418, #442).
- **Symbolic TrueType glyph rendering** — non-ASCII bytes (ú=0xFA, á=0xE1,
  fi=0x85) in DC/EC-style LaTeX fonts with Mac Roman cmap no longer
  suppressed as spaces (partially fixes #307; follow-up cases reported
  by FireMasterK on 2026-04-29 remain open).
- **Image XObject deduplication** — same image embedded twice no longer
  re-encoded as two separate compressed streams; PDF size matches the
  sum of source images (#443).
- **Diagonal-line artifact in transparent images fixed** — missing
  `DecodeParms` in the soft-mask XObject caused a visible diagonal stripe
  in any PNG with an alpha channel (#450).
- **Barcode SVG generation** — `pdf_barcode_get_svg` no longer returns
  `ERR_UNSUPPORTED`; generates real SVG for all 8 barcode types including
  QR (#421).
- **CLI output routing** — `crop`, `decrypt`, `delete`, and `reorder` now
  write default output beside the input file instead of the current working
  directory; `merge` now requires `-o/--output` and errors up front instead
  of silently defaulting to the first input's directory. Completes #412.

### Real PDF/A conversion (#418, #442)

`convert_to_pdf_a()` previously recorded conversion actions and returned
success, but the document bytes were unchanged — the XMP metadata stream
was constructed in memory and then discarded. This release rewrites the
conversion core end-to-end:

- **XMP metadata stream** — a standards-compliant XMP packet is serialised
  and written as an indirect object, then wired into the document catalog as
  `/Metadata`. `pdfaid:part` and `pdfaid:conformance` are set per level:
  A1b → `1/B`, A2b → `2/B`, A2u → `2/U`, A3b → `3/B`.
- **OutputIntents** — a `GTS_PDFA1` output intent referencing sRGB is
  injected when none is present. Idempotent: a second call detects the
  existing intent and does not duplicate it.
- **Language tag** — `/Lang` is written to the catalog when the validator
  raises `MissingLanguage`.
- **JavaScript removal** — `/Names/JavaScript` entries are stripped when
  present.
- **Source bytes patched** — `doc.source_bytes` is updated in-place; the
  document is immediately re-parseable after conversion.
- **Font embedding** (`rendering` feature) — `embed_font()` now resolves the
  14 standard PDF Type1 PostScript names (Helvetica, Courier, Times-Roman, …)
  to the metrically-equivalent URW Base 35 open-source fonts shipped by default
  on Linux (`Nimbus Sans`, `Nimbus Mono PS`, `Nimbus Roman`). With
  `--features rendering` all B-level PDFs convert to **0 remaining errors**,
  including `FontNotEmbedded`. Three bugs were fixed in the embedding pipeline:
  - `try_fix_error` dedup applied to error codes, so only the first
    `FontNotEmbedded` error was processed; remaining fonts were skipped — fixed
    to dedup per-error-code for non-font errors only.
  - `write_full_to_writer` wrote font objects from the original source instead
    of preferring staged `modified_objects` — fixed to use the same priority
    order as the general object sweep.
  - `add_structure()` only added `/StructTreeRoot` but not `/MarkInfo /Marked
    true`; the validator requires both for PDF/A-\*a conformance — fixed.

**Test coverage** — 17 new end-to-end roundtrip tests in
`tests/test_pdfa_roundtrip.rs` verify every fixable scenario
(validate → convert → validate). The `showcase_pdfa_conversion` CI example
is rewritten to assert correctness and panics on any regression.

All seven bindings expose the updated function:

| Binding | API |
|---------|-----|
| Rust    | `convert_to_pdf_a(&mut doc, PdfALevel::A2b)?` |
| Python  | `pdf_oxide.convert_to_pdf_a(doc, "A2b")` |
| WASM    | `convertToPdfA(doc, "A2b")` |
| C FFI   | `pdf_oxide_convert_to_pdf_a(doc, level, &out)` |
| C#      | `Compliance.ConvertToPdfA(doc, PdfALevel.A2b)` |
| Go      | `compliance.ConvertToPdfA(doc, compliance.PdfALevelA2b)` |
| Node.js | `compliance.convertToPdfA(doc, "A2b")` |

### Symbolic TrueType glyph rendering fix (#307)

LaTeX-generated PDFs using DC/EC fonts (`Dcr10`, `Dcsl10`, etc.) embed
symbolic TrueType fonts with these characteristics:

- `/Flags` has the symbolic bit set (bit 3 = 4)
- No `/Encoding` dictionary
- Mac Roman format-0 cmap (platform 1, encoding 0): byte code → glyph ID
- No Windows Unicode cmap

pdf_oxide correctly routes these through the `render_cid_direct()` path,
which resolves each content-stream byte to a glyph ID via the Mac Roman
cmap. The bug was one line in the space-detection guard:

```rust
// Before — bytes without a Unicode mapping fell through to unwrap_or(' ')
let char_at_pos = char_str.chars().next().unwrap_or(' ');
if char_at_pos.is_whitespace() { /* skip draw */ }
```

Any byte whose Unicode mapping returned `None` — including ú (0xFA → GID 85),
á (0xE1 → GID 83), and fi (0x85 → GID 75) — was treated as a space, so the
`is_whitespace()` guard blocked glyph drawing entirely.

```rust
// After — '\0' is not whitespace; GID ≠ 0 glyphs are drawn correctly
let char_at_pos = char_str.chars().next().unwrap_or('\0');
```

Verified pixel-perfect against Poppler and MuPDF on the #307 reproduction
PDF. Regression-tested across 69 PDFs (120 page comparisons) — zero
regressions in rendering, plain text, Markdown, and HTML extraction.

### Text extraction fixes (#444)

Two issues surfaced while investigating #444 (Calibri ligature mis-mapping,
which is an upstream macOS Quartz PDF producer bug with no fix possible on
our side):

**ICCBased color space warn spam** — PDF producers that register ICCBased
profiles under user-defined names (e.g. `Cs1`, `Cs2`) caused the text
extractor to fire a `WARN` log on every `sc`/`SC`/`scn`/`SCN` operator
that used such a name. The catch-all `_` branch in the color-space handler
did not know how to handle named references, so it logged and left the
color unchanged. The fix: apply a component-count fallback in that branch
(1 component → gray, 3 → RGB, 4 → CMYK) and demote the log to `DEBUG`.
Affected PDFs with large amounts of colored text (like typical Office
documents) emitted 96+ spurious warnings per page; now silent.

**Text span reading-order scrambling** — `reorder_rowspan_labels`, a
function that promotes vertically-centered table row labels to sort at
the top of their row block, was incorrectly activating on single-column
prose documents (resumes, reports). It identified spans at rightward X
positions as a "sparse column" and promoted them to wrong Y coordinates,
causing line-continuation text like `"to assess technical needs and"` or
`"-making."` to appear before the earlier line they followed.

Root cause: the label-candidate filter did not exclude spans whose Y-band
already appears in the dense column. Genuine rowspan labels are vertically
*between* data rows, so their Y-band is absent from the dense column.
Line-continuation spans share the Y-band of the main column text and must
not be treated as labels. The fix adds that exclusion:

```rust
// Before — any sparse-column span in the data Y range
y > data_bot && y < data_top

// After — additionally exclude spans that align with a dense-column row
y > data_bot && y < data_top && !dense_bands.contains(&band_of(y))
```

The original rowspan-label behavior for actual table layouts (CJK lab
reports, mixed-column tables) is preserved; the existing test confirms
that genuine between-row labels are still promoted correctly.

### Image XObject deduplication (#443)

When the same image data was passed to `page.image()` or `from_bytes()` on
multiple pages, pdf_oxide encoded it as independent XObjects — each carrying
the full compressed pixel data. A 760 KB PNG embedded twice contributed
1.52 MB instead of 760 KB; the #443 reproduction produced 2.32 MB from
images totalling under 1.6 MB.

The fix hashes the normalised stream bytes **after** calling
`image_content_to_xobject_stream()`. Hashing before normalisation failed
across API paths: an image supplied via `page.image()` (which accepts raw
file bytes and decodes them internally) and the same image supplied via
`ImageContent::from_bytes()` produced different pre-encoding byte strings but
identical post-normalisation compressed streams. Hashing after normalisation
ensures the key is stable regardless of which API path the caller used. The
key is `(hash, byte_length)` over the compressed pixel data; if a matching
entry is already registered in the document's XObject map, the existing
reference is reused and no new stream is written.

### Diagonal-line artifact in images with transparency (#450)

PDFs with PNG images that have an alpha channel displayed a diagonal stripe
across the image when opened in Acrobat, Preview, and most other viewers.

Root cause: `compress_image_data()` prepends a PNG None-filter byte (`0x00`)
before every scanline before Flate-compressing the pixel data. This is
required by `FlateDecode` with `DecodeParms/Predictor=15`. The main image
XObject carried the correct `DecodeParms` dictionary — but `build_soft_mask_dict()`,
which builds the `/SMask` XObject for the alpha channel, emitted no
`DecodeParms` at all. Viewers therefore decompressed the raw Flate stream,
then treated the leading `0x00` filter byte of each row as an alpha pixel,
shifting every row one byte to the right. The cumulative horizontal offset
over hundreds of rows appears as a diagonal stripe.

Fixed by adding the same `DecodeParms` dictionary to the soft-mask stream:

```
DecodeParms { Predictor=15, Colors=1, BitsPerComponent=8, Columns=<width> }
```

Reported by **[@RubberDuckShobe](https://github.com/RubberDuckShobe)** in
[#450](https://github.com/yfedoseev/pdf_oxide/issues/450). Any PDF built with
`page.image()` or `ImageContent::from_bytes()` where the source PNG has an
alpha channel was affected; the fix is purely in the soft-mask stream header
and does not change pixel data.

### Barcode SVG generation (#421)

`pdf_barcode_get_svg` was a stub returning `ERR_UNSUPPORTED`. Two root
causes were blocking a real implementation:

1. **Format sentinel collision** — `pdf_generate_qr_code` stored
   `FfiBarcodeImage.format = 0`, the same value as `pdf_generate_barcode`
   with `format = 0` (Code128). The `get_svg` function had no way to
   distinguish QR from Code128. Fixed: QR codes now use the internal
   sentinel value `100` (outside the 0–7 range of 1D barcode types); the
   public `pdf_barcode_get_format` return value for QR codes changes from
   `0` to `100` accordingly.

2. **Missing SVG rendering path** — `barcoders` 2.0 ships
   `barcoders::generators::svg::SVG` (enabled by default via `features =
   ["svg"]`), so no new dependency was required. For 1D barcodes, the
   encoding step is now factored into a private `encode_1d` helper shared
   by both `generate_1d` (PNG) and the new `generate_1d_svg` (SVG). For
   QR codes, `generate_qr_svg` rebuilds the code matrix from
   `qrcode::QrCode::to_colors()` and emits a compact inline SVG with
   `<rect>` elements — no raster stage.

`pdf_barcode_get_svg` now returns a valid SVG string for all supported
barcode types (Code128, Code39, EAN-13, EAN-8, UPC-A, ITF, Code93,
Codabar, QR) when the `barcodes` feature is enabled.

### CLI output routing (#412, #452)

A previous partial fix (commit `9dd94c0`) introduced `output_beside()` /
`output_dir_beside()` helpers and converted five commands (`watermark`,
`compress`, `flatten`, `rotate`, `split`). Four binary-output commands
were missed and continued resolving the default output path relative to
the current working directory:

- **`crop`** — now writes `<stem>_cropped.pdf` beside the input file.
- **`decrypt`** — now writes `<stem>_decrypted.pdf` beside the input file.
- **`delete`** — now writes `<stem>_deleted.pdf` beside the input file.
- **`reorder`** — now writes `<stem>_reordered.pdf` beside the input file.

**`merge`** previously silently defaulted to writing `merged.pdf` in the
directory of the first input file when `-o/--output` was omitted. This
silent fallback was the riskiest behavior in the CLI: callers who expected
output beside a specific file got a surprise in a potentially unrelated
directory. `merge` now requires `-o/--output` and exits with a clear error
message if it is missing.

No library code was changed — all five files are in `pdf_oxide_cli`.

---

## [0.3.40] - 2026-04-27

> Image rendering fixes, dashed stroke + streaming table batch, digital
> signature verification, binding completeness sweep, security hardening,
> dependency freshness, and a new clean image API — all 7 bindings.

### Community contributors

This release exists because of the community. Special thanks to:

- **[@sparkyandrew](https://github.com/sparkyandrew)** — six detailed bug
  reports (#382, #385, #386, #397, #401, #425) that drove the CJK font
  subsetter, encryption, font-name handling, and now the image rendering
  overhaul. Every report came with a reproduction case. Issue #425 specifically
  identified four separate rendering bugs and raised the API design question
  that led to `ImageContent::from_bytes()` and the new `image()` method across
  all bindings.

- **[@potatochipcoconut](https://github.com/potatochipcoconut)** — three
  well-targeted reports (#409, #416, #417) that directly drove the manylinux
  glibc fix, the OCR wheel fix, and the discovery of the missing in-memory
  encrypted save API. Terse, precise, actionable every time.

### Scope at-a-glance

- **Image rendering** — four bugs fixed in PNG/JPEG embed path (#425).
- **New image API** — `ImageContent::from_bytes()` + plain `image()` on all bindings; no pixel dims needed (#425).
- **Dashed stroke + streaming table batch** — `StrokeRectDashed`/`StrokeLineDashed` + `StreamingTable` bounded-batch API across all 7 bindings (#400).
- **Digital signature verification** — real RSA-PSS / ECDSA / TSA cryptographic checks (#420).
- **Binding completeness** — encrypted bytes (#423), barcode via C FFI (#421), Node.js validation (#424) and page extraction (#384), Python/Go `convert_to_pdfa` (#418/#419).
- **Platform fixes** — Python glibc 2.34 compat (#416), OCR wheels (#417), WASM rendering (#422), CLI output path.
- **Security & hygiene** — unsafe audit, dep freshness, SLSA provenance, SBOM, CodeQL, DCO (#415).

### Image rendering fixes (#425)

This release closes the image-rendering bugs reported in
[#425](https://github.com/yfedoseev/pdf_oxide/issues/425) by
[@sparkyandrew](https://github.com/sparkyandrew). Four bugs, all in the same
family of incorrect assumptions in `image_handler.rs` / `pdf_writer.rs`:

- **PNG color corruption** (`Predictor=15` mismatch) — `FlateDecode` with
  `DecodeParms/Predictor=15` promises PNG-style per-scanline filter bytes. The
  encoder was compressing raw pixels without prepending the required `0x00`
  (None-filter) byte before each row; viewers applied PNG unfiltering to raw
  data, corrupting every pixel. Fixed: `compress_image_data()` now prepends one
  `0x00` per scanline before Flate compression.

- **Blank PNG via `ImageContent::new()`** — `image_content_to_xobject_stream()`
  assumed `data` was already decoded pixel bytes. Passing raw PNG file bytes
  caused the PNG header to be treated as pixels — blank / garbage output. Fixed:
  magic-byte detection (`89 50 4E 47`) routes raw bytes through
  `ImageData::from_png()`.

- **JPEG zoom / wrong dimensions** — same root cause; JPEG file bytes were not
  routed through `ImageData::from_jpeg()`, so the pixel dimensions stored in the
  XObject were wrong. Fixed by the same `FF D8` magic-byte detection.

- **Soft-mask (alpha) lost** — PNG transparency was discarded when raw bytes were
  passed through `ImageContent::new()`. The new auto-detect path correctly
  threads the alpha channel through to the PDF `/SMask` XObject.

### New image API — `from_bytes()` and `image()` (#425)

The bug report also identified a legitimate API design problem: every other PDF
library (ReportLab, fpdf2, iText, PDFBox, PDFKit, printpdf, Prawn) auto-detects
pixel dimensions from the image header — users only specify where the image
appears on the page. `ImageContent::new()` required passing `width` and `height`
explicitly, which callers typically had to look up from a separate decode step.

```rust
// Before — pixel dims required even though the library could read them itself
let img = ImageContent::new(bbox, ImageFormat::Png, raw_bytes, width, height);

// After — just bytes + on-page display rect; everything else auto-detected
let img = ImageContent::from_bytes(bbox, raw_bytes)?;
```

`from_bytes()` detects JPEG/PNG by magic number and reads `width`, `height`,
`color_space`, `bits_per_component`, and the soft-mask channel from the image
header. A plain `image()` method (no accessibility wrapper) was also missing
from Go, C#, and Node.js — added to all three:

| Binding | Method |
|---------|--------|
| Rust    | `ImageContent::from_bytes(bbox, data)?` |
| Go      | `page.Image(bytes, x, y, w, h)` |
| C#      | `page.Image(bytes, x, y, w, h)` |
| Node.js | `page.image(bytes, x, y, w, h)` |
| Python  | `page.image_from_bytes(bytes, x, y, w, h)` *(pre-existing)* |
| WASM    | `page.image_from_bytes(bytes, x, y, w, h)` *(pre-existing)* |

Use `imageWithAlt` / `ImageWithAlt` for PDF/UA-1 accessible figures and
`imageArtifact` / `ImageArtifact` for decorative images.

### Dashed stroke + streaming table batch (#400)

Two `FluentPageBuilder` additions shipping across all 7 bindings:

- **`stroke_rect_dashed` / `stroke_line_dashed`** — stroke a rectangle or line
  with an explicit dash pattern (`&[f32]` on/off lengths + phase) and RGB colour.
  Complements the existing solid `stroke_rect` / `stroke_line`.

- **`StreamingTable` bounded-batch API** — `set_batch_size(n)`, `pending_row_count()`,
  `batch_count()`, `flush()` — lets callers control how many rows accumulate in
  memory before being flushed to the PDF content stream. Useful when streaming
  very large tables from a source that itself has natural chunk boundaries.

Both surfaces are available in Rust, Python, WASM, Go, C#, and Node.js /
TypeScript. New `examples/*/09-new-features/dashed_stroke/` examples ship in
all four binding example directories.

### Digital signature verification (#420)

`SignatureInfo.verify()` now performs real cryptographic verification instead of
returning a stub result:

- **RSA-PSS** and **RSA-PKCS#1 v1.5** — verified against the embedded
  certificate public key via the `rsa` + `sha2` crates.
- **ECDSA (P-256 / P-384)** — verified via the `p256` / `p384` crates.
- **TSA timestamp** (`Timestamp.verify()`) — full RFC 3161 countersignature
  verification: CMS structure, signer certificate, and TSTInfo hash match.

### Binding completeness sweep

Several APIs present in the Rust core and some bindings were missing from
others. All are now consistent across all 7 bindings:

- **In-memory encrypted save** (#423) — `PdfDocument.to_bytes_encrypted(user_pw, owner_pw)` saves with AES-256 encryption directly to `bytes` / `Buffer` / `Vec<u8>` without touching disk. Available in Python, Node.js, C#, Go, and the C FFI. Driven by [@potatochipcoconut](https://github.com/potatochipcoconut) in #409.

- **Barcode via C FFI** (#421) — `pdf_add_barcode_to_page()` embeds a generated barcode PNG onto a page at a given rect. Previously the function returned `ERR_UNSUPPORTED`; it now calls the new `DocumentEditor::add_image_bytes_to_page()` helper internally. C FFI only in this release — Go and C# wrappers are follow-up work.

- **PDF/A, PDF/X, PDF/UA validation on Node.js** (#424) — `PdfDocument.validatePdfA()`, `.validatePdfX()`, `.validatePdfUA()` now available in the Node.js binding, matching Python, Go, C#, WASM, and Rust.

- **Page extraction in Node.js** (#384) — `DocumentEditor.extractPagesToBytes(pageIndices)` splits a multi-page PDF into per-chunk `Buffer` objects entirely in memory, no temp files needed.

  ```js
  const chunk = editor.extractPagesToBytes([0, 1, 2]); // → Buffer
  ```

- **PDF/A conversion** (#418/#419) — `PdfDocument.convert_to_pdfa(output_path, level)` exposed in Python; `pdf_convert_to_pdfa()` C FFI + Go `ConvertToPdfA()`.

### Platform fixes

- **Python glibc 2.34 compatibility** (#416) — LLVM emits `__memcmpeq` (a
  glibc 2.35 symbol) in some optimised builds; wheels built against glibc 2.35
  failed to load on Amazon Linux 2023 (glibc 2.34) and similar systems. Fixed by
  adding a `global_asm!` weak-symbol alias in `src/lib.rs` that maps
  `__memcmpeq` → `memcmp`. This works with both GNU ld and lld (unlike
  `--defsym` which lld rejects for PLT-resolved symbols). Reported
  by [@potatochipcoconut](https://github.com/potatochipcoconut).

- **Python OCR wheels** (#417) — published wheels omitted the `ocr` feature, so
  `pip install pdf-oxide[ocr]` installed silently but failed at runtime. Wheels
  now compile with `--features ocr`; ORT library path auto-detected on import.
  Reported by [@potatochipcoconut](https://github.com/potatochipcoconut).

- **WASM rendering** (#422) — `wasm-pack` builds were missing the `rendering`
  feature flag, producing blank page images. All WASM targets now build with
  `--features rendering`.

- **CLI binary output path** — `pdf-oxide render`, `pdf-oxide thumbnail`, and
  other commands that produce binary output were writing next to the working
  directory instead of next to the input file when no explicit output path was
  given. Fixed.

### Security & hygiene (#415)

- `#[forbid(unsafe_code)]` on all modules that have no FFI business being
  unsafe; remaining unsafe consolidated into audited FFI helpers with
  `handle_mut!` / `handle_ref!` macros.
- `lazy_static` replaced with `std::sync::OnceLock` throughout.
- `cargo update` dep freshness sweep; lock file refreshed.
- `cargo-geiger` unsafe audit + `cargo-outdated` dependency check added to CI
  (both run monthly).
- CI: action SHAs pinned, OIDC publish, SLSA provenance level 3, SBOM
  (CycloneDX), OpenSSF Scorecard, CodeQL static analysis, DCO enforcement.
- Dependabot configured for all three ecosystems (`cargo`, `npm`, `github-actions`).
- SPDX licence headers added to source files; `CODEOWNERS` and `CONTRIBUTING`
  (DCO) added.

---

## [0.3.39] - 2026-04-23

> Tables (streaming + buffered), PDF/UA-1, digital signing (CMS/PKCS#7),
> AcroForm flatten, interior-mutability thread safety, encryption +
> UTF-8 encoding fixes, L4 font cache, and `to_bytes` — all 7 bindings.

### Scope at-a-glance

v0.3.39 originally shipped as a single release themed around table
generation (issue #393). Mid-release we expanded the scope to close
the broader post-#393 programmatic-builder gap audit
(docs/v0.3.39/design/builder_gaps_plan.md, 26 items in 4 tiers).
The release now delivers:

- **Bundle C** — shape primitives (`circle`, `ellipse`, `polygon`, `arc`, `bezier_curve`) + dash patterns on `LineStyle`.
- **Bundle A** — image placement (`image_from_file` / `_from_bytes` / `_with`) + 2D affine transforms (`rotated`, `scaled`, `translated`, `with_transform`; v0.3.39 scope text-only, path/image/table in v0.3.40).
- **Bundle B** — document outline (`bookmark`, `bookmark_tree`), page labels (`with_page_labels`), ToC auto-generator (`insert_toc`).
- **Bundle D (partial)** — `list_box` form widget, fluent field metadata (`required` / `read_only` / `tooltip`), page `tab_order` (`TabOrder::{Row, Column, Structure}`).
- **Bundle E + F (research)** — RFCs for rich-text accumulator (`docs/v0.3.39/design/e_rich_text_rfc.md`) and PDF/UA compliance (`docs/v0.3.39/research/e_pdf_ua_compliance.md`). Implementation deferred to v0.3.40 (#400).
- **Bundle D (deferred)** — signature_field widget, barcode-bound fields, JS-action field validation, calculated fields, XFA write-side → v0.3.40 (#400).

### DocumentBuilder tables (original #393 scope)

This release closes issue [#393](https://github.com/yfedoseev/pdf_oxide/issues/393).
Users who previously had to build giant HTML strings or drop to PdfSharp
(the .NET community's canonical pain point — MigraDoc halts around 30 k
rows with an O(rows²) autosize) can now stream tables of arbitrary size
directly through `DocumentBuilder`. The release gate is a criterion
benchmark that proves linear scaling from 1 k → 30 k rows; see the
"Release gate" section below.

Design + research anchors live under `docs/v0.3.39/`:

- `research/a_table_api_landscape.md` — survey of 20 OSS PDF libraries across 6 ecosystems.
- `research/b_scalable_layout_algorithms.md` — why MigraDoc fails at 30 k rows + how to not repeat it.
- `research/c_api_ergonomics.md` — idiomatic API shape per binding.
- `research/d_builder_gap_analysis.md` — primitives we were missing to make tables compose.
- `design/393_tables_decision.md` — synthesis + scope split v0.3.39 / v0.3.40.

### Two table surfaces, one type vocabulary

- **Buffered `Table`** (`page.table(Table::new(rows).with_header_row()...)`) — takes the full row matrix, supports `colspan` / `rowspan` / rich per-cell styling, splits at row boundaries, emits `ContentElement`s so the v0.3.38 subsetter continues re-keying CJK glyph IDs. Best for tables under ~1 k rows.

- **Streaming `StreamingTable`** (`page.streaming_table(StreamingTableConfig::new().column(...).column(...))`) — row-at-a-time, `TableMode::Fixed` only (explicit widths, zero look-ahead), O(cols) persistent memory, auto page-break with repeat-header. Best for 1 k → ∞ rows. Solves the motivating MigraDoc 30 k-row failure directly.

```rust
use pdf_oxide::writer::{
    CellAlign, DocumentBuilder, StreamingColumn, StreamingTableConfig,
};

let mut doc = DocumentBuilder::new();
let page = doc.letter_page().font("Helvetica", 10.0).at(72.0, 720.0);

let mut t = page.streaming_table(
    StreamingTableConfig::new()
        .column(StreamingColumn::new("SKU").width_pt(72.0))
        .column(StreamingColumn::new("Item").width_pt(240.0))
        .column(
            StreamingColumn::new("Qty")
                .width_pt(48.0)
                .align(CellAlign::Right),
        )
        .repeat_header(true),
);

for record in huge_dataset {       // never materialised
    t.push_row(|r| {
        r.cell(&record.sku);
        r.cell(&record.name);
        r.cell(record.qty.to_string());
    })?;
}
t.finish().done();
```

Both surfaces ship with idiomatic per-binding wrappers (Python, WASM, C#, Go, Node/TS). See each binding's README / guide for the native shape.

### Four supporting `FluentPageBuilder` primitives

Shipped alongside tables because a credible table API needs them:

- `measure(&str) -> f32` — text width in points for the current font/size. Pure query; used to pick explicit column widths.
- `text_in_rect(rect, text, align)` — wraps `text` to `rect.width`, aligns each line horizontally per `TextAlign::{Left, Center, Right}`. Cursor is deliberately NOT advanced — the rect has its own geometry. Finally honours `TextConfig.align` which was a dead field for seven releases.
- `stroke_rect(x, y, w, h, LineStyle)` + `stroke_line(p1, p2, LineStyle)` — stroke with explicit width + RGB colour. Previously `rect()` and `line()` only stroked at 1 pt black. `LineStyle { width, color }` is the new public type.
- `remaining_space()` + `new_page_same_size()` — the missing page-break signal. `remaining_space()` returns vertical points from cursor to the bottom margin; `new_page_same_size()` commits pending annotations and opens a fresh page with the same dimensions + carried `text_config`.

### Release gate

A criterion benchmark at `benches/streaming_table_scaling.rs` runs `StreamingTable` at 1 k / 5 k / 10 k / 30 k rows. Local numbers on the contributor machine (`--quick`):

| Size   | Time     | Throughput      |
|--------|----------|-----------------|
| 1 000  | 21.7 ms  | 46.0 K rows/sec |
| 10 000 | 217.0 ms | 46.0 K rows/sec |

10× rows → 10× time → O(rows). MigraDoc's failure mode would have shown
~100× time at 10× input. Cargo-bench-invoked as `cargo bench --bench streaming_table_scaling`.

### Rendering-correctness fixes surfaced during the refactor

- **Multi-line cell rendering.** The existing `src/writer/table_renderer.rs` computed row heights from wrapped text (`wrap_text` at `:817`) but only emitted the first line on render (`:968-969` — flagged as `// Simple single-line rendering for now`). Fixed by pre-computing wrapped lines + per-line widths once inside `TableLayout.cell_layouts` and looping them at render time.
- **Per-line alignment.** `Center` and `Right` alignment used `cell_x + content_width / 2` and `cell_x + content_width` as the drawn-from x, which placed the text's left edge at the centre or right edge of the cell (so centre text was offset, right text was pushed off-cell). Fixed by using each wrapped line's measured width: `cell_x + (content_width - line_width) / 2` for Centre, `cell_x + content_width - line_width` for Right.

### Expansion bundles — builder-gap closure

#### Bundle A — images + transforms

```rust
page.image_from_file("logo.png", Rect::new(72.0, 720.0, 120.0, 40.0))?
    .rotated(15.0, |p| p.text("tilted caption"))
    .scaled(1.5, 1.5, |p| p.text("enlarged footnote"));
```

- `image_from_file(path, rect)` / `image_from_bytes(&[u8], rect)` / `image_with(ImageData, rect)` — auto-detect JPEG + PNG, alpha channels become `/SMask` XObjects for transparent placement.
- `rotated(deg, |p| ...)`, `scaled(sx, sy, |p| ...)`, `translated(tx, ty, |p| ...)`, `with_transform([a b c d e f], |p| ...)` — closure-scoped 2D affine transforms. Compose naturally (`translated(50, 100, |p| p.rotated(45, |p| p.text("tilted")))` produces the expected composed matrix). **v0.3.39 scope is text-only** — Path / Image / Table elements gain a matrix field in v0.3.40. Rotated watermarks + stamps + captions are the common-case target today.

#### Bundle B — navigation + document structure

```rust
doc.bookmark("Intro", 0)
   .bookmark_tree(|o| {
       o.add_item(OutlineItem::new("Chapter 1", 1));
       o.add_child(OutlineItem::new("Section 1.1", 2));
   })
   .with_page_labels(
       PageLabelsBuilder::new()
           .add_range(PageLabelRange::new(0).with_style(PageLabelStyle::RomanLower))
           .add_range(PageLabelRange::new(4).with_style(PageLabelStyle::Decimal)),
   )
   .insert_toc(0, "Table of Contents");
```

- `bookmark(title, page_index)` + `bookmark_tree(|b| ...)` — outline / bookmarks emitted as the catalog `/Outlines` tree. Pre-existing `OutlineBuilder` was unused; this release is the fluent wiring + the end-to-end catalog emission it was missing.
- `with_page_labels(PageLabelsBuilder)` — Roman preface + Arabic body or any PageLabelStyle mix, emitted as `/PageLabels` number-tree.
- `insert_toc(insert_at, title)` — walks the bookmark tree and renders an indented ToC page with right-aligned page numbers. v0.3.39 limitation: doesn't auto-renumber existing bookmark targets (call before further bookmarks, or re-issue after).

#### Bundle C — shapes + dash patterns

```rust
page.circle(cx, cy, r, Some(LineStyle::new(1.5, 0.1, 0.2, 0.3)), None)
    .ellipse(cx, cy, rx, ry, None, Some((0.9, 0.1, 0.1)))
    .polygon(&points, Some(LineStyle::default()), Some((0.5, 0.5, 0.9)))
    .arc(cx, cy, r, start, end, LineStyle::new(1.0, 0.0, 0.0, 0.0))
    .bezier_curve(x0, y0, c1x, c1y, c2x, c2y, x3, y3, style, None)
    .stroke_line(10, 100, 500, 100, LineStyle::new(0.5, 0, 0, 0).with_dash(&[3.0, 2.0], 0.0));
```

- `circle`, `ellipse`, `polygon`, `arc`, `bezier_curve` — five fluent shape primitives, each emitting one `ContentElement::Path` with optional stroke + fill. `circle` reuses `PathContent::circle`; `ellipse` / `arc` / `bezier_curve` build their quarter-Bezier approximations inline.
- `LineStyle::with_dash(&[f32], phase)` / `.solid()` — dash patterns propagate into `PathContent.dash_pattern`, emitted as `[...] phase d` before stroke and reset to solid after.

#### Bundle D — form fields (partial)

```rust
page.list_box("interests", 72, 600, 200, 80,
              vec!["Hiking".into(), "Reading".into(), "Coding".into()],
              Some("Coding".into()), true /* multi_select */)
    .required()
    .tooltip("Pick one or more")
    .text_field("email", 72, 500, 200, 20, None)
    .required()
    .read_only()
    .tab_order(TabOrder::Column);
```

- `list_box(name, x, y, w, h, options, selected, multi_select)` — wires the existing `ListBoxWidget` (fully implemented in `form_fields/choice_fields.rs`) through the public fluent surface.
- `.required()` / `.read_only()` / `.tooltip(text)` — chainable metadata that mutates the most-recently-added form field on the current page (no-op if no field has been added yet).
- `page.tab_order(TabOrder::{Row, Column, Structure})` — emits `/Tabs` on the page dict for reader tab-navigation order. `Structure` requires tagged PDF (Bundle F) to be meaningful.

#### Bundle E (partial) — layout primitives

```rust
page.heading(1, "Shopping list")
    .bullet_list(&["Apples", "Bananas", "Cherries"])
    .space(12.0)
    .numbered_list(&["First chapter", "Second chapter"], ListStyle::Decimal)
    .code_block("rust", "fn main() {\n    println!(\"hi\");\n}");
```

- `page.bullet_list(items)` — bullets (•) with indent + per-item
  wrapping.
- `page.numbered_list(items, ListStyle::{Decimal, RomanLower, AlphaLower})`
  — Arabic, lowercase Roman, or lowercase alpha markers.
- `page.code_block(language, source)` — monospace text over a
  light-grey filled rectangle. `language` reserved for Bundle F
  accessibility tagging; no syntax highlighting in v0.3.39.
- Helpers: `to_roman_lower(n)` and `to_alpha_lower(n)` exposed
  internally.

Inline rich text (`ParagraphBuilder` with `.bold()` / `.italic()` /
`.color()`), multi-column flow, and footnotes remain deferred to
v0.3.40 — see the E-0 RFC at `docs/v0.3.39/design/e_rich_text_rfc.md`.

#### Bundles E + F — RFC + research only

- `docs/v0.3.39/design/e_rich_text_rfc.md` — RFC for v0.3.40 inline-styling `ParagraphBuilder` with `.bold()` / `.italic()` / `.color(rgb, text)` cascading runs. ~770 LOC estimated for v0.3.40.
- `docs/v0.3.39/research/e_pdf_ua_compliance.md` — PDF/UA-1 compliance audit. Repo has ~40 % of the plumbing (StructureElement, MCID counter, ArtifactType) but MCIDs are orphaned — no StructTreeRoot emission. Bundle F lands in v0.3.40 as ~490 Rust LoC + 1,450 across 6 bindings.

### FFI / bindings

- **C FFI (`include/pdf_oxide_c/pdf_oxide.h`)** — six new entry points: `pdf_page_builder_stroke_rect`, `_stroke_line`, `_text_in_rect`, `_new_page_same_size`, `_table` (buffered), and the streaming trio `_streaming_table_begin` / `_push_row` / `_finish`. Handle-lifetime contract documented inline.
- **Python** (pyo3) — new classes `Align`, `Column`, `Table`, `StreamingTable`; new `FluentPageBuilder` methods mirroring the Rust surface. `align` kwargs accept string, enum, or raw int interchangeably.
- **WASM** (wasm-bindgen) — `Align` enum + `StreamingTable` class; buffered `table({columns, rows, hasHeader})` via serde-wasm-bindgen; `stroke_rect`, `stroke_line`, `text_in_rect`, `new_page_same_size`, `measure`, `remaining_space` on the page builder.
- **C#** — `Alignment`, `Column`, `TableSpec`, `StreamingTable : IDisposable`; fluent methods on `PageBuilder` including managed-side streaming buffer that flushes on `.Build()`.
- **Go** (cgo) — `Alignment`, `Column`, `TableSpec`, `StreamingTableConfig` under `go/types.go`; fluent methods on `*PageBuilder`; managed streaming adapter. Purego backend untouched (table surface is cgo-only in v0.3.39).
- **Node/TS** — `Align` enum + `StreamingTable` class in `js/src/builders/streaming-table.ts` with `pushRow`, `pushAll` (sync + async iterables), `finish`. All new types in `js/index.d.ts`.

### Scope deferred to v0.3.40 (tracked in #400)

**Tables**
- `TableMode::Sample` — measure first N rows, freeze widths, stream the rest.
- `TableMode::AutoAll` — opt-in O(rows × cols) with documentation warning.
- Cross-page cell splitting for tall rich cells.
- Bounded-lookahead rowspan in streaming mode.
- Arrow-style bounded batching on binding StreamingTables (current impl buffers all rows managed-side between `begin` and `finish`).
- Mixed-font exact metrics inside a single table (currently measures against the table default font).
- Pandas DataFrame first-class adapter in Python.

**Transforms**
- `TableContent`-as-a-whole matrix (individual cells compose naturally
  through their own `TextContent` / `PathContent` matrix fields, which
  now ship — but wrapping an entire Table in one transform needs a new
  field on `TableContent` itself).

**Forms (rest of Bundle D)**
- Signature-field form widget (coordinates with #208 signing half).
- Barcode-bound form field (auto-generate from another field's value at fill time).
- Field validation — regex mask, numeric range, JavaScript actions.

**Layout (Bundle E) — blocked on E-0 RFC which ships in v0.3.39**
- Inline rich-text styling (`ParagraphBuilder` with `.bold()` / `.italic()` / `.color()`).
- Multi-column flow on `DocumentBuilder` (currently only available through `Pdf::from_html_css`).
- Footnotes / endnotes.

**Accessibility (Bundle F) — blocked on F-0 research which ships in v0.3.39**
- Tagged PDF / logical structure tree emission.
- `/Lang` per content run.
- `/Artifact` marking for headers/footers on the write side.
- `/RoleMap` for non-standard structure types.

**Advanced forms (Bundle G) — pick up on concrete customer demand**
- Calculated fields / JavaScript actions.
- XFA write-side.

### Bug fixes

- **#401** — Encrypted PDFs were missing embedded-font sub-objects (`/Widths`, `/FontDescriptor`, `/FontFile2`); they are now included and referenced correctly. Reported by [@sparkyandrew](https://github.com/sparkyandrew).
- **#402 / #406** — Systemic UTF-8 encoding loss: every PDF string object (metadata titles, annotation contents, bookmark titles, content streams) was written as raw UTF-8 bytes instead of PDFDocEncoding (Latin-1 code point for chars ≤ U+00FF) or UTF-16BE with BOM (for chars > U+00FF). Reported by [@AngeloBestetti](https://github.com/AngeloBestetti) (#402) and internally audited as #406.
- **#407** — L4 font cache cross-contamination: when two pages share the same `/Font` resource key (e.g. both use key `F1`), the CMap of the first-loaded face silently overwrote the second's glyph mapping, causing glyphs to be dropped or mis-decoded. Fixed by keying the combined-font hash over all font objects. Reported by [@ChadThackray](https://github.com/ChadThackray).
- **#395** — `SignatureException` on `PdfDocument.open()` for PDFs containing digital signatures. Fixed as a side-effect of the signing infrastructure (#208). Reported by [@gevorgter](https://github.com/gevorgter).
- **#398** — Native PDF parser was non-reentrant: concurrent FFI reads on the same handle returned spurious parse errors. Resolved by the interior-mutability refactor (`Mutex<…>` on internal caches).
- **#409** — Python (and all bindings) lacked `to_bytes()` / in-memory output; `compress` and `garbage_collect` were not wired into the write path. Reported by [@potatochipcoconut](https://github.com/potatochipcoconut).
- **#411** — `p12 = "0.6"` (yanked / unmaintained) replaced with `p12-keystore = "0.2.1"` (RustCrypto-ecosystem, pure Rust, actively maintained). No public API change; `SigningCredentials::from_pkcs12` behaviour is unchanged.
- **StreamingTable rowspan flush** — `finish()` was silently dropping the in-progress rowspan group if the table ended mid-span. Added a flush of any partial `rowspan_buf` before finalising the page.
- **`draw_rowspan_group` bounds guard** — accessing `rows[0][col_idx].rowspan` was not guarded against `col_idx ≥ rows[0].len()`, causing a panic on narrow tables with rowspan cells. Added the bounds check `col_idx < rows[0].len()`.
- **`scan_root_ref` anchoring** — the digital-signature helper scanned the entire document for `/Root`, so a `/Root` reference embedded inside an annotation value or stream body could silently win over the real XRef `/Root` at the end of the file. Now mirrors `scan_startxref` by restricting the search to the last 4 KB of the file.
- **Signature reason/location PDFDocEncoding** — `/Reason` and `/Location` entries in CMS-signature dictionaries were written as raw UTF-8 bytes, bypassing the `encode_pdf_text_string` path. Non-ASCII characters (accents, CJK, etc.) were stored as illegal UTF-8 sequences in the PDF string. Now uses the same hex-encoded PDFDocEncoding/UTF-16BE path as all other string objects, closing the last #402-class gap in the signing path.
- **#394** — Mixed-size inline runs (superscripts, footnote markers) were incorrectly split onto separate lines because the newline gate used a hard-coded 2 pt Y-tolerance. Replaced with `PdfDocument::same_line_threshold` — a font-size-relative helper (`max(prev_fs, cur_fs) × 0.5`) shared across all seven Tagged-PDF assembly paths and `should_insert_space`. A forward-gap guard was added to prevent the widened threshold from merging spans across column gutters. Contributed by [@RolandWArnold](https://github.com/RolandWArnold) (#394).
- **#403** — Simple fonts without an explicit `/Widths` array fell back to a uniform 0.55 em default for every glyph. For standard-14 fonts (Helvetica, Times, Courier, etc.) this inflated span widths by up to 40 %, collapsing inter-column gaps from real values (e.g. 47 pt) to near-zero (5 pt) and breaking gap-dependent layout heuristics. The fast path now populates the byte-to-width table from `get_standard_font_width` when `/Widths` is absent; non-standard fonts and unmapped codepoints still fall back to the generic default. Contributed by [@RolandWArnold](https://github.com/RolandWArnold) (#403).
- **#404** — Span right-edges could drift ~0.02 pt outside the detected table bbox due to float accumulation in upstream width arithmetic. The strict `Rect::contains_rect` check then rejected those spans from the table's retain set, so they were emitted via both the table path and the flow path, producing duplicated text. Introduced a 0.1 pt tolerance at the two retain call sites in `document.rs` via `PdfDocument::contains_rect_with_tolerance`; the geometry primitive itself remains strict. Contributed by [@RolandWArnold](https://github.com/RolandWArnold) (#404).

### CI / test-suite fixes

- Resolved all Clippy, `rustfmt`, and `cargo check` failures that were blocking CI (`fix(ci)` commit `6c95bada`): unused-mut across 80+ files after the interior-mutability refactor, late-init variables, doc-comment ordering, non-minimal boolean conditions, deprecated function references.
- Renamed six test files from issue-number / benchmark-code names to functional descriptive names (`refactor(tests)` commit `fa071380`): `test_b1_*` → `test_shared_form_xobject_per_page_ctm`, `test_b3_*` → `test_running_header_first_occurrence_kept`, `test_b4_*` → `test_two_column_reading_order`, `test_b7_*` → `test_stroke_fill_duplicate_text_dedup`, `test_issue_346_*` → `test_extract_text_sort_comparator_stability`, `test_issue_395_*` → `test_signed_pdf_opens_and_renders`.
- **Example smoke-tests in CI** — all code examples are now compiled and executed on every CI run, catching binding API drift before it reaches a release. A dedicated `rust-examples` job runs all 13 Rust examples (`tutorial_*` + `showcase_*`). The Python, Go, Node.js, and C# binding jobs each gained an equivalent step that runs the per-language examples against `tests/fixtures/simple.pdf`. This means any breaking change to a public binding API will fail CI immediately rather than being discovered post-release by users.
- **Example restructuring** — the single monolithic `09-new-features` showcase file per language was replaced with one standalone file per feature (`streaming-table`, `pdf-ua-image`, `in-memory-roundtrip`, `pkcs12-signing`, `rfc3161-timestamp`) across all 5 languages. Each file is a self-contained runnable program. The tutorial examples `01-08` were also repaired: Go examples gained `go.mod` + `go.sum` and had three API-drift regressions fixed (`OpenEditor`, `pdf.Save`, `RowCount`/`CellText`); JavaScript examples were migrated from CommonJS `require()` to ESM `import`; C# examples gained `.csproj` files referencing the local `PdfOxide` project.

### Community Contributors

- **[@RolandWArnold](https://github.com/RolandWArnold)** — First
  contribution to PDFOxide, and a substantial one at that. Roland
  identified three independent text-extraction correctness issues, traced
  each one to its root cause in the Rust source, wrote focused fixes with
  synthetic `PdfWriter`-based regression tests, and documented the
  behaviour thoroughly in PR descriptions that made review straightforward.
  [#394](https://github.com/yfedoseev/pdf_oxide/pull/394) fixes the
  long-standing mixed-size inline run / superscript line-grouping
  problem; [#403](https://github.com/yfedoseev/pdf_oxide/pull/403)
  restores correct span widths for standard-14 fonts without `/Widths`;
  [#404](https://github.com/yfedoseev/pdf_oxide/pull/404) eliminates
  duplicate text caused by sub-pixel float drift at the table-retain
  boundary. Thank you, Roland — we look forward to more! 🚀

- **[@AngeloBestetti](https://github.com/AngeloBestetti)** — Filed
  [#402](https://github.com/yfedoseev/pdf_oxide/issues/402) with the
  concrete word `"Lógico"`: a Portuguese term that, when saved to PDF,
  came back as mojibake because every accented byte was being stored as
  raw UTF-8. That single reproducer uncovered a systemic encoding bug —
  _all_ PDF string objects (metadata titles, annotation contents,
  bookmark labels, content-stream text) were silently corrupted for any
  non-ASCII character. The internal audit that followed produced #406
  and a full rewrite of `write_escaped_string` +
  `encode_pdf_text_string` to emit PDFDocEncoding for chars ≤ U+00FF
  and UTF-16BE with BOM for anything above. Thank you.

- **[@sparkyandrew](https://github.com/sparkyandrew)** — Filed
  [#401](https://github.com/yfedoseev/pdf_oxide/issues/401) after
  discovering that AES-256 encrypted PDFs built with `DocumentBuilder`
  opened successfully but rendered blank — the embedded font was gone.
  The root cause: `collect_reachable_ids` followed the top-level `Font`
  dictionary but stopped there, so `/Widths`, `/FontDescriptor`, and
  `/FontFile2` were garbage-collected as "unreachable" during the
  encrypted write pass. The fix traces the full font sub-object graph
  before encryption so the complete font survives. Thank you.

- **[@ChadThackray](https://github.com/ChadThackray)** — Filed
  [#407](https://github.com/yfedoseev/pdf_oxide/issues/407) after
  noticing that glyphs from one page silently replaced those of another
  whenever two pages shared the same `/Font` resource-key name (both
  using key `F1` but mapped to different faces). The L4 cache was
  keying the combined glyph-map on a spot-check of a single font
  object; the fix computes a combined hash over the _complete_ font set,
  so any change to any face invalidates the entry. Thank you.

- **[@gevorgter](https://github.com/gevorgter)** — Filed
  [#395](https://github.com/yfedoseev/pdf_oxide/issues/395) after a
  `SignatureException` from `RenderPage` on a 9-page signed PDF — the
  renderer was propagating a signature-parse failure as the page-render
  verdict even though no interactive widget lived on that page. The fix
  treats unparseable signature-field metadata as non-fatal at render
  time. @gevorgter also supplied the reproducer PDF that became the
  regression fixture (`tests/test_signed_pdf_opens_and_renders.rs`),
  ensuring this class of error can never silently return. Thank you.

- **[@potatochipcoconut](https://github.com/potatochipcoconut)** —
  Asked [#409](https://github.com/yfedoseev/pdf_oxide/issues/409) how
  to get a `PdfDocument` as raw bytes from Python without writing to
  disk, and whether `compress` and `garbage_collect` were available.
  Neither worked. The question drove the `to_bytes()` / `SaveOptions`
  kwargs work that shipped in-memory output, compression, and
  garbage-collection across all 7 bindings, plus 18 missing
  `DocumentEditor` methods. Thank you.

## [0.3.38] - 2026-04-23
> DocumentBuilder fluent API across every language binding, real font subsetting, DocumentBuilder encryption, multi-target WASM packaging, and the first cryptographic slice of PDF signature verification

This release closes the "Rust-only `DocumentBuilder` gap": the fluent
write-side builder, embedded fonts, the HTML+CSS pipeline, annotations,
form-field creation, and low-level graphics primitives are now reachable
from **Python, WASM, C#, Go, and Node/TypeScript** — the Rust
implementation is the single source of truth and every binding is a
thin translation layer. On top of that it lands the first
cryptographic signature-verification path (RSA-PKCS#1 v1.5) across
every binding and a pdf.js-parity fix for scanned / bilevel pages
rendered under a Multiply-blended overlay.

> **Scope note.** "Write-side API" here refers specifically to the
> `DocumentBuilder` surface listed below. Reader / editor /
> rendering-options parity across bindings is ongoing work; see the
> post-release audit (`docs/api-coverage-audit.md` / issue tracker
> `#384` follow-ups) for the full picture.

### Write-side API × every binding (#384)

Every binding now exposes the full `DocumentBuilder` fluent API:

```python
# Python — the same shape ships in WASM, C#, Go, and Node/TS
font = EmbeddedFont.from_file("DejaVuSans.ttf")
(DocumentBuilder()
  .register_embedded_font("DejaVu", font)
  .a4_page()
    .font("DejaVu", 12).at(72, 720).text("Привет, мир!")
    .highlight((1.0, 1.0, 0.0))
    .text_field("name", 150, 680, 200, 20, "Jane Doe")
    .checkbox("subscribe", 72, 650, 15, 15, True)
    .rect(50, 50, 500, 700)
  .done()
  .build())
```

Surface shipped in all 6 bindings:

- **DocumentBuilder** + **FluentPageBuilder** + **EmbeddedFont** —
  multi-page construction with CJK / Cyrillic / Greek support (closes
  **#382** cross-language).
- **HTML+CSS pipeline** — `Pdf.from_html_css(...)` and
  `from_html_css_with_fonts(...)` for multi-font cascades.
- **15 annotation methods** — link (URL / page / named), highlight,
  underline, strikeout, squiggly, sticky note, stamp (14 standard
  types + custom), free text, watermark (custom / DRAFT /
  CONFIDENTIAL).
- **5 AcroForm widget types** — text_field, checkbox, combo_box,
  radio_group, push_button.
- **Graphics primitives** — `rect`, `filled_rect`, `line`.
- **AES-256 encryption** — `save_encrypted` / `to_bytes_encrypted` on
  every binding.

Per-binding regression tests for every capability above; ~70 new
integration tests pass across Python (20), C FFI (11), C# (11), Go
(11), Node/TS (10), and WASM (9).

### Real font subsetting on the write path (#385 — FONT-3b)

Documents that embed a CJK face now ship a **subset**, not the full
font. A PDF with 5 characters from `NotoSansCJKtc-Regular.otf`
(~17 MB original) is typically under 100 KB. Content streams, `/W`
widths, and `ToUnicode` CMap are all re-keyed onto the subset GID
space; `extract_text` round-trips unchanged.

*Breaking (v0.3.x semver-acceptable):* `EmbeddedFont::encode_string` /
`encode_shaped_run` now return `Vec<u16>` instead of a hex `String`,
and `build_embedded_font_objects` returns a `GlyphRemapper` that
callers must pass to `ContentStreamBuilder::build_with_remappers`.
Internal writer-library consumers only — no change to high-level APIs.

### DocumentBuilder encryption (#386)

AES-256 encryption is now available on programmatically-built PDFs:

```rust
DocumentBuilder::new()
    .a4_page().text("secret").done()
    .save_encrypted("out.pdf", "user-pw", "owner-pw")?;
```

Also: `save_with_encryption` (custom algorithm + permissions) and
`to_bytes_encrypted` for in-memory output.

### Multi-target WASM packaging (#392)

`pdf-oxide-wasm` now ships three builds side-by-side and routes each
consumer through `package.json` conditional exports:

| Environment                                       | Build      |
|---------------------------------------------------|------------|
| Node.js                                           | `nodejs/`  |
| Bundlers (Vite, webpack, Rollup, esbuild, Bun)    | `bundler/` |
| Browsers / Deno / Cloudflare Workers              | `web/`     |

Fixes `ReferenceError: Can't find variable: __dirname` thrown in any
browser bundler. Subpath imports (`pdf-oxide-wasm/web` etc.) are also
available for manual routing.

### Digital signature verification (#208 — verification half)

First cryptographically-backed signature surface on the reader side.
Every binding (`Signature.verify()` / `.verifyDetached()` / equivalents)
now runs the RFC 5652 §5.4 signer-attributes check against the
embedded certificate and the §11.2 `messageDigest` check against the
caller's document bytes:

```python
for sig in doc.signatures():
    print(sig.signer_name, "→", sig.verify())            # signer-attrs only
    print("detached ok =", sig.verify_detached(pdf_bytes))  # + content hash
```

- **RSA-PKCS#1 v1.5** over SHA-1 / SHA-256 / SHA-384 / SHA-512 — the
  padding used by effectively every signed PDF in the wild — returns
  `Valid` / `Invalid`.
- **RSA-PSS** and **ECDSA** surface as `Unknown` /
  `UnsupportedFeatureException` for now; callers that need those can
  still read the signer certificate via `Signature.GetCertificate()`
  and drive their own check.
- `SignatureVerifier::verify` (Rust) also stamps the verification
  result with trust-root lookup, expiry window, and signer DN pulled
  from the embedded certificate.

Supporting surface shipped alongside:

- `Certificate` — DER inspection (subject, issuer, serial, validity,
  `is_valid`) via `x509-parser` — **every binding**.
- `Signature` — enumerate + inspect + `.GetCertificate()` —
  **every binding**.
- `Timestamp` — RFC 3161 `TSTInfo` parsing (time, serial, policy,
  TSA name, hash algorithm, message imprint) — **every binding**.
- `TsaClient` — RFC 3161 HTTP POST with nonce + HTTP Basic auth,
  behind a new `tsa-client` Cargo feature — **every binding except
  WASM**. Intentionally not wired on WASM (ureq is wasm-incompatible).
- `DocumentEditor::set_producer` / `set_creation_date` — metadata
  writers.
- `render_page_region` / `render_page_fit` — clipped / fitted
  rendering surface.
- Bicubic image filtering (pdf.js#19978 parity) — scanned / bilevel
  pages with a Multiply-blended overlay no longer collapse their
  grayscale range on downscale.

Signing (as opposed to verification) is not covered by this release;
#208 remains open for the signing half.

### Binding parity follow-ups

Five thin-wrapper commits closed the last coverage holes in this
release's signature surface — Python/Go/WASM `Certificate` inspect,
Node `Timestamp` parse+verify, Node `TsaClient` HTTP. Every capability
in the Supporting Surface list above is now the language-idiomatic
shape across all six non-Rust bindings (modulo the principled
WASM-TsaClient omission).

### Go binding — purego backend + cache-dir install

Go users can now build with `CGO_ENABLED=0` via a second backend that
uses [ebitengine/purego](https://github.com/ebitengine/purego) to
`dlopen` `libpdf_oxide.{so,dylib,dll}` at runtime — no C toolchain
required. Backend selection is automatic via Go's built-in `cgo` tag
(`//go:build cgo` → full CGo API, `//go:build !cgo` → purego).

The purego backend covers the read-side `PdfDocument` surface — open
(path / bytes / password), page count, version, text / Markdown / HTML
/ plain-text extraction, fonts, annotations, page elements, search,
page dimensions, logging — plus `PdfCreator.FromMarkdown` for test
fixtures. Editor, `DocumentBuilder`, barcode, signature, TSA,
rendering, OCR, and forms stay CGo-only; using them under `!cgo` is a
compile-time error. Full parity is tracked for a follow-up.

Installer:

- **New `-shared` flag** fetches the cdylib instead of the staticlib
  and prints `CGO_ENABLED=0` + `PDF_OXIDE_LIB_PATH=…` to export.
- **Install dir moved to `os.UserCacheDir()`** — `~/.cache/pdf_oxide`
  on Linux, `~/Library/Caches/pdf_oxide` on macOS,
  `%LocalAppData%\pdf_oxide` on Windows. Matches Go's own `GOCACHE`
  convention; existing installs re-fetch once into the new path.

Release assets now include `pdf_oxide-go-ffi-shared-<platform>.tar.gz`
for every Tier-1 platform alongside the existing staticlib archives.

### Bug fixes

- **#395** — `PdfOxide.Exceptions.SignatureException: '[8500]
  Signature error...'` raised by `doc.RenderPage(0, 0)` on a
  specific 9-page PDF reported by
  [@gevorgter](https://github.com/gevorgter). The failure was the
  renderer propagating a signature-parse error up as the page-render
  verdict even though the page itself had no interactive signature
  widget on it. Fixed by treating unparseable signature-field
  metadata as non-fatal at render time; pinned by
  `tests/test_issue_395_render_signature_exception.rs` + the C#
  regression test so this can't silently come back.

### Thanks

Reports and feature requests from
[@sparkyandrew](https://github.com/sparkyandrew) (#382 CJK via
`DocumentBuilder`, #385 subsetter),
[@arthurlassagne](https://github.com/arthurlassagne) (#392 browser
build breakage), and
[@gevorgter](https://github.com/gevorgter) (#395 RenderPage
SignatureException). All three surfaced the gaps that drove this
release.

## [0.3.37] - 2026-04-20
> HTML + CSS → PDF (issue #248) — first credible pure-Rust pipeline

### API — `Pdf::from_html_css` (#248)

```rust
let font = std::fs::read("DejaVuSans.ttf")?;
let mut pdf = Pdf::from_html_css(
    "<h1>Hello</h1><p>World</p>",
    "h1 { color: blue; font-size: 24pt }",
    font,
)?;
pdf.save("out.pdf")?;
```

The whole feature: pass HTML + CSS + font bytes, get a paginated PDF
back. Pure Rust, MIT/Apache only (no MPL transitive deps),
`extract_text` round-trips byte-equal so produced PDFs participate in
the existing test infrastructure.

End-to-end test suite at `tests/test_html_to_pdf_e2e.rs` covers
simple paragraph, multi-paragraph, nested HTML, CSS-styled text, and
Unicode (Latin + Latin-Extended + Cyrillic + symbols) round-trips.

### Phase FONT — embedded TTF/OTF subsystem

- **Subsetter wrapper** around the `subsetter` crate (Typst's,
  MIT/Apache): `crate::fonts::subset_font_bytes(bytes, used_glyphs)`
  produces a subset face, and `EmbeddedFont` tracks used glyph IDs
  via the `FontSubsetter` type. The writer path currently embeds the
  full font face in `FontFile2` (full-face embedding + Identity-H is
  valid PDF 1.7 and round-trips correctly); switching to the
  subsetter's output requires remapping glyph IDs in the already-
  emitted content streams, which lands as a later follow-up. The
  standalone API + glyph tracking still ship so callers that use the
  subsetter directly (e.g. CLI tools shelling out to `subset_font_bytes`)
  get the size benefit today.
- **Type 0 / CIDFontType2 / Identity-H / ToUnicode emission** wired
  into `PdfWriter` so `add_embedded_text(text, x, y, "EFn", size)`
  produces a font dict graph that PDF readers handle correctly.
  Round-trip via `extract_text` returns the input string for Latin,
  Cyrillic, Greek, Hebrew, Arabic.
- **System font discovery** via `fontdb` (RazrFalcon, MIT). New
  `system-fonts` feature gates discovery + shaping; default-on for
  language bindings, off for WASM and the bare Rust crate.
- **Text shaping** via `rustybuzz` (HarfBuzz port, MIT). Returns
  positioned glyph runs with `cluster` info so the inline formatter
  can map glyphs back to source bytes.

### Phase CSS — hand-rolled engine

10 modules, ~6,500 LoC, no MPL anywhere:

- **Tokenizer** (CSS Syntax L3) with full token coverage including
  CDO/CDC, hex+named entities resolution in url(), source locations.
- **Parser** producing `Stylesheet { rules: Vec<Rule> }` with
  forgiving recovery per spec.
- **Selectors** L3 + L4 subset: `:is`/`:where`/`:not`/`:has`,
  structural pseudo-classes, attribute matchers with `i`/`s` flags,
  specificity computation packed into a sortable u32.
- **Matcher** with `Element` trait so the engine isn't tied to one
  DOM implementation.
- **Cascade** with origin/specificity/source-order sorting,
  inheritance from parent for the spec's inherited-property list,
  inline-style merge, custom-property storage.
- **`calc()` / `min()` / `max()` / `clamp()`** evaluator with mixed-
  unit math against a `CalcContext`.
- **`var()`** substitution with DFS cycle detection.
- **Typed property values** for colour (~150 named, hex, rgb/rgba/
  hsl), length (every CSS Values L4 unit), display, font-size/
  weight/style/family, margin/padding shorthand expansion, line-
  height, etc.
- **At-rules**: `@media print` always-true + `(min/max-width)`
  predicates, `@page` with `:first`/`:left`/`:right`/`:blank`
  selectors and margin boxes, `@font-face` descriptor extraction,
  `@import` URL forwarding, `@supports` against our supported set.
- **Counters** (`counter`/`counters`/`counter-reset`/`-increment`/
  `-set` with Roman/Greek/alpha numbering) and pseudo-element
  content evaluation.

### Phase HTML

- **HTML5 tokenizer** with attribute parsing (quoted/unquoted/bare),
  void-element implicit self-closing, `<style>`/`<script>` raw-text
  contexts, named + numeric entity decoding, comments, DOCTYPE.
- **Flat arena DOM** implementing the CSS-4 `Element` trait so the
  cascade matches against real document nodes. Implicit close
  handling for the common `<p>` and `<li>` cases.
- **Stylesheet extraction**: `<style>` blocks, `<link
  rel="stylesheet">` (URL forwarded; `media` attribute preserved),
  per-element inline `style="..."`.
- **Resource extraction**: `<img>` with srcset DPR selection,
  `<picture>`/`<source>` first-match, `<a href>` (internal anchor
  detection).

### Phase LAYOUT

- **Box tree** from DOM × ComputedStyles with display-split
  (outer/inner), anonymous-block insertion per CSS 2.1 §9.2.1.1,
  `display: none`/`contents` handling, UA default display table for
  common HTML elements.
- **Taffy integration** for block / flex / grid layout (Dioxus, MIT,
  default-features-off + only the features we need).
- **Inline formatting** with greedy line breaker via UAX #14
  (`unicode-linebreak`), `text-align`/`white-space` modes, hard
  breaks, atomic inline boxes.
- **Float scaffolding** with line-shortening helpers.
- **Margin collapsing** per CSS 2.1 §8.3.1.
- **Multi-column** distribution (`column-count`/`column-width`/
  `column-gap` with greedy line distribution).
- **Tables** with auto + fixed column-width algorithms, row-group
  classification (header/body/footer for paginator repetition).

### Phase PAGINATE

- Slices a positioned box tree across pages at `floor(box.y /
  content_height)` boundaries.
- Multi-page boxes emit one PaginatedBox per page with the visible
  y-slice; preserves source IDs so PAINT can look up styles.
- A4 portrait (96dpi) and Letter (8.5×11) page presets.

### Phase PAINT

- Walks each PageFragment and emits text + borders into the existing
  `PdfWriter` / `PageBuilder`.
- HTML→PDF Y-flip applied once at emission time so all internal
  coordinates stay top-down.

### Corner-case fixes and follow-ups

After the initial cut of the HTML+CSS pipeline, corner-case
validation surfaced a set of regressions and missing features. All
of the below also ship in v0.3.37:

- **Tokenizer char-boundary safety.** The CSS tokenizer's
  `ignore_case` lookahead indexed raw byte offsets on multi-byte
  characters, panicking on any CSS source that put non-ASCII inside a
  keyword-adjacent position. Fixed.
- **Block sizing for inline-text flow.** Block boxes with only-inline
  children were given zero intrinsic height, so paint-time
  `y`-coordinates collapsed; multi-paragraph documents dropped every
  paragraph but the first, and long single paragraphs retained only
  ~20 % of their words. `run_layout` now reserves intrinsic height
  from the body font size and the inline run count.
- **Arabic / RTL shaping.** Paint now routes RTL paragraphs through
  the rustybuzz shaper (feature `system-fonts`) so contextual forms,
  ligatures, and visual reordering all work.
- **Multi-font cascade.** New
  `Pdf::from_html_css_with_fonts(html, css, Vec<(family, bytes)>)`.
  CSS `font-family` on any element resolves against the registered
  families (case-insensitive, with/without quotes); unknown families
  fall back to the first registered font. Walks up the box tree so
  inline children inherit their ancestor's family.
- **Page breaks.** `page-break-before: always` and
  `page-break-after: always` now open a fresh page, both via CSS
  rules and via inline `style="..."`. Multiple breaks accumulate.
- **`::before` / `::after` generated content.** New
  `cascade::pseudo_content_for(ss, element, PseudoKind::{Before,After})`.
  Literal strings, `attr(name)`, and `open-quote`/`close-quote` all
  resolve.
- **Opacity + `transform: translate*()`.** `opacity <= 0.01` on any
  ancestor hides an element and all its text descendants.
  `transform: translateX/Y/translate(…)` applies as a pre-paint
  offset on the box's x/y.
- **`<img>` data-URI embedding.** `<img src="data:image/png;base64,…">`
  (and `data:image/jpeg;…`, percent-encoded plain payloads) now
  decode to a real PDF Image XObject. The paint pipeline emits
  `/Do` operators against a per-page `/XObject` resource dictionary
  which `PdfWriter::finish()` now serializes — the missing
  resource-dict wiring was why prior `page.add_element(Image(…))`
  calls rendered as silent no-ops. External URLs / filesystem paths
  return `None` from `decode_image_src` so callers can resolve those
  themselves.
- **List markers.** `<ul>` items get `•` (U+2022) and `<ol>` items
  get `N.` numbering, painted in the gutter to the left of the
  `<li>`'s content box. Nested lists work on both levels.
- **`<a href>` link annotations.** Every anchor box with a non-empty
  `href` emits a PDF `/Link` annotation carrying a `/URI` action;
  inline text inside the anchor inherits the link by walking up the
  box tree. Anchors with no `href` emit no annotation.
- **Embedded fonts via `DocumentBuilder` (#382).** New
  `DocumentBuilder::register_embedded_font(name, EmbeddedFont)`.
  Text emitted through the fluent builder
  (`FluentPageBuilder::font(name, size).text(...)`, or any
  `ContentElement::Text` whose `FontSpec.name` matches a registered
  embedded font — including template headers/footers) is now routed
  through the Type-0 / CIDFontType2 path instead of silently falling
  back to Helvetica. CJK, Cyrillic, Greek, Hebrew, Arabic text
  emitted via the high-level API now actually embeds and renders.
  Unregistered font names continue to resolve against the base-14
  set. Reported by @sparkyandrew.

### Bug fixes surfaced during pre-release review

- **Base-14 bold text rendered non-bold.** The page `/Resources /Font`
  dictionary keyed entries with dashes stripped (`HelveticaBold`)
  while content streams emitted `Tf /Helvetica-Bold`. PDF readers
  silently fell back to the default font, so every bold or italic
  base-14 run came out regular. Resource-dict keys now match the
  `Tf` operator names exactly.
- **TTC system fonts (Helvetica.ttc, msgothic.ttc, …).** `fontdb`
  surfaces collection fonts as `Source::SharedFile(path, …)`, which
  the resolver previously rejected as `NoPath`. SharedFile entries
  are now read the same way as regular files, so a huge swathe of
  macOS/Windows system fonts become resolvable.
- **Unquoted multi-word `font-family`.** `font-family: DejaVu Sans,
  sans-serif` tokenises as two separate `Ident`s, so the registered-
  family lookup never matched them as a single name. The resolver
  now collects consecutive idents (whitespace-separated) into one
  candidate and flushes at top-level commas, so quoted and unquoted
  forms behave the same.
- **Memory leak in `Pdf::from_html_css` / `from_html_css_with_fonts`.**
  The factories leaked the combined CSS source, parsed stylesheet,
  DOM, and family map on every call (four `Box::leak` sites). Long-
  running processes (HTTP servers, batch converters) grew unbounded.
  The downstream APIs all accept non-'static references; the
  function now holds them in locals scoped to the call.
- **PNG alpha / soft-mask now renders.** `ImageData::from_png`
  already decoded and compressed the alpha channel, but
  `ImageContent` had no field for it and the XObject emitter hard-
  coded `SMask = None`. `ImageContent` gains a `soft_mask`, the
  html_css paint pipeline propagates it, and the XObject path
  actually emits a `/SMask` stream.
- **Shaped text round-trips via `extract_text`.** The shaped path
  (`add_shaped_embedded_text`) only recorded glyph IDs in the
  subsetter, leaving shaped runs absent from the ToUnicode CMap and
  uncopy-paste-able. The new `encode_shaped_run` maps glyph clusters
  back to source codepoints so the ToUnicode entries are complete
  for simple scripts and exact-leading-char for ligatures.
- **Reproducible PDF output.** `PdfWriter::finish` iterated
  `embedded_fonts` directly from the HashMap, randomising object-ID
  order across runs. Embedded fonts are now emitted in registration
  order via an explicit `embedded_font_order` vector.
- **Embedded-font name collisions.** Registering two fonts with the
  same display name silently overwrote the first. `embedded_fonts`
  is keyed by its `EFn` resource name (unique, monotonic) so
  registrations are independent regardless of display name.
- **fontdb Mutex serialised on slow disks.** `SystemFontDb::resolve`
  held the fontdb lock across the font-bytes `fs::read`.
  Concurrent resolve calls are now lock-free during I/O — the lock
  is released once the face path + PostScript metadata are picked.
- **Misleading docs corrected.** Module documentation previously
  claimed `background-color` rendered as a filled rect (currently a
  no-op stub) and that the writer embedded a subset of the face
  (currently embeds the full face + Identity-H, subsetter output is
  a later follow-up). Both are now reflected accurately in the
  relevant docstrings.

### Tests added in the corner-case pass

- **E2E** (`tests/test_html_to_pdf_e2e.rs`): 36 tests (was 14),
  covering every feature above plus a kitchen-sink document that
  exercises `::before`, list markers, page-break, opacity, translate,
  and `<a href>` in a single round-trip.
- **Unit**: 4 cascade pseudo-element tests, 7 paint tests (opacity /
  translate / data-URI decode), 3 inline-text sizing tests, 1 RTL
  shaper test, 1 multi-font cascade test, 1 tokenizer multi-byte
  regression test.
- Total test count: 4772 lib + 36 e2e; 168 integration suites all
  green, 0 regressions on the existing corpus.

### Limits

The supported CSS surface is documented in detail in
[`docs/HTML_TO_PDF_GUIDE.md`](docs/HTML_TO_PDF_GUIDE.md). Out of
scope: CSS filters, 3D transforms, animations, SVG-in-HTML (every
viable Rust SVG crate is MPL), MathML, `hyphens: auto`,
`shape-outside`, JavaScript execution, full-matrix `transform`
(scale/rotate), gradients, and `box-shadow`.

### Licence audit

`cargo deny check licenses` passes with **zero** MPL transitive
dependencies. The Mozilla CSS stack (`cssparser`, `selectors`,
`html5ever`, `lightningcss`, `stylo`) is all MPL-2.0; v0.3.37 hand-
rolls the equivalents to keep pdf_oxide entirely under MIT/Apache.

### Community Contributors

- **[@jmriebold](https://github.com/jmriebold)** — Filed
  [#248](https://github.com/yfedoseev/pdf_oxide/issues/248)
  ("CSS support"). That single issue is the root of this release's
  entire HTML+CSS→PDF pipeline — the hand-rolled CSS engine, the
  HTML5 tokenizer + arena DOM, Taffy-backed layout, the `::before`/
  `::after`, `page-break-*`, `<img>` data-URI, multi-font cascade,
  opacity / transform, `<a href>` link, and RTL shaping work all
  exist because he asked for it. Thank you.

## [0.3.36] - 2026-04-19
> Markdown structural extraction quality vs pdfium — Tagged-PDF
> heading and list emission, multi-column reading-order fixes,
> safer RTL handling, inline-image cap

### Markdown structural extraction (#377)

The headline change of this release. `to_markdown()` previously
consumed only the MCID *order* from `/StructTreeRoot` and then
re-derived heading levels from font-size heuristics and list
markers from glyph detection. For Word/Acrobat tagged PDFs whose
body and heading text share a point size, this dropped every
heading; for tagged lists where `LI → LBody → MCR` nests the
actual content under a Span/P, this dropped every bullet; for
tagged paragraphs whose inter-paragraph gap was less than 1.5×
line height, this merged adjacent paragraphs.

This release wires the structure tree directly into the markdown
pipeline:

- **Heading and list emission from `/StructTreeRoot`.** New
  `StructRole` (Heading(1..6), ListItem, ListItemLabel, ListItemBody)
  attached to every span via the per-MCID lookup map. The converter
  prefers the explicit role over font-size heuristics so Word-tagged
  documents recover their full heading hierarchy. Lists emit `- item`
  with paragraph breaks at every role transition. (D1)
- **Heading / list role propagated through nested MCRs.** Tagged PDFs
  commonly wrap heading content as `H1 → Span → MCR` and list bodies
  as `LI → LBody → Span → MCR`. The traversal now threads
  `InheritedContext { heading_level, list_role }` down both
  `traverse_element` and `traverse_element_all_pages`, so deeply
  nested MCRs carry the right semantic role. (D8b)
- **Per-`/StructTreeRoot` block boundary forces paragraph break.** New
  `OrderedContent.block_id` increments on every entry into a block
  element (`/P`, `/H1..6`, `/LI`, `/Lbl`, `/LBody`, `/Sect`, `/Div`,
  `/Art`, `/TR`, `/TH`, `/TD`, `/Note`, `/Reference`, `/BibEntry`,
  `/Code`); the converter splits paragraphs whenever this changes
  between adjacent spans. Tight-gap layouts (pdfa_049-style) no
  longer merge. (D5)
- **Same-baseline gate against form-heading over-fragmentation.** D5
  alone over-split horizontal heading bands like
  `# Form / # 1040 / # U.S. Individual Income Tax Return` into three
  separate headings. The block-id transition now fires only when the
  spans are also on different visual lines; same-baseline pieces
  re-join into one heading. (D5b)
- **Multi-column gutter detection.** Two spans on the same baseline
  separated by a horizontal gap > `max(3 × font_size, 30 pt)` are
  treated as belonging to different columns even when their
  block_ids would say otherwise — newspapers and two-column academic
  papers no longer concatenate cross-column tokens. (D5c)
- **Backward-x reading-order wrap detection.** When the structure
  tree's reading order goes column-major (last span of column 1 at
  x=976 immediately followed by first span of column 2 at x=192,
  same baseline), the converter now recognises the wrap as a
  paragraph break instead of joining the two into a nonsense token
  like `constitutionAssailing`. (D5d)
- **Geometric heading + list-prefix detection for untagged docs.**
  Bold + 5 % size bump promotes to H4. New
  `is_ordered_list_marker(text) -> Option<u32>` recognises `1.` /
  `12.` / `a)` / `iv.` / `A.` while conservatively rejecting figure
  captions (`1.1 Foo`) and years (`1986`). Bullet or ordered
  marker on a new line forces a paragraph break regardless of the
  geometric gap. (D2 / D3 / D4)

### RTL text — safe-by-default

- **Spurious `**bold**` markers around Arabic contextual glyphs are
  now stripped.** Initial / medial / final shape transitions
  routinely flipped the font-weight detector and emitted single-letter
  emphasis runs; the converter now recognises and removes them.
- **Bidi reorder is OFF by default.** An earlier draft of D7 ran
  `unicode-bidi`'s visual→logical reorder on every RTL line; that
  broke previously-correct logical-order PDFs (Hebrew name `בנימין`
  was being reversed to `ןימינב`). Without a reliable signal for
  source order, the safer behaviour is to preserve the input
  ordering. The reorder helper remains exported from
  `text::bidi::reorder_visual_to_logical` for callers that *know*
  their input is in visual order.

### Markdown output

- **Inline-image base64 data URIs capped at 200 KB.** PDFs with
  high-resolution diagrams previously inflated markdown output by
  10–20× (one 1.9 MB academic paper produced 11.3 MB of markdown).
  Images that exceed the cap now emit an HTML-comment placeholder
  noting the suppression and the original size. File-based image
  output (`image_output_dir`) is unaffected.

### Tests

- 80+ new unit tests in `pipeline::converters::markdown::tests`,
  `structure::traversal::tests`, and `text::bidi::tests` covering
  every defect with TDD-shaped RED→GREEN cases plus parametrised
  variations (all six heading levels, all three list roles, edge
  cases like clamped levels, baseline jitter, three-column layouts,
  the IA_0047 backward-x reproducer, etc.).

### Empirical impact

Validated against v0.3.35 baseline on a 369-PDF regression spanning
academic, government, forms, newspapers, technical, theses, IRS,
pdfium, pdfjs, safedocs, and slow-corpus subsets:

- **0 catastrophic regressions** (no `HEAD_FAIL`, no `SHRUNK_BIG` on
  real content; the three sub-50-byte SHRUNK cases are pdfjs test
  fixtures where D5b same-line joining suppresses geometric heading
  detection on minimal content).
- **Token Jaccard vs pdfium and pdftotext: median 1.000 (perfect),
  ≥0.95 on 95/106 fixtures.**
- **Token Jaccard vs pymupdf4llm: median 0.978**, ≥0.95 on 65/106
  fixtures.
- **~2× more headings emitted than pymupdf4llm** across the corpus —
  the structure-tree wiring lets pdf_oxide pick up section titles
  that font-only heuristics miss.
- Per fixture (issue #377): nougat_002 0→4 H1s + 5→34 bullets;
  nougat_011 64→266 lines; word365_structure 0→1 H1 + 2→3 bullets;
  2023-06-20-PV 0→4 H + 0→5 bullets.

### Community Contributors

- **[@Goldziher](https://github.com/Goldziher)** ([kreuzberg](https://github.com/kreuzberg-dev/kreuzberg)) —
  filed [#377](https://github.com/yfedoseev/pdf_oxide/issues/377)
  with a 727-document benchmark methodology (block-level SF1 +
  token-level TF1) comparing pdf_oxide against pdfium, plus 9
  reproducer PDFs covering the worst structural-extraction
  regressions. The clarity of that report (per-pattern bucketing,
  per-fixture gaps, and an explicit "TF1 within ±3 % so text content
  is fine, structure is the issue" framing) made the entire
  investigation tractable. The single-PR unlock that drove this
  release was identifying that pdf_oxide had a complete structure-tree
  parser whose output the markdown converter was discarding — that
  framing came directly from the issue.

## [0.3.35] - 2026-04-19
> Narrow-glyph doublet preservation in text extraction

### Text extraction correctness

- **Adjacent narrow-glyph doublets no longer collapsed at small font sizes (#378, PR #379).**
  `TextExtractor::deduplicate_overlapping_chars` and
  `deduplicate_overlapping_spans` used a hardcoded 2 pt absolute threshold to
  detect duplicate glyphs from stroke+fill render passes. For narrow glyphs
  (`l`, `r`, `I`, `i`) in compact fonts at small sizes the per-glyph advance
  width drops to ≤ 2 pt (Helvetica `l` ≈ 2.5 pt at 9 pt), so legitimate
  adjacent doublets one full advance apart fell inside the dedup window and
  one of the two glyphs was silently dropped. Visible corruption included
  `controller → controler`, `billed → biled`, `warranty → warrnty`,
  `following → folowing`, and `VIII → VII`. Builds on prior #102 / #253,
  which added same-text and same-character identity guards but kept the 2 pt
  threshold — this fix addresses the residual case where both glyphs are
  identical (passing the identity check) yet still legitimate neighbours.
  Threshold now scales with each glyph's own `advance_width` (fallback
  `bbox.width`) as `min(advance_width * 0.30, 2.0)`. Real render-pass
  duplicates sit well under 5 % of one advance apart and continue to
  collapse; heaviest kerning observed in the wild is ≤ 20 % of advance, so
  legitimate kerned neighbours are preserved. Tunables hoisted to
  `TextExtractor::DEDUP_OVERLAP_RATIO` / `DEDUP_OVERLAP_CAP_PT` associated
  constants so both dedup paths share one source of truth. Regression
  coverage spans the matrix of four narrow glyphs × three small body-text
  sizes (7 / 9 / 11 pt) on both the per-char and per-span paths, plus
  positive cases proving stroke+fill duplicates at ~0 pt offset still
  collapse.

### Community Contributors

- **[@Hugues-DTANKOUO](https://github.com/Hugues-DTANKOUO)** — Reported #378
  with a precise root-cause analysis (the 2 pt absolute threshold falling
  below one advance width for narrow glyphs in compact fonts at small sizes)
  and authored PR #379 with the advance-scaled threshold and a
  parametrised regression matrix covering the four narrow glyphs across
  three body-text sizes.

## [0.3.34] - 2026-04-18
> Idiomatic page API, structured tables, column-order, image, and ICC colour fixes

### API — Page abstraction (#371)

All four language bindings now expose a page object so callers can iterate a
document and call extraction methods on the page directly. Named consistently
as `Page` in Python, Node.js, C#, and Go.

```python
with PdfDocument("paper.pdf") as doc:
    for page in doc:           # len(doc), doc[i], doc[-1] also work
        text = page.text
        md   = page.markdown(detect_headings=True)
```

- **Python** — `Page` with lazy properties: `text`, `chars`, `words`, `lines`,
  `spans`, `tables`, `images`, `paths`, `annotations`; methods: `markdown()`,
  `plain_text()`, `html()`, `render()`, `search()`, `region()`. The pre-existing
  editor `PdfPage` is unchanged.
- **Node.js** — `Page` with cached `width`/`height`/`rotation` and extraction
  methods. `[Symbol.iterator]` and `page(index)` added to `PdfDocument`. Six
  previously native-only methods wired into the TS layer: `extractWords`,
  `extractTextLines`, `extractTables`, `extractPaths`, `getEmbeddedImages`,
  `ocrExtractText`.
- **C#** — `Page` with full sync + async surface. `doc.Pages`
  (`IReadOnlyList<Page>`) and `doc[i]` indexer added to `PdfDocument`.
- **Go** — `Page` struct with full method surface. `doc.Page(i)` and
  `doc.Pages()` added to `PdfDocument`.

### API — Structured table extraction with consistent naming (#289)

`extract_tables()` returns structured data — rows, cells with text and bounding
boxes — not just Markdown. Available on both `PdfDocument` and the new `Page`
objects across all bindings, with a single consistent type name `Table`:

| Language | Type | Cell access |
|---|---|---|
| Rust   | `Table`             | iterate `rows[i].cells[j]` |
| Python | `dict`              | `row["cells"][i]["text"]` |
| Go     | `Table`             | `table.CellText(row, col)` |
| C#     | `Table`             | `table.CellText(row, col)` |
| Node.js| `Table` (interface) | `table.cells[row][col]` |

C# previously returned only `(int RowCount, int ColCount)` tuples — now returns
a proper `Table[]` with cell text accessors, matching Go and Rust.

### Text extraction correctness

- **Multi-column reading-order interleaving fixed (#319).** On untagged
  multi-column PDFs (academic textbooks, genetics references), `extract_text`
  was applying XY-cut column ordering inside `extract_spans()` and then
  re-sorting with row-aware sort in `extract_text_with_options`, undoing the
  column structure. Result: garbled fragments like `accompaally` (= "accompa"
  from column 1 + "ally" from column 2). Fix: skip the row-aware re-sort when
  the page is genuinely multi-column. Verified on Hartwell Genetics, Murphy ML,
  and Kandel Neural Science textbooks — all known garbled tokens eliminated.
- **XY-cut column-detection improvements** for mixed-layout pages (table + body
  text). Wide spans (>55% of region width) excluded from the projection density
  so tab-expanded table rows no longer fill the column gutter. Single-character
  spans (table cell values like `G`, `T`) excluded from projection so they
  don't scatter across the gutter. Coverage check uses character-count estimate
  rather than bbox width so tab-padded rows don't masquerade as dense body text.
- **Sparse-layout false-positive guard** for `is_multi_column_page`. Copyright
  pages, title pages, and colophons can produce two X-center peaks with only
  7-10 spans per "column" — these are no longer treated as multi-column,
  preventing XY-cut from splitting sentences whose halves are at different X
  positions on the same line.
- **Font-aware column-shape gate** in `is_multi_column_page`. Fax-style and
  scattered-fragment layouts (each row built from several individually
  positioned word fragments) used to clear every prior multi-column check
  and routed through XY-cut, which then read the page column-major and could
  reverse fragments within a row. The new gate measures the fraction of
  side-spans falling into the largest X-cluster (cluster gap derived from the
  page's dominant em); body text scores ≥ 0.5 while scattered layouts score
  < 0.4. Pages that fail either side fall back to row-aware sort, so
  scanned-fax PDFs again read left-to-right line-by-line. Per-page font
  statistics are computed once via the new `pdf_oxide::layout::PageFontStats`
  type and reused by every threshold the layout pipeline derives.
- **Newline insertion on backwards-X jumps in span join.** When the upstream
  sort handed the join loop two same-baseline spans whose X positions went
  backwards (a multi-column page whose XY-cut routing groups column-side
  spans across rows so adjacent iteration items share a Y band but belong
  to different visual rows), no separator was being inserted and texts
  glued together — producing tokens like `instancesinstancesinstances` from
  three table-header cells in a stats grid. Same-baseline pairs whose
  delta-x is more negative than 3 em now emit a newline.

### Distribution

- **Node.js Linux prebuild now portable across glibc 2.35+ systems.** Previous
  builds were dynamically linked against `libstdc++.so.6` requiring
  `GLIBCXX_3.4.31` (GCC 13+), failing to load on Debian 12 stable, Ubuntu
  22.04, and RHEL 8/9. Fix: `binding.gyp` now passes `-static-libstdc++` and
  `-static-libgcc`, and the Linux runner is pinned to `ubuntu-22.04` /
  `ubuntu-22.04-arm` (glibc 2.35). The resulting `.node` is fully self-contained
  for C++ runtime — `ldd` shows only `libm`/`libc`. Size impact: +210 KB.
- **Go installer documents `@latest`.** `go run github.com/yfedoseev/pdf_oxide/go/cmd/install@latest`
  is now the recommended install command (the installer auto-resolves the
  matching version via `runtime/debug.ReadBuildInfo()`).
- **pkg.go.dev now shows Go documentation.** The Go module (rooted at
  `go/go.mod` with module path `github.com/yfedoseev/pdf_oxide/go`) was
  returning `Documentation not displayed due to license restrictions`
  because pkg.go.dev's licensecheck only inspects the module's own
  subtree — it does not walk up to the repo root where
  `LICENSE-APACHE` + `LICENSE-MIT` live. Fix: duplicate both files into
  `go/LICENSE-APACHE` and `go/LICENSE-MIT`, filenames both on
  pkg.go.dev's accepted list. Takes effect on the next tag.
- **npm, NuGet, and PyPI packages now embed both licence files.** Same
  class of gap as the Go fix: `js/package.json`'s `files` list, the C#
  `.csproj`, and the maturin `[tool.maturin] include` all omitted the
  licence text so shipped artifacts lacked the notice MIT requires.
  `js/package.json`'s `license` field also flattened to `"MIT"`,
  contradicting the crate's declared `MIT OR Apache-2.0`; corrected to
  match. The C# csproj carried a deprecated `<LicenseUrl>` alongside
  `<PackageLicenseExpression>` that NuGet warns on — removed.
- **`LICENSE-MIT` copyright corrected.** All four `LICENSE-MIT` copies
  (root, `go/`, `js/`, `csharp/PdfOxide/`) carried `Copyright (c) The
  Rust Project Contributors` left over from the `cargo init` template.
  Updated to `Copyright (c) 2025-present Yury Fedoseev`. Verified with
  google/licensecheck — all four still classify as 100% MIT, so
  pkg.go.dev / NuGet / npm license detection is unaffected.

### CI

- **Free-disk-space step added to all Ubuntu jobs that do heavy Rust + Python
  builds.** A v0.3.33 release-pipeline failure (`No space left on device` on
  `actions-runner` log writes) traced to GitHub Ubuntu runners filling up at
  the `maturin build --release` step. Now applied to `python.yml` test job
  (was only one fixed initially), `ci.yml` Python Bindings + WASM Build jobs,
  and `release.yml` Python wheel build matrix (Linux targets only via
  `if: runner.os == 'Linux'` guard).

### Image extraction correctness

- **4-bit-per-component Indexed images no longer decode to vertical-stripe
  noise (#375).** The PNG predictor decoder was honouring the numeric
  `/Predictor` value from `/DecodeParms` instead of the per-row filter
  tag byte written into each row. ISO 32000-1:2008 §7.4.4.4 makes the
  per-row tag authoritative: a producer may declare `/Predictor 12` (Up)
  on the parameters and still write tag 0 (None) on every row. Reading
  the declared predictor instead produced Up-cascade on raw index bytes,
  rendering a 710×1012 scanned-book page as a diagonal-stripe noise
  pattern. Reported by @Charltsing.
- **Indexed palette streams whose first byte is `0x0D` (CR) or `0x0A` (LF)
  no longer decode to solid black (#375).** `decode_stream_data` was
  running a post-parse `trim_leading_stream_whitespace` pass that
  stripped CR/LF bytes from the start of every unencrypted stream. The
  parser already consumes exactly one EOL after the `stream` keyword per
  ISO 32000-1:2008 §7.3.8.1, so re-trimming corrupted binary streams
  that legitimately start with those bytes. For an Indexed-backed image,
  shrinking a 4-byte CMYK palette `0d 0c 0c 04` to 3 bytes pushed every
  lookup into the expander's out-of-range branch, producing `(0,0,0)`
  for every pixel. Reported by @Charltsing.
- **DeviceCMYK → DeviceRGB fallback now matches ISO 32000-1:2008 §10.3.5
  (#375).** All CMYK→RGB paths — image-level bulk conversion,
  Indexed-CMYK palette expansion, content-stream fill/stroke colour
  state, JPEG CMYK decoding — now use the spec's additive-clamp formula
  `R = 1 − min(1, C + K)`. Four inline copies and three helper functions
  were collapsed onto this single form; the common multiplicative
  `(1-C)(1-K)` variant differed on heavily-inked samples and was the
  default we inherited from imaging libraries, not what the spec specifies.

### Colour management (new)

- **Real ICC profile-driven colour conversion via qcms (#375; opt-out
  `icc` feature, on by default).** When a PDF's `/ICCBased` colour space
  or `/OutputIntents → DestOutputProfile` provides an ICC profile, image
  extraction now compiles it to a `qcms::Transform` and routes CMYK
  samples through the CMM instead of the §10.3.5 fallback. RGB- and
  gray-ICCBased profiles use the same pipeline. The graphics-state
  rendering intent (`/Intent` on image dictionaries, `/RI`, or the `ri`
  operator) is honoured; unrecognised intent names fall through to
  `RelativeColorimetric` per §8.6.5.8. qcms is pure Rust (no C/FFI) so
  WASM and C# AOT builds keep working; opt out with
  `default-features = false`. Reported by @Charltsing.
- **New `pdf_oxide::color` module** exposes `IccProfile`, `IccHeader`,
  `RenderingIntent`, and `Transform` for consumers that want to drive
  colour conversion directly.
- **Measured impact on a representative CMYK-heavy fixture** (218
  images, `/ICCBased 4` throughout): mean PSNR vs poppler's reference
  rendering improved from 27.9 dB (§10.3.5 fallback) to 39.2 dB
  (qcms). Worst-case PSNR rose from 16.4 dB ("visibly wrong saturation")
  to 33.8 dB ("perceptually indistinguishable"). A representative blue
  swatch shifted from `RGB(62, 142, 252)` to `RGB(58, 123, 190)` vs the
  ICC reference's `RGB(62, 124, 191)`.

### Community Contributors

- **[@SeanPedersen](https://github.com/SeanPedersen)** — Proposed the page-first
  API (#371) with lazy evaluation and sequence semantics. Python follows his
  design exactly; extended to Node.js, C#, and Go.
- **[@pdenapo](https://github.com/pdenapo)** — Requested structured table
  extraction returning data structures rather than Markdown (#289), which
  prompted the cell-text API surfacing in C# / Node.js and the `Table` rename
  for cross-language consistency.

## [0.3.33] - 2026-04-16
> Text extraction, image correctness, and memory safety fixes

### Text extraction correctness

- **ToUnicode CMap miss returns U+FFFD instead of ASCII ciphertext (#363).** Subset Type0 fonts whose ToUnicode CMap doesn't cover a CID now emit the replacement character instead of falling through to the Identity-H `cid-as-Unicode` path that produced strings like `%B+$%8A//$2*%01*1%6APP`.
- **Intra-word TJ kerning no longer splits words (#365).** Letter-pair kerning of 0.10–0.20 em inside single words (`[(diffe) -150 (rent)]`) no longer triggers space insertion. Validated on 5 Kreuzberg fixtures — zero split-word patterns.
- **Cyrillic / non-Latin text recovered from UTF-8 mojibake (#317).** Fonts with Latin-only encoding and no ToUnicode CMap that carry raw UTF-8 byte sequences now decode correctly. Validated on `issue20232.pdf` — Russian engineering text readable.
- **FlateDecode partial-recovery rejects garbage output (#364).** MS Reporting Services PDFs (`nougat_026.pdf`) whose content streams failed mid-decompress were returning 128 bytes of pseudo-random data. Partial-recovery paths now validate output via `looks_like_real_stream` before accepting. Pages 1/2/5 go from 0 → 848/792/321 bytes.

### Image extraction

- **Indexed + ICCBased palette correctly resolves component count (#373).** Unresolved ICC stream references inside the Indexed base array caused `/N` to default to 3 instead of reading the actual value (4 = CMYK), producing diagonal-stripe artifacts. Reported by @Charltsing.
- **Lab-base Indexed palettes converted to sRGB (#337).** Palette bytes in CIE L\*a\*b\* are now converted through Lab→XYZ→sRGB instead of being reinterpreted as raw RGB.

### Memory and performance

- **All internal caches bounded (PR #369, #354).** Object cache (64 MB), font caches (256–512 entries), XObject span/image caches (1024 entries), and global CMap cache (1024 entries) all use FIFO eviction. Cache utilities extracted to `src/cache.rs`.
- **Path extraction OOM on chart-heavy PDFs fixed (PR #369).** Added CTM-aware `processed_xobjects` dedup — same XObject at same position is deduplicated, same XObject at different positions processes separately.
- **Mutex poison resilience.** `MutexExt::lock_or_recover()` replaces 72 `.lock().unwrap()` calls.

### Dependencies

- **RustCrypto cipher 0.5 ecosystem (PRs #352, #295, #291).** `aes` 0.8→0.9, `cbc` 0.1→0.2, `sha2` 0.10→0.11, `sha1` 0.10→0.11, `md-5` 0.10→0.11.

### Test suite

- 13 dead/stale ignored tests removed; 3 previously-ignored tests fixed and un-ignored.
- Regression tests added: ToUnicode CID-miss (3 tests), FlateDecode stream boundary framing (4 variants), TJ intra-word kerning, Cyrillic encoding and UTF-8 sniff (2 tests), dedup flow-prose preference, reading-order glyph sort stability (2 tests), Indexed Lab palette conversion.
- **Suite: 6,300 passed, 0 failed, 228 ignored.**

### Community Contributors

Thank you to everyone who reported issues, filed reproducers, or contributed code for this release!

- **[@Charltsing](https://github.com/Charltsing)** — Reported the Indexed + CMYK image extraction failure (#373) with a reproduction PDF and screenshot comparison against pdfimages (xpdf), which exposed the unresolved ICC stream reference bug that had been silently producing garbled diagonal-stripe artifacts since the Indexed palette support landed in v0.3.27.
- **[@ddxtanx](https://github.com/ddxtanx)** — Reported the unbounded memory growth during multi-page extraction (#354) with profiling data that showed object and font caches consuming 200 MB+ on a 609 KB arXiv PDF. This drove the bounded-cache work in PR #369.
- **[@andrewjradcliffe](https://github.com/andrewjradcliffe)** — Authored PR #369 implementing bounded FIFO caches for all internal caches, CTM-aware XObject dedup for the path extractor OOM, `MutexExt` poison-recovery trait, Python binding hardening, and markdown inter-group spacing. The PR also included comprehensive unit tests for all new cache types.

## [0.3.32] - 2026-04-15

### Release pipeline

- **Fix `x86_64-pc-windows-gnu` native-lib build failing the v0.3.31 release.** The new `scripts/shrink-staticlib.sh` introduced in v0.3.31 ran `objcopy --strip-debug` on every archive member. The MinGW cross-compile toolchain emits split-debug `.dwo` members that contain *only* DWARF sections; after stripping those sections the member has no sections left and objcopy aborted the whole archive with `'...rcgu.dwo' has no sections`, failing the job that produces the Go Windows-x64 FFI tarball. Fix: drop `.dwo` archive members via `ar d` before invoking `objcopy`. No functional change to Rust, Python, Node, WASM, or C# artifacts — those built and uploaded successfully in v0.3.31; this release exists solely to unblock the Windows-x64 Go install path.

## [0.3.31] - 2026-04-15

### Text extraction correctness

- **`extract_text(n)` returned page 0's content for every `n` on PDFs that share one Form XObject across all pages (#346, B1).** Certain producers (notably ExpertPdf) emit one big Form XObject containing every page's text and give each page's content stream a different CTM translation to clip into its slice. Two cache/filter bugs stacked: (1) `xobject_spans_cache` keyed spans by `ObjectRef` only and returned CTM-transformed page-0 coordinates to every subsequent page; (2) even once the cache was CTM-gated, the extractor had no awareness of the content-stream `W n` clipping operator, so every page emitted the whole stack at distinct but out-of-bounds Y coordinates. Fix: cache only when the caller CTM is identity, and post-filter extracted spans by the page's MediaBox (with a 2pt bleed tolerance). Added `Matrix::is_identity()`. Regression test at `tests/test_b1_shared_form_xobject_per_page_ctm.rs`. Largest single-fixture improvement in this release — `nougat_005.pdf` TF1 **0.254 → 0.901**.

- **Running-artifact detector stripped the cover-page title when it happened to repeat as the per-page running header (B3).** Reports like "Fiscal Year 2010 Appropriations Act" or "University of Oklahoma 2009" appear at the top of every page *and* are the document title on page 1. The detector classified them as chrome and removed them from page 1 too. Fix: track first-seen page per signature (across all pages, not just body-content pages), skip the artifact marking on that first occurrence. Covers the edge case where the cover page is all-chrome and would otherwise be skipped by the body-content gate. Regression test at `tests/test_b3_first_occurrence_of_running_header_kept.rs`.

- **Multi-column reading order via XY-cut (B4).** `extract_text` used a row-aware Y-band sort that interleaved left/right columns on newspaper / academic layouts: `LeftCol-row1 RightCol-row1 LeftCol-row2 …`. Added `is_multi_column_page` heuristic (body-span X-center histogram with vertical-overlap confirmation, a 15% chrome-band exclusion so header banners don't trip the detector, 25%-per-side minimum column mass) that routes detected multi-column pages through the existing `XYCutStrategy`. Single-column pages stay on the cheap row-aware path. Regression test synthesises a 2×20 interleaved grid at `tests/test_b4_two_column_reading_order.rs`.

- **Stroke+fill labels no longer produce doubled words (B7).** Map/poster PDFs render every label twice — once stroked for the outline, once filled — and both passes landed as distinct `TextSpan`s at essentially the same CTM. The downstream merge step concatenated them, producing `"EverestEverest"`, `"CentralCentral"`. New `dedup_stroke_fill_overlap` runs before existing positional dedup: bucket by lowercased text, drop any later span whose bbox overlaps an earlier same-text span by ≥ 70% IoU. Conservative thresholds (≥2-char minimum, using `.chars().count()` not `.len()` so non-ASCII glyphs are handled correctly). Regression test at `tests/test_b7_stroke_fill_dedup.rs`.

- **Soft-hyphen line-break rejoin (B8a).** Typographic hyphenation — `"scruti-\nneer"` for `"scrutineer"`, `"disinfec-\ntion"` for `"disinfection"` — previously preserved the hyphen and newline. Added `dehyphenate_line_breaks` to the plain-text cleanup pipeline: rewrites `<lowercase>-[ \t]*\n[ \t]*<lowercase>` → concatenation. Conservative on both sides (requires ASCII lowercase before and after) so compound hyphens (`"state-of-the-art"`), proper-noun fragments (`"co-\nWorker"`), and bullet markers stay intact.

- **TrueType cmap format 0 parser (B9).** Microsoft Office Word/Excel subset fonts (Calibri, Times New Roman) sometimes ship only a format-0 (legacy 1-byte Mac Roman) cmap. Previously these fonts bailed with "Unsupported cmap format: 0" and the font had no glyph→Unicode mapping, which cascaded into text extraction losing content from that font. Added `parse_cmap_format0` — reads the 6-byte header + 256 glyph IDs, maps byte codes 0x00–0x7F as ASCII pass-through and 0x80–0xFF through the full Mac Roman → Unicode table (so byte 0x8A correctly decodes to `ä`). Truncated glyph arrays surface as parse errors rather than silent zero-glyph output.

### Verification infrastructure

- **`tools/benchmark-harness/` — TF1/SF1 extraction quality measurement crate (#320).** New workspace member that computes **TF1** (bag-of-words F1 on lowercase alphanumeric tokens) and **SF1** (block-weighted structural F1 with LIS ordering penalty) against ground-truth markdown. Methodology mirrors Kreuzberg's harness so numbers are directly comparable. Includes engine adapters for `pdf_oxide` (in-process), `pdftotext` (poppler subprocess), and `pdfium` (gated behind `--features pdfium`); a consensus-baseline mode for corpora without manual ground truth; and a `diff` subcommand with regression gates (default: fail on mean TF1 drop > 0.5pp or per-fixture drop > 5pp). `scripts/fetch-fixtures.sh` clones Kreuzberg's Apache-2.0 fixture set without vendoring PDFs into our repo. Makefile targets: `make benchmark-fetch`, `make benchmark-run`, `make benchmark-compare`. 18 unit tests.

  **Cumulative impact on the 78-unique-fixture Kreuzberg corpus vs v0.3.30 baseline:**
  - TF1 mean: 0.919 → **0.930** (+1.1pp)
  - TF1 p10 (hard tail): 0.776 → **0.849** (+7.3pp — tied with pdftotext)
  - SF1 mean: 0.337 → 0.355 (+1.8pp) — pdf_oxide leads pdftotext by +10.8pp SF1 on this corpus
  - Runtime: −42%
  - Zero per-fixture TF1 regressions > 0.5pp

  Four follow-up work items filed with precise reproducers: #363 (ToUnicode CID-miss on specific MS Office subset fonts), #364 (FlateDecode stream offset bug on MS Reporting Services PDFs), #365 (intra-word TJ space calibration), #366 (CI wiring for the harness), #367 (docling-parse adapter), #368 (markdown adapter for the pdf_oxide engine).

### Earlier bug fixes included in this release

- **Rendering: `Page index N not found by scanning` on PDFs whose xref mis-flags page objects as free** — when a producer emits a corrupted xref table with `f` entries pointing at real objects (common in several large regulatory PDFs in the wild), `load_object` previously resolved every such object to `Null` per §7.3.10. If the page objects were uncompressed the page tree traversal would bottom out in nulls; if they were packed into `/Type /ObjStm` the `get_page_by_scanning` fallback never reached the content at all. Two recovery paths now trigger before the `Null` fallback: (1) if the file body contains an `N G obj` marker for the supposedly-free id, load it from the scanned offset — the same mechanism already used for objects missing from the xref entirely; (2) if not found in the body, perform a one-time raw-pattern scan for every `/Type /ObjStm` in the file, parse each, and cache all contained objects (overwriting the earlier `Null` entries). The `get_page_by_scanning` fallback now unions `xref.all_object_numbers()` with the newly-cached ObjStm ids so pages whose xref slot says `f` but whose content lives inside an object stream are visible to the scanner; its heuristic second pass (page-shaped dicts without `/Type`) now also runs as a complement rather than only when pass 1 finds zero pages. Also unifies the `id <= 10` code path with the general path — previously low-numbered free-flagged objects hit a broken "fall through" branch that still ended up Null.
- **Rendering: `Invalid object number in header: j` on PDFs whose xref offsets are off by a handful of bytes** — the same SEBI-style PDF also carries a second corruption shape: `in_use=true` xref entries whose byte offsets point ~3 bytes BEFORE the real `N G obj` header (into the previous object's `endobj` tail). The existing `find_object_header_backwards` fallback only triggered when no `obj` keyword was found at all, not when the keyword parsed but the preceding tokens were junk like `j`. `load_uncompressed_object_impl` now catches the parse-as-number failure, re-queries `scan_for_object` for the same id, and if the scan recorded a different offset retries from there. With both fixes live, the report's 253-page PDF goes from 0 → **253 pages renderable** (was 0 before the free-flag recovery, 200 with just that fix, 253 with the combined offset recovery). Regression tests in `tests/test_xref_free_flag_corruption.rs` cover both corruption shapes plus the clean-baseline and genuinely-free negative cases.

### BREAKING — Go module install flow

- **Native Rust libraries are no longer committed to `go/lib/`** — after landing the 63% staticlib shrink (below), the committed payload would still have been ~130 MB per release, accumulating indefinitely in git history. v0.3.31 instead publishes per-platform `pdf_oxide-go-ffi-<platform>.tar.gz` as GitHub Release assets and ships a small Go installer at `go/cmd/install`. Consumers run it once per machine:
  ```
  go get github.com/yfedoseev/pdf_oxide/go
  go run github.com/yfedoseev/pdf_oxide/go/cmd/install@latest
  # Installer prints the CGO_CFLAGS / CGO_LDFLAGS to export
  ```
  The installer downloads the matching asset into `~/.pdf_oxide/v<version>/`, SHA-256 verifies it against the signed `.sha256` published alongside, and either prints the env vars to export or (with `--write-flags=<dir>`) generates a `cgo_flags.go` next to the user's code. Monorepo / source-tree builds use `-tags pdf_oxide_dev` which points CGo at `target/release/libpdf_oxide.a` directly — no installer needed.

  **`@latest` just works** — the installer reads its own module version from `runtime/debug.ReadBuildInfo()`, so every tagged release auto-matches its FFI assets without a release-time sed step. `go run .../cmd/install@latest` always resolves to the matching `.tar.gz`.

  **Why:** shipping ~130 MB of binary in git per release was bloating clone time and accumulating to GBs over dozens of releases. This is the approach Kreuzberg (https://github.com/kreuzberg-dev/kreuzberg/blob/main/packages/go/v4/cmd/install/main.go) and similar Rust-in-Go projects use. Repo size per release bump drops to ~0 KB; clone stays fast forever.

  **Migration:** consumers upgrading from v0.3.30 must run the install command once and export the printed `CGO_*` env vars (or add them to their shell profile / CI env). No code changes to the Go API. `go get` without the install step will fail at link time with `undefined reference to pdf_document_open ...` — the installer fixes this.

- **Go release pipeline hardening** — three ordering + integrity fixes landed alongside the install-flow switch:
  - **SHA-256 gate end-to-end.** `package-go-ffi` emits `pdf_oxide-go-ffi-<platform>.tar.gz.sha256` next to each tarball (attached to the GitHub Release). The Go installer downloads both and aborts with a checksum mismatch if they don't match; `--skip-checksum` bypasses for offline/air-gapped installs. The same `.sha256` is verified by `verify-go-install` in CI before the release is even published.
  - **verify-go-install is now a publish gate.** `create-release` depends on `verify-go-install` — that job extracts the freshly-built tarball, matches the sha256, then builds a `FromMarkdown → Save → Open → PageCount` consumer against a local `replace` directive. A broken `.a`, a missing symbol, or a stale `cgo_dev.go` blocks the release instead of leaking through.
  - **`go/v<version>` tag is pushed last, not first.** A new `tag-go-module` job runs *after* `create-release` has uploaded the FFI tarballs. Previously the tag was pushed during packaging, creating a window where `go install @latest` could resolve a tag whose FFI assets 404'd. Tag creation is gated on `!contains(version, '-')` so prerelease `-rc.1` tags never reach sum.golang.org.

### Release Infrastructure (artifact size reductions)

- **Shrink Rust static libs 62.8% before packaging** — Rust-produced staticlibs carry 35 MB of `.llvmbc` (LLVM bitcode for cross-crate LTO) + 4 MB of DWARF per platform, none of which CGo's linker or node-gyp ever uses. New `scripts/shrink-staticlib.sh` strips both via `objcopy --remove-section=.llvmbc --remove-section=.llvmcmd --strip-debug` (Linux / Windows-GNU) or `strip -S` (macOS) inside the `build-native-libs` job. Per-platform `libpdf_oxide.a` drops from ~71 MB to ~26 MB. All 85 Go-consumed FFI symbols verified intact post-strip.
- **Strip the npm `.node` addon** — `node-gyp rebuild` left the addon unstripped (17 MB, `with debug_info, not stripped`). Added post-build `strip --strip-unneeded` (Linux) / `strip -x` (macOS) in the `build-nodejs` job. Combined with the upstream staticlib shrink, the Linux `.node` is expected to drop from 17 MB to ~7 MB.
- **Drop sourcemaps from the npm tarball** — `js/tsconfig.json` sets `declarationMap: false` + `sourceMap: false` for the published build. File count falls from 211 → 107 (removes 104 `.js.map` / `.d.ts.map` files). `.d.ts.map` was never useful to consumers; `.js.map` is moot without TS sources, which we don't ship.
- **Fix crate sdist leak (pulled in 47 unrelated files)** — Cargo's `include` uses gitignore-style globs, so the bare `"README.md"` entry was matching every README.md recursively, including 27 `js/node_modules/*/README.md` dependency READMEs and 20 subdirectory READMEs. Anchored all patterns with a leading `/` — sdist file count 308 → 264.
- **Tighten NuGet symbol package** — `EmbedAllSources` dropped from `true` to `false` in `csharp/PdfOxide/PdfOxide.csproj`. SourceLink + the embedded PDB already serve sources on demand from the git SHA, so embedding every source file into the `.snupkg` was pure bloat. Added defensive `<None Remove="..\..\target\**\*.pdb" />` to prevent native PDBs from landing in `runtimes/` (nuget.org's snupkg validator rejects native PDBs).

### Thanks

Issues reported or features requested by: [@Goldziher](https://github.com/Goldziher) (#320 benchmark harness), [@ddxtanx](https://github.com/ddxtanx) (#346 sort-order panic, #354 memory leak on page 12), [@frederikhors](https://github.com/frederikhors) (#325 rendering regression), [@Charltsing](https://github.com/Charltsing) (#344 CMYK JPEGs), [@FireMasterK](https://github.com/FireMasterK) (#345 page-scan failures), [@Jeevaanandh](https://github.com/Jeevaanandh) (#353 yanked libflate dep).

## [0.3.27] - 2026-04-12

### Language Bindings

- **Go: migrate from cdylib to staticlib for self-contained binaries (#334)** — `pdf_oxide` now produces `libpdf_oxide.a` alongside the cdylib (new `staticlib` entry in `Cargo.toml`'s `crate-type`), and `go/pdf_oxide.go` links the archive directly via per-platform `#cgo ... LDFLAGS` with the exact system-library list rustc needs. The resulting Go binary is fully self-contained — no `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` / `PATH` configuration required. Windows x64 is produced via a new `x86_64-pc-windows-gnu` cross-compile row in the release matrix; Windows ARM64 temporarily stays on dynamic `pdf_oxide.dll` until `aarch64-pc-windows-gnullvm` stabilises.
- **Node.js: ship prebuilt native bindings via platform subpackages (#335)** — switched to the napi-rs style prebuilt-binary model: the main `pdf-oxide` package drops the install hook, declares per-platform `pdf_oxide-<triple>` subpackages as `optionalDependencies`, and ships only compiled `lib/` + `README.md`. `binding.gyp` links the `libpdf_oxide.a` / `pdf_oxide.lib` staticlib with per-OS system-library lists, so the resulting `.node` is self-contained. `npm install pdf-oxide` now works out of the box with no TypeScript, Python, C++ toolchain, or native lib on the consumer's machine.
- **C#: migrate all 881 P/Invoke declarations from DllImport to LibraryImport for NativeAOT (#333)** — `PdfOxide` on NuGet is now NativeAOT-publish-ready and trim-safe. Target frameworks trimmed to `net8.0;net10.0`. `IsAotCompatible=true` and `IsTrimmable=true` flags enabled. The `build-csharp` release job gains a `Verify NativeAOT publish` step that `dotnet publish` a tiny consumer with `PublishAot=true` + `TreatWarningsAsErrors=true` on net10.0. Requested by @Charltsing.
- **OCR FFI bridge for Go, C#, and Node.js** — added 4 `pub extern "C" fn` declarations to `src/ffi.rs` wrapping `src/ocr::OcrEngine`: `pdf_ocr_engine_create`, `pdf_ocr_engine_free`, `pdf_ocr_page_needs_ocr`, `pdf_ocr_extract_text`. Each has `#[cfg(feature = "ocr")]` with the real implementation and `#[cfg(not(feature = "ocr"))]` stub returning `ERR_UNSUPPORTED`. Previously only Python had OCR (via direct pyo3); now Go, C#, and Node.js can also use OCR when built with `--features ocr`. Go gains `NewOcrEngine()`, `NeedsOcr()`, `ExtractTextWithOcr()`.
- **Node.js binding.cc cleanup** — deleted 12 hallucinated C++ class methods that referenced nonexistent FFI functions (ML/analysis ×7, XFA parse/free ×2, rendering extras ×3). Wired 6 rendering/annotation/PDF-A functions to their real Rust FFI names using Go's working code as the reference. Fixed macOS framework linking (`xcode_settings.OTHER_LDFLAGS`) and MSVC C++20 (`/std:c++20`).

### Bug Fixes

- **Image extraction: `Invalid RGB image dimensions` error on PDFs with Indexed color space images (#311)** — `extract_image_from_xobject` now resolves Indexed palettes via `resolve_indexed_palette` and expands indices through `expand_indexed_to_rgb`, supporting 1/2/4/8 bpc with RGB/Grayscale/CMYK base color spaces. Reported by @Charltsing.
- **Encryption: AES-256 (V=5, R=6) PDFs returned empty or garbled text (#313)** — three independent fixes: uncompressed-object string decryption, push-button widget `/MK /CA` caption extraction, and Algorithm 2.B termination off-by-one correction.
- **Reading order: `ColumnAware` fragmented single-column body text (#314)** — added `is_single_column_region` guard, fixed vertical-split partition inversion. Verified on RFC 2616, Berkeley theses, EU GDPR.
- **Tables: product data sheet label/value rows rendered far from their section (#315)** — replaced with inline-table-insertion scheme that drains tables at their spatial position.
- **Reading order: tabular content interleaved by Y jitter (#316)** — added `row_aware_span_cmp` with 3pt Y-band quantisation. CJK rowspan-label columns preserved through spatial table detector (#329).
- **Text extraction: adjacent Tj/TJ operators concatenated without spaces (#326)** — lowered word-separation threshold to match pdfium's heuristic.
- **Text extraction: fallback-width inflation on fonts with no `/Widths` array (#328)** — added `FontInfo::has_explicit_widths()` and `space_gap` correction for proportional fonts.
- **Text extraction: Arabic content in visual order instead of reading order (#330)** — added Pass 0 pre-shaped Arabic span reversal.
- **Encryption: object cache not invalidated after successful late authenticate() (#323)** — drops `object_cache` on the authenticated transition.
- **Images: Indexed palette expander hardened against DoS and truncation (#324)** — `checked_mul` + 256 MiB guard + truncation rejection.
- **Rendering: slow cold-cache start, dropped ligatures, text missing on subset-CID fonts (#325, #331 R1/R2/R4)** — fixed multi-character cluster width accumulation, Arabic/Latin ligature expansion, and system fontdb caching. Reported by @frederikhors.

### Release Infrastructure

- **Go module tag creation moved to end of pipeline** — `update-go-native-libs` now depends on ALL build + verify jobs. The Go tag is created only after the full build matrix is green, and publishes are gated on it. This prevents sum.golang.org from permanently caching a broken tag hash on failed runs.
- **`verify-go-install` uses local path verification** — uses `go mod edit -replace` against the locally-staged checkout instead of `go get @vX.Y.Z`, eliminating sum.golang.org contact entirely during CI.
- **Go tag creation guarded against re-push** — skips if tag already exists on remote.

### Tooling

- **`scripts/regression_harness.py`** — new self-contained regression harness. Subcommands: `collect` / `run` / `diff` / `groundtruth` / `show`. 60-PDF curated corpus with `text`, `markdown`, and `html` format support.

### Community Contributors

Thank you to everyone who reported issues or filed detailed reproducers for this release!

- **@Charltsing** — Reported the Indexed color space image extraction failure (#311) with a reproduction PDF that exposed a long-standing gap in palette handling, and requested the `DllImport → LibraryImport` migration for NativeAOT-ready C# bindings (#333).
- **@Goldziher** — Reported four extraction issues (#313, #314, #315, #316) with clear repro snippets that let us localise the AES-256 string-decryption gap, the Algorithm 2.B termination off-by-one, the single-column XYCut fragmentation, the inline-rendering gap on product data sheets, and the row-aware sort gap for tabular content. Also raised the pdfium-parity bar (#320) that drove the corpus-wide quality audit and the regression harness.
- **@frederikhors** — Reported the rendering-path bugs on the `rendering` feature (#325): cold-cache slowness, dropped ligatures, missing text on subset-CID fonts, and a font-specific vertical flip. Triage of the report surfaced four distinct signatures (#331 R1-R4); the three that we could reproduce ship in this release.

## [0.3.24] - 2026-04-09
> New Language Bindings: JavaScript / TypeScript, Go, and C#

This release ships official bindings for JavaScript/TypeScript, Go, and C#, built on a shared C FFI layer. 100% Rust FFI parity across all three.

### Features

- **JavaScript / TypeScript bindings** (`pdf-oxide` on npm) — N-API native module with `Buffer`/`Uint8Array` input, `openWithPassword()`, worker thread pool, `Symbol.dispose`, rich error hierarchy, and complete API coverage: document editor, forms, rendering, signatures/TSA, compliance, annotations, extraction with bbox. Full TypeScript type definitions included.
- **Go bindings** (`github.com/yfedoseev/pdf_oxide/go`) — Full API with goroutine-safe `PdfDocument` (`sync.RWMutex`), `io.Reader` support, functional options pattern, `SetLogLevel()`, and ARM64 CGo targets.
- **C# / .NET bindings** (`PdfOxide` on NuGet) — P/Invoke with `NativeHandle` (SafeHandle), `IDisposable`, `ReaderWriterLockSlim` thread safety, `async Task<T>` + `CancellationToken`, fluent builders, LINQ extensions, plugin system. ARM64 NuGet targets (linux-arm64, osx-arm64, win-arm64).
- **C FFI layer (`src/ffi.rs`)** — 270+ `extern "C"` functions covering the full Rust API surface.
- **Shared C header (`include/pdf_oxide_c/pdf_oxide.h`)** — Portable header for all FFI consumers.
- **`pdf_oxide_set_log_level()` / `pdf_oxide_get_log_level()`** — Global log level control exposed to all language bindings.

## [0.3.23] - 2026-04-09

### Bug Fixes

- **Text extraction: SIGABRT on pages with degenerate CTM coordinates (#308)** — extracting text from certain rotated dvips-generated pages (e.g., arXiv papers with `Page rot: 90`) caused a 38 petabyte allocation and SIGABRT. Degenerate CTM transforms produced text spans with bounding boxes ~19 quadrillion points wide, which blew up the column detection histogram in `detect_page_columns()`. Per PDF 32000-1:2008 §8.3.2.3, the visible page region is defined by MediaBox/CropBox, not by raw user-space coordinates. Now `detect_page_columns()` uses median-based outlier rejection to exclude degenerate spans from the histogram, with a 10,000pt hard cap as defense-in-depth. Preserves all 1516 characters on the affected page (matching v0.3.19 output). Reported by @ddxtanx.
- **Editor: images and XObjects stripped on save (#306)** — opening a PDF containing images, making any edit (or none), and saving produced an output with all images removed. The cause was that `write_full_to_writer` only serialized Font resources from the page Resources dictionary, silently dropping XObject (images, form XObjects) and ExtGState entries. Now writes XObject and ExtGState dictionary entries alongside fonts. Also wires up pending image XObject references from `generate_content_stream` into the page Resources dictionary. The `create_pdf_with_images` example was also affected — output contained no images. Reported by @RubberDuckShobe.
- **Rendering: garbled text on systems without common fonts (#307)** — rendering any PDF with text produced random symbols or black rectangles on Linux systems without Arial/Times New Roman installed (e.g., minimal EndeavourOS). The PDF's non-embedded fonts (ArialMT, Arial-BoldMT, TimesNewRomanPSMT) relied on system font availability, but font parsing failures were silent and the fallback font list was too narrow. Now logs a warning with the font name when parsing fails, added DejaVu Sans, Noto Sans, and FreeSans to the system font fallback chain, and logs an actionable message suggesting which font packages to install (`liberation-fonts`, `dejavu-fonts`, or `noto-fonts`). Reported by @FireMasterK.
- **Editor: form field page index always reported as 0** — `get_form_fields()` hardcoded `page_index` to 0 for all fields read from the source document, so fields on page 2+ were incorrectly placed. Now builds a page-ref-to-index map and resolves the actual page from each widget annotation's `/P` entry.
- **Text extraction: fix Tf inside q/Q test** — the `test_extract_save_restore` unit test was ignored due to malformed PDF syntax (`q 14 Tf` missing font name operand). Fixed to valid syntax and unignored. The save/restore mechanism itself was already correct.

### Docs

- **Remove stale CID font widths TODO** — the comment claimed Type0 CID font widths were "not yet implemented", but `parse_cid_widths` and `get_glyph_width` already handled them correctly.

### Community Contributors

Thank you to everyone who reported issues for this release!

- **@ddxtanx** — Reported SIGABRT crash on rotated dvips PDFs (#308) with a clear reproduction case and backtrace. Identified it as a regression from #272.
- **@RubberDuckShobe** — Reported images being stripped on save (#306). Confirmed the issue also affected the `create_pdf_with_images` example.
- **@FireMasterK** — Reported garbled text rendering on EndeavourOS (#307) and provided the test PDF with non-embedded Arial fonts.

## [0.3.22] - 2026-04-08
> Thread-Safe PdfDocument, Async API, Performance, Community Fixes

### Breaking Changes

None. All changes are backward-compatible.

### Features

- **Thread-safe `PdfDocument` — Send + Sync (#302)** — replaced all 16 `RefCell<T>` with `Mutex<T>` and `Cell<usize>` with `AtomicUsize`. `PdfDocument` can now safely cross thread boundaries. Removes `unsendable` from `PdfDocument`, `FormField`, and `PdfPage` Python classes. Enables `asyncio.to_thread()`, free-threaded Python (cp314t), and thread pool usage without `RuntimeError`. Reported by @FireMasterK (#298).
- **`AsyncPdfDocument`, `AsyncPdf`, `AsyncOfficeConverter` (#217)** — complete async API with auto-generated method wrappers. All sync methods are available as async. Requested by @j-mendez.
- **Free-threaded Python support (#296)** — `#[pymodule(gil_used = false)]` declares GIL-free compatibility for cp314t. Requested by @pcen.
- **Word/line segmentation thresholds (#249)** — `extract_words()` and `extract_text_lines()` accept optional `word_gap_threshold`, `line_gap_threshold`, and `profile` kwargs. New `page_layout_params()` method and `ExtractionProfile` class expose adaptive parameters. Contributed by @tboser.

### Bug Fixes

- **CLI split/merge blank pages (#297)** — merge now writes merged page refs; split now filters removed pages from Kids. Reported by @Suleman-Elahi.
- **Rendering: skip malformed images (#299, #300)** — images with missing `/ColorSpace` or invalid dimensions are skipped with a warning instead of crashing the page render. Also handles malformed images inside Form XObjects. Reported by @FireMasterK.
- **Structure tree cycle SIGSEGV (#301)** — cyclic `/K` indirect references in malformed tagged PDFs caused stack overflow. A visited-object set now breaks cycles. Contributed by @hoesler.
- **`horizontal_strategy: 'lines'` text fallback gate (#290)** — setting `horizontal_strategy` to `lines` now correctly suppresses text-based row detection. Each axis is checked independently. Contributed by @hoesler.
- **`vertical_strategy` Python parsing (#290)** — `vertical_strategy` was never read from the Python `table_settings` dict, always defaulting to `Both`. Contributed by @hoesler.

### Performance

- **Cache structure tree** — parsed once and cached; non-tagged PDFs skip parsing via MarkInfo check.
- **Cache decompressed page content stream** — avoids re-decompression when multiple extractors access the same page.
- **Shared XObject stream cache for path extraction** — reuses decompressed Form XObject streams already cached by text extraction.
- **Cached XObject dictionary for path extraction** — avoids re-resolving Resources -> XObject dict chain on every Do operator.
- **Byte-level path extraction parser** — skips BT/ET text blocks and parses path/state/color operators without Object allocation.
- **Allocation-free graphics state for paths** — Copy-only state struct eliminates heap allocations on q/Q save/restore.
- **Index-based font tracking in prescan** — replaces String cloning on every q operator with index into font table.
- **Prescan: drop Do positions when Do-dominated** — prevents region merging that defeats the prescan optimization.
- **Reuse spans in table detection** — reuses pre-extracted spans instead of re-parsing the content stream.
- **Pre-filter non-table paths** — filters to lines/rectangles before the detection pipeline.
- **O(1) MCID lookup** — HashSet instead of linear search for marked-content identifier matching.
- **O(log n) page tree traversal** — uses /Count to skip subtrees instead of linear counting.
- **Lazy page tree population** — defers bulk page tree walk until needed.

### Dependencies

- Bump `zip` 8.5.0 -> 8.5.1
- Bump `pdfium-render` 0.8.37 -> 0.9.0
- Bump `tokenizers` 0.15.2 -> 0.22.2

### Community Contributors

Thank you to everyone who reported issues and contributed PRs for this release!

- **@hoesler** — Structure tree cycle SIGSEGV fix (#301) and table strategy gating fix (#290). Two high-quality PRs with tests and clean code.
- **@tboser** — Word/line segmentation thresholds feature (#249). Well-designed API with 14 tests and responsive to review feedback.
- **@FireMasterK** — Reported thread-safety crash (#298), rendering crashes with missing ColorSpace (#299) and invalid image dimensions (#300). Three critical bug reports that drove the Send+Sync refactor.
- **@Suleman-Elahi** — Reported CLI split/merge blank pages bug (#297) with clear reproduction steps.
- **@pcen** — Requested free-threaded Python compatibility (#296).
- **@j-mendez** — Requested async Python API (#217).

## [0.3.21] - 2026-04-04
> Log Level Honored in Python, Multi-Arch Wheels

### Bug Fixes

- **Log level now fully respected in Python (#283)** — `extract_log_debug!` / `extract_log_trace!` / etc. were printing to stderr directly via `eprintln!`, bypassing the `log` crate and therefore ignoring `pdf_oxide.set_log_level(...)` and Python's `logging.basicConfig(level=...)`. Messages like `[DEBUG] Parsing content stream for text extraction` and `[TRACE] Detected document script: Latin` leaked through at ERROR level. The macros now forward to `log::debug!` / `log::trace!` / etc. and are properly gated by the `log` crate's max level filter. Reported by @marph91 as a follow-up to #280.

### Packaging

- **Multi-arch Python wheels (#284)** — Added wheels for Linux aarch64 (`manylinux_2_28_aarch64`), Linux musl x86_64 and aarch64 (`musllinux_1_2_*`), and Windows ARM64 (`win_arm64`). Lowered the manylinux glibc floor from `2_34` to `2_28` to cover RHEL 8, Debian 11, Ubuntu 20.04, and Amazon Linux 2023. A source distribution (sdist) is now published for any platform with a Rust toolchain. Reported by @jhhayashi.

## [0.3.20] - 2026-04-04
> Table Extraction Engine — Intersection Pipeline, Text-Edge Detection, Converter Improvements

### Table Extraction Engine

Major rewrite of the table detection system, implementing the universal `Edges → Snap/Merge → Intersections → Cells → Groups` pipeline — the gold-standard approach used by Tabula, pdfplumber, and PyMuPDF, now in pure Rust.

#### New Detection Capabilities
- **Intersection-based table detection** — Finds H×V line crossings, builds cells from 4-corner rectangles, groups into tables via union-find. The gold-standard approach used by Tabula/pdfplumber/PyMuPDF, now in pure Rust.
- **Extended grid for non-crossing lines** — When H and V lines are in different page regions, creates virtual grid from Cartesian product of all coordinates.
- **Column-aware text detection** — Segments 2-column layouts via X-projection histogram, runs text-only table detection per column.
- **H-rule-bounded text tables** — Detects tables bounded by horizontal rules but no vertical lines (common in academic papers).
- **Hybrid row detection** — Infers row boundaries from text Y-positions when only vertical borders exist (e.g. invoice line items).
- **Dotted/dashed line reconstitution** — Merges short line segments into continuous edges for row separator detection.
- **Section divider splitting** — Splits multi-section forms at full-width horizontal dividers.
- **Edge coverage filtering** — Removes orphan edges that don't participate in any potential grid.
- **Configurable V-line split gap** — `v_split_gap` field in `TableDetectionConfig` (default 20pt, was hardcoded 4pt).

#### Table Rendering
- **Space-padded column alignment** — Clean, readable output replacing ASCII box drawing (`+--+|`). Right-aligns currency/number columns.
- **Form numbering artifact stripping** — Removes single-digit prefixes from PDF form templates ("1 Apr 11" → "Apr 11").
- **Dash/underscore cell stripping** — Removes decorative `------` separators from table cells.

### Text Extraction Quality

- **Adjacent value spacing** — Inserts space between consecutive currency values in table cells.
- **Split decimal merging** — Rejoins integer and decimal parts rendered in separate fixed-width boxes.
- **Bold span consolidation** — Merges adjacent single-character bold spans into a single `**WORD**` in markdown.
- **HTML heading hierarchy** — Content-aware detection; addresses and box numbers no longer tagged as `<h1>`/`<h2>`.
- **Image bloat fix** — `include_images` defaults to `false`, dramatically reducing output size.
- **Label-value pairing** — Same-Y spans from different reading-order groups rendered on the same output line.
- **Content ordering** — XYCut group_id propagation keeps spatial regions as contiguous blocks.
- **Columnar group merging** — Detects column-by-column layouts and re-interleaves into rows.
- **Orphaned span recovery** — Text spans inside rejected table regions are preserved at correct Y-position.
- **Key-value pair merging** — `Label\n$Value` patterns merged to `Label $Value` in post-processing.

### Bug Fixes

- **Encrypted PDF clear error** — Returns `Error::EncryptedPdf` with helpful message instead of silent zero output.
- **ObjStm/XRef stream decryption** — Object streams are no longer incorrectly decrypted per ISO 32000-2 Section 7.6.3.
- **Stream parser trailing newline** — Strips CR/LF before `endstream` keyword, fixing AES block-size errors on encrypted PDFs.
- **Table detection enabled by default** — `extract_text()` now uses `extract_tables: true`.
- **`to_plain_text()` includes tables** — Was silently dropping all detected tables.
- **Python `extract_tables()` config** — Now uses `default()` (Both strategy) instead of `relaxed()` (Text-only).
- **MD table cell dropping** — Row padding and centroid drift fix in spatial detector.
- **Box label spacing** — Inserts space between box number and adjacent currency value.
- **Dash cell artifact** — `------` cells cleared from table output.
- **Orphaned dollar values** — Dollar values no longer silently dropped when table detector misses them.
- **Digit→currency spacing** — Any positive gap between digit/text and `$`/`€`/`£` inserts a space.

### Refactoring (SOLID/DRY/KISS)

- **UnionFind struct** — Extracted from two duplicated inline implementations (DRY).
- **`snap_and_merge()` decomposed** — Split into `snap_edges()`, `join_collinear_edges()`, `reconstitute_dotted_lines()` (SRP).
- **Shared converter helpers** — `span_in_table()` and `has_horizontal_gap()` extracted from 3 duplicated copies to `converters/mod.rs` (DRY).
- **`detect_tables_from_intersections()` decomposed** — 229-line 6-responsibility function split into `build_grid_from_lines()`, `assign_spans_to_intersection_grid()`, `finalize_intersection_tables()` + 20-line orchestrator (SRP).
- **Collinear segment joining** — Relaxed coord tolerance from `f32::EPSILON` to `SNAP_TOL` for proper chain joining.

### API Consistency

- Python, Rust, and WASM `extract_tables()` all use the same `TableDetectionConfig::default()` (Both strategy) for consistent results across languages.

### Logging (#280)

Library logging now follows standard best practices — **silent by default** across all bindings.

- **Python** — Rust `log` macros now flow through Python's `logging` module via `pyo3-log`. Configure with the normal API:
  ```python
  import logging
  logging.basicConfig(level=logging.WARNING)
  ```
  New helpers: `pdf_oxide.set_log_level("warn")` and `pdf_oxide.disable_logging()`. The `setup_logging()` function is kept for backward compatibility (the bridge is initialized automatically on module import).
- **WASM** — New `setLogLevel(level)` / `disableLogging()` functions. Logs are forwarded to the browser console via `console_log`. Accepts `"off"`, `"error"`, `"warn"`, `"info"`, `"debug"`, `"trace"`.
- **Rust** — No change; the library continues to use the `log` crate facade without initializing a backend (standard Rust library practice). Applications choose their own logger (`env_logger`, `tracing`, etc.).

### 🏆 Community Contributors

🥇 **@marph91** — Thank you for reporting the logging flood issue (#280) and the thoughtful proposal. This pushed us to audit the bindings against the logging best practices used by `pyo3-log`-based projects (cryptography, polars) and ship a clean fix across Python, WASM, and Rust! 🚀

## [0.3.19] - 2026-04-02
> Text Extraction Accuracy, Column-Aware Reading Order, and Community Contributions

### Features

- **`extract_page_text()` Single-Call DTO** (#268) — New `PageText` struct returns spans, characters, and page dimensions from a single extraction pass, eliminating redundant content stream parsing. Available across Rust, Python, and WASM.
- **Column-Aware Reading Order** (#270) — New `extract_spans_with_reading_order()` method accepts a `ReadingOrder` parameter. `ReadingOrder::ColumnAware` uses XY-Cut spatial partitioning to detect columns and read each column top-to-bottom, fixing garbled text for multi-column PDFs.
- **Per-Character Bounding Boxes from Font Metrics** (#269) — `TextSpan` now carries per-glyph advance widths captured during extraction. `to_chars()` produces accurate per-character bounding boxes using font metrics instead of uniform width division. Available as `span.char_widths` in Python and `span.charWidths` in WASM (omitted when empty).
- **`is_monospace` Flag on TextSpan/TextChar** (#271) — Exposes the PDF font descriptor FixedPitch bit, with fallback name heuristic (Courier, Consolas, Mono, Fixed). Eliminates the need for fragile font-name string matching.
- **`Pdf::from_bytes()` Constructor** (#252) — Opens existing PDFs from in-memory bytes without requiring a file path. Available across Rust, Python (`Pdf.from_bytes(data)`), and WASM (`WasmPdf.fromBytes(data)`).
- **Path Operations in Python** (#261) — `extract_paths()` now includes an `operations` list with individual path commands (move_to, line_to, curve_to, rectangle, close_path) and their coordinates. WASM `extractPaths()` also aligned.

### Bug Fixes

- **Fixed panic on multi-byte UTF-8 in debug log slicing** (#251) — Replaced raw byte-offset string slices with char-boundary-safe helpers, preventing panics when extracting text from CJK/emoji PDFs with debug logging enabled.
- **Fixed markdown spacing around styled text** (#273) — Markdown output no longer merges words across annotation/style span boundaries (e.g., "visitwww.example.comto" → "visit www.example.com to").
- **Fixed Form XObject /Matrix application** (#266) — Text extraction now correctly applies Form XObject transformation matrices and wraps in implicit q/Q save/restore per PDF spec Section 8.10.1.
- **Fixed text matrix advance for rotated text** (#266) — Replaced incorrect `total_width / text_matrix.d.abs()` division (divide-by-zero for 90° rotation) with correct `Tm_new = T(tx, 0) × Tm` per ISO 32000-1 Section 9.4.4.
- **Fixed prescan CTM loss for deeply nested text** (#267) — Replaced backward 4KB scan with forward CTM tracking across the full content stream, capturing outer scaling transforms for text in streams >256KB (e.g., chart axis labels).
- **Fixed prescan dropping marked content (BDC/BMC) for tagged PDFs** — The forward CTM scan now includes preceding BDC/BMC operators and following EMC operators in region boundaries, preserving MCID, ActualText, and artifact tagging for tagged PDFs in large content streams.
- **Fixed deduplication dropping distinct characters** (#253) — `deduplicate_overlapping_chars` now checks character identity, not just position. Distinct characters close together (e.g., space followed by 'r' at 1.5pt) are no longer incorrectly removed.
- **Fixed text dropped with font-size-as-Tm-scale pattern** (#254) — Corrected TD/T* matrix multiplication order per ISO 32000-1 Section 9.4.2. PDFs using `/F1 1 Tf` + scaled `Tm` (common in InDesign, LaTeX) no longer silently lose lines. Also tightened containment filter to require text identity match.
- **Fixed markdown merging words in single-word BT/ET blocks** (#260) — `to_markdown()` now detects horizontal gaps between consecutive same-line spans and inserts spaces, matching `extract_text()` behavior. Fixes PDFs generated by PDFKit.NET/DocuSign.
- **Fixed CLI merge creating blank documents** (#262) — `merge_from`/`merge_from_bytes` now properly imports page objects with deep recursive copy of all dependent objects (content streams, fonts, images), remapping indirect references.

### Dependencies

- **pyo3** 0.27.2 → 0.28.2 — Added `skip_from_py_object` / `from_py_object` annotations per new `FromPyObject` opt-in requirement.
- **clap** 4.5.60 → 4.6.0
- **codecov/codecov-action** 5 → 6

### Breaking Changes (WASM only)

- **WASM JSON field names now use camelCase** — `TextSpan`, `TextChar`, `PageText`, `TextBlock`, and `TextLine` serialized fields changed from snake_case to camelCase (e.g., `font_name` → `fontName`, `font_size` → `fontSize`, `is_italic` → `isItalic`, `page_width` → `pageWidth`) when the `wasm` feature is enabled. This aligns with JavaScript naming conventions. **Rust JSON serialization via serde is only affected when the `wasm` feature is enabled. Python uses PyO3 getters and is unaffected.**

### 🏆 Community Contributors

🥇 **@Goldziher** — Thank you for the comprehensive feature requests (#252, #268, #269, #270, #271) that shaped the text extraction improvements in this release. Your detailed issue reports with code examples and spec references made implementation straightforward! 🚀

🥈 **@bsickler** — Thank you for the Form XObject matrix fix (#266) and prescan CTM rewrite (#267). These are critical correctness fixes for text extraction in rotated documents and large content streams! 🚀

🥉 **@hansmrtn** — Thank you for the UTF-8 panic fix (#251). This prevents crashes for any user processing non-ASCII PDFs with debug logging! 🚀

🏅 **@jorlow** — Thank you for the markdown spacing fix (#273). Clean, well-tested fix for a common user-facing issue! 🚀

🏅 **@willywg** — Thank you for exposing path operations in Python (#261), giving downstream tools access to individual vector path commands! 🚀

🏅 **@titusz** — Thank you for reporting the character deduplication (#253) and Tm-scale text dropping (#254) bugs with clear root cause analysis! 🚀

🏅 **@oscmejia** — Thank you for reporting the markdown word merging issue (#260) with a clear reproduction case! 🚀

🏅 **@Inklikdevteam** — Thank you for reporting the CLI merge blank pages bug (#262)! 🚀

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
