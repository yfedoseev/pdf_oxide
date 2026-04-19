# Changelog

All notable changes to PDFOxide are documented here.

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
