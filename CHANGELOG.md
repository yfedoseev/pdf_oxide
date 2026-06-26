# Changelog

All notable changes to PDFOxide are documented here.

## [Unreleased]

### Fixed

- **Non-Identity-ordered Type0 fonts no longer emit a wrong character for CIDs missing from `/ToUnicode` (#773)** — for an embedded Type0 font whose `/ToUnicode` CMap omits some drawn CIDs (e.g. a ligature glyph with no single Unicode codepoint), the decode path fell back to a numeric guess — the GID via the standard glyph-name table → AGL, or the CID itself as a code point (`char::from_u32`) — emitting a plausible-but-wrong, content-like character that varied per subset (e.g. a `ti` ligature → `:` / `D`, so `notificacao` → `no:ficacao`). The glyph has no Unicode anywhere in the file (no `/ToUnicode` entry, no `post` name, no GSUB), so the letters are unrecoverable, but substituting a wrong character is silent corruption. When a usable `/ToUnicode` is present, the GID→AGL guess is now suppressed for all Type0 fonts, and the CID-as-Unicode guess is suppressed for fonts whose `CIDSystemInfo` ordering is **not** `Identity`, so an uncovered CID there decodes to `U+FFFD` instead. Identity-ordered (Adobe-Identity-0) fonts retain the CID-as-Unicode mapping that Identity ordering conventionally implies; the residual wrong-character case for ligature/private-use slots in Identity-ordered fonts remains a known limitation. Subtractive for the no-`/ToUnicode` path — a font with no `/ToUnicode` still uses the CID-as-Unicode heuristic exactly as before, and the authoritative embedded-`cmap`/`post` lookups are unchanged.

## [0.3.68] - 2026-06-24

> Extraction fidelity release — symbolic TrueType character mis-decoding corrected via the `(3,0)`/`(1,0)` cmap, same-row span ordering preserved in plain-text output, JPEG 2000 (`JPXDecode`) image XObjects decoded via OpenJPEG, and RTL Farsi body text recovered from tagged Type0/CID PDFs.

### Added

- **JPEG 2000 (`JPXDecode`) image XObjects decoded via OpenJPEG** — `render_page` previously skipped image XObjects whose stream was compressed with `/JPXDecode`, silently dropping page content. The OpenJPEG library (via `jpeg2k`) now decodes them at render time; multi-component images are colour-managed and alpha-composited exactly as other image types. Thanks @potatochipcoconut for the report.

### Fixed

- **RTL Farsi body text recovered from tagged Type0/CID PDFs (#758)** — Type0/CID composite fonts with a valid `/ToUnicode` CMap had ~92% of their body text silently dropped in v0.3.66 on RTL (Farsi) documents. The tagged-structure traversal now correctly assembles CID-encoded spans before the RTL reconstruction pass, recovering the full body. Thanks @Goldziher for the report.
- **Symbolic TrueType fonts no longer mis-decode characters (#760)** — a simple symbolic TrueType font (FontDescriptor `Flags` bit 3, no `/Encoding`, no `/ToUnicode`) decoded its content bytes by treating each byte directly as a glyph ID, producing wrong-but-plausible characters (e.g. `Ç` → `Ê`, `SOLUÇÃO` → `SOLUÊÃO`). The fix parses the embedded font's `(3,0)` symbol (or `(1,0)` Macintosh) `cmap` subtable into a byte→GID map so the correct byte→GID→Unicode hop is applied; fonts without such a subtable still use the byte as the GID. Thanks @schelip for the report and fix.
- **Same-row spans no longer reordered or split in plain-text output (#752)** — when one logical line was emitted as spans at the same Y in different reading-order groups whose boxes overlapped by a fraction of a point, `to_plain_text` interleaved the overlapping group as a vertical column (hoisting a fragment to the front) and forced a space between the overlapping fragments (splitting a word). A group whose spans share a Y row is now excluded from columnar detection, and the cross-group same-Y space rule is replaced by the standard `has_horizontal_gap` threshold used by the other converters. Thanks @schelip for the report and fix.

### Documentation

- **macOS/Rust OCR setup guide corrected** — `ORT_LIB_LOCATION` is inert with the `load-dynamic` ONNX Runtime feature; the guide now documents `ORT_DYLIB_PATH`, the variable actually read at runtime.

### Dependencies

- **pyo3 0.28 → 0.29** — fixes two security vulnerabilities: a missing `Sync` bound on `PyCFunction::new_closure` closures, and a possible out-of-bounds read in `BoundTupleIterator::nth_back` / `BoundListIterator::nth_back`.
- **phf 0.13 → 0.14**, **bytes 1.11 → 1.12**, **log 0.4.32 → 0.4.33**, **p12-keystore 0.3.0 → 0.3.1**.
- GitHub Actions: `actions/checkout` v7.0.0, `actions/setup-java` v5.3.0, `softprops/action-gh-release` v3.0.1, `ruby/setup-ruby` v1.314.0, `taiki-e/install-action` v2.82.2.
- Patch/minor updates for `rustls`, `time`, `zerocopy`, `zeroize`, `wasm-bindgen`, `wide`, and ~35 other transitive crates.

## [0.3.67] - 2026-06-20

> Reading-order quality release for untagged scientific papers — manuscript-line-number rails lifted out of the body, dense two-column bodies kept apart at a measured gutter, comma-bearing statistic subscripts rejoined, repeated journal pagination footers suppressed, whole-page placed-PDF article bodies recovered, and single-column pages protected from false column detection.

### Fixed

- **Whole-page placed-PDF article bodies are no longer dropped** — when a publisher places an entire typeset article body inside one Adobe InDesign `/PlacedPDF` marked-content region (e.g. MATEC Web of Conferences), its text is now extracted instead of suppressed. The placed-figure suppression — which removes a decorative draft-galley overlay whose glyphs duplicate logical text living *outside* the region — is now coverage-aware: a cheap content-stream pre-scan keeps the placed text when it dominates the page and the non-placed text is negligible (the placed region *is* the content), and suppresses it only when it is a minority overlay. This matches pymupdf/pdftotext, recovers the full body on whole-body-placed pages, and preserves the de-duplication win on decorative-overlay pages.
- **Single-column pages are no longer scrambled by false column detection** — the topological block-reordering pass now requires each side-by-side column to span at least four text rows. A single-column page whose few body lines happen to have a wide mid-line word gap (a sentence space after a period) was previously split into two phantom columns and read out of order; it now stays on the row-aware path and reads correctly.
- **Marginalia line-number rails no longer interleave into the body** — a narrow, sparse, body-aligned left-margin rail of numerals (manuscript line numbers `118 119 120 …`, a folio rail) is now lifted out before geometric column ordering and re-appended at the end of the reading order, instead of being woven into the prose by a flat row-aware sort. Tightly gated (narrow outer band, sparse, ≥3 lines, predominantly numeric, a clear detached gutter, running alongside the body) so genuine narrow first columns, table label columns, and ordinary pages are unaffected.
- **Dense two-column bodies are kept apart at a measured gutter** — the multi-region block builder now refuses to join two spans that straddle the page's measured central gutter, even when a tight-leading over-wide advance makes the cross-gutter gap look like a normal word space. This stops the two columns from fusing into one block (which previously left the page on a row-major interleave) on dense, tightly-led academic two-column pages. Purely subtractive — single-column, grid, and off-centre pages are unchanged.
- **Comma-bearing statistic subscripts rejoin to their base** — an F-statistic's degrees of freedom (`4,176` in `F4,176`) or a multi-affiliation marker (`1,2`), which the producer set as a single comma-joined subscript, is now merged back into its base instead of being stranded as separate `F` / `4` / `176` tokens.
- **Repeated journal pagination footers are suppressed** — a constant-text running citation/pagination line (a DOI, a journal `Volume … | Article …` reference, or a journal URL host with a digit) that recurs in the top/bottom margin on a strong majority of pages is now treated as furniture and excluded from the body, matching how reference extractors handle running heads and folios (ISO 32000-1 §14.8.2.2). The shape gate is deliberately narrow — government-form control numbers such as `OMB No. 1545-0115` are kept.

## [0.3.66] - 2026-06-18

> Extraction-quality release — multi-region and two-column reading order, right-to-left Arabic/Hebrew reconstruction in Markdown and HTML, soft-wrap de-gluing and de-hyphenation, and a subset-font cipher-encoding fix — plus destructive redaction of Identity-H (composite) text, CCITT Group 3 fax decoding, and a 16-bit-per-component image fix.

### Added

- **Destructive redaction of Identity-H composite (Type 0) text (#748)** — destructive redaction previously refused all Type 0 fonts. It now supports the common Identity-H case (`Type0` + `Encoding /Identity-H`, horizontal writing mode): 2-byte big-endian CIDs are decoded and widths read from `/W` (ISO 32000-1 §9.7.4.3 / §9.7.5.2), target glyphs are removed, and the engine stays fail-closed — refusing Identity-V, legacy embedded CMaps, and odd-length strings it cannot safely rewrite. Thanks @shota-sh.
- **CCITT Group 3 (T.4) fax decoding (#738)** — extends the in-house CCITT decoder (Group 4 landed in v0.3.65) to Group 3: pure 1-D Modified Huffman (`K = 0`) and mixed 1-D/2-D (`K > 0`), with end-of-line handling and the same partial-row recovery contract as the Group 4 path. Group-3-encoded fax images that previously rendered blank now decode to their real content. Thanks @potatochipcoconut.

### Fixed

- **16-bit-per-component images no longer panic on extraction (#750)** — extracting an image with `/BitsPerComponent 16` hit an internal assertion (an uncatchable panic across the FFI boundary) because 16-bit big-endian samples were stored verbatim into a double-sized buffer. 16-bit samples are now down-converted to 8-bit at parse time, and the PNG encoder rejects a mismatched buffer with a recoverable error instead of asserting. Thanks @shtyker.
- **Multi-region and two-column reading order** — pages with genuine side-by-side regions (two-column bodies, a publisher sidebar beside the body, multi-column newsletters and forms) are now linearised with a topological block order — a precede relation over text blocks — instead of a flat row-aware sort that interleaved columns line by line. Two stacked text lines merged by the same-line Y tolerance are de-interleaved by baseline rather than X-sorted into each other. Self-gating: pages without horizontally-disjoint, vertically-overlapping regions are unchanged.
- **Right-to-left Markdown and HTML reconstruction** — Arabic and Hebrew now read correctly in Markdown and HTML, not only in plain text. RTL lines are reconstructed at glyph fidelity from the raw spans before the converter orders them; an inter-word punctuation span between two RTL words is kept (instead of being dropped as decoration and gluing the words); a line-final word that sits just below its baseline band is rejoined to its line; and the producer's authoritative inter-word space after a dual-joining letter is preserved.
- **Soft-wrap de-gluing and de-hyphenation** — generator PDFs that emit one show-string per visual line with no trailing separator no longer glue the last word of a wrapped line to the first word of the next line; a line-break separator is synthesised, since text positioning is purely geometric and a line break encodes no space character (ISO 32000-1 §9.4). The spurious space left after a line-final hyphen on a wrapped line is suppressed so hyphenated words rejoin cleanly.
- **Markdown/HTML gutter-crossing word rejoin** — a word the producer split into two show-strings across the column gutter (a lone name initial, the two digits of a split number) is rejoined for Markdown and HTML, while genuinely separate adjacent column lines are kept apart and an interior emphasis (bold/italic) span inside a rejoined run is no longer dropped.
- **Subset-font cipher encoding no longer overrides the named encoding** — a Type1/CFF subset font whose re-indexed built-in encoding disagrees with the producer-declared named `/Encoding` (e.g. WinAnsi / MacRoman) on most overlapping codes is detected as a subset cipher and the named encoding is kept, fixing whole-page mojibake on affected documents.

### Dependencies

- Bumped `taffy` 0.10.1 → 0.11.0 (#741), `pdfium-render` 0.9.1 → 0.9.2 (#743), `regex` 1.12.3 → 1.12.4 (#746), `brotli` 8.0.3 → 8.0.4 (#744), and `pyo3-log` 0.13.3 → 0.13.4 (#745); CI bumps for `ruby/setup-ruby` (#742) and `taiki-e/install-action` (#740).

## [0.3.65] - 2026-06-16

> Multilingual and layout extraction quality — right-to-left bidi reconstruction for Arabic and Hebrew, multi-region reading order for publisher sidebars and two-column academic pages, and CJK/Indic word segmentation — plus an in-house CCITT Group 4 fax decoder that honours `EncodedByteAlign`, structured two-column surfacing, and a batch of O(n²) hot-path removals.

### Added

- **Two-column structured extraction and tagged-structure surfacing (#734)** — `extract_structured` now reports a per-line `column_index` for multi-column pages and, on tagged PDFs, surfaces marginal labels (`Lbl` → marginal label) and the nearest enclosing section (`Sect`/`Art`/`Part` → a document-stable `section_id` with cross-page continuity), per ISO 32000-1 §14.8.4. Additive and zero-risk for untagged input. Thanks @lggcs.
- **Reading-order threads for linked content (#458)** — article-thread (`/Threads` → `/B` bead) ordering is surfaced so content that flows across columns and pages can be read in author-intended order.
- **In-house CCITT Group 4 (T.6) fax decoder (#738)** — a from-scratch decoder for `CCITTFaxDecode` images that correctly honours `EncodedByteAlign` (ITU-T T.6 2D mode codes, Modified-Huffman run tables, reference-line changing-element walk), with partial-row recovery on truncated streams. Replaces a path that could silently fall back to an all-white image; bilevel fax images now decode to their real content. Thanks @potatochipcoconut.

### Fixed

- **Right-to-left Arabic/Hebrew text reconstructed in logical order** — several classes of RTL extraction defect are corrected so Arabic and Hebrew read correctly instead of scrambled:
  - **Cross-span cluster reversal (SEG-AR)** — producers that draw an Arabic word as interleaved base-glyph and zero-width mark spans (the mark's x falling *inside* a neighbouring word) had their letters atom-sorted to word edges, scrambling e.g. `الثدييات` → `ثالدييات`. Pure-RTL lines with such zero-width-inside-a-span runs are now collapsed into a single visual-order span — glyphs ordered by x, combining marks bound to their base, word boundaries taken from the producer's own standalone space spans — then reversed to logical order (UAX #9 L2). A representative Arabic page improved from a heavily garbled paragraph to fully correct text.
  - **RTL number preservation (SEG-AR / SEG-HE)** — Arabic-Indic and Latin digit runs embedded in RTL text are no longer reversed: `٤٣٤١` now reads `١٤٣٤` (1434) and a Hebrew `ל ,2009-` now reads `ל-2009,`, matching a conformant bidi reorder.
  - **Glyph-advance preservation when merging scrambled-RTL spans** — merging adjacent RTL spans no longer corrupts true glyph positions, and a real word break bordering non-cursive punctuation is kept (rather than suppressed as a cursive-shatter space) on `/ReversedChars` producers.
- **Multi-region reading order for publisher sidebars and two-column pages** — narrow publisher-metadata sidebars are now segregated from the body and emitted after it (title and body merged top-to-bottom, sidebar last) instead of being interleaved, across text, Markdown, and HTML. Bottom-spanning blocks that follow a multi-column region are peeled correctly, numbered-list markers are skipped in rowspan-label reordering, and two-column prose is linearised column-major. A figure Form XObject's `/BBox` clip (ISO 32000-1 §8.10.1) now drops a draft-galley underlay a conformant renderer would clip — gated to figure-sized forms so a full-page content-frame wrapper keeps its body.
- **CJK and Indic word segmentation** — Korean number/counter spacing and line-break rejoining (`1 만년` → `1만년`), and stray spaces before Bengali/Devanagari/Latin punctuation (`प्राणी ।` → `प्राणी।`), are corrected. Adobe predefined CIDFont collections decode through the documented CID → Unicode path (ISO 32000-1 §9.3.3).

### Changed

- **Performance — O(n²) and O(n·m) hot-path removals** — drop-cap initial pairing uses a windowed binary search; the rotated-character filter is skipped entirely on unrotated pages; and table filtering, XY-cut, hyphen merging, and word extraction lose their quadratic hot paths. Text/Markdown/HTML output is unchanged by these changes.
- **Redundant clip-mask clone dropped in `apply_pending_clip` (#654)** — the render path no longer clones the clip mask when it is about to be replaced, trimming an allocation per clipped paint. Pixel output is sub-perceptually unchanged. Thanks @RayVR.

## [0.3.64] - 2026-06-12

> Composite-CJK page rendering — a bundled Droid Sans fallback now paints embed-less Type 0 fonts and the Adobe predefined CIDFont collections — plus a §11 transparency compositing surface with optional lcms2 colour management, cross-document font-cache correctness, valid annotation appearance streams, and math/CJK text-extraction polish (prime-notation spacing, signed unit exponents, CJK bracket spacing, table-header Markdown).

### Added

- **Substitute Adobe predefined CIDFonts with a bundled CJK fallback when rendering (#730)** — a composite (Type 0) font that references an Adobe predefined character collection (Adobe-Japan1, Adobe-GB1, Adobe-CNS1, Adobe-KR — e.g. `Ryumin-Light`, `GothicBBB-Medium`, `STSong-Light`, `HYSMyeongJo-Medium`) without embedding glyph outlines previously painted nothing. ISO 32000-2 §9.7.5.2 requires a conforming reader to ship those collections; the page renderer now maps CID → Unicode (via the bundled `cid_mappings` tables) → a Droid Sans Fallback glyph, producing a real sans-serif rendering even when the document asked for Mincho / Gothic. Gated behind the opt-in `cjk-render-fallback` feature (~3.4 MB), independent of `cjk-form-fonts`.
- **Bundled CJK fallback for embed-less composite fonts on CJK-fontless hosts** — the same `cjk-render-fallback` feature loads Droid Sans Fallback into the page renderer's font database so composite fonts that reference a glyph collection but embed no outlines still paint real glyphs instead of blank on machines with no system CJK font installed.
- **§11 transparency compositing surface and a pluggable `IccBackend` trait (#674)** — adds an ISO 32000-1 §11 transparency surface to the rendering pipeline and an `IccBackend` colour-management abstraction with an optional `lcms2`-backed implementation, alongside the default `qcms` path.

### Fixed

- **Cross-document font cache now keys Type0/Identity-H fonts by `/ToUnicode` content (#733, extends #595, #597, #598)** — the cross-document cache hardening in #595/#597/#598 folds the `/ToUnicode` *reference* (object id/gen) into the font identity hash and keeps *canonical* subset fonts (`AAAAAA+`) out of the shared cache. This extends that coverage to two cases the reference-based key doesn't reach: a non-canonical subset tag such as `/CIDFont+F1` (emitted by some generators) stays eligible for cross-document sharing, and PDFs produced from a common template reuse the same `/ToUnicode` object number — so two genuinely different fonts that merely share a `/BaseFont` name produced an identical key. Processed in one long-lived process, a later document was then served an earlier font's parsed `FontInfo` and its glyphs decoded through the wrong `/ToUnicode` — a constant-offset cipher (`SUMMARY` → `6800$5<`) or control/PUA characters — though each document extracted correctly in isolation. The identity hash now folds the `/ToUnicode` stream *bytes*, the embedded `/FontFile{,2,3}` bytes, the descendant `/Subtype`, and a stream-form descendant `/CIDToGIDMap`, so same-named-but-different fonts get distinct keys regardless of subset-tag form or object reuse, while genuinely identical fonts still deduplicate across documents (the cache's purpose is preserved). Same bug class as the `/Widths` poisoning fixed in #598.
- **Watermark annotations rendered as nothing in compliant viewers (#713)** — a watermark's `/AP` appearance was serialized as a stream nested *directly* inside the annotation dictionary (`/AP <</N <<…>> stream … endstream>>`). A PDF stream must be an indirect object (ISO 32000-1:2008 §7.3.8); the inline form is invalid, so spec-compliant readers (e.g. MuPDF/PyMuPDF) rejected the annotation with "invalid key in dict" and the watermark never appeared — even though the bytes were present in the file. A shared `hoist_appearance_streams` helper now lifts nested `/N`, `/D`, and `/R` appearance streams (including named-state sub-dictionaries) into freshly allocated indirect objects and replaces the slot with a reference, applied on both the `DocumentBuilder` writer and the existing-page `DocumentEditor::save_page` paths. Verified end-to-end with MuPDF: the watermark now parses and renders on both paths.
- **Fixed Python type stubs leaking the pyo3 `Py<Self>` receiver as a positional parameter (#728)** — methods implemented in Rust with a by-value receiver (`fn page(slf_handle: Py<Self>, …)` — the idiom pyo3 uses to hand a method an owned handle to its own instance) were emitted by the rylai stub generator with that receiver re-exposed *alongside* the injected `self`.
- **Bookmark titles encoded in UTF-16BE/LE decoded incorrectly (#729)** — PDF outline `/Title` strings may be UTF-16BE/LE with a leading BOM or PDFDocEncoding; decoding now routes through `optional_content::decode_pdf_text_string` so non-Latin bookmark labels read correctly.
- **A damaged ICC profile no longer aborts the OutputIntents scan (#712, #731)** — a single broken entry in `/OutputIntents` previously stopped the whole colour-profile scan; the scan now skips the damaged profile, logs it, and continues so the remaining intents are still read.
- **Spurious space inside prime-notation numbers (math/astronomy)** — a prime's narrow metric advance (`w₀`, ISO 32000-1 §9.4.4) is small relative to its inked form, so the geometric word-break heuristic injected a space before a following decimal or period: `0′′.28` → `0′′. 28`, `Y′.` → `Y ′ .`. Arc-second values and prime-suffixed tokens (`0′′.28`, `1′′.47`, `δnn′.`) now stay intact, while genuine feet-and-inches like `5′ 6″` are preserved. Surfaced by the py-pdf/benchmarks text-extraction corpus.
- **Signed unit exponents wrongly rewritten to Unicode subscripts** — the super/subscript pass synthesized Unicode sub/superscript glyphs from glyph geometry, overriding the authoritative ToUnicode decode (§9.10) and firing inconsistently on identical occurrences; a scientific unit exponent such as `s−1` could become `s₋₁`. A run that opens with a minus/hyphen sign and contains a digit is now left as ASCII, matching the plaintext convention every reference extractor follows.
- **Markdown table header cells no longer force-bolded** — emitted table headers (`| **CRN** | **Type** |`) dropped the `**` wrappers so header text matches the source weight (`| CRN | Type |`).
- **Spurious spaces between CJK/Hangul text and brackets** — a stray space a producer inserted between an ideograph/Hangul syllable and an adjacent bracket (`(Xa, Za) 的` → `(Xa, Za)的`, `消灯 (0%)` → `消灯(0%)`, `관찰되었다 (Fig. 3)` → `관찰되었다(Fig. 3)`) is dropped; full-width CJK brackets are left alone.
- **Line-end hyphen joins kept space-free** — a word split across a line break by a hyphen no longer gains a stray space at the join.

### Changed

- **Dependencies / CI** — `taiki-e/install-action` 2.81.8 → 2.81.9 (#717); CI hardening for single-OS flake isolation and network-setup retries (macOS Homebrew, crates.io publish-check). The page renderer is now exercised by a dedicated CI tier (`rendering,test-support,cjk-render-fallback`), which previously compiled to zero tests under the default feature set.

### Thanks

- **RayVR (Ray V. Roberts)** — contributed bundled-fallback substitution of Adobe predefined CIDFonts and the §11 transparency compositing surface with the optional `lcms2` ICC backend (#730, #674).
- **@regularkevvv (Kevin Castro)** — reported and fixed the cross-document font-cache `/ToUnicode` key collision that could serve one document's font to another in a long-lived process (#733).
- **Norbert Preining** — fixed invalid inline annotation `/AP` appearance streams, the leaked pyo3 `Py<Self>` receiver parameter in the generated `.pyi`, and UTF-16BE/LE bookmark-title decoding (#713, #728, #729).
- **Mike Beschastnov** — hardened the OutputIntents scan to skip a damaged ICC profile instead of aborting (#712, #731).

## [0.3.63] - 2026-06-09

> CJK extraction-quality fixes — vertical-CJK (tategaki) reading order no longer mis-fires on horizontal Japanese text, CJK glyphs no longer surface as Kangxi-radical codepoints, and Korean number spacing is preserved — plus recovery of dropped inter-word spaces on tightly-typeset PDFs, and routine dependency updates.

### Fixed

- **Dropped inter-word spaces on tightly-typeset PDFs (#724, #725)** — text drawn one glyph per `Tj` with incremental `Td` offsets and **no space glyph** (inter-word gaps are just slightly larger `Td` moves, ~0.3–0.8 × the glyph advance — common in résumé generators) was concatenated: `"JOHN DOE"` → `"JOHNDOE"`, `"Master of Science"` → `"MasterofScience"`. The characters were correct; only the word boundaries were lost, which breaks search/indexing (`"science"` can't be matched inside `"masterofscience"`). PyMuPDF and poppler `pdftotext` both infer these spaces, so oxide was the outlier against its own calibration reference. Two root causes, both fixed: (1) `FontInfo::get_space_glyph_width` returned a CID (Type0) font's `/DW` *default* width (often ≥ 0.5 em) as the "space advance" when code `0x20` has no explicit `/W` entry — the norm for Identity-H subsets, where the space glyph is rarely at `0x20` — inflating the geometric word-gap threshold above real word gaps; it now falls back to the 0.25 em typographic default unless `0x20` carries an explicit `/W` width (preserving the #656 Arabic-subset fallback). (2) the intra-word kerning guard in `should_insert_space` suppressed lowercase→lowercase boundaries up to 2.4 × the geometric threshold (≈ 0.33 em for Helvetica) — far wider than any real inter-letter kerning — swallowing genuine tight word gaps; the ceiling is lowered to 1.5 × (≈ 0.2 em), which still clears worst-case ~0.19 em intra-word kerning while admitting 0.2-em-and-wider word gaps, the same ~0.18–0.2 em word-break point PyMuPDF / poppler use.
- **Vertical-CJK reading order mis-fired on horizontal Japanese text** — the v0.3.62 tategaki assembler counted vertical-vs-horizontal glyph adjacencies over all pairs, which false-positived on grid-aligned *horizontal* CJK (genkō-yōshi regulatory/academic layouts align glyphs in both rows and columns), scrambling the reading order and collapsing line breaks. Detection now requires single-glyph CJK spans to dominate the page — genuine tategaki positions each glyph on its own origin, while horizontal CJK is emitted as multi-character runs — and discriminates by each glyph's nearest-neighbour direction with a clear vertical majority. Ambiguous or horizontal pages fall back to the normal horizontal assembler (the pre-vertical-CJK behaviour), so a missed detection can never regress against that baseline.
- **CJK glyphs surfaced as Kangxi/CJK *radical* codepoints** — a glyph shared between a radical and its unified ideograph could reverse-resolve to the radical form (e.g. 欠→⽋, 立→⽴, 言→⾔, 金→⾦). The CID decode path now NFKC-normalizes the CJK Radicals Supplement (U+2E80–2EFF) and Kangxi Radicals (U+2F00–2FDF) blocks — which never appear in running text — to their unified ideograph; ordinary text and fullwidth forms are untouched.
- **Korean number spacing was wrongly stripped** — the CJK number-boundary cleanup that correctly removes stray spaces between Chinese/Japanese ideographs and embedded digits (`公元前 1000 年` → `公元前1000年`) over-fired on Korean. Korean is written with inter-word spaces, so a space between a Hangul syllable and a number is a real word boundary (`14 예` = "14 cases"); Hangul is now excluded from the cleanup.

### Changed

- **Dependencies** — `p12-keystore` 0.2.1 → 0.3.0 (`from_pkcs12` adapted to its new `Pkcs12ImportPolicy` API), `uuid` 1.23.2 → 1.23.3, `chrono` 0.4.44 → 0.4.45; CI actions: `actions/checkout` → 6.0.3, `astral-sh/setup-uv` → 8.2.0, `ruby/setup-ruby` → 1.312.0, `codecov/codecov-action` → 7.0.0, `taiki-e/install-action` → 2.81.8.

### Thanks

- **@regularkevvv (Kevin Castro)** — reported the tightly-typeset inter-word spacing bug and contributed the fix (#724, #725).

## [0.3.62] - 2026-06-09

> Best-in-class text, Markdown and HTML extraction: right-to-left (Arabic/Hebrew) and vertical-CJK (tategaki) reading order across every format, untagged multi-column reading order, table / list / heading / hyperlink structure fidelity, paragraph reflow and fenced code blocks, correct display of flattened form fields (incl. CJK/emoji), hyperlink-URI XSS hardening, and an OCR-augmented `auto` extraction mode that is a strict superset of native output on text, Markdown and HTML.

### Added

- **`auto` extraction mode is now a true superset of native on every format — text, Markdown, and HTML (#661)** — `extract_text_auto` / `AutoExtractor::extract_text` / `extract_markdown` / `extract_html` keep the native text layer verbatim and **add** text recovered from image regions (a chart with rasterized data labels, a figure caption baked into a bitmap), never replacing good native text with a worse full-page OCR pass. The recovered text is positioned at the figure's reading-order slot, not appended after the page: it is wrapped in a synthetic span anchored to the image's page-space bounding box, borrowing the marked-content id (MCID) of the native line immediately above the figure so a tagged page (assembled in marked-content order, §14.7.4.3) drops it into the right place; an untagged page positions it geometrically by its y-centre. Recovered lines already present natively are de-duplicated. New reusable building blocks back this: `PdfDocument::extract_text_with_extra_spans`, `to_markdown_with_extra_spans`, and `to_html_with_extra_spans` (the converters were refactored to share an `*_inner` core). On the `figure-image-text` golden case this lifts `auto` from worse-than-native to best in class (text 0.508 → 0.074, Markdown 0.376 → 0.167, HTML 0.444 → 0.264 CER); across the golden corpus `auto` is never worse than native on any document.
- **Hyperlinks in Markdown and HTML output (#670)** — `/Link` annotation URIs are now emitted as `[anchor](uri)` / `<a href>` instead of dropping the destination and keeping only the anchor text. URIs are tagged onto the covered spans in the converter pipeline and survive reading-order reassembly.

### Changed

- **Untagged multi-column reading order — short pages (#658)** — the multi-column detector gained a clean-gutter test (`has_clean_column_gutter`): a vertical channel that no text span crosses, with multi-line content of overlapping vertical extent on both sides, is the unambiguous geometric signature of side-by-side columns even on a short article the span-count histogram cannot classify. Such pages now route through the existing XY-Cut column ordering (recovered from layout because untagged pages carry no logical-structure reading-order hint, §14.8) instead of reading row-by-row across the gutter. `twocol-untagged` text CER 0.442 → 0.007, with zero change to single-column or already-detected layouts.

### Fixed

- **Right-to-left (Arabic / Hebrew) reading order across text, Markdown, and HTML (#656, #657, #669, #667)** — RTL runs are now reassembled in logical order on every surface: the tagged-extraction path, the plain-text assembler, and the Markdown/HTML converter pipeline (which previously emitted spans in visual order). Display headings and prose drawn right-to-left reconstruct as a single heading/paragraph rather than fragmenting; producer-inserted interior spaces inside cursive Arabic runs are stripped (§14.8.2.3.3 — a show string "shall not contain interior SPACEs"); neutral punctuation (the Hebrew comma case, #669) attaches to the correct word; and the word-gap heuristic no longer injects spurious spaces inside complex-script (Hangul/Indic) clusters (#667).
- **Vertical CJK writing mode (tategaki) across all formats (#659)** — pages whose majority of spans are vertical (WMode 1, `-V` encodings) are read top-to-bottom, right-to-left in `extract_text` and now equally in `to_markdown` / `to_html` (the converter pipeline mirrors the plain-text column-major path instead of shredding columns). Vertical CJK prose is no longer mis-detected as a spatial table.
- **Spurious spaces between CJK glyphs and embedded numbers (#665)** — a literal space glyph some producers emit at a CJK↔number script transition ("公元前 1000 年") is dropped at end of assembly, so the number attaches directly to the surrounding ideographs ("公元前1000年") as the script requires.
- **Tagged two-column assembler — duplication and block reorder (#660)** — wrapped lines no longer split into one paragraph per visual line; a column line-wrap reflows into one paragraph, hyphenated words split across lines are rejoined, and a leading number is treated as a sentence continuation rather than a list marker.
- **Table, list, and heading structure in Markdown / HTML (#668, #664, #671, #672)** — tagged table rows linearize with a single newline between cells (one row per line) instead of a newline per cell (#668); Markdown lists emit tight (no blank line between items) and with correct block boundaries for the structure score (#664); HTML list items render as `<ul>/<ol>/<li>` instead of `<p>` (#671); and HTML headings keep their level (no H1→H2 shift) without wrapping the heading text in a stray inline element (#672). Bulleted lists and monospace code are no longer mis-detected as spatial tables; monospace line runs render as fenced Markdown code blocks.
- **Flattened form fields now display their filled values, including CJK and emoji (#647, #675)** — flattening a filled form regenerates each field's appearance from `/DA` + `/DR` so the value is actually painted (Latin), and an opt-in `cjk-form-fonts` feature embeds CJK + emoji fonts so non-Latin values (やまだたろう🍺) render too, instead of appearing blank. Thanks **@mitslabo**.
- **`extract_structured` column routing now activates for reference-edition layouts (#673)** — `column_index` was always `null` on every region for documents such as two-column Bibles, so the body of both columns coalesced into a single interleaved text string. The gutter detector was center-of-mass based with a 12 %-of-page-width gate, which collapses exactly where column routing matters most — short ragged verse lines, word-level spans, and a narrow gutter leave no wide centre-gap. It now detects the gutter by edge-based empty-corridor analysis (the valley in the horizontal projection of span extents that no span crosses) with an absolute minimum-width gate, so narrow real gutters are found and inter-word gaps are still rejected. Columns now receive `column_index` `0` / `1`, and body spans split into per-column regions instead of merging. Thanks **@lggcs**.

### Security

- **Hyperlink-URI allowlist in Markdown / HTML output** — the hyperlink emission added for #670 now passes destination URIs through a shared `is_safe_link_uri` scheme allowlist (http, https, mailto, tel, ftp, …), so a malicious `/Link` action such as `javascript:alert(1)` cannot reach a consumer-facing `<a href>` / `[text](uri)` as an XSS vector.

### Maintenance

- **Pre-commit hooks and pre-existing lint errors fixed (#677)** — Thanks **@norbusan**.
- **Documentation for the page-oriented Python API (#678)** — Thanks **@SeanPedersen**.

## [0.3.61] - 2026-06-07

> Press-accurate CMYK→RGB rendering via document `/OutputIntents` ICC profiles, vertical writing mode (WMode 1 / tategaki) support, RTL (Hebrew/Arabic) and Indic text-extraction fixes, separation-plate image rendering and ActualText extraction, path flattening (`PathContent::to_points`), Node.js quickstart and form-display fixes, macOS OCR detection, faster table-heavy extraction, and cross-OS + cross-language CI verification

### Added

- **Vertical writing mode (WMode 1 / tategaki) support across extraction, rendering, and reading-order pipelines (#645)** — Japanese tategaki, Traditional Chinese vertical packaging, and similar `-V`-suffixed encodings (Identity-V, UniJIS-UTF16-V, UniGB-UTF16-V, UniCNS-UTF16-V, UniKS-UTF16-V) plus CMap streams with `/WMode 1 def` now drive vertical glyph advance along the y-axis instead of being silently rendered as horizontal. The §9.4.4 axis-swap math lives in a single helper (`GraphicsState::advance_text_matrix`) consumed by the extractor, page renderer, separation renderer, and text rasterizer — horizontal text pays one predicted-not-taken branch per advance. Per-CID `/W2` (§9.7.4.3) and `/DW2` arrays are parsed for vertical metrics; ToUnicode `/WMode` is intentionally ignored per §9.10.2 so a stale tooling leftover can't flip the document. Vertical-majority pages (≥50% of spans tagged `wmode == 1`) bypass the configured `ReadingOrderStrategyType` and route through a dedicated right-to-left column-ordering path, since none of the horizontal strategies can produce correct vertical reading order. Thanks @RayVR.
- **`PathContent::to_points(tolerance)` path flattening (#147)** — flattens an extracted vector path into polylines (`Vec<Vec<(f32, f32)>>`, one inner vec per subpath) for consumers that need sampled coordinates rather than drawing operators (chart/ECG/CAD digitisation). `MoveTo`/`LineTo` pass through unchanged; cubic Béziers are adaptively subdivided to stay within `tolerance` of the true curve. Subpath handling follows ISO 32000-1:2008 §8.5.2 (Table 59). Thanks @mbeschastn0v, and @joelparkerhenderson for the use case.
- **Separation-plate image rendering (#631)** — raster Image XObjects are now routed to the matching ink plates in the separation renderer (previously only Form XObjects were handled, so photo content, gradients, and sample-based artwork were absent from per-ink output). Per-pixel routing dispatches by image colour space (ISO 32000-1:2008 §8.9). Thanks @RayVR.
- **`/ActualText` extraction for structure-tree spans (#646)** — `/ActualText` on a `StructElem` (the form InDesign emits for drop caps, ligature spans, and stylized text in tagged PDFs, §14.9.4) is now applied correctly in `extract_text` / `to_markdown` / `to_html` — emitting the replacement text once, at the right position, instead of duplicating it with the raw descendant glyphs. Marked-content-scope `/ActualText` already worked. Thanks @RayVR.
- **Article-thread (`/Threads`) parsing (#458)** — a new parser reads a document's article threads (ISO 32000-1:2008 §12.4.3) into per-page bead rectangles, with an accompanying reading-order strategy, shipped as tested public building blocks. The default reading order is unchanged; auto-wiring threads into it is tracked for a future release.
- **Cross-language test-parity suite** — one shared functional spec (open, extract, convert, search, structured extraction, create, encrypt, version) is now implemented idiomatically in all nine bindings (Rust, Python, Node, Go, Java, Ruby, PHP, C#, WASM), so every binding is verified to expose the same core behavior.
- **Press-accurate CMYK→RGB via document `/OutputIntents` ICC profile (#652)** — the composite render path now consumes the document's `/OutputIntents` CMYK `DestOutputProfile` and routes `/DeviceCMYK` paint, `/Separation` / `/DeviceN` colourants resolving to a `/DeviceCMYK` alternate, and `/ICCBased N=4` spaces lacking a usable embedded profile through `qcms` (ISO 32000-1:2008 §14.11.5, §10). The conversion is built as `qcms::Transform::new_to(src = OutputIntent, dst = sRGB)`, so it uses the OutputIntent profile's AToB ("device-to-PCS") direction into the CIE PCS and then the sRGB profile's PCS-to-device direction out — composite direction CMYK → CIE PCS → sRGB. Closes the press-vs-screen colour divergence on heavy-yellow / saturated-mid-tone branding artwork that previously rendered through the §10.3.5 additive-clamp fallback. When no `/OutputIntents` is declared, §10.3.5 is preserved byte-for-byte. Thanks @RayVR.
- **Page-level `/DefaultGray` / `/DefaultRGB` / `/DefaultCMYK` overrides (§8.6.5.6)** — when a page or Form XObject's `/Resources /ColorSpace` declares these defaults, the canonical `g` / `rg` / `k` / `K` operators (and their stroking siblings) are routed through the override colour space before any document-level `/OutputIntents` lookup. A `/DefaultCMYK [/ICCBased <N=4 stream>]` override drives the conversion through its embedded profile; the override takes precedence over the document `/OutputIntents` for bare device-family paint. Form XObject overrides take precedence inside the form's scope (§7.8.3).
- **Rendering-intent operator (`/RI`) honoured in the render path (§10.7.3)** — the `/RI` operator was being parsed but its value never reached the colour conversion. The graphics-state intent (`/AbsoluteColorimetric` / `/RelativeColorimetric` / `/Saturation` / `/Perceptual`, defaulting to `/RelativeColorimetric`) now flows into every qcms `Transform::new_srgb_target` build. Two `/RI` settings on the same page now compile two distinct transforms instead of silently sharing one.
- **ICC v2 and ICC v4 `DestOutputProfile` profiles both supported** through qcms 0.3.0's unconditional header-version check. A v4 LUT8-tag-form profile compiles through the same code path as the v2 equivalent and produces byte-identical RGB.
- **Per-page compiled-transform cache** (`IccTransformCache`, lives on `PageRenderer`) keyed on `(profile.content_hash, intent)`. Amortises the 17⁴ CLUT precomputation `qcms::Transform::new_to` runs for CMYK input across paint operators that share a profile and intent: a page emitting 1 000 identical CMYK paints builds one transform, not one thousand. The cache is dropped per page so memory stays bounded across renders.

### Changed

- **Faster extraction on table-heavy pages** — an output-preserving optimization to the table detector cuts ~30% off extraction time for large, dense documents (e.g. regulatory volumes) with byte-identical results.
- **Cross-OS + FIPS example verification in CI** — the core example scenarios for every language binding now run **and assert their output** on Linux, macOS, and Windows (previously Linux-only — the gap that let #648 reach users), including a FIPS-safe run. This is the guard that prevents another platform-specific quickstart regression.
- **Renderer resolution pipeline refactor (#649)** — the copy-pasted paint-resolution arms in `page_renderer` and `separation_renderer` are unified into a single layered `rendering/resolution` module (colour resolution, overprint, blend-mode, clip, per-plate routing), removing quiet divergence between the two renderers. This also fixes PostScript Type 4 calculator tint transforms for `/Separation` and `/DeviceN` spot colours over `DeviceCMYK`/`ICCBased` alternates, which previously fell through to a flat fallback. Thanks @RayVR.
- **CI reliability hardening (#544)** — SHA-pinned the remaining floating GitHub Action and added network retries to reduce transient CI failures.
- **Dependency & CI-action updates** — `imageproc` 0.27, `subsetter` 0.2.6, `log` 0.4.32, and the `actions/checkout`, `taiki-e/install-action`, and `astral-sh/setup-uv` actions were bumped (dependabot, #639/#637/#636/#643/#641/#640).
- **`FontInfo` gains three new `pub` fields (`wmode: u8`, `cid_vertical_metrics: Option<HashMap<u16, VerticalMetrics>>`, `cid_default_vertical_metrics: VerticalMetrics`) (#645)** — source-breaking for downstream code that constructs `FontInfo` with struct-literal syntax; add the three new fields to fix. Horizontal-only fonts pay no allocation cost (`cid_vertical_metrics: None`, `cid_default_vertical_metrics` is `Copy`). `FontInfo` continues to NOT be `#[non_exhaustive]`, consistent with the 0.3.60 `ascent`/`descent` addition. The natural construction values for the new fields are `wmode: 0`, `cid_vertical_metrics: None`, `cid_default_vertical_metrics: VerticalMetrics::SPEC_DEFAULT`.
- **`ReadingOrderConfig::strategy` is now overridden on vertical-majority pages (#645)** — the configured horizontal strategy (Simple, Geometric, XYCut, StructureTreeFirst) is bypassed when a page has ≥50% vertical-writing spans, and the page is ordered through the tategaki path instead. Per-span `wmode` is preserved on every output span so consumers can still distinguish the two modes. See `ReadingOrderConfig::strategy` rustdoc for the rule.
- **`ResolvedColor` gains an `IccCmyk { rgba, cmyk }` dual-payload variant (#652)** — `/ICCBased N=4` paint with a parseable embedded profile (and `/DefaultCMYK [/ICCBased N=4]` overrides) emits both the pre-converted RGBA (consumed by the composite backend) and the original CMYK quadruple (consumed by the per-plate separation router). Source-breaking for downstream code that exhaustively matches on `ResolvedColor`; add the new arm to fix. The type is not `#[non_exhaustive]`.
- **`/ICCBased N=4` with an embedded profile now wins over document `/OutputIntents`** (§8.6.5.5). Pre-this-change, an embedded `/ICCBased N=4` colour space with a parseable qcms profile emitted `ResolvedColor::Cmyk` and was projected through the document `/OutputIntents` ICC profile by the composite pipeline — inverting the spec's "embedded ICC trumps OutputIntent". The four components are now routed through the embedded profile directly and the OutputIntent is consulted only when the embedded profile fails to parse or qcms refuses to build a CMM.

### Fixed

- **Node.js quickstart (#648)** — the documented Quick Start used CommonJS `require` (the package is ESM-only) and `new PdfDocument(path)` (not a public constructor), so the first example threw. Samples now use `import` + `PdfDocument.open(path)`, and the constructor gives an actionable error if handed a path. Thanks @abeq for the report and @lihouwenbin for the docs fix (#651).
- **Form fields filled but not displayed (#647)** — for PDFs with an inline AcroForm, a filled field's value was written but the `/NeedAppearances` flag was dropped on full-rewrite save, so viewers showed the field blank (ISO 32000-1:2008 §12.7.3.3). The flag now survives, and viewers render the filled value. Thanks @mitslabo.
- **macOS OCR engine detection (#632)** — `onnxruntime` auto-discovery missed the versioned `libonnxruntime.<version>.dylib` macOS actually ships, so OCR was silently skipped; detection is now version-tolerant across Linux/macOS/Windows. Thanks @paliwalvimal.
- **Deterministic table detection** — several table-detection steps could order results by per-process hash iteration, yielding run-to-run differences on table-heavy pages; these are now deterministic.
- **Decimal points in `CMSY`/Symbol math fonts** — numbers whose decimal point is drawn from a math font's `logicalnot` glyph (e.g. `1¬00`, and the spaced `1¬ 00`) now extract as `1.00`.
- **Consistent empty output for unreadable encrypted PDFs** — all text surfaces (`extract_text`, markdown, HTML, plain text) now uniformly return empty for an encrypted PDF that can't be decrypted, instead of one surface diverging (ISO 32000-1:2008 §7.6).
- **RTL (Hebrew/Arabic) word order in tagged PDFs (#656, #657)** — `extract_text` on a tagged PDF assembles text from the structure tree and never reached the untagged `reverse_rtl_visual_order_runs` pass, so a pure-RTL run's word-spans were emitted in visual (left-to-right) order — the whole line reversed. A new geometric pass (`order_mcid_spans` → `row_aware_span_cmp_rtl`) now emits each pure-RTL row right-to-left (rightmost word first), reconstructing logical reading order from page geometry independent of how the producer stored the run (ISO 32000-1:2008 §14.8, UAX #9 §3.3.4). Hebrew text-extraction error rate on the multilingual benchmark drops from worst-tier to parity with poppler/pdfium. Two companion fixes improve Arabic: a **grapheme-cluster reversal** keeps combining marks (kasra/shadda) attached to their base letter instead of floating off, and a **space-gap threshold fallback** (0.25 em) when a CID subset font omits a space glyph stops the geometric word-gap threshold from collapsing to zero and shattering cursive words into single letters (§9.3.3 — `Tw` does not apply to composite-font Arabic). Remaining Arabic intra-word phantom gaps (glyph ink-width vs advance-width) and the markdown/HTML converter paths are tracked in #656/#657. Mixed RTL+Latin runs are left untouched pending full bidi.
- **Node.js binding missing barrel exports (#653)** — the package's ESM entry re-exports `ContentType`, `ImageFormat`, `ThumbnailManager`/`ThumbnailSize`, and the `OCRDetectionMode`/`OCRLanguage` aliases through `managers/index.js`, but that barrel never re-exported the hybrid-ml and thumbnail modules (the CJS `require` path tolerated the gap silently). Strict ESM consumers — including the new cross-language core-parity test — failed at import with "does not provide an export named 'ContentType'". The managers barrel now re-exports all of them.
- **Java binding native-library load on macOS/Windows (#653)** — `NativeLoader` loaded the explicit `fyi.oxide.pdf.lib.path` override unconditionally, but the Maven build defaults that property to a Linux `.so` path. On macOS (`.dylib`) and Windows (`.dll`) the override file is absent, so `System.load` hard-failed with `UnsatisfiedLinkError` even though the correct platform native is bundled in the JAR. The loader now checks the override exists and otherwise falls through to the bundled resource. Surfaced by v0.3.61's new cross-OS Java JNI test runs.
- **Number corruption in plain-text table cells** — per-glyph table cells (`Td <hex> Tj`, e.g. `0.99`, `Q1`) were merged into one span without keeping the `char_widths` array in sync, so the width-based column-spanning-decimal and letter→digit splitters misfired and **dropped the decimal point** (`0.99` → `0 99`) or inserted a spurious space (`Q1` → `Q 1`). `char_widths` is now re-synced on merge (benchmark table CER 0.117→0.067, 0.091→0.061).
- **Spurious word spaces in Indic scripts (Tamil/Bengali/Devanagari)** — Brahmic text extracted the right codepoints but inserted a space after nearly every dependent vowel sign (matra), because a matra carries its own advance and the geometric gap test read matra→consonant as a word break. Tamil/Bengali/Telugu/Kannada/Malayalam are now recognised by the complex-script word-boundary path, the matra→base-consonant boundary is suppressed, and `should_insert_space` gained a combining-mark guard (benchmark text CER: Tamil 0.095→0.035, Bengali 0.175→0.032, Devanagari 0.066→0.016; real word breaks carry an explicit space glyph, §9.3.3).

### Known limitations

These are limitations of the upstream `qcms` 0.3.0 colour engine (items 1–2) and one test-coverage gap (item 3), tracked in #655. The test suite documents each with a `HONEST_GAP_*` marker wired as an upgrade gate, so a future engineer (or a qcms upgrade) flips the gated test RED on landing:

- **qcms 0.3.0 ignores the CMYK rendering intent**. The end-to-end intent chain inside pdf_oxide is correct — `gs.rendering_intent` → `ResolutionContext::rendering_intent` → `Transform::new_srgb_target`'s `intent` parameter → qcms — but qcms 0.3.0 declares the intent as `_intent` for CLUT-based CMYK conversion (`transform.rs:1283-1289`) and dispatches the same CLUT for every PDF intent. A qcms upgrade that honours the parameter, or a CMM swap, will surface intent-sensitive behaviour without further code changes; the test `qa_round3_qcms_030_treats_cmyk_intent_as_informational` is the upgrade gate.
- **qcms 0.3.0 has no Black-Point Compensation** (`lib.rs:29-36` — upstream documents the choice as intentional). `qa_round4_bpc_paper_white_preservation_under_relative_colorimetric` is `#[ignore]`-marked with `HONEST_GAP_QCMS_030_NO_BPC`.
- **No real-corpus branding-logo regression fixture** (`HONEST_GAP_NO_REAL_BRANDING_FIXTURE`). The synthetic green-mark probe (`qa_round4_branding_green_mark_routes_through_output_intent`) pins the press-target direction-of-shift through saturation collapse; a vendor-issued press profile plus a CIEDE2000 ΔE assertion against a commercial-viewer baseline would tighten the bound.

## [0.3.60] - 2026-06-03

> Converter performance sweep (no double per-page extraction, cached structure-tree traversal) + Arabic/Persian CIDFont extraction, ZapfDingbats coverage, graceful encrypted-PDF text extraction, and an `extract_tables` opt-out for speed-first text extraction

### Added

- **`PathContent::to_points(tolerance)`** — flattens an extracted vector path into polylines (`Vec<Vec<(f32, f32)>>`, one inner vec per subpath) for consumers that need sampled coordinates rather than drawing operators (chart/ECG/CAD digitisation). `MoveTo`/`LineTo` pass through unchanged; cubic Béziers (`CurveTo`) are adaptively subdivided so the polyline stays within `tolerance` of the true curve. Subpath handling follows ISO 32000-1:2008 §8.5.2 (Table 59): `re` is a complete closed 5-point subpath, `h` closes and terminates the subpath (a following segment starts a new one), and a lone or overridden `m` leaves no vestige. `tolerance` is in the path's coordinate units (PDF points for paths from `extract_paths`); non-positive/non-finite values are floored so subdivision always terminates. Purely additive. (#147) Thanks @joelparkerhenderson for the use case.
- **`TextChar::ascent` and `TextChar::descent`** — glyph ascent and descent in device space (pre-multiplied by effective font size, matching the units of `advance_width` and `rendered_advance`). Sourced from the font's `FontDescriptor` (`/Ascent` / `/Descent`), with fallbacks to built-in metrics for the 14 standard PDF fonts and then Poppler-compatible defaults (0.95em / −0.35em). For Type0/CID fonts the values are now read from the CIDFont descendant's descriptor (§9.7.4) rather than silently falling back to 0.95em / −0.35em. Use `origin_y + ascent` / `origin_y + descent` directly to get glyph bounding-box edges. Thanks @haberman.

- **Arabic/Persian CIDFont text extraction (Adobe-Arabic-1 / Adobe-Persian-1)** — Type0 CIDFonts that declare `/CIDSystemInfo /Ordering (Arabic)` or `(Persian)` and ship without an embedded `/ToUnicode` (Nazanin, Yagut, Mitra, Lotus and similar) now decode through the existing Arabic-block CID→Unicode mapping instead of falling through to Latin-Extended-B garbage. Both predefined-CMap dispatch sites (`character_mapper.rs`, `font_dict.rs`) gained `"Arabic"`/`"Persian"` ordering arms (ISO 32000-1:2008 §9.7.3 / §9.7.5 / §9.10.3 step-3 identity fallback).
- **ZapfDingbats circled-digit and arrow glyphs (①–➓, ➔ ➾, → ↔ ↕)** — the standard-14 ZapfDingbats built-in encoding now maps the circled-digit ranges (`①–⑩`, `❶–❿`, `➀–➉`, `➊–➓`) and the arrow ranges (ISO 32000-1:2008 Annex D.6, octal codes 254–376) that were previously dropped, recovering this content from ZapfDingbats showcase documents.
- **Symbol-font math operators ≤ ≥ ∞** — the Adobe Symbol built-in encoding now maps `lessequal`/`greaterequal`/`infinity` (Annex D.5, octal 243/263/245), previously unmapped.
- **CLI `text --format structured` and MCP `extract` `format: "structured"`** (#626) — both surfaces now expose the library's `extract_structured` API, emitting `StructuredPage` JSON: typed regions (`kind` = RegionRole — body, heading, marginal label, header/footer, page number, artifact) with per-region `column_index`, so two-column PDFs (Bibles, dictionaries, papers with side notes) come out as separate column blocks instead of line-interleaved. Previously this was reachable only from the Rust library and Python binding. Thanks @lggcs.
- **`extract_tables` keyword on the Python `extract_text`** — `doc.extract_text(page, extract_tables=False)` skips the table-detection sweep for speed-first raw-text extraction (the dense-academic-page hot spot). Default `True` reproduces previous behaviour byte-for-byte.

### Changed

- **`TextChar` and `FontInfo` gain two new `pub` fields (`ascent: f32`, `descent: f32`)** — source-breaking for downstream code that constructs these structs with struct-literal syntax; add the two new fields to fix. Both structs are not `#[non_exhaustive]`.
- **Encrypted PDFs that cannot be decrypted with the empty password now extract empty text instead of erroring** — `extract_text`/`extract_spans`/`to_markdown`/`to_html` and their whole-document variants warn and return empty output, matching `pdftotext`/PyMuPDF (ISO 32000-1:2008 §7.6). `page_count` still returns `Err(EncryptedPdf)` so callers that query document structure are not silently handed zero pages; image extraction still fails closed.

### Fixed

- **Decimal points in Computer-Modern math subsets no longer decode as `¬`** — when a CM/Symbol subset draws its decimal from the `logicalnot` slot, a `¬` directly between two digits (e.g. `1¬00`) is recovered as `.` (`1.00`); spaced logic/set `¬` is left untouched.
- **Oversized drop-cap / table-title initials re-attach to their word** — a lone uppercase initial set in a larger font (so it became its own span) is merged with the body run to its right before reading-order sorting, fixing `TABLE` → `T … ABLE` stranding on regulatory tables. Gated to genuine initials (oversized vs the page's median body text, touching its continuation, on the same baseline) so inline math (`A_st`), word-spaced capitals (`A Perspective`), and tall initials reaching the line above are left alone.
- **Rotated text runs are read as their own blocks instead of scrambling the page** — a run drawn with a rotated text matrix (`atan2(b,a)` of `T_m × CTM`, ISO 32000-1 §9.4.4) — the vertical `arXiv:…` margin stamp, rotated plot axis labels, rotated table column headers — was interleaved into the horizontal row-band / XY-cut sort, whose axis-aligned assumptions it violates. Such runs now carry a `rotation_degrees` and are stably lifted out of the horizontal flow (which keeps its exact prior order — pages with no rotated text are byte-identical) and re-emitted as their own blocks ordered in an upright frame. Recovers e.g. the rotated `Array`/`Boolean`/… headers of a PDF-syntax matrix and stops a chart's axis labels from fusing into its data (`0CardSort` → `0` + `CardSort`).
- **Borderless numeric results tables keep one value per column** — a dense ML/benchmark grid laid out on a tight, regular numeric pitch (no ruling lines) had adjacent columns fused by greedy clustering (`0.69 0.76` sharing one cell). When the spans are predominantly numeric and a finer text-edge column set recurs across ≥3 rows on a regular pitch — and still forms a valid grid — each value now lands in its own column. The validity probe guarantees the refinement never demotes an otherwise-valid table to prose.
- **`to_markdown` / `to_html` now honour `exclude_regions` and `include_region`** (#609) — these `ConversionOptions` region filters were applied only by the plain-text path (`extract_text` / `to_plain_text`), so markdown and HTML emitted the whole page regardless of the requested exclusions. The filter now runs up front for all three surfaces (shared `apply_region_filters`), before table/heading/reading-order processing, so excluded content is gone everywhere; tables in excluded regions drop too. No-op when neither field is set (default output unchanged). Thanks @alexanderameye.
- **Dense numeric tables are no longer flattened into bold-label + run-on numbers** — the spatial-table quality gate rejected any ≥5-column table whose cells are >70% single-word, to suppress prose accidentally split into one-word columns. A genuine numeric data table (financial/metrics slides, benchmark grids) is legitimately almost all single tokens — every cell is a number — so it was wrongly rejected and emitted as flattened text. The gate now bypasses that rule when a table is numeric-dominated (≥50% of cells are data values like `5,012` / `+2%` / `240`); number-heavy prose stays below the threshold and is still rejected. Recovers e.g. the tracemonkey SunSpider benchmark table and arXiv result/count matrices that were previously flattened.
- **A stray superscript ordinal is no longer promoted to a heading** — when a superscript ordinal (`st`/`nd`/`rd`/`th`) is split from its number ("May 5th" → "May 5" + superscript "th"), the lone suffix was emitted as its own `#### th` heading under `detect_headings`, fragmenting the document outline. Heading detection now rejects a bare ordinal suffix.
- **Scanned/image pages are marked instead of rendering silently blank** — a page with no extractable text that classifies as scanned/image now emits a `> [OCR REQUIRED — page N]` block-quote in `to_markdown`/`to_markdown_all`, so a reader of a scanned document sees where content was lost and OCR is needed rather than half the document silently missing. Gated to genuinely scanned/image pages (legitimately-blank pages are untouched) and suppressible via the new `ConversionOptions::annotate_skipped_pages` (default `true`).
- **OCR reading-order sort no longer panics the host process** — the detection-box reading-order comparator (`ocr::engine`) used a non-transitive rule ("compare X when the Y gap is < 10 px, else compare Y"), which Rust's sort detects on image-text pages with a few near-aligned labels and turns into a `comparison function does not correctly implement a total order` panic — aborting the host across every binding (C#/Go/Node/Rust). It is now a genuine total order: group by a fixed 10 px Y band, then X, then Y.
- **Text extraction no longer panics on spans with out-of-range coordinates** — a span whose centre maps to the `i32` limits could overflow the dedup-grid neighbour scan in an overflow-checked build; the cell-index arithmetic now saturates.
- **Line-end hyphen rejoin is more conservative** — a hard `-` wrap only rejoins when both fragments are lowercase-alpha (keeps `COVID-19` / `well-Known`); soft hyphens (U+00AD) still rejoin per §14.8.2.2.3.
- **Bogus `U+FFFF`/`U+FFFE` ToUnicode placeholders fall back instead of being emitted** — some producers stuff the BMP noncharacters into a `/ToUnicode` CMap as a "no glyph" marker (e.g. an Identity-H subset mapping every CID to `<ffff>`). Noncharacters are never valid interchange text, so for Identity-encoded fonts these are now treated as a CMap miss and routed through the CID→GID→embedded-`cmap` / CID-as-Unicode fallback (recovering real text when the embedded font carries a usable `cmap`), consistent with the existing notdefrange-`U+FFFD` handling.
- **Filling a merged field+widget form field no longer blanks the widget** — when an AcroForm text field's dictionary is also its widget annotation (the common single-widget case, ISO 32000-1:2008 §12.7.4.1), `DocumentEditor::set_form_field_value` followed by `save_to_bytes` wrote the value onto a freshly-allocated bare field object (no `/T`, `/Subtype`, `/Rect`, `/AP`) appended to the page `/Annots`, and left the real widget the reader displays with an empty `/V`. Readers (PyMuPDF, Acrobat) then showed an empty field and could not build an appearance stream. The value (and `/AS` for buttons) is now written **in place** on the existing field/widget object, preserving its full dictionary and its `/Fields` / `/Annots` membership, with `/NeedAppearances` set so viewers regenerate; CJK values round-trip (`山田太郎` → `/V <FEFF…>`). Thanks @mitslabo.
- **Subset CFF (Type 1C) fonts resolve glyphs through the PDF `/Encoding` (#629)** — for a simple CFF font the byte→glyph-name mapping is now taken from the font dictionary's `/Encoding` (its `/BaseEncoding` — WinAnsi / MacRoman / StandardEncoding per Annex D — plus `/Differences`) and resolved to a GID through the CFF Charset (parsed over the full `nGlyphs`), per ISO 32000-1:2008 §9.6.6 — instead of the font program's own, frequently sparse, built-in Encoding table. Aggressively-subset CFF fonts (common in prepress/packaging artwork) whose internal Encoding mapped only a handful of bytes previously dropped every other byte to `.notdef`, painting a single fallback glyph; they now render the full subset. Thanks @RayVR.

### Performance

All performance changes preserve output (byte-identical for inputs that previously decoded correctly), except where noted; a small number carry a floating-point/tie-break corpus note in the release notes.

- **Converters no longer extract every page twice** — `to_markdown`/`to_html` extracted a page's spans once for their own use and again inside the table-detection path (`extract_page_tables` → `extract_words` → `page_reading_order`). Postprocessed spans are now memoized per page (invalidated by redaction), removing the second full glyph-decode + postprocess pass.
- **`to_html` structure-tree traversal is now cached (O(pages²) → O(pages))** — `to_html`'s reading-order context used the un-cached per-page `traverse_structure_tree` walk; on a tagged N-page document that is O(N × tree) ≈ O(N²). It now uses the document-level all-pages traversal cache (the pattern `to_markdown` already used, #608).
- **`find_table_elements` is computed once per document** — the converter table path walked the whole structure tree per page (another O(pages²) on tagged docs); a single all-pages walk now buckets `Table` elements by page for O(1) per-page lookup.
- **XY-cut depth bound + clustering memoization** — `partition_indexed` gained a recursion depth cap (bounds the O(n)-deep singleton-peel pathology on header/footer-heavy pages); and `classify_region_kind` is now computed once per node instead of being re-run by each prose detector.
- **Type0/CID `char_to_unicode` is memoized per font** — composite-font glyphs re-ran the full decode cascade (and re-allocated) on every occurrence; a per-font cache makes repeat decodes O(1), also removing the redundant CJK-fallback rescan and TJ/RTL double-decodes.
- **Superscript/subscript band detection is O(n)** — `apply_super_sub_script_substitutions` replaced its per-span Y-window walk (O(n²) on wide rows) with a sliding-window maximum.
- **TJ-distribution analysis is O(1) per query** — `analyze_tj_distribution` maintains running mean/variance accumulators instead of re-scanning the offset history (was O(n²)/page on justified text).
- **Table detection fail-fast gate** — a page with thousands of line/rectangle paths (an engineering drawing/chart, not a ruled table) skips the O(E²) cell-reconstruction sweep.
- **Large contiguous ToUnicode `bfrange`s are stored compressed** — a `<0000><FFFF>`-style range no longer persists ~65 536 individual `String`s in the cached CMap; long contiguous runs are collapsed to range entries resolved by binary search. Document-order overrides (§9.10.3, #619) are preserved — the compression runs over the final mapping state.
- **Global font cache uses a concurrent read lock** — the cross-document font cache moved from `Mutex<LRU>` to `RwLock<FIFO>`, so concurrent readers don't serialize (single-threaded behaviour unchanged; cache is correctness-neutral, #408/#595 isolation preserved).
- **Superscript/subscript neighbour scan is spatially indexed** — `span_is_token_internal`'s same-line scan (O(n) per candidate) now queries a Y-band index, byte-identically.
- **Misc** — span-sort caches its row-band key once per span (`sort_by_row_band`); stroke/fill dedup uses a grid index (exact IoU preserved); the Layer-4 font-set cache guard memoizes each font's identity hash per object (no per-page re-load + re-hash); whole-document converters pre-reserve output buffers and write page wrappers without per-page `format!` temporaries; detector binning/cloning micro-optimised.

## [0.3.59] - 2026-06-01

> Community contributions — Type 4 PostScript calculator functions, optional-content (OCG/OCMD) render + extraction filtering, document-order ToUnicode parsing, per-variant standard-font width tables, subset-font cache isolation, and inline-image NUL-whitespace handling

### Added

- **Type 4 (PostScript calculator) function evaluator** — a complete, standalone evaluator for PDF Type 4 functions (ISO 32000-1:2008 §7.10.5). All Table 42 operators are implemented with spec-faithful semantics: the trigonometric operators (`sin`/`cos`/`atan`) operate in **degrees** with `atan` mapped to `[0, 360)`, `round`/`truncate`/`floor`/`ceiling` tie behaviour, strict int-vs-real typing, and `i64`-overflow handling for `idiv`/`mod`/`mul`/`add`/`sub`/`bitshift`. A dedicated `Error::Type4Runtime` distinguishes stack underflow, typecheck, sqrt-of-negative, and divide-by-zero from invalid input. *Note: this lands as a tested capability not yet wired into the Separation/DeviceN tint-transform colour path — that integration is a tracked follow-up, so rendering behaviour is unchanged for now.* (#603) Thanks @RayVR.
- **Optional-content (OCG / OCMD) filtering for rendering and extraction** — `render_page` and the text extractors now resolve optional-content visibility through a shared `optional_content` resolver, honouring OCMD `/P` policy (`AnyOn`/`AllOn`/`AnyOff`/`AllOff`) and `/VE` visibility expressions (§8.11.2.2), the default configuration `/OCProperties/D` with `/BaseState`/`ON`/`OFF` (§8.11.4), and the hidden-content text-advance rule (§8.11.3). Marked-content (`BDC /OC … EMC`) on both the extraction and rendering paths is filtered consistently, fixing a prior duplication where the renderer mis-decoded UTF-16LE/PDFDocEncoding layer names. PDFs without optional content are byte-for-byte unchanged. By design, `render_page` honours the PDF's own default configuration while `extract_text` filters only caller-supplied layers (§8.11.3 NOTE 4). (#604) Thanks @RayVR.

### Fixed

- **ToUnicode CMaps process `bfchar` and `bfrange` sections in document order (#619)** — a ToUnicode CMap is a single combined mapping space where a later definition overrides an earlier one for the same code (ISO 32000-1:2008 §9.10.3); sections are now applied in the order they appear so the last definition wins, matching Adobe/pdf.js/MuPDF/Poppler. Thanks @haberman.
- **Null byte (`0x00`) is treated as PDF white-space when locating the inline-image `EI` operator (#618)** — `NUL` is one of the six PDF white-space characters (ISO 32000-1:2008 §7.2, Table 1) but was previously omitted, so an `EI` delimited by a null byte in inline-image (`BI`/`ID`/`EI`) binary data could be missed. Thanks @haberman.
- **Bold/italic variants of the standard Times and Helvetica fonts use correct per-variant width tables (#615)** — the Bold and Italic variants of Times and the Bold variant of Helvetica were falling back to the Regular-weight width table, drifting character positions and word-break detection in documents using these common standard-14 fonts without a `/Widths` array (ISO 32000-1:2008 §9.6.2.2). Per-variant widths are now sourced from the Adobe Core 14 AFM metrics. Thanks @haberman.
- **Subset fonts with colliding BaseFont names no longer poison the cross-document font cache (#595)** — two PDFs that reuse the same subset BaseFont name (a six-uppercase-letter `+` tag such as `AAAAAA+TestFont`, ISO 32000-1:2008 §9.6.4) but embed *different* document-specific ToUnicode CMaps could be served the first document's cached `FontInfo`, decoding the second document's text to the wrong characters. Subset fonts — whose glyph subset and ToUnicode are inherently document-specific — are now excluded from the cross-document global font cache (the per-document caches are unaffected), so each document decodes with its own mapping. Thanks @RayVR for the root-cause analysis and the fix.

### Dependencies

- **CI actions**: `taiki-e/install-action` 2.79.12 → 2.81.0 (#580).

## [0.3.58] - 2026-05-31

> Structure-tree reading order with /Suspects handling, structured page extraction, two-column reference/verse routing, math-font punctuation + emoji spacing, mixed bidirectional text, /Rotate 90°/270°, image colour-space resolution, and a dependency refresh

### Added

- **`PdfDocument::extract_structured(page) -> StructuredPage`** — additive typed page surface (#536). Returns the page's text grouped into reading-order `StructuredRegion`s with a `RegionRole` (`BodyBlock`, `StructuralHeading { level }`, `MarginalLabel`, `Header`, `Footer`, `PageNumber`, `Artifact`) and a best-effort `column_index` for two-column bodies. Roles reuse signals already on each span — `/Artifact` marked content (ISO 32000-1:2008 §14.8.2.2), structure-tree heading levels (§14.7.2), and geometry (§14.8.2.3.1) — so a trustworthy tagged PDF yields tree-driven roles for free. New public types `StructuredPage` / `StructuredRegion` / `RegionRole` in `pdf_oxide::structured` (serde-serializable). Available across all bindings with idiomatic names — Python `extract_structured`, JS/WASM `extractStructured`, Go `ExtractStructured`, Ruby `extract_structured`, PHP `extractStructured`, Java `extractStructured`, C# `ExtractStructured`, and the C ABI `pdf_document_extract_structured_to_json` (all returning the serialized `StructuredPage`). Thanks @lggcs.
- **`PdfDocument::prefers_structure_reading_order() -> bool`** — read-only introspection accessor reporting whether text extraction will use the Tagged-PDF *logical structure order* (a depth-first traversal of `/StructTreeRoot`, ISO 32000-1:2008 §14.8.2.3.1 / §14.7.1) rather than geometric page-content order for this document. (#608)
- **`PdfImageHandle::indexed_base() -> Option<ColorSpace>`** — for an Indexed image (`[/Indexed base hival lookup]`, ISO 32000-1:2008 §8.6.6.3) the de-indexed base colour space; `None` for non-Indexed images. (#588)

### Fixed

- **AcroForm fill: non-ASCII (CJK/Japanese) field values no longer mojibake (#616)** — `DocumentEditor::set_form_field_value` wrote field values (and field names `/T`, tooltips `/TU`) as raw UTF-8 bytes, which a conformant reader (Acrobat, Stirling-PDF, Preview, pdf.js) interprets as PDFDocEncoding, so `山田太郎` rendered as `å±±ç"°…`. Values are now encoded as proper PDF text strings per ISO 32000-1:2008 §7.9.2.2 — a PDFDocEncoding-compatible literal for ASCII/Latin-1, and UTF-16BE with a `U+FEFF` BOM for anything above U+00FF (`/V <FEFF5C71753059 2A90CE>`). ASCII values stay plain literals.
- **AcroForm fill: a filled field no longer disappears on save (#617)** — after `set_form_field_value` + `save_to_bytes`, re-reading the document returned **0 form fields**, because a standalone terminal field lost its `/FT` (field type) on rewrite, leaving an untyped widget annotation `FormExtractor` could not classify (`/FT` is an inheritable, required key — §12.7.4.1). Parentless terminal fields now re-emit `/FT` (and `/Ff`), so fill → save → re-read round-trips correctly. (Both #616 and #617 surfaced building the issue #611 Japanese form-fill round-trip; they fix the published crate, so every binding's form-fill benefits.)
- **Two-column reference lists and short-verse bodies read column-by-column (#536)** — pages whose left/right columns share line baselines but have *short* lines (bibliographies, Bible/lexicon verse editions) were read straight across, because the table-safety guard rejected anything with a low per-line character count. A length-independent admission path now routes them down the left column then the right when a single persistent central gutter is present, gated by concentration, coverage, column char-balance, and a grid-row signal so multi-cell numeric tables stay off the column path. Display-equation rows are excluded from the gutter-coverage measurement so dense-math pages still route. Validated table-safe: `google_doc_document.pdf` markdown is byte-identical.
- **Punctuation glyphs parked at non-standard codes in symbolic fonts decode correctly (#536)** — when a font's `/Differences` names a code `period` / `comma` / `hyphen` / `minus` (ISO 32000-1:2008 §9.6.6.1) but its program / base encoding resolves that code to a non-sensible symbol, the Adobe Glyph List value (§9.10.2) is now preferred, recovering the intended punctuation. Tightly gated: correctly-mapped fonts and genuine `logicalnot` (`¬`) / math-symbol glyphs are untouched. (A figure axis that draws a symbol-font glyph whose encoding genuinely *is* `¬` where a decimal point belongs — the font itself is wrong — remains out of scope, since recovering it would require unsafe context guessing that corrupts legitimate `5×3` / `2±1`.)
- **Mixed Arabic/Latin/numeral lines settle in logical order** — a right-to-left line embedding European/Arabic-Indic numerals or Latin words (e.g. a date `14 april 1434 ٤٣٤١`) now gives each embedded left-to-right sub-run its own LTR level per the Unicode Bidirectional Algorithm (UAX #9 §3.3.4), instead of only reversing pure-RTL runs. The pass is gated to confidently-RTL mixed lines; pure-RTL, pure-LTR, and all ASCII/Latin extraction are byte-for-byte unchanged.
- **Image `decode()` resolves resource-name colour spaces (#588)** — `page_image_handles()` `decode()` previously failed on an image whose `/ColorSpace` is a resource name (e.g. `/CS0` resolved via `/Resources/ColorSpace`, ISO 32000-1:2008 §8.6.6 / §8.9.7), returning `Unsupported color space`. The active resource colour-space map is now threaded into the handle (page, Form-XObject, and inline scopes) so such images decode. Indexed image handles now report `ColorSpace::Indexed` (1-component sample layout, §8.6.6.3) with the de-indexed base available via the new `indexed_base()` accessor — an API refinement of the v0.3.57-only handle surface; decoded pixels are unchanged.
- **Suspect tagged PDFs now extract in geometric reading order (#608)** — a document advertising `/MarkInfo /Suspects true` (the `/TagSuspect /Ordering` signal that the producer could not guarantee page content order matches logical structure order, §14.8.2.3.1) was previously read through its structure tree by `extract_text`, which could emit content out of visual order. Such documents now fall back to geometric order, while trustworthy tagged PDFs — `/Marked`, or a catalog `/StructTreeRoot` on PDF-1.4 files that predate `/MarkInfo` — read in logical structure order across **all four text accessors** (`extract_text`, `to_plain_text`, `to_markdown`, `to_html`). A single shared trustworthiness predicate gates every reading-order path so they cannot drift apart, and a marked-content element whose spans cross multiple visual lines is now emitted in reading order. Non-suspect and untagged documents are byte-for-byte unchanged.

- **`/Rotate 90` and `/Rotate 270` pages now read in display orientation** — pages with a 90°- or 270°-clockwise `/Rotate` (ISO 32000-1:2008 §7.7.3.3) were previously extracted in raw user-space coordinates and came out sideways, with lines and words in the wrong order. Every span is now mapped into the displayed coordinate frame — including the page width/height swap for 90°/270° (§8.3.3) — before reading-order assembly, so all four rotation angles read upright. Annotation text on rotated pages is mapped into the same displayed frame. Unrotated and 180° pages are byte-for-byte unchanged (verified on the issue14415 180° contract, which now reads in correct order).
- **Space after an emoji is no longer dropped** — a pictographic glyph (e.g. 📄) immediately followed by a word (`📄 README.md`) previously merged into `📄README.md`, because the residual gap after the wide glyph fell below the proportional-font space threshold. An emoji→letter boundary with a positive gap now keeps the inter-token space. Word segmentation is reader latitude (ISO 32000-1:2008 §9.10); the rule is gated on pictographic codepoints (arrows and math-operator blocks excluded), so technical text is unaffected.

### Changed

- **`PdfImageHandle::decode()` / `raw_compressed_bytes()` now borrow `&self`** and `PdfImageHandle` is `Clone` (refining the v0.3.57 two-phase image API, #588). A single handle now supports an inspect → raw-bytes → decode flow without re-enumerating the page. Borrowing is a strict superset of the previous by-value signatures, so existing callers still compile.
- **Indexed image handles now report `ColorSpace::Indexed`** instead of the de-indexed base (refining the v0.3.57-only handle metadata, #588). The base remains available via the new `indexed_base()` accessor; this makes the handle's `color_space` reflect the 1-component sample layout per ISO 32000-1:2008 §8.6.6.3 and is consistent across the direct-array, indirect-reference, and inline-image forms. Decoded pixels are unchanged (`decode()` re-resolves the palette independently).

### Dependencies

- **Rust**: `serde_json` 1.0.149 → 1.0.150 (#578), `log` 0.4.29 → 0.4.30 (#592), `quick-xml` 0.40.0 → 0.40.1 (#579), `cbc` 0.2.0 → 0.2.1 (pulls `cipher` 0.5.2 / `crypto-common` 0.2.2, #577). **Go**: `ebitengine/purego` 0.10.0 → 0.10.1 (#593). **Ruby**: `rubocop-rspec` 2.20 → 3.9 (#581). **CI actions**: `actions/setup-java` v4 → v5.2.0 (#585), `github/codeql-action` v3 4.35.5 → 4.36.0 (#582), `taiki-e/install-action` 2.79.2 → 2.79.12 (#580), `EmbarkStudios/cargo-deny-action` 2.0.18 → 2.0.20 (#584), `golangci/golangci-lint-action` 9.2.0 → 9.2.1 (#583).

## [0.3.57] - 2026-05-30

> Community contributions + extraction-quality sweep — separation plates, OCG ink filtering, two-phase images, rendered-advance metrics, plus multi-column reading order, page-rotation, CJK/UTF-8 CMap decoding, RTL logical order, indirect-ref page boxes, and font-cache correctness

### Added

- **`TextChar::rendered_advance`** — per-glyph cursor advance to the next character's origin, including character spacing (Tc) and word spacing (Tw) per the PDF Tx formula, distinct from the shape-only `advance_width`. Enables accurate word-boundary detection and cursor reconstruction. Thanks @haberman. (#602)
- **Separation plate rendering** — `render_separations(page, dpi)` / `render_separation(page, ink_name, dpi)` (Rust + Python) emit one grayscale image per ink, pixel value = ink coverage (0 = none, 255 = full tint). Routes DeviceCMYK / Separation / DeviceN content per ISO 32000-1 §8.6 and honours the reserved colorant names `/All` and `/None` per §8.6.6.4 so registration / crop marks land on every plate. New `SeparationPlate` namedtuple in Python. Thanks @RayVR. (#605)
- **OCG (Optional Content Group) ink filtering for text extraction** — `extract_text_filtered(page, excluded_layers, excluded_inks)` and the Python equivalent route through the full text-assembly pipeline (structure-tree ordering, table detection) while filtering by PDF layer and DeviceN/Separation ink. Handles OCMD membership dictionaries and DeviceN all-or-nothing ink semantics. Thanks @RayVR. (#600)
- **`page_image_handles()` two-phase image API** — enumerate image handles on a page first, then materialize pixels on demand, including images nested inside Form XObjects via recursion. Avoids decoding every image up front. Thanks @kh3rld. (#588)
- **Optional Content Group (PDF "layer") name on extracted paths** — `PathContent` gains a `layer: Option<String>` carrying the human-readable OCG name from the surrounding `BDC /OC … EMC` markers (e.g. `A-GRID`, `S-COLS` from Revit/AutoCAD exports), surfaced in the Python path dict too. Resolves OCMD membership dictionaries via `/OCGs` (§8.11.3.2, depth-bounded), decodes names through PDFDocEncoding/UTF-16, and honours Form-XObject-scoped `/Resources/Properties` with leak-isolation across XObject boundaries (§14.6.2, §8.10.1). Thanks @willywg. (#587)
- **olmOCR-bench regression harness** — `tools/benchmark-harness/olmocr/` runs the public `allenai/olmOCR-bench` corpus (999 single-page PDFs, checkable substring/order/absent assertions) for CI regression tracking. Corpus fetched on demand (gitignored, not vendored). (#567)
- **Configurable non-text drop heuristics** — `NonTextDetector` thresholds (`non_ascii_drop_threshold`, `drop_suspicious_unicode`) are now configurable so callers can tune the markdown garbage-glyph filter rather than relying on hard-coded constants. (PDX-7)

### Changed

- **`TextChar` gained a required `rendered_advance` field** — external callers constructing `TextChar { .. }` literals must add `rendered_advance` (set it equal to `advance_width` to preserve prior behaviour). (#602)
- **Documented the three plain-text APIs** — `extract_text`, `to_plain_text`, and markdown-strip now carry guidance on when each is the right choice and why their output differs, so callers stop picking the wrong mode per-PDF. (#554)

### Fixed

- **Hebrew and Arabic text now extracts in correct reading order (#557)** — right-to-left runs were emitted in visual (reversed) order; they now read in logical order in plain-text, Markdown/HTML, and tagged (structure-tree) extraction alike. Previously a tagged Hebrew document such as `אבג דהו` came out reversed. Latin text is never reordered.
- **Two-column references and bibliographies are read column-by-column (#549, #536, #607)** — pages whose left and right columns share the same line baselines were read straight across, interleaving the two columns line by line (`…genetic exchange Kashtan, N., … divergence in prokaryotes reveals…`). They now read down the left column, then the right. Validated across the corpus: 15 academic pages jumped to ~0.98–0.99 similarity vs pdftotext + PyMuPDF, with no regression to tables.
- **Chinese / Japanese / Korean text in UTF-8 CMap fonts is now extracted (#610)** — Type0 fonts encoded with a UTF-8 CMap (`Uni-Utf8-H` and the Adobe `UniGB-/UniCNS-/UniJIS-/UniKS-UTF8-H` family) previously returned **no text at all**; their 1–4-byte codes are now decoded correctly, recovering Latin and CJK including rare 4-byte ideographs.
- **Non-embedded Japanese (JIS) fonts no longer produce garbled Latin** — text using the bare predefined `H`/`V` CMaps with an Adobe-Japan1 collection (e.g. `あいうえお`) was emitted as nonsense ASCII; it now decodes to the correct kana/kanji.
- **Pages with indirect-reference page boxes no longer come back empty** — when a page's `/MediaBox` or `/CropBox` stored its coordinates as indirect references (`/MediaBox [4 0 R 5 0 R 6 0 R 7 0 R]`, ISO 32000-1 §7.3.10) the page collapsed to zero area and dropped **all** text; the references are now resolved per element.
- **180°-rotated pages read in the right order** — a page with `/Rotate 180` was extracted in unrotated coordinates, so its lines and words came out fully reversed (a rotated English agreement read bottom-up, words backwards). The page geometry is now corrected before reading-order assembly. (90°/270° remain a follow-up.)
- **Signature and form-field text stored only in the widget appearance is recovered** — signed-signature fields and form widgets whose value lives in the `/AP` appearance stream (not a `/V` entry) were dropped from extraction; their visible text is now included.
- **Unchecked checkboxes no longer inject `[ ]` noise** — an unchecked checkbox widget previously emitted a stray `[ ]` marker into the surrounding text; it now contributes nothing.
- **Page numbers and running headers no longer leak into body text (#553)** — a standalone page number or running-header line isolated on its own baseline is no longer spliced into the adjacent paragraph.
- **Glyph corruption between documents that reuse a font name (#597, #598)** — Type 3 fonts (whose glyphs are document-scoped content streams) are no longer shared via the cross-document font cache, and the cache key now includes glyph-width metrics, so two fonts that share a BaseFont name but differ in `/Widths` no longer alias to one another.
- **Type 3 font spacing now honours the font's `FontMatrix` (#606)** — glyph advances for Type 3 fonts were scaled by a hard-coded 1/1000 em; they now apply the font's own `FontMatrix[0]`, so Type 3 fonts with a non-standard (e.g. identity `[1 0 0 1 0 0]`) matrix get correct character and word spacing. Thanks @haberman.
- **Faster, no double rescan on damaged PDFs (#572)** — a reconstructed cross-reference table now seeds the object-scan cache, removing a redundant second full-file sweep on corrupt/polyglot PDFs.
- **Form XObject image cache poisoning when fonts/XObjects collide on basename** — the OCG ink-filtering work also fixed three latent bugs in OCG/ink handling: a parser edge case, a Form XObject cache keyed too coarsely, and ink-state restore on graphics-state pop. Thanks @RayVR. (#600)

### Performance

- **`extract_text` no longer hangs on heavily OCR-layered scans (#575)** — superscript-baseline snapping was quadratic in the number of text spans; it is now windowed, so pages with tens of thousands of OCR spans extract promptly instead of stalling.
- **Regression guards added** for previously-fixed word-spacing, character-clustering scaling, `to_html` table handling, and multi-column detection, so they cannot silently regress.

## [0.3.56] - 2026-05-28

> Text-extraction fidelity sweep — XY-cut routing, typed extraction status, OCR API repair, Persian font support, encryption authentication enforcement

### Added

- **`ExtractionSignal` enum + `Warning` / `WarningSink` types** (`src/extractors/status.rs`, `src/extractors/warnings.rs`) — typed signal surface so callers can distinguish "no text" from "extraction failure" (`Ok` / `Truncated{at_op}` / `NoTextLayer` / `UnmappedGlyphs{count}` / `OcrUnavailable{reason}` / `PasswordRequired` / `Multiple`). Foundation for `flatten_warnings()` and per-call status accessors.
- **`PdfPermissions` accessor** (`src/encryption/permissions.rs`) — decodes `/P` flags per PDF spec §7.6.3.2 Table 22; surfaces `print`, `modify`, `copy`, `annotate`, `form_fill`, `extract_for_accessibility`, `assemble`, `print_high_quality`. Closes #562 API gap.
- **`PdfDocument::has_text_layer(page) -> Result<bool>`** — predicate wrapping `page_cannot_have_text` + `may_contain_text` so callers know whether to fall back to OCR. Closes #563.
- **`PdfDocument::extract_text_ocr_only(page, engine, options)`** — additive companion that always invokes OCR (no text-layer peek). Closes #574.
- **`set_max_ops_per_stream(Option<usize>)`** (`src/content/parser.rs`) — global `AtomicUsize` override for the 1,000,000-operator cap; all 6 runtime check-sites gated through `effective_max_operators()`. Closes #559.
- **`set_preserve_unmapped_glyphs(bool)`** (`src/extractors/text.rs`) — global atomic + 8 filter-site gating so U+FFFD glyphs from glyph-name-only fonts (MSAM10, dingbats) can be preserved for downstream repair. Closes #571.
- **`assemble_text_via_reading_order(spans)`** helper + Python wrappers + 4 per-class detector predicates (`DenseSingleLine`, `SubSuperBaselineReattach`, `NarrowTrackedJustified`, `DramaticScript`) in `src/pipeline/reading_order/detectors.rs`. The detectors return a `ReadingOrderClass` callers can route on. Bench-visible improvements on the same issue cluster (#549/#556/#561/#565/#568/#576) come from the parallel `XYCutStrategy` work documented under **Changed** — full detector-driven assembly is a follow-up.
- **`ExtractionProfile::TJ_HEAVY`** (`src/config/extraction_profiles.rs`) — opt-in profile for TJ-kerned word-boundary recovery (threshold -100 vs CONSERVATIVE -120). Closes #564.
- **Adobe-Persian-1 / Adobe-Arabic-1 stub CMaps** (`src/fonts/cid_mappings/adobe_arabic.rs`) + DescendantFonts inline-dict parse path in `src/fonts/font_dict.rs`. Closes #566.
- **`PyPageCount` dual-shape** — Python `doc.page_count` works as both attribute and method (`__call__` + `__index__`) so `range(doc.page_count)` works again post-v0.3.54 regression. Closes #550.
- **`PdfDocument::structured_warnings()` accessor** + `WarningSink` wired as the per-document field type, with 5 `log::warn!` migration sites pushing into the global sink. Per-document and global drains merge on each call. Closes #558 (second half).

### Changed

- **`extract_text` / `extract_text_auto` / `extract_page_auto` no longer panic** when ONNX Runtime fails to dlopen — `std::panic::catch_unwind` around the full `ort::Session::builder()` chain converts the dylib-load panic to a typed `OcrError::ModelLoadError`. Closes #569, #573.
- **`extract_text_ocr` honours its contract** — was previously a passthrough that returned the empty text layer of scanned PDFs; now invokes the supplied engine when `text_layer_empty` is true. Closes #574.
- **`get_form_fields` walks parent fields with `/T` even without `/FT`** — matches pypdf's traversal of IRS AcroForm logical-group dictionaries (~15-30% of checkbox `/Off` values were silently dropped). Closes #570.
- **Default Python logger no longer captures `pdf_oxide.{parser,content,fonts,document}` WARN records** — at import, `_setup_default_log_levels` attaches a `NullHandler` and sets `propagate = False` on those four high-frequency targets (standard library convention per PEP 282); records stop at the pdf_oxide logger boundary instead of bubbling to root's default stderr handler. Callers re-enable bubbling via `logger.propagate = True` or read the typed feed via `doc.structured_warnings()`. Closes #558 (first half).
- **Span-boundary spacing tightened** — bucket-1/2/3/4 fixes: span `bbox.x` matches first char after TJ word boundary; font-transition with small positive gap inserts a space; super/sub digit runs substituted to U+2070–2089/U+00B2/B3/B9; spacing diacritics (U+00B4, U+0060) folded into following base letter via NFC.
- **`merge_adjacent_spans` joins same-font multi-char small-caps / drop-cap runs** so "SUBTITLE A—OFFICE OF THE" is no longer split mid-word; bench small-caps clusters now read cleanly. Closes #555 root-cause path; complements #560.
- **Super/subscript glyphs snap onto base baseline pre-sort** so arxiv affiliation markers ("name¹,²") stay inline with the author run instead of being lifted into a stray top-of-page band. Improves bench by lifting 6 papers in the py-pdf 14-PDF set.
- **AcroForm widget text-capacity bound** — `truncate_to_widget_capacity` caps `/V` to `0.0175 * w * h + 64` chars so scrollable fields with embedded Lorem-ipsum (pdfbox AcroFormsBasicFields) no longer dump 90 KB of phantom text per widget.
- **Forward-scan CTM tracker skips inline image data (BI…EI)** so the marker bytes inside JPX/JBIG2 streams stop being parsed as text operators (2201.00151 p2: 758 of 789 garbage spans → 0).
- **XY-cut `find_horizontal_split_indexed` rejects sliver sub-splits** (`MIN_RESULT_WIDTH_PT = 60.0 pt`) so 8 pt monospaced columns no longer fragment into 48 pt strips.
- **Narrow-gutter prose detector** (`detect_narrow_gutter_prose`) uses gap-position clustering to recover 2-column body text where the legacy detector saw outlier singleton clusters and bailed — fixes arXiv 2201.00151-class column interleave. Bench: py-pdf 14-PDF mean 89.4% → **90.2%**.

### Fixed

- **#549** — `extract_text` and `to_plain_text` now route through XY-cut on multi-column / dramatic-script layouts (parity with `to_markdown_all`). Largest single bench improvement of the cycle.
- **#551** — Latin ligatures (fi/fl/ff/ffi) preserved at `should_insert_space` threshold + post-processing repair pass for the merged-cluster cases not visible at extractor time.
- **#552** — Combining diacritics (`´`, `` ` ``, `^`, etc.) NFC-composed into the following base letter at `apply_combining_mark_composition` instead of being emitted as a separate token before the letter.
- **#555** — Missing space at run/font boundary repaired at `should_insert_space` (root-cause) + `repair_run_boundary_space` (post-processing for case-change boundaries like `theEditor`).
- **#556** — Figure-region math glyphs (subscripts/superscripts) no longer interleave captions; same reading-order plumbing as #549.
- **#558** — `SPEC VIOLATION` / xref-reconstruction / font-warn log noise gated behind opt-in `setup_logging()`; structured `flatten_warnings()` accessor available for programmatic consumers.
- **#560** — Monospace code blocks no longer emit intra-token whitespace at every glyph boundary; `is_monospace_font` helper bumps the space-insertion threshold from 0.5× to 1.2× space-width.
- **#561** — Subscript/superscript glyphs no longer reordered into wrong inline position; `SubSuperBaselineReattach` detector + pre-sort baseline snap.
- **#562** — `password_protected.pdf` and `copy_protected.pdf` no longer extract text without authentication; `permissions()` accessor + existing `require_authenticated()` guard verified end-to-end.
- **#563** — Image-only / rasterized PDFs surface `has_text_layer() == false` so callers can dispatch to OCR instead of receiving silent empty strings.
- **#564** — Word boundaries no longer lost on TJ-kerned text; opt-in `ExtractionProfile::TJ_HEAVY` calibration.
- **#565** — Narrow-tracked / justified column intra-word spaces eliminated via per-line median-gap normalisation in `NarrowTrackedJustified` detector.
- **#566** — Persian/Farsi Type0 fonts (NazaninNormal, YagutBold) without bundled CMap now resolve via Adobe-Persian-1 / Adobe-Arabic-1 stub lookup; DescendantFonts inline-dict parse path accepted.
- **#568** — Dense 8 pt body lines no longer split into two interleaved character streams; `DenseSingleLine` detector + min-result-width filter.
- **#576** — Dramatic-script layouts (centered speaker tags + indented dialogue) read in performance order; `DramaticScript` detector.

### Security

- **#562** — Verified no text extraction path bypasses authentication on encrypted PDFs. `EncryptionHandler::raw_permissions()` accessor for cross-binding `/P` consumption; spec-aligned audit at `docs/releases/issues/password-bypass-audit.md`.

### Deprecated

- `PdfDocument.page_count()` method-call form (Python binding) — supported but deprecated; use `doc.page_count` attribute form instead. Removal scheduled for v0.4.0 (#414). Closes #550.

## [0.3.55] - 2026-05-25

> Ruby + PHP language bindings + multi-line heading reading-order fix

### Added
- **Ruby binding (9th language)** — full PDF toolkit for Ruby. Install
  with `gem install pdf_oxide`. Prebuilt native gems for linux-x86_64,
  linux-aarch64, darwin-x86_64, darwin-arm64, windows-x64 plus a
  source gem. API mirrors the Java binding's 9-class shape
  (`PdfDocument`, `Pdf`, `PdfPage`, `PdfPolicy`, `PdfSigner`,
  `PdfValidator`, `DocumentEditor`, `AutoExtractor`,
  `MarkdownConverter`) so docs and examples are uniform across
  languages. Full feature parity with Python / Java including
  auto-extraction, PAdES B/T/LT signing, destructive redaction,
  office round-trip, and split-by-bookmarks. (#545)
- **PHP binding (10th language)** — full PDF toolkit for PHP. Install
  with `composer require oxide/pdf-oxide`. PHP 8.2 / 8.3 / 8.4 / 8.5
  on Linux / macOS / Windows. Same 9-class API shape as Ruby and
  Java. Composer post-install hook fetches the matching prebuilt
  `libpdf_oxide` per platform with SHA-256 verification. Full feature
  parity with Python / Java. Requires `ext-ffi`. (#546)

### Fixed
- **#543** — Long subsection titles in multi-column academic papers
  no longer split when they wrap across column boundaries.
  Discovered while regression-testing v0.3.54 on a 75-PDF corpus.
- **#537** *(follow-up)* — Markdown output now emits Unicode
  bidi-isolation markers around RTL runs detected by the v0.3.54
  Hebrew / RTL detector, so extracted Hebrew / Arabic text renders
  correctly inside mixed-direction paragraphs. Original report by
  **alexagr**; this completes the markdown emission half deferred
  from v0.3.54.
- **#535** *(follow-up)* — Type 1 built-in encodings, CFF charset,
  and `/Differences`-array glyph lookups now go through the same
  Adobe Glyph List fallback chain as v0.3.54's main extractor, with
  variant-suffix stripping (`A.sc` → `A`, `bullet.alt` → `•`,
  `fi.001` → `ﬁ`). Resolves replacement-character (`�`) leakage on
  PDFs using simple-font encodings without a ToUnicode CMap.

## [0.3.54] - 2026-05-23

> Text-extraction fidelity pass. Hebrew / RTL reads in correct
> reading order thanks to a new geometric visual-vs-logical
> detector that finally fires on Hebrew (Pass 0's
> Arabic-Presentation-Forms gate never matched Hebrew because
> Hebrew has no presentation forms in Unicode). Bullets and
> `fi` / `fl` ligatures decode to their canonical codepoints
> via a new ISO 32000-1 §9.10.2 fallback level — Type0
> Identity-H fonts without a `CIDToGIDMap` now consult the
> embedded TrueType `cmap` + `post` tables before the
> CID-as-Unicode last resort. Tight two-column prose bodies
> read column-by-column instead of row-by-row interleave (the
> known v0.3.53 issue), gated by a new table-vs-prose
> classifier so the v0.3.53 corpus-revert lesson holds —
> tables stay byte-identical. Two-column reference PDFs
> (Bibles / dictionaries / encyclopedias) inherit the same
> reading-order fix. Four fixes; zero new public surface.

### Fixed

- **Hebrew / RTL visual-vs-logical detection
  ([#537](https://github.com/yfedoseev/pdf_oxide/issues/537))**
  — Hebrew PDFs that store text in visual order (the PDF
  content stream draws glyphs left-to-right even though the
  script reads right-to-left) now extract in correct logical
  order. New per-RTL-run X-coordinate-monotonicity detector
  gates the existing UAX #9 `bidi::reorder_visual_to_logical`
  pass; logical-order PDFs (the pdfium `hebrew_mirrored.pdf`
  test fixture and Arabic CID-TrueType samples) stay
  byte-identical. Pass 0's Arabic-Presentation-Forms trigger
  was the only Arabic-specific gate; Hebrew has no
  presentation forms in Unicode, so the geometric gate is
  what unblocks it. Reported by **alexagr** with the Magic
  Palace Eilat hotel PDF as the canonical case; reproduced
  empirically before the fix landed.

- **ToUnicode CMap miss: extended ISO 32000-1 §9.10.2
  fallback chain
  ([#535](https://github.com/yfedoseev/pdf_oxide/issues/535))**
  — Type0 Identity-H fonts without a `CIDToGIDMap` (the
  PowerPoint / Acrobat-exported subset-font shape that
  produced ~25,350 "ToUnicode CMap MISS" warnings on the
  PowerPoint corpus) now consult the embedded TrueType
  font's `cmap` + `post` tables to recover glyph names,
  then look up the Adobe Glyph List (including the AGL §6
  `uniXXXX` / `uXXXXX` synthetic-name patterns). Resolves
  the bullet → `❍` substitution and the `fi` / `fl`
  ligature corruption. The post-extraction band-aid
  `normalize_bullet_glyphs` stays retired (it was wrong for
  documents that legitimately use `❍`); the fix is in the
  decoder, where it belongs. Same shape pdf.js / MuPDF /
  PDFBox 3.x use.

- **Multi-column prose reading order
  ([#534](https://github.com/yfedoseev/pdf_oxide/issues/534))**
  — tight two-column PROSE bodies (~12pt gutters) now read
  column-by-column instead of row-by-row interleave. The
  v0.3.53 known issue, explicitly documented in that
  release's notes. New table-vs-prose classifier (region
  is "prose" when lines are tall stacks of wide content,
  "table" when lines are short cells; the v0.3.53 revert
  comment in `xycut.rs:73-101` is the spec) gates a
  tight-gutter column cut so the same XY-cut recursion no
  longer corrupts table cells — the v0.3.53 lesson where two
  prior attempts were reverted by the 70-PDF corpus sweep
  after the `google_doc_document.pdf` population table
  ("273.879.7501" → "1273.879.750") regressed. Adds
  recursive band-separation of full-width header / footer
  rows BEFORE column detection so the multi-column body
  gets analyzed as its own region. Same 70-PDF sweep is the
  acceptance gate.

- **Two-column reference PDFs: column-aware reading order
  ([#536](https://github.com/yfedoseev/pdf_oxide/issues/536))**
  — reference-style two-column PDFs (Bibles, dictionaries,
  encyclopedias, academic editions with marginal section
  numbers) now extract column-by-column thanks to the #534
  classifier. The cascade where the spatial-table-detector
  latched onto the interleaved structure and rendered the
  whole multi-column body as a Markdown table-with-
  one-word-per-cell is resolved: clean column reading order
  produces clean prose, not a degenerate table. Concerns
  about marginal-label classification, structural heading
  isolation, cross-page section continuity, and
  header / footer separation that the reporter usefully
  called out are captured for the v0.4.0
  structured-extraction track and acknowledged on the
  issue. Reported by **lggcs (Luke)** with a 1256-page
  French Louis Segond Bible as the canonical case;
  reproduced empirically before the fix landed.

## [0.3.53] - 2026-05-22

> Java is the 8th binding, plus a markdown-extraction quality pass
> and OCR parity across every prebuilt. Native Maven-Central
> artifact on jni-rs 0.22 (JDK 11+, five-arch fat JAR), full v0.3.52
> surface parity across text / markdown / AutoExtractor / forms /
> render / PAdES B-B+B-T+B-LT / destructive redaction /
> split-by-bookmarks / compliance / crypto-policy. Free Kotlin
> interop via the same JAR. Published Python wheels and the Java JAR
> now ship OCR (parity with Node / Go / C#). Markdown extraction
> fixes: table-cell bold/italic preserved, CamelCase brand names no
> longer split, spatial cell words no longer fragment into columns,
> centered titles read in order. The May-2026 language promise
> ([README:3](README.md)) lands.

### Added

- **Java binding (`fyi.oxide:pdf-oxide:0.3.53`, [#NNN](https://github.com/yfedoseev/pdf_oxide/issues/NNN))**
  — native JNI binding to pdf_oxide via jni-rs 0.22 with the same
  Rust core the existing seven bindings sit on. Maven Central
  publish via `central-publishing-maven-plugin` 0.9.0 under groupId
  `fyi.oxide` (matching the `pdf.oxide.fyi` brand), Java package
  `fyi.oxide.pdf.*`. **JDK 11 LTS floor** — broadest enterprise
  reach, Polars/Lance/RocksDB precedent (not kreuzberg-style
  FFM+Java 25 which excludes the JDK 17/21 majority). Five native
  arches embedded in the published fat JAR (linux x86_64, linux
  aarch64, macOS x86_64, macOS aarch64, windows x86_64). 52 JNI
  symbols across 9 wired classes; 82 JUnit tests green.

- **`PdfDocument`** — `open(Path/byte[]/InputStream/String)`,
  `open(Path, String password)` + bytes variant, `authenticate`,
  `pageCount`, `extractText(int)`, `extractTextAuto(int)` (v0.3.51
  graceful auto-routing), `render(int)` + DPI overload (PNG bytes),
  `producer`/`creator` Info dict, `formFields()`,
  `search(query, caseInsensitive, regex, maxResults)`,
  `toMarkdown`/`toHtml` convenience, `page(int)` /
  `pages()` / `pagesStream()`. `AutoCloseable` with idempotent
  `close()` (shared `AtomicLong` + Cleaner backstop — multi-class-
  loader safe).

- **`PdfPage`** — `mediaBox` / `cropBox`, `width` / `height`,
  `rotation`, `text()`, `text(BBox region)`, `words()`, `lines()`
  (nested `List<TextWord>` per line), `chars()`, `images()`
  (`ExtractedImage` with bytes + format enum + bbox + dimensions),
  `tables()` (flat `List<TableCell>` with row/col indices + spans),
  `annotations()` (13-subtype enum + URI extraction for Link).

- **`MarkdownConverter`** — `toMarkdown(doc)` /
  `toMarkdown(doc, page)` / `toHtml(doc)` / `toHtml(doc, page)`.

- **`Pdf`** — `fromMarkdown(String)` / `fromHtml(String)` /
  `fromImages(List<byte[]>)` (auto-detects JPEG/PNG), `save()` /
  `saveTo(Path)`, `planSplitByBookmarksCount(byte[], int)`,
  `splitByBookmarksFromBytes(byte[], int) -> byte[][]` (v0.3.50
  #482 — round-trip proven: outlined PDF → segments → each
  reopenable).

- **`DocumentEditor`** — `open(Path/byte[]/String)`,
  `setFormField(name, String/boolean)`, `addRedaction(page, BBox)`,
  `redactionCount(page)`, `applyRedactionsDestructive()` (v0.3.50
  #231 — full Phase 3 T11 pipeline; default `RedactionOptions`
  scrub metadata + strip JS + remove embedded files + hide OCG;
  fail-closed on composite/Type0/unknown fonts), `scrubMetadata()`,
  `save()` / `saveTo(Path)`.

- **`AutoExtractor`** (v0.3.51 #517) — `of(doc)` /
  `fast(doc)` / `balanced(doc)` / `highFidelity(doc)` presets,
  `classifyPageKind(int)` / `classifyDocumentKinds()` (returns
  per-page `PageClass` enum), `extractText()` /
  `extractTextForPage(int)` (graceful OCR fallback), `extractAutoPage(int)`
  / `extractAutoDocument()` (simplified `AutoResult`), and the
  rich-shape escape hatch **`extractPageJson(int)` /
  `extractDocumentJson()`** returning serde-JSON of the full
  v0.3.51 `PageExtraction` / `DocumentExtraction` (typed reasons +
  per-region bboxes + confidence + ocr_used + pages_needing_ocr).

- **`PdfSigner`** (v0.3.50 #235) — `fromPkcs12(Path/byte[], String)`,
  `sign(byte[] pdf, SignOptions opts)` supporting PAdES **B-B**
  (no TSA needed), **B-T** and **B-LT** (RFC 3161 TSA HTTP via the
  `tsa-client` Cargo feature; `opts.tsaUrl()` required for B-T/B-LT),
  `verify(byte[])`, `classifyLevel(byte[])` (static — returns highest
  PAdES level present in a signed PDF without needing key material).

- **`PdfValidator`** — `isPdfA(doc, PdfALevel)` /
  `isPdfUa(doc, PdfUaLevel)` (simplified boolean verdict);
  `validatePdfA` / `validatePdfUa` return `ValidationResult`. PDF/A
  levels 1a/1b/2a/2b/2u/3a/3b/3u supported; PDF/A-4 + PDF/UA-2
  surface as `PdfUnsupportedException` (pdf_oxide core gaps).

- **`PdfPolicy`** (v0.3.50 #230) — `current()` / `set(PolicyMode)`
  + `compat/strict/fipsStrict` presets. **Set-once enforced** at
  process startup per the v0.3.50 design (second `set` throws with
  a clear `"already set"` message).

- **Exception taxonomy** — `PdfException extends RuntimeException`
  (unchecked, modern Java consensus per Effective Java Item 71) +
  8 typed subclasses (`PdfParseException`, `PdfEncryptedException`,
  `PdfPermissionException`, `PdfIoException`,
  `PdfOcrUnavailableException`, `PdfSignatureException`,
  `PdfInvalidStateException`, `PdfUnsupportedException`) +
  `PdfErrorKind` enum for switch-on-enum dispatch. Rust `Error::*`
  variants mapped 1:1 in `pdf_oxide_jni/src/error.rs`.

- **Value types** — `geometry.{BBox, Point, Rect, Color}`,
  `text.{TextStyle, TextWord, TextLine, TextChar, TextSpan}`,
  `table.{Table, TableCell}`, `image.{ImageFormat, ExtractedImage}`,
  `form.{FormField, FormFieldType}`,
  `auto.{ExtractMode, ExtractReason, PageClass, RegionResult,
  AutoResult, ClassifyResult, AutoExtractConfig + Builder}`,
  `compliance.{PdfALevel, PdfXLevel, PdfUaLevel, ValidationResult,
  ValidationViolation}`,
  `signature.{SignatureLevel, SignOptions + Builder}`,
  `policy.{PolicyMode, SecurityPolicy + Builder}`,
  `render.PixelFormat`, `redaction.RedactResult`,
  `split.{SplitByBookmarksOptions + Builder, BookmarkSegment}`,
  `metadata.{DocumentInfo, XmpMetadata}`,
  `search.{SearchOptions + Builder, SearchMatch, SearchResult}`,
  `annotation.{Annotation, AnnotationType}`. JDK 11 floor → final
  classes with manual `equals`/`hashCode`/`toString` and
  record-shaped accessor names (drop-in `record` migration when
  floor moves to 17+). JSpecify `@Nullable` annotations throughout.

- **`NativeLoader`** — multi-classloader-safe UUID-suffixed temp
  extraction (snappy-java pattern, avoids the Tomcat/OSGi
  `UnsatisfiedLinkError` trap from FLINK-5408). Honors
  `-Dfyi.oxide.pdf.lib.path` / `-Dfyi.oxide.pdf.use.systemlib` /
  `-Dfyi.oxide.pdf.tempdir` overrides for FIPS / locked-down
  `/tmp` / read-only-rootfs deployments.

### Fixed

- **OCR now ships in the published Python wheels and Java JAR** — CI
  test builds compiled OCR (`--features python,ocr,barcodes`) but the
  released wheels used `--features python`, so PyPI users got a wheel
  without OCR even though CI exercised it. Both glibc and musl Python
  wheels, and the Java JNI fat JAR, now build with OCR for parity with
  the Node / Go / C# prebuilts. FIPS variants deliberately exclude OCR
  (no ONNX in FIPS deployments).

- **Markdown table cells preserve bold/italic** — the tagged-PDF table
  extractor built `TableCell`s from joined text only, discarding the
  per-span font weight/style, so `**bold**` / `*italic*` inside table
  cells was lost on the way out. Cells now carry their span styles
  end-to-end (`table_extractor` populates `cell.spans`).

- **Words no longer split mid-word by phantom spacing** — words whose
  glyph runs are positioned edge-to-edge (common in presentation
  exports) could be emitted with a spurious internal space when the
  source font lacked a `/Widths` array. Per ISO 32000-1 §9.4.4,
  inter-glyph spacing is the displacement between glyph origins; the
  fallback-width correction that compensates for missing width metrics
  now applies only when glyph boxes actually overlap, never to
  cleanly-adjacent glyphs. Legitimate word spacing — including after a
  token that ends in a capital letter — is preserved.

- **Spatially-positioned cell words no longer fragment into columns** —
  a single table cell whose words are laid out with wide gaps was split
  into one column per word. A row-coverage filter drops phantom columns
  present in too few rows, gated so it only refines an already-detected
  table and never fabricates one from prose.

- **Prose pages no longer mis-detected as tables** — a single-column
  page whose wrapped paragraph lines' inter-word gaps coincidentally
  aligned could be emitted as a fragmented table. A prose gate rejects a
  spatially-detected (no-rulings) table when a row crosses a sentence
  boundary, a structure genuine data tables do not exhibit. Ruled and
  tagged tables are unaffected.

- **Centered titles read in document order** — a centered multi-word
  title plus subtitle/byline was misread as multiple columns,
  scrambling the heading. A centered-block guard (scattered leftmost
  edges, small block) keeps such blocks as a single column.

- **Fewer fragmented headings** — runs of same-level heading fragments
  (PowerPoint word-per-heading exports, wrapped headings) are merged
  when the run is unambiguous; KPI numeric-only heading runs collapse
  to a list.

- **Stray pipe characters escaped** — a `|` outside a markdown table
  block is escaped so downstream renderers do not misread it as a
  malformed table row.

- **Content-preservation policy for markdown post-processing** — the
  post-process pass never drops or rewrites legitimate text. Earlier
  band-aids that filtered "Page N" lines, rewrote bullet-glyph
  codepoints, flattened sparse-but-real tables, or deduped repeated
  content were removed after a 70-PDF baseline-vs-HEAD regression sweep
  proved they damaged real documents; the correct upstream fixes are
  tracked as follow-ups.

### Known issues

- Tight two-column **prose** bodies can still interleave row-by-row in
  reading order
  ([#534](https://github.com/yfedoseev/pdf_oxide/issues/534)). A safe
  fix needs a table-vs-prose classifier so it does not regress
  table-cell ordering; two threshold/structural attempts were reverted
  after the regression sweep caught table-data corruption.

- Bullet and ligature glyphs in fonts with no usable `/ToUnicode` CMap
  can decode to an incorrect code point or be dropped
  ([#535](https://github.com/yfedoseev/pdf_oxide/issues/535)). The fix
  is a §9.10 decode fallback (glyph-name / encoding) in the font layer,
  not a markdown-layer code-point rewrite (which was removed as content
  corruption — see the content-preservation note above).

### CI / Release

- **`.github/workflows/ci.yml`** — new `build-lib` variant
  `java-jni` builds the JNI cdylib with `--features rendering,
  signatures,tsa-client`. New `java` job (matrix: ubuntu × JDK
  {11, 17, 21}) downloads the native, stages into the Maven
  resource path, runs `mvn compile/test/package`, validates JAR
  contents + manifest, uploads the JAR artifact. New `java-lint`
  job runs the Java code-quality gates — Spotless
  (palantir-java-format) formatting check and SpotBugs static
  analysis — bringing the Java binding to parity with the
  format+lint gates the other bindings already enforce (rustfmt +
  clippy / gofmt + golangci-lint / Biome / dotnet-format / ruff).

- **`.github/workflows/ci-fips.yml`** — new `fips-java` job
  (ubuntu + macOS) builds `pdf_oxide_jni` with `--no-default-features
  --features fips,signatures` and runs the full JUnit suite against
  the FIPS-compiled cdylib. Validates the `legacy-crypto` exclusion
  holds end-to-end.

- **`.github/workflows/release.yml`** — new `build-java-native`
  matrix (5 arches: linux x86_64/aarch64, macOS x86_64/aarch64,
  windows x86_64) cross-compiles the JNI cdylib per target with
  `ocr,rendering,signatures,barcodes,tsa-client` (OCR-enabled parity
  with the Node/Go/C# native cdylib; `system-fonts` arrives
  transitively via `rendering`). New
  `package-java-jar` job assembles the fat JAR (all 5 natives
  embedded). New `publish-maven` job uploads to Maven Central via
  `central-publishing-maven-plugin` with `autoPublish=false` per
  `feedback_release_gate` — the upload reaches `VALIDATED` state and
  the maintainer flips Publish from the Central Portal UI. Python
  wheel jobs (glibc + musl) build `--features python,ocr,barcodes`
  so the published wheels ship OCR. `validate` job extended to
  enforce `java/pom.xml` version matches Cargo workspace.

- **`pdf_oxide_jni`** — new workspace member crate (`crate-type =
  ["cdylib", "rlib"]`; jni 0.22; feature-mirrored `ocr` /
  `signatures` / `tsa-client` / `rendering` / `barcodes` / `full`
  / `fips` / `legacy-crypto`; not published to crates.io — the
  consumable artifact is the Maven Central jar).

### Thanks

<!-- TBD on issue close + Suleman-Elahi / other reporters -->

## [0.3.52] - 2026-05-18

> Out-of-the-box OCR for the Node.js, Go and C# prebuilts, a Node
> worker-teardown fix that silenced a spurious exit warning, an OCR
> detection-unclip fix that restores recognition on wide text lines
> (native and WASM bindings alike), a Markdown→PDF styling fix that
> restores headings, bold/italic and monospace, strict CI
> toolchain-drift gating, and a dependency-maintenance batch.

### Added

- **OCR in the prebuilt native library for Node.js, Go and C#
  ([#520](https://github.com/yfedoseev/pdf_oxide/issues/520))** — the
  `build-native-libs` release job now compiles the shared library with
  the `ocr` feature (alongside
  `rendering,signatures,barcodes,tsa-client,system-fonts`), so auto-mode
  OCR / `extractTextAuto` works straight from the published Node, Go and
  C# packages with **no `--build-from-source`**. `docs/OCR_GUIDE.md`
  gains an "OCR Support by Binding" matrix and a pure-npm Node recipe
  (`npm install pdf-oxide onnxruntime-node`, `ORT_DYLIB_PATH` via
  `require.resolve(...)`, `prefetchModels` + `extractTextAuto`). The
  default `pdf-oxide-wasm` still ships without OCR; see the opt-in
  `wasm-ocr` build below for the experimental WASM OCR path.

- **WASM OCR backend (#524, experimental, opt-in)** — pure-Rust
  `tract` inference under a new `wasm-ocr` build feature
  ([`ocr-tract`](https://github.com/yfedoseev/pdf_oxide/issues/524) is
  the bare backend; `wasm-ml` is retained as a thin back-compat
  alias). The default `pdf-oxide-wasm` package is **unchanged** and
  still ships without OCR; only `wasm-ocr` builds include it. The
  pure-Rust path is **output-equivalent to the native `ort`
  backend** — verified at the inference-engine level (max abs diff
  ≤ 3e-6 on the real PaddleOCR det/rec graphs) and end-to-end
  (byte-identical recognized text on a shared fixture). Cross-target
  (browser / Deno / edge) integration tests and a release `.wasm`
  size gate are pending and will land in #524's own dedicated release
  cycle. See `docs/OCR_GUIDE.md` for the JS fetch+Cache recipe and the
  build command.

### Fixed

- **OCR: wide text lines were misread (detection unclip bug)
  ([#524](https://github.com/yfedoseev/pdf_oxide/issues/524))** — the
  DBNet box-unclip step scaled each corner by a *percent of its own
  dimension* from the centre instead of PaddleOCR's uniform
  `area·ratio/perimeter` polygon offset. On a long, short text line
  that is badly anisotropic: it over-grew the long axis (pushing the
  box's x origin off-image, negative) and barely grew the short axis
  (the box stayed ~one glyph-band tall), so the recognizer received a
  horizontally-shifted, vertically-clipped sliver. A clean single
  line "OCR fidelity test hello world 2024" came out
  "OcR tdenfy test neno woridZoZ4 s" (confidence 0.66). Replaced with
  the standard uniform offset: the same line now reads exactly, at
  0.98 confidence. Affects the **native** OCR path (all bindings), not
  just WASM — the two backends are bit-equivalent. Pinned by new
  postprocessor unit tests and an end-to-end regression guard.

- **Node `prefetchModels()` no longer emits a spurious
  `Worker N exited with code 1`
  ([#521](https://github.com/yfedoseev/pdf_oxide/issues/521))** — the
  `worker_threads` pool is now spawned **lazily** on the first real task
  (importing the library, or calling the synchronous native APIs such as
  `extractText*` / `classifyPage` / `prefetchModels`, spawns zero
  workers); pooled workers are `unref()`'d so an idle pool never keeps
  the event loop alive; and teardown does an async graceful
  `terminate()` on `beforeExit` (deliberately **no** `SIGINT`/`SIGTERM`
  listeners — a library must not change the host's default signal
  semantics) with a synchronous `terminated` flag flipped on the hard
  `exit` so a normal process exit killing an unref'd worker is no longer
  reported as an abnormal exit.

- **Markdown → PDF now renders styling instead of flat body text
  ([#525](https://github.com/yfedoseev/pdf_oxide/issues/525))** —
  `Pdf::from_markdown` (and `from_html`, which funnels through it)
  computed heading sizes but only used them for line spacing, then drew
  every line in a single regular font, and *stripped* `**bold**` /
  `*italic*` markers to plain text. Headings (`#`–`####`) now render in
  the bold face at 2.0/1.5/1.25/1.1× scale; inline `**bold**`,
  `*italic*` and `` `code` `` produce real per-run font switches
  (Helvetica-Bold/-Oblique, Courier) measured and positioned so a line
  stays visually contiguous; fenced code blocks and GFM tables render
  monospace. Two writer-layer bugs that masked this are also fixed:
  `map_font_name` discarded explicit Standard-14 weight/oblique names
  (`Helvetica-Bold` → `Helvetica`) and had no italic path, and the page
  `/Font` resource set registered only 6 of the 12 Latin Standard-14
  faces, so any `Tf` to an oblique/bold-serif face resolved to a missing
  resource and silently fell back to regular. Underscores are no longer
  treated as emphasis, so `snake_case` identifiers survive intact.
  Reported by @Jethril.

### CI / Release

- **Strict toolchain-drift gating
  ([#522](https://github.com/yfedoseev/pdf_oxide/issues/522))** —
  `RUSTFLAGS=-D warnings` and `RUSTDOCFLAGS=-D warnings` are enforced on
  the **stable** matrix leg only, and `continue-on-error` has been
  removed from the beta/nightly legs so upstream-rustc drift surfaces as
  a real signal instead of being silently swallowed. Residual nightly
  `rust-lld` SIGBUS risk is mitigated with `CARGO_BUILD_JOBS=2` and
  documented inline in the workflow.
- **Dependency maintenance** — quick-xml `0.39 → 0.40`
  ([#494](https://github.com/yfedoseev/pdf_oxide/issues/494)) with all
  seven `BytesText::xml_content()` call sites migrated to the
  behaviour-identical zero-arg `xml11_content()` (0.40 added a
  `version: XmlVersion` parameter; 0.39's `xml_content()` was literally
  `self.xml11_content()`, so this is a byte-for-byte no-op);
  tokenizers `0.22 → 0.23`
  ([#498](https://github.com/yfedoseev/pdf_oxide/issues/498));
  weezl `0.1 → 0.2`
  ([#527](https://github.com/yfedoseev/pdf_oxide/pull/527)) — picks
  up the upstream LZW-decoder fix for streams that overwrote initial
  table entries (benefits **pdf_oxide's direct PDF `LZWDecode` filter**
  in `src/decoders/lzw.rs`; the `image` crate's TIFF reader still pins
  weezl 0.1 transitively and is unaffected by this bump); aws-lc-rs
  `1.16.3 → 1.17.0`
  ([#526](https://github.com/yfedoseev/pdf_oxide/pull/526)) — build-
  hygiene only (jitterentropy `CFLAGS` filtering for FreeBSD, MinGW
  Win7 fixes, nightly clippy); and
  SHA-pinned action bumps `actions/upload-artifact → v7.0.1`
  ([#495](https://github.com/yfedoseev/pdf_oxide/issues/495)) with
  `actions/download-artifact → v8.0.1` for compat and
  `astral-sh/setup-uv → v8.1.0`
  ([#502](https://github.com/yfedoseev/pdf_oxide/issues/502)), unified
  across all nine workflows. Additional pinned-action bumps in this
  release: `pypa/gh-action-pypi-publish → v1.14.0`
  ([#530](https://github.com/yfedoseev/pdf_oxide/pull/530)) which
  carries a security fix
  ([GHSA-vxmw-7h4f-hqxh](https://github.com/pypa/gh-action-pypi-publish/security/advisories/GHSA-vxmw-7h4f-hqxh))
  for the action that publishes our Python wheel;
  `github/codeql-action → v4.35.5`
  ([#528](https://github.com/yfedoseev/pdf_oxide/pull/528));
  `taiki-e/install-action → v2.79.2`
  ([#529](https://github.com/yfedoseev/pdf_oxide/pull/529)) — the
  upstream deprecations (`mdbook-alerts`, `iai-callgrind-runner`) do
  not affect us (our tool list is `cargo-cyclonedx`, `wasm-tools`,
  `taplo-cli`, `cargo-shear`); and `codecov/codecov-action → v6.0.1`
  ([#531](https://github.com/yfedoseev/pdf_oxide/pull/531)).
  The Dependabot `ort 2.0.0-rc.11 → rc.12`
  bump ([#496](https://github.com/yfedoseev/pdf_oxide/issues/496)) was
  **declined** — rc.12 is an upstream regression (missing
  `SessionOptionsAppendExecutionProvider_VitisAI` on `OrtApi`); the pin
  is held at `=2.0.0-rc.11`.

### Thanks

- **@Jethril** for reporting
  [#525](https://github.com/yfedoseev/pdf_oxide/issues/525) — the
  PDF-from-markdown-has-no-styling bug — with a minimal repro that led
  directly to the renderer fix.

## [0.3.51] - 2026-05-17

> Comprehensive auto extraction — per-page text-vs-OCR with typed
> reason codes, graceful native fallback, and image-table recovery —
> across all seven bindings plus the CLI and MCP server; a pre-merge
> release-pipeline dry-run; and five bundled fixes.

### Added

- **Comprehensive auto extraction
  ([#517](https://github.com/yfedoseev/pdf_oxide/issues/517))** — a
  new, **strictly additive** surface that returns recoverable text
  decided **per page/region** with a machine-readable reason for every
  degraded result, and a **graceful warn-and-fall-back-to-native**
  policy (never a crash, never a silent empty). The classifier consumes
  pdf_oxide *internals* (Tr render-mode-3, GlyphlessFont/no-embedded
  ratio, notdef/U+FFFD, union of CTM-transformed image boxes, image
  codec, structure tree, producer/XMP) — strictly more accurate than a
  post-hoc heuristic on the flattened text. New: a configured-once
  `AutoExtractor` (`new`/`text_only`/`with` + `fast`/`balanced`/
  `high_fidelity` presets + builder), `extract_text`/`extract_markdown`/
  `extract_html`/`extract_page`/`extract_document`, the cheap
  `classify_page`/`classify_document` preflight (+ `pages_needing_ocr`),
  a one-shot `PdfDocument::extract_text_auto`, an enriched T0.5
  text-quality gate (U+FFFD ratio + critical-fragmentation hard-trigger
  + a column-scramble/consecutive-repeat detector), an optional
  `force_ocr_pages` per-page OCR override, and build-time
  `AutoExtractor::prefetch_models()` / `model_manifest()` (the
  `pdf-oxide models prefetch`/`manifest` Dockerfile contract). Exposed
  across all seven bindings (Rust, C-ABI, Python, WASM, Node, C#,
  Go cgo+purego — Go via idiomatic functional options) as a frozen
  JSON envelope, plus CLI subcommands `classify`/`auto`/`models` and
  MCP tools `classify`/`auto`. Existing `extract_text`/CLI/MCP
  behaviour is byte-identical.
- AutoExtractor semantics are precisely specified: `TextOnly` returns
  native text **without** classifying (the cheapest path); each
  per-page result reports its **actual** source/reason, so a native
  fallback after a failed/empty/absent OCR is `Fallback` +
  `OcrRequestedButUnavailable` — never mislabelled `Ocr`;
  `classify_page`/`classify_document` **fail closed** on
  encrypted-unauthenticated PDFs (a security op) while non-security
  per-page errors degrade gracefully; the "OCR unavailable" warning
  is emitted only when the `ocr` feature is absent; and
  `model_cache_dir()` resolves cross-platform (Windows
  `%LOCALAPPDATA%`/`%USERPROFILE%`, else `$XDG_CACHE_HOME` or
  `$HOME/.cache`; dependency-free).
- The local-CPU tier ships via the existing ONNX OCR engine + spatial
  table detector; the SLANet + PP-DocLayout-S ONNX *models* are a
  documented **zero-API-change point-release follow-up**
  (`tier-model-strategy.md` §5) — the API, prefetch and manifest
  contracts are stable now.

### Fixed

- **CSS `background-color` ignored in HTML/CSS→PDF
  ([#516](https://github.com/yfedoseev/pdf_oxide/issues/516))** — a
  v0.3.50 regression where output was byte-identical with/without a
  page/`body` `background-color`. Implemented CSS 2.1 §14.2 / CSS
  Backgrounds 3 §3.11.2 **canvas background propagation** (root → else
  `body`, painted over the whole page under content); guarded by a
  core-level Rust test plus the existing Python/Go oracles.
- **OCR-only reading-order parity
  ([#460](https://github.com/yfedoseev/pdf_oxide/issues/460))** —
  `detect_page_type`/`needs_ocr` now route through the unified
  classifier so OCR detection matches `extract_page_auto` exactly;
  `extract_text_ocr` is retained as the documented forced-OCR escape
  hatch; `extract_text` is unchanged.
- **Opaque OCR error on Windows
  ([#513](https://github.com/yfedoseev/pdf_oxide/issues/513))** — the
  bare `RuntimeError("OCR feature not enabled.")` is replaced with an
  actionable message (which wheel/extra, how to supply models, and the
  graceful `extract_text_auto` path); plus a cross-platform Python
  feature-guard test (runs on `windows-latest`).
- **Stale PAdES module rustdoc
  ([#514](https://github.com/yfedoseev/pdf_oxide/issues/514))** —
  `src/signatures/pades/mod.rs` no longer claims the B-T/B-LT/B-LTA
  pieces are "deferred / must not be shipped" (they shipped in
  v0.3.50).
- **Per-glyph `Tm`+`Tj` jitter scrambled reading order
  ([#518](https://github.com/yfedoseev/pdf_oxide/issues/518))** —
  Microsoft Word emits broken-image placeholder text as one
  `BT Tm Tj ET` block per glyph with ±2.5–5pt sinusoidal Y-jitter;
  the `Tm`-run merge tolerated only ±0.5pt, splitting jittered
  glyphs into separate Y-banded spans that the reading-order sort
  then emitted top-to-bottom (e.g. `"Hello"` → `"elH l o"`). The
  same-line tolerance is now scale-relative (0.5× the text-space
  glyph height, ≥0.5pt floor) so typographic jitter merges while
  genuine line breaks (leading ≳ 1.0× font size) still split.
  Pinned by an end-to-end regression suite (the reported repro plus
  a max-amplitude case and an anti-over-merge two-line case).
- **Go `purego` backend panicked at runtime on the first call** — the
  `CGO_ENABLED=0` backend registers every FFI symbol in one
  `sync.Once`; `pdf_sign_bytes_pades` has 18 scalar parameters, which
  exceeds `purego`'s SysV/AMD64 argument limit, so
  `purego.RegisterLibFunc` panicked (`too many stack arguments`) and
  the entire pure-Go backend was unusable (any first call aborted).
  A pre-existing v0.3.50 defect — `cgo` is unaffected and CI only
  *built* (never *ran*) the `purego` backend, so it went unnoticed.
  Fixed additively: a new C-ABI `pdf_sign_bytes_pades_opts` collapses
  the parameters into one `#[repr(C)]` options struct (5-argument
  call surface; delegates to `pdf_sign_bytes_pades`, byte-identical
  behaviour — the 18-argument function is unchanged for existing
  C/C++/C#/Node callers). The `purego` binding now uses it; a Go
  regression test exercises the registration path (closing the
  build-only CI gap). Surfaced by a cross-binding smoke pass of the
  full v0.3.51 + v0.3.50 API.
- **Auto-extract reported a complete native result as
  `partial_success` / `ocr_requested_but_unavailable`** — when the
  classifier routed a page to OCR but OCR was unavailable, the native
  fallback was *unconditionally* labelled degraded, even when that
  native text was itself high quality. A downstream consumer trusting
  `status` / `reason` / `pages_needing_ocr` would run needless OCR and
  treat a perfect extraction as incomplete. Now `route` re-checks the
  T0.5 quality gate on the fallback text: high-quality native text is
  reported `Complete` / `NativeText` / `NativeTextHighConfidence`
  (only genuinely poor fallback stays `partial`). Also: a short,
  clean, image-free text page is classified `TextLayer` (not
  `Scanned`) so it is no longer wrongly listed in
  `pages_needing_ocr`; only *garbled* glyphs route to OCR. Pinned by
  a semantic regression suite that additionally asserts
  `AutoExtractor::extract_text` is byte-identical to the canonical
  `extract_text` per page, plus a default-running **fidelity** suite
  (known prose extracts verbatim, in reading order, ungarbled, and
  `extract_markdown`/`extract_html` delegate faithfully and carry the
  content). Surfaced by the cross-binding smoke pass.
- **AutoExtractor never actually ran OCR** — `route()` invoked
  `extract_text_with_ocr(.., None, ..)` with a `None` engine and there
  was no default engine loader, so the function returned native text
  *without OCR*. The Auto surface silently fell back to native for
  *every* image page even with the `ocr` feature and models present —
  text-from-images was non-functional, not merely untested (#519).
  Fixed: `route()` now builds an `OcrEngine` from the documented
  `model_cache_dir()` (`$PDF_OXIDE_MODEL_DIR` / the `prefetch_models`
  layout: `det.onnx` / `rec.onnx` / `en_dict.txt`) and passes
  `Some(&engine)`; unprovisioned → graceful native fallback (never
  fail-loud — only security ops fail-closed). Pinned by a model-gated
  `#[cfg(feature = "ocr")]` end-to-end test (real image-only PDF →
  AutoExtractor recovers the text, `source = Ocr`) **plus a new CI
  `ocr` lane** that provisions the models + ONNX Runtime and runs it,
  so the path is genuinely exercised. Multi-script note: native
  CJK/Arabic/Hebrew/Cyrillic extraction via the auto surface is
  guaranteed by the byte-identical-to-canonical invariant over the
  repo's running script suites (a direct CJK auto test is included);
  OCR *recognition* of non-Latin images is bounded by which
  PaddleOCR language models are provisioned (provisioning, not a code
  defect).
- **Multi-language OCR + a real model-provisioning API**: the engine
  loader honors `AutoExtractOptions.ocr_languages` and, when unset, a
  cheap script heuristic (`detect_ocr_language`) reads the document's
  own native text so a scanned Chinese/Arabic/Cyrillic/Devanagari PDF
  is not OCR'd with the English model; it selects the per-language
  recognition model + dictionary from the model cache dir (shared
  script-agnostic detector), falling back English → native (never
  fail-loud). `AutoExtractor::prefetch_models(&[OcrLanguage])` is **no
  longer a stub — it actually downloads** (idempotent, atomic) the
  detector + requested language packs into `model_cache_dir()`; new
  `prefetch_models_default()`, instance `AutoExtractor::prefetch()`
  (uses the configured `ocr_languages`), `prefetch_available()`,
  `OcrLanguage` enum + `OcrLanguage::ALL`, and a real
  `model_manifest()` (det + every language's files/URLs). The
  provisioning trio (`prefetch_models` / `model_manifest` /
  `prefetch_available`) is exposed **across all bindings** — C-ABI
  (`pdf_oxide_prefetch_models`/`_model_manifest`/`_prefetch_available`),
  Python, Node, Go (cgo+purego), C# — so the Docker/CI build-time
  predownload story works from any consumer language, not just Rust;
  WASM exposes `modelManifest()` only (browser has no
  filesystem/network-to-disk — host-side provisioning, stated
  honestly). CLI: `pdf-oxide models prefetch [-l <lang>… | --all]`,
  and (real fix) the CLI now **warns instead of silently lying** when
  built without the `ocr` feature (the downloader is `ocr`-gated;
  `pdf_oxide_cli` gained an `ocr` feature forwarding `pdf_oxide/ocr`).
  Honest scope (empirically verified end-to-end through the auto
  surface, **10/12**): english · chinese (Simplified) · **cyrillic** ·
  arabic · korean · latin · devanagari · tamil · telugu · kannada.
  **japanese & chinese-traditional**: the loader/prefetch/detect
  pipeline is correct and their packs download fine, but the specific
  deepghs `japan_PP-OCRv3_rec` / `chinese_cht_PP-OCRv3_rec` models do
  not produce output through the current recognizer (model/engine
  compat — `source=Fallback`; the same pipeline works for the other
  10 incl. Simplified Chinese); their tests are `#[ignore]` with that
  reason — a tracked follow-up, not a code defect, not hidden.
  **Hebrew**: a genuine hard limit — PaddleOCR publishes a Hebrew
  *dict* but no recognition model anywhere, so it cannot be fetched
  (the loader is ready the instant a pair is provided — upstream
  limit, not our code). Pinned by a network-gated `prefetch_models`
  download test (proves real fetch-to-disk), the model-gated
  per-language auto-OCR matrix, the cross-binding manifest-parity
  tests (C-ABI/Python/Node/Go/C#), and the new CI `ocr` lane
  (provisions models + ONNX Runtime and runs them).

### CI / Release

- **Release pipeline unverifiable pre-merge
  ([#515](https://github.com/yfedoseev/pdf_oxide/issues/515))** —
  `release.yml` now runs a no-publish **dry-run on `release/*`
  pull requests** (parity with `release-fips.yml`) plus
  `workflow_dispatch{publish}`; every mutating publish job is
  hard-gated so a `pull_request` can **never** publish, while the
  full build/validate/package matrix runs on the release PR. Scoped
  to `release/*` PRs so ordinary feature PRs are unaffected.

### Thanks

- [@Suleman-Elahi](https://github.com/Suleman-Elahi) for reporting
  [#513](https://github.com/yfedoseev/pdf_oxide/issues/513).
- [@kh3rld](https://github.com/kh3rld) for reporting
  [#518](https://github.com/yfedoseev/pdf_oxide/issues/518).

## [0.3.50] - 2026-05-16

> True destructive PDF redaction, PAdES-B-T/B-LT long-term-validation
> signatures, a runtime cryptographic algorithm-governance policy, and
> split-PDF-by-bookmarks across all seven bindings, plus a
> signature-date correctness fix.

### Added

- **True destructive redaction
  ([#231](https://github.com/yfedoseev/pdf_oxide/issues/231))** — the
  prior "redaction" only drew a filled rectangle over content whose
  bytes survived (recoverable by copy-paste / `pdftotext` / a hex
  editor). Redaction is now **destructive**: the text under each region
  is physically removed from the content stream — every glyph whose
  ISO 32000-1:2008 §9.4.4 text-rendering box intersects the
  (edge-padded) region is deleted, survivors are re-emitted with a
  fresh absolute `Tm` and **no** `TJ` deltas so neither the glyphs nor
  a width/shift side channel (Bland et al., PETS 2023) remain; the page
  is rewritten so the original content object is dropped by the
  garbage-collected full rewrite (no residual recoverable bytes); an
  opaque overlay marks the area (ISO 32000-1:2008 §12.5.6.23, "remove
  all traces … clipping shall not be used"). Composite/Type0/unknown
  fonts are **refused** rather than risk a silent under-redaction
  (fail-closed). New `DocumentEditor::add_redaction` /
  `redaction_count` / `apply_redactions_destructive` plus the
  `pdf_redaction_add/count/apply/scrub_metadata` C ABI and Python,
  WASM, Node, C#, Go bindings and a `pdf-oxide redact INPUT --rect
  PAGE:x0,y0,x1,y1 [--from-annotations] [--fill R,G,B]
  [--no-scrub-metadata]` CLI. The legacy
  `apply_page_redactions`/`apply_all_redactions` keep their signatures.
  Standalone document sanitization (`DocumentEditor::sanitize_document`,
  the live `pdf_redaction_scrub_metadata` C ABI, Python
  `sanitize_document`, WASM `sanitizeDocument`, and the already-wired
  Node/C#/Go scrub paths) strips the `/Info` dictionary, the catalog
  XMP `/Metadata` stream, document JavaScript (`/OpenAction`, `/AA`,
  `/Names/JavaScript`) and `/Names/EmbeddedFiles`; the removed object
  subtrees are hard-excluded from the rewritten file so a secret cannot
  survive even as a GC-missed orphan (G6). Geometric image/path/XObject
  pruning remains roadmap; composite-font text and encrypted documents
  are refused (not under-redacted).
- **PAdES long-term-validation signatures
  ([#235](https://github.com/yfedoseev/pdf_oxide/issues/235))** —
  signing now produces ETSI EN 319 142-1 PAdES baseline signatures, not
  just bare `adbe.pkcs7.detached`: **B-B** embeds the RFC 5035 ESS
  `signing-certificate-v2` signed attribute; **B-T** adds an RFC 3161
  `signature-time-stamp` unsigned attribute over the signature value;
  **B-LT** appends a Document Security Store (ISO 32000-2:2020
  §12.8.4.3 — certs/CRLs/OCSPs + a per-signature `/VRI` keyed by the
  uppercase-hex SHA-1 of the signature's `/Contents`) as an
  **append-only second incremental update**, so the original
  signature's byte range is untouched and stays `Valid`. Read side:
  `read_dss` parses a `/DSS` and `classify_pades_level` reports a
  signature's level (B-B/B-T/B-LT). New
  `sign_pdf_bytes_pades` / `PadesLevel` / `RevocationMaterial` /
  `DocumentSecurityStore` in core, the `pdf_sign_bytes_pades` /
  `pdf_signature_get_pades_level` / `pdf_document_get_dss` /
  `pdf_dss_*` C ABI, and Python, WASM, Node, C#, Go bindings. **B-LTA**
  is also produced: a `/Type /DocTimeStamp` (`/SubFilter
  /ETSI.RFC3161`) RFC 3161 timestamp over the whole file *including*
  the DSS, appended as a third incremental update so the archival
  timestamp covers the signature and its validation material;
  `has_document_timestamp` is the document-scoped reader signal
  (`classify_pades_level` stays signature-scoped and tops out at B-LT
  by design — the frozen `pdf_signature_get_pades_level` C ABI has no
  document handle). The legacy `sign_pdf_bytes` `adbe.pkcs7.detached`
  path is byte-for-byte unchanged. Final ETSI conformance is gated on
  the EU DSS demonstration-validator release check (online TSA fetch is
  CGo/native-only — WASM takes a pre-fetched RFC 3161 token).
- **Runtime crypto-governance policy
  ([#230](https://github.com/yfedoseev/pdf_oxide/issues/230))** — a
  process-wide `crypto::SecurityPolicy` (modes `compat` / `strict` /
  `fips-strict`, plus an `allow:`/`deny:<alg>@<read|write>` override
  grammar) layered as an orthogonal, set-once decorator over the
  existing `CryptoProvider`. Read/write asymmetry lets a deployment
  *read* legacy RC4/MD5 PDFs while *forbidding* weak crypto on write or
  new signatures; fail-closed throughout (unknown algorithm /
  unparseable spec ⇒ deny). Includes a content-keyed `inventory()`
  governance report and a pluggable `AuditSink`. Exposed across all
  seven surfaces (Rust, Python, C ABI, Go, C#, WASM, Node) as
  `set_crypto_policy` / `crypto_policy` / `crypto_inventory`. Default
  (`compat`) behaviour is byte-for-byte unchanged. The residual
  password-key-derivation MD5 (ISO 32000-1 §7.6.3 Algorithm 1/2/3/5/7)
  is now also routed through the governed provider, so a
  `strict`/`fips-strict` policy denies legacy R≤4 at the **primitive**
  level, not only the operation gate — closing the gap noted in the
  v0.3.50 slice. The hashing is byte-identical under `compat`
  (existing encrypted PDFs still decrypt; newly written ones are
  bit-for-bit unchanged). Non-security opaque MD5 (file identifier,
  embedded-file `/CheckSum`) is deliberately left direct so a strict
  policy still permits AES-256 writes. A machine-readable **CycloneDX
  1.6 Cryptographic Bill of Materials** of the algorithms a run
  actually exercised is exported via `crypto_cbom` (core `cbom_json` +
  C ABI / Python / WASM / Go / Node / C# bindings) — the structured
  complement to `crypto_inventory` for CBOM/SPDX-crypto governance.
  The policy now also **recognises and governs post-quantum
  algorithms**: `PolicyMode::Cnsa2` (CNSA 2.0 — new crypto must be
  FIPS-approved *and* 192-bit-class or stronger; 128-bit classical and
  L1/L2 PQC denied for write) and `PolicyMode::PqcReady` (Strict
  semantics that additionally recognise/permit ML-DSA/ML-KEM for
  classical+PQC dual-stacking during migration), plus ML-DSA-44/65/87
  (FIPS 204)
  and ML-KEM-512/768/1024 (FIPS 203) `AlgorithmId`s in
  `inventory()`/CBOM/the policy grammar. This is governance vocabulary
  (the policy decides; the actual ML-DSA/ML-KEM primitives are a
  separate provider concern — a sign attempt fails closed until they
  land). Set via the string grammar (`crypto_policy("cnsa2")`), so all
  seven bindings get it with no API change; frozen `AlgorithmId` bit
  indices are preserved (PQC ids appended). A governed **RSA
  modulus-size floor** is also enforced for *signing*:
  `SecurityPolicy::min_rsa_modulus_bits` (per-mode default — Compat 0,
  Strict/PqcReady 2048, FipsStrict/Cnsa2 3072 per NIST SP 800-131A /
  CNSA 2.0) makes `sign_pdf_bytes`/`sign_pdf_bytes_pades` fail closed
  with a weak RSA key — the key-strength gate the algorithm-level
  `min_security_bits` cannot see. Default `compat` keeps no floor
  (byte-for-byte unchanged). (Finer X.509 cert-policy governance —
  keyUsage / extendedKeyUsage / validity-window enforcement for the
  signing certificate — is the remaining #230 roadmap item, tracked as
  a focused follow-up. Per-document policy override (Phase G) was
  design-assessed and deliberately deferred: the active policy is
  set-once specifically because a mid-flight downgrade is an attack
  vector, so a runtime *widening* override (e.g. relax-for-one-document)
  cannot be added safely; the only sound shape is an explicit
  per-document policy threaded through every crypto call site — a large
  cross-cutting change, tracked as a separate follow-up, not a set-once
  relaxation.)
- **Split a PDF by bookmarks
  ([#482](https://github.com/yfedoseev/pdf_oxide/issues/482))** — new
  `pdf-oxide split --by-bookmarks [--bookmark-prefix P]
  [--bookmark-level N] [--ignore-case] [--no-front-matter]` CLI, plus
  `plan_split_by_bookmarks` / `split_by_bookmarks*` in core and every
  binding (Python, WASM, C ABI, Go, C#, Node). Splits at outline
  boundaries into one PDF per (optionally prefix-filtered) bookmark,
  with collision-free, filesystem-safe filenames. Outline parsing now
  resolves **named destinations** (catalog `/Dests` dictionary and the
  `/Names` → `/Dests` name tree, ISO 32000-1 §12.3.2.3 / §7.9.6),
  bounded against malformed/cyclic name trees. Plain per-page `split`
  is unchanged (backward compatible).
- **Full idiomatic cross-binding parity for #230/#231/#235/#482** —
  every feature is now exposed *idiomatically* in **all** supported
  bindings (Rust, Python, C ABI, WASM, C#, Go-cgo, Go-purego, Node/TS):
  - A new additive C ABI `pdf_document_has_timestamp(doc)` exposes the
    document-scoped PAdES-**B-LTA** reader signal that
    `pdf_signature_get_pades_level` (signature-scoped, ≤B-LT by design)
    cannot report; surfaced as Python `has_document_timestamp`, WASM
    `hasDocumentTimestamp`, C# `PdfDocument.HasDocumentTimestamp`, Go
    `(*PdfDocument).HasDocumentTimestamp`, and Node
    `PdfDocument.hasDocumentTimestamp` / `SignatureManager`.
  - Python now re-exports the entire signing/PAdES surface
    (`sign_pdf_bytes`, `sign_pdf_bytes_pades`, `Certificate`,
    `Signature`, `PadesLevel`, `RevocationMaterial`, `Dss`) plus
    `crypto_cbom` from the top-level `pdf_oxide` package under idiomatic
    names (the functions were previously reachable only as
    `py_`-prefixed symbols on the private extension module).
  - The standalone document **sanitization** entrypoint (#231) is now a
    first-class `SanitizeDocument()` on the C# and Go (cgo + purego)
    `DocumentEditor` (previously the live `pdf_redaction_scrub_metadata`
    C ABI had no managed/Go wrapper).
  - The Go **purego** (CGO-free) backend, previously read-side only,
    now covers crypto-governance (#230), destructive redaction +
    sanitize (#231), PAdES signing + DSS read + B-LTA (#235), and
    split-by-bookmarks (#482) with signatures identical to the cgo
    backend.
  - Node/TS gains idiomatic `signPdfBytesPades`, `PadesLevel`,
    `PdfDocument.getDocumentSecurityStore/hasDocumentTimestamp/
    planSplitByBookmarks`, `setCryptoPolicy/cryptoPolicy/
    cryptoInventory/cryptoCbom`, and `SecurityManager` /
    `SignatureManager` / `OutlineManager` methods, all with generated
    TypeScript declarations. Behaviour and the frozen `PadesLevel`
    integer mapping are unchanged.

### Fixed

- **Wrong dates in digital-signature timestamps** — `format_pdf_date`
  hard-coded the month/day to `0101` and approximated the year as
  `1970 + days/365`, so every signature `/M` value (and document
  timestamps) was an incorrect ≈Jan-1-of-leap-drifted-year
  (ISO 32000-1 §7.9.4). Replaced with one leap-year-correct,
  de-duplicated implementation (the two divergent copies are gone).

### Security

- **Redaction now actually removes content
  ([#231](https://github.com/yfedoseev/pdf_oxide/issues/231))** — the
  Node `editing-manager` redaction methods previously called native
  `pdf_redaction_*` symbols that did not exist (silently no-op'ing — a
  security-critical operation pretending to succeed while removing
  nothing). Those C ABI symbols now exist and perform **true
  destructive** redaction (see Added); the binding gap is closed across
  all surfaces. A `[BLOCK]` integration test builds a real PDF
  containing a secret, redacts it through the public API, and asserts
  the secret is absent from **both** re-extracted text and the raw
  saved bytes (idempotent).
- **PAdES long-term-validation signatures
  ([#235](https://github.com/yfedoseev/pdf_oxide/issues/235))** — PDF
  signatures can now carry the ESS `signing-certificate-v2` binding
  (RFC 5035, defeats certificate-substitution), an RFC 3161 timestamp
  (B-T), and a Document Security Store for offline long-term validation
  (B-LT). The DSS is added as an append-only incremental update so
  pre-existing signatures provably remain `Valid` (asserted by the
  I1–I7 integrity-invariant suite in `tests/pades_ltv.rs`); a tampered
  signed region still fails verification (negative test). See Added for
  scope and the EU-DSS conformance gate.

### Thanks

- [@Suleman-Elahi](https://github.com/Suleman-Elahi) for requesting
  split-by-bookmarks (#482).
- [@jedzill4](https://github.com/jedzill4) for volunteering on
  destructive redaction (#231).

## [0.3.49] - 2026-05-15

> Off-byte-0 PDF header recovery, sparse-trailer Catalog discovery,
> a render-path thread-safety fix, and release-automation hardening.

### Fixed

- **Linearized PDFs with a non-zero `%PDF-` header offset
  ([#509](https://github.com/yfedoseev/pdf_oxide/issues/509))** — files
  whose `%PDF-` header is preceded by leading bytes (e.g. a captive-
  portal HTML redirect injected ahead of a Linearized PDF) are now read
  instead of rejected with `Trailer missing /Root entry`. The xref-
  offset shift for header-offset PDFs no longer requires the final
  trailer to carry `/Root`; xref reconstruction now rejects a parsed-
  but-`/Root`-less trailer and falls through to Catalog discovery; and
  `catalog()` scans for `/Type /Catalog` when the trailer omits `/Root`
  (matching Poppler / PDFium behaviour, ISO 32000-2 §7.5.2 / 1.7
  Implementation Note G.6).

- **Render-path data race under concurrent rendering
  ([#505](https://github.com/yfedoseev/pdf_oxide/issues/505))** — the
  process-wide embedded-font classification cache keyed on
  `Arc::as_ptr` could return a stale `(is_byte_indexed,
  has_unicode_cmap)` for an unrelated font when an allocation address
  was recycled across threads, intermittently surfacing as
  `ParseException [1000]` from `RenderPage` / `RenderPageFit` under
  `Parallel.ForEach`. The unsound global cache is removed; the cmap
  classification is now computed locally per call (a cheap `ttf_parser`
  table probe), so concurrent renders can no longer collide.

- **Test helper `make_type0_font` used a non-production `Encoding`
  variant ([#504](https://github.com/yfedoseev/pdf_oxide/issues/504))**
  — the helper now maps `Identity-H` / `Identity-V` to
  `Encoding::Identity` exactly as the real font parser does, so the
  affected Type0 tests exercise the production code path instead of a
  variant production never produces. Purely test-correctness; no user-
  facing behaviour change.

### CI / Infrastructure

- **Release-notes title extraction hardened
  ([#506](https://github.com/yfedoseev/pdf_oxide/issues/506))** —
  `extract-release-notes.sh` now bounds the subtitle scan to the
  requested version's section (no longer silently inheriting an older
  version's `>` blockquote), concatenates multi-line blockquotes
  instead of truncating at the first line, and fails loudly when the
  version section or its subtitle is missing. A `validate-changelog`
  PR/release-branch gate plus a release-title sanity check stop a
  malformed CHANGELOG from ever reaching the publish step, and a self-
  contained regression test covers the missing-section, missing-
  subtitle, multi-line, and cross-version false-scrape cases.

- **GitHub Deployments visibility for regular publishes
  ([#493](https://github.com/yfedoseev/pdf_oxide/issues/493))** — each
  publish job in `release.yml` (crates.io, PyPI, npm, npm-native,
  NuGet, Homebrew/Scoop) now declares an `environment:`, so standard-
  pipeline publishes appear under the Deployments view with their
  artifact URL, matching what the FIPS pipeline already did.

### Thanks

- [@Goldziher](https://github.com/Goldziher) (kreuzberg-dev) — opened
  [#509](https://github.com/yfedoseev/pdf_oxide/issues/509) with a clean
  standalone reproducer (no app code), a pinned test file, a full
  multi-engine cross-check against Poppler, and a 156-PDF corpus survey
  that isolated this as the single legitimate file the parser rejected.
  That report turned a vague "Linearized PDF fails" into a precise
  header-offset + sparse-trailer root cause.

The remaining fixes ([#506](https://github.com/yfedoseev/pdf_oxide/issues/506),
[#505](https://github.com/yfedoseev/pdf_oxide/issues/505),
[#504](https://github.com/yfedoseev/pdf_oxide/issues/504),
[#493](https://github.com/yfedoseev/pdf_oxide/issues/493)) were surfaced
internally while reviewing the v0.3.45–v0.3.47 release automation, the
post-merge `main` CI runs, and the v0.3.47 PR review.

## [0.3.48] - 2026-05-14

> Bidirectional PDF ↔ DOCX/PPTX/XLSX office converter integration across all seven bindings.

This release lands the **office converter integration**
([#159](https://github.com/yfedoseev/pdf_oxide/issues/159)):
bidirectional PDF ↔ DOCX/PPTX/XLSX round-trip with layout-preserving
fidelity, exposed through all seven bindings (Rust, Python, Node,
WASM, C FFI, C#, Go). Typical text-heavy PDFs round-trip through an
Office file and back at near-pixel parity to the source. The corpus
harness used to validate the integration covers 26 PDFs spanning
academic papers, hymnals, multi-column newspapers, slide decks,
government forms, and policy documents.

Closes the v0.3.14-milestone feature request "PDF to Word/DOCX export":
text styling (fonts / sizes / colours) preserved via layout-mode
writers + Unicode/CJK system-font fallback; paragraphs / headings /
lists preserved via positional frame anchors; image placement preserved
via raster Image XObject + Form XObject rasterization. Tables flow
through positional shapes (grid-aware reconstruction is still
follow-up work).

### Added

- **Bidirectional PDF ↔ DOCX/PPTX/XLSX conversion
  ([#159](https://github.com/yfedoseev/pdf_oxide/issues/159))** — new
  `OfficeConverter` API converts in both directions across DOCX, PPTX,
  and XLSX. Layout-preserving writers
  (`src/converters/{docx,pptx,xlsx}_layout.rs`) emit one positionally-
  anchored shape / frame per PDF text span; the back-direction render
  path (`render_positional_ir` / `render_pptx_positional`) reproduces
  the source page near-identically. Available on every binding via the
  `09-new-features/office_conversion/` examples.

- **Unicode + CJK system-font fallback for office round-trip**
  (`src/fonts/unicode_fallback.rs`) — when the source PDF embeds a CID-
  only font subset the writer can't re-embed, a system Unicode face
  (DejaVu Sans → FreeSans → Noto Sans → Tinos / Arimo) and a CJK face
  (DroidSansFallbackFull → IPAGothic → NanumGothic → Unifont) are
  registered automatically. `needs_unicode_fallback` is WinAnsi-aware
  (curly quotes / em-en dashes / bullet / ellipsis / trademark stay on
  the source font); CJK ranges (Han / Hiragana / Katakana / Hangul /
  Compatibility Forms / Halfwidth–Fullwidth) route to the CJK face
  first. Restores Hebrew, Arabic, Latin Extended, Chinese, Japanese,
  and Korean characters that previously rendered as `?` glyphs across
  all three formats.

- **Music-notation region detection + rasterization**
  (`src/converters/music_region_finder.rs`) — hymnals and sheet-music
  PDFs (Finale Maestro, SMuFL Bravura, Sibelius Petrucci / Opus, Adobe
  Sonata, LilyPond Emmentaler, …) are detected by combining a music-
  font allowlist with a 5-line staff-clustering pass on
  `extract_paths`. Detected music systems are rasterized once at
  150 DPI and embedded as positioned PNGs; the source spans / shapes
  inside each music region are suppressed so glyph substitutions don't
  overlay the bitmap. Hymnal-style PDFs now round-trip with their
  staves and noteheads preserved instead of emitting random Latin
  characters from the missing music face.

- **Form XObject + inline-image rasterizer shared helper**
  (`src/converters/form_xobject_finder.rs::rasterize_form_and_inline_regions`)
  — the layout-mode writers and the flow-mode `pdf_to_ir` path share
  one helper that renders each page once at 150 DPI and crops per
  region. Vector figures (academic-paper charts, agency logos drawn
  as Form XObjects) survive the office round-trip; the prior per-
  region full-page render was replaced.

- **Per-run text colour preservation** — PDF→DOCX/PPTX/XLSX now
  emits `<w:color>` / `<a:solidFill>` for spans carrying explicit
  colour; the back-render path drops to `rich_paragraph` instead of
  `text_in_rect` when any inline run has a colour so the colour
  survives the PDF render. Sibling `office_oxide` parser changes
  expose the colour on `TextSpan` for the docx, pptx slide, and
  pptx shape paths.

### Fixed

- **Rotated-text watermark filter
  (`src/converters/pdf_to_ir.rs::span_overlaps_rotated_chars`)** —
  page-edge `arXiv:NNNN.NNNNN [cat] DATE` watermarks were leaking
  into the office round-trip as horizontal text strips mid-page.
  The new origin-based filter matches each span to its nearest
  `extract_chars` glyph by `(origin_x, origin_y)` distance and uses
  that glyph's `rotation_degrees` to decide drop. Gated by a page-
  level `chars_horizontal_dominant` heuristic (≥75 % chars at ~0°)
  so PDFs whose text-matrix decomposition spuriously reports
  rotation = 90° for every glyph (Finale slide-mode decks) are left
  alone. Catches the watermark family across multiple arxiv papers.

- **Multi-column page handling in layout-mode line grouping
  (`src/converters/layout_lines.rs::group_spans_into_lines`)** —
  refuses to merge a candidate span into the active line when its
  `bbox.x` sits more than `max_font_size * 4` past the line's right
  edge. Threshold (~36-48 pt for body text) is wider than any
  justified inter-word gap but narrower than typical column gutters
  (60+ pt). Fixes German multi-column newspapers and 2-column
  arxiv papers where columns previously merged into one frame.

- **Drop-cap guard for layout-mode line grouping** — `group_spans_
  into_lines` rejects merges when the candidate span's font size
  differs from the line's existing spans by > 2×. Anchors Nature-
  Methods-style drop-cap "A" wraps at the correct visual position
  instead of fusing them into a single heading-class frame with
  the body text below.

- **OpenType / CFF cmap rebuild and injection
  (`src/fonts/cmap_injector.rs`, `src/document.rs`)** — two real
  bugs in the cmap-injection path that produced corrupted lowercase
  glyphs on strict OS renderers:
  - `build_format4_cmap` over-reported subtable length by 2 bytes
    (double-counted the `reservedPad` field). Strict ttf-parser /
    CoreText paths silently rejected the cmap; some Win/macOS
    renderers then mapped the affected codepoints to the wrong
    glyph.
  - `extract_embedded_fonts_with_unicode_maps_and_widths` was driving
    its Unicode→GID table off `char_to_unicode`, whose CID-as-
    Unicode fallback overwrote authoritative ToUnicode entries with
    identity mappings on Identity-H fonts. Now reads the ToUnicode
    CMap directly and filters U+FFFD plus C0 controls.

- **Shape-artefact filter for layout-mode DOCX
  (`src/converters/docx_layout.rs`)** — drop solid-black rects > 25%
  page area (slide-background artefacts), solid-white rects > 50%
  page area (page-background rects emitted before text — would
  occlude the rendered text in the back-PDF), and rects > 1.2× page
  extent (extractor noise that wiped the entire frame).

- **XLSX layout-mode page count gate raised
  (`src/document.rs::to_xlsx_bytes`)** — `LAYOUT_MAX_PAGES` raised
  30 → 200. The 134-page arxiv dissertation was being routed to
  flow-mode `ir_to_xlsx`, whose column-A row-N layout collapses the
  centered cover page into the top of column A. Layout-mode handles
  100+ page documents fine; the gate now triggers only for very
  large reports.

### Performance

- **ExtGState resolve cache: 75× speedup on vector-heavy PDFs
  (`src/rendering/page_renderer.rs`)** — `apply_ext_g_state` was
  deep-cloning the per-Form ExtGState HashMap on every `gs`
  operator. Vector figures (scatter / contour plots emitted as Form
  XObjects) trigger this thousands of times per page — a typical
  academic paper with a dense plot can hit ~10 000 `gs` ops with
  10 000+ unique ExtGState names. The clone dominated render time.
  The resource dict is now resolved once at the top of
  `execute_operators` and parsed-effect (`ParsedExtGState`) results
  are cached per `dict_name`. Measured on a ~10-page vector-heavy
  arXiv paper: PDF→DOCX dropped from 263 s to 3 s.

- **Debug-only path-rasterizer clones gated by log level
  (`src/rendering/path_rasterizer.rs`)** — `path.clone().transform`
  was unconditional, used only to populate `pixel_bounds` in a
  `log::debug!` line. Same vector figures hit this path tens of
  thousands of times per page. Gated behind
  `log::log_enabled!(Level::Debug)`.

## [0.3.47] - 2026-05-11

> Text-extraction quality, CJK + RTL fixes, table-detection hardening, and a WASM SystemTime fix.

This release closes the remaining bugs surfaced by the kreuzberg
integration (issue [#484](https://github.com/yfedoseev/pdf_oxide/issues/484))
and ships the related text-extraction quality fixes.  Word-F1 against the
pdftotext-derived ground truth corpus now meets the kreuzberg quality
floor for every PDF in the issue 484 set.

### Fixed

- **kreuzberg regression suite — all 24 PDFs now meet the F1 floor
  ([#484](https://github.com/yfedoseev/pdf_oxide/issues/484))** —
  `extract_text` previously failed three documents reported by
  @Goldziher on the kreuzberg corpus: `pdfa_039.pdf` (swimming-results
  table) returned F1 0.810, `pr-136-example.pdf` (CJK financial document)
  returned F1 0.709, and `annotations.pdf` returned F1 0.545.  Three
  separate root-cause fixes restore them to F1 ≥ 0.85:
  - `eliminate duplicate emission of multi-row table labels` — the
    text-only spatial fallback in `detect_tables_with_lines` now
    requires `config.text_fallback=true` (which `extract_text` does
    not pass) so report-style PDFs with decorative ruling lines no
    longer get their cell content emitted twice; `span_in_table` adds
    a text-match fallback to catch label spans whose font ascent
    extends slightly above the cell's ink box (issue-53-example.pdf
    F1 0.867 → 0.992).
  - `tighten cross-font glue and decimal merge for CJK + Latin
    layouts` — `cross_font_word_glue` no longer fires on a CJK ↔
    non-CJK boundary (CJK ideographs satisfy `is_alphabetic()` per
    Unicode and were being concatenated with adjacent Latin); the
    `decimal_merge` heuristic requires a column-boundary-sized gap
    (gap > 0.4 em) so per-glyph Tj operators in CJK documents stop
    mangling "2013" into "201.3" (pr-136 F1 0.709 → 0.884).
  - `narrow CJK boundary forced-space to script glyphs only` —
    `should_insert_space` now actively inserts a space at the
    CJK ↔ non-CJK boundary to match pdftotext tokenisation, but
    restricted to actual script glyphs (ideographs, kana, hangul);
    fullwidth ASCII operators like ＜ ＞ ＝ μ stay inline with
    adjacent digits/Latin so compound tokens like "60000≤Q＜80000"
    are preserved (issue-336 text quality gate stays at PASS).
  Reported by @Goldziher.

- **`extract_spans` now exposes a `merge_tm_tj_runs` opt-out
  ([#488](https://github.com/yfedoseev/pdf_oxide/issues/488))** —
  Same-line Tm+Tj runs were unconditionally batched into a single
  `TextSpan`, throwing away the per-Tm positioning that downstream
  layout-analysis code (e.g. column-aware table detection) needs.
  `SpanMergingConfig::merge_tm_tj_runs` (default `true` for backward
  compatibility) now flushes the span buffer at every Tm operator so
  callers can opt in to one span per Tm+Tj group, matching the
  granularity of `pdftotext -bbox-layout`.  Reported by @haberman.

- **`saveEncryptedToBytes` no longer panics in browser WASM
  ([#492](https://github.com/yfedoseev/pdf_oxide/issues/492))** —
  `generate_file_id` (per ISO 32000-1 §14.4) called
  `std::time::SystemTime::now()`, which is unimplemented on
  `wasm32-unknown-unknown`.  Cfg-gated so the WASM build derives the
  file identifier from `uuid::Uuid::new_v4()` only — still a unique
  opaque 16-byte ID per the spec.  Reported by @eersis-byte.

- **CJK fullwidth operator spacing in `to_markdown` / `to_html`
  ([#485](https://github.com/yfedoseev/pdf_oxide/issues/485))** —
  Four coordinated changes restore `issue-336-example.pdf` to PASS
  on all three quality gates (text, markdown, html):
  - `pipeline/converters/has_horizontal_gap` suppresses space
    insertion when one side is CJK and the other is CJK or a
    fullwidth/math operator (≤, ＜, ＞, ＝, μ, etc.), mirroring the
    text-extraction CJK-pair suppression.
  - `extract_cell_text` no longer inserts an unconditional space
    between adjacent spans on the same row of a table cell — uses
    the same gap-aware separator rules as the inline-flow path so
    multi-span cells like `60000≤Q＜80000` (rendered as 5 separate
    Tj operators) keep their compound tokens intact.
  - `consolidate_adjacent_table_fragments` (new helper in
    `spatial_table_detector`) merges vertically-adjacent tables that
    share an identical column structure.  The line-based detector
    emits one fragment per ruling-rule strip on PDFs that draw a
    horizontal rule between every pair of rows; each fragment was
    failing `is_real_grid` and falling through to paragraph flow
    with column-based reading order, producing orphan
    `<p>40000≤Q</p>` / `<p>＜55000</p>` pairs.  Consolidating before
    the filter lets the merged multi-row table survive.
  - `is_real_grid` accepts wide consolidated tables that have
    dense data rows alongside sparse header / multi-row-label rows
    — the strict 70 % dense-ratio gate was rejecting real tables
    whose column headers split across multiple visual rows.
  Score improvements on `issue-336-example.pdf`:
  text 0.612 → 0.820, markdown 0.577 → 0.863, html 0.632 → 0.646
  (all PASS their thresholds).

- **Text-only spatial table fallback for line-less tables in
  `to_markdown`
  ([#486](https://github.com/yfedoseev/pdf_oxide/issues/486)) —
  partial fix.** `extract_page_tables` now opts in to a relaxed
  text-only detection when the caller is a converter (text_fallback=
  true), with the column ceiling raised from 15 to 25 so that
  sailing-score grids with 16-18 score columns are no longer
  rejected outright.  The fragmented-table consolidation from #485
  also kicks in here, recovering most of the row labels and
  identifier columns.  `nougat_018.pdf` markdown still trails its
  threshold (0.656 vs 0.90) because the score columns themselves —
  variable-width sparse cells with parenthesised drop-scores —
  evade column detection; that is the remaining piece tracked
  separately.

- **HTML table cell rendering aligned with markdown
  ([#487](https://github.com/yfedoseev/pdf_oxide/issues/487)) —
  partial fix.** `to_html` now uses the same span-walking and
  bold/italic preservation as `to_markdown`'s
  `render_table_markdown`.  Three of four affected docs improved
  by 1-4 % Jaccard but two (nougat_018, nougat_026) still trail the
  threshold pending the table-fragmentation work above.

- **RTL inline emphasis stripping in markdown extraction
  ([#459](https://github.com/yfedoseev/pdf_oxide/issues/459))** —
  RTL detection now strips `<strong>` / `<em>` markers from
  visually-reversed runs in `to_markdown` consistently with the
  plain-text path; spec basis ISO 32000-1 §14.8.2.3.3 (Reverse-
  Order Show Strings).  46 unit tests in
  `tests/test_rtl_script_support.rs` cover the detector, BiDi
  algorithm, and inline-flow integration.

- **Multi-byte CMap parsing and array-form `beginbfrange`
  (§9.7.5)** — `beginbfrange ... endbfrange` array notation
  `<src> <src> [<dst1> <dst2> ...]` was not fully covered; the
  CMap parser now matches the spec's allowed grammar so multi-byte
  CIDs map correctly through ToUnicode CMaps.

- **`/StructTreeRoot`-only tagged PDFs (§14.7.4)** — Documents
  that declare `/StructTreeRoot` in the catalog without a
  `/MarkInfo` dictionary (PDF 1.4 documents, valid per the spec)
  now correctly use the structure tree for table-cell content
  extraction.  Resolves `/OBJR` content-item references during
  tree traversal so OBJR-referenced annotations and XObjects are
  no longer lost.

- **Indirect references in MediaBox/CropBox accessors (§7.7.3.4)**
  — Page attribute accessors now resolve `/MediaBox` and `/CropBox`
  through indirect references and the `/Pages` inheritance chain.
  This is what made the Bucket A errors in the issue 484 retest
  comment (`annotations*.pdf`, `pdfa_039.pdf`) parse successfully.

- **CTM-aware cache key for Form XObject span extraction** — Form
  XObject spans were cached by XObject reference alone, returning
  stale coordinates for the same XObject reused on multiple pages
  with different CTM transforms.  Cache key now includes the CTM
  so repeated XObjects produce correctly-positioned spans on each
  invocation.

- **`notdefrange` U+FFFD no longer blocks the CID-as-Unicode
  fallback (§9.10.2)** — Per the spec, U+FFFD (REPLACEMENT
  CHARACTER) signals "no proper Unicode mapping", so a notdefrange
  hit must not stop the priority list.  The Identity CID-as-Unicode
  fallback (Priority 3) now fires correctly for composite fonts
  whose ToUnicode CMap returns U+FFFD.

- **ToUnicode Priority-3 fallback guarded for composite fonts
  (§9.10.2)** — The CID-as-Unicode fallback is now only applied
  to fonts whose CMap is one of the predefined composite-font
  CMaps or whose CIDFont uses one of the Adobe character
  collections, matching the spec's enumeration; misapplication on
  other fonts could produce mojibake on previously-working files.

- **Reject prose / TOC / underline-annotation false-positive
  tables in `to_html` and `to_markdown`** — Wide pages of
  ordinary paragraph text were sometimes detected as multi-column
  tables: word x-positions cluster into "columns" by accident, and
  decorative horizontal rules (newsletter mastheads, annotation
  underlines, page borders) tricked the line-based detector into
  treating two adjacent lines as a header + data row.  The
  detection pipeline now applies several post-`is_real_grid`
  guards that look at the *shape* of the candidate's cell
  content rather than just its grid geometry:
  - `looks_like_prose_table` rejects a candidate when more than
    12 % of cells end with a mid-sentence `,` or `;`, more than
    25 % of cells start with a lowercase ASCII letter
    (continuation fragments like "and", "the", "to"), or more
    than 10 % of cells are pure leader dots (the `. . . . . .`
    runs in tables of contents).
  - The text-only spatial fallback and the horizontal-rule-
    bounded path both now require ≥ 3 rows of evidence.  A
    title plus a wrapped body line is the signature of prose,
    not a table; only the line-based intersection / cluster
    paths (which have authoritative visual evidence) still
    accept 2-row tables.
  - `should_insert_space` no longer forces a space at the
    CJK ↔ ASCII-punctuation boundary.  The boundary forced-
    space added in v0.3.47 was correctly inserting a space at
    "神鹰集团" + "2015" but was wrongly producing "する ."
    instead of "する." in Japanese technical text; ASCII
    clause punctuation hugs the preceding token in every
    script, so the rule is now suppressed when the
    transitioning glyph IS the punctuation.
  - `text_fallback` defaults back to `true` on
    `TableDetectionConfig`.  The new prose-shape filter
    replaces the gate-based protection added earlier in the
    cycle, so the public `extract_tables` API again detects
    line-less data tables out of the box.

### Notes

- `tests/test_corpus_extraction_quality.rs` now strips markdown
  formatting markers (`**bold**`, `*italic*`, `|` separators,
  `---|---|---` rule, `# heading`, ```` ``` ```` fences) before
  computing Jaccard against the plain-text GT — mirrors the HTML
  test's existing `strip_html` step so the score reflects text
  content rather than formatting markup.
- All 19 quality-gate Jaccard tests in
  `tests/test_corpus_extraction_quality.rs` now pass (up from
  13 at the start of this branch).  The kreuzberg issue 484
  corpus passes its F1 floor on every PDF.

### Thanks

This release was driven entirely by community bug reports and the
kreuzberg integration test feedback loop:

- @Goldziher (kreuzberg-dev) — opened #484 with a calibrated 166-PDF
  regression suite and follow-up retest comments that turned every
  remaining gap into a focused root-cause fix
- @haberman — opened #488 with a minimal Rust reproducer for the
  Tm+Tj merging issue
- @eersis-byte — opened #492 with the WASM `SystemTime` panic backtrace

## [0.3.46] - 2026-05-10

> Extraction quality, raw RGBA output, JBIG2 decode, editor fixes, and FIPS CI hardening.

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

> legacy-crypto gate, FIPS RSA-PKCS#1 v1.5, CJK font subsetter fix, and render_page_fit precision.

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
