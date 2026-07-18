# AGENTS.md — guidance for AI coding agents

This is a PDF-spec-compliant Rust library with 20+ language bindings over one
core. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) in full before proposing changes.
The rules there apply to agent-assisted work; the essentials:

## Contribution rules (must follow)

1. **Issue-first.** Non-trivial changes require an issue the maintainer has
   accepted, with the approach agreed *before* code is written. Do not open
   drive-by PRs — they may be closed without review. Bug/typo/docs fixes may
   skip this; the reproducer may not.
2. **No autonomous PRs.** An autonomous agent must not open issues or pull
   requests on its own. A human directs the work, reviews every line, and is
   accountable for it. Disclose AI assistance in the PR; the human — not the
   agent — signs off (DCO) and owns correctness, licensing, and provenance.
3. **The human must understand and be able to explain every line.** If the
   change can't be explained without the AI, it isn't ready.
4. **Every bug fix ships a regression test that fails before the fix.** Build the
   reproducer as a **minimal synthetic PDF in code** — never commit a
   third-party/reporter/real-world PDF (the `fixture-hygiene` CI job blocks it).
5. **Prove no corpus regression.** For any extraction/layout/rendering/font
   change, run `corpus_sig` on a corpus **you supply** (the project corpus is
   private, not distributed) and diff against **both `main` and the latest
   release**; report what you tested in the PR. Judge with a structural metric
   (word-Jaccard + spacing outliers), not char-Levenshtein.
6. **House rules:** no issue/PR numbers or contributor/company names in code,
   comments, or fixture names — name tests by defect *class*
   (`type0_identity_h_tj_word_seam`, not `issue847`); credit reporters only in
   `CHANGELOG.md`. One logical change per PR. Fail loudly, never fall back to a
   silent plausible-but-wrong result.
7. **Green across the whole feature matrix**, not just `cargo test` — run the
   tiers you touch (`rendering`, `fips,icc`, `ml`, the affected binding). Never
   `--all-features` (`fips` and `legacy-crypto` are mutually exclusive).

## PDF Specification Reference

The authoritative PDF specification is at `docs/spec/pdf.md`
(ISO 32000-1:2008, PDF 1.7).

**Key sections for text extraction:**
- Section 9: Text (fonts, text state, text objects)
- Section 9.10: Extraction of Text Content (ToUnicode CMaps, character mapping)
- Section 9.4: Text Objects (BT/ET, positioning operators Td, TD, Tm, T*)
- Section 9.3: Text State Parameters (Tc, Tw, Tz, TL, Tf, Tr, Ts)
- Section 14.7: Logical Structure (Tagged PDF structure trees)
- Section 14.8: Tagged PDF (semantic structure)

**Compliance Rules:**
1. All text extraction MUST follow Section 9.10 character-to-Unicode mapping priority
2. Word boundary detection should use TJ offset values (Section 9.4.4) and geometric positioning
3. Do NOT use linguistic heuristics (CamelCase, pattern matching) for word segmentation
4. Prefer Tagged PDF structure (Section 14.7) when available for reading order
5. Font metrics from PDF spec (Section 9.6-9.8) are acceptable for spacing calculations
