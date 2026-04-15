# pdf_oxide Benchmark Harness — Implementation Plan

Closes: #320. Branch: `feat/benchmark-harness` (off `release/v0.3.31`).

## Why this exists

Release validation today is a 170-PDF byte/word diff. That catches crashes
and gross regressions but can't answer "did markdown extraction quality
go up or down by N percentage points". Without TF1/SF1 scoring against
ground-truth markdown, every release ships on gut-feel. #320 is right
that this is verification infrastructure, not a feature.

## Scoring methodology

Mirrors Kreuzberg's `tools/benchmark-harness` so external numbers are
comparable. Formulas:

- **TF1**: bag-of-words F1 on lowercase alphanumeric tokens between
  extracted markdown and ground-truth markdown.
- **SF1**: block-level F1 with per-block-type weights
  (`heading=2.0`, `code/formula/table=1.5`, `list=1.0`,
  `paragraph/image=0.5`). `match_score = content_TF1 × type_compat`
  with a type-compatibility matrix (exact match = 1.0, heading-to-
  paragraph = 0.25, etc.). Greedy assignment, threshold 0.10 (0.20
  for short blocks < 5 tokens).
- **Order score**: LIS length / match count; 1.0 = perfectly ordered,
  0.0 = reversed.

## Deliverables

1. `tools/benchmark-harness/` Rust crate, workspace member.
2. `cargo run -p benchmark-harness -- run --engine <E> --corpus <DIR> --ground-truth <DIR> --output <JSON>`.
3. `cargo run -p benchmark-harness -- diff BASE.json HEAD.json`
   — exit non-zero on meaningful regression (tunable thresholds).
4. Engine adapters: `pdf_oxide` (in-process), `pdftotext` (subprocess,
   poppler), `pdfium` (pdfium-render crate). Docling deferred.
5. Fixture corpus: vendor Kreuzberg's Apache-2.0 fixtures +
   attribution; extend with pdf_oxide-specific fixtures later.
6. `make benchmark-compare BASE=<rev> HEAD=<rev>` target for
   per-release validation.
7. README covering scoring, engine setup, CI integration.

## Non-goals

- Performance benchmarking (timings are reported but not gated).
- GPU/OCR engines.
- Real-time visualization / dashboards.

## Sequencing

| Phase | Subject                                       | Cut-off |
| ----- | --------------------------------------------- | ------- |
| 1     | Crate scaffold + CLI skeleton                 | D1      |
| 2     | TF1 scorer + pdf_oxide adapter                | D1      |
| 3     | SF1 scorer (block parser + weighted F1 + LIS) | D2      |
| 4     | pdftotext + pdfium adapters                   | D3      |
| 5     | Consensus fallback ground-truth mode          | D3      |
| 6     | Vendor Kreuzberg fixtures                     | D4      |
| 7     | Regression gate + diff subcommand             | D4      |
| 8     | Makefile + README + CI wiring                 | D5      |

Every phase produces usable output on its own. After phase 2 we can
already diff two branches' JSON reports on our existing corpus.

## Risks / open questions

- **License of fixtures**: Kreuzberg is Apache-2.0. We vendor with
  attribution (NOTICE file). Need to confirm per-fixture licenses
  inside their corpus aren't stricter (some fixtures may be CC-BY-SA).
- **pdfium-render toolchain**: requires a prebuilt `pdfium` shared
  library. CI will need to fetch it; local dev can skip the engine.
- **Consensus baseline quality**: when we fall back to "median of
  N engines" as ground truth, the scores are relative, not absolute.
  Clearly labelled in the report.
- **pymupdf4llm license**: AGPL. We can call its output from our
  tooling (no linkage), but we don't redistribute it. Optional
  adapter only.
