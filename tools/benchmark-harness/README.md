# pdf_oxide benchmark-harness

Release-verification infrastructure for `pdf_oxide`. Computes **TF1**
(token F1) and **SF1** (block-weighted structural F1 with LIS ordering)
against ground-truth markdown, so "did this release improve extraction
quality?" has an answer beyond gut feel and byte diffs.

Closes #320.

## Quick start

```bash
# 1. Fetch an external fixture corpus (Kreuzberg's Apache-2.0 set).
make benchmark-fetch

# 2. Score the current branch.
make benchmark-run OUTPUT=head.json

# 3. Diff two runs and gate on regression.
git checkout main
cargo build --release -p benchmark-harness
make benchmark-run OUTPUT=base.json
make benchmark-compare BASE=base.json HEAD=head.json
```

The `compare` step exits non-zero when:

- mean TF1 drops > 0.5pp (configurable `--mean-tf1-drop-pp`),  or
- any single fixture drops > 5pp (configurable `--per-fixture-tf1-drop-pp`).

## Scoring

### TF1 — token F1

```
precision = |ext ∩ gt| / |ext|
recall    = |ext ∩ gt| / |gt|
TF1       = 2 · P · R / (P + R)
```

Tokens are lowercase alphanumeric; bag-of-words (set-based). Matches
Kreuzberg's methodology so numbers are comparable across projects.

### SF1 — structural F1

```
weight(heading)                    = 2.0
weight(code | formula | table)     = 1.5
weight(list)                       = 1.0
weight(paragraph | image)          = 0.5

type_compat:
  exact match                      = 1.0
  heading↔heading(|Δlevel|)        = max(0.6, 1.0 − 0.1·|Δlevel|)
  list ↔ paragraph                 = 0.5
  heading ↔ paragraph              = 0.25
  code ↔ formula                   = 0.3
  table ↔ paragraph                = 0.25
  code ↔ paragraph                 = 0.2
  everything else                  = 0.0

match_score = content_TF1 · type_compat
greedy assignment (threshold 0.10, or 0.20 if either block < 5 tokens)

matched_w = Σ weight(block) · match_score
recall    = matched_w(gt)  / Σ weight(gt_blocks)
precision = matched_w(ext) / Σ weight(ext_blocks)
SF1       = 2 · P · R / (P + R)
order     = LIS(matched ext indices sorted by gt index) / matches
```

Block types come from a `pulldown-cmark` parse with tables, math, and
GFM enabled. Math inside a paragraph promotes it to `Formula`.

### Consensus mode (no ground truth)

Pass `--consensus-peers pdftotext,pdfium` (instead of `--ground-truth`)
and the harness will build a per-PDF token set from the intersection of
≥2 peer engines and score the target against it. The report records
`reference=consensus(pdftotext,pdfium)` so downstream readers never
confuse this with absolute quality.

## Engine adapters

| Engine       | Flag                | Cost          | Dependencies                                   |
| ------------ | ------------------- | ------------- | ---------------------------------------------- |
| `pdf_oxide`  | `--engine pdf_oxide` | in-process    | workspace member                               |
| `pdftotext`  | `--engine pdftotext` | subprocess    | `poppler-utils` on PATH, or `$PDFTOTEXT_BIN`   |
| `pdfium`     | `--engine pdfium`   | native linked | `cargo build --features pdfium`, `$PDFIUM_DYNAMIC_LIB_PATH` |

More engines go in `src/engine.rs`; one enum arm + one trait impl per
engine.

## Report format

```jsonc
{
  "engine": "pdf_oxide",
  "corpus": "tools/benchmark-harness/fixtures/kreuzberg",
  "reference": "manual",              // or "consensus(pdftotext,pdfium)"
  "ground_truth": "…/kreuzberg",      // null under consensus
  "fixtures": [
    {
      "name": "arxiv_2510.21411v1",
      "tf1": 0.847,
      "sf1": 0.712,
      "sf1_precision": 0.69,
      "sf1_recall": 0.73,
      "order_score": 1.0,
      "matched_blocks": 42,
      "duration_ms": 184,
      "error": null
    }
  ],
  "aggregate": {
    "count": 318, "ok": 316,
    "tf1_mean": 0.83, "tf1_p50": 0.86, "tf1_p90": 0.52,
    "sf1_mean": 0.67, "sf1_p50": 0.71, "sf1_p90": 0.38,
    "order_mean": 0.94,
    "duration_ms_total": 58321
  }
}
```

`tf1_p90` / `sf1_p90` are **lower-tail** percentiles — the worst 10%,
not the best — so regressions surface first. Aggregate means filter out
failed extractions.

## Sequencing

See `PLAN.md` for the full plan and open risks. Phases 1–7 are done.
Phase 8 (this file + Makefile + fetch script) is complete; CI wiring
(a `benchmark` job that runs `make benchmark-run` on every release
branch and uploads the JSON artifact) is the remaining stretch item.

## License

This crate is MIT, matching the workspace. Fixtures fetched via
`scripts/fetch-fixtures.sh` are Kreuzberg's (Apache-2.0, per-fixture
licenses vary — inspect `fixtures/kreuzberg/*/LICENSE*` before
redistributing).
