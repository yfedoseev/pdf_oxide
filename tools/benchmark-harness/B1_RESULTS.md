# B1 fix — before/after measurements

Run: `benchmark-harness run --engine pdf-oxide --corpus kreuzberg/pdfs
--ground-truth kreuzberg/gt` (102 stem-matched fixtures, 30 s timeout per
fixture).

| Metric       | Before (v0.3.31) | After (B1 fix) |   Δ   |
| ------------ | ---------------: | -------------: | ----: |
| **TF1 mean** |            0.919 |      **0.925** | +0.64pp |
| TF1 p50      |            0.965 |          0.965 |    0 |
| **TF1 p10**  |            0.776 |      **0.848** | +7.2pp |
| SF1 mean     |            0.337 |          0.339 | +0.22pp |
| SF1 p10      |            0.121 |          0.128 | +0.75pp |
| order mean   |            0.804 |          0.808 | +0.45pp |
| total runtime|            8.3 s |          5.7 s | −31 % |

**Zero per-fixture regressions** above threshold (diff: "no regression
above thresholds").

## Key fixture: nougat_005.pdf

| Metric | Before | After |
| ------ | -----: | ----: |
| TF1    |  0.254 | 0.901 |
| SF1    |  0.071 | 0.274 |

Single fixture moved from worst-in-corpus to essentially at parity with
pdftotext (0.924). Accounts for most of the p10 improvement.

## Takeaways

- The hard-tail gap vs pdftotext at p10 shrank from 10.5pp (0.776 vs
  0.881) to 3.3pp (0.848 vs 0.881). The remaining gap is mostly B2–B4
  territory (empty text-heavy pages, running-artifact over-aggression,
  multi-column reading order).
- Per-fixture runtime dropped 31 % because we no longer re-run the full
  text pipeline from the cache-poisoned state.
- SF1 barely moved, as expected: pdf_oxide still emits plain text
  (newlines, not markdown blocks) so structural F1 is dominated by
  parser-specific paragraph matching, not our fix.

## Reproduce

```bash
git checkout main
cargo build --release -p benchmark-harness
make benchmark-run OUTPUT=base.json

git checkout fix/b1-linearized-page-resolution
cargo build --release -p benchmark-harness
make benchmark-run OUTPUT=head.json

make benchmark-compare BASE=base.json HEAD=head.json
```
