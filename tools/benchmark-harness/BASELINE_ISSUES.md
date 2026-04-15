# Baseline benchmark findings — `release/v0.3.31`

First run on the Kreuzberg PDF corpus (102 stem-matched fixtures out of 154
PDFs / 180 GT markdown files), engine = `pdf_oxide` vs `pdftotext`.

## Headline numbers

|                 | pdf_oxide | pdftotext |       Δ |
| --------------- | --------: | --------: | ------: |
| TF1 mean        |     0.919 |     0.946 | -2.7 pp |
| TF1 p50         |     0.965 |     0.984 | -1.9 pp |
| TF1 p10 (worst) |     0.776 |     0.881 | -10.5pp |
| SF1 mean        |     0.337 |     0.232 | +10.5pp |
| SF1 p50         |     0.340 |     0.190 | +15.0pp |
| order mean      |     0.804 |     0.863 | -5.9 pp |
| total runtime   |     8.3 s |     6.8 s |   +22 % |

Per-fixture breakdown (TF1 delta):

|         | count |   % |
| ------- | ----: | --: |
| wins (Δ>+1pp)   |     3 |  3% |
| ties (|Δ|<1pp)  |    59 | 58% |
| losses (Δ<-1pp) |    40 | 39% |
| big losses (>5pp) |  12 | 12% |
| **net mean Δ**  |     − | -2.7pp |

**Bottom line.** On content coverage (TF1) we're noticeably behind poppler,
especially on the hard tail. We make up ground on structure (SF1) because
our output happens to retain more paragraph-like structure than poppler's
layout-mode dump — but our SF1 is still objectively low (0.337 / 1.0),
because we emit plain text, not markdown. Once we swap the adapter to the
markdown converter, SF1 will rise *or* the real structure gap will become
visible — either is better than the current "can't tell".

## Confirmed bugs

### B1 — `extract_text(n)` returns page-0 content on linearized PDFs

`tools/benchmark-harness/fixtures/kreuzberg/pdfs/nougat_005.pdf` (ExpertPdf,
`/Linearized 1`, 5 pages):

```
=== page 0 (942 bytes) | "2021\n\n\nGeneral merchandise and apparel …"
=== page 1 (942 bytes) | "2021\n\n\nGeneral merchandise and apparel …"
=== page 2 (942 bytes) | "2021\n\n\nGeneral merchandise and apparel …"
=== page 3 (942 bytes) | "2021\n\n\nGeneral merchandise and apparel …"
=== page 4 (942 bytes) | "2021\n\n\nGeneral merchandise and apparel …"
```

Every page index returns identical bytes. pdftotext on the same PDF emits
distinct content per page including the "SIGN OFF / Nigel Chadwick /
Chief Financial Officer / Friday 28 May 2021" and the DISCLAIMER block
on page 5 (both completely absent from `pdf_oxide` output).

Scored TF1: pdf_oxide 0.254 vs pdftotext 0.924 → **single worst fixture,
Δ -67 pp**.

Hypothesis: the linearized page tree resolves every leaf Kid to the Root
page object. Needs a targeted fix in the page resolution code path.
**Issue to file post-benchmark.**

### B2 — Empty-page false positives on text-heavy PDFs

`pdfa_010.pdf` (14 pages): `extract_text` returns 0 bytes for pages 2, 9,
11. pdftotext returns 400–2000 bytes each. These are text-heavy medical
report pages, not scanned images (verified from pdfinfo). TF1 0.626 vs
0.813 (Δ -18.6 pp).

Hypothesis: our content-stream parser is bailing early on some specific
operator combination these pages use.

### B3 — Running-artifact detector removes cover-page titles

Seen on `pdfa_010` (drops "University of Oklahoma 2009") and the earlier
`5PFVA6…` case from the 170-PDF byte sweep. The detector from commit
`c3d3e3f` treats any line that repeats on every page as chrome and
suppresses it — correct for running headers, wrong when the document
title happens to be included in the header block.

Fix direction: require at least one page (cover/first) to retain the
repeating text when it appears above the page fold; only suppress from
the *second* occurrence onward.

### B4 — Reading-order degradation on multi-column pages

`order_mean` is 5.9 pp lower than pdftotext across the corpus. Inspection
of the big-loss fixtures (nougat_005, nougat_004, nougat_016) shows the
XY-cut strategy breaking interleaved text and figure-caption columns on
dashboard-style layouts.

## Dashboard — 12 worst fixtures by TF1 delta

| Fixture                              | pdf_oxide | pdftotext | Δpp    | likely cause |
| ------------------------------------ | --------: | --------: | -----: | --- |
| nougat_005                           |     0.254 |     0.924 |  -67.0 | B1 linearized, page-repeat |
| nougat_026 / pdfa_001                |     0.775 |     0.986 |  -21.0 | B4 reading-order |
| nougat_035 / pdfa_010                |     0.626 |     0.813 |  -18.6 | B2 empty pages + B3 |
| nougat_016                           |     0.645 |     0.792 |  -14.7 | B4 |
| pdfa_050, pdfa_036                   |  0.91     |  0.99     |  -8.7  | B4 tail |
| nougat_046 / pdfa_021                |     0.906 |     0.979 |  -7.3  | B4 |
| pdfa_044                             |     0.924 |     0.992 |  -6.7  | marginal |
| pdfa_026                             |     0.897 |     0.962 |  -6.5  | marginal |

## Recommended issue filings

| Ref | Title                                                    | Scope          |
| --- | -------------------------------------------------------- | -------------- |
| B1  | extract_text returns identical content per page on some linearized PDFs | fix + regression test |
| B2  | extract_text emits empty string on some text-heavy pages  | investigate + fix |
| B3  | Running-artifact detector suppresses cover-page titles when they repeat in header area | refine detector |
| B4  | XY-cut reading-order drops / reorders content on dashboard / figure-caption layouts | reading-order tuning |

## What the harness proved

1. It finds real bugs (B1). A 170-PDF byte diff would not have caught
   "every page returns page 0" — bytes came out the same size on both
   branches because both branches had the bug.
2. TF1/SF1 surface *quality gaps*, not just crashes. pdftotext isn't
   necessarily "better" — it has no structure claim — but its TF1 lead
   of 10.5pp at p10 proves pdf_oxide is losing content on hard PDFs
   that nobody would have flagged by eyeball.
3. The harness runs in under 15 seconds per engine on this corpus. Fast
   enough to gate every release.

## Next

1. Open issues B1–B4 upstream on pdf_oxide so they're tracked separately
   from the benchmark work.
2. Fix B1 first (largest TF1 hit, easiest repro).
3. Swap the pdf_oxide adapter to the markdown converter so SF1 becomes a
   real measurement instead of a proxy for paragraph structure.
4. Rerun: expect mean TF1 gap to narrow by ≥2pp just from B1 + B2.
