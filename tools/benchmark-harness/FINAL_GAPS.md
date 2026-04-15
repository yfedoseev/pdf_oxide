# Deep-dive: all remaining gaps after B1+B3+B4+B8a

Combined branch: `fix/b1-b3-b4-combined-with-harness`.
Run: `benchmark-harness run --engine pdf-oxide` vs `--engine pdftotext`
on 102 stem-matched Kreuzberg fixtures, deduplicated by PDF content hash
to 78 unique inputs (Kreuzberg ships some files under both `nougat_*`
and `pdfa_*` names — 24 duplicates).

## Aggregate (78 unique fixtures)

|            | pdf_oxide | pdftotext |      Δ |
| ---------- | --------: | --------: | -----: |
| TF1 mean   |     0.930 |     0.944 |  -1.4pp |
| TF1 p50    |     0.973 |     0.985 |  -1.2pp |
| **TF1 p10**|     0.849 |     0.849 |  **0.0** |
| SF1 mean   |     0.354 |     0.247 | +10.7pp |
| SF1 p50    |     0.351 |     0.203 | +14.8pp |
| SF1 p10    |     0.129 |     0.017 | +11.2pp |
| order mean |     0.818 |     0.860 |  -4.3pp |
| order p50  |     0.846 |     0.964 | -11.8pp |

TF1 p10 is **tied**. TF1 mean sits 1.4pp behind because of a handful of
specific layouts. SF1 is 10.7pp ahead because we keep paragraph
structure that pdftotext's plain-text dump destroys.

25 fixtures (of 78) have a TF1 loss > 1pp. 8 of those are big losses
(>5pp). Of those 8 big losses, **5 are distinct root causes** — the
others are either duplicates (Kreuzberg corpus artefacts) or secondary
manifestations of the same bug.

## Five root-cause bug classes, ranked by corpus ROI

### B5 — multi-page PDFs drop body content after the first page

**Canary: `nougat_035 / pdfa_010` (TF1 0.627 vs 0.813, Δ −18.7pp).
Headline impact: estimated −0.4pp TF1 mean.**

5-page Celsius weight-loss study (Adobe Acrobat 9). We extract page 1
cleanly — title, abstract, first results block — and then subsequent
pages emit very little. 45 % of the GT vocabulary is missing
("abstract", "accepted", "acknowledgements", "adaptations",
"activation", "aerobic", "acute", "aids", …).

Per-page byte comparison with pdftotext:

| page | pdf_oxide | pdftotext |
| ---: | --------: | --------: |
|    1 |   ~1.6 kB |    ~1.6 kB |
|    2 |    0.2 kB |     1.5 kB |
|    3 |    0.3 kB |     1.8 kB |
|    4 |    0.2 kB |     1.1 kB |
|    5 |    0.1 kB |     0.7 kB |

Hypothesis: content-stream state bleed between pages (font metrics
stack, current transformation matrix). Not B1 (pages return *distinct*
content, just sparse). Probably a bail-out in the content-stream
parser triggered by a specific operator sequence used on pages 2+ of
Adobe-Acrobat-9 exports. pdfa_050 (28pg Nitro), pdfa_036 (2pg MS
Reporting) show the same symptom.

**How to fix**: turn on `RUST_LOG=pdf_oxide::extractors::text=trace`
on `extract_text(1)` for nougat_035; look for `parse_and_execute…`
bailing early. Likely a `Tm` or `Tj` edge case with specific numeric
representation.

### B6 — form/table PDFs lose rows after first several

**Canary: `nougat_026 / pdfa_001` (TF1 0.808 vs 0.987, Δ −17.9pp).
Estimated corpus impact: −0.23pp.**

St. Mary's Medical Center facility report (Microsoft Reporting
Services). Classic form layout:

```
| Facility Number: | 12460 |
| Facility Name:   | St. Mary's Medical Center |
| Address:         | 450 Stanyan Street |
| City:            | San Francisco |
| Hospital Owner:  | St. Mary's |
…
```

pdftotext emits every row. We emit the first three to five, then stop.
50 of 169 words missing ("adolescent", "alternate", "applicable",
"completion", "construction", "dates", "deadline", …).

Hypothesis: the spatial table detector finds the key-value grid and
passes it through the table-rendering pipeline, but one of
`detect_tables_with_lines` / `extract_page_tables` is returning only
the header rows. Interacts with acroform widget extraction — this
fixture has form widgets overlaid on the visual table. Same shape in
pdfa_044 (MS Word 2013) and pdfa_049 (MS Word 2010).

**How to fix**: run the extraction with `extract_tables: false` in
`ConversionOptions` — if the output suddenly gains the rest of the
rows, the table detector is the culprit. Inspect its row-filter logic.

### B7 — stroke + fill renders produce doubled text

**Canary: `nougat_016` (TF1 0.651 vs 0.793, Δ −14.2pp).
Estimated corpus impact: −0.20pp.**

City of Kirkland Lakeview neighbourhood map. Map labels are drawn as
a stroked outline + solid fill so they stand out over raster tiles.
Each label shows up twice in our output:

- `"EverestEverest"`, `"CentralCentral"`, `"HoughtonHoughton"` —
  the concatenation happens because our span-merging pass treats the
  two drawings as one span when they land at nearly identical
  positions.
- Side-effect: character drops mid-word in the merged output:
  `"war anties"` for "warranties", `"al rights"` for "all rights",
  because the two stroke + fill passes write each glyph at slightly
  different X positions and our merge logic picks the wrong side.

pdfa_026 (Adobe Acrobat Pro DC Paper Capture) shows a related pattern
with underscore form-field labels: `"____"` and `"________"` dupes.

**How to fix**: after span extraction, run an overlap-aware dedup
that, for any two spans with (a) byte-identical text after
normalisation (b) CTM bbox overlap > 80 % (c) font size match within
1 % — keep only one. Currently `deduplicate_overlapping_spans` in
`src/extractors/text.rs` handles some of this; needs extension to
catch this rendering pattern.

### B8 — intra-word spaces from aggressive space-insertion heuristic

**Canary: `nougat_047` (TF1 0.927 vs 0.964, Δ −3.8pp). Pattern also
dominates pdfa_044, nougat_029, pdfa_049, nougat_040, pdfa_037,
nougat_020. Estimated corpus impact: −0.4pp across 10+ fixtures.**

Our space-insertion heuristic in `TextExtractor` fires too often on
`TJ`-positioned glyphs, producing splits inside words:

```
"diffe rent partner regions"          (different)
"cha nge" "equivalen t"               (change, equivalent)
"w ere collected"                     (were)
"f ollows"                            (follows)
"Compa\n(the 'AGM')"                  (Company + line break)
"Dir\nectors"                         (Directors across line)
"comple\ntion"  "inspec\ntion"        (completion, inspection)
"dioactive"                           (radioactive first char dropped)
```

Two sub-patterns:
- **B8a** (line-break hyphenation): `word1-\nword2` should become
  `word1word2`. **Fixed in this branch** (`a90a31c`), +0.08pp TF1
  mean.
- **B8b** (intra-word TJ spaces): `"diffe rent"` comes from
  `(diffe)[-2](rent)` in the content stream — our extractor reads
  the negative TJ offset as a word boundary. Not a line break; a
  mid-line space.

**How to fix (B8b)**: the `space_threshold_em_ratio` of 0.25 in
`TextExtractor` is too low for fonts that use aggressive kerning.
pdftotext uses a denser signal (font's advance width × character
count) to predict expected X for the next glyph and only inserts a
space when actual vs. predicted X exceeds a font-dependent gap.
Rework the gap-computation to use font-advance prediction instead
of a fixed em-ratio.

### B9 — TrueType `cmap format 0` unsupported

**Impact: surfaces as warnings on ~8 fixtures (`pdfa_036`, `pdfa_037`,
pdfa_049, nougat_033 etc.), each losing 3–7pp TF1. Estimated corpus
impact: −0.25pp.**

```
[WARN] Font 'KPSHBO+Calibri': TrueType cmap extraction failed:
       Unsupported cmap format: 0
```

cmap format 0 is a legacy byte-indexed 1-to-1 encoding (Mac classic).
Some Microsoft Office exports still emit it for subset fonts.
`ttf-parser` supports format 0; we need to call the right code path
when encountering it.

**How to fix**: small, bounded — `src/fonts/font_dict.rs` cmap
extraction needs a `format 0` arm that reads the 262 glyph table
directly. Same class of fix as #325 (TrueType simple subset).

## The 8 big losses, attributed

| Fixture      | TF1 loss | Root cause |
| ------------ | -------: | --- |
| nougat_035 / pdfa_010 | −18.7pp | B5 |
| nougat_026 / pdfa_001 | −17.9pp | B6 |
| nougat_016            | −14.2pp | B7 |
| pdfa_050              |  −8.8pp | B5 |
| pdfa_036              |  −7.7pp | B5 + B9 |
| nougat_046            |  −7.3pp | B8b |
| pdfa_026              |  −6.6pp | B7 (underscore dupes) |
| pdfa_044              |  −5.4pp | B6 + B8b |

## What a full sweep buys

| Fix | Estimated TF1-mean impact |
| --- | ---: |
| B5 (multi-page parse bail-out) | +0.4pp |
| B6 (table detector row filter) | +0.2pp |
| B7 (stroke+fill dedup)         | +0.2pp |
| B8a (hyphenation, landed)      | +0.1pp |
| B8b (intra-word TJ spaces)     | +0.4pp |
| B9  (cmap format 0)            | +0.25pp |
| **total**                      | **+1.6pp** |

That matches the observed gap to pdftotext, which means the
remaining aggregate is fully explained by these five bug classes —
no hidden "we're worse at everything" signal.

Shipping all of them would push pdf_oxide's TF1 to parity with
pdftotext (or slightly ahead on the hard tail) while keeping our
10.7pp SF1 lead. That's the next release's target.

## Sequencing recommendation

1. **B9** first — small, bounded, well-understood (reuse the
   TrueType simple-subset code from fix #325). Lowest risk.
2. **B7** — overlap-aware dedup is a localised change in
   `deduplicate_overlapping_spans`. Adds unit coverage by
   re-rendering a toy PDF with stroke+fill.
3. **B8b** — requires calibrating the space heuristic on real PDFs.
   High value but needs the most care to avoid regressing currently
   good cases.
4. **B5** — investigate with `RUST_LOG=trace` on nougat_035 page 2.
5. **B6** — test with `extract_tables: false` first to confirm the
   attribution, then dig into table detector.

Every fix lands with a benchmark-harness comparison. The diff gate
rejects any change that drops mean TF1 > 0.5pp or any per-fixture
> 5pp, so we have automatic safety netting.

## Workflow reproduction

```bash
make benchmark-fetch
cargo build --release -p benchmark-harness

# before
git checkout v0.3.31
make benchmark-run OUTPUT=target/bench/v0.3.31.json

# after
git checkout fix/b1-b3-b4-combined-with-harness
make benchmark-run OUTPUT=target/bench/head.json

make benchmark-compare BASE=target/bench/v0.3.31.json HEAD=target/bench/head.json
```
