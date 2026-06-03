# olmOCR-bench regression harness (#567)

[`allenai/olmOCR-bench`](https://huggingface.co/datasets/allenai/olmOCR-bench)
(ODC-BY) as a second public regression corpus for pdf_oxide, alongside the
py-pdf/benchmarks set. It ships single-page PDFs across prose-oriented subsets
with *checkable assertions* (substring present/absent, substring order,
baseline non-empty, and LaTeX `math`) rather than full-text ground truth — so
it is cheap to run and diff in CI.

The corpus is **never committed** (it is fetched on demand and gitignored under
`../fixtures/olmocr-bench/`).

## Fetch

```bash
python -c "from huggingface_hub import snapshot_download as s; \
  s(repo_id='allenai/olmOCR-bench', repo_type='dataset', \
    local_dir='tools/benchmark-harness/fixtures/olmocr-bench', \
    allow_patterns=['bench_data/*.jsonl','bench_data/pdfs/**'])"
```

## Run

```bash
# build the python extension first
maturin develop --release --features python

python tools/benchmark-harness/olmocr/run_olmocr_bench.py \
  --bench-data tools/benchmark-harness/fixtures/olmocr-bench/bench_data \
  --baseline multi_column=0 --baseline headers_footers=90
```

## Scope

pdf_oxide is a plain-text/markdown extractor, not an OCR/LLM pipeline. The
harness scores the **present / absent / order / baseline** assertion types and
reports **math / table-structure** assertions as *out-of-scope* (pdf_oxide
emits plain text, not LaTeX). Consequently:

- `arxiv_math` ≈ 0% checkable is **expected**, not a regression (math is OOS).
- `multi_column` is the meaningful reading-order signal.
- `headers_footers` exercises chrome handling (#553).

Pass a `--baseline subset=pct` floor per subset to make the run a CI gate; it
exits non-zero if a subset drops below its recorded floor. It is **opt-in**
(not part of default CI) and disk-light only after the one-time fetch.
