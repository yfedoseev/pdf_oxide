#!/usr/bin/env python3
"""olmOCR-bench regression harness for pdf_oxide (#567).

Scores pdf_oxide's plain-text extraction against the allenai/olmOCR-bench
assertion set. Checkable assertion types for a plain-text (non-OCR, non-LLM)
extractor: present / absent / order / baseline. `math` (LaTeX) and table-
structure assertions are reported as out-of-scope — pdf_oxide emits plain
text, not LaTeX, so counting them as failures would be misleading (this is
why the issue's arxiv_math baseline is ~0% by construction while multi_column
is the meaningful signal).

The corpus itself is NOT committed; fetch it with:

    python -c "from huggingface_hub import snapshot_download as s; \\
      s(repo_id='allenai/olmOCR-bench', repo_type='dataset', \\
        local_dir='tools/benchmark-harness/fixtures/olmocr-bench', \\
        allow_patterns=['bench_data/*.jsonl','bench_data/pdfs/**'])"

Then build the wheel (maturin develop --release --features python) and run:

    python tools/benchmark-harness/olmocr/run_olmocr_bench.py \\
      --bench-data tools/benchmark-harness/fixtures/olmocr-bench/bench_data

Exit code is non-zero if any subset's checkable pass-rate falls below the
matching --baseline floor (CI regression gate; opt-in, not run by default).
"""

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path


def norm(s):
    return " ".join((s or "").split())


def extract_text(python_bin, pdf_path, timeout):
    """Extract full text via a pdf_oxide subprocess (crash/hang isolated)."""
    code = (
        "import sys,pdf_oxide;"
        "d=pdf_oxide.PdfDocument(sys.argv[1]);"
        "n=d.page_count if isinstance(d.page_count,int) else d.page_count();"
        "sys.stdout.write(chr(10).join(d.extract_text(i) for i in range(n)))"
    )
    try:
        r = subprocess.run(
            [python_bin, "-c", code, str(pdf_path)], capture_output=True, timeout=timeout
        )
        return r.stdout.decode("utf-8", "replace") if r.returncode == 0 else ""
    except Exception:
        return ""


def check(assertion, text):
    nt = norm(text)
    t = assertion.get("type")
    if t == "present":
        return norm(assertion["text"]) in nt
    if t == "absent":
        return norm(assertion["text"]) not in nt
    if t == "order":
        b, a = norm(assertion["before"]), norm(assertion["after"])
        ib, ia = nt.find(b), nt.find(a)
        return ib != -1 and ia != -1 and ib < ia
    if t == "baseline":
        return len(nt) > 0
    return None  # out of scope (math / table-structure)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bench-data", required=True, help="olmOCR bench_data dir (contains *.jsonl + pdfs/)"
    )
    ap.add_argument("--python", default=sys.executable, help="python with pdf_oxide installed")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument(
        "--baseline",
        action="append",
        default=[],
        help="subset=pct floor, e.g. multi_column=0 (repeatable)",
    )
    args = ap.parse_args()

    root = Path(args.bench_data)
    pdfs_root = root / "pdfs"
    floors = {}
    for b in args.baseline:
        k, _, v = b.partition("=")
        floors[k] = float(v)

    cache = {}
    bysub = collections.defaultdict(lambda: collections.Counter())
    for jf in sorted(root.glob("*.jsonl")):
        sub = jf.stem
        for line in jf.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                a = json.loads(line)
            except Exception:
                continue
            t = a.get("type", "?")
            if t not in ("present", "absent", "order", "baseline"):
                bysub[sub]["oos"] += 1
                continue
            pdf = a["pdf"]
            if pdf not in cache:
                cache[pdf] = extract_text(args.python, pdfs_root / pdf, args.timeout)
            ok = check(a, cache[pdf])
            bysub[sub]["pass" if ok else "fail"] += 1
        print(f"scored {sub}", flush=True)

    print("\n=== olmOCR-bench (pdf_oxide) ===")
    print(f"{'subset':22} {'pass-rate':>14} {'oos':>6}")
    regressions = []
    for sub in sorted(bysub):
        c = bysub[sub]
        denom = c["pass"] + c["fail"]
        pct = (100.0 * c["pass"] / denom) if denom else 0.0
        print(f"{sub:22} {c['pass']:>5}/{denom:<5} ({pct:5.1f}%) {c['oos']:>6}")
        if sub in floors and pct < floors[sub]:
            regressions.append(f"{sub}: {pct:.1f}% < floor {floors[sub]:.1f}%")

    if regressions:
        print("\nREGRESSION:", "; ".join(regressions))
        sys.exit(1)


if __name__ == "__main__":
    main()
