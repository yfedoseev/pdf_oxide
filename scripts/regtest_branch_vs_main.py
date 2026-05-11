#!/usr/bin/env python3
"""Regression harness: release/v0.3.46 branch vs main.

Compares dump_all_formats (text/markdown/html) for both builds
across a curated 120-PDF corpus sampled from ~/projects/pdf_oxide_tests.

Subcommands:
  collect  -- sample corpus, write scripts/regtest_corpus_v0346.txt
  run      -- execute both builds on all PDFs
  report   -- diff outputs, write /tmp/regtest_v0346/report.md + report.tsv
  show     -- dump side-by-side for one PDF
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
TESTS_ROOT = Path(os.path.expanduser("~/projects/pdf_oxide_tests"))

CORPUS_FILE = SCRIPTS_DIR / "regtest_corpus_v0346.txt"
RUN_ROOT = Path("/tmp/regtest_v0346")
BRANCH_BIN_DEFAULT = RUN_ROOT / "bin" / "dump_all_formats.branch"
MAIN_BIN_DEFAULT = RUN_ROOT / "bin" / "dump_all_formats.main"

MAX_BYTES = 50 * 1024 * 1024  # 50 MB cap; raised to 100 MB for ocr bucket
TIMEOUT = 30  # seconds per invocation
FORMATS = ("text", "markdown", "html")

# ---------------------------------------------------------------------------
# Corpus sampling
# ---------------------------------------------------------------------------

BUCKETS = [
    # (name, source_dirs, target_n, max_bytes_override, pattern_hint)
    (
        "multi_column",
        [
            TESTS_ROOT / "pdfs" / "academic",
            TESTS_ROOT / "pdfs_1000" / "academic" / "arxiv",
            TESTS_ROOT / "pdfs_1000" / "academic" / "journals",
        ],
        30,
        None,
        None,
    ),
    (
        "tables_dense",
        [
            TESTS_ROOT / "irs",
            TESTS_ROOT / "fixtures_policy",
            TESTS_ROOT / "pdfs_1000" / "academic" / "10k",
            TESTS_ROOT / "pdfs_1000" / "academic" / "reports",
        ],
        20,
        None,
        None,
    ),
    (
        "single_column",
        [
            TESTS_ROOT / "pdfs" / "diverse",
            TESTS_ROOT / "pdfs" / "theses",
            TESTS_ROOT / "pdfs" / "government",
            TESTS_ROOT / "pdfs_1000" / "academic" / "policy",
            TESTS_ROOT / "pdfs_1000" / "academic" / "standards",
        ],
        20,
        None,
        None,
    ),
    (
        "ocr_scanned",
        [
            TESTS_ROOT / "fixtures_ocr",
            TESTS_ROOT / "pdfs" / "mixed",
        ],
        10,
        100 * 1024 * 1024,
        None,
    ),
    (
        "pdfjs_torture",
        [
            TESTS_ROOT / "pdfs_pdfjs",
        ],
        15,
        None,
        None,
    ),
    (
        "safedocs_edge",
        [
            TESTS_ROOT / "pdfs_safedocs",
        ],
        10,
        None,
        None,
    ),
    (
        "fixtures_regression",
        [
            TESTS_ROOT / "fixtures_regression",
        ],
        10,
        None,
        None,
    ),
    (
        "mixed_other",
        [
            TESTS_ROOT / "pdfs" / "mixed",
            TESTS_ROOT / "pdfs_pdfium",
            TESTS_ROOT / "pdfs_1000" / "academic" / "magazines",
            TESTS_ROOT / "pdfs_1000" / "academic" / "manuals",
        ],
        10,
        None,
        None,
    ),
]


def _collect_bucket(dirs, target_n, max_bytes, rng):
    candidates = []
    for d in dirs:
        if not d.exists():
            continue
        for p in d.rglob("*.pdf"):
            try:
                sz = p.stat().st_size
            except OSError:
                continue
            limit = max_bytes or MAX_BYTES
            if sz < 2048 or sz > limit:
                continue
            candidates.append(p)
    candidates = sorted(set(candidates))
    rng.shuffle(candidates)
    return candidates[:target_n]


def cmd_collect(args):
    total_target = args.target
    default_total = sum(t for _, _, t, _, _ in BUCKETS)
    rng = random.Random(0xC0FFEE)
    seen: set = set()
    rows: List[Tuple[str, Path]] = []

    for bucket, dirs, target_n, max_bytes, _ in BUCKETS:
        bucket_target = max(1, round(target_n * total_target / default_total))
        picked = _collect_bucket(dirs, bucket_target * 3, max_bytes, rng)
        added = 0
        for p in picked:
            if str(p) in seen:
                continue
            seen.add(str(p))
            rows.append((bucket, p))
            added += 1
            if added >= bucket_target:
                break
        print(f"  {bucket:<22} {added:>3} / {target_n} PDFs")

    total = len(rows)
    print(f"\nTotal: {total} PDFs")
    CORPUS_FILE.write_text("\n".join(f"{b}\t{p}" for b, p in rows) + "\n")
    print(f"Written → {CORPUS_FILE}")


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def _run_one(bin_path: str, pdf: str, fmt: str, out_path: str, timeout: int) -> dict:
    t0 = time.monotonic()
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    try:
        r = subprocess.run(
            [bin_path, pdf, fmt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.monotonic() - t0
        if r.returncode == 0:
            out_p.write_text(r.stdout, errors="replace")
            status = "ok"
        else:
            out_p.write_text(r.stdout or "", errors="replace")
            status = "crash"
        return {
            "status": status,
            "rc": r.returncode,
            "elapsed": round(elapsed, 2),
            "bytes": len(r.stdout),
            "stderr": r.stderr[-500:] if r.stderr else "",
        }
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        out_p.write_text("", errors="replace")
        return {
            "status": "timeout",
            "rc": None,
            "elapsed": round(elapsed, 2),
            "bytes": 0,
            "stderr": "",
        }
    except Exception as e:
        out_p.write_text("", errors="replace")
        return {"status": "error", "rc": None, "elapsed": 0, "bytes": 0, "stderr": str(e)}


def _worker(args_tuple):
    build, bin_path, pdf, fmt, out_path, timeout = args_tuple
    meta = _run_one(bin_path, pdf, fmt, out_path, timeout)
    return (build, pdf, fmt, meta)


def cmd_run(args):
    corpus = [
        (b, p)
        for line in Path(args.corpus).read_text().splitlines()
        for b, p in [line.split("\t", 1)]
    ]
    branch_bin = args.branch_bin
    main_bin = args.main_bin
    out_root = Path(args.out)
    timeout = args.timeout
    jobs = args.jobs

    manifest_path = out_root / "manifest.jsonl"
    done: set = set()
    if manifest_path.exists() and not args.force:
        for line in manifest_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["build"], r["pdf"], r["format"]))
            except Exception:
                pass

    tasks = []
    for bucket, pdf in corpus:
        stem = Path(pdf).stem
        for fmt in FORMATS:
            for build, bin_path in (("branch", branch_bin), ("main", main_bin)):
                if (build, pdf, fmt) in done:
                    continue
                out_path = str(out_root / build / bucket / f"{stem}.{fmt}")
                tasks.append((build, bin_path, pdf, fmt, out_path, timeout))

    total = len(tasks)
    print(f"Running {total} tasks ({jobs} workers, {timeout}s timeout each) …")

    completed = 0
    with open(manifest_path, "a") as mf, ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}
        for fut in as_completed(futs):
            build, pdf, fmt, meta = fut.result()
            bucket = next(b for b, p in corpus if p == pdf)
            stem = Path(pdf).stem
            row = {
                "build": build,
                "pdf": pdf,
                "bucket": bucket,
                "stem": stem,
                "format": fmt,
                **meta,
            }
            mf.write(json.dumps(row) + "\n")
            mf.flush()
            completed += 1
            status_sym = {"ok": "✓", "crash": "✗", "timeout": "T", "error": "E"}.get(
                meta["status"], "?"
            )
            print(f"  [{completed:>4}/{total}] {status_sym} {build:6} {fmt:8} {stem[:40]}")

    print(f"\nDone. Manifest → {manifest_path}")


# ---------------------------------------------------------------------------
# Tokenizer / diff helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[\w/\-\.@]+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text)


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(_tokenize(a)), set(_tokenize(b))
    u = len(sa | sb)
    return len(sa & sb) / u if u else 1.0


def _classify(main_txt: str, branch_txt: str, main_ok: bool, branch_ok: bool) -> Tuple[str, float]:
    if not main_ok and not branch_ok:
        return "CRASH_BOTH", 0.0
    if not branch_ok:
        return "CRASH_BRANCH", 0.0
    if not main_ok:
        return "CRASH_MAIN", 1.0  # branch fixes main crash → positive
    if main_txt == branch_txt:
        return "IDENTICAL", 1.0

    def norm(s):
        return re.sub(r"\s+", " ", s).strip()

    if norm(main_txt) == norm(branch_txt):
        return "WHITESPACE_ONLY", 1.0
    j = _jaccard(main_txt, branch_txt)
    byte_d = abs(len(branch_txt) - len(main_txt))
    byte_pct = byte_d / max(len(main_txt), 1)
    if j >= 0.98 and byte_pct < 0.02:
        return "MINOR_DIFF", j
    if not main_txt and branch_txt:
        return "NEW_OUTPUT", 1.0
    if main_txt and not branch_txt:
        return "LOST_OUTPUT", 0.0
    return "CONTENT_DIFF", j


def _wdiff(main_txt: str, branch_txt: str, n: int = 3) -> str:
    import difflib

    mt = _tokenize(main_txt)
    bt = _tokenize(branch_txt)
    diff = list(difflib.unified_diff(mt, bt, fromfile="main", tofile="branch", lineterm="", n=n))
    return "\n".join(diff[:80])


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

STATUS_ORDER = [
    "CRASH_BRANCH",
    "LOST_OUTPUT",
    "CONTENT_DIFF",
    "MINOR_DIFF",
    "WHITESPACE_ONLY",
    "NEW_OUTPUT",
    "CRASH_MAIN",
    "CRASH_BOTH",
    "IDENTICAL",
]


def cmd_report(args):
    run_root = Path(args.run)
    # manifest lives under out/ (written by cmd_run)
    manifest_path = run_root / "out" / "manifest.jsonl"
    if not manifest_path.exists():
        manifest_path = run_root / "manifest.jsonl"
    diff_dir = run_root / "diffs"
    diff_dir.mkdir(exist_ok=True)

    rows: Dict[Tuple[str, str, str], dict] = {}  # (pdf, fmt, build) → row
    for line in manifest_path.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        rows[(r["pdf"], r["format"], r["build"])] = r

    # Build set of (pdf, fmt) pairs we have both sides for
    pairs: set = set()
    for pdf, fmt, _build in rows:
        pairs.add((pdf, fmt))

    tsv_rows = []
    content_diffs = []
    counts: Dict[Tuple[str, str], int] = {}  # (status, fmt) → count
    bucket_status: Dict[Tuple[str, str], int] = {}  # (bucket, status) → count

    for pdf, fmt in sorted(pairs):
        main_row = rows.get((pdf, fmt, "main"), {})
        branch_row = rows.get((pdf, fmt, "branch"), {})

        bucket = main_row.get("bucket") or branch_row.get("bucket") or "?"
        stem = Path(pdf).stem

        # Load text outputs
        def _load(build, _pdf=pdf, _fmt=fmt, _stem=stem):
            r = rows.get((_pdf, _fmt, build), {})
            out_p = run_root / "out" / build / r.get("bucket", "?") / f"{_stem}.{_fmt}"
            if out_p.exists():
                return out_p.read_text(errors="replace")
            return ""

        main_txt = _load("main")
        branch_txt = _load("branch")

        main_ok = main_row.get("status") == "ok"
        branch_ok = branch_row.get("status") == "ok"

        status, jaccard = _classify(main_txt, branch_txt, main_ok, branch_ok)

        bd = len(branch_txt) - len(main_txt)
        em = main_row.get("elapsed", 0)
        eb = branch_row.get("elapsed", 0)

        note = ""
        if status == "CONTENT_DIFF":
            wdiff_p = diff_dir / f"{bucket}__{stem}__{fmt}.wdiff"
            wdiff_p.write_text(_wdiff(main_txt, branch_txt))
            note = f"see diffs/{wdiff_p.name}"
            content_diffs.append((pdf, fmt, jaccard, bd, bucket, stem, note))

        tsv_rows.append(
            {
                "bucket": bucket,
                "pdf": Path(pdf).name,
                "format": fmt,
                "status": status,
                "jaccard": round(jaccard, 4),
                "bytes_main": len(main_txt),
                "bytes_branch": len(branch_txt),
                "byte_delta": bd,
                "elapsed_main": em,
                "elapsed_branch": eb,
                "notes": note,
            }
        )

        counts[(status, fmt)] = counts.get((status, fmt), 0) + 1
        bucket_status[(bucket, status)] = bucket_status.get((bucket, status), 0) + 1

    # Write TSV
    tsv_path = run_root / "report.tsv"
    tsv_hdr = "bucket\tpdf\tformat\tstatus\tjaccard\tbytes_main\tbytes_branch\tbyte_delta\telapsed_main\telapsed_branch\tnotes"
    with open(tsv_path, "w") as f:
        f.write(tsv_hdr + "\n")
        for r in tsv_rows:
            f.write("\t".join(str(r[k]) for k in tsv_hdr.split("\t")) + "\n")

    # Build markdown report
    all_statuses = sorted(
        {s for s, f in counts}, key=lambda s: STATUS_ORDER.index(s) if s in STATUS_ORDER else 99
    )
    all_fmts = FORMATS

    md_lines = [
        "# Regression Report: release/v0.3.46 vs main",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Corpus: {len(pairs) // 3} PDFs × 3 formats = {len(pairs)} comparisons",
        "",
        "## Summary",
        "",
        "| Status | text | markdown | html | total |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for st in all_statuses:
        tc = sum(counts.get((st, f), 0) for f in all_fmts)
        cols = " | ".join(str(counts.get((st, f), 0)) for f in all_fmts)
        flag = " ← review" if st in ("CRASH_BRANCH", "LOST_OUTPUT", "CONTENT_DIFF") else ""
        flag += " ← expected improvement" if st in ("CRASH_MAIN", "NEW_OUTPUT") else ""
        md_lines.append(f"| {st} | {cols} | {tc} |{flag}")

    # Content diffs table
    md_lines += [
        "",
        "## Content diffs requiring review",
        "",
        "| PDF | fmt | jaccard | Δbytes | bucket | wdiff |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for pdf, fmt, j, bd, bucket, _stem, note in sorted(content_diffs, key=lambda x: x[2]):
        md_lines.append(f"| {Path(pdf).name} | {fmt} | {j:.3f} | {bd:+} | {bucket} | {note} |")

    # Crashes / new outputs
    crashes = [r for r in tsv_rows if "CRASH" in r["status"] or r["status"] in ("LOST_OUTPUT",)]
    new_out = [r for r in tsv_rows if r["status"] in ("NEW_OUTPUT",)]

    if crashes:
        md_lines += [
            "",
            "## Crashes / lost output",
            "",
            "| PDF | fmt | status | bucket |",
            "| --- | --- | --- | --- |",
        ]
        for r in crashes:
            md_lines.append(f"| {r['pdf']} | {r['format']} | {r['status']} | {r['bucket']} |")

    if new_out:
        md_lines += [
            "",
            "## New output (branch fixes main crash/empty)",
            "",
            "| PDF | fmt | bucket |",
            "| --- | --- | --- |",
        ]
        for r in new_out:
            md_lines.append(f"| {r['pdf']} | {r['format']} | {r['bucket']} |")

    # Top 30 worst by jaccard (content diff only)
    worst = sorted(
        [r for r in tsv_rows if r["status"] == "CONTENT_DIFF"], key=lambda r: r["jaccard"]
    )[:30]
    if worst:
        md_lines += [
            "",
            "## Top 30 worst content diffs (lowest Jaccard)",
            "",
            "| PDF | fmt | jaccard | Δbytes | bucket |",
            "| --- | --- | ---: | ---: | --- |",
        ]
        for r in worst:
            md_lines.append(
                f"| {r['pdf']} | {r['format']} | {r['jaccard']:.3f} | {r['byte_delta']:+} | {r['bucket']} |"
            )

    md_path = run_root / "report.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"\nReport → {md_path}")
    print(f"TSV    → {tsv_path}")
    print(f"Diffs  → {diff_dir}/")

    # Console summary
    print("\n=== SUMMARY ===")
    for st in all_statuses:
        tc = sum(counts.get((st, f), 0) for f in all_fmts)
        print(
            f"  {st:<22} {tc:>4}  ({', '.join(f'{f}:{counts.get((st, f), 0)}' for f in all_fmts)})"
        )


# ---------------------------------------------------------------------------
# Show
# ---------------------------------------------------------------------------


def cmd_show(args):
    run_root = Path(args.run)
    manifest_path = run_root / "manifest.jsonl"
    target_pdf = args.pdf
    fmt = args.format

    for line in manifest_path.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if Path(r["pdf"]).name != Path(target_pdf).name:
            continue
        if r["format"] != fmt:
            continue
        out_p = run_root / "out" / r["build"] / r["bucket"] / f"{r['stem']}.{fmt}"
        print(f"\n{'=' * 60}\n{r['build'].upper()} ({r['status']}, {r['elapsed']}s)\n{'=' * 60}")
        if out_p.exists():
            print(out_p.read_text(errors="replace")[:3000])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="branch vs main regression harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("collect", help="Sample corpus")
    pc.add_argument("--target", type=int, default=120)

    pr = sub.add_parser("run", help="Execute both builds")
    pr.add_argument("--corpus", default=str(CORPUS_FILE))
    pr.add_argument("--branch-bin", default=str(BRANCH_BIN_DEFAULT))
    pr.add_argument("--main-bin", default=str(MAIN_BIN_DEFAULT))
    pr.add_argument("--out", default=str(RUN_ROOT / "out"))
    pr.add_argument("--timeout", type=int, default=TIMEOUT)
    pr.add_argument("--jobs", type=int, default=4)
    pr.add_argument("--force", action="store_true")

    pd = sub.add_parser("report", help="Diff outputs and write report")
    pd.add_argument("--run", default=str(RUN_ROOT))

    ps = sub.add_parser("show", help="Side-by-side for one PDF")
    ps.add_argument("--run", default=str(RUN_ROOT))
    ps.add_argument("--pdf", required=True)
    ps.add_argument("--format", default="text")

    args = p.parse_args()
    {"collect": cmd_collect, "run": cmd_run, "report": cmd_report, "show": cmd_show}[args.cmd](args)


if __name__ == "__main__":
    main()
